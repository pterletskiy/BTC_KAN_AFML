"""
src/7_models.py
---------------
Model definitions and training logic for the MLDP quantitative pipeline.

Executes Kolmogorov-Arnold Networks (PureKAN, TKAN, KASPER) and 
equivalent baseline benchmarks. Incorporates AFML sample weight 
normalizations inside an overlapping PyTorch / Scikit framework.

References:
  - AFML Ch. 4: Sample Weighting mapped to N observations
  - AFML Ch. 9: Objective evaluation and log-loss objective
  - AFML Ch. 10: Probability calibration for bet sizing
"""

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import IsotonicRegression

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

logger = logging.getLogger(__name__)


# ==============================================================================
# LOSS FUNCTION (AFML Weighted Log-Loss)
# ==============================================================================
def weighted_neg_log_loss(y_true: torch.Tensor, y_pred_proba: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    """
    Computes sample-weighted binary cross-entropy.
    Why: AFML requires penalizing high-confidence misclassifications because overconfident 
    incorrect predictions lead to blown-out portfolio drawdowns during bet sizing.
    
    Args:
        y_true: Ground truth target labels [0, 1].
        y_pred_proba: Predictive probabilities.
        sample_weight: Uniqueness attributes applied per sample.

    Returns:
        torch.Tensor: Normalized loss scalar.
    """
    p = torch.clamp(y_pred_proba, 1e-7, 1.0 - 1e-7)
    
    # Scale sample weights to sum to N (batch size).
    # Why: Maintains learning rate stability dynamically regardless of batch sample weights.
    weight_scaled = sample_weight * (len(sample_weight) / (sample_weight.sum() + 1e-8))
    
    loss = -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p))
    return torch.mean(loss * weight_scaled)


# ==============================================================================
# BASELINE WRAPPERS (Interface uniform `.fit()` / `.predict_proba()`)
# ==============================================================================
class ARLogistic:
    """
    Autoregressive baseline mapping only temporal autocorrelation of targets.
    Why: Serves as a control to test if our features are predicting anything beyond 
    simple trend persistence.
    """
    
    def __init__(self, config: dict):
        self.lags = config.get('lags', 1)
        self.model = LogisticRegression(C=config.get('C', 1.0), solver=config.get('solver', 'lbfgs'))
        self.last_y = None  # Buffer for most recent target values seen during fit
        
    def _make_lags(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build historical target momentum mapping bounds."""
        N = len(y)
        if N <= self.lags:
            return np.zeros((N, self.lags)), y
            
        X_lag = np.zeros((N - self.lags, self.lags))
        for i in range(self.lags):
            X_lag[:, i] = y[i : N - self.lags + i]
            
        y_out = y[self.lags:]
        return X_lag, y_out

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> 'ARLogistic':
        self.last_y = y.copy()
        X_lag, y_adj = self._make_lags(X, y)
        w_adj = sample_weight[self.lags:] if sample_weight is not None else None
        
        if len(np.unique(y_adj)) > 1:
            self.model.fit(X_lag, y_adj, sample_weight=w_adj)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Produce predictions recursively over the test timeline.
        Why: We don't have true `y` at test time, so we must feed our
        own hard predictions back in autoregressively to generate the path.
        """
        N = len(X)
        if hasattr(self.model, 'classes_') is False or self.last_y is None or len(self.last_y) < self.lags:
            return np.ones((N, 2)) * 0.5
            
        preds = []
        curr_lags = self.last_y[-self.lags:].copy()
        
        for _ in range(N):
            X_in = curr_lags.reshape(1, -1)
            p = self.model.predict_proba(X_in)[0]
            preds.append(p)
            
            y_pred = self.model.predict(X_in)[0]
            curr_lags = np.roll(curr_lags, -1)
            curr_lags[-1] = y_pred
            
        return np.array(preds)


class SklearnBaseline:
    """Wraps MLDP Random Forest or Logistic Regression implementations."""
    
    def __init__(self, clf_type: str, config: dict):
        self.clf_type = clf_type
        if clf_type == 'logistic':
            self.model = LogisticRegression(C=config.get('C', 1.0), solver=config.get('solver', 'lbfgs'), max_iter=1000)
        elif clf_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=config.get('n_estimators', 500),
                class_weight='balanced_subsample',
                max_depth=config.get('max_depth', None)
            )
        elif clf_type == 'xgb':
            if XGBClassifier is None:
                raise ImportError("XGBoost not installed.")
            self.model = XGBClassifier(
                n_estimators=config.get('n_estimators', 500),
                eval_metric='logloss',
                max_depth=config.get('max_depth', 3),
                learning_rate=config.get('learning_rate', 0.1)
            )
            
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> 'SklearnBaseline':
        if self.clf_type == 'rf' and sample_weight is not None:
            # Why: Setting max_samples to the average uniqueness replicates MLDP Sequential Bootstrap logic securely.
            avg_u = np.mean(sample_weight)
            max_samples = min(1.0, max(0.1, avg_u))
            self.model.set_params(max_samples=max_samples)
            
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return np.ones((len(X), 2)) * 0.5


# ==============================================================================
# PYTORCH MLP BASELINE
# ==============================================================================
class MLPModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ==============================================================================
# KAN ARCHITECTURES
# ==============================================================================
class KANLayer(nn.Module):
    """
    Computes learnable B-spline math replacing traditional linear weight matrices.
    Why: B-splines capture complex non-linear feature interactions natively without
    requiring arbitrary layer depths.
    """
    
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, k: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.k = k

        step = 2.0 / grid_size
        grid = torch.arange(-1 - k * step, 1 + (k + 1) * step, step)
        self.register_buffer('grid', grid)

        self.coef = nn.Parameter(torch.randn(out_features, in_features, grid_size + k) * 0.1)

    def compute_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates contiguous differentiable B-spline bases recursive formula maps."""
        x_expanded = x.unsqueeze(-1)
        
        bases = ((x_expanded >= self.grid[:-1]) & (x_expanded < self.grid[1:])).float()
        
        for d in range(1, self.k + 1):
            left_denom = self.grid[d:-1] - self.grid[:-d-1]
            right_denom = self.grid[d+1:] - self.grid[1:-d]
            
            left_term = (x_expanded - self.grid[:-d-1]) / torch.where(left_denom == 0, torch.ones_like(left_denom), left_denom)
            right_term = (self.grid[d+1:] - x_expanded) / torch.where(right_denom == 0, torch.ones_like(right_denom), right_denom)
            
            bases = left_term * bases[..., :-1] + right_term * bases[..., 1:]
            
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bases = self.compute_spline_basis(x)
        return torch.einsum('bis,ois->bo', bases, self.coef)


class PureKAN(nn.Module):
    """
    Standalone Kolmogorov-Arnold Network extracting features via learnable
    B-spline activation grids rather than linear combinations.
    """
    
    def __init__(self, in_features: int, layer_dims: list, grid_size: int = 5, k: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_in = in_features
        
        for dim in layer_dims:
            self.layers.append(KANLayer(curr_in, dim, grid_size, k))
            # Why: Normalization between KAN layers prevents B-spline output magnitudes
            # from drifting uncontrollably away from the fixed [-1, 1] grid mapping.
            self.layers.append(nn.LayerNorm(dim))
            curr_in = dim
            
        self.head = nn.Linear(curr_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.head(x)).squeeze(-1)

    def get_activation_functions(self) -> dict:
        """Returns spline arrays required for downstream exact expression extraction."""
        return {
            f"layer_{i}": {'grid': l.grid.cpu().numpy(), 'coef': l.coef.detach().cpu().numpy(), 'k': l.k}
            for i, l in enumerate(self.layers) if isinstance(l, KANLayer)
        }


class TKAN(nn.Module):
    """
    Temporal KAN incorporating Recurrent dependencies integrating explicit LSTM-styled topologies natively.
    """
    
    def __init__(self, in_features: int, hidden_dim: int, grid_size: int = 5, k: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.kan_i = KANLayer(in_features + hidden_dim, hidden_dim, grid_size, k)
        self.kan_f = KANLayer(in_features + hidden_dim, hidden_dim, grid_size, k)
        self.kan_o = KANLayer(in_features + hidden_dim, hidden_dim, grid_size, k)
        self.kan_g = KANLayer(in_features + hidden_dim, hidden_dim, grid_size, k)
        
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x_seq.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            xh = torch.cat([x_t, h], dim=1)
            
            i = torch.sigmoid(self.kan_i(xh))
            f = torch.sigmoid(self.kan_f(xh))
            o = torch.sigmoid(self.kan_o(xh))
            g = torch.tanh(self.kan_g(xh))
            
            c = f * c + i * g
            h = o * torch.tanh(c)
            
        return torch.sigmoid(self.head(h)).squeeze(-1)


class KASPER(nn.Module):
    """
    Regime Adaptive Model extracting Gumbel probabilities configuring dynamic soft weights targeting isolated clusters natively.
    """
    
    def __init__(self, in_features: int, num_regimes: int, kan_dims: list, grid_size: int = 5, k: int = 3, tau: float = 1.0):
        super().__init__()
        self.num_regimes = num_regimes
        self.tau = tau
        
        hidden_r = max(4, in_features // 2)
        self.detector = nn.Sequential(
            nn.Linear(in_features, hidden_r),
            nn.ReLU(),
            nn.Linear(hidden_r, num_regimes)
        )
        
        self.regime_kans = nn.ModuleList([
            PureKAN(in_features, kan_dims, grid_size, k)
            for _ in range(num_regimes)
        ])
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.detector(x)
        # Why: Gumbel-Softmax allows completely differentiable regime detection mapping while cleanly bounding outputs.
        r = F.gumbel_softmax(logits, tau=self.tau, hard=False)
        
        preds = []
        for i in range(self.num_regimes):
            preds.append(self.regime_kans[i](x))
            
        preds = torch.stack(preds, dim=1)  # (B, K)
        out = torch.einsum('bk,bk->b', r, preds)
        
        return out, r, logits

    def compute_regime_losses(self, r: torch.Tensor, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates internal losses driving the KASPER models apart structurally to prevent identical ensembles."""
        ortho_loss = torch.sum(r.T @ r) - torch.trace(r.T @ r)
        
        W = self.detector[-1].weight
        dist = torch.cdist(W, W)
        mask = 1.0 - torch.eye(self.num_regimes, device=W.device)
        contrastive_loss = (F.relu(margin - dist) * mask).sum() / max(1, (self.num_regimes * (self.num_regimes - 1)))
        
        return contrastive_loss, ortho_loss

    def get_regime_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.detector(x)
        return F.softmax(logits, dim=-1)


# ==============================================================================
# PIPELINE TRAINER 
# ==============================================================================
class ModelTrainer:
    """Wraps PyTorch models, tracking AFML-compliant testing distributions natively."""
    
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.config = config
        
        # Why: AdamW decouples weight limits directly improving sparse coefficients tracking.
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.get('lr', 1e-3), 
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Why: ReduceLROnPlateau prevents cyclical oscillations when the loss surface flattens explicitly natively smoothly.
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config.get('scheduler_factor', 0.7), 
            patience=config.get('scheduler_patience', 7)
        )
        
        self.lamb_1 = config.get('lamb_1', 1e-4)
        self.lamb_group = config.get('lamb_group', 1e-3)
        self.patience = config.get('patience', 20)
        self.margin = config.get('margin', 1.0)
        self.lambda_contrastive = config.get('lambda_contrastive', 0.1)
        self.lambda_ortho = config.get('lambda_ortho', 0.1)
        
        self.calibrator = None
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_bce': [],
            'val_accuracy': []
        }

    def _get_l1_loss(self) -> torch.Tensor:
        """Calculates standard L1 penalization natively across KAN layer coefficients."""
        loss = 0.0
        for m in self.model.modules():
            if isinstance(m, KANLayer):
                loss += torch.abs(m.coef).mean()
        if isinstance(loss, float):
            return torch.tensor(0.0, device=self.device)
        return loss
        
    def _get_group_lasso_loss(self) -> torch.Tensor:
        """
        Group Lasso penalty over spline coefficients.
        Why: Encourages the network to entirely drop uninformative basis functions
        (sparsity at the grid level) rather than just shrinking individual coefficients.
        """
        loss = 0.0
        for m in self.model.modules():
            if isinstance(m, KANLayer):
                # dim=-1 groups all grid points strictly for every explicit input-output connection mapping organically.
                loss += torch.norm(m.coef, p=2, dim=-1).mean()
        if isinstance(loss, float):
            return torch.tensor(0.0, device=self.device)
        return loss

    def fit_fold(self, X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, w_val: np.ndarray) -> 'ModelTrainer':
        epochs = self.config.get('steps', 200)
        batch_size = self.config.get('batch_size', 32)
        seq_len = self.config.get('seq_len', 1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        w_t = torch.tensor(w_train, dtype=torch.float32).to(self.device)
        
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        w_v = torch.tensor(w_val, dtype=torch.float32).to(self.device)

        # Why: Provides structural protection adapting the standard 2D feature matrix into 3D LSTM shapes for TKAN.
        if isinstance(self.model, TKAN):
            if X_t.shape[1] % seq_len != 0:
                raise ValueError("For TKAN, the explicit number of features must be strictly divisible by seq_len.")
            feat_dim = X_t.shape[1] // seq_len
            X_t = X_t.view(X_t.shape[0], seq_len, feat_dim)
            X_v = X_v.view(X_v.shape[0], seq_len, feat_dim)

        dataset = TensorDataset(X_t, y_t, w_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            train_loss_total = 0.0
            
            for bx, by, bw in loader:
                self.optimizer.zero_grad()
                
                loss_bce = 0.0
                L_contrastive = 0.0
                L_ortho = 0.0
                
                if isinstance(self.model, KASPER):
                    preds, r, logits = self.model(bx)
                    loss_bce = weighted_neg_log_loss(by, preds, bw)
                    L_contrastive, L_ortho = self.model.compute_regime_losses(r, self.margin)
                else:
                    preds = self.model(bx)
                    loss_bce = weighted_neg_log_loss(by, preds, bw)
                    
                loss = loss_bce + self.lambda_contrastive * L_contrastive + self.lambda_ortho * L_ortho
                loss = loss + self.lamb_1 * self._get_l1_loss()
                loss = loss + self.lamb_group * self._get_group_lasso_loss()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss_total += loss.item() * bx.size(0)

            avg_train_loss = train_loss_total / len(y_t)
            
            self.model.eval()
            with torch.no_grad():
                val_Lc, val_Lo = 0.0, 0.0
                if isinstance(self.model, KASPER):
                    val_preds, val_r, _ = self.model(X_v)
                    val_loss_bce = weighted_neg_log_loss(y_v, val_preds, w_v)
                    val_Lc, val_Lo = self.model.compute_regime_losses(val_r, self.margin)
                else:
                    val_preds = self.model(X_v)
                    val_loss_bce = weighted_neg_log_loss(y_v, val_preds, w_v)
                
                val_total_loss = val_loss_bce + self.lambda_contrastive * val_Lc + self.lambda_ortho * val_Lo
                val_total_loss = val_total_loss + self.lamb_1 * self._get_l1_loss() + self.lamb_group * self._get_group_lasso_loss()
                
                val_total_loss_float = val_total_loss.item()
                val_bce_float = val_loss_bce.item()
                val_accuracy = ((val_preds >= 0.5) == y_v).float().mean().item()
                
            self.metrics_history['train_loss'].append(avg_train_loss)
            self.metrics_history['val_loss'].append(val_total_loss_float) 
            self.metrics_history['val_bce'].append(val_bce_float)
            self.metrics_history['val_accuracy'].append(val_accuracy)
            
            self.scheduler.step(val_total_loss_float)
                
            if val_total_loss_float < best_val_loss:
                best_val_loss = val_total_loss_float
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                logger.info("Early stopping triggered at Epoch %d | Best Total Val Loss: %.4f", epoch, best_val_loss)
                break
                
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> 'ModelTrainer':
        """
        Fit Isotonic Regression to map raw native predictions to reliable empirical probabilities.
        Why: AFML Ch. 10 notes bet-sizing formulas demand well-calibrated metrics mirroring
        true real-world likelihood. Uncalibrated deep models generally return overconfident bounds.
        """
        raw_preds = self._predict_raw(X_val)
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        # Calibrating directly matching raw probabilities bounds targeting truth labels uniquely correctly.
        self.calibrator.fit(raw_preds, y_val)
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            seq_len = self.config.get('seq_len', 1)
            
            if isinstance(self.model, TKAN):
                feat_dim = X_t.shape[1] // seq_len
                X_t = X_t.view(X_t.shape[0], seq_len, feat_dim)
                
            if isinstance(self.model, KASPER):
                preds, _, _ = self.model(X_t)
            else:
                preds = self.model(X_t)
                
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self._predict_raw(X)
        
        if self.calibrator is not None:
            p1 = self.calibrator.transform(p1)
            
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

    def get_fold_metrics(self) -> dict:
        """Returns objective evaluation arrays natively spanning validation matrices seamlessly."""
        return self.metrics_history
