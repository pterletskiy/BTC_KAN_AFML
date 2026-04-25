"""
8.4) LSTM
==============
LSTM classifier in PyTorch that consumes windowed sequences of features.
Wraps the PyTorch module behind the BaseModel interface for seamless
integration with the CPCV pipeline.

Training improvements:
  - Temporal attention: learned weighting over all timesteps instead of
    just the final hidden state, so early window information isn't lost
  - Tanh input normalization: squash features to [-1, 1]
  - LayerNorm on attended context vector
  - Gradient clipping (max_norm=1.0)
  - Label smoothing (0.1) for noisy financial labels
  - Cosine annealing with warm restarts LR schedule
"""

import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.cpcv.models.base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_WINDOW = 30            # lookback window (~1 crypto trading month)
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 100
LSTM_LR = 1e-3
LSTM_PATIENCE = 15          # early stopping patience
LSTM_LABEL_SMOOTHING = 0.1  # label smoothing for noisy financial labels
LSTM_GRAD_CLIP_NORM = 1.0   # max gradient norm for clipping

# Warm restarts: T_0=25 epochs per cycle, T_mult=2 doubles each cycle.
# T_0 is set below early-stopping patience (15) × 2 so at least one restart
# fires before early stopping could terminate training.
LSTM_WARMRESTART_T0 = 25
LSTM_WARMRESTART_TMULT = 2


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def create_sequences(
    X, y=None, w=None, window: int = LSTM_WINDOW
) -> tuple:
    """Reshape 2D features into 3D windowed sequences.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix, shape (T, n_features).
    y : np.ndarray or pd.Series, optional
        Labels. If None, prediction mode.
    w : np.ndarray or pd.Series, optional
        Sample weights.
    window : int
        Lookback window length.

    Returns
    -------
    Training mode (y provided):
        (X_seq, y_seq, w_seq, valid_indices)
    Prediction mode (y is None):
        (X_seq, valid_indices)

    X_seq has shape (T - window + 1, window, n_features).
    valid_indices maps each sequence to its original positional index.
    """
    X_arr = X.values if hasattr(X, "values") else X
    T, n_feat = X_arr.shape

    n_seq = T - window + 1
    if n_seq <= 0:
        raise ValueError(
            f"Input length {T} is shorter than window {window}. "
            "Cannot create sequences."
        )

    X_seq = np.zeros((n_seq, window, n_feat), dtype=np.float32)
    for i in range(n_seq):
        X_seq[i] = X_arr[i : i + window]

    valid_indices = np.arange(window - 1, T)

    if y is None:
        return X_seq, valid_indices

    # training mode: align y and w
    y_arr = y.values if hasattr(y, "values") else y
    y_seq = y_arr[valid_indices]

    w_seq = None
    if w is not None:
        w_arr = w.values if hasattr(w, "values") else w
        w_seq = w_arr[valid_indices]

    return X_seq, y_seq, w_seq, valid_indices


# ---------------------------------------------------------------------------
# PyTorch module
# ---------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """LSTM network with temporal attention, LayerNorm, and classification head.

    Instead of using only the final hidden state, a learned attention
    mechanism weights all timestep outputs. This preserves information
    from early lookback days that would otherwise be washed out through
    21 recurrent steps.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
    ):
        super().__init__()
        # Read module constants at call time (not definition time) so that
        # tuning overrides via `lstm_mod.LSTM_HIDDEN_SIZE = ...` are picked up.
        hidden_size = LSTM_HIDDEN_SIZE if hidden_size is None else hidden_size
        num_layers = LSTM_NUM_LAYERS if num_layers is None else num_layers
        dropout = LSTM_DROPOUT if dropout is None else dropout

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # temporal attention: learns which timesteps matter most
        self.attn_W = nn.Linear(hidden_size, 1, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal attention.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, window, n_features).

        Returns
        -------
        torch.Tensor
            Raw logits, shape (batch, n_classes). No softmax applied.
        """
        # lstm_out: (batch, window, hidden_size) — all timestep outputs
        lstm_out, _ = self.lstm(x)

        # attention: score each timestep and compute weighted context
        attn_scores = self.attn_W(lstm_out)                     # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)        # (B, T, 1)
        context = (attn_weights * lstm_out).sum(dim=1)           # (B, H)

        context = self.layer_norm(context)
        context = self.dropout(context)
        logits = self.fc(context)                                # (B, n_classes)
        return logits


# ---------------------------------------------------------------------------
# BaseModel wrapper
# ---------------------------------------------------------------------------
class LSTMModel(BaseModel):
    """LSTM classifier wrapped in the BaseModel interface."""

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.net = None
        self.window = LSTM_WINDOW
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_valid_indices = None
        # tanh normalization parameters (fitted on training data)
        self._input_mean = None
        self._input_std = None

    # ------------------------------------------------------------------
    # Input normalization (matching KAN)
    # ------------------------------------------------------------------
    def _fit_input_norm(self, X: np.ndarray) -> None:
        """Fit tanh normalization: z = tanh((x - mean) / (std + eps)).

        Maps features into [-1, 1], stabilizing LSTM training on
        fat-tailed financial data.
        """
        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0) + 1e-8

    def _apply_input_norm(self, X: np.ndarray) -> np.ndarray:
        """Apply tanh normalization using stored parameters."""
        if self._input_mean is None:
            return X
        z = (X - self._input_mean) / self._input_std
        return np.tanh(z)

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_val=None,
        y_val=None,
    ) -> None:
        # ── reproducibility ───────────────────────────────────────────
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # ── tanh input normalization ──────────────────────────────────
        X_train_arr = X_train.values if hasattr(X_train, "values") else X_train
        self._fit_input_norm(X_train_arr)
        X_train_normed = self._apply_input_norm(X_train_arr)

        # ── build sequences ───────────────────────────────────────────
        import pandas as pd
        X_train_df = pd.DataFrame(X_train_normed, index=getattr(X_train, 'index', None))
        X_seq, y_seq, w_seq, _ = create_sequences(
            X_train_df, y_train, sample_weight, window=self.window
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_arr = X_val.values if hasattr(X_val, "values") else X_val
            X_val_normed = self._apply_input_norm(X_val_arr)
            X_val_df = pd.DataFrame(X_val_normed, index=getattr(X_val, 'index', None))
            X_val_seq, y_val_seq, _, _ = create_sequences(
                X_val_df, y_val, window=self.window
            )

        # ── tensors ───────────────────────────────────────────────────
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.long).to(self.device)
        w_t = (
            torch.tensor(w_seq, dtype=torch.float32).to(self.device)
            if w_seq is not None
            else torch.ones(len(y_seq), dtype=torch.float32).to(self.device)
        )

        if has_val:
            X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val_seq, dtype=torch.long).to(self.device)

        # ── dataloader ────────────────────────────────────────────────
        train_ds = TensorDataset(X_t, y_t, w_t)
        train_dl = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True)

        # ── model ─────────────────────────────────────────────────────
        self.net = LSTMClassifier(
            n_features=self.n_features, n_classes=self.n_classes
        ).to(self.device)

        # ── class weights ─────────────────────────────────────────────
        class_counts = np.bincount(y_seq, minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * self.n_classes
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # ── loss, optimizer, scheduler ────────────────────────────────
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_t, reduction="none",
            label_smoothing=LSTM_LABEL_SMOOTHING,
        )
        criterion_val = nn.CrossEntropyLoss(
            weight=class_weights_t,
            label_smoothing=LSTM_LABEL_SMOOTHING,
        )
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=LSTM_LR, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=LSTM_WARMRESTART_T0,
            T_mult=LSTM_WARMRESTART_TMULT,
            eta_min=1e-5,
        )

        # ── training loop ─────────────────────────────────────────────
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(LSTM_EPOCHS):
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_b, y_b, w_b in train_dl:
                optimizer.zero_grad()
                logits = self.net(X_b)
                per_sample_loss = criterion(logits, y_b)
                weighted_loss = (per_sample_loss * w_b).mean()
                weighted_loss.backward()
                # gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), LSTM_GRAD_CLIP_NORM
                )
                optimizer.step()

                epoch_loss += weighted_loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)

            # ── validation ────────────────────────────────────────────
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    val_logits = self.net(X_val_t)
                    val_loss = criterion_val(val_logits, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= LSTM_PATIENCE:
                    logger.info(
                        "LSTM early stopping at epoch %d (best val loss: %.4f).",
                        epoch + 1, best_val_loss,
                    )
                    break

        # restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.net.eval()
        logger.info(
            "LSTM fitted: %d sequences, %d epochs completed, device=%s.",
            len(y_seq), epoch + 1, self.device,
        )

    def predict_proba(self, X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else X
        X_normed = self._apply_input_norm(X_arr)
        import pandas as pd
        X_df = pd.DataFrame(X_normed, index=getattr(X, 'index', None))
        X_seq, valid_indices = create_sequences(X_df, window=self.window)
        self.last_valid_indices = valid_indices

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_logits(self, X) -> np.ndarray:
        """Return raw pre-softmax logits."""
        X_arr = X.values if hasattr(X, "values") else X
        X_normed = self._apply_input_norm(X_arr)
        import pandas as pd
        X_df = pd.DataFrame(X_normed, index=getattr(X, 'index', None))
        X_seq, valid_indices = create_sequences(X_df, window=self.window)
        self.last_valid_indices = valid_indices

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(X_t).cpu().numpy()

        return logits

    def get_name(self) -> str:
        return "LSTM"