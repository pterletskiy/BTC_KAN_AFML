"""
8.5) KAN (Kolmogorov-Arnold Network)
=========================================
KAN classifier using efficient-kan for CPCV prediction. Trains as a
standard PyTorch nn.Module with AdamW, producing well-scaled logits
that calibrate without extreme temperature correction.

Symbolic extraction is handled separately by symbolic_extraction.py
using PyKAN (same architecture, independent retraining).

Architecture: [n_features, HIDDEN, n_classes] — a narrow bottleneck
with B-spline activations. Inputs are tanh-normalized to [-1, 1]
to match the spline grid range.
"""

import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn

from src.cpcv.models.base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (shared with symbolic_extraction.py)
# ---------------------------------------------------------------------------
# Architecture
KAN_HIDDEN = 5                     # 1st hidden layer width
KAN_HIDDEN2 = 0                    # 2nd hidden layer (0 = single hidden layer)
KAN_GRID = 5                      # grid size
KAN_K = 3                         # B-spline order (cubic)

# Training
KAN_LR = 1e-3                     # AdamW learning rate
KAN_WEIGHT_DECAY = 1e-4           # L2 regularization
KAN_EPOCHS = 200                   # maximum training epochs
KAN_PATIENCE = 20                  # early stopping patience
KAN_VAL_INTERVAL = 1              # validate every epoch
KAN_LABEL_SMOOTHING = 0.1         # label smoothing for noisy financial labels
KAN_GRAD_CLIP_NORM = 1.0          # max gradient norm for clipping
KAN_ENTROPY_REG = 0.01            # entropy penalty weight (encourages decisive predictions)
KAN_SWA_START_FRAC = 0.6          # start SWA after 60% of epochs
KAN_SWA_LR = 1e-4                 # SWA learning rate

# Warm restarts: T_0=60 epochs per cycle, T_mult=2 doubles each cycle
KAN_WARMRESTART_T0 = 60
KAN_WARMRESTART_TMULT = 2


# =====================================================================
# KAN Model (efficient-kan)
# =====================================================================
class KANModel(BaseModel):
    """KAN classifier using efficient-kan for CPCV prediction.

    Uses the same B-spline basis as PyKAN but wrapped in a standard
    nn.Module that trains reliably with AdamW. Symbolic extraction
    is handled downstream by symbolic_extraction.py using PyKAN.
    """

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        if KAN_HIDDEN2 > 0:
            self.widths = [n_features, KAN_HIDDEN, KAN_HIDDEN2, n_classes]
        else:
            self.widths = [n_features, KAN_HIDDEN, n_classes]
        self.kan_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # tanh normalization parameters (fitted on training data)
        self._input_mean = None
        self._input_std = None

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    def _fit_input_norm(self, X_t: torch.Tensor) -> None:
        """Fit tanh normalization: z = tanh((x - mean) / (std + eps)).

        Maps features into [-1, 1], matching efficient-kan's default
        grid_range=[-1, 1].
        """
        self._input_mean = X_t.mean(dim=0)
        self._input_std = X_t.std(dim=0) + 1e-8

    def _apply_input_norm(self, X_t: torch.Tensor) -> torch.Tensor:
        """Apply tanh normalization using stored parameters."""
        if self._input_mean is None:
            return X_t
        z = (X_t - self._input_mean.to(X_t.device)) / self._input_std.to(X_t.device)
        return torch.tanh(z)

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_val=None,
        y_val=None,
    ) -> None:

        from efficient_kan import KAN

        # ── reproducibility ───────────────────────────────────────────
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # ── tensors ───────────────────────────────────────────────────
        X_t = torch.tensor(
            X_train.values if hasattr(X_train, "values") else X_train,
            dtype=torch.float32,
        ).to(self.device)
        y_t = torch.tensor(
            y_train.values if hasattr(y_train, "values") else y_train,
            dtype=torch.long,
        ).to(self.device)

        if sample_weight is not None:
            w_t = torch.tensor(
                sample_weight.values if hasattr(sample_weight, "values") else sample_weight,
                dtype=torch.float32,
            ).to(self.device)
        else:
            w_t = torch.ones(len(y_t), dtype=torch.float32).to(self.device)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(
                X_val.values if hasattr(X_val, "values") else X_val,
                dtype=torch.float32,
            ).to(self.device)
            y_val_t = torch.tensor(
                y_val.values if hasattr(y_val, "values") else y_val,
                dtype=torch.long,
            ).to(self.device)

        # ── tanh input normalization ──────────────────────────────────
        self._fit_input_norm(X_t)
        X_t = self._apply_input_norm(X_t)
        if has_val:
            X_val_t = self._apply_input_norm(X_val_t)

        # ── class weights ─────────────────────────────────────────────
        y_np = y_train.values if hasattr(y_train, "values") else y_train
        class_counts = np.bincount(y_np, minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * self.n_classes
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # ── build model ───────────────────────────────────────────────
        model = KAN(
            layers_hidden=self.widths,
            grid_size=KAN_GRID,
            spline_order=KAN_K,
            grid_range=[-1, 1],
        ).to(self.device)

        # ── optimizer, scheduler, and loss ─────────────────────────────
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=KAN_LR, weight_decay=KAN_WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=KAN_WARMRESTART_T0,
            T_mult=KAN_WARMRESTART_TMULT,
            eta_min=1e-5,
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_t, reduction="none",
            label_smoothing=KAN_LABEL_SMOOTHING,
        )
        criterion_val = nn.CrossEntropyLoss(
            weight=class_weights_t, label_smoothing=KAN_LABEL_SMOOTHING,
        )

        # ── SWA: stochastic weight averaging ──────────────────────────
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_start_epoch = int(KAN_SWA_START_FRAC * KAN_EPOCHS)
        swa_active = False

        # ── training loop ─────────────────────────────────────────────
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(KAN_EPOCHS):
            model.train()
            optimizer.zero_grad()

            logits = model(X_t)
            per_sample = criterion(logits, y_t)
            ce_loss = (per_sample * w_t).mean()

            # entropy regularization: penalize uncertain predictions
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            loss = ce_loss + KAN_ENTROPY_REG * entropy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), KAN_GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            # SWA: collect weight snapshots after swa_start_epoch
            if epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_active = True

            # validation
            if has_val and (epoch + 1) % KAN_VAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion_val(model(X_val_t), y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= KAN_PATIENCE:
                    logger.info(
                        "Early stopping at epoch %d (best val=%.4f).",
                        epoch + 1, best_val_loss,
                    )
                    break

        # restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # SWA: use averaged model if it was active
        if swa_active:
            # SWA model needs BN update — KAN has no BN, so just use it directly
            swa_model.eval()
            self.kan_model = swa_model
            logger.info("KAN using SWA-averaged weights (started at epoch %d).", swa_start_epoch)
        else:
            model.eval()
            self.kan_model = model

        # ── store dataset reference for downstream use ────────────────
        self._dataset = {
            "train_input": X_t,
            "train_label": y_t.float(),
            "test_input": X_val_t if has_val else X_t,
            "test_label": y_val_t.float() if has_val else y_t.float(),
        }

        # ── log results ───────────────────────────────────────────────
        with torch.no_grad():
            test_input = X_val_t if has_val else X_t
            test_label = y_val_t if has_val else y_t
            pred = self.kan_model(test_input)
            val_acc = (pred.argmax(dim=1) == test_label).float().mean().item()
            logit_range = (pred.min().item(), pred.max().item())

        final_epoch = epoch + 1
        swa_tag = " [SWA]" if swa_active else ""
        logger.info(
            "efficient-KAN%s fitted: widths=%s, grid=%d, epochs=%d, "
            "val_acc=%.4f, val_loss=%.4f, logit_range=[%.2f, %.2f], device=%s.",
            swa_tag, self.widths, KAN_GRID, final_epoch, val_acc, best_val_loss,
            logit_range[0], logit_range[1], self.device,
        )
        print(
            f"  [KAN{swa_tag}] widths={self.widths}, grid={KAN_GRID}, "
            f"epochs={final_epoch}, val_acc={val_acc:.4f}, "
            f"val_loss={best_val_loss:.4f}"
        )

    # ------------------------------------------------------------------
    # Inference (all methods apply tanh normalization)
    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        X_t = torch.tensor(
            X.values if hasattr(X, "values") else X,
            dtype=torch.float32,
        ).to(self.device)
        X_t = self._apply_input_norm(X_t)

        self.kan_model.eval()
        with torch.no_grad():
            logits = self.kan_model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_logits(self, X) -> np.ndarray:
        """Return raw pre-softmax logits."""
        X_t = torch.tensor(
            X.values if hasattr(X, "values") else X,
            dtype=torch.float32,
        ).to(self.device)
        X_t = self._apply_input_norm(X_t)

        self.kan_model.eval()
        with torch.no_grad():
            logits = self.kan_model(X_t).cpu().numpy()
        return logits

    def get_name(self) -> str:
        return "KAN"