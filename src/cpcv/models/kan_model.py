"""
10.5) KAN (Kolmogorov-Arnold Network)
=========================================
KAN classifier using ``efficient-kan``: trains as a standard PyTorch nn.Module
with AdamW, producing well-scaled logits that calibrate without extreme
temperature correction.

Architecture: ``[n_features, KAN_HIDDEN, n_classes]`` — a narrow bottleneck
with B-spline activations. Inputs are tanh-normalised to ``[-1, 1]`` so they
match efficient-kan's default grid range.

Symbolic-formula extraction (the thesis's interpretability contribution) is
handled separately by ``symbolic_extraction.py`` using PyKAN with an
independently retrained model on the same architecture.
"""

import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn

from src.cpcv.models.base import BaseModel

logger = logging.getLogger(__name__)

# --- Module-level constants (shared with symbolic_extraction.py) -----------

# Architecture: single hidden layer by default (KAN_HIDDEN2 = 0 disables the second).
KAN_HIDDEN = 5
KAN_HIDDEN2 = 0
KAN_GRID = 5                      # B-spline grid size
KAN_K = 3                         # B-spline order (cubic)

# Training stack.
KAN_LR = 1e-3
KAN_WEIGHT_DECAY = 1e-4
KAN_EPOCHS = 200
KAN_PATIENCE = 20
KAN_VAL_INTERVAL = 1
KAN_LABEL_SMOOTHING = 0.1         # smoothing for noisy financial labels
KAN_GRAD_CLIP_NORM = 1.0

# Warm restarts: T_0=30 epochs per cycle sits above patience (20) so at least one
# restart fires before early stopping could terminate training.
KAN_WARMRESTART_T0 = 30
KAN_WARMRESTART_TMULT = 2


# --- 1. KAN Model (efficient-kan) ------------------------------------------
# Bridge between the CPCV pipeline (BaseModel contract) and the efficient-kan B-spline stack.
class KANModel(BaseModel):
    """KAN classifier built on efficient-kan; same B-spline basis as PyKAN, simpler training.

    Reliable AdamW training inside the CPCV loop is the goal here; symbolic
    extraction is delegated to ``symbolic_extraction.py``.
    """

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        # Single or two-layer architecture depending on the KAN_HIDDEN2 toggle.
        if KAN_HIDDEN2 > 0:
            self.widths = [n_features, KAN_HIDDEN, KAN_HIDDEN2, n_classes]
        else:
            self.widths = [n_features, KAN_HIDDEN, n_classes]
        self.kan_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Tanh-normalisation parameters fitted on training data only.
        self._input_mean = None
        self._input_std = None

    # Fit tanh normalisation on the training tensor; maps features into [-1, 1] to match grid range.
    def _fit_input_norm(self, X_t: torch.Tensor) -> None:
        """Fit ``z = tanh((x - mean) / (std + eps))`` on the training tensor."""
        self._input_mean = X_t.mean(dim=0)
        self._input_std = X_t.std(dim=0) + 1e-8

    # Apply the stored tanh normalisation (identity if not yet fitted).
    def _apply_input_norm(self, X_t: torch.Tensor) -> torch.Tensor:
        """Apply tanh normalisation using stored mean/std."""
        if self._input_mean is None:
            return X_t
        z = (X_t - self._input_mean.to(X_t.device)) / self._input_std.to(X_t.device)
        return torch.tanh(z)

    # Full training routine: normalise → fit → log final metrics.
    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_val=None,
        y_val=None,
        sample_weight_val=None,
    ) -> None:

        from efficient_kan import KAN

        # Seed every RNG so the same data + hyperparameters produce the same fitted weights.
        # The cuDNN flags force deterministic kernels for any operation that has both
        # a fast non-deterministic and a slower deterministic implementation.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Move features, labels, and (optional) sample weights onto the selected device.
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
            if sample_weight_val is not None:
                w_val_t = torch.tensor(
                    sample_weight_val.values
                    if hasattr(sample_weight_val, "values")
                    else sample_weight_val,
                    dtype=torch.float32,
                ).to(self.device)
            else:
                w_val_t = torch.ones(len(y_val_t), dtype=torch.float32).to(self.device)

        # Tanh normalisation: fit on train, apply to train and (if present) validation.
        self._fit_input_norm(X_t)
        X_t = self._apply_input_norm(X_t)
        if has_val:
            X_val_t = self._apply_input_norm(X_val_t)

        # Class-frequency-inverse weighting handles label imbalance inside the loss.
        y_np = y_train.values if hasattr(y_train, "values") else y_train
        class_counts = np.bincount(y_np, minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * self.n_classes
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # Instantiate the efficient-kan stack with the configured grid and order.
        model = KAN(
            layers_hidden=self.widths,
            grid_size=KAN_GRID,
            spline_order=KAN_K,
            grid_range=[-1, 1],
        ).to(self.device)

        # AdamW + cosine warm restarts: same stack used for LSTM, just longer cycles.
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=KAN_LR, weight_decay=KAN_WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=KAN_WARMRESTART_T0,
            T_mult=KAN_WARMRESTART_TMULT,
            eta_min=1e-5,
        )
        # Per-sample reduction on BOTH train and val so AFML sample weights multiply
        # correctly into each term, keeping the early-stopping criterion weighted on
        # the same basis as the training loss (AFML Snippet 8.3 symmetry).
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_t, reduction="none",
            label_smoothing=KAN_LABEL_SMOOTHING,
        )
        criterion_val = nn.CrossEntropyLoss(
            weight=class_weights_t, reduction="none",
            label_smoothing=KAN_LABEL_SMOOTHING,
        )

        # Full-batch training: per-epoch convergence is logged at DEBUG;
        # the outer pipeline tqdm bar (pipeline.py) drives notebook-level progress.
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(KAN_EPOCHS):
            model.train()
            optimizer.zero_grad()

            # Forward pass, sample-weighted loss, backward, gradient clip, step.
            logits = model(X_t)
            per_sample = criterion(logits, y_t)
            loss = (per_sample * w_t).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), KAN_GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            train_loss_val = float(loss.item())

            # Validation pass: track best weights and the patience counter for early stopping.
            if has_val and (epoch + 1) % KAN_VAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    val_per_sample = criterion_val(model(X_val_t), y_val_t)
                    val_loss = (val_per_sample * w_val_t).mean().item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                logger.debug(
                    "KAN epoch %d/%d: train=%.4f val=%.4f best=%.4f patience=%d/%d",
                    epoch + 1, KAN_EPOCHS, train_loss_val, val_loss,
                    best_val_loss, patience_counter, KAN_PATIENCE,
                )

                if patience_counter >= KAN_PATIENCE:
                    logger.debug(
                        "Early stopping at epoch %d (best val=%.4f).",
                        epoch + 1, best_val_loss,
                    )
                    break
            else:
                logger.debug(
                    "KAN epoch %d/%d: train=%.4f (no validation)",
                    epoch + 1, KAN_EPOCHS, train_loss_val,
                )

        # Restore the best validation weights (no-op if no validation was supplied).
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        self.kan_model = model

        # Final-fit summary at INFO; per-epoch diagnostics stay at DEBUG to keep notebook output small.
        with torch.no_grad():
            test_input = X_val_t if has_val else X_t
            test_label = y_val_t if has_val else y_t
            pred = self.kan_model(test_input)
            val_acc = (pred.argmax(dim=1) == test_label).float().mean().item()
            logit_range = (pred.min().item(), pred.max().item())

        final_epoch = epoch + 1
        val_loss_str = f"{best_val_loss:.4f}" if has_val else "n/a"
        logger.info(
            "KAN fitted: widths=%s, grid=%d, epochs=%d, "
            "val_acc=%.4f, val_loss=%s, logit_range=[%.2f, %.2f], device=%s.",
            self.widths, KAN_GRID, final_epoch, val_acc, val_loss_str,
            logit_range[0], logit_range[1], self.device,
        )

    # Class probabilities at inference, with the stored tanh normalisation applied.
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

    # Hard-label prediction via argmax.
    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    # Raw pre-softmax logits for downstream calibration.
    def predict_logits(self, X) -> np.ndarray:
        """Return raw pre-softmax logits, shape ``(n_samples, n_classes)``."""
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