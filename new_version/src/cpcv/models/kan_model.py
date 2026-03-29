"""
8.5) KAN (Kolmogorov-Arnold Network)
=========================================
KAN classifier using efficient-kan with coarse-to-fine grid refinement.
Implements the spline-based architecture as an alternative to MLPs,
with L1 regularization on spline weights for interpretability.
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
KAN_GRID_SCHEDULE = [
    {"grid_size": 3,  "steps": 100, "lr": 1.0,  "optimizer": "LBFGS"},
    {"grid_size": 5,  "steps": 100, "lr": 1.0,  "optimizer": "LBFGS"},
    {"grid_size": 10, "steps": 80,  "lr": 1e-3, "optimizer": "Adam"},
    {"grid_size": 20, "steps": 50,  "lr": 1e-3, "optimizer": "Adam"},
]
KAN_SPLINE_ORDER = 3
KAN_GRID_RANGE = [-3.0, 3.0]
KAN_LAMB_L1 = 1.0
KAN_LAMB_ENTROPY = 2.0
KAN_BATCH_SIZE = 256
KAN_PATIENCE = 15


# =====================================================================
# KAN Model
# =====================================================================
class KANModel(BaseModel):
    """KAN classifier with coarse-to-fine grid refinement schedule."""

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.widths = [
            n_features,
            2 * n_features,
            max(n_features // 2, 4),
            n_classes,
        ]
        self.best_model = None
        self.best_grid_size = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # ── class weights ─────────────────────────────────────────────
        y_np = y_train.values if hasattr(y_train, "values") else y_train
        class_counts = np.bincount(y_np, minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * self.n_classes
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # ── dataloader ────────────────────────────────────────────────
        train_ds = TensorDataset(X_t, y_t, w_t)
        train_dl = DataLoader(train_ds, batch_size=KAN_BATCH_SIZE, shuffle=True)

        # ── grid schedule loop ────────────────────────────────────────
        best_val_loss = float("inf")
        best_state = None
        best_grid = None
        patience_counter = 0
        early_stopped = False

        for phase_idx, phase in enumerate(KAN_GRID_SCHEDULE):
            if early_stopped:
                break

            grid_size = phase["grid_size"]
            steps = phase["steps"]
            lr = phase["lr"]
            opt_name = phase["optimizer"]

            # fresh model at each grid level
            model = KAN(
                layers_hidden=self.widths,
                grid_size=grid_size,
                spline_order=KAN_SPLINE_ORDER,
            ).to(self.device)

            # optimizer
            if opt_name == "LBFGS":
                optimizer = torch.optim.LBFGS(
                    model.parameters(),
                    lr=lr,
                    max_iter=20,
                    line_search_fn="strong_wolfe",
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=lr, weight_decay=1e-5
                )

            criterion = nn.CrossEntropyLoss(weight=class_weights_t, reduction="none")

            phase_train_loss = 0.0
            phase_val_loss = None

            for step in range(steps):
                if early_stopped:
                    break

                model.train()

                if opt_name == "LBFGS":
                    # LBFGS uses full batch via closure
                    def closure():
                        optimizer.zero_grad()
                        logits = model(X_t)
                        per_sample = criterion(logits, y_t)
                        loss = (per_sample * w_t).mean()
                        loss = loss + _l1_regularization(model)
                        loss.backward()
                        return loss

                    loss_val = optimizer.step(closure)
                    phase_train_loss = loss_val.item()

                else:
                    # Adam: mini-batch
                    epoch_loss = 0.0
                    n_batches = 0
                    for X_b, y_b, w_b in train_dl:
                        optimizer.zero_grad()
                        logits = model(X_b)
                        per_sample = criterion(logits, y_b)
                        loss = (per_sample * w_b).mean()
                        loss = loss + _l1_regularization(model)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        n_batches += 1

                    phase_train_loss = epoch_loss / max(n_batches, 1)

                # ── validation check every 10 steps ───────────────────
                if has_val and (step + 1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_logits = model(X_val_t)
                        val_loss = nn.CrossEntropyLoss(weight=class_weights_t)(
                            val_logits, y_val_t
                        ).item()

                    phase_val_loss = val_loss

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(model.state_dict())
                        best_grid = grid_size
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= KAN_PATIENCE:
                        logger.info(
                            "KAN early stopping at grid=%d, step=%d (best val=%.4f).",
                            grid_size, step + 1, best_val_loss,
                        )
                        early_stopped = True

            # end of phase logging
            val_str = f", val_loss={phase_val_loss:.4f}" if phase_val_loss is not None else ""
            stopped_str = " [EARLY STOPPED]" if early_stopped else ""
            logger.info(
                "KAN phase %d: grid=%d, train_loss=%.4f%s%s",
                phase_idx, grid_size, phase_train_loss, val_str, stopped_str,
            )
            print(
                f"  [KAN] grid={grid_size:>2d}, steps={steps:>3d}, "
                f"train_loss={phase_train_loss:.4f}{val_str}{stopped_str}"
            )

            # if no val set, keep the last phase's model as best
            if not has_val:
                best_state = copy.deepcopy(model.state_dict())
                best_grid = grid_size

        # ── restore best model ────────────────────────────────────────
        if best_state is not None:
            self.best_grid_size = best_grid
            self.best_model = KAN(
                layers_hidden=self.widths,
                grid_size=best_grid,
                spline_order=KAN_SPLINE_ORDER,
            ).to(self.device)
            self.best_model.load_state_dict(best_state)
            self.best_model.eval()
        else:
            # fallback: use last model
            self.best_model = model
            self.best_model.eval()
            self.best_grid_size = grid_size

        logger.info(
            "KAN fitted: widths=%s, best_grid=%d, best_val_loss=%.4f, device=%s.",
            self.widths, self.best_grid_size, best_val_loss, self.device,
        )

    def predict_proba(self, X) -> np.ndarray:
        X_t = torch.tensor(
            X.values if hasattr(X, "values") else X,
            dtype=torch.float32,
        ).to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            logits = self.best_model(X_t)
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

        self.best_model.eval()
        with torch.no_grad():
            logits = self.best_model(X_t).cpu().numpy()

        return logits

    def get_name(self) -> str:
        return "KAN"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _l1_regularization(model: nn.Module) -> torch.Tensor:
    """L1 penalty on spline_weight parameters for sparsity."""
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if "spline_weight" in name:
            l1 = l1 + param.abs().mean()
    return KAN_LAMB_L1 * l1