"""
Models — KAN (Kolmogorov-Arnold Network)
=========================================
KAN classifier using PyKAN, following the VIX KAN paper's Algorithm 1:
  1. Train with L1 + entropy regularization (LBFGS)
  2. (downstream) Prune low-importance edges and nodes
  3. (downstream) Symbolify activation functions
  4. (downstream) Fine-tune affine parameters

Uses PyKAN for both prediction and symbolic extraction, eliminating the
need for a separate library. The trained model object is stored as
``self.kan_model`` and can be passed directly to symbolic_extraction.py.

Architecture follows the VIX paper: [n_features, 2*n_features, n_classes],
deliberately kept small since pruning will remove unnecessary complexity.
"""

import copy
import logging
import random

import numpy as np
import torch
import torch.nn as nn

from kan import KAN

from src.cpcv.models.base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (aligned with VIX KAN paper)
# ---------------------------------------------------------------------------
KAN_GRID = 3                      # grid size (small, pruning will simplify further)
KAN_K = 3                         # B-spline order (cubic)
KAN_LR = 0.04                     # LBFGS learning rate (from VIX paper)
KAN_TRAIN_STEPS = 200              # maximum training steps
KAN_LAMB = 0.01                    # overall regularization strength
KAN_LAMB_L1 = 1.0                  # L1 on activation magnitudes (sparsity)
KAN_LAMB_ENTROPY = 2.0             # entropy regularization (binary on/off)
KAN_PATIENCE = 15                  # early stopping patience
KAN_LR_DECAY_PATIENCE = 5         # reduce LR after this many stale evals
KAN_LR_DECAY_FACTOR = 0.1         # LR multiplier on decay
KAN_VAL_INTERVAL = 10              # validate every N steps


# =====================================================================
# KAN Model (PyKAN)
# =====================================================================
class KANModel(BaseModel):
    """KAN classifier using PyKAN with L1 + entropy regularization.

    The trained PyKAN model is stored as ``self.kan_model`` and can be
    passed directly to ``symbolic_extraction.py`` for pruning,
    symbolification, and formula extraction.
    """

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.widths = [n_features, 2 * n_features, n_classes]
        self.kan_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # ── instantiate PyKAN ─────────────────────────────────────────
        model = KAN(
            width=self.widths,
            grid=KAN_GRID,
            k=KAN_K,
            seed=self.seed,
        ).to(self.device)

        # ── prepare dataset dict for PyKAN ────────────────────────────
        dataset = {
            "train_input": X_t,
            "train_label": y_t.float(),
            "test_input": X_val_t if has_val else X_t,
            "test_label": y_val_t.float() if has_val else y_t.float(),
        }

        # ── attempt PyKAN's native fit() ──────────────────────────────
        trained_via_fit = False
        try:
            model.fit(
                dataset,
                opt="LBFGS",
                lr=KAN_LR,
                steps=KAN_TRAIN_STEPS,
                lamb=KAN_LAMB,
                lamb_l1=KAN_LAMB_L1,
                lamb_entropy=KAN_LAMB_ENTROPY,
                loss_fn=nn.CrossEntropyLoss(weight=class_weights_t),
            )
            trained_via_fit = True
            logger.info("PyKAN trained via model.fit(): %d steps.", KAN_TRAIN_STEPS)
        except (TypeError, AttributeError, Exception) as e:
            logger.info("PyKAN model.fit() failed (%s). Using custom loop.", e)

        # ── custom training loop (fallback) ───────────────────────────
        if not trained_via_fit:
            model = self._custom_train(
                model, X_t, y_t, w_t, class_weights_t,
                X_val_t if has_val else None,
                y_val_t if has_val else None,
            )

        model.eval()
        self.kan_model = model

        # ── store dataset reference for symbolic extraction ───────────
        self._dataset = dataset

        # ── log validation accuracy ───────────────────────────────────
        with torch.no_grad():
            test_input = X_val_t if has_val else X_t
            test_label = y_val_t if has_val else y_t
            pred = model(test_input)
            val_acc = (pred.argmax(dim=1) == test_label).float().mean().item()

        logger.info(
            "PyKAN fitted: widths=%s, grid=%d, val_acc=%.4f, device=%s.",
            self.widths, KAN_GRID, val_acc, self.device,
        )
        print(
            f"  [KAN] widths={self.widths}, grid={KAN_GRID}, "
            f"val_acc={val_acc:.4f}"
        )

    def _custom_train(
        self, model, X_t, y_t, w_t, class_weights_t,
        X_val_t=None, y_val_t=None,
    ):
        """Custom LBFGS loop when PyKAN's model.fit() doesn't support CrossEntropyLoss."""
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=KAN_LR, max_iter=20,
            line_search_fn="strong_wolfe",
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, reduction="none")

        has_val = X_val_t is not None
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        lr_patience_counter = 0
        current_lr = KAN_LR
        final_step = 0

        for step in range(KAN_TRAIN_STEPS):
            final_step = step
            model.train()

            def closure():
                optimizer.zero_grad()
                logits = model(X_t)
                per_sample = criterion(logits, y_t)
                loss = (per_sample * w_t).mean()
                # L1 + entropy regularization via PyKAN API
                try:
                    reg_l1 = model.regularization_loss(
                        regularize_activation=1.0, regularize_entropy=0.0
                    )
                    reg_ent = model.regularization_loss(
                        regularize_activation=0.0, regularize_entropy=1.0
                    )
                    loss = loss + KAN_LAMB * (
                        KAN_LAMB_L1 * reg_l1 + KAN_LAMB_ENTROPY * reg_ent
                    )
                except (AttributeError, TypeError):
                    # fallback: manual L1 on spline parameters
                    l1 = sum(
                        p.abs().mean() for n, p in model.named_parameters()
                        if "coef" in n or "spline" in n
                    )
                    loss = loss + KAN_LAMB * KAN_LAMB_L1 * l1
                loss.backward()
                return loss

            optimizer.step(closure)

            # ── validation ────────────────────────────────────────────
            if has_val and (step + 1) % KAN_VAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val_t)
                    val_loss = nn.CrossEntropyLoss(weight=class_weights_t)(
                        val_logits, y_val_t
                    ).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    lr_patience_counter = 0
                else:
                    patience_counter += 1
                    lr_patience_counter += 1

                # LR decay
                if lr_patience_counter >= KAN_LR_DECAY_PATIENCE:
                    current_lr *= KAN_LR_DECAY_FACTOR
                    for pg in optimizer.param_groups:
                        pg["lr"] = current_lr
                    lr_patience_counter = 0
                    logger.info("KAN step %d: LR → %.6f", step + 1, current_lr)

                # early stopping
                if patience_counter >= KAN_PATIENCE:
                    logger.info(
                        "KAN early stopping at step %d (best val=%.4f).",
                        step + 1, best_val_loss,
                    )
                    break

        # restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        logger.info(
            "Custom PyKAN: %d steps, best val loss: %.4f",
            final_step + 1, best_val_loss,
        )
        return model

    def predict_proba(self, X) -> np.ndarray:
        X_t = torch.tensor(
            X.values if hasattr(X, "values") else X,
            dtype=torch.float32,
        ).to(self.device)

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

        self.kan_model.eval()
        with torch.no_grad():
            logits = self.kan_model(X_t).cpu().numpy()

        return logits

    def get_name(self) -> str:
        return "KAN"