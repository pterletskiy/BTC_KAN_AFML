"""
8.5) KAN (Kolmogorov-Arnold Network)
=========================================
KAN classifier using PyKAN, following the VIX KAN paper's Algorithm 1:
  1. Train with L1 + entropy regularization
  2. (downstream) Prune low-importance edges and nodes
  3. (downstream) Symbolify activation functions
  4. (downstream) Fine-tune affine parameters

Uses PyKAN for both prediction and symbolic extraction. The trained model
object is stored as ``self.kan_model`` and can be passed directly to
symbolic_extraction.py.

Architecture: [n_features, HIDDEN, n_classes] — a narrow bottleneck that
forces compressed representations amenable to symbolic extraction.
Inputs are tanh-normalized to the B-spline active range.

Training is staged: Adam for exploration, then LBFGS for refinement.
Regularization ramps from zero (Adam phase) to full strength (LBFGS phase).
Grid is extended from 3→5 after initial convergence.
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
# Module-level constants
# ---------------------------------------------------------------------------
# Architecture
KAN_HIDDEN = 5                     # narrow bottleneck (was 2*n_features ≈ 40)
KAN_GRID = 3                      # initial grid size (refined to 5 later)
KAN_GRID_REFINE = 5               # grid after extension
KAN_K = 3                         # B-spline order (cubic)

# Training — Phase 1 (Adam: explore)
ADAM_STEPS = 100                   # Adam exploration phase
ADAM_LR = 1e-3                     # standard Adam learning rate
ADAM_LAMB = 0.0                    # NO regularization during exploration

# Training — Phase 2 (LBFGS: refine)
LBFGS_STEPS = 150                  # LBFGS refinement phase
LBFGS_LR = 0.02                   # LBFGS learning rate
KAN_LAMB = 0.005                   # regularization (lower than old 0.01)
KAN_LAMB_L1 = 1.0                  # L1 on activation magnitudes (sparsity)
KAN_LAMB_ENTROPY = 2.0             # entropy regularization (binary on/off)

# Early stopping & scheduling (applied during LBFGS phase)
KAN_PATIENCE = 15                  # early stopping patience (in val checks)
KAN_LR_DECAY_PATIENCE = 5         # reduce LR after this many stale evals
KAN_LR_DECAY_FACTOR = 0.1         # LR multiplier on decay
KAN_VAL_INTERVAL = 5              # validate every N steps (more frequent)


# =====================================================================
# KAN Model (PyKAN)
# =====================================================================
class KANModel(BaseModel):
    """KAN classifier using PyKAN with staged training and grid extension.

    Key improvements over the baseline:
      1. Narrow bottleneck: [n_features, 5, 2] instead of [n_features, 40, 2]
      2. Staged optimizer: Adam (explore) → LBFGS (refine)
      3. Regularization ramp: zero during Adam, full during LBFGS
      4. Grid extension: 3 → 5 after Adam convergence
      5. Tanh input normalization to B-spline active range

    The trained PyKAN model is stored as ``self.kan_model`` and can be
    passed directly to ``symbolic_extraction.py`` for pruning,
    symbolification, and formula extraction.
    """

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
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

        This maps features into roughly [-1, 1], ensuring all inputs land
        within the B-spline grid's active range.
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

        from kan import KAN

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

        # ── Step 5: tanh input normalization ──────────────────────────
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

        # ── Step 1: narrow architecture ───────────────────────────────
        model = KAN(
            width=self.widths,
            grid=KAN_GRID,
            k=KAN_K,
            seed=self.seed,
        ).to(self.device)

        # ── prepare dataset dict for PyKAN (grid extension needs it) ──
        dataset = {
            "train_input": X_t,
            "train_label": y_t.float(),
            "test_input": X_val_t if has_val else X_t,
            "test_label": y_val_t.float() if has_val else y_t.float(),
        }

        # ── Steps 2+3: staged training ───────────────────────────────
        model = self._staged_train(
            model, X_t, y_t, w_t, class_weights_t,
            X_val_t if has_val else None,
            y_val_t if has_val else None,
            dataset,
        )

        model.eval()
        self.kan_model = model
        self._dataset = dataset

        # ── log validation accuracy ───────────────────────────────────
        with torch.no_grad():
            test_input = X_val_t if has_val else X_t
            test_label = y_val_t if has_val else y_t
            pred = model(test_input)
            val_acc = (pred.argmax(dim=1) == test_label).float().mean().item()

        logger.info(
            "PyKAN fitted: widths=%s, grid=%d→%d, val_acc=%.4f, device=%s.",
            self.widths, KAN_GRID, KAN_GRID_REFINE, val_acc, self.device,
        )
        print(
            f"  [KAN] widths={self.widths}, grid={KAN_GRID}→{KAN_GRID_REFINE}, "
            f"val_acc={val_acc:.4f}"
        )

    # ------------------------------------------------------------------
    # Staged training: Adam → grid extend → LBFGS
    # ------------------------------------------------------------------
    def _staged_train(
        self, model, X_t, y_t, w_t, class_weights_t,
        X_val_t, y_val_t, dataset,
    ):
        """Two-phase training with grid extension between phases.

        Phase 1 (Adam): explore loss landscape with no regularization.
        Grid extension: refine B-spline resolution from 3→5.
        Phase 2 (LBFGS): polish with second-order optimization + regularization.
        """
        has_val = X_val_t is not None
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, reduction="none")
        criterion_val = nn.CrossEntropyLoss(weight=class_weights_t)

        # ── Phase 1: Adam (explore, no regularization) ────────────────
        logger.info("Phase 1: Adam (%d steps, lr=%.4f, lamb=%.4f)", ADAM_STEPS, ADAM_LR, ADAM_LAMB)
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=ADAM_LR)

        best_val_loss = float("inf")
        best_state = None

        for step in range(ADAM_STEPS):
            model.train()
            optimizer_adam.zero_grad()

            logits = model(X_t)
            per_sample = criterion(logits, y_t)
            loss = (per_sample * w_t).mean()

            # minimal regularization (can be zero)
            if ADAM_LAMB > 0:
                loss = loss + self._compute_reg(model, ADAM_LAMB)

            loss.backward()
            optimizer_adam.step()

            # validation check
            if has_val and (step + 1) % KAN_VAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion_val(model(X_val_t), y_val_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())

        # restore best Adam state
        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info("Phase 1 complete. Best val loss: %.4f", best_val_loss)

        # ── Step 4: grid extension (3 → 5) ───────────────────────────
        try:
            # PyKAN's refine() extends the grid without reinitializing
            model = model.refine(KAN_GRID_REFINE)
            model = model.to(self.device)
            logger.info("Grid extended: %d → %d", KAN_GRID, KAN_GRID_REFINE)
        except (AttributeError, TypeError, Exception) as e:
            logger.warning("Grid extension failed (%s). Continuing with grid=%d.", e, KAN_GRID)

        # ── Phase 2: LBFGS (refine with regularization) ──────────────
        logger.info(
            "Phase 2: LBFGS (%d steps, lr=%.4f, lamb=%.4f)",
            LBFGS_STEPS, LBFGS_LR, KAN_LAMB,
        )
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=LBFGS_LR, max_iter=20,
            line_search_fn="strong_wolfe",
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        lr_patience_counter = 0
        current_lr = LBFGS_LR
        final_step = 0

        for step in range(LBFGS_STEPS):
            final_step = step
            model.train()

            def closure():
                optimizer_lbfgs.zero_grad()
                logits = model(X_t)
                per_sample = criterion(logits, y_t)
                loss = (per_sample * w_t).mean()
                # full regularization in LBFGS phase
                loss = loss + self._compute_reg(model, KAN_LAMB)
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)

            # validation
            if has_val and (step + 1) % KAN_VAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion_val(model(X_val_t), y_val_t).item()

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
                    for pg in optimizer_lbfgs.param_groups:
                        pg["lr"] = current_lr
                    lr_patience_counter = 0
                    logger.info("LBFGS step %d: LR → %.6f", step + 1, current_lr)

                # early stopping
                if patience_counter >= KAN_PATIENCE:
                    logger.info(
                        "LBFGS early stopping at step %d (best val=%.4f).",
                        step + 1, best_val_loss,
                    )
                    break

        # restore best LBFGS state
        if best_state is not None:
            model.load_state_dict(best_state)

        logger.info(
            "Phase 2 complete: %d steps, best val loss: %.4f",
            final_step + 1, best_val_loss,
        )
        return model

    # ------------------------------------------------------------------
    # Regularization helper
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_reg(model, lamb: float) -> torch.Tensor:
        """Compute L1 + entropy regularization via PyKAN API with fallback."""
        if lamb <= 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        try:
            reg_l1 = model.regularization_loss(
                regularize_activation=1.0, regularize_entropy=0.0
            )
            reg_ent = model.regularization_loss(
                regularize_activation=0.0, regularize_entropy=1.0
            )
            return lamb * (KAN_LAMB_L1 * reg_l1 + KAN_LAMB_ENTROPY * reg_ent)
        except (AttributeError, TypeError):
            # fallback: manual L1 on spline parameters
            l1 = sum(
                p.abs().mean() for n, p in model.named_parameters()
                if "coef" in n or "spline" in n
            )
            return lamb * KAN_LAMB_L1 * l1

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