"""
12) Symbolic Extraction
================================
Take the best CPCV fold (or full dataset), retrain a fresh PyKAN model
with the same architecture as the efficient-kan used in CPCV, then
apply Algorithm 1 from the VIX KAN paper:
  1. Train with staged optimizer (Adam → grid extend → LBFGS)
  2. Prune low-importance edges and nodes
  3. Symbolify activation functions with closed-form candidates
  4. Fine-tune affine parameters

The CPCV pipeline uses efficient-kan (standard nn.Module, AdamW).
This module uses PyKAN independently because only PyKAN supports
prune(), suggest_symbolic(), fix_symbolic(), and symbolic_formula().
Both share the same [n_features, HIDDEN, n_classes] architecture
and B-spline basis.
"""

import copy
import logging
import os

import numpy as np
import pandas as pd
import sympy
import torch
import torch.nn as nn

from src.cpcv.cv import generate_cpcv_splits
from src.cpcv.preprocessing import apply_ffd

# shared architecture constants (same for efficient-kan and PyKAN)
from src.cpcv.models.kan_model import KAN_HIDDEN, KAN_GRID, KAN_K

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyKAN-specific training constants (not used by efficient-kan)
# ---------------------------------------------------------------------------
# Phase 1: Adam with weight decay (no KAN-specific regularization)
PYKAN_ADAM_STEPS = 600             # longer Adam phase — main learning happens here
PYKAN_ADAM_LR = 1e-3
PYKAN_ADAM_WEIGHT_DECAY = 1e-3     # L2 penalty to prevent memorization
PYKAN_NOISE_STD = 0.05            # Gaussian noise injected into inputs each step
                                   # (acts as dropout-like regularizer for small data)

# Phase 2: LBFGS (much shorter — only light refinement, not memorization)
PYKAN_LBFGS_STEPS = 40            # reduced from 150 — LBFGS memorizes small datasets
PYKAN_LBFGS_LR = 0.01             # reduced from 0.02 — less aggressive steps
PYKAN_LBFGS_WARMUP_FRAC = 0.5     # first 50% of LBFGS steps: no regularization
PYKAN_LAMB = 0.002                 # reduced from 0.005 — gentler sparsity
PYKAN_LAMB_L1 = 1.0
PYKAN_LAMB_ENTROPY = 2.0

PYKAN_PATIENCE = 10                # reduced — stop faster if overfitting
PYKAN_VAL_INTERVAL = 5
PYKAN_GRID_INIT = 3               # start coarse
PYKAN_GRID_EXTEND = False          # DISABLED — with 358 samples, grid extension
                                   # adds parameters and causes memorization.
                                   # Only enable for datasets > 1000 samples.

# Data-aware architecture: override KAN_HIDDEN when data is small
PYKAN_MIN_SAMPLES_PER_PARAM = 5    # want at least 5 samples per parameter
PYKAN_HIDDEN_OVERRIDE = None       # set dynamically in train_pykan()

# Minimum accuracy gate — skip symbolification if PyKAN can't predict
PYKAN_MIN_ACCURACY = 0.53          # must beat random (50%) by a margin

# ---------------------------------------------------------------------------
# Symbolic extraction constants
# ---------------------------------------------------------------------------
PRUNE_THRESHOLD = 0.01

SYMBOLIC_LIBRARY = [
    "x", "x^2", "x^3", "exp", "log", "sqrt", "tanh", "sin", "abs",
    "sigmoid", "x*abs(x)", "1/x", "0",
]
SYMBOLIC_R2_THRESHOLD = 0.3        # lowered from 0.5 to see what R² values
                                   # actually exist before filtering too aggressively
SYMBOLIC_TOPK = 5

AFFINE_FINETUNE_STEPS = 30
AFFINE_LR = 0.0004          # from VIX paper

CACHE_DIR = "cache/"


# =====================================================================
# Diagnostic helpers
# =====================================================================
def _compute_accuracy(model, X: torch.Tensor, y: torch.Tensor) -> float:
    """Compute classification accuracy (no grad)."""
    model.eval()
    with torch.no_grad():
        pred = model(X)
        acc = (pred.argmax(dim=1) == y).float().mean().item()
    return acc


def _count_active_edges(model, threshold: float) -> tuple[int, int]:
    """Count total and active (above threshold) edges in the KAN.

    Inspects activation magnitudes to determine which edges carry
    meaningful signal vs. near-zero activations.
    """
    total = 0
    active = 0
    try:
        for l in range(len(model.width) - 1):
            n_in = model.width[l]
            n_out = model.width[l + 1]
            if isinstance(n_in, (list, tuple)):
                n_in = n_in[0] if n_in else 0
            if isinstance(n_out, (list, tuple)):
                n_out = n_out[0] if n_out else 0
            for i in range(n_in):
                for j in range(n_out):
                    total += 1
                    try:
                        # check activation magnitude via act_fun attribute
                        act = model.act_fun[l]
                        # different PyKAN versions store activations differently
                        if hasattr(act, 'coef'):
                            coef_norm = act.coef[j, i].abs().mean().item()
                        else:
                            coef_norm = 1.0  # assume active if we can't check
                        if coef_norm > threshold:
                            active += 1
                    except (IndexError, AttributeError, RuntimeError):
                        active += 1  # assume active if we can't check
    except (AttributeError, TypeError):
        return -1, -1  # can't inspect
    return total, active


def _log_diagnostic(label: str, model, X_train, y_train, X_val, y_val):
    """Print a diagnostic checkpoint with train/val accuracy."""
    train_acc = _compute_accuracy(model, X_train, y_train)
    val_acc = _compute_accuracy(model, X_val, y_val)
    print(f"    [{label}] train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    logger.info("%s: train_acc=%.4f, val_acc=%.4f", label, train_acc, val_acc)
    return train_acc, val_acc


# =====================================================================
# 1. Select best extraction fold
# =====================================================================
def select_extraction_fold(cpcv_results: dict) -> tuple[int, dict]:
    """Identify the CPCV fold where KAN achieved best F1 macro."""
    predictions = cpcv_results["predictions"]
    n_splits = cpcv_results["n_splits"]
    n_seeds = cpcv_results["n_seeds"]

    split_f1s = {}
    split_prep = {}

    for split_idx in range(n_splits):
        seed_f1s = []
        prep_info = None
        for seed in range(n_seeds):
            key = ("kan", split_idx, seed)
            if key not in predictions:
                continue
            pred = predictions[key]
            f1 = pred.get("f1_macro", np.nan)
            if not np.isnan(f1):
                seed_f1s.append(f1)
            if prep_info is None:
                prep_info = pred.get("prep_info", {})

        if seed_f1s:
            split_f1s[split_idx] = np.mean(seed_f1s)
            split_prep[split_idx] = prep_info

    if not split_f1s:
        logger.warning("No KAN predictions found. Using first available split.")
        for key, pred in predictions.items():
            return key[1], pred.get("prep_info", {})

    best_split = max(split_f1s, key=split_f1s.get)
    logger.info(
        "Best KAN fold: split %d (F1=%.4f). All: %s",
        best_split, split_f1s[best_split],
        {k: f"{v:.4f}" for k, v in split_f1s.items()},
    )

    return best_split, split_prep[best_split]


# =====================================================================
# 2. Prepare extraction data
# =====================================================================
def prepare_extraction_data(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    t1: pd.Series,
    cpcv_results: dict,
    best_split_idx: int,
    prep_info: dict,
) -> tuple[dict, list[str]]:
    """Reconstruct preprocessed data for the extraction fold.

    Applies tanh normalization to match both efficient-kan and PyKAN
    input preprocessing.
    """
    splits = generate_cpcv_splits(X, t1)
    train_idx, _ = splits[best_split_idx]

    y_mapped = ((y + 1) // 2).astype(int)

    ffd_info = prep_info.get("ffd", {})
    scaler = prep_info.get("scaler", None)
    selected_features = prep_info.get("selected_features", list(X.columns))

    # apply FFD to full series then extract training fold
    X_transformed = X.copy()
    for col, d_star in ffd_info.items():
        if col in X_transformed.columns:
            X_transformed[col] = apply_ffd(X_transformed[col], d_star)

    X_train = X_transformed.iloc[train_idx].copy()
    X_train = X_train.loc[X_train.notna().all(axis=1)]
    y_train = y_mapped.loc[X_train.index]

    # scale using the stored scaler (same transform as pipeline)
    if scaler is not None:
        X_train = pd.DataFrame(
            scaler.transform(X_train),
            index=X_train.index, columns=X_train.columns,
        )

    X_train = X_train[selected_features]

    # 80/20 split (same as pipeline)
    cal_boundary = int(len(X_train) * 0.8)
    X_model = X_train.iloc[:cal_boundary]
    X_val = X_train.iloc[cal_boundary:]
    y_model = y_train.iloc[:cal_boundary]
    y_val = y_train.iloc[cal_boundary:]

    # convert to tensors
    X_model_t = torch.tensor(X_model.values, dtype=torch.float32)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)

    # tanh normalization (fit on training split, apply to both)
    input_mean = X_model_t.mean(dim=0)
    input_std = X_model_t.std(dim=0) + 1e-8
    X_model_t = torch.tanh((X_model_t - input_mean) / input_std)
    X_val_t = torch.tanh((X_val_t - input_mean) / input_std)

    dataset = {
        "train_input": X_model_t,
        "train_label": torch.tensor(y_model.values, dtype=torch.float32),
        "test_input": X_val_t,
        "test_label": torch.tensor(y_val.values, dtype=torch.float32),
    }

    feature_names = list(selected_features)
    logger.info(
        "Extraction data: %d train, %d val, %d features (tanh-normalized).",
        len(X_model), len(X_val), len(feature_names),
    )
    return dataset, feature_names


# =====================================================================
# 3. Train PyKAN (same architecture, staged training)
# =====================================================================
def train_pykan(dataset: dict, n_features: int, n_classes: int = 2):
    """Train a fresh PyKAN model for symbolic extraction.

    Uses a data-aware architecture: hidden width is chosen so that the
    total parameter count stays well below the number of training samples,
    preventing LBFGS memorization.

    Phase 1: Adam with weight decay + input noise (generalization)
    Grid extension: only if dataset is large enough
    Phase 2a: LBFGS warmup (short, no regularization)
    Phase 2b: LBFGS sparsity (short, gentle L1 + entropy)
    """
    from kan import KAN

    X_t = dataset["train_input"]
    y_t = dataset["train_label"].long()
    X_val_t = dataset["test_input"]
    y_val_t = dataset["test_label"].long()
    n_train = X_t.shape[0]

    # ── Data-aware architecture ───────────────────────────────────────
    # Each edge has ~(grid + k) parameters. With grid=3, k=3, that's ~6
    # params per edge. We want n_train / total_params >= 5.
    params_per_edge = PYKAN_GRID_INIT + KAN_K
    max_edges = n_train // PYKAN_MIN_SAMPLES_PER_PARAM

    # Architecture [n_features, h, n_classes] has n_features*h + h*n_classes edges
    # Solve for h: h * (n_features + n_classes) <= max_edges / params_per_edge
    max_hidden = max_edges // (params_per_edge * (n_features + n_classes))
    max_hidden = max(2, min(max_hidden, KAN_HIDDEN))  # clamp to [2, KAN_HIDDEN]

    hidden = max_hidden
    width = [n_features, hidden, n_classes]
    total_edges = n_features * hidden + hidden * n_classes
    total_params_est = total_edges * params_per_edge

    print(
        f"    Data-aware architecture: {width} "
        f"({total_edges} edges, ~{total_params_est} params for {n_train} samples, "
        f"ratio={n_train/max(total_params_est,1):.1f}x)"
    )

    if n_train / max(total_params_est, 1) < 2:
        print(
            f"    ⚠ WARNING: samples/params ratio < 2. "
            f"Memorization is very likely."
        )

    model = KAN(width=width, grid=PYKAN_GRID_INIT, k=KAN_K, seed=42)

    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Adam with weight decay + noise injection ─────────────
    logger.info(
        "Extraction Phase 1: Adam (%d steps, wd=%.4f, noise=%.3f)",
        PYKAN_ADAM_STEPS, PYKAN_ADAM_WEIGHT_DECAY, PYKAN_NOISE_STD,
    )
    print(
        f"    Phase 1: Adam ({PYKAN_ADAM_STEPS} steps, "
        f"wd={PYKAN_ADAM_WEIGHT_DECAY}, noise_std={PYKAN_NOISE_STD})..."
    )
    optimizer_adam = torch.optim.Adam(
        model.parameters(), lr=PYKAN_ADAM_LR,
        weight_decay=PYKAN_ADAM_WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state = None

    for step in range(PYKAN_ADAM_STEPS):
        model.train()
        optimizer_adam.zero_grad()

        # noise injection: perturb inputs to prevent memorization
        X_noisy = X_t + PYKAN_NOISE_STD * torch.randn_like(X_t)
        X_noisy = X_noisy.clamp(-1, 1)  # keep within tanh-normalized range

        logits = model(X_noisy)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer_adam.step()

        if (step + 1) % PYKAN_VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Diagnostic: check if Adam learned anything ────────────────────
    train_acc, val_acc = _log_diagnostic(
        "After Adam", model, X_t, y_t, X_val_t, y_val_t
    )

    if val_acc < PYKAN_MIN_ACCURACY:
        logger.warning(
            "Adam phase val_acc=%.4f < %.2f minimum. "
            "PyKAN may not have learned meaningful patterns.",
            val_acc, PYKAN_MIN_ACCURACY,
        )
        print(
            f"    ⚠ WARNING: val_acc={val_acc:.4f} barely above random. "
            f"Continuing, but symbolic extraction may yield constants."
        )

    # ── Grid extension (conditional) ──────────────────────────────────
    if PYKAN_GRID_EXTEND and n_train > 1000:
        try:
            model = model.refine(KAN_GRID)
            logger.info("Grid extended: %d → %d", PYKAN_GRID_INIT, KAN_GRID)
            print(f"    Grid extended: {PYKAN_GRID_INIT} → {KAN_GRID}")
            _log_diagnostic("After grid extend", model, X_t, y_t, X_val_t, y_val_t)
        except (AttributeError, TypeError, Exception) as e:
            logger.warning("Grid extension failed (%s).", e)
            print(f"    Grid extension failed: {e}")
    else:
        print(
            f"    Grid extension SKIPPED (n_train={n_train}, "
            f"grid stays at {PYKAN_GRID_INIT}). "
            f"Too few samples for finer grid."
        )

    # ── Phase 2: LBFGS (short, to refine not memorize) ────────────────
    lbfgs_warmup_steps = int(PYKAN_LBFGS_STEPS * PYKAN_LBFGS_WARMUP_FRAC)
    lbfgs_sparse_steps = PYKAN_LBFGS_STEPS - lbfgs_warmup_steps

    # ── Phase 2a: LBFGS warmup (no regularization) ────────────────────
    logger.info("Extraction Phase 2a: LBFGS warmup (%d steps, no reg)", lbfgs_warmup_steps)
    print(f"    Phase 2a: LBFGS warmup ({lbfgs_warmup_steps} steps, lamb=0)...")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=PYKAN_LBFGS_LR, max_iter=10,
        line_search_fn="strong_wolfe",
    )

    # track best val from Adam phase as starting point
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for step in range(lbfgs_warmup_steps):
        model.train()

        def closure_warmup():
            optimizer_lbfgs.zero_grad()
            logits = model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            return loss

        optimizer_lbfgs.step(closure_warmup)

        if (step + 1) % PYKAN_VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PYKAN_PATIENCE:
                logger.info("LBFGS warmup early stop at step %d", step + 1)
                break

    model.load_state_dict(best_state)

    # ── Diagnostic after LBFGS warmup ─────────────────────────────────
    _log_diagnostic("After LBFGS warmup", model, X_t, y_t, X_val_t, y_val_t)

    # ── Phase 2b: LBFGS with sparsity regularization ──────────────────
    logger.info(
        "Extraction Phase 2b: LBFGS sparsity (%d steps, lamb=%.4f)",
        lbfgs_sparse_steps, PYKAN_LAMB,
    )
    print(f"    Phase 2b: LBFGS sparsity ({lbfgs_sparse_steps} steps, lamb={PYKAN_LAMB})...")

    optimizer_lbfgs2 = torch.optim.LBFGS(
        model.parameters(), lr=PYKAN_LBFGS_LR, max_iter=10,
        line_search_fn="strong_wolfe",
    )

    best_val_loss_sparse = float("inf")
    best_state_sparse = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for step in range(lbfgs_sparse_steps):
        model.train()

        def closure_sparse():
            optimizer_lbfgs2.zero_grad()
            logits = model(X_t)
            loss = criterion(logits, y_t)
            try:
                reg_l1 = model.regularization_loss(
                    regularize_activation=1.0, regularize_entropy=0.0
                )
                reg_ent = model.regularization_loss(
                    regularize_activation=0.0, regularize_entropy=1.0
                )
                loss = loss + PYKAN_LAMB * (
                    PYKAN_LAMB_L1 * reg_l1 + PYKAN_LAMB_ENTROPY * reg_ent
                )
            except (AttributeError, TypeError):
                pass
            loss.backward()
            return loss

        optimizer_lbfgs2.step(closure_sparse)

        if (step + 1) % PYKAN_VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            if val_loss < best_val_loss_sparse:
                best_val_loss_sparse = val_loss
                best_state_sparse = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PYKAN_PATIENCE:
                logger.info("LBFGS sparsity early stop at step %d", step + 1)
                break

    model.load_state_dict(best_state_sparse)
    model.eval()

    # ── Final diagnostic ──────────────────────────────────────────────
    train_acc, val_acc = _log_diagnostic(
        "After LBFGS sparsity (final)", model, X_t, y_t, X_val_t, y_val_t
    )

    model._pre_symbolic_accuracy = val_acc
    logger.info("PyKAN trained. Val accuracy: %.4f, width: %s", val_acc, width)
    return model


# =====================================================================
# 4. Prune (Algorithm 1, Step 2)
# =====================================================================
def prune_network(model, dataset: dict):
    """Prune dead edges and nodes.

    Includes diagnostic logging of edge survival counts to identify
    whether regularization is too aggressive.
    """
    _ = model(dataset["train_input"])

    try:
        original_width = list(model.width)
    except AttributeError:
        original_width = ["unknown"]

    # ── Diagnostic: edge survival before pruning ──────────────────────
    total_edges, active_edges = _count_active_edges(model, PRUNE_THRESHOLD)
    if total_edges > 0:
        print(
            f"    Edge analysis (threshold={PRUNE_THRESHOLD}): "
            f"{active_edges}/{total_edges} edges active "
            f"({active_edges/total_edges:.0%} survival rate)"
        )
        logger.info(
            "Pre-prune edge analysis: %d/%d active (%.1f%%)",
            active_edges, total_edges, 100 * active_edges / max(total_edges, 1),
        )

        if active_edges < 3:
            print(
                "    ⚠ WARNING: Very few active edges. Regularization may have "
                "been too aggressive. Symbolic extraction will likely yield constants."
            )

    # ── Accuracy gate ─────────────────────────────────────────────────
    val_acc = _compute_accuracy(
        model, dataset["test_input"], dataset["test_label"].long()
    )
    if val_acc < PYKAN_MIN_ACCURACY:
        print(
            f"    ⚠ WARNING: Pre-prune val_acc={val_acc:.4f} < {PYKAN_MIN_ACCURACY}. "
            f"Model hasn't learned meaningful patterns."
        )

    try:
        model.attribute()
    except Exception as e:
        logger.warning("model.attribute() failed: %s", e)

    pre_prune_model = model

    # try different PyKAN prune APIs (varies by version)
    pruned = False
    try:
        model = model.prune(threshold=PRUNE_THRESHOLD)
        pruned = True
    except TypeError:
        try:
            model = model.prune(node_th=PRUNE_THRESHOLD, edge_th=PRUNE_THRESHOLD)
            pruned = True
        except TypeError:
            try:
                model = model.prune()
                pruned = True
            except Exception as e:
                logger.warning("model.prune() failed: %s. Returning unpruned.", e)
                return pre_prune_model
    except Exception as e:
        logger.warning("model.prune() failed: %s. Returning unpruned.", e)
        return pre_prune_model

    # verify pruned model can still do a forward pass
    if pruned:
        try:
            _ = model(dataset["train_input"])
        except (RuntimeError, Exception) as e:
            logger.warning(
                "Pruned model forward pass failed (%s). Returning unpruned.", e
            )
            return pre_prune_model

    try:
        pruned_width = list(model.width)
    except AttributeError:
        pruned_width = ["unknown"]

    # ── Diagnostic: post-prune edge count ─────────────────────────────
    post_total, post_active = _count_active_edges(model, PRUNE_THRESHOLD)
    print(f"    Pruned: {original_width} → {pruned_width}")
    if post_total > 0:
        print(f"    Post-prune edges: {post_total} remaining")
    logger.info("Pruned: %s → %s", original_width, pruned_width)

    # ── Post-prune accuracy ───────────────────────────────────────────
    post_acc = _compute_accuracy(
        model, dataset["test_input"], dataset["test_label"].long()
    )
    print(f"    Post-prune val_acc: {post_acc:.4f}")

    _save_plot(model, "kan_pruned_network.png")
    return model


# =====================================================================
# 5. Symbolify (Algorithm 1, Step 3 + 4)
# =====================================================================
def symbolify_network(model, dataset: dict):
    """Replace B-spline activations with symbolic functions, then fine-tune."""
    # ensure activations are cached from a clean forward pass
    model.eval()
    _ = model(dataset["train_input"])

    # ── Activation sanity check ───────────────────────────────────────
    try:
        with torch.no_grad():
            out = model(dataset["train_input"])
            if torch.isnan(out).any():
                print("    ⚠ Model produces NaN outputs. Symbolification will fail.")
            elif (out.std(dim=0) < 1e-6).all():
                print("    ⚠ Model outputs are near-constant. Activations may be flat.")
            else:
                logit_diff = (out[:, 1] - out[:, 0]) if out.shape[1] > 1 else out[:, 0]
                print(
                    f"    Activation check: logit_diff std={logit_diff.std().item():.4f}, "
                    f"range=[{logit_diff.min().item():.3f}, {logit_diff.max().item():.3f}]"
                )
    except Exception as e:
        print(f"    Activation check failed: {e}")

    with torch.no_grad():
        pre_pred = model(dataset["test_input"])
        pre_acc = (pre_pred.argmax(dim=1) == dataset["test_label"].long()).float().mean().item()

    pre_state = copy.deepcopy(model.state_dict())

    total_edges = 0
    symbolified_edges = 0
    skipped_edges = 0
    r2_values = []               # collect all R² values for diagnostics

    for l in range(len(model.width) - 1):
        n_in = model.width[l]
        n_out = model.width[l + 1]

        if isinstance(n_in, (list, tuple)):
            n_in = n_in[0] if n_in else 0
        if isinstance(n_out, (list, tuple)):
            n_out = n_out[0] if n_out else 0

        for i in range(n_in):
            for j in range(n_out):
                total_edges += 1

                # ── each edge gets its own try/except ─────────────
                try:
                    suggestions = model.suggest_symbolic(
                        l, i, j, topk=SYMBOLIC_TOPK, lib=SYMBOLIC_LIBRARY,
                    )
                except Exception as e:
                    # PRINT first 5 errors to diagnose why all edges are skipped
                    if skipped_edges < 5:
                        print(
                            f"    ⚠ suggest_symbolic({l},{i},{j}) error: "
                            f"{type(e).__name__}: {e}"
                        )
                    logger.debug("suggest_symbolic(%d,%d,%d) failed: %s", l, i, j, e)
                    skipped_edges += 1
                    continue

                if suggestions is None:
                    skipped_edges += 1
                    continue

                # ── safe parsing of suggestions ───────────────────
                best_fn = None
                best_r2 = 0.0

                try:
                    if hasattr(suggestions, "iloc"):
                        if len(suggestions) == 0:
                            skipped_edges += 1
                            continue
                        best_fn = str(suggestions.iloc[0, 0])
                        r2_col = [
                            c for c in suggestions.columns
                            if "r2" in c.lower() and "loss" not in c.lower()
                        ]
                        if r2_col:
                            raw_r2 = suggestions.iloc[0][r2_col[0]]
                            best_r2 = _safe_float(raw_r2, default=0.0)
                    elif isinstance(suggestions, (list, tuple)):
                        if len(suggestions) == 0:
                            skipped_edges += 1
                            continue
                        best_fn = str(suggestions[0][0])
                        if len(suggestions[0]) > 2:
                            best_r2 = _safe_float(suggestions[0][2], default=0.0)
                    else:
                        skipped_edges += 1
                        continue
                except Exception as e:
                    if skipped_edges < 5:
                        print(
                            f"    ⚠ Edge ({l},{i},{j}) parse error: "
                            f"{type(e).__name__}: {e}"
                        )
                    logger.debug(
                        "Edge (%d,%d,%d): failed to parse suggestions: %s",
                        l, i, j, e,
                    )
                    skipped_edges += 1
                    continue

                if best_fn is None:
                    skipped_edges += 1
                    continue

                # ── collect R² for diagnostics ────────────────────
                r2_values.append((l, i, j, best_fn, best_r2))

                # ── apply symbolic replacement if R² is good enough ──
                if best_r2 >= SYMBOLIC_R2_THRESHOLD:
                    try:
                        model.fix_symbolic(l, i, j, best_fn)
                        symbolified_edges += 1
                        logger.info(
                            "Edge (%d,%d,%d): %s (R²=%.4f) ✓",
                            l, i, j, best_fn, best_r2,
                        )
                    except Exception as e:
                        logger.warning(
                            "fix_symbolic(%d,%d,%d) failed: %s",
                            l, i, j, e,
                        )
                else:
                    logger.debug(
                        "Edge (%d,%d,%d): best=%s R²=%.4f < %.2f, keeping spline.",
                        l, i, j, best_fn, best_r2, SYMBOLIC_R2_THRESHOLD,
                    )

    sym_rate = symbolified_edges / max(total_edges, 1)
    print(
        f"  Symbolified: {symbolified_edges}/{total_edges} edges ({sym_rate:.0%})"
        f"  [skipped: {skipped_edges}]"
    )

    # ── R² diagnostic summary ─────────────────────────────────────────
    if r2_values:
        r2_scores = [v[4] for v in r2_values]
        print(
            f"  R² distribution: min={min(r2_scores):.4f}, "
            f"median={np.median(r2_scores):.4f}, "
            f"max={max(r2_scores):.4f}, "
            f"above threshold ({SYMBOLIC_R2_THRESHOLD}): "
            f"{sum(1 for r in r2_scores if r >= SYMBOLIC_R2_THRESHOLD)}/{len(r2_scores)}"
        )
        # show top 5 edges by R²
        top5 = sorted(r2_values, key=lambda x: x[4], reverse=True)[:5]
        print("  Top 5 edges by R²:")
        for l, i, j, fn, r2 in top5:
            print(f"    Edge ({l},{i},{j}): {fn} (R²={r2:.4f})")
    else:
        print("  ⚠ No R² values collected — all edges were skipped.")

    # ── Step 4: fine-tune affine parameters ───────────────────────────
    if symbolified_edges > 0:
        print(f"  Fine-tuning affine parameters ({AFFINE_FINETUNE_STEPS} steps)...")
        try:
            dataset_fit = {
                "train_input": dataset["train_input"],
                "train_label": dataset["train_label"].long(),
                "test_input": dataset["test_input"],
                "test_label": dataset["test_label"].long(),
            }
            model.fit(
                dataset_fit, opt="LBFGS", lr=AFFINE_LR,
                steps=AFFINE_FINETUNE_STEPS,
                loss_fn=nn.CrossEntropyLoss(),
                update_grid=False,
            )
        except (TypeError, AttributeError, RuntimeError):
            try:
                optimizer = torch.optim.LBFGS(model.parameters(), lr=AFFINE_LR, max_iter=10)
                loss_fn = nn.CrossEntropyLoss()
                for step in range(AFFINE_FINETUNE_STEPS):
                    def closure():
                        optimizer.zero_grad()
                        pred = model(dataset["train_input"])
                        loss = loss_fn(pred, dataset["train_label"].long())
                        if torch.isnan(loss):
                            raise ValueError("NaN loss")
                        loss.backward()
                        return loss
                    try:
                        optimizer.step(closure)
                    except ValueError:
                        logger.warning("NaN at affine step %d. Reverting.", step)
                        model.load_state_dict(pre_state)
                        break
            except Exception as e:
                logger.warning("Affine fine-tuning failed: %s. Reverting.", e)
                model.load_state_dict(pre_state)
    else:
        print("  Skipping affine fine-tuning (no edges were symbolified).")

    _save_plot(model, "kan_symbolified_network.png")

    model._symbolification_rate = sym_rate
    model._pre_symbolic_accuracy = pre_acc
    return model


# =====================================================================
# 6. Extract formulas
# =====================================================================
def extract_formulas(model, dataset: dict, feature_names: list[str]) -> dict:
    """Extract closed-form expressions from the symbolified PyKAN model."""
    try:
        formulas = model.symbolic_formula()
    except Exception as e:
        logger.error("model.symbolic_formula() failed: %s", e)
        return _empty_result(model, feature_names)

    try:
        if isinstance(formulas, (list, tuple)) and len(formulas) > 0:
            expr_list = formulas[0] if isinstance(formulas[0], (list, tuple)) else formulas
            logit_bearish = expr_list[0]
            logit_bullish = expr_list[1] if len(expr_list) > 1 else sympy.Integer(0)
        else:
            return _empty_result(model, feature_names)
    except Exception as e:
        logger.error("Formula parsing failed: %s", e)
        return _empty_result(model, feature_names)

    # substitute x_0, x_1, ... with actual feature names
    for i, name in enumerate(feature_names):
        old = sympy.Symbol(f"x_{i}")
        new = sympy.Symbol(name)
        try:
            logit_bearish = logit_bearish.subs(old, new)
            logit_bullish = logit_bullish.subs(old, new)
        except (AttributeError, TypeError):
            pass

    try:
        decision = sympy.simplify(logit_bullish - logit_bearish)
    except Exception:
        decision = logit_bullish - logit_bearish

    try:
        logit_bearish = sympy.nsimplify(logit_bearish, tolerance=1e-4)
        logit_bullish = sympy.nsimplify(logit_bullish, tolerance=1e-4)
        decision = sympy.nsimplify(decision, tolerance=1e-4)
    except Exception:
        pass

    pre_acc = getattr(model, "_pre_symbolic_accuracy", np.nan)
    sym_rate = getattr(model, "_symbolification_rate", np.nan)

    try:
        with torch.no_grad():
            post_pred = model(dataset["test_input"])
            post_acc = (
                (post_pred.argmax(dim=1) == dataset["test_label"].long())
                .float().mean().item()
            )
    except Exception:
        post_acc = np.nan

    surviving = [
        name for name in feature_names
        if sympy.Symbol(name) in decision.free_symbols
    ]

    try:
        pruned_arch = list(model.width)
    except AttributeError:
        pruned_arch = ["unknown"]

    return {
        "logit_bearish": str(logit_bearish),
        "logit_bullish": str(logit_bullish),
        "decision_function": str(decision),
        "p_up_formula": f"1 / (1 + exp(-({decision})))",
        "sympy_objects": {
            "bearish": logit_bearish,
            "bullish": logit_bullish,
            "decision": decision,
        },
        "pre_symbolic_accuracy": pre_acc,
        "post_symbolic_accuracy": post_acc,
        "symbolification_rate": sym_rate,
        "pruned_architecture": pruned_arch,
        "surviving_features": surviving,
    }


# =====================================================================
# Top-level orchestration
# =====================================================================
def run_symbolic_extraction(
    cpcv_results: dict,
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    t1: pd.Series,
) -> dict:
    """Run the full symbolic extraction pipeline (Algorithm 1).

    Called from the notebook after CPCV results are available.
    The CPCV pipeline used efficient-kan; this retrains a fresh PyKAN
    on the best fold for symbolic analysis.
    """
    print("=" * 60)
    print("Symbolic Extraction (VIX KAN Paper, Algorithm 1)")
    print("  CPCV model: efficient-kan | Extraction model: PyKAN")
    print("=" * 60)

    # 1. select best fold
    best_split, prep_info = select_extraction_fold(cpcv_results)
    print(f"  Best fold: split {best_split}")

    # 2. prepare data
    dataset, feature_names = prepare_extraction_data(
        X, y, w, t1, cpcv_results, best_split, prep_info,
    )
    print(
        f"  Data: {dataset['train_input'].shape[0]} train, "
        f"{dataset['test_input'].shape[0]} val, "
        f"{len(feature_names)} features: {feature_names}"
    )

    # 3. train PyKAN from scratch
    print("\n  Step 1: Training PyKAN (Adam → grid extend → LBFGS)...")
    model = train_pykan(dataset, n_features=len(feature_names))

    # 4. prune
    print("\n  Step 2: Pruning...")
    model = prune_network(model, dataset)

    # 5. symbolify + fine-tune
    print("\n  Steps 3+4: Symbolification + affine fine-tuning...")
    model = symbolify_network(model, dataset)

    # 6. extract formulas
    print("\n  Extracting formulas...")
    result = extract_formulas(model, dataset, feature_names)

    # summary
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Pre-symbolic accuracy:  {result['pre_symbolic_accuracy']:.4f}")
    print(f"  Post-symbolic accuracy: {result['post_symbolic_accuracy']:.4f}")
    print(f"  Symbolification rate:   {result['symbolification_rate']:.0%}")
    print(f"  Surviving features:     {result['surviving_features']}")
    print(f"  Pruned architecture:    {result['pruned_architecture']}")
    print(f"\n  Decision function:")
    print(f"    {result['decision_function']}")
    print(f"\n  P(up) = {result['p_up_formula']}")
    print(f"{'='*60}")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float, handling 'nan', 'n', and other junk.

    PyKAN's suggest_symbolic() can return DataFrames with string values
    like 'nan', truncated strings, or other non-numeric entries that
    cause ``float()`` to raise ValueError.
    """
    if value is None:
        return default
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        # handle truncated strings like 'n' from 'nan', etc.
        logger.debug("_safe_float: could not convert %r, using default %.2f", value, default)
        return default


def _save_plot(model, filename: str) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            model.plot(mask=True, beta=10)
        except TypeError:
            model.plot(beta=10)
        import matplotlib.pyplot as plt
        plt.savefig(
            os.path.join(CACHE_DIR, filename),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        logger.info("Saved %s", filename)
    except Exception as e:
        logger.warning("Could not save %s: %s", filename, e)


def _empty_result(model, feature_names: list[str]) -> dict:
    """Fallback when formula extraction fails."""
    return {
        "logit_bearish": "extraction_failed",
        "logit_bullish": "extraction_failed",
        "decision_function": "extraction_failed",
        "p_up_formula": "extraction_failed",
        "sympy_objects": {"bearish": None, "bullish": None, "decision": None},
        "pre_symbolic_accuracy": getattr(model, "_pre_symbolic_accuracy", np.nan),
        "post_symbolic_accuracy": np.nan,
        "symbolification_rate": getattr(model, "_symbolification_rate", np.nan),
        "pruned_architecture": [],
        "surviving_features": [],
    }