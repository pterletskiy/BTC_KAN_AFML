"""
13) Symbolic Extraction
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
import io
import logging
import os
import re
import sys
import threading

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

# Symbolic extraction simplifications (when using tuned CPCV KAN params)
PYKAN_SYMBOLIC_WIDTH_CAP = 8       # soft ceiling on width1; tuned KAN width1 is
                                   # capped at this to keep formulas readable.
                                   # A width=12 KAN produces ~24 summed terms
                                   # pre-pruning, which overwhelms sympy.
PYKAN_SYMBOLIC_DROP_WIDTH2 = True  # force single hidden layer for extraction.
                                   # Two hidden layers produce nested f(g(x))
                                   # compositions that sympy struggles with.
PYKAN_SYMBOLIC_FORCE_GRID = 3      # force coarse grid for extraction. Coarser
                                   # splines → cleaner symbolic matches.

# Minimum accuracy gate — skip symbolification if PyKAN can't predict
PYKAN_MIN_ACCURACY = 0.53          # must beat random (50%) by a margin

# ---------------------------------------------------------------------------
# Symbolic extraction constants
# ---------------------------------------------------------------------------
PRUNE_THRESHOLD = 0.01

SYMBOLIC_LIBRARY = [
    "x", "x^2", "x^3", "x^4",     # polynomials
    "exp", "log", "sqrt",           # standard transforms
    "tanh", "sin", "cos",           # bounded nonlinearities
    "abs", "sgn",                   # piecewise
    "arctan",                       # bounded monotonic
    "0",                            # constant (zero)
]
# NOTE: 'sigmoid' and 'x*abs(x)' are NOT in PyKAN's internal
# SYMBOLIC_LIB and cause KeyError. '1/x' can cause division-by-zero
# issues. Only use names that PyKAN recognizes natively.
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
def select_extraction_fold(
    cpcv_results: dict,
    fold_selection: str | int = "best",
) -> tuple[int, dict]:
    """Select which CPCV fold to use for symbolic extraction.

    Parameters
    ----------
    fold_selection : str or int
        - "best": fold with highest KAN F1 macro (default)
        - "last": last fold (most recent data, closest to rolling window)
        - int: specific fold index
    """
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

    # log all fold F1s for reference
    logger.info(
        "Fold F1 scores: %s",
        {k: f"{v:.4f}" for k, v in split_f1s.items()},
    )

    # select fold based on strategy
    if isinstance(fold_selection, int):
        selected = fold_selection
        if selected not in split_f1s:
            logger.warning("Fold %d not found. Falling back to best fold.", selected)
            selected = max(split_f1s, key=split_f1s.get)
        reason = f"manual (F1={split_f1s.get(selected, np.nan):.4f})"

    elif fold_selection == "last":
        selected = max(split_f1s.keys())
        reason = f"last/most recent (F1={split_f1s[selected]:.4f})"

    else:  # "best"
        selected = max(split_f1s, key=split_f1s.get)
        reason = f"best F1 (F1={split_f1s[selected]:.4f})"

    logger.info("Selected fold: split %d — %s", selected, reason)
    print(f"  Fold selection: split {selected} — {reason}")

    return selected, split_prep[selected]


# =====================================================================
# 2. Rank features by CPCV selection frequency
# =====================================================================
def rank_features_by_stability(cpcv_results: dict) -> list[tuple[str, float]]:
    """Rank features by how often they were selected across CPCV folds.

    Returns a list of (feature_name, selection_frequency) sorted descending.
    Features selected in more folds are more stable/important.
    """
    predictions = cpcv_results["predictions"]
    feature_counts = {}
    total_folds = 0

    for key, pred in predictions.items():
        model_name = key[0]
        if model_name != "kan":
            continue

        prep_info = pred.get("prep_info", {})
        selected = prep_info.get("selected_features", [])
        if not selected:
            continue

        total_folds += 1
        for feat in selected:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    if total_folds == 0:
        logger.warning("No KAN folds found for feature ranking.")
        return []

    ranked = [
        (feat, count / total_folds)
        for feat, count in feature_counts.items()
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =====================================================================
# 3. Prepare extraction data
# =====================================================================
def prepare_extraction_data(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    t1: pd.Series,
    cpcv_results: dict,
    best_split_idx: int,
    prep_info: dict,
    feature_subset: list[str] | None = None,
    n_groups: int | None = None,
    k: int | None = None,
    embargo_pct: float | None = None,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict, list[str]]:
    """Reconstruct preprocessed data for the extraction fold.

    Applies tanh normalization to match both efficient-kan and PyKAN
    input preprocessing.

    Parameters
    ----------
    feature_subset : list of str, optional
        If provided, only these features are used for symbolic extraction.
        This allows extracting simpler formulas with fewer variables,
        independent of the CPCV feature selection.
    n_groups, k, embargo_pct : optional
        CPCV configuration. Must match the values used to produce
        ``cpcv_results``. If omitted, defaults are inferred from
        ``cpcv_results["split_info"]`` so that the regenerated splits
        align with the original training fold. Pass them explicitly
        only when ``cpcv_results`` lacks a ``split_info`` entry.
    splits : list of (train_idx, test_idx), optional
        Pre-computed CPCV splits, identical to the list produced by
        the notebook's CV cell. When provided, the function uses them
        directly and skips internal regeneration. Recommended pattern:
        pass the same ``splits`` list the notebook produced upstream.
    """
    # Resolve CPCV configuration: prefer explicit splits → cpcv_results
    # split_info → explicit n_groups/k/embargo_pct → cv.py defaults.
    if splits is None:
        if n_groups is None or k is None or embargo_pct is None:
            si = cpcv_results.get("split_info") or {}
            if n_groups is None:
                n_groups = si.get("n_groups")
            if k is None:
                k = si.get("k")
            if embargo_pct is None:
                embargo_pct = si.get("embargo_pct")
        if n_groups is None or k is None or embargo_pct is None:
            raise ValueError(
                "prepare_extraction_data: CPCV configuration could not be "
                "resolved. Pass ``splits`` directly, or pass ``n_groups`` / "
                "``k`` / ``embargo_pct``, or ensure ``cpcv_results`` "
                "contains a ``split_info`` entry with these keys. "
                "Regenerating with cv.py defaults would silently use a "
                "different fold from the one ``cpcv_results`` was trained "
                "on, which is unsafe."
            )
        splits = generate_cpcv_splits(
            X, t1, n_groups=n_groups, k=k, embargo_pct=embargo_pct,
        )
    train_idx, _ = splits[best_split_idx]

    # ``y`` may arrive as a numpy array, a pandas Series, or anything
    # array-like depending on how the notebook assembles the inputs.
    # Coerce to a pandas Series indexed on ``X.index`` so subsequent
    # ``.loc`` lookups work uniformly regardless of input type.
    if not isinstance(y, pd.Series):
        y_arr = np.asarray(y)
        if len(y_arr) != len(X):
            raise ValueError(
                f"prepare_extraction_data: ``y`` has length {len(y_arr)} "
                f"but ``X`` has length {len(X)}. The function expects the "
                "event-aligned label series produced by the pre-CPCV "
                "alignment step, not the pooled prediction labels from "
                "``cpcv_results``. Verify that ``y`` in the notebook is "
                "the same variable that was passed to ``run_cpcv_pipeline``."
            )
        y = pd.Series(y_arr, index=X.index)
    elif len(y) != len(X) or not y.index.equals(X.index):
        raise ValueError(
            f"prepare_extraction_data: ``y`` (length {len(y)}) and ``X`` "
            f"(length {len(X)}) do not share an aligned index. Re-run "
            "the pre-CPCV alignment step to restore them."
        )
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

    # override with explicit feature subset for symbolic extraction
    if feature_subset is not None:
        # only keep features that exist in the data
        valid_subset = [f for f in feature_subset if f in X_train.columns]
        if len(valid_subset) < len(feature_subset):
            missing = set(feature_subset) - set(valid_subset)
            logger.warning("Feature subset: %d missing features: %s", len(missing), missing)
        if valid_subset:
            X_train = X_train[valid_subset]
            selected_features = valid_subset
            logger.info(
                "Feature subset applied: %d → %d features.",
                len(selected_features), len(valid_subset),
            )
        else:
            logger.warning("No valid features in subset. Using all selected features.")

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
def train_pykan(
    dataset: dict,
    n_features: int,
    n_classes: int = 2,
    use_multkan: bool = False,
    tuned_kan_params: dict | None = None,
):
    """Train a fresh PyKAN model for symbolic extraction.

    Architecture selection priority:
    1. If ``tuned_kan_params`` is provided (from CPCV tuning), uses the tuned
       ``width1`` (capped at ``PYKAN_SYMBOLIC_WIDTH_CAP``) with simplifications:
       drops ``width2`` and forces ``grid=PYKAN_SYMBOLIC_FORCE_GRID``. This
       keeps the symbolic model architecturally linked to the benchmarked
       CPCV KAN while ensuring formulas remain tractable for sympy.
    2. Otherwise, falls back to data-aware sizing: hidden width chosen so
       ``n_train / total_params >= PYKAN_MIN_SAMPLES_PER_PARAM``.

    In either case, the final width is additionally clamped by the
    data-aware safety floor to prevent memorization.

    Parameters
    ----------
    use_multkan : bool
        If True, use MultKAN (KAN 2.0) with multiplication nodes instead
        of standard additive KAN. MultKAN can discover multiplicative
        relationships (e.g., rsi * stoch_k) that standard KAN cannot
        represent without fragile log/exp decomposition. Same symbolic
        extraction pipeline works for both.
    tuned_kan_params : dict, optional
        Best params dict from CPCV tuning for this fold, e.g.
        ``{"width1": 10, "width2": 4, "grid": 5, "lr": 0.01, ...}``.
        Only ``width1`` is consulted; ``width2`` and ``grid`` are
        deliberately overridden for symbolic tractability.

    Phase 1: Adam with weight decay + input noise (generalization)
    Grid extension: only if dataset is large enough
    Phase 2a: LBFGS warmup (short, no regularization)
    Phase 2b: LBFGS sparsity (short, gentle L1 + entropy)
    """
    if use_multkan:
        from kan import MultKAN as KANClass
        model_type = "MultKAN"
    else:
        from kan import KAN as KANClass
        model_type = "KAN"

    X_t = dataset["train_input"]
    y_t = dataset["train_label"].long()
    X_val_t = dataset["test_input"]
    y_val_t = dataset["test_label"].long()
    n_train = X_t.shape[0]

    # ── Data-aware safety floor ───────────────────────────────────────
    # Regardless of source, we want n_train / total_params >= PYKAN_MIN_SAMPLES_PER_PARAM.
    # Used both as fallback sizing and as a clamp on tuned widths.
    params_per_edge = PYKAN_SYMBOLIC_FORCE_GRID + KAN_K
    max_edges = n_train // PYKAN_MIN_SAMPLES_PER_PARAM
    max_hidden_safety = max_edges // (params_per_edge * (n_features + n_classes))
    max_hidden_safety = max(2, max_hidden_safety)  # never below 2

    # ── Architecture selection ────────────────────────────────────────
    tuned_w1 = None
    if tuned_kan_params is not None:
        tuned_w1 = tuned_kan_params.get("width1")

    if tuned_w1 is not None:
        # Use tuned width1, but apply simplifications + safety clamp
        hidden = min(tuned_w1, PYKAN_SYMBOLIC_WIDTH_CAP, max_hidden_safety)
        hidden = max(2, hidden)  # enforce minimum
        arch_source = (
            f"tuned (raw width1={tuned_w1} → "
            f"capped at {PYKAN_SYMBOLIC_WIDTH_CAP}, safety floor {max_hidden_safety})"
        )
    else:
        # Fallback: data-aware sizing (original behavior), clamped by KAN_HIDDEN
        hidden = min(max_hidden_safety, KAN_HIDDEN)
        hidden = max(2, hidden)
        arch_source = "data-aware fallback (no tuned params)"

    width = [n_features, hidden, n_classes]
    total_edges = n_features * hidden + hidden * n_classes
    total_params_est = total_edges * params_per_edge

    print(
        f"    {model_type} architecture: {width} "
        f"[{arch_source}]"
    )
    print(
        f"    {total_edges} edges, ~{total_params_est} params for {n_train} samples, "
        f"ratio={n_train/max(total_params_est,1):.1f}x"
    )
    if PYKAN_SYMBOLIC_DROP_WIDTH2 and tuned_kan_params is not None:
        tuned_w2 = tuned_kan_params.get("width2", 0) or 0
        if tuned_w2 > 0:
            print(
                f"    Note: dropped tuned width2={tuned_w2} "
                f"(PYKAN_SYMBOLIC_DROP_WIDTH2=True for tractable formulas)."
            )
    if tuned_kan_params is not None:
        tuned_grid = tuned_kan_params.get("grid")
        if tuned_grid is not None and tuned_grid != PYKAN_SYMBOLIC_FORCE_GRID:
            print(
                f"    Note: forcing grid={PYKAN_SYMBOLIC_FORCE_GRID} "
                f"(tuned grid={tuned_grid} overridden for symbolic clarity)."
            )

    if n_train / max(total_params_est, 1) < 2:
        print(
            f"    ⚠ WARNING: samples/params ratio < 2. "
            f"Memorization is very likely."
        )

    if use_multkan:
        model = KANClass(
            width=width, grid=PYKAN_SYMBOLIC_FORCE_GRID, k=KAN_K,
            seed=42, mult_arity=2,
        )
    else:
        model = KANClass(
            width=width, grid=PYKAN_SYMBOLIC_FORCE_GRID, k=KAN_K, seed=42,
        )

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
            logger.info("Grid extended: %d → %d", PYKAN_SYMBOLIC_FORCE_GRID, KAN_GRID)
            print(f"    Grid extended: {PYKAN_SYMBOLIC_FORCE_GRID} → {KAN_GRID}")
            _log_diagnostic("After grid extend", model, X_t, y_t, X_val_t, y_val_t)
        except (AttributeError, TypeError, Exception) as e:
            logger.warning("Grid extension failed (%s).", e)
            print(f"    Grid extension failed: {e}")
    else:
        print(
            f"    Grid extension SKIPPED (n_train={n_train}, "
            f"grid stays at {PYKAN_SYMBOLIC_FORCE_GRID}). "
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
    logger.info("%s trained. Val accuracy: %.4f, width: %s", model_type, val_acc, width)
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
    fallback_count = 0               # count edges that needed PyKAN default lib
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
                except (KeyError, Exception) as e:
                    # custom library may contain names PyKAN doesn't know;
                    # fall back to PyKAN's built-in default library
                    try:
                        suggestions = model.suggest_symbolic(
                            l, i, j, topk=SYMBOLIC_TOPK,
                        )
                        if fallback_count < 3:
                            print(
                                f"    ℹ Edge ({l},{i},{j}): custom lib failed "
                                f"({type(e).__name__}), used PyKAN defaults."
                            )
                        fallback_count += 1
                    except Exception as e2:
                        if skipped_edges < 5:
                            print(
                                f"    ⚠ suggest_symbolic({l},{i},{j}) error: "
                                f"{type(e2).__name__}: {e2}"
                            )
                        logger.debug("suggest_symbolic(%d,%d,%d) failed: %s", l, i, j, e2)
                        skipped_edges += 1
                        continue

                if suggestions is None:
                    skipped_edges += 1
                    continue

                # ── safe parsing of suggestions ───────────────────
                # PyKAN's suggest_symbolic() return type varies by version:
                #   - Some versions: DataFrame (rows = candidates)
                #   - Some versions: flat tuple (fn_name, r2, r2_loss, ...)
                #   - Some versions: nested tuple of tuples
                # We handle all cases and always skip the constant "0".
                best_fn = None
                best_r2 = 0.0

                try:
                    # debug: dump full structure for first 2 edges
                    if total_edges <= 2:
                        print(
                            f"    [DEBUG] Edge ({l},{i},{j}): "
                            f"type={type(suggestions).__name__}, "
                            f"repr={repr(suggestions)[:200]}"
                        )

                    # ── CASE 1: DataFrame ──────────────────────────────
                    if hasattr(suggestions, "to_dict"):
                        records = suggestions.to_dict("records")
                        if not records:
                            skipped_edges += 1
                            continue

                        sample_keys = list(records[0].keys())
                        fn_key = sample_keys[0]
                        r2_key = None
                        for k in sample_keys:
                            if "r2" in str(k).lower() and "loss" not in str(k).lower():
                                r2_key = k
                                break

                        for rec in records:
                            fn_name = str(rec.get(fn_key, ""))
                            if fn_name == "0":
                                continue
                            if r2_key is not None:
                                try:
                                    r2_val = float(rec[r2_key])
                                except (ValueError, TypeError):
                                    r2_val = 0.0
                                r2_val = max(0.0, min(1.0, r2_val))
                            else:
                                r2_val = 0.0
                            if r2_val > best_r2:
                                best_r2 = r2_val
                                best_fn = fn_name

                    # ── CASE 2: flat tuple (fn_name, r2, ...) ─────────
                    elif isinstance(suggestions, (tuple, list)):
                        if len(suggestions) == 0:
                            skipped_edges += 1
                            continue

                        first = suggestions[0]

                        if isinstance(first, str):
                            # flat tuple: ('cos', <fitted_lambdas>, R², complexity)
                            # index: [0]=name, [1]=lambdas, [2]=R², [3]=complexity
                            fn_name = first
                            if fn_name != "0" and len(suggestions) > 2:
                                try:
                                    r2_val = float(suggestions[2])
                                except (ValueError, TypeError):
                                    r2_val = 0.0
                                r2_val = max(0.0, min(1.0, r2_val))
                                best_fn = fn_name
                                best_r2 = r2_val
                            elif fn_name == "0":
                                # "0" won by total_loss due to zero complexity,
                                # but non-constant functions may have excellent R².
                                # Brute-force: try each candidate via fix_symbolic,
                                # capture the R² PyKAN prints, keep the best.
                                original_state = copy.deepcopy(model.state_dict())
                                best_direct_fn = None
                                best_direct_r2 = 0.0

                                for candidate in SYMBOLIC_LIBRARY:
                                    if candidate == "0":
                                        continue
                                    try:
                                        # capture stdout to extract R²
                                        old_stdout = sys.stdout
                                        sys.stdout = buffer = io.StringIO()
                                        try:
                                            model.fix_symbolic(l, i, j, candidate)
                                            output = buffer.getvalue()
                                        finally:
                                            sys.stdout = old_stdout

                                        # parse "r2 is 0.XXXX" from PyKAN's output
                                        r2_match = re.search(r"r2 is ([\d.eE+-]+)", output)
                                        if r2_match:
                                            cand_r2 = float(r2_match.group(1))
                                            cand_r2 = max(0.0, min(1.0, cand_r2))
                                            if cand_r2 > best_direct_r2:
                                                best_direct_r2 = cand_r2
                                                best_direct_fn = candidate

                                        # restore original state for next candidate
                                        model.load_state_dict(original_state)
                                    except Exception:
                                        model.load_state_dict(original_state)
                                        continue

                                if best_direct_fn is not None and best_direct_r2 >= SYMBOLIC_R2_THRESHOLD:
                                    best_fn = best_direct_fn
                                    best_r2 = best_direct_r2

                        elif isinstance(first, (tuple, list)):
                            # nested: (('cos', lambdas, R², complexity), ...)
                            for entry in suggestions:
                                if not isinstance(entry, (tuple, list)) or len(entry) < 3:
                                    continue
                                fn_name = str(entry[0])
                                if fn_name == "0":
                                    continue
                                try:
                                    r2_val = float(entry[2])
                                except (ValueError, TypeError):
                                    r2_val = 0.0
                                r2_val = max(0.0, min(1.0, r2_val))
                                if r2_val > best_r2:
                                    best_r2 = r2_val
                                    best_fn = fn_name

                        elif hasattr(first, "to_dict"):
                            # tuple containing a DataFrame as first element
                            records = first.to_dict("records")
                            sample_keys = list(records[0].keys())
                            fn_key = sample_keys[0]
                            r2_key = None
                            for k in sample_keys:
                                if "r2" in str(k).lower() and "loss" not in str(k).lower():
                                    r2_key = k
                                    break
                            for rec in records:
                                fn_name = str(rec.get(fn_key, ""))
                                if fn_name == "0":
                                    continue
                                if r2_key:
                                    try:
                                        r2_val = float(rec[r2_key])
                                    except (ValueError, TypeError):
                                        r2_val = 0.0
                                    r2_val = max(0.0, min(1.0, r2_val))
                                else:
                                    r2_val = 0.0
                                if r2_val > best_r2:
                                    best_r2 = r2_val
                                    best_fn = fn_name
                        else:
                            skipped_edges += 1
                            continue
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
        f"  [skipped: {skipped_edges}, fallback to defaults: {fallback_count}]"
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
    # ── Build explicit sympy variable list for symbolic_formula() ──────
    # PyKAN internally evaluates expressions containing variable names
    # (e.g. x1, x2, ...) via sympy.  Without passing a `var=` list,
    # those names are undefined in its namespace, causing:
    #   NameError: name 'x1' is not defined
    n_inputs = len(feature_names)
    # Try multiple naming conventions that PyKAN might expect internally
    # Convention 1: x1, x2, ... (no underscore, 1-based)
    var_symbols_x = [sympy.Symbol(f"x{i+1}") for i in range(n_inputs)]
    # Convention 2: x_1, x_2, ... (underscore, 1-based)
    var_symbols_x_ = [sympy.Symbol(f"x_{i+1}") for i in range(n_inputs)]
    # Convention 3: x_0, x_1, ... (underscore, 0-based)
    var_symbols_x0 = [sympy.Symbol(f"x_{i}") for i in range(n_inputs)]

    formulas = None
    used_vars = None
    for var_candidate in [var_symbols_x, var_symbols_x_, var_symbols_x0]:
        try:
            formulas = model.symbolic_formula(var=var_candidate)
            used_vars = var_candidate
            break
        except Exception:
            continue

    # last resort: try without var= (may work on some PyKAN versions)
    if formulas is None:
        try:
            formulas = model.symbolic_formula()
            used_vars = None
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

    # ── Substitute placeholder variables with feature names ───────────
    # If we passed var= explicitly, we know exactly which symbols to sub.
    # Otherwise, detect the naming convention from the formula.
    if used_vars is not None:
        for old_sym, name in zip(used_vars, feature_names):
            new_sym = sympy.Symbol(name)
            try:
                logit_bearish = logit_bearish.subs(old_sym, new_sym)
                logit_bullish = logit_bullish.subs(old_sym, new_sym)
            except (AttributeError, TypeError):
                pass
    else:
        # Fallback: detect convention from free symbols
        all_symbols = set()
        for expr in [logit_bearish, logit_bullish]:
            if hasattr(expr, "free_symbols"):
                all_symbols |= expr.free_symbols

        symbol_names = {str(s) for s in all_symbols}
        has_x0 = "x_0" in symbol_names
        has_xn = f"x_{len(feature_names)}" in symbol_names

        # if x_{n} exists but x_0 doesn't → 1-based indexing
        offset = 1 if (has_xn and not has_x0) else 0
        if offset == 1:
            logger.info("Detected PyKAN 1-based variable naming (x_1..x_%d).", len(feature_names))

        for i, name in enumerate(feature_names):
            old = sympy.Symbol(f"x_{i + offset}")
            new = sympy.Symbol(name)
            try:
                logit_bearish = logit_bearish.subs(old, new)
                logit_bullish = logit_bullish.subs(old, new)
            except (AttributeError, TypeError):
                pass

    # ── simplification with timeout (sympy can hang on complex expressions) ──
    def _simplify_with_timeout(expr, timeout_sec=30):
        """Run sympy.simplify with a timeout. Returns original expr if timeout."""
        result = [expr]  # mutable container for thread result

        def _worker():
            try:
                result[0] = sympy.simplify(expr)
            except Exception:
                pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)
        if t.is_alive():
            logger.warning("sympy.simplify timed out after %ds. Using unsimplified.", timeout_sec)
            print(f"    ⚠ sympy.simplify timed out after {timeout_sec}s. Skipping.")
        return result[0]

    decision = logit_bullish - logit_bearish
    decision = _simplify_with_timeout(decision, timeout_sec=30)

    # nsimplify for cleaner rational numbers (also with timeout protection)
    try:
        logit_bearish = sympy.nsimplify(logit_bearish, tolerance=1e-3, rational=False)
        logit_bullish = sympy.nsimplify(logit_bullish, tolerance=1e-3, rational=False)
        decision = sympy.nsimplify(decision, tolerance=1e-3, rational=False)
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
    n_top_features: int | None = None,
    use_multkan: bool = False,
    fold_selection: str | int = "best",
) -> dict:
    """Run the full symbolic extraction pipeline (Algorithm 1).

    Called from the notebook after CPCV results are available.
    The CPCV pipeline used efficient-kan; this retrains a fresh PyKAN
    on the best fold for symbolic analysis.

    Parameters
    ----------
    n_top_features : int, optional
        If provided, only the top N most stable features (by CPCV selection
        frequency) are used for symbolic extraction. Recommended values:
        5-7 for interpretable formulas, 10 for moderate complexity.
        If None, all features selected by the best fold are used.
    use_multkan : bool
        If True, use MultKAN (KAN 2.0) with multiplication nodes.
        MultKAN can discover multiplicative feature interactions
        (e.g., rsi * stoch_k) that standard KAN cannot represent
        efficiently. Default: False (standard additive KAN).
    fold_selection : str or int
        Which CPCV fold to use for extraction:
        - "best": fold with highest KAN F1 macro (default)
        - "last": last fold (most recent data, rolling-window style)
        - int: specific fold index (e.g., 0, 5, 14)
    """
    model_label = "MultKAN" if use_multkan else "PyKAN"
    print("=" * 60)
    print("Symbolic Extraction (VIX KAN Paper, Algorithm 1)")
    print(f"  CPCV model: efficient-kan | Extraction model: {model_label}")
    print("=" * 60)

    # 1. select fold
    best_split, prep_info = select_extraction_fold(cpcv_results, fold_selection=fold_selection)

    # 1b. retrieve tuned KAN params for this fold (if available)
    tuned_kan_params = None
    tuning_results = cpcv_results.get("tuning_results", {})
    if tuning_results and best_split in tuning_results:
        kan_tuning = tuning_results[best_split].get("kan", {})
        tuned_kan_params = kan_tuning.get("best_params")
    if tuned_kan_params:
        print(f"\n  Tuned KAN params for split {best_split}: {tuned_kan_params}")
    else:
        print(f"\n  No tuned KAN params found for split {best_split} "
              f"(falling back to data-aware sizing).")

    # 2. rank features and select top N if requested
    feature_subset = None
    if n_top_features is not None:
        ranked = rank_features_by_stability(cpcv_results)
        if ranked:
            feature_subset = [f for f, _ in ranked[:n_top_features]]
            print(f"\n  Feature ranking (top {n_top_features} of {len(ranked)}):")
            for i, (feat, freq) in enumerate(ranked[:n_top_features]):
                print(f"    {i+1}. {feat} ({freq:.0%} selection frequency)")
            if len(ranked) > n_top_features:
                print(f"    ... ({len(ranked) - n_top_features} features excluded)")
        else:
            print("  ⚠ Could not rank features. Using all selected features.")

    # 3. prepare data
    dataset, feature_names = prepare_extraction_data(
        X, y, w, t1, cpcv_results, best_split, prep_info,
        feature_subset=feature_subset,
    )
    print(
        f"  Data: {dataset['train_input'].shape[0]} train, "
        f"{dataset['test_input'].shape[0]} val, "
        f"{len(feature_names)} features: {feature_names}"
    )

    # 3. train model from scratch
    print(f"\n  Step 1: Training {model_label} (Adam → grid extend → LBFGS)...")
    model = train_pykan(
        dataset,
        n_features=len(feature_names),
        use_multkan=use_multkan,
        tuned_kan_params=tuned_kan_params,
    )

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
    print(f"  Model:                  {model_label}")
    print(f"  Fold:                   split {best_split} (selection='{fold_selection}')")
    if n_top_features is not None:
        print(f"  Features used:          {n_top_features} (top by CPCV stability)")
    else:
        print(f"  Features used:          {len(feature_names)} (all selected)")
    print(f"  Feature names:          {feature_names}")
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