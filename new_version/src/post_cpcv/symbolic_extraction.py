"""
15) Symbolic Extraction
================================
Take the best CPCV fold (or full dataset), retrain a fresh PyKAN model with
the same architecture as the efficient-kan used in CPCV, then apply Algorithm 1
from the VIX KAN paper:
  1. Train with staged optimiser (Adam → optional grid extend → LBFGS)
  2. Prune low-importance edges and nodes
  3. Symbolify activation functions with closed-form candidates
  4. Fine-tune affine parameters

The CPCV pipeline uses efficient-kan (standard ``nn.Module``, AdamW); this
module uses PyKAN independently because only PyKAN supports ``prune()``,
``suggest_symbolic()``, ``fix_symbolic()``, and ``symbolic_formula()``. Both
share the same ``[n_features, HIDDEN, n_classes]`` architecture and B-spline
basis, so the symbolic re-training is faithful to the CPCV-evaluated model.
"""

import contextlib
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
from tqdm.auto import tqdm

from src.cpcv.cv import generate_cpcv_splits
from src.cpcv.preprocessing import apply_ffd

# Shared architecture constants between efficient-kan and PyKAN.
from src.cpcv.models.kan_model import KAN_HIDDEN, KAN_GRID, KAN_K

logger = logging.getLogger(__name__)


# Context manager to suppress pykan's internal print() noise (suggest_symbolic's pandas table,
# "saving model version 0.x", "checkpoint directory created", etc). Use sparingly and only
# around pykan API calls; our own logger.warning lines go to stderr so they survive this redirect.
@contextlib.contextmanager
def _suppress_pykan_stdout():
    """Redirect stdout to /dev/null during pykan calls; stderr (warnings) unaffected."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf

# --- PyKAN-specific training constants -------------------------------------
# Phase 1: Adam with weight decay (main learning phase, no KAN-specific regularisation).
PYKAN_ADAM_STEPS = 600
PYKAN_ADAM_LR = 1e-3
PYKAN_ADAM_WEIGHT_DECAY = 1e-3
PYKAN_NOISE_STD = 0.05            # Gaussian input noise (dropout-like regulariser for small data)

# Phase 2: LBFGS (short refinement only; LBFGS will memorise on small datasets if run too long).
PYKAN_LBFGS_STEPS = 40
PYKAN_LBFGS_LR = 0.01
PYKAN_LBFGS_WARMUP_FRAC = 0.5     # first 50% of LBFGS steps: no sparsity regularisation
PYKAN_LAMB = 0.002                # gentle sparsity multiplier
PYKAN_LAMB_L1 = 1.0
PYKAN_LAMB_ENTROPY = 2.0

PYKAN_PATIENCE = 10
PYKAN_VAL_INTERVAL = 5
PYKAN_GRID_INIT = 3
PYKAN_GRID_EXTEND = True          # Default-intent: enable grid extension. The runtime gate
                                  # `n_train > 1000` in train_pykan handles small-data safety,
                                  # so this constant being True is safe for the project's
                                  # current ~358-sample data (the gate skips automatically).
                                  # Flip to False to disable unconditionally.

# Data-aware architecture sizing.
PYKAN_MIN_SAMPLES_PER_PARAM = 5   # target at least 5 training samples per parameter
PYKAN_HIDDEN_OVERRIDE = None      # set dynamically in train_pykan()

# --- Symbolic extraction architecture controls ------------------------------
# Defaults honour the tuned width1 / width2 / grid from CPCV so the symbolic formula
# represents the model that actually won the CPCV evaluation. Override toggles below
# revert to a simplified-for-readability mode when sympy struggles with the tuned depth.
PYKAN_SYMBOLIC_WIDTH_CAP = 8       # safety ceiling on width1; matches tuning maximum
                                   # so it never bites in practice, but guards against
                                   # future tuning expansions producing intractable widths.
PYKAN_SYMBOLIC_DROP_WIDTH2 = False # set True to force width2=0 regardless of tuning
                                   # (legacy single-hidden-layer behaviour; use when
                                   # the tuned depth produces nested compositions sympy
                                   # cannot simplify in 30s).
PYKAN_SYMBOLIC_FORCE_GRID = None   # set int (e.g. 3) to force a specific grid density;
                                   # default None honours the tuned grid.
PYKAN_FALLBACK_GRID = 3            # used only when both PYKAN_SYMBOLIC_FORCE_GRID is None
                                   # AND no tuned grid is available (CPCV ran untuned).

# Accuracy gate: skip symbolification if PyKAN can't even beat random.
PYKAN_MIN_ACCURACY = 0.53

# --- Symbolic extraction constants -----------------------------------------
PRUNE_THRESHOLD = 0.01

# Library of symbolic primitives PyKAN tries to fit on each edge.
# Note: 'sigmoid' and 'x*abs(x)' are NOT in PyKAN's internal SYMBOLIC_LIB and cause
# KeyError; '1/x' can produce division-by-zero. Only names PyKAN recognises natively.
SYMBOLIC_LIBRARY = [
    "x", "x^2", "x^3", "x^4",     # polynomials
    "exp", "log", "sqrt",          # standard transforms
    "tanh", "sin", "cos",          # bounded nonlinearities
    "abs", "sgn",                  # piecewise
    "arctan",                      # bounded monotonic
    "0",                           # constant (zero)
]
SYMBOLIC_R2_THRESHOLD = 0.3        # below this R² the edge keeps its spline
SYMBOLIC_TOPK = 5

AFFINE_FINETUNE_STEPS = 30
AFFINE_LR = 0.0004                 # from VIX paper

CACHE_DIR = "cache/"


# --- 1. Diagnostic helpers --------------------------------------------------
# Classification accuracy under no_grad; used at every staged-training checkpoint.
def _compute_accuracy(model, X: torch.Tensor, y: torch.Tensor) -> float:
    """Return classification accuracy of ``model`` on ``(X, y)`` under ``torch.no_grad``."""
    model.eval()
    with torch.no_grad():
        pred = model(X)
        acc = (pred.argmax(dim=1) == y).float().mean().item()
    return acc


# Count total and "active" edges (activation coefficient above threshold); diagnostic for over-regularisation.
def _count_active_edges(model, threshold: float) -> tuple[int, int]:
    """Return ``(total_edges, active_edges)`` for the KAN; ``(-1, -1)`` if the structure can't be inspected.

    An edge is "active" when its activation coefficient norm exceeds ``threshold``, signalling
    non-trivial signal flow. Low survival rates suggest the sparsity regulariser was too aggressive.
    """
    total = 0
    active = 0
    try:
        # Walk every (layer, in_node, out_node) triple in the KAN.
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
                        # Inspect the activation coefficient magnitude; PyKAN versions vary in storage.
                        act = model.act_fun[l]
                        if hasattr(act, 'coef'):
                            coef_norm = act.coef[j, i].abs().mean().item()
                        else:
                            coef_norm = 1.0  # treat as active when introspection fails
                        if coef_norm > threshold:
                            active += 1
                    except (IndexError, AttributeError, RuntimeError):
                        active += 1
    except (AttributeError, TypeError):
        return -1, -1
    return total, active


# Per-checkpoint diagnostic: returns (train_acc, val_acc); only prints when verbose=True.
def _log_diagnostic(label: str, model, X_train, y_train, X_val, y_val, verbose: bool = True):
    """Compute train + val accuracy. Print only when ``verbose``; always log to file."""
    train_acc = _compute_accuracy(model, X_train, y_train)
    val_acc = _compute_accuracy(model, X_val, y_val)
    if verbose:
        print(f"    [{label}] train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    logger.info("%s: train_acc=%.4f, val_acc=%.4f", label, train_acc, val_acc)
    return train_acc, val_acc


# --- 2. Fold selection ------------------------------------------------------
# Pick which CPCV fold's KAN to re-train symbolically; default is the best-F1 fold.
def select_extraction_fold(
    cpcv_results: dict,
    fold_selection: str | int = "best",
) -> tuple[int, dict]:
    """Choose the CPCV split for symbolic extraction; return ``(split_idx, prep_info)``.

    ``fold_selection``:
      - ``"best"``  : split with highest KAN F1 macro (default)
      - ``"last"``  : most recent split (rolling-window style)
      - ``int``     : specific split index
    """
    predictions = cpcv_results["predictions"]
    n_splits = cpcv_results["n_splits"]
    n_seeds = cpcv_results["n_seeds"]

    # Build per-split F1 by seed-averaging the stored KAN entries.
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

    # Fallback: no KAN predictions stored at all.
    if not split_f1s:
        logger.warning("No KAN predictions found. Using first available split.")
        for key, pred in predictions.items():
            return key[1], pred.get("prep_info", {})

    logger.info(
        "Fold F1 scores: %s",
        {k: f"{v:.4f}" for k, v in split_f1s.items()},
    )

    # Dispatch on fold_selection mode.
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


# --- 3. Feature selection ---------------------------------------------------
# Rank features by how often they were MDA-selected across all KAN folds.
def rank_features_by_stability(cpcv_results: dict) -> list[tuple[str, float]]:
    """Return ``[(feature_name, selection_frequency), ...]`` sorted descending."""
    predictions = cpcv_results["predictions"]
    feature_counts = {}
    total_folds = 0

    # Walk every KAN prediction entry and tally selected features.
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


# Resolve which features to feed PyKAN; supports two strategies with different methodological framings.
def select_features_for_extraction(
    cpcv_results: dict,
    fold_idx: int,
    n_top_features: int | None = None,
    strategy: str = "per_fold",
) -> list[str]:
    """Return the feature list for symbolic re-training under one of two strategies.

    ``per_fold`` (default) uses the MDA selection from the chosen extraction fold itself, so
    the symbolic formula represents the actual model whose performance is reported in the
    comparison table. If the per-fold selection contains more features than ``n_top_features``,
    the cap is enforced by ranking the fold's selection by cross-fold stability (highest wins,
    ties broken alphabetically). This keeps the formula focused on the features that both this
    fold valued AND that consistently mattered in other folds.

    ``stability`` (legacy) uses the top ``n_top_features`` by selection frequency across all KAN
    CPCV folds, so the symbolic formula reflects features robustly important across the dataset's
    history rather than features specific to one fold. Falls back to ``stability`` automatically
    if ``per_fold`` is requested but no MDA selection is available for the chosen fold.
    """
    if strategy == "per_fold":
        # prep_info is shared across models within a fold, so any model entry serves.
        prep_info_for_fold = None
        for key, pred in cpcv_results["predictions"].items():
            if key[1] != fold_idx:
                continue
            pi = pred.get("prep_info", {})
            if pi.get("selected_features"):
                prep_info_for_fold = pi
                break

        if prep_info_for_fold is None:
            logger.warning(
                "No MDA selection found for fold %d; falling back to "
                "stability ranking.", fold_idx,
            )
            strategy = "stability"
        else:
            fold_selected = list(prep_info_for_fold["selected_features"])
            if n_top_features is None or n_top_features >= len(fold_selected):
                return fold_selected
            # Cap: intersect fold-local MDA selection with cross-fold stability ranking.
            stability = dict(rank_features_by_stability(cpcv_results))
            scored = [
                (f, stability.get(f, 0.0))
                for f in fold_selected
            ]
            scored.sort(key=lambda x: (-x[1], x[0]))  # desc by freq, ties alpha
            return [f for f, _ in scored[:n_top_features]]

    # strategy == "stability" (either requested directly or falling back).
    ranked = rank_features_by_stability(cpcv_results)
    if not ranked:
        return []
    if n_top_features is None:
        return [f for f, _ in ranked]
    return [f for f, _ in ranked[:n_top_features]]


# --- 4. Data preparation ---------------------------------------------------
# Reconstruct the preprocessed training fold (FFD + scaler + MDA subset + tanh normalisation).
def prepare_extraction_data(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    t1: pd.Series,
    cpcv_results: dict,
    best_split_idx: int,
    prep_info: dict,
    feature_subset: list[str] | None = None,
) -> tuple[dict, list[str]]:
    """Rebuild the training fold for the chosen split and return ``(dataset, feature_names)``.

    Applies FFD then scaling then tanh normalisation, matching both efficient-kan and PyKAN
    input preprocessing. ``feature_subset`` overrides the per-fold MDA selection (used for the
    cross-fold stability strategy); if omitted, the per-fold MDA selection is used.
    """
    # S2: read split parameters from cpcv_results['split_info'] rather than relying on
    # the module-level defaults of generate_cpcv_splits. The defaults match today's locked
    # N=8, k=2, embargo=0.01 config so this is currently a no-op, but the previous code was
    # fragile: any change to cv.py's N_GROUPS/K_TEST_GROUPS constants would silently put the
    # extraction on a different split list than CPCV evaluated, making fold-selection meaningless.
    split_info = cpcv_results.get("split_info", {})
    splits = generate_cpcv_splits(
        X, t1,
        n_groups=split_info.get("n_groups", 8),
        k=split_info.get("k", 2),
        embargo_pct=split_info.get("embargo_pct", 0.01),
    )
    train_idx, _ = splits[best_split_idx]

    # Coerce ``y`` to a Series aligned to X.index regardless of input type (np.ndarray, Series, etc.).
    # The notebook can pass either depending on its assembly path, and downstream .loc lookups need
    # the index alignment.
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

    # Apply FFD to the full series before slicing the training fold.
    X_transformed = X.copy()
    for col, d_star in ffd_info.items():
        if col in X_transformed.columns:
            X_transformed[col] = apply_ffd(X_transformed[col], d_star)

    X_train = X_transformed.iloc[train_idx].copy()
    X_train = X_train.loc[X_train.notna().all(axis=1)]
    y_train = y_mapped.loc[X_train.index]

    # Apply the stored scaler so the symbolic re-training sees the same transform as the pipeline.
    if scaler is not None:
        X_train = pd.DataFrame(
            scaler.transform(X_train),
            index=X_train.index, columns=X_train.columns,
        )

    # Feature-subset resolution. When ``feature_subset`` is provided (typically the cross-fold
    # stability ranking), allow any column present in the scaled X, even if NOT selected by MDA
    # for this fold. The scaler is fitted on the full feature matrix before MDA, so every original
    # column is available in scaled form. When ``feature_subset`` is None, fall back to the per-fold
    # MDA selection — the right default for diagnostic uses that mirror the pipeline exactly.
    if feature_subset is not None:
        valid_subset = [f for f in feature_subset if f in X_train.columns]
        if len(valid_subset) < len(feature_subset):
            missing = set(feature_subset) - set(valid_subset)
            logger.warning(
                "Feature subset: %d feature(s) requested but not present "
                "in the post-FFD scaled matrix: %s. These will be skipped.",
                len(missing), missing,
            )
        if valid_subset:
            X_train = X_train[valid_subset]
            selected_features = valid_subset
            logger.info(
                "Symbolic extraction feature subset: %d feature(s) "
                "(overrides per-fold MDA selection).",
                len(selected_features),
            )
        else:
            logger.warning(
                "No valid features in subset; falling back to per-fold "
                "MDA-selected features (%d features).",
                len(selected_features),
            )
            X_train = X_train[selected_features]
    else:
        X_train = X_train[selected_features]

    # 80/20 train/val split. The CPCV pipeline uses 70/15/15 because its KAN predictions get
    # fed through a calibration step, but pykan's symbolic extraction consumes none of those
    # probabilistic outputs downstream — the val set's only role here is best-state tracking
    # inside train_pykan. So 80/20 (no calibration partition) is the right local choice,
    # not a stale leftover from a previous pipeline convention.
    cal_boundary = int(len(X_train) * 0.8)
    X_model = X_train.iloc[:cal_boundary]
    X_val = X_train.iloc[cal_boundary:]
    y_model = y_train.iloc[:cal_boundary]
    y_val = y_train.iloc[cal_boundary:]

    X_model_t = torch.tensor(X_model.values, dtype=torch.float32)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)

    # Tanh normalisation fit on the training split, applied to both train and val.
    input_mean = X_model_t.mean(dim=0)
    input_std = X_model_t.std(dim=0) + 1e-8
    X_model_t = torch.tanh((X_model_t - input_mean) / input_std)
    X_val_t = torch.tanh((X_val_t - input_mean) / input_std)

    # S1: build the raw → tanh transform parameters per feature so downstream sensitivity
    # helpers can substitute correctly. The full chain is:
    #   x_raw -> scaler(RobustScaler) -> (x_raw - median) / iqr  =: x_scaled
    #   x_scaled -> tanh-normalise   -> tanh((x_scaled - input_mean) / input_std) =: z
    # Combined: z = tanh((x_raw - a) / b) where
    #   a = median + input_mean * iqr   (raw value mapping to z = 0)
    #   b = iqr * input_std             (raw scale factor)
    # If no scaler was used, a = input_mean and b = input_std directly.
    feature_a: dict[str, float] = {}
    feature_b: dict[str, float] = {}
    input_mean_np = input_mean.cpu().numpy()
    input_std_np = input_std.cpu().numpy()

    if scaler is not None:
        # scaler.center_ / scaler.scale_ are sized to whatever X had at scaler-fit time
        # (all 73 features in this project's pipeline). We need each selected feature's
        # index in that original column ordering to pull the right entry.
        scaler_cols = list(X.columns)
        for i, feat in enumerate(selected_features):
            try:
                col_idx = scaler_cols.index(feat)
                center_i = float(scaler.center_[col_idx])
                scale_i = float(scaler.scale_[col_idx]) if scaler.scale_[col_idx] != 0 else 1.0
                feature_a[feat] = center_i + float(input_mean_np[i]) * scale_i
                feature_b[feat] = scale_i * float(input_std_np[i])
            except (ValueError, IndexError, AttributeError):
                # Defensive fallback: feature missing from scaler's column ordering, or
                # scaler doesn't expose center_/scale_ (non-RobustScaler in future).
                feature_a[feat] = float(input_mean_np[i])
                feature_b[feat] = float(input_std_np[i])
    else:
        for i, feat in enumerate(selected_features):
            feature_a[feat] = float(input_mean_np[i])
            feature_b[feat] = float(input_std_np[i])

    dataset = {
        "train_input": X_model_t,
        "train_label": torch.tensor(y_model.values, dtype=torch.float32),
        "test_input": X_val_t,
        "test_label": torch.tensor(y_val.values, dtype=torch.float32),
        # S1: the transform parameters travel WITH the dataset so any downstream code that
        # uses the formula can convert raw inputs into the tanh-normalised space the formula
        # actually lives in. Without these, sensitivity / marginal-effect computations
        # substitute raw values into a formula that expects normalised values and silently
        # produce wrong numbers.
        "input_transform": {
            "feature_a": feature_a,
            "feature_b": feature_b,
        },
    }

    feature_names = list(selected_features)
    logger.info(
        "Extraction data: %d train, %d val, %d features (tanh-normalized).",
        len(X_model), len(X_val), len(feature_names),
    )
    return dataset, feature_names


# --- 5. Training (Algorithm 1, Step 1) -------------------------------------
# Staged PyKAN training: Adam (Phase 1) → optional grid extend → LBFGS warmup → LBFGS sparsity.
def train_pykan(
    dataset: dict,
    n_features: int,
    n_classes: int = 2,
    use_multkan: bool = False,
    tuned_kan_params: dict | None = None,
):
    """Train a fresh PyKAN (or MultKAN) for symbolic extraction.

    METHODOLOGY NOTE (S3): the efficient-kan that ran during CPCV is not directly portable to
    pykan, because the two libraries use different spline parameterisations under the hood.
    What's transferred here is the *hyperparameter set* (width1, width2, grid, k), not the
    learned weights. The pykan is then retrained from scratch on the same training fold via
    the staged optimiser below. The extracted symbolic formula therefore represents a pykan
    with matching architecture and training data, whose decision boundary approximates but
    does not equal the CPCV-evaluated efficient-kan's. Pre/post symbolic accuracy and the
    pre-symbolic val accuracy reported during training quantify how close this approximation
    is for any given fold; large gaps signal the symbolic formula is not faithful to the
    CPCV-evaluated KAN and the thesis chapter should flag the fold accordingly.

    Architecture: when ``tuned_kan_params`` is provided from CPCV, the symbolic model honours
    the tuned ``width1`` / ``width2`` / ``grid``. ``width1`` is still capped at
    ``PYKAN_SYMBOLIC_WIDTH_CAP`` and clamped by a data-aware safety floor; ``width2`` and
    ``grid`` are honoured verbatim unless the override constants are set. Without tuned params,
    falls back to data-aware sizing so ``n_train / total_params >= PYKAN_MIN_SAMPLES_PER_PARAM``.

    ``use_multkan=True`` switches to MultKAN (KAN 2.0) with multiplication nodes, which can
    discover multiplicative interactions (e.g., ``rsi * stoch_k``) that standard additive KAN
    cannot represent without fragile log/exp decomposition. The same symbolic-extraction
    pipeline works for both.

    Phases:
      1. Adam with weight decay + input noise (the main generalisation phase).
      2. Grid extension, conditional on dataset size (gated to n_train > 1000 for small data).
      3. LBFGS warmup (short, no regularisation).
      4. LBFGS sparsity (short, gentle L1 + entropy).
    """
    if use_multkan:
        from kan import MultKAN as KANClass
        model_type = "MultKAN"
    else:
        from kan import KAN as KANClass
        model_type = "KAN"

    # S11: full reproducibility seeding. The KAN constructor's seed=42 covers only the
    # spline initialisation; the Adam optimiser, the input-noise tensor (line 631), and any
    # pykan-internal random calls are otherwise unseeded. Seeding all RNG sources at function
    # entry makes the extracted formula reproducible across runs on the same machine. Note
    # that determinism across machines also requires matching CUDA/cuDNN versions; this is
    # CPU-deterministic but not strictly cross-hardware-deterministic for GPU runs.
    import random
    _PYKAN_SEED = 42
    torch.manual_seed(_PYKAN_SEED)
    np.random.seed(_PYKAN_SEED)
    random.seed(_PYKAN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_PYKAN_SEED)

    X_t = dataset["train_input"]
    y_t = dataset["train_label"].long()
    X_val_t = dataset["test_input"]
    y_val_t = dataset["test_label"].long()
    n_train = X_t.shape[0]

    # Resolve grid first because the safety floor depends on params_per_edge = grid + k.
    if PYKAN_SYMBOLIC_FORCE_GRID is not None:
        extraction_grid = int(PYKAN_SYMBOLIC_FORCE_GRID)
        grid_source = (
            f"forced (PYKAN_SYMBOLIC_FORCE_GRID={extraction_grid})"
        )
    elif tuned_kan_params is not None and tuned_kan_params.get("grid") is not None:
        extraction_grid = int(tuned_kan_params["grid"])
        grid_source = f"tuned (matches CPCV-fold setting)"
    else:
        extraction_grid = int(PYKAN_FALLBACK_GRID)
        grid_source = (
            f"fallback (no tuned grid; PYKAN_FALLBACK_GRID={extraction_grid})"
        )

    # Data-aware safety floor: enforce n_train / total_params >= PYKAN_MIN_SAMPLES_PER_PARAM.
    # Used both as fallback sizing and as a clamp on any tuned width1 that's too large for the data.
    params_per_edge = extraction_grid + KAN_K
    max_edges = n_train // PYKAN_MIN_SAMPLES_PER_PARAM
    max_hidden_safety = max_edges // (params_per_edge * (n_features + n_classes))
    max_hidden_safety = max(2, max_hidden_safety)

    # Architecture selection: honour tuned width1 within the safety + cap envelope.
    tuned_w1 = None
    tuned_w2 = None
    if tuned_kan_params is not None:
        tuned_w1 = tuned_kan_params.get("width1")
        tuned_w2 = tuned_kan_params.get("width2", 0) or 0

    if tuned_w1 is not None:
        hidden = min(tuned_w1, PYKAN_SYMBOLIC_WIDTH_CAP, max_hidden_safety)
        hidden = max(2, hidden)
        # Annotate the source for diagnostic output.
        clamp_notes = []
        if tuned_w1 > PYKAN_SYMBOLIC_WIDTH_CAP:
            clamp_notes.append(f"capped at {PYKAN_SYMBOLIC_WIDTH_CAP}")
        if tuned_w1 > max_hidden_safety:
            clamp_notes.append(f"safety floor {max_hidden_safety}")
        if clamp_notes:
            arch_source = (
                f"tuned width1={tuned_w1} → {hidden} ({', '.join(clamp_notes)})"
            )
        else:
            arch_source = f"tuned width1={tuned_w1} (no clamping needed)"
    else:
        # Fallback: data-aware sizing clamped by the CPCV-side KAN_HIDDEN.
        hidden = min(max_hidden_safety, KAN_HIDDEN)
        hidden = max(2, hidden)
        arch_source = "data-aware fallback (no tuned params)"

    # Resolve width2: honour tuned value unless the override toggle forces single-hidden-layer.
    if PYKAN_SYMBOLIC_DROP_WIDTH2:
        extraction_w2 = 0
        if tuned_w2 and tuned_w2 > 0:
            width2_source = (
                f"forced 0 (PYKAN_SYMBOLIC_DROP_WIDTH2=True; "
                f"tuned width2={tuned_w2} ignored)"
            )
        else:
            width2_source = "forced 0 (PYKAN_SYMBOLIC_DROP_WIDTH2=True)"
    else:
        extraction_w2 = int(tuned_w2) if tuned_w2 else 0
        if extraction_w2 > 0:
            width2_source = f"tuned (width2={extraction_w2}, two hidden layers)"
        elif tuned_kan_params is not None:
            width2_source = "tuned (width2=0, single hidden layer)"
        else:
            width2_source = "default 0 (no tuned params)"

    # Build the width list.
    if extraction_w2 > 0:
        width = [n_features, hidden, extraction_w2, n_classes]
        total_edges = n_features * hidden + hidden * extraction_w2 + extraction_w2 * n_classes
    else:
        width = [n_features, hidden, n_classes]
        total_edges = n_features * hidden + hidden * n_classes
    total_params_est = total_edges * params_per_edge

    print(
        f"    {model_type} architecture: {width}  "
        f"(width1={hidden} [{arch_source}], grid={extraction_grid} [{grid_source}])  "
        f"→ {total_edges} edges, ~{total_params_est} params for {n_train} samples "
        f"(ratio={n_train/max(total_params_est,1):.1f}x)"
    )

    if n_train / max(total_params_est, 1) < 2:
        print(
            f"    ⚠ samples/params ratio < 2. Memorization risk."
        )

    if use_multkan:
        model = KANClass(
            width=width, grid=extraction_grid, k=KAN_K,
            seed=42, mult_arity=2,
        )
    else:
        model = KANClass(
            width=width, grid=extraction_grid, k=KAN_K, seed=42,
        )

    criterion = nn.CrossEntropyLoss()

    # Phase 1: Adam with weight decay + input-noise injection.
    logger.info(
        "Extraction Phase 1: Adam (%d steps, wd=%.4f, noise=%.3f)",
        PYKAN_ADAM_STEPS, PYKAN_ADAM_WEIGHT_DECAY, PYKAN_NOISE_STD,
    )
    print(
        f"    Phase 1 — Adam ({PYKAN_ADAM_STEPS} steps, "
        f"wd={PYKAN_ADAM_WEIGHT_DECAY}, noise_std={PYKAN_NOISE_STD})"
    )
    optimizer_adam = torch.optim.Adam(
        model.parameters(), lr=PYKAN_ADAM_LR,
        weight_decay=PYKAN_ADAM_WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state = None

    # Adam training loop with periodic val-loss tracking for best-state restoration.
    adam_pbar = tqdm(range(PYKAN_ADAM_STEPS), desc="    Adam", leave=False, unit="step")
    for step in adam_pbar:
        model.train()
        optimizer_adam.zero_grad()

        # Input noise: dropout-like regularisation, clamped to keep within the tanh-normalised range.
        X_noisy = X_t + PYKAN_NOISE_STD * torch.randn_like(X_t)
        X_noisy = X_noisy.clamp(-1, 1)

        logits = model(X_noisy)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer_adam.step()

        if (step + 1) % PYKAN_VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            adam_pbar.set_postfix(val_loss=f"{val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
    adam_pbar.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Diagnostic: did Adam actually learn anything beyond random?
    train_acc, val_acc = _log_diagnostic(
        "After Adam", model, X_t, y_t, X_val_t, y_val_t, verbose=False,
    )
    print(f"    ✓ Adam done. train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # S6: gate val_acc against the majority-class baseline, not just the absolute floor.
    # If the post-drop_rare class balance is e.g. 55/45, a model achieving 0.53 is BELOW
    # the trivial "always-predict-majority" baseline. The 0.53 floor is kept as a hard
    # minimum for very-balanced classes; the +0.01 margin requires "better than majority"
    # by a small margin to avoid declaring a trivial classifier "good".
    y_train_np = y_t.cpu().numpy() if y_t.is_cuda else y_t.numpy()
    majority_baseline = max(
        float(np.mean(y_train_np == 0)),
        float(np.mean(y_train_np == 1)),
    )
    effective_min_acc = max(PYKAN_MIN_ACCURACY, majority_baseline + 0.01)
    if val_acc < effective_min_acc:
        logger.warning(
            "Adam phase val_acc=%.4f < %.4f (majority baseline %.4f + 0.01 margin, "
            "floored at PYKAN_MIN_ACCURACY=%.2f). PyKAN may not have learned meaningful patterns.",
            val_acc, effective_min_acc, majority_baseline, PYKAN_MIN_ACCURACY,
        )
        print(
            f"    ⚠ WARNING: val_acc={val_acc:.4f} below effective threshold "
            f"{effective_min_acc:.4f} (majority baseline {majority_baseline:.4f}). "
            f"Continuing, but symbolic extraction may yield constants."
        )

    # Phase 2 (optional): grid extension. Skipped by default for small data per the constant.
    if PYKAN_GRID_EXTEND and n_train > 1000:
        try:
            with _suppress_pykan_stdout():
                model = model.refine(KAN_GRID)
            logger.info("Grid extended: %d → %d", extraction_grid, KAN_GRID)
            print(f"    ✓ Grid extended: {extraction_grid} → {KAN_GRID}")
        except Exception as e:
            # S7: broad catch is intentional. pykan's refine API has varied across versions:
            # different method names, different signatures, returning either a new model or
            # mutating in place, and various internal failures on small/degenerate data.
            # Grid extension is optional, so the safe behaviour is to skip on any failure.
            logger.warning("Grid extension failed (%s: %s).", type(e).__name__, e)
            print(f"    Grid extension failed: {type(e).__name__}: {e}")
    else:
        print(
            f"    Grid extension skipped (n_train={n_train} ≤ 1000, "
            f"grid stays at {extraction_grid})"
        )

    # Phase 2: LBFGS, split into a warmup half (no regularisation) and a sparsity half.
    lbfgs_warmup_steps = int(PYKAN_LBFGS_STEPS * PYKAN_LBFGS_WARMUP_FRAC)
    lbfgs_sparse_steps = PYKAN_LBFGS_STEPS - lbfgs_warmup_steps

    # Phase 2a: LBFGS warmup with no regularisation, so the loss can settle before sparsity kicks in.
    logger.info("Extraction Phase 2a: LBFGS warmup (%d steps, no reg)", lbfgs_warmup_steps)
    print(f"    Phase 2a — LBFGS warmup ({lbfgs_warmup_steps} steps, lamb=0)")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=PYKAN_LBFGS_LR, max_iter=10,
        line_search_fn="strong_wolfe",
    )

    # Reset best-state tracking for the LBFGS phase.
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    warmup_pbar = tqdm(range(lbfgs_warmup_steps), desc="    LBFGS warmup", leave=False, unit="step")
    for step in warmup_pbar:
        model.train()

        # LBFGS requires a closure that re-evaluates loss and computes gradients on each step.
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
            warmup_pbar.set_postfix(val_loss=f"{val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PYKAN_PATIENCE:
                logger.info("LBFGS warmup early stop at step %d", step + 1)
                break
    warmup_pbar.close()

    model.load_state_dict(best_state)

    train_acc, val_acc = _log_diagnostic(
        "After LBFGS warmup", model, X_t, y_t, X_val_t, y_val_t, verbose=False,
    )
    print(f"    ✓ LBFGS warmup done. train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # Phase 2b: LBFGS with sparsity regularisation (L1 + entropy) to prepare for symbolification.
    logger.info(
        "Extraction Phase 2b: LBFGS sparsity (%d steps, lamb=%.4f)",
        lbfgs_sparse_steps, PYKAN_LAMB,
    )
    print(f"    Phase 2b — LBFGS sparsity ({lbfgs_sparse_steps} steps, lamb={PYKAN_LAMB})")

    optimizer_lbfgs2 = torch.optim.LBFGS(
        model.parameters(), lr=PYKAN_LBFGS_LR, max_iter=10,
        line_search_fn="strong_wolfe",
    )

    best_val_loss_sparse = float("inf")
    best_state_sparse = copy.deepcopy(model.state_dict())
    patience_counter = 0

    sparse_pbar = tqdm(range(lbfgs_sparse_steps), desc="    LBFGS sparsity", leave=False, unit="step")
    for step in sparse_pbar:
        model.train()

        # Sparsity closure: add lamb * (L1 + entropy) terms when PyKAN exposes regularization_loss.
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
                pass  # PyKAN version doesn't expose regularization_loss; train without sparsity
            loss.backward()
            return loss

        optimizer_lbfgs2.step(closure_sparse)

        if (step + 1) % PYKAN_VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            sparse_pbar.set_postfix(val_loss=f"{val_loss:.4f}")
            if val_loss < best_val_loss_sparse:
                best_val_loss_sparse = val_loss
                best_state_sparse = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PYKAN_PATIENCE:
                logger.info("LBFGS sparsity early stop at step %d", step + 1)
                break
    sparse_pbar.close()

    model.load_state_dict(best_state_sparse)
    model.eval()

    train_acc, val_acc = _log_diagnostic(
        "After LBFGS sparsity", model, X_t, y_t, X_val_t, y_val_t, verbose=False,
    )
    print(f"    ✓ LBFGS sparsity done. train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # S9: return diagnostic state explicitly via a dict instead of stashing on the model
    # instance. The previous side-channel `model._pre_symbolic_accuracy` was fragile: re-running
    # symbolify_network on the same model overwrote it silently with no warning. Explicit
    # return values force the orchestrator to manage the state lifecycle.
    train_state = {"post_train_val_acc": float(val_acc)}
    logger.info("%s trained. Val accuracy: %.4f, width: %s", model_type, val_acc, width)
    return model, train_state


# --- 6. Pruning (Algorithm 1, Step 2) --------------------------------------
# Drop dead edges and nodes via PyKAN's prune API; logs edge survival for over-regularisation diagnostics.
def prune_network(model, dataset: dict):
    """Prune low-importance edges and return the pruned model; logs edge counts before / after.

    Falls back to the unpruned model when PyKAN's prune API raises or the pruned model fails its
    forward-pass sanity check.
    """
    _ = model(dataset["train_input"])

    try:
        original_width = list(model.width)
    except AttributeError:
        original_width = ["unknown"]

    # Pre-prune diagnostic: which edges are active under the threshold.
    total_edges, active_edges = _count_active_edges(model, PRUNE_THRESHOLD)
    if total_edges > 0:
        logger.info(
            "Pre-prune edge analysis: %d/%d active (%.1f%%)",
            active_edges, total_edges, 100 * active_edges / max(total_edges, 1),
        )

        # Very low edge survival is usually a symptom of over-aggressive sparsity regularisation.
        if active_edges < 3:
            print(
                "    ⚠ Very few active edges; regularization may have been too aggressive."
            )

    # Accuracy gate before pruning: warn if PyKAN never learned past random.
    # S6: majority-class baseline check.
    val_acc = _compute_accuracy(
        model, dataset["test_input"], dataset["test_label"].long()
    )
    y_test = dataset["test_label"].long()
    y_test_np = y_test.cpu().numpy() if y_test.is_cuda else y_test.numpy()
    majority_baseline = max(
        float(np.mean(y_test_np == 0)),
        float(np.mean(y_test_np == 1)),
    )
    effective_min_acc = max(PYKAN_MIN_ACCURACY, majority_baseline + 0.01)
    if val_acc < effective_min_acc:
        print(
            f"    ⚠ Pre-prune val_acc={val_acc:.4f} < {effective_min_acc:.4f} "
            f"(majority baseline {majority_baseline:.4f}). Model hasn't learned meaningful patterns."
        )

    try:
        with _suppress_pykan_stdout():
            model.attribute()
    except Exception as e:
        # S12: attribute() populates the importance scores that prune() uses. When it fails
        # silently, subsequent prune may operate on stale/uninitialised attribution and yield
        # an unreliable pruned architecture rather than an outright error. Surface this clearly.
        logger.warning(
            "model.attribute() failed (%s: %s). Pruning may use stale attribution.",
            type(e).__name__, e,
        )
        print(
            f"    ⚠ model.attribute() failed ({type(e).__name__}); pruning may be unreliable."
        )

    pre_prune_model = model

    # PyKAN's prune API varies by version: try threshold=, then node_th=/edge_th=, then no-args.
    pruned = False
    try:
        with _suppress_pykan_stdout():
            model = model.prune(threshold=PRUNE_THRESHOLD)
        pruned = True
    except TypeError:
        try:
            with _suppress_pykan_stdout():
                model = model.prune(node_th=PRUNE_THRESHOLD, edge_th=PRUNE_THRESHOLD)
            pruned = True
        except TypeError:
            try:
                with _suppress_pykan_stdout():
                    model = model.prune()
                pruned = True
            except Exception as e:
                logger.warning("model.prune() failed: %s. Returning unpruned.", e)
                return pre_prune_model
    except Exception as e:
        logger.warning("model.prune() failed: %s. Returning unpruned.", e)
        return pre_prune_model

    # Sanity check: a successfully-pruned model must still produce a valid forward pass.
    if pruned:
        try:
            _ = model(dataset["train_input"])
        except Exception as e:
            # S7: broad catch — the pruned model can fail in many ways (RuntimeError from
            # tensor-shape mismatch, ValueError from empty layers, version-specific
            # AttributeErrors). Any forward-pass failure means the prune produced an
            # unusable model, so we fall back to the unpruned version uniformly.
            logger.warning(
                "Pruned model forward pass failed (%s: %s). Returning unpruned.",
                type(e).__name__, e,
            )
            return pre_prune_model

    try:
        pruned_width = list(model.width)
    except AttributeError:
        pruned_width = ["unknown"]

    # Post-prune diagnostic counts.
    post_total, post_active = _count_active_edges(model, PRUNE_THRESHOLD)
    post_acc = _compute_accuracy(
        model, dataset["test_input"], dataset["test_label"].long()
    )
    print(
        f"    ✓ Pruned: {original_width} → {pruned_width}  "
        f"({active_edges}/{total_edges} → {post_active}/{post_total} active edges)  "
        f"val_acc={post_acc:.4f}"
    )
    logger.info("Pruned: %s → %s", original_width, pruned_width)

    _save_plot(model, "kan_pruned_network.png")
    return model


# --- 7. Symbolification + affine fine-tune (Algorithm 1, Steps 3+4) --------
# Replace B-spline activations with closed-form primitives per edge, then fine-tune the affine params.
def symbolify_network(model, dataset: dict):
    """Symbolify every edge whose best symbolic candidate has R² ≥ ``SYMBOLIC_R2_THRESHOLD``, then fine-tune.

    Per edge, PyKAN's ``suggest_symbolic`` returns a ranked list of (function, R²) candidates from
    ``SYMBOLIC_LIBRARY``. We pick the highest-R² non-constant candidate and call ``fix_symbolic``.
    Edges that fall below the R² threshold keep their spline. Three return-format cases are handled
    (DataFrame, flat tuple, nested tuple) because PyKAN versions differ.
    """
    # Cached forward pass first so activations are populated.
    model.eval()
    _ = model(dataset["train_input"])

    # Activation sanity check: flag NaN outputs or collapsed (near-constant) logits early.
    try:
        with torch.no_grad():
            out = model(dataset["train_input"])
            if torch.isnan(out).any():
                print("    ⚠ Model produces NaN outputs. Symbolification will fail.")
            elif (out.std(dim=0) < 1e-6).all():
                print("    ⚠ Model outputs are near-constant. Activations may be flat.")
            else:
                logit_diff = (out[:, 1] - out[:, 0]) if out.shape[1] > 1 else out[:, 0]
                logger.info(
                    "Activation check: logit_diff std=%.4f, range=[%.3f, %.3f]",
                    logit_diff.std().item(),
                    logit_diff.min().item(),
                    logit_diff.max().item(),
                )
    except Exception as e:
        logger.warning("Activation check failed: %s", e)

    # Record pre-symbolic accuracy and snapshot state for potential rollback after fine-tuning.
    with torch.no_grad():
        pre_pred = model(dataset["test_input"])
        pre_acc = (pre_pred.argmax(dim=1) == dataset["test_label"].long()).float().mean().item()

    pre_state = copy.deepcopy(model.state_dict())

    total_edges = 0
    symbolified_edges = 0
    skipped_edges = 0
    fallback_count = 0
    r2_values = []

    # Pre-count edges so the progress bar has a known total. Mirrors model.width's structure.
    total_edges_expected = 0
    for _l in range(len(model.width) - 1):
        _n_in = model.width[_l]
        _n_out = model.width[_l + 1]
        if isinstance(_n_in, (list, tuple)):
            _n_in = _n_in[0] if _n_in else 0
        if isinstance(_n_out, (list, tuple)):
            _n_out = _n_out[0] if _n_out else 0
        total_edges_expected += _n_in * _n_out

    # Walk every (layer, in_node, out_node) triple and try to symbolify that edge.
    # pykan's per-edge "function fitting r2" table is suppressed via the stdout redirect.
    edge_pbar = tqdm(total=total_edges_expected, desc="    Symbolify", leave=False, unit="edge")
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
                edge_pbar.update(1)

                # Try the curated SYMBOLIC_LIBRARY first; if PyKAN doesn't recognise a name, fall back
                # to its built-in default library before giving up.
                try:
                    with _suppress_pykan_stdout():
                        suggestions = model.suggest_symbolic(
                            l, i, j, topk=SYMBOLIC_TOPK, lib=SYMBOLIC_LIBRARY,
                        )
                except Exception as e:
                    # S7: broad catch absorbs the various TypeError/AttributeError/RuntimeError
                    # failures pykan raises across versions when symbolic fitting hits degenerate
                    # edge data. Fall back to pykan's default library.
                    try:
                        with _suppress_pykan_stdout():
                            suggestions = model.suggest_symbolic(
                                l, i, j, topk=SYMBOLIC_TOPK,
                            )
                        if fallback_count < 3:
                            logger.info(
                                "Edge (%d,%d,%d): custom lib failed (%s), using pykan defaults.",
                                l, i, j, type(e).__name__,
                            )
                        fallback_count += 1
                    except Exception as e2:
                        if skipped_edges < 5:
                            logger.warning(
                                "suggest_symbolic(%d,%d,%d) failed: %s: %s",
                                l, i, j, type(e2).__name__, e2,
                            )
                        skipped_edges += 1
                        continue

                if suggestions is None:
                    skipped_edges += 1
                    continue

                # Parse the return value safely. PyKAN's return type varies across versions:
                #   - DataFrame (rows = candidates)
                #   - flat tuple (fn_name, fitted_lambdas, r2, complexity)
                #   - nested tuple of tuples
                # All cases are handled, and the constant "0" is always skipped because complexity
                # alone makes it win on total_loss even when non-trivial candidates have better R².
                best_fn = None
                best_r2 = 0.0

                try:
                    # Debug dump for the first few edges so version mismatches surface in the log.
                    if total_edges <= 2:
                        logger.debug(
                            "Edge (%d,%d,%d): type=%s, repr=%s",
                            l, i, j, type(suggestions).__name__,
                            repr(suggestions)[:200],
                        )

                    # CASE 1: DataFrame return.
                    if hasattr(suggestions, "to_dict"):
                        records = suggestions.to_dict("records")
                        if not records:
                            skipped_edges += 1
                            continue

                        # Detect the R² column heuristically (avoid matching "r2_loss").
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

                    # CASE 2: flat or nested tuple return.
                    elif isinstance(suggestions, (tuple, list)):
                        if len(suggestions) == 0:
                            skipped_edges += 1
                            continue

                        first = suggestions[0]

                        if isinstance(first, str):
                            # Flat tuple shape: (fn_name, fitted_lambdas, R², complexity).
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
                                # "0" can win on total_loss due to zero complexity even when
                                # non-constant functions have excellent R². Brute-force override:
                                # try each candidate via fix_symbolic, capture the R² PyKAN prints,
                                # keep the best non-zero match.
                                original_state = copy.deepcopy(model.state_dict())
                                best_direct_fn = None
                                best_direct_r2 = 0.0
                                # S5: track regex-miss count so we can warn after the loop if
                                # pykan changed its log format. The fallback degrades silently
                                # otherwise, since every candidate would score r2=0 by default.
                                n_regex_misses = 0
                                n_candidates_tried = 0

                                # S5: tolerate variants in pykan's print format. The original
                                # regex was r"r2 is ([\d.eE+-]+)"; cover capitalisation, "=" vs "is",
                                # and "R^2" variants that have appeared across versions.
                                R2_PATTERNS = [
                                    re.compile(r"r2 is ([\d.eE+-]+)", re.IGNORECASE),
                                    re.compile(r"r\^?2\s*[:=]\s*([\d.eE+-]+)", re.IGNORECASE),
                                ]

                                for candidate in SYMBOLIC_LIBRARY:
                                    if candidate == "0":
                                        continue
                                    n_candidates_tried += 1
                                    try:
                                        # Capture stdout so we can extract R² from
                                        # PyKAN's print output (no programmatic R² return value).
                                        old_stdout = sys.stdout
                                        sys.stdout = buffer = io.StringIO()
                                        try:
                                            model.fix_symbolic(l, i, j, candidate)
                                            output = buffer.getvalue()
                                        finally:
                                            sys.stdout = old_stdout

                                        r2_match = None
                                        for pattern in R2_PATTERNS:
                                            r2_match = pattern.search(output)
                                            if r2_match:
                                                break
                                        if r2_match:
                                            cand_r2 = float(r2_match.group(1))
                                            cand_r2 = max(0.0, min(1.0, cand_r2))
                                            if cand_r2 > best_direct_r2:
                                                best_direct_r2 = cand_r2
                                                best_direct_fn = candidate
                                        else:
                                            n_regex_misses += 1

                                        # Restore original state before trying the next candidate.
                                        model.load_state_dict(original_state)
                                    except Exception:
                                        model.load_state_dict(original_state)
                                        continue

                                # S5: warn if every candidate missed the regex; this means
                                # pykan changed its log format and the fallback isn't recovering
                                # any R² values, even when candidates fit well.
                                if n_candidates_tried > 0 and n_regex_misses == n_candidates_tried:
                                    logger.warning(
                                        "Edge (%d,%d,%d): R² regex matched 0/%d candidates. "
                                        "PyKAN's fix_symbolic log format may have changed; "
                                        "the brute-force '0' fallback is not recovering R² "
                                        "and this edge will keep its spline. Inspect a recent "
                                        "fix_symbolic call's stdout to update R2_PATTERNS.",
                                        l, i, j, n_candidates_tried,
                                    )

                                if best_direct_fn is not None and best_direct_r2 >= SYMBOLIC_R2_THRESHOLD:
                                    best_fn = best_direct_fn
                                    best_r2 = best_direct_r2

                        elif isinstance(first, (tuple, list)):
                            # Nested tuple shape: ((fn_name, lambdas, R², complexity), ...).
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
                            # Edge case: tuple wrapping a DataFrame as the first element.
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
                        logger.warning(
                            "Edge (%d,%d,%d) parse error: %s: %s",
                            l, i, j, type(e).__name__, e,
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

                # Record the candidate for the post-loop R² distribution diagnostic.
                r2_values.append((l, i, j, best_fn, best_r2))

                # Apply the symbolic replacement only when R² clears the threshold.
                if best_r2 >= SYMBOLIC_R2_THRESHOLD:
                    try:
                        with _suppress_pykan_stdout():
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
    edge_pbar.close()

    # Single-line summary; the per-edge / R²-distribution / top-5 detail goes to the log.
    sym_rate = symbolified_edges / max(total_edges, 1)
    if r2_values:
        r2_scores = [v[4] for v in r2_values]
        r2_min, r2_med, r2_max = min(r2_scores), float(np.median(r2_scores)), max(r2_scores)
        print(
            f"    ✓ Symbolified {symbolified_edges}/{total_edges} edges ({sym_rate:.0%})  "
            f"[skipped {skipped_edges}, fallback {fallback_count}, "
            f"R² min/med/max = {r2_min:.3f}/{r2_med:.3f}/{r2_max:.3f}]"
        )
        # Top 5 edges by R² goes to logger.info for thesis-audit traceability.
        top5 = sorted(r2_values, key=lambda x: x[4], reverse=True)[:5]
        for l_, i_, j_, fn, r2 in top5:
            logger.info("Top edge (%d,%d,%d): %s (R²=%.4f)", l_, i_, j_, fn, r2)
    else:
        print(
            f"    ⚠ No R² values collected; symbolification skipped all "
            f"{total_edges} edges."
        )

    # Step 4 of Algorithm 1: fine-tune the affine parameters around the now-fixed symbolic activations.
    if symbolified_edges > 0:
        print(f"    Affine fine-tune ({AFFINE_FINETUNE_STEPS} steps)")
        try:
            dataset_fit = {
                "train_input": dataset["train_input"],
                "train_label": dataset["train_label"].long(),
                "test_input": dataset["test_input"],
                "test_label": dataset["test_label"].long(),
            }
            # pykan's own tqdm bar shows progress here; we don't suppress its stdout because
            # silencing it would also silence the per-step train/test loss readout.
            model.fit(
                dataset_fit, opt="LBFGS", lr=AFFINE_LR,
                steps=AFFINE_FINETUNE_STEPS,
                loss_fn=nn.CrossEntropyLoss(),
                update_grid=False,
            )
        except (TypeError, AttributeError, RuntimeError):
            # PyKAN version differences: fall back to a hand-rolled LBFGS loop with NaN guard.
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
                        # NaN at step N: revert to the pre-symbolic state and exit fine-tuning.
                        logger.warning("NaN at affine step %d. Reverting.", step)
                        model.load_state_dict(pre_state)
                        break
            except Exception as e:
                logger.warning("Affine fine-tuning failed: %s. Reverting.", e)
                model.load_state_dict(pre_state)
    else:
        print("  Skipping affine fine-tuning (no edges were symbolified).")

    _save_plot(model, "kan_symbolified_network.png")

    # S9: return state explicitly. See the comment in train_pykan for the rationale.
    sym_state = {
        "pre_symbolic_accuracy": float(pre_acc),
        "symbolification_rate": float(sym_rate),
    }
    return model, sym_state


# --- 8. Formula extraction --------------------------------------------------
# Pull the closed-form sympy expressions out of the symbolified PyKAN; handles version variation in variable naming.
def extract_formulas(
    model,
    dataset: dict,
    feature_names: list[str],
    sym_state: dict | None = None,
) -> dict:
    """Return a dict of closed-form expressions and metadata: bearish / bullish logits, decision function,
    P(up) formula, pre / post symbolic accuracy, symbolification rate, pruned architecture, surviving features.

    ``sym_state`` (S9): the dict returned by ``symbolify_network`` containing ``pre_symbolic_accuracy``
    and ``symbolification_rate``. When provided, these values are read from the dict; when omitted,
    we fall back to ``getattr(model, ...)`` for backward compatibility with any caller that still
    relies on the legacy side-channel attributes.

    Robust to PyKAN's variable-naming convention drift (``x1``, ``x_1``, ``x_0``) by trying all three.
    """
    # PyKAN evaluates expressions through sympy. Without passing var= explicitly, undefined variable
    # names (x1, x2, ...) raise NameError in PyKAN's namespace. Try all three conventions in order.
    n_inputs = len(feature_names)
    var_symbols_x = [sympy.Symbol(f"x{i+1}") for i in range(n_inputs)]      # x1, x2, ... (1-based, no underscore)
    var_symbols_x_ = [sympy.Symbol(f"x_{i+1}") for i in range(n_inputs)]    # x_1, x_2, ... (1-based)
    var_symbols_x0 = [sympy.Symbol(f"x_{i}") for i in range(n_inputs)]      # x_0, x_1, ... (0-based)

    formulas = None
    used_vars = None
    for var_candidate in [var_symbols_x, var_symbols_x_, var_symbols_x0]:
        try:
            formulas = model.symbolic_formula(var=var_candidate)
            used_vars = var_candidate
            break
        except Exception:
            continue

    # Last resort: try without var= (works on some PyKAN versions where var is inferred).
    if formulas is None:
        try:
            formulas = model.symbolic_formula()
            used_vars = None
        except Exception as e:
            logger.error("model.symbolic_formula() failed: %s", e)
            return _empty_result(model, feature_names, sym_state=sym_state)

    # Pull the per-class expressions from PyKAN's nested return shape.
    try:
        if isinstance(formulas, (list, tuple)) and len(formulas) > 0:
            expr_list = formulas[0] if isinstance(formulas[0], (list, tuple)) else formulas
            logit_bearish = expr_list[0]
            logit_bullish = expr_list[1] if len(expr_list) > 1 else sympy.Integer(0)
        else:
            return _empty_result(model, feature_names, sym_state=sym_state)
    except Exception as e:
        logger.error("Formula parsing failed: %s", e)
        return _empty_result(model, feature_names, sym_state=sym_state)

    # Substitute placeholder symbols with feature names so the formula reads in domain terms.
    if used_vars is not None:
        # We passed var= explicitly, so we know exactly which symbols to substitute.
        for old_sym, name in zip(used_vars, feature_names):
            new_sym = sympy.Symbol(name)
            try:
                logit_bearish = logit_bearish.subs(old_sym, new_sym)
                logit_bullish = logit_bullish.subs(old_sym, new_sym)
            except (AttributeError, TypeError):
                pass
    else:
        # Fallback: detect the naming convention from free symbols (x_0-based vs x_1-based).
        all_symbols = set()
        for expr in [logit_bearish, logit_bullish]:
            if hasattr(expr, "free_symbols"):
                all_symbols |= expr.free_symbols

        symbol_names = {str(s) for s in all_symbols}
        has_x0 = "x_0" in symbol_names
        has_xn = f"x_{len(feature_names)}" in symbol_names

        # If x_{n} exists but x_0 doesn't, the naming is 1-based.
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

    # sympy.simplify can hang on complex KAN expressions, so run it in a thread with a 30s timeout.
    def _simplify_with_timeout(expr, timeout_sec=30):
        """Run ``sympy.simplify(expr)`` with a wall-clock timeout; return the original on timeout."""
        result = [expr]

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

    # nsimplify for cleaner rational numbers; also protected against hangs.
    try:
        logit_bearish = sympy.nsimplify(logit_bearish, tolerance=1e-3, rational=False)
        logit_bullish = sympy.nsimplify(logit_bullish, tolerance=1e-3, rational=False)
        decision = sympy.nsimplify(decision, tolerance=1e-3, rational=False)
    except Exception:
        pass

    # S9: prefer sym_state when present; fall back to legacy side-channel for callers that
    # haven't migrated yet.
    if sym_state is not None:
        pre_acc = sym_state.get("pre_symbolic_accuracy", np.nan)
        sym_rate = sym_state.get("symbolification_rate", np.nan)
    else:
        pre_acc = getattr(model, "_pre_symbolic_accuracy", np.nan)
        sym_rate = getattr(model, "_symbolification_rate", np.nan)

    # Post-symbolic accuracy: the model's accuracy after symbolic activations + fine-tuning.
    try:
        with torch.no_grad():
            post_pred = model(dataset["test_input"])
            post_acc = (
                (post_pred.argmax(dim=1) == dataset["test_label"].long())
                .float().mean().item()
            )
    except Exception:
        post_acc = np.nan

    # Surviving features = the subset that the simplified decision function still references.
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
        # S1: the raw → tanh transform parameters travel with the result so sensitivity
        # and marginal-effect helpers can substitute correctly. The decision_function above
        # is expressed in tanh-normalised space; consumers MUST convert raw values via
        # z_i = tanh((x_i - feature_a[i]) / feature_b[i]) before evaluating.
        "input_transform": dataset.get("input_transform"),
    }


# --- 9. Top-level orchestration --------------------------------------------
# Public entry point: take CPCV results + raw data, return the extracted symbolic formula dict.
def run_symbolic_extraction(
    cpcv_results: dict,
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    t1: pd.Series,
    n_top_features: int | None = None,
    use_multkan: bool = False,
    fold_selection: str | int = "best",
    feature_selection_strategy: str = "per_fold",
) -> dict:
    """Run the full Algorithm 1 pipeline: fold pick → feature pick → data prep → train → prune → symbolify → extract.

    ``feature_selection_strategy``: ``per_fold`` (default) uses the chosen fold's MDA selection so
    the symbolic formula represents the model whose performance is in the comparison table;
    ``stability`` (legacy) uses the top ``n_top_features`` by cross-fold selection frequency.
    Both are defensible; per_fold is more methodologically faithful to the CPCV evaluation.

    ``fold_selection``: ``"best"`` (highest KAN F1, default), ``"last"`` (most recent), or an int.
    ``use_multkan=True`` switches to KAN 2.0 with multiplication nodes for multiplicative interactions.
    """
    model_label = "MultKAN" if use_multkan else "PyKAN"
    print("=" * 60)
    print("Symbolic Extraction (VIX KAN Paper, Algorithm 1)")
    print(f"  CPCV model: efficient-kan | Extraction model: {model_label}")
    print("=" * 60)

    # Step 1: fold selection.
    best_split, prep_info = select_extraction_fold(cpcv_results, fold_selection=fold_selection)

    # Step 1b: pull the tuned KAN params for this fold so the extraction model can match them.
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

    # Step 2: feature selection. Resolved before data prep so we can override the per-fold MDA set if asked.
    feature_subset = None
    if n_top_features is not None or feature_selection_strategy == "per_fold":
        feature_subset = select_features_for_extraction(
            cpcv_results=cpcv_results,
            fold_idx=best_split,
            n_top_features=n_top_features,
            strategy=feature_selection_strategy,
        )
        if feature_subset:
            print(
                f"\n  Feature selection (strategy='{feature_selection_strategy}', "
                f"fold={best_split}): {len(feature_subset)} features"
            )
            if feature_selection_strategy == "per_fold":
                # Annotate the per-fold pick with cross-fold stability so the reader sees the trade-off
                # between fold-specific and broadly-stable selection.
                stability = dict(rank_features_by_stability(cpcv_results))
                for i, feat in enumerate(feature_subset):
                    freq = stability.get(feat, 0.0)
                    print(f"    {i+1}. {feat} ({freq:.0%} cross-fold stability)")
            else:
                # stability strategy: show the frequency that drove the rank.
                ranked = rank_features_by_stability(cpcv_results)
                ranked_dict = dict(ranked)
                for i, feat in enumerate(feature_subset):
                    freq = ranked_dict.get(feat, 0.0)
                    print(f"    {i+1}. {feat} ({freq:.0%} selection frequency)")
                if n_top_features is not None and len(ranked) > n_top_features:
                    print(f"    ... ({len(ranked) - n_top_features} features excluded)")
        else:
            print("  ⚠ Could not resolve feature subset. Using all selected features.")
            feature_subset = None

    # Step 3: data preparation.
    dataset, feature_names = prepare_extraction_data(
        X, y, w, t1, cpcv_results, best_split, prep_info,
        feature_subset=feature_subset,
    )
    print(
        f"  Data: {dataset['train_input'].shape[0]} train, "
        f"{dataset['test_input'].shape[0]} val, "
        f"{len(feature_names)} features: {feature_names}"
    )

    # Step 4: train, prune, symbolify, extract.
    print(f"\n  Step 1: Training {model_label} (Adam → grid extend → LBFGS)...")
    # S9: capture the explicit state dict from train_pykan instead of relying on a
    # side-channel attribute. The state from train_pykan (post-training val acc) is
    # superseded by symbolify_network's pre-symbolic accuracy measurement, which is the
    # canonical value because it's measured immediately before symbolification.
    model, _train_state = train_pykan(
        dataset,
        n_features=len(feature_names),
        use_multkan=use_multkan,
        tuned_kan_params=tuned_kan_params,
    )

    print("\n  Step 2: Pruning...")
    model = prune_network(model, dataset)

    print("\n  Steps 3+4: Symbolification + affine fine-tuning...")
    # S9: symbolify_network now returns (model, sym_state) explicitly.
    model, sym_state = symbolify_network(model, dataset)

    print("\n  Extracting formulas...")
    # S9: pass sym_state directly to avoid any dependence on the legacy side-channel.
    result = extract_formulas(model, dataset, feature_names, sym_state=sym_state)

    # Headline summary block.
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
    print(f"  Surviving features:    {result['surviving_features']}")
    print(f"  Pruned architecture:    {result['pruned_architecture']}")
    # S3: methodology disclaimer surfaced at the cell-output level.
    print(
        f"\n  Note: the formula below represents a {model_label} retrained from scratch with"
        f"\n  matching architecture on the same training fold. Its decision boundary approximates"
        f"\n  but does not exactly equal the CPCV-evaluated efficient-kan's. The pre/post symbolic"
        f"\n  accuracy gap above quantifies how faithful the symbolic substitution is."
    )
    print(f"\n  Decision function (first 240 chars):")
    decision_str = result["decision_function"]
    if len(decision_str) > 240:
        print(f"    {decision_str[:240]}…")
        print(f"    ({len(decision_str)} chars total; access via result['decision_function'])")
    else:
        print(f"    {decision_str}")
    print(f"\n  P(up) = 1 / (1 + exp(-decision))  [see result['p_up_formula']]")
    print(f"{'='*60}")

    return result


# --- 10. Private helpers ----------------------------------------------------
# Safe float coercion: PyKAN's suggest_symbolic() can return DataFrames with 'nan' strings or truncated entries.
def _safe_float(value, default: float = 0.0) -> float:
    """Coerce ``value`` to float; return ``default`` on ValueError, NaN, or inf."""
    if value is None:
        return default
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        logger.debug("_safe_float: could not convert %r, using default %.2f", value, default)
        return default


# Save the current PyKAN plot to the cache directory; falls back silently when matplotlib backend differs.
def _save_plot(model, filename: str) -> None:
    """Save the current PyKAN visualisation to ``CACHE_DIR/filename``; warn (don't raise) on failure."""
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
        # S13: include the exception type so the user can distinguish matplotlib backend
        # failures from pykan API mismatches from filesystem permission errors.
        logger.warning(
            "Could not save %s (%s: %s).", filename, type(e).__name__, e,
        )


# Sentinel return value when extraction fails before reaching extract_formulas.
def _empty_result(
    model,
    feature_names: list[str],
    sym_state: dict | None = None,
) -> dict:
    """Return a placeholder result dict for the failure path.

    ``sym_state`` (S9): same dict that ``extract_formulas`` accepts; supplies accuracy and
    symbolification-rate values when the legacy side-channel isn't present on the model.
    """
    if sym_state is not None:
        pre_acc = sym_state.get("pre_symbolic_accuracy", np.nan)
        sym_rate = sym_state.get("symbolification_rate", np.nan)
    else:
        pre_acc = getattr(model, "_pre_symbolic_accuracy", np.nan)
        sym_rate = getattr(model, "_symbolification_rate", np.nan)
    return {
        "logit_bearish": "extraction_failed",
        "logit_bullish": "extraction_failed",
        "decision_function": "extraction_failed",
        "p_up_formula": "extraction_failed",
        "sympy_objects": {"bearish": None, "bullish": None, "decision": None},
        "pre_symbolic_accuracy": pre_acc,
        "post_symbolic_accuracy": np.nan,
        "symbolification_rate": sym_rate,
        "pruned_architecture": [],
        "surviving_features": [],
        # S1: input transform params, populated only when the full pipeline runs.
        "input_transform": None,
    }


# --- 11. Notebook display / analysis helpers -------------------------------
# These helpers consume the dict returned by run_symbolic_extraction and produce the formatted
# output / plots that the notebook used to generate inline. The singularity handling in
# _safe_eval_at_point and compute_feature_sensitivity is non-obvious and is documented inline.

# Safe sympy-derivative evaluation: returns NaN at poles rather than +/-inf.
def _safe_eval_at_point(deriv, point: dict) -> float:
    """Evaluate ``deriv`` at ``point``; return NaN if the result is non-finite or substitution raises.

    PyKAN's symbolic library includes reciprocal and logarithmic primitives that can produce poles
    in the learned activation. For features whose centrality measure (mean / median) lands near
    such a pole, the symbolic gradient is non-finite. Without this guard, ``+inf`` / ``-inf`` would
    propagate into the sensitivity table and mislead the reader; NaN reads correctly as "undefined
    at this evaluation point".
    """
    try:
        val = float(deriv.subs(point))
    except (ZeroDivisionError, TypeError, ValueError, OverflowError):
        return float("nan")
    if not np.isfinite(val):
        return float("nan")
    return val


# Print the headline outputs of run_symbolic_extraction.
def print_symbolic_decision(symbolic: dict, max_chars: int = 240) -> None:
    """Print the decision function (truncated to ``max_chars`` for readability), P(up)
    formula reference, and surviving features.

    Long sympy expressions can take noticeable time for Jupyter to render due to the
    cell's syntax highlighting and word-wrap calculations. Passing ``max_chars=None``
    prints the full expression; the default 240 keeps the cell snappy and the full
    string remains available via ``symbolic['decision_function']``.
    """
    decision_str = symbolic["decision_function"]
    print("Decision function:")
    if max_chars is not None and len(decision_str) > max_chars:
        print(f"  {decision_str[:max_chars]}…")
        print(f"  ({len(decision_str)} chars total; access via symbolic['decision_function'])")
    else:
        print(f"  {decision_str}")
    print(f"\nP(up) = 1 / (1 + exp(-decision))  [full string in symbolic['p_up_formula']]")
    print(f"\nSurviving features: {symbolic['surviving_features']}")


# Print the pre/post symbolic accuracy gap; near-zero gap means symbolic substitution faithfully captured the spline.
def print_extraction_metrics(symbolic: dict) -> None:
    """Print pre / post symbolic accuracy, symbolification rate, and pruned architecture."""
    pre = symbolic.get("pre_symbolic_accuracy", float("nan"))
    post = symbolic.get("post_symbolic_accuracy", float("nan"))
    rate = symbolic.get("symbolification_rate", float("nan"))
    arch = symbolic.get("pruned_architecture", "N/A")
    print(f"Pre-symbolic accuracy:  {pre:.4f}")
    print(f"Post-symbolic accuracy: {post:.4f}")
    if isinstance(rate, (int, float)) and np.isfinite(rate):
        print(f"Symbolification rate:   {rate:.0%}")
    else:
        print(f"Symbolification rate:   N/A")
    print(f"Pruned architecture:    {arch}")


# Print the closed-form partial derivative of the decision function with respect to each surviving feature.
def print_partial_derivatives(symbolic: dict) -> None:
    """Print the symbolic gradient ``∂(decision)/∂(feature)`` for every surviving feature."""
    decision_expr = symbolic["sympy_objects"]["decision"]
    features = symbolic["surviving_features"]

    if decision_expr is None or decision_expr == "extraction_failed":
        print("Symbolic extraction failed; no partial derivatives to display.")
        return

    print("Partial derivatives (symbolic form):\n")
    for feat in features:
        sensitivity = sympy.diff(decision_expr, sympy.Symbol(feat))
        print(f"  ∂(decision)/∂({feat}) =")
        print(f"    {sensitivity}\n")


def _evaluate_decision(decision_fn, decision_expr, sym_vars, features, z_values):
    """Evaluate the decision function at a tanh-space point; prefer the lambdified callable."""
    if decision_fn is not None:
        try:
            return float(decision_fn(*[z_values[f] for f in features]))
        except (ZeroDivisionError, TypeError, ValueError, OverflowError):
            return float("nan")
    try:
        return float(decision_expr.subs({sym_vars[f]: z_values[f] for f in features}))
    except (ZeroDivisionError, TypeError, ValueError, OverflowError):
        return float("nan")


# Compute per-feature marginal sensitivity at the dataset mean (or median).
# Uses central finite differences in raw space, which avoids the analytic poles that sympy.diff
# produces when the formula contains log/sqrt/1-over primitives evaluated near zero.
def compute_feature_sensitivity(
    symbolic: dict,
    X: pd.DataFrame,
    eval_point: str = "mean",
    fd_step_frac: float = 0.01,
) -> pd.DataFrame:
    """Return per-feature gradient + per-σ effect on the decision logit + per-σ ΔP(up).

    The symbolic ``decision_function`` is expressed in tanh-normalised feature space, not raw
    feature space. The variable rename in ``extract_formulas`` is purely cosmetic; substituting
    raw feature values into the formula evaluates it at the wrong point. This function uses
    ``symbolic['input_transform']`` to map raw values into tanh space.

    **Gradient via central finite differences.** The earlier implementation used ``sympy.diff``
    of the decision function and substituted the centrality point into the analytic derivative.
    For formulas containing primitives like ``log(g(z))``, ``sqrt(g(z))`` or ``1/g(z)`` whose
    ``g`` vanishes near the scaled origin, the analytic derivative has a pole at exactly the
    point the RobustScaler-then-tanh pipeline maps the median to (``z = 0``). The result was a
    table of pure NaN even when the formula itself was perfectly well-defined a step away. This
    implementation evaluates the lambdified formula at ``f(x_raw + h)`` and ``f(x_raw - h)`` in
    raw space (converting each to tanh space with the stored transform) and takes the central
    difference. The two evaluation points are not at the singularity, so the gradient is finite
    whenever the formula itself is. ``fd_step_frac`` is the step size as a fraction of the raw
    feature std; the default 0.01 is small enough to be local for typical features.

    The per-σ effect uses a finite difference across a full raw σ, computed identically.

    ``sigma_delta_p`` uses the actual sigmoid slope at the centrality point,
    ``p_at_center * (1 - p_at_center)``, instead of the previous hardcoded 1/4. The 1/4 value
    is the maximum slope at P=0.5; for confident predictions (P close to 0 or 1) it overstates
    the actual rate of change of P(up) substantially.

    ``eval_point`` selects the centrality measure on the raw data:
      - ``"mean"`` (default): substitution point is ``X[features].mean()``
      - ``"median"``: substitution point is ``X[features].median()`` (more robust on skewed features)

    Returns a DataFrame indexed by feature, with columns:
      - ``mean_value``, ``std_value``: raw-space statistics from the input X
      - ``d_decision/d_feature_at_center``: gradient in raw-feature units (df/dx_raw)
      - ``sigma_effect_on_decision``: change in decision logit from bumping the feature by 1 raw σ
      - ``sigma_delta_p``: corresponding change in P(up), using the slope at the centrality point
    """
    decision_expr = symbolic["sympy_objects"]["decision"]
    features = symbolic["surviving_features"]

    if decision_expr is None or decision_expr == "extraction_failed":
        return pd.DataFrame()

    transform = symbolic.get("input_transform")
    if transform is None:
        raise ValueError(
            "compute_feature_sensitivity: symbolic result has no 'input_transform' key. "
            "Re-run run_symbolic_extraction to obtain a result dict that includes the raw "
            "to tanh-normalised transform parameters; without these the sensitivity values "
            "cannot be computed in raw-feature units."
        )
    feature_a = transform.get("feature_a", {})
    feature_b = transform.get("feature_b", {})

    sym_vars = {f: sympy.Symbol(f) for f in features}
    X_features = X[features]
    means_raw = X_features.mean()
    stds_raw = X_features.std()

    # Build the centrality point in raw space.
    if eval_point == "median":
        center_raw = X_features.median()
    elif eval_point == "mean":
        center_raw = means_raw
    else:
        raise ValueError(f"eval_point must be 'mean' or 'median'; got {eval_point!r}")

    # Helper: convert one raw feature value to tanh-normalised space.
    def _to_tanh(feat: str, raw_val: float) -> float:
        a = feature_a.get(feat, 0.0)
        b = feature_b.get(feat, 1.0)
        if b == 0:
            b = 1.0  # defensive; shouldn't happen since input_std has +1e-8
        return float(np.tanh((raw_val - a) / b))

    # Tanh-space centrality point (used for the decision-at-center evaluation and as the
    # base point for per-feature finite differences).
    z_center = {f: _to_tanh(f, float(center_raw[f])) for f in features}

    # Lambdify the decision function once for fast repeated evaluation; fall back to
    # per-call subs if lambdify chokes on exotic sympy expressions.
    try:
        decision_fn = sympy.lambdify(
            [sym_vars[f] for f in features],
            decision_expr,
            modules="numpy",
        )
    except Exception:
        decision_fn = None

    # Decision value and sigmoid slope at the centrality point.
    decision_at_center = _evaluate_decision(
        decision_fn, decision_expr, sym_vars, features, z_center
    )
    if np.isfinite(decision_at_center):
        p_at_center = 1.0 / (1.0 + np.exp(-decision_at_center))
        p_slope = p_at_center * (1.0 - p_at_center)
    else:
        p_at_center = float("nan")
        p_slope = float("nan")

    rows = []
    n_singular = 0
    for feat in features:
        sigma_raw = float(stds_raw[feat])
        if not np.isfinite(sigma_raw) or sigma_raw == 0:
            sigma_raw = 1.0  # defensive

        h_raw = fd_step_frac * sigma_raw  # small step for the gradient finite difference

        # --- Gradient: central finite difference in RAW space ---
        z_minus = z_center.copy()
        z_minus[feat] = _to_tanh(feat, float(center_raw[feat]) - h_raw)
        z_plus_small = z_center.copy()
        z_plus_small[feat] = _to_tanh(feat, float(center_raw[feat]) + h_raw)

        f_minus = _evaluate_decision(decision_fn, decision_expr, sym_vars, features, z_minus)
        f_plus_small = _evaluate_decision(
            decision_fn, decision_expr, sym_vars, features, z_plus_small
        )

        if np.isfinite(f_minus) and np.isfinite(f_plus_small):
            deriv_raw = (f_plus_small - f_minus) / (2.0 * h_raw)
            if not np.isfinite(deriv_raw):
                deriv_raw = float("nan")
        else:
            deriv_raw = float("nan")

        # --- Per-σ effect: finite difference across a full raw σ ---
        z_sigma = z_center.copy()
        z_sigma[feat] = _to_tanh(feat, float(center_raw[feat]) + sigma_raw)
        f_sigma = _evaluate_decision(decision_fn, decision_expr, sym_vars, features, z_sigma)

        if np.isfinite(f_sigma) and np.isfinite(decision_at_center):
            sigma_effect = f_sigma - decision_at_center
            sigma_delta_p = sigma_effect * p_slope if np.isfinite(p_slope) else float("nan")
        else:
            sigma_effect = float("nan")
            sigma_delta_p = float("nan")

        # Track features with no usable gradient OR no usable per-σ value.
        if not np.isfinite(deriv_raw) and not np.isfinite(sigma_effect):
            n_singular += 1

        rows.append({
            "feature": feat,
            "mean_value": means_raw[feat],
            "std_value": stds_raw[feat],
            "d_decision/d_feature_at_center": deriv_raw,
            "sigma_effect_on_decision": sigma_effect,
            "sigma_delta_p": sigma_delta_p,
        })

    if n_singular > 0:
        logger.warning(
            "Sensitivity table: %d/%d features have non-finite gradient and per-sigma effect "
            "at the %s; reported as NaN. The symbolic formula has poles near these features' "
            "%s in tanh-normalised space.",
            n_singular, len(features), eval_point, eval_point,
        )

    df = pd.DataFrame(rows).set_index("feature")
    df.attrs["eval_point"] = eval_point  # for the printer to label the table correctly
    return df


# Print the sensitivity DataFrame with NaN-aware formatting; renders singular gradients as 'N/A'.
def print_feature_sensitivity(sensitivity_df: pd.DataFrame) -> None:
    """Print a feature-sensitivity DataFrame; non-finite gradients render as ``N/A``, not ``+nan``."""
    if sensitivity_df.empty:
        print("Sensitivity table is empty (extraction failed).")
        return

    # Format helper for the to_string float_format.
    def fmt(x: float) -> str:
        if not np.isfinite(x):
            return "    N/A "
        return f"{x:+.4f}"

    eval_point = sensitivity_df.attrs.get("eval_point", "mean")
    label = "dataset median" if eval_point == "median" else "dataset mean"
    print(f"Feature sensitivity at the {label}:\n")
    print(sensitivity_df.to_string(float_format=fmt))

    # Footer note when any row has a non-finite entry.
    n_singular = int((~np.isfinite(sensitivity_df.values)).any(axis=1).sum())
    if n_singular > 0:
        alt = "mean" if eval_point == "median" else "median"
        print(
            f"\n  Note: {n_singular} feature(s) have a non-finite gradient at "
            f"the {label} (symbolic formula has a pole near that point); "
            f"rendered as N/A. Try eval_point='{alt}' for a different point, "
            f"or inspect the closed-form decision function for log/sqrt/1-over "
            f"primitives that vanish near the centrality point."
        )


# One-subplot-per-feature plot of the marginal effect on P(up); other features pinned to their median.
def plot_marginal_effects(
    symbolic: dict,
    X: pd.DataFrame,
    n_points: int = 100,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    figsize_per_panel: tuple = (4.5, 4.0),
):
    """Plot the marginal effect of each surviving feature on P(up).

    S1 fix: the symbolic decision function lives in tanh-normalised feature space, so the
    raw sweep must be converted to tanh space (per-feature, via the stored ``input_transform``
    parameters) before being substituted into the formula. Without this conversion, the curve
    plotted is f(raw_value) for a formula that was trained on tanh(raw_value), and the entire
    figure is wrong even though its x-axis labels look correct.

    Sweeps each feature across its empirical [q_low, q_high] range in raw space while pinning
    the others at their raw-data medians, converts each substitution point to tanh space,
    evaluates the symbolic decision function there, and plots the sigmoid-transformed P(up)
    against the raw sweep values on the x-axis. Median (rather than mean) is used for the
    pinned features because it's more robust on skewed distributions.
    """
    import matplotlib.pyplot as plt

    decision_expr = symbolic["sympy_objects"]["decision"]
    features = symbolic["surviving_features"]

    if decision_expr is None or decision_expr == "extraction_failed":
        raise ValueError("Symbolic extraction failed; no decision function to plot.")

    # S1: pull the raw → tanh transform; fail loudly if missing rather than plotting wrong.
    transform = symbolic.get("input_transform")
    if transform is None:
        raise ValueError(
            "plot_marginal_effects: symbolic result has no 'input_transform' key. "
            "Re-run run_symbolic_extraction to obtain a result dict that includes the raw "
            "to tanh-normalised transform parameters; without these the marginal-effect "
            "plot would evaluate the formula at wrong points and mislead readers."
        )
    feature_a = transform.get("feature_a", {})
    feature_b = transform.get("feature_b", {})

    # Lambdify the symbolic decision function for fast numeric evaluation across the sweep.
    sym_vars = {f: sympy.Symbol(f) for f in features}
    decision_fn = sympy.lambdify(
        [sym_vars[f] for f in features],
        decision_expr,
        modules="numpy",
    )

    def _to_tanh(feat: str, raw_val: float) -> float:
        a = feature_a.get(feat, 0.0)
        b = feature_b.get(feat, 1.0) or 1.0
        return float(np.tanh((raw_val - a) / b))

    X_features = X[features]
    n_feat = len(features)
    fig, axes = plt.subplots(
        1, n_feat,
        figsize=(figsize_per_panel[0] * n_feat, figsize_per_panel[1]),
        sharey=True,
    )
    if n_feat == 1:
        axes = [axes]

    # Precompute the tanh-space values of pinned features (raw medians of the OTHER features).
    fixed_raw = {f: float(X_features[f].median()) for f in features}
    fixed_z = {f: _to_tanh(f, fixed_raw[f]) for f in features}

    # One subplot per feature: sweep it from q_low to q_high in raw space, hold others at median.
    for ax, feat in zip(axes, features):
        sweep_raw = np.linspace(
            X_features[feat].quantile(quantile_low),
            X_features[feat].quantile(quantile_high),
            n_points,
        )

        # For each sweep point: bump only this feature in tanh space, leave others pinned.
        p_up = []
        for v_raw in sweep_raw:
            z_args = dict(fixed_z)
            z_args[feat] = _to_tanh(feat, float(v_raw))
            try:
                d = decision_fn(*[z_args[f] for f in features])
            except (ZeroDivisionError, TypeError, ValueError, OverflowError):
                p_up.append(np.nan)
                continue
            # Guard against NaN/Inf from the lambdified function at poles.
            if not np.isfinite(d):
                p_up.append(np.nan)
            else:
                p_up.append(1.0 / (1.0 + np.exp(-d)))

        ax.plot(sweep_raw, p_up, linewidth=2)
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5,
                   label="P=0.5 (abstention)")
        ax.axvline(X_features[feat].median(), color="red", linestyle=":",
                   alpha=0.5, label="Median")
        ax.set_xlabel(feat)
        ax.set_title(f"Marginal effect: {feat}")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("P(up)")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(
        "Marginal effect of each feature on P(up)\n"
        "(other features held at their median, formula evaluated in tanh-normalised space)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return fig


# Print the per-feature term count + sensitivity table side by side; surfaces structural vs numeric importance.
def print_term_structure_summary(
    sensitivity_df: pd.DataFrame,
    symbolic: dict,
) -> None:
    """Print per-feature term counts (a structural importance proxy) joined to the sensitivity table.

    ``n_terms_in_formula`` counts how many times each feature's name appears in the closed-form
    decision expression, which is a structural-importance proxy independent of the numeric gradient.
    """
    decision_expr = symbolic["sympy_objects"]["decision"]
    features = symbolic["surviving_features"]

    if decision_expr is None or decision_expr == "extraction_failed":
        print("Symbolic extraction failed; no term-structure summary available.")
        return

    formula_str = str(decision_expr)
    term_counts = {feat: formula_str.count(feat) for feat in features}

    summary = sensitivity_df.copy()
    summary["n_terms_in_formula"] = pd.Series(term_counts)
    # S1: column names updated (d_feature_at_mean → d_feature_at_center to match the new
    # eval_point semantics; approx_sigma_delta_p → sigma_delta_p because the value is no
    # longer an approximation tied to P=0.5).
    summary = summary[[
        "mean_value", "std_value", "n_terms_in_formula",
        "d_decision/d_feature_at_center", "sigma_effect_on_decision",
        "sigma_delta_p",
    ]]

    # NaN-aware formatter shared with print_feature_sensitivity.
    def fmt(x: float) -> str:
        if not np.isfinite(x):
            return "    N/A "
        return f"{x:+.4f}"

    print("Formula term-structure summary:\n")
    print(summary.to_string(float_format=fmt))