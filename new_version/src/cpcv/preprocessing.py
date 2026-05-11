"""
9) Preprocessing
====================
The three per-fold transformations that must happen inside the CPCV loop,
fitted on training data only:

  1. Fractional differentiation (AFML Ch. 5)
  2. Feature scaling (RobustScaler, median/IQR)
  3. Feature selection via Multi-Model MDA with purged inner CV (AFML §8.4)

Multi-Model MDA averages permutation importance from a Random Forest (nonlinear
interactions) and a Logistic Regression (linear effects), so no single model
architecture dominates selection. Lag features sit inside the MDA pool on equal
footing with TA, math, and external features; AR Logistic still consumes its
ten lag columns by name from the pre-MDA matrix and is unaffected by what MDA
selects for the other models.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

# --- Module-level constants -------------------------------------------------

# FFD (AFML Ch. 5) — d* sweep and weight truncation.
FFD_D_RANGE = (0.0, 1.0, 0.05)     # (start, stop, step) for d sweep
FFD_THRESHOLD = 1e-4                 # weight truncation threshold τ
FFD_ADF_SIGNIFICANCE = 0.05          # ADF rejection threshold
FFD_MAX_LOOKBACK = 200               # hard cap on FFD lookback length

# Multi-Model MDA (AFML §8.4). The notebook overrides MDA_TOP_K_FRAC to lock
# the per-fold selected-feature count for the production configuration.
MDA_N_ESTIMATORS = 500
MDA_N_INNER_FOLDS = 3
MDA_TOP_K_FRAC = 0.4                 # cap at top 40% of total features (notebook overrides this)

# Hard floor on the number of selected features.
MIN_FEATURES = 5


# --- 1. FFD — Fractional Differentiation (AFML Ch. 5) ----------------------
# Recursive FFD weight kernel: ω_0 = 1, ω_k = -ω_{k-1} · (d - k + 1) / k.
def _compute_ffd_weights(d: float, threshold: float = FFD_THRESHOLD,
                         max_lookback: int = FFD_MAX_LOOKBACK) -> np.ndarray:
    """Return the FFD weight vector for order ``d``, truncated when |ω_k| < threshold."""
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold or k >= max_lookback:
            break
        weights.append(w)
        k += 1
    return np.array(weights, dtype=np.float64)


# d* search: minimum order achieving ADF-stationary FFD output on the training slice.
def find_optimal_d(
    series: pd.Series,
    d_range: tuple[float, float, float] = FFD_D_RANGE,
    threshold: float = FFD_THRESHOLD,
    significance: float = FFD_ADF_SIGNIFICANCE,
) -> float:
    """Sweep d ∈ d_range and return the smallest value whose FFD output rejects the ADF unit root."""
    d_values = np.arange(d_range[0], d_range[1] + d_range[2] / 2, d_range[2])
    sweep_results = {}

    # For each candidate d, run FFD on the series and store the ADF p-value.
    for d in d_values:
        d = round(d, 4)
        if d == 0.0:
            sweep_results[d] = 1.0
            continue

        ffd_series = apply_ffd(series, d, threshold).dropna()

        if len(ffd_series) < 50:
            sweep_results[d] = 1.0
            continue

        try:
            adf_stat, pval, *_ = adfuller(ffd_series, maxlag=14, autolag="AIC")
            sweep_results[d] = pval
        except (ValueError, np.linalg.LinAlgError):
            sweep_results[d] = 1.0

    # Take the smallest d achieving stationarity; AFML's "minimum memory loss" principle.
    d_star = None
    for d in sorted(sweep_results.keys()):
        if sweep_results[d] < significance:
            d_star = d
            break

    if d_star is None:
        d_star = 1.0
        logger.warning(
            "No d in range achieved stationarity (p < %.2f). Defaulting to d*=1.0.",
            significance,
        )

    logger.debug(
        "FFD d* search: optimal d*=%.2f (p=%.4f).",
        d_star,
        sweep_results.get(d_star, np.nan),
    )

    return d_star


# Fixed-width window FFD: convolve the series with the truncated weight vector.
def apply_ffd(
    series: pd.Series, d: float, threshold: float = FFD_THRESHOLD
) -> pd.Series:
    """Return ``series`` fractionally-differenced at order ``d``; first ``K-1`` rows are NaN."""
    weights = _compute_ffd_weights(d, threshold)
    K = len(weights)
    values = series.values
    n = len(values)

    # Each output is the dot product of the K most recent values (reversed) with the weights.
    result = np.full(n, np.nan)
    for t in range(K - 1, n):
        result[t] = np.dot(weights, values[t - K + 1 : t + 1][::-1])

    return pd.Series(result, index=series.index, name=series.name)


# Per-fold FFD orchestrator: estimates d* from train only, then applies FFD to the full series.
def ffd_transform(
    X_full: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ffd_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply FFD to the full series (so test rows have lookback) with d* from train only.

    Returns the train and test slices with FFD-induced NaN rows dropped on the
    FFD columns. Non-FFD columns are forward/backward-filled to handle external-data gaps.
    """
    ffd_info = {}
    X_transformed = X_full.copy()

    # Fit d* on each FFD column using only the training slice, then apply to the full series.
    for col in ffd_columns:
        if col not in X_transformed.columns:
            logger.warning("FFD: column '%s' not found, skipping.", col)
            continue

        train_series = X_full[col].iloc[train_idx]
        d_star = find_optimal_d(train_series)
        ffd_info[col] = d_star

        X_transformed[col] = apply_ffd(X_full[col], d_star)
        lookback = len(_compute_ffd_weights(d_star)) - 1
        logger.debug("FFD: col='%s', d*=%.2f, lookback=%d obs.", col, d_star, lookback)

    X_train = X_transformed.iloc[train_idx].copy()
    X_test = X_transformed.iloc[test_idx].copy()

    # Non-FFD columns are ffill/bfill'd for external-data availability gaps.
    ffd_cols_present = [c for c in ffd_columns if c in X_train.columns]
    non_ffd_cols = [c for c in X_train.columns if c not in ffd_cols_present]

    if non_ffd_cols:
        X_train[non_ffd_cols] = X_train[non_ffd_cols].ffill().bfill()
        X_test[non_ffd_cols] = X_test[non_ffd_cols].ffill().bfill()

    # Drop rows only where the FFD columns themselves are NaN (the lookback head).
    if ffd_cols_present:
        train_mask = X_train[ffd_cols_present].notna().all(axis=1)
        test_mask = X_test[ffd_cols_present].notna().all(axis=1)
    else:
        train_mask = pd.Series(True, index=X_train.index)
        test_mask = pd.Series(True, index=X_test.index)

    n_train_dropped = (~train_mask).sum()
    n_test_dropped = (~test_mask).sum()

    X_train = X_train.loc[train_mask]
    X_test = X_test.loc[test_mask]

    if n_train_dropped > 0 or n_test_dropped > 0:
        logger.debug(
            "FFD: dropped %d train, %d test NaN rows from FFD lookback.",
            n_train_dropped, n_test_dropped,
        )

    logger.debug("FFD transform complete: %s", ffd_info)
    return X_train, X_test, ffd_info


# --- 2. Scaling ------------------------------------------------------------
# RobustScaler chosen over StandardScaler: median/IQR is resilient to fat-tailed BTC features.
def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Fit a RobustScaler on ``X_train`` and apply to both sets; return the fitted scaler too."""
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )

    # Guard against empty test slices (can happen if FFD drops the whole test head).
    if len(X_test) == 0:
        logger.warning("Scaling: test set is empty (0 rows). Returning empty DataFrame.")
        X_test_scaled = X_test.copy()
    else:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns,
        )

    logger.debug("Scaling: RobustScaler fitted on %d training rows.", len(X_train))
    return X_train_scaled, X_test_scaled, scaler


# --- 3. Feature Selection — Multi-Model MDA (AFML Ch. 8) -------------------
# Permutation-importance loop on a single classifier with purged inner CV.
def _compute_mda_single_model(
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    t1_train: pd.Series,
) -> pd.Series:
    """Per-feature mean MDA score across ``MDA_N_INNER_FOLDS`` purged inner folds."""
    n_folds = MDA_N_INNER_FOLDS
    T = len(X_train)
    fold_size = T // n_folds
    rng = np.random.RandomState(42)

    mda_scores = pd.DataFrame(index=X_train.columns)

    for fold_i in range(n_folds):
        # Contiguous-block inner test slice; everything else is candidate inner train.
        inner_test_start = fold_i * fold_size
        inner_test_end = (fold_i + 1) * fold_size if fold_i < n_folds - 1 else T

        inner_test_idx = list(range(inner_test_start, inner_test_end))
        inner_train_idx = list(range(0, inner_test_start)) + list(range(inner_test_end, T))

        # Purge inner-train rows whose t1 overlaps the inner-test window (AFML Snippet 7.1).
        if len(inner_test_idx) > 0:
            t_test_start = X_train.index[inner_test_idx[0]]
            t_test_end = X_train.index[inner_test_idx[-1]]

            purged = set()
            for i in inner_train_idx:
                t_i_end = t1_train.iloc[i]
                if pd.isna(t_i_end):
                    continue
                t_i_start = X_train.index[i]
                if t_test_start <= t_i_start <= t_test_end:
                    purged.add(i)
                elif t_test_start <= t_i_end <= t_test_end:
                    purged.add(i)
                elif t_i_start <= t_test_start and t_test_end <= t_i_end:
                    purged.add(i)
            inner_train_idx = [i for i in inner_train_idx if i not in purged]

        if len(inner_train_idx) < 20 or len(inner_test_idx) < 10:
            continue

        X_tr = X_train.iloc[inner_train_idx]
        y_tr = y_train.iloc[inner_train_idx]
        w_tr = w_train.iloc[inner_train_idx]
        X_te = X_train.iloc[inner_test_idx]
        y_te = y_train.iloc[inner_test_idx]

        # Clone-and-fit so the original estimator object stays unfitted across folds.
        from sklearn.base import clone
        model = clone(clf)
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        baseline_f1 = f1_score(y_te, model.predict(X_te), average="macro")

        # Permute one column at a time and measure the F1 drop; that drop is the MDA.
        fold_mda = {}
        for col in X_train.columns:
            X_te_perm = X_te.copy()
            X_te_perm[col] = rng.permutation(X_te_perm[col].values)
            perm_f1 = f1_score(y_te, model.predict(X_te_perm), average="macro")
            fold_mda[col] = baseline_f1 - perm_f1

        mda_scores[f"fold_{fold_i}"] = pd.Series(fold_mda)

    avg_mda = mda_scores.mean(axis=1)
    return avg_mda


# Multi-Model MDA: average permutation importance from a Random Forest and a Logistic Regression.
def compute_multi_model_mda(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    t1_train: pd.Series,
) -> pd.DataFrame:
    """Return ``DataFrame[MDA_RF, MDA_LR, MDA]`` sorted by averaged MDA descending.

    A feature ranks high only if it contributes meaningfully to a linear AND a
    nonlinear classifier, or very strongly to one of them; this guards against
    selection bias toward any single model architecture.
    """
    # Random Forest captures nonlinear and interaction effects.
    rf_clf = RandomForestClassifier(
        n_estimators=MDA_N_ESTIMATORS,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    logger.info("Computing MDA (Random Forest)...")
    mda_rf = _compute_mda_single_model(rf_clf, X_train, y_train, w_train, t1_train)

    # Logistic Regression captures linear effects on the standardised feature scale.
    lr_clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    logger.info("Computing MDA (Logistic Regression)...")
    mda_lr = _compute_mda_single_model(lr_clf, X_train, y_train, w_train, t1_train)

    # The final score is the simple average; a feature wins only by performing in both worlds.
    results = pd.DataFrame({
        "MDA_RF": mda_rf,
        "MDA_LR": mda_lr,
    })
    results["MDA"] = results[["MDA_RF", "MDA_LR"]].mean(axis=1)

    return results.sort_values("MDA", ascending=False)


# Public feature-selection entry point: two-stage filter (MDA > 0, then top-K cap).
def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    t1_train: pd.Series,
    top_k_frac: float | None = None,
    *,
    split_idx: int | None = None,
    n_splits: int | None = None,
) -> list[str]:
    """Select features via multi-model MDA (AFML §8.4); return their column names sorted.

    Two-stage selection:
      1. Keep features whose averaged MDA > 0 (permuting them hurts at least one model).
      2. Cap the surviving pool at ``top_k_frac`` of total features, with ``MIN_FEATURES`` floor.

    Lag features participate in MDA on equal footing with TA, math, and external
    features; AR Logistic still consumes its ten lag columns by name from the
    pre-MDA matrix via the pipeline's ``X_tr_full`` route, so its behaviour is
    independent of what MDA selects for the other models.

    ``split_idx`` and ``n_splits`` are used to prefix any warning with ``[split K/N]``
    so the deferred-warning summary printed by ``run_cpcv_pipeline`` identifies
    which folds actually triggered the warning.
    """
    if top_k_frac is None:
        top_k_frac = MDA_TOP_K_FRAC

    n_total = X_train.shape[1]

    # Build a per-fold prefix so warnings can be attributed to a specific split downstream.
    if split_idx is not None and n_splits is not None:
        split_prefix = f"[split {split_idx + 1}/{n_splits}] "
    elif split_idx is not None:
        split_prefix = f"[split {split_idx + 1}] "
    else:
        split_prefix = ""

    # Stage 1: compute averaged MDA across the full feature universe.
    mda_results = compute_multi_model_mda(X_train, y_train, w_train, t1_train)

    # Filter to strictly positive MDA; everything else is eliminated by the linear/nonlinear consensus.
    mda_positive = mda_results[mda_results["MDA"] > 0]
    mda_eliminated = mda_results[mda_results["MDA"] <= 0]
    n_passed = len(mda_positive)
    n_eliminated = len(mda_eliminated)

    # Fallback: if positive-MDA set is too small, take the top MIN_FEATURES by score.
    if n_passed < MIN_FEATURES:
        logger.warning(
            "%sOnly %d features with MDA > 0. Taking top %d by MDA value.",
            split_prefix, n_passed, MIN_FEATURES,
        )
        selected_df = mda_results.head(MIN_FEATURES)
    else:
        selected_df = mda_positive

    # Stage 2: cap at top_k_frac · n_total, with a hard floor at MIN_FEATURES.
    if top_k_frac is not None:
        max_features = max(int(n_total * top_k_frac), MIN_FEATURES)
        if len(selected_df) > max_features:
            selected_df = selected_df.head(max_features)
            logger.debug(
                "MDA pool capped: %d → %d features (top_k_frac=%.2f).",
                n_passed, max_features, top_k_frac,
            )

    selected = sorted(selected_df.index.tolist())

    # Full ranking is only emitted at DEBUG to keep notebook output manageable.
    mda_results["selected"] = mda_results.index.isin(selected)
    logger.debug(
        "Feature selection rankings:\n%s", mda_results.to_string()
    )

    logger.info(
        "Multi-model MDA: %d/%d passed (MDA > 0), %d eliminated",
        n_passed, n_total, n_eliminated,
    )
    if n_passed > len(selected):
        logger.info(
            "  Capped at %d features (top_k_frac=%s)",
            len(selected), top_k_frac,
        )
    logger.info("  Selected (%d): %s", len(selected), selected)

    return selected


# --- 4. Orchestration ------------------------------------------------------
# Per-fold pipeline: FFD → re-align labels → scale → select. Returns full-column matrices.
def preprocess_fold(
    X_full: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: pd.Series,
    w_train: pd.Series,
    t1_train: pd.Series,
    ffd_columns: list[str],
    top_k_frac: float | None = None,
    skip_selection: bool = False,
    *,
    split_idx: int | None = None,
    n_splits: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Run full preprocessing for one CPCV fold; return (X_train, X_test, selected, info).

    Returned ``X_train`` and ``X_test`` carry ALL columns (pre-selection) so the
    pipeline can route the full pre-MDA matrix to AR Logistic via ``X_tr_full``.
    The selected feature list is returned separately and applied by the caller.

    ``skip_selection=True`` is used when only AR Logistic is being evaluated.
    """
    # 1. FFD on price-level columns: d* from train slice, applied to the full series.
    X_train, X_test, ffd_info = ffd_transform(
        X_full, train_idx, test_idx, ffd_columns
    )

    # 2. Re-align labels/weights/t1 because FFD drops the lookback-head NaN rows.
    common_train = X_train.index.intersection(y_train.index)
    y_train = y_train.loc[common_train]
    w_train = w_train.loc[common_train]
    t1_train = t1_train.loc[common_train]

    # 3. Scale every feature using a RobustScaler fitted on train only.
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # 4. Feature selection (skipped when only AR Logistic is being evaluated).
    if skip_selection:
        selected = sorted(X_train.columns.tolist())
        logger.info("Feature selection skipped (AR Logistic uses lagged returns only)")
    else:
        selected = select_features(
            X_train, y_train, w_train, t1_train, top_k_frac,
            split_idx=split_idx, n_splits=n_splits,
        )

    # Per-fold sample sizes are surfaced once by the calling pipeline; we no longer print here.
    info = {
        "ffd": ffd_info,
        "scaler": scaler,
        "selected_features": selected,
    }

    return X_train, X_test, selected, info