"""
7) Preprocessing
====================
The three transformations that MUST happen inside the CPCV loop, fitted
on training data only: fractional differentiation (AFML Chapter 5),
feature scaling, and feature selection (AFML Chapter 8).

Each function takes train/test data and returns transformed data,
ensuring zero leakage.

Feature selection uses Multi-Model Mean Decrease Accuracy (MDA) with
purged inner cross-validation (AFML §8.4). Permutation importance is
computed using both a Random Forest (captures nonlinear interactions)
and a Logistic Regression (captures linear effects), then averaged.
This prevents selection bias toward any single model architecture.
A feature is selected if its averaged MDA > 0 and it ranks in the
top K by averaged MDA value.

Lag features are part of the global feature universe alongside TA,
mathematical, and external features. All non-AR models receive lag
features through the standard MDA selection step (lags compete on
equal footing with engineered features for the top-k cap). AR
Logistic continues to consume the six precomputed lag columns by
name from the pre-MDA feature matrix, preserving its role as the
pure-autoregressive baseline.
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

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# FFD
FFD_D_RANGE = (0.0, 1.0, 0.05)     # (start, stop, step) for d sweep
FFD_THRESHOLD = 1e-4                 # weight truncation threshold τ
FFD_ADF_SIGNIFICANCE = 0.05          # ADF rejection threshold
FFD_MAX_LOOKBACK = 200               # hard cap on FFD lookback length

# Feature selection (multi-model MDA with purged inner CV)
MDA_N_ESTIMATORS = 500
MDA_N_INNER_FOLDS = 3
MDA_TOP_K_FRAC = 0.4                 # cap at top 40% of total features

# Minimum features to keep (hard floor)
MIN_FEATURES = 5


# =====================================================================
# FFD — Fractional Differentiation (AFML Chapter 5)
# =====================================================================
def _compute_ffd_weights(d: float, threshold: float = FFD_THRESHOLD,
                         max_lookback: int = FFD_MAX_LOOKBACK) -> np.ndarray:
    """Compute FFD weights via the recursive formula.

    ω_0 = 1, ω_k = -ω_{k-1} × (d - k + 1) / k, truncated when
    |ω_k| < threshold or k reaches max_lookback.
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold or k >= max_lookback:
            break
        weights.append(w)
        k += 1
    return np.array(weights, dtype=np.float64)


def find_optimal_d(
    series: pd.Series,
    d_range: tuple[float, float, float] = FFD_D_RANGE,
    threshold: float = FFD_THRESHOLD,
    significance: float = FFD_ADF_SIGNIFICANCE,
) -> float:
    """Find the minimum fractional differentiation order d* that achieves stationarity.

    Parameters
    ----------
    series : pd.Series
        Price-level series (e.g., log close) from the training fold only.
    d_range : tuple
        (start, stop, step) for the d sweep.
    threshold : float
        FFD weight truncation threshold.
    significance : float
        ADF p-value rejection threshold.

    Returns
    -------
    float
        Optimal d* value.
    """
    d_values = np.arange(d_range[0], d_range[1] + d_range[2] / 2, d_range[2])
    sweep_results = {}

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


def apply_ffd(
    series: pd.Series, d: float, threshold: float = FFD_THRESHOLD
) -> pd.Series:
    """Apply fixed-width window fractional differentiation at order *d*.

    Parameters
    ----------
    series : pd.Series
        Input series (price-level or log-price).
    d : float
        Fractional differentiation order.
    threshold : float
        Weight truncation threshold.

    Returns
    -------
    pd.Series
        FFD-transformed series, same index as input. First K rows are NaN
        where K is the lookback length (number of FFD weights - 1).
    """
    weights = _compute_ffd_weights(d, threshold)
    K = len(weights)
    values = series.values
    n = len(values)

    result = np.full(n, np.nan)
    for t in range(K - 1, n):
        result[t] = np.dot(weights, values[t - K + 1 : t + 1][::-1])

    return pd.Series(result, index=series.index, name=series.name)


def ffd_transform(
    X_full: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ffd_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Orchestrate FFD for the current fold.

    Applies FFD to the FULL series (so test observations have lookback
    history), but estimates d* from training data only.

    Parameters
    ----------
    X_full : pd.DataFrame
        Complete feature matrix (all observations, not yet split).
    train_idx, test_idx : np.ndarray
        Positional indices for this fold.
    ffd_columns : list[str]
        Columns requiring FFD transformation.

    Returns
    -------
    X_train_ffd, X_test_ffd : pd.DataFrame
        Transformed and NaN-free feature matrices for this fold.
    ffd_info : dict
        ``{column_name: d_star_value}``.
    """
    ffd_info = {}
    X_transformed = X_full.copy()

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

    # forward-fill NaN in non-FFD columns (from external data availability gaps)
    ffd_cols_present = [c for c in ffd_columns if c in X_train.columns]
    non_ffd_cols = [c for c in X_train.columns if c not in ffd_cols_present]

    if non_ffd_cols:
        X_train[non_ffd_cols] = X_train[non_ffd_cols].ffill().bfill()
        X_test[non_ffd_cols] = X_test[non_ffd_cols].ffill().bfill()

    # only drop rows where FFD columns have NaN (from lookback)
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


# =====================================================================
# Scaling
# =====================================================================
def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Fit a RobustScaler on X_train and transform both sets.

    RobustScaler uses median/IQR, making it resistant to fat-tailed
    outliers common in BTC features.

    Returns
    -------
    X_train_scaled, X_test_scaled : pd.DataFrame
    scaler : RobustScaler
        Fitted scaler for potential inverse-transformation.
    """
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )

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


# =====================================================================
# Feature Selection — Multi-Model MDA (AFML Chapter 8)
# =====================================================================
def _compute_mda_single_model(
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    t1_train: pd.Series,
) -> pd.Series:
    """Compute MDA using a single classifier via purged inner CV.

    This is the core permutation importance loop. Called once per
    inner model (RF and Logistic Regression).

    Parameters
    ----------
    clf : sklearn estimator (unfitted)
        Classifier to use. Will be cloned/refitted per inner fold.
    X_train, y_train, w_train, t1_train : pd.DataFrame / pd.Series
        Training data for the current outer CPCV fold.

    Returns
    -------
    pd.Series
        Mean MDA per feature, sorted descending.
    """
    n_folds = MDA_N_INNER_FOLDS
    T = len(X_train)
    fold_size = T // n_folds
    rng = np.random.RandomState(42)

    mda_scores = pd.DataFrame(index=X_train.columns)

    for fold_i in range(n_folds):
        inner_test_start = fold_i * fold_size
        inner_test_end = (fold_i + 1) * fold_size if fold_i < n_folds - 1 else T

        inner_test_idx = list(range(inner_test_start, inner_test_end))
        inner_train_idx = list(range(0, inner_test_start)) + list(range(inner_test_end, T))

        # purge: remove inner-train observations whose t1 overlaps inner-test
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

        # clone and fit the classifier
        from sklearn.base import clone
        model = clone(clf)
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        baseline_f1 = f1_score(y_te, model.predict(X_te), average="macro")

        fold_mda = {}
        for col in X_train.columns:
            X_te_perm = X_te.copy()
            X_te_perm[col] = rng.permutation(X_te_perm[col].values)
            perm_f1 = f1_score(y_te, model.predict(X_te_perm), average="macro")
            fold_mda[col] = baseline_f1 - perm_f1

        mda_scores[f"fold_{fold_i}"] = pd.Series(fold_mda)

    avg_mda = mda_scores.mean(axis=1)
    return avg_mda


def compute_multi_model_mda(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    t1_train: pd.Series,
) -> pd.DataFrame:
    """Multi-Model MDA: average permutation importance from RF and LR.

    Computes MDA separately using a Random Forest (captures nonlinear
    interactions and ensemble effects) and a Logistic Regression
    (captures linear relationships). The final MDA is the average of
    both, ensuring no single model architecture dominates feature
    selection.

    A feature ranks high only if it demonstrably contributes to BOTH
    a linear and nonlinear classifier, or contributes very strongly
    to one of them.

    Returns
    -------
    pd.DataFrame
        Columns: MDA_RF, MDA_LR, MDA (average). Sorted by MDA descending.
    """
    # ── Random Forest MDA ─────────────────────────────────────────────
    rf_clf = RandomForestClassifier(
        n_estimators=MDA_N_ESTIMATORS,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    logger.info("Computing MDA (Random Forest)...")
    mda_rf = _compute_mda_single_model(rf_clf, X_train, y_train, w_train, t1_train)

    # ── Logistic Regression MDA ───────────────────────────────────────
    lr_clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    logger.info("Computing MDA (Logistic Regression)...")
    mda_lr = _compute_mda_single_model(lr_clf, X_train, y_train, w_train, t1_train)

    # ── Average ───────────────────────────────────────────────────────
    results = pd.DataFrame({
        "MDA_RF": mda_rf,
        "MDA_LR": mda_lr,
    })
    results["MDA"] = results[["MDA_RF", "MDA_LR"]].mean(axis=1)

    return results.sort_values("MDA", ascending=False)


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
    """Select features via multi-model MDA (AFML §8.4).

    Two-stage selection:
      1. Compute averaged MDA (RF + Logistic Regression).
         Keep features with MDA > 0 (permuting them hurts at least
         one model type on average).
      2. Cap at top_k_frac of total features (by MDA rank).
         Minimum floor of MIN_FEATURES enforced.

    Using both a linear and nonlinear model for permutation importance
    prevents selection bias toward any single architecture. Features
    must demonstrate value across model families to rank high.

    All columns of ``X_train`` enter the MDA pool, including lag
    features. Lag and engineered features compete on equal footing
    for the top-k cap; whether a given fold selects lags depends on
    their permutation importance relative to TA, math, and external
    features. AR Logistic still consumes its six lag columns by name
    from the pre-MDA matrix routed through ``X_tr_full`` in the
    pipeline, so its behaviour is unaffected by what MDA selects.

    Parameters
    ----------
    top_k_frac : float, optional
        Cap selection at this fraction of total features.
        Default from MDA_TOP_K_FRAC module constant.
    split_idx : int, optional
        Zero-based index of the current CPCV split. When provided,
        any warning emitted by this function is prefixed with the
        split identifier so the deferred-warning summary printed by
        ``run_cpcv_pipeline`` can show which fold(s) actually
        triggered the warning. Pass via keyword only.
    n_splits : int, optional
        Total number of CPCV splits. Used together with ``split_idx``
        to render the prefix as ``[split K/N]``.

    Returns
    -------
    list[str]
        Sorted list of selected feature names.
    """
    if top_k_frac is None:
        top_k_frac = MDA_TOP_K_FRAC

    n_total = X_train.shape[1]

    # Build a per-fold prefix for warning messages so the post-run
    # summary printed by ``run_cpcv_pipeline`` identifies which split
    # triggered the warning rather than leaving the reader to assume
    # the issue applied to every fold.
    if split_idx is not None and n_splits is not None:
        split_prefix = f"[split {split_idx + 1}/{n_splits}] "
    elif split_idx is not None:
        split_prefix = f"[split {split_idx + 1}] "
    else:
        split_prefix = ""

    # ── Compute multi-model MDA on the full feature universe ──────────
    mda_results = compute_multi_model_mda(X_train, y_train, w_train, t1_train)

    # ── Select: all features with averaged MDA > 0 ────────────────────
    mda_positive = mda_results[mda_results["MDA"] > 0]
    mda_eliminated = mda_results[mda_results["MDA"] <= 0]
    n_passed = len(mda_positive)
    n_eliminated = len(mda_eliminated)

    # fallback: if too few pass, take top MIN_FEATURES by MDA
    if n_passed < MIN_FEATURES:
        logger.warning(
            "%sOnly %d features with MDA > 0. Taking top %d by MDA value.",
            split_prefix, n_passed, MIN_FEATURES,
        )
        selected_df = mda_results.head(MIN_FEATURES)
    else:
        selected_df = mda_positive

    # ── Cap via top_k_frac ────────────────────────────────────────────
    if top_k_frac is not None:
        max_features = max(int(n_total * top_k_frac), MIN_FEATURES)
        if len(selected_df) > max_features:
            selected_df = selected_df.head(max_features)
            logger.debug(
                "MDA pool capped: %d → %d features (top_k_frac=%.2f).",
                n_passed, max_features, top_k_frac,
            )

    selected = sorted(selected_df.index.tolist())

    # ── Log full rankings at DEBUG (kept available for diagnostics, but
    # not flooded into the notebook by default; raise the logger to DEBUG
    # to see them again).
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
    # The dropped-features list is the complement of `selected`; it adds
    # ~44 names per split with no new information and dominates notebook
    # output size. Use `set(X_train.columns) - set(selected)` if you
    # ever need it outside the pipeline output.

    return selected


# =====================================================================
# Orchestration
# =====================================================================
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
    """Full preprocessing for one CPCV fold: FFD → scaling → selection.

    Returns DataFrames with ALL columns (pre-selection) so the pipeline
    can provide full-column DataFrames to AR Logistic. The selected
    feature list is returned separately for the pipeline to apply.

    Lag columns sit alongside engineered features in the returned
    DataFrames and are now eligible for MDA selection on equal footing
    with TA, math, and external features. AR Logistic continues to
    consume its six lag columns by name from the pre-selection matrix
    via the pipeline's ``X_tr_full`` route, independently of whether
    MDA happens to select any lag columns for the other models.

    Parameters
    ----------
    X_full : pd.DataFrame
        Complete feature matrix (all observations).
    train_idx, test_idx : np.ndarray
        Positional indices for this fold.
    y_train, w_train, t1_train : pd.Series
        Labels, weights, and barrier timestamps for training observations.
    ffd_columns : list[str]
        Columns requiring FFD transformation.
    top_k_frac : float, optional
        Cap selection at this fraction of total features. Default from
        MDA_TOP_K_FRAC module constant.
    skip_selection : bool
        If True, skip feature selection and return all columns.
        Used when only AR Logistic is being evaluated.
    split_idx : int, optional
        Zero-based index of the current CPCV split. Forwarded to
        ``select_features`` so any warning it emits identifies the
        fold that triggered it. Pass via keyword only.
    n_splits : int, optional
        Total number of CPCV splits. Forwarded together with
        ``split_idx`` to render warning prefixes as ``[split K/N]``.

    Returns
    -------
    X_train : pd.DataFrame
        Processed training features (all columns, pre-selection).
    X_test : pd.DataFrame
        Processed test features (all columns, pre-selection).
    selected : list[str]
        Names of selected features.
    info : dict
        Preprocessing metadata (FFD d* values, scaler, selected features).
    """
    # 1. FFD on price-level columns (applied to full series, d* from train only)
    X_train, X_test, ffd_info = ffd_transform(
        X_full, train_idx, test_idx, ffd_columns
    )

    # 2. Re-align y, w, t1 after FFD drops NaN rows
    common_train = X_train.index.intersection(y_train.index)
    y_train = y_train.loc[common_train]
    w_train = w_train.loc[common_train]
    t1_train = t1_train.loc[common_train]

    # 3. Scale all features
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # 4. Select features
    if skip_selection:
        selected = sorted(X_train.columns.tolist())
        logger.info("Feature selection skipped (AR Logistic uses lagged returns only)")
    else:
        selected = select_features(
            X_train, y_train, w_train, t1_train, top_k_frac,
            split_idx=split_idx, n_splits=n_splits,
        )

    n_cal = int(len(X_train) * 0.2)
    # Note: the per-fold sample sizes (train + cal + test) are stored in
    # `info` and surfaced once per split by the calling pipeline, so we no
    # longer print them here per call. Two `preprocess_fold` invocations
    # per split (one per model in some configurations) was producing two
    # near-identical lines that bloated the cell output.

    info = {
        "ffd": ffd_info,
        "scaler": scaler,
        "selected_features": selected,
    }

    return X_train, X_test, selected, info