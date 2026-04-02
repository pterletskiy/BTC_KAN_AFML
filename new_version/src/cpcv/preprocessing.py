"""
7) Preprocessing
====================
The three transformations that MUST happen inside the CPCV loop, fitted
on training data only: fractional differentiation (AFML Chapter 5),
feature scaling, and feature selection (AFML Chapter 8).

Each function takes train/test data and returns transformed data,
ensuring zero leakage.
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
FFD_THRESHOLD = 1e-4                 # weight truncation threshold τ (1e-4 keeps lookback manageable)
FFD_ADF_SIGNIFICANCE = 0.05          # ADF rejection threshold
FFD_MAX_LOOKBACK = 200               # hard cap on FFD lookback length

# Feature selection (MDA + SFI composite scoring)
MDA_N_ESTIMATORS = 500
MDA_N_INNER_FOLDS = 3
SFI_INNER_TRAIN_PCT = 0.6            # chronological split for SFI
COMPOSITE_TOP_K_FRAC = 0.5           # select top 50% of features by composite score


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
            # d=0 means no differentiation, skip
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

    # find minimum d where p-value < significance
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

    logger.info(
        "FFD d* search: optimal d*=%.2f (p=%.4f). Sweep: %s",
        d_star,
        sweep_results.get(d_star, np.nan),
        {round(k, 2): round(v, 4) for k, v in sweep_results.items()},
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

        # estimate d* from training data only
        train_series = X_full[col].iloc[train_idx]
        d_star = find_optimal_d(train_series)
        ffd_info[col] = d_star

        # apply FFD to the FULL series (causal filter, no leakage)
        X_transformed[col] = apply_ffd(X_full[col], d_star)
        lookback = len(_compute_ffd_weights(d_star)) - 1
        logger.info("FFD: col='%s', d*=%.2f, lookback=%d obs.", col, d_star, lookback)

    # extract train/test from the transformed full series
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
        logger.info(
            "FFD: dropped %d train, %d test NaN rows from FFD lookback.",
            n_train_dropped, n_test_dropped,
        )

    logger.info("FFD transform complete: %s", ffd_info)
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

    logger.info("Scaling: RobustScaler fitted on %d training rows.", len(X_train))
    return X_train_scaled, X_test_scaled, scaler


# =====================================================================
# Feature Selection (AFML Chapter 8)
# =====================================================================
def compute_mda(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    t1_train: pd.Series,
) -> pd.Series:
    """Mean Decrease Accuracy (AFML Chapter 8.4).

    Uses internal purged K-fold CV on the training set to compute
    out-of-sample permutation importance.
    """
    n_folds = MDA_N_INNER_FOLDS
    T = len(X_train)
    fold_size = T // n_folds
    rng = np.random.RandomState(42)

    mda_scores = pd.DataFrame(index=X_train.columns)

    for fold_i in range(n_folds):
        # define inner test range
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

        # positional to iloc
        X_tr = X_train.iloc[inner_train_idx]
        y_tr = y_train.iloc[inner_train_idx]
        w_tr = w_train.iloc[inner_train_idx]
        X_te = X_train.iloc[inner_test_idx]
        y_te = y_train.iloc[inner_test_idx]

        clf = RandomForestClassifier(
            n_estimators=MDA_N_ESTIMATORS,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        # baseline F1
        baseline_f1 = f1_score(y_te, clf.predict(X_te), average="macro")

        # permutation importance per feature
        fold_mda = {}
        for col in X_train.columns:
            X_te_perm = X_te.copy()
            X_te_perm[col] = rng.permutation(X_te_perm[col].values)
            perm_f1 = f1_score(y_te, clf.predict(X_te_perm), average="macro")
            fold_mda[col] = baseline_f1 - perm_f1

        mda_scores[f"fold_{fold_i}"] = pd.Series(fold_mda)

    # average across folds
    avg_mda = mda_scores.mean(axis=1).rename("MDA")
    return avg_mda.sort_values(ascending=False)


def compute_sfi(
    X_train: pd.DataFrame, y_train: pd.Series, w_train: pd.Series
) -> pd.Series:
    """Single Feature Importance (AFML Chapter 8.4).

    For each feature individually, fit a logistic regression on a
    chronological inner split and measure F1 on the held-out portion.
    """
    split_point = int(len(X_train) * SFI_INNER_TRAIN_PCT)
    sfi_scores = {}

    X_inner_train = X_train.iloc[:split_point]
    y_inner_train = y_train.iloc[:split_point]
    w_inner_train = w_train.iloc[:split_point]
    X_inner_val = X_train.iloc[split_point:]
    y_inner_val = y_train.iloc[split_point:]

    for col in X_train.columns:
        X_tr_1d = X_inner_train[[col]]
        X_val_1d = X_inner_val[[col]]

        try:
            clf = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )
            clf.fit(X_tr_1d, y_inner_train, sample_weight=w_inner_train)
            preds = clf.predict(X_val_1d)
            sfi_scores[col] = f1_score(y_inner_val, preds, average="macro")
        except Exception as e:
            logger.warning("SFI failed for '%s': %s", col, e)
            sfi_scores[col] = 0.0

    result = pd.Series(sfi_scores, name="SFI")
    return result.sort_values(ascending=False)


def compute_composite_score(
    mda: pd.Series, sfi: pd.Series
) -> pd.DataFrame:
    """Combine MDA and SFI into a single composite score via percentile ranks.

    Percentile ranks normalise the two methods to [0, 1] regardless of
    their native scales (MDA can be negative, SFI is bounded in [0, 1]).
    Equal weighting (0.5/0.5) treats both signals as equally informative.

    Each feature is also assigned a *profile* that characterises the
    nature of its predictive contribution:

    - **consensus**: top half in both MDA and SFI. The feature is
      individually predictive AND contributes unique information to the
      ensemble. Strongest selection signal.
    - **substitution**: top half in SFI but not MDA. The feature
      predicts well alone, but permuting it in the RF barely hurts
      because correlated features compensate (AFML §8.3 substitution
      effect). Still selected, but its apparent importance may be
      shared with neighbours.
    - **joint**: top half in MDA but not SFI. The feature contributes
      through interactions with other features in the ensemble but is
      not predictive in isolation. Valuable for non-linear models.
    - **weak**: bottom half in both. Little evidence of predictive
      value under either criterion.

    Parameters
    ----------
    mda : pd.Series
        Mean Decrease Accuracy scores, indexed by feature name.
    sfi : pd.Series
        Single Feature Importance scores, indexed by feature name.

    Returns
    -------
    pd.DataFrame
        Columns: MDA, SFI, MDA_rank, SFI_rank, composite, profile.
        Sorted by composite descending.
    """
    rankings = pd.DataFrame({"MDA": mda, "SFI": sfi})

    # percentile ranks (0 = worst, 1 = best)
    rankings["MDA_rank"] = rankings["MDA"].rank(pct=True)
    rankings["SFI_rank"] = rankings["SFI"].rank(pct=True)

    # composite: equal-weighted average of percentile ranks
    rankings["composite"] = 0.5 * rankings["MDA_rank"] + 0.5 * rankings["SFI_rank"]

    # feature profile based on median split of each method's rank
    mda_high = rankings["MDA_rank"] >= 0.5
    sfi_high = rankings["SFI_rank"] >= 0.5

    conditions = [
        mda_high & sfi_high,       # consensus
        ~mda_high & sfi_high,      # substitution (high SFI, low MDA)
        mda_high & ~sfi_high,      # joint (high MDA, low SFI)
    ]
    choices = ["consensus", "substitution", "joint"]
    rankings["profile"] = np.select(conditions, choices, default="weak")

    return rankings.sort_values("composite", ascending=False)


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    t1_train: pd.Series,
) -> list[str]:
    """Select features via MDA + SFI composite scoring.

    Uses only MDA (permutation importance with purged inner CV) and SFI
    (single-feature logistic regression). MDI is excluded to avoid
    over-engineering from tree-based impurity bias (AFML §8.3).

    A feature is selected if its composite percentile-rank score places
    it in the top ``COMPOSITE_TOP_K_FRAC`` of all features (default 50%).
    A minimum floor of 5 features is enforced.

    Diagnostic profiles are logged for each feature:
    - consensus:    high MDA + high SFI (individually and jointly useful)
    - substitution: low MDA + high SFI  (predictive alone, redundant in ensemble)
    - joint:        high MDA + low SFI  (useful only through interactions)
    - weak:         low MDA + low SFI   (dropped)

    Returns
    -------
    list[str]
        Sorted list of selected feature names.
    """
    n_features = X_train.shape[1]
    top_k = max(int(n_features * COMPOSITE_TOP_K_FRAC), 5)

    print("[preprocessing] Computing MDA...")
    mda = compute_mda(X_train, y_train, w_train, t1_train)
    print("[preprocessing] Computing SFI...")
    sfi = compute_sfi(X_train, y_train, w_train)

    # composite scoring
    rankings = compute_composite_score(mda, sfi)

    # select top-K by composite score
    selected = sorted(rankings.head(top_k).index.tolist())

    # fallback: ensure at least 5 features
    if len(selected) < 5:
        logger.warning(
            "Only %d features above composite threshold. "
            "Taking top 5 by composite score.", len(selected),
        )
        selected = sorted(rankings.head(5).index.tolist())

    rankings["selected"] = rankings.index.isin(selected)

    # log full rankings with profiles
    logger.info("Feature selection rankings:\n%s", rankings.to_string())

    # profile summary
    profile_counts = rankings.loc[rankings["selected"], "profile"].value_counts()

    print(
        f"[preprocessing] Feature selection: {len(selected)}/{n_features} features kept "
        f"(top {top_k} by MDA+SFI composite)"
    )
    print(f"  Selected: {selected}")
    print(f"  Profiles: {profile_counts.to_dict()}")
    dropped = sorted(set(X_train.columns) - set(selected))
    if dropped:
        print(f"  Dropped:  {dropped}")

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
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Full preprocessing for one CPCV fold: FFD → scaling → selection.

    Returns DataFrames with ALL columns (pre-selection) so the pipeline
    can provide full-column DataFrames to AR Logistic. The selected
    feature list is returned separately for the pipeline to apply.

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

    # 4. Select features (but do NOT apply here, return the list)
    selected = select_features(X_train, y_train, w_train, t1_train)

    info = {
        "ffd": ffd_info,
        "scaler": scaler,
        "selected_features": selected,
    }

    return X_train, X_test, selected, info