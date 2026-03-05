"""
preproc.py — Data cleaning, normalization, and feature selection pipeline.

This module orchestrates the full preprocessing flow and enforces the
**Golden Rule** from ``econometrics.md`` §1:

    Scalers are ONLY fitted on training data.
    Validation / test sets are ONLY transformed.

Depends on:
  - ``features.py``      (create_ta_features, create_onchain_features)
  - ``econometrics.py``   (frac_diff_ffd, find_optimal_d)

Does NOT import matplotlib / seaborn (§4).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import RobustScaler

from src.econometrics import find_optimal_d, frac_diff_ffd
from src.features import create_onchain_features, create_ta_features

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Cleaning — Noisy / Meaningless Feature Identification
# ═══════════════════════════════════════════════════════════════════════════
def get_low_variance_numerical(
    df: pd.DataFrame,
    target: str = "Price_Direction",
    threshold: float = 0.0099,
) -> List[str]:
    """Identify numerical features with near-zero variance.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    target : str
        Target column name to exclude from the check.
    threshold : float
        Variance threshold (default ``0.99 * (1 - 0.99)``).

    Returns
    -------
    list of str
        Column names that fall below the variance threshold.
    """
    num_df = df.select_dtypes(include=["number"]).drop(
        columns=[target], errors="ignore"
    )
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(num_df)

    to_drop = [
        col for col in num_df.columns if col not in num_df.columns[sel.get_support()]
    ]
    if to_drop:
        logger.info("Low-variance numerical features (%d): %s", len(to_drop), to_drop)
    return to_drop


def get_low_variance_categorical(
    df: pd.DataFrame,
    threshold: float = 0.99,
) -> List[str]:
    """Identify categorical features where one value dominates.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    threshold : float
        If the most frequent value accounts for ≥ *threshold* of rows,
        the feature is flagged.

    Returns
    -------
    list of str
        Column names to drop.
    """
    to_drop: List[str] = []
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in cat_cols:
        top_ratio = df[col].value_counts(normalize=True).iloc[0]
        if top_ratio >= threshold:
            to_drop.append(col)
            logger.info("  Low-variance categorical: %s (%.1f%%)", col, top_ratio * 100)

    return to_drop


def get_noisy_metadata_columns(df: pd.DataFrame) -> List[str]:
    """Identify administrative / metadata columns that are not market features.

    Catches:
      - ``*-status-time`` columns
      - ``*CompletionTime*`` columns
      - ``PriceUSD`` / ``volume_reported_spot_usd_1d`` (duplicates of yfinance data)

    Parameters
    ----------
    df : pd.DataFrame
        Training data.

    Returns
    -------
    list of str
        Column names to drop.
    """
    to_drop: List[str] = []

    to_drop.extend(c for c in df.columns if "-status-time" in c)
    to_drop.extend(c for c in df.columns if "CompletionTime" in c)
    to_drop.extend(
        c for c in df.columns if "PriceUSD" in c or "volume_reported_spot_usd_1d" in c
    )

    if to_drop:
        logger.info("Noisy metadata columns (%d): %s", len(to_drop), to_drop[:10])
    return to_drop


def get_high_missing_features(
    df: pd.DataFrame,
    threshold: float = 30.0,
) -> List[str]:
    """Identify features exceeding a missing-value percentage threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    threshold : float
        Maximum allowed percentage of NaN values.

    Returns
    -------
    list of str
        Column names to drop.
    """
    pct = df.isnull().mean() * 100
    to_drop = pct[pct > threshold].index.tolist()

    if to_drop:
        logger.info(
            "High-missing features (>%.0f%%): %d columns", threshold, len(to_drop)
        )
    return to_drop


def identify_noisy_features(
    df: pd.DataFrame,
    target: str = "Price_Direction",
    missing_threshold: float = 30.0,
    variance_threshold: float = 0.0099,
    categorical_threshold: float = 0.99,
) -> List[str]:
    """Orchestrate all noisy-feature detection steps.

    Combines:
      1. Low-variance numerical
      2. Low-variance categorical
      3. Noisy metadata columns
      4. High missing-value percentage

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    target : str
        Target column name.
    missing_threshold, variance_threshold, categorical_threshold
        Thresholds forwarded to the individual detectors.

    Returns
    -------
    list of str
        Deduplicated master list of column names to drop.
    """
    noisy: List[str] = []
    noisy.extend(get_low_variance_numerical(df, target, variance_threshold))
    noisy.extend(get_low_variance_categorical(df, categorical_threshold))
    noisy.extend(get_noisy_metadata_columns(df))
    noisy.extend(get_high_missing_features(df, missing_threshold))

    noisy = sorted(set(noisy))
    logger.info("Total noisy features to remove: %d", len(noisy))
    return noisy


# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature Normalization
# ═══════════════════════════════════════════════════════════════════════════
def apply_log_transform(
    df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """Apply ``np.log1p`` (log(1 + x)) to the specified columns.

    Safe for columns containing zeros.  Columns must **not** contain
    negative values (use :func:`econometrics.run_distribution_profile`
    to check first).

    Parameters
    ----------
    df : pd.DataFrame
        Training data (modified in-place on a copy).
    features : list of str
        Column names to transform.

    Returns
    -------
    pd.DataFrame
        DataFrame with transformed columns.
    """
    df = df.copy()
    cols = [c for c in features if c in df.columns]
    if cols:
        skew_before = df[cols].skew().mean()
        df[cols] = np.log1p(df[cols])
        skew_after = df[cols].skew().mean()
        logger.info(
            "Log transform on %d features — avg skew %.4f → %.4f",
            len(cols), skew_before, skew_after,
        )
    return df


def apply_fractional_differencing(
    df: pd.DataFrame,
    features: List[str],
    thres: float = 1e-4,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Find optimal *d* and apply FFD to non-stationary features.

    Uses :func:`econometrics.find_optimal_d` per feature, then applies
    :func:`econometrics.frac_diff_ffd`.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (modified in-place on a copy).
    features : list of str
        Non-stationary feature column names.
    thres : float
        Weight threshold for FFD.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, float]]
        - DataFrame with differenced features.
        - Mapping ``{feature_name: optimal_d}`` — **save this** for
          ``preprocess_evaluation_set``.
    """
    df = df.copy()
    optimal_d: Dict[str, float] = {}

    logger.info("Applying FFD to %d features …", len(features))
    for col in features:
        series = df[col].dropna()
        d = find_optimal_d(series, thres=thres)
        optimal_d[col] = d
        df[col] = frac_diff_ffd(df[col], d, thres)
        logger.info("  %s → d = %.1f", col, d)

    return df, optimal_d


def fit_robust_scaler(
    df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, RobustScaler]:
    """Fit a ``RobustScaler`` on training data and transform it.

    **Golden Rule (§1):** The scaler is fitted **only** here, on the
    training partition.  Validation / test sets must use the returned
    scaler's ``.transform()`` method.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    features : list of str
        Columns to scale.

    Returns
    -------
    tuple[pd.DataFrame, RobustScaler]
        - Scaled training DataFrame.
        - The fitted scaler — **save this** for
          ``preprocess_evaluation_set``.
    """
    df = df.copy()
    cols = [c for c in features if c in df.columns]
    scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    logger.info("RobustScaler fitted on %d features", len(cols))
    return df, scaler


# ═══════════════════════════════════════════════════════════════════════════
# 3. Feature Selection
# ═══════════════════════════════════════════════════════════════════════════
def remove_highly_correlated_features(
    X: pd.DataFrame,
    features: List[str],
    threshold: float = 0.85,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop one of each pair of highly Spearman-correlated features.

    For every pair above *threshold*, the feature with the **higher**
    average absolute correlation with all others is removed (it carries
    more redundant information).

    **§3 Reminder:** Call this on **training data only**.

    Parameters
    ----------
    X : pd.DataFrame
        Training data.
    features : list of str
        Columns to evaluate.
    threshold : float
        Correlation threshold.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - Reduced DataFrame.
        - Sorted list of removed column names.
    """
    X = X.copy()
    corr = X[features].corr(method="spearman").abs()
    to_remove: set = set()

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] > threshold:
                f1, f2 = corr.columns[i], corr.columns[j]
                avg1 = corr[f1].drop([f1, f2]).mean()
                avg2 = corr[f2].drop([f1, f2]).mean()
                to_remove.add(f1 if avg1 > avg2 else f2)

    removed = sorted(to_remove)
    X_reduced = X.drop(columns=removed)

    logger.info(
        "Correlation filter (>%.2f): removed %d features, %d remaining",
        threshold, len(removed), X_reduced.shape[1],
    )
    return X_reduced, removed


def rank_feature_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 30,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Rank features by combined Mutual Information + Random Forest importance.

    **§3 Reminder:** Call this on **training data only**.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix (training partition).
    y_train : pd.Series
        Target vector (training partition).
    top_n : int
        Number of top features to return.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - Importance DataFrame sorted by ``Combined_Score`` (descending),
          with columns: ``Mutual Information``, ``Random Forest Importance``,
          ``MI_Scaled``, ``RF_Scaled``, ``Combined_Score``.
        - List of the top *top_n* feature names.
    """
    # 1. Mutual Information
    mi_scores = mutual_info_classif(X_train, y_train, random_state=random_seed)
    mi_series = pd.Series(mi_scores, index=X_train.columns)

    # 2. Random Forest (shallow to avoid overfitting)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=random_seed, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_series = pd.Series(rf.feature_importances_, index=X_train.columns)

    # 3. Combine
    imp_df = pd.DataFrame({
        "Mutual Information": mi_series,
        "Random Forest Importance": rf_series,
    }).fillna(0)

    mi_max = imp_df["Mutual Information"].max()
    rf_max = imp_df["Random Forest Importance"].max()

    imp_df["MI_Scaled"] = imp_df["Mutual Information"] / mi_max if mi_max else 0
    imp_df["RF_Scaled"] = imp_df["Random Forest Importance"] / rf_max if rf_max else 0
    imp_df["Combined_Score"] = (imp_df["MI_Scaled"] + imp_df["RF_Scaled"]) / 2
    imp_df = imp_df.sort_values("Combined_Score", ascending=False)

    best = imp_df.head(top_n).index.tolist()
    logger.info("Feature importance: top %d selected from %d", top_n, len(X_train.columns))

    return imp_df, best


# ═══════════════════════════════════════════════════════════════════════════
# 4. Master Evaluation Pipeline (Val / Test)
# ═══════════════════════════════════════════════════════════════════════════
def preprocess_evaluation_set(
    df_target: pd.DataFrame,
    raw_full_data: pd.DataFrame,
    noisy_features: List[str],
    log_features: List[str],
    d_values_dict: Dict[str, float],
    fitted_scaler: RobustScaler,
    scale_features: List[str],
    final_features: List[str],
    buffer_days: int = 600,
) -> pd.DataFrame:
    """Apply the full preprocessing pipeline to a validation or test set.

    Uses a historical buffer so that rolling-window warm-up rows are
    consumed from the buffer — **not** from the target data.

    **Golden Rule (§1):**
      - Scaler is ``transform``-ed only (never ``fit``).
      - Optimal *d* values come from training.
      - Feature importance ranking comes from training.

    Parameters
    ----------
    df_target : pd.DataFrame
        The val/test partition (date-sliced).
    raw_full_data : pd.DataFrame
        The **complete** raw merged dataset (output of
        ``data_loader.load_dataset``), used to create the lookback buffer.
    noisy_features : list of str
        Columns to drop (from ``identify_noisy_features``).
    log_features : list of str
        Columns to log-transform (from ``run_distribution_profile``).
    d_values_dict : dict
        ``{feature: optimal_d}`` from ``apply_fractional_differencing``.
    fitted_scaler : RobustScaler
        Scaler fitted on training data (from ``fit_robust_scaler``).
    scale_features : list of str
        Columns to scale.
    final_features : list of str
        Final column selection (features + target).
    buffer_days : int
        Number of calendar days to prepend as a warm-up buffer.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed evaluation set, indexed by the original
        target dates.
    """
    logger.info(
        "Preprocessing evaluation set: %d rows, buffer=%d days",
        len(df_target), buffer_days,
    )

    # 1. Create historical buffer
    start_date = df_target.index[0]
    buffer_start = start_date - pd.Timedelta(days=buffer_days)
    df_work = raw_full_data.loc[buffer_start : df_target.index[-1]].copy()

    # 2. Drop noisy / metadata features
    cols_to_drop = [c for c in noisy_features if c in df_work.columns]
    df_work = df_work.drop(columns=cols_to_drop, errors="ignore")

    # 3. Feature engineering (rolling windows consume the buffer)
    df_work, _ = create_ta_features(df_work)
    df_work, _ = create_onchain_features(df_work)

    # 4. Log transformation
    cols_to_log = [c for c in log_features if c in df_work.columns]
    if cols_to_log:
        df_work[cols_to_log] = np.log1p(df_work[cols_to_log])

    # 5. Fractional differencing (using optimal d from training)
    for col, d_val in d_values_dict.items():
        if col in df_work.columns:
            df_work[col] = frac_diff_ffd(df_work[col], d_val)

    # Drop NaNs from lookback (only buffer rows should be affected)
    df_work = df_work.dropna()

    # 6. SLICE — isolate the exact target dates (drop buffer)
    df_clean = df_work.loc[df_target.index].copy()

    # 7. Robust scaling (transform ONLY — never fit)
    cols_to_scale = [c for c in scale_features if c in df_clean.columns]
    if cols_to_scale:
        df_clean[cols_to_scale] = fitted_scaler.transform(df_clean[cols_to_scale])

    # 8. Feature selection — keep only the final powerful features
    cols_to_keep = [c for c in final_features if c in df_clean.columns]
    df_clean = df_clean[cols_to_keep]

    logger.info("Evaluation set ready: %d rows × %d cols", *df_clean.shape)
    return df_clean
