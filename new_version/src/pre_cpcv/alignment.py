"""
6) Alignment
==================
Intersect the full daily feature matrix with the CUSUM-filtered labels and
weights, producing the ``(X, y, w, t1)`` tuple consumed by the CPCV loop.
``t1`` is preserved because purging requires it.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


# Build the (X, y, w, t1) tuple consumed by the CPCV loop from upstream-stage outputs.
def align_for_cv(features: pd.DataFrame, bins: pd.DataFrame, weights: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Align features, labels, and weights on their common date index."""
    # The CPCV loop expects a single index shared by all four objects; find it here.
    common_idx = features.index.intersection(bins.index).intersection(weights.index)

    if len(common_idx) == 0:
        raise ValueError(
            "No overlapping indices between features, labels, and weights.")
    if common_idx.duplicated().any():
        raise ValueError("Duplicate dates in common index.")

    common_idx = common_idx.sort_values()
    if not common_idx.is_monotonic_increasing:
        raise ValueError("Index not sorted.")

    # Slice each upstream object down to the shared index in one place.
    X = features.loc[common_idx]
    y = bins.loc[common_idx, "bin"].astype(int)
    w = weights.loc[common_idx]
    t1 = bins.loc[common_idx, "t1"]

    # Cross-check that the slicing produced consistent shapes and no all-NaN columns.
    if not (X.shape[0] == y.shape[0] == w.shape[0] == t1.shape[0]):
        raise ValueError(
            f"Shape mismatch after alignment: X={X.shape[0]}, y={y.shape[0]}, "
            f"w={w.shape[0]}, t1={t1.shape[0]}.")

    all_nan_cols = X.columns[X.isnull().all(axis=0)].tolist()
    if len(all_nan_cols) > 0:
        raise ValueError(
            f"Feature column(s) entirely NaN after alignment: {all_nan_cols}")

    nan_counts = X.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        logger.warning(
            "Remaining NaNs in %d feature column(s) after alignment:\n%s",
            len(nan_cols), nan_cols.to_string())

    print(
        f"[alignment] {X.shape[0]} samples, {X.shape[1]} features | "
        f"date range: {X.index[0].date()} → {X.index[-1].date()} | "
        f"classes: {y.value_counts().to_dict()}")

    return X, y, w, t1


# Defensive recheck called inside the CPCV loop before any model sees the data.
def validate_alignment(X: pd.DataFrame, y: pd.Series, w: pd.Series, t1: pd.Series) -> bool:
    """Re-run all alignment invariants inside the CV loop. Raises on failure.

    Cheap belt-and-suspenders check: if a preprocessing step elsewhere has
    silently broken alignment, this catches it before the model sees the data.
    """
    if not (X.shape[0] == y.shape[0] == w.shape[0] == t1.shape[0]):
        raise ValueError(
            f"Shape mismatch: X={X.shape[0]}, y={y.shape[0]}, "
            f"w={w.shape[0]}, t1={t1.shape[0]}.")

    if not X.index.equals(y.index):
        raise ValueError("X and y indices differ.")
    if not X.index.equals(w.index):
        raise ValueError("X and w indices differ.")
    if not X.index.equals(t1.index):
        raise ValueError("X and t1 indices differ.")
    if not X.index.is_monotonic_increasing:
        raise ValueError("Index not sorted ascending.")
    if X.index.duplicated().any():
        raise ValueError("Duplicate dates in index.")

    if not set(y.unique()).issubset({-1, 0, 1}):
        raise ValueError(
            f"Labels contain unexpected values: {set(y.unique()) - {-1, 0, 1}}")

    if not (w > 0).all():
        raise ValueError(
            f"Non-positive sample weights found: {w[w <= 0].shape[0]} value(s).")

    # t1 is required for CPCV purging; missing values would silently corrupt folds
    if not t1.notna().all():
        raise ValueError(
            f"Missing t1 values ({t1.isna().sum()}) would break CPCV purging.")

    all_nan_cols = X.columns[X.isnull().all(axis=0)].tolist()
    if len(all_nan_cols) > 0:
        raise ValueError(
            f"Feature column(s) entirely NaN: {all_nan_cols}")

    logger.info("Alignment validation passed (%d samples, %d features).", *X.shape)
    return True