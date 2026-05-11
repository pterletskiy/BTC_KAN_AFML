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

    assert len(common_idx) > 0, (
        "No overlapping indices between features, labels, and weights.")
    assert not common_idx.duplicated().any(), "Duplicate dates in common index."

    common_idx = common_idx.sort_values()
    assert common_idx.is_monotonic_increasing, "Index not sorted."

    # Slice each upstream object down to the shared index in one place.
    X = features.loc[common_idx]
    y = bins.loc[common_idx, "bin"].astype(int)
    w = weights.loc[common_idx]
    t1 = bins.loc[common_idx, "t1"]

    # Cross-check that the slicing produced consistent shapes and no all-NaN columns.
    assert X.shape[0] == y.shape[0] == w.shape[0] == t1.shape[0], (
        f"Shape mismatch after alignment: X={X.shape[0]}, y={y.shape[0]}, "
        f"w={w.shape[0]}, t1={t1.shape[0]}.")

    all_nan_cols = X.columns[X.isnull().all(axis=0)].tolist()
    assert len(all_nan_cols) == 0, (
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
    assert X.shape[0] == y.shape[0] == w.shape[0] == t1.shape[0], (
        f"Shape mismatch: X={X.shape[0]}, y={y.shape[0]}, "
        f"w={w.shape[0]}, t1={t1.shape[0]}.")

    assert X.index.equals(y.index), "X and y indices differ."
    assert X.index.equals(w.index), "X and w indices differ."
    assert X.index.equals(t1.index), "X and t1 indices differ."
    assert X.index.is_monotonic_increasing, "Index not sorted ascending."
    assert not X.index.duplicated().any(), "Duplicate dates in index."

    assert set(y.unique()).issubset({-1, 0, 1}), (
        f"Labels contain unexpected values: {set(y.unique()) - {-1, 0, 1}}")

    assert (w > 0).all(), (
        f"Non-positive sample weights found: {w[w <= 0].shape[0]} value(s).")

    # t1 is required for CPCV purging; missing values would silently corrupt folds
    assert t1.notna().all(), (
        f"Missing t1 values ({t1.isna().sum()}) would break CPCV purging.")

    all_nan_cols = X.columns[X.isnull().all(axis=0)].tolist()
    assert len(all_nan_cols) == 0, (
        f"Feature column(s) entirely NaN: {all_nan_cols}")

    logger.info("Alignment validation passed (%d samples, %d features).", *X.shape)
    return True