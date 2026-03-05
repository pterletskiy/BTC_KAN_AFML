"""
econometrics.py — Econometric tests and transformations for the MFW pipeline.

Pure statistical / mathematical functions.  Stateless, no side effects.
This module does NOT perform any data cleaning, scaling, or feature selection.

Follows ``econometrics.md``:
  §2  Preserve datetime indices during fractional differencing.
  §4  Type hints on all functions.  No matplotlib / seaborn imports.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Fractional Differencing (AFML Chapter 5)
# ═══════════════════════════════════════════════════════════════════════════
def get_weights_ffd(d: float, thres: float = 1e-4) -> np.ndarray:
    """Generate Fixed-Width Fractional Differencing (FFD) weights.

    Based on *Advances in Financial Machine Learning* (de Prado), Ch. 5.

    Parameters
    ----------
    d : float
        Differencing order (0 < d ≤ 1).
    thres : float
        Minimum absolute weight to include.

    Returns
    -------
    np.ndarray
        Weight vector of shape ``(n, 1)``, oldest-first.
    """
    w: List[float] = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    thres: float = 1e-4,
) -> pd.Series:
    """Apply Fixed-Width Fractional Differencing to a Pandas Series.

    **Critical (§2):** The returned Series preserves the original
    ``DatetimeIndex`` exactly.  Dropping the time index is a fatal error.

    Parameters
    ----------
    series : pd.Series
        Input time series (must have a DatetimeIndex).
    d : float
        Differencing order (0 < d ≤ 1).
    thres : float
        Minimum absolute weight for the FFD expansion.

    Returns
    -------
    pd.Series
        Fractionally differenced series with the **same index** as *series*.
        Leading values that lack sufficient history are set to ``NaN``.
    """
    # Shortcut: standard differencing when d ≈ 1
    if round(d, 2) == 1.0:
        return series.diff()

    w = get_weights_ffd(d, thres)
    width = len(w) - 1

    vals = series.values
    res = np.full_like(vals, np.nan, dtype=float)

    for i in range(width, len(vals)):
        res[i] = np.dot(w.T, vals[i - width : i + 1])[0]

    return pd.Series(res, index=series.index, name=series.name)


def find_optimal_d(
    series: pd.Series,
    d_range: Optional[np.ndarray] = None,
    thres: float = 1e-4,
    alpha: float = 0.05,
) -> float:
    """Search for the minimum differencing order *d* that achieves stationarity.

    Iterates through *d_range* and returns the first *d* for which the
    ADF test p-value is below *alpha*.  Falls back to ``1.0`` (standard
    differencing) if no fractional *d* suffices.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    d_range : np.ndarray, optional
        Candidate *d* values to try (default ``0.1, 0.2, …, 0.9``).
    thres : float
        Weight threshold for FFD.
    alpha : float
        Significance level for the ADF test.

    Returns
    -------
    float
        Optimal differencing order.
    """
    if d_range is None:
        d_range = np.arange(0.1, 1.0, 0.1)

    for d in d_range:
        diff = frac_diff_ffd(series, d, thres).dropna()
        if len(diff) > 10:
            pval = adfuller(diff)[1]
            if pval < alpha:
                return round(d, 1)

    return 1.0  # fallback


# ═══════════════════════════════════════════════════════════════════════════
# Stationarity Testing (ADF + KPSS)
# ═══════════════════════════════════════════════════════════════════════════
def run_stationarity_tests(
    df: pd.DataFrame,
    features: List[str],
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, List[str]]:
    """Batch ADF + KPSS stationarity tests on selected features.

    **Strict logic:** if *either* test flags the series as non-stationary,
    the feature is marked as needing differencing.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the feature columns.
    features : list of str
        Column names to test.
    alpha : float
        Significance level.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - Results DataFrame with columns: ``Feature``, ``ADF p-value``,
          ``ADF Result``, ``KPSS p-value``, ``KPSS Result``,
          ``Needs Differencing?``.
        - List of feature names that require differencing.
    """
    results: List[list] = []
    features_to_difference: List[str] = []

    for col in features:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # ADF: H0 = Non-Stationary → reject (p < α) means stationary
        adf_pval = adfuller(series)[1]
        adf_result = "Stationary" if adf_pval < alpha else "Non-Stationary"

        # KPSS: H0 = Stationary → reject (p < α) means non-stationary
        kpss_pval = kpss(series, regression="c", nlags="auto")[1]
        kpss_result = "Stationary" if kpss_pval >= alpha else "Non-Stationary"

        needs_diff = (
            "Yes"
            if adf_result == "Non-Stationary" or kpss_result == "Non-Stationary"
            else "No"
        )
        if needs_diff == "Yes":
            features_to_difference.append(col)

        results.append(
            [col, adf_pval, adf_result, kpss_pval, kpss_result, needs_diff]
        )

    stat_df = pd.DataFrame(
        results,
        columns=[
            "Feature",
            "ADF p-value",
            "ADF Result",
            "KPSS p-value",
            "KPSS Result",
            "Needs Differencing?",
        ],
    )

    logger.info(
        "Stationarity tests: %d/%d features need differencing",
        len(features_to_difference),
        len(features),
    )
    return stat_df, features_to_difference


# ═══════════════════════════════════════════════════════════════════════════
# Distribution Profiling (Skewness, Kurtosis, Jarque-Bera)
# ═══════════════════════════════════════════════════════════════════════════
def run_distribution_profile(
    df: pd.DataFrame,
    features: List[str],
    skew_threshold: float = 1.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute skewness, kurtosis, and Jarque-Bera for each feature.

    Returns a results DataFrame and a list of features recommended for
    log transformation (highly skewed **and** contain no negative values).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the feature columns.
    features : list of str
        Column names to profile.
    skew_threshold : float
        Absolute skewness above which log transform is recommended.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - Profile DataFrame with columns: ``Feature``, ``Skewness``,
          ``Skew Level``, ``Kurtosis``, ``Fat Tails (Kurtosis > 3)``,
          ``JB Stat``, ``JB p-value``, ``Normally Distributed?``,
          ``Contains Negative Values?``, ``Log Transform Recommended``.
        - List of feature names safe for and recommended for log transform.
    """
    results: List[list] = []
    features_to_log: List[str] = []

    for col in features:
        series = df[col].dropna()
        skew_val = series.skew()
        kurt_val = series.kurtosis()
        has_negatives = bool((series < 0).any())

        jb_stat, jb_pval = jarque_bera(series)
        is_normal = "Yes" if jb_pval >= 0.05 else "No"

        # Skew classification
        if abs(skew_val) <= 0.5:
            skew_level = "Symmetric"
        elif abs(skew_val) <= 1.0:
            skew_level = "Moderately Skewed"
        else:
            skew_level = "Highly Skewed"

        fat_tails = "Yes (Leptokurtic)" if kurt_val > 3 else "No"

        # Log transform recommendation
        if abs(skew_val) > skew_threshold:
            if has_negatives:
                recommend_log = "Cannot (Has Negatives)"
            else:
                recommend_log = "Yes"
                features_to_log.append(col)
        else:
            recommend_log = "No"

        results.append([
            col,
            skew_val,
            skew_level,
            kurt_val,
            fat_tails,
            jb_stat,
            jb_pval,
            is_normal,
            "Yes" if has_negatives else "No",
            recommend_log,
        ])

    dist_df = pd.DataFrame(
        results,
        columns=[
            "Feature",
            "Skewness",
            "Skew Level",
            "Kurtosis",
            "Fat Tails (Kurtosis > 3)",
            "JB Stat",
            "JB p-value",
            "Normally Distributed?",
            "Contains Negative Values?",
            "Log Transform Recommended",
        ],
    )

    logger.info(
        "Distribution profile: %d features recommended for log transform",
        len(features_to_log),
    )
    return dist_df, features_to_log


# ═══════════════════════════════════════════════════════════════════════════
# Pandas Styler Helpers (for Notebook display)
# ═══════════════════════════════════════════════════════════════════════════
_GREEN = "background-color: #1e7e46"
_RED = "background-color: #a0382c"
_AMBER = "background-color: #b67c00"


def highlight_stat(val: str) -> str:
    """CSS for stationarity result cells."""
    if val in ("Stationary", "No"):
        return _GREEN
    if val in ("Non-Stationary", "Yes"):
        return _RED
    return ""


def highlight_skew(val: float) -> str:
    """CSS for skewness cells."""
    if abs(val) > 1:
        return _RED
    if abs(val) > 0.5:
        return _AMBER
    return _GREEN


def highlight_kurt(val: float) -> str:
    """CSS for kurtosis cells."""
    if val > 3:
        return _RED
    if val > 1:
        return _AMBER
    return _GREEN


def highlight_fat_tails(val: str) -> str:
    """CSS for fat-tails cells."""
    return _RED if "Yes" in str(val) else _GREEN


def highlight_normal(val: str) -> str:
    """CSS for normality cells."""
    return _GREEN if val == "Yes" else _RED


def highlight_negatives(val: str) -> str:
    """CSS for negative-values cells."""
    return _AMBER if val == "Yes" else _GREEN


def highlight_log_rec(val: str) -> str:
    """CSS for log-transform recommendation cells."""
    if val == "Yes":
        return _RED
    if "Cannot" in str(val):
        return _AMBER
    return _GREEN


def style_stationarity_df(stat_df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Apply notebook-friendly styling to a stationarity results DataFrame.

    Parameters
    ----------
    stat_df : pd.DataFrame
        Output of :func:`run_stationarity_tests`.

    Returns
    -------
    pd.io.formats.style.Styler
        Styled DataFrame ready for ``display()`` in a Jupyter Notebook.
    """
    return (
        stat_df.style
        .map(
            lambda x: highlight_stat(x) if isinstance(x, str) else "",
            subset=["ADF Result", "KPSS Result", "Needs Differencing?"],
        )
        .format({"ADF p-value": "{:.4f}", "KPSS p-value": "{:.4f}"})
    )


def style_distribution_df(dist_df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Apply notebook-friendly styling to a distribution profile DataFrame.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Output of :func:`run_distribution_profile`.

    Returns
    -------
    pd.io.formats.style.Styler
        Styled DataFrame ready for ``display()`` in a Jupyter Notebook.
    """
    return (
        dist_df.style
        .map(
            lambda x: highlight_skew(x) if isinstance(x, (int, float)) else "",
            subset=["Skewness"],
        )
        .map(
            lambda x: highlight_kurt(x) if isinstance(x, (int, float)) else "",
            subset=["Kurtosis"],
        )
        .map(highlight_fat_tails, subset=["Fat Tails (Kurtosis > 3)"])
        .map(highlight_normal, subset=["Normally Distributed?"])
        .map(highlight_negatives, subset=["Contains Negative Values?"])
        .map(highlight_log_rec, subset=["Log Transform Recommended"])
        .format({
            "Skewness": "{:.4f}",
            "Kurtosis": "{:.4f}",
            "JB Stat": "{:.2f}",
            "JB p-value": "{:.4e}",
        })
    )
