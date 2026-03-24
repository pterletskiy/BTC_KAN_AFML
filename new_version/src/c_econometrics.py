# 3_econometrics.py — Econometric transformations for the MFW pipeline.
#
# This module acts as both a continuous pre-processor (structural logs, SADF,
# SMT bubble detection) and a mathematical toolkit for the downstream
# Cross-Validation loop (FFD).
#
# Strictly adheres to Marcos López de Prado's methodologies (AFML).
# Called ONCE before the CV loop; output saved to data/interim/.
# FFD optimization is intentionally deferred to the inner CV loop.

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Smart Structural Log Transformations
# ═══════════════════════════════════════════════════════════════════════════
def apply_structural_log(
    df: pd.DataFrame, feature_metadata: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Apply np.log1p to strictly non-negative ``raw_level`` features.

    After transformation, updates the tag from ``"raw_level"`` to
    ``"log_level"`` in ``feature_metadata`` so downstream processing
    knows the column has been log-transformed.

    Explicitly skips columns tagged ``"bounded_oscillator"``,
    ``"zero_centered"``, ``"ratio"``, or ``"cyclical"`` — these must
    never be log-transformed.

    Args:
        df: Dataset containing the engineered features.
        feature_metadata: Dictionary mapping column names to MLDP
            statistical type tags.

    Returns:
        Tuple of (transformed DataFrame, updated feature_metadata).

    Notes:
        AFML Ch. 2 — log transforms apply only to strictly positive,
        scale-skewed variables (prices, volumes, market cap).
        Uses ``np.log1p`` instead of ``np.log`` to handle near-zero
        values safely.
    """
    df = df.copy()
    transformed_count = 0

    for col in list(feature_metadata.keys()):
        tag = feature_metadata[col]

        if tag != "raw_level":
            continue

        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        if (series < 0).any():
            logger.debug(
                "Skipping log transform for '%s': contains negative values.", col
            )
            continue

        # Strictly non-negative raw level — apply log1p and upgrade tag
        df[col] = np.log1p(df[col])
        feature_metadata[col] = "log_level"
        transformed_count += 1

    logger.info("[LOG] Applied log1p to %d raw_level features.", transformed_count)
    return df, feature_metadata


# ═══════════════════════════════════════════════════════════════════════════
# 2. Supremum ADF (SADF) Bubble Detection
# ═══════════════════════════════════════════════════════════════════════════
def compute_sadf_signal(
    series: pd.Series, min_window: int = 100
) -> pd.Series:
    """Compute the Supremum ADF (SADF) bubble detection signal.

    Optimized O(n²) implementation using running normal equation accumulators 
    replacing O(n³) expanding window regressions explicitly.

    Args:
        series: The asset price or level series (e.g., log-Close).
        min_window: Minimum number of observations for each inner
            ADF window (default 100).

    Returns:
        A pd.Series of SADF t-statistics, explicitly shifted by 1
        to prevent look-ahead bias. Named ``"BTC_SADF_Bubble_Signal"``.
    """
    n = len(series)
    sadf_vals = np.full(n, np.nan)
    vals = series.values
    dy = np.diff(vals)
    y_lag = vals[:-1]

    for t in range(min_window, n):
        max_stat = -np.inf
        
        XtX = np.zeros((2, 2))
        Xty = np.zeros(2)
        dy_sq_sum = 0.0
        
        # Walk t0 backward from t-1 down to 0
        for t0 in range(t - 1, -1, -1):
            x_new = np.array([1.0, y_lag[t0]])
            dy_new = dy[t0]
            
            # O(1) Accumulators natively building matrices
            XtX += np.outer(x_new, x_new)
            Xty += x_new * dy_new
            dy_sq_sum += dy_new * dy_new
            
            window_len = t - t0
            if window_len < min_window - 1:
                continue
                
            try:
                # O(1) Mathematical solving avoiding matrix inversions algebraically
                XtX_inv = np.linalg.pinv(XtX)
                beta_hat = XtX_inv @ Xty
                
                # O(1) Residual Variance boundaries 
                ssr = max(0.0, dy_sq_sum - np.dot(beta_hat, Xty))
                sigma2 = ssr / max(window_len - 2, 1)
                se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])
                
                if se_beta >= 1e-15:
                    stat = beta_hat[1] / se_beta
                    if stat > max_stat:
                        max_stat = stat
            except (np.linalg.LinAlgError, FloatingPointError):
                continue
                
        if max_stat > -np.inf:
            sadf_vals[t] = max_stat

    # CRITICAL: Shift by 1 to prevent look-ahead bias (AFML information barrier)
    result = pd.Series(
        sadf_vals, index=series.index, name="BTC_SADF_Bubble_Signal"
    ).shift(1)

    logger.info(
        "SADF: Computed Supremum ADF signal (min_window=%d, lagged=True)", min_window
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. Sub/Super-Martingale Test (SMT) Bubble Detection
# ═══════════════════════════════════════════════════════════════════════════
def compute_smt_signal(
    series: pd.Series,
    min_window: int = 100,
    phi: float = 0.5,
    trend_type: str = "poly1",
) -> pd.Series:
    """Compute the Sub/Super-Martingale Test (SMT) bubble detection signal.

    Optimized O(n²) implementation utilizing pre-calculated inverse trend 
    matrices alongside mathematical state accumulators eliminating inverse bottlenecks completely.

    Args:
        series: The asset price or level series (e.g., log-Close).
        min_window: Minimum observations per inner OLS window (default 100).
        phi: Window length penalty exponent. Lower values favour short-run
            bubbles; higher values penalise short windows (default 0.5).
        trend_type: Type of time trend regressor ('poly1', 'poly2', 'exp').

    Returns:
        A pd.Series of SMT statistics, explicitly shifted by 1.
    """
    n = len(series)
    dy = series.diff().values
    smt_vals = np.full(n, np.nan)

    # Pre-calculate explicit boundaries eliminating inverse bounds 
    XtX_inv_cache = np.zeros((n + 1, 2, 2))
    
    for w_len in range(3, n + 1):
        tau = np.arange(1, w_len + 1, dtype=float)
        if trend_type == "poly1":
            trend = tau
        elif trend_type == "poly2":
            trend = tau ** 2
        elif trend_type == "exp":
            trend = np.exp(0.01 * tau)
        else:
            raise ValueError(f"Unknown trend_type: '{trend_type}'")
            
        X = np.column_stack([np.ones(w_len), trend])
        try:
            XtX_inv_cache[w_len] = np.linalg.pinv(X.T @ X)
        except np.linalg.LinAlgError:
            pass

    for t in range(min_window, n):
        max_stat = -np.inf
        
        # Accumulators natively tracking shifting sequential trends mathematically completely efficiently
        S0, S1, S2, S_exp = 0.0, 0.0, 0.0, 0.0
        dy_sq_sum = 0.0
        
        # Recursively walk t0 sequences bridging O(n^2) bounds
        for t0 in range(t - 1, -1, -1):
            dy_new = dy[t0]
            win_len = t - t0
            
            dy_sq_sum += dy_new * dy_new
            if trend_type == "poly1":
                S1 = dy_new + S1 + S0
                S0 = dy_new + S0
                Xty = np.array([S0, S1])
            elif trend_type == "poly2":
                S2 = dy_new + S2 + 2 * S1 + S0
                S1 = dy_new + S1 + S0
                S0 = dy_new + S0
                Xty = np.array([S0, S2])
            elif trend_type == "exp":
                S_exp = np.exp(0.01) * (dy_new + S_exp)
                S0 = dy_new + S0
                Xty = np.array([S0, S_exp])
                
            if win_len < min_window - 1: 
                continue
                
            XtX_inv = XtX_inv_cache[win_len]
            beta_0 = XtX_inv[0, 0] * Xty[0] + XtX_inv[0, 1] * Xty[1]
            beta_1 = XtX_inv[1, 0] * Xty[0] + XtX_inv[1, 1] * Xty[1]
            
            ssr = max(0.0, dy_sq_sum - (beta_0 * Xty[0] + beta_1 * Xty[1]))
            sigma2 = ssr / max(win_len - 2, 1)
            se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])
            
            if se_beta >= 1e-15:
                stat = abs(beta_1) / (se_beta * (win_len ** phi + 1e-8))
                if stat > max_stat:
                    max_stat = stat

        if max_stat > -np.inf:
            smt_vals[t] = max_stat

    col_name = f"BTC_SMT_{trend_type}_phi{phi}"
    
    # CRITICAL: Shift by 1 to prevent look-ahead bias (AFML information barrier)
    result = pd.Series(smt_vals, index=series.index, name=col_name).shift(1)

    logger.info(
        "SMT: Computed %s signal (min_window=%d, phi=%.1f, lagged=True)",
        trend_type, min_window, phi,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. Fractional Differencing (FFD) — CV Toolkit
# ═══════════════════════════════════════════════════════════════════════════
def get_weights_ffd(d: float, thres: float = 1e-5) -> np.ndarray:
    """Generate Fixed-Width Fractional Differencing (FFD) weights.

    Args:
        d: Differencing order (0 < d ≤ 1).
        thres: Minimum absolute weight to include (default 1e-5,
            the threshold τ from AFML Ch. 5).

    Returns:
        Weight vector of shape (n, 1), oldest-first.

    Notes:
        AFML Ch. 5 — Weight sequence:
        ω_0 = 1, ω_k = −ω_{k−1} · (d − k + 1) / k.
        Truncated when |ω_k| < thres.
    """
    if d == 0.0:
        return np.array([1.0]).reshape(-1, 1)
        
    assert 0 < d <= 1.0, "d must be in (0, 1]"
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
    series: pd.Series, d: float, thres: float = 1e-5
) -> pd.Series:
    """Apply Fixed-Width Window Fractional Differencing to a pd.Series.

    Args:
        series: Input time series (must have a DatetimeIndex).
        d: Differencing order (0 < d ≤ 1).
        thres: Minimum absolute weight for the FFD expansion.

    Returns:
        Fractionally differenced series with the same index as
        ``series``. Leading values that lack sufficient history
        are set to NaN.

    Notes:
        AFML Ch. 5 — Fixed-width window avoids the expanding memory
        problem of the standard fracdiff, making the transform
        applicable to production systems.
    """
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
    series: pd.Series, pval_threshold: float = 0.05
) -> Tuple[float, pd.Series, float]:
    """Find the minimum FFD order d* that achieves ADF stationarity.

    Grid-searches d over [0.00, 0.05, ..., 1.00]. Selects the absolute
    minimum d* where the ADF p-value falls below ``pval_threshold``.
    Performs a Pearson correlation memory check against the original
    level series.

    Args:
        series: The input time series to fractionally difference.
        pval_threshold: Maximum ADF p-value to consider the series
            stationary (default 0.05).

    Returns:
        Tuple of (d_star, ffd_series, corr):
            - d_star: The optimal fractional differencing order.
            - ffd_series: The fractionally differenced pd.Series at d*.
            - corr: Pearson correlation between the original and FFD
              series. Defaults to 1.0 if d*=0, or np.nan if
              correlation cannot be computed.

    Notes:
        AFML Ch. 5 — This function exists ONLY to be called from inside
        the CV loop (from 5_cv.py or 6_preproc.py). Calling it inside
        apply_continuous_econometrics constitutes a lookahead bias
        violation — the optimal d would be fit on the full dataset
        including future test folds.

        The memory check targets |ρ| > 0.90 between the original and
        FFD series (AFML §2.2 correlation check).
    """
    d_range = np.arange(0.00, 1.05, 0.05)
    optimal_d = 1.0
    optimal_series = series.diff()  # fallback: standard differencing

    # Check d=0: if already stationary, no transform needed
    valid = series.dropna()
    if len(valid) > 10:
        try:
            base_pval = adfuller(valid, autolag="AIC")[1]
            if base_pval < pval_threshold:
                logger.info("[FFD] Series already stationary at d=0.00.")
                return 0.0, series.copy(), 1.0
        except Exception:
            pass

    # Grid search d > 0
    for d in d_range[1:]:
        d_rounded = round(d, 2)
        ffd_series = frac_diff_ffd(series, d_rounded)
        ffd_clean = ffd_series.dropna()

        if len(ffd_clean) < 10:
            continue

        try:
            adf_pval = adfuller(ffd_clean, autolag="AIC")[1]
        except Exception:
            continue

        if adf_pval < pval_threshold:
            optimal_d = d_rounded
            optimal_series = ffd_series
            break

    if optimal_d >= 1.0:
        logger.warning(
            "[FFD] No fractional d < 1.0 achieved stationarity "
            "(pval_threshold=%.2f). Falling back to d=1.0 (standard diff).",
            pval_threshold,
        )

    # Memory correlation check
    corr = np.nan  # default if cannot be computed
    if optimal_d == 0.0:
        corr = 1.0
    elif optimal_d > 0.0:
        aligned = pd.concat([series, optimal_series], axis=1).dropna()
        if len(aligned) > 10:
            corr, _ = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
            if abs(corr) < 0.90:
                logger.warning(
                    "[FFD] Memory loss warning: correlation with original = "
                    "%.4f at d=%.2f. Consider reviewing feature engineering.",
                    corr, optimal_d,
                )
            else:
                logger.info(
                    "[FFD] Memory check passed: correlation = %.4f at d*=%.2f.",
                    corr, optimal_d,
                )

    return optimal_d, optimal_series, corr


# ═══════════════════════════════════════════════════════════════════════════
# 5. Pre-CV Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def apply_continuous_econometrics(
    df: pd.DataFrame, feature_metadata: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Execute all continuous, pre-CV econometric transformations.

    This is the single public entry point called by MFW_Pipeline.ipynb
    before the cross-validation loop. Its output is saved to
    ``data/interim/``.

    Steps:
        1. Structural log transforms on ``raw_level`` features.
        2. SADF bubble detection signal on Close.
        3. Two SMT signals (poly1/phi=0.1 and poly2/phi=0.9).

    Args:
        df: DataFrame with DatetimeIndex containing all raw and
            engineered features from 2_features.py. Must contain
            a ``"Close"`` column.
        feature_metadata: Dictionary mapping column names to MLDP
            statistical type tags.

    Returns:
        Tuple of (transformed DataFrame, updated feature_metadata).

    Notes:
        AFML Ch. 5, 17 — FFD deferred to CV loop (5_cv.py / 6_preproc.py)
        to prevent train-test leakage of the optimal d parameter.
        The SADF and SMT signals use .shift(1) to enforce the AFML
        information barrier principle.
    """
    # Step 0: Preserve raw Close for downstream dollar-barrier labels
    if "Close" in df.columns:
        df["Raw_Close"] = df["Close"].copy()
        # Strictly mask transformations targeting tracking values skipping evaluations structurally
        feature_metadata["Raw_Close"] = "target_tracking"

    # Step 1: Structural Log Transforms
    df, feature_metadata = apply_structural_log(df, feature_metadata)

    # Step 2: SADF signal on (log-transformed) Close
    new_cols = []
    if "Close" in df.columns:
        sadf = compute_sadf_signal(df["Close"], min_window=100)
        df["BTC_SADF_Bubble_Signal"] = sadf
        feature_metadata["BTC_SADF_Bubble_Signal"] = "zero_centered"
        new_cols.append("BTC_SADF_Bubble_Signal")

    # Step 3: SMT signals — two complementary configurations
    if "Close" in df.columns:
        # Signal A: short-run bubble sensitivity (low phi, linear trend)
        smt_a = compute_smt_signal(
            df["Close"], min_window=100, phi=0.1, trend_type="poly1"
        )
        col_a = smt_a.name  # "BTC_SMT_poly1_phi0.1"
        df[col_a] = smt_a
        feature_metadata[col_a] = "zero_centered"
        new_cols.append(col_a)

        # Signal B: long-run trend sensitivity (high phi, quadratic trend)
        smt_b = compute_smt_signal(
            df["Close"], min_window=100, phi=0.9, trend_type="poly2"
        )
        col_b = smt_b.name  # "BTC_SMT_poly2_phi0.9"
        df[col_b] = smt_b
        feature_metadata[col_b] = "zero_centered"
        new_cols.append(col_b)

    # FFD deferred to CV loop (5_cv.py / 6_preproc.py) to prevent
    # train-test leakage of the optimal d parameter.

    logger.info(
        "[ECONOMETRICS] Pre-CV pipeline complete. New columns added: %s",
        new_cols,
    )
    return df, feature_metadata
