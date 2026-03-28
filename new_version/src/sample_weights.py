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

import numba
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
    """Apply np.log1p to strictly non-negative ``raw_level`` features."""
    df = df.copy()
    feature_metadata = dict(feature_metadata)  # Avoid in-place mutation
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
# _numba inner loops
# ═══════════════════════════════════════════════════════════════════════════

@numba.njit
def _sadf_inner_loop(vals: np.ndarray, dy: np.ndarray, y_lag: np.ndarray, n: int, min_window: int, lags: int) -> np.ndarray:
    """Numba-compiled O(n^3) accumulator logic for SADF, supporting AR-lags."""
    sadf_vals = np.full(n, np.nan)
    num_params = 2 + lags
    
    for t in range(min_window, n):
        max_stat = -np.inf
        XtX = np.zeros((num_params, num_params))
        Xty = np.zeros(num_params)
        dy_sq_sum = 0.0
        
        # Walk backward to build accumulators avoiding loop duplication internally algebraically
        for t0 in range(t - 1, lags - 1, -1):
            x_new = np.zeros(num_params)
            x_new[0] = 1.0
            x_new[1] = y_lag[t0]
            for lag_idx in range(lags):
                x_new[2 + lag_idx] = dy[t0 - 1 - lag_idx]
                
            dy_new = dy[t0]
            
            # outer product update:
            for i in range(num_params):
                for j in range(num_params):
                    XtX[i, j] += x_new[i] * x_new[j]
                Xty[i] += x_new[i] * dy_new
                
            dy_sq_sum += dy_new * dy_new
            
            window_len = t - t0
            if window_len < min_window - 1:
                continue
                
            try:
                # Analytical 2x2 inverse if lags=0, otherwise fallback to numpy.linalg.inv inside Numba natively securely
                if num_params == 2:
                    A, B = XtX[0, 0], XtX[0, 1]
                    C, D = XtX[1, 0], XtX[1, 1]
                    det = A * D - B * C
                    if abs(det) < 1e-15:
                        continue
                    XtX_inv = np.array([[D, -B], [-C, A]]) / det
                else:
                    XtX_inv = np.linalg.inv(XtX)
                
                beta_hat = np.empty(num_params)
                for i in range(num_params):
                    acc = 0.0
                    for j in range(num_params):
                        acc += XtX_inv[i, j] * Xty[j]
                    beta_hat[i] = acc
                
                # Residual variance
                fitted_sq_sum = 0.0
                for i in range(num_params):
                    fitted_sq_sum += beta_hat[i] * Xty[i]
                    
                ssr = max(0.0, dy_sq_sum - fitted_sq_sum)
                sigma2 = ssr / max(window_len - num_params, 1)
                se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])
                
                if se_beta >= 1e-15:
                    stat = beta_hat[1] / se_beta
                    if stat > max_stat:
                        max_stat = stat
            except Exception:
                continue
                
        if max_stat > -np.inf:
            sadf_vals[t] = max_stat
            
    return sadf_vals


@numba.njit
def _smt_inner_loop(dy: np.ndarray, n: int, min_window: int, phi: float, trend_type_code: int, XtX_inv_cache: np.ndarray) -> np.ndarray:
    """Numba-compiled O(n^3) accumulators tracking SMT exponential/polynomial matrices."""
    smt_vals = np.full(n, np.nan)
    
    exp_factor = np.exp(0.01)

    for t in range(min_window, n):
        max_stat = -np.inf
        S0, S1, S2, S_exp = 0.0, 0.0, 0.0, 0.0
        dy_sq_sum = 0.0
        
        for t0 in range(t - 1, -1, -1):
            dy_new = dy[t0]
            win_len = t - t0
            
            dy_sq_sum += dy_new * dy_new
            Xty = np.zeros(2)
            
            if trend_type_code == 1:  # poly1
                S1 = dy_new + S1 + S0
                S0 = dy_new + S0
                Xty[0] = S0
                Xty[1] = S1
            elif trend_type_code == 2:  # poly2
                S2 = dy_new + S2 + 2 * S1 + S0
                S1 = dy_new + S1 + S0
                S0 = dy_new + S0
                Xty[0] = S0
                Xty[1] = S2
            else:  # exp (code 3)
                S_exp = exp_factor * (dy_new + S_exp)
                S0 = dy_new + S0
                Xty[0] = S0
                Xty[1] = S_exp
                
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

    return smt_vals


# ═══════════════════════════════════════════════════════════════════════════
# 2. Supremum ADF (SADF) Bubble Detection
# ═══════════════════════════════════════════════════════════════════════════
def compute_sadf_signal(
    series: pd.Series, min_window: int = 100, lags: int = 1, asset_prefix: str = "BTC"
) -> pd.Series:
    """Compute the Supremum ADF (SADF) bubble detection signal.

    O(n³) mathematical implementation containing a reduced sequential constant factor
    calculating AR residual boundaries evaluating lag targets analytically cleanly.

    Args:
        series: The asset price or level series (e.g., log-Close).
        min_window: Minimum number of observations for each inner
            ADF window (default 100).
        lags: Number of lagged terms tracking design matrix outputs evaluating serial
            correlation inside errors inherently logically cleanly perfectly seamlessly.
        asset_prefix: Explicit tracking variable natively cleanly bounding outputs explicitly optimally.

    Returns:
        A pd.Series of SADF t-statistics, explicitly shifted by 1
        to prevent look-ahead bias. Named ``f"{asset_prefix}_SADF_Bubble_Signal"``.
    """
    n = len(series)
    vals = series.values
    dy = np.diff(vals)
    y_lag = vals[:-1]

    # Pre-compiled boundary execution natively tracking matrices analytically
    sadf_vals = _sadf_inner_loop(vals, dy, y_lag, n, min_window, lags)

    col_name = f"{asset_prefix}_SADF_Bubble_Signal"
    result = pd.Series(
        sadf_vals, index=series.index, name=col_name
    ).shift(1)

    logger.info(
        "SADF: Computed Supremum ADF signal (min_window=%d, lags=%d, lagged=True)", min_window, lags
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
    asset_prefix: str = "BTC",
) -> pd.Series:
    """Compute the Sub/Super-Martingale Test (SMT) bubble detection signal.

    O(n³) pre-calculated boundaries utilizing exact sum equations avoiding Numba linear
    inversions internally natively cleanly!

    Args:
        series: The asset price or level series (e.g., log-Close).
        min_window: Minimum observations per inner OLS window (default 100).
        phi: Window length penalty exponent. Lower values favour short-run
            bubbles; higher values penalise short windows (default 0.5).
        trend_type: Type of time trend regressor ('poly1', 'poly2', 'exp').
        asset_prefix: Targets prefix dynamically.

    Returns:
        A pd.Series of SMT statistics, explicitly shifted by 1.
    """
    n = len(series)
    dy = series.diff().values

    # Pre-calculate explicit boundaries eliminating inverse bounds ONLY for range > min_window
    XtX_inv_cache = np.zeros((n + 1, 2, 2))
    
    for w_len in range(min_window - 1, n + 1):
        N = float(w_len)
        
        if trend_type == "poly1":
            sum_tau = N * (N + 1) / 2
            sum_tau2 = N * (N + 1) * (2 * N + 1) / 6
            det = N * sum_tau2 - sum_tau * sum_tau
            if det != 0:
                XtX_inv_cache[w_len] = np.array([[sum_tau2, -sum_tau], [-sum_tau, N]]) / det
                
        elif trend_type == "poly2":
            sum_tau2 = N * (N + 1) * (2 * N + 1) / 6
            sum_tau4 = N * (N + 1) * (2 * N + 1) * (3 * N**2 + 3 * N - 1) / 30
            det = N * sum_tau4 - sum_tau2 * sum_tau2
            if det != 0:
                XtX_inv_cache[w_len] = np.array([[sum_tau4, -sum_tau2], [-sum_tau2, N]]) / det
                
        elif trend_type == "exp":
            tau = np.arange(1, w_len + 1, dtype=float)
            trend = np.exp(0.01 * tau)
            X = np.column_stack([np.ones(w_len), trend])
            try:
                XtX_inv_cache[w_len] = np.linalg.pinv(X.T @ X)
            except np.linalg.LinAlgError:
                pass
        else:
            raise ValueError(f"Unknown trend_type: '{trend_type}'")

    trend_type_map = {"poly1": 1, "poly2": 2, "exp": 3}
    trend_type_code = trend_type_map[trend_type]

    # Pre-compiled njit array dynamically evaluating logic natively
    smt_vals = _smt_inner_loop(dy, n, min_window, phi, trend_type_code, XtX_inv_cache)

    col_name = f"{asset_prefix}_SMT_{trend_type}_phi{phi}"
    
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
    if d == 0.0:
        return np.array([1.0]).reshape(-1, 1)
        
    if not (0 < d <= 1.0):
        raise ValueError("d must be in (0, 1]")
        
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
    df: pd.DataFrame, feature_metadata: Dict[str, str], asset_prefix: str = "BTC"
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    feature_metadata = dict(feature_metadata)
    
    if "Close" in df.columns:
        df["Raw_Close"] = df["Close"].copy()
        feature_metadata["Raw_Close"] = "target_tracking"

    df, feature_metadata = apply_structural_log(df, feature_metadata)

    new_cols = []
    if "Close" in df.columns:
        sadf = compute_sadf_signal(df["Close"], min_window=100, lags=1, asset_prefix=asset_prefix)
        col_name = sadf.name
        df[col_name] = sadf
        feature_metadata[col_name] = "zero_centered"
        new_cols.append(col_name)

    if "Close" in df.columns:
        smt_a = compute_smt_signal(
            df["Close"], min_window=100, phi=0.1, trend_type="poly1", asset_prefix=asset_prefix
        )
        col_a = smt_a.name
        df[col_a] = smt_a
        feature_metadata[col_a] = "zero_centered"
        new_cols.append(col_a)

        smt_b = compute_smt_signal(
            df["Close"], min_window=100, phi=0.9, trend_type="poly2", asset_prefix=asset_prefix
        )
        col_b = smt_b.name
        df[col_b] = smt_b
        feature_metadata[col_b] = "zero_centered"
        new_cols.append(col_b)

    logger.info(
        "[ECONOMETRICS] Pre-CV pipeline complete. New columns added: %s",
        new_cols,
    )
    return df, feature_metadata
