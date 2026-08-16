"""
4.1) Features
============================
Compute the TA + math + lag feature matrix on the full daily OHLCV bar set.
Row-level alignment to labelled events happens later in ``alignment.py``.

All time windows use the crypto calendar (365 days/year, 7 days/week):
  1 week = 7,  1 month = 30,  1 quarter = 90,  6 months = 180
"""

import logging
import os
import numpy as np
import pandas as pd
import time
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

# --- TA parameters ----------------------------------------------------------
RSI_PERIOD = 14
BB_PERIOD = 20
ATR_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

# Generic rolling window for skew/kurt/realized-vol/GK-vol moments.
ROLLING_WINDOW = 30
EMA_SHORT, EMA_MID, EMA_LONG = 20, 50, 200
CHAIKIN_FAST, CHAIKIN_SLOW = 3, 10
STOCH_PERIOD, STOCH_SMOOTH = 14, 3
ROC_PERIOD = 14
CCI_PERIOD = 14
MFI_PERIOD = 14
YZ_WINDOW = 30

# --- Volatility term-structure windows --------------------------------------
VOL_SHORT, VOL_MID, VOL_LONG = 7, 30, 90      # 1 crypto week, month, quarter

# --- Mathematical feature parameters ----------------------------------------
SADF_MIN_SL = 90              # minimum sample length (~1 crypto quarter)
SADF_LAGS = 1
ENTROPY_WINDOW = 30           # ~1 crypto month
LZ_WINDOW = 90                # ~1 crypto quarter
HURST_WINDOW = 180            # ~6 crypto months
VR_WINDOW = 90                # ~1 crypto quarter
VR_LAG = 7                    # 1 crypto week
JB_WINDOW = 90                # ~1 crypto quarter
GAUSS_ENT_WINDOW = 30         # ~1 crypto month

# --- Lag features (AR Logistic baseline) ------------------------------------
# AR_LAGS covers the full weekday cycle (1-7) plus 2-week, 3-week, and
# 1-month markers; the calendar-day convention matches BTC's 24/7 trading.
AR_LAGS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]
LAG_COLUMN_PREFIX = "log_returns_lag"

# --- Cache ------------------------------------------------------------------
# Parquet cache for the O(n²) mathematical features (SADF, SMT especially).
CACHE_DIR = "cache/"
MATH_CACHE_FILE = "math_features.parquet"


# --- 1. Technical Analysis features ----------------------------------------
# Backward-looking price/volume features computed in one pass over the OHLCV bars.
def compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 25 backward-looking TA features from OHLCV data, indexed identically to ``df``."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    open_ = df["Open"]

    features = pd.DataFrame(index=df.index)

    # 1. Log returns
    log_returns = np.log(close / close.shift(1))
    features["log_returns"] = log_returns

    # 2. RSI (Wilder smoothing via EWMA)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    features["rsi"] = 100.0 - 100.0 / (1.0 + rs)

    # 3. MACD: signal line + histogram capture both trend and trend acceleration
    ema_fast = close.ewm(span=MACD_FAST, min_periods=MACD_FAST).mean()
    ema_slow = close.ewm(span=MACD_SLOW, min_periods=MACD_SLOW).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, min_periods=MACD_SIGNAL).mean()
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_line - macd_signal

    # 4. Bollinger Band width: dimensionless volatility-of-volatility proxy
    bb_mid = close.rolling(BB_PERIOD).mean()
    bb_std = close.rolling(BB_PERIOD).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    features["bb_width"] = (bb_upper - bb_lower) / bb_mid

    # 5. ATR (Wilder-smoothed True Range)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    features["atr"] = tr.ewm(alpha=1.0 / ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    # 6. OBV: cumulative signed volume
    sign = np.sign(close.diff()).fillna(0)
    features["obv"] = (volume * sign).cumsum()

    # 7-8. Rolling moments of the return distribution
    features["skewness"] = log_returns.rolling(ROLLING_WINDOW).skew()
    features["kurtosis"] = log_returns.rolling(ROLLING_WINDOW).kurt()

    # 9. Realised vol (annualised): the simplest volatility estimator
    vol_medium = log_returns.rolling(ROLLING_WINDOW).std()
    features["realized_vol"] = vol_medium * np.sqrt(365)

    # 10. Garman-Klass: uses OHLC, lower variance than close-to-close
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_daily = 0.5 * log_hl ** 2 - (2.0 * np.log(2) - 1.0) * log_co ** 2
    features["gk_vol"] = np.sqrt(gk_daily.rolling(ROLLING_WINDOW).mean().clip(lower=0))

    # 11. Yang-Zhang: best unbiased OHLC volatility estimator (handles overnight gap)
    features["yz_vol"] = _yang_zhang_volatility(
        open_, high, low, close, window=YZ_WINDOW,
    )

    # 12-13. EMA ratios: short/mid and mid/long trend signals (golden-cross-style)
    ema_20 = close.ewm(span=EMA_SHORT, min_periods=EMA_SHORT).mean()
    ema_50 = close.ewm(span=EMA_MID, min_periods=EMA_MID).mean()
    ema_200 = close.ewm(span=EMA_LONG, min_periods=EMA_LONG).mean()
    features["ema_ratio_20_50"] = ema_20 / ema_50
    features["ema_ratio_50_200"] = ema_50 / ema_200

    # 14. VWMA ratio: volume-weighted trend confirmation
    vwma_20 = (close * volume).rolling(EMA_SHORT).sum() / volume.rolling(EMA_SHORT).sum()
    vwma_50 = (close * volume).rolling(EMA_MID).sum() / volume.rolling(EMA_MID).sum()
    features["vwma_ratio_20_50"] = vwma_20 / vwma_50

    # 15. Rate of Change
    features["roc_14"] = (close / close.shift(ROC_PERIOD) - 1.0) * 100.0

    # 16-17. Stochastic oscillator: %K raw, %D smoothed
    lowest_low = low.rolling(STOCH_PERIOD).min()
    highest_high = high.rolling(STOCH_PERIOD).max()
    stoch_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    features["stoch_k"] = stoch_k
    features["stoch_d"] = stoch_k.rolling(STOCH_SMOOTH).mean()

    # 18. Williams %R: complementary momentum oscillator
    features["williams_r"] = -100.0 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

    # 19. CCI: typical-price deviation from its rolling SMA, scaled by MAD
    typical_price = (high + low + close) / 3.0
    tp_sma = typical_price.rolling(CCI_PERIOD).mean()
    tp_mad = typical_price.rolling(CCI_PERIOD).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True,
    )
    features["cci_14"] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)

    # 20. Chaikin Oscillator: MACD applied to the accumulation/distribution line
    money_flow_mult = ((close - low) - (high - close)) / (high - low + 1e-10)
    money_flow_vol = money_flow_mult * volume
    adl = money_flow_vol.cumsum()
    features["chaikin_osc"] = (
        adl.ewm(span=CHAIKIN_FAST, min_periods=CHAIKIN_FAST).mean()
        - adl.ewm(span=CHAIKIN_SLOW, min_periods=CHAIKIN_SLOW).mean()
    )

    # 21. Money Flow Index: volume-weighted RSI
    features["mfi_14"] = _money_flow_index(high, low, close, volume, period=MFI_PERIOD)

    # 22-23. Volatility term structure: short/mid and mid/long ratios.
    # Annualisation cancels in ratios, so we use raw rolling std.
    vol_short = log_returns.rolling(VOL_SHORT).std()
    vol_long = log_returns.rolling(VOL_LONG).std()
    features["vol_term_7_30"] = vol_short / vol_medium.replace(0, np.nan)
    features["vol_term_30_90"] = vol_medium / vol_long.replace(0, np.nan)

    logger.info("TA features: %d columns, %d rows.", features.shape[1], features.shape[0])
    return features


# Yang-Zhang OHLC volatility estimator (Yang & Zhang, 2000).
def _yang_zhang_volatility(open_: pd.Series, high: pd.Series, low: pd.Series,
                           close: pd.Series, window: int = 21) -> pd.Series:
    """Combine overnight, open-to-close, and Rogers-Satchell components for an
    unbiased OHLC volatility estimate that captures the overnight gap."""
    log_oc = np.log(open_ / close.shift(1))
    log_co = np.log(close / open_)
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    log_hc = np.log(high / close)
    log_lc = np.log(low / close)

    rs = log_ho * log_hc + log_lo * log_lc

    # k optimally balances the overnight and intraday components for unbiasedness.
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    var_overnight = log_oc.rolling(window).var()
    var_close_open = log_co.rolling(window).var()
    var_rs = rs.rolling(window).mean()

    yz_var = var_overnight + k * var_close_open + (1 - k) * var_rs

    return np.sqrt(yz_var.clip(lower=0))


# Money Flow Index: volume-weighted variant of RSI.
def _money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series, period: int = 14) -> pd.Series:
    """Compute the period-window MFI from OHLC + volume."""
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    tp_diff = typical_price.diff()

    # Split each bar's money flow into positive/negative bucket by the sign of typical-price change.
    pos_flow = pd.Series(0.0, index=close.index)
    neg_flow = pd.Series(0.0, index=close.index)

    pos_mask = tp_diff > 0
    neg_mask = tp_diff < 0

    pos_flow[pos_mask] = raw_money_flow[pos_mask]
    neg_flow[neg_mask] = raw_money_flow[neg_mask]

    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()

    money_ratio = pos_sum / (neg_sum + 1e-10)
    mfi = 100.0 - 100.0 / (1.0 + money_ratio)

    return mfi


# --- 2. Mathematical features (AFML Part 4) --------------------------------
# Cached on Parquet because SADF and SMT are O(n²) and dominate the build time.
def compute_math_features(df: pd.DataFrame, which: list[str] | str = "all") -> pd.DataFrame:
    """Compute AFML mathematical features with Parquet caching.

    ``which`` selects a subset: 'shannon_entropy', 'lz_complexity', 'hurst',
    'variance_ratio', 'jarque_bera', 'negentropy', 'sadf', 'smt'. Pass "all"
    for everything.
    """
    ALL_MATH = ["shannon_entropy", "lz_complexity", "hurst", "variance_ratio",
                "jarque_bera", "negentropy", "sadf", "smt"]
    if which == "all":
        which = ALL_MATH

    # Resolve the column set the caller actually wants once; both the cache path
    # and the recompute path return the same final shape.
    requested_cols: set[str] = set()
    for w in which:
        if w == "smt":
            requested_cols.update(["smt_poly1", "smt_exp"])
        else:
            requested_cols.add(w)

    cache_path = os.path.join(CACHE_DIR, MATH_CACHE_FILE)

    # Cache hit requires the date range to match exactly AND every requested column to be present.
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        cached_cols = set(cached.columns)

        if (cached.index[0] == df.index[0]
                and cached.index[-1] == df.index[-1]
                and requested_cols.issubset(cached_cols)):
            logger.info("Math features loaded from cache (%s).", requested_cols)
            return cached[sorted(requested_cols)]
        logger.info("Cache miss (date range or columns). Recomputing requested features.")

    close = df["Close"]
    log_price = np.log(close)
    log_returns = np.log(close / close.shift(1)).dropna()

    features = pd.DataFrame(index=df.index)

    # Compute requested features, ordered from cheapest to most expensive
    # so an interrupted run still saves the easy results.
    if "shannon_entropy" in which:
        t0 = time.time()
        print("[features] Computing Shannon entropy...")
        features["shannon_entropy"] = _compute_shannon_entropy(log_returns, window=ENTROPY_WINDOW)
        print(f"  Shannon entropy took {(time.time() - t0) / 60:.1f} min")

    if "negentropy" in which:
        t0 = time.time()
        print("[features] Computing Negentropy (Gaussian − Shannon)...")
        gauss_ent = _compute_rolling_gaussian_entropy(
            log_returns, window=GAUSS_ENT_WINDOW,
        )
        shannon_ent = _compute_shannon_entropy(
            log_returns, window=GAUSS_ENT_WINDOW,
        )
        # Distance from Gaussian: a return distribution with fat tails has high negentropy.
        features["negentropy"] = gauss_ent - shannon_ent
        print(f"  Negentropy took {(time.time() - t0) / 60:.1f} min")

    if "lz_complexity" in which:
        t0 = time.time()
        print("[features] Computing Lempel-Ziv complexity...")
        features["lz_complexity"] = _compute_rolling_lz(log_returns, window=LZ_WINDOW)
        print(f"  LZ took {(time.time() - t0) / 60:.1f} min")

    if "variance_ratio" in which:
        t0 = time.time()
        print("[features] Computing Variance Ratio (Lo & MacKinlay)...")
        features["variance_ratio"] = _compute_rolling_variance_ratio(
            log_returns, window=VR_WINDOW, lag=VR_LAG,
        )
        print(f"  Variance Ratio took {(time.time() - t0) / 60:.1f} min")

    if "jarque_bera" in which:
        t0 = time.time()
        print("[features] Computing Jarque-Bera statistic...")
        features["jarque_bera"] = _compute_rolling_jarque_bera(
            log_returns, window=JB_WINDOW,
        )
        print(f"  Jarque-Bera took {(time.time() - t0) / 60:.1f} min")

    if "hurst" in which:
        t0 = time.time()
        print("[features] Computing Hurst exponent...")
        features["hurst"] = _compute_rolling_hurst(log_returns, window=HURST_WINDOW)
        print(f"  Hurst took {(time.time() - t0) / 60:.1f} min")

    if "sadf" in which:
        t0 = time.time()
        print("[features] Computing SADF (O(n²), may take ~15 min)...")
        features["sadf"] = _compute_sadf(log_price)
        print(f"  SADF took {(time.time() - t0) / 60:.1f} min")

    if "smt" in which:
        t0 = time.time()
        print("[features] Computing SMT (O(n²), may take ~30 min)...")
        smt_poly1, smt_exp = _compute_smt(log_price)
        features["smt_poly1"] = smt_poly1
        features["smt_exp"] = smt_exp
        print(f"  SMT took {(time.time() - t0) / 60:.1f} min")

    # Merge new columns into the existing cache if compatible, otherwise overwrite.
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if cached.index[0] == df.index[0] and cached.index[-1] == df.index[-1]:
            for col in features.columns:
                cached[col] = features[col]
            cached.to_parquet(cache_path)
            logger.info("Cache updated with %s.", list(features.columns))
            return cached[sorted(requested_cols)]
    features.to_parquet(cache_path)
    logger.info("Math features cached to %s.", cache_path)

    return features[sorted(requested_cols)]


# Rolling Variance Ratio (Lo & MacKinlay, 1988): random-walk null hypothesis test.
def _compute_rolling_variance_ratio(log_returns: pd.Series, window: int = VR_WINDOW, lag: int = VR_LAG) -> pd.Series:
    """``VR(q) = Var(r_q) / (q · Var(r_1))``. VR > 1 ⇒ momentum, < 1 ⇒ mean-reversion, ≈ 1 ⇒ random walk."""
    result = pd.Series(np.nan, index=log_returns.index)
    values = log_returns.values
    n = len(values)

    for i in range(window - 1, n):
        w = values[i - window + 1 : i + 1]
        if np.any(np.isnan(w)):
            continue

        var_1 = np.var(w, ddof=1)
        if var_1 < 1e-20:
            continue

        # Build the q-period aggregated returns and compute their variance.
        m = len(w) - lag
        if m < 2:
            continue
        multi_ret = np.array([w[j:j + lag].sum() for j in range(m)])
        var_q = np.var(multi_ret, ddof=1)

        result.iloc[i] = var_q / (lag * var_1)

    return result


# Rolling Jarque-Bera statistic (Jarque & Bera, 1987): scalar measure of non-normality.
def _compute_rolling_jarque_bera(log_returns: pd.Series, window: int = JB_WINDOW) -> pd.Series:
    """``JB = (n/6) · (S² + K²/4)`` where S is skewness and K is excess kurtosis."""
    result = pd.Series(np.nan, index=log_returns.index)
    values = log_returns.values
    n = len(values)

    for i in range(window - 1, n):
        w = values[i - window + 1 : i + 1]
        if np.any(np.isnan(w)):
            continue

        m = len(w)
        mean_w = np.mean(w)
        std_w = np.std(w, ddof=1)
        if std_w < 1e-20:
            continue

        # Standardise the window so skew/kurt are scale-invariant.
        z = (w - mean_w) / std_w
        skew = np.mean(z ** 3)
        kurt = np.mean(z ** 4) - 3.0

        jb = (m / 6.0) * (skew ** 2 + kurt ** 2 / 4.0)
        result.iloc[i] = jb

    return result


# Rolling Gaussian entropy (AFML Ch. 18.6): the reference point for negentropy.
def _compute_rolling_gaussian_entropy(log_returns: pd.Series, window: int = GAUSS_ENT_WINDOW) -> pd.Series:
    """``H_gauss = 0.5 · ln(2πe·σ²)``. The gap to empirical Shannon entropy is negentropy."""
    result = pd.Series(np.nan, index=log_returns.index)
    values = log_returns.values
    n = len(values)

    for i in range(window - 1, n):
        w = values[i - window + 1 : i + 1]
        if np.any(np.isnan(w)):
            continue

        var = np.var(w, ddof=1)
        if var < 1e-20:
            continue

        result.iloc[i] = 0.5 * np.log(2.0 * np.pi * np.e * var)

    return result


# SADF (AFML Snippet 17.1): supremum ADF test for explosive-bubble detection.
def _compute_sadf(log_price: pd.Series) -> pd.Series:
    """For each time t, take the supremum ADF t-stat across all expanding sub-samples ending at t."""
    result = pd.Series(np.nan, index=log_price.index)
    y = log_price.values
    n = len(y)

    # O(n²): for each t, sweep over all start points t0 to find the worst-case t-stat.
    for t in range(SADF_MIN_SL, n):
        sup_adf = -np.inf
        for t0 in range(0, t - SADF_MIN_SL + 1):
            segment = y[t0 : t + 1]
            tstat = _adf_tstat(segment, lags=SADF_LAGS)
            if tstat > sup_adf:
                sup_adf = tstat
        result.iloc[t] = sup_adf if sup_adf > -np.inf else np.nan

    return result


# OLS-based ADF t-statistic; the workhorse for SADF.
def _adf_tstat(y: np.ndarray, lags: int = 1) -> float:
    """T-stat for β in Δy_t = α + β·y_{t-1} + Σγ_k·Δy_{t-k} + ε. NaN on rank deficiency."""
    dy = np.diff(y)
    n = len(dy)

    if n <= lags + 2:
        return np.nan

    start = lags
    dep = dy[start:]
    m = len(dep)

    if m <= lags + 2:
        return np.nan

    # Build the design matrix: intercept, level, then `lags` differences.
    regressors = np.ones((m, 2 + lags))
    regressors[:, 1] = y[start : start + m]
    for k in range(1, lags + 1):
        regressors[:, 1 + k] = dy[start - k : start - k + m]

    try:
        beta, residuals, _, _ = np.linalg.lstsq(regressors, dep, rcond=None)
        if len(residuals) == 0:
            resid = dep - regressors @ beta
            sse = np.sum(resid ** 2)
        else:
            sse = residuals[0]
        mse = sse / (m - regressors.shape[1])
        cov = mse * np.linalg.inv(regressors.T @ regressors)
        se_beta = np.sqrt(cov[1, 1])
        return beta[1] / se_beta if se_beta > 0 else np.nan
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


# SMT (AFML §17.4.3): sub/super-martingale specification tests under polynomial and exponential trends.
def _compute_smt(log_price: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return the SMT statistic series for the polynomial-1 and exponential alternatives."""
    n = len(log_price)
    y = log_price.values
    idx = log_price.index

    smt_poly1 = pd.Series(np.nan, index=idx)
    smt_exp = pd.Series(np.nan, index=idx)

    # O(n²) like SADF, with two alternative regressors per inner loop.
    for t in range(SADF_MIN_SL, n):
        sup_poly = -np.inf
        sup_exp = -np.inf

        for t0 in range(0, t - SADF_MIN_SL + 1):
            seg = y[t0 : t + 1]
            length = len(seg)
            phi = 0.5  # AFML's recommended length-penalty exponent

            # Polynomial-1: regress log price on linear time trend.
            t_vec = np.arange(length, dtype=np.float64)
            tstat_p = _ols_tstat(seg, t_vec)
            if not np.isnan(tstat_p):
                val = abs(tstat_p) / (length ** phi)
                if val > sup_poly:
                    sup_poly = val

            # Exponential: regress log price on exp(t/length).
            exp_vec = np.exp(t_vec / length)
            tstat_e = _ols_tstat(seg, exp_vec)
            if not np.isnan(tstat_e):
                val = abs(tstat_e) / (length ** phi)
                if val > sup_exp:
                    sup_exp = val

        smt_poly1.iloc[t] = sup_poly if sup_poly > -np.inf else np.nan
        smt_exp.iloc[t] = sup_exp if sup_exp > -np.inf else np.nan

    return smt_poly1, smt_exp


# OLS t-statistic for a simple two-column regression; helper for SMT.
def _ols_tstat(y: np.ndarray, x: np.ndarray) -> float:
    """T-stat for β in y = α + β·x + ε. NaN on rank deficiency."""
    n = len(y)
    if n < 3:
        return np.nan
    X = np.column_stack([np.ones(n), x])
    try:
        beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        if len(residuals) == 0:
            resid = y - X @ beta
            sse = np.sum(resid ** 2)
        else:
            sse = residuals[0]
        mse = sse / (n - 2)
        cov = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(cov[1, 1])
        return beta[1] / se if se > 0 else np.nan
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


# Shannon entropy over equal-width return-distribution bins.
def _compute_shannon_entropy(log_returns: pd.Series, window: int = ENTROPY_WINDOW) -> pd.Series:
    """Equal-width histogram Shannon entropy over a rolling window.

    Equal-width binning rather than quantile binning, because clustered values
    can collapse all quantile edges and produce a degenerate zero entropy.
    """
    result = pd.Series(np.nan, index=log_returns.index)
    values = log_returns.values
    n = len(values)
    n_bins = 10

    for i in range(window - 1, n):
        w = values[i - window + 1 : i + 1]
        if np.any(np.isnan(w)):
            continue

        w_min, w_max = w.min(), w.max()
        spread = w_max - w_min
        if spread < 1e-14:
            # All values identical → zero information.
            result.iloc[i] = 0.0
            continue

        try:
            # Bin edges padded by 1e-10 on each side so endpoint values land cleanly.
            edges = np.linspace(w_min - 1e-10, w_max + 1e-10, n_bins + 1)
            digitized = np.digitize(w, edges[1:-1])
            counts = np.bincount(digitized, minlength=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log2(probs))
        except (ValueError, IndexError):
            ent = np.nan
        result.iloc[i] = ent

    return result


# Rolling Lempel-Ziv complexity on the binary-encoded return sign sequence.
def _compute_rolling_lz(log_returns: pd.Series, window: int = LZ_WINDOW) -> pd.Series:
    """LZ-76 complexity of the up/down indicator sequence, normalised by window/log2(window)."""
    result = pd.Series(np.nan, index=log_returns.index)
    values = log_returns.values
    n = len(values)

    for i in range(window - 1, n):
        w = values[i - window + 1 : i + 1]
        if np.any(np.isnan(w)):
            continue
        # Encode each return as 1 (up) or 0 (down) and count distinct sub-patterns.
        binary_str = "".join("1" if r > 0 else "0" for r in w)
        c = _lempel_ziv_76(binary_str)
        norm = window / np.log2(window) if window > 1 else 1.0
        result.iloc[i] = c / norm

    return result


# Lempel-Ziv-76 sub-pattern counter on a binary string.
def _lempel_ziv_76(s: str) -> int:
    """Standard LZ-76 implementation; returns the count of distinct sub-patterns."""
    n = len(s)
    if n == 0:
        return 0
    complexity = 1
    i = 0
    k = 1
    k_max = 1
    while i + k <= n:
        if s[i + 1 : i + k + 1] in s[0 : i + k]:
            k += 1
            if i + k > n:
                complexity += 1
                break
        else:
            complexity += 1
            if k > k_max:
                k_max = k
            i += k_max
            k = 1
            k_max = 1
            if i >= n - 1:
                break
    return complexity


# Rolling Hurst exponent via Rescaled-Range (R/S) analysis at multiple sub-period scales.
def _compute_rolling_hurst(log_returns: pd.Series, window: int = HURST_WINDOW) -> pd.Series:
    """Hurst > 0.5 ⇒ persistence (trending), < 0.5 ⇒ anti-persistence (mean-reverting)."""
    result = pd.Series(np.nan, index=log_returns.index)
    values = log_returns.values
    n = len(values)
    # Sub-period sizes chosen from the crypto calendar; we fit slope across them.
    sub_periods = np.array([14, 30, 60, 90])
    sub_periods = sub_periods[sub_periods < window]

    for i in range(window - 1, n):
        w = values[i - window + 1 : i + 1]
        if np.any(np.isnan(w)):
            continue
        result.iloc[i] = _hurst_rs(w, sub_periods)

    return result


# R/S estimator: fit a line to log(R/S) vs log(period); the slope is the Hurst exponent.
def _hurst_rs(data: np.ndarray, sub_periods: np.ndarray) -> float:
    """Compute Hurst as the slope of log(R/S) on log(period) across sub_periods."""
    log_ns = []
    log_rs = []

    for sp in sub_periods:
        n_chunks = len(data) // sp
        if n_chunks < 1:
            continue
        # For each chunk, the rescaled-range R/S is range-of-cumulative-deviations / std.
        rs_values = []
        for j in range(n_chunks):
            chunk = data[j * sp : (j + 1) * sp]
            mean_c = chunk.mean()
            deviations = np.cumsum(chunk - mean_c)
            r = deviations.max() - deviations.min()
            s = chunk.std(ddof=1)
            if s > 1e-12:
                rs_values.append(r / s)
        if rs_values:
            log_ns.append(np.log(sp))
            log_rs.append(np.log(np.mean(rs_values)))

    if len(log_ns) < 2:
        return np.nan

    log_ns = np.array(log_ns)
    log_rs = np.array(log_rs)
    slope, _ = np.polyfit(log_ns, log_rs, 1)
    return slope


# --- 3. Lag features (AR Logistic baseline) --------------------------------
# Helper that other modules use to reference the lag columns by name.
def lag_column_names(lags: list[int] | None = None) -> list[str]:
    """Canonical lag column names in the order matching ``lags`` (defaults to ``AR_LAGS``)."""
    if lags is None:
        lags = AR_LAGS
    return [f"{LAG_COLUMN_PREFIX}{k}" for k in lags]


# Precompute lagged log returns once on the full series; consumed by AR Logistic.
def compute_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Precompute lagged log-return features on the full daily series.

    Computing on the global series rather than inline at AR Logistic fit/predict
    time guarantees that every aligned event has valid lookback values respecting
    chronological order. The first ``max(lags)`` rows contain NaN; downstream
    CUSUM filtering drops them because the volatility EWMA warm-up exceeds
    ``max(AR_LAGS)``.
    """
    if lags is None:
        lags = AR_LAGS

    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # Build one shifted-return column per lag k.
    lag_df = pd.DataFrame(index=df.index)
    for k in lags:
        lag_df[f"{LAG_COLUMN_PREFIX}{k}"] = log_ret.shift(k)

    logger.info(
        "Lag features: %d columns, %d rows (lags=%s).",
        lag_df.shape[1], lag_df.shape[0], lags,
    )
    return lag_df


# --- 4. Compression transforms ---------------------------------------------
# Symmetric log: sign-preserving variant of natural log for signed wide-range features.
def apply_sym_log(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply ``sign(x) · log(|x| + 1)`` to listed columns; returns a modified copy.

    Used for signed features whose magnitudes span many orders of magnitude (e.g.
    ``obv`` and ``chaikin_osc`` reach 10⁷-10¹⁰ during high-volume regimes).
    Preserves sign and zero, preserves rank ordering, and asymptotically behaves
    like ``sign(x) · log(|x|)`` for ``|x| ≫ 1``. The derivative discontinuity at
    zero is acceptable here because the transform feeds downstream MDA, FFD, and
    scaling rather than a gradient-based optimiser.
    """
    features = features.copy()
    applied = []
    # Skip silently if a target column isn't in the frame, so the same list can be reused.
    for col in columns:
        if col not in features.columns:
            continue
        features[col] = np.sign(features[col]) * np.log(features[col].abs() + 1)
        applied.append(col)
    if applied:
        logger.info("symmetric log applied to: %s", applied)
    return features


# Unsigned log with ε floor: variance-stabilising transform for strictly non-negative features.
def apply_log(features: pd.DataFrame, columns: list[str], eps: float = 1e-8) -> pd.DataFrame:
    """Apply ``log(|x| + eps)`` to listed columns; returns a modified copy.

    Standard variance-stabilising transform for non-negative wide-range features
    (e.g. ATR, realised volatility). ``|x|`` guards against any spurious negative
    input; ``eps`` keeps the transform finite at zero.
    """
    features = features.copy()
    applied = []
    for col in columns:
        if col not in features.columns:
            continue
        features[col] = np.log(features[col].abs() + eps)
        applied.append(col)
    if applied:
        logger.info("log applied to: %s (eps=%g)", applied, eps)
    return features


# --- 5. Orchestration -------------------------------------------------------
# Build the TA + math half of the feature matrix; the notebook concatenates the rest.
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TA + math features and return them concatenated; no compression applied.

    Compression transforms (``apply_sym_log``, ``apply_log``) are applied in the
    notebook so the choice of which columns get which transform is visible at
    the call site rather than hidden in this function. External and lag features
    are added separately in the notebook because they have their own fetchers
    and the canonical concat order is ``[ta, math, external, lag]``.
    """
    ta = compute_ta_features(df)
    math = compute_math_features(df)
    features = pd.concat([ta, math], axis=1)

    print(
        f"[features] {features.shape[1]} features, {features.shape[0]} rows | "
        f"NaN rows (any): {features.isna().any(axis=1).sum()}"
    )

    return features