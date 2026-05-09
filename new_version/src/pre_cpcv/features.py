"""
4.1) Features
============================
Compute all features from the full OHLCV DataFrame. Returns a feature
matrix covering every daily bar. Feature selection by row (restricting
to labeled events) happens later in alignment.

Implements technical analysis features, AFML Part 4 mathematical
features, lag features (AR Logistic baseline), and log transforms.

All time windows use crypto calendar (365 days/year, 7 days/week):
  1 week = 7,  1 month = 30,  1 quarter = 90,  6 months = 180

TA features (25 total):
  log_returns, rsi, macd, macd_signal, macd_hist, bb_width, atr, obv,
  skewness, kurtosis, realized_vol, gk_vol, yz_vol, ema_ratio_20_50,
  ema_ratio_50_200, vwma_ratio_20_50, roc_14, stoch_k, stoch_d,
  williams_r, cci_14, chaikin_osc, mfi_14,
  vol_term_7_30, vol_term_30_90

Mathematical features (9 total):
  shannon_entropy, lz_complexity, hurst, variance_ratio, jarque_bera,
  negentropy, sadf, smt_poly1, smt_exp

Lag features (6 total, AR Logistic only):
  log_returns_lag1, log_returns_lag2, log_returns_lag3,
  log_returns_lag7, log_returns_lag14, log_returns_lag30

  Precomputed once on the full daily series so the AR Logistic baseline
  does not need to build lags inline at fit/predict time. Excluded
  from MDA feature selection in preprocessing — they are a separate
  category routed only to AR Logistic.
"""

import logging
import os
import numpy as np
import pandas as pd
import time
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

# ── TA parameters ─────────────────────────────────────────────────────
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
ATR_PERIOD = 14
ROLLING_WINDOW = 30              

EMA_SHORT = 20
EMA_MID = 50
EMA_LONG = 200
ROC_PERIOD = 14
STOCH_PERIOD = 14
STOCH_SMOOTH = 3
CCI_PERIOD = 14
MFI_PERIOD = 14
CHAIKIN_FAST = 3
CHAIKIN_SLOW = 10
YZ_WINDOW = 30                   

# ── Volatility term-structure windows ─────────────────────────────────
VOL_SHORT = 7                    # 1 crypto week
VOL_MID = 30                     # 1 crypto month
VOL_LONG = 90                    # 1 crypto quarter

# ── Mathematical feature parameters ───────────────────────────────────
SADF_MIN_SL = 90                 # minimum sample length (~1 crypto quarter)
SADF_LAGS = 1
ENTROPY_WINDOW = 30              # ~1 crypto month
LZ_WINDOW = 90                   # ~1 crypto quarter
HURST_WINDOW = 180               # ~6 crypto months
VR_WINDOW = 90                   # ~1 crypto quarter
VR_LAG = 7                       # 1 crypto week
JB_WINDOW = 90                   # ~1 crypto quarter
GAUSS_ENT_WINDOW = 30            # ~1 crypto month

# ── Lag features (consumed only by AR Logistic baseline) ──────────────
AR_LAGS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]
LAG_COLUMN_PREFIX = "log_returns_lag"

# ── Cache ─────────────────────────────────────────────────────────────
CACHE_DIR = "cache/"
MATH_CACHE_FILE = "math_features.parquet"


# =====================================================================
# Technical Analysis Features
# =====================================================================
def compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute backward-looking TA features from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        One column per feature, indexed identically to *df*.
    """
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

    # 3. MACD
    ema_fast = close.ewm(span=MACD_FAST, min_periods=MACD_FAST).mean()
    ema_slow = close.ewm(span=MACD_SLOW, min_periods=MACD_SLOW).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, min_periods=MACD_SIGNAL).mean()
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_line - macd_signal

    # 4. Bollinger Band width (dimensionless)
    bb_mid = close.rolling(BB_PERIOD).mean()
    bb_std = close.rolling(BB_PERIOD).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    features["bb_width"] = (bb_upper - bb_lower) / bb_mid

    # 5. ATR (EWMA smoothed)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    features["atr"] = tr.ewm(span=ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    # 6. OBV
    sign = np.sign(close.diff()).fillna(0)
    features["obv"] = (volume * sign).cumsum()

    # 7. Rolling skewness
    features["skewness"] = log_returns.rolling(ROLLING_WINDOW).skew()

    # 8. Rolling kurtosis
    features["kurtosis"] = log_returns.rolling(ROLLING_WINDOW).kurt()

    # 9. Rolling realized volatility (annualized)
    vol_medium = log_returns.rolling(ROLLING_WINDOW).std()
    features["realized_vol"] = vol_medium * np.sqrt(365)

    # 10. Garman-Klass volatility (rolling)
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_daily = 0.5 * log_hl ** 2 - (2.0 * np.log(2) - 1.0) * log_co ** 2
    features["gk_vol"] = gk_daily.rolling(ROLLING_WINDOW).mean()

    # 11. Yang-Zhang volatility (best unbiased OHLC estimator)
    features["yz_vol"] = _yang_zhang_volatility(
        open_, high, low, close, window=YZ_WINDOW,
    )

    # 12. EMA ratio 20/50 (short vs medium trend)
    ema_20 = close.ewm(span=EMA_SHORT, min_periods=EMA_SHORT).mean()
    ema_50 = close.ewm(span=EMA_MID, min_periods=EMA_MID).mean()
    features["ema_ratio_20_50"] = ema_20 / ema_50

    # 13. EMA ratio 50/200 (golden/death cross signal)
    ema_200 = close.ewm(span=EMA_LONG, min_periods=EMA_LONG).mean()
    features["ema_ratio_50_200"] = ema_50 / ema_200

    # 14. VWMA ratio 20/50 (volume-weighted trend confirmation)
    vwma_20 = (close * volume).rolling(EMA_SHORT).sum() / volume.rolling(EMA_SHORT).sum()
    vwma_50 = (close * volume).rolling(EMA_MID).sum() / volume.rolling(EMA_MID).sum()
    features["vwma_ratio_20_50"] = vwma_20 / vwma_50

    # 15. Rate of Change
    features["roc_14"] = (close / close.shift(ROC_PERIOD) - 1.0) * 100.0

    # 16. Stochastic %K
    lowest_low = low.rolling(STOCH_PERIOD).min()
    highest_high = high.rolling(STOCH_PERIOD).max()
    stoch_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    features["stoch_k"] = stoch_k

    # 17. Stochastic %D (smoothed %K)
    features["stoch_d"] = stoch_k.rolling(STOCH_SMOOTH).mean()

    # 18. Williams %R (momentum confirmation)
    features["williams_r"] = -100.0 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

    # 19. Commodity Channel Index
    typical_price = (high + low + close) / 3.0
    tp_sma = typical_price.rolling(CCI_PERIOD).mean()
    tp_mad = typical_price.rolling(CCI_PERIOD).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True,
    )
    features["cci_14"] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-10)

    # 20. Chaikin Oscillator (volume + price momentum)
    money_flow_mult = ((close - low) - (high - close)) / (high - low + 1e-10)
    money_flow_vol = money_flow_mult * volume
    adl = money_flow_vol.cumsum()
    features["chaikin_osc"] = (
        adl.ewm(span=CHAIKIN_FAST, min_periods=CHAIKIN_FAST).mean()
        - adl.ewm(span=CHAIKIN_SLOW, min_periods=CHAIKIN_SLOW).mean()
    )

    # 21. Money Flow Index (volume-weighted RSI)
    features["mfi_14"] = _money_flow_index(high, low, close, volume, period=MFI_PERIOD)

    # 22. Volatility term structure: short vol / long vol
    #     Annualization cancels in ratios, so use raw rolling std
    vol_short = log_returns.rolling(VOL_SHORT).std()
    vol_long = log_returns.rolling(VOL_LONG).std()
    features["vol_term_7_30"] = vol_short / vol_medium.replace(0, np.nan)
    features["vol_term_30_90"] = vol_medium / vol_long.replace(0, np.nan)

    logger.info("TA features: %d columns, %d rows.", features.shape[1], features.shape[0])
    return features


# ---------------------------------------------------------------------------
# Yang-Zhang volatility
# ---------------------------------------------------------------------------
def _yang_zhang_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Yang-Zhang (2000) volatility estimator.

    Combines overnight (close-to-open), open-to-close, and
    Rogers-Satchell components for the most efficient unbiased
    OHLC volatility estimate.
    """
    log_oc = np.log(open_ / close.shift(1))
    log_co = np.log(close / open_)
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    log_hc = np.log(high / close)
    log_lc = np.log(low / close)

    rs = log_ho * log_hc + log_lo * log_lc

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    var_overnight = log_oc.rolling(window).var()
    var_close_open = log_co.rolling(window).var()
    var_rs = rs.rolling(window).mean()

    yz_var = var_overnight + k * var_close_open + (1 - k) * var_rs

    return np.sqrt(yz_var.clip(lower=0))


# ---------------------------------------------------------------------------
# Money Flow Index
# ---------------------------------------------------------------------------
def _money_flow_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index: volume-weighted RSI."""
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    tp_diff = typical_price.diff()

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


# =====================================================================
# Mathematical Features (AFML Part 4)
# =====================================================================
def compute_math_features(
    df: pd.DataFrame,
    which: list[str] | str = "all",
) -> pd.DataFrame:
    """Compute AFML mathematical features with Parquet caching.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    which : list[str] or "all"
        Features to compute. Options: 'shannon_entropy', 'lz_complexity',
        'hurst', 'variance_ratio', 'jarque_bera', 'negentropy',
        'sadf', 'smt'. Pass "all" for everything.
    """
    ALL_MATH = ["shannon_entropy", "lz_complexity", "hurst", "variance_ratio",
                "jarque_bera", "negentropy", "sadf", "smt"]
    if which == "all":
        which = ALL_MATH

    cache_path = os.path.join(CACHE_DIR, MATH_CACHE_FILE)

    # check cache
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        cached_cols = set(cached.columns)
        requested_cols = set()
        for w in which:
            if w == "smt":
                requested_cols.update(["smt_poly1", "smt_exp"])
            else:
                requested_cols.add(w)

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

    # ordered from least to most expensive computation
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
        # Negentropy: how far the empirical distribution is from Gaussian
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

    # save/update cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if cached.index[0] == df.index[0] and cached.index[-1] == df.index[-1]:
            for col in features.columns:
                cached[col] = features[col]
            cached.to_parquet(cache_path)
            logger.info("Cache updated with %s.", list(features.columns))
            return cached
    features.to_parquet(cache_path)
    logger.info("Math features cached to %s.", cache_path)

    return features


# ---------------------------------------------------------------------------
# Rolling Variance Ratio (Lo & MacKinlay, 1988)
# ---------------------------------------------------------------------------
def _compute_rolling_variance_ratio(
    log_returns: pd.Series, window: int = VR_WINDOW, lag: int = VR_LAG,
) -> pd.Series:
    """Variance ratio VR(q) = Var(r_q) / (q * Var(r_1)).

    VR > 1 indicates momentum (positive autocorrelation).
    VR < 1 indicates mean-reversion (negative autocorrelation).
    VR = 1 is consistent with a random walk.
    """
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

        m = len(w) - lag
        if m < 2:
            continue
        multi_ret = np.array([w[j:j + lag].sum() for j in range(m)])
        var_q = np.var(multi_ret, ddof=1)

        result.iloc[i] = var_q / (lag * var_1)

    return result.reindex(log_returns.index)


# ---------------------------------------------------------------------------
# Rolling Jarque-Bera statistic (Jarque & Bera, 1987)
# ---------------------------------------------------------------------------
def _compute_rolling_jarque_bera(
    log_returns: pd.Series, window: int = JB_WINDOW,
) -> pd.Series:
    """JB = (n/6) * (S² + K²/4).

    Measures departure from normality. S = skewness, K = excess kurtosis.
    Higher values indicate stronger non-Gaussianity.
    """
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

        z = (w - mean_w) / std_w
        skew = np.mean(z ** 3)
        kurt = np.mean(z ** 4) - 3.0

        jb = (m / 6.0) * (skew ** 2 + kurt ** 2 / 4.0)
        result.iloc[i] = jb

    return result.reindex(log_returns.index)


# ---------------------------------------------------------------------------
# Rolling Gaussian Entropy (AFML Ch. 18.6)
# ---------------------------------------------------------------------------
def _compute_rolling_gaussian_entropy(
    log_returns: pd.Series, window: int = GAUSS_ENT_WINDOW,
) -> pd.Series:
    """H_gauss = 0.5 * ln(2 * pi * e * sigma²).

    The entropy of a Gaussian process with the same variance.
    The gap between this and empirical Shannon entropy measures
    how non-Gaussian the return distribution is.
    """
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

    return result.reindex(log_returns.index)


# ---------------------------------------------------------------------------
# SADF (AFML Snippet 17.1)
# ---------------------------------------------------------------------------
def _compute_sadf(log_price: pd.Series) -> pd.Series:
    """Supremum Augmented Dickey-Fuller test on log prices."""
    result = pd.Series(np.nan, index=log_price.index)
    y = log_price.values
    n = len(y)

    for t in range(SADF_MIN_SL, n):
        sup_adf = -np.inf
        for t0 in range(0, t - SADF_MIN_SL + 1):
            segment = y[t0 : t + 1]
            tstat = _adf_tstat(segment, lags=SADF_LAGS)
            if tstat > sup_adf:
                sup_adf = tstat
        result.iloc[t] = sup_adf

    return result


def _adf_tstat(y: np.ndarray, lags: int = 1) -> float:
    """Compute the ADF t-statistic for β in Δy_t = α + β*y_{t-1} + Σγ_k*Δy_{t-k} + ε."""
    dy = np.diff(y)
    n = len(dy)

    if n <= lags + 2:
        return np.nan

    start = lags
    dep = dy[start:]
    m = len(dep)

    if m <= lags + 2:
        return np.nan

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


# ---------------------------------------------------------------------------
# SMT (AFML Section 17.4.3)
# ---------------------------------------------------------------------------
def _compute_smt(log_price: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Sub/Super-Martingale tests: polynomial-1 and exponential specifications."""
    n = len(log_price)
    y = log_price.values
    idx = log_price.index

    smt_poly1 = pd.Series(np.nan, index=idx)
    smt_exp = pd.Series(np.nan, index=idx)

    for t in range(SADF_MIN_SL, n):
        sup_poly = -np.inf
        sup_exp = -np.inf

        for t0 in range(0, t - SADF_MIN_SL + 1):
            seg = y[t0 : t + 1]
            length = len(seg)
            phi = 0.5

            t_vec = np.arange(length, dtype=np.float64)
            tstat_p = _ols_tstat(seg, t_vec)
            if not np.isnan(tstat_p):
                val = abs(tstat_p) / (length ** phi)
                if val > sup_poly:
                    sup_poly = val

            exp_vec = np.exp(t_vec / length)
            tstat_e = _ols_tstat(seg, exp_vec)
            if not np.isnan(tstat_e):
                val = abs(tstat_e) / (length ** phi)
                if val > sup_exp:
                    sup_exp = val

        smt_poly1.iloc[t] = sup_poly if sup_poly > -np.inf else np.nan
        smt_exp.iloc[t] = sup_exp if sup_exp > -np.inf else np.nan

    return smt_poly1, smt_exp


def _ols_tstat(y: np.ndarray, x: np.ndarray) -> float:
    """T-statistic for β in y = α + β*x + ε."""
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


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------
def _compute_shannon_entropy(
    log_returns: pd.Series, window: int = ENTROPY_WINDOW
) -> pd.Series:
    """Equal-width histogram Shannon entropy over a rolling window.

    Uses equal-width bins instead of quantile bins to avoid the
    degenerate case where clustered values collapse all quantile
    edges, producing constant zero entropy.
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
            # All values identical → zero information
            result.iloc[i] = 0.0
            continue

        try:
            edges = np.linspace(w_min - 1e-10, w_max + 1e-10, n_bins + 1)
            digitized = np.digitize(w, edges[1:-1])
            counts = np.bincount(digitized, minlength=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log2(probs))
        except (ValueError, IndexError):
            ent = np.nan
        result.iloc[i] = ent

    return result.reindex(log_returns.index)


# ---------------------------------------------------------------------------
# Rolling Lempel-Ziv complexity
# ---------------------------------------------------------------------------
def _compute_rolling_lz(
    log_returns: pd.Series, window: int = LZ_WINDOW
) -> pd.Series:
    """Binary-encoded Lempel-Ziv-76 complexity over a rolling window."""
    result = pd.Series(np.nan, index=log_returns.index)
    values = log_returns.values
    n = len(values)

    for i in range(window - 1, n):
        w = values[i - window + 1 : i + 1]
        if np.any(np.isnan(w)):
            continue
        binary_str = "".join("1" if r > 0 else "0" for r in w)
        c = _lempel_ziv_76(binary_str)
        norm = window / np.log2(window) if window > 1 else 1.0
        result.iloc[i] = c / norm

    return result.reindex(log_returns.index)


def _lempel_ziv_76(s: str) -> int:
    """Lempel-Ziv-76 complexity: count distinct sub-patterns."""
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


# ---------------------------------------------------------------------------
# Rolling Hurst exponent (R/S analysis)
# ---------------------------------------------------------------------------
def _compute_rolling_hurst(
    log_returns: pd.Series, window: int = HURST_WINDOW
) -> pd.Series:
    """Rescaled Range (R/S) Hurst exponent over a rolling window."""
    result = pd.Series(np.nan, index=log_returns.index)
    values = log_returns.values
    n = len(values)
    sub_periods = np.array([14, 30, 60, 90])  # crypto-calendar sub-periods
    sub_periods = sub_periods[sub_periods < window]

    for i in range(window - 1, n):
        w = values[i - window + 1 : i + 1]
        if np.any(np.isnan(w)):
            continue
        result.iloc[i] = _hurst_rs(w, sub_periods)

    return result.reindex(log_returns.index)


def _hurst_rs(data: np.ndarray, sub_periods: np.ndarray) -> float:
    """Estimate Hurst exponent from R/S statistics at multiple scales."""
    log_ns = []
    log_rs = []

    for sp in sub_periods:
        n_chunks = len(data) // sp
        if n_chunks < 1:
            continue
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


# =====================================================================
# Lag features (AR Logistic baseline — separate category)
# =====================================================================
def lag_column_names(lags: list[int] | None = None) -> list[str]:
    """Return the canonical column names for the lag features.

    Parameters
    ----------
    lags : list[int], optional
        Lag periods. Defaults to ``AR_LAGS``.

    Returns
    -------
    list[str]
        Column names in the same order as ``lags``.
    """
    if lags is None:
        lags = AR_LAGS
    return [f"{LAG_COLUMN_PREFIX}{k}" for k in lags]


def compute_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Precompute lagged log-return features on the full daily series.

    These columns are kept as a separate feature category because they
    are consumed only by the AR Logistic baseline. ML models (Logistic
    Regression, Random Forest, XGBoost, LSTM, KAN) do not see them and
    the multi-model MDA feature selection in ``preprocessing.py``
    excludes them by name prefix.

    Computing the lags here on the full daily series instead of inline
    inside the AR Logistic model removes a look-ahead artefact: the
    previous inline ``ffill().bfill()`` imputation at predict time
    backfilled NaN lags at the head of each test fold from later test
    observations. Precomputing on the global series gives every aligned
    event valid lookback values that respect chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a 'Close' column and DatetimeIndex.
    lags : list[int], optional
        Lag periods to compute. Defaults to ``AR_LAGS``.

    Returns
    -------
    pd.DataFrame
        One column per lag, named ``log_returns_lag{k}``. Indexed
        identically to ``df``. The first ``max(lags)`` rows contain
        NaN; downstream alignment with CUSUM events drops these because
        CUSUM filtering only fires after the volatility EWMA warmup,
        which exceeds ``max(AR_LAGS)``.
    """
    if lags is None:
        lags = AR_LAGS

    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    lag_df = pd.DataFrame(index=df.index)
    for k in lags:
        lag_df[f"{LAG_COLUMN_PREFIX}{k}"] = log_ret.shift(k)

    logger.info(
        "Lag features: %d columns, %d rows (lags=%s).",
        lag_df.shape[1], lag_df.shape[0], lags,
    )
    return lag_df


# =====================================================================
# Log transforms
# =====================================================================
def apply_sym_log(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply the symmetric log transform to listed columns.

    ``sign(x) * log(|x| + 1)`` — the sign-preserving variant of the
    natural log. Used for signed quantities whose raw values can be
    either positive or negative and whose magnitudes span many orders
    of magnitude (e.g. ``obv``, ``chaikin_osc`` reach 10⁷-10¹⁰
    during high-volume regimes).

    Properties:

    - preserves sign and zero (``sym_log(0) = 0``);
    - preserves rank ordering;
    - asymptotically log-like: for ``|x| ≫ 1`` it behaves like
      ``sign(x) · log(|x|)``, compressing the tails by orders of
      magnitude;
    - has a derivative discontinuity at zero (``+1`` from the right,
      ``-1`` from the left), which is acceptable in a feature-engineering
      context where the transformed values feed downstream MDA, FFD,
      and scaling steps rather than a gradient-based optimiser.

    An asinh-based variant (``np.arcsinh``) was tested in May 2026 as
    a smoother alternative; in sensitivity analysis it produced higher
    out-of-sample variance on the neural-network models and was reverted
    to the symmetric log transform here, which matches the convention
    used in earlier rounds of the pipeline.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix. A modified copy is returned; the original is
        not mutated.
    columns : list of str
        Column names to transform. Columns not present in ``features``
        are silently skipped, so the same target list can be reused
        across feature-set variants.

    Returns
    -------
    pd.DataFrame
        Copy of ``features`` with the symmetric log applied to the
        named columns.
    """
    features = features.copy()
    applied = []
    for col in columns:
        if col not in features.columns:
            continue
        features[col] = np.sign(features[col]) * np.log(features[col].abs() + 1)
        applied.append(col)
    if applied:
        logger.info("symmetric log applied to: %s", applied)
    return features


def apply_log(
    features: pd.DataFrame,
    columns: list[str],
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Apply ``log(|x| + eps)`` to listed columns.

    Standard variance-stabilising transform for strictly non-negative
    quantities whose magnitudes span many orders of magnitude (e.g.
    ATR, realised volatility, dollar volume). The small additive
    constant ``eps`` prevents ``log(0)`` for the rare exact-zero
    observation. ``|x|`` is taken so that any spurious negative input
    (which would not be expected for the target columns) does not
    crash the transform.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix. Returned a modified copy; original is not
        mutated.
    columns : list of str
        Column names to transform. Columns not present in
        ``features`` are silently skipped.
    eps : float, default 1e-8
        Additive constant to keep the transform finite at zero.

    Returns
    -------
    pd.DataFrame
        Copy of ``features`` with log applied to ``columns``.
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


# =====================================================================
# Orchestration
# =====================================================================
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Chain TA features and math features into a single DataFrame.

    Compression transforms (symmetric log for signed wide-range
    columns, log for unsigned wide-range columns) are NOT applied
    here. They are applied in the notebook via ``apply_sym_log``
    and ``apply_log`` so the choice of which columns get which
    transform is visible at the call site rather than hidden inside
    this function. Earlier iterations of this pipeline applied a
    combined ``apply_log_transforms`` (and briefly an asinh variant
    in ``apply_asinh``) here; both were removed in favour of explicit
    notebook-level transform application.

    External features and lag features are added separately in the
    notebook (they have their own loaders / fetchers and are kept in
    distinct categories).

    Parameters
    ----------
    df : pd.DataFrame
        Clean OHLCV DataFrame from data_loader.

    Returns
    -------
    pd.DataFrame
        TA + math feature columns covering every daily bar, with no
        compression transforms applied.
    """
    ta = compute_ta_features(df)
    math = compute_math_features(df)
    features = pd.concat([ta, math], axis=1)

    print(
        f"[features] {features.shape[1]} features, {features.shape[0]} rows | "
        f"NaN rows (any): {features.isna().any(axis=1).sum()}"
    )

    return features