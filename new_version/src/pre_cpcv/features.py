"""
4) Features
============================
Compute all features from the full OHLCV DataFrame. Returns a feature
matrix covering every daily bar. Feature selection by row (restricting
to labeled events) happens later in alignment.

Implements technical analysis features (Step 6a), AFML Part 4
mathematical features (Step 6b), and log transforms (Step 7).
"""

import logging
import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

# ── TA parameters ─────────────────────────────────────────────────────
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
ATR_PERIOD = 14
ROLLING_WINDOW = 21

# ── Mathematical feature parameters ──────────────────────────────────
SADF_MIN_SL = 63        # minimum sample length (~1 quarter)
SADF_LAGS = 1
ENTROPY_WINDOW = 21
LZ_WINDOW = 63
HURST_WINDOW = 126

# ── Log transform targets ────────────────────────────────────────────
LOG_TRANSFORM_COLUMNS = ["atr", "obv"]

# ── Cache ─────────────────────────────────────────────────────────────
CACHE_DIR = "cache/"
MATH_CACHE_FILE = "math_features.parquet"


# =====================================================================
# Step 6a — Technical Analysis Features
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

    # 7. Garman-Klass volatility (rolling)
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_daily = 0.5 * log_hl ** 2 - (2.0 * np.log(2) - 1.0) * log_co ** 2
    features["gk_vol"] = gk_daily.rolling(ROLLING_WINDOW).mean()

    # 8. Rolling realized volatility (annualized)
    features["realized_vol"] = log_returns.rolling(ROLLING_WINDOW).std() * np.sqrt(365)

    # 9. Rolling skewness
    features["skewness"] = log_returns.rolling(ROLLING_WINDOW).skew()

    # 10. Rolling kurtosis
    features["kurtosis"] = log_returns.rolling(ROLLING_WINDOW).kurt()

    logger.info("TA features: %d columns, %d rows.", features.shape[1], features.shape[0])
    return features


# =====================================================================
# Step 6b — Mathematical Features (AFML Part 4)
# =====================================================================
def compute_math_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AFML mathematical features with Parquet caching.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Columns: sadf, smt_poly1, smt_exp, entropy, lz_complexity, hurst.
    """
    cache_path = os.path.join(CACHE_DIR, MATH_CACHE_FILE)

    # check cache
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if (cached.index[0] == df.index[0]) and (cached.index[-1] == df.index[-1]):
            logger.info("Math features loaded from cache.")
            return cached
        logger.info("Cache date range mismatch, recomputing.")

    close = df["Close"]
    log_price = np.log(close)
    log_returns = np.log(close / close.shift(1)).dropna()

    # compute each feature
    print("[features] Computing SADF (this may take a few minutes)...")
    sadf = _compute_sadf(log_price)

    print("[features] Computing SMT...")
    smt_poly1, smt_exp = _compute_smt(log_price)

    print("[features] Computing rolling entropy...")
    entropy = _compute_rolling_entropy(log_returns, window=ENTROPY_WINDOW)

    print("[features] Computing Lempel-Ziv complexity...")
    lz = _compute_rolling_lz(log_returns, window=LZ_WINDOW)

    print("[features] Computing Hurst exponent...")
    hurst = _compute_rolling_hurst(log_returns, window=HURST_WINDOW)

    # assemble
    features = pd.DataFrame(index=df.index)
    features["sadf"] = sadf
    features["smt_poly1"] = smt_poly1
    features["smt_exp"] = smt_exp
    features["entropy"] = entropy
    features["lz_complexity"] = lz
    features["hurst"] = hurst

    # save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    features.to_parquet(cache_path)
    logger.info("Math features computed and cached to %s.", cache_path)

    return features


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
        # backward-expanding start points
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

    # build regressor matrix: [const, y_{t-1}, Δy_{t-1}, ..., Δy_{t-lags}]
    # trim to align everything
    start = lags
    dep = dy[start:]  # Δy_t
    m = len(dep)

    if m <= lags + 2:
        return np.nan

    regressors = np.ones((m, 2 + lags))
    regressors[:, 1] = y[start : start + m]  # y_{t-1}
    for k in range(1, lags + 1):
        regressors[:, 1 + k] = dy[start - k : start - k + m]  # Δy_{t-k}

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

            # SM-Poly1: log[y] = α + β*t + ε
            t_vec = np.arange(length, dtype=np.float64)
            tstat_p = _ols_tstat(seg, t_vec)
            if not np.isnan(tstat_p):
                val = abs(tstat_p) / (length ** phi)
                if val > sup_poly:
                    sup_poly = val

            # SM-Exp: log[y] = α + β*exp(t/length) + ε
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
# Rolling Shannon entropy
# ---------------------------------------------------------------------------
def _compute_rolling_entropy(
    log_returns: pd.Series, window: int = ENTROPY_WINDOW
) -> pd.Series:
    """Quantile-encoded Shannon entropy over a rolling window."""
    result = pd.Series(np.nan, index=log_returns.index)
    values = log_returns.values
    n = len(values)
    n_bins = 5

    for i in range(window - 1, n):
        w = values[i - window + 1 : i + 1]
        if np.any(np.isnan(w)):
            continue
        # quantile-encode into 5 bins based on the window itself
        try:
            edges = np.quantile(w, np.linspace(0, 1, n_bins + 1))
            edges[0] -= 1e-10
            edges[-1] += 1e-10
            digitized = np.digitize(w, edges[1:-1])
            counts = np.bincount(digitized, minlength=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log2(probs))
        except (ValueError, IndexError):
            ent = np.nan
        result.iloc[i] = ent

    # reindex to full OHLCV index (log_returns is one row shorter)
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
        # normalize
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
        # check if s[i+1 : i+k+1] exists in s[0 : i+k]
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
    sub_periods = np.array([10, 21, 42, 63])
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

    # slope of log(R/S) vs log(n)
    log_ns = np.array(log_ns)
    log_rs = np.array(log_rs)
    slope, _ = np.polyfit(log_ns, log_rs, 1)
    return slope


# =====================================================================
# Step 7 — Log transforms
# =====================================================================
def apply_log_transforms(features: pd.DataFrame) -> pd.DataFrame:
    """Apply log transforms to scale-compress specified columns.

    - ATR: ``log(|x| + 1e-8)`` (always positive)
    - OBV: ``sign(x) * log(|x| + 1)`` (preserves sign)

    Returns a modified copy of *features*.
    """
    features = features.copy()

    for col in LOG_TRANSFORM_COLUMNS:
        if col not in features.columns:
            continue
        if col == "obv":
            features[col] = np.sign(features[col]) * np.log(features[col].abs() + 1)
        else:
            features[col] = np.log(features[col].abs() + 1e-8)

    logger.info("Log-transformed columns: %s", LOG_TRANSFORM_COLUMNS)
    return features


# =====================================================================
# Orchestration
# =====================================================================
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Chain TA features, math features, and log transforms.

    Parameters
    ----------
    df : pd.DataFrame
        Clean OHLCV DataFrame from data_loader.

    Returns
    -------
    pd.DataFrame
        ~20+ feature columns covering every daily bar.
    """
    ta = compute_ta_features(df)
    math = compute_math_features(df)
    features = pd.concat([ta, math], axis=1)
    features = apply_log_transforms(features)

    print(
        f"[features] {features.shape[1]} features, {features.shape[0]} rows | "
        f"NaN rows (any): {features.isna().any(axis=1).sum()}"
    )

    return features