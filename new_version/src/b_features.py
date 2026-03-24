# features.py — Feature engineering for the MFW Asset Direction Predictor.
#
# This module contains **only** feature creation logic. It produces strictly
# causal (backward-looking) features aligned with the MLDP (López de Prado)
# quantitative pipeline.
#
# It also provides a Metadata Tagging system (feature_metadata dictionary)
# that classifies every engineered feature into a strict statistical type
# ('raw_level', 'ratio', 'bounded_oscillator', 'zero_centered', 'cyclical',
# 'event_time') so downstream econometric transformations know exactly how
# to handle them.

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

# Raw OHLCV columns — kept in the DataFrame for downstream label creation,
# but NEVER included in the returned feature metadata dictionary.
_RAW_OHLCV = {"Open", "High", "Low", "Close", "Volume"}

# Tags exempt from the universal .shift(1) lag rule:
# - 'cyclical': perfectly predictable calendar features
# - 'event_time': deterministic event counters (e.g. Days_Since_Halving)
_SHIFT_EXEMPT_TAGS = {"cyclical", "event_time"}

# Default subset of features to create autoregressive lags for.
# Chosen to avoid the combinatorial explosion of lagging the entire feature set.
DEFAULT_FEATURES_TO_LAG = ["RSI_14", "Realized_Vol_7d", "ROC_7d"]


# ═══════════════════════════════════════════════════════════════════════════
# Technical Analysis Features
# ═══════════════════════════════════════════════════════════════════════════
def create_ta_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Generate Technical Analysis features from OHLCV data.

    All indicators use **causal** (backward-looking) rolling windows.
    Outputs a dictionary mapping each new column name to its MLDP
    statistical metadata tag.

    KAN Orthogonality: Only EMA (time-based trends) and VWMA (volume
    conviction) are retained as base moving averages. SMA and WMA are
    dropped to eliminate substitution effects that dilute MDI importance.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``Open, High, Low, Close, Volume``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        The augmented DataFrame and the feature metadata dictionary.
    """
    df = df.copy()
    meta: Dict[str, str] = {}

    def _add(col: str, series: pd.Series, tag: str) -> None:
        """Helper to append a column and register its metadata tag."""
        df[col] = series
        meta[col] = tag

    # ------------------------------------------------------------------
    # 1. Momentum & Oscillators (7d and 14d variants)
    # ------------------------------------------------------------------
    _add("RSI_7", ta.momentum.RSIIndicator(df["Close"], window=7).rsi(), "bounded_oscillator")
    _add("RSI_14", ta.momentum.RSIIndicator(df["Close"], window=14).rsi(), "bounded_oscillator")

    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3
    )
    _add("Stoch_K", stoch.stoch(), "bounded_oscillator")
    _add("Stoch_D", stoch.stoch_signal(), "bounded_oscillator")

    _add("Williams_R", ta.momentum.WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], lbp=14
    ).williams_r(), "bounded_oscillator")

    # ROC centers around zero in expectation → zero_centered
    _add("ROC_7d", (df["Close"] / df["Close"].shift(7) - 1) * 100, "zero_centered")
    _add("ROC_14d", (df["Close"] / df["Close"].shift(14) - 1) * 100, "zero_centered")

    _add("ADX_7", ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=7
    ).adx(), "bounded_oscillator")
    _add("ADX_14", ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).adx(), "bounded_oscillator")

    # ------------------------------------------------------------------
    # 2. Moving Averages — EMA + VWMA only (KAN Orthogonality)
    # ------------------------------------------------------------------
    # Exponential Moving Averages
    _add("EMA_7", df["Close"].ewm(span=7, adjust=False).mean(), "raw_level")
    _add("EMA_14", df["Close"].ewm(span=14, adjust=False).mean(), "raw_level")
    _add("EMA_50", df["Close"].ewm(span=50, adjust=False).mean(), "raw_level")
    _add("EMA_200", df["Close"].ewm(span=200, adjust=False).mean(), "raw_level")

    # Volume Weighted Moving Averages
    def _calc_vwma(close: pd.Series, vol: pd.Series, window: int) -> pd.Series:
        return (close * vol).rolling(window).sum() / vol.rolling(window).sum()

    _add("VWMA_7", _calc_vwma(df["Close"], df["Volume"], 7), "raw_level")
    _add("VWMA_14", _calc_vwma(df["Close"], df["Volume"], 14), "raw_level")
    _add("VWMA_50", _calc_vwma(df["Close"], df["Volume"], 50), "raw_level")
    _add("VWMA_200", _calc_vwma(df["Close"], df["Volume"], 200), "raw_level")

    # ------------------------------------------------------------------
    # 3. Moving Average Ratios (Stationary)
    # ------------------------------------------------------------------
    # Price-to-MA ratios (EMA only)
    _add("Price_to_EMA_7", df["Close"] / df["EMA_7"], "ratio")
    _add("Price_to_EMA_14", df["Close"] / df["EMA_14"], "ratio")
    _add("Price_to_EMA_50", df["Close"] / df["EMA_50"], "ratio")
    _add("Price_to_EMA_200", df["Close"] / df["EMA_200"], "ratio")

    # Price-to-VWMA ratios
    _add("Price_to_VWMA_7", df["Close"] / df["VWMA_7"], "ratio")
    _add("Price_to_VWMA_14", df["Close"] / df["VWMA_14"], "ratio")
    _add("Price_to_VWMA_50", df["Close"] / df["VWMA_50"], "ratio")
    _add("Price_to_VWMA_200", df["Close"] / df["VWMA_200"], "ratio")

    # Cross ratios — trend momentum (EMA and VWMA only)
    _add("EMA7_EMA14_ratio", df["EMA_7"] / df["EMA_14"], "ratio")
    _add("EMA50_EMA200_ratio", df["EMA_50"] / df["EMA_200"], "ratio")
    _add("VWMA7_VWMA14_ratio", df["VWMA_7"] / df["VWMA_14"], "ratio")
    _add("VWMA50_VWMA200_ratio", df["VWMA_50"] / df["VWMA_200"], "ratio")

    # MACD (26/12/9 standard) — MACD_Hist DROPPED to avoid perfect
    # linear dependency (Hist = MACD - Signal) that corrupts KAN extraction.
    macd = ta.trend.MACD(close=df["Close"])
    _add("MACD", macd.macd(), "zero_centered")
    _add("Signal_Line", macd.macd_signal(), "zero_centered")

    # OSCP (fast–slow MA spread)
    _add("OSCP", df["Close"].rolling(5).mean() - df["Close"].rolling(10).mean(), "zero_centered")

    # ------------------------------------------------------------------
    # 4. Volatility & Volume
    # ------------------------------------------------------------------
    _add("ATR_7", ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=7
    ).average_true_range(), "raw_level")
    _add("ATR_14", ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range(), "raw_level")

    # Log_Return guard: ensure Realized_Vol never silently fails
    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    _add("Realized_Vol_7d", df["Log_Return"].rolling(7).std(), "raw_level")
    _add("Realized_Vol_14d", df["Log_Return"].rolling(14).std(), "raw_level")

    _add("CCI", ta.trend.CCIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=20
    ).cci(), "zero_centered")

    # OBV_pct centers around zero in expectation → zero_centered
    _add("OBV_pct", ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"]
    ).on_balance_volume().pct_change(), "zero_centered")

    _add("MFI", ta.volume.MFIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14
    ).money_flow_index(), "bounded_oscillator")

    adl = ta.volume.AccDistIndexIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
    ).acc_dist_index()
    _add("Chaikin_Oscillator", adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean(), "zero_centered")

    # Bollinger Bands (normalized width and position)
    bbands = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    upper_band = bbands.bollinger_hband()
    lower_band = bbands.bollinger_lband()
    moving_average = bbands.bollinger_mavg()
    # BB_Width is strictly positive → ratio
    _add("BB_Width", (upper_band - lower_band) / moving_average, "ratio")
    _add("BB_Position", (df["Close"] - lower_band) / (upper_band - lower_band), "bounded_oscillator")

    # ------------------------------------------------------------------
    # 5. Pivot Points — Normalized to Ratios (Scale Invariance)
    # ------------------------------------------------------------------
    pp = (df["High"] + df["Low"] + df["Close"]) / 3
    s1 = (pp * 2) - df["High"]
    s2 = pp - (df["High"] - df["Low"])
    r1 = (pp * 2) - df["Low"]
    r2 = pp + (df["High"] - df["Low"])

    _add("Price_to_PP", df["Close"] / pp, "ratio")
    # Distance ratios normalized by ATR_14 for scale invariance
    atr_safe = df["ATR_14"].replace(0, np.nan)
    _add("Distance_to_S1", (df["Close"] - s1) / atr_safe, "zero_centered")
    _add("Distance_to_S2", (df["Close"] - s2) / atr_safe, "zero_centered")
    _add("Distance_to_R1", (df["Close"] - r1) / atr_safe, "zero_centered")
    _add("Distance_to_R2", (df["Close"] - r2) / atr_safe, "zero_centered")

    # ------------------------------------------------------------------
    # 6. Calendar Anomalies & Cycles
    # ------------------------------------------------------------------
    _add("DoW_sin", np.sin(2 * np.pi * df.index.dayofweek / 7), "cyclical")
    _add("DoW_cos", np.cos(2 * np.pi * df.index.dayofweek / 7), "cyclical")
    _add("Month_sin", np.sin(2 * np.pi * df.index.month / 12), "cyclical")
    _add("Month_cos", np.cos(2 * np.pi * df.index.month / 12), "cyclical")

    HALVING_DATES = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])

    def days_since_halving(index):
        if getattr(index, 'tz', None) is not None:
            halvings = HALVING_DATES.tz_localize(index.tz)
        else:
            halvings = HALVING_DATES

        days = np.empty(len(index))
        days[:] = np.nan
        for dt in halvings:
            mask = index >= dt
            days[mask] = (index[mask] - dt).days

        mask_before = index < halvings[0]
        if mask_before.any():
            days[mask_before] = (index[mask_before] - halvings[0]).days

        return days

    # event_time: deterministic counter, exempt from .shift(1)
    _add("Days_Since_Halving", days_since_halving(df.index), "event_time")

    # Safety Drop: Ensure raw OHLCV never leaks into metadata
    meta = {k: v for k, v in meta.items() if k not in _RAW_OHLCV}

    logger.info("TA features created: %d", len(meta))
    return df, meta


# ═══════════════════════════════════════════════════════════════════════════
# On-Chain Engineered Features
# ═══════════════════════════════════════════════════════════════════════════
def create_onchain_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Generate engineered on-chain features from CoinMetrics / Blockchain.com data.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        The augmented DataFrame and the feature metadata dictionary.
    """
    df = df.copy()
    meta: Dict[str, str] = {}

    def _add(col: str, series: pd.Series, tag: str) -> None:
        df[col] = series
        meta[col] = tag

    if "FlowInExNtv" in df.columns and "FlowOutExNtv" in df.columns:
        _add("Net_Exchange_Flow", df["FlowInExNtv"] - df["FlowOutExNtv"], "zero_centered")
        _add("Flow_Ratio", df["FlowInExNtv"] / df["FlowOutExNtv"].replace(0, np.nan), "ratio")

    if "AdrActCnt" in df.columns:
        _add("AdrAct_ROC_7d", df["AdrActCnt"].pct_change(7) * 100, "zero_centered")
        if "TxTfrCnt" in df.columns:
            _add("TxTfr_per_Active_Adr", df["TxTfrCnt"] / df["AdrActCnt"].replace(0, np.nan), "ratio")

    if "IssTotUSD" in df.columns and "volume_reported_spot_usd_1d" in df.columns:
        _add("Miner_Sell_Pressure", df["IssTotUSD"] / df["volume_reported_spot_usd_1d"].replace(0, np.nan), "ratio")

    if "CapMVRVCur" in df.columns:
        _add("MVRV_Momentum", df["CapMVRVCur"] - df["CapMVRVCur"].rolling(7).mean(), "zero_centered")

    # Hash Rate momentum (if available from CoinMetrics)
    if "HashRate" in df.columns:
        _add("HashRate_ROC_30d", df["HashRate"].pct_change(30) * 100, "zero_centered")

    # NVT Adjusted ratio (if available from CoinMetrics)
    if "NVTAdj" in df.columns:
        _add("NVTAdj", df["NVTAdj"], "raw_level")

    logger.info("On-chain features created: %d", len(meta))
    return df, meta


# ═══════════════════════════════════════════════════════════════════════════
# NaN Cleanup
# ═══════════════════════════════════════════════════════════════════════════
def drop_warmup_nans(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Drop rows where any *feature_cols* column is NaN."""
    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    logger.info(
        "Dropped %d warm-up rows (%.1f%%). Remaining: %d",
        dropped, 100 * dropped / max(before, 1), len(df),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Autoregressive Lag Features (Restricted Subset)
# ═══════════════════════════════════════════════════════════════════════════
def create_lagged_features(
    df: pd.DataFrame,
    base_metadata: Dict[str, str],
    lags: int = 3,
    drop_na: bool = True,
    features_to_lag: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Create shifted (lagged) columns for autoregressive modelling.

    To prevent feature explosion (270+ cols), only a restricted subset
    of features is lagged by default. Override with ``features_to_lag``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with the features to lag.
    base_metadata : dict of [str, str]
        Metadata dictionary containing the base features and their tags.
    lags : int
        Number of lags to create (default 3 → t-1, t-2, t-3).
    drop_na : bool
        If True, drop rows with NaNs introduced by shifting.
    features_to_lag : list of str, optional
        Explicit list of feature names to lag. If ``None``, defaults to
        ``DEFAULT_FEATURES_TO_LAG``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        - DataFrame with new lag columns appended.
        - Dictionary of ONLY the newly created lag metadata tags.
    """
    df = df.copy()
    lag_meta: Dict[str, str] = {}

    if features_to_lag is None:
        features_to_lag = DEFAULT_FEATURES_TO_LAG

    for feat in features_to_lag:
        if feat not in df.columns or feat not in base_metadata:
            continue
        tag = base_metadata[feat]
        for lag in range(1, lags + 1):
            col_name = f"{feat}_t-{lag}"
            df[col_name] = df[feat].shift(lag)
            lag_meta[col_name] = tag

    if drop_na and lag_meta:
        before = len(df)
        df = df.dropna(subset=list(lag_meta.keys()))
        logger.info(
            "Lagged features: %d cols created, %d warm-up rows dropped",
            len(lag_meta), before - len(df),
        )

    return df, lag_meta


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def create_all_features(
    df: pd.DataFrame,
    include_ta: bool = True,
    include_onchain: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Run all feature engineering and build the metadata dictionary.

    The returned DataFrame retains raw OHLCV columns (``Open, High, Low,
    Close, Volume``) because downstream ``4_labels.py`` needs ``Close``
    for Triple-Barrier Labeling.  However, those raw columns are
    **excluded** from the returned ``feature_metadata`` dict so the model
    only trains on explicitly engineered and tagged features.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``data_loader.load_dataset``.
    include_ta : bool
        Whether to create Technical Analysis features (default True).
    include_onchain : bool
        Whether to create on-chain engineered features (default True).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        The fully augmented DataFrame, and the ``feature_metadata`` dictionary
        mapping column names to MLDP statistical types:
        ``('raw_level', 'ratio', 'bounded_oscillator', 'zero_centered',
        'cyclical', 'event_time')``.
    """
    feature_metadata: Dict[str, str] = {}

    if include_ta:
        df, ta_meta = create_ta_features(df)
        feature_metadata.update(ta_meta)

    if include_onchain:
        df, oc_meta = create_onchain_features(df)
        feature_metadata.update(oc_meta)

    # Final safety net: ensure raw OHLCV never leaks into metadata
    feature_metadata = {k: v for k, v in feature_metadata.items() if k not in _RAW_OHLCV}

    # Universal Lag Rule (skill_mldp_pipeline.md):
    # Shift all rolling/expanding predictor features by 1 to prevent look-ahead bias,
    # except 'cyclical' and 'event_time' tagged features which are exempt.
    cols_to_shift = [col for col, tag in feature_metadata.items() if tag not in _SHIFT_EXEMPT_TAGS]
    if cols_to_shift:
        df[cols_to_shift] = df[cols_to_shift].shift(1)
        logger.info("Universal Lag Rule: Applied .shift(1) to %d features", len(cols_to_shift))

    logger.info("Total engineered features tracked in metadata: %d", len(feature_metadata))
    return df, feature_metadata
