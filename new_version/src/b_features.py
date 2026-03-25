"""
features.py — Feature engineering for the MFW Asset Direction Predictor.

This module contains **only** feature creation logic. It produces strictly
causal (backward-looking) features aligned with the MLDP (López de Prado)
quantitative pipeline.

Features flagged with `_REQUIRES_FFD` MUST undergo Fractional Differentiation
downstream before being passed into models.
"""

import logging
import re
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

# Explicit mapping requiring FFD handling sequentially.
_REQUIRES_FFD = {"raw_level"}

# Default subset of features to create autoregressive lags for.
DEFAULT_FEATURES_TO_LAG = ["RSI_14", "Realized_Vol_7d", "ROC_7d"]

HALVING_DATES = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

class FeatureBuilder:
    """Accumulates feature columns and their metadata tags."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.meta: Dict[str, str] = {}

    def add(self, col: str, series: pd.Series, tag: str) -> None:
        self.df[col] = series
        self.meta[col] = tag


def days_since_halving(index: pd.DatetimeIndex, halving_dates: pd.DatetimeIndex = HALVING_DATES) -> np.ndarray:
    if getattr(index, 'tz', None) is not None:
        halvings = halving_dates.tz_localize(index.tz)
    else:
        halvings = halving_dates

    days = np.empty(len(index))
    days[:] = np.nan
    for dt in halvings:
        mask = index >= dt
        days[mask] = (index[mask] - dt).days

    mask_before = index < halvings[0]
    if mask_before.any():
        days[mask_before] = (index[mask_before] - halvings[0]).days

    return days


# ═══════════════════════════════════════════════════════════════════════════
# Technical Analysis Features
# ═══════════════════════════════════════════════════════════════════════════

def create_ta_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Generate Technical Analysis features from OHLCV data."""
    missing = _RAW_OHLCV - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required OHLCV columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex for calendar feature extraction.")

    b = FeatureBuilder(df)

    # 1. Momentum & Oscillators (7d and 14d variants)
    b.add("RSI_7", ta.momentum.RSIIndicator(df["Close"], window=7).rsi(), "bounded_oscillator")
    b.add("RSI_14", ta.momentum.RSIIndicator(df["Close"], window=14).rsi(), "bounded_oscillator")

    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3
    )
    b.add("Stoch_K", stoch.stoch(), "bounded_oscillator")
    b.add("Stoch_D", stoch.stoch_signal(), "bounded_oscillator")

    b.add("Williams_R", ta.momentum.WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], lbp=14
    ).williams_r(), "bounded_oscillator")

    b.add("ROC_7d", (df["Close"] / df["Close"].shift(7).replace(0, np.nan) - 1) * 100, "zero_centered")
    b.add("ROC_14d", (df["Close"] / df["Close"].shift(14).replace(0, np.nan) - 1) * 100, "zero_centered")

    b.add("ADX_7", ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=7
    ).adx(), "bounded_oscillator")
    b.add("ADX_14", ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).adx(), "bounded_oscillator")

    # 2. Moving Averages — EMA + VWMA only (KAN Orthogonality)
    b.add("EMA_7", df["Close"].ewm(span=7, adjust=False).mean(), "raw_level")
    b.add("EMA_14", df["Close"].ewm(span=14, adjust=False).mean(), "raw_level")
    b.add("EMA_50", df["Close"].ewm(span=50, adjust=False).mean(), "raw_level")
    b.add("EMA_200", df["Close"].ewm(span=200, adjust=False).mean(), "raw_level")

    def _calc_vwma(close: pd.Series, vol: pd.Series, window: int) -> pd.Series:
        return (close * vol).rolling(window).sum() / vol.rolling(window).sum().replace(0, np.nan)

    b.add("VWMA_7", _calc_vwma(df["Close"], df["Volume"], 7), "raw_level")
    b.add("VWMA_14", _calc_vwma(df["Close"], df["Volume"], 14), "raw_level")
    b.add("VWMA_50", _calc_vwma(df["Close"], df["Volume"], 50), "raw_level")
    b.add("VWMA_200", _calc_vwma(df["Close"], df["Volume"], 200), "raw_level")

    # 3. Moving Average Ratios (Stationary)
    b.add("Price_to_EMA_7", df["Close"] / b.df["EMA_7"].replace(0, np.nan), "ratio")
    b.add("Price_to_EMA_14", df["Close"] / b.df["EMA_14"].replace(0, np.nan), "ratio")
    b.add("Price_to_EMA_50", df["Close"] / b.df["EMA_50"].replace(0, np.nan), "ratio")
    b.add("Price_to_EMA_200", df["Close"] / b.df["EMA_200"].replace(0, np.nan), "ratio")

    b.add("Price_to_VWMA_7", df["Close"] / b.df["VWMA_7"].replace(0, np.nan), "ratio")
    b.add("Price_to_VWMA_14", df["Close"] / b.df["VWMA_14"].replace(0, np.nan), "ratio")
    b.add("Price_to_VWMA_50", df["Close"] / b.df["VWMA_50"].replace(0, np.nan), "ratio")
    b.add("Price_to_VWMA_200", df["Close"] / b.df["VWMA_200"].replace(0, np.nan), "ratio")

    b.add("EMA7_EMA14_ratio", b.df["EMA_7"] / b.df["EMA_14"].replace(0, np.nan), "ratio")
    b.add("EMA50_EMA200_ratio", b.df["EMA_50"] / b.df["EMA_200"].replace(0, np.nan), "ratio")
    b.add("VWMA7_VWMA14_ratio", b.df["VWMA_7"] / b.df["VWMA_14"].replace(0, np.nan), "ratio")
    b.add("VWMA50_VWMA200_ratio", b.df["VWMA_50"] / b.df["VWMA_200"].replace(0, np.nan), "ratio")

    macd = ta.trend.MACD(close=df["Close"])
    b.add("MACD", macd.macd(), "zero_centered")
    b.add("Signal_Line", macd.macd_signal(), "zero_centered")

    b.add("OSCP", df["Close"].ewm(span=5, adjust=False).mean() - df["Close"].ewm(span=10, adjust=False).mean(), "zero_centered")

    # 4. Volatility & Volume
    b.add("ATR_7", ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=7
    ).average_true_range(), "raw_level")
    b.add("ATR_14", ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range(), "raw_level")

    if "Log_Return" not in df.columns:
        b.df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1).replace(0, np.nan))

    b.add("Realized_Vol_7d", b.df["Log_Return"].rolling(7).std(), "raw_level")
    b.add("Realized_Vol_14d", b.df["Log_Return"].rolling(14).std(), "raw_level")

    b.add("CCI", ta.trend.CCIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=20
    ).cci(), "zero_centered")

    b.add("OBV_pct", ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"]
    ).on_balance_volume().pct_change(), "zero_centered")

    b.add("MFI", ta.volume.MFIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14
    ).money_flow_index(), "bounded_oscillator")

    adl = ta.volume.AccDistIndexIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
    ).acc_dist_index()
    b.add("Chaikin_Oscillator", adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean(), "zero_centered")

    bbands = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    upper_band = bbands.bollinger_hband()
    lower_band = bbands.bollinger_lband()
    moving_average = bbands.bollinger_mavg()
    
    b.add("BB_Width", (upper_band - lower_band) / moving_average.replace(0, np.nan), "ratio")
    b.add("BB_Position", (df["Close"] - lower_band) / (upper_band - lower_band).replace(0, np.nan), "bounded_oscillator")

    # 5. Pivot Points
    pp = (df["High"] + df["Low"] + df["Close"]) / 3
    s1 = (pp * 2) - df["High"]
    s2 = pp - (df["High"] - df["Low"])
    r1 = (pp * 2) - df["Low"]
    r2 = pp + (df["High"] - df["Low"])

    b.add("Price_to_PP", df["Close"] / pp.replace(0, np.nan), "ratio")
    atr_safe = b.df["ATR_14"].replace(0, np.nan)
    b.add("Distance_to_S1", (df["Close"] - s1) / atr_safe, "zero_centered")
    b.add("Distance_to_S2", (df["Close"] - s2) / atr_safe, "zero_centered")
    b.add("Distance_to_R1", (df["Close"] - r1) / atr_safe, "zero_centered")
    b.add("Distance_to_R2", (df["Close"] - r2) / atr_safe, "zero_centered")

    # 6. Calendar Anomalies & Cycles
    b.add("DoW_sin", np.sin(2 * np.pi * df.index.dayofweek / 7), "cyclical")
    b.add("DoW_cos", np.cos(2 * np.pi * df.index.dayofweek / 7), "cyclical")
    b.add("Month_sin", np.sin(2 * np.pi * df.index.month / 12), "cyclical")
    b.add("Month_cos", np.cos(2 * np.pi * df.index.month / 12), "cyclical")
    b.add("Days_Since_Halving", days_since_halving(df.index), "event_time")

    # Safety Drop: Ensure raw OHLCV never leaks into metadata
    b.meta = {k: v for k, v in b.meta.items() if k not in _RAW_OHLCV}

    VALID_TAGS = {"raw_level", "ratio", "bounded_oscillator", "zero_centered", "cyclical", "event_time"}
    assert all(tag in VALID_TAGS for tag in b.meta.values()), f"Invalid metadata tag found: {set(b.meta.values()) - VALID_TAGS}"
    assert len(b.meta) == len(set(b.meta)), "Duplicate feature names detected in metadata"
    nan_only = [c for c in b.meta if b.df[c].isna().all()]
    if nan_only:
        logger.warning("Features with all-NaN values detected: %s", nan_only)

    logger.info("TA features created: %d", len(b.meta))
    return b.df, b.meta


# ═══════════════════════════════════════════════════════════════════════════
# On-Chain Engineered Features
# ═══════════════════════════════════════════════════════════════════════════

def create_onchain_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Generate engineered on-chain features from CoinMetrics / Blockchain.com data."""
    b = FeatureBuilder(df)

    if "FlowInExNtv" in df.columns and "FlowOutExNtv" in df.columns:
        b.add("Net_Exchange_Flow", df["FlowInExNtv"] - df["FlowOutExNtv"], "zero_centered")
        b.add("Flow_Ratio", df["FlowInExNtv"] / df["FlowOutExNtv"].replace(0, np.nan), "ratio")

    if "AdrActCnt" in df.columns:
        b.add("AdrAct_ROC_7d", (df["AdrActCnt"] / df["AdrActCnt"].shift(7).replace(0, np.nan) - 1) * 100, "zero_centered")
        if "TxTfrCnt" in df.columns:
            b.add("TxTfr_per_Active_Adr", df["TxTfrCnt"] / df["AdrActCnt"].replace(0, np.nan), "ratio")

    if "IssTotUSD" in df.columns and "Volume" in df.columns:
        b.add("Miner_Sell_Pressure", df["IssTotUSD"] / df["Volume"].replace(0, np.nan), "ratio")

    if "CapMVRVCur" in df.columns:
        b.add("MVRV_Momentum", df["CapMVRVCur"] - df["CapMVRVCur"].rolling(7).mean(), "zero_centered")

    if "HashRate" in df.columns:
        b.add("HashRate_ROC_30d", (df["HashRate"] / df["HashRate"].shift(30).replace(0, np.nan) - 1) * 100, "zero_centered")

    if "NVTAdj" in df.columns:
        b.add("NVTAdj", df["NVTAdj"], "raw_level")

    VALID_TAGS = {"raw_level", "ratio", "bounded_oscillator", "zero_centered", "cyclical", "event_time"}
    assert all(tag in VALID_TAGS for tag in b.meta.values()), f"Invalid metadata tag found: {set(b.meta.values()) - VALID_TAGS}"
    assert len(b.meta) == len(set(b.meta)), "Duplicate feature names detected in metadata"
    nan_only = [c for c in b.meta if b.df[c].isna().all()]
    if nan_only:
        logger.warning("Features with all-NaN values detected: %s", nan_only)

    logger.info("On-chain features created: %d", len(b.meta))
    return b.df, b.meta


# ═══════════════════════════════════════════════════════════════════════════
# Multicollinearity Filter
# ═══════════════════════════════════════════════════════════════════════════

def filter_correlated_features(
    df: pd.DataFrame, 
    meta: Dict[str, str], 
    threshold: float = 0.95
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Drop highly correlated features (keeping the one with the longer lookback)."""
    df_filtered = df.copy()
    meta_filtered = meta.copy()
    
    feature_cols = list(meta.keys())
    corr_matrix = df[feature_cols].corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()

    def extract_window(name: str) -> int:
        nums = re.findall(r'\d+', name)
        return int(nums[-1]) if nums else 0

    for col in upper.columns:
        highly_correlated = upper.index[upper[col] > threshold].tolist()
        
        for corr_col in highly_correlated:
            if col in to_drop or corr_col in to_drop:
                continue
                
            win_col = extract_window(col)
            win_corr = extract_window(corr_col)
            
            if win_col < win_corr:
                to_drop.add(col)
            elif win_col > win_corr:
                to_drop.add(corr_col)
            else:
                to_drop.add(col) 

    if to_drop:
        logger.info("Dropping %d highly correlated features (>%s): %s", len(to_drop), threshold, to_drop)
        df_filtered = df_filtered.drop(columns=list(to_drop))
        for col in to_drop:
            if col in meta_filtered:
                del meta_filtered[col]
                
    return df_filtered, meta_filtered


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
    """Create shifted (lagged) columns for autoregressive modelling."""
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
    include_onchain: bool = True,
    drop_correlated: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Run all feature engineering and build the metadata dictionary.

    The returned DataFrame retains raw OHLCV columns (``Open, High, Low,
    Close, Volume``) because downstream ``4_labels.py`` needs ``Close``
    for Triple-Barrier Labeling.  However, those raw columns are
    **excluded** from the returned ``feature_metadata`` dict so the model
    only trains on explicitly engineered and tagged features.
    
    Features with tags in `_REQUIRES_FFD` (such as `raw_level`) MUST 
    undergo Fractional Differentiation (FFD) before being fed to a model.
    """
    feature_metadata: Dict[str, str] = {}

    if include_ta:
        df, ta_meta = create_ta_features(df)
        feature_metadata.update(ta_meta)

    if include_onchain:
        df, oc_meta = create_onchain_features(df)
        feature_metadata.update(oc_meta)

    if drop_correlated:
        df, feature_metadata = filter_correlated_features(df, feature_metadata, threshold=0.95)

    # Final safety net: ensure raw OHLCV never leaks into metadata
    feature_metadata = {k: v for k, v in feature_metadata.items() if k not in _RAW_OHLCV}

    # Universal Lag Rule (skill_mldp_pipeline.md):
    # Shift all rolling/expanding predictor features by 1 to prevent look-ahead bias,
    # except 'cyclical' and 'event_time' tagged features which are exempt.
    cols_to_shift = [col for col, tag in feature_metadata.items() if tag not in _SHIFT_EXEMPT_TAGS]
    if cols_to_shift:
        df[cols_to_shift] = df[cols_to_shift].shift(1)
        logger.info("Universal Lag Rule: Applied .shift(1) to %d features", len(cols_to_shift))

    raw_level_count = sum(1 for tag in feature_metadata.values() if tag in _REQUIRES_FFD)
    if raw_level_count > 0:
        logger.warning(
            "Feature builder: %d 'raw_level' features generated. "
            "These strictly require Fractional Differentiation (FFD) before modeling.", 
            raw_level_count
        )

    logger.info("Total engineered features tracked in metadata: %d", len(feature_metadata))
    return df, feature_metadata
