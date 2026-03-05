"""
features.py — Feature engineering for the MFW Asset Direction Predictor.

This module contains **only** feature creation logic.  It does NOT perform:
  - Train/test splitting  (→ preprocessing.py)
  - Scaling / normalising  (→ preprocessing.py)
  - Stationarity tests     (→ econometrics.py)

Follows:
  financial_data.md  §2  — Log_Return already exists from data_loader.py
  econometrics.md    §4  — Type hints, no matplotlib/seaborn imports
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Technical Analysis Features (28 indicators)
# ═══════════════════════════════════════════════════════════════════════════
def create_ta_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Generate Technical Analysis features from OHLCV data.

    All indicators use **causal** (backward-looking) rolling windows,
    so no look-ahead bias is introduced.  Leading NaN values from
    warm-up periods are left in place — call :func:`drop_warmup_nans`
    after all feature creation is complete.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``Open, High, Low, Close, Volume``.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The augmented DataFrame and a list of newly created column names.
    """
    df = df.copy()
    initial_columns = set(df.columns)

    # ------------------------------------------------------------------
    # 1. Momentum & Oscillators
    # ------------------------------------------------------------------
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"],
        window=14, smooth_window=3,
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    df["Williams_R"] = ta.momentum.WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], lbp=14,
    ).williams_r()

    df["ROC"] = ta.momentum.ROCIndicator(
        close=df["Close"], window=12,
    ).roc()

    df["Momentum"] = df["Close"] - df["Close"].shift(4)

    # ------------------------------------------------------------------
    # 2. Moving Averages & Trend
    # ------------------------------------------------------------------
    df["EMA_14"] = ta.trend.EMAIndicator(
        close=df["Close"], window=14,
    ).ema_indicator()

    weights = np.arange(1, 15)
    df["WMA_14"] = (
        df["Close"]
        .rolling(14)
        .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    )

    df["Disparity_5"] = (df["Close"] / df["Close"].rolling(5).mean()) * 100
    df["Disparity_14"] = (df["Close"] / df["Close"].rolling(14).mean()) * 100

    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["Signal_Line"] = macd.macd_signal()

    df["OSCP"] = df["Close"].rolling(5).mean() - df["Close"].rolling(10).mean()

    # ------------------------------------------------------------------
    # 3. Volatility & Volume
    # ------------------------------------------------------------------
    df["RSI"] = ta.momentum.RSIIndicator(
        close=df["Close"], window=14,
    ).rsi()

    df["CCI"] = ta.trend.CCIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=20,
    ).cci()

    df["ATR"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14,
    ).average_true_range()

    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"],
    ).on_balance_volume()

    df["MFI"] = ta.volume.MFIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"],
        volume=df["Volume"], window=14,
    ).money_flow_index()

    adl = ta.volume.AccDistIndexIndicator(
        high=df["High"], low=df["Low"], close=df["Close"],
        volume=df["Volume"],
    ).acc_dist_index()
    df["Chaikin_Oscillator"] = (
        adl.ewm(span=3, adjust=False).mean()
        - adl.ewm(span=10, adjust=False).mean()
    )

    bbands = ta.volatility.BollingerBands(
        close=df["Close"], window=20, window_dev=2,
    )
    df["Upper_Band"] = bbands.bollinger_hband()
    df["Lower_Band"] = bbands.bollinger_lband()

    # ------------------------------------------------------------------
    # 4. Pivot Points (Support & Resistance)
    # ------------------------------------------------------------------
    df["PP"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["S1"] = (df["PP"] * 2) - df["High"]
    df["S2"] = df["PP"] - (df["High"] - df["Low"])
    df["R1"] = (df["PP"] * 2) - df["Low"]
    df["R2"] = df["PP"] + (df["High"] - df["Low"])

    # ------------------------------------------------------------------
    # 5. Calendar Anomalies
    # ------------------------------------------------------------------
    df["Day_of_Week"] = df.index.dayofweek
    df["Week_of_Month"] = (df.index.day - 1) // 7 + 1

    # ------------------------------------------------------------------
    # Track newly added columns
    # ------------------------------------------------------------------
    ta_features = [c for c in df.columns if c not in initial_columns]
    logger.info("TA features created: %d", len(ta_features))

    return df, ta_features


# ═══════════════════════════════════════════════════════════════════════════
# On-Chain Engineered Features (up to 6, conditional on available columns)
# ═══════════════════════════════════════════════════════════════════════════
def create_onchain_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Generate engineered on-chain features from CoinMetrics / Blockchain.com data.

    Each feature is created **only** if its required source columns are
    present in *df*, so the function works regardless of which on-chain
    provider was selected in ``data_loader.py``.

    Leading NaN values from rolling windows or ``pct_change()`` are left
    in place — call :func:`drop_warmup_nans` afterwards.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``data_loader.load_dataset`` (may or may not contain
        on-chain columns).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The augmented DataFrame and a list of newly created column names.
    """
    df = df.copy()
    initial_columns = set(df.columns)

    # --- Exchange flow ---
    if "FlowInExNtv" in df.columns and "FlowOutExNtv" in df.columns:
        df["Net_Exchange_Flow"] = df["FlowInExNtv"] - df["FlowOutExNtv"]
        df["Flow_Ratio"] = (
            df["FlowInExNtv"] / df["FlowOutExNtv"].replace(0, np.nan)
        )

    # --- Active addresses momentum ---
    if "AdrActCnt" in df.columns:
        df["AdrAct_ROC_7d"] = df["AdrActCnt"].pct_change(7) * 100
        if "TxTfrCnt" in df.columns:
            df["TxTfr_per_Active_Adr"] = (
                df["TxTfrCnt"] / df["AdrActCnt"].replace(0, np.nan)
            )

    # --- Miner sell pressure ---
    if (
        "IssTotUSD" in df.columns
        and "volume_reported_spot_usd_1d" in df.columns
    ):
        df["Miner_Sell_Pressure"] = (
            df["IssTotUSD"]
            / df["volume_reported_spot_usd_1d"].replace(0, np.nan)
        )

    # --- MVRV momentum ---
    if "CapMVRVCur" in df.columns:
        df["MVRV_Momentum"] = (
            df["CapMVRVCur"] - df["CapMVRVCur"].rolling(7).mean()
        )

    # ------------------------------------------------------------------
    # Track newly added columns
    # ------------------------------------------------------------------
    onchain_features = [c for c in df.columns if c not in initial_columns]
    logger.info("On-chain features created: %d", len(onchain_features))

    return df, onchain_features


# ═══════════════════════════════════════════════════════════════════════════
# NaN Cleanup (rolling-window warm-up)
# ═══════════════════════════════════════════════════════════════════════════
def drop_warmup_nans(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Drop rows where any *feature_cols* column is NaN.

    This removes the leading warm-up rows produced by rolling windows
    without touching the target column.  **Never** forward-fills —
    per the user's explicit requirement.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features already created.
    feature_cols : list of str
        Column names to check for NaNs.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no NaNs in the specified feature columns.
    """
    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    logger.info(
        "Dropped %d warm-up rows (%.1f%%). Remaining: %d",
        dropped, 100 * dropped / max(before, 1), len(df),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Autoregressive Lag Features
# ═══════════════════════════════════════════════════════════════════════════
def create_lagged_features(
    df: pd.DataFrame,
    features_to_lag: List[str],
    lags: int = 3,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create shifted (lagged) columns for autoregressive modelling.

    For each feature in *features_to_lag* and each lag ``1 … lags``,
    creates a new column ``{feature}_t-{lag}`` using ``pd.Series.shift``.

    This provides the "memory" term for an AR Logistic Regression:
    ``Y = logistic(X_t, X_{t-1}, X_{t-2}, …)``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with the features to lag.
    features_to_lag : list of str
        Column names to create lagged versions of.
    lags : int
        Number of lags to create (default 3 → t-1, t-2, t-3).
    drop_na : bool
        If True, drop rows with NaNs introduced by shifting
        (default True — prevents training errors).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - DataFrame with new lag columns appended.
        - List of newly created lag column names.
    """
    df = df.copy()
    lag_cols: List[str] = []

    for feat in features_to_lag:
        if feat not in df.columns:
            logger.warning("Skipping lag for missing column: %s", feat)
            continue
        for lag in range(1, lags + 1):
            col_name = f"{feat}_t-{lag}"
            df[col_name] = df[feat].shift(lag)
            lag_cols.append(col_name)

    if drop_na and lag_cols:
        before = len(df)
        df = df.dropna(subset=lag_cols)
        logger.info(
            "Lagged features: %d cols created, %d warm-up rows dropped",
            len(lag_cols), before - len(df),
        )
    else:
        logger.info("Lagged features: %d cols created (NaNs retained)", len(lag_cols))

    return df, lag_cols


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def create_all_features(
    df: pd.DataFrame,
    include_ta: bool = True,
    include_onchain: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run all feature engineering steps and return the combined feature list.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``data_loader.load_dataset`` or ``load_from_config``.
    include_ta : bool
        Whether to create Technical Analysis features (default True).
    include_onchain : bool
        Whether to create on-chain engineered features (default True).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The fully augmented DataFrame and the merged list of all newly
        created feature column names.
    """
    all_features: List[str] = []

    if include_ta:
        df, ta_feats = create_ta_features(df)
        all_features.extend(ta_feats)

    if include_onchain:
        df, oc_feats = create_onchain_features(df)
        all_features.extend(oc_feats)

    logger.info("Total engineered features: %d", len(all_features))
    return df, all_features
