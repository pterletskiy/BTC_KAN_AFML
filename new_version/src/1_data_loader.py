# data_loader.py — Modular data fetching for the MFW Asset Direction Predictor
# Follows the rules defined in financial_data.md

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from coinmetrics.api_client import CoinMetricsClient

# ═══════════════════════════════════════════════════════════════════════════
# Constants & paths
# ═══════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "raw"

DEFAULT_COINMETRICS_METRICS = [
    "AdrActCnt", "TxCnt", "TxTfrValAdjUSD", "FeeMeanUSD", "HashRate", 
    "DiffMean", "NVTAdj", "CapMrktCurUSD", "CapRealUSD", "CapMVRVCur", 
    "SplyAct1yr", "FlowInExUSD", "FlowOutExUSD"
]

# ═══════════════════════════════════════════════════════════════════════════
# 4) Idempotent Caching Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _cache_path(source: str, key: str, start: str, end: str) -> Path:
    """Return a deterministic Parquet cache file path under ``data/raw/``."""
    safe_key = key.replace("/", "_").replace("^", "").replace("-", "_").replace(".", "_")
    return _CACHE_DIR / f"{source}_{safe_key}_{start}_{end}.parquet"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    """Read a cached DataFrame if the file exists, else return None."""
    if path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > 24:
            logger.warning("Cache may be stale (%s): %.0f hours old", path.name, age_hours)
            
        logger.info("Cache hit: %s", path.name)
        return pd.read_parquet(path)
    return None


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    logger.info("Cached → %s", path.name)

def _validate_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{name}: empty DataFrame returned.")
    if df.index.duplicated().any():
        n_dupes = df.index.duplicated().sum()
        logger.warning("%s: %d duplicate dates found — keeping last.", name, n_dupes)
        df = df[~df.index.duplicated(keep="last")]
    if not df.index.is_monotonic_increasing:
        logger.warning("%s: index not sorted — sorting now.", name)
        df.sort_index(inplace=True)
    return df

# ═══════════════════════════════════════════════════════════════════════════
# 3) Timezone Alignment Helper
# ═══════════════════════════════════════════════════════════════════════════
def _to_utc_midnight(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize any DatetimeIndex to UTC, midnight-aligned."""
    if index.tz is None:
        return index.tz_localize("UTC").normalize()
    return index.tz_convert("UTC").normalize()

# ═══════════════════════════════════════════════════════════════════════════
# 1) Primary Asset (OHLCV via yfinance)
# ═══════════════════════════════════════════════════════════════════════════
def fetch_primary_asset(
    ticker: str,
    start: str = "2014-09-17",
    end: str = "2026-03-07",
    interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data for a single primary asset via yfinance.

    Parameters
    ----------
    ticker : str
        yfinance ticker symbol (e.g. ``'BTC-USD'``, ``'SPY'``, ``'GLD'``).
    start, end : str
        Date strings in ``'YYYY-MM-DD'`` format.
    interval : str
        Data frequency (default ``'1d'``).

    Returns
    -------
    pd.DataFrame
        Columns: ``Open, High, Low, Close, Volume``.
        Index: ``DatetimeIndex`` with ``tz='UTC'``, name ``'Date'``.
    """
    cache = _cache_path("yfinance_ohlcv", ticker, start, end)
    cached = _read_cache(cache)
    if cached is not None:
        return cached

    logger.info("Downloading OHLCV for %s …", ticker)
    df = yf.download(ticker, start=start, end=end, interval=interval)

    # yfinance may return MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = _to_utc_midnight(df.index)
    df.index.name = "Date"

    if (df["Close"] <= 0).any():
        n_before = len(df)
        df = df[df["Close"] > 0]
        logger.warning("yfinance_ohlcv:%s: Dropped %d rows with non-positive Close prices", ticker, n_before - len(df))
    
    df = _validate_df(df, f"yfinance_ohlcv:{ticker}")

    _write_cache(df, cache)
    return df

# ═══════════════════════════════════════════════════════════════════════════
# 1) Secondary Features: CoinMetrics On-Chain
# ═══════════════════════════════════════════════════════════════════════════
def fetch_coinmetrics(
    asset: str = "btc",
    metrics: Optional[List[str]] = None,
    start: str = "2014-09-17",
    end: str = "2026-03-07") -> pd.DataFrame:
    """Fetch on-chain metrics from the CoinMetrics community API.

    If *metrics* is ``None``, **all** available daily metrics for the
    asset are fetched (mirrors the original notebook behaviour).

    A 1-day forward shift is applied so that the metric value at row *T*
    represents data known at end-of-day *T−1* (no look-ahead).

    Parameters
    ----------
    asset : str
        Crypto asset ticker for CoinMetrics (e.g. ``'btc'``).
    metrics : list of str, optional
        Specific metric names.  Pass ``None`` for all daily metrics.
    start, end : str
        Date range in ``'YYYY-MM-DD'``.

    Returns
    -------
    pd.DataFrame
        On-chain features with UTC midnight ``DatetimeIndex``.
    """
    key = f"{asset}_{'all' if metrics is None else '_'.join(sorted(metrics))}"
    cache = _cache_path("coinmetrics", key, start, end)
    cached = _read_cache(cache)
    if cached is not None:
        return cached

    logger.info("Fetching CoinMetrics data for %s …", asset)
    client = CoinMetricsClient()

    # Resolve metric list if not provided
    if metrics is None:
        metrics = DEFAULT_COINMETRICS_METRICS
        logger.info("Using default %d CoinMetrics metrics for %s", len(metrics), asset)

    raw = client.get_asset_metrics(
        assets=asset,
        metrics=metrics,
        start_time=start,
        end_time=end,
        frequency="1d")
    df = raw.to_dataframe()

    # Drop the 'asset' column if present
    if "asset" in df.columns:
        df = df.drop(columns=["asset"])

    # Datetime index
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].dt.tz_localize(None).dt.normalize()
    df = df.set_index("time")
    df.index = _to_utc_midnight(df.index)
    df.index.name = "Date"
    
    df = _validate_df(df, f"coinmetrics:{asset}")

    # Shift by 1 day to prevent look-ahead and drop resulting NaN
    df = df.shift(1).dropna()

    # Coerce all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    _write_cache(df, cache)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1) Secondary Features: Blockchain.com On-Chain
# ═══════════════════════════════════════════════════════════════════════════

#: Default Blockchain.com API chart names → DataFrame column names.
BLOCKCHAIN_COM_METRICS: Dict[str, str] = {
    "n-transactions":     "bc_transactions",
    "n-unique-addresses": "bc_unique_addresses",
    "hash-rate":          "bc_hash_rate",
    "difficulty":         "bc_difficulty",
    "miners-revenue":     "bc_miners_revenue",
    "transaction-fees":   "bc_transaction_fees",
    "market-price":       "bc_market_price",
    "total-bitcoins":     "bc_total_bitcoins",
    "mempool-size":       "bc_mempool_size",}

def _fetch_with_retry(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3, timeout: int = 30) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("Retry %d/%d for %s (%.1fs)", attempt+1, retries, url, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unreachable")

def fetch_blockchain_com(
    metrics: Optional[Dict[str, str]] = None,
    start: str = "2014-09-17",
    end: str = "2026-03-07") -> pd.DataFrame:
    """Fetch on-chain data from the Blockchain.com public charts API.

    Parameters
    ----------
    metrics : dict, optional
        Mapping ``{api_chart_name: column_name}``.
        Defaults to :data:`BLOCKCHAIN_COM_METRICS`.
    start, end : str
        Date range in ``'YYYY-MM-DD'``.

    Returns
    -------
    pd.DataFrame
        Daily-resampled on-chain features with UTC midnight index.
        A 1-day forward shift is applied (same rationale as CoinMetrics).
    """
    if metrics is None:
        metrics = BLOCKCHAIN_COM_METRICS

    key = "_".join(sorted(metrics.values()))
    cache = _cache_path("blockchain_com", key, start, end)
    cached = _read_cache(cache)
    if cached is not None:
        return cached

    from datetime import datetime as _dt

    start_dt = _dt.strptime(start, "%Y-%m-%d")
    end_dt = _dt.strptime(end, "%Y-%m-%d")
    timespan_days = (end_dt - start_dt).days

    base_url = "https://api.blockchain.info/charts/"
    frames: Dict[str, pd.DataFrame] = {}

    logger.info("Fetching Blockchain.com data (%d metrics) …", len(metrics))
    for api_key, col_name in metrics.items():
        try:
            resp = _fetch_with_retry(
                f"{base_url}{api_key}",
                params={
                    "timespan": f"{timespan_days}days",
                    "format": "json",
                    "sampled": "false",
                },
                timeout=30,
            )
            values = resp.json().get("values", [])
            tmp = pd.DataFrame(values)
            tmp["date"] = pd.to_datetime(tmp["x"], unit="s")
            tmp = tmp.rename(columns={"y": col_name})[["date", col_name]]
            frames[col_name] = tmp
            logger.info("  ✓ %s: %d rows", col_name, len(tmp))
            time.sleep(0.5)  # courtesy rate-limit pause
        except Exception as exc:
            logger.warning("  ✗ %s: %s", col_name, exc)

    if not frames:
        logger.warning("No Blockchain.com data retrieved.")
        return pd.DataFrame()

    # Merge all single-column frames on the date column
    merged: Optional[pd.DataFrame] = None
    for _, tmp in frames.items():
        if merged is None:
            merged = tmp
        else:
            merged = pd.merge(merged, tmp, on="date", how="left")

    merged = merged.sort_values("date").set_index("date")

    # Blockchain.com timestamps can be intra-day → resample to daily
    merged = merged.resample("1D").last()

    merged.index = _to_utc_midnight(merged.index)
    merged.index.name = "Date"
    
    merged = _validate_df(merged, "blockchain_com")

    # 1-day shift to prevent look-ahead and drop resulting NaN
    merged = merged.shift(1).dropna()

    _write_cache(merged, cache)
    return merged

# ═══════════════════════════════════════════════════════════════════════════
# 1) On-Chain Routing: coinmetrics | blockchain_com | both
# ═══════════════════════════════════════════════════════════════════════════
def fetch_onchain_features(
    provider: str = "coinmetrics",
    asset: str = "btc",
    coinmetrics_metrics: Optional[List[str]] = None,
    blockchain_com_metrics: Optional[Dict[str, str]] = None,
    start: str = "2014-09-17",
    end: str = "2026-03-07",
) -> pd.DataFrame:
    """Unified on-chain data router.

    Parameters
    ----------
    provider : str
        One of ``'coinmetrics'``, ``'blockchain_com'``, or ``'both'``.
    asset : str
        Crypto asset ticker for CoinMetrics (e.g. ``'btc'``).
    coinmetrics_metrics : list of str, optional
        Specific CoinMetrics metrics (``None`` = all daily).
    blockchain_com_metrics : dict, optional
        Blockchain.com metrics mapping (``None`` = defaults).
    start, end : str
        Date range.

    Returns
    -------
    pd.DataFrame
        On-chain features merged on the ``Date`` index.
    """
    provider = provider.lower().strip()

    if provider == "coinmetrics":
        return fetch_coinmetrics(asset, coinmetrics_metrics, start, end)

    if provider == "blockchain_com":
        return fetch_blockchain_com(blockchain_com_metrics, start, end)

    if provider == "both":
        cm = fetch_coinmetrics(asset, coinmetrics_metrics, start, end)
        bc = fetch_blockchain_com(blockchain_com_metrics, start, end)
        return pd.merge(cm, bc, left_index=True, right_index=True, how="outer")

    raise ValueError(
        f"Unknown on-chain provider '{provider}'. "
        "Choose 'coinmetrics', 'blockchain_com', or 'both'.")

# ═══════════════════════════════════════════════════════════════════════════
# 1) Secondary Features: Macro Indicator via yfinance
# ═══════════════════════════════════════════════════════════════════════════
def fetch_macro_feature(ticker: str, start: str = "2014-09-17", end: str = "2026-03-07") -> pd.DataFrame:
    """Fetch a single macro/market indicator via yfinance.

    Useful for indices such as DXY (``DX-Y.NYB``), VIX (``^VIX``),
    Federal Funds Rate, etc.

    Returns a single-column DataFrame named after the ticker
    (sanitised), containing the Close price.
    """
    cache = _cache_path("yfinance_macro", ticker, start, end)
    cached = _read_cache(cache)
    if cached is not None:
        return cached

    logger.info("Downloading macro feature %s …", ticker)
    df = yf.download(ticker, start=start, end=end, interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    safe_name = ticker.replace("^", "").replace("-", "_").replace(".", "_")
    df = df[["Close"]].rename(columns={"Close": safe_name})

    df.index = _to_utc_midnight(df.index)
    df.index.name = "Date"

    df = _validate_df(df, f"yfinance_macro:{ticker}")
    df = df.shift(1).dropna()

    _write_cache(df, cache)
    return df

# ═══════════════════════════════════════════════════════════════════════════
# 1) General Secondary Features Router
# ═══════════════════════════════════════════════════════════════════════════
def fetch_secondary_features(feature_list: List[Dict[str, Any]],
                             start: str = "2014-09-17", end: str = "2026-03-07") -> List[pd.DataFrame]:
    """Dispatch each feature request to the appropriate fetcher.

    Parameters
    ----------
    feature_list : list of dict
        Each dict must contain a ``"source"`` key.  Examples::

            {"source": "coinmetrics",    "asset": "btc", "metrics": [...]}
            {"source": "blockchain_com", "metrics": {...}}
            {"source": "onchain",        "provider": "both", "asset": "btc"}
            {"source": "yfinance",       "ticker": "^VIX"}

    start, end : str
        Date range.

    Returns
    -------
    list of pd.DataFrame
        One DataFrame per feature source, ready for merging.
    """
    results: List[pd.DataFrame] = []

    for spec in feature_list:
        source = spec["source"].lower()

        if source == "coinmetrics":
            results.append(fetch_coinmetrics(
                asset=spec.get("asset", "btc"),
                metrics=spec.get("metrics"),
                start=start, end=end,
            ))
        elif source == "blockchain_com":
            results.append(fetch_blockchain_com(
                metrics=spec.get("metrics"),
                start=start, end=end,
            ))
        elif source == "onchain":
            results.append(fetch_onchain_features(
                provider=spec.get("provider", "coinmetrics"),
                asset=spec.get("asset", "btc"),
                coinmetrics_metrics=spec.get("coinmetrics_metrics"),
                blockchain_com_metrics=spec.get("blockchain_com_metrics"),
                start=start, end=end,
            ))
        elif source == "yfinance":
            results.append(fetch_macro_feature(
                ticker=spec["ticker"],
                start=start, end=end,
            ))
        else:
            raise ValueError(f"Unknown feature source: '{source}'")

    return results

# ═══════════════════════════════════════════════════════════════════════════
# 2) Log Returns & Target Calculation
# ═══════════════════════════════════════════════════════════════════════════
def compute_log_return_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns and binary direction target.

    Following ``financial_data.md`` §2:

    * ``r_t  = ln(Close_t) − ln(Close_{t−1})``
    * ``Price_Direction`` at time *T* = 1 if ``r_{T+1} > 0``, else 0.

    ``Log_Return`` (``r_t``) is kept as an independent feature.
    The shifted future return is **not** kept to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``'Close'`` column.

    Returns
    -------
    pd.DataFrame
        Original *df* augmented with ``Log_Return`` and
        ``Price_Direction``.  First and last rows are dropped
        (NaN log-return and NaN target respectively).
    """
    df = df.copy()

    # Daily log return: r_t = ln(Close_t) - ln(Close_{t-1})
    df["Log_Return"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))

    # Target: sign of the NEXT day's log return (r_{T+1})
    future_return = df["Log_Return"].shift(-1)
    df["Price_Direction"] = (future_return > 0).astype(int)

    # Drop rows with NaN target (last row) and NaN log return (first row)
    n_before = len(df)
    last_dropped_date = df.index[-1].strftime('%Y-%m-%d')
    df = df.dropna(subset=["Price_Direction", "Log_Return"])
    logger.info("compute_log_return_target: Dropped %d rows due to NaN target or returns (last dropped: %s)", n_before - len(df), last_dropped_date)

    return df

# ═══════════════════════════════════════════════════════════════════════════
# 3)Calendar Merging
# ═══════════════════════════════════════════════════════════════════════════
def merge_datasets(primary_df: pd.DataFrame, *secondary_dfs: pd.DataFrame) -> pd.DataFrame:
    """Left-join secondary DataFrames onto the primary asset's date index.

    Lower-frequency features (e.g. monthly macro) are **forward-filled**
    (§3: ``ffill`` only — never ``bfill``).  Primary columns are never
    forward-filled.

    Parameters
    ----------
    primary_df : pd.DataFrame
        Must have a UTC ``DatetimeIndex`` named ``'Date'``.
    *secondary_dfs : pd.DataFrame
        Any number of secondary feature DataFrames.

    Returns
    -------
    pd.DataFrame
        Merged dataset on the primary's date grid.
    """
    merged = primary_df.copy()
    primary_cols = set(primary_df.columns)

    for sec_df in secondary_dfs:
        if sec_df.empty:
            continue
        merged = pd.merge(
            merged, sec_df,
            left_index=True, right_index=True,
            how="left",
        )

    # Forward-fill ONLY secondary columns (§3: never bfill)
    secondary_cols = [c for c in merged.columns if c not in primary_cols]
    if secondary_cols:
        for col in secondary_cols:
            nans = merged[col].isnull()
            max_gap = nans.groupby((~nans).cumsum()).sum().max()
            if max_gap > 5:
                logger.warning("merge_datasets: secondary column '%s' has a gap of %d consecutive NaNs (> 5 limit).", col, max_gap)
        merged[secondary_cols] = merged[secondary_cols].ffill(limit=5)

    return merged

# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def load_dataset(ticker: str, feature_list: Optional[List[Dict[str, Any]]] = None,
                 start: str = "2014-09-17", end: str = "2026-03-07") -> pd.DataFrame:
    """End-to-end dataset loader: fetch → target → merge.

    Parameters
    ----------
    ticker : str
        Primary asset yfinance ticker.
    feature_list : list of dict, optional
        Secondary features to fetch (see :func:`fetch_secondary_features`).
        If ``None``, only the primary OHLCV + target is returned.
    start, end : str
        Date range.

    Returns
    -------
    pd.DataFrame
        Final merged dataset ready for feature engineering.
    """
    logger.info("Loading dataset for %s [%s → %s]", ticker, start, end)

    # 1. Primary asset
    primary = fetch_primary_asset(ticker, start, end)

    # 2. Target calculation
    primary = compute_log_return_target(primary)

    # 3. Secondary features
    if feature_list:
        secondary = fetch_secondary_features(feature_list, start, end)
        df = merge_datasets(primary, *secondary)
    else:
        df = primary

    logger.info("Final dataset: %d rows × %d columns", *df.shape)
    return df


def load_from_config(config: Dict[str, Any]) -> pd.DataFrame:
    """Convenience wrapper — call :func:`load_dataset` from a config dict.

    Parameters
    ----------
    config : dict
        Must contain ``"ticker"``.  Optionally ``"start"``, ``"end"``,
        and ``"feature_list"``.

    Returns
    -------
    pd.DataFrame
    """
    return load_dataset(ticker=config["ticker"], feature_list=config.get("feature_list"),
                        start=config.get("start", "2014-09-17"), end=config.get("end", "2026-03-07"))


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Asset Catalog & Date Helper
# ═══════════════════════════════════════════════════════════════════════════

#: Registry of known assets and their available secondary data sources.
#: ``category`` controls which menus appear in the interactive wizard.
#: Crypto assets get on-chain options; traditional assets do not.
ASSET_CATALOG: Dict[str, Dict[str, Any]] = {
    # ── Crypto ─────────────────────────────────────────────────────────
    "BTC-USD": {
        "name": "Bitcoin",
        "category": "crypto",
        "onchain_asset": "btc",
        "onchain_providers": ["coinmetrics", "blockchain_com", "both"],
        "macro_options": {
            "DX-Y.NYB": "US Dollar Index (DXY)",
            "^VIX":     "CBOE Volatility Index (VIX)",
            "^TNX":     "10-Year Treasury Yield",
            "^IRX":     "3-Month T-Bill Rate",
            "^GSPC":    "S&P 500 Index",
        },
    },
    "ETH-USD": {
        "name": "Ethereum",
        "category": "crypto",
        "onchain_asset": "eth",
        "onchain_providers": ["coinmetrics"],  # blockchain.com is BTC-only
        "macro_options": {
            "DX-Y.NYB": "US Dollar Index (DXY)",
            "^VIX":     "CBOE Volatility Index (VIX)",
            "^TNX":     "10-Year Treasury Yield",
            "^IRX":     "3-Month T-Bill Rate",
        },
    },
    "SOL-USD": {
        "name": "Solana",
        "category": "crypto",
        "onchain_asset": "sol",
        "onchain_providers": ["coinmetrics"],
        "macro_options": {
            "DX-Y.NYB": "US Dollar Index (DXY)",
            "^VIX":     "CBOE Volatility Index (VIX)",
            "^TNX":     "10-Year Treasury Yield",
        },
    },
    # ── Traditional ───────────────────────────────────────────────────
    "GLD": {
        "name": "SPDR Gold Shares (Gold ETF)",
        "category": "traditional",
        "macro_options": {
            "DX-Y.NYB": "US Dollar Index (DXY)",
            "^VIX":     "CBOE Volatility Index (VIX)",
            "^TNX":     "10-Year Treasury Yield",
            "^IRX":     "3-Month T-Bill Rate",
            "^GSPC":    "S&P 500 Index",
            "SI=F":     "Silver Futures",
        },
    },
    "SLV": {
        "name": "iShares Silver Trust (Silver ETF)",
        "category": "traditional",
        "macro_options": {
            "DX-Y.NYB": "US Dollar Index (DXY)",
            "^VIX":     "CBOE Volatility Index (VIX)",
            "^TNX":     "10-Year Treasury Yield",
            "^IRX":     "3-Month T-Bill Rate",
            "GC=F":     "Gold Futures",
        },
    },
    "SPY": {
        "name": "SPDR S&P 500 ETF",
        "category": "traditional",
        "macro_options": {
            "DX-Y.NYB": "US Dollar Index (DXY)",
            "^VIX":     "CBOE Volatility Index (VIX)",
            "^TNX":     "10-Year Treasury Yield",
            "^IRX":     "3-Month T-Bill Rate",
            "GC=F":     "Gold Futures",
            "CL=F":     "Crude Oil Futures",
        },
    },
    "QQQ": {
        "name": "Invesco QQQ Trust (Nasdaq-100 ETF)",
        "category": "traditional",
        "macro_options": {
            "DX-Y.NYB": "US Dollar Index (DXY)",
            "^VIX":     "CBOE Volatility Index (VIX)",
            "^TNX":     "10-Year Treasury Yield",
            "^GSPC":    "S&P 500 Index",
        },
    },
}


def prompt_date_range(default_start: str = "2014-09-17", default_end: str = "2026-03-07") -> Tuple[str, str]:
    """Prompt the user for a date range and return consistent dates.

    This is the **single point of date input**.  Both returned strings
    are used by every fetching function so that all data is aligned
    to the same period.

    Parameters
    ----------
    default_start, default_end : str
        Fallback dates shown if the user presses Enter.

    Returns
    -------
    tuple[str, str]
        ``(start_date, end_date)`` in ``'YYYY-MM-DD'`` format.
    """
    print("\n📅 Date Range")
    print(f"  All data sources will use the same period.")
    try:
        start = input(f"  Start date [YYYY-MM-DD] (default {default_start}): ").strip()
        start = start if start else default_start
        end = input(f"  End date   [YYYY-MM-DD] (default {default_end}): ").strip()
        end = end if end else default_end
        
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", start) or not re.match(r"^\d{4}-\d{2}-\d{2}$", end):
            raise ValueError("Dates must be in YYYY-MM-DD format.")
    except EOFError:
        logger.warning("Non-interactive environment detected (EOFError). Using default dates.")
        start, end = default_start, default_end
        
    print(f"  → Period: {start} to {end}")
    return start, end

# ═══════════════════════════════════════════════════════════════════════════
# Interactive Notebook Config Builder
# ═══════════════════════════════════════════════════════════════════════════
def interactive_config() -> Dict[str, Any]:
    """Build a dataset configuration interactively via ``input()`` prompts.

    Multi-step wizard designed for Jupyter Notebook cells:

    1. **Asset Selection** — presets from :data:`ASSET_CATALOG` or custom.
    2. **Date Range** — via :func:`prompt_date_range`.
    3. **Secondary Data** — dynamic menu based on asset category:
       - *Crypto* assets: on-chain provider selection + macro checklist.
       - *Traditional* assets: macro checklist only.

    Returns
    -------
    dict
        A config dict compatible with :func:`load_from_config`.
    """
    print("=" * 60)
    print("  MFW Multi-Asset Dataset Configuration Wizard")
    print("=" * 60)

    # ── Step 1: Asset Selection ────────────────────────────────────────
    print("\n📈 Step 1 — Select the asset you want to predict")
    catalog_keys = list(ASSET_CATALOG.keys())
    for i, key in enumerate(catalog_keys, 1):
        entry = ASSET_CATALOG[key]
        cat_tag = "🟠 Crypto" if entry["category"] == "crypto" else "🔵 Traditional"
        print(f"    [{i}] {key:10s}  {entry['name']:35s}  {cat_tag}")
    print(f"    [{len(catalog_keys) + 1}] Custom ticker")

    choice = input(f"  Select [1-{len(catalog_keys) + 1}]: ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(catalog_keys):
            ticker = catalog_keys[idx]
            catalog_entry = ASSET_CATALOG[ticker]
        else:
            ticker = input("  Enter custom yfinance ticker (e.g. AAPL): ").strip()
            cat_q = input("  Is this a crypto asset? [y/n] (default n): ").strip().lower()
            category = "crypto" if cat_q == "y" else "traditional"
            catalog_entry: Dict[str, Any] = {
                "name": ticker,
                "category": category,
                "macro_options": {
                    "DX-Y.NYB": "US Dollar Index (DXY)",
                    "^VIX":     "CBOE Volatility Index (VIX)",
                    "^TNX":     "10-Year Treasury Yield",
                },
            }
            if category == "crypto":
                oc_asset = input("  CoinMetrics asset id (e.g. btc, eth): ").strip().lower()
                catalog_entry["onchain_asset"] = oc_asset or "btc"
                catalog_entry["onchain_providers"] = ["coinmetrics"]
    except ValueError:
        ticker = choice
        catalog_entry = {
            "name": ticker, "category": "traditional",
            "macro_options": {"DX-Y.NYB": "DXY", "^VIX": "VIX", "^TNX": "10Y Yield"},
        }

    print(f"  → Primary asset: {ticker} ({catalog_entry['name']})")
    is_crypto = catalog_entry["category"] == "crypto"

    # ── Step 2: Date Range ────────────────────────────────────────────
    start, end = prompt_date_range()

    # ── Step 3: Secondary Data Sources ────────────────────────────────
    feature_list: List[Dict[str, Any]] = []

    # 3a. On-chain data (crypto only)
    if is_crypto:
        providers = catalog_entry.get("onchain_providers", [])
        oc_asset = catalog_entry.get("onchain_asset", "btc")

        print("\n🔗 Step 3a — On-Chain Data")
        print(f"  Available providers for {ticker}:")
        provider_options: List[str] = []
        for i, prov in enumerate(providers, 1):
            label = {
                "coinmetrics": "CoinMetrics",
                "blockchain_com": "Blockchain.com",
                "both": "Both (CoinMetrics + Blockchain.com)",
            }.get(prov, prov)
            print(f"    [{i}] {label}")
            provider_options.append(prov)
        skip_idx = len(provider_options) + 1
        print(f"    [{skip_idx}] None — skip on-chain data")

        oc_choice = input(f"  Select [1-{skip_idx}]: ").strip()
        try:
            oc_idx = int(oc_choice) - 1
            if 0 <= oc_idx < len(provider_options):
                feature_list.append({
                    "source": "onchain",
                    "provider": provider_options[oc_idx],
                    "asset": oc_asset,
                })
                print(f"  → On-chain: {provider_options[oc_idx]} ({oc_asset})")
            else:
                print("  → On-chain: skipped")
        except ValueError:
            print("  → On-chain: skipped")

    # 3b. Macro features (all assets)
    macro_opts = catalog_entry.get("macro_options", {})
    if macro_opts:
        step_label = "3b" if is_crypto else "3"
        print(f"\n🌍 Step {step_label} — Macro / Market Features")
        print(f"  Available for {ticker}:")
        macro_keys = list(macro_opts.keys())
        for i, (tk, desc) in enumerate(macro_opts.items(), 1):
            print(f"    [{i}] {tk:12s}  {desc}")
        print("  Enter the numbers you want, separated by commas.")
        print("  Or press Enter to skip, or type 'all' to select everything.")

        macro_input = input("  Selection: ").strip().lower()
        if macro_input == "all":
            selected_macros = macro_keys
        elif macro_input:
            selected_macros = []
            for part in macro_input.split(","):
                part = part.strip()
                try:
                    m_idx = int(part) - 1
                    if 0 <= m_idx < len(macro_keys):
                        selected_macros.append(macro_keys[m_idx])
                except ValueError:
                    # Allow direct ticker input as fallback
                    if part:
                        selected_macros.append(part)
        else:
            selected_macros = []

        for tk in selected_macros:
            feature_list.append({"source": "yfinance", "ticker": tk})

        if selected_macros:
            print(f"  → Macro features: {selected_macros}")
        else:
            print("  → Macro features: skipped")

    # ── Build config ──────────────────────────────────────────────────
    config: Dict[str, Any] = {
        "ticker": ticker,
        "start": start,
        "end": end,
    }
    if feature_list:
        config["feature_list"] = feature_list

    print("\n" + "=" * 60)
    print("  ✅ Configuration Ready")
    print("=" * 60)
    n_secondary = len(feature_list)
    print(f"  Asset:    {ticker}")
    print(f"  Period:   {start} → {end}")
    print(f"  Category: {catalog_entry['category']}")
    print(f"  Secondary sources: {n_secondary}")
    for spec in feature_list:
        if spec["source"] == "onchain":
            print(f"    • On-chain ({spec['provider']}, asset={spec['asset']})")
        elif spec["source"] == "yfinance":
            print(f"    • Macro: {spec['ticker']}")
    return config


# ═══════════════════════════════════════════════════════════════════════════
# Default Preset Configs (Quick-Run Mode)
# ═══════════════════════════════════════════════════════════════════════════

#: Minimal BTC config — CoinMetrics on-chain only.
DEFAULT_BTC_CONFIG: Dict[str, Any] = {
    "ticker": "BTC-USD",
    "start": "2014-09-17",
    "end": "2026-03-07",
    "feature_list": [
        {"source": "onchain", "provider": "coinmetrics", "asset": "btc"},
    ],
}

#: Full BTC config — both on-chain providers + macro features.
DEFAULT_BTC_FULL_CONFIG: Dict[str, Any] = {
    "ticker": "BTC-USD",
    "start": "2014-09-17",
    "end": "2026-03-07",
    "feature_list": [
        {"source": "onchain", "provider": "both", "asset": "btc"},
        {"source": "yfinance", "ticker": "DX-Y.NYB"},
        {"source": "yfinance", "ticker": "^VIX"},
        {"source": "yfinance", "ticker": "^TNX"},
    ],
}

#: ETH config — CoinMetrics on-chain + macro.
DEFAULT_ETH_CONFIG: Dict[str, Any] = {
    "ticker": "ETH-USD",
    "start": "2017-01-01",
    "end": "2026-03-07",
    "feature_list": [
        {"source": "onchain", "provider": "coinmetrics", "asset": "eth"},
        {"source": "yfinance", "ticker": "DX-Y.NYB"},
        {"source": "yfinance", "ticker": "^VIX"},
    ],
}

#: GLD config — Traditional macro features only.
DEFAULT_GLD_CONFIG: Dict[str, Any] = {
    "ticker": "GLD",
    "start": "2010-01-01",
    "end": "2026-03-07",
    "feature_list": [
        {"source": "yfinance", "ticker": "DX-Y.NYB"},
        {"source": "yfinance", "ticker": "^VIX"},
        {"source": "yfinance", "ticker": "^TNX"},
    ],
}

#: SPY config — Traditional macro features only.
DEFAULT_SPY_CONFIG: Dict[str, Any] = {
    "ticker": "SPY",
    "start": "2010-01-01",
    "end": "2026-03-07",
    "feature_list": [
        {"source": "yfinance", "ticker": "DX-Y.NYB"},
        {"source": "yfinance", "ticker": "^VIX"},
        {"source": "yfinance", "ticker": "^TNX"},
        {"source": "yfinance", "ticker": "CL=F"},
    ],
}

#: BTC config — OHLCV only, no secondary features.
#: The absence of the "feature_list" key is intentional and handled downstream.
DEFAULT_BTC_OHLCV_CONFIG: Dict[str, Any] = {
    "ticker": "BTC-USD",
    "start": "2014-09-17",
    "end": "2026-03-07",
}

#: GLD config — OHLCV only, no secondary features.
DEFAULT_GLD_OHLCV_CONFIG: Dict[str, Any] = {
    "ticker": "GLD",
    "start": "2010-01-01",
    "end": "2026-03-07",
}

#: QQQ config — OHLCV only, no secondary features.
DEFAULT_QQQ_OHLCV_CONFIG: Dict[str, Any] = {
    "ticker": "QQQ",
    "start": "2010-01-01",
    "end": "2026-03-07",
}
