"""
data_loader.py — Modular data fetching for the MFW Asset Direction Predictor.

Responsibilities:
  ✅  Fetch raw OHLCV data via yfinance
  ✅  Fetch raw macro features via yfinance
  ✅  Fetch raw on-chain data (CoinMetrics, Blockchain.com)
  ✅  Cache fetched data to Parquet (always RAW — never transformed)
  ✅  Align all sources to a common UTC midnight DatetimeIndex
  ✅  Left-join secondary features onto the primary asset's date grid
  ✅  Provide config presets and an asset catalog
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from coinmetrics.api_client import CoinMetricsClient

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONSTANTS & PATHS
# ══════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = Path(os.getenv(
    "MFW_CACHE_DIR",
    str(_PROJECT_ROOT / "data" / "raw"),
))

DEFAULT_COINMETRICS_METRICS = [
    "AdrActCnt", "TxCnt", "TxTfrValAdjUSD", "FeeMeanUSD", "HashRate",
    "DiffMean", "NVTAdj", "CapMrktCurUSD", "CapRealUSD", "CapMVRVCur",
    "SplyAct1yr", "FlowInExUSD", "FlowOutExUSD",
]

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
    "mempool-size":       "bc_mempool_size",
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. PRIVATE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

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
    """Validate a DataFrame: check empty, duplicates, sort order."""
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


def _to_utc_midnight(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize any DatetimeIndex to UTC, midnight-aligned."""
    if index.tz is None:
        return index.tz_localize("UTC").normalize()
    return index.tz_convert("UTC").normalize()


def _with_retry(fn: Callable, *args, retries: int = 3, **kwargs):
    """Call *fn* with exponential-backoff retries.

    Works for any callable (``yf.download``, CoinMetrics client calls,
    ``requests.get``, etc.).  On the last failed attempt the exception
    is re-raised.

    Parameters
    ----------
    fn : callable
        The function to call.
    *args, **kwargs
        Forwarded to *fn*.
    retries : int
        Total number of attempts (default 3).

    Returns
    -------
    object
        Whatever *fn* returns on success.
    """
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    "Retry %d/%d for %s (wait %.1fs): %s",
                    attempt + 1, retries, fn.__name__ if hasattr(fn, '__name__') else str(fn),
                    wait, exc,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unreachable")  # pragma: no cover


# ══════════════════════════════════════════════════════════════════════════════
# 4. PRIMARY ASSET FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_primary_asset(
    ticker: str,
    start: str = "2014-09-17",
    end: str = "2026-03-07",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download OHLCV data for a single primary asset via yfinance.

    Note: No look-ahead shift is applied here. OHLCV at row T represents
    data available at end-of-day T. Targets are derived downstream in the
    pipeline using the NEXT day's close.

    Parameters
    ----------
    ticker : str
        yfinance ticker symbol (e.g. ``'BTC-USD'``, ``'SPY'``, ``'GLD'``).
    start, end : str
        Date strings in ``'YYYY-MM-DD'`` format.
    interval : str
        Data frequency (default ``'1d'``).
    force_refresh : bool
        If ``True``, skip reading from cache and re-download.

    Returns
    -------
    pd.DataFrame
        Columns: ``Open, High, Low, Close, Volume``.
        Index: ``DatetimeIndex`` with ``tz='UTC'``, name ``'Date'``.
    """
    cache = _cache_path("yfinance_ohlcv", ticker, start, end)

    if not force_refresh:
        cached = _read_cache(cache)
        if cached is not None:
            return cached

    logger.info("Downloading OHLCV for %s …", ticker)
    df = _with_retry(yf.download, ticker, start=start, end=end, interval=interval)

    # yfinance may return MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = _to_utc_midnight(df.index)
    df.index.name = "Date"

    if (df["Close"] <= 0).any():
        n_before = len(df)
        df = df[df["Close"] > 0]
        logger.warning(
            "yfinance_ohlcv:%s: Dropped %d rows with non-positive Close prices",
            ticker, n_before - len(df),
        )

    df = _validate_df(df, f"yfinance_ohlcv:{ticker}")

    _write_cache(df, cache)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. MACRO FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_macro_feature(
    ticker: str,
    start: str = "2014-09-17",
    end: str = "2026-03-07",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a single macro/market indicator via yfinance.

    Useful for indices such as DXY (``DX-Y.NYB``), VIX (``^VIX``),
    Federal Funds Rate, etc.

    Returns a single-column DataFrame named after the ticker
    (sanitised), containing the Close price.  A 1-day forward shift is
    applied so that the value at row *T* represents data known at
    end-of-day *T−1* (no look-ahead).

    Parameters
    ----------
    ticker : str
        yfinance ticker symbol.
    start, end : str
        Date range in ``'YYYY-MM-DD'``.
    force_refresh : bool
        If ``True``, skip reading from cache and re-download.
    """
    cache = _cache_path("yfinance_macro", ticker, start, end)

    if not force_refresh:
        cached = _read_cache(cache)
        if cached is not None:
            # Cache stores RAW data; apply look-ahead prevention after read
            return cached.shift(1).dropna()

    logger.info("Downloading macro feature %s …", ticker)
    df = _with_retry(yf.download, ticker, start=start, end=end, interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    safe_name = ticker.replace("^", "").replace("-", "_").replace(".", "_")
    df = df[["Close"]].rename(columns={"Close": safe_name})

    df.index = _to_utc_midnight(df.index)
    df.index.name = "Date"

    df = _validate_df(df, f"yfinance_macro:{ticker}")

    # Always cache RAW
    _write_cache(df, cache)

    # Apply look-ahead prevention shift AFTER caching
    return df.shift(1).dropna()


# ══════════════════════════════════════════════════════════════════════════════
# 6. ON-CHAIN FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_coinmetrics(
    asset: str = "btc",
    metrics: Optional[List[str]] = None,
    start: str = "2014-09-17",
    end: str = "2026-03-07",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch on-chain metrics from the CoinMetrics community API.

    If *metrics* is ``None``, the default metric set
    (:data:`DEFAULT_COINMETRICS_METRICS`) is used.

    A 1-day forward shift is applied so that the metric value at row *T*
    represents data known at end-of-day *T−1* (no look-ahead).

    Parameters
    ----------
    asset : str
        Crypto asset ticker for CoinMetrics (e.g. ``'btc'``).
    metrics : list of str, optional
        Specific metric names.  Pass ``None`` for defaults.
    start, end : str
        Date range in ``'YYYY-MM-DD'``.
    force_refresh : bool
        If ``True``, skip reading from cache and re-download.

    Returns
    -------
    pd.DataFrame
        On-chain features with UTC midnight ``DatetimeIndex``.
    """
    key = f"{asset}_{'all' if metrics is None else '_'.join(sorted(metrics))}"
    cache = _cache_path("coinmetrics", key, start, end)

    if not force_refresh:
        cached = _read_cache(cache)
        if cached is not None:
            # Cache stores RAW data; apply look-ahead prevention after read
            return cached.shift(1).dropna()

    logger.info("Fetching CoinMetrics data for %s …", asset)
    client = CoinMetricsClient()

    # Resolve metric list if not provided
    if metrics is None:
        metrics = DEFAULT_COINMETRICS_METRICS
        logger.info("Using default %d CoinMetrics metrics for %s", len(metrics), asset)

    raw = _with_retry(
        client.get_asset_metrics,
        assets=asset,
        metrics=metrics,
        start_time=start,
        end_time=end,
        frequency="1d",
    )
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

    # Coerce all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Always cache RAW
    _write_cache(df, cache)

    # Apply look-ahead prevention shift AFTER caching
    return df.shift(1).dropna()





def fetch_blockchain_com(
    metrics: Optional[Dict[str, str]] = None,
    start: str = "2014-09-17",
    end: str = "2026-03-07",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch on-chain data from the Blockchain.com public charts API.

    Parameters
    ----------
    metrics : dict, optional
        Mapping ``{api_chart_name: column_name}``.
        Defaults to :data:`BLOCKCHAIN_COM_METRICS`.
    start, end : str
        Date range in ``'YYYY-MM-DD'``.
    force_refresh : bool
        If ``True``, skip reading from cache and re-download.

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

    if not force_refresh:
        cached = _read_cache(cache)
        if cached is not None:
            # Cache stores RAW data; apply look-ahead prevention after read
            return cached.shift(1).dropna()

    from datetime import datetime as _dt

    start_dt = _dt.strptime(start, "%Y-%m-%d")
    end_dt = _dt.strptime(end, "%Y-%m-%d")
    timespan_days = (end_dt - start_dt).days

    base_url = "https://api.blockchain.info/charts/"
    frames: Dict[str, pd.DataFrame] = {}

    logger.info("Fetching Blockchain.com data (%d metrics) …", len(metrics))
    for api_key, col_name in metrics.items():
        try:
            resp = _with_retry(
                requests.get,
                f"{base_url}{api_key}",
                params={
                    "timespan": f"{timespan_days}days",
                    "format": "json",
                    "sampled": "false",
                },
                timeout=30,
            )
            resp.raise_for_status()
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

    # Always cache RAW
    _write_cache(merged, cache)

    # Apply look-ahead prevention shift AFTER caching
    return merged.shift(1).dropna()


def fetch_onchain_features(
    provider: str = "coinmetrics",
    asset: str = "btc",
    coinmetrics_metrics: Optional[List[str]] = None,
    blockchain_com_metrics: Optional[Dict[str, str]] = None,
    start: str = "2014-09-17",
    end: str = "2026-03-07",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Unified on-chain data router.

    Note: When using provider='both', the outer merge may introduce NaNs
    at the chronological boundaries of the datasets. Downstream robust
    scaling or imputation is required.

    Parameters
    ----------
    provider : str
        One of ``'coinmetrics'``, ``'blockchain_com'``, or ``'both'``.
    asset : str
        Crypto asset ticker for CoinMetrics (e.g. ``'btc'``).
    coinmetrics_metrics : list of str, optional
        Specific CoinMetrics metrics (``None`` = defaults).
    blockchain_com_metrics : dict, optional
        Blockchain.com metrics mapping (``None`` = defaults).
    start, end : str
        Date range.
    force_refresh : bool
        If ``True``, skip reading from cache and re-download.

    Returns
    -------
    pd.DataFrame
        On-chain features merged on the ``Date`` index.
    """
    provider = provider.lower().strip()

    if provider == "coinmetrics":
        return fetch_coinmetrics(asset, coinmetrics_metrics, start, end, force_refresh=force_refresh)

    if provider == "blockchain_com":
        return fetch_blockchain_com(blockchain_com_metrics, start, end, force_refresh=force_refresh)

    if provider == "both":
        cm = fetch_coinmetrics(asset, coinmetrics_metrics, start, end, force_refresh=force_refresh)
        bc = fetch_blockchain_com(blockchain_com_metrics, start, end, force_refresh=force_refresh)
        return pd.merge(cm, bc, left_index=True, right_index=True, how="outer")

    raise ValueError(
        f"Unknown on-chain provider '{provider}'. "
        "Choose 'coinmetrics', 'blockchain_com', or 'both'."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. SECONDARY FEATURES ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_secondary_features(
    feature_list: List[Dict[str, Any]],
    start: str = "2014-09-17",
    end: str = "2026-03-07",
    force_refresh: bool = False,
) -> List[pd.DataFrame]:
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
    force_refresh : bool
        If ``True``, skip cache reads and re-download all sources.

    Returns
    -------
    list of pd.DataFrame
        One DataFrame per feature source, ready for merging.
        Empty DataFrames are logged as warnings and excluded.
    """
    results: List[pd.DataFrame] = []

    for spec in feature_list:
        source = spec["source"].lower()
        df: Optional[pd.DataFrame] = None

        if source == "coinmetrics":
            df = fetch_coinmetrics(
                asset=spec.get("asset", "btc"),
                metrics=spec.get("metrics"),
                start=start, end=end,
                force_refresh=force_refresh,
            )
        elif source == "blockchain_com":
            df = fetch_blockchain_com(
                metrics=spec.get("metrics"),
                start=start, end=end,
                force_refresh=force_refresh,
            )
        elif source == "onchain":
            df = fetch_onchain_features(
                provider=spec.get("provider", "coinmetrics"),
                asset=spec.get("asset", "btc"),
                coinmetrics_metrics=spec.get("coinmetrics_metrics"),
                blockchain_com_metrics=spec.get("blockchain_com_metrics"),
                start=start, end=end,
                force_refresh=force_refresh,
            )
        elif source == "yfinance":
            df = fetch_macro_feature(
                ticker=spec["ticker"],
                start=start, end=end,
                force_refresh=force_refresh,
            )
        else:
            raise ValueError(f"Unknown feature source: '{source}'")

        # Rule 9: warn and skip empty DataFrames
        if df is None or df.empty:
            logger.warning(
                "fetch_secondary_features: source '%s' returned an empty "
                "DataFrame — skipping.", source,
            )
            continue

        results.append(df)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 8. ALIGNMENT & MERGING
# ══════════════════════════════════════════════════════════════════════════════

def merge_datasets(
    primary_df: pd.DataFrame,
    *secondary_dfs: pd.DataFrame,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Left-join secondary DataFrames onto the primary asset's date index.

    Lower-frequency features (e.g. monthly macro) are **forward-filled**
    (``ffill`` only — never ``bfill``).  Primary columns are never
    forward-filled.

    Parameters
    ----------
    primary_df : pd.DataFrame
        Must have a UTC ``DatetimeIndex`` named ``'Date'``.
    *secondary_dfs : pd.DataFrame
        Any number of secondary feature DataFrames.
    ffill_limit : int
        Maximum number of consecutive NaN values to forward-fill in
        secondary columns (default ``5``).

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

    # Forward-fill ONLY secondary columns (never bfill)
    secondary_cols = [c for c in merged.columns if c not in primary_cols]
    if secondary_cols:
        for col in secondary_cols:
            nans = merged[col].isnull()
            max_gap = nans.groupby((~nans).cumsum()).sum().max()
            if max_gap > ffill_limit:
                logger.warning(
                    "merge_datasets: secondary column '%s' has a gap of %d "
                    "consecutive NaNs (> %d limit).", col, max_gap, ffill_limit,
                )
        merged[secondary_cols] = merged[secondary_cols].ffill(limit=ffill_limit)

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 9. ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(
    ticker: str,
    feature_list: Optional[List[Dict[str, Any]]] = None,
    start: str = "2014-09-17",
    end: str = "2026-03-07",
    force_refresh: bool = False,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """End-to-end dataset loader: fetch → merge (raw, no targets).

    Parameters
    ----------
    ticker : str
        Primary asset yfinance ticker.
    feature_list : list of dict, optional
        Secondary features to fetch (see :func:`fetch_secondary_features`).
        If ``None``, only the primary OHLCV is returned.
    start, end : str
        Date range.
    force_refresh : bool
        If ``True``, skip all caches and re-download every source.
    ffill_limit : int
        Maximum consecutive NaN forward-fill for secondary columns
        (passed through to :func:`merge_datasets`).

    Returns
    -------
    pd.DataFrame
        Raw, aligned dataset ready for feature engineering.
    """
    logger.info("Loading dataset for %s [%s → %s]", ticker, start, end)

    # 1. Primary asset
    primary = fetch_primary_asset(ticker, start, end, force_refresh=force_refresh)

    # 2. Secondary features
    if feature_list:
        secondary = fetch_secondary_features(feature_list, start, end, force_refresh=force_refresh)
        df = merge_datasets(primary, *secondary, ffill_limit=ffill_limit)
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
        ``"feature_list"``, ``"force_refresh"``, and ``"ffill_limit"``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If ``config`` does not contain the required ``"ticker"`` key.
    """
    if "ticker" not in config:
        raise ValueError("Config dict must contain a 'ticker' key.")

    return load_dataset(
        ticker=config["ticker"],
        feature_list=config.get("feature_list"),
        start=config.get("start", "2014-09-17"),
        end=config.get("end", "2026-03-07"),
        force_refresh=config.get("force_refresh", False),
        ffill_limit=config.get("ffill_limit", 5),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 10. ASSET CATALOG & CONFIG PRESETS
# ══════════════════════════════════════════════════════════════════════════════

#: Registry of known assets and their available secondary data sources.
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

# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "fetch_primary_asset",
    "fetch_onchain_features",
    "fetch_macro_feature",
    "fetch_secondary_features",
    "merge_datasets",
    "load_dataset",
    "load_from_config",
    "ASSET_CATALOG",
    "DEFAULT_BTC_CONFIG",
    "DEFAULT_BTC_FULL_CONFIG",
    "DEFAULT_ETH_CONFIG",
    "DEFAULT_GLD_CONFIG",
    "DEFAULT_SPY_CONFIG",
    "DEFAULT_BTC_OHLCV_CONFIG",
    "DEFAULT_GLD_OHLCV_CONFIG",
    "DEFAULT_QQQ_OHLCV_CONFIG",
]
