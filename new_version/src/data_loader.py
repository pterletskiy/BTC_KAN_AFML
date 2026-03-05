"""
data_loader.py — Modular data fetching for the MFW Asset Direction Predictor.

Follows the rules defined in skills/financial_data.md:
  §1  Modular Fetching Architecture
  §2  Log Returns & Target Calculation
  §3  Asynchronous Frequencies & Calendar Merging
  §4  Idempotent Caching
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from coinmetrics.api_client import CoinMetricsClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "raw"


# ═══════════════════════════════════════════════════════════════════════════
# §4 — Idempotent Caching Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _cache_path(source: str, key: str, start: str, end: str) -> Path:
    """Return a deterministic Parquet cache file path under ``data/raw/``."""
    safe_key = key.replace("/", "_").replace("^", "").replace("-", "_").replace(".", "_")
    return _CACHE_DIR / f"{source}_{safe_key}_{start}_{end}.parquet"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    """Read a cached DataFrame if the file exists, else return None."""
    if path.exists():
        logger.info("Cache hit: %s", path.name)
        return pd.read_parquet(path)
    return None


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    logger.info("Cached → %s", path.name)


# ═══════════════════════════════════════════════════════════════════════════
# §3 — Timezone Alignment Helper
# ═══════════════════════════════════════════════════════════════════════════
def _to_utc_midnight(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize any DatetimeIndex to UTC, midnight-aligned."""
    if index.tz is None:
        return index.tz_localize("UTC").normalize()
    return index.tz_convert("UTC").normalize()


# ═══════════════════════════════════════════════════════════════════════════
# §1 — Primary Asset (OHLCV via yfinance)
# ═══════════════════════════════════════════════════════════════════════════
def fetch_primary_asset(
    ticker: str,
    start: str = "2014-09-17",
    end: str = "2026-02-07",
    interval: str = "1d",
) -> pd.DataFrame:
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

    _write_cache(df, cache)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# §1 — Secondary Features: CoinMetrics On-Chain
# ═══════════════════════════════════════════════════════════════════════════
def fetch_coinmetrics(
    asset: str = "btc",
    metrics: Optional[List[str]] = None,
    start: str = "2014-09-17",
    end: str = "2026-02-07",
) -> pd.DataFrame:
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
        catalog = client.catalog_asset_metrics_v2(assets=asset).to_dataframe()
        metrics = catalog[catalog["frequency"] == "1d"]["metric"].tolist()
        logger.info("Resolved %d daily metrics for %s", len(metrics), asset)

    raw = client.get_asset_metrics(
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

    # Shift by 1 day to prevent look-ahead (matches notebook)
    df = df.shift(1)

    # Coerce all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    _write_cache(df, cache)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# §1 — Secondary Features: Blockchain.com On-Chain
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
    "mempool-size":       "bc_mempool_size",
}


def fetch_blockchain_com(
    metrics: Optional[Dict[str, str]] = None,
    start: str = "2014-09-17",
    end: str = "2026-02-07",
) -> pd.DataFrame:
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
            resp = requests.get(
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
        merged = tmp if merged is None else pd.merge(merged, tmp, on="date", how="outer")

    merged = merged.sort_values("date").set_index("date")

    # Blockchain.com timestamps can be intra-day → resample to daily
    merged = merged.resample("1D").last()

    merged.index = _to_utc_midnight(merged.index)
    merged.index.name = "Date"

    # 1-day shift to prevent look-ahead (same rationale as CoinMetrics)
    merged = merged.shift(1)

    _write_cache(merged, cache)
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# §1 — On-Chain Routing: coinmetrics | blockchain_com | both
# ═══════════════════════════════════════════════════════════════════════════
def fetch_onchain_features(
    provider: str = "coinmetrics",
    asset: str = "btc",
    coinmetrics_metrics: Optional[List[str]] = None,
    blockchain_com_metrics: Optional[Dict[str, str]] = None,
    start: str = "2014-09-17",
    end: str = "2026-02-07",
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
        "Choose 'coinmetrics', 'blockchain_com', or 'both'."
    )


# ═══════════════════════════════════════════════════════════════════════════
# §1 — Secondary Features: Macro Indicator via yfinance
# ═══════════════════════════════════════════════════════════════════════════
def fetch_macro_feature(
    ticker: str,
    start: str = "2014-09-17",
    end: str = "2026-02-07",
) -> pd.DataFrame:
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

    _write_cache(df, cache)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# §1 — General Secondary Features Router
# ═══════════════════════════════════════════════════════════════════════════
def fetch_secondary_features(
    feature_list: List[Dict[str, Any]],
    start: str = "2014-09-17",
    end: str = "2026-02-07",
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
# §2 — Log Returns & Target Calculation
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
    df = df.dropna(subset=["Price_Direction", "Log_Return"])

    return df


# ═══════════════════════════════════════════════════════════════════════════
# §3 — Calendar Merging
# ═══════════════════════════════════════════════════════════════════════════
def merge_datasets(
    primary_df: pd.DataFrame,
    *secondary_dfs: pd.DataFrame,
) -> pd.DataFrame:
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
        merged[secondary_cols] = merged[secondary_cols].ffill()

    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def load_dataset(
    ticker: str,
    feature_list: Optional[List[Dict[str, Any]]] = None,
    start: str = "2014-09-17",
    end: str = "2026-02-07",
) -> pd.DataFrame:
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
    return load_dataset(
        ticker=config["ticker"],
        feature_list=config.get("feature_list"),
        start=config.get("start", "2014-09-17"),
        end=config.get("end", "2026-02-07"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Interactive Notebook Config Builder
# ═══════════════════════════════════════════════════════════════════════════
def interactive_config() -> Dict[str, Any]:
    """Build a dataset configuration interactively via ``input()`` prompts.

    Designed to be called from a Jupyter Notebook cell.  Walks the user
    through selecting:

    1. Primary asset ticker (with common presets).
    2. Date range.
    3. On-chain data provider (CoinMetrics / Blockchain.com / both / none).
    4. Macro features (DXY, VIX, etc.).

    Returns
    -------
    dict
        A config dict compatible with :func:`load_from_config`.
    """
    print("=" * 60)
    print("  MFW Dataset Configuration Wizard")
    print("=" * 60)

    # --- Primary asset ---
    print("\n📈 Primary Asset")
    print("  Common presets:")
    print("    [1] BTC-USD  (Bitcoin)")
    print("    [2] ETH-USD  (Ethereum)")
    print("    [3] SPY      (S&P 500 ETF)")
    print("    [4] GLD      (Gold ETF)")
    print("    [5] Custom ticker")
    choice = input("  Select [1-5]: ").strip()
    preset_map = {"1": "BTC-USD", "2": "ETH-USD", "3": "SPY", "4": "GLD"}
    if choice in preset_map:
        ticker = preset_map[choice]
    else:
        ticker = input("  Enter custom yfinance ticker: ").strip()
    print(f"  → Primary asset: {ticker}")

    # --- Date range ---
    print("\n📅 Date Range")
    start = input("  Start date [YYYY-MM-DD] (default 2014-09-17): ").strip()
    start = start if start else "2014-09-17"
    end = input("  End date   [YYYY-MM-DD] (default 2026-02-07): ").strip()
    end = end if end else "2026-02-07"
    print(f"  → Period: {start} to {end}")

    feature_list: List[Dict[str, Any]] = []

    # --- On-chain features ---
    print("\n🔗 On-Chain Features")
    print("  [1] CoinMetrics only")
    print("  [2] Blockchain.com only")
    print("  [3] Both providers")
    print("  [4] None")
    oc_choice = input("  Select [1-4]: ").strip()
    if oc_choice in ("1", "2", "3"):
        provider_map = {"1": "coinmetrics", "2": "blockchain_com", "3": "both"}
        oc_asset = input(f"  Crypto asset for on-chain data (default 'btc'): ").strip()
        oc_asset = oc_asset if oc_asset else "btc"
        feature_list.append({
            "source": "onchain",
            "provider": provider_map[oc_choice],
            "asset": oc_asset,
        })

    # --- Macro features ---
    print("\n🌍 Macro Features (via yfinance)")
    print("  Enter tickers separated by commas, or press Enter to skip.")
    print("  Common: DX-Y.NYB (DXY), ^VIX, ^TNX (10Y yield), ^IRX (3M T-bill)")
    macro_input = input("  Tickers: ").strip()
    if macro_input:
        for t in [s.strip() for s in macro_input.split(",") if s.strip()]:
            feature_list.append({"source": "yfinance", "ticker": t})

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
    print(f"  {config}")
    return config


# ═══════════════════════════════════════════════════════════════════════════
# Default Preset Configs (Quick-Run Mode)
# ═══════════════════════════════════════════════════════════════════════════

#: Minimal BTC config — CoinMetrics on-chain only.
DEFAULT_BTC_CONFIG: Dict[str, Any] = {
    "ticker": "BTC-USD",
    "start": "2014-09-17",
    "end": "2026-02-07",
    "feature_list": [
        {
            "source": "onchain",
            "provider": "coinmetrics",
            "asset": "btc",
        },
    ],
}

#: Full BTC config — both on-chain providers + macro features.
DEFAULT_BTC_FULL_CONFIG: Dict[str, Any] = {
    "ticker": "BTC-USD",
    "start": "2014-09-17",
    "end": "2026-02-07",
    "feature_list": [
        {
            "source": "onchain",
            "provider": "both",
            "asset": "btc",
        },
        {"source": "yfinance", "ticker": "DX-Y.NYB"},   # DXY
        {"source": "yfinance", "ticker": "^VIX"},        # VIX
    ],
}
