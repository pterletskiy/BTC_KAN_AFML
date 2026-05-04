"""
4.2) External Features
============================
Fetch, cache, and align external data sources to BTC's daily calendar.

Features are organized into three groups:

Macro (13):
  Traditional finance signals reflecting the broader economic environment.
  dxy_roc_30, us2y, us10y, yield_curve_2y10y, yield_curve_10y30y, vix,
  sp500_ret_30, nasdaq_ret_30, gold_ret_30, silver_ret_30, copper_ret_30,
  oil_ret_30, natgas_ret_30

Crypto-Macro (1):
  Market-level cross-crypto signal (not blockchain fundamentals).
  eth_btc_ratio

On-Chain (8):
  Blockchain network activity from CoinMetrics Community API.
  active_addr_roc_14, tx_count_roc_14, hashrate_roc_30, mvrv,
  net_exchange_flow, fee_per_tx, exchange_supply_pct, issuance_ntv

Total: 22 external features.

All external data is forward-filled to BTC's trading calendar via
pd.merge_asof(direction='backward') so that each BTC day uses the
most recent available value (no look-ahead bias).
"""

import logging
import os
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = "cache/"
EXTERNAL_CACHE_FILE = "external_features.parquet"
ONCHAIN_CACHE_FILE = "onchain_raw.parquet"

# yfinance tickers for macro data
MACRO_TICKERS = {
    "dxy": "DX-Y.NYB",       # US Dollar Index
    "us10y": "^TNX",          # 10Y Treasury yield
    "us30y": "^TYX",          # 30Y Treasury yield
    "vix": "^VIX",            # CBOE Volatility Index
    "sp500": "^GSPC",         # S&P 500
    "nasdaq": "^IXIC",        # Nasdaq Composite
    "gold": "GC=F",           # Gold futures
    "silver": "SI=F",         # Silver futures
    "copper": "HG=F",         # Copper futures
    "oil": "CL=F",            # WTI Crude Oil futures
    "natgas": "NG=F",         # Natural gas futures
}

# rolling return period (~1 crypto month)
RET_WINDOW = 30

# CoinMetrics metrics to fetch (daily, community tier)
COINMETRICS_METRICS = [
    "AdrActCnt",       # active addresses
    "TxCnt",           # transaction count
    "HashRate",        # hash rate
    "CapMVRVCur",      # MVRV ratio
    "FlowInExNtv",     # exchange inflows (BTC)
    "FlowOutExNtv",    # exchange outflows (BTC)
    "FeeTotNtv",       # total fees (BTC)
    "SplyExNtv",       # supply on exchanges (BTC)
    "SplyCur",         # current supply
    "IssTotNtv",       # daily issuance (BTC)
]


# =====================================================================
# Data fetching helpers
# =====================================================================
def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch daily Close prices from yfinance."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        logger.warning("yfinance returned empty data for %s.", ticker)
        return pd.Series(dtype=float)

    # handle MultiIndex columns (newer yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def _fetch_us2y(start: str, end: str) -> pd.Series:
    """Fetch 2Y Treasury yield from FRED.

    Returns an empty Series on failure. The caller is expected to
    fall back to the T10Y2Y spread derivation in compute_macro_features.
    """
    try:
        import pandas_datareader.data as web
        from pandas_datareader.fred import FredReader
        FredReader.timeout = 60  # default 30s is too aggressive
        series = web.DataReader("DGS2", "fred", start, end)["DGS2"]
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series.dropna()
        if len(series) > 2000:
            return series
        logger.warning("FRED DGS2 returned only %d bars.", len(series))
    except (ImportError, Exception) as e:
        logger.warning("FRED DGS2 failed: %s", e)

    return pd.Series(dtype=float)


def _fetch_coinmetrics_eth(start: str, end: str) -> pd.Series:
    """Fetch daily ETH/USD prices from CoinMetrics.

    CoinMetrics serves ETH/USD history back to 2015-08-07 (the day
    of Ethereum's Frontier launch), substantially earlier than
    yfinance's ``ETH-USD`` ticker which only begins around
    2017-11-09. The earlier history is needed so that
    ``eth_btc_ratio`` has valid values throughout the BTC dataset's
    full span (2014 → present), preventing entire-test-partition NaN
    in the early CPCV folds.

    The Community tier gates which metrics are free per asset.
    ``ReferenceRateUSD`` (the multi-source weighted reference rate)
    is Pro-only for non-BTC assets at the time of writing, so this
    function tries metrics in priority order:

    1. ``ReferenceRateUSD`` — most curated, may be Pro-only for ETH.
    2. ``PriceUSD`` — single-exchange price, typically free.
    3. ``CapMrktCurUSD / SplyCur`` — derived market-cap / supply ratio.

    The first one that returns >100 rows wins. Returns an empty Series
    if all three fail; the caller then falls back to yfinance.

    Uses the same ``coinmetrics-api-client`` Community-tier endpoint
    already used for on-chain BTC metrics, so no additional
    credentials or dependencies are required.
    """
    try:
        from coinmetrics.api_client import CoinMetricsClient
    except ImportError:
        logger.warning(
            "coinmetrics-api-client not installed; "
            "cannot fetch ETH from CoinMetrics."
        )
        return pd.Series(dtype=float)

    client = CoinMetricsClient()

    # Try the curated reference rate first; fall back to price.
    for metric in ("ReferenceRateUSD", "PriceUSD"):
        try:
            result = client.get_asset_metrics(
                assets="eth",
                metrics=[metric],
                start_time=start,
                end_time=end,
                frequency="1d",
            )
            df = result.to_dataframe()
        except Exception as e:
            logger.warning(
                "CoinMetrics ETH %s request failed: %s", metric, e,
            )
            continue

        if df.empty or metric not in df.columns:
            logger.info(
                "CoinMetrics ETH %s returned no data; trying next metric.",
                metric,
            )
            continue

        df["time"] = pd.to_datetime(df["time"])
        df["time"] = df["time"].dt.tz_localize(None).dt.normalize()
        df = df.set_index("time").sort_index()

        eth = pd.to_numeric(df[metric], errors="coerce").dropna()
        eth.index.name = None
        eth.name = None

        if len(eth) > 100:
            logger.info(
                "CoinMetrics ETH %s returned %d daily rows.",
                metric, len(eth),
            )
            return eth

        logger.info(
            "CoinMetrics ETH %s returned only %d rows; trying next metric.",
            metric, len(eth),
        )

    # Last resort: derive price from market cap and supply.
    try:
        result = client.get_asset_metrics(
            assets="eth",
            metrics=["CapMrktCurUSD", "SplyCur"],
            start_time=start,
            end_time=end,
            frequency="1d",
        )
        df = result.to_dataframe()
    except Exception as e:
        logger.warning("CoinMetrics ETH cap/supply request failed: %s", e)
        return pd.Series(dtype=float)

    if df.empty or "CapMrktCurUSD" not in df.columns or "SplyCur" not in df.columns:
        logger.warning(
            "CoinMetrics ETH cap/supply derivation: missing required metrics."
        )
        return pd.Series(dtype=float)

    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].dt.tz_localize(None).dt.normalize()
    df = df.set_index("time").sort_index()

    cap = pd.to_numeric(df["CapMrktCurUSD"], errors="coerce")
    sply = pd.to_numeric(df["SplyCur"], errors="coerce")
    eth = (cap / sply).dropna()
    eth.index.name = None
    eth.name = None

    if len(eth) > 100:
        logger.info(
            "CoinMetrics ETH derived from CapMrktCurUSD / SplyCur: "
            "%d daily rows.",
            len(eth),
        )
        return eth

    logger.warning(
        "CoinMetrics ETH cap/supply derivation: only %d rows; "
        "falling back to yfinance.",
        len(eth),
    )
    return pd.Series(dtype=float)


def _align_to_btc(
    series: pd.Series, btc_index: pd.DatetimeIndex, name: str,
) -> pd.Series:
    """Forward-fill an external series to BTC's daily calendar.

    Uses merge_asof with direction='backward' so each BTC day gets the
    most recent available value from the external source.
    """
    if series.empty:
        return pd.Series(np.nan, index=btc_index, name=name)

    df_ext = series.to_frame("value").reset_index()
    df_ext.columns = ["date", "value"]
    df_ext["date"] = pd.to_datetime(df_ext["date"])

    df_btc = pd.DataFrame({"date": btc_index})

    merged = pd.merge_asof(
        df_btc.sort_values("date"),
        df_ext.sort_values("date"),
        on="date",
        direction="backward",
    )

    result = merged.set_index("date")["value"]
    result.name = name
    return result


# =====================================================================
# Macro Features (13)
# =====================================================================
def compute_macro_features(btc_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute 13 macro features aligned to BTC's daily calendar.

    Parameters
    ----------
    btc_index : pd.DatetimeIndex
        BTC's daily trading dates (the master timeline).

    Returns
    -------
    pd.DataFrame
        13 macro feature columns indexed on btc_index.
    """
    # buffer for rolling calculations
    start = str(btc_index[0].date() - pd.Timedelta(days=250))
    end = str(btc_index[-1].date() + pd.Timedelta(days=1))

    features = pd.DataFrame(index=btc_index)

    print("[external] Fetching macro data from yfinance...")

    # fetch all yfinance tickers
    raw = {}
    for name, ticker in MACRO_TICKERS.items():
        t0 = time.time()
        try:
            raw[name] = _fetch_yfinance(ticker, start, end)
            n = len(raw[name])
            print(f"  {name:>10s} ({ticker:>10s}): {n} bars ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.warning("Failed to fetch %s (%s): %s", name, ticker, e)
            raw[name] = pd.Series(dtype=float)

    # fetch 2Y yield separately (FRED)
    t0 = time.time()
    raw["us2y"] = _fetch_us2y(start, end)
    print(f"  {'us2y':>10s} ({'DGS2':>10s}): {len(raw['us2y'])} bars ({time.time()-t0:.1f}s)")

    # align all to BTC calendar
    aligned = {name: _align_to_btc(series, btc_index, name) for name, series in raw.items()}

    # 1. DXY 30-day rate of change (%)
    dxy = aligned["dxy"]
    features["dxy_roc_30"] = (dxy / dxy.shift(RET_WINDOW) - 1.0) * 100.0

    # 2. US 10Y yield (must come before 2Y so fallback can derive us2y)
    us10y = aligned["us10y"]
    if us10y.median() > 10:
        us10y = us10y / 10.0
    features["us10y"] = us10y

    # 3 & 4. US 2Y yield + yield curve 2y10y (with FRED spread fallback)
    us2y = aligned["us2y"]
    if us2y.notna().sum() > 2000:
        if us2y.median() > 10:
            us2y = us2y / 10.0
        features["us2y"] = us2y
        features["yield_curve_2y10y"] = features["us10y"] - features["us2y"]
    else:
        # fallback: fetch the T10Y2Y spread directly from FRED
        try:
            import pandas_datareader.data as web
            spread = web.DataReader("T10Y2Y", "fred", start, end)["T10Y2Y"]
            spread.index = pd.to_datetime(spread.index).tz_localize(None)
            spread_aligned = _align_to_btc(spread.dropna(), btc_index, "spread")
            features["yield_curve_2y10y"] = spread_aligned
            features["us2y"] = features["us10y"] - spread_aligned
            logger.info("Derived us2y from T10Y2Y spread (%d valid).",
                        features["us2y"].notna().sum())
        except Exception as e:
            logger.warning("T10Y2Y fallback failed: %s. us2y will be NaN.", e)
            features["us2y"] = np.nan
            features["yield_curve_2y10y"] = np.nan

    # 5. Yield curve 10y30y
    us30y = aligned["us30y"]
    if us30y.median() > 10:
        us30y = us30y / 10.0
    features["yield_curve_10y30y"] = us30y - features["us10y"]

    # 6. VIX (level)
    features["vix"] = aligned["vix"]

    # 7. S&P 500 rolling 30-day log return
    sp500 = aligned["sp500"]
    features["sp500_ret_30"] = np.log(sp500 / sp500.shift(RET_WINDOW))

    # 8. Nasdaq rolling 30-day log return
    nasdaq = aligned["nasdaq"]
    features["nasdaq_ret_30"] = np.log(nasdaq / nasdaq.shift(RET_WINDOW))

    # 9. Gold rolling 30-day log return
    gold = aligned["gold"]
    features["gold_ret_30"] = np.log(gold / gold.shift(RET_WINDOW))

    # 10. Silver rolling 30-day log return
    silver = aligned["silver"]
    features["silver_ret_30"] = np.log(silver / silver.shift(RET_WINDOW))

    # 11. Copper rolling 30-day log return
    copper = aligned["copper"]
    features["copper_ret_30"] = np.log(copper / copper.shift(RET_WINDOW))

    # 12. Oil rolling 30-day log return
    oil = aligned["oil"]
    features["oil_ret_30"] = np.log(oil / oil.shift(RET_WINDOW))

    # 13. Natural gas rolling 30-day log return
    natgas = aligned["natgas"]
    features["natgas_ret_30"] = np.log(natgas / natgas.shift(RET_WINDOW))

    n_valid = features.notna().all(axis=1).sum()
    logger.info("Macro features: %d columns, %d/%d rows fully valid.",
                features.shape[1], n_valid, len(features))

    return features


# =====================================================================
# Crypto-Macro Features (1)
# =====================================================================
def compute_crypto_macro_features(
    btc_close: pd.Series,
    btc_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute 1 crypto-macro feature aligned to BTC's calendar.

    Currently produces only ETH/BTC ratio. ``btc_dominance`` was
    removed because the CoinGecko endpoint returned BTC market cap in
    USD (not the bounded dominance percentage), and the proxy fallback
    was a price-correlated approximation. ETH/BTC ratio carries the
    alt-rotation signal alone, distinct from blockchain fundamentals
    in compute_onchain_features.

    Parameters
    ----------
    btc_close : pd.Series
        BTC daily close prices.
    btc_index : pd.DatetimeIndex
        BTC's daily trading dates.

    Returns
    -------
    pd.DataFrame
        Single-column ('eth_btc_ratio') DataFrame indexed on btc_index.
    """
    start = str(btc_index[0].date() - pd.Timedelta(days=30))
    end = str(btc_index[-1].date() + pd.Timedelta(days=1))

    features = pd.DataFrame(index=btc_index)

    print("[external] Fetching crypto-macro data...")

    # ETH/BTC ratio
    # Source priority: CoinMetrics first (ETH price back to 2015-08-07,
    # the Ethereum Frontier launch date), falling back to yfinance
    # ETH-USD (history back to ~2017-11-09 only). The earlier
    # CoinMetrics history is needed to avoid entire-test-partition NaN
    # in eth_btc_ratio for the early CPCV folds. Both sources return
    # daily-close-equivalent USD prices, so the ratio computation is
    # source-agnostic once the series is aligned to btc_index.
    # CoinMetrics' Community-tier API is already used elsewhere in
    # this module for on-chain BTC metrics, so no new credentials
    # are required. The CoinMetrics helper internally tries
    # ReferenceRateUSD, then PriceUSD, then a CapMrktCurUSD/SplyCur
    # derivation, before signalling the caller to fall back to
    # yfinance.
    t0 = time.time()
    try:
        eth = _fetch_coinmetrics_eth(start, end)
        if len(eth) > 100:
            print(
                f"  {'ETH-USD':>10s}: {len(eth)} bars from CoinMetrics "
                f"(first valid: {eth.index[0].date()}) "
                f"({time.time()-t0:.1f}s)"
            )
        else:
            logger.warning(
                "CoinMetrics ETH returned only %d rows; "
                "falling back to yfinance.",
                len(eth),
            )
            eth = _fetch_yfinance("ETH-USD", start, end)
            print(
                f"  {'ETH-USD':>10s}: {len(eth)} bars from yfinance "
                f"(fallback) ({time.time()-t0:.1f}s)"
            )

        eth_aligned = _align_to_btc(eth, btc_index, "eth_close")
        features["eth_btc_ratio"] = eth_aligned / btc_close.reindex(btc_index)

    except Exception as e:
        logger.warning("ETH/BTC ratio failed: %s", e)
        features["eth_btc_ratio"] = np.nan

    return features


# =====================================================================
# On-Chain Features (8)
# =====================================================================
def _fetch_coinmetrics(
    metrics: list[str], start: str, end: str,
) -> pd.DataFrame:
    """Fetch BTC on-chain metrics from CoinMetrics Community API."""
    try:
        from coinmetrics.api_client import CoinMetricsClient
    except ImportError:
        raise ImportError(
            "coinmetrics-api-client not installed. "
            "Run: pip install coinmetrics-api-client"
        )

    client = CoinMetricsClient()

    print(f"  Fetching {len(metrics)} metrics from CoinMetrics...")
    t0 = time.time()

    result = client.get_asset_metrics(
        assets="btc",
        metrics=metrics,
        start_time=start,
        end_time=end,
        frequency="1d",
    )
    df = result.to_dataframe()

    # parse timestamps
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].dt.tz_localize(None).dt.normalize()
    df = df.set_index("time")
    df.index.name = "Date"

    if "asset" in df.columns:
        df = df.drop(columns=["asset"])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # shift by 1 day to avoid look-ahead bias
    # (CoinMetrics reports end-of-day; we align to next day's open)
    df = df.shift(1)

    elapsed = time.time() - t0
    print(f"  CoinMetrics: {len(df)} days, {len(df.columns)} metrics ({elapsed:.1f}s)")

    return df


def compute_onchain_features(btc_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute 8 on-chain features from CoinMetrics Community API.

    These are blockchain fundamentals, distinct from crypto-macro
    market-level signals.

    Parameters
    ----------
    btc_index : pd.DatetimeIndex
        BTC's daily trading dates (the master timeline).

    Returns
    -------
    pd.DataFrame
        8 on-chain feature columns indexed on btc_index.
    """
    start = str(btc_index[0].date() - pd.Timedelta(days=60))
    end = str(btc_index[-1].date() + pd.Timedelta(days=1))

    features = pd.DataFrame(index=btc_index)

    print("[external] Fetching on-chain data from CoinMetrics...")

    # fetch raw metrics (with cache)
    raw_cache = os.path.join(CACHE_DIR, ONCHAIN_CACHE_FILE)

    if os.path.exists(raw_cache):
        raw = pd.read_parquet(raw_cache)
        if raw.index[0] <= btc_index[0] and raw.index[-1] >= btc_index[-1] - pd.Timedelta(days=3):
            print(f"  On-chain raw data loaded from cache ({len(raw)} days)")
        else:
            raw = _fetch_coinmetrics(COINMETRICS_METRICS, start, end)
            os.makedirs(CACHE_DIR, exist_ok=True)
            raw.to_parquet(raw_cache)
    else:
        raw = _fetch_coinmetrics(COINMETRICS_METRICS, start, end)
        os.makedirs(CACHE_DIR, exist_ok=True)
        raw.to_parquet(raw_cache)

    # align to BTC calendar
    aligned = {col: _align_to_btc(raw[col], btc_index, col) for col in raw.columns}

    # 1. Active addresses: 14-day rate of change
    adr = aligned.get("AdrActCnt", pd.Series(np.nan, index=btc_index))
    features["active_addr_roc_14"] = (adr / adr.shift(14) - 1.0) * 100.0

    # 2. Transaction count: 14-day rate of change
    tx = aligned.get("TxCnt", pd.Series(np.nan, index=btc_index))
    features["tx_count_roc_14"] = (tx / tx.shift(14) - 1.0) * 100.0

    # 3. Hash rate: 30-day rate of change
    hr = aligned.get("HashRate", pd.Series(np.nan, index=btc_index))
    features["hashrate_roc_30"] = (hr / hr.shift(30) - 1.0) * 100.0

    # 4. MVRV ratio (level, already stationary and mean-reverting)
    features["mvrv"] = aligned.get("CapMVRVCur", pd.Series(np.nan, index=btc_index))

    # 5. Net exchange flow: inflows minus outflows (already stationary)
    flow_in = aligned.get("FlowInExNtv", pd.Series(0.0, index=btc_index))
    flow_out = aligned.get("FlowOutExNtv", pd.Series(0.0, index=btc_index))
    features["net_exchange_flow"] = flow_in - flow_out

    # 6. Fee per transaction (ratio, stationary)
    fee = aligned.get("FeeTotNtv", pd.Series(np.nan, index=btc_index))
    tx_safe = tx.replace(0, np.nan)
    features["fee_per_tx"] = fee / tx_safe

    # 7. Exchange supply percentage (ratio, stationary)
    sply_ex = aligned.get("SplyExNtv", pd.Series(np.nan, index=btc_index))
    sply_cur = aligned.get("SplyCur", pd.Series(np.nan, index=btc_index))
    sply_cur_safe = sply_cur.replace(0, np.nan)
    features["exchange_supply_pct"] = (sply_ex / sply_cur_safe) * 100.0

    # 8. Daily issuance (BTC, level, changes only at halvings)
    features["issuance_ntv"] = aligned.get("IssTotNtv", pd.Series(np.nan, index=btc_index))

    n_valid = features.notna().all(axis=1).sum()
    logger.info("On-chain features: %d columns, %d/%d rows fully valid.",
                features.shape[1], n_valid, len(features))

    return features


# =====================================================================
# Orchestration
# =====================================================================
def build_external_features(
    btc_df: pd.DataFrame,
    include_macro: bool = True,
    include_crypto_macro: bool = True,
    include_onchain: bool = True,
) -> pd.DataFrame:
    """Fetch, compute, and cache all external features.

    Total feature count: 13 macro + 1 crypto-macro + 8 on-chain = 22.

    Parameters
    ----------
    btc_df : pd.DataFrame
        BTC OHLCV DataFrame with DatetimeIndex (from data_loader).
    include_macro : bool
        Whether to include macro features (traditional finance).
    include_crypto_macro : bool
        Whether to include crypto-macro features (ETH/BTC ratio).
    include_onchain : bool
        Whether to include on-chain features (requires coinmetrics-api-client).

    Returns
    -------
    pd.DataFrame
        External features aligned to BTC's daily calendar.
    """
    cache_path = os.path.join(CACHE_DIR, EXTERNAL_CACHE_FILE)
    btc_index = btc_df.index
    btc_close = btc_df["Close"]

    # check cache (date-range AND column-set match, the latter so dropping
    # a feature like btc_dominance does not silently return a stale cache)
    expected_cols = _expected_external_columns(
        include_macro, include_crypto_macro, include_onchain,
    )
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if (cached.index[0] == btc_index[0]
                and cached.index[-1] == btc_index[-1]
                and set(cached.columns) == set(expected_cols)):
            logger.info("External features loaded from cache (%d cols).", cached.shape[1])
            print(f"[external] Loaded from cache: {list(cached.columns)}")
            return cached
        if set(cached.columns) != set(expected_cols):
            logger.info(
                "External cache column mismatch (cached=%s, expected=%s), refetching.",
                sorted(cached.columns), sorted(expected_cols),
            )
        else:
            logger.info("External cache date range mismatch, refetching.")

    print("=" * 60)
    print("Fetching External Features")
    print("=" * 60)
    t_start = time.time()

    parts = []

    if include_macro:
        parts.append(compute_macro_features(btc_index))

    if include_crypto_macro:
        parts.append(compute_crypto_macro_features(btc_close, btc_index))

    if include_onchain:
        try:
            parts.append(compute_onchain_features(btc_index))
        except ImportError as e:
            print(f"  ⚠ Skipping on-chain features: {e}")
            logger.warning("On-chain features skipped: %s", e)

    if not parts:
        return pd.DataFrame(index=btc_index)

    features = pd.concat(parts, axis=1)

    # report
    elapsed = time.time() - t_start
    n_valid = features.notna().all(axis=1).sum()
    print(f"\n[external] {features.shape[1]} features, "
          f"{n_valid}/{len(features)} fully valid rows, "
          f"fetched in {elapsed:.1f}s")
    print(f"[external] Columns: {list(features.columns)}")

    # NaN summary
    nan_pct = features.isna().mean() * 100
    for col in features.columns:
        if nan_pct[col] > 5:
            print(f"  ⚠ {col}: {nan_pct[col]:.1f}% NaN")

    # cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    features.to_parquet(cache_path)
    logger.info("External features cached to %s.", cache_path)

    return features


def _expected_external_columns(
    include_macro: bool,
    include_crypto_macro: bool,
    include_onchain: bool,
) -> list[str]:
    """Return the column set the cache must match for a hit.

    Used by build_external_features to invalidate the cache when the
    feature set changes (e.g., dropping btc_dominance), since the prior
    cache check only compared index endpoints.
    """
    cols: list[str] = []
    if include_macro:
        cols += [
            "dxy_roc_30", "us10y", "us2y",
            "yield_curve_2y10y", "yield_curve_10y30y",
            "vix",
            "sp500_ret_30", "nasdaq_ret_30",
            "gold_ret_30", "silver_ret_30", "copper_ret_30",
            "oil_ret_30", "natgas_ret_30",
        ]
    if include_crypto_macro:
        cols += ["eth_btc_ratio"]
    if include_onchain:
        cols += [
            "active_addr_roc_14", "tx_count_roc_14", "hashrate_roc_30",
            "mvrv", "net_exchange_flow", "fee_per_tx",
            "exchange_supply_pct", "issuance_ntv",
        ]
    return cols