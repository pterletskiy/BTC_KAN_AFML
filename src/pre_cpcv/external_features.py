"""
4.2) External Features
============================
Fetch, cache, and align macro, crypto-macro, and on-chain external features
to BTC's daily calendar. 29 features total: 20 macro, 1 crypto-macro, 8 on-chain.

All external series are forward-filled to BTC's trading calendar via
``pd.merge_asof(direction='backward')`` so each BTC day uses the most recent
available value with no look-ahead.
"""

import logging
import os
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cache paths for the assembled external matrix and the raw CoinMetrics pull.
CACHE_DIR = "cache/"
EXTERNAL_CACHE_FILE = "external_features.parquet"
ONCHAIN_CACHE_FILE = "onchain_raw.parquet"

# yfinance tickers for the macro group.
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

# Rolling return windows. Faster-moving assets get both 30-day and 14-day returns
# so MDA can pick the responsive horizon per fold; slow-moving variables (DXY,
# yields, yield curves) keep only the 30-day window.
RET_WINDOW, RET_WINDOW_SHORT = 30, 14  # ~1 crypto month, ~2 crypto weeks

# CoinMetrics Community-tier metrics used by the on-chain group.
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


# --- 1. Data fetching helpers ----------------------------------------------
# Thin yfinance wrapper that returns just the Close series and handles MultiIndex columns.
def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily Close prices from yfinance; return empty Series on failure."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        logger.warning("yfinance returned empty data for %s.", ticker)
        return pd.Series(dtype=float)

    # yfinance >= 0.2.31 returns MultiIndex columns; flatten to the ticker level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


# 2Y Treasury yield from FRED, with empty-return signal so callers can fall back.
def _fetch_us2y(start: str, end: str) -> pd.Series:
    """Fetch DGS2 from FRED. Returns empty Series on failure; caller derives from T10Y2Y spread."""
    try:
        import pandas_datareader.data as web
        from pandas_datareader.fred import FredReader
        FredReader.timeout = 60  # default 30s times out on long history pulls
        series = web.DataReader("DGS2", "fred", start, end)["DGS2"]
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series.dropna()
        if len(series) > 2000:
            return series
        logger.warning("FRED DGS2 returned only %d bars.", len(series))
    except Exception as e:
        logger.warning("FRED DGS2 failed: %s", e)

    return pd.Series(dtype=float)


# CoinMetrics ETH price fetcher with a three-step metric-priority fallback.
def _fetch_coinmetrics_eth(start: str, end: str) -> pd.Series:
    """Fetch daily ETH/USD from CoinMetrics; required for early-history ETH/BTC ratio.

    CoinMetrics serves ETH back to 2015-08-07 (Ethereum's Frontier launch),
    whereas yfinance's ETH-USD only begins around 2017-11-09. The earlier
    history avoids entire-test-partition NaN in early CPCV folds.

    The Community tier gates which metrics are free per asset. Tries metrics in
    priority order — ReferenceRateUSD, then PriceUSD, then a CapMrktCurUSD/SplyCur
    derivation — and returns the first one returning more than 100 rows. Empty
    Series if all three fail; the caller then falls back to yfinance.
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

    # Try the curated reference rate first, then the raw single-exchange price.
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

    # Last-resort derivation: price ≈ market cap / supply when neither rate metric serves.
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


# Forward-fill an external series onto BTC's daily calendar via merge_asof.
def _align_to_btc(
    series: pd.Series, btc_index: pd.DatetimeIndex, name: str,
) -> pd.Series:
    """Snap each BTC day to the most recent available value of the external series.

    Direction='backward' guarantees no look-ahead: weekends carry Friday's value,
    weekly releases persist until the next print.
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


# --- 2. Macro features ------------------------------------------------------
# 20 traditional-finance signals: DXY, yields, yield curves, VIX, and asset returns.
def compute_macro_features(btc_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute 20 macro features aligned to BTC's daily calendar."""
    # Buffer the start so rolling calculations have warm-up room.
    start = str(btc_index[0].date() - pd.Timedelta(days=250))
    end = str(btc_index[-1].date() + pd.Timedelta(days=1))

    features = pd.DataFrame(index=btc_index)

    print("[external] Fetching macro data from yfinance...")

    # Fetch every yfinance ticker into a raw dict; per-ticker failures are isolated.
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

    # 2Y yield comes from FRED separately so the spread-derived fallback can run.
    t0 = time.time()
    raw["us2y"] = _fetch_us2y(start, end)
    print(f"  {'us2y':>10s} ({'DGS2':>10s}): {len(raw['us2y'])} bars ({time.time()-t0:.1f}s)")

    # Align every series onto the BTC index before any feature computation.
    aligned = {name: _align_to_btc(series, btc_index, name) for name, series in raw.items()}

    # 1. DXY 30-day rate of change (%)
    dxy = aligned["dxy"]
    features["dxy_roc_30"] = (dxy / dxy.shift(RET_WINDOW) - 1.0) * 100.0

    # 2. US 10Y yield (must come before 2Y so the fallback can derive us2y).
    # yfinance sometimes returns yields as bps × 10 instead of percent; auto-rescale.
    us10y = aligned["us10y"]
    if us10y.median() > 10:
        us10y = us10y / 10.0
    features["us10y"] = us10y

    # 3 & 4. US 2Y yield + 2y10y curve, with FRED T10Y2Y spread as fallback when DGS2 is sparse.
    us2y = aligned["us2y"]
    if us2y.notna().sum() > 2000:
        if us2y.median() > 10:
            us2y = us2y / 10.0
        features["us2y"] = us2y
        features["yield_curve_2y10y"] = features["us10y"] - features["us2y"]
    else:
        # Spread-derived path: us2y = us10y - T10Y2Y spread.
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

    # 5. 10y30y yield curve (long-end slope).
    us30y = aligned["us30y"]
    if us30y.median() > 10:
        us30y = us30y / 10.0
    features["yield_curve_10y30y"] = us30y - features["us10y"]

    # 6. VIX (level)
    features["vix"] = aligned["vix"]

    # 7. S&P 500 rolling log returns at both horizons; MDA picks per fold.
    sp500 = aligned["sp500"]
    features["sp500_ret_30"] = np.log(sp500 / sp500.shift(RET_WINDOW))
    features["sp500_ret_14"] = np.log(sp500 / sp500.shift(RET_WINDOW_SHORT))

    # 8. Nasdaq rolling log returns (30-day and 14-day).
    nasdaq = aligned["nasdaq"]
    features["nasdaq_ret_30"] = np.log(nasdaq / nasdaq.shift(RET_WINDOW))
    features["nasdaq_ret_14"] = np.log(nasdaq / nasdaq.shift(RET_WINDOW_SHORT))

    # 9. Gold rolling log returns (30-day and 14-day).
    gold = aligned["gold"]
    features["gold_ret_30"] = np.log(gold / gold.shift(RET_WINDOW))
    features["gold_ret_14"] = np.log(gold / gold.shift(RET_WINDOW_SHORT))

    # 10. Silver rolling log returns (30-day and 14-day).
    silver = aligned["silver"]
    features["silver_ret_30"] = np.log(silver / silver.shift(RET_WINDOW))
    features["silver_ret_14"] = np.log(silver / silver.shift(RET_WINDOW_SHORT))

    # 11. Copper rolling log returns (30-day and 14-day).
    copper = aligned["copper"]
    features["copper_ret_30"] = np.log(copper / copper.shift(RET_WINDOW))
    features["copper_ret_14"] = np.log(copper / copper.shift(RET_WINDOW_SHORT))

    # 12. Oil rolling log returns (30-day and 14-day).
    oil = aligned["oil"]
    features["oil_ret_30"] = np.log(oil / oil.shift(RET_WINDOW))
    features["oil_ret_14"] = np.log(oil / oil.shift(RET_WINDOW_SHORT))

    # 13. Natural gas rolling log returns (30-day and 14-day).
    natgas = aligned["natgas"]
    features["natgas_ret_30"] = np.log(natgas / natgas.shift(RET_WINDOW))
    features["natgas_ret_14"] = np.log(natgas / natgas.shift(RET_WINDOW_SHORT))

    n_valid = features.notna().all(axis=1).sum()
    logger.info("Macro features: %d columns, %d/%d rows fully valid.",
                features.shape[1], n_valid, len(features))

    return features


# --- 3. Crypto-macro features ----------------------------------------------
# Single market-level cross-crypto signal (ETH/BTC ratio); distinct from on-chain.
def compute_crypto_macro_features(
    btc_close: pd.Series,
    btc_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute the single crypto-macro feature ``eth_btc_ratio`` aligned to BTC's calendar."""
    start = str(btc_index[0].date() - pd.Timedelta(days=30))
    end = str(btc_index[-1].date() + pd.Timedelta(days=1))

    features = pd.DataFrame(index=btc_index)

    print("[external] Fetching crypto-macro data...")

    # ETH source priority: CoinMetrics (history back to 2015-08-07) → yfinance (~2017-11-09 onward).
    # The earlier history is needed so eth_btc_ratio is non-NaN throughout the BTC series.
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

        # Align ETH to the BTC index and form the ratio.
        eth_aligned = _align_to_btc(eth, btc_index, "eth_close")
        features["eth_btc_ratio"] = eth_aligned / btc_close

    except Exception as e:
        logger.warning("ETH/BTC ratio failed: %s", e)
        features["eth_btc_ratio"] = np.nan

    return features


# --- 4. On-chain features --------------------------------------------------
# Raw CoinMetrics pull shared by all 8 on-chain derivations; shifted by 1 day for anti-leakage.
def _fetch_coinmetrics(
    metrics: list[str], start: str, end: str,
) -> pd.DataFrame:
    """Fetch BTC on-chain metrics from CoinMetrics Community API; raises on missing client."""
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

    # Normalise the time column to a clean tz-naive DatetimeIndex.
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].dt.tz_localize(None).dt.normalize()
    df = df.set_index("time")
    df.index.name = "Date"

    if "asset" in df.columns:
        df = df.drop(columns=["asset"])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Critical anti-leakage step: CoinMetrics reports end-of-day, but a model
    # predicting tomorrow from today's open cannot use today's close-time values.
    df = df.shift(1)

    elapsed = time.time() - t0
    print(f"  CoinMetrics: {len(df)} days, {len(df.columns)} metrics ({elapsed:.1f}s)")

    return df


# 8 blockchain-fundamental features derived from CoinMetrics raw metrics.
def compute_onchain_features(btc_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute 8 on-chain features from the CoinMetrics Community API, aligned to BTC's calendar."""
    start = str(btc_index[0].date() - pd.Timedelta(days=60))
    end = str(btc_index[-1].date() + pd.Timedelta(days=1))

    features = pd.DataFrame(index=btc_index)

    print("[external] Fetching on-chain data from CoinMetrics...")

    # Cache the raw CoinMetrics pull because it is the slow step; refresh on stale cache.
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

    # Align every raw metric onto the BTC index before deriving features.
    aligned = {col: _align_to_btc(raw[col], btc_index, col) for col in raw.columns}

    # 1. Active addresses: 14-day rate of change (turns level into stationary differential).
    adr = aligned.get("AdrActCnt", pd.Series(np.nan, index=btc_index))
    features["active_addr_roc_14"] = (adr / adr.shift(14) - 1.0) * 100.0

    # 2. Transaction count: 14-day rate of change.
    tx = aligned.get("TxCnt", pd.Series(np.nan, index=btc_index))
    features["tx_count_roc_14"] = (tx / tx.shift(14) - 1.0) * 100.0

    # 3. Hash rate: 30-day rate of change (slower mining capacity adjustments).
    hr = aligned.get("HashRate", pd.Series(np.nan, index=btc_index))
    features["hashrate_roc_30"] = (hr / hr.shift(30) - 1.0) * 100.0

    # 4. MVRV ratio (already stationary and mean-reverting; kept as level).
    features["mvrv"] = aligned.get("CapMVRVCur", pd.Series(np.nan, index=btc_index))

    # 5. Net exchange flow: inflows − outflows. Positive ⇒ selling pressure to exchanges.
    flow_in = aligned.get("FlowInExNtv", pd.Series(np.nan, index=btc_index))
    flow_out = aligned.get("FlowOutExNtv", pd.Series(np.nan, index=btc_index))
    features["net_exchange_flow"] = flow_in - flow_out

    # 6. Fee per transaction (stationary ratio, BTC per tx).
    fee = aligned.get("FeeTotNtv", pd.Series(np.nan, index=btc_index))
    tx_safe = tx.replace(0, np.nan)
    features["fee_per_tx"] = fee / tx_safe

    # 7. Exchange supply percentage (stationary ratio, fraction of supply on exchanges).
    sply_ex = aligned.get("SplyExNtv", pd.Series(np.nan, index=btc_index))
    sply_cur = aligned.get("SplyCur", pd.Series(np.nan, index=btc_index))
    sply_cur_safe = sply_cur.replace(0, np.nan)
    features["exchange_supply_pct"] = (sply_ex / sply_cur_safe) * 100.0

    # 8. Daily issuance (kept as level; step-function changes at halvings).
    features["issuance_ntv"] = aligned.get("IssTotNtv", pd.Series(np.nan, index=btc_index))

    n_valid = features.notna().all(axis=1).sum()
    logger.info("On-chain features: %d columns, %d/%d rows fully valid.",
                features.shape[1], n_valid, len(features))

    return features


# --- 5. Orchestration ------------------------------------------------------
# Public entry point: fetch every category, concatenate, cache to Parquet.
def build_external_features(
    btc_df: pd.DataFrame,
    include_macro: bool = True,
    include_crypto_macro: bool = True,
    include_onchain: bool = True,
) -> pd.DataFrame:
    """Fetch, compute, and cache all external features.

    Each category is independently toggleable; the on-chain fetch is wrapped in
    try/except so a missing ``coinmetrics-api-client`` does not break the pipeline.
    """
    cache_path = os.path.join(CACHE_DIR, EXTERNAL_CACHE_FILE)
    btc_index = btc_df.index
    btc_close = btc_df["Close"]

    # Cache hit requires both date-range AND column-set to match. The column check
    # prevents a stale cache from silently surviving a feature-list change.
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

    # Build each enabled category in turn and concatenate.
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

    # Surface column-level NaN concerns so a quick eyeball check catches feed issues.
    elapsed = time.time() - t_start
    n_valid = features.notna().all(axis=1).sum()
    print(f"\n[external] {features.shape[1]} features, "
          f"{n_valid}/{len(features)} fully valid rows, "
          f"fetched in {elapsed:.1f}s")
    print(f"[external] Columns: {list(features.columns)}")

    nan_pct = features.isna().mean() * 100
    for col in features.columns:
        if nan_pct[col] > 5:
            print(f"  ⚠ {col}: {nan_pct[col]:.1f}% NaN")

    # Persist the assembled matrix for the next run.
    os.makedirs(CACHE_DIR, exist_ok=True)
    features.to_parquet(cache_path)
    logger.info("External features cached to %s.", cache_path)

    return features


# Cache schema fingerprint; invalidates the cache when the configured feature set changes.
def _expected_external_columns(
    include_macro: bool,
    include_crypto_macro: bool,
    include_onchain: bool,
) -> list[str]:
    """Return the column set the cache must contain to count as a hit."""
    cols: list[str] = []
    if include_macro:
        cols += [
            "dxy_roc_30", "us10y", "us2y",
            "yield_curve_2y10y", "yield_curve_10y30y",
            "vix",
            "sp500_ret_30", "sp500_ret_14",
            "nasdaq_ret_30", "nasdaq_ret_14",
            "gold_ret_30", "gold_ret_14",
            "silver_ret_30", "silver_ret_14",
            "copper_ret_30", "copper_ret_14",
            "oil_ret_30", "oil_ret_14",
            "natgas_ret_30", "natgas_ret_14",
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