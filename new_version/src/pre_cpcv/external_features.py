"""
4.2) External Features
==================
Fetch, cache, and align external data sources to BTC's daily calendar.

Macro features (10):
  dxy_roc_21, us2y, us10y, yield_curve_2y10y, yield_curve_10y30y,
  vix, sp500_ret_21, nasdaq_ret_21, gold_ret_21, oil_ret_21

Crypto-specific features (2):
  eth_btc_ratio, btc_dominance

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

# rolling return period
RET_WINDOW = 21

# CoinGecko API
COINGECKO_BTC_DOM_URL = (
    "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    "?vs_currency=usd&days=max&interval=daily"
)


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
    """Fetch 2Y Treasury yield. Falls back through multiple sources."""
    # attempt 1: pandas_datareader from FRED (full history)
    try:
        import pandas_datareader.data as web
        from pandas_datareader.fred import FredReader
        FredReader.timeout = 60  # increase from default 30s
        series = web.DataReader("DGS2", "fred", start, end)["DGS2"]
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series.dropna()
        if len(series) > 2000:
            return series
        logger.warning("FRED DGS2 returned only %d bars.", len(series))
    except (ImportError, Exception) as e:
        logger.warning("FRED DGS2 failed: %s", e)

    # attempt 2: derive from FRED T10Y2Y spread + 10Y yield
    try:
        import pandas_datareader.data as web
        from pandas_datareader.fred import FredReader
        FredReader.timeout = 60  # increase from default 30s
        spread = web.DataReader("T10Y2Y", "fred", start, end)["T10Y2Y"]
        spread.index = pd.to_datetime(spread.index).tz_localize(None)
        spread = spread.dropna()
        if len(spread) > 2000:
            logger.info("Using T10Y2Y + 10Y to derive 2Y yield (%d bars).", len(spread))
            return spread  # return spread directly, handle in caller
    except (ImportError, Exception) as e:
        logger.warning("FRED T10Y2Y failed: %s", e)

    # attempt 3: yfinance futures
    try:
        series = _fetch_yfinance("2YY=F", start, end)
        if len(series) > 100:
            return series
    except Exception:
        pass

    logger.warning("Could not fetch 2Y yield from any source. Column will be NaN.")
    return pd.Series(dtype=float)


def _fetch_btc_dominance(start: str, end: str) -> pd.Series:
    """Fetch BTC dominance from CoinGecko API."""
    try:
        import requests

        # fetch BTC market cap history
        resp = requests.get(COINGECKO_BTC_DOM_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        btc_mcap = pd.DataFrame(data["market_caps"], columns=["timestamp", "btc_mcap"])
        btc_mcap["date"] = pd.to_datetime(btc_mcap["timestamp"], unit="ms").dt.normalize()
        btc_mcap = btc_mcap.drop_duplicates("date", keep="last").set_index("date")["btc_mcap"]
        btc_mcap.index = btc_mcap.index.tz_localize(None)

        # get current dominance percentage for scaling
        time.sleep(1.5)  # CoinGecko rate limit
        resp_g = requests.get("https://api.coingecko.com/api/v3/global", timeout=30)
        resp_g.raise_for_status()
        current_dom = resp_g.json()["data"]["market_cap_percentage"]["btc"]

        # current total market cap
        current_total = btc_mcap.iloc[-1] / (current_dom / 100.0)

        # approximate historical total market cap assuming BTC mcap / total is
        # roughly proportional (this is an approximation for early years)
        # Better approach: use the ratio trend
        btc_mcap = btc_mcap.loc[start:end]

        if len(btc_mcap) > 100:
            logger.info("BTC market cap from CoinGecko: %d days. Current dominance: %.1f%%",
                        len(btc_mcap), current_dom)
            # return raw market cap; the ratio with total is what matters
            # and feature selection will handle it
            return btc_mcap

        return pd.Series(dtype=float)

    except Exception as e:
        logger.warning("CoinGecko fetch failed: %s", e)
        return pd.Series(dtype=float)


def _compute_btc_dominance_proxy(btc_close: pd.Series, start: str, end: str) -> pd.Series:
    """Approximate BTC dominance using inverse ETH/BTC ratio.

    When CoinGecko API is unavailable, 1/(1 + ETH/BTC) serves as a
    proxy: when ETH/BTC rises, BTC dominance falls.
    """
    try:
        eth = _fetch_yfinance("ETH-USD", start, end)
        if len(eth) < 100:
            return pd.Series(dtype=float)

        common = btc_close.index.intersection(eth.index)
        if len(common) < 100:
            return pd.Series(dtype=float)

        ratio = eth.loc[common] / btc_close.loc[common]
        dominance_proxy = 100.0 / (1.0 + ratio)

        return dominance_proxy

    except Exception as e:
        logger.warning("BTC dominance proxy failed: %s", e)
        return pd.Series(dtype=float)


def _align_to_btc(series: pd.Series, btc_index: pd.DatetimeIndex, name: str) -> pd.Series:
    """Forward-fill an external series to BTC's daily calendar."""
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
# Macro features
# =====================================================================
def compute_macro_features(btc_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute 10 macro features aligned to BTC's daily calendar.

    Parameters
    ----------
    btc_index : pd.DatetimeIndex
        BTC's daily trading dates (the master timeline).

    Returns
    -------
    pd.DataFrame
        10 macro feature columns indexed on btc_index.
    """
    # buffer for rolling calculations
    start = str(btc_index[0].date() - pd.Timedelta(days=250))
    end = str(btc_index[-1].date() + pd.Timedelta(days=1))

    features = pd.DataFrame(index=btc_index)

    print("[external] Fetching macro data from yfinance...")

    # ── fetch all tickers ─────────────────────────────────────────────
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

    # fetch 2Y yield separately
    t0 = time.time()
    raw["us2y"] = _fetch_us2y(start, end)
    print(f"  {'us2y':>10s} ({'DGS2':>10s}): {len(raw['us2y'])} bars ({time.time()-t0:.1f}s)")

    # ── align all to BTC calendar ─────────────────────────────────────
    aligned = {name: _align_to_btc(series, btc_index, name) for name, series in raw.items()}

    # ── compute features ──────────────────────────────────────────────

    # 1. DXY 21-day rate of change (%)
    dxy = aligned["dxy"]
    features["dxy_roc_21"] = (dxy / dxy.shift(RET_WINDOW) - 1.0) * 100.0

    # 2. US 10Y yield (must come before 2Y so fallback can derive us2y)
    us10y = aligned["us10y"]
    if us10y.median() > 10:
        us10y = us10y / 10.0
    features["us10y"] = us10y

    # 3. US 2Y yield
    us2y = aligned["us2y"]
    if us2y.notna().sum() > 2000:
        if us2y.median() > 10:
            us2y = us2y / 10.0
        features["us2y"] = us2y
        features["yield_curve_2y10y"] = features["us10y"] - features["us2y"]
    else:
        # fallback: fetch spread directly from FRED
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

    # 4. Yield curve: 30Y minus 10Y spread
    us30y = aligned["us30y"]
    if us30y.median() > 10:
        us30y = us30y / 10.0
    features["yield_curve_10y30y"] = us30y - features["us10y"]

    # 5. VIX (level)
    features["vix"] = aligned["vix"]

    # 6. S&P 500 rolling 21-day log return
    sp500 = aligned["sp500"]
    features["sp500_ret_21"] = np.log(sp500 / sp500.shift(RET_WINDOW))

    # 7. Nasdaq rolling 21-day log return
    nasdaq = aligned["nasdaq"]
    features["nasdaq_ret_21"] = np.log(nasdaq / nasdaq.shift(RET_WINDOW))

    # 8. Gold rolling 21-day log return
    gold = aligned["gold"]
    features["gold_ret_21"] = np.log(gold / gold.shift(RET_WINDOW))

    # 9. Silver rolling 21-day log return
    silver = aligned["silver"]
    features["silver_ret_21"] = np.log(silver / silver.shift(RET_WINDOW))

    # 10. Copper rolling 21-day log return
    copper = aligned["copper"]
    features["copper_ret_21"] = np.log(copper / copper.shift(RET_WINDOW))
    
    # 11. Oil rolling 21-day log return
    oil = aligned["oil"]
    features["oil_ret_21"] = np.log(oil / oil.shift(RET_WINDOW))

    # 12. Natural gas rolling 21-day log return
    natgas = aligned["natgas"]
    features["natgas_ret_21"] = np.log(natgas / natgas.shift(RET_WINDOW))



    n_valid = features.notna().all(axis=1).sum()
    logger.info("Macro features: %d columns, %d/%d rows fully valid.",
                features.shape[1], n_valid, len(features))

    return features


# =====================================================================
# Crypto-specific features
# =====================================================================
def compute_crypto_features(
    btc_close: pd.Series,
    btc_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute 2 crypto-specific features aligned to BTC's calendar.

    Parameters
    ----------
    btc_close : pd.Series
        BTC daily close prices.
    btc_index : pd.DatetimeIndex
        BTC's daily trading dates.

    Returns
    -------
    pd.DataFrame
        2 crypto feature columns indexed on btc_index.
    """
    start = str(btc_index[0].date() - pd.Timedelta(days=30))
    end = str(btc_index[-1].date() + pd.Timedelta(days=1))

    features = pd.DataFrame(index=btc_index)

    print("[external] Fetching crypto data...")

    # ── 1. ETH/BTC ratio ─────────────────────────────────────────────
    t0 = time.time()
    try:
        eth = _fetch_yfinance("ETH-USD", start, end)
        print(f"  {'ETH-USD':>10s}: {len(eth)} bars ({time.time()-t0:.1f}s)")

        eth_aligned = _align_to_btc(eth, btc_index, "eth_close")
        features["eth_btc_ratio"] = eth_aligned / btc_close.reindex(btc_index)

    except Exception as e:
        logger.warning("ETH/BTC ratio failed: %s", e)
        features["eth_btc_ratio"] = np.nan

    # ── 2. BTC dominance ──────────────────────────────────────────────
    t0 = time.time()
    print("  Fetching BTC dominance from CoinGecko...")

    btc_dom = _fetch_btc_dominance(start, end)

    if btc_dom.empty or btc_dom.notna().sum() < 100:
        print("  CoinGecko unavailable, computing dominance proxy from ETH/BTC...")
        btc_dom = _compute_btc_dominance_proxy(btc_close, start, end)

    if not btc_dom.empty and btc_dom.notna().sum() > 100:
        features["btc_dominance"] = _align_to_btc(btc_dom, btc_index, "btc_dominance")
        print(f"  BTC dominance: {features['btc_dominance'].notna().sum()} valid days ({time.time()-t0:.1f}s)")
    else:
        features["btc_dominance"] = np.nan
        logger.warning("BTC dominance unavailable. Column will be NaN.")

    return features


# =====================================================================
# On-chain features (CoinMetrics Community API)
# =====================================================================

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

ONCHAIN_CACHE_FILE = "onchain_raw.parquet"


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

    # drop non-metric columns
    if "asset" in df.columns:
        df = df.drop(columns=["asset"])

    # convert to numeric
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

    # ── fetch raw metrics (with cache) ────────────────────────────────
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

    # ── align to BTC calendar ─────────────────────────────────────────
    aligned = {}
    for col in raw.columns:
        aligned[col] = _align_to_btc(raw[col], btc_index, col)

    # ── compute derived features ──────────────────────────────────────

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
    include_crypto: bool = True,
    include_onchain: bool = True,
) -> pd.DataFrame:
    """Fetch, compute, and cache all external features.

    Parameters
    ----------
    btc_df : pd.DataFrame
        BTC OHLCV DataFrame with DatetimeIndex (from data_loader).
    include_macro : bool
        Whether to include macro features.
    include_crypto : bool
        Whether to include crypto-specific features.
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

    # check cache
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if (cached.index[0] == btc_index[0]
                and cached.index[-1] == btc_index[-1]):
            logger.info("External features loaded from cache (%d cols).", cached.shape[1])
            print(f"[external] Loaded from cache: {list(cached.columns)}")
            return cached
        logger.info("External cache date range mismatch, refetching.")

    print("=" * 60)
    print("Fetching External Features")
    print("=" * 60)
    t_start = time.time()

    parts = []

    if include_macro:
        macro = compute_macro_features(btc_index)
        parts.append(macro)

    if include_crypto:
        crypto = compute_crypto_features(btc_close, btc_index)
        parts.append(crypto)

    if include_onchain:
        try:
            onchain = compute_onchain_features(btc_index)
            parts.append(onchain)
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