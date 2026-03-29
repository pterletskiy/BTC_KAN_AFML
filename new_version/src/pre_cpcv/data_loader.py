"""
1) Data Loader
====================
Fetch BTC-USD daily OHLCV from yfinance, validate data quality,
and return a clean DatetimeIndex DataFrame.

Future expansion: add load_macro_features(), load_onchain_features(),
and merge_all_sources() using pd.merge_asof(direction='backward')
to align everything onto BTC's daily calendar.
"""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
_MAX_FFILL_DAYS = 3  # forward-fill limit for small yfinance gaps
_START_DATE = "2014-01-01"
_END_DATE = "2026-03-28"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_btc_daily(ticker: str = "BTC-USD",
                   start: str = _START_DATE, end: str = _END_DATE) -> pd.DataFrame:
    """Download daily OHLCV for *ticker* and return a validated DataFrame.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start, end : str
        Date strings understood by yfinance (inclusive start, exclusive end).

    Returns
    -------
    pd.DataFrame
        Columns ``['Open','High','Low','Close','Volume']``, indexed by a
        tz-naive ``DatetimeIndex`` at daily frequency, sorted ascending.

    Raises
    ------
    ValueError
        If the download is empty, contains duplicate dates, or has a calendar
        gap longer than ``_MAX_FFILL_DAYS`` days.
    """
    # ----- download --------------------------------------------------------
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # flatten MultiIndex columns (yfinance >= 0.2.31 returns multi-level)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(level=1, axis=1)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker} [{start}, {end})")

    df = raw[_OHLCV_COLS].copy()

    # ----- normalise index -------------------------------------------------
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df.sort_index()

    # ----- drop rows where Close is NaN ------------------------------------
    nan_close = df["Close"].isna()
    if nan_close.any():
        n = nan_close.sum()
        logger.warning("Dropped %d row(s) with NaN Close.", n)
        df = df.loc[~nan_close]

    # ----- validation ------------------------------------------------------
    warnings_log: list[str] = []

    # 1. duplicate dates
    if df.index.duplicated().any():
        raise ValueError("Duplicate dates found in downloaded data.")

    # 2. calendar-day gaps (BTC trades 7 d/wk)
    df, gap_warnings = _fill_small_gaps(df)
    warnings_log.extend(gap_warnings)

    # 3. OHLCV consistency
    ohlcv_warnings = _check_ohlcv_consistency(df)
    warnings_log.extend(ohlcv_warnings)

    # ----- summary ---------------------------------------------------------
    for w in warnings_log:
        logger.warning(w)

    summary = (
        f"[data_loader] {ticker}: {df.index[0].date()} → {df.index[-1].date()} | "
        f"{len(df)} rows | {len(warnings_log)} warning(s)"
    )
    print(summary)
    logger.info(summary)

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _fill_small_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Detect calendar-day gaps, forward-fill up to ``_MAX_FFILL_DAYS``.

    Raises ``ValueError`` if any gap exceeds the limit.
    """
    warnings: list[str] = []
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = full_range.difference(df.index)

    if missing.empty:
        return df, warnings

    # group consecutive missing dates to measure each gap
    missing_s = pd.Series(missing, name="date").sort_values()
    groups = (missing_s.diff().dt.days.fillna(1) > 1).cumsum()

    for _, grp in missing_s.groupby(groups):
        gap_len = len(grp)
        if gap_len > _MAX_FFILL_DAYS:
            raise ValueError(
                f"Calendar gap of {gap_len} days starting {grp.iloc[0].date()} "
                f"exceeds the {_MAX_FFILL_DAYS}-day fill limit. "
                "The data source may have a structural problem."
            )
        warnings.append(
            f"Forward-filled {gap_len}-day gap starting {grp.iloc[0].date()}."
        )

    # reindex and forward-fill
    df = df.reindex(full_range)
    df.index.name = "Date"
    df = df.ffill(limit=_MAX_FFILL_DAYS)

    return df, warnings


def _check_ohlcv_consistency(df: pd.DataFrame) -> list[str]:
    """Flag rows where High/Low/Volume constraints are violated."""
    warnings: list[str] = []

    high_ok = df["High"] >= df[["Open", "Close"]].max(axis=1)
    low_ok = df["Low"] <= df[["Open", "Close"]].min(axis=1)
    vol_ok = df["Volume"] >= 0

    n_high = (~high_ok).sum()
    n_low = (~low_ok).sum()
    n_vol = (~vol_ok).sum()

    if n_high:
        warnings.append(f"OHLCV: {n_high} row(s) where High < max(Open, Close).")
    if n_low:
        warnings.append(f"OHLCV: {n_low} row(s) where Low > min(Open, Close).")
    if n_vol:
        warnings.append(f"OHLCV: {n_vol} row(s) with negative Volume.")

    return warnings