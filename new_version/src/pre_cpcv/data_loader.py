"""
1) Data Loader
====================
Fetch BTC-USD daily OHLCV from yfinance and return a validated, gap-filled
DataFrame on a tz-naive daily DatetimeIndex.
"""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Canonical OHLCV column order; what yfinance returns and what downstream expects.
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Cap on forward-fill of short calendar gaps. Longer gaps raise rather than fill,
# because BTC trades 24/7 and a multi-day gap is a data-source problem, not a holiday.
_MAX_FFILL_DAYS = 3

# Default download window. Start covers full BTC history; end is the locked thesis cutoff.
_START_DATE, _END_DATE = "2014-01-01", "2026-05-01"

# Public entry point: fetch BTC daily OHLCV, validate, and return a clean DataFrame.
def load_btc_daily(ticker: str = "BTC-USD", start: str = _START_DATE, end: str = _END_DATE) -> pd.DataFrame:
    """Download daily OHLCV for ``ticker`` and return a validated DataFrame.

    Raises ``ValueError`` if the download is empty, has duplicate dates, or
    contains a calendar gap longer than ``_MAX_FFILL_DAYS``.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance >= 0.2.31 returns MultiIndex columns; flatten to the ticker level
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(level=1, axis=1)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker} [{start}, {end})")

    df = raw[_OHLCV_COLS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df.sort_index()

    nan_close = df["Close"].isna()
    if nan_close.any():
        logger.warning("Dropped %d row(s) with NaN Close.", nan_close.sum())
        df = df.loc[~nan_close]

    if df.index.duplicated().any():
        raise ValueError("Duplicate dates found in downloaded data.")

    warnings_log: list[str] = []

    # BTC trades 7 days a week, so any calendar gap is a data-source issue
    df, gap_warnings = _fill_small_gaps(df)
    warnings_log.extend(gap_warnings)
    warnings_log.extend(_check_ohlcv_consistency(df))

    for w in warnings_log:
        logger.warning(w)

    summary = (
        f"[data_loader] {ticker}: {df.index[0].date()} → {df.index[-1].date()} | "
        f"{len(df)} rows | {len(warnings_log)} warning(s)")
    print(summary)
    logger.info(summary)

    return df


# Detect and forward-fill calendar gaps in the 7-day trading week; raise on long gaps.
def _fill_small_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Forward-fill calendar gaps up to ``_MAX_FFILL_DAYS``; raise on anything longer.

    A multi-day gap is treated as a structural data issue rather than a holiday,
    since BTC has no exchange calendar.
    """
    warnings: list[str] = []
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = full_range.difference(df.index)

    if missing.empty:
        return df, warnings

    # Group consecutive missing dates into contiguous gaps so each gap can be sized.
    missing_s = pd.Series(missing, name="date").sort_values()
    groups = (missing_s.diff().dt.days.fillna(1) > 1).cumsum()

    # Validate each gap; raise on anything beyond the fill limit, warn on small ones.
    for _, grp in missing_s.groupby(groups):
        gap_len = len(grp)
        if gap_len > _MAX_FFILL_DAYS:
            raise ValueError(
                f"Calendar gap of {gap_len} days starting {grp.iloc[0].date()} "
                f"exceeds the {_MAX_FFILL_DAYS}-day fill limit. "
                "The data source may have a structural problem.")
        warnings.append(
            f"Forward-filled {gap_len}-day gap starting {grp.iloc[0].date()}.")

    # Reindex onto the full calendar and forward-fill prices; volume gets 0 (no trades).
    df = df.reindex(full_range)
    df.index.name = "Date"
    df[["Open", "High", "Low", "Close"]] = (df[["Open", "High", "Low", "Close"]].ffill(limit=_MAX_FFILL_DAYS))
    df["Volume"] = df["Volume"].fillna(0)

    return df, warnings


# Audit the OHLCV row-level invariants and return human-readable warnings.
def _check_ohlcv_consistency(df: pd.DataFrame) -> list[str]:
    """Warn on rows where High/Low ordering or non-negative Volume is violated."""
    warnings: list[str] = []

    high_ok = df["High"] >= df[["Open", "Close"]].max(axis=1)
    low_ok = df["Low"] <= df[["Open", "Close"]].min(axis=1)
    vol_ok = df["Volume"] >= 0

    if (n := (~high_ok).sum()):
        warnings.append(f"OHLCV: {n} row(s) where High < max(Open, Close).")
    if (n := (~low_ok).sum()):
        warnings.append(f"OHLCV: {n} row(s) where Low > min(Open, Close).")
    if (n := (~vol_ok).sum()):
        warnings.append(f"OHLCV: {n} row(s) with negative Volume.")

    return warnings