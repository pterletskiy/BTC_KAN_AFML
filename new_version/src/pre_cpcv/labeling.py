"""
2) Labeling
=====================
Take a clean close price series and produce a DataFrame of labels (bins)
with columns ['ret', 'bin', 't1'], where:
  - ret  : return at barrier touch
  - bin  : class label {-1, 0, +1}
  - t1   : timestamp when the label was resolved (first barrier touch)

The t1 column is critical for downstream purging in CPCV.

Implements AFML Snippets 2.4, 3.1, 3.2, 3.4, 3.5, 3.8.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Daily volatility (AFML Snippet 3.1)
# ---------------------------------------------------------------------------
def compute_daily_volatility(close: pd.Series, span: int = 50) -> pd.Series:
    """Exponentially weighted moving std of log returns.

    Parameters
    ----------
    close : pd.Series
        Close prices with a DatetimeIndex.
    span : int
        EWMA span. Default 50 is calibrated for BTC's faster regime
        transitions (De Prado uses 100 for equities).

    Returns
    -------
    pd.Series
        Daily volatility estimate, indexed identically to *close*.
        Leading NaNs from the EWMA warm-up are preserved.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.ewm(span=span).std()


# ---------------------------------------------------------------------------
# Step 2 — CUSUM filter (AFML Snippet 2.4)
# ---------------------------------------------------------------------------
def cusum_filter(log_returns: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter that fires when cumulative deviation exceeds *threshold*.

    Parameters
    ----------
    log_returns : pd.Series
        Log returns series (NaN-free).
    threshold : float
        Barrier *h* for both positive and negative accumulators.

    Returns
    -------
    pd.DatetimeIndex
        Timestamps where the filter triggered an event.
    """
    events = []
    s_pos, s_neg = 0.0, 0.0

    for t, r in log_returns.items():
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)

        if s_pos >= threshold:
            events.append(t)
            s_pos = 0.0

        if s_neg <= -threshold:
            events.append(t)
            s_neg = 0.0

    return pd.DatetimeIndex(events)


# ---------------------------------------------------------------------------
# Step 3 — Vertical barriers
# ---------------------------------------------------------------------------
def get_vertical_barriers(
    close: pd.Series, t_events: pd.DatetimeIndex, num_days: int
) -> pd.Series:
    """For each event, find the close-index timestamp *num_days* calendar days ahead.

    Parameters
    ----------
    close : pd.Series
        Close prices (used only for its index).
    t_events : pd.DatetimeIndex
        Event timestamps from the CUSUM filter.
    num_days : int
        Horizon in calendar days.

    Returns
    -------
    pd.Series
        Indexed on *t_events*, values are the vertical barrier timestamps
        (or NaN if the horizon falls beyond the data).
    """
    idx = close.index
    t1 = t_events + pd.Timedelta(days=num_days)
    # searchsorted finds the nearest index position at or after the target
    locs = idx.searchsorted(t1, side="right") - 1
    # clamp to valid range; mark out-of-bounds as NaN
    locs = pd.Series(locs, index=t_events)
    mask = (locs < 0) | (locs >= len(idx))
    locs[mask] = np.nan
    result = locs.map(lambda x: idx[int(x)] if not np.isnan(x) else np.nan)
    return pd.Series(result, index=t_events, name="t1")


# ---------------------------------------------------------------------------
# Step 4 — Triple-barrier labels (AFML Snippets 3.2 + 3.4 + 3.5)
# ---------------------------------------------------------------------------
def triple_barrier_labels(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    trgt: pd.Series,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    num_days: int = 10,
    min_return: float = 0.0,
) -> pd.DataFrame:
    """Apply the triple-barrier method to each event.

    Parameters
    ----------
    close : pd.Series
        Close prices with DatetimeIndex.
    t_events : pd.DatetimeIndex
        Event timestamps (e.g., from CUSUM filter).
    trgt : pd.Series
        Daily volatility at each timestamp (target width).
    pt_sl : tuple[float, float]
        Multipliers for (upper, lower) horizontal barriers.
        Set either to 0 to disable that barrier.
    num_days : int
        Vertical barrier horizon in calendar days.
    min_return : float
        Minimum absolute return to avoid the 0 label at the vertical barrier.

    Returns
    -------
    pd.DataFrame
        Columns ['ret', 'bin', 't1'], indexed on event timestamps.
    """
    # align events to timestamps present in trgt (drops warm-up NaNs)
    t_events = t_events[t_events.isin(trgt.dropna().index)]

    vertical = get_vertical_barriers(close, t_events, num_days)

    # build events DataFrame
    events = pd.DataFrame({"t1": vertical, "trgt": trgt.loc[t_events].values},
                          index=t_events)
    events = events.dropna(subset=["trgt"])

    results = []
    for t0, row in events.iterrows():
        t1 = row["t1"]
        target = row["trgt"]

        # if vertical barrier is NaN, skip this event
        if pd.isna(t1):
            continue

        # price path from t0 to t1
        path = close.loc[t0:t1]
        if len(path) < 2:
            continue

        p0 = close.loc[t0]

        # horizontal barriers
        upper = p0 * (1.0 + pt_sl[0] * target) if pt_sl[0] > 0 else np.inf
        lower = p0 * (1.0 - pt_sl[1] * target) if pt_sl[1] > 0 else -np.inf

        # find first touch of each barrier
        upper_touch = path[path >= upper].index.min() if pt_sl[0] > 0 else pd.NaT
        lower_touch = path[path <= lower].index.min() if pt_sl[1] > 0 else pd.NaT

        # earliest barrier touch
        touches = pd.Series(
            {"upper": upper_touch, "lower": lower_touch, "vertical": t1}
        ).dropna()
        first_touch = touches.min()

        # return at first touch
        ret = close.loc[first_touch] / p0 - 1.0

        # assign label
        if first_touch == upper_touch:
            label = 1
        elif first_touch == lower_touch:
            label = -1
        else:
            # vertical barrier: sign of return, or 0 if below min_return
            if abs(ret) < min_return:
                label = 0
            else:
                label = int(np.sign(ret))

        results.append({"t0": t0, "ret": ret, "bin": label, "t1": first_touch})

    out = pd.DataFrame(results).set_index("t0")
    out.index.name = None
    logger.info(
        "Triple-barrier: %d events labeled. Class distribution: %s",
        len(out),
        out["bin"].value_counts().to_dict(),
    )
    return out


# ---------------------------------------------------------------------------
# Drop rare labels (AFML Snippet 3.8)
# ---------------------------------------------------------------------------
def drop_rare_labels(
    bins: pd.DataFrame, min_pct: float = 0.05
) -> pd.DataFrame:
    """Remove rows whose class appears in fewer than *min_pct* of samples.

    Parameters
    ----------
    bins : pd.DataFrame
        Must contain a 'bin' column.
    min_pct : float
        Minimum fraction of total samples for a class to survive.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *bins*.
    """
    counts = bins["bin"].value_counts(normalize=True)
    rare = counts[counts < min_pct].index.tolist()

    if rare:
        n_before = len(bins)
        bins = bins[~bins["bin"].isin(rare)].copy()
        n_dropped = n_before - len(bins)
        logger.warning(
            "Dropped %d row(s) belonging to rare class(es) %s (< %.1f%%).",
            n_dropped, rare, min_pct * 100,
        )

    return bins


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_labeling_pipeline(
    close: pd.Series,
    vol_span: int = 50,
    cusum_enabled: bool = True,
    cusum_threshold_multiplier: float = 1.0,
    pt_sl: tuple[float, float] = (1, 1),
    num_days: int = 10,
    min_return: float = 0.0,
    drop_rare: bool = True,
    min_rare_pct: float = 0.05,
) -> pd.DataFrame:
    """Chain volatility estimation, CUSUM filtering, triple-barrier labeling,
    and rare-label pruning into one call.

    Parameters
    ----------
    close : pd.Series
        Clean close prices with DatetimeIndex.
    vol_span : int
        EWMA span for daily volatility.
    cusum_enabled : bool
        If False, every close timestamp becomes an event (no filtering).
    cusum_threshold_multiplier : float
        CUSUM threshold = multiplier × mean(daily_vol).
    pt_sl : tuple[float, float]
        (upper, lower) barrier multipliers.
    num_days : int
        Vertical barrier horizon in calendar days.
    min_return : float
        Minimum return to avoid the 0 label at vertical barrier.
    drop_rare : bool
        Whether to drop classes below *min_rare_pct*.
    min_rare_pct : float
        Threshold for rare-class removal.

    Returns
    -------
    pd.DataFrame
        Columns ['ret', 'bin', 't1'], indexed by event timestamps.
    """
    daily_vol = compute_daily_volatility(close, span=vol_span)
    log_rets = np.log(close / close.shift(1)).dropna()

    if cusum_enabled:
        h = cusum_threshold_multiplier * daily_vol.mean()
        t_events = cusum_filter(log_rets, h)
        logger.info("CUSUM filter: threshold=%.6f, %d events detected.", h, len(t_events))
    else:
        t_events = close.index
        logger.info("CUSUM disabled: using all %d timestamps as events.", len(t_events))

    bins = triple_barrier_labels(
        close, t_events, trgt=daily_vol,
        pt_sl=pt_sl, num_days=num_days, min_return=min_return,
    )

    if drop_rare:
        bins = drop_rare_labels(bins, min_pct=min_rare_pct)

    print(
        f"[labeling] {len(bins)} labels | "
        f"classes: {bins['bin'].value_counts().to_dict()} | "
        f"date range: {bins.index[0].date()} → {bins.index[-1].date()}"
    )

    return bins