"""
2) Labeling
=====================
Produce CUSUM-filtered triple-barrier labels for the downstream CPCV loop.
Output columns are ``['ret', 'bin', 't1']`` where ``t1`` is the timestamp of
first barrier touch (required for purging). Implements AFML §2.4 and §3.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Volatility estimator that sets the per-event barrier width in triple_barrier_labels.
def compute_daily_volatility(close: pd.Series, span: int = 50) -> pd.Series:
    """EWMA std of log returns (AFML Snippet 3.1).

    Span 50 instead of De Prado's 100 to track BTC's faster regime transitions.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.ewm(span=span).std()


# Reduce ~4,200 daily bars to a sparse event set where genuine drift has accumulated.
def cusum_filter(log_returns: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter (AFML Snippet 2.4); fires when |cumulative drift| ≥ h."""
    events = []
    s_pos, s_neg = 0.0, 0.0

    # Two running accumulators: s_pos tracks positive drift, s_neg tracks negative.
    # Each fires an event and resets when its absolute value crosses the threshold,
    # so a sustained move in either direction produces a CUSUM event.
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


# Compute the vertical (time) barrier for each event; the calendar-day horizon cap.
def get_vertical_barriers(close: pd.Series, t_events: pd.DatetimeIndex, num_days: int) -> pd.Series:
    """For each event, return the close-index timestamp ``num_days`` calendar days ahead.

    NaN where the horizon falls beyond the available price history.
    """
    idx = close.index
    t1 = t_events + pd.Timedelta(days=num_days)

    # Snap each target date to the closest preceding bar in the close index.
    # searchsorted returns the insertion position; subtracting 1 gives the previous bar.
    locs = idx.searchsorted(t1, side="right") - 1
    locs = pd.Series(locs, index=t_events)

    # Mark out-of-range positions (event too close to dataset end) as NaN.
    mask = (locs < 0) | (locs >= len(idx))
    locs[mask] = np.nan
    result = locs.map(lambda x: idx[int(x)] if not np.isnan(x) else np.nan)
    return pd.Series(result, index=t_events, name="t1")


# Core labelling step: assign {-1, 0, +1} per event based on which barrier is hit first.
def triple_barrier_labels(close: pd.Series, t_events: pd.DatetimeIndex, trgt: pd.Series,
                          pt_sl: tuple[float, float] = (1.0, 1.0), num_days: int = 10, min_return: float = 0.0) -> pd.DataFrame:
    """Triple-barrier labelling (AFML Snippets 3.2, 3.4, 3.5).

    ``pt_sl`` are the upper/lower barrier multipliers applied to ``trgt`` (daily
    vol). Setting either to 0 disables that barrier. Returns ``['ret', 'bin', 't1']``
    indexed on event timestamps.
    """
    # Drop events that fall in the vol warm-up (no trgt available) and attach vertical barriers.
    t_events = t_events[t_events.isin(trgt.dropna().index)]
    vertical = get_vertical_barriers(close, t_events, num_days)

    events = pd.DataFrame({"t1": vertical, "trgt": trgt.loc[t_events].values},
                          index=t_events)
    events = events.dropna(subset=["trgt"])

    # For each event, walk the price path to the vertical barrier and label by first touch.
    results = []
    for t0, row in events.iterrows():
        t1 = row["t1"]
        target = row["trgt"]
        if pd.isna(t1):
            continue

        path = close.loc[t0:t1]
        if len(path) < 2:
            continue

        # Convert the vol-scaled multipliers into absolute price barriers around p0.
        p0 = close.loc[t0]
        upper = p0 * (1.0 + pt_sl[0] * target) if pt_sl[0] > 0 else np.inf
        lower = p0 * (1.0 - pt_sl[1] * target) if pt_sl[1] > 0 else -np.inf

        # Earliest crossing of each horizontal barrier (NaT if never touched).
        upper_touch = path[path >= upper].index.min() if pt_sl[0] > 0 else pd.NaT
        lower_touch = path[path <= lower].index.min() if pt_sl[1] > 0 else pd.NaT

        # The label is decided by whichever barrier was hit first in time.
        touches = pd.Series(
            {"upper": upper_touch, "lower": lower_touch, "vertical": t1}
        ).dropna()
        first_touch = touches.min()

        ret = close.loc[first_touch] / p0 - 1.0

        if first_touch == upper_touch:
            label = 1
        elif first_touch == lower_touch:
            label = -1
        else:
            # Vertical barrier hit: sign-of-return, with the |ret| < min_return
            # → 0 escape hatch to avoid labelling pure noise
            label = 0 if abs(ret) < min_return else int(np.sign(ret))

        results.append({"t0": t0, "ret": ret, "bin": label, "t1": first_touch})

    out = pd.DataFrame(results).set_index("t0")
    out.index.name = None
    logger.info(
        "Triple-barrier: %d events labeled. Class distribution: %s",
        len(out),
        out["bin"].value_counts().to_dict(),
    )
    return out


# Drop labels too rare for the classifier to learn; protects per-fold class balance.
def drop_rare_labels(bins: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """Remove classes appearing in fewer than ``min_pct`` of samples (AFML Snippet 3.8).

    Prevents the classifier from being penalised for a class it sees too rarely
    to learn meaningfully.
    """
    counts = bins["bin"].value_counts(normalize=True)
    rare = counts[counts < min_pct].index.tolist()

    if rare:
        n_before = len(bins)
        bins = bins[~bins["bin"].isin(rare)].copy()
        logger.warning(
            "Dropped %d row(s) belonging to rare class(es) %s (< %.1f%%).",
            n_before - len(bins), rare, min_pct * 100)

    return bins


# Public orchestrator: one call from the notebook produces the labelled bins DataFrame.
def run_labeling_pipeline(close: pd.Series, vol_span: int = 50, cusum_enabled: bool = True,
                          cusum_threshold_multiplier: float = 1.0, pt_sl: tuple[float, float] = (1, 1),
                          num_days: int = 10, min_return: float = 0.0, drop_rare: bool = True,
                          min_rare_pct: float = 0.05) -> pd.DataFrame:
    """Chain vol estimation → CUSUM → triple-barrier → rare-class pruning."""
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
        pt_sl=pt_sl, num_days=num_days, min_return=min_return)

    if drop_rare:
        bins = drop_rare_labels(bins, min_pct=min_rare_pct)

    print(
        f"[labeling] {len(bins)} labels | "
        f"classes: {bins['bin'].value_counts().to_dict()} | "
        f"date range: {bins.index[0].date()} → {bins.index[-1].date()}")

    return bins