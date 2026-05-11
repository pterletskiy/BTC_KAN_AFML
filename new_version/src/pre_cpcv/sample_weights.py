"""
3) Sample Weights
=======================
Per-sample weights for overlapping triple-barrier labels, implementing AFML
Chapter 4 (Snippets 4.1, 4.2, 4.10, 4.11). Reduces the influence of redundant
overlapping observations during classifier training.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Bar-level overlap count: how many active labels are running at each timestamp.
def get_num_concurrent_labels(bins_t1: pd.Series, num_bars_index: pd.DatetimeIndex) -> pd.Series:
    """Count how many labels are alive at each bar (AFML Snippet 4.1)."""
    c_t = pd.Series(0, index=num_bars_index, dtype=np.int64)

    # For each label [t0, t1], add 1 to every bar inside the slice so the resulting
    # series tells us how many concurrent labels are alive at each calendar bar.
    for t0, t1 in bins_t1.items():
        if pd.isna(t1):
            continue
        c_t.loc[t0:t1] += 1

    return c_t


# Per-label uniqueness score from the bar-level concurrency series.
def get_average_uniqueness(bins_t1: pd.Series, concurrent_labels: pd.Series) -> pd.Series:
    """Average uniqueness ``mean(1/c_t)`` over each label's lifetime (AFML Snippet 4.2).

    Labels with no overlap → 1.0; heavily overlapping labels → 0.
    """
    avg_uniq = pd.Series(index=bins_t1.index, dtype=np.float64)

    # Each label's uniqueness is the mean of (1 / concurrent_count) over its lifespan:
    # at bars where many labels are alive, the contribution shrinks toward 0.
    for t0, t1 in bins_t1.items():
        if pd.isna(t1):
            avg_uniq.loc[t0] = np.nan
            continue
        c_slice = concurrent_labels.loc[t0:t1]
        c_slice = c_slice[c_slice > 0]
        avg_uniq.loc[t0] = (1.0 / c_slice).mean()

    return avg_uniq.dropna()


# Combine label magnitude with uniqueness to upweight informative, non-redundant events.
def get_return_attribution_weights(bins: pd.DataFrame, avg_uniqueness: pd.Series) -> pd.Series:
    """Weight each label by ``|ret| × uniqueness`` and normalise (AFML Snippet 4.10).

    Normalisation ``sum(w) == len(w)`` so the mean weight ≈ 1, matching
    sklearn's default ``sample_weight`` semantics.
    """
    common = bins.index.intersection(avg_uniqueness.index)
    ret = bins.loc[common, "ret"].abs()
    uniq = avg_uniqueness.loc[common]

    weights = ret * uniq
    w_sum = weights.sum()
    if w_sum > 0:
        weights = weights * len(weights) / w_sum
    else:
        logger.warning("All return-attribution weights are zero; setting uniform.")
        weights[:] = 1.0

    return weights


# Linear time decay favouring recent observations; recovers from ``oldest_weight=1`` as no-op.
def apply_time_decay(weights: pd.Series, oldest_weight: float = 1.0) -> pd.Series:
    """Linear time decay (AFML Snippet 4.11); ``oldest_weight=1`` disables decay.

    ``oldest_weight`` is the ``c`` parameter in AFML; setting it to 0.5 halves
    the oldest sample's weight relative to the newest.
    """
    if oldest_weight == 1.0:
        return weights.copy()

    n = len(weights)
    if n <= 1:
        return weights.copy()

    # Linear ramp from oldest_weight at the start to 1.0 at the most recent observation.
    decay = np.linspace(oldest_weight, 1.0, n)
    decayed = weights.values * decay

    # Re-normalise so the mean weight stays ≈ 1 after scaling.
    d_sum = decayed.sum()
    if d_sum > 0:
        decayed = decayed * len(decayed) / d_sum

    return pd.Series(decayed, index=weights.index, name=weights.name)


# Public orchestrator: returns the final sample-weight Series consumed by every classifier.
def compute_sample_weights(bins: pd.DataFrame, num_bars_index: pd.DatetimeIndex,
                           time_decay_factor: float = 1.0, weight_cap_quantile: float = 0.99) -> pd.Series:
    """Chain concurrency → uniqueness → return attribution → time decay → outlier cap.

    The quantile cap is a defensive step against a few extreme-return events
    dominating the training-sample gradient.
    """
    # Build the weight Series stage by stage; each stage transforms the previous.
    concurrent = get_num_concurrent_labels(bins["t1"], num_bars_index)
    avg_uniq = get_average_uniqueness(bins["t1"], concurrent)
    weights = get_return_attribution_weights(bins, avg_uniq)
    weights = apply_time_decay(weights, oldest_weight=time_decay_factor)

    # Clip the tail: a handful of extreme-return events would otherwise dominate the gradient.
    if weight_cap_quantile < 1.0:
        cap = weights.quantile(weight_cap_quantile)
        n_capped = (weights > cap).sum()
        weights = weights.clip(upper=cap)
        if n_capped > 0:
            logger.info(
                "Capped %d weights at %.2f (%.0f%% percentile).",
                n_capped, cap, weight_cap_quantile * 100)

    print(
        f"[sample_weights] {len(weights)} weights | "
        f"mean={weights.mean():.4f}, std={weights.std():.4f}, "
        f"min={weights.min():.4f}, max={weights.max():.4f}")

    return weights