"""
3) Sample Weights
=======================
Compute per-sample weights that account for overlapping triple-barrier
labels, following AFML Chapter 4 (Snippets 4.1, 4.2, 4.10, 4.11).

These weights are passed to the classifier's ``sample_weight`` parameter
during training so that overlapping, redundant labels receive less
influence than unique ones.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snippet 4.1 — Number of concurrent labels at each bar
# ---------------------------------------------------------------------------
def get_num_concurrent_labels(
    bins_t1: pd.Series, num_bars_index: pd.DatetimeIndex
) -> pd.Series:
    """Count how many labels are alive at each bar.

    Parameters
    ----------
    bins_t1 : pd.Series
        The 't1' column from the bins DataFrame (index = t₀, values = t₁).
    num_bars_index : pd.DatetimeIndex
        Full bar timeline (e.g., ``df_raw.index``).

    Returns
    -------
    pd.Series
        ``c_t`` indexed on *num_bars_index*, where each value is the count
        of labels active at that bar.
    """
    # initialise count series over the full bar index
    c_t = pd.Series(0, index=num_bars_index, dtype=np.int64)

    for t0, t1 in bins_t1.items():
        if pd.isna(t1):
            continue
        c_t.loc[t0:t1] += 1

    return c_t


# ---------------------------------------------------------------------------
# Snippet 4.2 — Average uniqueness per label
# ---------------------------------------------------------------------------
def get_average_uniqueness(
    bins_t1: pd.Series, concurrent_labels: pd.Series
) -> pd.Series:
    """Compute average uniqueness for each label.

    For label *i* spanning [t₀, t₁], uniqueness is
    ``mean(1 / c_t)`` over all bars in that interval. Labels with no
    overlap get uniqueness ≈ 1.0; heavily overlapping labels approach 0.

    Parameters
    ----------
    bins_t1 : pd.Series
        The 't1' column from the bins DataFrame.
    concurrent_labels : pd.Series
        Output of :func:`get_num_concurrent_labels`.

    Returns
    -------
    pd.Series
        Average uniqueness in (0, 1], indexed on event timestamps.
    """
    avg_uniq = pd.Series(index=bins_t1.index, dtype=np.float64)

    for t0, t1 in bins_t1.items():
        if pd.isna(t1):
            avg_uniq.loc[t0] = np.nan
            continue
        c_slice = concurrent_labels.loc[t0:t1]
        # avoid division by zero (bars with c_t == 0 shouldn't exist in the span)
        c_slice = c_slice[c_slice > 0]
        avg_uniq.loc[t0] = (1.0 / c_slice).mean()

    return avg_uniq.dropna()


# ---------------------------------------------------------------------------
# Snippet 4.10 — Return-attribution weights
# ---------------------------------------------------------------------------
def get_return_attribution_weights(
    bins: pd.DataFrame, avg_uniqueness: pd.Series
) -> pd.Series:
    """Weight each label by ``|ret| × uniqueness``, then normalise.

    Normalisation ensures ``weights.sum() == len(weights)`` so that
    the mean weight ≈ 1, which is compatible with sklearn's default
    sample_weight behaviour.

    Parameters
    ----------
    bins : pd.DataFrame
        Must contain a 'ret' column.
    avg_uniqueness : pd.Series
        Output of :func:`get_average_uniqueness`.

    Returns
    -------
    pd.Series
        Normalised weights indexed on event timestamps.
    """
    # align on common index
    common = bins.index.intersection(avg_uniqueness.index)
    ret = bins.loc[common, "ret"].abs()
    uniq = avg_uniqueness.loc[common]

    weights = ret * uniq

    # normalise: sum(w) == len(w)
    w_sum = weights.sum()
    if w_sum > 0:
        weights = weights * len(weights) / w_sum
    else:
        logger.warning("All return-attribution weights are zero; setting uniform.")
        weights[:] = 1.0

    return weights


# ---------------------------------------------------------------------------
# Snippet 4.11 — Time decay
# ---------------------------------------------------------------------------
def apply_time_decay(
    weights: pd.Series, oldest_weight: float = 1.0
) -> pd.Series:
    """Apply a linear time decay so that older samples weigh less.

    Parameters
    ----------
    weights : pd.Series
        Pre-decay weights (e.g., return-attribution weights), assumed
        sorted chronologically.
    oldest_weight : float
        Decay factor for the oldest observation (the ``c`` parameter in
        AFML). 1.0 means no decay; 0.5 halves the oldest sample's weight
        relative to the newest.

    Returns
    -------
    pd.Series
        Decayed weights, same index as input.
    """
    if oldest_weight == 1.0:
        return weights.copy()

    n = len(weights)
    if n <= 1:
        return weights.copy()

    # linear decay factors: d[0] = oldest_weight, d[-1] = 1.0
    decay = np.linspace(oldest_weight, 1.0, n)
    decayed = weights.values * decay

    # re-normalise so sum(w) == len(w)
    d_sum = decayed.sum()
    if d_sum > 0:
        decayed = decayed * len(decayed) / d_sum

    return pd.Series(decayed, index=weights.index, name=weights.name)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def compute_sample_weights(
    bins: pd.DataFrame,
    num_bars_index: pd.DatetimeIndex,
    time_decay_factor: float = 1.0,
    weight_cap_quantile: float = 0.99,
) -> pd.Series:
    """Chain concurrency, uniqueness, return attribution, time decay, and capping.

    Parameters
    ----------
    bins : pd.DataFrame
        Output of the labeling pipeline, must contain ['ret', 'bin', 't1'].
    num_bars_index : pd.DatetimeIndex
        Full bar timeline (``df_raw.index``).
    time_decay_factor : float
        Passed to :func:`apply_time_decay` as ``oldest_weight``.
    weight_cap_quantile : float
        Percentile at which to cap outlier weights. Set to 1.0 to disable.

    Returns
    -------
    pd.Series
        Final sample weights indexed on event timestamps.
    """
    concurrent = get_num_concurrent_labels(bins["t1"], num_bars_index)
    avg_uniq = get_average_uniqueness(bins["t1"], concurrent)
    weights = get_return_attribution_weights(bins, avg_uniq)
    weights = apply_time_decay(weights, oldest_weight=time_decay_factor)

    # cap outlier weights
    if weight_cap_quantile < 1.0:
        cap = weights.quantile(weight_cap_quantile)
        n_capped = (weights > cap).sum()
        weights = weights.clip(upper=cap)
        if n_capped > 0:
            logger.info(
                "Capped %d weights at %.2f (%.0f%% percentile).",
                n_capped, cap, weight_cap_quantile * 100,
            )

    print(
        f"[sample_weights] {len(weights)} weights | "
        f"mean={weights.mean():.4f}, std={weights.std():.4f}, "
        f"min={weights.min():.4f}, max={weights.max():.4f}"
    )

    return weights