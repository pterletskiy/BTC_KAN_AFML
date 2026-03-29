"""
6) Cross-Validation Splits
===============================
Generate all C(N,k) combinatorial train/test splits per AFML Chapter 12.4,
apply purging (Chapter 7.4.1) and embargoing (Chapter 7.4.2) to prevent
information leakage from overlapping triple-barrier labels, and compute
the path-assignment matrix that maps each (group, split) pair to one of
the φ[N,k] backtest paths.
"""

import logging
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
N_GROUPS = 6
K_TEST_GROUPS = 2
EMBARGO_PCT = 0.01  # fraction of T to embargo after each test boundary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_cpcv_splits(
    X: pd.DataFrame,
    t1: pd.Series,
    n_groups: int = N_GROUPS,
    k: int = K_TEST_GROUPS,
    embargo_pct: float = EMBARGO_PCT,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate all C(N,k) combinatorial purged cross-validation splits.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix, chronologically sorted (from alignment).
    t1 : pd.Series
        Barrier touch timestamps for each observation, same index as X.
    n_groups : int
        Number of contiguous groups to partition the data into.
    k : int
        Number of groups in each test set.
    embargo_pct : float
        Fraction of T observations to embargo after each test group.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples, one per split.
        Indices are positional (integer) into X.
    """
    T = len(X)
    embargo_len = int(embargo_pct * T)

    # partition into groups
    group_bounds = _compute_group_bounds(T, n_groups)

    # all C(N,k) combinations of test groups
    all_combos = list(combinations(range(n_groups), k))

    splits = []
    for test_groups in all_combos:
        # build test index set
        test_idx = set()
        for g in test_groups:
            start, end = group_bounds[g]
            test_idx.update(range(start, end))

        # build initial train index set (everything not in test)
        train_idx = set(range(T)) - test_idx

        # purge: remove training observations whose labels overlap with test
        purged = _purge_train(
            train_idx, test_groups, group_bounds, X.index, t1
        )
        train_idx -= purged

        # embargo: remove training observations immediately after each test group
        embargoed = _embargo_train(
            train_idx, test_groups, group_bounds, T, embargo_len
        )
        train_idx -= embargoed

        train_arr = np.sort(np.array(list(train_idx), dtype=np.int64))
        test_arr = np.sort(np.array(list(test_idx), dtype=np.int64))

        splits.append((train_arr, test_arr))

    logger.info(
        "CPCV: %d splits generated (N=%d, k=%d, embargo=%d obs).",
        len(splits), n_groups, k, embargo_len,
    )

    return splits


def build_path_matrix(
    n_groups: int = N_GROUPS, k: int = K_TEST_GROUPS
) -> tuple[int, dict]:
    """Compute the path-assignment matrix per AFML Chapter 12.4.1.

    For N=6, k=2 there are φ[6,2] = N-1 = 5 backtest paths. Each group
    appears in exactly φ[N,k] test sets.

    Parameters
    ----------
    n_groups : int
        Number of groups (N).
    k : int
        Test groups per split.

    Returns
    -------
    n_paths : int
        Number of backtest paths (φ[N,k]).
    path_map : dict
        ``{path_id: [(group_id, split_id), ...]}``. Each path entry
        contains N tuples specifying which split's predictions to use
        for that group.
    """
    n_paths = comb(n_groups - 1, k - 1)  # φ[N,k] = C(N-1, k-1)
    all_combos = list(combinations(range(n_groups), k))

    # for each group, collect which splits include it in the test set
    group_splits = {g: [] for g in range(n_groups)}
    for split_id, test_groups in enumerate(all_combos):
        for g in test_groups:
            group_splits[g].append(split_id)

    # assign each (group, split) occurrence to a path
    # each group appears in exactly n_paths test sets
    path_map = {p: [] for p in range(n_paths)}
    for g in range(n_groups):
        for path_id, split_id in enumerate(group_splits[g]):
            path_map[path_id].append((g, split_id))

    logger.info(
        "Path matrix: %d paths, %d groups, %d splits.",
        n_paths, n_groups, len(all_combos),
    )

    return n_paths, path_map


def get_split_info(
    X: pd.DataFrame,
    t1: pd.Series,
    n_groups: int = N_GROUPS,
    k: int = K_TEST_GROUPS,
    embargo_pct: float = EMBARGO_PCT,
) -> dict:
    """Compute and print a summary of the CPCV split configuration.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    t1 : pd.Series
        Barrier touch timestamps.
    n_groups, k, embargo_pct
        CPCV parameters.

    Returns
    -------
    dict
        Summary statistics for logging and sanity checking.
    """
    T = len(X)
    embargo_len = int(embargo_pct * T)
    group_bounds = _compute_group_bounds(T, n_groups)

    splits = generate_cpcv_splits(X, t1, n_groups, k, embargo_pct)
    n_paths, _ = build_path_matrix(n_groups, k)

    # group boundary info
    group_boundaries = []
    for g, (start, end) in enumerate(group_bounds):
        group_boundaries.append((
            start, end - 1,
            X.index[start].date(),
            X.index[end - 1].date(),
        ))

    # per-split statistics
    train_sizes = []
    test_sizes = []
    for train_idx, test_idx in splits:
        train_sizes.append(len(train_idx))
        test_sizes.append(len(test_idx))

    # compute purge/embargo counts by comparing to unpurged sizes
    all_combos = list(combinations(range(n_groups), k))
    purge_counts = []
    embargo_counts = []
    for i, test_groups in enumerate(all_combos):
        test_size = sum(group_bounds[g][1] - group_bounds[g][0] for g in test_groups)
        unpurged_train = T - test_size
        actual_train = len(splits[i][0])
        total_removed = unpurged_train - actual_train

        # estimate embargo portion
        est_embargo = min(embargo_len * len(test_groups), total_removed)
        est_purged = total_removed - est_embargo

        purge_counts.append(max(0, est_purged))
        embargo_counts.append(max(0, est_embargo))

    info = {
        "n_splits": len(splits),
        "n_paths": n_paths,
        "group_boundaries": group_boundaries,
        "avg_train_size": np.mean(train_sizes),
        "avg_test_size": np.mean(test_sizes),
        "avg_purged_count": np.mean(purge_counts),
        "avg_embargoed_count": np.mean(embargo_counts),
        "train_pct": np.mean(train_sizes) / T,
    }

    # print summary
    print("=" * 60)
    print("CPCV Split Summary")
    print("=" * 60)
    print(f"  Total observations (T):  {T}")
    print(f"  Groups (N):              {n_groups}")
    print(f"  Test groups per split:   {k}")
    print(f"  Splits C(N,k):           {info['n_splits']}")
    print(f"  Backtest paths φ[N,k]:   {info['n_paths']}")
    print(f"  Embargo length:          {embargo_len} obs ({embargo_pct*100:.1f}%)")
    print()
    print("  Group boundaries:")
    for g, (s, e, d0, d1) in enumerate(group_boundaries):
        print(f"    G{g}: idx [{s:>5d}, {e:>5d}]  ({d0} → {d1})")
    print()
    print(f"  Avg train size:          {info['avg_train_size']:.0f} ({info['train_pct']*100:.1f}%)")
    print(f"  Avg test size:           {info['avg_test_size']:.0f}")
    print(f"  Avg purged per split:    {info['avg_purged_count']:.1f}")
    print(f"  Avg embargoed per split: {info['avg_embargoed_count']:.1f}")
    print("=" * 60)

    return info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _compute_group_bounds(T: int, n_groups: int) -> list[tuple[int, int]]:
    """Return (start, end) positional index pairs for each group.

    Groups 0..N-2 have size ⌊T/N⌋, group N-1 gets the remainder.
    End indices are exclusive: [start, end).
    """
    base_size = T // n_groups
    bounds = []
    for g in range(n_groups):
        start = g * base_size
        if g < n_groups - 1:
            end = (g + 1) * base_size
        else:
            end = T
        bounds.append((start, end))
    return bounds


def _purge_train(
    train_idx: set[int],
    test_groups: tuple[int, ...],
    group_bounds: list[tuple[int, int]],
    index: pd.DatetimeIndex,
    t1: pd.Series,
) -> set[int]:
    """Purge training observations whose labels overlap with any test group.

    Implements AFML Snippet 7.1 overlap conditions:
      1. t_test_start <= t_i_start <= t_test_end
      2. t_test_start <= t1[i]     <= t_test_end
      3. t_i_start <= t_test_start  AND  t_test_end <= t1[i]
    """
    to_purge = set()

    for g in test_groups:
        g_start, g_end = group_bounds[g]
        t_test_start = index[g_start]
        t_test_end = index[g_end - 1]

        for i in train_idx:
            t_i_start = index[i]
            t_i_end = t1.iloc[i]

            if pd.isna(t_i_end):
                continue

            # condition 1: training observation starts within test period
            if t_test_start <= t_i_start <= t_test_end:
                to_purge.add(i)
                continue

            # condition 2: training label resolves within test period
            if t_test_start <= t_i_end <= t_test_end:
                to_purge.add(i)
                continue

            # condition 3: training label spans the entire test period
            if t_i_start <= t_test_start and t_test_end <= t_i_end:
                to_purge.add(i)
                continue

    return to_purge


def _embargo_train(
    train_idx: set[int],
    test_groups: tuple[int, ...],
    group_bounds: list[tuple[int, int]],
    T: int,
    embargo_len: int,
) -> set[int]:
    """Remove training observations in the embargo zone after each test group.

    Per AFML Chapter 7.4.2, embargo is only needed after the test set
    (not before), because training labels that resolve before the test
    begins contain no future information.
    """
    to_embargo = set()

    for g in test_groups:
        _, g_end = group_bounds[g]
        embargo_end = min(g_end + embargo_len, T)
        for i in range(g_end, embargo_end):
            if i in train_idx:
                to_embargo.add(i)

    return to_embargo