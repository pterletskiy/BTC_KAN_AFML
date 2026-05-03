"""
6) Cross-Validation Splits
===============================
Generate all C(N,k) combinatorial train/test splits per AFML Chapter 12.4,
apply purging (Chapter 7.4.1) and embargoing (Chapter 7.4.2) to prevent
information leakage from overlapping triple-barrier labels, and compute
the path-assignment matrix that maps each (group, split) pair to one of
the φ[N,k] backtest paths.

The notebook decides N and k by passing them explicitly. The module
constants ``N_GROUPS`` and ``K_TEST_GROUPS`` exist only as fallback
defaults for legacy callers; new code should pass values directly.
"""

import logging
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (defaults only; pass explicitly from the notebook)
# ---------------------------------------------------------------------------
N_GROUPS = 8
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
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    path_map: dict | None = None,
    n_paths: int | None = None,
    print_summary: bool = True,
) -> dict:
    """Compute (or accept) and optionally print a CPCV split-configuration
    summary.

    The function originally computed splits and paths every time it was
    called, which led to duplicate work and duplicate log lines when the
    notebook also called ``generate_cpcv_splits`` and
    ``build_path_matrix`` separately. The current signature lets the
    caller pass already-computed values via ``splits`` / ``path_map`` /
    ``n_paths``, in which case no recomputation occurs.

    The recommended notebook pattern is now::

        splits = generate_cpcv_splits(X, t1, n_groups=N, k=k)
        n_paths, path_map = build_path_matrix(n_groups=N, k=k)
        info = get_split_info(
            X, t1, n_groups=N, k=k,
            splits=splits, path_map=path_map, n_paths=n_paths,
        )

    which computes splits and paths once, then summarises without
    re-running anything. Passing only ``X`` and ``t1`` (the legacy
    pattern) still works but recomputes splits and paths internally.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    t1 : pd.Series
        Barrier touch timestamps.
    n_groups, k, embargo_pct
        CPCV parameters. Defaults come from the module constants.
    splits : list, optional
        Pre-computed splits from ``generate_cpcv_splits``. If None,
        will be computed. If passed, ``n_groups`` / ``k`` /
        ``embargo_pct`` must match the values used to compute it.
    path_map, n_paths : dict and int, optional
        Pre-computed path matrix from ``build_path_matrix``. If None,
        will be computed.
    print_summary : bool, default True
        Whether to print the human-readable summary block.

    Returns
    -------
    dict
        Summary statistics. The dict also includes the ``splits`` and
        ``path_map`` if they were computed inside the function, so the
        caller can avoid a second round of computation.
    """
    T = len(X)
    embargo_len = int(embargo_pct * T)
    group_bounds = _compute_group_bounds(T, n_groups)

    # compute splits and paths only if not provided
    if splits is None:
        splits = generate_cpcv_splits(X, t1, n_groups, k, embargo_pct)
    if path_map is None or n_paths is None:
        n_paths, path_map = build_path_matrix(n_groups, k)

    info = _compute_split_info(
        X=X, splits=splits,
        n_groups=n_groups, k=k,
        embargo_len=embargo_len, embargo_pct=embargo_pct,
        n_paths=n_paths, group_bounds=group_bounds,
    )

    if print_summary:
        print_split_summary(info)

    return info


def print_split_summary(info: dict) -> None:
    """Print the human-readable CPCV split-configuration summary.

    Pure presentation: takes the dict returned by
    ``_compute_split_info`` and writes the formatted table to stdout.
    Separated from computation so the caller can recompute or render
    without redundant work.

    Parameters
    ----------
    info : dict
        Output of ``_compute_split_info`` (or of ``get_split_info``).
        Must contain: ``T``, ``n_groups``, ``k``, ``n_splits``,
        ``n_paths``, ``embargo_len``, ``embargo_pct``,
        ``group_boundaries``, ``avg_train_size``, ``train_pct``,
        ``avg_test_size``, ``avg_purged_count``, ``avg_embargoed_count``.
    """
    print("=" * 60)
    print("CPCV Split Summary")
    print("=" * 60)
    print(f"  Total observations (T):  {info['T']}")
    print(f"  Groups (N):              {info['n_groups']}")
    print(f"  Test groups per split:   {info['k']}")
    print(f"  Splits C(N,k):           {info['n_splits']}")
    print(f"  Backtest paths φ[N,k]:   {info['n_paths']}")
    print(
        f"  Embargo length:          {info['embargo_len']} obs "
        f"({info['embargo_pct'] * 100:.1f}%)"
    )
    print()
    print("  Group boundaries:")
    for g, (s, e, d0, d1) in enumerate(info["group_boundaries"]):
        print(f"    G{g}: idx [{s:>5d}, {e:>5d}]  ({d0} → {d1})")
    print()
    print(
        f"  Avg train size:          {info['avg_train_size']:.0f} "
        f"({info['train_pct'] * 100:.1f}%)"
    )
    print(f"  Avg test size:           {info['avg_test_size']:.0f}")
    print(f"  Avg purged per split:    {info['avg_purged_count']:.1f}")
    print(f"  Avg embargoed per split: {info['avg_embargoed_count']:.1f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _compute_split_info(
    X: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_groups: int,
    k: int,
    embargo_len: int,
    embargo_pct: float,
    n_paths: int,
    group_bounds: list[tuple[int, int]],
) -> dict:
    """Compute the summary statistics dict from already-generated splits.

    Pure data assembly with no logging or printing. Separated from the
    public-facing ``get_split_info`` so unit tests can call it
    independently and so a notebook can re-render the summary without
    re-generating splits.
    """
    T = len(X)

    # group boundary info
    group_boundaries = []
    for g, (start, end) in enumerate(group_bounds):
        group_boundaries.append((
            start, end - 1,
            X.index[start].date(),
            X.index[end - 1].date(),
        ))

    # per-split statistics
    train_sizes = [len(tr) for tr, _ in splits]
    test_sizes = [len(te) for _, te in splits]

    # estimate purge/embargo counts per split by comparing to unpurged sizes
    all_combos = list(combinations(range(n_groups), k))
    purge_counts = []
    embargo_counts = []
    for i, test_groups in enumerate(all_combos):
        test_size = sum(
            group_bounds[g][1] - group_bounds[g][0] for g in test_groups
        )
        unpurged_train = T - test_size
        actual_train = len(splits[i][0])
        total_removed = unpurged_train - actual_train

        # estimate embargo portion: at most embargo_len per test group
        est_embargo = min(embargo_len * len(test_groups), total_removed)
        est_purged = total_removed - est_embargo

        purge_counts.append(max(0, est_purged))
        embargo_counts.append(max(0, est_embargo))

    return {
        "T": T,
        "n_groups": n_groups,
        "k": k,
        "n_splits": len(splits),
        "n_paths": n_paths,
        "embargo_len": embargo_len,
        "embargo_pct": embargo_pct,
        "group_boundaries": group_boundaries,
        "avg_train_size": float(np.mean(train_sizes)),
        "avg_test_size": float(np.mean(test_sizes)),
        "avg_purged_count": float(np.mean(purge_counts)),
        "avg_embargoed_count": float(np.mean(embargo_counts)),
        "train_pct": float(np.mean(train_sizes)) / T,
    }


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