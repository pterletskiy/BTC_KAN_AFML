"""
7) Cross-Validation Splits
===============================
Generate all C(N,k) combinatorial train/test splits (AFML §12.4), apply
purging (AFML §7.4.1) and embargoing (AFML §7.4.2) to prevent label-overlap
leakage, and compute the φ[N,k] path-assignment matrix.

The notebook passes N and k explicitly; the module-level constants are
fallback defaults only.
"""

import logging
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Module-level defaults (notebook passes these explicitly per run) ------
N_GROUPS, K_TEST_GROUPS, EMBARGO_PCT = 8, 2, 0.01

# --- 1. Public API ----------------------------------------------------------
# Build every C(N,k) combinatorial split with AFML purging + embargoing.
def generate_cpcv_splits(
    X: pd.DataFrame,
    t1: pd.Series,
    n_groups: int = N_GROUPS,
    k: int = K_TEST_GROUPS,
    embargo_pct: float = EMBARGO_PCT,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``[(train_idx, test_idx), ...]`` covering every C(N,k) test combination.

    Indices are positional integers into ``X``; ``t1`` supplies the label
    end-times that drive the purge condition.
    """
    T = len(X)
    embargo_len = int(embargo_pct * T)

    # Partition the timeline into N contiguous groups; combinations of these become test sets.
    group_bounds = _compute_group_bounds(T, n_groups)
    all_combos = list(combinations(range(n_groups), k))

    splits = []
    for test_groups in all_combos:
        # Build the union of test groups as a positional index set.
        test_idx = set()
        for g in test_groups:
            start, end = group_bounds[g]
            test_idx.update(range(start, end))

        # Start with everything else as training, then strip overlap-purged and embargoed rows.
        train_idx = set(range(T)) - test_idx

        purged = _purge_train(
            train_idx, test_groups, group_bounds, X.index, t1
        )
        train_idx -= purged

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


# Path-assignment matrix (AFML §12.4.1): maps each (group, occurrence) to a backtest path.
def build_path_matrix(
    n_groups: int = N_GROUPS, k: int = K_TEST_GROUPS
) -> tuple[int, dict]:
    """Return ``(n_paths, path_map)`` where ``path_map[p] = [(group, split), ...]``.

    ``n_paths = φ[N,k] = C(N-1, k-1)``: each group appears in exactly φ test sets,
    and the p-th occurrence of every group forms path p.
    """
    n_paths = comb(n_groups - 1, k - 1)
    all_combos = list(combinations(range(n_groups), k))

    # For each group, list the split IDs where it appears in the test set.
    group_splits = {g: [] for g in range(n_groups)}
    for split_id, test_groups in enumerate(all_combos):
        for g in test_groups:
            group_splits[g].append(split_id)

    # Assign the p-th appearance of every group to path p; by construction every path
    # ends up with exactly one (group, split) pair per group.
    path_map = {p: [] for p in range(n_paths)}
    for g in range(n_groups):
        for path_id, split_id in enumerate(group_splits[g]):
            path_map[path_id].append((g, split_id))

    logger.info(
        "Path matrix: %d paths, %d groups, %d splits.",
        n_paths, n_groups, len(all_combos),
    )

    return n_paths, path_map


# Summary builder that prefers pre-computed inputs to avoid double work.
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
    """Compute (or accept) and optionally print a CPCV configuration summary.

    Recommended notebook pattern, which computes splits and paths exactly once::

        splits = generate_cpcv_splits(X, t1, n_groups=N, k=k)
        n_paths, path_map = build_path_matrix(n_groups=N, k=k)
        info = get_split_info(
            X, t1, n_groups=N, k=k,
            splits=splits, path_map=path_map, n_paths=n_paths,
        )

    Passing only ``X`` and ``t1`` still works but recomputes splits internally.
    """
    T = len(X)
    embargo_len = int(embargo_pct * T)
    group_bounds = _compute_group_bounds(T, n_groups)

    # Recompute only what the caller hasn't already produced.
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


# Pure presentation: format the summary dict for human inspection.
def print_split_summary(info: dict) -> None:
    """Print the CPCV configuration summary block built by ``_compute_split_info``."""
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


# --- 2. Internal helpers ----------------------------------------------------
# Assemble the summary statistics dict from already-generated splits.
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
    """Build the human-readable summary dict; no logging, no side effects."""
    T = len(X)

    # Translate positional bounds into (index, date) tuples for the print block.
    group_boundaries = []
    for g, (start, end) in enumerate(group_bounds):
        group_boundaries.append((
            start, end - 1,
            X.index[start].date(),
            X.index[end - 1].date(),
        ))

    train_sizes = [len(tr) for tr, _ in splits]
    test_sizes = [len(te) for _, te in splits]

    # Reconstruct per-split purge and embargo counts by comparing the actual train
    # size to the unpurged baseline (T − test_size); the residual splits into the
    # bounded embargo portion and the rest as purge.
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


# Contiguous partitioning of [0, T) into N groups; remainder absorbed by the last group.
def _compute_group_bounds(T: int, n_groups: int) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` with exclusive end; groups 0..N-2 have size ⌊T/N⌋."""
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


# Overlap-based purging (AFML Snippet 7.1): remove training rows whose label horizon touches any test group.
def _purge_train(
    train_idx: set[int],
    test_groups: tuple[int, ...],
    group_bounds: list[tuple[int, int]],
    index: pd.DatetimeIndex,
    t1: pd.Series,
) -> set[int]:
    """Return training indices to purge based on AFML's three label-overlap conditions."""
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

            # Condition 1: training observation starts inside the test window.
            if t_test_start <= t_i_start <= t_test_end:
                to_purge.add(i)
                continue

            # Condition 2: training label resolves inside the test window.
            if t_test_start <= t_i_end <= t_test_end:
                to_purge.add(i)
                continue

            # Condition 3: training label horizon straddles the entire test window.
            if t_i_start <= t_test_start and t_test_end <= t_i_end:
                to_purge.add(i)
                continue

    return to_purge


# One-sided embargo (AFML §7.4.2): only training rows AFTER each test group are blocked.
def _embargo_train(
    train_idx: set[int],
    test_groups: tuple[int, ...],
    group_bounds: list[tuple[int, int]],
    T: int,
    embargo_len: int,
) -> set[int]:
    """Return training indices in the post-test embargo zone.

    No pre-test embargo is needed because training labels resolving before the test
    starts cannot carry future information.
    """
    to_embargo = set()

    for g in test_groups:
        _, g_end = group_bounds[g]
        embargo_end = min(g_end + embargo_len, T)
        for i in range(g_end, embargo_end):
            if i in train_idx:
                to_embargo.add(i)

    return to_embargo