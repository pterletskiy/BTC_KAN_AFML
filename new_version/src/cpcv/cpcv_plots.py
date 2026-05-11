"""
8) CV EDA
============================
Visualisation and verification helpers for the Combinatorial Purged
Cross-Validation (CPCV) configuration.

Each function takes pre-built CPCV inputs (``X``, ``t1``, ``splits``,
``path_map``, etc.) from the CV cell of the notebook and returns either a
matplotlib Figure or a pandas DataFrame. None of the plotting functions call
``plt.show()``; the notebook is responsible for display.

``X`` is assumed to be already truncated to the analysis window (post
``CUSUM_START_DATE``), so ``X.index[0]`` reflects the first labelled event
and every visualisation auto-spans the correct date range.
"""

import logging
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse cv.py's group-partition helper rather than maintaining a parallel copy here.
# Importing the private name across modules is a small convention break, but it
# eliminates the drift risk that a local duplicate would carry.
from src.cpcv.cv import _compute_group_bounds

logger = logging.getLogger(__name__)


# Eight-colour palette: one entry per CPCV group, distinct under normal vision.
DEFAULT_GROUP_COLORS = [
    "#e74c3c",  # red
    "#e67e22",  # orange
    "#f1c40f",  # yellow
    "#2ecc71",  # green
    "#3498db",  # blue
    "#9b59b6",  # purple
    "#1abc9c",  # teal
    "#7f8c8d",  # grey
]


# --- 1. Group-partition helpers --------------------------------------------
# _compute_group_bounds is imported from cv.py (see top of file); this section
# only contains helpers that are specific to the plotting / audit logic.


# Choose three illustrative splits for the train/test timeline plot.
def pick_demo_splits(
    all_combos: list[tuple[int, ...]], n_groups: int,
) -> list[int]:
    """Return positional indices for [contiguous-early, one-gap, contiguous-tail].

    The three splits cover the visually informative cases: a head-of-series test,
    a split with a gap between test groups, and an end-of-series test. Works for
    any ``k`` by deriving the contiguous and gapped target tuples from the combos
    themselves rather than assuming ``k=2``.
    """
    # Recover k from the first combination; all combos share the same length.
    k = len(all_combos[0])

    # Contiguous-early: the first k groups, e.g. (0,1) for k=2, (0,1,2) for k=3.
    contiguous_early = all_combos.index(tuple(range(k)))
    # Contiguous-tail: the last k groups, e.g. (N-2,N-1) for k=2, (N-3,N-2,N-1) for k=3.
    contiguous_tail = all_combos.index(tuple(range(n_groups - k, n_groups)))

    # Gap split: first combination that contains at least one non-adjacent pair of
    # test groups. Falls back to contiguous-early when none exist (k=N case).
    gap_split = next(
        (
            i for i, c in enumerate(all_combos)
            if any(c[j + 1] - c[j] > 1 for j in range(len(c) - 1))
        ),
        contiguous_early,
    )

    return [contiguous_early, gap_split, contiguous_tail]


# --- 2. BTC price with CPCV group partitions -------------------------------
# Big-picture plot that shows the N contiguous groups on top of the price series.
def plot_btc_with_groups(
    X: pd.DataFrame,
    df_raw: pd.DataFrame,
    n_groups: int,
    use_log_scale: bool = True,
    figsize: tuple[int, int] = (14, 5),
    colors: list[str] | None = None,
) -> plt.Figure:
    """BTC close price with CPCV groups shaded as coloured bands.

    Log scale is the default because BTC spans roughly 250× in price across the
    analysis window; on linear scale the early groups visually collapse.
    """
    if colors is None:
        colors = DEFAULT_GROUP_COLORS

    if n_groups > len(colors):
        raise ValueError(
            f"plot_btc_with_groups: n_groups={n_groups} exceeds "
            f"palette length {len(colors)}; pass a longer ``colors`` "
            "list or reduce N."
        )

    T = len(X)
    group_bounds = _compute_group_bounds(T, n_groups)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        X.index, df_raw.loc[X.index, "Close"],
        color="black", linewidth=0.8,
    )
    if use_log_scale:
        ax.set_yscale("log")

    # Shade each group with its palette colour at low alpha so the price line stays readable.
    for g, (start, end) in enumerate(group_bounds):
        ax.axvspan(
            X.index[start], X.index[end - 1],
            color=colors[g], alpha=0.15,
        )

    ymin, ymax = ax.get_ylim()
    # Label offset: multiplicative for log scale, additive for linear.
    if use_log_scale:
        label_y = ymin * 1.5
    else:
        label_y = ymin + (ymax - ymin) * 0.03

    for g, (start, end) in enumerate(group_bounds):
        mid_idx = (start + end) // 2
        ax.text(
            X.index[mid_idx], label_y, f"G{g}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=colors[g],
        )

    scale_label = "log scale" if use_log_scale else "linear scale"
    ax.set_ylabel(f"BTC Close (USD, {scale_label})")
    ax.set_xlabel("")
    ax.set_title(
        "BTC Price with CPCV Group Partitions",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    return fig


# --- 3. Train/test timelines for representative splits ---------------------
# Three-panel view of how train and test points scatter along the date axis.
def plot_train_test_timelines(
    X: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_groups: int,
    k: int,
    demo_splits: list[int] | None = None,
    panel_height: float = 1.8,
    figsize_width: int = 14,
) -> plt.Figure:
    """Three-panel scatter showing train (blue) and test (red) dates per representative split.

    If ``demo_splits`` is None, the panels show contiguous-early, one-gap, and
    contiguous-tail combinations chosen by ``pick_demo_splits``.
    """
    T = len(X)
    group_bounds = _compute_group_bounds(T, n_groups)
    all_combos = list(combinations(range(n_groups), k))

    if demo_splits is None:
        demo_splits = pick_demo_splits(all_combos, n_groups)

    fig, axes = plt.subplots(
        len(demo_splits), 1,
        figsize=(figsize_width, panel_height * len(demo_splits)),
        sharex=True,
    )

    for ax, split_id in zip(axes, demo_splits):
        train_idx, test_idx = splits[split_id]
        test_groups = all_combos[split_id]

        # Scatter train and test along y=0 so the panel reads as a 1-D date strip.
        ax.scatter(
            X.index[train_idx], np.zeros(len(train_idx)),
            c="steelblue", s=2, label="Train", zorder=2,
        )
        ax.scatter(
            X.index[test_idx], np.zeros(len(test_idx)),
            c="crimson", s=2, label="Test", zorder=3,
        )

        # Shade test groups so the reader sees which groups are out-of-sample.
        for g in test_groups:
            g_start, g_end = group_bounds[g]
            ax.axvspan(
                X.index[g_start], X.index[g_end - 1],
                color="crimson", alpha=0.08, zorder=1,
            )

        # Dashed verticals at every group boundary anchor the eye.
        for g in range(n_groups):
            g_start, _ = group_bounds[g]
            ax.axvline(
                X.index[g_start],
                color="gray", ls="--", lw=0.5, alpha=0.4,
            )

        ax.set_yticks([])
        ax.set_title(
            f"Split {split_id}: test groups {test_groups} -- "
            f"train={len(train_idx)}, test={len(test_idx)}",
            fontsize=9, fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=7, markerscale=3)

    plt.xlabel("Date")
    plt.suptitle(
        "CPCV Train/Test Timelines (3 Representative Splits)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    return fig


# --- 4. Purging and embargo boundary inspection ----------------------------
# Verification helper: dumps the rows around each test boundary for hand-checking.
def print_purge_embargo_detail(
    X: pd.DataFrame,
    t1: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_groups: int,
    k: int,
    demo_splits: list[int] | None = None,
) -> None:
    """Print the last 3 pre-test training rows and first 3 post-test training rows per demo split.

    For each pre-test row the function also checks whether its label end-time ``t1``
    resolves before the test starts: ``OK`` confirms the row's been correctly
    handled by purging, ``OVERLAP`` flags a leakage candidate that the audit
    function (``audit_cpcv_leakage``) should also catch.
    """
    T = len(X)
    group_bounds = _compute_group_bounds(T, n_groups)
    all_combos = list(combinations(range(n_groups), k))

    if demo_splits is None:
        demo_splits = pick_demo_splits(all_combos, n_groups)

    print("Purging and Embargo Verification (boundary observations)\n")

    sep = "-" * 60
    for split_id in demo_splits:
        train_idx, _ = splits[split_id]
        test_groups = all_combos[split_id]

        print(sep)
        print(f"Split {split_id} -- test groups {test_groups}")
        print(sep)

        for g in test_groups:
            g_start, g_end = group_bounds[g]
            t_test_start = X.index[g_start]
            t_test_end = X.index[g_end - 1]

            # Training rows immediately before and after the test region.
            train_before = sorted([i for i in train_idx if i < g_start])
            train_after = sorted([i for i in train_idx if i >= g_end])

            print(f"\n  Test G{g}: [{t_test_start.date()} -> {t_test_end.date()}]")

            if train_before:
                last_train_before = train_before[-1]
                purge_gap = g_start - last_train_before - 1
                print(f"  Pre-test gap (purged): {purge_gap} obs removed")
                # Inspect the three closest pre-test rows; their t1 must resolve before t_test_start.
                # NaN-t1 rows are flagged separately because they carry no label and so cannot leak;
                # _purge_train skips them too.
                for i in train_before[-3:]:
                    if pd.isna(t1.iloc[i]):
                        flag = "NaN-label"
                        t1_str = "NaT"
                    else:
                        safe = t1.iloc[i] < t_test_start
                        flag = "OK" if safe else "OVERLAP"
                        t1_str = str(t1.iloc[i].date())
                    print(
                        f"      idx={i:>4d} ({X.index[i].date()})  "
                        f"t1={t1_str}  {flag}"
                    )

            if train_after:
                first_train_after = train_after[0]
                embargo_gap = first_train_after - g_end
                print(f"  Post-test gap (embargo): {embargo_gap} obs removed")
                # Embargo is positional, so we just show the three closest post-test indices.
                for i in train_after[:3]:
                    print(f"      idx={i:>4d} ({X.index[i].date()})")

        print()


# --- 5. Leakage audit across all splits ------------------------------------
# Full audit: confirm no training label resolves inside any test group, and no
# training label horizon straddles any test group, across every split.
def _recover_test_groups(
    test_idx: np.ndarray,
    group_bounds: list[tuple[int, int]],
) -> tuple[int, ...]:
    """Recover the tuple of test-group IDs by checking which group bounds enclose ``test_idx``.

    Rather than trusting that the i-th split corresponds to the i-th combination from
    ``itertools.combinations`` (which is true at construction time but fragile if splits
    are later reordered), this reconstructs the test_groups from the indices themselves.
    """
    test_set = set(int(i) for i in test_idx)
    groups = []
    for g, (s, e) in enumerate(group_bounds):
        # A group is "in the test set" iff every one of its positional indices appears in test_idx.
        if all(i in test_set for i in range(s, e)):
            groups.append(g)
    return tuple(groups)


def audit_cpcv_leakage(
    X: pd.DataFrame,
    t1: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_groups: int,
    k: int,
    split_info: dict | None = None,
) -> pd.DataFrame:
    """Audit every split for label-overlap leakage; print a summary and return the per-split table.

    Checks AFML §7.4.1 conditions 2 and 3 (condition 1 cannot fire for contiguous group
    partitioning because train and test indices are disjoint by construction):
      - Condition 2: training label resolves inside a test group.
      - Condition 3: training label horizon straddles the entire test group.

    A clean run reports zero leaks for every split. NaN-t1 training rows are skipped
    because they carry no label and so cannot leak — ``_purge_train`` treats them the
    same way.
    """
    # Defensive: t1 must be positionally aligned with X for ``t1.iloc[train_idx]``
    # to return the right labels.
    if len(t1) != len(X):
        raise ValueError(
            f"audit_cpcv_leakage: t1 has length {len(t1)} but X has length "
            f"{len(X)}; they must be positionally aligned."
        )

    T = len(X)
    group_bounds = _compute_group_bounds(T, n_groups)

    n_leaks_total = 0
    audit_results = []

    for i, (train_idx, test_idx) in enumerate(splits):
        # Recover test_groups from the indices rather than trusting split position.
        test_groups = _recover_test_groups(test_idx, group_bounds)
        leak_count = 0

        # Sum leaks across every test group within the split.
        for g in test_groups:
            g_start, g_end = group_bounds[g]
            t_test_start = X.index[g_start]
            t_test_end = X.index[g_end - 1]

            train_t1 = t1.iloc[train_idx]
            train_times = X.index[train_idx]

            # Condition 2: training row starts before test AND its label resolves inside test.
            # NaN comparisons evaluate to False, so NaN-labelled rows fall out naturally.
            leaks_cond2 = train_t1[
                (train_times < t_test_start)
                & (train_t1 >= t_test_start)
                & (train_t1 <= t_test_end)
            ]
            # Condition 3: training label horizon straddles the entire test window
            # (row starts at or before test_start AND resolves at or after test_end).
            leaks_cond3 = train_t1[
                (train_times <= t_test_start)
                & (train_t1 >= t_test_end)
            ]
            leak_count += len(leaks_cond2) + len(leaks_cond3)

        n_leaks_total += leak_count
        audit_results.append({
            "split": i,
            "test_groups": test_groups,
            "train": len(train_idx),
            "test": len(test_idx),
            "leaks": leak_count,
        })

    audit_df = pd.DataFrame(audit_results)
    audit_df["status"] = audit_df["leaks"].apply(
        lambda x: "OK" if x == 0 else "FAIL"
    )

    print("CPCV Leakage Audit")
    print("=" * 65)
    for _, row in audit_df.iterrows():
        print(
            f"  Split {row['split']:2d} | groups {str(row['test_groups']):>7s} | "
            f"train={row['train']:>4d}  test={row['test']:>4d} | "
            f"leaks={row['leaks']:>2d} {row['status']}"
        )
    print("=" * 65)

    if n_leaks_total == 0:
        msg = f"OK -- all {len(splits)} splits passed. Zero leakage detected."
        print(msg)
        if split_info is not None and "n_paths" in split_info:
            print(
                f"     {split_info['n_paths']} backtest paths available "
                "for Sharpe ratio distribution."
            )
    else:
        print(f"FAIL -- {n_leaks_total} total leaking observations detected.")

    return audit_df