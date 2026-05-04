"""
CPCV Plotting and Diagnostic Utilities
=======================================
Visualisation and verification helpers for the Combinatorial Purged
Cross-Validation (CPCV) configuration. The functions in this module
are meant to be called after the notebook has produced ``X``, ``t1``,
``splits``, ``path_map``, ``n_paths``, and ``split_info`` from the
CV cell, so they can show the partitioning, the train/test
arrangement, the purging/embargo behaviour, and a leakage audit.

Each function takes the CPCV inputs as parameters and returns either
a matplotlib Figure (for the visual diagnostics) or a pandas
DataFrame (for the leakage audit). The functions do not call
``plt.show()``; the notebook is responsible for displaying or
saving the returned Figure.

The functions assume ``X`` is already truncated to the analysis
window (post-CUSUM_START_DATE), so ``X.index[0]`` reflects the
first labelled event and the visualisations automatically span the
correct date range without an explicit start-date parameter.

Functions
---------
pick_demo_splits
    Helper to choose three illustrative split indices
    (contiguous-early, one-gap, contiguous-tail) given the
    combinations list.
plot_btc_with_groups
    BTC close price with the N CPCV groups shaded as coloured
    bands. Toggleable log/linear y-axis.
plot_train_test_timelines
    Three sub-panels showing train/test date partitioning for
    three representative splits.
print_purge_embargo_detail
    Per-split text dump of the boundary observations near each
    test group, showing which training rows were purged
    (overlapping labels) and which were embargoed.
audit_cpcv_leakage
    Full audit across all splits, returning a DataFrame and a
    print summary indicating whether any training observation
    has a label end-time inside a test group.
"""

import logging
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Eight-color palette for groups. Extends the original six-color set
# with teal and grey so each of up to N=8 groups is visually distinct.
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


# =====================================================================
# Helpers
# =====================================================================
def _compute_group_bounds(T: int, n_groups: int) -> list[tuple[int, int]]:
    """Return inclusive-start / exclusive-end positional bounds per group.

    Mirrors the partitioning used inside cv.py so the visualisations
    show exactly the same group boundaries the splits were generated
    from. Groups 0..N-2 have size floor(T/N); group N-1 absorbs the
    remainder.
    """
    base_size = T // n_groups
    bounds = []
    for g in range(n_groups):
        start = g * base_size
        end = (g + 1) * base_size if g < n_groups - 1 else T
        bounds.append((start, end))
    return bounds


def pick_demo_splits(
    all_combos: list[tuple[int, ...]], n_groups: int,
) -> list[int]:
    """Choose three illustrative splits for visualisation.

    Picks one contiguous-early split (test groups (0, 1)), one
    one-gap split (test groups (1, 3) with G2 between them), and
    one contiguous-tail split (test groups (n_groups-2, n_groups-1)).
    Returns their positional indices into ``all_combos`` so the
    caller can index into the ``splits`` list directly.

    Parameters
    ----------
    all_combos : list of tuple
        Output of ``list(combinations(range(n_groups), k))``.
    n_groups : int
        Number of CPCV groups.

    Returns
    -------
    list of int
        Three split indices, in the order
        [contiguous_early, one_gap, contiguous_tail].
    """
    contiguous_early = all_combos.index((0, 1))
    contiguous_tail = all_combos.index((n_groups - 2, n_groups - 1))
    gap_split = all_combos.index((1, 3))
    return [contiguous_early, gap_split, contiguous_tail]


# =====================================================================
# 1. BTC price with CPCV group partitions
# =====================================================================
def plot_btc_with_groups(
    X: pd.DataFrame,
    df_raw: pd.DataFrame,
    n_groups: int,
    use_log_scale: bool = True,
    figsize: tuple[int, int] = (14, 5),
    colors: list[str] | None = None,
) -> plt.Figure:
    """BTC close price with CPCV groups shaded as coloured bands.

    The plot spans only the analysis window (``X.index[0]`` to
    ``X.index[-1]``), which after CUSUM truncation starts at the
    first labelled event rather than at the raw-data start. The
    coloured bands show the N contiguous groups the CPCV splits
    are built from; group labels (G0, G1, ...) sit near the bottom
    of each band.

    Parameters
    ----------
    X : pd.DataFrame
        Aligned feature matrix; only its index is used.
    df_raw : pd.DataFrame
        Raw OHLCV data with a ``Close`` column. Must cover all of
        ``X.index``.
    n_groups : int
        Number of CPCV groups (the same N used in
        ``generate_cpcv_splits``).
    use_log_scale : bool
        If True (default), the y-axis is log-scaled so all groups
        across BTC's 250x price range are visually balanced. Set
        False to emphasise the price-level shift between early
        and recent groups.
    figsize : tuple
        Figure size in inches.
    colors : list of str, optional
        Eight-color palette overriding ``DEFAULT_GROUP_COLORS``.

    Returns
    -------
    plt.Figure
        The constructed figure.
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

    for g, (start, end) in enumerate(group_bounds):
        ax.axvspan(
            X.index[start], X.index[end - 1],
            color=colors[g], alpha=0.15,
        )

    ymin, ymax = ax.get_ylim()
    if use_log_scale:
        # Multiplicative offset for log scale.
        label_y = ymin * 1.5
    else:
        # 3% above bottom for linear scale.
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


# =====================================================================
# 2. Train/test timelines for representative splits
# =====================================================================
def plot_train_test_timelines(
    X: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_groups: int,
    k: int,
    demo_splits: list[int] | None = None,
    panel_height: float = 1.8,
    figsize_width: int = 14,
) -> plt.Figure:
    """Three-panel visualisation of CPCV train/test partitioning.

    For each of three representative splits (chosen automatically
    by ``pick_demo_splits`` if ``demo_splits`` is None), draws train
    points in steelblue and test points in crimson along a date
    axis. Test groups are also shaded with a faint crimson
    background so the reader can see which groups make up the test
    fold for each split.

    Parameters
    ----------
    X : pd.DataFrame
        Aligned feature matrix; only its index is used.
    splits : list of (train_idx, test_idx)
        Output of ``generate_cpcv_splits``.
    n_groups : int
        Number of CPCV groups.
    k : int
        Number of test groups per split.
    demo_splits : list of int, optional
        Three split indices to plot. If None, the function calls
        ``pick_demo_splits`` to choose contiguous-early, one-gap,
        and contiguous-tail.
    panel_height : float
        Per-panel height in inches.
    figsize_width : int
        Figure width in inches.

    Returns
    -------
    plt.Figure
        The constructed figure.
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

        ax.scatter(
            X.index[train_idx], np.zeros(len(train_idx)),
            c="steelblue", s=2, label="Train", zorder=2,
        )
        ax.scatter(
            X.index[test_idx], np.zeros(len(test_idx)),
            c="crimson", s=2, label="Test", zorder=3,
        )

        for g in test_groups:
            g_start, g_end = group_bounds[g]
            ax.axvspan(
                X.index[g_start], X.index[g_end - 1],
                color="crimson", alpha=0.08, zorder=1,
            )

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


# =====================================================================
# 3. Purging and embargo detail
# =====================================================================
def print_purge_embargo_detail(
    X: pd.DataFrame,
    t1: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_groups: int,
    k: int,
    demo_splits: list[int] | None = None,
) -> None:
    """Per-split text dump showing the rows around each test boundary.

    For the three (or however many were passed) demo splits, prints
    the last three training rows immediately before each test group
    (the rows most likely to overlap and be purged) and the first
    three training rows immediately after (the embargo zone). Each
    "before" row also gets a check on whether its label end-time
    ``t1`` resolves before the test starts; ``OK`` means the row
    is correctly purged, ``OVERLAP`` flags a leakage candidate the
    leakage audit should catch.

    This is a verification helper, not a plot. It produces no
    figure and returns nothing -- the value is in the printed
    output, which the notebook reader uses to confirm the purging
    and embargo logic is behaving as expected.

    Parameters
    ----------
    X : pd.DataFrame
        Aligned feature matrix; only its index is used.
    t1 : pd.Series
        Label end-times indexed identically to X.
    splits : list of (train_idx, test_idx)
        Output of ``generate_cpcv_splits``.
    n_groups : int
        Number of CPCV groups.
    k : int
        Number of test groups per split.
    demo_splits : list of int, optional
        Split indices to inspect. If None, uses ``pick_demo_splits``.
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

            train_before = sorted([i for i in train_idx if i < g_start])
            train_after = sorted([i for i in train_idx if i >= g_end])

            print(f"\n  Test G{g}: [{t_test_start.date()} -> {t_test_end.date()}]")

            if train_before:
                last_train_before = train_before[-1]
                purge_gap = g_start - last_train_before - 1
                print(f"  Pre-test gap (purged): {purge_gap} obs removed")
                for i in train_before[-3:]:
                    safe = t1.iloc[i] < t_test_start
                    flag = "OK" if safe else "OVERLAP"
                    print(
                        f"      idx={i:>4d} ({X.index[i].date()})  "
                        f"t1={t1.iloc[i].date()}  {flag}"
                    )

            if train_after:
                first_train_after = train_after[0]
                embargo_gap = first_train_after - g_end
                print(f"  Post-test gap (embargo): {embargo_gap} obs removed")
                for i in train_after[:3]:
                    print(f"      idx={i:>4d} ({X.index[i].date()})")

        print()


# =====================================================================
# 4. Leakage audit across all splits
# =====================================================================
def audit_cpcv_leakage(
    X: pd.DataFrame,
    t1: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_groups: int,
    k: int,
    split_info: dict | None = None,
) -> pd.DataFrame:
    """Per-split leakage audit checking that no training label end-time
    resolves inside a test group.

    For every split, walks the training rows whose entry timestamp
    falls before the test region and checks whether their label
    end-time ``t1`` resolves *during* the test region. Any such
    row is a leak: information about the test fold's outcome would
    bleed into the training set through the overlapping label
    horizon. AFML's purging step is supposed to remove these rows
    before they enter the train index, so a clean run reports zero
    leaks across all splits.

    The function prints a summary table (one line per split) and
    returns the same data as a DataFrame for further analysis or
    thesis-table export.

    Parameters
    ----------
    X : pd.DataFrame
        Aligned feature matrix; only its index is used.
    t1 : pd.Series
        Label end-times indexed identically to X.
    splits : list of (train_idx, test_idx)
        Output of ``generate_cpcv_splits``.
    n_groups : int
        Number of CPCV groups.
    k : int
        Number of test groups per split.
    split_info : dict, optional
        Output of ``get_split_info``. If passed, the function
        prints the number of backtest paths in the success message;
        otherwise the path count is omitted.

    Returns
    -------
    pd.DataFrame
        One row per split with columns
        ``["split", "test_groups", "train", "test", "leaks", "status"]``.
    """
    T = len(X)
    group_bounds = _compute_group_bounds(T, n_groups)
    all_combos = list(combinations(range(n_groups), k))

    n_leaks_total = 0
    audit_results = []

    for i, (train_idx, test_idx) in enumerate(splits):
        test_groups = all_combos[i]
        leak_count = 0

        for g in test_groups:
            g_start, g_end = group_bounds[g]
            t_test_start = X.index[g_start]
            t_test_end = X.index[g_end - 1]

            train_t1 = t1.iloc[train_idx]
            train_times = X.index[train_idx]

            leaks = train_t1[
                (train_times < t_test_start)
                & (train_t1 >= t_test_start)
                & (train_t1 <= t_test_end)
            ]
            leak_count += len(leaks)

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