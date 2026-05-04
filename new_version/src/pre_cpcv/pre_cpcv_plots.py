"""
4.3) Plotting Utilities
============================
Visualisation helpers for the pre-CPCV stages of the pipeline:
labelling diagnostics (CUSUM filter, triple-barrier examples, label
distribution) and feature EDA (distributions, correlation matrix,
mutual information against the label).

Each function accepts a zoom window for the time-series plots and
returns the matplotlib Figure so the caller can save, restyle, or
embed it as needed. The functions do not call ``plt.show()``; the
notebook is responsible for displaying or saving the returned
Figure. This keeps the helpers re-usable for the thesis figure
exports, where a different rendering backend may be active.

Conventions
-----------
- All functions take pandas objects (Series or DataFrame), never
  numpy arrays, so the index is preserved for date-aware plotting.
- Time-window parameters (``zoom_start``, ``zoom_end``) accept
  anything ``pd.Timestamp`` can parse: ISO strings, ``datetime``
  objects, or ``pd.Timestamp`` directly.
- Threshold parameters (kurtosis, correlation, MI) default to the
  values used in the previous notebook iteration. Override at the
  call site to surface different feature subsets.

Functions
---------
plot_cusum_filter
    Two-panel zoom plot of log returns and CUSUM cumulative sums.
plot_tbl_examples
    Side-by-side panels showing one event per barrier-touch type
    (profit / stop loss / vertical) within a zoom window.
plot_label_distribution
    Donut chart of label counts with absolute and relative numbers
    printed below.
plot_feature_distributions
    Grid of histograms with kurtosis flagging.
plot_feature_correlation
    Annotated heat-map of the feature correlation matrix.
plot_feature_label_mutual_info
    Horizontal bar chart of mutual information between each feature
    and the binary label.
plot_adf_stationarity
    Augmented Dickey-Fuller test per feature, with a horizontal bar
    chart of p-values and a threshold line at the chosen
    significance level.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


# =====================================================================
# 1. CUSUM filter visualisation
# =====================================================================
def plot_cusum_filter(
    log_returns: pd.Series,
    t_events: pd.DatetimeIndex,
    h: float,
    zoom_start: str = "2026-01-01",
    zoom_end: str = "2026-03-27",
    figsize: tuple[int, int] = (16, 8),
) -> plt.Figure:
    """Two-panel CUSUM diagnostic for a chosen zoom window.

    The top panel shows daily log returns coloured by sign, with
    blue dashed verticals at every CUSUM event in the window. The
    bottom panel shows the running ``s_pos`` and ``s_neg``
    accumulators with the +/- ``h`` thresholds and the same event
    markers, so the reader can see the cumulative drift cross the
    threshold and reset.

    Parameters
    ----------
    log_returns : pd.Series
        Daily log returns indexed on date.
    t_events : pd.DatetimeIndex
        CUSUM event timestamps as returned by ``cusum_filter``.
    h : float
        CUSUM threshold (the same value used to generate
        ``t_events``).
    zoom_start, zoom_end : str
        Inclusive date range for the zoom window.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    plt.Figure
        The constructed figure (caller is responsible for showing
        or saving).
    """
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    mask = (log_returns.index >= zoom_start) & (log_returns.index <= zoom_end)
    lr_zoom = log_returns.loc[mask]
    events_zoom = t_events[(t_events >= zoom_start) & (t_events <= zoom_end)]

    # Top panel: log returns + CUSUM events
    ax1 = axes[0]
    ax1.bar(
        lr_zoom.index, lr_zoom.values, width=0.8, alpha=0.6,
        color=["#2ecc71" if r > 0 else "#e74c3c" for r in lr_zoom.values],
        label="Log returns",
    )
    for ev in events_zoom:
        ax1.axvline(ev, color="#3498db", alpha=0.7, linewidth=1.2, linestyle="--")
    if len(events_zoom) > 0:
        ax1.axvline(
            events_zoom[0], color="#3498db", alpha=0.7,
            linewidth=1.2, linestyle="--",
            label=f"CUSUM events ({len(events_zoom)})",
        )
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.set_ylabel("Log return")
    ax1.set_title(
        f"CUSUM filter -- {pd.Timestamp(zoom_start).date()} to "
        f"{pd.Timestamp(zoom_end).date()}",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=9)

    # Bottom panel: cumulative sums S+ and S-
    ax2 = axes[1]
    s_pos_series, s_neg_series = [], []
    s_pos, s_neg = 0.0, 0.0
    for t, r in lr_zoom.items():
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if t in events_zoom:
            s_pos_series.append(0.0)
            s_neg_series.append(0.0)
            s_pos, s_neg = 0.0, 0.0
        else:
            s_pos_series.append(s_pos)
            s_neg_series.append(s_neg)

    ax2.fill_between(
        lr_zoom.index, s_pos_series, 0,
        alpha=0.3, color="#2ecc71", label="S+ (upward)",
    )
    ax2.fill_between(
        lr_zoom.index, s_neg_series, 0,
        alpha=0.3, color="#e74c3c", label="S- (downward)",
    )
    ax2.axhline(h, color="#2ecc71", linewidth=1, linestyle=":",
                label=f"h = +{h:.4f}")
    ax2.axhline(-h, color="#e74c3c", linewidth=1, linestyle=":",
                label=f"h = -{h:.4f}")
    for ev in events_zoom:
        ax2.axvline(ev, color="#3498db", alpha=0.5,
                    linewidth=1, linestyle="--")
    ax2.set_ylabel("Cumulative sum")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    print(f"CUSUM events in window: {len(events_zoom)}")
    print(f"Total events (full sample): {len(t_events)}")

    return fig


# =====================================================================
# 2. Triple-barrier examples
# =====================================================================
def plot_tbl_examples(
    bins: pd.DataFrame,
    close: pd.Series,
    daily_vol: pd.Series,
    pt_sl: tuple[float, float],
    num_days: int,
    zoom_start: str = "2026-01-01",
    zoom_end: str = "2026-04-07",
    panel_width: int = 6,
    panel_height: float = 5.5,
) -> plt.Figure | None:
    """Side-by-side panels: one TBL example per barrier-touch type.

    Walks ``bins`` within the zoom window, classifies each event by
    which barrier was actually hit (profit target, stop loss, or
    vertical), then picks one representative event per type
    (preferring 3+ day holds for visible price action) and plots
    its price path with the barriers and entry/exit markers.

    Parameters
    ----------
    bins : pd.DataFrame
        Output of ``triple_barrier_labels``, with columns
        ``['ret', 'bin', 't1']`` indexed on event timestamps.
    close : pd.Series
        Close prices from the data loader.
    daily_vol : pd.Series
        EWMA daily volatility (used to recompute the barriers for
        each example).
    pt_sl : tuple[float, float]
        ``(upper_multiplier, lower_multiplier)`` used in labelling.
    num_days : int
        Vertical-barrier horizon in calendar days.
    zoom_start, zoom_end : str
        Inclusive date range to search for examples.
    panel_width, panel_height : numeric
        Per-panel size in inches.

    Returns
    -------
    plt.Figure or None
        ``None`` if no events fall inside the zoom window.
    """
    bins_zoom = bins[(bins.index >= zoom_start) & (bins.index <= zoom_end)].copy()

    classified: dict[str, list] = {"profit": [], "stop_loss": [], "vertical": []}

    for t0 in bins_zoom.index:
        row = bins_zoom.loc[t0]
        t1 = row["t1"]
        label = int(row["bin"])

        p0 = close.loc[t0]
        vol = daily_vol.loc[t0]
        upper = p0 * (1.0 + pt_sl[0] * vol)
        lower = p0 * (1.0 - pt_sl[1] * vol)

        path = close.loc[t0:t1]
        hit_upper = (path >= upper).any()
        hit_lower = (path <= lower).any()

        if hit_upper and label == 1:
            classified["profit"].append(t0)
        elif hit_lower and label == -1:
            classified["stop_loss"].append(t0)
        else:
            classified["vertical"].append(t0)

    print(
        f"Barrier classification "
        f"({pd.Timestamp(zoom_start).date()} to "
        f"{pd.Timestamp(zoom_end).date()}):"
    )
    print(f"  Profit target hit:  {len(classified['profit'])}")
    print(f"  Stop loss hit:      {len(classified['stop_loss'])}")
    print(f"  Vertical barrier:   {len(classified['vertical'])}")

    examples: dict[str, pd.Timestamp] = {}
    for key in ["profit", "stop_loss", "vertical"]:
        candidates = classified[key]
        if not candidates:
            continue
        holds = [(t0, (bins.loc[t0, "t1"] - t0).days) for t0 in candidates]
        long_holds = [(t0, h) for t0, h in holds if h >= 3]
        if long_holds:
            examples[key] = long_holds[len(long_holds) // 2][0]
        else:
            examples[key] = holds[0][0]

    plot_order = [
        (key, examples[key])
        for key in ["profit", "stop_loss", "vertical"]
        if key in examples
    ]

    if not plot_order:
        print("No TBL events found in the zoom window. Try a wider range.")
        return None

    n_examples = len(plot_order)
    fig, axes = plt.subplots(
        1, n_examples,
        figsize=(panel_width * n_examples, panel_height),
        sharey=False,
    )
    if n_examples == 1:
        axes = [axes]

    panel_colors = {
        "profit": "#2ecc71", "stop_loss": "#e74c3c", "vertical": "#95a5a6",
    }
    panel_titles = {
        "profit": "Take profit",
        "stop_loss": "Stop loss",
        "vertical": "Vertical barrier",
    }

    for ax, (panel_key, t0) in zip(axes, plot_order):
        row = bins.loc[t0]
        t1 = row["t1"]
        ret = row["ret"]

        p0 = close.loc[t0]
        vol = daily_vol.loc[t0]
        upper = p0 * (1.0 + pt_sl[0] * vol)
        lower = p0 * (1.0 - pt_sl[1] * vol)
        vert_date = t0 + pd.Timedelta(days=num_days)

        path_start = t0 - pd.Timedelta(days=1)
        path_end = t1 + pd.Timedelta(days=2)
        path = close.loc[path_start:path_end]

        if len(path) < 5:
            path_end = t0 + pd.Timedelta(days=max(num_days + 2, 7))
            path = close.loc[path_start:path_end]

        pre_entry = path.loc[:t0]
        during = path.loc[t0:t1]
        post_exit = path.loc[t1:]

        if len(pre_entry) > 0:
            ax.plot(pre_entry.index, pre_entry.values,
                    color="#bdc3c7", linewidth=1, zorder=2)
        ax.plot(during.index, during.values,
                color="#2c3e50", linewidth=2, zorder=3)
        if len(post_exit) > 1:
            ax.plot(post_exit.index, post_exit.values,
                    color="#bdc3c7", linewidth=1, zorder=2)

        ax.scatter([t0], [p0], color="#3498db", s=100, zorder=5,
                   edgecolors="white", linewidth=1.5, label="Entry")

        exit_price = (
            close.loc[t1] if t1 in close.index else p0 * (1 + ret)
        )
        ax.scatter([t1], [exit_price], color=panel_colors[panel_key], s=100,
                   zorder=5, marker="D", edgecolors="white", linewidth=1.5,
                   label=f"Exit ({(t1 - t0).days}d)")

        barrier_x = [t0, min(vert_date, path.index[-1])]
        ax.hlines(upper, barrier_x[0], barrier_x[1],
                  color="#2ecc71", linewidth=1.2, linestyle="--", alpha=0.8,
                  label=f"Profit: ${upper:,.0f}")
        ax.hlines(lower, barrier_x[0], barrier_x[1],
                  color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.8,
                  label=f"Stop: ${lower:,.0f}")

        if vert_date <= path.index[-1]:
            ax.axvline(vert_date, color="#95a5a6", linewidth=1.2,
                       linestyle=":", alpha=0.8,
                       label=f"Vertical ({num_days}d)")

        zone = path.loc[t0:vert_date]
        if len(zone) > 0:
            ax.fill_between(zone.index, lower, upper,
                            alpha=0.05, color="#3498db")

        ax.annotate(
            f"ret = {ret:+.2%}",
            xy=(t1, exit_price),
            xytext=(15, 15 if ret > 0 else -20),
            textcoords="offset points",
            fontsize=9, fontweight="bold",
            color=panel_colors[panel_key],
            arrowprops=dict(arrowstyle="->",
                            color=panel_colors[panel_key], lw=0.8),
        )

        ax.set_title(
            f"{panel_titles[panel_key]}\n"
            f"{t0.strftime('%b %d')} -> {t1.strftime('%b %d')} "
            f"({(t1 - t0).days} days)",
            fontsize=11, fontweight="bold", color=panel_colors[panel_key],
        )
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(alpha=0.15)
        ax.set_xlabel("")

    fig.suptitle(
        f"Triple-barrier labeling -- pt_sl={pt_sl}, num_days={num_days}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    print(f"\nTBL events in window: {len(bins_zoom)}")
    holding = (bins_zoom["t1"] - bins_zoom.index).dt.days
    if len(holding) > 0:
        print(f"  Avg holding period: {holding.mean():.1f} days")

    return fig


# =====================================================================
# 3. Label distribution
# =====================================================================
def plot_label_distribution(
    bins: pd.DataFrame,
    figsize: tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Donut chart of label distribution with class-count summary.

    Parameters
    ----------
    bins : pd.DataFrame
        Output of ``triple_barrier_labels`` (uses the ``bin`` column).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    plt.Figure
        The constructed figure.
    """
    counts = bins["bin"].value_counts().sort_index()
    label_map = {-1: "-1 (Down)", 1: "1 (Up)"}
    labels = [label_map.get(x, str(x)) for x in counts.index]
    colors = ["#800000", "#0b6623"][:len(counts)]

    fig = plt.figure(figsize=figsize)
    patches, _, _ = plt.pie(
        counts, labels=labels, autopct="%1.1f%%",
        startangle=140, colors=colors, pctdistance=0.85,
    )

    centre_circle = plt.Circle((0, 0), 0.25, fc="white")
    plt.gcf().gca().add_artist(centre_circle)

    plt.legend(
        patches, labels, title="Price Movement",
        loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
    )
    plt.title("Label Distribution", fontsize=16, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()

    print(f"Total samples: {len(bins)}")
    for cls, n in counts.items():
        print(f"  Class {label_map.get(cls, cls)}: {n} ({n/len(bins)*100:.1f}%)")

    return fig


# =====================================================================
# 4. Feature distributions
# =====================================================================
def plot_feature_distributions(
    feature_matrix: pd.DataFrame,
    n_cols: int = 4,
    kurtosis_threshold: float = 10.0,
    bins_per_hist: int = 50,
    panel_height: int = 3,
) -> plt.Figure:
    """Grid of feature histograms with kurtosis flagging.

    Each panel shows the distribution of one feature. Panels for
    features with kurtosis above ``kurtosis_threshold`` get a red
    title to surface candidates that may saturate KAN spline
    ranges or otherwise cause numerical issues during training.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Aligned feature matrix (TA + math + external + lag).
    n_cols : int
        Number of columns in the histogram grid.
    kurtosis_threshold : float
        Excess kurtosis above which to flag a feature (Fisher's
        definition; normal distribution has kurtosis 0).
    bins_per_hist : int
        Number of histogram bins.
    panel_height : int
        Per-row height in inches.

    Returns
    -------
    plt.Figure
        The constructed figure.
    """
    feat_cols = feature_matrix.columns.tolist()
    n_rows = int(np.ceil(len(feat_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, panel_height * n_rows))
    axes = axes.flatten()

    flagged: list[tuple[str, float]] = []
    last_idx = -1
    for i, col in enumerate(feat_cols):
        data = feature_matrix[col].dropna()
        axes[i].hist(data, bins=bins_per_hist, edgecolor="black", alpha=0.7)
        kurt = data.kurtosis()
        title_suffix = f" [kurt={kurt:.1f}]" if kurt > kurtosis_threshold else ""
        if kurt > kurtosis_threshold:
            axes[i].set_title(f"{col}{title_suffix}", fontsize=9, color="red")
            flagged.append((col, kurt))
        else:
            axes[i].set_title(f"{col}{title_suffix}", fontsize=9)
        last_idx = i

    for j in range(last_idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions", y=1.01, fontsize=13)
    plt.tight_layout()

    if flagged:
        print(
            f"Flagged: features with kurtosis > {kurtosis_threshold} "
            f"(may saturate KAN spline ranges):"
        )
        for col, k in flagged:
            print(f"  {col}: kurtosis = {k:.1f}")
    else:
        print(f"OK: no features with kurtosis > {kurtosis_threshold}.")

    return fig


# =====================================================================
# 5. Feature correlation matrix
# =====================================================================
def plot_feature_correlation(
    feature_matrix: pd.DataFrame,
    corr_threshold: float = 0.9,
    annotate: bool = True,
) -> plt.Figure:
    """Annotated heat-map of the feature correlation matrix.

    Drops rows with any NaN before computing correlations. Flags
    pairs whose absolute correlation exceeds ``corr_threshold`` so
    the caller can consider redundancy.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Aligned feature matrix.
    corr_threshold : float
        Absolute correlation above which to flag a pair.
    annotate : bool
        Whether to write the numeric value in each heat-map cell.
        Set False for very large feature sets where the annotation
        becomes illegible.

    Returns
    -------
    plt.Figure
        The constructed figure.
    """
    corr = feature_matrix.dropna().corr()
    n = len(corr.columns)

    fig, ax = plt.subplots(
        figsize=(max(20, n * 0.55), max(18, n * 0.5))
    )
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(corr.columns, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    if annotate:
        for i in range(n):
            for j in range(n):
                val = corr.iloc[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color=color)

    ax.set_title(
        "Feature Correlation Matrix",
        fontsize=18, fontweight="bold", pad=20,
    )
    plt.tight_layout()

    high_corr = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr.iloc[i, j]
            if abs(r) > corr_threshold:
                high_corr.append((corr.columns[i], corr.columns[j], r))

    if high_corr:
        print(f"Flagged: {len(high_corr)} pair(s) with |correlation| > {corr_threshold}:")
        for c1, c2, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f"  {c1} <-> {c2}: {r:.3f}")
    else:
        print(f"OK: no feature pairs with |correlation| > {corr_threshold}.")

    return fig


# =====================================================================
# 6. Feature-label mutual information
# =====================================================================
def plot_feature_label_mutual_info(
    feature_matrix: pd.DataFrame,
    bins: pd.DataFrame,
    mi_threshold: float = 1e-6,
    n_neighbors: int = 5,
    seed: int = 42,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Horizontal bar chart of feature-vs-label mutual information.

    Aligns the feature matrix to labelled events, drops rows with
    any NaN, and computes mutual information between each feature
    and the binary label using ``sklearn.feature_selection.
    mutual_info_classif``. Flags features with near-zero MI as
    removal candidates.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Aligned feature matrix.
    bins : pd.DataFrame
        Output of ``triple_barrier_labels`` (uses the ``bin`` column).
    mi_threshold : float
        MI value below which to flag a feature as near-zero.
    n_neighbors : int
        ``mutual_info_classif`` k-NN parameter.
    seed : int
        Random seed for reproducibility (the k-NN MI estimator is
        stochastic).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    plt.Figure
        The constructed figure.
    """
    aligned = feature_matrix.loc[feature_matrix.index.isin(bins.index)].copy()
    y_aligned = bins.loc[aligned.index, "bin"]

    mask = aligned.notna().all(axis=1)
    X_clean = aligned.loc[mask]
    y_clean = y_aligned.loc[mask]

    mi = mutual_info_classif(
        X_clean, y_clean, random_state=seed, n_neighbors=n_neighbors,
    )
    mi_series = pd.Series(mi, index=X_clean.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    mi_series.plot.barh(ax=ax, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Mutual Information (nats)")
    ax.set_title("Feature-Label Mutual Information")
    ax.invert_yaxis()
    plt.tight_layout()

    zero_mi = mi_series[mi_series < mi_threshold]
    if len(zero_mi):
        print(
            f"Flagged: {len(zero_mi)} feature(s) with near-zero MI "
            f"(removal candidates): {list(zero_mi.index)}"
        )
    else:
        print("OK: all features show non-zero mutual information with the label.")

    return fig


# =====================================================================
# 7. ADF stationarity test
# =====================================================================
def plot_adf_stationarity(
    feature_matrix: pd.DataFrame,
    significance: float = 0.05,
    maxlag: int = 14,
    autolag: str = "AIC",
    min_obs: int = 50,
    figsize: tuple[int, int] = (12, 10),
) -> tuple[plt.Figure, pd.DataFrame]:
    """Augmented Dickey-Fuller test per feature with a p-value bar chart.

    Runs the ADF test on each feature column, reporting the test
    statistic and p-value. The null hypothesis is that the series
    has a unit root (i.e. is non-stationary), so a p-value below
    ``significance`` rejects the null and flags the feature as
    stationary. Constant or near-constant features are skipped
    (the test is undefined on them).

    The bar chart sorts features by p-value ascending so the
    strongest stationarity rejections appear at the top. A vertical
    line at ``significance`` makes the threshold visible. Bars
    extending past the threshold (failing to reject the null)
    are coloured red, the others green.

    Use the returned DataFrame to decide which columns need
    fractional differencing (FFD) before they enter the CPCV
    pipeline. Cumulative or trending features (e.g. OBV, ATR)
    typically fail this test and need FFD; stationary features
    do not.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Aligned feature matrix.
    significance : float
        Significance level for the stationary / non-stationary
        decision. Default 0.05.
    maxlag : int
        Maximum lag for the ADF test. The autolag procedure
        chooses the optimal lag up to this cap.
    autolag : str
        Information criterion for autolag selection. ``"AIC"``,
        ``"BIC"``, ``"t-stat"``, or None.
    min_obs : int
        Minimum number of non-NaN observations required to run
        the test. Features with fewer observations are skipped
        with a warning.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : plt.Figure
        The constructed bar chart.
    adf_df : pd.DataFrame
        Per-feature ADF results indexed by feature name with
        columns ``["adf_stat", "p_value", "stationary"]``. The
        ``stationary`` column is a boolean derived from
        ``p_value < significance``.
    """
    adf_results = []
    for col in feature_matrix.columns:
        series = feature_matrix[col].dropna()
        if len(series) < min_obs:
            print(f"Skipping {col}: only {len(series)} non-NaN observations.")
            continue
        if series.nunique() <= 1:
            print(f"Skipping constant feature: {col}")
            continue
        stat, pval, *_ = adfuller(series, maxlag=maxlag, autolag=autolag)
        adf_results.append({
            "feature": col, "adf_stat": stat, "p_value": pval,
        })

    if not adf_results:
        raise ValueError(
            "plot_adf_stationarity: no features passed the min-obs / "
            "constant filter; nothing to test."
        )

    adf_df = pd.DataFrame(adf_results).set_index("feature")
    adf_df["stationary"] = adf_df["p_value"] < significance
    adf_df = adf_df.sort_values("p_value", ascending=True)

    # Bar chart: sort by p-value ascending so the most-stationary
    # features sit at the top and the borderline non-stationary ones
    # cluster near the threshold line.
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2ecc71" if s else "#e74c3c" for s in adf_df["stationary"]]
    ax.barh(
        adf_df.index, adf_df["p_value"],
        color=colors, edgecolor="black", alpha=0.75,
    )
    ax.axvline(
        significance, color="black", linestyle="--", linewidth=1.2,
        label=f"alpha = {significance}",
    )
    ax.set_xlabel("ADF p-value (smaller = more stationary)")
    ax.set_xlim(0, max(1.0, adf_df["p_value"].max() * 1.05))
    ax.set_title(
        "ADF Stationarity Test per Feature\n"
        "H0: unit root (non-stationary)",
        fontsize=12, fontweight="bold",
    )
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    n_total = len(adf_df)
    n_stationary = int(adf_df["stationary"].sum())
    n_non_stat = n_total - n_stationary
    print(f"\nADF test results (alpha = {significance}):")
    print(f"  Stationary:     {n_stationary} / {n_total}")
    print(f"  Non-stationary: {n_non_stat} / {n_total}")

    if n_non_stat > 0:
        non_stat_features = adf_df.loc[~adf_df["stationary"]].index.tolist()
        print(
            f"\nNon-stationary feature(s) "
            f"(candidates for fractional differencing):"
        )
        for col in non_stat_features:
            pval = adf_df.loc[col, "p_value"]
            print(f"  {col:25s}  p = {pval:.4f}")

    return fig, adf_df