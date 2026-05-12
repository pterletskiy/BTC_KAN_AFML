"""
5) Pre-CPCV EDA
============================
Plotting helpers for the labelling diagnostics (CUSUM filter, triple-barrier
examples, label distribution) and feature-side EDA (distributions, correlation,
mutual information, ADF stationarity). Each function returns the matplotlib
Figure and does not call ``plt.show()``; the notebook is responsible for
display or export.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


# Replay labeling.cusum_filter and record per-timestamp accumulator state.
def _replay_cusum_state(returns: pd.Series, h: float) -> tuple[pd.Series, pd.Series]:
    """Return ``(s_pos, s_neg)`` Series indexed identically to ``returns``.

    Used by ``plot_cusum_filter`` so the displayed trajectories match the
    filter's internal state, including carry-over at the zoom-window start.
    Resets only the accumulator that triggered, mirroring ``cusum_filter``.
    """
    s_pos, s_neg = 0.0, 0.0
    s_pos_hist, s_neg_hist = [], []
    for t, r in returns.items():
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos >= h:
            s_pos = 0.0
        elif s_neg <= -h:
            s_neg = 0.0
        s_pos_hist.append(s_pos)
        s_neg_hist.append(s_neg)
    return (
        pd.Series(s_pos_hist, index=returns.index, name="s_pos"),
        pd.Series(s_neg_hist, index=returns.index, name="s_neg"),
    )


# --- 1. CUSUM filter: zoom-window diagnostic of returns + S+/S- accumulators ---
def plot_cusum_filter(returns: pd.Series, t_events: pd.DatetimeIndex, h: float,
                      zoom_start: str = "2026-01-01", zoom_end: str = "2026-04-01",
                      figsize: tuple[int, int] = (16, 8)) -> plt.Figure:
    """Two-panel CUSUM diagnostic: returns on top, S+/S- accumulators below.

    The bottom panel makes the cumulative-drift crossings visible against the
    +/- ``h`` thresholds, which is the conceptually load-bearing part of the
    filter.
    """
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]})

    mask = (returns.index >= zoom_start) & (returns.index <= zoom_end)
    r_zoom = returns.loc[mask]
    events_zoom = t_events[(t_events >= zoom_start) & (t_events <= zoom_end)]

    # Top panel: returns, colour-coded by sign, with a vertical line at each event.
    ax1 = axes[0]
    ax1.bar(
        r_zoom.index, r_zoom.values, width=0.8, alpha=0.6,
        color=["#2ecc71" if r > 0 else "#e74c3c" for r in r_zoom.values],
        label="Returns")
    # Single labelled axvline per event; label only the first so the legend stays clean.
    for i, ev in enumerate(events_zoom):
        label = f"CUSUM events ({len(events_zoom)})" if i == 0 else None
        ax1.axvline(
            ev, color="#3498db", alpha=0.7, linewidth=1.2, linestyle="--",
            label=label)
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.set_ylabel("Return")
    ax1.set_title(
        f"CUSUM filter -- {pd.Timestamp(zoom_start).date()} to "
        f"{pd.Timestamp(zoom_end).date()}",
        fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)

    ax2 = axes[1]

    # Faithful state reconstruction: replay over the full series, then slice by mask,
    # so the accumulator inherits its actual historical context at the zoom start.
    s_pos_full, s_neg_full = _replay_cusum_state(returns, h)
    s_pos_zoom = s_pos_full.loc[mask]
    s_neg_zoom = s_neg_full.loc[mask]

    ax2.fill_between(
        s_pos_zoom.index, s_pos_zoom.values, 0,
        alpha=0.3, color="#2ecc71", label="S+ (upward)")
    ax2.fill_between(
        s_neg_zoom.index, s_neg_zoom.values, 0,
        alpha=0.3, color="#e74c3c", label="S- (downward)")
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


# --- 2. Triple-barrier examples: one labelled event per barrier-touch type ---
def plot_tbl_examples(bins: pd.DataFrame, close: pd.Series, daily_vol: pd.Series, pt_sl: tuple[float, float],
                      num_days: int, zoom_start: str = "2026-01-01", zoom_end: str = "2026-04-07", panel_width: int = 6,
                      panel_height: float = 5.5) -> plt.Figure | None:
    """One representative event per barrier-touch type within the zoom window.

    Prefers events with 3+ day holding periods so the price action is visible.
    Returns ``None`` when no events fall inside the window.
    """
    bins_zoom = bins[(bins.index >= zoom_start) & (bins.index <= zoom_end)].copy()

    # Bucket every event by which barrier its label corresponds to: profit (+1 and upper hit),
    # stop-loss (-1 and lower hit), or vertical (anything else, including 0).
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
        f"{pd.Timestamp(zoom_end).date()}):")
    print(f"  Profit target hit:  {len(classified['profit'])}")
    print(f"  Stop loss hit:      {len(classified['stop_loss'])}")
    print(f"  Vertical barrier:   {len(classified['vertical'])}")

    # Pick one representative per type; prefer events with >=3 day holds for visible action.
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
        if key in examples]

    if not plot_order:
        print("No TBL events found in the zoom window. Try a wider range.")
        return None

    n_examples = len(plot_order)
    fig, axes = plt.subplots(
        1, n_examples,
        figsize=(panel_width * n_examples, panel_height),
        sharey=False)
    if n_examples == 1:
        axes = [axes]

    panel_colors = {
        "profit": "#2ecc71", "stop_loss": "#e74c3c", "vertical": "#95a5a6"}
    panel_titles = {
        "profit": "Take profit",
        "stop_loss": "Stop loss",
        "vertical": "Vertical barrier"}

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
            close.loc[t1] if t1 in close.index else p0 * (1 + ret))
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
                            color=panel_colors[panel_key], lw=0.8))

        ax.set_title(
            f"{panel_titles[panel_key]}\n"
            f"{t0.strftime('%b %d')} -> {t1.strftime('%b %d')} "
            f"({(t1 - t0).days} days)",
            fontsize=11, fontweight="bold", color=panel_colors[panel_key])
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(alpha=0.15)
        ax.set_xlabel("")

    fig.suptitle(
        f"Triple-barrier labeling -- pt_sl={pt_sl}, num_days={num_days}",
        fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    print(f"\nTBL events in window: {len(bins_zoom)}")
    holding = (bins_zoom["t1"] - bins_zoom.index).dt.days
    if len(holding) > 0:
        print(f"  Avg holding period: {holding.mean():.1f} days")

    return fig


# --- 3. Label distribution: donut chart of class counts, surfaces class imbalance ---
def plot_label_distribution(bins: pd.DataFrame, figsize: tuple[int, int] = (8, 5)) -> plt.Figure:
    """Donut chart of label distribution with class-count summary."""
    counts = bins["bin"].value_counts().sort_index()
    label_map = {-1: "-1 (Down)", 1: "1 (Up)"}
    labels = [label_map.get(x, str(x)) for x in counts.index]
    colors = ["#800000", "#0b6623"][:len(counts)]

    fig = plt.figure(figsize=figsize)
    patches, _, _ = plt.pie(
        counts, labels=labels, autopct="%1.1f%%",
        startangle=140, colors=colors, pctdistance=0.85)

    centre_circle = plt.Circle((0, 0), 0.25, fc="white")
    plt.gcf().gca().add_artist(centre_circle)

    plt.legend(
        patches, labels, title="Price Movement",
        loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title("Label Distribution", fontsize=16, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()

    print(f"Total samples: {len(bins)}")
    for cls, n in counts.items():
        print(f"  Class {label_map.get(cls, cls)}: {n} ({n/len(bins)*100:.1f}%)")

    return fig


# --- 4. Feature distributions: histogram grid, flags high-kurtosis compression candidates ---
def plot_feature_distributions(feature_matrix: pd.DataFrame, n_cols: int = 4, kurtosis_threshold: float = 10.0,
                               bins_per_hist: int = 50, panel_height: int = 3) -> plt.Figure:
    """Histogram grid with red-titled panels for kurtosis > ``kurtosis_threshold``.

    Flags high-kurtosis features as candidates for compression transforms,
    since they tend to saturate KAN spline ranges and slow LSTM convergence.
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
            f"(may saturate KAN spline ranges):")
        for col, k in flagged:
            print(f"  {col}: kurtosis = {k:.1f}")
    else:
        print(f"OK: no features with kurtosis > {kurtosis_threshold}.")

    return fig


# --- 5. Correlation heat-map: surfaces redundant feature pairs above the threshold ---
def plot_feature_correlation(feature_matrix: pd.DataFrame, corr_threshold: float = 0.9, annotate: bool = True) -> plt.Figure:
    """Annotated correlation heat-map; surfaces pairs with |r| > ``corr_threshold``.

    Set ``annotate=False`` for very large feature sets where the cell labels
    become illegible.
    """
    # Complete-case analysis: every (i, j) cell uses the same row subset.
    corr = feature_matrix.dropna().corr()
    n = len(corr.columns)

    fig, ax = plt.subplots(
        figsize=(max(20, n * 0.55), max(18, n * 0.5)))
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
        fontsize=18, fontweight="bold", pad=20)
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


# --- 6. Mutual information bar chart: flags features with near-zero predictive signal ---
def plot_feature_label_mutual_info(feature_matrix: pd.DataFrame, bins: pd.DataFrame, mi_threshold: float = 1e-6,
                                   n_neighbors: int = 5, seed: int = 42, figsize: tuple[int, int] = (14, 10)) -> plt.Figure:
    """Per-feature mutual information against the binary label.

    Near-zero MI features are flagged as removal candidates; the ``seed`` and
    ``n_neighbors`` parameters are exposed because the k-NN MI estimator is
    stochastic.
    """
    aligned = feature_matrix.loc[feature_matrix.index.isin(bins.index)].copy()
    y_aligned = bins.loc[aligned.index, "bin"]

    mask = aligned.notna().all(axis=1)
    X_clean = aligned.loc[mask]
    y_clean = y_aligned.loc[mask]

    mi = mutual_info_classif(
        X_clean, y_clean, random_state=seed, n_neighbors=n_neighbors)
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
            f"(removal candidates): {list(zero_mi.index)}")
    else:
        print("OK: all features show non-zero mutual information with the label.")

    return fig


# --- 7. ADF test: per-feature stationarity scan; non-stationary cols flagged for FFD ---
def plot_adf_stationarity(feature_matrix: pd.DataFrame, significance: float = 0.05, maxlag: int = 14, autolag: str = "AIC",
                          min_obs: int = 50, figsize: tuple[int, int] = (12, 10)) -> tuple[plt.Figure, pd.DataFrame]:
    """ADF per feature; bars sorted by p-value with non-stationary candidates in red.

    H0 is a unit root, so rejection (p < significance) flags the series as
    stationary. Non-stationary features are returned as FFD candidates for the
    preprocessing layer; constant or near-constant series are skipped.
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
            "constant filter; nothing to test.")

    adf_df = pd.DataFrame(adf_results).set_index("feature")
    adf_df["stationary"] = adf_df["p_value"] < significance
    adf_df = adf_df.sort_values("p_value", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2ecc71" if s else "#e74c3c" for s in adf_df["stationary"]]
    ax.barh(
        adf_df.index, adf_df["p_value"],
        color=colors, edgecolor="black", alpha=0.75)
    ax.axvline(
        significance, color="black", linestyle="--", linewidth=1.2,
        label=f"alpha = {significance}")
    ax.set_xlabel("ADF p-value (smaller = more stationary)")
    ax.set_xlim(0, max(1.0, adf_df["p_value"].max() * 1.05))
    ax.set_title(
        "ADF Stationarity Test per Feature\n"
        "H0: unit root (non-stationary)",
        fontsize=12, fontweight="bold")
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
            f"(candidates for fractional differencing):")
        for col in non_stat_features:
            pval = adf_df.loc[col, "p_value"]
            print(f"  {col:25s}  p = {pval:.4f}")

    return fig, adf_df