"""
14.3) Path Explorer
===================
Per-model multi-path equity curves (Brownian-motion-style visualisations).

Three views of the same idea, each driven by ``analysis['path_results']``:

  1. STATIC PER-MODEL — one chart for one model showing all paths. Exports
     cleanly to image for thesis PDFs.
  2. STATIC GRID — one subplot per model with all paths. Best for
     side-by-side variance comparison across the model lineup.
  3. INTERACTIVE SELECTOR — ipywidgets dropdown for live notebook
     exploration; not for the written thesis.

All three plot individual paths as thin lines and overlay the median
across paths as a thick reference line, so the reader sees both the
central tendency and the dispersion.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- 1. Helper: collect all path equity curves into a single DataFrame -----
# Forward-fill each path onto a common index so subsequent plotting routines align cleanly.
def collect_path_equities(model_path_data: dict, n_paths: int) -> pd.DataFrame:
    """Return a DataFrame where each column is one path's cumulative-equity curve.

    All paths are forward-filled to share a common index. Duplicate-index entries (which can
    appear when LSTM windowing produces overlapping timestamps across folds) are collapsed
    by taking the last value per timestamp before concatenation.
    """
    equities = []
    # Build a per-path cumulative-equity series.
    for path_id in range(n_paths):
        if path_id in model_path_data and len(model_path_data[path_id]["returns"]) > 0:
            equity = (1 + model_path_data[path_id]["returns"]).cumprod()
            equity.name = f"path_{path_id}"
            equities.append(equity)
    if not equities:
        return pd.DataFrame()

    # Standard concat; if any path has duplicate timestamps, retry after collapsing them.
    try:
        df = pd.concat(equities, axis=1)
    except Exception:
        equities = [e.groupby(e.index).last() for e in equities]
        df = pd.concat(equities, axis=1)

    df = df.ffill()
    # Force a DatetimeIndex so callers can slice with strings like ``df.loc["2023":]``.
    df.index = pd.to_datetime(df.index)
    return df


# --- 2. View 1: STATIC PER-MODEL — one model, all paths -------------------
# Single figure showing every path for one model, with the median overlaid as a thick line.
def plot_paths_for_model(
    model_name: str,
    results: dict,
    analysis: dict,
    log_scale: bool = True,
    since: str | None = None,
    figsize: tuple = (14, 5.5),
):
    """Plot all path equity curves for one model with the median overlaid.

    ``since`` accepts a year string (e.g. ``"2023"``) to slice and re-normalise to 1.0 at
    the start of the window. ``log_scale=True`` is recommended for full-history plots where
    compounding makes early values invisible on linear axes.
    """
    df = collect_path_equities(
        analysis["path_results"][model_name],
        results["n_paths"],
    )
    if df.empty:
        print(f"No path data for {model_name}.")
        return

    # Optional time slice + renormalisation to 1.0 at the start of the window.
    if since is not None:
        try:
            df = df.loc[since:]
        except Exception:
            pass
        if df.empty:
            print(f"No data for {model_name} since {since}.")
            return
        # Defensive: bfill+replace(0, ε) prevents division-by-zero on paths that start with NaN or 0.
        first_valid = df.bfill().iloc[0].replace(0, 1e-10)
        df = df / first_valid

    median = df.median(axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    # Individual paths as thin alpha-blended lines.
    for col in df.columns:
        ax.plot(df.index, df[col].values,
                linewidth=0.9, alpha=0.55,
                label=col.replace("_", " "))

    # Median as a thick reference line on top.
    ax.plot(median.index, median.values,
            color="black", linewidth=2.0,
            label="median", zorder=10)

    if log_scale:
        try:
            ax.set_yscale("log")
        except Exception:
            pass

    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.4)

    title_suffix = f" since {since}" if since else " (full history)"
    ax.set_title(
        f"{model_name}: path-by-path equity curves{title_suffix}",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Cumulative Equity" + (" (log scale)" if log_scale else ""))
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.show()


# --- 3. View 2: STATIC GRID — all models side by side ---------------------
# Grid of subplots for cross-model variance comparison; same per-subplot logic as View 1.
def plot_paths_grid(
    results: dict,
    analysis: dict,
    log_scale: bool = True,
    since: str | None = None,
    n_cols: int = 2,
):
    """Grid of subplots (one per model) each showing all path equity curves with the median overlaid."""
    models = results.get("models", [])
    if not models:
        print("No models found in results.")
        return
    n_models = len(models)
    n_rows = max(1, (n_models + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(7 * n_cols, 4 * n_rows),
                              squeeze=False)

    # One subplot per model; same plotting recipe as View 1 but compressed for grid use.
    for idx, model_name in enumerate(models):
        ax = axes[idx // n_cols][idx % n_cols]

        df = collect_path_equities(
            analysis["path_results"][model_name],
            results["n_paths"],
        )
        if df.empty:
            ax.set_title(f"{model_name} (no data)")
            continue

        if since is not None:
            try:
                df = df.loc[since:]
            except Exception:
                pass
            if df.empty:
                ax.set_title(f"{model_name} (no data since {since})")
                continue
            # Same defensive renormalisation as View 1.
            first_valid = df.bfill().iloc[0].replace(0, 1e-10)
            df = df / first_valid

        median = df.median(axis=1)

        for col in df.columns:
            ax.plot(df.index, df[col].values, linewidth=0.8, alpha=0.5)
        ax.plot(median.index, median.values,
                color="black", linewidth=1.8, zorder=10)

        if log_scale:
            try:
                ax.set_yscale("log")
            except Exception:
                pass
        ax.axhline(1.0, color="grey", linestyle="--", alpha=0.4)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.2)

    # Hide trailing unused subplots when n_models doesn't fill the grid.
    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    title_suffix = f" since {since}" if since else " (full history)"
    fig.suptitle(
        f"Path-by-path equity curves per model{title_suffix}",
        fontsize=14, fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


# --- 4. View 3: INTERACTIVE SELECTOR — ipywidgets dropdown ----------------
# Live notebook explorer with model + period + log-scale controls; calls View 1 on every change.
def interactive_path_explorer(results: dict, analysis: dict):
    """Build an ipywidgets panel (model dropdown + period dropdown + log toggle) that re-renders View 1 on change."""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        print("ipywidgets is not installed. Run: pip install ipywidgets")
        return

    out = widgets.Output()
    model_dd = widgets.Dropdown(
        options=results["models"],
        value=results["models"][0],
        description="Model:",
    )
    log_toggle = widgets.Checkbox(value=True, description="Log scale")
    since_dd = widgets.Dropdown(
        options=[("Full history", None), ("Since 2020", "2020"),
                 ("Since 2022", "2022"), ("Since 2023", "2023"),
                 ("Since 2024", "2024")],
        value=None,
        description="Period:",
    )

    # Re-render handler: clears the output area and replots View 1 with current widget values.
    def render(*_):
        with out:
            clear_output(wait=True)
            plot_paths_for_model(
                model_dd.value, results, analysis,
                log_scale=log_toggle.value,
                since=since_dd.value,
            )

    # Wire change events on every control to the same render handler.
    model_dd.observe(render, names="value")
    log_toggle.observe(render, names="value")
    since_dd.observe(render, names="value")

    controls = widgets.HBox([model_dd, since_dd, log_toggle])
    display(controls, out)
    render()


# --- Notebook usage examples -----------------------------------------------
# Reference snippet for the notebook; not executed at import time.
'''
from src.post_cpcv.path_explorer import (
    plot_paths_for_model,
    plot_paths_grid,
    interactive_path_explorer,
)

# Single model, full history
plot_paths_for_model("kan", results, analysis)

# Grid of all models
plot_paths_grid(results, analysis)

# Interactive selector
interactive_path_explorer(results, analysis)
'''