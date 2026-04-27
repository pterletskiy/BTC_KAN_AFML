"""
12.2) Path Explorer
===================
Per-model multi-path equity curves (Brownian-motion-style).

Three views of the same idea:

1. STATIC PER-MODEL — a single chart for one model showing all 5 paths.
   Best for thesis PDFs, exports cleanly to image.

2. STATIC GRID — one subplot per model, all paths shown. Best for
   side-by-side variance comparison across the model lineup.

3. INTERACTIVE SELECTOR — dropdown to flip between models in the
   notebook. Uses ipywidgets. Best for live exploration.

All three use the median across paths as a thick reference line and
plot individual paths as thin lines, so you can see both the central
tendency and the dispersion.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# Helper: collect all paths for a model into a single DataFrame
# =====================================================================
def collect_path_equities(model_path_data: dict, n_paths: int) -> pd.DataFrame:
    """Collect equity curves for all paths of a model into a DataFrame.

    Each column is a path. Columns are forward-filled across the timeline
    so all paths share a common index for plotting.
    """
    equities = []
    for path_id in range(n_paths):
        if path_id in model_path_data and len(model_path_data[path_id]["returns"]) > 0:
            equity = (1 + model_path_data[path_id]["returns"]).cumprod()
            equity.name = f"path_{path_id}"
            equities.append(equity)
    if not equities:
        return pd.DataFrame()
    
    # Concatenate and handle duplicate index just in case
    try:
        df = pd.concat(equities, axis=1)
    except Exception:
        # If there are duplicate indices in individual series, align them by grouping
        equities = [e.groupby(e.index).last() for e in equities]
        df = pd.concat(equities, axis=1)
        
    df = df.ffill()
    # Ensure index is DatetimeIndex for proper slicing
    df.index = pd.to_datetime(df.index)
    return df


# =====================================================================
# View 1: STATIC PER-MODEL — single model, all paths
# =====================================================================
def plot_paths_for_model(
    model_name: str,
    results: dict,
    analysis: dict,
    log_scale: bool = True,
    since: str | None = None,
    figsize: tuple = (14, 5.5),
):
    """Plot all 5 path equity curves for one model.

    Parameters
    ----------
    model_name : str
        e.g. "kan", "xgboost", "logistic".
    results, analysis : dict
        The output dicts of run_cpcv_pipeline and analyze_results.
    log_scale : bool
        If True, use log y-axis. Recommended for full-history plots
        where any model has compounded heavily.
    since : str, optional
        e.g. "2023" to slice from 2023 onward and re-normalize to 1.0.
    figsize : tuple
        Figure size.
    """
    df = collect_path_equities(
        analysis["path_results"][model_name],
        results["n_paths"],
    )
    if df.empty:
        print(f"No path data for {model_name}.")
        return

    # optional time slice + renormalization
    if since is not None:
        try:
            df = df.loc[since:]
        except Exception:
            pass
        if df.empty:
            print(f"No data for {model_name} since {since}.")
            return
        # Normalize by the first valid value of each path, preventing NaN propagation
        first_valid = df.bfill().iloc[0].replace(0, 1e-10)
        df = df / first_valid

    median = df.median(axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    # individual paths as thin lines
    for col in df.columns:
        ax.plot(df.index, df[col].values,
                linewidth=0.9, alpha=0.55,
                label=col.replace("_", " "))

    # median as a thick reference line
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


# =====================================================================
# View 2: STATIC GRID — all models, one subplot each
# =====================================================================
def plot_paths_grid(
    results: dict,
    analysis: dict,
    log_scale: bool = True,
    since: str | None = None,
    n_cols: int = 2,
):
    """Grid of subplots, one per model, each showing all path equity curves.

    Useful for side-by-side variance comparison across the model lineup.
    """
    models = results.get("models", [])
    if not models:
        print("No models found in results.")
        return
    n_models = len(models)
    n_rows = max(1, (n_models + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(7 * n_cols, 4 * n_rows),
                              squeeze=False)

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

    # hide unused subplots
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


# =====================================================================
# View 3: INTERACTIVE SELECTOR — dropdown in the notebook
# =====================================================================
def interactive_path_explorer(results: dict, analysis: dict):
    """Dropdown selector for model + log/linear toggle + since-year slider.

    Requires ipywidgets. Designed for use inside a Jupyter notebook.
    """
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

    def render(*_):
        with out:
            clear_output(wait=True)
            plot_paths_for_model(
                model_dd.value, results, analysis,
                log_scale=log_toggle.value,
                since=since_dd.value,
            )

    model_dd.observe(render, names="value")
    log_toggle.observe(render, names="value")
    since_dd.observe(render, names="value")

    controls = widgets.HBox([model_dd, since_dd, log_toggle])
    display(controls, out)
    render()


# =====================================================================
# Notebook usage examples
# =====================================================================

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


plot_paths_grid for the main results chapter — it shows the full lineup variance in one figure, which is the strongest visual argument that DSR and PBO are correctly identifying low-confidence results.
plot_paths_for_model for the KAN-specific deep dive section — focuses attention on KAN's path variance while showing the median trajectory, supporting the claim that KAN's symbolic extraction (the actual contribution) is meaningful regardless of moderate predictive performance.
interactive_path_explorer only for your live defense — if a committee member asks "what does XGBoost actually look like across paths?", you can flip to it instantly. Don't include it in the written thesis.

'''