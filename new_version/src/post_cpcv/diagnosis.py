"""
12.3) Diagnostics
=====================
Interactive inspection helpers for examining trained model predictions
after ``run_cpcv_pipeline`` and ``analyze_results`` have produced their
outputs. These helpers operate on the raw predictions dict and on
``analysis["path_results"]`` and never touch the canonical event-aligned
X / y / w / t1 series. Use freely from notebooks without polluting
global state.

Sections:
  1. Calibration audit
  2. Path-level dispersion and regime concentration
  3. Bet-size distribution
  4. Reliability curves
"""

import numpy as np
import pandas as pd


# =====================================================================
# 1) Calibration audit
# =====================================================================
def pool_predictions(model_name, results, n_seeds=2, n_splits=15):
    """Pool calibrated P(y=1) and true labels across splits and seeds.

    Parameters
    ----------
    model_name : str
        One of the model keys used in ``results["predictions"]``.
    results : dict
        Output of ``run_cpcv_pipeline``.
    n_seeds : int, default 2
        Number of seeds the model was trained with.
    n_splits : int, default 15
        Number of CPCV splits.

    Returns
    -------
    proba_pool : np.ndarray
        Concatenated calibrated P(class=1).
    y_pool : np.ndarray
        Concatenated ground-truth labels (0 or 1).
    """
    proba_pool, y_pool = [], []
    for split_idx in range(n_splits):
        for seed in range(n_seeds):
            key = (model_name, split_idx, seed)
            if key in results["predictions"]:
                proba_pool.append(results["predictions"][key]["cal_proba"][:, 1])
                y_pool.append(results["predictions"][key]["y_true"])

    proba_pool = np.concatenate(proba_pool) if proba_pool else np.array([])
    y_pool = np.concatenate(y_pool) if y_pool else np.array([])
    return proba_pool, y_pool


def calibration_audit(model_name, results, n_seeds=2, n_splits=15, n_bins=10):
    """Print a calibration table comparing predicted P(y=1) to empirical
    P(y=1) within fixed-width probability bins.

    Diagnoses systematic directional bias, miscalibrated sharpness, or
    distributional collapse. Reports only bins with at least 10 samples.
    """
    proba_pool, y_pool = pool_predictions(model_name, results, n_seeds, n_splits)

    if len(proba_pool) == 0:
        print(f"{model_name}: no predictions found.")
        return

    print(f"{model_name}: mean P̂(y=1) = {proba_pool.mean():.4f}, "
          f"empirical = {y_pool.mean():.4f}")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(proba_pool, bin_edges) - 1

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 10:
            print(f"  P̂ ∈ [{bin_edges[b]:.1f}, {bin_edges[b+1]:.1f}): "
                  f"n={mask.sum()}, empirical = {y_pool[mask].mean():.3f}")


# =====================================================================
# 2) Path-level dispersion and regime concentration
# =====================================================================
def compute_top_k_concentration(strategy_returns: pd.Series, k: int = 5) -> dict:
    """Quantify how much of a path's cumulative return is driven by the
    top-K returns by absolute magnitude.

    A high concentration share (e.g., > 50%) indicates regime-fluke:
    most of the path's apparent profitability comes from a handful of
    extreme returns clustered in a specific market regime.

    Parameters
    ----------
    strategy_returns : pd.Series
        Per-event strategy returns indexed by timestamp. Typically
        ``analysis["path_results"][model][path_id]["returns"]``.
    k : int, default 5
        Number of largest-magnitude returns to inspect.

    Returns
    -------
    dict with keys:
        cum_full : float          full cumulative return
        cum_ex_top_k : float      cumulative return excluding top-K
        top_k_share : float       fraction of full growth from top-K
        top_k_dates : list        timestamps of the top-K returns
        top_k_values : list       the top-K return values
        date_range : str          shorthand "min-max" of top-K dates
    """
    if len(strategy_returns) == 0:
        return {
            "cum_full": 0.0, "cum_ex_top_k": 0.0, "top_k_share": np.nan,
            "top_k_dates": [], "top_k_values": [], "date_range": "n/a",
        }

    sr = strategy_returns.copy()
    cum_full = float(np.prod(1.0 + sr.values) - 1.0)

    if cum_full == 0 or len(sr) <= k:
        return {
            "cum_full": cum_full, "cum_ex_top_k": cum_full, "top_k_share": np.nan,
            "top_k_dates": list(sr.index), "top_k_values": list(sr.values),
            "date_range": "n/a",
        }

    top_k_idx = sr.abs().nlargest(k).index
    cum_ex = float(np.prod(1.0 + sr.drop(top_k_idx).values) - 1.0)

    # Share of growth attributable to the top-K returns.
    # Defined as the relative drop in (1 + cum) when removing them.
    share = (cum_full - cum_ex) / (1.0 + cum_full) if cum_full != 0 else np.nan

    top_k_dates = sorted(top_k_idx.tolist())
    date_range = (
        f"{pd.Timestamp(top_k_dates[0]).date()} → "
        f"{pd.Timestamp(top_k_dates[-1]).date()}"
    )

    return {
        "cum_full": cum_full,
        "cum_ex_top_k": cum_ex,
        "top_k_share": float(share),
        "top_k_dates": top_k_dates,
        "top_k_values": [float(sr.loc[d]) for d in top_k_idx],
        "date_range": date_range,
    }


def build_path_dispersion_table(analysis: dict, k: int = 5) -> pd.DataFrame:
    """Build a (model, path) dispersion table with regime-concentration
    diagnostics for every path of every model.

    Useful for assessing how much of each model's apparent performance
    comes from a small number of regime-driven trades versus accumulated
    edge across many trades.

    Parameters
    ----------
    analysis : dict
        Output of ``analyze_results``.
    k : int, default 5
        Number of top-magnitude returns to inspect per path.

    Returns
    -------
    pd.DataFrame indexed by (model, path) with columns:
        sharpe, cum_return, max_dd, n_trades, n_returns,
        top_k_share, top_k_date_range
    """
    rows = []
    for model_name, paths in analysis["path_results"].items():
        for path_id, path_data in paths.items():
            perf = path_data["performance"]
            sr = path_data["returns"]
            conc = compute_top_k_concentration(sr, k=k)

            rows.append({
                "model": model_name,
                "path": path_id,
                "sharpe": perf["annualized_sharpe"],
                "cum_return": perf["cumulative_return"],
                "max_dd": perf["max_drawdown"],
                "n_trades": perf["n_trades"],
                "n_returns": perf.get("n_returns", len(sr)),
                "top_k_share": conc["top_k_share"],
                "top_k_date_range": conc["date_range"],
            })

    df = pd.DataFrame(rows).set_index(["model", "path"]).sort_index()
    return df


def summarize_path_dispersion(dispersion: pd.DataFrame) -> pd.DataFrame:
    """Collapse the (model, path) dispersion table to one row per model
    with min/median/max statistics across the model's paths.

    Returns a DataFrame indexed by model with columns:
        sharpe_min, sharpe_median, sharpe_max,
        cum_min, cum_median, cum_max,
        avg_top_k_share
    """
    summary = dispersion.groupby(level="model").agg(
        sharpe_min=("sharpe", "min"),
        sharpe_median=("sharpe", "median"),
        sharpe_max=("sharpe", "max"),
        cum_min=("cum_return", "min"),
        cum_median=("cum_return", "median"),
        cum_max=("cum_return", "max"),
        avg_top_k_share=("top_k_share", "mean"),
    )
    return summary.sort_values("sharpe_median", ascending=False)


# =====================================================================
# 3) Bet-size distribution
# =====================================================================
def compute_bet_size_summary(analysis: dict, min_bet_size: float = 0.05,
                             max_bet_size: float = 0.75) -> pd.DataFrame:
    """Summarise the bet-size distribution for each model, pooled
    across all paths.

    Parameters
    ----------
    analysis : dict
        Output of ``analyze_results``.
    min_bet_size : float
        Threshold below which bets are abstentions (snapped to 0).
    max_bet_size : float
        Cap on raw bets, before discretisation.

    Returns
    -------
    pd.DataFrame indexed by model with columns:
        n_events           total events across all paths
        abstention_rate    share of events with bet_size == 0
        mean_abs_bet       mean |bet| among traded events
        median_abs_bet     median |bet| among traded events
        share_at_max       share of traded events at bet=±max_bet_size
        long_share         share of traded events with bet > 0
        short_share        share of traded events with bet < 0
    """
    rows = []
    for model_name, paths in analysis["path_results"].items():
        all_bets = []
        for path_data in paths.values():
            bs = path_data.get("bet_sizes", np.array([]))
            if len(bs):
                all_bets.append(bs)
        if not all_bets:
            continue

        bets = np.concatenate(all_bets)
        traded = bets[bets != 0]

        rows.append({
            "model": model_name,
            "n_events": int(len(bets)),
            "abstention_rate": float(np.mean(bets == 0)),
            "mean_abs_bet": float(np.mean(np.abs(traded))) if len(traded) else 0.0,
            "median_abs_bet": float(np.median(np.abs(traded))) if len(traded) else 0.0,
            "share_at_max": (
                float(np.mean(np.isclose(np.abs(traded), max_bet_size)))
                if len(traded) else 0.0
            ),
            "long_share": float(np.mean(traded > 0)) if len(traded) else 0.0,
            "short_share": float(np.mean(traded < 0)) if len(traded) else 0.0,
        })

    return pd.DataFrame(rows).set_index("model")


def collect_bet_sizes(analysis: dict, model_name: str) -> np.ndarray:
    """Return all bet sizes for a model, pooled across paths.

    Convenience helper for plotting histograms of bet-size distributions.
    """
    all_bets = []
    for path_data in analysis["path_results"].get(model_name, {}).values():
        bs = path_data.get("bet_sizes", np.array([]))
        if len(bs):
            all_bets.append(bs)
    return np.concatenate(all_bets) if all_bets else np.array([])


# =====================================================================
# 4) Reliability curves
# =====================================================================
def compute_reliability_curve(
    model_name: str, results: dict,
    n_seeds: int = 2, n_splits: int = 15, n_bins: int = 10,
    min_count: int = 10,
) -> pd.DataFrame:
    """Compute the binned (predicted, empirical) pairs for a reliability
    diagram.

    Parameters
    ----------
    n_bins : int, default 10
        Number of equal-width bins on [0, 1].
    min_count : int, default 10
        Bins with fewer samples are dropped.

    Returns
    -------
    pd.DataFrame with columns:
        bin_lo, bin_hi, bin_mid, predicted_mean, empirical_mean, n_samples
    """
    proba_pool, y_pool = pool_predictions(model_name, results, n_seeds, n_splits)

    if len(proba_pool) == 0:
        return pd.DataFrame(columns=[
            "bin_lo", "bin_hi", "bin_mid",
            "predicted_mean", "empirical_mean", "n_samples",
        ])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(proba_pool, bin_edges) - 1

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n < min_count:
            continue
        rows.append({
            "bin_lo": float(bin_edges[b]),
            "bin_hi": float(bin_edges[b + 1]),
            "bin_mid": float((bin_edges[b] + bin_edges[b + 1]) / 2),
            "predicted_mean": float(proba_pool[mask].mean()),
            "empirical_mean": float(y_pool[mask].mean()),
            "n_samples": n,
        })
    return pd.DataFrame(rows)