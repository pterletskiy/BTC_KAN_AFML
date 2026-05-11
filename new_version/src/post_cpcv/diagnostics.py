"""
14.2) Diagnostics
=====================
Interactive inspection helpers for examining trained model predictions after
``run_cpcv_pipeline`` and ``analyze_results`` have produced their outputs.

These helpers operate on the raw predictions dict and on ``analysis["path_results"]``
and never touch the canonical event-aligned X / y / w / t1 series, so they're safe
to call freely from notebooks without polluting global state.

Sections:
  1. Calibration audit (text)
  2. Path-level dispersion and regime concentration
  3. Bet-size distribution
  4. Reliability curves
  5. Calibration mean audit
  6. Confusion-matrix renderer
  7. Feature-stability bar chart and table
  8. Sharpe-distribution boxplot
  9. Reliability-diagram grid
  10. Bet-size histogram grid
  11. Regime-concentration scatter
"""

import numpy as np
import pandas as pd


# --- 1. Calibration audit (text) -------------------------------------------
# Pool calibrated P(y=1) and ground-truth labels across every (split, seed) for one model.
def pool_predictions(model_name, results, n_seeds=None, n_splits=None):
    """Return ``(proba_pool, y_pool)`` concatenated across all splits and seeds.

    ``n_seeds`` and ``n_splits`` default to ``results["n_seeds"]`` and ``results["n_splits"]``
    so the diagnostic stays in sync with the pipeline configuration; pass explicit integers to
    override (e.g. for sensitivity checks on a seed subset).
    """
    if n_seeds is None:
        n_seeds = int(results.get("n_seeds", 1))
    if n_splits is None:
        n_splits = int(results.get("n_splits", 15))

    proba_pool, y_pool = [], []
    # Walk every (split, seed) and accumulate P(y=1) and y_true.
    for split_idx in range(n_splits):
        for seed in range(n_seeds):
            key = (model_name, split_idx, seed)
            if key in results["predictions"]:
                proba_pool.append(results["predictions"][key]["cal_proba"][:, 1])
                y_pool.append(results["predictions"][key]["y_true"])

    proba_pool = np.concatenate(proba_pool) if proba_pool else np.array([])
    y_pool = np.concatenate(y_pool) if y_pool else np.array([])
    return proba_pool, y_pool


# Print a text calibration table comparing predicted to empirical P(y=1) across fixed bins.
def calibration_audit(model_name, results, n_seeds=None, n_splits=None, n_bins=10):
    """Print binned predicted vs empirical P(y=1) for ``model_name``; flags directional bias or collapse.

    Reports only bins with at least 10 samples to avoid noisy single-sample bins.
    """
    proba_pool, y_pool = pool_predictions(model_name, results, n_seeds, n_splits)

    if len(proba_pool) == 0:
        print(f"{model_name}: no predictions found.")
        return

    print(f"{model_name}: mean P̂(y=1) = {proba_pool.mean():.4f}, "
          f"empirical = {y_pool.mean():.4f}")

    # Bin predictions into n_bins fixed-width slices on [0, 1] and report empirical mean per bin.
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(proba_pool, bin_edges) - 1

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 10:
            print(f"  P̂ ∈ [{bin_edges[b]:.1f}, {bin_edges[b+1]:.1f}): "
                  f"n={mask.sum()}, empirical = {y_pool[mask].mean():.3f}")


# --- 2. Path-level dispersion and regime concentration ---------------------
# Top-K concentration: how much of a path's cumulative return is driven by its K largest-magnitude returns.
def compute_top_k_concentration(strategy_returns: pd.Series, k: int = 5) -> dict:
    """Quantify regime-fluke risk by measuring the cumulative-return share contributed by top-K returns.

    A high share (e.g., > 50%) indicates the path's apparent profitability is concentrated in
    a handful of extreme returns from a specific market regime, rather than accumulated edge.
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

    # Pick the K largest-magnitude returns and recompute cumulative without them.
    top_k_idx = sr.abs().nlargest(k).index
    cum_ex = float(np.prod(1.0 + sr.drop(top_k_idx).values) - 1.0)

    # Share = relative drop in (1 + cum) when removing the top-K; bounded in [0, 1] for reasonable cases.
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


# Build a (model, path) dispersion table with regime-concentration diagnostics per path.
def build_path_dispersion_table(analysis: dict, k: int = 5) -> pd.DataFrame:
    """Return a ``DataFrame`` indexed by ``(model, path)`` with Sharpe, cum return, drawdown,
    n_trades, n_returns, top_k_share, and top_k_date_range columns."""
    rows = []
    # Walk every (model, path) and assemble the per-path diagnostic row.
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


# Collapse the (model, path) dispersion table to one row per model with min/median/max statistics.
def summarize_path_dispersion(dispersion: pd.DataFrame) -> pd.DataFrame:
    """Return per-model summary of path dispersion (Sharpe and cum-return min/median/max, avg top-K share)."""
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


# --- 3. Bet-size distribution ----------------------------------------------
# Per-model bet-size distribution summary pooled across all paths.
def compute_bet_size_summary(analysis: dict, min_bet_size: float = 0.05,
                             max_bet_size: float = 0.75) -> pd.DataFrame:
    """Return per-model bet-size diagnostics: abstention rate, mean/median |bet|, share at max, long/short balance.

    All metrics are pooled across the model's paths.
    """
    rows = []
    # Walk every model, pool bet sizes across paths, and derive the diagnostic row.
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


# Convenience helper that pools all bet sizes for a single model into a flat array.
def collect_bet_sizes(analysis: dict, model_name: str) -> np.ndarray:
    """Return a flat ``np.ndarray`` of all bet sizes for ``model_name`` pooled across paths."""
    all_bets = []
    for path_data in analysis["path_results"].get(model_name, {}).values():
        bs = path_data.get("bet_sizes", np.array([]))
        if len(bs):
            all_bets.append(bs)
    return np.concatenate(all_bets) if all_bets else np.array([])


# --- 4. Reliability curves --------------------------------------------------
# Compute the binned (predicted_mean, empirical_mean) pairs that feed reliability diagrams.
def compute_reliability_curve(
    model_name: str, results: dict,
    n_seeds: int | None = None, n_splits: int | None = None,
    n_bins: int = 10, min_count: int = 10,
) -> pd.DataFrame:
    """Return a binned ``DataFrame[bin_lo, bin_hi, bin_mid, predicted_mean, empirical_mean, n_samples]``.

    Bins with fewer than ``min_count`` samples are dropped.
    """
    proba_pool, y_pool = pool_predictions(model_name, results, n_seeds, n_splits)

    if len(proba_pool) == 0:
        return pd.DataFrame(columns=[
            "bin_lo", "bin_hi", "bin_mid",
            "predicted_mean", "empirical_mean", "n_samples",
        ])

    # Fixed-width binning on [0, 1]; np.digitize returns 1-indexed bins, hence the −1.
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


# --- 5. Calibration mean audit ---------------------------------------------
# First-moment calibration check: every model's mean P(Up) should hug the empirical base rate.
def calibration_mean_audit(
    results: dict,
    y: pd.Series | np.ndarray,
    tolerance: float = 0.03,
) -> pd.DataFrame:
    """Audit each model's mean calibrated P(Up) against the empirical base rate; prints + returns the table.

    Deviation beyond ``tolerance`` flags possible calibration failure or distributional collapse.
    Both ``{0, 1}`` and ``{-1, +1}`` label spaces are accepted; the base rate is computed as
    ``mean(y == max(y))``.
    """
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    pos_label = int(y_arr.max())
    base_rate = float((y_arr == pos_label).mean())

    n_seeds = int(results.get("n_seeds", 1))

    # Walk every model and aggregate P(Up) across all (split, seed) entries.
    rows = []
    for model_name in results["models"]:
        all_probas = []
        for split_idx in range(results["n_splits"]):
            for seed in range(n_seeds):
                key = (model_name, split_idx, seed)
                if key in results["predictions"]:
                    all_probas.append(
                        results["predictions"][key]["cal_proba"][:, 1]
                    )
        if not all_probas:
            continue
        avg = float(np.concatenate(all_probas).mean())
        deviation = avg - base_rate
        flagged = abs(deviation) >= tolerance
        rows.append({
            "model": model_name,
            "mean_p_up": avg,
            "base_rate": base_rate,
            "deviation": deviation,
            "flag": flagged,
        })

    df = pd.DataFrame(rows).set_index("model")

    print("Calibration first-moment audit")
    print("=" * 60)
    print(f"  Empirical base rate (P(Up) in data): {base_rate:.4f}")
    print(f"  Models below should be close to this baseline.")
    print(f"  Tolerance: ±{tolerance:.2f} (flagged if exceeded)\n")
    for model_name, row in df.iterrows():
        glyph = "WARN" if row["flag"] else "  ok"
        print(
            f"  [{glyph}] {model_name:>20s}: mean P(Up) = "
            f"{row['mean_p_up']:.4f}  (Δ={row['deviation']:+.4f})"
        )
    print("=" * 60)

    return df


# --- 6. Confusion-matrix renderer ------------------------------------------
# Grid of per-model confusion matrices on seed-averaged probabilities (matches Sharpe/DSR/AUC convention).
def render_confusion_matrices(
    results: dict,
    seed_mode: str = "average",
    n_cols: int = 3,
    figsize_per_subplot: tuple = (5.0, 4.0),
    save_path: str | None = None,
):
    """Render a grid of confusion matrices, one per model; ``seed_mode='average'`` matches the financial-metrics convention.

    Label space is auto-detected from the data: both ``{0, 1}`` and ``{-1, +1}`` work without further configuration.
    Use ``seed_mode='seed_0'``/``'seed_1'``/``'seed_2'`` to inspect a single seed instead.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    model_names = results["models"]
    n_seeds = int(results.get("n_seeds", 1))

    # Detect label space from the first available prediction; the whole pipeline uses one convention.
    sample_key = next(iter(results["predictions"]))
    unique_labels = sorted(
        np.unique(results["predictions"][sample_key]["y_true"]).tolist()
    )
    if unique_labels == [0, 1]:
        pos_label, neg_label = 1, 0
    elif unique_labels == [-1, 1]:
        pos_label, neg_label = 1, -1
    else:
        raise ValueError(f"Unexpected label space: {unique_labels}")
    cm_label_order = [neg_label, pos_label]

    n_rows = int(np.ceil(len(model_names) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot[0] * n_cols,
                 figsize_per_subplot[1] * n_rows),
    )
    axes_flat = np.atleast_1d(axes).flatten()

    last_filled = -1
    for i, model_name in enumerate(model_names):
        all_true, all_pred = [], []

        # Per-split: either seed-average probabilities then threshold at 0.5, or read the per-seed y_pred.
        for split_idx in range(results["n_splits"]):
            if seed_mode == "average":
                seed_probas, seed_y_true = [], None
                for seed in range(n_seeds):
                    key = (model_name, split_idx, seed)
                    if key not in results["predictions"]:
                        continue
                    pred = results["predictions"][key]
                    seed_probas.append(pred["cal_proba"][:, 1])
                    if seed_y_true is None:
                        seed_y_true = pred["y_true"]
                if not seed_probas:
                    continue
                # Truncate to shortest before averaging (LSTM windowing edge case).
                min_len = min(len(p) for p in seed_probas)
                avg_proba = np.stack(
                    [p[:min_len] for p in seed_probas]
                ).mean(axis=0)
                y_true_split = seed_y_true[:min_len]
                y_pred_split = np.where(
                    avg_proba >= 0.5, pos_label, neg_label
                )
            else:
                # Single-seed diagnostic mode: parse the seed index from the seed_mode string.
                seed = int(seed_mode.split("_")[1])
                key = (model_name, split_idx, seed)
                if key not in results["predictions"]:
                    continue
                pred = results["predictions"][key]
                y_true_split = pred["y_true"]
                y_pred_split = pred["y_pred"]

            all_true.append(y_true_split)
            all_pred.append(y_pred_split)

        if not all_true:
            continue

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        cm = confusion_matrix(y_true, y_pred, labels=cm_label_order)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"])
        disp.plot(ax=axes_flat[i], cmap="Blues", colorbar=False)

        acc = float(np.mean(y_true == y_pred))
        axes_flat[i].set_title(
            f"{model_name}\n(acc={acc:.3f}, n={len(y_true)})", fontsize=10,
        )
        last_filled = i

    # Hide trailing unused subplots so the grid renders cleanly when n_models doesn't fill the grid.
    for j in range(last_filled + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if seed_mode == "average":
        title_suffix = (
            f"probabilities averaged across {n_seeds} "
            f"seed{'s' if n_seeds != 1 else ''}"
        )
    else:
        title_suffix = f"using {seed_mode} only"

    fig.suptitle(
        f"Confusion Matrices ({title_suffix}, "
        "aggregated across all splits)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# --- 7. Feature-stability bar chart and table ------------------------------
# Explicit TA and Math feature sets so _categorize_feature can place names without a unifying prefix.
_TA_FEATURES = frozenset({
    "log_returns", "rsi", "macd", "macd_signal", "macd_hist", "bb_width",
    "atr", "obv", "skewness", "kurtosis", "realized_vol", "gk_vol", "yz_vol",
    "ema_ratio_20_50", "ema_ratio_50_200", "vwma_ratio_20_50", "roc_14",
    "stoch_k", "stoch_d", "williams_r", "cci_14", "chaikin_osc", "mfi_14",
    "vol_term_7_30", "vol_term_30_90",
})

_MATH_FEATURES = frozenset({
    "shannon_entropy", "negentropy", "lz_complexity", "variance_ratio",
    "jarque_bera", "hurst", "sadf", "smt_poly1", "smt_exp",
})


# Classify a feature name into TA / Math / Lag / External (the catch-all for macro/onchain).
def _categorize_feature(name: str) -> str:
    """Return one of ``TA``, ``Math``, ``Lag``, ``External`` based on the feature name."""
    name = str(name)
    if name in _TA_FEATURES:
        return "TA"
    if name in _MATH_FEATURES:
        return "Math"
    if name.startswith("log_returns_lag"):
        return "Lag"
    return "External"


# Render the per-feature selection-frequency bar chart with green/blue/grey thresholds.
def render_feature_stability(
    feat_stab: dict,
    figsize: tuple | None = None,
    threshold_stable: float = 0.80,
    threshold_moderate: float = 0.50,
    save_path: str | None = None,
):
    """Bar chart of per-feature selection frequency across CPCV folds.

    Bars are coloured by stability tier: stable (>80%, green), moderate (>50%, blue), low (grey).
    """
    import matplotlib.pyplot as plt

    freq = feat_stab["feature_frequency"].sort_values(ascending=True)

    # Default figsize scales with the number of features so labels don't overlap.
    if figsize is None:
        figsize = (8, max(6, len(freq) * 0.3))

    fig, ax = plt.subplots(figsize=figsize)
    # Per-bar colour from the stability tier.
    colors = [
        "#0b6623" if f > threshold_stable
        else "#4a90d9" if f > threshold_moderate
        else "#999999"
        for f in freq.values
    ]
    ax.barh(freq.index, freq.values, color=colors)
    ax.axvline(
        threshold_stable, color="red", linestyle="--", alpha=0.7,
        label=f"{int(threshold_stable * 100)}% threshold",
    )
    ax.axvline(
        threshold_moderate, color="orange", linestyle="--", alpha=0.5,
        label=f"{int(threshold_moderate * 100)}% threshold",
    )
    ax.set_xlabel("Selection frequency across folds")
    ax.set_title("Feature selection stability")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# Print a categorised table of features above a threshold + four-tier summary block.
def print_feature_stability_table(
    feat_stab: dict,
    all_features: list,
    threshold_pct: float = 50.0,
) -> pd.DataFrame:
    """Print the per-feature stability table + the four-tier summary; return the full ranking DataFrame.

    Only features above ``threshold_pct`` are printed in the main table. The summary tracks
    stable (>80%), moderate (50-80%), low (0-50%), and never-selected (0%) features; the
    never-selected list is grouped by category so structural patterns are visible at a glance.
    """
    freq = feat_stab["feature_frequency"]
    # Reindex onto the full MDA pool so never-selected features (absent from freq) get 0.
    freq_full = freq.reindex(all_features, fill_value=0.0)

    ranking = pd.DataFrame({
        "feature": freq_full.sort_values(ascending=False).index,
        "selection_pct": (freq_full.sort_values(ascending=False).values * 100).round(1),
    })
    ranking["category"] = ranking["feature"].apply(_categorize_feature)
    ranking["status"] = ranking["selection_pct"].apply(
        lambda p: "stable (>80%)" if p > 80
        else "moderate (>50%)" if p > 50
        else "low"
    )

    printed = ranking[ranking["selection_pct"] > threshold_pct].copy()

    print("\n" + "=" * 70)
    print(f"{'Feature':<28} {'Category':<10} {'Selection %':>12} {'Status':<18}")
    print("=" * 70)

    if printed.empty:
        print(f"  (no features selected in more than {threshold_pct:.0f}% of folds)")
    else:
        # Insert a separator row whenever the stability tier changes.
        prev_status = None
        for _, row in printed.iterrows():
            if prev_status is not None and row["status"] != prev_status:
                print("-" * 70)
            print(
                f"{row['feature']:<28} {row['category']:<10} "
                f"{row['selection_pct']:>10.1f}%  {row['status']:<18}"
            )
            prev_status = row["status"]
    print("=" * 70)

    # Tier counts.
    n_total = len(ranking)
    n_stable = int((ranking["selection_pct"] > 80).sum())
    n_moderate = int(
        ((ranking["selection_pct"] > 50) & (ranking["selection_pct"] <= 80)).sum()
    )
    n_low = int(
        ((ranking["selection_pct"] > 0) & (ranking["selection_pct"] <= 50)).sum()
    )
    n_never = int((ranking["selection_pct"] == 0).sum())

    print(f"\nTotal: {n_total} features in the MDA pool.")
    print(f"  Stable (>80%):       {n_stable} feature{'s' if n_stable != 1 else ''}")
    print(f"  Moderate (50-80%):   {n_moderate} feature{'s' if n_moderate != 1 else ''}")
    print(f"  Low (0-50%):         {n_low} feature{'s' if n_low != 1 else ''}")
    print(f"  Never selected (0%): {n_never} feature{'s' if n_never != 1 else ''}")

    # Never-selected breakdown grouped by category — surfaces "whole category unused" patterns.
    if n_never > 0:
        never_features = ranking.loc[
            ranking["selection_pct"] == 0, ["feature", "category"]
        ]
        print(f"\nNever-selected features ({n_never}):")
        for category in ["TA", "Math", "External", "Lag"]:
            cat_feats = never_features.loc[
                never_features["category"] == category, "feature"
            ].tolist()
            if cat_feats:
                print(f"  {category}: {', '.join(cat_feats)}")

    return ranking


# --- 8. Sharpe-distribution boxplot ----------------------------------------
# Boxplot of per-model Sharpe across CPCV paths; shows median + spread side by side.
def render_sharpe_distribution(
    analysis: dict,
    models: list | None = None,
    figsize: tuple = (10.0, 5.0),
    save_path: str | None = None,
):
    """Boxplot of per-model Sharpe across CPCV paths; tight spread + high median is the goal.

    Reads ``analysis['path_sharpes']`` (shape ``(n_models, n_paths)``); row order must match ``models``.
    """
    import matplotlib.pyplot as plt

    if models is None:
        models = list(analysis["path_results"].keys())

    fig, ax = plt.subplots(figsize=figsize)
    sharpe_data = [analysis["path_sharpes"][i, :] for i in range(len(models))]

    ax.boxplot(sharpe_data, labels=models, patch_artist=True)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="Sharpe = 0")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title(
        "Sharpe Ratio Distribution Across 5 CPCV Paths",
        fontsize=13, fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# --- 9. Reliability-diagram grid -------------------------------------------
# Grid of per-model reliability diagrams: predicted vs empirical with dot size by sample count.
def render_reliability_diagrams(
    results: dict,
    analysis: dict,
    n_cols: int = 3,
    figsize_per_subplot: tuple = (4.0, 3.8),
    n_bins: int = 10,
    min_count: int = 10,
    save_path: str | None = None,
):
    """Grid of reliability diagrams, one per model; points on the diagonal indicate well-calibrated predictions.

    Dot size is proportional to bin sample count, so small dots correspond to bins with little
    statistical support and should be discounted visually.
    """
    import matplotlib.pyplot as plt

    models = list(analysis["path_results"].keys())
    n_models = len(models)
    n_rows = (n_models + n_cols - 1) // n_cols
    n_seeds = int(results.get("n_seeds", 1))
    n_splits = int(results.get("n_splits", 15))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot[0] * n_cols,
                 figsize_per_subplot[1] * n_rows),
    )
    axes_flat = np.atleast_1d(axes).flatten()

    # One subplot per model.
    for ax, model in zip(axes_flat, models):
        rel = compute_reliability_curve(
            model, results,
            n_seeds=n_seeds, n_splits=n_splits,
            n_bins=n_bins, min_count=min_count,
        )
        if len(rel) == 0:
            ax.set_visible(False)
            continue

        # Perfect-calibration diagonal as a reference line.
        ax.plot(
            [0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5,
            label="Perfect calibration",
        )
        # Dot size scaled by sqrt of sample count so visually small dots = low support.
        sizes = 50 * np.sqrt(rel["n_samples"] / rel["n_samples"].max())
        ax.plot(
            rel["predicted_mean"], rel["empirical_mean"],
            linewidth=1.5,
        )
        ax.scatter(
            rel["predicted_mean"], rel["empirical_mean"],
            s=sizes, alpha=0.7, edgecolors="black", linewidth=0.5,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{model}", fontsize=11)
        ax.set_xlabel("Mean predicted P(y=1)")
        ax.set_ylabel("Empirical P(y=1)")
        ax.grid(alpha=0.3)
        if model == models[0]:
            ax.legend(loc="upper left", fontsize=9)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    fig.suptitle(
        "Reliability diagrams (dot size proportional to bin sample count)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# --- 10. Bet-size histogram grid -------------------------------------------
# Grid of per-model bet-size histograms; bin edges target the four-step discretisation.
def render_bet_size_histograms(
    analysis: dict,
    bins: np.ndarray | None = None,
    n_cols: int = 3,
    figsize_per_subplot: tuple = (4.3, 3.5),
    save_path: str | None = None,
):
    """Grid of bet-size histograms, one per model; default bins target the canonical four-step discretisation.

    Bet sizes are pooled across all paths for each model.
    """
    import matplotlib.pyplot as plt

    # Default edges centred on the four-step grid ±0.25, ±0.50, ±0.75 with abstention straddling zero.
    if bins is None:
        bins = np.array(
            [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875]
        )

    models = list(analysis["path_results"].keys())
    n_models = len(models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot[0] * n_cols,
                 figsize_per_subplot[1] * n_rows),
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    # One histogram per model.
    for ax, model in zip(axes_flat, models):
        bets = collect_bet_sizes(analysis, model)
        ax.hist(bets, bins=bins, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{model} (n={len(bets):,})", fontsize=11)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Bet size")
        ax.grid(alpha=0.3, axis="y")

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel("Event count")
    fig.suptitle(
        "Bet-size distribution per model (pooled across paths)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# --- 11. Regime-concentration scatter --------------------------------------
# Scatter of (top-K share, path Sharpe) for every (model, path); identifies regime-fluke candidates.
def render_regime_concentration_scatter(
    dispersion: pd.DataFrame,
    k: int = 5,
    figsize: tuple = (9.0, 5.0),
    concentration_threshold: float = 0.5,
    save_path: str | None = None,
):
    """Scatter top-K concentration share vs path-level Sharpe; one point per (model, path).

    Paths with high concentration share are regime-fluke candidates: their apparent profitability
    rests on a handful of extreme outcomes rather than accumulated edge.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # One colour per model; tab10 palette gives up to 10 distinct hues.
    models = dispersion.index.get_level_values("model").unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    color_map = dict(zip(models, colors))

    for model in models:
        sub = dispersion.xs(model, level="model")
        ax.scatter(
            sub["top_k_share"], sub["sharpe"],
            s=80, alpha=0.7, label=model, c=[color_map[model]],
            edgecolors="black", linewidth=0.5,
        )

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(
        concentration_threshold,
        color="red", linestyle=":", linewidth=0.8,
        label=f"{int(concentration_threshold * 100)}% concentration",
    )
    ax.set_xlabel(f"Top-{k} share of cumulative return")
    ax.set_ylabel("Annualised Sharpe")
    ax.set_title(
        "Regime concentration vs path-level Sharpe (each point = one path)"
    )
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig