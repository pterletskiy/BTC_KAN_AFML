"""
12) Evaluation
=======================
Take raw predictions from the CPCV pipeline and produce:
  - Per-split classification metrics
  - Bet sizing via De Prado's S-curve (AFML Chapter 10)
  - Full-span CPCV backtest paths (AFML Chapter 12.4)
  - Path-level financial performance metrics
  - Deflated Sharpe Ratio (AFML Chapter 14)
  - Probability of Backtest Overfitting (AFML Chapter 11)
  - Final model comparison table
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
TRANSACTION_COST = 0.001          # 0.1% round-trip cost for BTC
MIN_BET_SIZE = 0.05               # absolute bet sizes below this → don't trade
MAX_BET_SIZE = 0.75               # cap to prevent explosive equity curves
BET_DISCRETIZATION = [0.0, 0.25, 0.50, 0.75]
ANNUALIZATION_FACTOR = 365        # BTC trades every calendar day
RISK_FREE_RATE = 0.0              # assume 0 for crypto


# =====================================================================
# Split-level classification metrics
# =====================================================================
def compute_split_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """Compute classification metrics for one split's test fold."""
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    f1_mac = f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)
    f1_per = f1_score(y_true, y_pred, average=None, sample_weight=sample_weight)
    prec = precision_score(
        y_true, y_pred, average="macro", sample_weight=sample_weight, zero_division=0
    )
    rec = recall_score(
        y_true, y_pred, average="macro", sample_weight=sample_weight, zero_division=0
    )

    try:
        ll = log_loss(y_true, y_proba, sample_weight=sample_weight)
    except ValueError:
        ll = np.nan

    brier = np.average((y_proba[:, 1] - y_true) ** 2, weights=sample_weight)

    try:
        auc = roc_auc_score(y_true, y_proba[:, 1], sample_weight=sample_weight)
    except ValueError:
        auc = np.nan

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_class_0": f1_per[0] if len(f1_per) > 0 else np.nan,
        "f1_class_1": f1_per[1] if len(f1_per) > 1 else np.nan,
        "precision_macro": prec,
        "recall_macro": rec,
        "log_loss": ll,
        "brier_score": brier,
        "auc_roc": auc,
    }


# =====================================================================
# Bet sizing (AFML Chapter 10.3)
# =====================================================================
def bet_size_from_proba(
    proba: np.ndarray,
    min_bet: float = MIN_BET_SIZE,
    discretize: bool = True,
) -> np.ndarray:
    """Convert calibrated probabilities into signed position sizes via S-curve."""
    # predicted side
    direction = np.where(proba[:, 1] > proba[:, 0], 1, -1)

    # confidence = probability assigned to predicted class
    p = np.maximum(proba[:, 0], proba[:, 1])

    # z-score transform
    z = (p - 0.5) / np.sqrt(p * (1 - p) + 1e-10)

    # S-curve mapping
    raw_bet = 2.0 * stats.norm.cdf(z) - 1.0

    # cap maximum position size
    raw_bet = np.clip(raw_bet, -MAX_BET_SIZE, MAX_BET_SIZE)

    # minimum threshold
    raw_bet = np.where(np.abs(raw_bet) < min_bet, 0.0, raw_bet)

    # apply direction
    signed_bet = direction * np.abs(raw_bet)

    # discretize
    if discretize:
        disc = np.array(BET_DISCRETIZATION)
        abs_bet = np.abs(signed_bet)
        for i in range(len(abs_bet)):
            closest_idx = np.argmin(np.abs(disc - abs_bet[i]))
            abs_bet[i] = disc[closest_idx]
        signed_bet = np.sign(signed_bet) * abs_bet

    return signed_bet


# =====================================================================
# Strategy returns
# =====================================================================
def compute_strategy_returns(
    bet_sizes: np.ndarray,
    label_returns: np.ndarray,
    timestamps: pd.DatetimeIndex,
    cost: float = TRANSACTION_COST,
) -> pd.Series:
    """Compute daily net strategy returns from bet sizes and realized returns."""
    gross = bet_sizes * label_returns

    # turnover
    turnover = np.abs(np.diff(bet_sizes, prepend=0))
    turnover[0] = np.abs(bet_sizes[0])

    tx_cost = cost * turnover
    net = gross - tx_cost

    return pd.Series(net, index=timestamps, name="strategy_return")


# =====================================================================
# Path-level financial performance
# =====================================================================
def compute_path_performance(
    strategy_returns: pd.Series, bet_sizes: np.ndarray
) -> dict:
    """Compute financial performance metrics from a backtest path."""
    returns = strategy_returns.values
    n = len(returns)

    if n == 0:
        return _empty_performance()

    # Sharpe ratio
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1) if n > 1 else 1e-10
    if std_r < 1e-10:
        sharpe = 0.0
    else:
        sharpe = (mean_r - RISK_FREE_RATE / ANNUALIZATION_FACTOR) / std_r * np.sqrt(ANNUALIZATION_FACTOR)

    # cumulative return
    cum_ret = np.prod(1.0 + returns) - 1.0

    # annualized return
    ann_ret = (1.0 + cum_ret) ** (ANNUALIZATION_FACTOR / n) - 1.0 if n > 0 else 0.0

    # maximum drawdown
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = np.min(drawdowns)

    # time under water
    underwater = equity < running_max
    if underwater.any():
        uw_groups = np.diff(np.where(np.concatenate(([False], underwater, [False])))[0])
        time_uw = int(np.max(uw_groups)) if len(uw_groups) > 0 else 0
    else:
        time_uw = 0

    # win rate (among traded observations)
    traded_mask = bet_sizes != 0
    traded_returns = returns[traded_mask]
    n_trades = int(traded_mask.sum())
    win_rate = np.mean(traded_returns > 0) if n_trades > 0 else 0.0

    # profit factor
    pos_returns = traded_returns[traded_returns > 0].sum()
    neg_returns = traded_returns[traded_returns < 0].sum()
    profit_factor = pos_returns / abs(neg_returns) if neg_returns != 0 else np.inf

    # average bet size
    avg_bet = np.mean(np.abs(bet_sizes[traded_mask])) if n_trades > 0 else 0.0

    # distributional
    skewness = float(stats.skew(returns)) if n > 2 else 0.0
    kurtosis = float(stats.kurtosis(returns)) if n > 3 else 0.0

    return {
        "annualized_sharpe": sharpe,
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "max_drawdown": max_dd,
        "time_under_water": time_uw,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_bet_size": avg_bet,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def _empty_performance() -> dict:
    return {
        "annualized_sharpe": 0.0, "cumulative_return": 0.0,
        "annualized_return": 0.0, "max_drawdown": 0.0,
        "time_under_water": 0, "win_rate": 0.0,
        "profit_factor": 0.0, "n_trades": 0,
        "avg_bet_size": 0.0, "skewness": 0.0, "kurtosis": 0.0,
    }


# =====================================================================
# Path stitching (AFML Chapter 12.4)
# =====================================================================
def stitch_paths(
    predictions: dict,
    path_map: dict,
    n_paths: int,
    model_name: str,
    seed: int = 0,
    n_seeds: int = 1,
) -> dict:
    """Assemble full backtest paths by stitching test-fold predictions.

    If n_seeds > 1, calibrated probabilities are averaged across all
    available seeds before computing bet sizes. This ensemble averaging
    reduces prediction variance by ~1/sqrt(n_seeds).
    """
    path_results = {}

    for path_id in range(n_paths):
        assignments = path_map[path_id]  # [(group_id, split_id), ...]

        all_timestamps = []
        all_proba = []
        all_ret = []

        for group_id, split_id in sorted(assignments, key=lambda x: x[0]):
            if n_seeds > 1:
                # seed-averaging: collect probabilities from all seeds
                seed_probas = []
                ref_key = None
                for s in range(n_seeds):
                    key = (model_name, split_id, s)
                    if key in predictions:
                        seed_probas.append(predictions[key]["cal_proba"])
                        if ref_key is None:
                            ref_key = key

                if not seed_probas or ref_key is None:
                    logger.warning(
                        "No seed predictions for %s, split=%d. Skipping.",
                        model_name, split_id,
                    )
                    continue

                # average calibrated probabilities across seeds
                proba = np.mean(seed_probas, axis=0)
                pred = predictions[ref_key]
            else:
                key = (model_name, split_id, seed)
                if key not in predictions:
                    logger.warning(
                        "Missing predictions for %s, split=%d, seed=%d. Skipping.",
                        model_name, split_id, seed,
                    )
                    continue
                pred = predictions[key]
                proba = pred["cal_proba"]

            ts = pred["timestamps"]
            ret = pred["ret"]

            all_timestamps.append(ts)
            all_proba.append(proba)
            all_ret.append(ret)

        if not all_timestamps:
            path_results[path_id] = {
                "returns": pd.Series(dtype=float),
                "performance": _empty_performance(),
                "bet_sizes": np.array([]),
                "timestamps": pd.DatetimeIndex([]),
            }
            continue

        # concatenate chronologically
        timestamps = np.concatenate(all_timestamps)
        proba_concat = np.vstack(all_proba)
        ret_concat = np.concatenate(all_ret)

        # sort by timestamp
        sort_idx = np.argsort(timestamps)
        timestamps = pd.DatetimeIndex(timestamps[sort_idx])
        proba_concat = proba_concat[sort_idx]
        ret_concat = ret_concat[sort_idx]

        # bet sizing
        bet_sizes = bet_size_from_proba(proba_concat)

        # strategy returns
        strat_returns = compute_strategy_returns(
            bet_sizes, ret_concat, timestamps
        )

        # performance
        perf = compute_path_performance(strat_returns, bet_sizes)

        path_results[path_id] = {
            "returns": strat_returns,
            "performance": perf,
            "bet_sizes": bet_sizes,
            "timestamps": timestamps,
        }

    return path_results


# =====================================================================
# Deflated Sharpe Ratio (AFML Chapter 14)
# =====================================================================
def compute_deflated_sharpe(
    observed_sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
) -> float:
    """Compute the Deflated Sharpe Ratio correcting for selection bias."""
    if n_trials <= 1:
        return 1.0

    log_n = np.log(n_trials)
    if log_n < 1e-10:
        return 1.0

    euler_gamma = 0.5772156649

    # expected maximum Sharpe under null (all true Sharpe = 0)
    e_max_sr = (
        np.sqrt(2 * log_n)
        * (1 - euler_gamma / (2 * log_n))
        + euler_gamma / (2 * np.sqrt(2 * log_n))
    )

    # standard error of Sharpe accounting for non-normality
    inner = (1 - skew * observed_sharpe
             + (kurtosis - 1) / 4 * observed_sharpe ** 2)
    inner = max(inner, 1e-10)  # clamp to prevent sqrt(negative) → NaN
    sr_std = np.sqrt(inner / max(n_obs - 1, 1))

    if sr_std < 1e-10:
        return 0.0

    dsr = stats.norm.cdf((observed_sharpe - e_max_sr) / sr_std)
    return float(dsr)


# =====================================================================
# Probability of Backtest Overfitting (AFML Chapter 11)
# =====================================================================
def compute_pbo(path_sharpes_matrix: np.ndarray) -> float:
    """Compute PBO via Combinatorially Symmetric Cross-Validation.

    Parameters
    ----------
    path_sharpes_matrix : np.ndarray
        Shape (n_models, n_paths). Sharpe ratio per model per path.
    """
    n_models, n_paths = path_sharpes_matrix.shape

    if n_models < 2 or n_paths < 2:
        return np.nan

    is_size = n_paths // 2
    all_combos = list(combinations(range(n_paths), is_size))

    underperform_count = 0
    total_count = 0

    for is_cols in all_combos:
        oos_cols = [c for c in range(n_paths) if c not in is_cols]

        # IS average Sharpe per model
        is_sharpes = path_sharpes_matrix[:, list(is_cols)].mean(axis=1)

        # IS-best model
        best_model = np.argmax(is_sharpes)

        # OOS Sharpe for IS-best model
        oos_sharpe_best = path_sharpes_matrix[best_model, oos_cols].mean()

        # OOS median across all models
        oos_sharpes_all = path_sharpes_matrix[:, oos_cols].mean(axis=1)
        oos_median = np.median(oos_sharpes_all)

        if oos_sharpe_best < oos_median:
            underperform_count += 1
        total_count += 1

    pbo = underperform_count / total_count if total_count > 0 else np.nan
    return float(pbo)


# =====================================================================
# DeLong AUC Significance Test
# =====================================================================
def _delong_auc_variance(y_true: np.ndarray, y_score: np.ndarray):
    """Compute AUC and its variance using the DeLong method.

    Returns (auc, var_auc).
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m = len(pos)
    n = len(neg)

    if m == 0 or n == 0:
        return np.nan, np.nan

    # structural components (placement values)
    V_pos = np.array([np.mean(neg < p) + 0.5 * np.mean(neg == p) for p in pos])
    V_neg = np.array([np.mean(pos > q) + 0.5 * np.mean(pos == q) for q in neg])

    auc = np.mean(V_pos)
    s10 = np.var(V_pos, ddof=1) if m > 1 else 0.0
    s01 = np.var(V_neg, ddof=1) if n > 1 else 0.0

    var_auc = s10 / m + s01 / n
    return auc, var_auc


def _delong_covariance(y_true, y_score_a, y_score_b):
    """Compute covariance between two AUC estimates (DeLong method)."""
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos_a, neg_a = y_score_a[pos_mask], y_score_a[neg_mask]
    pos_b, neg_b = y_score_b[pos_mask], y_score_b[neg_mask]
    m = len(pos_a)
    n = len(neg_a)

    if m == 0 or n == 0:
        return 0.0

    V_pos_a = np.array([np.mean(neg_a < p) + 0.5 * np.mean(neg_a == p) for p in pos_a])
    V_pos_b = np.array([np.mean(neg_b < p) + 0.5 * np.mean(neg_b == p) for p in pos_b])
    V_neg_a = np.array([np.mean(pos_a > q) + 0.5 * np.mean(pos_a == q) for q in neg_a])
    V_neg_b = np.array([np.mean(pos_b > q) + 0.5 * np.mean(pos_b == q) for q in neg_b])

    cov10 = np.cov(V_pos_a, V_pos_b, ddof=1)[0, 1] if m > 1 else 0.0
    cov01 = np.cov(V_neg_a, V_neg_b, ddof=1)[0, 1] if n > 1 else 0.0

    return cov10 / m + cov01 / n


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> dict:
    """Two-sided DeLong test for comparing two AUC values on the same sample.

    Parameters
    ----------
    y_true : binary labels
    y_score_a : predicted probabilities from model A (positive class)
    y_score_b : predicted probabilities from model B (positive class)

    Returns
    -------
    dict with keys: auc_a, auc_b, z_stat, p_value
    """
    auc_a, var_a = _delong_auc_variance(y_true, y_score_a)
    auc_b, var_b = _delong_auc_variance(y_true, y_score_b)
    cov_ab = _delong_covariance(y_true, y_score_a, y_score_b)

    var_diff = var_a + var_b - 2 * cov_ab
    var_diff = max(var_diff, 1e-15)  # prevent division by zero

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))  # two-sided

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "delta_auc": float(auc_a - auc_b),
        "z_stat": float(z),
        "p_value": float(p),
    }


def compute_auc_significance(
    predictions: dict,
    models: list[str],
    n_splits: int,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run pairwise DeLong AUC tests across all model pairs.

    Pools predictions from all CPCV splits (for a given seed) so that
    the test operates on the full sample. Since CPCV splits have
    non-overlapping test sets, pooling is valid.

    Parameters
    ----------
    predictions : dict from CPCV pipeline
    models : list of model names to compare
    n_splits : number of CPCV splits
    seed : which seed to use (default 0)
    alpha : significance level (default 0.05)

    Returns
    -------
    DataFrame with columns: model_a, model_b, auc_a, auc_b, delta_auc,
                            z_stat, p_value, significant
    """
    # pool predictions per model across splits
    pooled = {}
    for model_name in models:
        y_trues, y_probas = [], []
        for split_idx in range(n_splits):
            key = (model_name, split_idx, seed)
            if key not in predictions:
                logger.warning(
                    "Missing predictions for %s, split=%d, seed=%d. Skipping.",
                    model_name, split_idx, seed,
                )
                continue
            pred = predictions[key]
            y_trues.append(pred["y_true"])
            y_probas.append(pred["cal_proba"][:, 1])

        if y_trues:
            pooled[model_name] = {
                "y_true": np.concatenate(y_trues),
                "y_score": np.concatenate(y_probas),
            }

    # pairwise tests
    results = []
    tested_models = [m for m in models if m in pooled]
    for i, model_a in enumerate(tested_models):
        for model_b in tested_models[i + 1:]:
            # align samples: both models must have same test observations
            # (CPCV guarantees this for same split/seed)
            a = pooled[model_a]
            b = pooled[model_b]

            # use shorter array if lengths differ (e.g., LSTM drops
            # observations from sequencing)
            n = min(len(a["y_true"]), len(b["y_true"]))
            res = delong_test(
                a["y_true"][:n],
                a["y_score"][:n],
                b["y_score"][:n],
            )
            res["model_a"] = model_a
            res["model_b"] = model_b
            res["significant"] = res["p_value"] < alpha
            results.append(res)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)[
        ["model_a", "model_b", "auc_a", "auc_b", "delta_auc",
         "z_stat", "p_value", "significant"]
    ]

    return df
def compute_model_summary(
    model_name: str,
    path_performances: list[dict],
    split_metrics: list[dict],
    n_trials: int,
) -> dict:
    """Aggregate all metrics for a single model."""
    # path-level
    sharpes = [p["annualized_sharpe"] for p in path_performances]
    median_sharpe = float(np.median(sharpes))
    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0

    # pool returns for distributional stats
    all_skew = [p["skewness"] for p in path_performances]
    all_kurt = [p["kurtosis"] for p in path_performances]
    pooled_skew = float(np.mean(all_skew))
    pooled_kurt = float(np.mean(all_kurt))
    total_obs = sum(p["n_trades"] for p in path_performances)

    # DSR
    dsr = compute_deflated_sharpe(
        median_sharpe, max(total_obs, 2), pooled_skew, pooled_kurt, n_trials
    )

    # split-level averages
    def safe_mean(key):
        vals = [m[key] for m in split_metrics if key in m and not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else np.nan

    return {
        "model_name": model_name,
        "median_sharpe": median_sharpe,
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "dsr": dsr,
        "median_max_dd": float(np.median([p["max_drawdown"] for p in path_performances])),
        "median_cum_return": float(np.median([p["cumulative_return"] for p in path_performances])),
        "median_win_rate": float(np.median([p["win_rate"] for p in path_performances])),
        "median_profit_factor": float(np.median([p["profit_factor"] for p in path_performances])),
        "mean_f1": safe_mean("f1_macro"),
        "mean_accuracy": safe_mean("accuracy"),
        "mean_log_loss": safe_mean("log_loss"),
        "mean_auc_roc": safe_mean("auc_roc"),
        "mean_brier": safe_mean("brier_score"),
        "pooled_skew": pooled_skew,
        "pooled_kurt": pooled_kurt,
    }


# =====================================================================
# Model comparison table
# =====================================================================
def compare_models(all_summaries: list[dict]) -> pd.DataFrame:
    """Produce a ranked model comparison DataFrame."""
    df = pd.DataFrame(all_summaries)

    # rank by median_sharpe desc, std_sharpe asc as tiebreaker
    df = df.sort_values(
        ["median_sharpe", "std_sharpe"], ascending=[False, True]
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    display_cols = [
        "rank", "model_name", "median_sharpe", "std_sharpe", "dsr",
        "mean_f1", "mean_accuracy", "mean_auc_roc",
        "median_max_dd", "median_cum_return",
        "median_win_rate", "median_profit_factor",
    ]
    display = df[[c for c in display_cols if c in df.columns]]

    print("\n" + "=" * 80)
    print("Model Comparison Table")
    print("=" * 80)
    print(display.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 80)

    # highlights
    best = df.iloc[0]
    print(f"\n  Best model: {best['model_name']} (median Sharpe: {best['median_sharpe']:.4f})")

    dsr_pass = df[df["dsr"] > 0.95]
    if len(dsr_pass):
        print(f"  DSR > 0.95: {list(dsr_pass['model_name'].values)}")
    else:
        print("  DSR > 0.95: none (results may be statistical artifacts)")

    return df


# =====================================================================
# Feature stability
# =====================================================================
def compute_feature_stability(predictions: dict, models: list[str]) -> dict:
    """Compute how consistently each feature is selected across folds."""
    feature_counts = {}
    n_splits = 0

    for key, pred in predictions.items():
        model_name, split_idx, seed = key
        if seed != 0:
            continue
        if model_name != models[0]:
            continue

        prep = pred.get("prep_info", {})
        selected = prep.get("selected_features", [])

        for f in selected:
            feature_counts[f] = feature_counts.get(f, 0) + 1
        n_splits += 1

    if n_splits == 0:
        return {"feature_frequency": pd.Series(dtype=float),
                "stable_features": [], "n_splits": 0}

    freq = pd.Series(feature_counts).sort_values(ascending=False) / n_splits
    stable = freq[freq > 0.80].index.tolist()

    return {
        "feature_frequency": freq,
        "stable_features": stable,
        "n_splits": n_splits,
    }


# =====================================================================
# FFD stability
# =====================================================================
def compute_ffd_stability(predictions: dict) -> dict:
    """Compute how stable FFD d* values are across folds."""
    d_stars = {}

    for key, pred in predictions.items():
        _, _, seed = key
        if seed != 0:
            continue

        prep = pred.get("prep_info", {})
        ffd = prep.get("ffd", {})

        for col, d in ffd.items():
            if col not in d_stars:
                d_stars[col] = []
            d_stars[col].append(float(d))

    result = {
        "d_star_by_column": d_stars,
        "mean_d_star": {col: float(np.mean(vals)) for col, vals in d_stars.items()},
        "std_d_star": {col: float(np.std(vals)) for col, vals in d_stars.items()},
    }

    for col, std in result["std_d_star"].items():
        if std > 0.1:
            logger.warning(
                "FFD: column '%s' has heterogeneous d* (std=%.3f). "
                "Stationarity structure varies across time periods.", col, std,
            )

    return result


# =====================================================================
# Top-level orchestration
# =====================================================================
def analyze_results(cpcv_results: dict) -> dict:
    """Produce the complete post-CPCV analysis from pipeline results.

    Called from the notebook after ``run_cpcv_pipeline()`` completes.
    """
    predictions = cpcv_results["predictions"]
    path_map = cpcv_results["path_map"]
    n_paths = cpcv_results["n_paths"]
    models = cpcv_results["models"]
    n_seeds = cpcv_results["n_seeds"]
    n_trials = len(models)

    all_summaries = []
    path_sharpes_matrix = np.zeros((len(models), n_paths))

    for model_idx, model_name in enumerate(models):
        print(f"\n  Evaluating: {model_name}...")

        # ── split-level metrics (averaged across seeds) ───────────────
        split_metrics = []
        for split_idx in range(cpcv_results["n_splits"]):
            seed_metrics = []
            for seed in range(n_seeds):
                key = (model_name, split_idx, seed)
                if key not in predictions:
                    continue
                pred = predictions[key]
                metrics = compute_split_metrics(
                    pred["y_true"], pred["y_pred"], pred["cal_proba"]
                )
                seed_metrics.append(metrics)

            if seed_metrics:
                avg = {}
                for k in seed_metrics[0]:
                    vals = [m[k] for m in seed_metrics if not np.isnan(m[k])]
                    avg[k] = float(np.mean(vals)) if vals else np.nan
                split_metrics.append(avg)

        # ── stitch paths (seed-averaged for financial metrics) ────────
        path_results = stitch_paths(
            predictions, path_map, n_paths, model_name,
            seed=0, n_seeds=n_seeds,
        )

        path_performances = []
        for path_id in range(n_paths):
            perf = path_results[path_id]["performance"]
            path_performances.append(perf)
            path_sharpes_matrix[model_idx, path_id] = perf["annualized_sharpe"]

        # ── model summary ─────────────────────────────────────────────
        summary = compute_model_summary(
            model_name, path_performances, split_metrics, n_trials
        )
        all_summaries.append(summary)

    # ── model comparison ──────────────────────────────────────────────
    comparison_df = compare_models(all_summaries)

    # ── PBO ────────────────────────────────────────────────────────────
    pbo = compute_pbo(path_sharpes_matrix)
    print(f"\n  Probability of Backtest Overfitting (PBO): {pbo:.4f}")
    if pbo < 0.3:
        print("  → PBO < 0.3: model selection appears robust.")
    elif pbo > 0.5:
        print("  → PBO > 0.5: in-sample winner tends to underperform OOS. Caution.")
    else:
        print("  → PBO in [0.3, 0.5]: moderate overfitting risk.")

    # ── feature stability ─────────────────────────────────────────────
    feature_stability = compute_feature_stability(predictions, models)

    # ── AUC significance (DeLong pairwise tests) ─────────────────────
    auc_significance = compute_auc_significance(
        predictions, models, cpcv_results["n_splits"], seed=0
    )
    if len(auc_significance):
        print("\n" + "=" * 80)
        print("AUC Significance Tests (DeLong, pooled across splits, seed=0)")
        print("=" * 80)
        print(auc_significance.to_string(index=False, float_format="{:.4f}".format))
        n_sig = auc_significance["significant"].sum()
        n_total = len(auc_significance)
        print(f"\n  {n_sig}/{n_total} pairs significantly different (α=0.05)")
        print("=" * 80)

    # ── FFD stability ─────────────────────────────────────────────────
    ffd_stability = compute_ffd_stability(predictions)

    # ── stitch all models for return ──────────────────────────────────
    all_path_results = {}
    for m in models:
        all_path_results[m] = stitch_paths(
            predictions, path_map, n_paths, m,
            seed=0, n_seeds=n_seeds,
        )

    return {
        "comparison": comparison_df,
        "pbo": pbo,
        "path_sharpes": path_sharpes_matrix,
        "all_summaries": all_summaries,
        "path_results": all_path_results,
        "feature_stability": feature_stability,
        "ffd_stability": ffd_stability,
        "auc_significance": auc_significance,
    }