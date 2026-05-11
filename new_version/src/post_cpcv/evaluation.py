"""
14.1) Evaluation
=======================
Take raw predictions from the CPCV pipeline and produce:
  - Per-split classification metrics
  - Bet sizing via López de Prado's S-curve (AFML Ch. 10)
  - Full-span CPCV backtest paths (AFML §12.4)
  - Path-level financial performance metrics
  - Deflated Sharpe Ratio (AFML Ch. 14)
  - Probability of Backtest Overfitting (AFML Ch. 11)
  - DeLong AUC significance tests
  - Final model comparison table

Single orchestration entry point: ``analyze_results(cpcv_results)``.
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

# --- Module-level constants -------------------------------------------------
TRANSACTION_COST = 0.001          # 0.1% round-trip cost for BTC
MIN_BET_SIZE = 0.05               # absolute bet sizes below this snap to 0 (no trade)
MAX_BET_SIZE = 0.75               # cap to prevent explosive equity curves
BET_DISCRETIZATION = [0.0, 0.25, 0.50, 0.75]
ANNUALIZATION_FACTOR = 365        # BTC trades every calendar day
RISK_FREE_RATE = 0.0              # assume 0 for crypto


# --- 1. Split-level classification metrics ---------------------------------
# Per-test-fold classification metrics: accuracy, F1, precision/recall, log loss, Brier, AUC.
def compute_split_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """Compute the canonical seven-metric block for one split's test fold."""
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    f1_mac = f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)
    f1_per = f1_score(y_true, y_pred, average=None, sample_weight=sample_weight)
    prec = precision_score(
        y_true, y_pred, average="macro", sample_weight=sample_weight, zero_division=0
    )
    rec = recall_score(
        y_true, y_pred, average="macro", sample_weight=sample_weight, zero_division=0
    )

    # log_loss and AUC can fail on single-class folds; NaN-on-error keeps the row well-formed.
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


# --- 2. Bet sizing (AFML §10.3) --------------------------------------------
# Convert calibrated probabilities into signed position sizes via López de Prado's S-curve.
def bet_size_from_proba(
    proba: np.ndarray,
    min_bet: float = MIN_BET_SIZE,
    discretize: bool = True,
) -> np.ndarray:
    """Map calibrated probabilities to signed bet sizes in ``[-MAX_BET_SIZE, +MAX_BET_SIZE]``."""
    # Predicted direction is the argmax class; the |bet| comes from the S-curve below.
    direction = np.where(proba[:, 1] > proba[:, 0], 1, -1)

    # Confidence = probability assigned to predicted class.
    p = np.maximum(proba[:, 0], proba[:, 1])

    # Z-score transform: distance from p=0.5 in standard-error units.
    z = (p - 0.5) / np.sqrt(p * (1 - p) + 1e-10)

    # S-curve mapping (AFML §10.3): bet ∈ [-1, +1] scaled from z.
    raw_bet = 2.0 * stats.norm.cdf(z) - 1.0

    # Cap and threshold: positions above MAX_BET_SIZE are clipped, low confidence becomes no-trade.
    raw_bet = np.clip(raw_bet, -MAX_BET_SIZE, MAX_BET_SIZE)
    raw_bet = np.where(np.abs(raw_bet) < min_bet, 0.0, raw_bet)

    signed_bet = direction * np.abs(raw_bet)

    # Snap to the canonical four-step grid so bet sizes are interpretable.
    if discretize:
        disc = np.array(BET_DISCRETIZATION)
        abs_bet = np.abs(signed_bet)
        for i in range(len(abs_bet)):
            closest_idx = np.argmin(np.abs(disc - abs_bet[i]))
            abs_bet[i] = disc[closest_idx]
        signed_bet = np.sign(signed_bet) * abs_bet

    return signed_bet


# --- 3. Strategy returns ---------------------------------------------------
# Net strategy returns: gross PnL minus transaction cost on each |Δbet|.
def compute_strategy_returns(
    bet_sizes: np.ndarray,
    label_returns: np.ndarray,
    timestamps: pd.DatetimeIndex,
    cost: float = TRANSACTION_COST,
) -> pd.Series:
    """Return net per-event strategy returns indexed by timestamp."""
    gross = bet_sizes * label_returns

    # Turnover at each event = |Δbet|. ``prepend=0`` makes the first element |bet_sizes[0]|
    # automatically (cost of opening the initial position).
    turnover = np.abs(np.diff(bet_sizes, prepend=0))

    tx_cost = cost * turnover
    net = gross - tx_cost

    return pd.Series(net, index=timestamps, name="strategy_return")


# --- 4. Bootstrap utilities -------------------------------------------------
# Resample-with-replacement bootstrap for a CI on the median; the standard non-parametric tool.
def bootstrap_median_ci(
    values,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Return ``(lower, upper)`` percentile CI on the median; ``(NaN, NaN)`` when n < 2.

    Non-finite entries (NaN, inf) are dropped before resampling so paths with undefined
    Sharpe (e.g. zero-trade paths producing NaN) don't poison the percentile estimate.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = len(arr)
    medians = np.empty(n_bootstrap, dtype=float)
    # Standard bootstrap: resample with replacement, compute median, repeat n_bootstrap times.
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        medians[i] = np.median(sample)

    lower = float(np.percentile(medians, 100.0 * alpha / 2))
    upper = float(np.percentile(medians, 100.0 * (1 - alpha / 2)))
    return (lower, upper)


# --- 5. Path-level financial performance -----------------------------------
# Compute the full performance metric dict for one backtest path (Sharpe, Sortino, Calmar, drawdown, etc.).
def compute_path_performance(
    strategy_returns: pd.Series, bet_sizes: np.ndarray
) -> dict:
    """Compute Sharpe, Sortino, Calmar, drawdown, win rate, profit factor, and moments for one path."""
    returns = strategy_returns.values
    n = len(returns)

    if n == 0:
        return _empty_performance()

    # Sharpe ratio: annualised by sqrt of the event sampling factor.
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1) if n > 1 else 1e-10
    if std_r < 1e-10:
        sharpe = 0.0
    else:
        sharpe = (mean_r - RISK_FREE_RATE / ANNUALIZATION_FACTOR) / std_r * np.sqrt(ANNUALIZATION_FACTOR)

    cum_ret = np.prod(1.0 + returns) - 1.0

    # Annualised return is computed over calendar time, not event count, because CUSUM events
    # arrive at ~75/year whereas daily bars arrive at 365/year; using n directly would over-annualise.
    ts = strategy_returns.index
    if len(ts) < 2:
        years_elapsed = 0.0
        ann_ret = 0.0
    else:
        years_elapsed = max((ts[-1] - ts[0]).days / 365.25, 1.0 / 365.25)
        ann_ret = (1.0 + cum_ret) ** (1.0 / years_elapsed) - 1.0

    # Maximum drawdown: peak-to-trough percentage decline on the equity curve.
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = np.min(drawdowns)

    # Longest time under water = longest run of equity < running_max.
    underwater = equity < running_max
    if underwater.any():
        uw_groups = np.diff(np.where(np.concatenate(([False], underwater, [False])))[0])
        time_uw = int(np.max(uw_groups)) if len(uw_groups) > 0 else 0
    else:
        time_uw = 0

    # Win rate among traded observations (zero-bet rows excluded).
    traded_mask = bet_sizes != 0
    traded_returns = returns[traded_mask]
    n_trades = int(traded_mask.sum())
    win_rate = np.mean(traded_returns > 0) if n_trades > 0 else 0.0

    # Profit factor — three-way semantics for the edge cases:
    #   no trades at all  → NaN (undefined; matches _empty_performance)
    #   trades, no losses → inf (winning streak)
    #   trades, no winners → 0.0
    #   otherwise         → pos / |neg|
    pos_returns = traded_returns[traded_returns > 0].sum()
    neg_returns = traded_returns[traded_returns < 0].sum()
    if n_trades == 0:
        profit_factor = np.nan
    elif neg_returns == 0:
        profit_factor = np.inf
    else:
        profit_factor = pos_returns / abs(neg_returns)

    avg_bet = np.mean(np.abs(bet_sizes[traded_mask])) if n_trades > 0 else 0.0

    # Sortino ratio (Sortino & Price 1994): penalises only downside vol, uses the RMS of
    # negative returns (the standard construction). Annualised the same way as Sharpe.
    downside = returns[returns < 0]
    if len(downside) == 0:
        # Winning streak with no losing day: positive mean ⇒ +∞, exactly-zero mean ⇒ 0.
        sortino = np.inf if mean_r > 0 else 0.0
    else:
        downside_rms = np.sqrt(np.mean(downside ** 2))
        if downside_rms < 1e-10:
            sortino = np.inf if mean_r > 0 else 0.0
        else:
            sortino = (
                (mean_r - RISK_FREE_RATE / ANNUALIZATION_FACTOR)
                / downside_rms
                * np.sqrt(ANNUALIZATION_FACTOR)
            )

    # Calmar ratio: annualised return / |max drawdown|. Tail-risk-aware alternative to Sharpe.
    if abs(max_dd) < 1e-10:
        calmar = np.inf if ann_ret > 0 else 0.0
    else:
        calmar = ann_ret / abs(max_dd)

    # Distributional moments: scipy returns NaN on constant inputs, so clamp to 0 for safety.
    skewness = float(stats.skew(returns)) if n > 2 else 0.0
    kurtosis = float(stats.kurtosis(returns)) if n > 3 else 0.0
    if np.isnan(skewness):
        skewness = 0.0
    if np.isnan(kurtosis):
        kurtosis = 0.0

    return {
        "annualized_sharpe": sharpe,
        "annualized_sortino": sortino,
        "calmar": calmar,
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "max_drawdown": max_dd,
        "time_under_water": time_uw,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "n_returns": n,
        "years_elapsed": years_elapsed,
        "avg_bet_size": avg_bet,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


# Zero-filled performance dict for empty paths.
def _empty_performance() -> dict:
    """Return a zero-filled performance dict matching the schema of ``compute_path_performance``."""
    return {
        "annualized_sharpe": 0.0, "annualized_sortino": 0.0, "calmar": 0.0,
        "cumulative_return": 0.0,
        "annualized_return": 0.0, "max_drawdown": 0.0,
        "time_under_water": 0, "win_rate": 0.0,
        "profit_factor": np.nan, "n_trades": 0,
        "n_returns": 0, "years_elapsed": 0.0,
        "avg_bet_size": 0.0, "skewness": 0.0, "kurtosis": 0.0,
    }


# --- 6. Path stitching (AFML §12.4) -----------------------------------------
# Mirror of cv._compute_group_bounds; kept local so this module doesn't import a private symbol.
def _compute_group_bounds(T: int, n_groups: int) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` with exclusive end; groups 0..N-2 have size ⌊T/N⌋."""
    base_size = T // n_groups
    bounds = []
    for g in range(n_groups):
        start = g * base_size
        end = (g + 1) * base_size if g < n_groups - 1 else T
        bounds.append((start, end))
    return bounds


# Recover the full event index from stored predictions; needed for mapping timestamps to positions.
def _derive_event_index(predictions: dict) -> pd.DatetimeIndex:
    """Return the sorted, de-duplicated union of all stored timestamp slices.

    The pipeline does not store ``X.index`` directly; every prediction stores its own
    ``timestamps`` slice. Their union recovers the alignment-time event index.
    """
    seen = set()
    for pred in predictions.values():
        ts = pred.get("timestamps")
        if ts is None:
            continue
        for t in pd.DatetimeIndex(ts):
            seen.add(t)
    return pd.DatetimeIndex(sorted(seen))


# Assemble full backtest paths from per-split predictions using the path matrix; seed-averages when n_seeds > 1.
def stitch_paths(
    predictions: dict,
    path_map: dict,
    n_paths: int,
    model_name: str,
    seed: int = 0,
    n_seeds: int = 1,
    *,
    event_index: pd.DatetimeIndex | None = None,
    group_bounds: list[tuple[int, int]] | None = None,
    n_groups: int = 6,
) -> dict:
    """Build ``{path_id: {returns, performance, bet_sizes, timestamps}}`` for one model.

    For each path, ``path_map[path_id] = [(group_id, split_id), ...]`` tells the function
    which split's predictions to use for which chronological group. Each split's stored test
    set covers ``k`` groups; this function filters down to the events belonging to ``group_id``
    before stitching, so a single split's predictions can contribute to multiple paths cleanly.

    When ``n_seeds > 1``, calibrated probabilities are averaged across all available seeds
    before bet-sizing; the ensemble reduces prediction variance by ~1/sqrt(n_seeds).
    """
    # Derive the helpers when the caller doesn't pre-supply them (the standard analyse path does).
    if event_index is None:
        event_index = _derive_event_index(predictions)
    if group_bounds is None:
        group_bounds = _compute_group_bounds(len(event_index), n_groups)

    path_results = {}

    for path_id in range(n_paths):
        assignments = path_map[path_id]

        all_timestamps = []
        all_proba = []
        all_ret = []

        # Walk the path's (group, split) assignments in chronological group order.
        for group_id, split_id in sorted(assignments, key=lambda x: x[0]):
            if n_seeds > 1:
                # Seed-averaging branch: collect probabilities across all available seeds.
                seed_probas = []
                seed_keys = []
                for s in range(n_seeds):
                    key = (model_name, split_id, s)
                    if key in predictions:
                        seed_probas.append(predictions[key]["cal_proba"])
                        seed_keys.append(key)

                if not seed_probas or not seed_keys:
                    logger.warning(
                        "No seed predictions for %s, split=%d. Skipping.",
                        model_name, split_id,
                    )
                    continue

                # Different lengths across seeds can occur for LSTM (windowing edge cases);
                # truncate to the shortest before averaging.
                min_len = min(p.shape[0] for p in seed_probas)
                seed_probas = [p[:min_len] for p in seed_probas]

                proba = np.mean(seed_probas, axis=0)
                ref_key = seed_keys[0]
                pred = predictions[ref_key]
            else:
                # Single-seed branch.
                key = (model_name, split_id, seed)
                if key not in predictions:
                    logger.warning(
                        "Missing predictions for %s, split=%d, seed=%d. Skipping.",
                        model_name, split_id, seed,
                    )
                    continue
                pred = predictions[key]
                proba = pred["cal_proba"]

            ts = pd.DatetimeIndex(pred["timestamps"])
            ret = np.asarray(pred["ret"])

            # LSTM windowing can make timestamps slightly longer than proba; align by truncation.
            n_proba = len(proba)
            if len(ts) > n_proba:
                ts = ts[:n_proba]
                ret = ret[:n_proba]

            # Filter by CPCV group: only events whose positional index falls within
            # group_bounds[group_id] belong to this (path, group, split) assignment.
            positions = event_index.get_indexer(ts)
            start, end = group_bounds[group_id]
            mask = (positions >= start) & (positions < end)

            if not mask.any():
                logger.debug(
                    "No events for %s, path=%d, group=%d, split=%d after "
                    "group filter.", model_name, path_id, group_id, split_id,
                )
                continue

            ts = ts[mask]
            proba = proba[mask]
            ret = ret[mask]

            all_timestamps.append(np.asarray(ts))
            all_proba.append(proba)
            all_ret.append(ret)

        # Empty path: store the zero-filled performance dict and move on.
        if not all_timestamps:
            path_results[path_id] = {
                "returns": pd.Series(dtype=float),
                "performance": _empty_performance(),
                "bet_sizes": np.array([]),
                "timestamps": pd.DatetimeIndex([]),
            }
            continue

        # Concatenate and sort chronologically.
        timestamps = np.concatenate(all_timestamps)
        proba_concat = np.vstack(all_proba)
        ret_concat = np.concatenate(all_ret)

        sort_idx = np.argsort(timestamps)
        timestamps = pd.DatetimeIndex(timestamps[sort_idx])
        proba_concat = proba_concat[sort_idx]
        ret_concat = ret_concat[sort_idx]

        # Post-stitch sanity check: each event should appear at most once per path under CPCV.
        n_dup = len(timestamps) - timestamps.nunique()
        if n_dup > 0:
            logger.warning(
                "Path %d for %s has %d duplicate timestamps after group "
                "filter; this should not happen under standard CPCV.",
                path_id, model_name, n_dup,
            )

        # Bet sizing → strategy returns → performance metrics.
        bet_sizes = bet_size_from_proba(proba_concat)
        strat_returns = compute_strategy_returns(
            bet_sizes, ret_concat, timestamps
        )
        perf = compute_path_performance(strat_returns, bet_sizes)

        path_results[path_id] = {
            "returns": strat_returns,
            "performance": perf,
            "bet_sizes": bet_sizes,
            "timestamps": timestamps,
        }

    return path_results


# --- 7. Deflated Sharpe Ratio (AFML Ch. 14) --------------------------------
# Bailey & López de Prado (2014): adjusts Sharpe for non-normality and multiple-trials selection bias.
def compute_deflated_sharpe(
    observed_sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    annualization_factor: float = ANNUALIZATION_FACTOR
) -> float:
    """Return DSR ∈ [0, 1]; values > 0.95 indicate the Sharpe survives multiple-testing correction."""
    if n_trials <= 1:
        return 1.0

    log_n = np.log(n_trials)
    if log_n < 1e-10:
        return 1.0

    euler_gamma = 0.5772156649

    # The Mertens variance formula applies to the per-period (non-annualised) Sharpe,
    # so de-annualise before computing the standard error.
    non_ann_sr = observed_sharpe / np.sqrt(annualization_factor)

    # Expected maximum standardised Sharpe under the null, AFML §14.4.
    e_max_z = (
        np.sqrt(2 * log_n)
        * (1 - euler_gamma / (2 * log_n))
        + euler_gamma / (2 * np.sqrt(2 * log_n))
    )

    # Standard error of the non-annualised Sharpe accounting for non-normality (Mertens 2002 /
    # AFML Eq. 14.5). Note: scipy.stats.kurtosis returns *excess* kurtosis (γ_4 − 3); Mertens uses
    # raw γ_4, so the conversion is (γ_4 − 1)/4 = (excess + 2)/4.
    inner = (1 - skew * non_ann_sr
             + (kurtosis + 2) / 4 * non_ann_sr ** 2)
    inner = max(inner, 1e-10)  # clamp to prevent sqrt(negative)
    sr_std = np.sqrt(inner / max(n_obs - 1, 1))

    if sr_std < 1e-10:
        return 0.0

    expected_max_sr = e_max_z * sr_std
    dsr = stats.norm.cdf((non_ann_sr - expected_max_sr) / sr_std)
    return float(dsr)


# --- 8. Probability of Backtest Overfitting (AFML Ch. 11) ------------------
# Combinatorially-Symmetric CV: how often the in-sample winner underperforms the OOS median.
def compute_pbo(path_sharpes_matrix: np.ndarray) -> float:
    """Compute PBO from a ``(n_models, n_paths)`` Sharpe matrix; returns NaN when matrix is too small.

    For each balanced (IS, OOS) partition of the paths, the IS-best model's OOS Sharpe is
    compared to the OOS median; PBO is the share of partitions where the IS-best underperforms.
    """
    n_models, n_paths = path_sharpes_matrix.shape

    if n_models < 2 or n_paths < 2:
        return np.nan

    # Combinatorial partition into half IS, half OOS paths.
    is_size = n_paths // 2
    all_combos = list(combinations(range(n_paths), is_size))

    underperform_count = 0
    total_count = 0

    for is_cols in all_combos:
        oos_cols = [c for c in range(n_paths) if c not in is_cols]

        # Pick the IS-best model by mean Sharpe on the IS paths.
        is_sharpes = path_sharpes_matrix[:, list(is_cols)].mean(axis=1)
        best_model = np.argmax(is_sharpes)

        # Compare that model's OOS Sharpe to the OOS median across all models.
        oos_sharpe_best = path_sharpes_matrix[best_model, oos_cols].mean()
        oos_sharpes_all = path_sharpes_matrix[:, oos_cols].mean(axis=1)
        oos_median = np.median(oos_sharpes_all)

        if oos_sharpe_best < oos_median:
            underperform_count += 1
        total_count += 1

    pbo = underperform_count / total_count if total_count > 0 else np.nan
    return float(pbo)


# --- 9. DeLong AUC Significance Test ---------------------------------------
# AUC variance via DeLong's placement-value decomposition.
def _delong_auc_variance(y_true: np.ndarray, y_score: np.ndarray):
    """Return ``(auc, var_auc)`` from DeLong's structural components."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m = len(pos)
    n = len(neg)

    if m == 0 or n == 0:
        return np.nan, np.nan

    # Placement values: for each positive, the share of negatives below it (with 0.5 for ties).
    V_pos = np.array([np.mean(neg < p) + 0.5 * np.mean(neg == p) for p in pos])
    V_neg = np.array([np.mean(pos > q) + 0.5 * np.mean(pos == q) for q in neg])

    auc = np.mean(V_pos)
    s10 = np.var(V_pos, ddof=1) if m > 1 else 0.0
    s01 = np.var(V_neg, ddof=1) if n > 1 else 0.0

    var_auc = s10 / m + s01 / n
    return auc, var_auc


# Paired AUC covariance via DeLong; needed for the test on the AUC difference.
def _delong_covariance(y_true, y_score_a, y_score_b):
    """Return ``Cov(AUC_a, AUC_b)`` under the DeLong method on paired predictions."""
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


# Two-sided DeLong test on two AUC values evaluated on the same paired sample.
def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> dict:
    """Run the DeLong test; return ``{auc_a, auc_b, delta_auc, z_stat, p_value}``."""
    auc_a, var_a = _delong_auc_variance(y_true, y_score_a)
    auc_b, var_b = _delong_auc_variance(y_true, y_score_b)
    cov_ab = _delong_covariance(y_true, y_score_a, y_score_b)

    # Variance of the AUC difference: var_a + var_b − 2·cov_ab; clamp to avoid div-by-zero.
    var_diff = var_a + var_b - 2 * cov_ab
    var_diff = max(var_diff, 1e-15)

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))  # two-sided

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "delta_auc": float(auc_a - auc_b),
        "z_stat": float(z),
        "p_value": float(p),
    }


# Pairwise DeLong tests across all model pairs; predictions pooled across splits + seed-averaged.
def compute_auc_significance(
    predictions: dict,
    models: list[str],
    n_splits: int,
    n_seeds: int = 1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Return a DataFrame of all pairwise DeLong tests with ``significant`` column at level ``alpha``.

    Probabilities are seed-averaged per split (matching ``stitch_paths``) then pooled across
    splits before testing. Pooling is valid because CPCV splits have non-overlapping test sets.
    """
    # Build the per-model pool: y_true + seed-averaged y_score across every split.
    pooled = {}
    for model_name in models:
        y_trues, y_probas = [], []
        for split_idx in range(n_splits):
            seed_probas = []
            seed_y_true = None
            for seed in range(n_seeds):
                key = (model_name, split_idx, seed)
                if key not in predictions:
                    continue
                pred = predictions[key]
                seed_probas.append(pred["cal_proba"][:, 1])
                if seed_y_true is None:
                    seed_y_true = pred["y_true"]

            if not seed_probas:
                logger.warning(
                    "Missing predictions for %s, split=%d (all seeds). Skipping.",
                    model_name, split_idx,
                )
                continue

            # Align to shortest length across seeds (LSTM windowing edge cases).
            min_len = min(len(p) for p in seed_probas)
            seed_probas_aligned = np.stack([p[:min_len] for p in seed_probas])
            avg_proba = seed_probas_aligned.mean(axis=0)

            y_trues.append(seed_y_true[:min_len])
            y_probas.append(avg_proba)

        if y_trues:
            pooled[model_name] = {
                "y_true": np.concatenate(y_trues),
                "y_score": np.concatenate(y_probas),
            }

    # Pairwise tests; truncate to the shorter array when LSTM produces a different length.
    results = []
    tested_models = [m for m in models if m in pooled]
    for i, model_a in enumerate(tested_models):
        for model_b in tested_models[i + 1:]:
            a = pooled[model_a]
            b = pooled[model_b]

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


# --- 10. Per-model summary aggregator --------------------------------------
# Collapse path-level and split-level metrics into the one-row dict that feeds compare_models.
def compute_model_summary(
    model_name: str,
    path_performances: list[dict],
    split_metrics: list[dict],
    n_trials: int,
) -> dict:
    """Aggregate path + split metrics into a single per-model summary dict."""
    # Path-level Sharpe statistics.
    sharpes = [p["annualized_sharpe"] for p in path_performances]
    median_sharpe = float(np.median(sharpes))
    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0

    # Pool skewness and kurtosis across paths so DSR sees representative non-normality.
    all_skew = [p["skewness"] for p in path_performances]
    all_kurt = [p["kurtosis"] for p in path_performances]
    pooled_skew = float(np.mean(all_skew))
    pooled_kurt = float(np.mean(all_kurt))

    # DSR's n_obs must match the n used to estimate Sharpe. Sharpe is computed over the full
    # event series (zero-bet rows included), so n_obs is the average n_returns per path, not n_trades.
    avg_n_returns = int(np.mean([p["n_returns"] for p in path_performances]))

    dsr = compute_deflated_sharpe(
        median_sharpe, max(avg_n_returns, 2), pooled_skew, pooled_kurt, n_trials
    )

    # Split-level averages with NaN-safe reduction.
    def safe_mean(key):
        vals = [m[key] for m in split_metrics if key in m and not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else np.nan

    # Profit factor uses nanmedian rather than median because compute_path_performance returns
    # NaN for paths with zero trades; nanmedian skips them rather than letting one NaN poison
    # the whole reduction. Inf entries (winning streaks) are kept as legitimate large values.
    pf_values = [p["profit_factor"] for p in path_performances]
    pf_arr = np.asarray(pf_values, dtype=float)
    if np.all(np.isnan(pf_arr)):
        median_pf = np.nan
    else:
        median_pf = float(np.nanmedian(pf_arr))

    # Sortino and Calmar can be ±∞ on winning-streak paths; filter to finite entries before nanmedian.
    sortino_arr = np.asarray(
        [p["annualized_sortino"] for p in path_performances], dtype=float,
    )
    calmar_arr = np.asarray(
        [p["calmar"] for p in path_performances], dtype=float,
    )
    median_sortino = (
        float(np.nanmedian(sortino_arr[np.isfinite(sortino_arr)]))
        if np.any(np.isfinite(sortino_arr)) else np.nan
    )
    median_calmar = (
        float(np.nanmedian(calmar_arr[np.isfinite(calmar_arr)]))
        if np.any(np.isfinite(calmar_arr)) else np.nan
    )

    # Bootstrap 95% CI on the median Sharpe so the thesis can quantify estimate stability.
    sharpe_ci_lower, sharpe_ci_upper = bootstrap_median_ci(
        sharpes, n_bootstrap=1000, alpha=0.05, seed=42,
    )

    return {
        "model_name": model_name,
        "median_sharpe": median_sharpe,
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "sharpe_ci_lower": sharpe_ci_lower,
        "sharpe_ci_upper": sharpe_ci_upper,
        "dsr": dsr,
        "median_sortino": median_sortino,
        "median_calmar": median_calmar,
        "median_max_dd": float(np.median([p["max_drawdown"] for p in path_performances])),
        "median_cum_return": float(np.median([p["cumulative_return"] for p in path_performances])),
        "median_win_rate": float(np.median([p["win_rate"] for p in path_performances])),
        "median_profit_factor": median_pf,
        "mean_f1": safe_mean("f1_macro"),
        "mean_accuracy": safe_mean("accuracy"),
        "mean_log_loss": safe_mean("log_loss"),
        "mean_auc_roc": safe_mean("auc_roc"),
        "mean_brier": safe_mean("brier_score"),
        "pooled_skew": pooled_skew,
        "pooled_kurt": pooled_kurt,
    }


# --- 11. Buy-and-hold benchmark --------------------------------------------
# Reconstructs the BH equity curve on the same CPCV path structure so it slots into compare_models.
def compute_buy_and_hold_summary(
    predictions: dict,
    path_map: dict,
    n_paths: int,
    reference_model: str,
    seed: int = 0,
) -> dict:
    """Build a buy-and-hold summary row in the same shape as ``compute_model_summary``.

    For each path, reconstructs the chronological event sequence from the reference model's
    stored timestamps and returns (which are model-invariant for a given split/seed), holds a
    long position of size 1.0 over every event, and computes path-level metrics identically to
    the model paths. Predictive metrics (F1, accuracy, AUC, log loss, Brier) are NaN because BH
    makes no probabilistic predictions.

    The BH benchmark is fully leveraged (size 1.0) whereas the models cap at MAX_BET_SIZE = 0.75,
    so the asymmetry is conservative for the model side: a 1.0-leveraged BH is harder to beat
    than a 0.75-leveraged one.
    """
    path_performances = []

    for path_id in range(n_paths):
        assignments = path_map[path_id]

        all_timestamps = []
        all_ret = []

        # Walk the path's assignments and accumulate timestamps + returns from the reference model.
        for group_id, split_id in sorted(assignments, key=lambda x: x[0]):
            key = (reference_model, split_id, seed)
            if key not in predictions:
                continue
            pred = predictions[key]
            all_timestamps.append(pred["timestamps"])
            all_ret.append(pred["ret"])

        if not all_timestamps:
            path_performances.append(_empty_performance())
            continue

        # Sort chronologically.
        timestamps = np.concatenate(all_timestamps)
        ret_concat = np.concatenate(all_ret)
        sort_idx = np.argsort(timestamps)
        timestamps = pd.DatetimeIndex(timestamps[sort_idx])
        ret_concat = ret_concat[sort_idx]

        # Always-long, full-size position; tx cost applies only to the initial buy.
        bet_sizes = np.ones(len(ret_concat), dtype=float)
        strat_returns = compute_strategy_returns(
            bet_sizes, ret_concat, timestamps,
        )

        perf = compute_path_performance(strat_returns, bet_sizes)
        path_performances.append(perf)

    # Aggregate across paths using the same reductions as compute_model_summary.
    sharpes = [p["annualized_sharpe"] for p in path_performances]
    median_sharpe = float(np.median(sharpes))
    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0

    pf_arr = np.asarray(
        [p["profit_factor"] for p in path_performances], dtype=float,
    )
    if np.all(np.isnan(pf_arr)):
        median_pf = np.nan
    else:
        median_pf = float(np.nanmedian(pf_arr))

    sortino_arr = np.asarray(
        [p["annualized_sortino"] for p in path_performances], dtype=float,
    )
    calmar_arr = np.asarray(
        [p["calmar"] for p in path_performances], dtype=float,
    )
    median_sortino = (
        float(np.nanmedian(sortino_arr[np.isfinite(sortino_arr)]))
        if np.any(np.isfinite(sortino_arr)) else np.nan
    )
    median_calmar = (
        float(np.nanmedian(calmar_arr[np.isfinite(calmar_arr)]))
        if np.any(np.isfinite(calmar_arr)) else np.nan
    )

    sharpe_ci_lower, sharpe_ci_upper = bootstrap_median_ci(
        sharpes, n_bootstrap=1000, alpha=0.05, seed=42,
    )

    return {
        "model_name": "buy_and_hold",
        "median_sharpe": median_sharpe,
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "sharpe_ci_lower": sharpe_ci_lower,
        "sharpe_ci_upper": sharpe_ci_upper,
        "dsr": np.nan,  # DSR requires a multiple-trials selection context; BH is not selected.
        "median_sortino": median_sortino,
        "median_calmar": median_calmar,
        "median_max_dd": float(np.median([p["max_drawdown"] for p in path_performances])),
        "median_cum_return": float(np.median([p["cumulative_return"] for p in path_performances])),
        "median_win_rate": float(np.median([p["win_rate"] for p in path_performances])),
        "median_profit_factor": median_pf,
        "mean_f1": np.nan,
        "mean_accuracy": np.nan,
        "mean_log_loss": np.nan,
        "mean_auc_roc": np.nan,
        "mean_brier": np.nan,
        "pooled_skew": float(np.mean([p["skewness"] for p in path_performances])),
        "pooled_kurt": float(np.mean([p["kurtosis"] for p in path_performances])),
        "_path_performances": path_performances,
    }


# --- 12. Final model comparison table --------------------------------------
# Build the ranked side-by-side comparison DataFrame and print the headline table.
def compare_models(all_summaries: list[dict]) -> pd.DataFrame:
    """Return a ranked DataFrame across models; also prints the headline comparison table."""
    df = pd.DataFrame(all_summaries)

    # Rank by median_sharpe desc, std_sharpe asc as a tie-breaker (lower variance wins ties).
    df = df.sort_values(
        ["median_sharpe", "std_sharpe"], ascending=[False, True]
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    display_cols = [
        "rank", "model_name",
        "median_sharpe", "std_sharpe",
        "sharpe_ci_lower", "sharpe_ci_upper",
        "dsr",
        "median_sortino", "median_calmar",
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

    # Headline highlights for the cell output.
    best = df.iloc[0]
    print(f"\n  Best model: {best['model_name']} (median Sharpe: {best['median_sharpe']:.4f})")

    dsr_pass = df[df["dsr"] > 0.95]
    if len(dsr_pass):
        print(f"  DSR > 0.95: {list(dsr_pass['model_name'].values)}")
    else:
        print("  DSR > 0.95: none (results may be statistical artifacts)")

    return df


# --- 13. Feature stability --------------------------------------------------
# Count how consistently each feature is selected across all CPCV folds × seeds.
def compute_feature_stability(predictions: dict, models: list[str]) -> dict:
    """Return ``{feature_frequency, stable_features, n_splits}`` for the first non-AR model.

    With ``n_seeds`` > 1, counting across all (split, seed) pairs gives a denser frequency
    estimate than using only ``seed=0``, reducing seed-induced noise.
    """
    # AR Logistic skips feature selection (uses lag columns by name), so find the first non-AR model.
    reference_model = None
    for m in models:
        if m != "ar_logistic":
            reference_model = m
            break

    if reference_model is None:
        logger.warning(
            "Feature stability: no non-AR model found in %s; skipping.", models,
        )
        return {"feature_frequency": pd.Series(dtype=float),
                "stable_features": [], "n_splits": 0}

    feature_counts = {}
    n_splits_seen = set()
    n_observations = 0

    # Walk every (model, split, seed) entry, incrementing the count for each selected feature.
    for key, pred in predictions.items():
        model_name, split_idx, seed = key
        if model_name != reference_model:
            continue

        prep = pred.get("prep_info", {})
        selected = prep.get("selected_features", [])

        for f in selected:
            feature_counts[f] = feature_counts.get(f, 0) + 1
        n_observations += 1
        n_splits_seen.add(split_idx)

    if n_observations == 0:
        return {"feature_frequency": pd.Series(dtype=float),
                "stable_features": [], "n_splits": 0}

    freq = pd.Series(feature_counts).sort_values(ascending=False) / n_observations
    stable = freq[freq > 0.80].index.tolist()

    return {
        "feature_frequency": freq,
        "stable_features": stable,
        "n_splits": len(n_splits_seen),
    }


# --- 14. FFD stability ------------------------------------------------------
# Distribution of d* values per FFD column across folds; reveals whether the stationarity threshold is regime-stable.
def compute_ffd_stability(predictions: dict) -> dict:
    """Return ``{d_star_by_column, mean_d_star, std_d_star}`` aggregated across all stored folds.

    FFD is shared across models within a fold and is deterministic given the training fold, so
    the spread in d* across this collection reflects training-fold differences across splits.
    """
    d_stars = {}

    # Collect d* values from every entry's prep_info.
    for key, pred in predictions.items():
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

    # Warn on columns whose stationarity threshold varies a lot across time.
    for col, std in result["std_d_star"].items():
        if std > 0.1:
            logger.warning(
                "FFD: column '%s' has heterogeneous d* (std=%.3f). "
                "Stationarity structure varies across time periods.", col, std,
            )

    return result


# --- 15. Top-level orchestration -------------------------------------------
# Public entry point: take the pipeline result dict, return everything the notebook needs.
def analyze_results(cpcv_results: dict) -> dict:
    """Produce the complete post-CPCV analysis from a ``run_cpcv_pipeline`` result."""
    predictions = cpcv_results["predictions"]
    path_map = cpcv_results["path_map"]
    n_paths = cpcv_results["n_paths"]
    models = cpcv_results["models"]
    n_seeds = cpcv_results["n_seeds"]
    n_trials = len(models)

    # Derive the event index and group bounds once; ``stitch_paths`` reuses them per model.
    n_groups = cpcv_results.get("n_groups", 6)
    event_index = _derive_event_index(predictions)
    group_bounds = _compute_group_bounds(len(event_index), n_groups)

    all_summaries = []
    path_sharpes_matrix = np.zeros((len(models), n_paths))

    for model_idx, model_name in enumerate(models):
        print(f"\n  Evaluating: {model_name}...")

        # Per-split metrics averaged across seeds before aggregation.
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

        # Path stitching: produces the seed-averaged path-level financial metrics.
        path_results = stitch_paths(
            predictions, path_map, n_paths, model_name,
            seed=0, n_seeds=n_seeds,
            event_index=event_index, group_bounds=group_bounds,
        )

        path_performances = []
        for path_id in range(n_paths):
            perf = path_results[path_id]["performance"]
            path_performances.append(perf)
            path_sharpes_matrix[model_idx, path_id] = perf["annualized_sharpe"]

        summary = compute_model_summary(
            model_name, path_performances, split_metrics, n_trials
        )
        all_summaries.append(summary)

    # Ranked comparison table (also prints to the notebook).
    comparison_df = compare_models(all_summaries)

    # PBO with a verbal verdict for the cell output.
    pbo = compute_pbo(path_sharpes_matrix)
    print(f"\n  Probability of Backtest Overfitting (PBO): {pbo:.4f}")
    if pbo < 0.3:
        print("  → PBO < 0.3: model selection appears robust.")
    elif pbo > 0.5:
        print("  → PBO > 0.5: in-sample winner tends to underperform OOS. Caution.")
    else:
        print("  → PBO in [0.3, 0.5]: moderate overfitting risk.")

    feature_stability = compute_feature_stability(predictions, models)

    # Pairwise DeLong tests + formatted print block.
    auc_significance = compute_auc_significance(
        predictions, models, cpcv_results["n_splits"], n_seeds=n_seeds,
    )
    if len(auc_significance):
        print("\n" + "=" * 80)
        print(
            "AUC Significance Tests (DeLong, pooled across splits, "
            f"averaged across {n_seeds} seed{'s' if n_seeds != 1 else ''})"
        )
        print("=" * 80)
        print(auc_significance.to_string(index=False, float_format="{:.4f}".format))
        n_sig = auc_significance["significant"].sum()
        n_total = len(auc_significance)
        print(f"\n  {n_sig}/{n_total} pairs significantly different (α=0.05)")
        print("=" * 80)

    ffd_stability = compute_ffd_stability(predictions)

    # Re-stitch all models for the returned ``path_results`` so downstream plotters have everything.
    all_path_results = {}
    for m in models:
        all_path_results[m] = stitch_paths(
            predictions, path_map, n_paths, m,
            seed=0, n_seeds=n_seeds,
            event_index=event_index, group_bounds=group_bounds,
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


# --- 16. Statistical-robustness display helpers ----------------------------
# Three formatted-print helpers consumed by the notebook's robustness cell; factored out so
# the notebook reads as three calls instead of forty lines of print formatting.

# FFD d* stability table: per-column mean/std/min/max across folds.
def render_ffd_stability(analysis: dict) -> None:
    """Print the FFD d* stability table (mean, std, min, max per column)."""
    ffd_stab = analysis["ffd_stability"]
    print("FFD d* across CPCV training folds:")
    print("-" * 60)
    print(f"  {'Column':<15s}  {'Mean d*':>9s}  {'Std d*':>9s}  "
          f"{'Min':>7s}  {'Max':>7s}")
    print("-" * 60)
    for col in ffd_stab["d_star_by_column"]:
        vals = np.asarray(ffd_stab["d_star_by_column"][col])
        print(f"  {col:<15s}  "
              f"{ffd_stab['mean_d_star'][col]:>9.3f}  "
              f"{ffd_stab['std_d_star'][col]:>9.3f}  "
              f"{vals.min():>7.3f}  "
              f"{vals.max():>7.3f}")
    print("-" * 60)


# DSR-per-model table sorted descending; ✓pass/fail against the threshold.
def render_deflated_sharpe_table(analysis: dict, threshold: float = 0.95) -> None:
    """Print the per-model DSR table sorted descending; marks ``> threshold`` as a pass.

    DSR penalises Sharpe for non-normality of returns and for multiple-comparisons selection
    bias produced by considering N candidate models. A DSR above ``threshold`` (default 0.95)
    indicates a result that survives the AFML Ch. 14 correction.
    """
    print("Deflated Sharpe Ratios:")
    print("-" * 55)
    print(f"  {'Model':<15s}  {'DSR':>10s}  "
          f"{'>' + format(threshold, '.2f') + '?':>15s}")
    print("-" * 55)

    # NaN DSR (e.g. buy-and-hold) renders as 'n/a' and sorts last.
    sorted_summaries = sorted(
        analysis["all_summaries"],
        key=lambda s: s["dsr"] if not np.isnan(s["dsr"]) else -np.inf,
        reverse=True,
    )
    for s in sorted_summaries:
        if np.isnan(s["dsr"]):
            print(f"  {s['model_name']:<15s}  {'n/a':>10s}  "
                  f"{'(undefined)':>15s}")
        else:
            verdict = "✓ pass" if s["dsr"] > threshold else "fail"
            print(f"  {s['model_name']:<15s}  {s['dsr']:>10.4f}  "
                  f"{verdict:>15s}")
    print("-" * 55)
    print(
        f"  Threshold {threshold}: result survives multiple-testing "
        f"correction (AFML Ch. 14)."
    )


# Baseline PBO + leave-one-out sensitivity table; identifies which model contributes most to PBO.
def render_pbo_summary(
    analysis: dict, results: dict, drop_padding_blank_lines: bool = True,
) -> None:
    """Print baseline PBO plus a leave-one-out PBO sensitivity table.

    Leave-one-out PBO recomputes the metric after excluding each model in turn; a large
    negative Δ vs baseline means that model's exclusion sharply reduces overfitting (i.e.
    the model contributes PBO when present).
    """
    models_list = results["models"]
    path_sharpes = analysis["path_sharpes"]
    pbo_all = analysis["pbo"]

    if not drop_padding_blank_lines:
        print()
    print(f"Baseline PBO (all {len(models_list)} models): {pbo_all:.3f}")
    print()
    print("Leave-one-out PBO:")
    print("-" * 55)
    print(f"  {'Model excluded':<15s}  "
          f"{'PBO (n=' + str(len(models_list) - 1) + ')':>10s}  "
          f"{'Δ vs baseline':>15s}")
    print("-" * 55)

    # Recompute PBO excluding each model in turn.
    loo_results = []
    for i, m in enumerate(models_list):
        mask = np.arange(len(models_list)) != i
        sub_sharpes = path_sharpes[mask, :]
        pbo_loo = compute_pbo(sub_sharpes)
        delta = pbo_loo - pbo_all
        loo_results.append((m, pbo_loo, delta))

    # Sort ascending: the model whose exclusion most reduces PBO appears first.
    loo_results.sort(key=lambda r: r[1])

    for m, pbo_loo, delta in loo_results:
        print(f"  {m:<15s}  {pbo_loo:>10.3f}  {delta:>+15.3f}")

    print("-" * 55)