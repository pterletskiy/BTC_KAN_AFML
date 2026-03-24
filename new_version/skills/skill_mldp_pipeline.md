---
name: skill_MLdP_pipeline
description: >
  Governs the complete López de Prado (AFML) preprocessing and evaluation
  pipeline for financial time series machine learning. Use this skill whenever
  working on labeling (Triple-Barrier Method), cross-validation design
  (Purged K-Fold, CPCV), sample weights, feature importance analysis,
  stationarity transforms (FFD), leakage-safe evaluation, backtesting
  statistics (DSR, PSR), or bet sizing from predicted probabilities.
  Apply to any step in: data loading, feature engineering, labeling, CV
  splitting, feature importance, backtesting, or strategy evaluation.
  Covers BTC daily time-bar pipelines with KAN interpretability constraints.
---

# López de Prado Preprocessing & Evaluation Skill (AFML)

## 0. Scope and Fixed Constraints

This skill governs everything **before and around** the model:
data structure, feature transforms, labeling, cross-validation, sample weights,
feature importance, and backtesting evaluation.

**Fixed constraints for this project:**
- **Data frequency:** daily time bars only (yfinance / CoinMetrics).
- **No PCA / no orthogonalization of features.** Features must remain
  interpretable in their original form to allow symbolic math extraction
  from the KAN at the end of the pipeline. Any rule that references PCA
  is **suspended**.
- Modeling architecture and KAN-specific logic belong to `skill_kan_modeling.md`.

---

## 1. Chronological Processing Rule (Non-Negotiable)

All transformations must preserve temporal order.

- No operation may use future information relative to the observation
  being processed.
- All rolling or expanding statistics must use **only past data** and
  must be **explicitly lagged by at least one period** when the result
  is used as a predictor for the same timestamp.
- The lag must be documented in the feature definition record.

---

## 2. Continuous Feature Transformations (Pre-CV, Pre-Label)

Apply these transformations to **raw continuous predictors only**,
before labeling and before any cross-validation split.

### 2.1 Log Transformations

- Apply only to strictly positive, scale-skewed variables: price, volume,
  market cap, on-chain value flows.
- Never apply to bounded oscillators, signed indicators, binary flags,
  or already-standardized series.
- If a feature can be zero or negative, document a safe minimum shift
  before applying; or do not log-transform.

### 2.2 Fractional Differentiation (FFD) — Default Stationarity Transform

Use Fixed-Width Window Fractional Differentiation as the default
stationarity transform for price-like continuous series.

**Fitting protocol:**
1. Search over `d ∈ [0.0, 1.0]` in steps of 0.05 or finer.
2. Fit `d*` on the training fold only, using the ADF test at 5% significance.
3. Select the minimum `d*` that achieves stationarity.
4. Apply the same `d*` (and same window width `τ`) to the corresponding
   validation and test folds — no re-fitting outside the training fold.
5. Drop or mask the warm-up region created by the fixed-width filter.
6. Record `d*`, `τ`, and the ADF statistic at that `d*` in the
   preprocessing log.

**Never** apply integer differencing (`d=1`, i.e., standard returns) if a
smaller fractional order already achieves stationarity. The memory lost
by over-differencing directly reduces predictive signal.

**Correlation check:** after FFD, compute the Pearson correlation between
the FFD series and the original level series. Target `|ρ| > 0.95` at the
chosen `d*`. If correlation collapses below 0.90, reduce `d` or investigate
whether the series is truly non-stationary.

### 2.3 SADF as a Predictive Feature (Not Only a Diagnostic)

The Supremum Augmented Dickey-Fuller statistic measures explosiveness
in a rolling, time-adaptive way. On daily BTC data it functions as
a real-time bubble / regime indicator.

- Compute a rolling SADF series over a backward-expanding window.
- Lag by one day before using as a predictor.
- This feature is optional but strongly recommended.
- Never use raw SADF as a stationarity diagnostic on the full sample
  before splitting — that leaks regime information.

---

## 3. Labeling — Triple-Barrier Method (TBM)

Never use fixed-time-horizon labeling as the primary label.

### 3.1 Dynamic Volatility Threshold

- Compute an exponentially weighted daily volatility estimate
  (e.g., `span=20` days) using **only past observations**.
- Use this volatility to set the horizontal barrier widths as a
  multiple `pt_sl * vol`.
- Recommended default: symmetric barriers `pt_sl = [1, 1]` for
  side-learning; asymmetric `pt_sl = [2, 1]` for meta-labeling.

### 3.2 Event Selection (CUSUM Filter)

- Sample events using the symmetric CUSUM filter on daily returns,
  with threshold `h = σ_ewm` (the same rolling volatility estimate).
- This prevents over-sampling during quiet regimes and ensures that
  training examples represent actionable signal episodes.

### 3.3 Label Assignment

For each event at `t0`, inspect the price path between `t0` and
`t0 + max_holding_days` (vertical barrier):

| First barrier touched | Label |
|---|---|
| Upper horizontal | **+1** |
| Lower horizontal | **-1** |
| Vertical (timeout) | **0** |

- Record `t0` (event start) and `t1` (first barrier touch) for every event.
- The `t1` object is required by every downstream step: sample weights,
  purging, and embargo all depend on it.
- Neutral labels (`0`) may be dropped or retained. Document the decision.
  When using meta-labeling, neutral labels from the primary model
  become **0-class** (do not trade) — keep them.

### 3.4 Class Balance Reporting

After labeling, always report:
- Count and fraction of `+1`, `-1`, `0` labels.
- Average `t1 - t0` holding period in days.
- Fraction of events that hit the vertical barrier.

If `>60%` of labels are neutral, reconsider barrier widths or volatility
multipliers before proceeding.

---

## 4. Sample Weights

Standard ML assumes IID observations. Financial labels are not IID:
outcomes overlap in time. Correct this with two weight components.

### 4.1 Uniqueness-Based Weights

- For each event `i`, compute the average uniqueness
  `ū_i = mean(1 / c_t)` over `[t0_i, t1_i]`, where `c_t` is
  the number of concurrent labels at time `t`.
- Use `ū_i` as the base sample weight. Overlapping labels get
  down-weighted; rare non-overlapping labels get up-weighted.

### 4.2 Return-Attribution Multiplier (Optional)

- Multiply `ū_i` by the absolute return realized at `t1_i`
  (normalized so weights sum to the number of events).
- This increases the influence of large, unambiguous outcomes.

### 4.3 Time-Decay Factor (Optional)

- Apply a linear decay `d(x) = max(0, a + bx)` over cumulative
  uniqueness to reduce the weight of older events.
- Parameter `c ∈ (-1, 1]`: `c=1` = no decay; `c=0` = linear decay
  to zero; `c < 0` = oldest `|c|` fraction gets zero weight.
- Fit decay parameters on the training fold only.

### 4.4 Class Weights

- For binary classifiers (meta-labeling): set `class_weight='balanced'`
  to correct for the imbalance between true positives (1) and
  false positives (0).
- For three-class primary models: report whether balanced weights
  improve or hurt validation F1 before committing.

---

## 5. Cross-Validation Design

### 5.1 Permitted Methods Only

| Method | Permitted |
|---|---|
| Random K-Fold | ❌ Forbidden |
| Standard `TimeSeriesSplit` | ❌ Forbidden (no purge/embargo) |
| Walk-Forward (WF) | ⚠️ Permitted only if DSR is reported |
| Purged K-Fold | ✅ Default for model development |
| Combinatorial Purged CV (CPCV) | ✅ Required for final backtest paths |

### 5.2 Purging Rule

Remove from the training fold any observation `i` whose label window
`[t0_i, t1_i]` overlaps with any timestamp in the test fold.
Overlap condition: `t0_i ≤ t_test_end AND t1_i ≥ t_test_start`.

### 5.3 Embargo Rule

After each test fold, remove a buffer of `h ≈ 0.01 * T` observations
immediately following the test end. Prevents leakage from serial
correlation in features that persist across the boundary.

### 5.4 CPCV for Final Backtest Distribution

Use CPCV with `N` groups and `k=2` to generate `φ = N-1` backtest paths.

- Each path gives one Sharpe ratio estimate.
- Report the **distribution** of Sharpe ratios across paths,
  not a single point estimate.
- Recommended: `N = 6` to `N = 10` for ~2–5 years of daily data.

---

## 6. Inner-CV Fit Wall

Every transformation that learns from data must be fitted **only inside
the CV loop, on the purged and embargoed training fold**.

| Transform | Where to Fit |
|---|---|
| `RobustScaler` / `StandardScaler` | Training fold only |
| Feature selection (variance, correlation) | Training fold only |
| FFD `d*` search | Training fold only |
| CUSUM threshold `h` | Training fold only |
| Volatility estimate for barriers | Training fold only (use trailing window) |
| `d*` application to validation/test | Apply fitted `d*` from train fold |

**Never** apply a fitted transformer to the full dataset before splitting.

---

## 7. Feature Importance (Required Before Backtesting)

> "Backtesting is not a research tool. Feature importance is."
> — López de Prado, AFML Ch. 8

Feature importance analysis **must** be completed before any backtest
is run. This is not optional.

### 7.1 Mean Decrease Impurity (MDI)

- Use a Random Forest or Bagged Decision Tree auxiliary model
  (not the KAN) solely for importance estimation.
- Set `max_features=1` to prevent masking effects.
- Replace zero-importance entries with `NaN` before averaging.
- MDI is in-sample and fast; use it for initial screening.

### 7.2 Mean Decrease Accuracy (MDA)

- Permute each feature column one at a time; measure the OOS
  performance drop on purged K-Fold.
- Score with `neg_log_loss` for probability models, `F1` for
  meta-labeling models.
- MDA can conclude that **all features are unimportant** — if that
  happens, do not proceed to backtesting. Revisit feature engineering.

### 7.3 Single Feature Importance (SFI)

- Train the auxiliary classifier on each feature in isolation.
- SFI is immune to substitution effects; use it to confirm
  MDI/MDA findings.
- Features that rank high in SFI but low in MDI are suppressed
  by correlated substitutes — investigate why.

### 7.4 KAN Interpretability Alignment Check

After importance ranking, verify that the features the KAN
assigns large activation magnitudes to are also the features
flagged as important by MDI/MDA/SFI. Disagreement is a warning
sign of either overfitting or a spurious correlation.

### 7.5 Feature Importance Output

Report for each feature:
- MDI mean and standard deviation across trees.
- MDA score (absolute and relative to baseline).
- SFI score.
- Rank across all three methods.

Drop features that rank in the bottom quartile across **all three**
methods before final model training.

---

## 8. Meta-Labeling (Recommended)

Meta-labeling decouples side prediction (primary model) from
size/confidence filtering (secondary model).

**When to apply:**
- When the primary KAN achieves acceptable recall but low precision.
- When bet sizing based on raw predicted probabilities is unreliable.

**Protocol:**
1. Train primary KAN to predict direction (`+1` / `-1`) with
   high recall. Accept lower precision.
2. Take the primary model's positive predictions as events.
3. Generate meta-labels: `1` if the primary prediction was a
   true positive, `0` if false positive.
4. Train a secondary calibrated classifier (e.g., logistic regression
   or a small RF) on the meta-labels using `F1` scoring.
5. The final position size = (side from primary) × (probability
   from secondary).
6. Score the meta-model with `F1`, not accuracy. Accuracy is
   misleading when negatives dominate.

---

## 9. Bet Sizing from Predicted Probabilities

Do not use binary threshold predictions for position sizing.
Convert calibrated probabilities into continuous bet sizes.

For a two-class prediction where `p = P(label = +1)`:

```
z = (p - 0.5) / sqrt(p * (1 - p) / n_observations)
m = 2 * Φ(z) - 1       # m ∈ (-1, +1)
```

Where `Φ` is the standard Normal CDF.

- `m > 0` → long position, scaled by `|m|`.
- `m < 0` → short position (if applicable).
- `m ≈ 0` → no position (near-zero confidence).

Average active bets if multiple overlapping signals are active at the
same time (see AFML Ch. 10.4). Apply size discretization to avoid
excessive micro-trading from small changes in `m`.

---

## 10. Backtesting Statistics (Required Outputs)

### 10.1 Mandatory Statistics

Every backtest must report:

| Statistic | Notes |
|---|---|
| Annualized Sharpe Ratio (SR) | Non-annualized SR × √252 |
| Probabilistic Sharpe Ratio (PSR) | PSR > 0.95 target at 5% sig. |
| **Deflated Sharpe Ratio (DSR)** | **Required. See 10.2.** |
| Max Drawdown (95th percentile) | From CPCV path distribution |
| Time Under Water (95th percentile) | From CPCV path distribution |
| Hit Ratio | Fraction of bets with positive PnL |
| Average return: hits vs. misses | Asymmetry diagnostic |
| HHI on positive returns | Concentration diagnostic |
| HHI on negative returns | Concentration diagnostic |

### 10.2 Deflated Sharpe Ratio (DSR) — Non-Negotiable

The DSR corrects SR for the number of configurations tested.

```
SR* = sqrt(Var(SR_trials)) * (
    (1 - γ) * Φ^{-1}(1 - 1/N) +
    γ * Φ^{-1}(1 - 1/(N*e))
)
DSR = PSR(SR* as benchmark)
```

**Requirements:**
- Record every model configuration and hyperparameter combination
  tested. This count is `N` in the DSR formula.
- Record the variance of SR estimates across those `N` trials.
- A backtest result is not reportable without its `N` and `Var(SR)`.
- DSR must exceed 0.95 for a result to be considered non-spurious.

### 10.3 CPCV Sharpe Distribution

Report not a single SR but the **distribution** of SR across CPCV paths:
- Mean SR across paths.
- Standard deviation of SR across paths.
- Minimum and maximum path SR.
- Fraction of paths with SR > 0.

A strategy where >30% of CPCV paths have SR < 0 is unreliable
regardless of the mean SR.

---

## 11. Leakage Audit Checklist

Before any model training, confirm all of the following:

- [ ] All rolling indicators lagged by at least one period.
- [ ] No feature uses the prediction-day value to predict the same day.
- [ ] `d*` for FFD fitted on training fold only.
- [ ] CUSUM threshold `h` computed from training data only.
- [ ] Scaler fitted on purged + embargoed training fold only.
- [ ] Feature selection (if any) performed inside CV loop.
- [ ] No label-derived quantity used as a predictor.
- [ ] `t1` object verified as strictly forward-looking from `t0`.
- [ ] Class balance and neutral label fraction reported.
- [ ] SADF series lagged before use as predictor.

---

## 12. Forbidden Actions

| Action | Reason |
|---|---|
| Preprocess full dataset before splitting | Introduces look-ahead leakage |
| Random K-Fold or shuffled time series CV | Destroys temporal ordering |
| Fit any scaler / selector outside CV loop | Distribution leakage |
| Use integer differencing when FFD suffices | Memory destruction |
| Report SR without recording number of trials | DSR cannot be computed |
| Backtest before feature importance analysis | Overfitting by backtest cycle |
| Apply PCA to features | Destroys KAN symbolic interpretability |
| Use fixed-time-horizon labeling as primary | Path dependency ignored |
| Report single-path SR from WF as final result | High variance, overfittable |

---

## 13. Output Artefacts

The pipeline must produce the following saved artefacts:

| File | Content |
|---|---|
| `data/interim/*.parquet` | FFD-transformed continuous features, pre-CV |
| `data/processed/fold_*.parquet` | Purged, embargoed, scaled folds |
| `data/processed/labels.parquet` | `t0`, `t1`, label, sample weight per event |
| `models/feature_importance.json` | MDI, MDA, SFI scores per feature |
| `models/preprocessing_log.json` | `d*`, `τ`, ADF stats, CUSUM `h` per fold |
| `models/trial_registry.json` | All tested configs and their SR for DSR computation |
| `models/backtest_stats.json` | Full backtest statistics per CPCV path |