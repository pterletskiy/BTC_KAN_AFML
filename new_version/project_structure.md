# BTC Daily Direction Prediction Using KANs Within the AFML Framework
## Project Structure & Pipeline Architecture

**Student:** Petr Terletskiy (l63023)
**Programme:** Masters in Mathematical Finance, ISEG (2024/26)

---

## Overview

The project predicts Bitcoin daily price direction using Kolmogorov–Arnold Networks (KANs), benchmarked against AR Logistic, Logistic Regression, Random Forest, XGBoost, and LSTM models, evaluated under López de Prado's *Advances in Financial Machine Learning* (2018) framework. The pipeline is organized into three phases with strict leakage boundaries between them.

**Data:** BTC-USD daily OHLCV, November 2014 – May 2026 (~4,200 bars). Raw data starts in November 2014 to provide the 252-day lookback required by the longest-warmup features (Hurst, EMA 50/200 ratio, SADF). The CUSUM event filter is applied from August 8, 2015 onward — the date of Ethereum's Frontier launch and the first day with valid ETH/USD price data needed for the `eth_btc_ratio` feature. This buffer-and-truncate structure ensures that all engineered features have completed their warmup windows by the start of model evaluation, and that all cross-asset crypto features have full data availability across every CPCV fold.
**Features:** 73 columns total, all eligible for MDA feature selection (25 technical, 9 mathematical/AFML Part 4, 29 external: 20 macro, 1 crypto-macro, 8 on-chain from CoinMetrics, 10 autoregressive lags). The 10 lag features (`log_returns_lag1` … `log_returns_lag30`) compete with engineered features for the MDA top-k cap. AR Logistic continues to consume the 10 lag columns by name from the pre-MDA matrix as its pure-autoregressive baseline, independently of MDA's choices. The macro group includes 14-day and 30-day return horizons for the seven faster-moving assets (S&P 500, Nasdaq, gold, silver, copper, oil, natural gas), with MDA deciding per fold which horizon is more informative; slow-moving variables (DXY, yields, yield curves) keep the 30-day horizon only.
**Calendar:** All rolling windows use the BTC trading calendar (7-day week, 30-day month, 90-day quarter, 180-day semester, 365-day year).
**Evaluation:** CPCV (N=8, k=2) producing 28 splits and 7 backtest paths, with Deflated Sharpe Ratio, Probability of Backtest Overfitting, and DeLong pairwise AUC significance tests.
**Tuning:** Nested Optuna TPE + Purged K-Fold, per-split, inside each CPCV training fold (AFML Ch. 7 compliant).
**Interpretability contribution:** Symbolic formula extraction from KAN via PyKAN re-training, pruning, and symbolification.

---

## Repository Structure

```
project/
├── notebooks/
│   └── main.ipynb                          ← central orchestrator
├── src/
│   ├── __init__.py
│   │
│   ├── pre_cpcv/                           ← PHASE 1: Data Preparation
│   │   ├── __init__.py
│   │   ├── data_loader.py                  ← BTC OHLCV retrieval + validation
│   │   ├── labeling.py                     ← volatility, CUSUM, triple-barrier, rare-label drop
│   │   ├── sample_weights.py               ← concurrent labels, uniqueness, return attribution, time decay
│   │   ├── features.py                     ← TA + mathematical features, log transforms
│   │   ├── external_features.py            ← macro, crypto-macro, on-chain features
│   │   ├── alignment.py                    ← index intersection, validation assertions
│   │   └── pre_cpcv_plots.py               ← labelling diagnostics, feature EDA, ADF stationarity test
│   │
│   ├── cpcv/                               ← PHASE 2: Cross-Validation & Training
│   │   ├── __init__.py
│   │   ├── cv.py                           ← CPCV splits, purging, embargo, path matrix
│   │   ├── cpcv_plots.py                   ← group partitioning, train/test timelines, purging detail, leakage audit
│   │   ├── preprocessing.py                ← per-fold FFD, RobustScaler, multi-model MDA feature selection
│   │   ├── tuning.py                       ← Optuna TPE + Purged K-Fold nested per-split tuning
│   │   ├── calibration.py                  ← Platt scaling, vector scaling (default for PyTorch models)
│   │   ├── pipeline.py                     ← full CPCV loop orchestration (accepts n_trials parameter)
│   │   └── models/
│   │       ├── __init__.py                 ← model registry & factory
│   │       ├── base.py                     ← abstract BaseModel interface
│   │       ├── benchmarks.py               ← AR Logistic, Logistic Regression
│   │       ├── tree_models.py              ← Random Forest, XGBoost
│   │       ├── lstm_model.py               ← LSTM (PyTorch)
│   │       └── kan_model.py                ← KAN (efficient-kan, PyTorch)
│   │
│   └── post_cpcv/                          ← PHASE 3: Evaluation & Interpretability
│       ├── __init__.py
│       ├── evaluation.py                   ← metrics, path stitching, DSR, PBO, DeLong AUC, comparison
│       ├── diagnostics.py                  ← interactive inspection helpers (calibration audit, regime concentration, bet-size, reliability curves)
│       └── symbolic_extraction.py          ← PyKAN re-train, prune, symbolify, formula extraction
│
└── cache/                                  ← Parquet cache for expensive features
    ├── math_features.parquet               ← SADF, SMT, entropy, Hurst, etc.
    ├── external_features.parquet           ← macro + crypto-macro + on-chain
    └── onchain_raw.parquet                 ← raw CoinMetrics data
```

---

## Phase 1 — Pre-CPCV (Data Preparation)

Everything in this phase runs once, before any model training. It produces four aligned objects (X, y, w, t1) that enter the CPCV loop. No stateful transformation that could leak future information occurs here; those are deferred to Phase 2.

### Notebook sections: 1 (Dependencies) → 2 (Data Engineering)

---

### Step 0 — Data Loading (`data_loader.py`)

**Single function:** `load_btc_daily(ticker, start, end) → pd.DataFrame`

Fetches daily OHLCV from yfinance and returns a validated DataFrame with columns `['Open', 'High', 'Low', 'Close', 'Volume']` indexed by a tz-naive DatetimeIndex sorted ascending.

**Validation pipeline (runs automatically):**

| Check | Action on failure |
|-------|-------------------|
| Empty download | Raises `ValueError` |
| MultiIndex columns (yfinance ≥ 0.2.31) | Flattens automatically via `droplevel` |
| Duplicate dates | Raises `ValueError` |
| Calendar gaps ≤ 3 days | Forward-fills + logs warning |
| Calendar gaps > 3 days | Raises `ValueError` (structural data problem) |
| High < max(Open, Close) | Logs warning, keeps row |
| Low > min(Open, Close) | Logs warning, keeps row |
| Volume < 0 | Logs warning, keeps row |
| NaN Close | Drops row + logs warning |

**Parameters used in notebook:** `ticker="BTC-USD"`, `start="2014-11-01"`, `end="2026-05-01"`

**Internal helpers:** `_fill_small_gaps(df)` handles gap detection and forward-filling with the 3-day limit. `_check_ohlcv_consistency(df)` validates OHLCV relationships row-by-row.

---

### Step 1 — Daily Volatility (`labeling.py`)

**Function:** `compute_daily_volatility(close, span=50) → pd.Series`

Implements AFML Snippet 3.1. Computes `close.pct_change()` as log returns, then applies `ewm(span=50).std()`. The span of 50 (vs De Prado's 100 for equities) is calibrated for BTC's faster regime transitions: a smaller span reacts to regime changes more quickly, which matters for crypto's 24/7 trading and higher realized volatility. Returns a Series with NaN for the first ~50 rows (EWMA warm-up), indexed identically to `close`.

This series serves as the dynamic threshold for both CUSUM (Step 2) and triple-barrier widths (Step 3).

---

### Step 2 — CUSUM Filter (`labeling.py`)

**Function:** `cusum_filter(log_returns, threshold) → pd.DatetimeIndex`

Implements AFML Snippet 2.4 (symmetric CUSUM). Maintains two accumulators:
- `s_pos = max(0, s_pos + return)` — tracks upside runs
- `s_neg = min(0, s_neg + return)` — tracks downside runs

Fires an event and resets when `s_pos >= h` or `s_neg <= -h`. The zero floor ensures only **sustained directional moves** trigger events (hovering around a level keeps resetting the accumulator). This is a structural break detector, not a volatility filter: choppy sideways action produces few events, while small-but-persistent drifts can produce many.

Returns a DatetimeIndex of event timestamps, reducing ~4,200 bars to ~1,000 informative events.

**Parameters in notebook:** `h = 1.0 × daily_vol.mean()`. The multiplier was tightened from an earlier 1.5× choice after empirical sweeps: `0.5×` produces too many noisy events, `3.0×` reduces to ~200 (too few for ML), `1.0×` yields a workable event count with roughly balanced classes.

**CUSUM start-date truncation.** After the filter runs on the full raw return series, the notebook truncates the event index to a configured `CUSUM_START_DATE` (currently `"2015-08-08"`, the date of Ethereum's Frontier launch and the first day with valid ETH/USD price data from CoinMetrics). Events that fired before this date are dropped. This is structurally important: the EWMA daily-volatility estimator with `span=50` needs roughly 150 days of warmup to converge, which is satisfied for any event after February 2015 given the November 2014 raw-data start. More importantly, `eth_btc_ratio` (one of the cross-asset crypto features) cannot be computed before ETH started trading. Truncating CUSUM to start at the ETH availability date ensures every labelled event has full feature coverage across every CPCV fold and prevents the asymmetric data-availability problem that otherwise produces fully-NaN early test partitions under N=8. The CUSUM accumulators `s_pos` and `s_neg` themselves are computed on the full raw series, so events that survive this truncation reflect the dynamic state of the cumulative drift over the entire pre-event history. Empirically, the mean daily volatility over the truncated window matches the full-series mean to four decimal places, confirming the EWMA was fully converged by the truncation date.

---

### Step 3 — Triple-Barrier Labeling (`labeling.py`)

**Produces:** DataFrame `bins` with columns `['ret', 'bin', 't1']` where `ret` is the return at barrier touch, `bin` is the class label {-1, 0, +1}, and `t1` is the timestamp when the label was resolved (first barrier touch). The `t1` column is critical for downstream purging in CPCV.

**Functions:**

#### `get_vertical_barriers(close, t_events, num_days) → pd.Series`
For each event timestamp, finds the close-index timestamp `num_days` calendar days ahead using `searchsorted`. Returns NaN for events near the end of the data where the horizon exceeds available bars.

#### `triple_barrier_labels(close, t_events, trgt, pt_sl, num_days, min_return=0.0) → pd.DataFrame`
Implements AFML Snippets 3.2 + 3.4 + 3.5. For each CUSUM event at time t₀:
1. Sets upper barrier at `close[t₀] × (1 + pt_sl[0] × trgt[t₀])`
2. Sets lower barrier at `close[t₀] × (1 − pt_sl[1] × trgt[t₀])`
3. Sets vertical barrier at t₀ + num_days
4. Walks forward through the price path, records first barrier touch
5. Labels: +1 (upper), −1 (lower), sign(return) at vertical (or 0 if |return| < min_return)

Aligns events to non-NaN volatility timestamps. Skips events with NaN vertical barriers or path length < 2.

**Parameters in notebook:** `pt_sl=(1.5, 1.5)`, `num_days=10`, `min_return=0.02`.

**Parameter choice rationale:**
- Symmetric `pt_sl=(1.5, 1.5)` avoids imposing any prior directional bias. With σ ≈ 3% daily, barriers sit at roughly ±4.5% from entry.
- `num_days=10` gives horizontal barriers meaningful time to trigger without letting labels go stale. Observed mean holding period is ~5.1 days, with the vertical (time) barrier hit on only 19.9% of events — horizontal barriers (profit-take or stop-loss) close ~80% of positions before the time barrier triggers.
- `min_return=0.02` collapses very small vertical-barrier returns into class 0 (later dropped as a rare label). Without this floor, the symmetric pt_sl already produces few class-0 events; the floor pushes the few remaining near-zero vertical-barrier returns into the rare-label drop bucket, leaving cleaner binary {-1, +1} labels.

---

### Step 4 — Drop Rare Labels (`labeling.py`)

**Function:** `drop_rare_labels(bins, min_pct=0.05) → pd.DataFrame`

Implements AFML Snippet 3.8. Counts class frequencies, drops all rows belonging to classes below the threshold. **Notebook calls with `min_pct=0.085`** (raised from the 0.05 default to be more aggressive about removing residual class-0 events). Combined with the symmetric pt_sl and `min_return=0.02`, class 0 is eliminated, producing binary labels {-1, +1}.

**Orchestration:** `run_labeling_pipeline(close, ...) → pd.DataFrame` chains Steps 1–4 into a single call. All parameters have defaults matching De Prado's recommendations, overridable at call time from the notebook.

---

### Step 5 — Sample Weights (`sample_weights.py`)

**Produces:** A pd.Series of per-sample weights that account for overlapping triple-barrier labels (AFML Chapter 4).

**Functions (in dependency order):**

#### `get_num_concurrent_labels(bins_t1, num_bars_index) → pd.Series`
Implements AFML Snippet 4.1. For each bar in the full daily index, counts how many labels are "alive" (their [t₀, t₁] span includes that bar). Iterates through all labels, incrementing a counter at each bar in the label's span.

#### `get_average_uniqueness(bins_t1, concurrent_labels) → pd.Series`
Implements AFML Snippet 4.2. For each label i spanning [t₀, t₁], computes `ū_i = mean(1 / c_t)` across all bars in that interval. Labels with no overlap get uniqueness ≈ 1.0; labels overlapping with many others approach 0.

#### `get_return_attribution_weights(bins, avg_uniqueness) → pd.Series`
Implements AFML Snippet 4.10. Computes `w_i = |ret_i| × ū_i`, then normalizes so `sum(weights) == len(weights)` (mean weight ≈ 1, sklearn-compatible). Falls back to uniform weights if all return-attribution weights are zero.

#### `apply_time_decay(weights, oldest_weight=1.0) → pd.Series`
Implements AFML Snippet 4.11. Applies linear decay via `np.linspace(oldest_weight, 1.0, n)` multiplied element-wise with the weights. Re-normalizes after decay to preserve `sum == len`. When oldest_weight=1.0, returns unchanged weights.

#### `compute_sample_weights(bins, num_bars_index, time_decay_factor=1.0, weight_cap_quantile=0.99) → pd.Series`
Orchestration function. Chains: concurrent labels → uniqueness → return attribution → time decay → quantile capping. The `weight_cap_quantile` parameter clips extreme weights at the specified percentile to prevent single events from dominating training.

**Parameters in notebook:** `time_decay_factor=0.4` (oldest sample's weight is 40% of the newest), `weight_cap_quantile=0.99`.

---

### Step 6 — Feature Engineering (`features.py`, `external_features.py`)

**Produces:** A pd.DataFrame of 73 columns covering every daily bar, broken into four categories:

| Category | Count | Routed to |
|----------|-------|-----------|
| Technical (TA) | 25 | MDA pool |
| Mathematical (AFML Part 4) | 9 | MDA pool |
| External (macro / crypto-macro / on-chain) | 29 | MDA pool |
| Lag (autoregressive) | 10 | MDA pool + AR Logistic (by name, from pre-MDA matrix) |
| **Total** | **73** | **All 73 eligible for MDA; AR Logistic restricts itself to its 10 lag columns** |

**Module-level constants (all feature parameters defined at top of each file):**

```
TA: RSI_PERIOD=14, MACD_FAST=12, MACD_SLOW=26, MACD_SIGNAL=9, BB_PERIOD=20,
    ATR_PERIOD=14, ROLLING_WINDOW=30, EMA_SHORT=20, EMA_MID=50, EMA_LONG=200,
    ROC_PERIOD=14, STOCH_PERIOD=14, STOCH_SMOOTH=3, CCI_PERIOD=14, MFI_PERIOD=14,
    CHAIKIN_FAST=3, CHAIKIN_SLOW=10, YZ_WINDOW=30, VOL_SHORT=7, VOL_MID=30, VOL_LONG=90

Math: SADF_MIN_SL=90, SADF_LAGS=1, ENTROPY_WINDOW=30, LZ_WINDOW=90,
      HURST_WINDOW=180, VR_WINDOW=90, VR_LAG=7, JB_WINDOW=90, GAUSS_ENT_WINDOW=30

Lag: AR_LAGS=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30], LAG_COLUMN_PREFIX="log_returns_lag"

Cache: CACHE_DIR="cache/", MATH_CACHE_FILE="math_features.parquet",
       EXTERNAL_CACHE_FILE="external_features.parquet", ONCHAIN_CACHE_FILE="onchain_raw.parquet"
```

**Note on window choices:** All rolling-window parameters follow the BTC trading calendar (24/7/365). Named technical indicators (RSI, MACD, Bollinger, Stochastic, CCI, MFI, ATR) retain their conventional parameter settings since these are established TA defaults rather than calendar-dependent choices. Annualization uses √365 reflecting BTC's continuous trading.

#### Step 6.a — TA Features (`features.py`)

**Function:** `compute_ta_features(df) → pd.DataFrame` — 25 columns

| # | Feature | Formula | Scale |
|---|---------|---------|-------|
| 1 | `log_returns` | `log(close / close.shift(1))` | Dimensionless |
| 2 | `rsi` | Wilder RSI via EWMA(α=1/14) | Bounded [0, 100] |
| 3 | `macd` | EMA(12) − EMA(26) | Price units |
| 4 | `macd_signal` | EMA(MACD, 9) | Price units |
| 5 | `macd_hist` | MACD − Signal | Price units |
| 6 | `bb_width` | (Upper − Lower) / Middle, bands at ±2σ | Dimensionless ratio |
| 7 | `atr` | EWMA(True Range, 14) | Price units → log-transformed |
| 8 | `obv` | Cumulative sum of Volume × sign(ΔClose) | Volume units → log-transformed |
| 9 | `skewness` | Rolling 30-day skewness of log returns | Dimensionless |
| 10 | `kurtosis` | Rolling 30-day excess kurtosis of log returns | Dimensionless |
| 11 | `realized_vol` | Rolling 30-day std of log returns × √365 | Annualized, dimensionless |
| 12 | `gk_vol` | Garman-Klass: `0.5·ln(H/L)² − (2ln2−1)·ln(C/O)²`, rolling 30-day mean | Dimensionless |
| 13 | `yz_vol` | Yang-Zhang (2000) combining overnight, open-to-close, and Rogers-Satchell components. `k = 0.34/(1.34 + (w+1)/(w−1))` weighting, rolling 30-day | Dimensionless |
| 14 | `ema_ratio_20_50` | EMA(20) / EMA(50), short vs medium trend | Ratio ≈ 1.0 |
| 15 | `ema_ratio_50_200` | EMA(50) / EMA(200), golden/death cross signal | Ratio ≈ 1.0 |
| 16 | `vwma_ratio_20_50` | VWMA(20) / VWMA(50), volume-weighted trend confirmation | Ratio ≈ 1.0 |
| 17 | `roc_14` | `(close / close.shift(14) − 1) × 100`, rate of change | Percentage |
| 18 | `stoch_k` | `100 × (C − LL₁₄) / (HH₁₄ − LL₁₄)`, Stochastic %K | Bounded [0, 100] |
| 19 | `stoch_d` | SMA(%K, 3), smoothed Stochastic | Bounded [0, 100] |
| 20 | `williams_r` | `−100 × (HH₁₄ − C) / (HH₁₄ − LL₁₄)`, Williams %R | Bounded [−100, 0] |
| 21 | `cci_14` | `(TP − SMA(TP)) / (0.015 × MAD(TP))`, Commodity Channel Index | Unbounded, centered ~0 |
| 22 | `chaikin_osc` | EMA(ADL, 3) − EMA(ADL, 10), volume + price momentum | Volume units |
| 23 | `mfi_14` | Volume-weighted RSI using typical price × volume | Bounded [0, 100] |
| 24 | `vol_term_7_30` | `vol_7 / vol_30`, short-term stress signal (week vs month) | Ratio ≈ 1.0 |
| 25 | `vol_term_30_90` | `vol_30 / vol_90`, regime-level signal (month vs quarter) | Ratio ≈ 1.0 |

**Internal helpers:** `_yang_zhang_volatility()` implements the Yang-Zhang (2000) estimator. `_money_flow_index()` implements the volume-weighted RSI calculation.

#### Step 6.b — Mathematical Features (`features.py`)

**Function:** `compute_math_features(df, which="all") → pd.DataFrame` — 9 columns

All features are computationally expensive (O(n²) for SADF/SMT). The function checks for a cached Parquet file before computing; if cache exists with matching date range and requested columns, it loads from cache. Otherwise computes, saves, and returns.

The `which` parameter accepts `"all"` or a list of specific feature names for partial computation.

| # | Feature | AFML Reference | Method | Window | Approx. Time |
|---|---------|---------------|--------|--------|--------------|
| 1 | `shannon_entropy` | Ch. 18.2 | Equal-width-bin (10 bins) plug-in entropy `H = −Σ pᵢ log₂(pᵢ)`. Equal-width bins replaced an earlier quantile-bin scheme that collapsed to zero entropy whenever clustered values made all quantile edges coincide | 30 days | < 1 min |
| 2 | `lz_complexity` | Ch. 18.4 | Binary-encode returns ("1" if > 0), apply Lempel-Ziv-76 algorithm, normalize by `n/log₂(n)` | 90 days | < 1 min |
| 3 | `hurst` | Related | Rescaled Range (R/S) analysis at sub-periods [14, 30, 60, 90], slope of log(R/S) vs log(n) | 180 days | < 1 min |
| 4 | `variance_ratio` | Related | Lo & MacKinlay (1988): `VR(q) = Var(r_q) / (q × Var(r₁))`. VR > 1 = momentum, VR < 1 = mean-reversion | 90 days, lag=7 | < 1 min |
| 5 | `jarque_bera` | Related | `JB = (n/6)(S² + K²/4)` where S = skewness, K = excess kurtosis. Measures departure from normality | 90 days | < 1 min |
| 6 | `negentropy` | Ch. 18.6 | `gaussian_entropy − shannon_entropy`. Gap measures non-Gaussianity of return distribution | 30 days | < 1 min |
| 7 | `sadf` | Ch. 17.4.2 | Supremum ADF: backward-expanding ADF regressions on log prices, take supremum of β's t-statistic | minSL=90 | ~15 min |
| 8 | `smt_poly1` | Ch. 17.4.3 | Sub/Super-Martingale polynomial-1 spec: `log[yₜ] = α + β·t + ε`, backward-expanding, supremum of `|t-stat(β)| / length^0.5` | minSL=90 | ~15 min (joint) |
| 9 | `smt_exp` | Ch. 17.4.3 | Sub/Super-Martingale exponential spec: `log[yₜ] = α + β·exp(t/length) + ε`, same supremum structure | minSL=90 | ~15 min (joint) |

**Internal helpers:** `_compute_sadf()`, `_adf_tstat()`, `_compute_smt()`, `_ols_tstat()`, `_compute_rolling_entropy()`, `_compute_rolling_lz()`, `_lempel_ziv_76()`, `_compute_rolling_hurst()`, `_hurst_rs()`, `_compute_rolling_variance_ratio()`, `_compute_rolling_jarque_bera()`, `_compute_rolling_gaussian_entropy()`.

#### Step 6.c — Lag Features (`features.py`)

**Function:** `compute_lag_features(df, lags=AR_LAGS) → pd.DataFrame` — 10 columns

Precomputes lagged log-return features on the full daily series. Columns are named `log_returns_lag1`, `log_returns_lag2`, `log_returns_lag3`, `log_returns_lag4`, `log_returns_lag5`, `log_returns_lag6`, `log_returns_lag7`, `log_returns_lag14`, `log_returns_lag21`, `log_returns_lag30`.

**Why a separate category.** Lag features are precomputed once on the full daily series rather than inline inside `ARLogistic.fit` / `ARLogistic.predict`. The inline version had a look-ahead artefact: NaN lags at the head of each test fold were imputed with `bfill()`, which used later test observations to fill earlier ones. Precomputing on the global series gives every aligned event valid lookback values that respect chronological order.

**Why these lags.** The set `[1, 2, 3, 4, 5, 6, 7, 14, 21, 30]` follows a calendar-day convention rather than the trading-day convention `[1, 2, 3, 5, 10, 21]` common in equity ML literature. BTC trades 24/7 with no weekend or holiday gaps, so trading-day arithmetic has no natural meaning for this asset. Lags 1-7 cover the complete weekday cycle, capturing both immediate autocorrelation (lags 1-3, the conventional short-term momentum window) and the rest of the week (lags 4-7) so the AR Logistic baseline can resolve any day-of-week patterns BTC inherits from the macro/equity calendar despite trading 24/7. Lags 14, 21, and 30 capture two-week, three-week, and one-month cycles. The 14-day lag additionally aligns with the BTC mining-difficulty adjustment cycle (~2,016 blocks ≈ 14 days), a BTC-native cycle the equity convention does not capture.

The list was extended from the earlier `[1, 2, 3, 7, 14, 30]` configuration after observing that filling out the full 1-7 weekday cycle and adding the 21-day mid-month marker produced a better-spec'd autoregressive baseline. With ~900 training events per fold and 11 parameters (10 lags + intercept), AR Logistic still has roughly 80 events per parameter, well within healthy fitting territory.

**Helpers:**
- `lag_column_names(lags=None)`: returns the canonical column names in the order matching `lags` (defaulting to `AR_LAGS`).

**MDA inclusion (advisor-driven change).** Lag features now enter the multi-model MDA pool alongside TA, math, and external features. An earlier version excluded them from MDA on the rationale that they should be reserved for the AR Logistic baseline; the advisor argued this created an unfair information asymmetry, since AR Logistic received lag features the ML and neural models did not. The current pipeline lets all 73 features (was 62 before the lag extension and 66 before the macro horizon extension) compete for the top-k cap; whether MDA selects any lag columns for the other models depends on their permutation importance for that fold. AR Logistic still selects its ten lag columns by name from the pre-MDA matrix via the pipeline's `X_tr_full` route, so its behaviour is unchanged.

#### Step 6.d — Macro Features (`external_features.py`)

**Function:** `compute_macro_features(btc_index) → pd.DataFrame` — 20 columns

**Alignment method:** All external data is merged onto BTC's 7-day calendar via `pd.merge_asof(direction='backward')` through the helper `_align_to_btc()`. This ensures each BTC day uses the most recent available value (no look-ahead): weekends carry Friday's macro close, weekly data persists until the next release.

Data sources: yfinance for 11 tickers plus FRED (via pandas-datareader) for 2Y Treasury yield with spread-based fallback.

| # | Feature | Source Ticker | Transformation |
|---|---------|--------------|----------------|
| 1 | `dxy_roc_30` | DX-Y.NYB (US Dollar Index) | 30-day rate of change (%) |
| 2 | `us10y` | ^TNX (10Y Treasury) | Level (yield in %), auto-scaled if median > 10 |
| 3 | `us2y` | FRED DGS2 → spread-derived fallback | Level (yield in %), auto-scaled |
| 4 | `yield_curve_2y10y` | Derived: us10y − us2y (or FRED T10Y2Y directly) | Spread (percentage points) |
| 5 | `yield_curve_10y30y` | Derived: ^TYX − ^TNX | Spread (percentage points) |
| 6 | `vix` | ^VIX | Level |
| 7 | `sp500_ret_30` | ^GSPC | 30-day log return |
| 8 | `sp500_ret_14` | ^GSPC | 14-day log return |
| 9 | `nasdaq_ret_30` | ^IXIC | 30-day log return |
| 10 | `nasdaq_ret_14` | ^IXIC | 14-day log return |
| 11 | `gold_ret_30` | GC=F | 30-day log return |
| 12 | `gold_ret_14` | GC=F | 14-day log return |
| 13 | `silver_ret_30` | SI=F | 30-day log return |
| 14 | `silver_ret_14` | SI=F | 14-day log return |
| 15 | `copper_ret_30` | HG=F | 30-day log return |
| 16 | `copper_ret_14` | HG=F | 14-day log return |
| 17 | `oil_ret_30` | CL=F | 30-day log return |
| 18 | `oil_ret_14` | CL=F | 14-day log return |
| 19 | `natgas_ret_30` | NG=F | 30-day log return |
| 20 | `natgas_ret_14` | NG=F | 14-day log return |

**Dual-horizon convention for fast-moving assets.** The seven non-yield macro assets (S&P 500, Nasdaq, gold, silver, copper, oil, natural gas) are exposed at both 30-day and 14-day return horizons via the module-level constants `RET_WINDOW = 30` and `RET_WINDOW_SHORT = 14`. The 14-day variant captures bi-weekly cycles and is more responsive to recent news and macro shocks; the 30-day variant captures monthly cycles and is smoother. MDA decides per fold which horizon is more informative, with three possible outcomes: (1) MDA consistently prefers one horizon per asset across folds, indicating a stable preference; (2) MDA picks both for some assets, indicating that the 14d and 30d variants each capture marginally distinct information despite their correlation (typically Pearson r > 0.7); (3) MDA picks differently per fold, indicating regime-dependent optimal horizons. Slow-moving variables (DXY, US Treasury yields, yield-curve spreads) keep the 30-day horizon only because their information content is dominated by monthly settlement and release cycles where a 14-day variant would add noise without signal.

**2Y yield fallback:** (1) FRED DGS2 directly. (2) If DGS2 returns fewer than 2,000 bars, falls back to fetching the T10Y2Y spread from FRED, using it directly as `yield_curve_2y10y` and back-deriving `us2y = us10y − spread`. This two-step fallback ensures the yield curve feature is always populated even when DGS2 is unavailable.

#### Step 6.e — Crypto-Macro Features (`external_features.py`)

**Function:** `compute_crypto_macro_features(btc_close, btc_index) → pd.DataFrame` — 1 column

Single market-level cross-crypto signal, distinct from blockchain fundamentals (which are in Step 6.f).

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 1 | `eth_btc_ratio` | CoinMetrics ETH PriceUSD (with yfinance fallback) | `ETH_close / BTC_close` aligned via merge_asof |

**ETH source priority.** The earlier version of this column used yfinance's `ETH-USD` ticker as the sole source. yfinance's ETH-USD history begins around November 9, 2017 (3,092 daily bars over a ~4,200-day external dataset), leaving roughly 27% of the calendar series as NaN at the head and producing entire-test-partition NaN in early CPCV folds under N=8. The current implementation switches the primary source to CoinMetrics' Community-tier API, which serves ETH price data back to August 8, 2015 (Ethereum's Frontier launch date). The fetch tries three CoinMetrics metrics in priority order — `ReferenceRateUSD`, `PriceUSD`, then a `CapMrktCurUSD / SplyCur` derivation — because not every ETH metric is available on the Community tier (ReferenceRateUSD is currently Pro-gated for ETH and returns 1 row; PriceUSD returns ~3,917 daily rows back to August 2015). The first metric returning more than 100 rows is used. yfinance remains as a final fallback if all three CoinMetrics attempts fail; the source actually used is logged at fetch time. With the CoinMetrics PriceUSD source, the residual `eth_btc_ratio` NaN rate over the external dataframe is 7.7%, all of which falls in the November 2014 → August 2015 pre-ETH-trading window and is fully truncated out by the CUSUM start-date filter described in Step 2.

**Note on the dropped `btc_dominance` feature.** An earlier version included a second crypto-macro column, `btc_dominance`, fetched from CoinGecko's `/coins/bitcoin/market_chart` endpoint with a `100 / (1 + ETH/BTC)` proxy as fallback. The CoinGecko endpoint actually returns BTC market cap in USD (not the bounded [0, 100] dominance percentage the column name implied), and the proxy fallback was a price-correlated approximation that the methodology could not cleanly defend. The column was removed; `eth_btc_ratio` carries the alt-rotation signal alone.

**Why not CoinGecko for ETH directly?** CoinGecko's free public API was evaluated as a candidate ETH source. As of 2024 the free tier limits historical queries to the most recent 365 days only; the 2015-2017 ETH history needed for full BTC-dataset coverage sits behind a paid tier. CoinMetrics' Community tier has no such restriction and serves ETH prices back to genesis without rate limits or authentication, which is why it is the chosen primary source.

#### Step 6.f — On-Chain Features (`external_features.py`)

**Function:** `compute_onchain_features(btc_index) → pd.DataFrame` — 8 columns

Data source: CoinMetrics Community API (free tier), cached to `cache/onchain_raw.parquet`.

**Raw metrics fetched from CoinMetrics:**

```
AdrActCnt, TxCnt, HashRate, CapMVRVCur, FlowInExNtv, FlowOutExNtv,
FeeTotNtv, SplyExNtv, SplyCur, IssTotNtv
```

**Important:** Raw CoinMetrics data is shifted by 1 day (`df.shift(1)`) to avoid look-ahead bias, since CoinMetrics reports end-of-day values that wouldn't be available at market open.

| # | Feature | Derived From | Transformation | Stationarity |
|---|---------|-------------|----------------|-------------|
| 1 | `active_addr_roc_14` | AdrActCnt | 14-day rate of change (%) | Stationary (RoC) |
| 2 | `tx_count_roc_14` | TxCnt | 14-day rate of change (%) | Stationary (RoC) |
| 3 | `hashrate_roc_30` | HashRate | 30-day rate of change (%) | Stationary (RoC) |
| 4 | `mvrv` | CapMVRVCur | Level (already mean-reverting) | Stationary |
| 5 | `net_exchange_flow` | FlowInExNtv − FlowOutExNtv | Net BTC flow (positive = inflows, selling pressure) | Stationary |
| 6 | `fee_per_tx` | FeeTotNtv / TxCnt | Ratio (BTC per transaction) | Stationary |
| 7 | `exchange_supply_pct` | (SplyExNtv / SplyCur) × 100 | Percentage of total supply on exchanges | Stationary |
| 8 | `issuance_ntv` | IssTotNtv | Daily BTC issuance (changes at halvings) | Step function |

#### External features orchestration

`build_external_features(btc_df, include_macro=True, include_crypto_macro=True, include_onchain=True) → pd.DataFrame` fetches all three categories (macro 13, crypto-macro 1, on-chain 8 = 22 columns), concatenates, reports NaN summary, and caches to Parquet. Each category is independently toggleable. On-chain fetch is wrapped in try/except so missing `coinmetrics-api-client` doesn't break the pipeline.

**Cache invalidation.** The cache check verifies both the date-range endpoints and the column set. Changing the feature mix (e.g., dropping `btc_dominance`) invalidates the cache and triggers a refetch on next run; the date-range-only check would have silently returned a stale frame.

---

### Step 7 — Compression Transforms (`features.py`)

**Functions:** `apply_sym_log(features, columns) → pd.DataFrame` and `apply_log(features, columns, eps=1e-8) → pd.DataFrame`

Two scale-compression transforms applied in fixed order, with the column lists exposed at the notebook level rather than hardcoded inside the module so the choice of which columns get which transform is visible at the call site:

1. **Symmetric log** to `obv` and `chaikin_osc`: `np.sign(x) * np.log(np.abs(x) + 1)`. The sign-preserving variant of the natural log. Used because both features can be either positive or negative and span 10⁷-10¹⁰ over the 2014-2026 period as BTC daily volume grew by six orders of magnitude. Symmetric log preserves sign and zero, preserves rank ordering, and asymptotically behaves like `sign(x) · log(|x|)` for `|x| ≫ 1` (compressing the tails by orders of magnitude). The derivative discontinuity at zero is acceptable in a feature-engineering context where the transformed values feed downstream MDA, FFD, and scaling steps rather than a gradient-based optimiser.

2. **Unsigned log** to `atr`: `np.log(np.abs(x) + 1e-8)`. ATR is strictly non-negative, so sign preservation is irrelevant; the small additive constant prevents `log(0)` for the rare zero-volatility bar. The `np.abs()` guards against any spurious negative input (which would not be expected on ATR) without crashing the transform.

All other features are left untouched (already bounded or dimensionless and handled by the per-fold RobustScaler downstream).

**Target columns (notebook-level constants).** The notebook defines:

```python
SYM_LOG_COLUMNS = ["obv", "chaikin_osc"]
LOG_COLUMNS     = ["atr"]

feature_matrix = pd.concat([ta_features, math_features, external, lag_features], axis=1)
feature_matrix = apply_sym_log(feature_matrix, SYM_LOG_COLUMNS)
feature_matrix = apply_log(feature_matrix, LOG_COLUMNS)
```

This is a deliberate methodology choice: hiding column lists inside `features.py` (as a previous iteration did via a combined `apply_log_transforms` and module-level `SIGNED_*_COLUMNS` constants) makes the choice invisible to a reader scanning the notebook. Putting them at the call site makes the transformation policy explicit and inspectable, with both helper functions accepting the column list as an argument and silently skipping any column not present in the DataFrame so the same target list can be reused across feature-set variants.

**Methodology trail (asinh sensitivity, May 2026).** An asinh-based variant of the signed transform was tested in May 2026: `apply_asinh` replaced symmetric log with `np.arcsinh`, motivated by asinh's smoothness at zero (no derivative kink), parameter-free form, and the Burbidge-Magee-Robb 1988 econometric tradition for signed wide-range variables. In sensitivity analysis the asinh variant produced substantially higher path-Sharpe variance on the neural-network models and an inflated PBO (0.571 vs the symmetric-log run's 0.143), even though asinh and symmetric log are within `log(2) ≈ 0.69` of each other for `|x| ≫ 1`. The transform was reverted to symmetric log on grounds of empirical out-of-sample stability across CPCV folds, which the thesis methodology section discloses as an honest sensitivity finding rather than a result-driven choice. The `apply_asinh` function was removed; the symmetric-log convention is the locked thesis configuration.

**Orchestration:** `build_feature_matrix(df) → pd.DataFrame` chains `compute_ta_features(df)` → `compute_math_features(df)` → `pd.concat`, returning 34 columns × ~4,200 rows (25 TA + 9 math) with NO compression transforms applied. The transforms are now the notebook's responsibility (visible at the call site, see the snippet above). External features (22) and lag features (10) are assembled separately in the notebook via `build_external_features(df_raw)` and `compute_lag_features(df_raw)` respectively, and concatenated alongside the TA + math output before the transforms are applied. The notebook concat order is `[ta, math, external, lag]`.

---

### Step 8 — EDA (in the notebook, not in a `.py` file)

Lives directly in `main.ipynb` section 2.5 as inline plotting cells. Inspects data quality before committing to the CV loop.

| Check | Method | What to look for |
|-------|--------|-----------------|
| Feature distributions | Histograms per feature with kurtosis annotation | Features with kurtosis > 10 (flagged in red) may saturate KAN spline ranges |
| Pairwise correlations | Heatmap with values annotated, threshold at \|r\| > 0.9 | Redundant feature pairs (e.g., stoch_k ↔ williams_r) as drop candidates |
| Stationarity | ADF test per feature at 5% significance | Non-stationary features → candidates for FFD inside CPCV (currently only ATR) |
| Feature-label MI | `mutual_info_classif` with 5 nearest neighbors | Features with near-zero MI are removal candidates |

**Notebook also includes:** label distribution pie chart, holding period statistics, vertical barrier hit rate.

---

### Step 9 — Alignment (`alignment.py`)

**Produces:** `(X, y, w, t1)` — the four objects that enter the CPCV loop.

#### `align_for_cv(features, bins, weights) → (X, y, w, t1)`

Computes the triple intersection of features (all ~4,200 daily bars), bins (CUSUM-filtered events truncated to start at `CUSUM_START_DATE`, approximately **1,150** after dropping rare class 0 at 8.5% threshold), and weights (same index as bins) via `features.index.intersection(bins.index).intersection(weights.index)`.

**Assertions (raises on failure):**
- Non-empty intersection
- No duplicate dates
- Monotonically increasing index
- All four arrays have identical length
- No feature column entirely NaN

**Warnings (logged, doesn't fail):** Reports remaining NaN counts per column (from long-lookback features like Hurst in early rows).

#### `validate_alignment(X, y, w, t1) → bool`

Standalone validation for the CV loop to call. Checks everything `align_for_cv` checks plus:
- Labels ∈ {-1, 0, +1} only
- All weights > 0
- No NaN in t1 (would break CPCV purging)
- X and y share identical indices
- X and w share identical indices
- X and t1 share identical indices

---

### NaN Handling Across Phase 1

Feature computation produces NaN values for two reasons:

1. **Rolling-window warm-up.** Lookback features cannot be computed before their window has filled. RSI needs 14 bars, EMA-50 needs ~50, Bollinger Bands need 20, ATR needs 14, Shannon entropy needs 30, SADF needs 90, variance ratio needs 90, Hurst exponent needs 180. Each feature carries a leading NaN run of length equal to its rolling window.
2. **External-data calendar gaps.** Equity tickers (S&P 500, gold, oil, etc.) do not trade on weekends; FRED macro releases occur weekly or monthly with publication delays; CoinMetrics on-chain data has its own warm-up and occasional missing days. Each external column has scattered NaN cells where the source data was not yet released or not aligned to a BTC trading day.

**Phase 1 does not drop or impute these NaNs.** The feature-engineering layer simply produces them and logs per-column NaN percentages. Reasoning: dropping rows at this stage would lose the early period where some features are fully computable but others are not (e.g., RSI is valid by day 15 but Hurst is not valid until day 180); imputing would fabricate values from a future-looking window.

**Alignment** (`align_for_cv`) hardens the policy slightly. The step performs:
- A *hard assertion* that no feature column is entirely NaN (catches dead columns from a failed external-data fetch).
- A *hard assertion* that `t1` is fully populated (any missing barrier-touch timestamp would break CPCV purging).
- A *soft warning* listing per-column NaN counts for partial-NaN columns; the pipeline continues.

Most warm-up NaNs are eliminated naturally at alignment because `bins.index` (the CUSUM event index, ~1,150 events after the August 2015 truncation and the rare-class drop) starts firing well after the longest feature warmup window has completed. The 252-day raw-data buffer ahead of the CUSUM start date guarantees that even the slowest-warming features (Hurst at 252, EMA 50/200 ratio at 200, SADF at variable but bounded windows) have valid values for every CUSUM event. External columns may still have scattered NaN cells where the source data was not yet released or not aligned to a BTC trading day; these are handled in Phase 2 via per-fold ffill+bfill within partitions.

The X passed to CPCV may therefore contain partial-NaN rows. This is intentional: the per-fold preprocessing step resolves them with a partition-aware policy described in Step 11 (FFD).

### Pre-CPCV Output Summary

The shape numbers below reflect the locked configuration with `START_DATE="2014-11-01"`, `END_DATE="2026-05-01"`, `CUSUM_START_DATE="2015-08-08"`, `CUSUM_MULT=1.0`, `pt_sl=(1.5, 1.5)`, `num_days=10`, and `drop_rare_labels(min_pct=0.085)`. Approximate event count is given below; the figure depends on the exact CUSUM event density in the August 2015 → May 2026 window and is updated after the locked end-to-end run. The previous configuration (raw data from September 2014, no CUSUM truncation) produced 1,245 events; the buffer-and-truncate configuration drops events that fired before August 8, 2015, reducing the event count by approximately 80-100.

| Object | Shape | Description |
|--------|-------|-------------|
| `X` | ~1,150 × 73 | Feature matrix (25 TA + 9 math + 29 external + 10 lag, post log-transform). All 73 columns are eligible for MDA selection; AR Logistic restricts itself to the 10 lag columns by name from the pre-MDA matrix. |
| `y` | ~1,150 | Binary labels {-1, +1}. The 0 class is removed by `drop_rare_labels(min_pct=0.085)` after triple-barrier labeling. |
| `w` | ~1,150 | AFML sample weights (uniqueness × return attribution × time decay, capped at 99th pctile) |
| `t1` | ~1,150 | Barrier touch timestamps (DatetimeIndex, for CPCV purging) |

These four objects are the contract between Phase 1 and Phase 2. Everything from FFD onward happens inside the CPCV loop, fitted on training data only.

---

### Step 9.5 — Pre-CPCV Plotting and EDA (`pre_cpcv_plots.py`)

Visualisation and diagnostic helpers for the labelling and feature-engineering stages. Each function takes the relevant data, returns a matplotlib `Figure` (or `(Figure, DataFrame)` for the ADF test that produces both a chart and a result table), and prints diagnostic summaries. Functions do not call `plt.show()`; the notebook decides whether to display, save, or close each figure. This keeps the helpers reusable for thesis-figure exports at different sizes and DPIs.

| Function | Purpose | Returns |
|----------|---------|---------|
| `plot_cusum_filter(log_returns, t_events, h, zoom_start, zoom_end)` | Two-panel zoom: log returns coloured by sign with event markers, and CUSUM cumulative sums `s_pos` / `s_neg` against the threshold band | `Figure` |
| `plot_tbl_examples(bins, close, daily_vol, pt_sl, num_days, zoom_start, zoom_end)` | Side-by-side panels showing one event per barrier-touch type (profit / stop loss / vertical) with annotated price paths | `Figure` or `None` |
| `plot_label_distribution(bins)` | Donut chart of class counts with absolute and relative numbers printed below | `Figure` |
| `plot_feature_distributions(feature_matrix, kurtosis_threshold)` | Grid of histograms with kurtosis flagging for features that may saturate KAN spline ranges | `Figure` |
| `plot_feature_correlation(feature_matrix, corr_threshold, annotate)` | Annotated heat-map of the feature correlation matrix; flags pairs above the threshold | `Figure` |
| `plot_feature_label_mutual_info(feature_matrix, bins, mi_threshold)` | Horizontal bar chart of MI between each feature and the binary label, computed via `sklearn.mutual_info_classif` | `Figure` |
| `plot_adf_stationarity(feature_matrix, significance, maxlag, autolag)` | Per-feature Augmented Dickey-Fuller test with a p-value bar chart, threshold line at `significance`, and a returned DataFrame of test statistics for downstream use | `(Figure, DataFrame)` |

The ADF function returns the result table because its output drives the FFD column choice in the notebook (`COLUMNS_TO_FFD`); the table is also a candidate for a thesis appendix.

The CUSUM and TBL functions accept `zoom_start` / `zoom_end` parameters with notebook-overridable defaults. The defaults pick a recent zoom window (Jan-Apr 2026); call with different ranges to inspect earlier regimes.

---

## Phase 2 — CPCV (Cross-Validation & Training)

Everything in this phase runs inside the CPCV loop. Every stateful transformation (FFD d* estimation, scaling, feature selection, hyperparameter tuning) is fitted on training data only. This is the leakage-critical zone.

### Notebook sections: 3 (CV Framework) → 4 (Model Training)

---

### Step 10 — CPCV Split Generation (`cv.py`)

**Produces:** 28 train/test splits with purging and embargo, plus a 7-path assignment matrix.

**Module-level constants (defaults):** `N_GROUPS=8`, `K_TEST_GROUPS=2`, `EMBARGO_PCT=0.01`. The notebook passes `n_groups` and `k` explicitly per call so the configuration is visible at the top of the CV cell rather than buried in a module constant.

#### `generate_cpcv_splits(X, t1, n_groups=8, k=2, embargo_pct=0.01) → list[tuple[np.ndarray, np.ndarray]]`

Partitions T observations into N=8 contiguous groups (groups 0–6 of size ⌊T/N⌋, group 7 gets the remainder, ~156 events per group at the current dataset size). Generates all C(8,2) = 28 combinations of 2 test groups.

For each split, applies:

**Purging** (AFML Snippet 7.1): Removes training observations whose labels overlap with any test group. Checks three sufficient overlap conditions for each training observation i against each test group boundary [t_test_start, t_test_end]:
1. Training observation starts within test period: `t_test_start ≤ t_i_start ≤ t_test_end`
2. Training label resolves within test period: `t_test_start ≤ t1[i] ≤ t_test_end`
3. Training label spans the entire test period: `t_i_start ≤ t_test_start AND t_test_end ≤ t1[i]`

**Embargo** (AFML Section 7.4.2): Removes `int(embargo_pct × T)` training observations immediately after each test group boundary. Only applied after the test set (not before), since training labels resolving before test begins contain no future information.

Returns positional integer arrays into X for each split.

#### `build_path_matrix(n_groups=8, k=2) → (n_paths, path_map)`

Computes φ[N,k] = C(N-1, k-1) = C(7,1) = 7 backtest paths. For each group, collects all splits where it appears in the test set, then assigns each occurrence to a path so every path covers all N groups exactly once.

Returns `path_map: {path_id: [(group_id, split_id), ...]}` with N entries per path.

#### `get_split_info(X, t1, n_groups=8, k=2, embargo_pct=0.01, splits=None, path_map=None, n_paths=None, print_summary=True) → dict`

Computes (or accepts) and optionally prints a CPCV split-configuration summary. The function originally recomputed splits and paths every time it was called, which led to duplicate work and duplicate log lines when the notebook also called `generate_cpcv_splits` and `build_path_matrix` separately. The current signature lets the caller pass already-computed values via `splits` / `path_map` / `n_paths`, in which case no recomputation occurs.

The recommended notebook pattern is now:

```python
splits = generate_cpcv_splits(X, t1, n_groups=N_GROUPS, k=K_TEST, embargo_pct=EMBARGO_PCT)
n_paths, path_map = build_path_matrix(n_groups=N_GROUPS, k=K_TEST)
split_info = get_split_info(
    X, t1, n_groups=N_GROUPS, k=K_TEST, embargo_pct=EMBARGO_PCT,
    splits=splits, path_map=path_map, n_paths=n_paths,
)
```

Each function is called exactly once; the printed summary is rendered without redundant work. The accompanying helper `print_split_summary(info)` is also exposed for callers that want to render an existing info dict (e.g., when iterating on a configuration without rerunning the splitter).

---

### Step 10.5 — CPCV Plotting and Audits (`cpcv_plots.py`)

Visualisation and verification helpers for the CPCV configuration, called after the CV cell has produced `splits`, `path_map`, `n_paths`, and `split_info`. Each function takes the CPCV inputs as parameters and returns either a `Figure` (visual diagnostics) or a `DataFrame` (the leakage audit). All functions derive their date axis from `X.index` directly, so after the CUSUM start-date truncation they automatically span the analysis window (August 2015 onward) rather than the raw-data start date.

| Function | Purpose | Returns |
|----------|---------|---------|
| `pick_demo_splits(all_combos, n_groups)` | Helper that picks three illustrative split indices: contiguous-early `(0, 1)`, one-gap `(1, 3)`, and contiguous-tail `(n_groups-2, n_groups-1)` | `list[int]` |
| `plot_btc_with_groups(X, df_raw, n_groups, use_log_scale)` | BTC close price with the N CPCV groups shaded as coloured bands; toggleable log/linear y-axis | `Figure` |
| `plot_train_test_timelines(X, splits, n_groups, k, demo_splits)` | Three sub-panels showing train (steelblue) and test (crimson) date partitioning for representative splits, with test-group shading and group-boundary verticals | `Figure` |
| `print_purge_embargo_detail(X, t1, splits, n_groups, k, demo_splits)` | Per-split text dump showing the last three training rows before each test group (purge zone) and the first three after (embargo zone), with `OK` / `OVERLAP` flags on each pre-test row | `None` |
| `audit_cpcv_leakage(X, t1, splits, n_groups, k, split_info)` | Full audit across all splits checking that no training label end-time `t1` resolves inside a test group; prints a summary table and returns the audit data | `DataFrame` |

`audit_cpcv_leakage` returns the audit as a DataFrame so the notebook can write it to a thesis appendix table or filter to failures only; on a clean run all 28 splits should report `status="OK"` with zero leaks.

The 8-color palette used by the visual functions is exposed as a module-level constant `DEFAULT_GROUP_COLORS` for consistency across plots.

---

### Step 11 — Per-Fold Preprocessing (`preprocessing.py`)

**Produces:** Scaled, FFD-transformed, feature-selected train and test DataFrames. Shared across all models within the same fold.

**Module-level constants:**
```
FFD: FFD_D_RANGE=(0.0, 1.0, 0.05), FFD_THRESHOLD=1e-4, FFD_ADF_SIGNIFICANCE=0.05, FFD_MAX_LOOKBACK=200
Selection: MDA_N_ESTIMATORS=500, MDA_N_INNER_FOLDS=3, MDA_TOP_K_FRAC=0.30, MIN_FEATURES=5
```

#### Step 11.a — FFD (`ffd_transform`)

Applies FFD to the **full** series (so test observations have lookback history) but estimates d* from **training data only**. For each column in `ffd_columns`:

1. `find_optimal_d(train_series)`: Sweeps d from 0 to 1 in steps of 0.05. At each d, computes FFD weights via `ω_0 = 1, ω_k = -ω_{k-1} × (d - k + 1) / k`, truncated when `|ω_k| < 1e-4` or k reaches 200. Applies as convolution, runs ADF test. Returns minimum d where p-value < 0.05.
2. `apply_ffd(full_series, d_star)`: Applies FFD at d* to the complete series.
3. Splits into train/test by positional indices.
4. **Per-fold NaN resolution (asymmetric by column type).** The leading-NaN problem inherited from Phase 1 is resolved here, inside the CPCV loop, with a policy that differentiates FFD-eligible from non-FFD columns and respects the train-test boundary.
   - **Non-FFD columns (the majority).** External-data calendar gaps and remaining slow-warmup gaps are imputed via `ffill().bfill()` *applied independently within `X_train` and within `X_test`*. The forward fill carries the most recent valid observation across calendar gaps (the natural economic interpretation: the model uses the last published macro reading or last on-chain snapshot, exactly as a human trader would). The trailing back-fill handles any leading NaN at the very start of the partition. The two operations never cross the train-test boundary, so test-set values cannot leak into train and vice versa.
   - **FFD columns (currently ATR only).** Rows where any FFD column is NaN are dropped, not imputed. Forward-filling an FFD-transformed column would inject stale weighted-lookback values into the model's input; dropping the leading rows is the correct response. The function logs the dropped row count: `"FFD: dropped X train, Y test NaN rows from FFD lookback."` Typically 2-10 rows per partition, concentrated at the start.

**Currently applied to:** ATR only (identified as non-stationary in EDA ADF tests).

#### Step 11.b — Scaling (`scale_features`)

Fits `sklearn.preprocessing.RobustScaler` (median + IQR) on training fold. Transforms both train and test. Handles empty test sets gracefully. Returns the fitted scaler for potential inverse-transformation.

#### Step 11.c — Feature Selection (`select_features`)

**Multi-Model MDA** (differs from AFML's three-method MDI/MDA/SFI protocol):

Uses `compute_multi_model_mda()` which runs permutation importance with **two classifiers in parallel**:

| Model | Why included |
|-------|-------------|
| Random Forest (500 trees, balanced) | Captures nonlinear interactions and ensemble effects |
| Logistic Regression (balanced, L2) | Captures linear relationships, prevents RF-biased selection |

**Rationale for multi-model over RF-only MDA:** RF-only MDA introduces tree bias (features that tree models naturally exploit get inflated importance). SFI in weak-signal regimes produces uninformative near-uniform scores (~0.45–0.55). Averaging MDA from RF and LR balances these biases and reduces selection variance.

For each model, `_compute_mda_single_model()` runs a **purged inner 3-fold CV** on the training set:
1. Splits training data into 3 chronological inner folds
2. Purges inner-train observations whose t1 overlaps inner-test (same 3-condition check as outer CPCV)
3. Fits the classifier, computes baseline F1 on inner-test
4. For each feature: permutes column, recomputes F1, records `MDA = baseline_F1 − permuted_F1`
5. Averages MDA across inner folds

The final averaged MDA per feature = `mean(MDA_RF, MDA_LR)`. Selection rules:
1. Keep features with averaged MDA > 0 (permuting hurts at least one model on average)
2. Cap at `top_k_frac` of total features (default 30%, overridable from notebook)
3. Hard floor of 5 features minimum

**Lag features in the MDA pool.** `select_features` runs MDA over all 73 features, including the six `log_returns_lagN` columns. An earlier version of the pipeline excluded lag features from MDA on the rationale that they should be reserved for the AR Logistic baseline; the advisor argued this created an unfair information asymmetry, since AR Logistic received lag features the ML and neural models did not see. The current pipeline lets every feature compete on equal footing; lag columns appear in `selected` for a given fold only if their averaged MDA (RF + Logistic Regression permutation importance) is positive and ranks within the top-k cap. AR Logistic continues to consume the ten lag columns by name from the pre-MDA matrix via the pipeline's `X_tr_full` route, independently of MDA's choices.

**Typical result:** ~14–16 features selected per fold from the 73 candidates. The 10 lag columns sit alongside engineered features in `X_tr_proc` and may or may not appear in `selected` depending on per-fold MDA. AR Logistic always sees the lag columns regardless, because it bypasses MDA via `X_tr_full`.

**Per-fold warning provenance.** When fewer than `MIN_FEATURES = 5` features clear MDA > 0 on a given fold and the function falls back to the top-5 by MDA value, the emitted `logger.warning(...)` is prefixed with `[split K/N]` where `K` is the 1-indexed CPCV split number and `N` is the total number of splits (28 in the locked configuration). The prefix is plumbed via the keyword-only `split_idx` and `n_splits` arguments to `select_features` and `preprocess_fold`, which `pipeline.py` populates inside the per-split loop. The prefix surfaces in the deferred-warning summary that `run_cpcv_pipeline` prints after the ✅ completion notice, so the reader can tell which specific folds tripped the fallback rather than assuming the issue applied uniformly across all 28.

**TOP_K_FRAC tightening and sensitivity (from 0.4 to 0.30, with 0.25 as the sensitivity check).** The cap was tightened from common defaults of 0.40-0.50 after a high-PBO run revealed that only ~6 features cleared 50% selection frequency in the stability bar chart, indicating that the long tail of the MDA-ranked feature set was contributing variance rather than signal. Two cap settings were evaluated as candidates for the primary configuration: 0.25 forces ~16 features through the bottleneck, and 0.30 forces ~19. Both produced essentially identical predictive metrics across all six models (mean accuracy and pooled AUC differed by less than 0.5 percentage points per model), but the backtest-derived metrics diverged substantially: PBO was 0.69 at 0.25 and 0.26 at 0.30, with model rankings reshuffling between the two. The 0.30 setting was selected as primary because it produced the lower PBO (under the AFML interpretation, this indicates more robust model selection) while still being meaningfully tighter than the 0.40 default. The 0.25 setting is retained as a documented sensitivity check; the divergence between the two configurations on the path-Sharpe-derived metrics, despite stable predictive metrics, is itself the AFML rank-instability finding made operational on this dataset.

#### `preprocess_fold(X_full, train_idx, test_idx, y_train, w_train, t1_train, ffd_columns, top_k_frac, skip_selection=False)`

Orchestration chaining FFD → scaling → selection. Returns DataFrames with **all columns** (pre-selection) so the pipeline can provide full-column DataFrames to AR Logistic. The selected feature list is returned separately. The `skip_selection` flag is set when only AR Logistic is being evaluated; AR Logistic does not need MDA selection because it consumes only the precomputed lag columns from Step 6.c.

---

### Step 12 — Nested Hyperparameter Tuning (`tuning.py`)

**Produces:** Per-split optimal hyperparameters for each tuned model, derived using only the 4 training groups of the current CPCV split (AFML Ch. 7 compliant). Replaces the earlier walk-forward 3-fold majority vote approach.

**Module-level constants:**
```
N_INNER_FOLDS = 3              # Purged K-Fold folds inside each training fold
PURGE_EMBARGO = 10             # observations purged around inner-fold boundaries
N_TRIALS_CLASSICAL = 60        # default trials for Logistic, RF, XGBoost
N_TRIALS_NEURAL = 40           # default trials for LSTM, KAN
```

The `run_cpcv_pipeline()` function accepts an `n_trials` parameter that overrides these defaults. The notebook currently passes `n_trials=30` for every tuned model. The recommendations below give a wider menu balancing exploration against runtime:

| Model | Notebook `n_trials` | Lower-cost alternative | Rationale |
|-------|---------------------|------------------------|-----------|
| Logistic | 30 | — | Cheap per trial, only 2 hyperparameters |
| Random Forest | 30 | — | Moderate cost, 4 hyperparameters |
| XGBoost | 30 | — | Moderate cost, 8 hyperparameters (early stopping compensates) |
| LSTM | 30 | 20 | Expensive per trial; the notebook keeps 30 for parity with the classical models |
| KAN | 30 | 20 | Expensive per trial; the notebook keeps 30 for parity with the classical models |

#### `_purged_kfold_splits(X, y, w) → list[(train_idx, val_idx)]`

Creates 3 chronological inner folds with 10-observation embargo around boundaries (matches TBL num_days). Fewer folds (3 vs 5) improves runtime by ~40% and increases inner validation set size (~300 vs ~180 observations per fold), providing more reliable log loss estimates in a low-signal environment.

#### Per-model tuning functions

Each returns `{"best_params": {...}, "best_log_loss": float, "results_df": DataFrame}`:

**`tune_logistic(X, y, w, n_trials=None)`** — Search space:
- `C`: log-uniform [1e-4, 1e2]
- `penalty`: categorical {l1, l2}

**`tune_random_forest(X, y, w, n_trials=None)`** — Search space:
- `n_estimators`: int [100, 250] step 50 (capped from an earlier 300 ceiling; trees in a noisy regime do not benefit from more than 250)
- `max_depth`: int [2, 6] (tightened from earlier [3, 15]; depth 6 has 64 leaves which is plenty for ~900-sample training folds, and shallower forests vote in tighter agreement, reducing the disagreement that surfaces as path-Sharpe variance)
- `min_samples_leaf`: int [15, 40] (raised from earlier [1, 30]; a floor of 15 forces each leaf to represent ≥1.7% of the training fold, preventing leaves that fit just a handful of high-volatility events)
- `max_features`: categorical {sqrt, log2}

**`tune_xgboost(X, y, w, n_trials=None)`** — Search space:
- `max_depth`: int [1, 3] (tightened from earlier [2, 6]; XGBoost's sequential boosting compounds depth nonlinearly across rounds, so depth 3 across 50 boosting rounds already produces substantial nonlinear capacity, and depth 6 in this regime memorises residuals)
- `learning_rate`: log-uniform [0.01, 0.3] (floor at 0.01; below this, training takes forever and the model effectively underfits)
- `min_child_weight`: int [5, 30] (floor raised from 1 to align with RF's leaf-size discipline; with ~900 train samples a `min_child_weight=1` permits trees to split off single-event leaves)
- `subsample`, `colsample_bytree`: uniform [0.6, 1.0]
- `gamma`: log-uniform [1e-8, 1.0] (tightened upper bound; large gamma rarely helps on weak-signal financial data)
- `reg_alpha`, `reg_lambda`: log-uniform [1e-8, 10.0]
- `n_estimators` fixed at 500 with early stopping (20 rounds)

**`tune_lstm(X, y, w, n_features, n_trials=None)`** — Search space:
- `hidden_size`: int [16, 32] step 16 (capped further from 64 after empirical underperformance at higher capacities)
- `num_layers`: fixed at 1 (no longer searched; tightened from earlier [1, 2] then [1, 3]; two- and three-layer LSTMs on ~1,150 events are deep-overfit territory and the additional layer added variance to path-Sharpes without improving accuracy. Hardcoding to 1 frees Optuna trials for finer exploration of dropout and learning_rate)
- `dropout`: uniform [0.1, 0.5] (floor raised from 0.0 for regularization)
- `learning_rate`: log-uniform [1e-4, 5e-2]

**LSTM tuning sensitivity (May 2026).** A wider search space was tested in parallel with the KAN sensitivity check: `hidden_size ∈ {16, 32, 48, 64}`, `num_layers ∈ [1, 2]`, `dropout ∈ [0.0, 0.5]`. The expanded configuration produced a 0.11 decline in median Sharpe over the seven CPCV paths, consistent with the same memorisation hypothesis. The narrow ranges are retained as the locked thesis configuration.

**`tune_kan(X, y, w, n_features, n_trials=None)`** — Search space:
- `width1`: int [2, 6] (tightened from earlier [3, 12] then [3, 16]; the cap is set at 6 to keep the symbolic formula extracted in Phase 3 humanly readable, since each surviving width1 unit becomes one additive term plus interactions in the closed-form expression)
- `width2`: fixed at 0 (no longer searched; tightened from earlier [0, 10]; one hidden layer only, so the Phase 3 symbolic formula does not nest trigonometric primitives in trigonometric primitives, which would otherwise produce fourth-order compositions that lose interpretability. Hardcoding `width2=0` also ensures the architecture used for CPCV evaluation matches the architecture extracted in Phase 3, so the symbolic formula reflects the actual benchmark model rather than an unrelated KAN topology)
- `lr`: log-uniform [5e-4, 5e-2]
- `weight_decay`: log-uniform [1e-5, 5e-3]
- `grid`: categorical {3, 5} (dropped grid=8 to prevent memorisation)

**KAN tuning sensitivity (May 2026).** A wider search space was tested as a sensitivity check against the locked narrow ranges above: `width1 ∈ [3, 8]`, `width2 ∈ [0, 3]`, `grid ∈ {3, 5, 7}`, `lr ∈ log [5e-4, 5e-3]`. The expanded configuration produced a 0.07 decline in median Sharpe over the seven CPCV paths, with KAN's std_sharpe roughly unchanged. The decline is consistent with the small-sample memorisation hypothesis that motivated the original narrow ranges (deeper / wider KAN configurations passed inner-fold tuning but generalised slightly worse to the held-out CPCV test fold). The narrow ranges are retained as the locked thesis configuration; the expanded result is reported as a robustness check confirming the tuning ranges were not arbitrarily chosen.

**Optuna configuration:** TPE sampler with `seed=42` for reproducibility. MedianPruner with `n_startup_trials=5` (classical) or `n_startup_trials=3` (neural), `n_warmup_steps=1`. Pruner kills trials whose intermediate log loss after any inner fold falls below the median of all completed trials, saving compute on clearly bad regions.

**Rationale for capped search spaces:** With ~600 training samples per CPCV split, overly flexible architectures guarantee overfitting. Each cap was chosen to match parameter count to sample size (e.g., a `[22, 20, 15, 2]` KAN has ~8,500 parameters for 480 samples; capping widths drops this to manageable levels).

**Production-tuning consistency.** Tuning uses identical configurations to production along the dimensions that affect what gets learned: same window length (14 for LSTM), same warm restart schedule (T_0=25 for LSTM, T_0=30 for KAN), same loss function (cross-entropy with label smoothing 0.1, sample weights, class weights). Hyperparameters tuned under one regime would be suboptimal under another, so consistency matters along these axes.

The LSTM tuning loop deliberately runs fewer epochs and a shorter patience window than production (tuning: epochs=50, patience=7; production: epochs=100, patience=15) to keep the per-trial cost bounded across hundreds of inner-fold fits. The tuned hyperparameters are then re-fitted at the production budget. Hyperparameter rankings under the shorter budget have correlated well with full-budget rankings on this dataset, but the divergence is documented because it is the one axis where tuning and production differ.

#### `tune_all_models(X, y, w, n_features, models, seed, verbose, n_trials) → dict`

Orchestrates per-split tuning for a list of models. Returns `{model_name: tune_result}`. Called inside `pipeline.py`'s split loop; tuned parameters are then applied as module-level constants for model training in that split.

---

### Step 13 — Model Training (`models/`)

All six models implement the `BaseModel` abstract interface from `base.py`:

```python
class BaseModel(ABC):
    def __init__(self, n_features, n_classes=2, seed=42)
    def fit(self, X_train, y_train, sample_weight=None, X_val=None, y_val=None)
    def predict_proba(self, X) → np.ndarray        # (n_samples, n_classes)
    def predict(self, X) → np.ndarray               # argmax of predict_proba
    def get_name(self) → str                        # for logging/comparison
```

**Label convention:** Pipeline maps {-1, +1} → {0, 1} before passing to models. All models work with 0-indexed classes. Evaluation maps back.

The model registry in `models/__init__.py` provides `create_model(name, n_features, seed)` and `list_models()`.

#### Benchmarks (`benchmarks.py`)

| Model | Class | Key Behavior |
|-------|-------|-------------|
| **AR Logistic** | `ARLogistic` | Selects the 10 precomputed lag columns (`log_returns_lag1` … `log_returns_lag30`) from the pre-MDA feature matrix by name and ignores everything else. The lag columns are produced once on the full daily series by `pre_cpcv.features.compute_lag_features`. AR Logistic receives the pre-MDA matrix via the pipeline's `X_tr_full` route, so it always sees its lag columns regardless of whether MDA happens to select them for the other models. `predict_logits` returns log-odds via `log(p₁/p₀)` with a symmetric `np.clip(proba, 1e-10, 1 − 1e-10)` matching the tree-model convention. NaN lag columns at predict time raise (the previous inline-build path silently `bfill()`-imputed and is gone). Not tuned (deterministic baseline). |
| **Logistic Regression** | `LogisticRegressionModel` | Standard sklearn LogisticRegression, `class_weight='balanced'`, solver chosen based on penalty (lbfgs for L2, liblinear for L1). Tuned per split. `predict_logits` returns `decision_function` (raw log-odds). |

AR Logistic uses `LOGISTIC_MAX_ITER=1000` and L2 penalty as hardcoded defaults. Logistic Regression's `C` and `penalty` are tuned per split.

#### Tree Models (`tree_models.py`)

| Model | Class | Tuned Params | Fixed Params |
|-------|-------|-------------|-------------|
| **Random Forest** | `RandomForestModel` | n_estimators, max_depth, min_samples_leaf, max_features | max_features='sqrt' default, balanced_subsample, n_jobs=-1 |
| **XGBoost** | `XGBoostModel` | max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda | n_estimators=500 with early stopping at 20 rounds, binary:logistic, scale_pos_weight from class balance |

XGBoost's `predict_logits` converts proba to log-odds via `log(p₁/p₀)` with clipping at 1e-10.

#### LSTM (`lstm_model.py`)

**Architecture:** Single-layer `nn.LSTM` (`num_layers=1` hardcoded, `hidden_size` ∈ [16, 32], `dropout` ∈ [0.1, 0.5]) → last hidden state from the final layer → LayerNorm → dropout → linear classifier. The hidden size, dropout, and learning rate are tuned per split; `num_layers` is no longer searched after empirical evidence that the second layer added path-Sharpe variance without improving accuracy on this dataset.

**Last-hidden-state pooling.** The final timestep's hidden state from the last LSTM layer serves as the sequence representation. An earlier version used learned temporal attention pooling (weighted sum across all timesteps), but it was removed: with a 14-day window and ~900-sample folds, the additional attention parameters did not improve performance and the simpler standard approach proved more robust.

**Tanh input normalization.** Features are tanh-normalized: `z = tanh((x - μ) / σ)`. Maps features to [-1, 1] regardless of original scale, stabilizing training on fat-tailed financial data. Mean and std are fitted on training data only and stored for inference.

**Sequence construction:** `create_sequences(X, y, w, window=14)` reshapes 2D features into 3D windowed sequences of shape `(T-13, 14, n_features)`. First 13 observations are dropped (insufficient lookback). Returns `valid_indices` mapping sequences back to original positions. Window length is intentionally close to the 10-day TBL labeling horizon — longer windows attenuate gradient signal across recurrent steps and increase the parameter-to-sample ratio on small folds.

**Training stack:** AdamW (lr tuned, weight_decay=1e-4), CrossEntropyLoss with class weights and AFML sample weights (per-sample weighted loss), label smoothing (0.1), gradient clipping (max norm 1.0), cosine annealing warm restarts (T_0=25, T_mult=2), batch size=64, max 100 epochs, early stopping patience=15 on validation loss with best-state restoration.

**Tuning consistency.** `LSTMClassifier.__init__` reads `LSTM_HIDDEN_SIZE`, `LSTM_NUM_LAYERS`, `LSTM_DROPOUT` at call time (not as default arguments), ensuring tuning overrides via `lstm_mod.LSTM_HIDDEN_SIZE = ...` actually reach the model. The earlier default-argument pattern was a silent no-op for architectural tuning; now fixed.

**Pipeline interaction:** `last_valid_indices` attribute stores the index mapping after `predict_proba`/`predict_logits`. The pipeline uses this to align LSTM predictions with original timestamps (LSTM produces fewer predictions than other models). Calibration handles this via `calibrator.fit_from_logits()` with pre-aligned y_cal.

#### KAN (`kan_model.py`)

**Architecture:** `efficient_kan.KAN(layers_hidden=[n_features, width1, 2], grid_size=grid, spline_order=3, grid_range=[-1, 1])`. Width is tuned per split (`width1` ∈ [2, 6]) and grid size (∈ {3, 5}). The second hidden layer is permanently disabled (`width2=0` hardcoded in tuning.py): the CPCV-evaluated KAN matches the single-hidden-layer architecture used in the Phase 3 symbolic extraction, so the formula extracted reflects the same model the benchmark numbers describe.

**Input normalization:** Tanh normalization fitted on training data: `z = tanh((x - mean) / (std + ε))`. Maps features into [-1, 1] to match the spline grid range. Stored parameters applied at inference time.

**Training stack:** AdamW (lr and weight_decay tuned), CrossEntropyLoss with class weights and AFML sample weights, label smoothing (0.1), gradient clipping (max norm 1.0), cosine annealing warm restarts (T_0=30, T_mult=2), early stopping patience=20 on validation loss with best-state restoration. Max 200 epochs. Single grid level throughout training (no coarse-to-fine refinement).

**Why no SWA or entropy regularization.** Earlier experimentation included Stochastic Weight Averaging and entropy-of-prediction regularization. SWA conflicted with early stopping (either early stopping terminates before SWA activates, or SWA overrides `best_state` with potentially worse weights). Entropy regularization was redundant with `label_smoothing=0.1` (both discourage confident predictions). Both were removed for coherence and to simplify the methodology defense.

**Why a single grid level.** Unlike the literature's coarse-to-fine schedule (start at grid=3, refine to grid=5 mid-training), this implementation trains at a single grid level throughout. With ~900 training samples, grid refinement adds parameters faster than the data can support, causing memorization. Single-grid training is more stable.

**Dual-library strategy.** efficient-kan is used for CPCV training/inference across all 28 splits (fast, reliable, integrates with standard PyTorch tooling). PyKAN is re-trained independently for symbolic extraction only (Phase 3). This avoids the PyKAN parameter and training fragility while leveraging its symbolic features. Both libraries share the same B-spline basis and tanh input normalization.

---

### Step 14 — Calibration (`calibration.py`)

**Two methods, auto-selected by model type:**

| Method | Models | Input | Mechanism |
|--------|--------|-------|-----------|
| **Platt scaling** (Platt 1999) | AR Logistic, Logistic Regression, Random Forest, XGBoost | 1D log-odds | Fits `LogisticRegression(C=1e10)` mapping logits → calibrated proba (slope and intercept) |
| **Vector scaling** (Guo et al. 2017, §4.2) | LSTM, KAN | 2D logits (n, n_classes) | Fits temperature `T` and per-class bias `b` minimising NLL of `softmax((logits + b) / T)` via `scipy.optimize.minimize(L-BFGS-B)` with bounds `T ∈ [0.05, 20]`, `b_c ∈ [-5, 5]` |

Both methods are two-parameter (Platt) or three-parameter (vector). Each can correct both miscalibration sharpness *and* directional bias. `fit_temperature_scaling` is retained in the module for reference and unit tests but is no longer the default for any model.

**`Calibrator` class** provides a unified interface: `fit(model, X_cal, y_cal)` auto-detects method from `model.get_name()`, `calibrate(logits)` applies the fitted calibration. For LSTM, `fit_from_logits(logits, y_cal, method)` handles the index-aligned calibration data. Method tags: `"platt"`, `"vector"` (default for PyTorch), `"temperature"` (opt-in for backward compatibility).

Calibration is fitted on the held-out 20% of the training fold (chronological split, no shuffling). Never touches test data. If calibration fails (logged as warning), the pipeline falls back to uncalibrated `predict_proba`.

**Methodological note: why vector scaling rather than temperature scaling.** An earlier implementation used pure temperature scaling for PyTorch models. A calibration audit before final evaluation revealed that both LSTM and KAN exhibited systematic under-prediction of P(y=1) by 10-23 percentage points across the bulk of the predicted-probability distribution, while the empirical base rate of class 1 was approximately 0.55. Pure temperature scaling preserves the argmax of the raw logits by construction, so a single-parameter `T` cannot shift a "lean class 0" prediction to "lean class 1" no matter what value it takes. The bias propagated through bet sizing as systematic short bets in regimes where the market drifted upward, contributing to the negative path-level Sharpe ratio that an early version of the KAN equity curves displayed. Vector scaling adds a per-class bias `b` that lifts the directional constraint and is the natural extension recommended by Guo et al. (2017) for cases where temperature scaling alone is insufficient. The substitution was made before the final evaluation pass and constitutes a correction of methodological inadequacy rather than test-set-informed model selection.

**Methodological note (calibration set dual role):** The 20% calibration subset serves a dual role: early-stopping monitor for XGBoost and input for Platt/vector scaling. Since early stopping only controls ensemble size (no individual tree decisions are influenced by the cal set), and each calibration method fits at most three parameters, this shared use introduces minimal information leakage. Splitting the already-small cal set (~180 observations at N=8) further would degrade both purposes.

---

### Step 15 — Pipeline Orchestration (`pipeline.py`)

**Single entry point:** `run_cpcv_pipeline(X, y, w, t1, bins_ret, ..., tune=False, tune_models=None, n_trials=None)`

**Module-default reset.** The first thing the function does is call `_reset_module_defaults()`, which on the first invocation snapshots the pristine import-time values of every module-level constant that `_apply_tuned_params` mutates (LOGISTIC_C, RF_N_ESTIMATORS, KAN_HIDDEN, etc.) and on every subsequent invocation restores those snapshots. Without this, a second `run_cpcv_pipeline` call would inherit the previous run's tuned values: e.g. running once with tuning, then re-running without tuning, would give the second run's "untuned" models the first run's tuned widths. The tracked-constants list is in `_TRACKED_CONSTANTS` and mirrors exactly what `_apply_tuned_params` writes; new tuned hyperparameters must be added to both.

**Execution flow per split:**
1. Extract fold data using positional indices
2. `preprocess_fold()` (shared across all models): FFD → scale → select
3. Re-align y, w, t1 after FFD may drop NaN rows
4. Keep `X_tr_full` (all columns) for AR Logistic alongside `X_tr_sel` (selected) for other models
5. Chronological 80/20 split of training into model-train + calibration
6. If `tune=True`: run nested Optuna tuning on `X_tr_sel` using `tune_all_models()`, apply results to module-level constants
7. For each model × each seed:
   - `create_model(name, n_features, seed)`
   - Route correct X (full for AR Logistic, selected for others)
   - `model.fit(X_fit, y_model, sample_weight=w_model, X_val=X_c, y_val=y_cal)`
   - Calibrate (special LSTM handling via `fit_from_logits` for index alignment)
   - `model.predict_logits(X_test)` → `calibrator.calibrate(logits)` → argmax for predictions
   - Handle LSTM index mismatch via `last_valid_indices`
   - Compute inline metrics: F1 macro, AUC-ROC, log-loss
   - Store: y_true, y_pred, cal_proba, timestamps, returns, prep_info, calibrator repr, tuned_params

**Storage key:** `(model_name, split_idx, seed)` tuple.

**Models run separately in the notebook** (Section 4): AR Logistic first (~5s, no tuning), then LR, then RF+XGBoost, then LSTM, then KAN. Results merged in Section 4.6 via dictionary union before passing to post-CPCV evaluation.

**Error handling:** Failed model fits are caught, logged, and skipped (don't crash the pipeline). Summary prints successful/failed task counts.

---

### Model Specifications Summary

| Model | Seeds | Tuned per split | Notes |
|-------|-------|-----------------|-------|
| AR Logistic | 3 | No | Fixed: C=1.0, L2, lags [1,2,3,7,14,30]. Consumes precomputed lag columns from `pre_cpcv.features.compute_lag_features`; not subject to MDA. |
| Logistic Regression | 3 | Yes (30 trials) | C, penalty |
| Random Forest | 3 | Yes (30 trials) | n_estimators ≤ 300, depth, leaf, max_features |
| XGBoost | 3 | Yes (30 trials) | 8 params, depth ≤ 6, early stopping |
| LSTM | 2 | Yes (30 trials) | hidden ∈ {16, 32}, layers fixed at 1, dropout ∈ [0.1, 0.5], lr; window=14. Tuning runs at epochs=50, patience=7; production refits at epochs=100, patience=15. |
| KAN | 2 | Yes (30 trials) | width1 ≤ 6, width2 fixed at 0, grid ∈ {3,5}, lr ∈ [5e-4, 5e-2], weight_decay ∈ [1e-5, 5e-3] |

---

## Phase 3 — Post-CPCV (Evaluation & Interpretability)

Takes the raw predictions dictionary from Phase 2 and produces the thesis deliverables: model comparison, statistical robustness tests, diagnostics, and KAN symbolic formulas.

### Notebook sections: 5 (Post-CPCV Evaluation) → 6 (Symbolic Extraction)

---

### Step 16 — Evaluation (`evaluation.py`)

**Module-level constants:**
```
TRANSACTION_COST = 0.001          # 0.1% round-trip for BTC
MIN_BET_SIZE = 0.05               # |bet| below this → don't trade
MAX_BET_SIZE = 0.75               # cap on raw bet (after S-curve, before discretization)
BET_DISCRETIZATION = [0.0, 0.25, 0.50, 0.75]   # 1.0 dropped, consistent with the 0.75 cap
ANNUALIZATION_FACTOR = 365        # used by the Sharpe formula and DSR de-annualisation
RISK_FREE_RATE = 0.0              # assume 0 for crypto
```

#### Step 16.a — Split-level metrics (`compute_split_metrics`)

For each split's test fold: accuracy, F1 macro, F1 per class (class 0 = bearish, class 1 = bullish), precision macro, recall macro, log-loss, Brier score, AUC-ROC. All computed with sample weights where applicable. Handles single-class test folds (AUC returns NaN).

#### Step 16.b — Bet sizing (`bet_size_from_proba`)

Implements De Prado's S-curve (AFML Chapter 10.3):
1. Direction: `+1 if P(up) > P(down), else -1`
2. Confidence: `p = max(P(class_0), P(class_1))`, always in [0.5, 1.0]
3. Z-score: `z = (p - 0.5) / sqrt(p(1-p) + 1e-10)`
4. Raw bet: `2 × Φ(z) - 1` where Φ is the standard normal CDF
5. Maximum bet cap: `np.clip(raw_bet, -0.75, 0.75)` to prevent the highest-confidence predictions from dominating the equity curve
6. Minimum threshold: `|bet| < 0.05 → 0` (the "don't trade" decision)
7. Discretization: snap to nearest of {0.0, 0.25, 0.50, 0.75}
8. Apply sign: `bet = direction × |bet|`

**This is where the abstention mechanism lives.** Predictions with p ≈ 0.50 produce bet ≈ 0, meaning no capital is allocated despite having a prediction.

#### Step 16.c — Strategy returns (`compute_strategy_returns`)

For each observation: `gross_return = bet_size × label_return`, `turnover = |Δbet_size|`, `tx_cost = 0.1% × turnover`, `net_return = gross - tx_cost`. Returns a pd.Series indexed by timestamps.

#### Step 16.d — Path stitching (`stitch_paths`)

Assembles 7 full-span backtest paths from the 28 splits using the path-assignment matrix. For each path:
1. Collects `(group_id, split_id)` pairs from `path_map[path_id]`.
2. For each pair, retrieves the corresponding split's stored predictions (calibrated probabilities and returns) **and filters down to the events whose positional index falls within `group_bounds[group_id]`**. Each split's stored test set covers `k=2` chronological groups concatenated, so this filter is essential: without it, events from co-tested groups get pulled into the path multiple times.
3. With multiple seeds, calibrated probabilities are averaged across seeds before bet sizing (ensemble averaging reduces prediction variance by ~1/√n_seeds).
4. Concatenates chronologically, sorts by timestamp.
5. Asserts no duplicate timestamps after the group filter; emits a warning if any are detected so future regressions in `path_map` construction surface immediately.
6. Computes bet sizes → strategy returns → path performance.

`stitch_paths` accepts `event_index` and `group_bounds` as optional inputs. When not supplied, both are derived from `predictions` via `_derive_event_index` (union of all stored timestamp slices, sorted and de-duplicated) and `_compute_group_bounds` (mirroring the helper in `cv.py`). The orchestrator (`analyze_results`) computes them once and passes them to every per-model stitch call.

**Bug-fix disclosure.** An earlier implementation pulled each split's full test set whenever the split was referenced, double- or multiple-counting events from groups co-tested with the requested group. The bug surfaced as a duplication pattern in the stitched series (groups appearing more times than their CPCV path assignment intended) and was identified by direct timestamp inspection. The fix is the group filter described above. All path-level metrics in this thesis use the corrected stitching; the bug-fix history is preserved in the methodology chapter as a transparency disclosure.

#### Step 16.e — Path performance (`compute_path_performance`)

Per-path financial metrics:

| Metric | Formula |
|--------|---------|
| Annualized Sharpe | `(mean_r / std_r) × √365` |
| Annualized Sortino | `(mean_r / downside_rms) × √365`, where `downside_rms = sqrt(mean(neg_returns²))` over the strictly-negative returns. Returns inf for paths with no losing days and positive mean (winning streak), 0 if mean is also zero. |
| Calmar | `annualized_return / |max_drawdown|`. Tail-risk-aware alternative to Sharpe; returns inf when `|max_drawdown|` is exactly zero and ann_ret is positive. |
| Cumulative return | `∏(1 + rₜ) - 1` |
| Annualized return | `(1 + cum_ret)^(1 / years_elapsed) − 1`, where `years_elapsed = (timestamps[-1] − timestamps[0]).days / 365.25` (calendar-time CAGR; previously used `(1+cum_ret)^(365/n_events)` which assumed n was the number of daily bars and over-stated ann_ret by ~5–6× for typical multi-year paths) |
| Maximum drawdown | `min((equity - running_max) / running_max)` |
| Time under water | Longest consecutive run where equity < running_max (days) |
| Win rate | Fraction of traded observations with positive return |
| Profit factor | `Σ(positive returns) / |Σ(negative returns)|`. Three-way logic: NaN if `n_trades == 0` (undefined for an empty path), `inf` if there are trades but no losses, normal ratio otherwise. |
| n_trades | Count of observations where bet_size ≠ 0 |
| n_returns | Length of the strategy-returns series (= total events in the path, including zero-bet rows). Used by DSR's `n_obs` so it matches the n that estimated Sharpe. |
| years_elapsed | Calendar span of the path (used by the annualized-return formula) |
| Average bet size | Mean \|bet\| among traded observations |
| Skewness / kurtosis | Distribution shape of strategy returns |

#### Step 16.f — Deflated Sharpe Ratio (`compute_deflated_sharpe`)

Implements AFML Chapter 14. Corrects observed Sharpe for selection bias (n_trials = 6 models) and non-normal returns:

```
E[max SR] = √(2·ln(n)) × (1 - γ/(2·ln(n))) + γ/(2·√(2·ln(n)))
SR_std    = √((1 - skew×SR + (kurt + 2)/4 × SR²) / (n_obs - 1))
DSR       = Φ((SR_observed - E[max SR]) / SR_std)
```

**Kurtosis convention.** The Mertens (2002) variance formula assumes raw kurtosis (γ_4 = 3 for normal). `scipy.stats.kurtosis` returns *excess* kurtosis (γ_4 - 3 = 0 for normal). The implementation converts internally: `(γ_4 - 1)/4 = (excess + 3 - 1)/4 = (excess + 2)/4`. For normal returns (skew=0, excess=0), this correctly reduces to `1 + SR²/2` per the standard Lo (2002) approximation.

**`n_obs` source.** `compute_model_summary` passes `avg_n_returns` (mean across paths of `len(strategy_returns)`) as `n_obs`, matching the n used to estimate the Sharpe ratio. Earlier versions used `avg_n_trades` (subset where `bet_size ≠ 0`), which understated `n_obs` and inflated `sr_std` because the Mertens correction divides by `(n_obs − 1)`; the previous convention was conservative but inconsistent with the Sharpe estimation horizon.

**Sharpe annualisation convention.** The Sharpe ratio uses `(mean / std) × √365`, treating each event's return as a daily-equivalent observation. Strategy returns are sampled at CUSUM event timestamps, not at every daily bar (~75 events per year, not 365), so the bet-frequency annualisation `√(N_events / years_elapsed) ≈ √75 ≈ 8.7` would give roughly half the reported Sharpe. The implementation is internally consistent (DSR de-annualises by the same `√365` factor, so DSR verdicts and model rankings are convention-invariant), but the absolute Sharpe values reported in the comparison table sit on the daily-equivalent scale rather than the bet-frequency scale. The methodology chapter discloses this choice explicitly.

**NaN safety clamp.** The variance term `inner` is clamped to a minimum of 1e-10 before the square root, preventing `sqrt(negative)` for edge cases with extreme skew.

DSR > 0.95 → result survives multiple-testing correction. DSR < 0.95 → may be a statistical artifact.

#### Step 16.g — Probability of Backtest Overfitting (`compute_pbo`)

Implements AFML Chapter 11 via CSCV. Takes `path_sharpes_matrix` of shape (6 models, 7 paths):
1. Generates all C(7, 3) = 35 IS/OOS partitions of the 7 paths
2. For each partition: identifies the IS-best model, checks if it underperforms the OOS median
3. PBO = fraction of partitions where IS-best underperforms OOS

PBO < 0.3 → robust selection. PBO > 0.5 → anti-predictive (in-sample winner is out-of-sample loser).

#### Step 16.h — DeLong pairwise AUC tests (`compute_auc_significance`)

For each pair of models, tests the null hypothesis that their AUCs are equal using the DeLong (1988) method:

1. For each (model, split), averages predicted probabilities across all available seeds (3 for every model in the locked configuration). This matches what `stitch_paths` already does for path-level financial metrics, so the AUC test reflects the same averaged predictions that the Sharpe/DSR/PBO results are computed from.
2. Pools the seed-averaged predictions across all 28 CPCV splits per model. Pooling is valid because CPCV test sets are non-overlapping.
3. Computes AUC for each model on the pooled data.
4. Uses the non-parametric covariance estimator (`_delong_covariance`) via placement values (midranks).
5. Computes z-statistic: `z = (AUC_a − AUC_b) / sqrt(Var(AUC_a) + Var(AUC_b) − 2·Cov(AUC_a, AUC_b))`.
6. Two-sided p-value from standard normal.

An earlier version of `compute_auc_significance` used only `seed=0` predictions, which made the AUC values and z-statistics depend on which initialisation happened to be labelled seed 0. Averaging across seeds before pooling removes this arbitrary dependence and uses the full available signal.

Returns a DataFrame with columns: `model_a`, `model_b`, `auc_a`, `auc_b`, `delta_auc`, `z_stat`, `p_value`, `significant` (at α=0.05). The notebook reports "X/Y pairs significantly different" as a top-line robustness statistic.

#### Step 16.i — Model comparison table (`compare_models`)

Ranks models by median path Sharpe (primary, descending) with std Sharpe as tiebreaker (ascending, prefer consistency).

**Columns:** `rank`, `model_name`, `median_sharpe`, `std_sharpe`, `sharpe_ci_lower`, `sharpe_ci_upper`, `dsr`, `median_sortino`, `median_calmar`, `mean_f1`, `mean_accuracy`, `mean_auc_roc`, `median_max_dd`, `median_cum_return`, `median_win_rate`, `median_profit_factor`. The `sharpe_ci_lower` and `sharpe_ci_upper` columns hold the bounds of a non-parametric 95% confidence interval on the median Sharpe (1000 bootstrap resamples; details in 16.j). `median_sortino` and `median_calmar` are tail-risk-aware alternatives to Sharpe added so the comparison table doesn't reduce to a single risk-adjustment lens.

#### Step 16.j — Model summary aggregation (`compute_model_summary`)

Per model: pools path-level metrics (median/mean/std Sharpe, median Sortino, median Calmar, median drawdown, win rate, profit factor), split-level metrics (mean F1, accuracy, log-loss, AUC-ROC, Brier), and computes DSR using pooled skewness/kurtosis from all paths.

**Two distinct `n`'s.** The summary distinguishes `n_trades` (subset where `bet_size ≠ 0`, used for win rate and profit factor) from `n_returns` (full event series including zero-bet rows, used for Sharpe and DSR). DSR's `n_obs` is set to `avg_n_returns` so it matches the n used to estimate the Sharpe ratio. Earlier versions conflated these.

**Bootstrap confidence interval on the median Sharpe.** `compute_model_summary` calls `bootstrap_median_ci(sharpes, n_bootstrap=1000, alpha=0.05, seed=42)` per model and stores the bounds as `sharpe_ci_lower` and `sharpe_ci_upper` in the summary dict. The bootstrap resamples each model's seven path Sharpes with replacement 1000 times, computes the median for each resample, and returns the 2.5th and 97.5th percentiles of the resulting median distribution. Non-finite entries (NaN, inf) are dropped before resampling so paths with undefined Sharpe don't poison the percentile estimate. The CI complements DSR (parametric, AFML-corrected for selection bias and non-normality) with a non-parametric robustness check: a model whose CI crosses zero has a median Sharpe that is not statistically distinguishable from zero under simple resampling, regardless of what DSR says.

**Profit factor aggregation.** The median profit factor uses `np.nanmedian` rather than `np.median` because the per-path profit factor is NaN for paths with zero trades (the metric is mathematically undefined when there are no trades). Plain `np.median` propagates NaN: if any single path has NaN, the median collapses to NaN even when the other six paths have valid finite values. `np.nanmedian` skips the NaN entries and computes the median over the paths that actually traded. Inf entries (winning-streak paths with no losing trades) are kept and treated as legitimate large values during the median calculation. An earlier version used plain `np.median` and produced spurious NaN values for models whose calibrated probabilities clustered tightly around 0.5 on at least one path (typically the linear baselines, where the S-curve thresholding occasionally produced zero-bet paths).

#### Step 16.k — Buy-and-hold benchmark (`compute_buy_and_hold_summary`)

Produces a model-summary-compatible row for buy-and-hold using the same CPCV path structure the models see. For each path, the function reconstructs the chronological sequence of test-fold returns from the predictions dict (the timestamps and per-event returns are model-invariant for a given split/seed, so any model serves as a reference; `reference_model="logistic"` is the default). The benchmark holds a long position of size 1.0 over every event in the path, incurring transaction cost only on the initial buy. Path-level metrics are computed identically to the model paths so the benchmark row is directly comparable in the comparison table.

The benchmark is a more aggressive position-size baseline than the models, which cap at `MAX_BET_SIZE = 0.75` via the S-curve. This asymmetry is conservative for the model side: a fully-leveraged long benchmark is harder to beat than a 0.75-leveraged one. The benchmark contextualises the model Sharpe ratios against the trivial strategy of holding BTC throughout each test path; the resulting row in the comparison table answers the "did your models actually beat just holding?" question directly. Predictive metrics (F1, accuracy, AUC, log loss, Brier) are NaN for the benchmark row since buy-and-hold makes no probabilistic predictions; DSR is also NaN because the metric requires a Sharpe under multiple-trials selection, and buy-and-hold was not selected from a pool.

The benchmark is added by the notebook in section 5.1 immediately after `analyze_results(results)`: the user calls `compute_buy_and_hold_summary(...)`, appends the resulting dict to `analysis["all_summaries"]`, and re-runs `compare_models(...)` to regenerate the ranked comparison with the benchmark row sorted into the leaderboard by median Sharpe.

#### Step 16.l — Display helpers (`render_ffd_stability`, `render_deflated_sharpe_table`, `render_pbo_summary`)

Three thin renderer functions that consume `analysis` (the dict returned by `analyze_results`) and `results` (from `run_cpcv_pipeline`) and print formatted tables to the notebook output. They factor out the FFD-stability / DSR / PBO rendering logic that was previously inlined as a long notebook cell, so the statistical-robustness cell now reads as three calls instead of forty lines of print formatting.

- `render_ffd_stability(analysis)` prints a per-FFD-column table with mean d*, std d*, and the min/max observed d* across folds. Wide min-max spread suggests the FFD parameter is regime-dependent; tight spread indicates robust identification of the column's stationarity threshold.

- `render_deflated_sharpe_table(analysis, threshold=0.95)` prints the per-model DSR sorted descending, with a verdict column (`✓ pass` / `fail`) against the threshold. Models with NaN DSR (e.g. the buy-and-hold benchmark, which is not selected from a candidate pool) are rendered as `n/a` and sorted last. The threshold is a kwarg in case a sensitivity check at e.g. 0.90 is desired.

- `render_pbo_summary(analysis, results)` prints the baseline PBO followed by a leave-one-out table that recomputes PBO after excluding each model in turn, ranked by the magnitude of the resulting PBO change. A large negative `Δ vs baseline` means the excluded model contributes substantially to overall pool overfitting; a small or zero `Δ` means the model is a minor contributor; a positive `Δ` means removing the model worsens the pool's stability.

The notebook usage is:

```python
render_ffd_stability(analysis)
print()
render_deflated_sharpe_table(analysis)
print()
render_pbo_summary(analysis, results)
```

These are presentation-layer helpers only; all the statistical computation (FFD d*, DSR, PBO, leave-one-out PBO) happens in `analyze_results` and the underlying `compute_pbo`. The renderers exist solely to keep the notebook readable.

#### Diagnostics

**Feature stability** (`compute_feature_stability`): counts how often each feature is selected across all `(split, seed)` pairs for the first non-AR reference model. Because `prep_info` is computed once per split and stored under every `(model, split, seed)` key, all seeds for a given split contribute the same selection list — the seed loop scales numerator and denominator equally, leaving the per-feature frequency identical to a seed=0-only count but more symmetric with the AUC and FFD-stability diagnostics. Features selected in > 80% of folds are flagged as "stable." The notebook plots this as a horizontal bar chart.

**FFD stability** (`compute_ffd_stability`): collects d* values across all `(model, split, seed)` entries. FFD is shared across models within a fold and deterministic given the training fold, so the only meaningful source of dispersion is across-split training-fold variation. Including all seeds yields a denser histogram without changing the qualitative result; mean and population std are unaffected by the per-fold replication. Warns if std > 0.1 (heterogeneous stationarity structure across time periods).

#### Top-level orchestration (`analyze_results`)

Called from notebook as `analysis = analyze_results(results)`. Chains: split metrics → path stitching → path performance → model summaries → comparison table → PBO → DeLong AUC significance → feature stability → FFD stability. Returns a dictionary with all results for the notebook's visualization cells (confusion matrices, equity curves, Sharpe box plots, feature stability bar chart, FFD d* values, DeLong significance heatmap).

Each component (`compute_pbo`, `compute_auc_significance`, `compute_feature_stability`, `compute_ffd_stability`, `stitch_paths`, `compute_split_metrics`, `compute_model_summary`, `compare_models`) is also exposed as a standalone function. The notebook's results section can call them à la carte for individual subsections without going through the full orchestration. This is useful when iterating on a single subsection's figure without re-running the entire analysis.

---

### Step 17 — Symbolic Extraction (`symbolic_extraction.py`)

**1,412 lines.** The most complex file in the project. Re-trains a PyKAN model independently from efficient-kan, following Algorithm 1 from the VIX KAN paper, with extensive robustness engineering for PyKAN's fragile APIs.

**Architecture.** Inherits `KAN_HIDDEN=5`, `KAN_GRID=5`, `KAN_K=3` from `kan_model.py` as architecture defaults but applies a data-aware safety floor that can reduce these based on training-sample count (`PYKAN_MIN_SAMPLES_PER_PARAM=5`). The two libraries share the same B-spline basis and the same tanh input normalisation, but no per-instance state is passed between them — the symbolic pipeline reconstructs everything it needs from `prep_info` (FFD d* values, the fitted scaler, the selected feature list).

**Module-level constants (PyKAN-specific, not shared with efficient-kan):**
```
Training:
  PYKAN_ADAM_STEPS=600, PYKAN_ADAM_LR=1e-3, PYKAN_ADAM_WEIGHT_DECAY=1e-3
  PYKAN_NOISE_STD=0.05 (Gaussian noise injection as regularizer)
  PYKAN_LBFGS_STEPS=40, PYKAN_LBFGS_LR=0.01, PYKAN_LBFGS_WARMUP_FRAC=0.5
  PYKAN_LAMB=0.002, PYKAN_LAMB_L1=1.0, PYKAN_LAMB_ENTROPY=2.0
  PYKAN_GRID_INIT=3, PYKAN_GRID_EXTEND=False (disabled, too few samples)
  PYKAN_MIN_SAMPLES_PER_PARAM=5 (data-aware architecture sizing)
  PYKAN_MIN_ACCURACY=0.53 (gate: must beat random by 3%)

Symbolic:
  PRUNE_THRESHOLD=0.01
  SYMBOLIC_LIBRARY=['x','x^2','x^3','x^4','exp','log','sqrt','tanh',
                    'sin','cos','abs','sgn','arctan','0']
  SYMBOLIC_R2_THRESHOLD=0.3
  AFFINE_FINETUNE_STEPS=30, AFFINE_LR=0.0004
```

#### `run_symbolic_extraction(cpcv_results, X, y, w, t1, n_top_features=None, use_multkan=False, fold_selection="best", feature_selection_strategy="per_fold")`

Top-level entry point. Four control parameters:

| Parameter | Options | Effect |
|-----------|---------|--------|
| `n_top_features` | `None` or int (e.g., 5) | If set, caps the number of features used for symbolic extraction. The cap is enforced according to `feature_selection_strategy`. Fewer features → simpler formulas. |
| `feature_selection_strategy` | `"per_fold"` (default) / `"stability"` | Controls how features are chosen for the symbolic re-training. See Step 17.b below. |
| `use_multkan` | `False` / `True` | If True, uses MultKAN (KAN 2.0) with multiplication nodes, enabling discovery of multiplicative interactions (e.g., RSI × Stoch_K). Same symbolic pipeline works for both. |
| `fold_selection` | `"best"` / `"last"` / int | Which fold to use: best F1, most recent, or specific index. |

**Pipeline:**

#### Step 17.a — Fold selection (`select_extraction_fold`)

Scans all `(kan, split_idx, seed)` predictions, averages F1 across seeds per split, selects by strategy. Also retrieves `prep_info` (FFD d*, scaler, selected features) from that fold.

#### Step 17.b — Feature selection (`select_features_for_extraction`, `rank_features_by_stability`)

The symbolic re-training picks features via one of two strategies, controlled by `feature_selection_strategy`:

- **`"per_fold"` (default).** Uses the MDA selection from the chosen extraction fold itself (the same features the CPCV-evaluated KAN was trained on for that fold). Since MDA runs once per fold in `pipeline.py` and the selection is shared across all models within a fold, this strategy ensures the symbolic formula represents the actual KAN whose performance is reported in the comparison table rather than an idealised KAN trained on a different feature set. If `n_top_features` is set and the per-fold selection contains more features than the cap, the cap is enforced by ranking the fold's selection by cross-fold stability so the final pick is the intersection of "selected on this specific fold" and "consistently selected across other folds". The notebook output reports each chosen feature alongside its cross-fold stability percentage so a reader can see whether the per-fold selection broadly aligns with the stability strategy or made a fold-idiosyncratic choice.

- **`"stability"` (legacy).** Counts how often each feature was selected across all KAN CPCV folds and returns `[(feature_name, selection_frequency)]` sorted descending. When `n_top_features` is set, only the top N are used. This was the default in earlier iterations of the pipeline and reflects features that are robustly important across the dataset's history, at the cost of potentially selecting features the chosen extraction fold did not actually train on.

The per_fold strategy is the new default because it is more methodologically faithful to the CPCV evaluation: the symbolic formula corresponds to the same input space the comparison-table KAN saw on the chosen fold. The stability strategy remains available as a sensitivity check, and the docstring documents the trade-off so a reader of the source can understand why the choice was made.

#### Step 17.c — Data preparation (`prepare_extraction_data`)

Reconstructs the extraction fold's preprocessed data:
1. Resolves CPCV configuration (`n_groups`, `k`, `embargo_pct`) and either re-generates the splits or accepts them from the caller
2. Applies stored FFD d* values to full series, extracts training fold
3. Applies stored scaler transform
4. Selects features (stored selection or explicit subset override)
5. 80/20 chronological split into model-train / validation
6. **Tanh normalization** fitted on training split: `z = tanh((x - mean) / (std + ε))`, matching efficient-kan's input preprocessing

Returns PyKAN-format dataset dict with normalized float32 tensors.

**CPCV-config threading.** The function accepts `splits`, `n_groups`, `k`, and `embargo_pct` as optional parameters and resolves them in this priority order: explicit `splits` → explicit `n_groups`/`k`/`embargo_pct` → values stored in `cpcv_results["split_info"]` → `ValueError`. The previous version called `generate_cpcv_splits(X, t1)` with no arguments and silently fell back to the cv.py module defaults; if the module defaults differed from what `cpcv_results` was actually trained against, the function would silently regenerate the wrong fold (different N produces different group boundaries, so `splits[best_split_idx]` would point to a completely different training set). The current design refuses to guess: if the configuration cannot be resolved from any of the three sources, the function raises rather than silently using a mismatched fold.

#### Step 17.d — PyKAN training (`train_pykan`)

**Faithful-by-default architecture.** The function reads the tuned hyperparameters from `cpcv_results["tuning_results"][best_split]["kan"]["best_params"]` and uses them verbatim for the symbolic re-training (subject to a data-aware safety floor described below). Three resolution rules:

1. *width1* uses the tuned value, capped at `PYKAN_SYMBOLIC_WIDTH_CAP = 8` (the locked KAN tuning maximum is 6, so the cap never bites in practice; it remains as a guard against future tuning expansions producing widths the symbolic step cannot handle).
2. *width2* uses the tuned value when `PYKAN_SYMBOLIC_DROP_WIDTH2 = False` (the locked default). The locked KAN tuning hardcodes width2=0, so the symbolic re-training also uses width2=0; no two-hidden-layer compositions are produced. Setting `PYKAN_SYMBOLIC_DROP_WIDTH2 = True` is available as a legacy override that forces width2=0 even if a future tuning expansion picks width2 > 0; it is unused in the locked configuration.
3. *grid* uses the tuned value when `PYKAN_SYMBOLIC_FORCE_GRID = None` (the locked default). The locked KAN tuning searches grid ∈ {3, 5}, so the symbolic re-training inherits whichever of those won on the chosen fold. `PYKAN_FALLBACK_GRID = 3` is used only as a last-resort fallback when no tuned grid is available (e.g., CPCV ran without tuning).

This faithful-by-default behaviour replaces an earlier configuration that hardcoded `PYKAN_SYMBOLIC_DROP_WIDTH2 = True` and `PYKAN_SYMBOLIC_FORCE_GRID = 3` on the rationale that simpler architectures produce more readable formulas. With width2 already hardcoded to 0 in the CPCV tuning stage and grid ∈ {3, 5} producing tractable formulas in both cases, the architecture used for CPCV evaluation now matches the architecture extracted in Phase 3 by construction; the "force grid=3" override would only diverge when the tuned grid was 5, in which case forcing it to 3 produces a less faithful formula for marginal interpretability gains. The legacy override constants are retained so the user can fall back to the simpler-formula behaviour if the tuned configuration produces a sympy-intractable formula on a given fold.

**Data-aware safety floor.** Independent of the tuned width, the function applies a samples-per-parameter floor. If `n_train / total_params < 5`, the hidden width is reduced. For ~350 training samples (after the 80/20 cal split) with grid=3 and k=3, this typically caps hidden width at 4-5 regardless of the tuned value.

**Three-phase training protocol:**

| Phase | Optimizer | Steps | Key Feature |
|-------|-----------|-------|-------------|
| 1. Adam | Adam (lr=1e-3, wd=1e-3) | 600 | Gaussian noise injection (`std=0.05`) on inputs each step, clamped to [-1,1]. Acts as dropout-like regularizer. Early stopping on validation loss. |
| 2a. LBFGS warmup | LBFGS (lr=0.01) | 20 | No regularization. Light refinement only. |
| 2b. LBFGS sparsity | LBFGS (lr=0.01) | 20 | L1 + entropy regularization via `model.regularization_loss()`. Encourages sparse, interpretable activations. |

Grid extension is **disabled** (`PYKAN_GRID_EXTEND=False`) because with ~350 samples, increasing the grid further adds parameters and causes memorization. Only recommended for datasets > 1,000 samples.

**Accuracy gate:** If validation accuracy < 53% after Adam phase, logs warning but continues. Symbolic extraction may yield constants in this case.

**Diagnostic checkpoints:** Logs train/val accuracy after each phase, including the resolved width1, width2, and grid alongside their respective sources (`tuned`, `forced`, `fallback`, or `data-aware`).

#### Step 17.e — Pruning (`prune_network`)

1. Forward pass to populate cached activations
2. Edge survival analysis: counts active edges (above threshold) vs total
3. `model.attribute()` for importance scoring
4. `model.prune(threshold=0.01)` with multi-API fallback (PyKAN versions vary)
5. Verifies pruned model can still forward-pass; reverts if broken
6. Post-prune accuracy check + network visualization saved to `cache/kan_pruned_network.png`

Typical result: `[12, 5, 2] → [5, 3, 2]` or similar compression.

#### Step 17.f — Symbolification (`symbolify_network`)

The most complex function (~350 lines) due to PyKAN's inconsistent API. For each surviving edge:

1. `model.suggest_symbolic(l, i, j, topk=5, lib=SYMBOLIC_LIBRARY)` — tries custom library first
2. Falls back to PyKAN's built-in default library if custom fails (`KeyError` on unrecognized function names)
3. **Parses suggestions through three format handlers** (PyKAN versions return different types):
   - DataFrame (rows = candidates)
   - Flat tuple `(fn_name, lambdas, R², complexity)`
   - Nested tuple of tuples
4. **Constant-skip logic:** If `"0"` wins (due to zero complexity in total_loss), runs brute-force: tries each candidate via `fix_symbolic`, captures R² from PyKAN's stdout via regex `r"r2 is ([\d.eE+-]+)"`, keeps the best non-constant function
5. If best R² ≥ 0.3: `model.fix_symbolic(l, i, j, best_fn)`
6. If best R² < 0.3: keeps spline (partial symbolification)

**Affine fine-tuning** (Step 4 of Algorithm 1): After symbolification, trains remaining affine parameters (a, b, c, d per symbolic edge) with LBFGS for 30 steps at lr=0.0004. NaN detection reverts to pre-fine-tune state.

Saves `cache/kan_symbolified_network.png`.

#### Step 17.g — Formula extraction (`extract_formulas`)

1. Builds SymPy variable list, tries three naming conventions: `x1..xn` (no underscore, 1-based), `x_1..x_n` (underscore, 1-based), `x_0..x_{n-1}` (underscore, 0-based)
2. `model.symbolic_formula(var=...)` — falls back to no-var call
3. Extracts `logit_bearish` (class 0) and `logit_bullish` (class 1)
4. Substitutes placeholder variables with actual feature names
5. **Simplification with 30-second timeout** (sympy can hang on complex expressions) via threading
6. `nsimplify` for cleaner rational coefficients
7. Computes post-symbolic accuracy
8. Identifies surviving features (those appearing in `decision.free_symbols`)

**Output:**
```python
{
    'logit_bearish': str,              # e.g., "0.34*tanh(2.1*FFD_close + 0.7)"
    'logit_bullish': str,
    'decision_function': str,          # logit_bull - logit_bear
    'p_up_formula': str,               # "1 / (1 + exp(-(decision)))"
    'sympy_objects': {bearish, bullish, decision},  # SymPy expression objects
    'pre_symbolic_accuracy': float,
    'post_symbolic_accuracy': float,
    'symbolification_rate': float,     # fraction of edges symbolified
    'pruned_architecture': list,       # e.g., [5, 3, 2]
    'surviving_features': list,        # features in the final formula
}
```

**Notebook usage** (Section 6):
```python
symbolic = run_symbolic_extraction(results, X, y, w, t1,
                                   n_top_features=5, fold_selection="last",
                                   feature_selection_strategy="per_fold")

# Decision-function summary
print_symbolic_decision(symbolic)

# Pre/post symbolic accuracy + symbolification rate + pruned architecture
print_extraction_metrics(symbolic)

# Symbolic partial derivative of the decision function w.r.t. each feature
print_partial_derivatives(symbolic)

# Per-feature sensitivity at the dataset mean (with NaN-safe handling)
sensitivity_df = compute_feature_sensitivity(symbolic, X, eval_point="mean")
print_feature_sensitivity(sensitivity_df)

# Marginal-effect plot for each surviving feature
fig = plot_marginal_effects(symbolic, X)
plt.show()

# Term-count + sensitivity summary (n_terms_in_formula joined with sensitivity)
print_term_structure_summary(sensitivity_df, symbolic)
```

The `print_*`, `compute_feature_sensitivity`, and `plot_marginal_effects` helpers live in `symbolic_extraction.py` so the notebook stays slim; what was previously a six-cell block of inline sympy / matplotlib formatting is now seven one-line calls.

**Singular-gradient handling (`compute_feature_sensitivity`).** PyKAN's symbolic library includes `1/x`, `log(x)`, and similar reciprocal/logarithmic primitives that produce poles in the learned activation. When the symbolic gradient is evaluated at a point near such a pole (which can happen for heavily right-skewed features whose mean lands far from the bulk of the distribution, e.g. the `jarque_bera` feature with mean ≈ 218 and std ≈ 775), `float(deriv.subs(point))` returns `inf` or `-inf`, which then propagates silently through the sigma_effect and approx_delta_p columns. The `_safe_eval_at_point` helper detects non-finite gradients and substitutes NaN, which downstream formatters render as the string `"   N/A "` rather than `"+nan"`. The function reports a count of singular features in a footer line and accepts an `eval_point="median"` kwarg that evaluates gradients at the per-feature median instead of the mean, which is more robust for skewed distributions and typically avoids the singularity. An earlier version of this code computed gradients with no NaN-safety, producing `-inf` rows in the sensitivity table for any feature whose distribution had a near-pole at the mean; the fix is documented in code comments and presented as a known PyKAN-symbolic-library quirk in the methodology chapter.

**Known fragilities** (documented in code comments):
- `'sigmoid'` is NOT in PyKAN's internal `SYMBOLIC_LIB` → causes `KeyError`
- `'1/x'` causes division-by-zero at affine fine-tuning
- PyKAN uses 1-based variable naming (`x_1..x_n`), not 0-based
- `suggest_symbolic` return format varies across PyKAN versions
- `sympy.simplify` can hang indefinitely on complex expressions (→ 30s timeout)
- `"0"` (constant) always wins `total_loss` due to zero complexity penalty (→ brute-force fallback)

**Defensive input handling at `prepare_extraction_data`.** The function coerces `y` to a `pd.Series` indexed on `X.index` before any indexing, regardless of whether the caller passed a Series, a numpy array, or another array-like. If the supplied `y` length does not match `X`, the function raises a clear `ValueError` rather than letting pandas's generic length-mismatch error propagate. This catches a common notebook pattern where `y` gets shadowed by a pooled-prediction array (e.g., from a calibration audit cell that does `y = np.concatenate(...)`) and fails fast with a message naming the alignment requirement.

---

### Step 18 — Diagnostics (`diagnostics.py`)

A separate module in `post_cpcv/` housing interactive inspection helpers for the notebook's results section. None of these functions belong to the AFML evaluation protocol per se; they exist to support specific arguments in the thesis (calibration audit, regime-concentration check, bet-size distribution, reliability diagrams) and to keep that argument-supporting code out of `evaluation.py`.

All functions operate on `cpcv_results["predictions"]` or `analysis["path_results"]` and return DataFrames or arrays. They never touch the canonical event-aligned `X / y / w / t1` series, so they cannot accidentally shadow notebook globals.

**Function families:**

#### Step 18.a — Calibration audit
- `pool_predictions(model, results, n_seeds, n_splits) → (proba, y)`: concatenates calibrated `P(class=1)` and ground-truth labels across all `(split, seed)` combinations for a given model.
- `calibration_audit(model, results, n_seeds, n_splits, n_bins=10)`: prints a binned predicted-vs-empirical comparison table. Reports only bins with at least 10 samples. The diagnostic that exposed the temperature-vs-vector-scaling issue.

#### Step 18.b — Path-level dispersion and regime concentration
- `compute_top_k_concentration(returns, k=5) → dict`: quantifies how much of a path's cumulative return comes from the top-K largest-magnitude returns. Returns `cum_full`, `cum_ex_top_k`, `top_k_share`, `top_k_dates`, `top_k_values`, `date_range`. A high concentration share (e.g., > 50%) indicates regime-fluke.
- `build_path_dispersion_table(analysis, k=5) → DataFrame`: assembles a `(model, path)` indexed DataFrame with Sharpe, cumulative return, drawdown, `top_k_share`, and the date range of the top-K returns. One row per (model, path).
- `summarize_path_dispersion(dispersion) → DataFrame`: collapses to one row per model with `sharpe_min/median/max`, `cum_min/median/max`, and `avg_top_k_share`.

#### Step 18.c — Bet-size distribution
- `compute_bet_size_summary(analysis, min_bet_size=0.05, max_bet_size=0.75) → DataFrame`: per-model abstention rate, mean and median absolute bet size among traded events, share at the cap, long/short balance.
- `collect_bet_sizes(analysis, model) → np.ndarray`: raw bet-size array for a given model, pooled across paths. Convenience helper for histogram plotting.

#### Step 18.d — Reliability curves
- `compute_reliability_curve(model, results, n_seeds=None, n_splits=None, n_bins=10, min_count=10) → DataFrame`: returns binned `(predicted_mean, empirical_mean, n_samples)` triples ready for plotting as a reliability diagram. `n_seeds` and `n_splits` default to `None`; when not passed, they are read from `results["n_seeds"]` and `results["n_splits"]` so the diagnostic stays in sync with the pipeline configuration without requiring the caller to track those values. Pass explicit integers to override (e.g. for a sensitivity check that pools a subset of seeds). This default change replaced an earlier version where these arguments were hardcoded to `n_seeds=2, n_splits=15`, which silently produced incorrect pooling once the locked configuration moved to `n_seeds=3` and `n_splits=28`. The same default-resolution pattern is used by `pool_predictions` and `calibration_audit` in this module.

**Why a separate module.** `evaluation.py` is the canonical AFML evaluation pipeline (DSR, PBO, DeLong, comparison table). It runs once per CPCV experiment and produces the headline numbers. `diagnostics.py` holds inspection helpers called interactively while writing the thesis. Different lifetimes, different stability requirements; separating them keeps `evaluation.py` focused.

**Notebook usage:** Each function corresponds to one paragraph or one figure in the thesis chapters. `calibration_audit` populates the methodology chapter's calibration section. `build_path_dispersion_table` and `compute_top_k_concentration` populate the results chapter's path-dispersion and regime-concentration analysis. `compute_bet_size_summary` plus `collect_bet_sizes` populate the trading-behaviour subsection. `compute_reliability_curve` provides the visual companion to the calibration audit table.

---

## Leakage Prevention Summary

| Risk | Mitigation | Where |
|------|-----------|-------|
| FFD d* computed on full data | d* estimated per fold on training data only | `preprocessing.py` |
| Scaler fitted on full data | RobustScaler fitted on training fold only | `preprocessing.py` |
| Feature selection on full data | Multi-model MDA run on training fold only | `preprocessing.py` |
| Hyperparameter tuning on full data | Nested Optuna TPE inside each training fold (3-fold Purged K-Fold) | `tuning.py` |
| Overlapping triple-barrier labels | Purging removes training obs whose t1 extends into test | `cv.py` |
| Serial correlation across CV boundary | Embargo removes buffer after test boundary | `cv.py` |
| Calibration on test data | Calibrator fitted on held-out training partition (20%), never on test | `calibration.py` |
| Shared cal set for XGB early stop + calibration | Acknowledged trade-off; only ensemble size affected, no tree decisions | `pipeline.py` |
| On-chain data look-ahead | CoinMetrics data shifted by 1 day before alignment | `external_features.py` |
| CUSUM threshold on full data | Minor approximation (h uses full-sample vol mean); acknowledged, negligible impact | `labeling.py` |
| NaN imputation across train-test boundary | `ffill().bfill()` applied independently within each partition; FFD columns drop NaN rather than impute | `preprocessing.py` |