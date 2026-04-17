# BTC Daily Direction Prediction Using KANs Within the AFML Framework
## Project Structure & Pipeline Architecture

**Student:** Petr Terletskiy (l63023)
**Programme:** Masters in Mathematical Finance, ISEG (2024/26)

---

## Overview

The project predicts Bitcoin daily price direction using Kolmogorov–Arnold Networks (KANs), benchmarked against AR Logistic, Logistic Regression, Random Forest, XGBoost, and LSTM models, evaluated under López de Prado's *Advances in Financial Machine Learning* (2018) framework. The pipeline is organized into three phases with strict leakage boundaries between them.

**Data:** BTC-USD daily OHLCV, October 2014 – March 2026 (~4,200 bars).
**Features:** 55 total (23 technical, 9 mathematical/AFML Part 4, 23 external: 13 macro, 2 crypto-macro, 8 on-chain from CoinMetrics).
**Evaluation:** CPCV (N=6, k=2) producing 15 splits and 5 backtest paths, with Deflated Sharpe Ratio, Probability of Backtest Overfitting, and DeLong pairwise AUC significance tests.
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
│   │   └── alignment.py                    ← index intersection, validation assertions
│   │
│   ├── cpcv/                               ← PHASE 2: Cross-Validation & Training
│   │   ├── __init__.py
│   │   ├── cv.py                           ← CPCV splits, purging, embargo, path matrix
│   │   ├── preprocessing.py                ← per-fold FFD, RobustScaler, multi-model MDA feature selection
│   │   ├── tuning.py                       ← Optuna TPE + Purged K-Fold nested per-split tuning
│   │   ├── calibration.py                  ← Platt scaling, temperature scaling
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

**Parameters used in notebook:** `ticker="BTC-USD"`, `start="2014-10-01"`, `end="2026-03-27"`

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

**Parameters in notebook:** `h = 1.5 × daily_vol.mean()`. The multiplier of 1.5 was chosen as a balance: `0.5×` produces ~3,000 noisy events, `3.0×` reduces to ~200 (too few for ML), `1.5×` yields ~900 events with roughly balanced classes.

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

**Parameters in notebook:** `pt_sl=(1.5, 1.5)`, `num_days=10`, `min_return=0.0`.

**Parameter choice rationale:**
- Symmetric `pt_sl=(1.5, 1.5)` avoids imposing any prior directional bias. With σ ≈ 3% daily, barriers sit at roughly ±4.5% from entry.
- `num_days=10` gives horizontal barriers meaningful time to trigger without letting labels go stale. Observed mean holding period is ~5.1 days, indicating horizontal barriers hit roughly half the time before the vertical barrier.

---

### Step 4 — Drop Rare Labels (`labeling.py`)

**Function:** `drop_rare_labels(bins, min_pct=0.05) → pd.DataFrame`

Implements AFML Snippet 3.8. Counts class frequencies, drops all rows belonging to classes below the threshold. With current parameters (symmetric barriers, min_return=0.0), class 0 is eliminated, producing binary labels {-1, +1}.

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

**Parameters in notebook:** `time_decay_factor=0.5` (oldest sample's weight is halved), `weight_cap_quantile=0.99`.

---

### Step 6 — Feature Engineering (`features.py`, `external_features.py`)

**Produces:** A pd.DataFrame of 55 features (23 TA + 9 mathematical + 23 external) covering every daily bar.

**Module-level constants (all feature parameters defined at top of each file):**

```
TA: RSI_PERIOD=14, MACD_FAST=12, MACD_SLOW=26, MACD_SIGNAL=9, BB_PERIOD=20,
    ATR_PERIOD=14, ROLLING_WINDOW=21, EMA_SHORT=20, EMA_MID=50, EMA_LONG=200,
    ROC_PERIOD=14, STOCH_PERIOD=14, STOCH_SMOOTH=3, CCI_PERIOD=14, MFI_PERIOD=14,
    CHAIKIN_FAST=3, CHAIKIN_SLOW=10, YZ_WINDOW=21

Math: SADF_MIN_SL=63, SADF_LAGS=1, ENTROPY_WINDOW=21, LZ_WINDOW=63,
      HURST_WINDOW=126, VR_WINDOW=63, VR_LAG=5, JB_WINDOW=63, GAUSS_ENT_WINDOW=21

Cache: CACHE_DIR="cache/", MATH_CACHE_FILE="math_features.parquet",
       EXTERNAL_CACHE_FILE="external_features.parquet", ONCHAIN_CACHE_FILE="onchain_raw.parquet"
```

#### Step 6.a — TA Features (`features.py`)

**Function:** `compute_ta_features(df) → pd.DataFrame` — 23 columns

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
| 9 | `skewness` | Rolling 21-day skewness of log returns | Dimensionless |
| 10 | `kurtosis` | Rolling 21-day excess kurtosis of log returns | Dimensionless |
| 11 | `realized_vol` | Rolling 21-day std of log returns × √365 | Annualized, dimensionless |
| 12 | `gk_vol` | Garman-Klass: `0.5·ln(H/L)² − (2ln2−1)·ln(C/O)²`, rolling 21-day mean | Dimensionless |
| 13 | `yz_vol` | Yang-Zhang (2000) combining overnight, open-to-close, and Rogers-Satchell components. `k = 0.34/(1.34 + (w+1)/(w−1))` weighting, rolling 21-day | Dimensionless |
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

**Internal helpers:** `_yang_zhang_volatility()` implements the Yang-Zhang (2000) estimator. `_money_flow_index()` implements the volume-weighted RSI calculation.

#### Step 6.b — Mathematical Features (`features.py`)

**Function:** `compute_math_features(df, which="all") → pd.DataFrame` — 9 columns

All features are computationally expensive (O(n²) for SADF/SMT). The function checks for a cached Parquet file before computing; if cache exists with matching date range and requested columns, it loads from cache. Otherwise computes, saves, and returns.

The `which` parameter accepts `"all"` or a list of specific feature names for partial computation.

| # | Feature | AFML Reference | Method | Window | Approx. Time |
|---|---------|---------------|--------|--------|--------------|
| 1 | `shannon_entropy` | Ch. 18.2 | Quantile-encode returns into 5 bins, compute plug-in entropy `H = −Σ pᵢ log₂(pᵢ)` | 21 days | < 1 min |
| 2 | `lz_complexity` | Ch. 18.4 | Binary-encode returns ("1" if > 0), apply Lempel-Ziv-76 algorithm, normalize by `n/log₂(n)` | 63 days | < 1 min |
| 3 | `hurst` | Related | Rescaled Range (R/S) analysis at sub-periods [10, 21, 42, 63], slope of log(R/S) vs log(n) | 126 days | < 1 min |
| 4 | `variance_ratio` | Related | Lo & MacKinlay (1988): `VR(q) = Var(r_q) / (q × Var(r₁))`. VR > 1 = momentum, VR < 1 = mean-reversion | 63 days, lag=5 | < 1 min |
| 5 | `jarque_bera` | Related | `JB = (n/6)(S² + K²/4)` where S = skewness, K = excess kurtosis. Measures departure from normality | 63 days | < 1 min |
| 6 | `gaussian_entropy` | Ch. 18.6 | `H_gauss = 0.5 × ln(2πeσ²)`. Gap vs Shannon entropy measures non-Gaussianity | 21 days | < 1 min |
| 7 | `sadf` | Ch. 17.4.2 | Supremum ADF: backward-expanding ADF regressions on log prices, take supremum of β's t-statistic | minSL=63 | ~15 min |
| 8 | `smt_poly1` | Ch. 17.4.3 | Sub/Super-Martingale polynomial-1 spec: `log[yₜ] = α + β·t + ε`, backward-expanding, supremum of `|t-stat(β)| / length^0.5` | minSL=63 | ~15 min (joint) |
| 9 | `smt_exp` | Ch. 17.4.3 | Sub/Super-Martingale exponential spec: `log[yₜ] = α + β·exp(t/length) + ε`, same supremum structure | minSL=63 | ~15 min (joint) |

**Internal helpers:** `_compute_sadf()`, `_adf_tstat()`, `_compute_smt()`, `_ols_tstat()`, `_compute_rolling_entropy()`, `_compute_rolling_lz()`, `_lempel_ziv_76()`, `_compute_rolling_hurst()`, `_hurst_rs()`, `_compute_rolling_variance_ratio()`, `_compute_rolling_jarque_bera()`, `_compute_rolling_gaussian_entropy()`.

#### Step 6.c — Macro Features (`external_features.py`)

**Function:** `compute_macro_features(btc_index) → pd.DataFrame` — 13 columns

**Alignment method:** All external data is merged onto BTC's 7-day calendar via `pd.merge_asof(direction='backward')` through the helper `_align_to_btc()`. This ensures each BTC day uses the most recent available value (no look-ahead): weekends carry Friday's macro close, weekly data persists until the next release.

Data sources: yfinance for 11 tickers plus FRED (via pandas-datareader) for 2Y Treasury yield with spread-based fallback.

| # | Feature | Source Ticker | Transformation |
|---|---------|--------------|----------------|
| 1 | `dxy_roc_21` | DX-Y.NYB (US Dollar Index) | 21-day rate of change (%) |
| 2 | `us10y` | ^TNX (10Y Treasury) | Level (yield in %), auto-scaled if median > 10 |
| 3 | `us2y` | FRED DGS2 → spread-derived fallback | Level (yield in %), auto-scaled |
| 4 | `yield_curve_2y10y` | Derived: us10y − us2y (or FRED T10Y2Y directly) | Spread (percentage points) |
| 5 | `yield_curve_10y30y` | Derived: ^TYX − ^TNX | Spread (percentage points) |
| 6 | `vix` | ^VIX | Level |
| 7 | `sp500_ret_21` | ^GSPC | 21-day log return |
| 8 | `nasdaq_ret_21` | ^IXIC | 21-day log return |
| 9 | `gold_ret_21` | GC=F | 21-day log return |
| 10 | `silver_ret_21` | SI=F | 21-day log return |
| 11 | `copper_ret_21` | HG=F | 21-day log return |
| 12 | `oil_ret_21` | CL=F | 21-day log return |
| 13 | `natgas_ret_21` | NG=F | 21-day log return |

**2Y yield fallback:** (1) FRED DGS2 directly. (2) If DGS2 returns fewer than 2,000 bars, falls back to fetching the T10Y2Y spread from FRED, using it directly as `yield_curve_2y10y` and back-deriving `us2y = us10y − spread`. This two-step fallback ensures the yield curve feature is always populated even when DGS2 is unavailable.

#### Step 6.d — Crypto-Macro Features (`external_features.py`)

**Function:** `compute_crypto_macro_features(btc_close, btc_index) → pd.DataFrame` — 2 columns

These are market-level cross-crypto signals, distinct from blockchain fundamentals (which are in Step 6.e).

| # | Feature | Source | Method |
|---|---------|--------|--------|
| 1 | `eth_btc_ratio` | yfinance ETH-USD | `ETH_close / BTC_close` aligned via merge_asof |
| 2 | `btc_dominance` | CoinGecko API → fallback: `100 / (1 + ETH/BTC)` proxy | BTC market cap from CoinGecko `/coins/bitcoin/market_chart`, with dominance proxy as fallback when the API is unavailable |

**BTC dominance fallback:** When CoinGecko API fails or returns fewer than 100 data points, computes `100 / (1 + ETH/BTC)` as a dominance proxy. This is an approximation (ignores altcoins beyond ETH) but captures the main directional signal: when ETH outperforms BTC, the proxy falls.

#### Step 6.e — On-Chain Features (`external_features.py`)

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

`build_external_features(btc_df, include_macro=True, include_crypto_macro=True, include_onchain=True) → pd.DataFrame` fetches all three categories (macro, crypto-macro, on-chain), concatenates, reports NaN summary, and caches to Parquet. Each category is independently toggleable. On-chain fetch is wrapped in try/except so missing `coinmetrics-api-client` doesn't break the pipeline.

---

### Step 7 — Log Transforms (`features.py`)

**Function:** `apply_log_transforms(features) → pd.DataFrame`

Applies `log(|x| + 1e-8)` to `atr` (always positive). Applies `sign(x) × log(|x| + 1)` to `obv` (preserves sign while compressing scale). All other features are left untouched (already bounded or dimensionless).

**Target columns:** `LOG_TRANSFORM_COLUMNS = ["atr", "obv"]`

**Orchestration:** `build_feature_matrix(df) → pd.DataFrame` chains `compute_ta_features(df)` → `compute_math_features(df)` → `pd.concat` → `apply_log_transforms`. Returns 32 columns × ~4,200 rows. External features are assembled separately in the notebook via `build_external_features(df_raw)` and concatenated before alignment.

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

Computes the triple intersection of features (all ~4,200 daily bars), bins (CUSUM-filtered ~911 events), and weights (same index as bins) via `features.index.intersection(bins.index).intersection(weights.index)`.

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

### Pre-CPCV Output Summary

| Object | Shape | Description |
|--------|-------|-------------|
| `X` | ~911 × 55 | Feature matrix (23 TA + 9 math + 23 external, post log-transform) |
| `y` | ~911 | Binary labels {-1, +1} |
| `w` | ~911 | AFML sample weights (uniqueness × return attribution × time decay, capped at 99th pctile) |
| `t1` | ~911 | Barrier touch timestamps (DatetimeIndex, for CPCV purging) |

These four objects are the contract between Phase 1 and Phase 2. Everything from FFD onward happens inside the CPCV loop, fitted on training data only.

---

## Phase 2 — CPCV (Cross-Validation & Training)

Everything in this phase runs inside the CPCV loop. Every stateful transformation (FFD d* estimation, scaling, feature selection, hyperparameter tuning) is fitted on training data only. This is the leakage-critical zone.

### Notebook sections: 3 (CV Framework) → 4 (Model Training)

---

### Step 10 — CPCV Split Generation (`cv.py`)

**Produces:** 15 train/test splits with purging and embargo, plus a 5-path assignment matrix.

**Module-level constants:** `N_GROUPS=6`, `K_TEST_GROUPS=2`, `EMBARGO_PCT=0.01`

#### `generate_cpcv_splits(X, t1, n_groups=6, k=2, embargo_pct=0.01) → list[tuple[np.ndarray, np.ndarray]]`

Partitions T observations into N=6 contiguous groups (groups 0–4 of size ⌊T/N⌋, group 5 gets the remainder). Generates all C(6,2) = 15 combinations of 2 test groups.

For each split, applies:

**Purging** (AFML Snippet 7.1): Removes training observations whose labels overlap with any test group. Checks three sufficient overlap conditions for each training observation i against each test group boundary [t_test_start, t_test_end]:
1. Training observation starts within test period: `t_test_start ≤ t_i_start ≤ t_test_end`
2. Training label resolves within test period: `t_test_start ≤ t1[i] ≤ t_test_end`
3. Training label spans the entire test period: `t_i_start ≤ t_test_start AND t_test_end ≤ t1[i]`

**Embargo** (AFML Section 7.4.2): Removes `int(embargo_pct × T)` training observations immediately after each test group boundary. Only applied after the test set (not before), since training labels resolving before test begins contain no future information.

Returns positional integer arrays into X for each split.

#### `build_path_matrix(n_groups=6, k=2) → (n_paths, path_map)`

Computes φ[N,k] = C(N-1, k-1) = C(5,1) = 5 backtest paths. For each group, collects all splits where it appears in the test set, then assigns each occurrence to a path so every path covers all N groups exactly once.

Returns `path_map: {path_id: [(group_id, split_id), ...]}` with N entries per path.

#### `get_split_info(X, t1) → dict`

Prints formatted summary: T, N, k, C(N,k), φ[N,k], embargo length, group boundaries with dates, average train/test sizes, average purged/embargoed counts. The notebook uses this for the CPCV EDA section (partition overlay on BTC price, train/test timeline visualization, purging/embargo boundary verification, full leakage audit).

---

### Step 11 — Per-Fold Preprocessing (`preprocessing.py`)

**Produces:** Scaled, FFD-transformed, feature-selected train and test DataFrames. Shared across all models within the same fold.

**Module-level constants:**
```
FFD: FFD_D_RANGE=(0.0, 1.0, 0.05), FFD_THRESHOLD=1e-4, FFD_ADF_SIGNIFICANCE=0.05, FFD_MAX_LOOKBACK=200
Selection: MDA_N_ESTIMATORS=500, MDA_N_INNER_FOLDS=3, MDA_TOP_K_FRAC=0.4, MIN_FEATURES=5
```

#### Step 11.a — FFD (`ffd_transform`)

Applies FFD to the **full** series (so test observations have lookback history) but estimates d* from **training data only**. For each column in `ffd_columns`:

1. `find_optimal_d(train_series)`: Sweeps d from 0 to 1 in steps of 0.05. At each d, computes FFD weights via `ω_0 = 1, ω_k = -ω_{k-1} × (d - k + 1) / k`, truncated when `|ω_k| < 1e-4` or k reaches 200. Applies as convolution, runs ADF test. Returns minimum d where p-value < 0.05.
2. `apply_ffd(full_series, d_star)`: Applies FFD at d* to the complete series.
3. Splits into train/test by positional indices.
4. Forward-fills NaN in non-FFD columns (external data gaps). Drops rows where FFD columns have NaN (from lookback window).

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
2. Cap at `top_k_frac` of total features (default 40%, overridable from notebook)
3. Hard floor of 5 features minimum

**Typical result:** ~18–22 features selected per fold from 55 total.

#### `preprocess_fold(X_full, train_idx, test_idx, y_train, w_train, t1_train, ffd_columns, top_k_frac, skip_selection=False)`

Orchestration chaining FFD → scaling → selection. Returns DataFrames with **all columns** (pre-selection) so the pipeline can provide full-column DataFrames to AR Logistic. The selected feature list is returned separately. The `skip_selection` flag is set when only AR Logistic is being evaluated (it constructs its own features).

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

The `run_cpcv_pipeline()` function accepts an `n_trials` parameter that overrides these defaults. Recommended values balancing exploration against runtime:

| Model | Recommended `n_trials` | Rationale |
|-------|----------------------|-----------|
| Logistic | 30 | Cheap per trial, only 2 hyperparameters |
| Random Forest | 30 | Moderate cost, 4 hyperparameters |
| XGBoost | 30 | Moderate cost, 8 hyperparameters (early stopping compensates) |
| LSTM | 20 | Expensive per trial; tighter search space |
| KAN | 20 | Expensive per trial; tighter search space |

#### `_purged_kfold_splits(X, y, w) → list[(train_idx, val_idx)]`

Creates 3 chronological inner folds with 10-observation embargo around boundaries (matches TBL num_days). Fewer folds (3 vs 5) improves runtime by ~40% and increases inner validation set size (~200 vs ~120 observations per fold), providing more reliable log loss estimates in a low-signal environment.

#### Per-model tuning functions

Each returns `{"best_params": {...}, "best_log_loss": float, "results_df": DataFrame}`:

**`tune_logistic(X, y, w, n_trials=None)`** — Search space:
- `C`: log-uniform [1e-4, 1e2]
- `penalty`: categorical {l1, l2}

**`tune_random_forest(X, y, w, n_trials=None)`** — Search space:
- `n_estimators`: int [100, 300] step 50 (capped from 500 to avoid wasteful trials on small data)
- `max_depth`: int [3, 20]
- `min_samples_leaf`: int [1, 30]
- `max_features`: categorical {sqrt, log2}

**`tune_xgboost(X, y, w, n_trials=None)`** — Search space:
- `max_depth`: int [2, 6] (capped from 10 to prevent overfitting ~900 samples)
- `learning_rate`: log-uniform [0.01, 0.3]
- `min_child_weight`: int [1, 30]
- `subsample`, `colsample_bytree`: uniform [0.6, 1.0]
- `gamma`, `reg_alpha`, `reg_lambda`: log-uniform [1e-8, 10.0]
- `n_estimators` fixed at 500 with early stopping (20 rounds)

**`tune_lstm(X, y, w, n_features, n_trials=None)`** — Search space:
- `hidden_size`: int [16, 64] step 16 (capped from 128)
- `num_layers`: int [1, 3]
- `dropout`: uniform [0.1, 0.5] (floor raised from 0.0 for regularization)
- `learning_rate`: log-uniform [1e-4, 5e-2]

**`tune_kan(X, y, w, n_features, n_trials=None)`** — Search space:
- `width1`: int [3, 12] (capped from 20)
- `width2`: int [0, 10], 0 = skip second hidden layer (capped from 15)
- `lr`: log-uniform [1e-3, 0.1]
- `weight_decay`: log-uniform [1e-5, 1e-2]
- `grid`: categorical {3, 5} (dropped 8 to prevent memorization)

**Optuna configuration:** TPE sampler with `seed=42` for reproducibility. MedianPruner with `n_startup_trials=5` (classical) or `n_startup_trials=3` (neural), `n_warmup_steps=1`. Pruner kills trials whose intermediate log loss after any inner fold falls below the median of all completed trials, saving compute on clearly bad regions.

**Rationale for capped search spaces:** With ~600 training samples per CPCV split, overly flexible architectures guarantee overfitting. Each cap was chosen to match parameter count to sample size (e.g., a `[22, 20, 15, 2]` KAN has ~8,500 parameters for 480 samples; capping widths drops this to manageable levels).

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
| **AR Logistic** | `ARLogistic` | Constructs its own features: lagged log returns at lags [1, 2, 3, 5, 10, 21]. Ignores the selected feature set entirely. Requires a `log_returns` column in X (pipeline passes pre-selection DataFrame). `predict_logits` returns log-odds via `log(p₁/p₀)`. Not tuned (deterministic baseline). |
| **Logistic Regression** | `LogisticRegressionModel` | Standard sklearn LogisticRegression, `class_weight='balanced'`, solver chosen based on penalty (lbfgs for L2, liblinear for L1). Tuned per split. `predict_logits` returns `decision_function` (raw log-odds). |

AR Logistic uses `LOGISTIC_MAX_ITER=1000` and L2 penalty as hardcoded defaults. Logistic Regression's `C` and `penalty` are tuned per split.

#### Tree Models (`tree_models.py`)

| Model | Class | Tuned Params | Fixed Params |
|-------|-------|-------------|-------------|
| **Random Forest** | `RandomForestModel` | n_estimators, max_depth, min_samples_leaf, max_features | max_features='sqrt' default, balanced_subsample, n_jobs=-1 |
| **XGBoost** | `XGBoostModel` | max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda | n_estimators=500 with early stopping at 20 rounds, binary:logistic, scale_pos_weight from class balance |

XGBoost's `predict_logits` converts proba to log-odds via `log(p₁/p₀)` with clipping at 1e-10.

#### LSTM (`lstm_model.py`)

**Architecture:** `nn.LSTM(input_size=n_features, hidden_size, num_layers, dropout, batch_first=True)` → `nn.Dropout` → `nn.Linear(hidden_size, 2)`. Takes last hidden state from final layer. All architectural parameters tuned per split.

**Sequence construction:** `create_sequences(X, y, w, window=21)` reshapes 2D features into 3D windowed sequences of shape `(T-20, 21, n_features)`. First 20 observations are dropped (insufficient lookback). Returns `valid_indices` mapping sequences back to original positions.

**Training:** Adam optimizer (lr tuned), ReduceLROnPlateau (patience=5, factor=0.5), CrossEntropyLoss with class weights and AFML sample weights (per-sample weighted loss), early stopping patience=10 on validation loss, batch size=64, max 100 epochs. Best model state restored after early stop.

**Pipeline interaction:** `last_valid_indices` attribute stores the index mapping after `predict_proba`/`predict_logits`. The pipeline uses this to align LSTM predictions with original timestamps (LSTM produces fewer predictions than other models). Calibration handles this via `calibrator.fit_from_logits()` with pre-aligned y_cal.

#### KAN (`kan_model.py`)

**Architecture:** `efficient_kan.KAN(layers_hidden=[n_features, width1, (width2), 2], grid_size=grid, spline_order=3, grid_range=[-1, 1])`. Width parameters and grid size tuned per split; typical result `[n, 5, 2]` with grid=5.

**Input normalization:** Tanh normalization fitted on training data: `z = tanh((x - mean) / (std + ε))`. Maps features into [-1, 1] to match the spline grid range. Stored parameters applied at inference time.

**Training:** AdamW (lr and weight_decay tuned), CrossEntropyLoss with class weights and AFML sample weights, early stopping patience=20 on validation loss, max 200 epochs. No grid refinement schedule (single grid level throughout training). Best model state restored after early stop.

**Key difference from blueprint:** Uses a single grid level trained with AdamW instead of the coarse-to-fine LBFGS→Adam schedule. This was found to be more stable on the ~700-sample training folds.

**Dual-library strategy:** efficient-kan is used for CPCV training/inference across all 15 splits (fast, reliable). PyKAN is re-trained independently on the best or last fold for symbolic extraction only (Phase 3). This avoids the PyKAN vs efficient-kan parameter incompatibility while leveraging PyKAN's symbolic features.

---

### Step 14 — Calibration (`calibration.py`)

**Two methods, auto-selected by model type:**

| Method | Models | Input | Mechanism |
|--------|--------|-------|-----------|
| **Platt scaling** | AR Logistic, Logistic Regression, Random Forest, XGBoost | 1D log-odds | Fits `LogisticRegression(C=1e10)` mapping logits → calibrated proba |
| **Temperature scaling** | LSTM, KAN | 2D logits (n, n_classes) | Finds scalar T minimizing NLL of `softmax(logits/T)` via `scipy.optimize.minimize_scalar` over T ∈ [0.1, 10.0] |

**`Calibrator` class** provides unified interface: `fit(model, X_cal, y_cal)` auto-detects method from `model.get_name()`, `calibrate(logits)` applies the fitted calibration. For LSTM, a separate `fit_from_logits(logits, y_cal, method)` handles the index-aligned calibration data.

Calibration is fitted on the held-out 20% of the training fold (chronological split, no shuffling). Never touches test data. If calibration fails (logged as warning), the pipeline falls back to uncalibrated `predict_proba`.

**Methodological note:** The 20% calibration subset serves a dual role: early stopping monitor for XGBoost and input for temperature scaling across all models. Since early stopping only controls ensemble size (no individual tree decisions are influenced by the cal set), and temperature scaling fits a single scalar parameter, this shared use introduces minimal information leakage. Splitting the already-small cal set (~120 observations) further would degrade both purposes.

---

### Step 15 — Pipeline Orchestration (`pipeline.py`)

**Single entry point:** `run_cpcv_pipeline(X, y, w, t1, bins_ret, ..., tune=False, tune_models=None, n_trials=None)`

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
| AR Logistic | 3 | No | Fixed: C=1.0, L2, lags [1,2,3,5,10,21] |
| Logistic Regression | 3 | Yes (30 trials) | C, penalty |
| Random Forest | 3 | Yes (30 trials) | n_estimators ≤ 300, depth, leaf, max_features |
| XGBoost | 3 | Yes (30 trials) | 8 params, depth ≤ 6, early stopping |
| LSTM | 2 | Yes (20 trials) | hidden ≤ 64, layers, dropout ≥ 0.1, lr |
| KAN | 2 | Yes (20 trials) | width ≤ 12, grid ∈ {3,5}, lr, weight_decay |

---

## Phase 3 — Post-CPCV (Evaluation & Interpretability)

Takes the raw predictions dictionary from Phase 2 and produces the thesis deliverables: model comparison, statistical robustness tests, diagnostics, and KAN symbolic formulas.

### Notebook sections: 5 (Post-CPCV Evaluation) → 6 (Symbolic Extraction)

---

### Step 16 — Evaluation (`evaluation.py`)

**Module-level constants:**
```
TRANSACTION_COST = 0.001          # 0.1% round-trip for BTC
MIN_BET_SIZE = 0.10               # |bet| below this → don't trade
BET_DISCRETIZATION = [0.0, 0.25, 0.50, 0.75, 1.0]
ANNUALIZATION_FACTOR = 365        # BTC trades every calendar day
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
5. Minimum threshold: `|bet| < 0.10 → 0` (the "don't trade" decision)
6. Discretization: snap to nearest of {0.0, 0.25, 0.50, 0.75, 1.0}
7. Apply sign: `bet = direction × |bet|`

**This is where the abstention mechanism lives.** Predictions with p ≈ 0.50 produce bet ≈ 0, meaning no capital is allocated despite having a prediction.

#### Step 16.c — Strategy returns (`compute_strategy_returns`)

For each observation: `gross_return = bet_size × label_return`, `turnover = |Δbet_size|`, `tx_cost = 0.1% × turnover`, `net_return = gross - tx_cost`. Returns a pd.Series indexed by timestamps.

#### Step 16.d — Path stitching (`stitch_paths`)

Assembles 5 full-span backtest paths from the 15 splits using the path-assignment matrix. For each path:
1. Collects `(group_id, split_id)` pairs from `path_map[path_id]`
2. Retrieves calibrated probabilities and returns for each group from the corresponding split's predictions (seed=0)
3. Concatenates chronologically, sorts by timestamp
4. Computes bet sizes → strategy returns → path performance

#### Step 16.e — Path performance (`compute_path_performance`)

Per-path financial metrics:

| Metric | Formula |
|--------|---------|
| Annualized Sharpe | `(mean_r / std_r) × √365` |
| Cumulative return | `∏(1 + rₜ) - 1` |
| Annualized return | `(1 + cum_ret)^(365/n) - 1` |
| Maximum drawdown | `min((equity - running_max) / running_max)` |
| Time under water | Longest consecutive run where equity < running_max (days) |
| Win rate | Fraction of traded observations with positive return |
| Profit factor | `Σ(positive returns) / \|Σ(negative returns)\|` |
| Number of trades | Count of observations where bet_size ≠ 0 |
| Average bet size | Mean |bet| among traded observations |
| Skewness / kurtosis | Distribution shape of strategy returns |

#### Step 16.f — Deflated Sharpe Ratio (`compute_deflated_sharpe`)

Implements AFML Chapter 14. Corrects observed Sharpe for selection bias (n_trials = 6 models) and non-normal returns:

```
E[max SR] = √(2·ln(n)) × (1 - γ/(2·ln(n))) + γ/(2·√(2·ln(n)))
SR_std = √((1 - skew×SR + (kurt-1)/4 × SR²) / (n_obs - 1))
DSR = Φ((SR_observed - E[max SR]) / SR_std)
```

**NaN safety fix:** The `SR_std` formula can produce a negative value inside the square root when `skew × SR > 1`, causing `sqrt(negative) = NaN`. The implementation clamps `inner = max(inner, 1e-10)` before sqrt to prevent this while preserving correct output for well-behaved inputs.

DSR > 0.95 → result survives multiple-testing correction. DSR < 0.95 → may be a statistical artifact.

#### Step 16.g — Probability of Backtest Overfitting (`compute_pbo`)

Implements AFML Chapter 11 via CSCV. Takes `path_sharpes_matrix` of shape (6 models, 5 paths):
1. Generates all C(5, 2) = 10 IS/OOS partitions of the 5 paths
2. For each partition: identifies the IS-best model, checks if it underperforms the OOS median
3. PBO = fraction of partitions where IS-best underperforms OOS

PBO < 0.3 → robust selection. PBO > 0.5 → anti-predictive (in-sample winner is out-of-sample loser).

#### Step 16.h — DeLong pairwise AUC tests (`compute_auc_significance`)

New in the current pipeline. For each pair of models, tests the null hypothesis that their AUCs are equal using the DeLong (1988) method:

1. Pools predicted probabilities and true labels across all 15 CPCV splits (seed=0)
2. Computes AUC for each model on the pooled data
3. Uses the non-parametric covariance estimator (`_delong_covariance`) via placement values (midranks)
4. Computes z-statistic: `z = (AUC_a − AUC_b) / sqrt(Var(AUC_a) + Var(AUC_b) − 2·Cov(AUC_a, AUC_b))`
5. Two-sided p-value from standard normal

Returns a DataFrame with columns: `model_a`, `model_b`, `auc_a`, `auc_b`, `delta_auc`, `z_stat`, `p_value`, `significant` (at α=0.05). The notebook reports "X/Y pairs significantly different" as a top-line robustness statistic.

#### Step 16.i — Model comparison table (`compare_models`)

Ranks models by median path Sharpe (primary, descending) with std Sharpe as tiebreaker (ascending, prefer consistency).

**Columns:** rank, model_name, median_sharpe, std_sharpe, DSR, mean_f1, mean_accuracy, mean_log_loss, mean_auc_roc, median_max_dd, median_cum_return, median_win_rate, median_profit_factor.

#### Step 16.j — Model summary aggregation (`compute_model_summary`)

Per model: pools path-level metrics (median/mean/std Sharpe, median drawdown, win rate, profit factor), split-level metrics (mean F1, accuracy, log-loss, AUC-ROC, Brier), and computes DSR using pooled skewness/kurtosis from all paths.

#### Diagnostics

**Feature stability** (`compute_feature_stability`): Counts how often each feature is selected across 15 folds (seed=0, first model). Features selected in > 80% of folds are flagged as "stable." The notebook plots this as a horizontal bar chart.

**FFD stability** (`compute_ffd_stability`): Collects d* values per FFD column across all folds (seed=0). Reports mean and std. Warns if std > 0.1 (heterogeneous stationarity structure across time periods).

#### Top-level orchestration (`analyze_results`)

Called from notebook as `analysis = analyze_results(results)`. Chains: split metrics → path stitching → path performance → model summaries → comparison table → PBO → DeLong AUC significance → feature stability → FFD stability. Returns a dictionary with all results for the notebook's visualization cells (confusion matrices, equity curves, Sharpe box plots, feature stability bar chart, FFD d* values, DeLong significance heatmap).

---

### Step 17 — Symbolic Extraction (`symbolic_extraction.py`)

**1,412 lines.** The most complex file in the project. Re-trains a PyKAN model independently from efficient-kan, following Algorithm 1 from the VIX KAN paper, with extensive robustness engineering for PyKAN's fragile APIs.

**Architecture:** Shares `KAN_HIDDEN=5`, `KAN_GRID=5`, `KAN_K=3` as defaults but uses data-aware sizing that can reduce these based on training sample count.

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

#### `run_symbolic_extraction(cpcv_results, X, y, w, t1, n_top_features=None, use_multkan=False, fold_selection="best")`

Top-level entry point. Three control parameters:

| Parameter | Options | Effect |
|-----------|---------|--------|
| `n_top_features` | `None` or int (e.g., 5) | If set, only the N most stable features (by CPCV selection frequency) are used. Fewer features → simpler formulas. |
| `use_multkan` | `False` / `True` | If True, uses MultKAN (KAN 2.0) with multiplication nodes, enabling discovery of multiplicative interactions (e.g., RSI × Stoch_K). Same symbolic pipeline works for both. |
| `fold_selection` | `"best"` / `"last"` / int | Which fold to use: best F1, most recent, or specific index. |

**Pipeline:**

#### Step 17.a — Fold selection (`select_extraction_fold`)

Scans all `(kan, split_idx, seed)` predictions, averages F1 across seeds per split, selects by strategy. Also retrieves `prep_info` (FFD d*, scaler, selected features) from that fold.

#### Step 17.b — Feature ranking (`rank_features_by_stability`)

Counts how often each feature was selected across all KAN CPCV folds. Returns `[(feature_name, selection_frequency)]` sorted descending. When `n_top_features` is set, only the top N are used for extraction, producing simpler formulas.

#### Step 17.c — Data preparation (`prepare_extraction_data`)

Reconstructs the extraction fold's preprocessed data:
1. Re-generates CPCV splits to get training indices
2. Applies stored FFD d* values to full series, extracts training fold
3. Applies stored scaler transform
4. Selects features (stored selection or explicit subset override)
5. 80/20 chronological split into model-train / validation
6. **Tanh normalization** fitted on training split: `z = tanh((x - mean) / (std + ε))`, matching efficient-kan's input preprocessing

Returns PyKAN-format dataset dict with normalized float32 tensors.

#### Step 17.d — PyKAN training (`train_pykan`)

**Data-aware architecture:** Computes maximum hidden width such that `n_samples / total_params ≥ 5`. With ~350 training samples and grid=3, k=3, this typically yields hidden=2–5. Prevents LBFGS memorization.

**Three-phase training protocol:**

| Phase | Optimizer | Steps | Key Feature |
|-------|-----------|-------|-------------|
| 1. Adam | Adam (lr=1e-3, wd=1e-3) | 600 | Gaussian noise injection (`std=0.05`) on inputs each step, clamped to [-1,1]. Acts as dropout-like regularizer. Early stopping on validation loss. |
| 2a. LBFGS warmup | LBFGS (lr=0.01) | 20 | No regularization. Light refinement only. |
| 2b. LBFGS sparsity | LBFGS (lr=0.01) | 20 | L1 + entropy regularization via `model.regularization_loss()`. Encourages sparse, interpretable activations. |

Grid extension is **disabled** (`PYKAN_GRID_EXTEND=False`) because with ~350 samples, increasing grid from 3 to 5 adds parameters and causes memorization. Only recommended for datasets > 1,000 samples.

**Accuracy gate:** If validation accuracy < 53% after Adam phase, logs warning but continues. Symbolic extraction may yield constants in this case.

**Diagnostic checkpoints:** Logs train/val accuracy after each phase.

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
                                   n_top_features=5, fold_selection="last")
# prints P(up) formula, surviving features, pre/post accuracy

# Feature sensitivity via partial derivatives:
for feat in symbolic['surviving_features']:
    sensitivity = sympy.diff(symbolic['sympy_objects']['decision'], sympy.Symbol(feat))
```

**Known fragilities** (documented in code comments):
- `'sigmoid'` is NOT in PyKAN's internal `SYMBOLIC_LIB` → causes `KeyError`
- `'1/x'` causes division-by-zero at affine fine-tuning
- PyKAN uses 1-based variable naming (`x_1..x_n`), not 0-based
- `suggest_symbolic` return format varies across PyKAN versions
- `sympy.simplify` can hang indefinitely on complex expressions (→ 30s timeout)
- `"0"` (constant) always wins `total_loss` due to zero complexity penalty (→ brute-force fallback)

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