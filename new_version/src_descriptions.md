# Source Code Walkthrough
## BTC Daily Direction Prediction with KANs Within the AFML Framework

This document walks through every file in `pre_cpcv/`, `cpcv/`, `cpcv/models/`, and `post_cpcv/`. Each file is described in three layers: what it does, why it exists, and the core concepts behind its design.

---

# Phase 1 — `pre_cpcv/` (Data Preparation)

This phase runs once, before any model training. It produces four aligned objects — X, y, w, t1 — that enter the CPCV loop. No stateful transformation that could leak future information occurs here; those are deferred to Phase 2. All rolling-window parameters follow the BTC calendar (7-day week, 30-day month, 90-day quarter, 180-day semester, 365-day year).

---

## `pre_cpcv/data_loader.py` — BTC OHLCV Retrieval

### What it does

Downloads daily BTC-USD OHLCV from Yahoo Finance, validates the data, and returns a clean DataFrame indexed by a timezone-naive DatetimeIndex.

### Core concepts

**Why validation matters.** Raw downloads from yfinance have quirks: MultiIndex columns in recent versions, occasional NaN closes from failed scrapes, sporadic calendar gaps, and rare OHLCV inconsistencies (High < Close, negative Volume). A silent bad download would corrupt every downstream step. The loader enforces a strict contract: if anything looks structurally wrong, it raises; if anything looks minor, it logs a warning and continues.

**Validation pipeline.** Empty downloads raise. MultiIndex columns get flattened automatically. Duplicate dates raise. Calendar gaps of up to 3 days are forward-filled (yfinance occasionally drops weekends). Gaps larger than 3 days raise — this would indicate a structural problem with the data source rather than routine noise. OHLCV row-level inconsistencies (High below max(Open,Close), Low above min(Open,Close), negative Volume) generate warnings but don't fail, since they may reflect odd but real market behavior.

**Why daily bars.** Higher frequencies (hourly, minute) introduce microstructure noise that overwhelms the signal at the horizons this thesis targets. Daily bars match the economic horizon of the feature set (macro releases are daily to weekly, on-chain snapshots are daily) and keep the sample size manageable.

### Key function: `load_btc_daily`

Returns a validated DataFrame with columns `['Open', 'High', 'Low', 'Close', 'Volume']`, indexed by a timezone-naive DatetimeIndex sorted ascending. All downstream code assumes this contract.

---

## `pre_cpcv/labeling.py` — Triple-Barrier Labeling

### What it does

Implements the AFML labeling pipeline: computes daily volatility, filters bars via CUSUM to identify meaningful events, applies triple-barrier labeling at each event, and drops rare classes.

### Core concepts

**Why fixed-time-horizon labels are insufficient.** Standard ML tutorials label "did the price go up over the next N days?" This ignores path: a bar that rose 5% then crashed 15% over 10 days looks the same as one that drifted up 1% steadily. Triple-barrier labeling conditions on which event resolves first — a profit-taking move, a stop-loss move, or time expiring.

**Daily volatility (AFML Snippet 3.1).** Exponentially weighted standard deviation of log returns with span 50. The span is calibrated for BTC's faster regime transitions — De Prado uses 100 for equities, but crypto's 24/7 trading and higher realized volatility call for a shorter-memory estimator that adapts quickly. This volatility series scales both the CUSUM threshold and the triple-barrier widths.

**CUSUM filter (AFML Snippet 2.4).** Reduces ~4,200 bars to a smaller set of "meaningful" events by tracking cumulative deviation in log returns. Two accumulators: `s_pos = max(0, s_pos + r)` tracks upside runs, `s_neg = min(0, s_neg + r)` tracks downside runs. When either exceeds the threshold h, an event fires and the accumulator resets to zero. With h = 1.0 × mean(daily_vol), the filter produces roughly 1,300-1,400 events on the BTC dataset before the start-date truncation and rare-label dropping.

The notebook applies an additional truncation step after the CUSUM filter runs, dropping events whose timestamp falls before a configured `CUSUM_START_DATE` (currently `"2015-08-08"`, the day Ethereum's Frontier launch went live and the first day with valid ETH/USD price data from CoinMetrics). The CUSUM accumulators themselves are computed on the full raw return series, so events that survive the truncation reflect the full pre-event cumulative drift; only the event index is filtered. Empirically the EWMA daily volatility used as the CUSUM threshold is fully converged by August 2015 (the mean over the truncation window matches the full-series mean to four decimal places), so the truncation drops events without distorting the threshold dynamics. The purpose of the truncation is to align the analysis window with the earliest date for which all features (including `eth_btc_ratio`) are available, which prevents asymmetric data-availability across CPCV folds.

The zero floor is what makes CUSUM a structural-break detector rather than a volatility filter. Choppy sideways action keeps resetting the accumulator and produces few events, while small-but-persistent drifts can produce many. The threshold `h = 1.0 × mean(daily_vol)` balances event count against signal strength: smaller multipliers produce noisy events, larger multipliers produce too few for ML. The notebook uses 1.0 after empirical sweeps; an earlier version used 1.5 but the lower threshold gave a richer event set without compromising class balance.

**Triple-barrier labels (AFML Snippets 3.2, 3.4, 3.5).** For each CUSUM event at time t₀:

1. Upper barrier at `close[t₀] × (1 + pt_sl[0] × σ[t₀])`.
2. Lower barrier at `close[t₀] × (1 - pt_sl[1] × σ[t₀])`.
3. Vertical barrier at t₀ + num_days.

Walks forward through the price path, records the first barrier touch. Labels: +1 (upper hit), -1 (lower hit), sign(return) at vertical (or 0 if below `min_return` threshold).

**Parameter choices.** Symmetric `pt_sl=(1.5, 1.5)` avoids imposing a directional prior. With σ ≈ 3% daily, barriers sit at roughly ±4.5% from entry. `num_days=10` gives horizontal barriers meaningful time to trigger without letting labels go stale. Observed mean holding period is ~5.1 days, with the vertical (time) barrier hit on only 19.9% of events — horizontal barriers (profit-take or stop-loss) close ~80% of positions before the time barrier triggers. The function's `min_return` parameter defaults to 0.0, but the notebook calls it with `min_return=0.02`: very small vertical-barrier returns get bucketed into class 0, which is then dropped as a rare label. This produces clean binary {-1, +1} labels without forcing every near-flat vertical-barrier return into the sign-of-return camp.

**The t1 column.** Every label carries a timestamp t1 indicating when the label was resolved (first barrier touch). This is critical for CPCV purging in Phase 2: training labels whose t1 extends into the test period must be purged to prevent leakage.

**Drop rare labels (AFML Snippet 3.8).** With symmetric barriers and the `min_return=0.02` floor, the 0 class becomes rare and is dropped. The notebook uses an 8.5% threshold (raised from the 5% function default) to be more aggressive about removing residual class-0 events. Produces binary labels {-1, +1}.

### Key function: `run_labeling_pipeline`

Chains the four steps (volatility → CUSUM → triple-barrier → rare-label drop) into one call. Returns a DataFrame with columns `['ret', 'bin', 't1']` indexed by event timestamps.

---

## `pre_cpcv/sample_weights.py` — AFML Chapter 4 Weights

### What it does

Computes per-sample weights that account for label overlap, return attribution, and time decay. These weights are passed to models' `sample_weight` parameter during training.

### Core concepts

**Why sample weighting matters here.** Triple-barrier labels can overlap extensively: a label starting at t₀ with t1 = t₀+10 shares eight bars with a label starting at t₀+2. If both labels are positive, the model sees the same "positive regime" twice, effectively double-counting information. AFML Chapter 4 provides a weighting scheme that discounts overlapping labels proportionally.

**Step 1 — Concurrent labels (Snippet 4.1).** For each bar in the daily timeline, counts how many labels are "alive" (their [t₀, t1] span includes that bar). Produces a count series c_t over the full daily index.

**Step 2 — Average uniqueness (Snippet 4.2).** For each label i spanning [t₀, t1], computes `ū_i = mean(1 / c_t)` over the bars in its span. A label with no overlap has uniqueness ≈ 1.0; a label heavily overlapping with many others approaches 0.

**Step 3 — Return attribution (Snippet 4.10).** Combines uniqueness with return magnitude: `w_i = |return_i| × ū_i`. Labels that are both unique (informative) and associated with large price moves (signal) receive higher weight. The result is normalized so `sum(weights) == len(weights)`, making the mean weight ≈ 1 for sklearn compatibility.

**Step 4 — Time decay (Snippet 4.11).** Applies linear decay via `np.linspace(oldest_weight, 1.0, n)` so that older samples weigh less. The `oldest_weight` parameter controls decay strength: 1.0 means no decay, 0.4 means the oldest sample's weight is 40% of the newest. The thesis uses 0.4, reflecting that market regimes shift over the 2014-2026 window and more recent samples are more relevant to deployment.

**Step 5 — Quantile cap.** Clips extreme weights at the 99th percentile to prevent a single high-return unique event from dominating training. Without this cap, a rare 30%-move label with uniqueness 1.0 would have weight ~30× the mean, causing the model to overfit that single event.

### Why this matters

Without these weights, overlapping labels would inflate apparent class counts and mislead the model about signal frequency. The weights align the statistical treatment with the economic reality that overlapping labels carry redundant information.

### Key function: `compute_sample_weights`

Orchestrates the four steps. Returns a pd.Series of weights indexed on event timestamps, mean ≈ 1, max capped at the 99th percentile.

---

## `pre_cpcv/features.py` — Technical, Mathematical, and Lag Features

### What it does

Computes 44 columns per bar across three categories: 25 technical-analysis features, 9 mathematical features from AFML Part 4, and 10 precomputed autoregressive lag features. Applies log transforms to extreme-scale features. All rolling-window parameters follow the BTC calendar.

### Core concepts

**Technical analysis features (25).** Standard indicators from the trading literature: log returns, RSI, MACD (and signal + histogram), Bollinger Band width, ATR, OBV, rolling skewness and kurtosis, realized volatility, Garman-Klass and Yang-Zhang volatility, EMA ratios (20/50, 50/200), VWMA ratio, Rate of Change, Stochastic %K and %D, Williams %R, CCI, Chaikin Oscillator, MFI, and volatility term structures (7/30, 30/90).

Named indicators (RSI, MACD, Bollinger, Stochastic, CCI, MFI, ATR) use their conventional parameter settings. These are fixed trading conventions, not calendar-dependent choices, so they stay at their standard values regardless of crypto vs equity calendar.

**Volatility term structure.** Two ratios: `vol_ratio_7_30` captures short-term stress (flash crashes, rapid expansions over a week relative to a month); `vol_ratio_30_90` captures regime-level shifts (multi-week regime transitions relative to a quarter). Values above 1 indicate expansion, below 1 indicate compression. These are crypto-native adaptations of the VIX term structure concept.

**Mathematical features from AFML Part 4.** Higher-order statistical measures of the return distribution and price dynamics:

1. **Shannon entropy** (Ch. 18.2): encodes returns into 10 equal-width bins, computes `H = -Σ pᵢ log₂(pᵢ)`. Measures uncertainty in the return distribution. Equal-width bins replaced an earlier quantile-bin scheme that collapsed to zero entropy whenever clustered values made all quantile edges coincide.
2. **Lempel-Ziv complexity** (Ch. 18.4): binary-encodes returns, applies LZ76 compression, normalizes. Measures algorithmic complexity — near 0 = highly predictable, near 1 = random.
3. **Hurst exponent** (Rescaled Range analysis): quantifies long-memory behavior. H > 0.5 = trending (persistent), H < 0.5 = mean-reverting.
4. **Variance ratio** (Lo & MacKinlay 1988): `VR(q) = Var(r_q) / (q × Var(r₁))`. > 1 = momentum, < 1 = mean reversion, = 1 = random walk.
5. **Jarque-Bera**: `JB = (n/6)(S² + K²/4)`. Distance from normality. High values flag fat-tailed/skewed regimes.
6. **Negentropy**: `Gaussian entropy - Shannon entropy`. The "gap" that measures non-Gaussianity of the return distribution.
7. **SADF** (Ch. 17.4.2): Supremum ADF. Backward-expanding ADF regressions on log prices, takes the supremum of β's t-statistic. Detects explosive bubble behavior.
8. **SMT poly1** and **SMT exp** (Ch. 17.4.3): Sub/Super-Martingale tests against polynomial and exponential trends. Complement SADF for different bubble shapes.

**Why these matter.** TA features capture the surface structure (momentum, mean reversion, volatility regimes). Mathematical features capture the statistical structure (entropy, complexity, long-memory, non-normality). Together they test whether KAN's adaptive activations can combine both types of information better than fixed-activation models.

**Expensive computation and caching.** SADF and SMT are O(n²) and together take ~30 minutes on the full 4,200-bar dataset. The function caches results to `cache/math_features.parquet`. The cache validator checks date range and requested columns; if window constants change but column names don't (as happened during the BTC-calendar migration), the cache must be manually deleted to avoid returning stale values.

**Compression transforms.** Two transforms are applied at the notebook level (not inside `features.py`) so the policy is visible at the call site: `np.sign(x) * np.log(np.abs(x) + 1)` (symmetric log) on `obv` and `chaikin_osc`, and `np.log(np.abs(x) + 1e-8)` on `atr`. The two volume-flow features (OBV, Chaikin) span 10⁷-10¹⁰ over the 2014-2026 period as BTC daily volume grew by six orders of magnitude; without compression, linear models fit on raw values produce vanishing coefficients on otherwise informative features because the gradient is dominated by the largest observations. Symmetric log preserves sign and zero, preserves rank ordering, and asymptotically behaves like `sign(x) · log(|x|)` for `|x| ≫ 1`. ATR uses unsigned log because it is strictly non-negative; the `1e-8` additive constant prevents `log(0)` for the rare zero-volatility bar. All other features are on bounded or dimensionless scales that the per-fold RobustScaler handles adequately.

An asinh-based variant (`np.arcsinh`) was tested in May 2026 as a smoother alternative to symmetric log, motivated by asinh's smoothness at zero (no derivative kink), parameter-free form, and the Burbidge-Magee-Robb 1988 econometric tradition for signed wide-range variables. In sensitivity analysis the asinh variant produced substantially higher path-Sharpe variance on the neural-network models and an inflated PBO (0.571 vs the symmetric-log run's 0.143), even though asinh and symmetric log are within `log(2) ≈ 0.69` of each other for `|x| ≫ 1`. The transform was reverted to symmetric log on grounds of empirical out-of-sample stability across CPCV folds. The `apply_asinh` function was removed from this module; symmetric log is the locked thesis configuration.

**NaN policy.** Every rolling-window feature produces a leading NaN run equal to its window size: 14 bars for RSI and ATR, 20 for Bollinger Bands, 30 for Shannon entropy and EMA-30, 50 for EMA-50, 90 for SADF and variance ratio, 180 for Hurst. The feature-engineering layer does not drop or impute these NaNs. Dropping at this stage would lose the early period where some features are fully computable but others are not (e.g., RSI is valid by day 15 but Hurst is not valid until day 180); imputing would fabricate values from a future-looking window. The function logs `features.isna().any(axis=1).sum()` so per-column NaN density is visible. Resolution is deferred to the per-fold preprocessing step inside the CPCV loop.

**Lag features (10).** Ten precomputed columns named `log_returns_lag1` through `log_returns_lag7`, `log_returns_lag14`, `log_returns_lag21`, and `log_returns_lag30` produced by `compute_lag_features(df)`. Constants `AR_LAGS=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30]` and `LAG_COLUMN_PREFIX="log_returns_lag"` live at module scope so other files (preprocessing, benchmarks) can reference them without hardcoding the lag list. The list was extended from an earlier `[1, 2, 3, 7, 14, 30]` configuration after observing that filling out the full 1-7 weekday cycle and adding the 21-day mid-month marker produced a better-spec'd autoregressive baseline. With ~900 training events per fold and 11 parameters (10 lags + intercept), AR Logistic still has roughly 80 events per parameter, well within healthy fitting territory.

These are consumed by the AR Logistic baseline as its complete feature set, and are also part of the global feature universe seen by the other models through MDA selection. Computing them once on the full daily series, instead of inline inside `ARLogistic.fit` / `ARLogistic.predict`, removes a look-ahead artefact that the inline version had: previously, NaN lags at the head of each test fold were imputed with `bfill()`, which used later test observations to fill earlier ones. Precomputing on the global daily series gives every aligned event valid lookback values that respect chronological order. The lag columns sit alongside TA / math / external columns in the assembled feature matrix and enter the MDA pool on equal footing.

**Why these specific lag values.** The set follows a calendar-day convention rather than the trading-day convention `[1, 2, 3, 5, 10, 21]` common in equity ML literature. BTC trades 24/7 with no weekend or holiday gaps, so trading-day arithmetic has no natural meaning here. The chosen lags map to one to three days of short-term autocorrelation, one and two calendar weeks (regulatory cycles, retail trading patterns, futures expiries), and one calendar month (macro-release cycle). The 14-day lag additionally aligns with the BTC mining-difficulty adjustment cycle (~2,016 blocks ≈ 14 days), a BTC-native cycle the equity convention does not capture.

### Key functions

- `compute_ta_features(df)`: returns the 25 TA columns.
- `compute_math_features(df, which)`: returns the 9 math columns, with caching.
- `compute_lag_features(df, lags=AR_LAGS)`: returns the 10 lag columns.
- `apply_sym_log(features, columns)`: applies `sign(x) * log(|x| + 1)` to the listed columns and returns a copy. Skips columns not present in the DataFrame so the same target list can be reused across feature-set variants.
- `apply_log(features, columns, eps=1e-8)`: applies `log(|x| + eps)` to the listed columns and returns a copy.
- `build_feature_matrix(df)`: orchestrates `compute_ta_features` + `compute_math_features` + `pd.concat`. Returns 34 columns (25 TA + 9 math), with NO compression transforms applied; the notebook applies `apply_sym_log` and `apply_log` itself with the column lists visible at the call site. External features (29) and lag features (10) are assembled separately in the notebook and concatenated alongside before alignment; the canonical concat order is `[ta, math, external, lag]`, producing the final 73-column matrix. Earlier iterations of this module included `apply_log_transforms` (and briefly `apply_asinh`) which both applied the compression transforms inside `build_feature_matrix`; both were removed in favour of the explicit notebook-level transform application.
- `lag_column_names(lags=None)`: returns the canonical column names in the order matching `lags`, used by `benchmarks.py` to pull lag columns out of `X` without hardcoding strings.

---

## `pre_cpcv/external_features.py` — Macro, Crypto-Macro, and On-Chain

### What it does

Fetches and aligns 29 external features: 20 macro variables from Yahoo Finance and FRED, 1 crypto-macro signal, and 8 on-chain metrics from CoinMetrics.

### Core concepts

**Alignment via `merge_asof(direction='backward')`.** All external data is merged onto BTC's 7-day calendar. For each BTC day, the most recent available value from each external source is used. Weekends carry Friday's equity close; weekly macro releases persist until the next print. This is the only defensible alignment method — no look-ahead bias, no fabricated values.

**Macro (20).** Traditional finance signals: Dollar Index 30-day RoC, 2-year and 10-year Treasury yields, two yield curve spreads (2s10s and 10s30s), VIX as a level, and rolling log returns at both 30-day and 14-day horizons for the seven faster-moving assets (S&P 500, Nasdaq, gold, silver, copper, oil, natural gas). The two horizons are exposed simultaneously so MDA can pick whichever the per-fold permutation importance prefers; module constants `RET_WINDOW = 30` and `RET_WINDOW_SHORT = 14` parameterise the windows. The 14-day variant captures bi-weekly cycles and is more responsive to recent news/macro shocks; the 30-day variant captures monthly cycles and is smoother. Slow-moving variables (DXY, yields, yield curves) keep the 30-day window only because their information content is dominated by monthly settlement and release cycles where a 14-day variant would add noise without signal. The 14d/30d pairs of the same asset are correlated (Pearson r typically > 0.7); MDA's permutation-importance scheme handles correlated features by either spreading importance across both or, in folds where the redundancy is unhelpful, dropping the less-informative variant entirely. Either outcome is acceptable; the methodology is "let MDA decide which horizon matters per fold". Together the 20 macro features test whether crypto responds to traditional macro regimes — risk-on/risk-off, inflation, commodity cycles — at multiple timescales.

**2-year yield fallback chain.** FRED DGS2 is the primary source. If it returns insufficient data, the code falls back to fetching the T10Y2Y spread from FRED and back-deriving `us2y = us10y - spread`. This ensures the yield curve feature is always populated even when the preferred source fails.

**Crypto-macro (1).** A single market-level cross-crypto signal, `eth_btc_ratio` (ETH close / BTC close), captures altcoin rotation. The ETH price is fetched from CoinMetrics' Community-tier API (the same source used for the on-chain BTC metrics, no additional credentials needed). The fetch tries metrics in priority order — `ReferenceRateUSD`, `PriceUSD`, then a `CapMrktCurUSD / SplyCur` derivation — and uses the first one that returns more than 100 daily rows. `ReferenceRateUSD` is currently Pro-gated for ETH on the Community tier and returns 1 row, so `PriceUSD` is the operative source in practice; it serves daily ETH/USD data back to August 8, 2015 (the day Ethereum's Frontier launch went live), giving roughly 3,917 daily rows over the ~4,200-day external dataset. yfinance's `ETH-USD` ticker remains as a final fallback if all three CoinMetrics attempts fail; the source actually used is logged at fetch time.

The earlier implementation used yfinance as the sole source. yfinance's `ETH-USD` history begins around November 9, 2017 (3,092 daily rows), leaving roughly 27% of the calendar series as NaN at the head and producing entire-test-partition NaN in early CPCV folds under N=8. With CoinMetrics PriceUSD the residual `eth_btc_ratio` NaN rate over the full external dataframe drops to 7.7%, all of it concentrated in the November 2014 → August 2015 pre-ETH-trading window. That residual NaN window is fully truncated out by the CUSUM start-date filter (`CUSUM_START_DATE = "2015-08-08"` in the labeling cell), so every CUSUM event in the analysis window has a valid ETH/BTC ratio.

An earlier version also included a second crypto-macro column, `btc_dominance`, fetched from CoinGecko's `/coins/bitcoin/market_chart` endpoint with a `100 / (1 + ETH/BTC)` proxy as fallback. Two problems forced the column out: the CoinGecko endpoint returns BTC market cap in USD rather than the bounded [0, 100] dominance percentage the column name implied, and the proxy fallback was a price-correlated approximation the methodology could not cleanly defend. `eth_btc_ratio` carries the alt-rotation signal alone. CoinGecko was also evaluated as a primary ETH source: as of 2024 the free public tier limits historical queries to the most recent 365 days only, so the 2015-2017 ETH history needed for full BTC-dataset coverage sits behind a paid tier. CoinMetrics' Community tier has no such restriction, which is why it is the chosen primary source.

**On-chain (8).** Blockchain network activity from CoinMetrics Community API: 14-day RoC of active addresses and transaction count, 30-day RoC of hashrate, MVRV ratio, net exchange flow (inflows − outflows), fee per transaction, exchange supply percentage, and daily issuance. These are BTC-native signals that equity models cannot access.

**Critical anti-leakage: on-chain data shift.** Raw CoinMetrics values are shifted by 1 day (`df.shift(1)`) before alignment. CoinMetrics reports end-of-day values, but a model predicting tomorrow's direction from today's open cannot use today's close-time metrics. The 1-day shift ensures the feature at bar t uses data that was available at the start of bar t.

**Feature transformation philosophy.** RoC features for count-type metrics (addresses, transactions, hashrate) convert non-stationary levels into stationary differentials. Level features (MVRV, yields, VIX) are kept as levels because they are already mean-reverting or bounded. No log transforms are applied to external features because their raw scales are already well-behaved (returns, rates of change, yields, ratios, percentages).

**NaN sources.** External data has scattered NaN cells from three sources. (1) Equity tickers do not trade on weekends and US holidays, so `merge_asof(direction="backward")` carries Friday's value forward but the BTC weekend rows are NaN until the merge fills them. (2) FRED macro variables are released weekly or monthly with publication delays; values are NaN between releases. (3) On-chain data has its own warm-up period and occasional missing days from API outages. The function logs per-column NaN percentages via `features.isna().mean() * 100`. As with `features.py`, NaN values are not imputed at the external-features stage; resolution is deferred to per-fold preprocessing.

### Key function: `build_external_features`

Orchestrates macro, crypto-macro, and on-chain fetches, concatenates, caches to Parquet, and returns the combined DataFrame. Each category is independently toggleable via boolean flags. The on-chain fetch is wrapped in try/except so missing `coinmetrics-api-client` doesn't break the pipeline.

The cache check verifies both the date-range endpoints and the column set. Changing the feature mix (e.g., dropping `btc_dominance`) invalidates the cache and triggers a refetch on the next run; the date-range-only check would have silently returned a stale frame.

---

## `pre_cpcv/alignment.py` — Four-Object Contract with the CPCV Loop

### What it does

Aligns the feature matrix (covering all ~4,200 daily bars) with the labels and weights (covering only the CUSUM events that survive the start-date truncation and rare-label dropping, approximately 1,150 events), producing the four objects (X, y, w, t1) that enter the CPCV loop.

### Core concepts

**Why alignment happens here.** Features are computed per-bar so that downstream Phase 2 can pull any bar's features for any reason (e.g., reconstructing preprocessing for symbolic extraction). Labels exist only at CUSUM event timestamps. Weights share the label index. The CPCV loop needs all four indexed on the same set of event timestamps.

**Triple intersection.** `features.index ∩ bins.index ∩ weights.index`, producing the common set of approximately 1,150 event timestamps. Everything downstream operates on this aligned set.

**Hard assertions.** The function raises on any structural problem: empty intersection, duplicate dates, non-monotonic index, shape mismatch across the four arrays, entire feature columns being NaN, or any NaN in `t1` (which would break CPCV purging). These are all conditions that would silently corrupt downstream results if allowed to pass.

**Soft warnings.** Partial-NaN columns trigger a logger warning naming the column count and per-column NaN counts; the pipeline continues. The buffer-and-truncate dataset structure (raw data starting November 2014, CUSUM events starting August 2015) eliminates the bulk of warm-up NaN at the head of the series: by the time the first CUSUM event fires, the longest-warmup engineered features (Hurst at 252 bars, EMA 50/200 ratio at 200, SADF at variable but bounded windows) have all completed their warmup. A handful of external columns may still have scattered NaN cells where macro releases have not yet happened or weekend-aligned tickers have not yet been carried forward by `merge_asof`. The X passed to CPCV may therefore contain a small number of partial-NaN rows. This is intentional: alignment cannot drop rows because dropping would create gaps in the CPCV chronological group structure. Resolution is the per-fold preprocessing step's responsibility, where it can be done with a partition-aware policy that respects the train-test boundary.

### Key function: `align_for_cv`

Returns the four aligned objects:
- **X**: feature matrix (~1,150 × 73; 25 TA + 9 math + 29 external + 10 lag). All 73 columns are eligible for MDA selection. AR Logistic separately consumes its 10 lag columns by name from the pre-MDA matrix routed through the pipeline's `X_tr_full`.
- **y**: binary labels {-1, +1}.
- **w**: AFML sample weights (uniqueness × return attribution × time decay, capped at 99th percentile).
- **t1**: barrier touch timestamps (DatetimeIndex) for CPCV purging.

### Key function: `validate_alignment`

Standalone validator called by the CPCV loop before training. Checks everything `align_for_cv` checks, plus: label values are in {-1, 0, +1}, all weights are positive, no NaN in t1 (would break purging), and index consistency across X, y, w, t1.

### Why a separate validator

The CPCV loop runs for hours. If an alignment problem exists, it should fail loudly and immediately at the start, not corrupt results after 20 minutes of training. `validate_alignment` is the pre-flight check.

---

## `pre_cpcv/pre_cpcv_plots.py` — Labelling Diagnostics and Feature EDA

### What it does

Produces the visual diagnostics and statistical tests that sit between the data-engineering steps and the CPCV cell: CUSUM filter visualisation, triple-barrier examples, label distribution donut, feature distribution histograms with kurtosis flagging, feature correlation heat-map, feature-vs-label mutual information, and the Augmented Dickey-Fuller stationarity test that informs the FFD column choice.

### Core concepts

**Single-responsibility plotting.** Each function takes the data it needs as parameters, returns a `Figure` (or `(Figure, DataFrame)` for the ADF test), and prints a short diagnostic summary. None of the functions call `plt.show()`; the notebook decides whether to display, save, or close each figure. This is what makes the helpers reusable for thesis-figure exports at different resolutions and aspect ratios.

**Zoom windows are parameters.** The CUSUM filter plot and the TBL examples plot accept `zoom_start` and `zoom_end` parameters with notebook-overridable defaults. The defaults pick a recent window (Jan-Apr 2026); call with different ranges to inspect earlier regimes.

**ADF test informs FFD selection.** The stationarity test is a diagnostic, not a decision: the function returns the per-feature ADF result table, and the notebook reads it to decide which columns to fractionally difference. The default cuts at `significance=0.05`, but the column flagged for FFD in the locked configuration (`atr`) was confirmed visually rather than auto-derived; some TA features (e.g., RSI, bounded between 0 and 100) can reject the unit-root null at small samples without actually needing FFD treatment, so the ADF result is informative rather than definitive.

**MI handles binary labels directly.** The mutual information plot uses `sklearn.feature_selection.mutual_info_classif` against the binary `bin` column. Features with MI below `1e-6` are flagged as removal candidates. With 73 features and ~1,150 events the MI estimator is stable, but the function exposes `n_neighbors` and `seed` for sensitivity analysis.

**ASCII portability.** Status flags use `OK` / `Flagged` rather than Unicode marks (`✓` / `✗`) so the diagnostic output renders consistently in Windows console, terminal pipes, PDF exports, and LaTeX appendices.

### Key functions

- `plot_cusum_filter(log_returns, t_events, h, zoom_start, zoom_end)`: two-panel zoom showing log returns coloured by sign with event markers, and CUSUM cumulative sums against the threshold band.
- `plot_tbl_examples(bins, close, daily_vol, pt_sl, num_days, zoom_start, zoom_end)`: side-by-side panels showing one event per barrier-touch type (profit / stop loss / vertical) with annotated price paths.
- `plot_label_distribution(bins)`: donut chart of class counts.
- `plot_feature_distributions(feature_matrix, kurtosis_threshold)`: grid of histograms with kurtosis flagging.
- `plot_feature_correlation(feature_matrix, corr_threshold, annotate)`: annotated heat-map of the correlation matrix.
- `plot_feature_label_mutual_info(feature_matrix, bins, mi_threshold, n_neighbors, seed)`: horizontal bar chart of MI values.
- `plot_adf_stationarity(feature_matrix, significance, maxlag, autolag)`: returns `(Figure, DataFrame)` with per-feature ADF p-values and the `stationary` boolean flag for downstream use.

---

# Phase 2 — `cpcv/` (Cross-Validation & Training)

This phase handles everything that must be fitted on training data only, inside the leakage-protected zone defined by Combinatorial Purged Cross-Validation. Every stateful transformation lives here.

---

## `cpcv/cv.py` — Combinatorial Purged Cross-Validation

### What it does

Generates the 28 train/test splits used to evaluate the models, applies purging and embargo to prevent label leakage, and computes the path-assignment matrix that maps splits back to backtest paths.

### Core concepts

**Combinatorial splits.** With N=8 contiguous groups and k=2 test groups per split, the function generates all C(8,2) = 28 combinations. Each split holds out 2 groups as test and uses the remaining 6 as training. This produces more out-of-sample evaluations than standard k-fold, at no additional data cost. The N=8 configuration was chosen over an earlier N=6 setup to give 7 backtest paths (more PBO partitions, denser equity-curve grids) while still keeping roughly 140 events per group at the locked event count of approximately 1,150, comfortably above the rough lower bound for daily-bar AFML pipelines.

**Purging (AFML Snippet 7.1).** Removes training observations whose triple-barrier labels overlap with any test group. Three sufficient conditions are checked for each training observation i against each test group:

1. The training observation begins inside the test period.
2. The training label resolves inside the test period.
3. The training label spans the entire test period.

If any condition holds, the observation is purged. Without purging, a training label whose t1 falls inside the test period would leak future information into training.

**Embargo (AFML Section 7.4.2).** Removes a buffer of `embargo_pct × T` training observations immediately after each test group. This prevents serial correlation from carrying signal across the train-test boundary. Embargo is applied only after the test set, not before, because training labels resolving before test begins contain no future information.

**Path matrix.** For N=8, k=2 there are φ[8,2] = C(7,1) = 7 backtest paths. Each path is a full chronological reconstruction of the dataset, assembled from test predictions across multiple splits. The path matrix tells you which split's predictions to use for each group when assembling each path. Each group appears in exactly 7 test sets, so the assignment is unambiguous.

### Key function: `generate_cpcv_splits`

Returns a list of 28 `(train_idx, test_idx)` tuples with positional integer indices into X. Each tuple already has purging and embargo applied.

### Key function: `build_path_matrix`

Returns `(n_paths=7, path_map)` where `path_map[p]` is a list of `(group_id, split_id)` tuples. Each path covers all N groups exactly once.

### Key function: `get_split_info` and helper `print_split_summary`

`get_split_info` originally recomputed splits and paths every time it was called, which led to duplicate work and duplicate log lines when the notebook also called `generate_cpcv_splits` and `build_path_matrix` separately. The current signature accepts pre-computed `splits`, `path_map`, and `n_paths` arguments so the notebook can compute each exactly once and pass them in. The new helper `print_split_summary(info)` is exposed for callers that want to render an existing info dict (e.g., during config exploration without rerunning the splitter); `get_split_info(..., print_summary=False)` returns the dict silently for the same purpose.

---

## `cpcv/cpcv_plots.py` — CPCV Visualisation and Leakage Audits

### What it does

Renders four CPCV diagnostics: a BTC-with-groups overview showing how the N CPCV groups partition the price series, a train/test timeline grid for representative splits, a text dump of the purge and embargo zones around each test group, and a full leakage audit across all 28 splits. Each function takes the CPCV inputs as parameters and is called after the CV cell has produced `splits`, `path_map`, `n_paths`, and `split_info`.

### Core concepts

**Date axis derived from `X.index`.** Every visual function pulls the date axis from `X.index` directly rather than from a separate date parameter. This means after the CUSUM start-date truncation, the plots automatically span the analysis window (August 2015 onward) without needing date-related arguments to be threaded through. The BTC backdrop in `plot_btc_with_groups` uses `df_raw` clipped to `X.index[0]` and `X.index[-1]` so the price line and the group bands share endpoints exactly.

**Group bounds computed once, passed everywhere.** The helper `_compute_group_bounds(n_events, n_groups)` divides the event index into N contiguous chunks of equal size (with the remainder distributed to the last group). It is computed once per call site and shared across the plotting and audit functions, so the rendered group boundaries always match what the CPCV splitter actually used.

**Demonstration-split selection is deterministic.** `pick_demo_splits(all_combos, n_groups)` selects three illustrative split indices: contiguous-early `(0, 1)`, one-gap `(1, 3)`, and contiguous-tail `(n_groups-2, n_groups-1)`. These three patterns exhaust the qualitatively distinct CPCV scenarios for the train/test timeline plot, which would otherwise pick arbitrary splits and miss interesting cases. Callers can override by passing `demo_splits=[i, j, k]` explicitly.

**ASCII status flags.** The leakage audit and purge-detail dump use `OK` / `FAIL` / `OVERLAP` rather than Unicode characters (`✓` / `✗`), so the output renders correctly in Windows console, terminal pipes, and LaTeX appendix tables. The audit function returns a `DataFrame` with the per-split leak counts so the notebook can write it to a thesis-appendix CSV or filter to failures only.

**Leakage audit logic.** For each split, `audit_cpcv_leakage` walks the test groups, finds the time range of each group, and checks the training set for any label whose event time `t` falls before the test-group start while its barrier-touch end-time `t1` falls inside the test-group window. This is exactly the AFML purging condition; on a clean run with `embargo_pct=0.01` and `num_days=10` triple-barrier labels, all 28 splits should report zero leaks.

### Key functions

- `pick_demo_splits(all_combos, n_groups) → list[int]`: helper that picks three illustrative split indices.
- `plot_btc_with_groups(X, df_raw, n_groups, use_log_scale)`: BTC close price with the N CPCV groups shaded as coloured bands.
- `plot_train_test_timelines(X, splits, n_groups, k, demo_splits)`: three sub-panels showing train (steelblue) and test (crimson) date partitioning for representative splits, with test-group shading and group-boundary verticals.
- `print_purge_embargo_detail(X, t1, splits, n_groups, k, demo_splits)`: per-split text dump showing the last three training rows before each test group (purge zone) and the first three after (embargo zone), with `OK` / `OVERLAP` flags on each pre-test row.
- `audit_cpcv_leakage(X, t1, splits, n_groups, k, split_info)`: full audit across all splits; prints summary table and returns the audit `DataFrame`.

The 8-color palette used by the visual functions is exposed as the module-level constant `DEFAULT_GROUP_COLORS` for consistency across plots.

---

## `cpcv/preprocessing.py` — Per-Fold Preprocessing

### What it does

Inside each CPCV fold, applies three transformations in order: fractional differentiation, robust scaling, and feature selection. Every transformation is fitted on training data only, ensuring zero leakage.

### Core concepts

**Fractional differentiation (FFD, AFML Chapter 5).** Standard differencing (first or second differences) destroys long-memory information in price series. FFD applies a fractional power d ∈ [0, 1] to the differencing operator, yielding a series that is stationary while retaining as much memory as possible.

For each FFD column (currently only ATR), the function:

1. Sweeps d from 0 to 1 in steps of 0.05 on the training fold.
2. At each d, computes FFD weights via the recursive formula `ω_k = -ω_{k-1} × (d - k + 1) / k`, truncated when weights fall below 1e-4.
3. Applies the convolution and runs an Augmented Dickey-Fuller test.
4. Returns the minimum d* where ADF p-value falls below 0.05.
5. Applies FFD at d* to the full series so test observations have proper lookback history, but only training data was used to determine d*.

**NaN resolution policy (asymmetric by column type).** Phase 1 deliberately defers NaN resolution to this step. Inside the FFD function, after the train/test partition is formed, NaN handling diverges by column category, applied independently within each partition so the train-test boundary is never crossed.

- *Non-FFD columns (the majority).* Imputed via `ffill().bfill()` within each partition. Forward-fill carries the most recent valid observation across calendar gaps (the natural economic interpretation: the model uses the last published macro reading or last on-chain snapshot, exactly as a human trader would). The trailing back-fill handles any leading NaN at the very start of the partition. Both operations stay within the partition: a test-set NaN at the boundary fills only from test data, never from train, preserving the AFML purging discipline.
- *FFD columns (currently ATR).* Rows where any FFD column is NaN are dropped, not imputed. Forward-filling an FFD-transformed column would inject stale weighted-lookback values; dropping the leading rows is the correct response. The function logs the dropped row count: typically 2-10 rows per partition, concentrated at the start.

**Robust scaling.** Uses `sklearn.preprocessing.RobustScaler` (median + IQR) rather than `StandardScaler`. RobustScaler is more resilient to the heavy-tailed distributions typical of financial features. Fitted on training fold only; applied to both train and test.

**Multi-Model MDA feature selection.** Departs from AFML's three-method MDI/MDA/SFI protocol. Instead, runs Mean Decrease Accuracy (permutation importance) using two classifiers in parallel: a Random Forest (captures nonlinear interactions) and a Logistic Regression (captures linear relationships). Both run inside a purged inner 3-fold CV with one-sided post-test embargo (no pre-test embargo, since labels resolving before inner-test contain no future information).

For each classifier:

1. Splits training data into 3 chronological inner folds.
2. Purges inner-train observations whose t1 overlaps inner-test (AFML §7.4.1 label-overlap purge).
3. Applies a one-sided post-test embargo after each inner-test boundary.
4. Fits the classifier and computes weighted F1 on inner-test using the test-fold's sample weights.
5. For each feature, permutes the column, recomputes weighted F1, and records `MDA = baseline_F1 − permuted_F1`.

**Per-model z-scoring before averaging.** RF and Logistic Regression produce MDA scores on different absolute scales (RF's permutation drops F1 by ~0.005-0.05 per feature; Logistic's drops can be an order of magnitude smaller). Naive averaging of `(MDA_RF + MDA_LR) / 2` would let RF's larger magnitudes dominate, effectively recovering an RF-only ranking. The implementation z-scores each model's MDA vector across features before averaging, so both models contribute symmetrically to the final ranking regardless of their native magnitudes. Features with positive averaged z-MDA are kept (permuting them hurts at least one model on average, on a rank-normalised basis), capped at `MDA_TOP_K_FRAC = 0.20` of total features, with a hard floor of 5 features minimum.

**TOP_K_FRAC trail.** The cap was tightened from common defaults of 0.40-0.50 in an earlier round of methodology development. The first tightening from 0.40 to 0.30 came after a high-PBO run on the 66-feature pool revealed only ~6 features clearing 50% selection frequency in the stability bar chart, indicating that the long tail was contributing variance rather than signal. The May 2026 expansion of the feature pool from 66 to 73 columns (the 14d/30d macro pairs and the AR_LAGS extension from 6 to 10) was an advisor-driven structural change to the feature set; the cap was re-evaluated at the same time using absolute-feature-count arithmetic. At the locked 73-feature pool, `0.20 × 73 ≈ 14-15` features restores the absolute count to roughly the original working point of 0.30 × 66 ≈ 20. The 0.20 cap is the locked working configuration. Methodology rationale: with ~600 events in the model-training partition after the 70/15/15 three-way split (described below), the KAN and LSTM parameter counts at `width1=6`, `hidden=16` already approach 1 sample per parameter at 15 features; expanding to 20+ features would deepen overparameterisation in the two most-watched neural models. Sensitivity sweeps over `{0.10, 0.15, 0.20, 0.25, 0.30}` can be appended as an appendix without modifying the source.

The rationale for multi-model MDA: RF-only MDA introduces tree bias (features that tree ensembles naturally exploit get inflated importance). SFI in weak-signal regimes produces uninformative near-uniform scores. Averaging across model families reduces selection variance and prevents architecture bias.

### Key function: `preprocess_fold`

Orchestrates FFD → scaling → selection per fold and **returns a dict** (not a tuple) with ten keys: `X_tr_full` (full feature matrix for AR Logistic's lag pull), `X_tr_sel` / `X_te_sel` (selected-feature matrices for the other models), `y_train`, `w_train`, `t1_train`, `y_test`, `w_test`, `t1_test`, and `prep_info` (FFD d*, scaler, selected features). The dict return supersedes an earlier tuple-return signature; bundling the train and test alignments together with the prep_info means downstream code can route the dict through a single argument and never needs to know which positional index held the scaler vs the selected-feature list. The function returns the test-side alignment because FFD's per-partition NaN drop can shorten either partition independently, so the caller cannot reconstruct the test alignment from positional indices alone.

**Lag-feature inclusion (advisor-driven change).** All 73 features enter MDA together, including the 10 `log_returns_lagN` columns from `pre_cpcv/features.py`. An earlier version of `select_features` filtered out lag columns by name prefix on the rationale that they should be reserved for the AR Logistic baseline; the advisor argued this created an unfair information asymmetry, since AR Logistic received lag features the ML and neural models did not see. The current pipeline lets every feature compete on equal footing for the top-k cap. Whether a given fold selects any lag columns for the non-AR models depends on their permutation importance for that fold's RF and Logistic Regression inner classifiers. AR Logistic continues to pull its ten lag columns by name from the pre-MDA matrix via the pipeline's `X_tr_full` route, so its behaviour is unchanged regardless of MDA's choices.

The `skip_selection=True` flag is set when only AR Logistic is being evaluated. AR Logistic does not need MDA selection because it consumes only the precomputed lag columns from `pre_cpcv/features.compute_lag_features`; running MDA on its behalf would be wasted compute.

**Per-split warning prefix.** Both `preprocess_fold` and `select_features` accept optional keyword-only arguments `split_idx` and `n_splits`. When `pipeline.py` populates these inside the per-split CPCV loop, any warning emitted by `select_features` (currently only the `Only N features with MDA > 0; taking top {MIN_FEATURES} by MDA value` fallback) is prefixed with `[split K/N]` where `K` is the 1-indexed split number. The prefix surfaces in the deferred-warning summary that `pipeline.run_cpcv_pipeline` prints after the ✅ completion notice, so the reader can identify which specific folds tripped the fallback rather than assuming the warning applied uniformly across all splits. Backward-compatible: callers that omit `split_idx` and `n_splits` get unprefixed warnings as before.

---

## `cpcv/tuning.py` — Nested Hyperparameter Tuning

### What it does

For each CPCV training fold, runs Bayesian hyperparameter optimization (Optuna's TPE) with a 3-fold purged inner cross-validation. Returns optimal parameters per model per split, applied just before that split's models are trained.

### Core concepts

**Why nested tuning.** Tuning hyperparameters on the full dataset and then evaluating with CPCV would leak test-set information into the model selection process, invalidating DSR and PBO. Nested tuning restricts each split's hyperparameter search to its own training fold, preserving the integrity of out-of-sample metrics.

**TPE (Tree-structured Parzen Estimator).** Bergstra et al. (2011). A Bayesian optimizer that maintains two probability distributions over the hyperparameter space: one for "good" trials (low loss) and one for "bad" trials. New trials are sampled from regions where the ratio of good to bad density is highest. Concentrates compute in promising regions instead of grid-search exhaustiveness.

**Purged K-Fold inner CV.** Three chronological inner folds with a 10-observation embargo around each fold's boundaries. The embargo length matches the triple-barrier horizon (`num_days=10`), ensuring overlapping labels cannot leak between inner train and inner validation. Three folds (rather than five) increase inner validation size to ~300 observations per fold (vs ~180 with five folds), giving more reliable log-loss estimates in a low-signal environment. With outer-train sizes around 900 events at N=8, the three inner folds give the tuner enough samples per validation block to detect overfitting without the variance penalty of a five-fold split.

**Median pruner.** After each inner fold, Optuna can terminate underperforming trials whose intermediate log loss is worse than the median of completed trials. Saves compute on clearly bad regions of the search space.

### Search spaces (capped to match the regularization stance)

- **Logistic Regression**: `C` ∈ log-uniform [1e-4, 1e2], `penalty` ∈ {l1, l2}.
- **Random Forest**: `n_estimators` ∈ [100, 250], `max_depth` ∈ [2, 6], `min_samples_leaf` ∈ [15, 40], `max_features` ∈ {sqrt, log2}.
- **XGBoost**: `max_depth` ∈ [1, 3], `learning_rate` ∈ log [0.01, 0.3], `min_child_weight` ∈ [10, 50], plus subsample, colsample_bytree, gamma ∈ log [0.01, 1.0], reg_alpha, reg_lambda. Fixed `n_estimators=500` with early stopping at 20 rounds.
- **LSTM**: `hidden_size` ∈ {16, 32}, `num_layers` fixed at 1, `dropout` ∈ [0.1, 0.5], `learning_rate` ∈ log [1e-4, 5e-2]. Window=14 (matches production).
- **KAN**: `width1` ∈ [2, 6], `width2` fixed at 0, `lr` ∈ log [5e-4, 5e-2], `weight_decay` ∈ log [1e-5, 5e-3], `grid` ∈ {3, 5}.

The caps were tightened in the locked configuration relative to common defaults and to earlier iterations of this thesis. RF `n_estimators` capped at 250 (down from an earlier 300) because trees in a noisy regime do not benefit from more than 250 estimators. RF `max_depth` was tightened further from [3, 15] to [2, 6]: depth 6 has 64 leaves which is plenty for ~900-sample training folds, and shallower forests vote in tighter agreement, reducing the disagreement that surfaces as path-Sharpe variance. RF `min_samples_leaf` was raised from [1, 30] to [15, 40] so each leaf represents at least 1.7% of the training fold rather than fitting handfuls of high-volatility events. XGBoost `max_depth` was tightened from [2, 6] to [1, 3] because boosting compounds depth nonlinearly across rounds; depth 3 across 50 rounds already produces substantial nonlinear capacity, and depth 6 in this regime memorises residuals. XGBoost `min_child_weight` floor was raised across successive iterations (1 → 5 → 10) after a leave-one-out PBO audit showed XGBoost driving the bulk of the multi-model selection-bias signal; with ~600 events in the model-training partition under the 70/15/15 split, the current `[10, 50]` floor forces each leaf to represent ≥1.7% of the training fold, suppressing the single-event-leaf overfitting mode that inflates XGBoost's in-sample-rank variance. XGBoost `gamma` floor was raised from 1e-8 to 0.01 (still log-uniform up to 1.0) in the same iteration: the earlier floor was effectively "off" for the regulariser, allowing splits with arbitrarily small loss reduction; the new floor forces every split to clear a non-trivial gain threshold. LSTM `num_layers` was first tightened from [1, 3] to [1, 2] and then hardcoded to 1: two- and three-layer LSTMs on ~1,150 events are deep-overfit territory and the additional layer added variance to path-Sharpes without improving accuracy; hardcoding to 1 frees Optuna trials for finer exploration of dropout and learning_rate. KAN `width1` was tightened from [3, 12] to [2, 6] to keep the symbolic formula extracted in Phase 3 humanly readable, since each surviving width1 unit becomes one additive term plus interactions in the closed-form expression. KAN `width2` was hardcoded to 0 (one hidden layer only): a two-hidden-layer formula nests trigonometric primitives in trigonometric primitives and produces fourth-order compositions that lose interpretability. Hardcoding `width2=0` also ensures the architecture used for CPCV evaluation matches the architecture extracted in Phase 3, so the symbolic formula reflects the actual benchmark model rather than an unrelated KAN topology. With ~900 training samples per outer fold (at N=8) and ~300 inner-validation samples, overly flexible architectures guarantee overfitting; each cap is justified by the samples-to-parameters ratio.

**KAN and LSTM tuning sensitivity (May 2026).** A wider search space was tested as a sensitivity check against the locked narrow ranges. For KAN: `width1 ∈ [3, 8]`, `width2 ∈ [0, 3]`, `grid ∈ {3, 5, 7}`, `lr ∈ log [5e-4, 5e-3]`. For LSTM: `hidden_size ∈ {16, 32, 48, 64}`, `num_layers ∈ [1, 2]`, `dropout ∈ [0.0, 0.5]`. The expanded configurations produced declines in median Sharpe of 0.07 (KAN) and 0.11 (LSTM) over the seven CPCV paths, consistent with the small-sample memorisation hypothesis that motivated the original narrow ranges. The narrow ranges are retained as the locked thesis configuration; the failed expansion is documented in the tuning-function docstrings and reported in the methodology chapter as empirical evidence that the tuning ranges were not arbitrarily chosen.

### Production consistency

Tuning uses identical configurations to production along the dimensions that affect what gets learned: same window length (14 for LSTM), same warm restart schedule (T_0=25 for LSTM, T_0=30 for KAN), same loss function (cross-entropy with label smoothing 0.1, sample weights, class weights). Hyperparameters tuned under one regime would be suboptimal under another, so consistency matters along these axes.

The LSTM tuning loop deliberately runs fewer epochs and a shorter patience window than production (tuning: epochs=50, patience=7; production: epochs=100, patience=15) to keep per-trial cost bounded across hundreds of inner-fold fits. The tuned hyperparameters are then re-fitted at the production budget. The methodology chapter discloses this divergence explicitly.

### Default trial counts

`N_TRIALS_CLASSICAL = 30` (Logistic, RF, XGBoost), `N_TRIALS_NEURAL = 30` (LSTM, KAN). The `run_cpcv_pipeline()` function accepts an `n_trials` override, and the notebook currently passes `n_trials=30` for every tuned model.

### Key function: `tune_all_models`

Orchestrates per-split tuning for a list of models. Returns `{model_name: {"best_params": ..., "best_log_loss": ..., "results_df": ...}}`. Called inside `pipeline.py`'s split loop.

---

## `cpcv/calibration.py` — Probability Calibration

### What it does

Calibrates raw model probabilities so that predicted confidence levels correspond to empirical accuracy. Without calibration, downstream bet sizing produces systematically wrong position sizes.

### Core concepts

**Why calibration matters here.** The bet-sizing formula in evaluation converts probabilities directly into position sizes via De Prado's S-curve. If a model's predicted 0.70 probability actually corresponds to 0.55 empirical accuracy, the resulting bet size is too aggressive. Calibration aligns predicted confidence with reality.

**Platt scaling (Platt 1999).** For sklearn-compatible models (AR Logistic, Logistic Regression, Random Forest, XGBoost). Fits a logistic regression mapping raw logits to calibrated probabilities: `p_calibrated = σ(a × logit + b)`. The regularization is set to C=1e10 (effectively unregularized) so the sigmoid is purely data-driven. The two parameters `a` and `b` allow Platt scaling to correct both miscalibration sharpness (slope) and directional bias (intercept).

**Vector scaling (Guo et al. 2017, Section 4.2).** For PyTorch models (LSTM, KAN). Fits a temperature `T > 0` and a per-class additive bias vector `b` that minimise the negative log-likelihood of `softmax((logits + b) / T)`. The optimisation runs L-BFGS-B with bounds `T ∈ [0.05, 20.0]` and `b_c ∈ [-5.0, 5.0]` per class. The `T` parameter sharpens or softens predictions; the `b` vector shifts the decision boundary, allowing the calibrator to correct directional bias that pure temperature scaling cannot.

**Class-0 bias pinned to zero.** Softmax is invariant to a constant added to all class biases (`softmax(z + c) = softmax(z)` for any scalar `c`), so fitting both `b[0]` and `b[1]` leaves a one-dimensional ridge in the loss surface that L-BFGS-B can stall on. The implementation fits only `b[1]` as a free parameter and pins `b[0] = 0`, removing the overparameterisation while preserving the directional-bias correction (the relative bias `b[1] − b[0]` is the only identifiable component anyway). The `T` bound was widened from an earlier `(0.1, 10.0)` to `(0.05, 20.0)` after observing boundary hits on a small fraction of folds; the wider range keeps the optimum interior on every fold evaluated.

`fit_temperature_scaling` is retained in the module for reference and unit tests but is no longer the default for any model.

### Why two methods, and why vector scaling rather than temperature scaling

Sklearn models output 1D log-odds (a single scalar per sample), well-suited to a sigmoid mapping. PyTorch classifiers output 2D logits (one per class), better calibrated by a method that respects the multi-class softmax structure.

An earlier implementation used pure temperature scaling for PyTorch models. A calibration audit before final evaluation revealed that temperature scaling could not correct a systematic directional bias present in both LSTM and KAN outputs: predicted P(y=1) sat ~10–23 percentage points below the empirical rate of ~0.55 across the bulk of the predicted-probability distribution. Pure temperature scaling preserves the argmax of the raw logits by construction, so a single-parameter `T` cannot shift a "lean class 0" prediction to "lean class 1" no matter what value it takes. Vector scaling adds the per-class bias vector and is the natural remedy, recommended by Guo et al. (2017) for cases where temperature scaling alone is insufficient. The substitution was made before the final evaluation pass and constitutes a correction of methodological inadequacy rather than test-set-informed model selection.

### Calibration set

Calibration is fitted on the held-out 15% calibration partition of the training fold (chronological split, no shuffling, separate from the 15% validation partition used for early stopping). Never touches test data. If calibration fails for any reason, the pipeline falls back to uncalibrated `predict_proba`.

### Methodological note

Under the locked 70/15/15 three-way split, the 15% calibration partition serves Platt/vector scaling exclusively and is fully separate from the 15% validation partition that handles XGBoost's early stopping and the LSTM/KAN best-state tracking. The split was widened from an earlier 80/10/10 after a calibration audit revealed all six models systematically under-predicting P(Up) by 4-7 percentage points across folds; with ~750 events per outer training fold, 10% (~75 events) was insufficient for both the vector-scaling fit (T, b[1] estimated on ~75 binary labels) AND the LSTM/KAN early-stopping signal. The 80/20 split that preceded both used a single 20% subset for both purposes; the three-way split removes that dual-use coupling. The current 15% partitions raise both roles to ~112-129 events at the cost of ~75 fewer model-training events per fold (~600 vs ~688 under 80/10/10).

### Weighted vs unweighted calibration

Both `fit_platt_scaling` and `fit_vector_scaling` accept an optional `sample_weight` argument that weights the per-sample NLL by AFML sample weights before averaging. The argument is supported in the calibrator API but **intentionally not triggered from `pipeline.py`**. The rationale is empirical: an audit run on the full six-model pool with weighted Platt + weighted vector scaling enabled showed that the weighted variant pushed every model's calibration miss FURTHER from the empirical base rate (3/6 flagged → 5/6 flagged under weighted; calibration deltas grew on every single model relative to the unweighted baseline), and median Sharpes compressed across the board. The mechanism is that AFML weights over-represent the high-weight subset (rare events with little overlapping label structure) whose class balance differs from the population base rate; weighting the calibrator's loss against that subset's distribution tilts the calibrated probability away from the empirical class frequency on the full data. The unweighted path remains the locked configuration. The weighted code paths in `calibration.py` are kept available for future experiments without re-implementation cost; a comment block above the `calibrator.fit(...)` call site in `pipeline.py` documents the audit evidence so the next reader doesn't reintroduce the regression.

### Key class: `Calibrator`

Unified interface that auto-detects method from `model.get_name()`:
- `fit(model, X_cal, y_cal)`: fits the appropriate calibrator (Platt for sklearn models, vector for PyTorch models).
- `calibrate(logits)`: applies the fitted calibration to new logits, returning a (n_samples, n_classes) probability matrix.
- `fit_from_logits(logits, y_cal, method)`: alternative entry point for LSTM, where index alignment requires pre-computed logits. Defaults to `method="vector"`; accepts `"temperature"` or `"platt"` for opt-in use.

---

## `cpcv/pipeline.py` — CPCV Loop Orchestration

### What it does

The main entry point. Coordinates everything: split generation, preprocessing, optional tuning, model training across all model × seed combinations, calibration, and prediction storage. Produces the predictions dictionary that feeds Phase 3 evaluation.

### Core concepts

**One preprocessing per fold, multiple models per fold.** Preprocessing (FFD, scaling, selection) is shared across all models in a fold. This avoids repeating expensive computation and ensures all models compete on the same features. The preprocessing step returns a dict (see `preprocessing.py`); the pipeline routes its members to each model without unpacking by position.

**Three-way 70/15/15 chronological split for early stopping and calibration.** Within each training fold, the data is split chronologically into 70% model-train, 15% validation, and 15% calibration. The validation partition feeds the model's `X_val` / `y_val` for early stopping and best-state tracking; the calibration partition is held entirely separate and feeds Platt or vector scaling. The split was widened from an earlier 80/10/10 layout after a calibration audit revealed all six models systematically under-predicting P(Up) by 4-7 percentage points; 10% (~75 events) was insufficient for the vector-scaling fit and for the early-stopping signal. An even earlier 80/20 split used the same 20% subset for both early stopping and calibration, which introduced subtle leakage between the model's stopping decision and the calibration data; the three-way split removes that coupling.

**`sample_weight_val` for weighted early stopping.** When `tune=True` the pipeline passes the validation partition's AFML sample weights through to each model's `fit` call via a `sample_weight_val` argument. XGBoost forwards this to `sample_weight_eval_set`; LSTM and KAN use it inside the per-epoch validation-loss computation. Without this, training observed sample-weighted loss but early stopping observed unweighted loss, and the two diverged on folds with heavy weight concentration.

**Model dispatching.** AR Logistic receives the full pre-selection DataFrame and pulls its 10 precomputed lag columns from there (`log_returns_lag1` … `log_returns_lag30`, produced once on the full daily series by `pre_cpcv.features.compute_lag_features`). Other models receive the MDA-selected feature subset. The pipeline routes the correct X to each model.

**LSTM logits sliced before calibration.** The LSTM produces logits only for windows where all 14 lookback timesteps are non-NaN; the first ~14 events of any partition are window-incomplete and would produce NaN logits. The pipeline slices `raw_logits` to `valid_idx` BEFORE passing them to the calibrator. An earlier ordering calibrated on the full unsliced logits and sliced afterward, producing a fitted calibrator whose temperature reflected the NaN-padded positions rather than the valid windows; the bug manifested as `T` consistently hitting the lower bound on LSTM folds and was caught during a calibration audit.

**Per-split tuning application.** When `tune=True`, the pipeline calls `tune_all_models()` after preprocessing and before model training. The tuned parameters are then written to module-level constants (`lstm_mod.LSTM_HIDDEN_SIZE`, `kan_mod.KAN_HIDDEN`, etc.). Each model class reads these constants at instantiation time, so tuning takes effect for that split's models.

**Module-default reset.** The first thing `run_cpcv_pipeline` does is call `_reset_module_defaults()`. On the first invocation it snapshots the pristine import-time values of every constant `_apply_tuned_params` writes (LOGISTIC_C, RF_N_ESTIMATORS, KAN_HIDDEN, …); on every subsequent invocation it restores those snapshots. Without this, a second pipeline call would inherit the previous run's tuned values: tuning once and then re-running without tuning would silently use the tuned widths from the prior run. The tracked-constants list (`_TRACKED_CONSTANTS`) mirrors exactly what `_apply_tuned_params` writes; new tuned hyperparameters must be added to both.

**LSTM index handling.** LSTM consumes windowed sequences (length 14), so it produces fewer predictions than other models. The pipeline tracks `last_valid_indices` to align LSTM predictions back to original timestamps. Calibration uses `fit_from_logits` to handle this alignment.

**Error handling.** Failed model fits are caught, logged, and skipped without crashing the pipeline. Final summary prints successful and failed task counts.

### Stored predictions

Per `(model_name, split_idx, seed)` key, stores:
- `y_true`, `y_pred`, `cal_proba`: predictions and ground truth.
- `f1_macro`, `roc_auc`, `log_loss`: inline metrics.
- `timestamps`, `ret`: for path stitching downstream.
- `prep_info`: FFD d* values, scaler, selected features (for symbolic extraction).
- `calibrator`: string repr of the fitted calibrator.

### Tuning results

When `tune=True`, the result dict also contains `tuning_results[split_idx][model_name]` with each model's best params, best log loss, and full Optuna trial DataFrame.

### Key function: `run_cpcv_pipeline`

Returns a dict with `predictions`, `path_map`, `n_paths`, `n_splits`, `models`, `n_seeds`, and optionally `tuning_results`. This dict is the contract between Phase 2 and Phase 3.

---

# Phase 2 — `cpcv/models/` (Model Implementations)

Six models implementing a uniform interface. The pipeline iterates over them without knowing their internals.

---

## `cpcv/models/base.py` — Abstract Base Class

### What it does

Defines the `BaseModel` interface that every model must implement: `fit`, `predict_proba`, `predict`, `get_name`, plus a uniform constructor signature `(n_features, n_classes=2, seed=42)`.

### Core concepts

**Why an abstract base class.** Six different libraries (sklearn, xgboost, PyTorch with two architectures, custom AR construction) need to be exchangeable in the pipeline. The base class enforces that every model exposes the same methods, so the pipeline can iterate without conditional logic per model.

**Label convention.** The pipeline maps original labels {-1, +1} → {0, 1} before passing to models. All models work with 0-indexed classes. Evaluation maps back to {-1, +1} for economic interpretation.

### Key methods

- `fit(X_train, y_train, sample_weight, X_val, y_val)`: train the model. `sample_weight` is the AFML Chapter 4 weights. `X_val`/`y_val` enable early stopping for neural models.
- `predict_proba(X)`: returns class probabilities, shape (n_samples, n_classes).
- `predict(X)`: returns hard labels via argmax of `predict_proba`. Default implementation provided.
- `get_name()`: returns a human-readable model name for logging and comparison.

---

## `cpcv/models/benchmarks.py` — AR Logistic and Logistic Regression

### What it does

Implements two baseline models: an autoregressive logistic regression on lagged returns, and a standard logistic regression on the selected feature set.

### Core concepts

**AR Logistic — econometric baseline.** Selects 10 precomputed lag columns (`log_returns_lag1` through `log_returns_lag7`, `log_returns_lag14`, `log_returns_lag21`, `log_returns_lag30`) by name from the pre-MDA feature matrix and ignores everything else. The lag columns are produced once on the full daily series by `pre_cpcv.features.compute_lag_features`. The pipeline routes the pre-MDA matrix (`X_tr_full`) to AR Logistic, so it always sees its lag columns regardless of whether MDA selects them for the other models. The role of this baseline is to test whether the engineered features (TA, AFML mathematical, macro, on-chain) add value beyond simple price momentum. If a more complex model cannot beat AR Logistic, it has not learned anything beyond autocorrelation.

NaN lag columns at predict time raise an exception. An earlier inline-build version silently `bfill()`-imputed missing lags inside `fit` and `predict`, which leaked future test observations into earlier ones. Moving lag construction upstream to `pre_cpcv.features` (computed once on the full daily series, no fold-local bfill) eliminated that leakage path.

**Logistic Regression — linear ML baseline.** Uses sklearn's standard `LogisticRegression` on the full selected feature set. Class-weighted to handle label imbalance, with L1 or L2 penalty (chosen via tuning). Solver auto-selected based on penalty (LBFGS for L2, liblinear for L1).

**Why two baselines.** AR Logistic isolates pure autocorrelation. Logistic Regression captures linear effects across the full feature set. Together they bracket the linear-signal regime: any complex model claiming nonlinear value must outperform both.

### `predict_logits` method

Both models expose a `predict_logits` method that returns raw log-odds for downstream calibration. AR Logistic computes log-odds from `predict_proba` with a symmetric `np.clip(proba, 1e-10, 1 − 1e-10)` matching the tree-model convention (the earlier asymmetric `log(p₁ / (p₀ + 1e-10))` is gone, eliminating a mild numerical asymmetry between baselines and tree models). Logistic Regression uses sklearn's `decision_function` which returns the raw decision value before sigmoid.

---

## `cpcv/models/tree_models.py` — Random Forest and XGBoost

### What it does

Wraps sklearn's `RandomForestClassifier` and `XGBClassifier` in the BaseModel interface. Both return calibration-ready logits and accept AFML sample weights.

### Core concepts

**Random Forest.** 500 trees with `max_features="sqrt"` and `class_weight="balanced_subsample"`. The balanced subsampling reweights minority class within each bootstrap, more robust than fixed `class_weight` for time-varying class proportions. No max depth by default (trees grow until `min_samples_leaf=5`).

**XGBoost.** Gradient-boosted trees with 500 estimators and early stopping at 20 rounds when a validation set is provided. `scale_pos_weight` set automatically from training class balance. Uses `binary:logistic` objective.

**Logit conversion for calibration.** Both models output probabilities, but Platt scaling expects logits. Both `predict_logits` methods convert via `log(p₁ / p₀)` with clipping at 1e-10 to avoid infinities at the boundaries.

**Tunable hyperparameters.** When tuning runs, module-level constants (`RF_N_ESTIMATORS`, `XGB_MAX_DEPTH`, etc.) are overridden per split. The model classes read these at construction time. Without tuning, they use the defaults defined at the top of the file.

### Why these two tree methods

Random Forest provides bagged variance reduction, robust to noisy features. XGBoost provides boosted bias reduction, sequentially correcting errors. Together they cover the two main paradigms of tree ensembles. In financial ML, both are standard benchmarks.

---

## `cpcv/models/lstm_model.py` — LSTM Classifier

### What it does

A PyTorch LSTM with last-hidden-state pooling that consumes windowed sequences and outputs binary classifications. Wrapped in the BaseModel interface for seamless integration with the CPCV pipeline.

### Core concepts

**Sequence construction.** The 2D feature matrix is reshaped into 3D windowed sequences of length 14 (two BTC calendar weeks). Each timestep contains all selected features at that bar. The first 13 observations are dropped (insufficient lookback). The function returns `valid_indices` mapping sequences back to original positional indices.

**Architecture.** Single-layer LSTM (`num_layers=1` hardcoded, `hidden_size` ∈ {16, 32}, `dropout` ∈ [0.1, 0.5]) → last hidden state from the final layer → LayerNorm → dropout → linear classifier. Hidden size, dropout, and learning rate are tuned per split. `num_layers` is no longer searched after empirical evidence that the second layer added path-Sharpe variance without improving accuracy; the previous configuration searched [1, 2] and converged to single-layer almost every fold, so the search dimension is now hardcoded to free Optuna trials for finer exploration of the remaining hyperparameters.

**Last-hidden-state pooling.** The final timestep's hidden state from the last LSTM layer is used as the sequence representation. An earlier version used learned temporal attention pooling (weighted sum across all timesteps), but it was removed: with a 14-day window and ~900-sample folds, the additional attention parameters did not improve performance and the simpler standard approach proved more robust.

**Window length matched to labeling horizon.** The 14-day window (~2 BTC weeks) is intentionally close to the 10-day triple-barrier horizon. Longer windows (30+ days) caused two problems: gradient signal attenuation across many recurrent steps, and a parameter-to-sample ratio that encouraged overfitting on small training folds.

**Tanh input normalization.** Before the LSTM, features are tanh-normalized: `z = tanh((x - μ) / σ)`. Maps features to [-1, 1] regardless of original scale, stabilizing training on fat-tailed financial data. Mean and std are fitted on training data only and stored for inference.

**Training regularization stack.** Standard techniques chosen for small-sample financial time series:
- Label smoothing (0.1) softens noisy labels.
- Gradient clipping (max norm 1.0) prevents exploding gradients.
- Cosine annealing with warm restarts (T_0=25, T_mult=2) provides exploration via periodic LR resets.
- Class weights for imbalance.
- AFML sample weights for label uniqueness.
- Early stopping (patience 15) on validation loss with best-state restoration.

**Tuning consistency.** The `LSTMClassifier.__init__` reads `LSTM_HIDDEN_SIZE`, `LSTM_NUM_LAYERS`, `LSTM_DROPOUT` at call time (not as default arguments). This ensures that when tuning overrides these module constants, the new values reach the model.

### Why LSTM here

The LSTM is the standard sequential-modeling baseline in financial deep learning. It tests whether explicitly modeling temporal dependence in feature sequences improves over models that see only the bar at time t. Comparing KAN against LSTM isolates whether KAN's spline-based representation gains advantage from sequential context (it does not — KAN consumes single-bar features) or purely from richer per-bar function approximation.

If LSTM underperforms a 6-lag AR Logistic baseline on this dataset, that is itself a meaningful empirical finding: the temporal information in BTC daily features is largely already encoded in the per-bar feature representation (TA indicators, mathematical features, macro, on-chain), leaving little marginal signal for a recurrent architecture to exploit. The thesis reports such results honestly rather than tuning the LSTM until it beats simpler baselines.

---

## `cpcv/models/kan_model.py` — Kolmogorov-Arnold Network

### What it does

A KAN classifier using the `efficient-kan` library, trained as a standard PyTorch `nn.Module` with AdamW. Outputs well-scaled logits that calibrate without extreme temperature correction.

### Core concepts

**KAN as architecture.** Where MLPs apply learnable weights to fixed activation functions, KANs apply fixed weights (additions) to learnable activation functions. Each edge in the network is a B-spline basis function whose shape adapts during training. The Kolmogorov-Arnold representation theorem guarantees that any continuous multivariate function can be expressed as a finite composition of univariate functions and additions, providing the theoretical motivation.

**Architecture for this thesis.** `[n_features, KAN_HIDDEN, n_classes]` — a single hidden layer with B-spline activations forming a narrow bottleneck. The `width1` parameter is tuned per split within `[2, 6]` and grid size within `{3, 5}`; `width2` is hardcoded to 0, permanently disabling the second hidden layer. The single-hidden-layer constraint is essential for the Phase 3 symbolic-extraction deliverable: a two-hidden-layer KAN extracts as a formula that nests trigonometric primitives in trigonometric primitives (fourth-order compositions), which loses interpretability. With width1 capped at 6, the extracted formula has at most 6 input-derived terms plus a small number of interaction terms, which is humanly readable. Hardcoding `width2=0` also ensures the CPCV-evaluated KAN architecture matches the architecture used by `prepare_extraction_data` in Phase 3, so the symbolic formula reflects the same model the benchmark numbers describe rather than an unrelated KAN topology.

**Tanh input normalization to spline grid range.** Features are tanh-normalized to [-1, 1] to match efficient-kan's default `grid_range=[-1, 1]`. Without this, features outside the grid range would extrapolate to flat splines, losing all gradient signal.

**Why efficient-kan instead of PyKAN for prediction.** PyKAN is the canonical KAN library but trains via Adam → LBFGS schedules that are fragile for small samples. `efficient-kan` reimplements the same B-spline basis as a standard `nn.Module`, allowing reliable AdamW training. Symbolic extraction is handled separately by the `post_cpcv` symbolic extraction pipeline using PyKAN.

**Training stack.** AdamW (lr and weight_decay tuned), label smoothing (0.1), gradient clipping (max norm 1.0), cosine annealing warm restarts (T_0=30), early stopping (patience 20) with best-state restoration. The same regularization stack as LSTM, with no neural-net-specific tricks like SWA or entropy regularization (both tested and removed for being either redundant or non-coherent with early stopping).

**Single grid level.** Unlike the literature's coarse-to-fine schedule (start at grid=3, refine to grid=5 mid-training), this implementation trains at a single grid level throughout. With ~900 training samples, grid refinement adds parameters faster than the data can support, causing memorization. Single-grid training is more stable.

**No state passed to symbolic extraction.** The fitted KAN holds only its standard PyTorch state (parameters, normalization buffers). It does not store the training dataset on the instance — an earlier version cached a `_dataset` dict on the model so symbolic extraction could read its training tensors directly, but the symbolic pipeline now reconstructs everything it needs from `prep_info` (the FFD d* values, the fitted scaler, the selected feature list stored alongside predictions). Removing the cached dataset cuts a ~10 MB redundancy per fold and makes the prediction-vs-extraction handoff a clean data interface rather than a stateful coupling.

### Why KAN

The thesis tests whether KAN's adaptive activations improve over fixed-activation models (LR, RF, XGBoost, LSTM) for BTC direction prediction. The novel contribution beyond benchmarking is symbolic extraction: KAN's structure permits closed-form formula recovery (Phase 3), which no other model in the comparison set can offer.

---

# Phase 3 — `post_cpcv/` (Evaluation & Interpretability)

Takes the predictions dictionary from Phase 2 and produces the thesis deliverables: model comparison, statistical robustness tests, financial performance metrics, and KAN symbolic formulas.

---

## `post_cpcv/evaluation.py` — Metrics, Path Stitching, and Robustness Tests

### What it does

Comprehensive evaluation pipeline: split-level metrics, bet sizing, path stitching, financial performance per path, Deflated Sharpe Ratio, Probability of Backtest Overfitting, DeLong pairwise AUC tests, model comparison tables, and feature stability diagnostics.

### Core concepts

**Split-level metrics.** Per split's test fold: accuracy, F1 macro, F1 per class, precision, recall, log-loss, Brier score, AUC-ROC. Sample-weighted where applicable. Handles single-class test folds gracefully (AUC returns NaN).

**Bet sizing (AFML Chapter 10.3).** Implements De Prado's S-curve to convert calibrated probabilities into position sizes:

1. Direction: +1 if P(up) > P(down), else -1.
2. Confidence: `p = max(P(class_0), P(class_1))`, always in [0.5, 1].
3. Z-score: `z = (p - 0.5) / sqrt(p × (1-p))`.
4. Raw bet: `2 × Φ(z) - 1` where Φ is the standard normal CDF.
5. Maximum bet cap: `np.clip(raw_bet, -0.75, 0.75)` (`MAX_BET_SIZE=0.75`) prevents the highest-confidence predictions from dominating the equity curve.
6. Minimum threshold: bets below 0.05 (`MIN_BET_SIZE=0.05`) in absolute value snap to 0 (the abstention mechanism).
7. Discretization: snap to nearest of {0.0, 0.25, 0.50, 0.75}. The 1.0 level was dropped to stay consistent with the 0.75 cap.
8. Apply sign: `bet = direction × |bet|`.

The abstention mechanism is the crucial AFML innovation: predictions with probability near 0.5 produce zero bet, meaning no capital is allocated despite having a directional prediction. This separates classification accuracy (always evaluated) from trading behavior (selective).

**Strategy returns.** `gross_return = bet_size × label_return`, `turnover = |Δbet_size|`, `tx_cost = 0.1% × turnover`, `net_return = gross - tx_cost`. Transaction cost of 10 bps round-trip is reasonable for BTC on major exchanges.

**Path stitching.** Assembles 7 full-span backtest paths from the 28 splits using the path-assignment matrix from `cv.py`. For each `(group_id, split_id)` pair in `path_map[path_id]`, the function looks up that split's stored predictions and **filters down to the events whose positional index falls within `group_bounds[group_id]`** before concatenation. This filter is essential: each split's stored test set covers `k=2` chronological groups concatenated, so without the filter, events from co-tested groups get pulled into the path multiple times. With multiple seeds, calibrated probabilities are averaged across seeds before bet sizing (ensemble averaging reduces prediction variance by ~1/√n_seeds). After concatenation the function asserts no duplicate timestamps and emits a warning if any are detected, so future regressions in path-map construction surface immediately.

`stitch_paths` accepts `event_index` and `group_bounds` as optional inputs; when not supplied, both are derived from `predictions` via `_derive_event_index` (union of all stored timestamp slices) and `_compute_group_bounds` (mirroring the helper in `cv.py`). The orchestrator computes them once and passes them to every per-model stitch call.

**Bug-fix disclosure.** An earlier implementation pulled each split's full test set whenever the split was referenced, double- or quintuple-counting events from groups co-tested with the requested group. The fix was identified by inspecting timestamp duplication in stitched series (the 1/3/5 distribution exposed the path-map structure) and corrected by adding the group filter described above. All path-level metrics in this thesis use the corrected stitching. The bug-fix history is preserved in the methodology chapter as a transparency disclosure.

**Path performance metrics.** Per path: annualized Sharpe (using √365 for BTC's continuous trading), annualized Sortino (downside-only deviation), Calmar (annualized return divided by absolute max drawdown), cumulative return, annualized return, maximum drawdown, time under water, win rate, profit factor, number of trades (`n_trades`), full event count (`n_returns`), elapsed years, average bet size, return distribution skewness and kurtosis. Sortino and Calmar are tail-risk-aware alternatives to Sharpe so the comparison table doesn't reduce to a single risk-adjustment lens.

**Sortino convention.** Penalises only downside volatility. Implemented with target = 0 (any loss is downside) and the root-mean-square of negative returns rather than the std of negatives, which is the standard Sortino & Price (1994) construction. Annualised by `√365` matching Sharpe. Returns inf for paths with no losing days and positive mean (winning streak), 0 if mean is also zero.

**Calmar convention.** `annualized_return / |max_drawdown|`. Common tail-risk-aware metric in CTA / managed-futures literature. Returns inf when `|max_drawdown|` is exactly zero and ann_ret is positive.

**Annualized return formula.** Calendar-time CAGR: `(1 + cum_ret)^(1 / years_elapsed) − 1`, where `years_elapsed = (timestamps[-1] − timestamps[0]).days / 365.25`. An earlier version used `(1 + cum_ret)^(365 / n_events) − 1`, which assumed `n` was the number of daily bars but was actually the number of CUSUM events (~75/year, not 365). That over-stated `ann_ret` by roughly 5–6× for typical multi-year paths. The Sharpe formula's `√365` annualisation is internally consistent with the deflated-Sharpe de-annualisation downstream and is left unchanged.

**Profit factor.** Three-way logic: NaN if `n_trades == 0` (undefined for an empty path), `inf` if there are trades but no losses, normal `Σ(positive returns) / |Σ(negative returns)|` otherwise. The earlier convention returned `inf` for both "trades, no losses" and "no trades at all" and `0.0` for empty paths from `_empty_performance`, which were inconsistent under aggregation. The aggregation step in `compute_model_summary` uses `np.nanmedian` rather than `np.median` because plain `np.median` propagates NaN: if any single path has NaN profit factor (zero trades), the median collapses to NaN even when the other six paths have valid finite values. `np.nanmedian` skips the NaN entries and computes the median over the paths that actually traded. Inf entries (winning streaks) are kept and treated as legitimate large values.

**Deflated Sharpe Ratio (AFML Chapter 14, Bailey & Lopez de Prado 2014).** Corrects observed Sharpe for two biases:

1. **Selection bias**: when comparing N models, the maximum observed Sharpe is biased upward even under the null of zero true Sharpe. Correction adjusts for `E[max SR | n_trials]`.
2. **Non-normality**: the standard Sharpe variance formula assumes normal returns. Mertens (2002) corrects via skewness and kurtosis.

$$\text{DSR} = \Phi\left(\frac{SR_{obs} - E[\max SR]}{\sigma_{SR}}\right)$$

`E[max SR]` uses the exact AFML §14.4 Equation 14.13 inverse-CDF form:

$$E[\max SR] = (1 - \gamma)\,\Phi^{-1}\!\left(1 - \frac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \frac{1}{N\!\cdot\!e}\right)$$

where `γ ≈ 0.5772` is the Euler-Mascheroni constant and `Φ⁻¹` is the inverse standard-normal CDF. An earlier implementation used the asymptotic form `√(2·ln(N)) × (1 − γ/(2·ln(N))) + γ/(2·√(2·ln(N)))`, which is accurate for large N but overestimates `E[max]` by roughly 25-35% at the small N=6 the project actually evaluates. The asymptotic was making the multi-trials correction unnecessarily strict at small trial counts; the two forms converge for large N so there is no regression for hypothetical higher-N runs.

The Mertens (2002) variance term `σ_SR` includes the non-normality correction. The implementation correctly handles scipy's *excess* kurtosis (γ_4 − 3) by converting to the raw kurtosis the Mertens formula expects: `(γ_4 − 1)/4 = (excess + 2)/4`. Includes a clamp on the variance to prevent NaN from numerical edge cases.

The Mertens formula divides by `(n_obs − 1)`. `n_obs` is set to `avg_n_returns` (mean across paths of `len(strategy_returns)`), matching the n used to estimate the Sharpe ratio. Earlier versions used `avg_n_trades` (the strict subset where `bet_size ≠ 0`), which inflated `sr_std` and conservatively understated DSR; the previous convention was internally inconsistent with the Sharpe estimation horizon.

DSR > 0.95 indicates a result that survives multiple-testing correction. DSR < 0.95 may be a statistical artifact.

**Probability of Backtest Overfitting (AFML Chapter 11).** Implements Combinatorially Symmetric Cross-Validation (CSCV). Takes the matrix of (n_models × n_paths) Sharpe ratios:

1. Generates all C(n_paths, n_paths/2) IS/OOS partitions.
2. For each partition, identifies the IS-best model and checks whether it underperforms the OOS median.
3. PBO = fraction of partitions where the IS-best model is OOS-poor.

PBO < 0.3 indicates robust selection. PBO > 0.5 indicates anti-predictive behavior (the in-sample winner systematically loses out-of-sample). With 7 paths under N=8, k=2, the partition count is C(7, 3) = 35 — substantially more than the 10 partitions an N=6, k=2 setup would yield. The denser partition coverage reduces the standard error on the PBO estimate, so the reported point estimate is more trustworthy at this configuration than under the earlier N=6 setup.

**DeLong pairwise AUC tests.** Tests whether two models have significantly different AUC on the pooled predictions across all 28 splits, using the placement-value (midrank) approach of DeLong et al. (1988) with the closed-form covariance estimator. For each pair of models:

1. For each (model, split), average predicted probabilities across all available seeds (3 for the sklearn-compatible models, 2 for LSTM and KAN). This matches the seed-averaging that `stitch_paths` already performs for path-level financial metrics, so the AUC test reflects the same averaged predictions used by the Sharpe / DSR / PBO results.
2. Pool the seed-averaged predictions across all 28 CPCV splits. Pooling is valid because CPCV test sets are non-overlapping.
3. Compute AUC for each model on the pooled data.
4. Compute the variance of each AUC and the covariance between them.
5. z-statistic: `(AUC_a - AUC_b) / sqrt(Var_a + Var_b - 2 × Cov_ab)`.
6. Two-sided p-value from the standard normal.

An earlier version of `compute_auc_significance` used only `seed=0` predictions, which made the AUC values and z-statistics depend on which initialisation happened to be labelled seed 0. Averaging across seeds before pooling removes this arbitrary dependence and uses the full available signal.

Reports as a DataFrame with columns model_a, model_b, auc_a, auc_b, delta_auc, z_stat, p_value, significant (α=0.05). The notebook displays "X/Y pairs significantly different" as a top-line robustness statistic.

**Model comparison.** Ranks models by median path Sharpe (descending) with std Sharpe as tiebreaker (ascending, prefer consistency). Reports rank, model_name, median Sharpe, std Sharpe, bootstrap 95% CI lower and upper bounds on the median Sharpe, DSR, median Sortino, median Calmar, mean F1, accuracy, AUC, median drawdown, cumulative return, win rate, and profit factor in a single table.

**Bootstrap confidence interval on the median Sharpe.** `bootstrap_median_ci(values, n_bootstrap=1000, alpha=0.05, seed=42)` resamples each model's seven path Sharpes with replacement, computes the median for each resample, and returns the 2.5th and 97.5th percentiles of the resulting median distribution. Non-finite entries (NaN, inf) are dropped before resampling so paths with undefined Sharpe don't poison the percentile estimate. The CI complements DSR (parametric, AFML-corrected for selection bias and non-normality) with a non-parametric robustness check: a model whose CI crosses zero has a median Sharpe that is not statistically distinguishable from zero under simple resampling, regardless of what DSR says. Both bounds appear as columns in the comparison table.

**Buy-and-hold benchmark.** `compute_buy_and_hold_summary(predictions, path_map, n_paths, reference_model)` produces a model-summary-compatible row using the same CPCV path structure the models see. For each path it reconstructs the chronological sequence of test-fold returns from the predictions dict (timestamps and per-event returns are model-invariant for a given split/seed, so any model serves as a reference), applies bet_size = 1.0 for every event (always 100% long), incurs transaction cost only on the initial buy of each path, and computes path-level metrics identically to the model paths. Returns NaN for the predictive metrics (F1, accuracy, AUC, log loss, Brier) since the benchmark makes no probabilistic predictions, and NaN for DSR since the metric requires a Sharpe under multiple-trials selection. `analyze_results` calls this function automatically and appends its row to `all_summaries` before producing the comparison table; an earlier workflow required a manual append + `compare_models` rerun in the notebook, which has been removed from the locked cells (running it now would double-count the BH row). The benchmark intentionally is NOT added to `path_sharpes_matrix`: PBO is a model-selection diagnostic on candidate models, and BH is not a selectable candidate. Note: the benchmark is a more aggressive position-size baseline than the models, which cap at `MAX_BET_SIZE = 0.75` via the S-curve, so beating buy-and-hold's 1.0-leveraged Sharpe is methodologically harder than the asymmetry would suggest.

**Feature stability.** Counts how often each feature is selected across all `(split, seed)` pairs for the first non-AR reference model. Because `prep_info` is computed once per split in `pipeline.py` and then stored under every `(model, split, seed)` key, all seeds for a given split contribute the same selection list — the seed loop scales numerator and denominator equally, leaving the per-feature frequency identical to a seed=0-only count but symmetric with the AUC and FFD-stability diagnostics. Features selected in > 80% of folds are flagged as "stable." Plotted as a horizontal bar chart in the notebook.

**FFD stability.** Collects FFD d* values across all `(model, split, seed)` entries. FFD is shared across models within a fold and deterministic given the training fold, so the only meaningful source of dispersion is across-split training-fold variation. Including all seeds yields a denser histogram without changing the qualitative result; mean and population std are unaffected by the per-fold replication. Warns if std > 0.1 (heterogeneous stationarity structure across time periods).

**Display helpers.** Three thin renderers consume `analysis` (the dict returned by `analyze_results`) and `results` (from `run_cpcv_pipeline`) and print formatted tables to the notebook output, so the statistical-robustness cell now reads as three calls instead of forty lines of inline print formatting:

- `render_ffd_stability(analysis)` prints the per-FFD-column table with mean d*, std d*, and the min/max observed d* across folds.
- `render_deflated_sharpe_table(analysis, threshold=0.95)` prints per-model DSR sorted descending with a verdict column (`✓ pass` / `fail`) against the threshold; NaN DSR (e.g. the buy-and-hold benchmark) renders as `n/a` and sorts last.
- `render_pbo_summary(analysis, results)` prints baseline PBO followed by a leave-one-out table that recomputes PBO after excluding each model in turn, ranked by the magnitude of the resulting `Δ vs baseline`.

These are presentation-layer helpers only; all the statistical computation happens upstream in `analyze_results` and `compute_pbo`.

### Key function: `analyze_results`

Orchestrates the entire post-CPCV analysis. Called from the notebook as `analysis = analyze_results(cpcv_results)`. Chains: split metrics → path stitching → path performance → model summaries → comparison table → PBO → DeLong AUC → feature stability → FFD stability. Returns a dict consumed by the notebook's visualization cells.

Each component (`compute_pbo`, `compute_auc_significance`, `compute_feature_stability`, `compute_ffd_stability`, `stitch_paths`, `compute_split_metrics`, `compute_model_summary`, `compare_models`) is also exposed as a standalone function, so the notebook can call them à la carte for individual results subsections without going through the full `analyze_results` orchestration.

---

## `post_cpcv/diagnostics.py` — Interactive Inspection Helpers

### What it does

Provides standalone diagnostic and rendering helpers for the notebook's results section. The module has two families of functions: data-producing helpers that return DataFrames or arrays for further analysis, and rendering helpers that consume `results` or `analysis` directly and return matplotlib `Figure` objects. Both families operate on `cpcv_results["predictions"]` and `analysis["path_results"]` only; they never touch the canonical event-aligned X / y / w / t1 series, so they cannot accidentally shadow notebook globals.

### Core concepts

**Why a separate module.** `evaluation.py` is the canonical AFML evaluation pipeline (DSR, PBO, DeLong, comparison table). It runs once per CPCV experiment and produces the headline numbers. `diagnostics.py` holds inspection helpers that get called interactively during writing — auditing calibration choice, profiling bet behaviour, looking for regime concentration in path returns, rendering the figures that the thesis chapters need. The two modules have different lifetimes and different stability requirements, and separating them keeps `evaluation.py` focused.

**Rendering helpers do not call `plt.show()`.** Every `render_*` function returns the `Figure` object so the notebook controls when (and whether) to display, and so the same function can be called from a script that saves figures to disk via the `save_path` parameter without ever rendering them on screen.

**Function families.**

1. **Calibration audit.** `pool_predictions(model, results)` returns `(proba_pool, y_pool)` arrays concatenated across all (split, seed) combinations for a given model. `calibration_audit(model, results)` prints a binned predicted-vs-empirical comparison table — the diagnostic that exposed the temperature-vs-vector-scaling issue earlier in development.
2. **Path-level dispersion and regime concentration.** `compute_top_k_concentration(returns, k=5)` quantifies how much of a path's cumulative return comes from its top-K largest-magnitude returns; a high concentration share indicates regime-fluke. `build_path_dispersion_table(analysis)` assembles a (model, path) DataFrame with Sharpe, cumulative return, drawdown, top-K share, and the date range of the top-K returns. `summarize_path_dispersion(dispersion)` collapses this to one row per model with min/median/max statistics across paths.
3. **Bet-size distribution.** `compute_bet_size_summary(analysis)` reports per-model abstention rate, mean and median absolute bet size among traded events, share at the cap, and long/short balance. `collect_bet_sizes(analysis, model)` returns the raw bet-size array for a given model, pooled across paths, for histogram plotting.
4. **Reliability curves.** `compute_reliability_curve(model, results)` returns binned `(predicted_mean, empirical_mean, n_samples)` triples ready for plotting as a reliability diagram.
5. **Calibration mean audit.** `calibration_mean_audit(results, y, tolerance=0.03)` computes each model's mean calibrated P(Up) across all (split, seed) entries and compares to the empirical base rate. Models whose mean P(Up) deviates from the empirical base rate by more than `tolerance` are flagged with a warning glyph in the printed table. The function auto-handles both `{0, 1}` and `{-1, +1}` label spaces and returns the audit as a DataFrame for downstream use. This is the cheapest calibration check — a one-line numerical summary per model that frames the deeper per-bin diagnostic produced by `compute_reliability_curve` / `render_reliability_diagrams`.
6. **Confusion-matrix renderer.** `render_confusion_matrices(results, seed_mode="average", n_cols=3, save_path=None)` produces a grid of confusion matrices, one per model. The default `seed_mode="average"` averages calibrated probabilities across all available seeds and thresholds at 0.5, matching what `stitch_paths` and `compute_auc_significance` do for the financial and AUC metrics. The `seed_mode="seed_0"` / `"seed_1"` / `"seed_2"` modes use the per-seed `y_pred` directly for diagnostic inspection of single initialisations. Label space is auto-detected.
7. **Feature-stability bar chart and table.** `render_feature_stability(feat_stab, threshold_stable=0.80, threshold_moderate=0.50, save_path=None)` renders the colour-banded bar chart (green ≥ 80%, blue ≥ 50%, grey otherwise). `print_feature_stability_table(feat_stab, all_features, threshold_pct=50)` prints a categorised text table of features above the threshold (separated into stable and moderate tiers), then a four-tier summary (stable / moderate / low / never-selected) plus a list of never-selected features grouped by category (TA / Math / External / Lag). Reindexing against `all_features` recovers the never-selected set, which is absent from `feat_stab["feature_frequency"]`. Returns the full ranking DataFrame for downstream use.
8. **Sharpe-distribution boxplot.** `render_sharpe_distribution(analysis, models=None, figsize=(10, 5), save_path=None)` renders a per-model boxplot of path-level annualised Sharpe ratios. Each box summarises the 7 backtest paths produced by CPCV under N=8, so the spread of the box reflects sensitivity to the path subsampling.
9. **Reliability-diagram grid.** `render_reliability_diagrams(results, analysis, n_cols=3, n_bins=10, min_count=10, save_path=None)` renders a grid of per-bin reliability curves with marker size proportional to bin sample count. Pulls `n_seeds` from `results` and silently skips missing keys, so the same call works for models trained with different seed counts.
10. **Bet-size histogram grid.** `render_bet_size_histograms(analysis, bins=None, n_cols=3, save_path=None)` renders a grid of histograms over the four-step bet discretisation (default bin edges target ±0.25, ±0.50, ±0.75 with abstention straddling zero). Custom `bins` can be passed to match alternative discretisation schemes.
11. **Regime-concentration scatter.** `render_regime_concentration_scatter(dispersion, k=5, concentration_threshold=0.5, save_path=None)` renders top-K share against path Sharpe with one point per (model, path). The vertical reference at `concentration_threshold` flags the regime-fluke risk band. Consumes the DataFrame produced by `build_path_dispersion_table`.

**`n_seeds` and `n_splits` default-resolution.** Functions in this module that pool predictions across the `(model, split, seed)` grid (`pool_predictions`, `calibration_audit`, `compute_reliability_curve`) accept `n_seeds` and `n_splits` as optional keyword arguments. When not passed (default `None`), they read `results["n_seeds"]` and `results["n_splits"]` so the diagnostic stays in sync with the actual pipeline configuration without the caller having to track those values. Pass explicit integers to override (e.g. for sensitivity checks that pool a subset of seeds). This default change replaced an earlier version where these arguments were hardcoded to `n_seeds=2, n_splits=15`, which silently produced incorrect pooling once the locked configuration moved to `n_seeds=3` and `n_splits=28`. Callers that pass explicit values continue to work unchanged.

### Why this matters for the thesis

Each function corresponds to one paragraph or one figure in the thesis chapters:

- The calibration mean audit (5.5.a) is the one-line numerical readout per model used to confirm calibration is doing its job. Anchors the methodology chapter's defence of vector scaling.
- The reliability-diagram grid (5.5.b) provides the per-bin visual elaboration of the mean audit.
- The feature-stability bar chart and table (5.5.c, 5.5.d) audit MDA selection consistency across folds, including the never-selected list which surfaces structural patterns (e.g., entire feature category unused in a low-signal regime).
- The confusion-matrix grid (5.1.b) reveals the directional prediction bias that the comparison-table accuracy column alone cannot show — and which, combined with the calibration mean audit, supports the thesis argument that "classification accuracy is not the right metric for trading-strategy evaluation."
- The Sharpe-distribution boxplot (5.2.b) makes path-level variance visible, framing the DSR and PBO numbers that follow.
- The path dispersion summary, top-K concentration, and regime-concentration scatter (5.2.c) jointly address the "is this skill or one lucky window" question that examiners will ask of any path-level Sharpe above 1.
- The bet-size summary table and histograms (5.3.a, 5.3.b) surface the abstention mechanism's effect on each model's trading behaviour, completing the chain from calibration → bet sizing → financial performance.

The functions are designed to be quick to call from a notebook cell with no setup — every helper takes only the `results` or `analysis` dict (and occasionally `y` or `X.columns`) that already exists at that point in the notebook, and returns either presentation-ready output or a `Figure` object the notebook displays with `plt.show()`.

---

## `post_cpcv/path_explorer.py` — Per-Path Equity Curve Visualisation

### What it does

Three views over the per-path equity curves produced by `analyze_results`: a single-model figure (`plot_paths_for_model`), a multi-model grid (`plot_paths_grid`), and an ipywidgets-driven interactive selector (`interactive_path_explorer`). All three share a common `collect_path_equities` helper that walks `analysis["path_results"][model_name]` and assembles a DataFrame whose columns are the individual paths and whose rows are the per-event timestamps.

### Why this matters

The CPCV evaluation produces 7 prediction paths per model and the headline tables (Sharpe, DSR, PBO) collapse those into medians and confidence intervals. The path-by-path views let the reader see the dispersion the headline statistics summarise: a 0.48 median Sharpe with [0.10, 1.24] CI looks different on the page than the same number from seven tightly clustered paths. The interactive selector additionally supports a "since" period filter for inspecting performance during a specific market regime (e.g., "since 2022" for the institutional bull market).

### Implementation details

**Single-display contract on `plot_paths_grid`.** The grid plot calls `plt.close(fig)` immediately before returning the figure. This detaches the figure from matplotlib's inline-backend auto-display so the inline render doesn't fire; IPython then renders the returned figure exactly once via the figure's `_repr_png_`. Without this, calling `plot_paths_grid(results, analysis)` bare (no assignment) produces two identical plots side by side because (1) the inline backend auto-displays figures created in the cell, and (2) IPython also displays the function's return value. Callers who assign the return value (`fig = plot_paths_grid(...)`) still get a usable Figure for `savefig` or further customisation; the detach is invisible from the caller's perspective.

**Year coverage in `interactive_path_explorer`.** The "since" period dropdown derives its year options dynamically from the data so that 2025+ appears automatically as the data extends. The collection logic walks every path's full date index and unions all unique years present (`years_in_data.update(int(y) for y in sr.index.year.unique())`), then keeps the most recent five at or after 2020. An earlier version added only each path's first and last year, which with CPCV's non-uniform path ranges silently dropped intermediate years; with the user's 2014-2026 data range, years like 2022 and 2023 frequently failed to appear in the dropdown because no path endpoint happened to fall in them even though every path crossed those years. The current logic guarantees that any year covered by the data shows up in the option list.

---

## `post_cpcv/symbolic_extraction.py` — KAN Symbolic Formula Extraction

### What it does

The thesis's novel methodological contribution. Re-trains a PyKAN model on a selected fold using the CPCV-tuned architecture (with simplifications for tractability), prunes weak edges, replaces survivors with closed-form symbolic functions, and extracts a SymPy formula relating features to predicted probability. The output is a closed-form mathematical expression for `P(up | features)`.

### Core concepts

**Why a separate symbolic extraction.** The CPCV KAN uses `efficient-kan` for fast and reliable training across all 28 folds. PyKAN is the canonical library that supports symbolic operations (`prune`, `suggest_symbolic`, `fix_symbolic`, `symbolic_formula`) but is fragile and slow. Extracting symbolic formulas from `efficient-kan` is not supported. The solution: use `efficient-kan` for benchmarking, retrain with PyKAN on a single fold for symbolic extraction, ensure both share the same input normalization and architecture template.

**Workflow honesty.** What's transferred from CPCV to symbolic extraction is the *hyperparameter set* (width1, width2, grid, k), not the learned weights. The two libraries use different spline parameterisations under the hood, so direct weight transfer is not feasible. The PyKAN is retrained from scratch on the same training fold via the staged Adam → LBFGS schedule below. The extracted symbolic formula therefore represents a PyKAN with matching architecture and training data, whose decision boundary approximates but does not equal the CPCV-evaluated efficient-kan's. The pre-symbolic validation accuracy and the pre/post-symbolic accuracy gap together quantify how close this approximation is for any given fold; large gaps signal the symbolic formula is not faithful to the CPCV-evaluated KAN and the thesis chapter flags the fold accordingly.

**Algorithm 1 from the VIX KAN paper.** Four steps:

1. **Train.** Phase 1 uses Adam (600 steps, lr=1e-3, weight decay=1e-3, Gaussian noise injection on inputs). Phase 2 uses LBFGS in two stages: warmup (no regularization) then sparsity (L1 + entropy regularization).
2. **Prune.** Forward pass to populate cached activations, then `model.prune(threshold=0.01)` with multi-API fallback for PyKAN version differences. Verifies pruned model can still forward-pass.
3. **Symbolify.** For each surviving edge, calls `suggest_symbolic` with a custom function library {x, x², x³, x⁴, exp, log, sqrt, tanh, sin, cos, abs, sgn, arctan, 0}. Falls back to PyKAN's default library on `KeyError`. Implements brute-force candidate testing when `"0"` (constant) wins by zero complexity penalty. If best R² ≥ 0.3, the edge is replaced with the symbolic function via `fix_symbolic`.
4. **Affine fine-tune.** After symbolification, the remaining affine parameters (a, b, c, d per symbolic edge) are LBFGS-optimized for 30 steps to compensate for substitution error. NaN detection reverts to the pre-fine-tune state.

**Architecture coupling to CPCV (faithful by default).** The symbolic re-training reads tuned hyperparameters from `cpcv_results["tuning_results"][best_split]["kan"]["best_params"]` and uses them verbatim, subject to a data-aware safety floor. Three resolution rules:

1. *width1* uses the tuned value, capped at `PYKAN_SYMBOLIC_WIDTH_CAP = 8` (the locked KAN tuning maximum is 6, so the cap never bites in practice; it remains as a guard against future tuning expansions).
2. *width2* uses the tuned value when `PYKAN_SYMBOLIC_DROP_WIDTH2 = False` (the locked default). The locked KAN tuning hardcodes `width2 = 0`, so the symbolic re-training also uses single-hidden-layer architecture by construction.
3. *grid* uses the tuned value when `PYKAN_SYMBOLIC_FORCE_GRID = None` (the locked default). The locked KAN tuning searches `grid ∈ {3, 5}`, so the symbolic re-training inherits whichever of those won on the chosen fold. `PYKAN_FALLBACK_GRID = 3` is used only as a last-resort fallback when no tuned grid is available.

The architecture used for CPCV evaluation now matches the architecture extracted in Phase 3 by construction — the symbolic formula reflects the actual benchmark model rather than a simplified surrogate. An earlier configuration hardcoded `PYKAN_SYMBOLIC_DROP_WIDTH2 = True` and `PYKAN_SYMBOLIC_FORCE_GRID = 3` to force a single-hidden-layer, coarse-grid architecture for symbolic clarity even when the tuner picked otherwise. With width2 already hardcoded to 0 in the CPCV tuning stage and grid ∈ {3, 5} both producing tractable formulas, the legacy override no longer adds value and was switched off; the override constants remain in the source as escape hatches if a future tuning configuration produces a sympy-intractable formula on a given fold.

**Data-aware safety floor.** Independent of the tuned width, the function applies a samples-per-parameter floor. If `n_train / total_params < 5`, the hidden width is reduced. For a typical extraction fold (6 of 8 CPCV groups ≈ 860 events into `prepare_extraction_data`, then 80/20 split inside the symbolic pipeline → ~690 training events) with grid=3 and k=3, this typically caps hidden width at 4-5 regardless of the tuned value. The floor activates only when the tuned configuration would produce a parameter count the data cannot support.

**Reproducibility seeding.** `train_pykan` seeds `torch.manual_seed`, `torch.cuda.manual_seed_all` (when CUDA is available), `numpy.random.seed`, and `random.seed` at function entry. The PyKAN constructor's own `seed=42` covers only spline initialisation; the Adam optimiser, the Gaussian-noise tensor injected on each Adam step, and any PyKAN-internal RNG calls are otherwise unseeded. Without this, repeat extractions on the same fold produced slightly different formulas across runs.

**Console-output discipline.** Each long-running step is wrapped in a `tqdm.auto.tqdm` progress bar with `leave=False`, so the bar appears while the phase runs and is replaced by a single result line on completion. This applies to the Adam phase (600 steps), both LBFGS phases (20 steps each), and the per-edge symbolify loop (~15 edges). The architecture preamble that earlier versions printed across five lines (width1 / width2 / grid / edge count / params/sample ratio) is now consolidated into one dense line. The earlier per-phase `[After Adam]` / `[After LBFGS warmup]` / `[After LBFGS sparsity (final)]` checkpoint prints have been removed from stdout; `_log_diagnostic` now defaults to `verbose=False` and the file-level logger still receives the per-phase metrics for audit retrieval.

A module-level `_suppress_pykan_stdout()` context manager wraps the noisier pykan API calls (`model.suggest_symbolic`, `model.fix_symbolic`, `model.prune`, `model.attribute`, `model.refine`) so the cell output does not interleave the per-edge `function | fitting r2 | r2 loss | complexity | total loss` pandas tables or the `saving model version 0.x` / `checkpoint directory created` lines that pykan emits on every call. The one pykan call NOT wrapped is the affine fine-tune `model.fit`, whose own tqdm progress bar with `train_loss` / `test_loss` / `reg` postfix carries information worth showing. Per-edge debug detail (custom-library fallbacks, parse errors, the [DEBUG] suggestion dump for the first few edges, the Top-5 edges by R²) all flow to the file logger at the appropriate level rather than to stdout.

The Results block at the end of `run_symbolic_extraction` also truncates the printed decision function at 240 characters by default, with a note about accessing the full string via `result['decision_function']`. The follow-up `P(up) = 1/(1+exp(-decision))` print is now a one-line reference rather than a re-printing of the same expression wrapped in sigmoid. Long sympy expressions (typically 1k+ chars) carry a non-trivial Jupyter rendering cost that the truncation removes.

**Grid extension gated by sample count.** `PYKAN_GRID_EXTEND = True` enables the grid-refinement feature; a runtime check then disables it on folds where `n_train ≤ 1000` to avoid adding spline parameters faster than the data can support. The locked configuration's ~690-event training partition triggers the runtime guard on every fold, so in practice no grid extension occurs at the locked dataset size. Setting the constant to `False` disables the feature unconditionally.

**Majority-class-aware accuracy gate.** `PYKAN_MIN_ACCURACY = 0.53` is a floor; the actual gate is `max(0.53, majority_baseline + 0.01)` where `majority_baseline = max(P(y=0), P(y=1))` from the training labels. If the post-`drop_rare(0.085)` class balance is 55/45, the gate becomes 0.56 rather than 0.53; an absolute 0.53 floor would have let a "predict the majority class" trivial baseline pass on imbalanced folds. The gate logs a warning rather than raising, so symbolic extraction proceeds on weak folds with visible methodological caveat in the cell output.

**Fold selection.** The `fold_selection` parameter controls which CPCV fold is used: `"best"` picks the highest-F1 fold, `"last"` picks the most recent, or an integer specifies a fold directly. The thesis uses `"last"` to match what would be used in deployment (most recent training data).

**Feature ranking and subsetting.** When `n_top_features` is set, features are ranked by selection frequency across all KAN CPCV folds, and only the top N are used. Fewer features produce simpler formulas at some accuracy cost.

**Tanh input normalization match.** The symbolic re-training applies the same tanh normalization as the CPCV KAN (fitted on the symbolic re-training fold), ensuring inputs hit the spline grid range [-1, 1].

**Defensive input handling.** The entry point of `prepare_extraction_data` coerces `y` to a `pd.Series` indexed on `X.index` before any indexing, regardless of whether the caller passed a Series, a numpy array, or another array-like. If the supplied `y` length does not match `X`, the function raises a clear `ValueError` rather than letting pandas's generic length-mismatch error propagate. This catches a common notebook pattern where `y` gets shadowed by a pooled-prediction array (e.g., from a calibration audit cell that does `y = np.concatenate(...)`) and fails fast with a message naming the alignment requirement.

**SymPy simplification: deliberately omitted.** After formula extraction, only `sympy.nsimplify` is run (for cleaner rational coefficients). `sympy.simplify` itself is no longer called on the decision expression. On KAN-shape expressions (nested tanh, sin, cos, polynomial) it rarely produces a more readable form and routinely runs for many minutes. An earlier version wrapped the call in a daemon thread with a 30-second wall-clock timeout, but Python threads cannot be cancelled cooperatively — when the timeout fired, the worker thread continued running `sympy.simplify` in the background and held the GIL for hours afterward, starving every subsequent kernel operation. This surfaced in the notebook as multi-tens-of-minutes apparent wait times on trivial `print()` calls in cells run after `run_symbolic_extraction`. Removing the `sympy.simplify` call eliminates the zombie-thread failure mode; the `nsimplify` pass handles the rational-cleanup that's actually useful, and the unsimplified expression is what the user has been quoting from the Results block all along.

**Output.** The function returns a dict containing:
- `logit_bearish`, `logit_bullish`: closed-form expressions for each class logit.
- `decision_function`: their difference (logit_bull - logit_bear).
- `p_up_formula`: `1 / (1 + exp(-decision))`.
- `sympy_objects`: the SymPy expression objects for downstream differentiation/analysis.
- `pre_symbolic_accuracy`, `post_symbolic_accuracy`: validation accuracy before and after symbolification.
- `symbolification_rate`: fraction of edges successfully symbolified.
- `pruned_architecture`: final widths after pruning.
- `surviving_features`: feature names that appear in the final formula.
- `input_transform`: per-feature `(a, b)` parameters such that `z = tanh((x_raw − a) / b)` reproduces the raw → scaler → tanh-normalise chain. Downstream consumers (`compute_feature_sensitivity`, `plot_marginal_effects`) MUST use these to convert raw evaluation points to tanh space before substituting into the formula; the formula's variables represent tanh-normalised values, not raw values, despite being renamed to the original feature names.

**Notebook display helpers.** A second tier of functions in this module consumes the dict returned by `run_symbolic_extraction` and produces the formatted output / plots that the notebook used to generate inline. Factoring them here keeps the notebook readable and centralises the singularity-handling logic for derivative evaluation:

- `print_symbolic_decision(symbolic, max_chars=None)`: prints the decision function in full by default, a one-line reference to the P(up) formula (which is just `1 / (1 + exp(-decision))`), and the surviving features. The optional `max_chars` parameter accepts an integer for truncated previews if a particular Jupyter frontend struggles to render very long single lines. Earlier defensive variants of this helper used `sys.stdout.write` followed by explicit `flush()` calls per output chunk, which in Jupyter forces a synchronous kernel-to-frontend ZMQ round-trip per write; on a single 1-2 KB sympy expression that pattern ran 14× slower than `print()` in pure Python and dramatically worse in Jupyter where queued messages backed up across cells. The current implementation is three plain `print()` calls and runs in microseconds.
- `print_extraction_metrics(symbolic)`: prints pre/post symbolic accuracy, symbolification rate, and the pruned architecture. The accuracy gap (pre minus post) measures the cost of replacing spline activations with closed-form symbolic primitives. A fixed-size readout of pre-computed floats; if this helper ever appears to take tens of minutes the cause is a zombie `sympy.simplify` thread from an earlier `run_symbolic_extraction` call holding the GIL (see the SymPy-simplification paragraph above), not this helper itself. A kernel restart resolves it.
- `print_partial_derivatives(symbolic)`: prints the closed-form partial derivative of the decision function with respect to each surviving feature, useful as a thesis appendix item.
- `compute_feature_sensitivity(symbolic, X, eval_point="mean")`: returns a per-feature sensitivity DataFrame with `mean_value`, `std_value`, `d_decision/d_feature_at_center`, `sigma_effect_on_decision`, and `sigma_delta_p` columns. The `at_center` and `delta_p` suffixes are deliberate: the centre point is either the dataset mean or the per-feature median depending on the `eval_point` kwarg, and the probability response is the formula's literal value at the centre rather than a worst-case approximation (see the rewrite below). Set `eval_point="median"` to evaluate gradients at the per-feature median, which is more robust for skewed distributions and typically avoids the singularity issues that the singular-gradient handling paragraph documents.
- `print_feature_sensitivity(sensitivity_df)`: prints the dataframe with NaN-aware formatting (NaN renders as `"   N/A "` rather than `"+nan"`) and a footer line counting the singular features.
- `plot_marginal_effects(symbolic, X, n_points=100, quantile_low=0.05, quantile_high=0.95)`: returns a matplotlib figure with one subplot per surviving feature, sweeping each feature across its empirical `[q_low, q_high]` *raw* range, converting each sweep point to tanh space via the stored `input_transform`, holding the other features at their raw medians (also converted), and substituting the tanh-space values into the formula. The previous version swept in raw space and substituted raw values directly into the formula, producing curves that traced the formula's behaviour outside the spline grid where its values are not interpretable. Returns the figure rather than calling `plt.show()` so the caller can save it for the thesis appendix.
- `print_term_structure_summary(sensitivity_df, symbolic)`: prints the term-count summary alongside the sensitivity table — a per-feature count of how many times the feature's name appears in the closed-form decision expression, as a proxy for structural importance independent of the numerical sensitivity.

**`compute_feature_sensitivity` — four corrections applied together.** The earlier inline notebook version of this code accumulated four methodological gaps that the helper now resolves:

*1. Input-space evaluation via the stored transform.* The formula's variables live in tanh-normalised space even though they're renamed to the original feature names; substituting raw values silently evaluates the formula at the wrong point. The helper resolves this by reading `input_transform` from the symbolic result and converting every centrality point: given raw centre `x_raw` and per-feature `(a, b)`, the tanh-space value is `z = tanh((x_raw − a) / b)`. The symbolic partial derivative is evaluated in tanh space, then transformed back to raw space via the chain rule: `df/dx_raw = (df/dz) × (1 − z²) / b`. The function raises `ValueError` if `input_transform` is missing rather than producing wrong numbers from a raw-space substitution.

*2. Probability-effect uses the actual sigmoid slope, not the worst-case 1/4.* The reported probability impact of a one-σ feature move is `sigma_delta_p ≈ p_center × (1 − p_center) × Δlogit_σ`, where `p_center` is the symbolic-formula's predicted probability at the centre point and `Δlogit_σ` is the change in the decision function across `±σ` of the feature in raw space (computed by finite difference, since the chain-rule gradient is only locally accurate for non-trivial activations). The earlier `1/4 × Δlogit_σ` form used the maximum slope of the logistic at `p = 0.5` regardless of the actual operating point, overstating `sigma_delta_p` by up to 4× on high-confidence folds where the centre point sits in the saturation region. The renamed column is `sigma_delta_p` (was `approx_sigma_delta_p`).

*3. Column rename.* Centre points may be the dataset mean or the per-feature median depending on the `eval_point` kwarg, so the columns are renamed `d_decision/d_feature_at_center` and `sigma_delta_p` (the latter without the `approx_` prefix that suggested the value was a rough estimate when in fact it's now the formula's literal probability response at the centre).

*4. Singular-gradient handling (`_safe_eval_at_point`).* PyKAN's symbolic library includes `1/x`, `log(x)`, and similar reciprocal/logarithmic primitives that produce poles in the learned activation. When the symbolic gradient is evaluated at a point near such a pole — which is common for heavily right-skewed features whose mean lands far from the bulk of the distribution, e.g. `jarque_bera` with mean ≈ 218 and std ≈ 775 — `float(deriv.subs(point))` returns `inf` or `-inf`, which then propagates silently through the `sigma_delta_p` column. The `_safe_eval_at_point` helper catches `ZeroDivisionError`, `TypeError`, `ValueError`, and `OverflowError` from sympy substitution and applies an `np.isfinite` check, returning NaN for any non-finite result. The display helpers render NaN as `"   N/A "` and emit a footer line counting the singular features. The `eval_point="median"` kwarg on `compute_feature_sensitivity` provides a robust alternative for heavily-skewed distributions: the per-feature median typically falls in the bulk of the distribution and avoids the singularity.

### Known PyKAN fragilities

The ~2,300-line file contains extensive defensive code for PyKAN's inconsistent APIs:

- `'sigmoid'` is not in PyKAN's internal `SYMBOLIC_LIB`, causing `KeyError` (handled with fallback to default library).
- `'1/x'` causes division-by-zero during affine fine-tuning (excluded from custom library).
- PyKAN uses 1-based variable naming (`x_1..x_n`), not 0-based (handled by trying three naming conventions).
- `suggest_symbolic` return format varies across PyKAN versions (handled by three format parsers: DataFrame, flat tuple, nested tuple).
- `sympy.simplify` can hang indefinitely on KAN-shape expressions. The call is no longer made; an earlier 30-second threading timeout could not actually cancel the worker on expiry and left a zombie thread holding the GIL for hours afterward, starving every subsequent kernel operation.
- `"0"` (constant function) always wins `total_loss` due to zero complexity penalty. Handled by brute-force candidate testing with stdout regex parsing for R² extraction; two regex patterns (`r"r2 is ([\d.eE+-]+)"` and `r"r\^?2\s*[:=]\s*([\d.eE+-]+)"`, both case-insensitive) tolerate format drift across PyKAN versions, and a warning fires when every candidate misses both patterns on a given edge so silent regressions in PyKAN's log format surface immediately rather than through downstream sensitivity diagnostics.
- `model.attribute()` failures during pruning log a warning and print to stdout, since pruning that proceeds with stale attribution scores produces an unreliable architecture that should be visible in the notebook output rather than buried in the log handler.

### Why this matters for the thesis

KAN symbolic extraction is the novel methodological contribution. The benchmarking exercise (KAN vs LSTM vs trees vs logistic) is informative but not novel — comparable studies exist for crypto direction prediction. What no prior thesis or paper in this exact intersection has done is extract a closed-form formula from a CPCV-deployed KAN and present it alongside benchmark performance. The output formula is the deliverable that distinguishes this thesis from a standard ML benchmarking study.

### Honest limitations to disclose

1. The data-aware safety floor may reduce the architecture width below the tuned value when the samples-to-parameters ratio falls below 5, producing simpler but less expressive formulas than the CPCV-evaluated model used. This typically caps hidden width at 4-5 for the locked configuration's ~690 training events even when the tuner picked width1 = 6.
2. Symbolic accuracy is reported on the validation split of a single CPCV fold, not across all folds. The formula reflects one fold's signal structure.
3. PyKAN's `fix_symbolic` introduces substitution error that affine fine-tuning partially compensates for; the residual error contributes to the gap between pre- and post-symbolic accuracy.
4. What's transferred from the CPCV-evaluated KAN is the hyperparameter set, not the learned weights, so the symbolic formula reflects a PyKAN with matching architecture and training data rather than the exact efficient-kan model that produced the CPCV metrics. The pre-symbolic validation accuracy quantifies how close this approximation is for a given fold.
5. The legacy override constants `PYKAN_SYMBOLIC_DROP_WIDTH2` and `PYKAN_SYMBOLIC_FORCE_GRID` provide escape hatches for sympy-intractable cases. They are not used in the locked configuration (defaults: `False` and `None` respectively, which honor the tuned values), but their existence is disclosed since toggling them on a future fold would re-introduce a CPCV-vs-symbolic architecture divergence.

These limitations are presented transparently in the methodology chapter rather than papered over.