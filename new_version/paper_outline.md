# BTC Daily Direction Prediction Using KANs Within the AFML Framework

**Author:** Petr Terletskiy (l63023)
**Programme:** Master's in Mathematical Finance, ISEG (2024/26)
**Supervisor:** Prof. João Afonso Bastos

---

## How to use this file

Bullet-point content outline, not draft prose. Each subsection lists topics, configuration values, and citations to land in the final text. Convert each block into one to three paragraphs at writing time. The outline assumes the lit review covers all theoretical background; methodology subsections refer back rather than re-explaining.

The section numbering in this outline matches the current thesis table of contents exactly.

### Writing principles in force across the paper

- **Cochrane: structure.** Lead with the contribution. Three pages maximum for the introduction. Lit review is brief, set off, positioned against the 2 to 3 closest papers per stream. Tables and figures get self-contained captions. Two to three significant digits. Less math is better.
- **Cochrane: voice.** Active voice, present tense. Search out passive sentences.
- **Cochrane: style.** No opening quotation, no "the literature has long been interested in", no previews or recalls, no footnotes for parenthetical comments.
- **Thatcher rules.** R1 open boldly; R2 narrative hook (despite A, observe B); R3 lead with the contribution; R4 one construct per paragraph; R5 thread constructs (event, label, observation, fold, path) consistently; R6 parallel logic Q1 to C1 etc.; R7 link results back to mechanisms; R13 compress without hollowing; R15 write for reviewers; R16 end with resonance.
- **User-enforced style.** No em-dashes. No semicolons. British spelling. Italics for class and parameter names, no code style.

---

## 1. Introduction

> Three pages maximum. Lead with the contribution and the headline finding.

### Chapter opening (no numbered subsection)

- One paragraph, three sentences:
  - This thesis applies KANs to BTC daily direction prediction within the full AFML framework.
  - It benchmarks KAN against five models (AR Logistic, Logistic Regression, Random Forest, XGBoost, LSTM) under CPCV with DSR, PBO, and DeLong corrections.
  - The novel deliverable is a closed-form symbolic decision function extracted from the trained KAN.
- Headline finding stated immediately (T-R1, T-R3): no trained model achieves DSR ≥ 0.95, consistent with semi-strong-form EMH under leakage-free evaluation; the symbolic extraction pipeline produces an interpretable formula nonetheless.

### 1.1 Research Context and Motivation [T-R2 narrative hook]

- Hook: despite the crypto-ML literature reporting 85% to 95% daily-direction accuracy, no model in this thesis survives the AFML statistical-correction layer.
- Why BTC: largest cryptocurrency, highest cumulative return of any asset class since 2009, 24/7 transparent on-chain data, daily OHLCV from 2014.
- Why KANs: Liu et al. (2024) introduced an MLP alternative whose edges carry learnable B-splines that can be distilled into closed-form symbolic functions. No prior work applies KANs to BTC daily direction or extracts symbolic formulas from a classification KAN under AFML evaluation.

### 1.2 Problems and Research Questions

> All four questions in one numbered section. State the problem in the field, then state the question this thesis asks.

- **Q1 Predictability under leakage-free evaluation.** Crypto-ML papers report 85% to 95% daily-direction accuracy using fixed-horizon labels and naive train-test splits. Does BTC daily direction remain predictable once labels stop leaking and overlapping observations are downweighted? Does any positive Sharpe survive DSR, PBO, and DeLong?
- **Q2 Which feature families carry signal.** Most crypto-ML papers use one or two feature families. Among technical, statistical, macroeconomic, and on-chain features, which survive multi-model permutation-importance selection across CPCV folds? Are on-chain features the free lunch the literature suggests?
- **Q3 KAN vs standard model families.** KANs have been applied to VIX forecasting and stock prediction but never to BTC direction. Where does a KAN sit relative to AR Logistic, Logistic Regression, Random Forest, XGBoost, and LSTM under identical CPCV splits, features, sample weights, and metrics?
- **Q4 Closed-form formula extraction.** Existing KAN symbolic-extraction work targets regression on relatively predictable series. Can a human-readable expression for P(Up) be extracted from a CPCV-trained KAN in a weak-signal classification regime while preserving most of the predictive accuracy?

### 1.3 Contributions

> One short paragraph per contribution. Map one-to-one onto Q1 to Q4.

- **C1 → Q1. Honest evaluation under full AFML.** End-to-end pipeline applying CUSUM event sampling, triple-barrier labels, sample weights (uniqueness × return attribution × time decay), fractional differencing, CPCV with purging and embargo, plus DSR, PBO, and DeLong corrections.
- **C2 → Q2. A 73-feature universe across four families.** 25 TA, 9 mathematical (AFML Part 4), 29 external (20 macro, 1 crypto-macro, 8 on-chain), plus 10 autoregressive lags. All compete in multi-model MDA selection.
- **C3 → Q3. Six-model apples-to-apples benchmark.** Identical CPCV splits, features, sample weights, and metrics across all six models. Pairwise DeLong AUC tests determine which differences are statistically real.
- **C4 → Q4. Closed-form symbolic formula from KAN.** A human-readable expression for P(Up), pruned and substituted from the trained KAN via PyKAN's symbolic-extraction pipeline.

### 1.4 Document Structure

- One short paragraph naming Sections 2 to 6.

---

## 2. Literature Review

> Position this thesis against the closest 2 to 3 papers per stream. Three streams: KANs, BTC ML prediction, AFML methodology. Each stream ends with a gap statement.
>
> The lit review carries all theoretical background. The methodology chapter refers back rather than re-explaining the Kolmogorov-Arnold theorem, B-splines, CUSUM, TBL, FFD, sample weights, CPCV mechanics, DSR formula, or PBO mechanism.

### 2.1 Kolmogorov-Arnold Networks

> Opening claim: KANs are the only neural architecture whose learned activation functions can be distilled into closed-form symbolic expressions.

- **2.1.1 Mathematical foundation.** Kolmogorov-Arnold representation theorem (any continuous multivariate function decomposes as a finite sum of continuous univariate functions). Considered historically impractical because the inner functions can be non-smooth. Liu et al. (2024) made it practical by parameterising the inner functions with learnable B-splines. **Cite:** `kolmogorov_1957`, `arnold_1958`.
- **2.1.2 How KANs work.** MLPs put fixed activations on nodes; KANs put learnable activations on edges. Each edge is a B-spline `g(x) = Σ a_k · B_k(x)` of order `k` over `G` grid intervals. Architecture notation `[n_input, n_hidden, n_output]`. Training is Adam followed by LBFGS with L1-on-activations and entropy sparsity. Key property: after training, each edge's spline can be inspected, pruned, or replaced with a symbolic function. Training best practices: coarser grids generalise better on small data; Adam-to-LBFGS staging; tanh input normalisation matches the default grid range `[-1, 1]`. **Cite:** `liu_kan_2024`, `noorizadegan_2026`.
- **2.1.3 KAN variants.** KAN 2.0 / MultKAN (Liu 2024) adds multiplication nodes alongside addition; enables multiplicative-interaction discovery without log-exp decomposition. Future-work hook. Broader landscape catch-all: TKAN (Genet and Inzirillo 2024) as the canonical recurrent-KAN variant, ChebyKAN, DecoKAN, and others. The comprehensive review notes symbolic distillation in financial time series is underexplored. **Cite:** `liu_kan2_2024`, `genet_inzirillo_2024`, `yamak_et_al_2025`.
- **2.1.4 KANs for financial time series.**
  - VIX KAN (Cho, Lee, Kim 2025): Algorithm 1 (train → prune → symbolify → affine fine-tune). Direct methodological ancestor of this thesis's symbolic-extraction pipeline. VIX is regression and mean-reverting; BTC daily direction is binary classification in a weak-signal regime. **Cite:** `cho_lee_kim_2025`.
  - KASPER (Oad 2025): KAN with Gumbel-Softmax regime detection for stock prediction; reports R²=0.89 and Sharpe=12.02. Closest related interpretability work but on different target type with no AFML correction layer. **Cite:** `oad_kasper_2025`.
  - DecoKAN (Gao 2025): KAN with trend/seasonality/residual decomposition for crypto forecasting. Closest on the asset-class axis but uses regression target with naive splits. **Cite:** `gao_decokan_2025`.
- **2.1.5 Gap statement.** No prior work applies KANs to BTC daily direction. No prior work extracts symbolic formulas from a classification KAN. No prior work combines KAN symbolic extraction with AFML evaluation.

### 2.2 Bitcoin Price Prediction with Machine Learning

> Opening claim: despite extensive ML coverage of BTC prediction, the field's accuracy numbers do not survive leakage-free evaluation.

- **2.2.1 Overview of approaches.** Three families (statistical/econometric, classical ML, deep learning); two framings (price-level regression, direction classification).
- **2.2.2 Technical analysis and ML for BTC.** Confluence study identifies ROC as a top predictor (justifies `roc_14` inclusion). Limitation: standard split, no purging, no overlap accounting. **Cite:** `mate_confluence_2024`.
- **2.2.3 Deep learning for BTC direction.** Same problem as this thesis with DL models; reports competitive LSTM accuracy. Limitation: fixed-horizon labels, no CUSUM, no sample weights, no purging or embargo. **Cite:** `omole_enke_2024`.
- **2.2.4 Broader crypto-DL landscape and conservative prediction.** Catch-all citations for the wider literature: LSTM/GRU/attention dominance, recurring issues (small datasets, overfitting, no transaction costs, no risk-adjusted metrics). Nabar and Shroff (2023): in noisy financial settings, selective abstention beats always-predict; motivates the bet-sizing threshold in 3.10.1 (`p ≈ 0.50 → bet ≈ 0`). **Cite:** `bourday_crypto_dl_2024`, `wu_crypto_dl_review_2024`, `nabar_shroff_2023`.
- **2.2.5 Methodological weaknesses in the literature [bridge to 2.3].** Synthesise the common problems: naive train-test splits with overlapping labels; fixed-horizon labelling; no sample weighting; accuracy as primary metric; no multiple-testing correction. These are the failure modes the AFML framework was designed to address.

### 2.3 The AFML Methodology

> Opening claim: AFML treats financial ML as a methodology problem first and a model problem second.

- **2.3.1 Overview and motivation.** Lopez de Prado (2018) identifies systematic flaws in financial ML (backtest overfitting, label-overlap leakage, naive CV, inflated Sharpes). Proposes a complete framework addressing each. Asset-agnostic in principle. **Cite:** `lopez_de_prado_2018`.
- **2.3.2 Event-driven sampling, triple barrier labelling, and sample weights.**
  - Symmetric CUSUM (Snippet 2.4) identifies structurally meaningful events. Threshold calibrated as a multiple of mean daily volatility.
  - Triple barrier labelling assigns the label by the first of three barriers touched (take-profit, stop-loss, vertical time limit). Each label carries `t1` for downstream purging. Reflects real trading outcomes. Kang and Kim (2025) validate TBL on Korean equities (cross-asset transferability).
  - Sample weights chain (Snippets 4.1, 4.2, 4.10, 4.11): concurrent label count → uniqueness → return-attribution → time decay. Corrects redundancy in overlapping labels. Passed to classifiers via `sample_weight` or per-sample loss multiplier.
  - **Cite:** `lopez_de_prado_2018` Ch. 3 and 4, `kang_kim_2025`.
- **2.3.3 Fractional differentiation.** Integer differentiation destroys long memory; price levels preserve memory but are non-stationary. FFD finds the minimum `d* ∈ (0, 1)` that achieves stationarity (ADF) while preserving memory. Fixed-width window for efficiency. `d*` estimated per fold on training data only. **Cite:** `lopez_de_prado_2018` Ch. 5.
- **2.3.4 Combinatorial Purged Cross-Validation.** Standard k-fold is invalid because overlapping TBL labels leak across folds. CPCV: `N` contiguous groups, `k` test groups → `C(N, k)` splits, `φ(N, k)` backtest paths. Purging removes train observations whose labels overlap with test (three sufficient conditions). Embargo removes a buffer after each test group. Path matrix enables distribution-level Sharpe analysis. **Cite:** `lopez_de_prado_2018` Ch. 7 and 12.
- **2.3.5 Deflated Sharpe Ratio and Probability of Backtest Overfitting.** DSR corrects observed Sharpe for selection bias across `n_trials` models and for non-normal returns (skew, kurtosis); DSR > 0.95 is significant after correction. PBO via combinatorially symmetric CV: split paths into IS/OOS halves, check if IS-best underperforms OOS median; PBO ≈ 0 reliable, PBO > 0.5 adversarial. **Cite:** `lopez_de_prado_2018` Ch. 11 and 14.
- **2.3.6 AFML in practice: crypto applications and the fitting-scheme finding.**
  - Slepaczuk and Bieganowski (2024) combine FFD and TBL with supervised autoencoders on BTC, ETH, LTC; walk-forward `d*` estimation. Fu et al. (2024) use a GA-driven TBL for crypto pair trading. Both use walk-forward instead of CPCV; neither computes DSR or PBO. This thesis fills the evaluation gap with full CPCV + DSR + PBO and keeps TBL parameters at AFML defaults.
  - Audrino and Chassot (2024) test HAR vs RF, lasso, GBT, FFNN on 1,445 US equities for realised-volatility forecasting; a properly fitted HAR outperforms the ML cross-section. ML superiority results in the literature came from suboptimal fitting schemes. Directly relevant: this thesis's negative DSR results align with the same finding (properly evaluated baselines are hard to beat in weak-signal regimes).
  - **Cite:** `slepaczuk_bieganowski_2024`, `fu_et_al_2024`, `chassot_audrino_2026`, `corsi_2009`.
- **2.3.7 Gap statement.** Few crypto-ML studies adopt the full AFML pipeline; no study combines AFML and KAN for any asset; no study reports DSR and PBO for KAN-based predictions. This thesis is the first to apply the complete pipeline (CUSUM + TBL + FFD + sample weights + CPCV + DSR + PBO) to BTC direction prediction with KANs and five baselines.

---

## 3. Methodology

> Document the pipeline so a fellow graduate student can reproduce every number. Theory referenced from Section 2; this chapter covers the locked-run configuration and the implementation-specific decisions.
>
> Target length: 18 to 20 pages of prose. The current draft sits near 33 pages; cuts target per-model justifications (Section 3.7) and the long argumentation paragraphs in 3.9 and 3.10.

### Pipeline overview (chapter opening)

- Single short paragraph naming the three phases. Use the table for the at-a-glance map.

| Phase | Purpose | Output |
|-------|---------|--------|
| I. Pre-CPCV | Raw OHLCV to labelled events | `(X, y, w, t1)` |
| II. CPCV pipeline | Per-fold training and prediction | calibrated test-fold probabilities |
| III. Post-CPCV | Evaluation and interpretability | model comparison + symbolic formula |

### 3.1 Data

- **Source.** BTC-USD daily OHLCV from yfinance.
- **Range.** 2014-11-01 to 2026-05-01 (approximately 4,200 daily bars).
- **CUSUM truncation.** Raw series begins 2014-11-01 to provide the 252-day lookback the longest-warmup features require. The CUSUM event filter is truncated to start 2015-08-08, the first day with valid ETH/USD data needed for `eth_btc_ratio`. CUSUM accumulators are computed on the full raw series; only the event-firing window is restricted.
- **External sources.** Macro (yfinance, FRED), crypto-macro (CoinMetrics with yfinance fallback), on-chain (CoinMetrics Community API).
- **Anti-leakage.** Externals aligned via `merge_asof(direction='backward')`. CoinMetrics shifted by 1 day (end-of-day reporting convention).
- **Validation.** Empty downloads raise; MultiIndex flattened; duplicate dates raise; calendar gaps ≤ 3 days forward-filled; OHLCV consistency checked; NaN Close drops the row.
- **Calendar.** All rolling windows use the BTC calendar (7-day week, 365-day year).
- **Figure.** BTC-USD log price 2014 to 2026 with CPCV group boundaries and the 2015-08-08 truncation date.

### 3.2 Labelling [introduces event / label / observation thread]

> Theory in 2.3.2. This section covers locked-run configuration only.

- **3.2.1 Daily volatility.** EWMA std of log returns, `span=50` (vs AFML default 100 for equities; tightened for BTC's faster regime transitions). Two roles: target width for TBL barriers; CUSUM threshold calibration.
- **3.2.2 CUSUM event filter.** Symmetric `s_pos`, `s_neg` accumulators on the full raw series. Threshold `h = 1.0 × mean(daily_vol)` (tightened from 1.5×). After firing, events truncated to start 2015-08-08. Locked-run: 1,352 candidates → 82 dropped by truncation → 1,270 enter TBL.
- **3.2.3 Triple barrier labelling.** `pt_sl=(1.5, 1.5)` symmetric; `num_days=10`; `min_return=0.02`. Output `bins[ret, bin, t1]`. Observed mean holding period 5.0 days; vertical barrier hit on 19.7% of events. Rare-label removal closes the section: `min_pct=0.085` combined with the symmetric `pt_sl` and `min_return=0.02` eliminates class 0 → binary labels {-1, +1}. **Final: 1,168 events** at `{+1: 664, -1: 504}` = 56.85% Up / 43.15% Down.
- **Figures.** CUSUM accumulators with event markers and truncation boundary; triple-barrier visualisation for two representative events.
- **Table.** Label distribution before and after rare-label removal.

### 3.3 Sample Weights

> Theory in 2.3.2 (sample-weight chain). This section lists the four steps with locked-run parameter values.

- **Step 1. Concurrent count (Snippet 4.1).** For each bar `t`, count active labels `c_t = |{i : t0_i ≤ t ≤ t1_i}|`.
- **Step 2. Average uniqueness (Snippet 4.2).** `ū_i = mean(1/c_t)` over `[t0_i, t1_i]`.
- **Step 3. Return attribution (Snippet 4.10).** `w_i = |ret_i| · ū_i`, normalised to `sum(w) = len(w)`.
- **Step 4. Time decay (Snippet 4.11).** Linear decay, `time_decay_factor = 0.4`. Re-normalise.
- **Step 5. Outlier cap.** Clip at `weight_cap_quantile = 0.99`.
- **Integration.** sklearn models via `sample_weight`; PyTorch models via per-sample multiplier in CrossEntropyLoss.
- **Figure.** Sample weights over time, annotated with notable BTC events.

### 3.4 Feature Engineering

> Group features by function. Per-feature detail in Appendix E.

- 73 features in four groups, all eligible for MDA. AR Logistic restricts itself to the 10 lag columns by name.

| Group | Count | Source | Purpose |
|-------|-------|--------|---------|
| Technical (TA) | 25 | OHLCV | Price/volume patterns |
| Mathematical (AFML Part 4) | 9 | Returns/log-prices | Information-theoretic, structural breaks |
| External: macro | 20 | yfinance, FRED | Macro environment |
| External: crypto-macro | 1 | CoinMetrics + yfinance fallback | Cross-crypto signal |
| External: on-chain | 8 | CoinMetrics | Blockchain fundamentals |
| Lag (autoregressive) | 10 | Log returns | Pure-autoregressive baseline |
| **Total** | **73** | | |

- **3.4.1 Technical analysis features (25).** Returns and volatility (8): `log_returns`, realised vol, Garman-Klass, Yang-Zhang, ATR (EWMA span=14, log-transformed), Bollinger width, `vol_term_7_30`, `vol_term_30_90`. Momentum/trend (9): RSI(14), MACD/signal/hist, `roc_14`, Stoch %K/%D, Williams %R, CCI(14). Volume (3): OBV (log-transformed), Chaikin, MFI(14). Distribution shape (2): rolling skew/kurt (window=21). Trend ratios (3): EMA 20/50, EMA 50/200, VWMA 20/50. Standard TA periods kept (no per-indicator tuning). **Cite:** `garman_klass_1980`, `yang_zhang_2000`, `mate_confluence_2024`.
- **3.4.2 Mathematical features (9, AFML Part 4).** Information-theoretic (3): Shannon entropy, Lempel-Ziv complexity, negentropy. Random-walk (2): Hurst, variance ratio. Normality (1): Jarque-Bera. Structural breaks (3): SADF, SMT-polynomial-1, SMT-exponential. Cached due to O(n²) cost. **Cite:** `lopez_de_prado_2018` Part 4, `lo_mackinlay_1988`.
- **3.4.3 External features (29).** Macro (20): `dxy_roc_30`, `us2y`, `us10y`, yield curves, `vix`, plus 30-day and 14-day return windows on seven commodity/index series (SP500, Nasdaq, gold, silver, copper, oil, natgas). Crypto-macro (1): `eth_btc_ratio` from CoinMetrics (with yfinance fallback). On-chain (8): `active_addr_roc_14`, `tx_count_roc_14`, `hashrate_roc_30`, `mvrv`, `net_exchange_flow`, `fee_per_tx`, `exchange_supply_pct`, `issuance_ntv`. All shifted by 1 day. Anti-leakage via `merge_asof(direction='backward')`.
- **3.4.4 Lag features (10).** `AR_LAGS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]`. Lag features compete with engineered features in MDA. AR Logistic selects the 10 lag columns by name from the pre-MDA matrix.
- **3.4.5 Phase I output.** Log transforms applied to `atr` and `obv` only. NaN values from warmup and external gaps are not dropped in Phase I; per-fold resolution happens inside CPCV (3.6.1). Alignment `align_for_cv(features, bins, weights) → (X, y, w, t1)` via index intersection, with hard assertions (non-empty intersection, monotone index, identical lengths, no all-NaN columns, `t1` fully populated, weights > 0). **Output.** `X` shape `(1168, 73)`, `y` shape `(1168,)`, `w` shape `(1168,)`, `t1` shape `(1168,)`. **Table.** Alignment summary (daily bars → CUSUM events → labelled events → aligned size).

### 3.5 Cross-Validation Framework

> Theory in 2.3.4. This section covers the locked-run configuration, the path-matrix construction, and the leakage audit.

- **3.5.1 Why standard CV fails.** Single short paragraph: TBL labels span `[t0, t1]` multi-day intervals; overlapping spans leak training-label information into the test period; metrics inflate. Refer back to 2.3.4.
- **3.5.2 CPCV configuration.** `N_GROUPS=8`, `K_TEST_GROUPS=2`, `EMBARGO_PCT=0.01`. Groups 0 to 6 of size `⌊T/N⌋`; group 7 absorbs remainder. `C(8, 2) = 28` splits; `φ(8, 2) = C(7, 1) = 7` backtest paths. Each group appears in 7 test sets at 146 events per group. Justification for N=8, k=2: yields 146 events per group while keeping the training fold at approximately 858 events (73.4% of 1,168) after purging and embargo; 28 splits and 7 paths give denser combinatorial diversity for PBO than smaller N values. Path matrix construction per AFML Section 12.4.1: for group `g`, path `p` uses the `p`-th split that includes `g` in its test set; each path uses 4 of the 28 splits and the Sharpe distribution across the 7 paths feeds DSR and PBO. Locked-run audit: average train 858 events, test 292, purged 1.9, embargoed 16.5; group date ranges span 2015-08-08 to 2026-05-02. **Figure.** CPCV split visualisation for representative splits.
- **3.5.3 Purging and embargo.** Purging (AFML Snippet 7.1): three sufficient overlap conditions for training observation `i` against test window `[t_test_start, t_test_end]` (observation falls in window; label resolves in window; label spans the entire window). Embargo (AFML 7.4.2): `int(EMBARGO_PCT × T) = 11` observations removed immediately after each test group; one-sided because training labels resolving before the test starts contain no future test information. Leakage audit closes the section: zero training observations whose `t1` falls within the assigned test group's date range, across all 28 splits. **Table.** Leakage audit → Appendix B.

### 3.6 Per-Fold Preprocessing

> Opening claim: three transformations happen inside the CPCV loop, fitted on training data only, to prevent test-fold statistics from leaking into training.

- **3.6.1 Fractional differentiation.** Theory in 2.3.3. ADF test across all 73 features at α=0.05 identifies ATR as the only non-stationary feature; FFD applied to ATR only. Weights `ω_0=1, ω_k = -ω_{k-1}(d-k+1)/k`, truncated at `|ω_k| < 1e-4` or `k ≥ 200`. `d*` sweep `d ∈ [0, 1]` step 0.05; minimum `d` where ADF p < 0.05. FFD applied to the full ATR series using training-derived `d*` (strictly backward-looking convolution). Per-fold NaN policy: non-FFD columns get `ffill().bfill()` independently within train and test; FFD columns drop NaN rows. **Cite:** `lopez_de_prado_2018` Ch. 5; `slepaczuk_bieganowski_2024`.
- **3.6.2 Feature scaling.** `RobustScaler` (median + IQR), fitted on training fold only, applied to all 73 features. Chosen over StandardScaler because BTC features show fat tails.
- **3.6.3 Multi-model MDA feature selection.** Novel relative to single-model MDA. MDA computed independently with Random Forest (500 trees, balanced) and Logistic Regression (balanced). Per-model z-scoring before averaging because RF MDA drops F1 by 0.005 to 0.05 per feature while LR's drops are an order of magnitude smaller; naive averaging would let RF magnitudes dominate. Final score: `mean(zscore(MDA_RF), zscore(MDA_LR))`. Inner 3-fold purged CV with the same `t1`-overlap conditions as outer CPCV. Selection rule: keep features with averaged z-MDA > 0; cap at `MDA_TOP_K_FRAC = 0.20` (approximately 14 to 15 of 73); floor of 5. The 0.20 cap keeps the absolute feature count near the parameter-to-sample threshold for KAN width-6 and LSTM hidden-16. AR Logistic bypasses MDA and selects the 10 lag columns by name from the pre-MDA matrix.

### 3.7 Models

> Six models, four families. Detailed hyperparameter ranges and per-model justifications in Appendix C (model summary table). Per-model subsections cover architecture and tuned ranges only.

- **Opener (2 paragraphs).** Six classifiers, four families (econometric / linear ML / ensemble / neural). Three shared training-pipeline elements: AFML sample weights, class-balanced via `class_weight='balanced'` or `scale_pos_weight`, and five seeds with split-level metrics averaged. Five engineered-feature models enter Optuna at 40 trials per split. Summary table in Appendix C lists architecture and tuned ranges side by side.

- **3.7.1 AR Logistic (econometric baseline).** Tests pure price momentum. sklearn `LogisticRegression` with `C=1.0`, L2, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`. Consumes the 10 lag columns by name from the pre-MDA matrix. 5 seeds, no per-split tuning.
- **3.7.2 Logistic Regression (linear ML baseline).** Same MDA-selected features as the ensembles. Tuned: `C` log-uniform [1e-4, 1e2], `penalty ∈ {L1, L2}`. Solver auto-selected. 5 seeds, 40 trials per split.
- **3.7.3 Random Forest.** 500 trees, `class_weight='balanced_subsample'`, `n_jobs=-1`. Tuned: `n_estimators ∈ {100, 150, 200, 250}`, `max_depth ∈ [2, 6]`, `min_samples_leaf ∈ [15, 40]`, `max_features ∈ {sqrt, log2}`. 5 seeds, 40 trials per split. **Cite:** `breiman_2001`.
- **3.7.4 XGBoost.** 500 trees with early stop at 20 rounds, `binary:logistic`, `scale_pos_weight` from class balance. Tuned: `max_depth ∈ [1, 3]`, `lr` log-uniform [0.01, 0.3], `min_child_weight ∈ [5, 30]`, `subsample` and `colsample_bytree ∈ [0.6, 1.0]`, `gamma` log-uniform [1e-8, 1], `reg_alpha` and `reg_lambda` log-uniform [1e-8, 10]. Calibration set dual role (eval-set for early stopping + Platt-fit data) is a mild dependency bounded by the 500-tree cap and 20-round patience. 5 seeds, 40 trials per split. **Cite:** `chen_guestrin_2016`.
- **3.7.5 LSTM.** Single-layer `nn.LSTM` → final hidden state → LayerNorm → dropout → linear → 2 logits. `num_layers=1` hardcoded. Sliding window `LSTM_WINDOW=14` (close to TBL `num_days=10`); reduces sequence count from `N` to `N-13`. Tanh input normalisation. Training stack: AdamW (`weight_decay=1e-4`), CrossEntropyLoss with class + AFML weights, label smoothing 0.1, grad clip max norm 1.0, cosine annealing warm restarts (`T_0=25`, `T_mult=2`), batch 64, max 100 epochs, early stopping patience 15. Tuned: `hidden ∈ {16, 32}`, `dropout ∈ [0.1, 0.5]`, `lr` log-uniform [1e-4, 5e-2]. Tuning at 50 epochs / patience 7; production refit at 100 / 15 (compute-vs-fidelity trade-off). 5 seeds, 40 trials per split. **Cite:** `hochreiter_schmidhuber_1997`, `loshchilov_adamw_2019`.
- **3.7.6 KAN.** `efficient_kan.KAN([n, width_1, 2], grid_size=grid, spline_order=3, grid_range=[-1, 1])`. Single hidden layer (`width_2=0` hardcoded). Training stack: AdamW (lr and weight_decay tuned), CrossEntropyLoss with class + AFML weights, label smoothing 0.1, grad clip 1.0, cosine annealing (`T_0=30`, `T_mult=2`), max 200 epochs, patience 20. Tanh input normalisation matching the spline grid range. No SWA (conflicts with early stopping); no entropy regularisation (redundant with label smoothing). Tuned: `width_1 ∈ {2, 3, 4, 5, 6}`, `grid ∈ {3, 5}`, `lr` log-uniform [5e-4, 5e-2], `weight_decay` log-uniform [1e-5, 5e-3]. Single-grid training (no coarse-to-fine refinement; fold size insufficient). Dual-library design: `efficient_kan` for CPCV; PyKAN re-trained from scratch for symbolic extraction (Section 3.11), since only PyKAN exposes `prune`, `suggest_symbolic`, `fix_symbolic`, `symbolic_formula`. 5 seeds, 40 trials per split. **Cite:** `liu_kan_2024`.

### 3.8 Hyperparameter Tuning

- Nested per-split Optuna inside each CPCV training fold (AFML Ch. 7 compliant).
- Inner 3-fold purged CV with the same `t1`-overlap conditions as outer CPCV and 10-observation embargo (matches TBL `num_days`).
- TPE sampler, `seed=42`. MedianPruner with `n_startup_trials=5` (classical) or 3 (neural), `n_warmup_steps=1`.
- Trial budget: `n_trials=40` per tuned model per split, uniform across LR, RF, XGBoost, LSTM, KAN. AR Logistic excluded.
- Per-split tuned params overwrite module-level constants before training; pristine defaults restored on subsequent runs.
- DSR validity: `n_trials` in DSR counts compared models (6), not Optuna trials per split.
- **Cite:** `akiba_optuna_2019`, `lopez_de_prado_2018` Ch. 7.

### 3.9 Calibration

> Opening claim: bet sizing depends on calibrated probabilities.

- **70/15/15 chronological split of the training fold.** 70% model-train, 15% validation (early stopping for neural), 15% calibration (Platt or vector scaling). Approximately 600 / 129 / 129 events at the locked configuration. None overlaps the outer test fold.
- **Three-partition justification (1 short paragraph).** Earlier 80/10/10 was insufficient for both calibrator fit and neural early stopping; 80/20 with shared cal/val coupled stopping decisions to calibration data. Three-way split removes the coupling at the cost of ~75 fewer model-training events per fold.
- **Two methods, auto-selected by model type.**

| Method | Models | Input | Mechanism |
|--------|--------|-------|-----------|
| Platt (Platt 1999) | AR Logistic, LR, RF, XGBoost | 1D log-odds | `LogisticRegression(C=1e10)` |
| Vector (Guo 2017, §4.2) | LSTM, KAN | 2D logits | Fit `T ∈ [0.05, 20]` and `b ∈ ℝ²` `∈ [-5, 5]` minimising NLL of `softmax((logits+b)/T)` via L-BFGS-B |

- **Vector-over-temperature (1 short paragraph).** Pre-final audit showed LSTM and KAN under-predicting P(y=1) by 10 to 23 pp against a 0.55 base rate. Pure temperature scaling preserves argmax and cannot shift class lean. Vector scaling adds a per-class bias `b` (Guo 2017's recommended extension). Substitution made before final evaluation: methodological correction, not test-set-informed selection.
- **Unweighted calibration locked.** Audit showed weighted Platt/vector pushed every model's calibration miss further from the base rate and compressed Sharpes; `sample_weight=None` in the calibrator call site.
- **LSTM logit slicing.** Logits sliced to valid-index set (windows with full 14 lookback timesteps) before calibration; earlier ordering produced `T` consistently at the lower bound; caught in audit.
- **Cite:** `platt_1999`, `guo_temperature_2017`.

### 3.10 Evaluation Framework

> Module constants: `TRANSACTION_COST=0.001`, `MIN_BET_SIZE=0.05`, `MAX_BET_SIZE=0.75`, `BET_DISCRETIZATION=[0, 0.25, 0.50, 0.75]`, `ANNUALIZATION_FACTOR=365`, `RISK_FREE_RATE=0.0`.
>
> Theory for DSR and PBO in 2.3.5; this section covers configuration and methodological choices.

- **3.10.1 Trading simulation.**
  - Bet sizing (AFML 10.3, compressed to 2 sentences + formula): `sign = +1 if P(up) > P(down) else -1`; confidence `p = max(P(0), P(1))`; z-score `z = (p - 0.5) / sqrt(p(1-p) + 1e-10)`; raw bet `2·Φ(z) - 1`; clip to [-0.75, 0.75]; threshold `|bet| < 0.05 → 0`; discretise to {0, 0.25, 0.5, 0.75}; apply sign. Implicit abstention per Conservative Predictions (Section 2.2.4). **Cite:** `lopez_de_prado_2018` Ch. 10, `nabar_shroff_2023`.
  - Strategy returns: `gross = bet · ret_label`; `turnover = |Δbet|`; `tx_cost = 0.001 · turnover`; `net = gross - tx_cost`. 0.1% per trade is conservative for major BTC exchanges. Annualise at 365 (BTC trades 24/7); risk-free rate 0.
  - Path stitching (AFML 12.4.1): 7 backtest paths assembled from the 28 splits via `path_map[path_id]`. For each `(group_id, split_id)`: retrieve predictions AND filter to events whose positional index falls within `group_bounds[group_id]`. The group filter prevents duplication when co-tested groups appear in multiple splits. Multi-seed: average calibrated probabilities across 5 seeds before bet sizing (variance reduction ≈ 1/√5). Concatenate, sort, assert no duplicate timestamps.[^stitching]

[^stitching]: Earlier implementation omitted the group filter, double- or quintuple-counting events. Caught by direct timestamp inspection.

- **3.10.2 Path performance metrics.**
  - Table of metrics: annualised Sharpe `(mean/std)·√365`, cumulative return `Π(1+r_t)-1`, annualised return `(1+cum_ret)^(1/years)-1`, max drawdown, time under water, win rate, profit factor, n_trades, n_returns, skew, kurt.
  - Sharpe annualisation disclosure (1 short paragraph): strategy returns sampled at CUSUM events (~75 per year), not daily bars (365). `√365` yields the daily-equivalent scale; DSR de-annualises by the same `√365`, so DSR verdicts and rankings are convention-invariant.

- **3.10.3 Statistical correction layer: DSR, PBO, and DeLong.**
  - DSR (AFML Ch. 14). Refer to 2.3.5 for the formula. `n = n_trials = 6` (compared models); `n_obs = avg_n_returns` per path. Kurtosis convention: Mertens (2002) raw kurtosis; conversion applied from `scipy.stats.kurtosis` excess. NaN-safety clamp on variance term. DSR > 0.95 → significant after correction.
  - PBO via CSCV (AFML Ch. 11). Refer to 2.3.5. Path-Sharpe matrix shape (6 models × 7 paths). All `C(7, 3) = 35` IS/OOS partitions (IS=3 paths, OOS=4 paths). PBO = fraction where IS-best underperforms OOS median. N=8 yields 35 partitions vs 10 under N=6, raising resolution from 0.10 to 0.029 per partition.
  - DeLong pairwise AUC (DeLong 1988). Per `(model, split)`: average predicted probabilities across 5 seeds. Pool across all 28 splits per model (valid: CPCV test sets are non-overlapping). Z-statistic from standard formula; two-sided p-value from normal. 15 pairwise comparisons. No Bonferroni applied; acknowledged in 5.4. **Cite:** `delong_1988`.

- **3.10.4 Stability diagnostics, model comparison, and ranking.**
  - Feature stability. Per-feature selection frequency across 28 folds. Frequency > 0.80 stable; flat profile → diffuse signal.
  - FFD stability. Mean and std of `d*` for ATR across 28 folds. `std(d*) < 0.1` consistent; `> 0.1` time-varying.
  - Ranking. Primary criterion median path Sharpe (descending); tiebreaker std Sharpe (ascending, prefer consistency). Columns: rank, model, median Sharpe, std Sharpe, DSR, mean F1, mean acc, mean log_loss, mean AUC, median max_dd, median cum_ret, median win_rate, median profit_factor.

### 3.11 Symbolic Extraction

> Frame as exploratory analysis (per advisor guidance), not a core contribution. Main text approximately 1.5 pages. Detailed training logs, edge-by-edge R², unsimplified formulas → Appendix F.

- **3.11.1 Purpose, fold selection, and data preparation.**
  - Operates after all 28 CPCV splits have been evaluated. Reconstructs preprocessing state (stored `d*`, RobustScaler, selected feature list) and re-trains a PyKAN model on the same training fold, because only PyKAN exposes `prune`, `suggest_symbolic`, `fix_symbolic`, `symbolic_formula`.
  - Architecture inherits 3.7.6: single hidden layer, width cap 6, hidden 5 / grid 5 / spline order 3 defaults, safety floor 5 training samples per parameter.
  - Fold selection: last split (split 27); test set covers most recent observations, closest to deployment. Formula fold-specific by design.
  - Data preparation: re-generate CPCV splits, apply stored `d*` to full ATR series, extract training fold, apply stored RobustScaler, reduce to top-3 features by 28-fold MDA selection frequency (`n_top_features=3`, strategy `"stability"`), 80/20 chronological model-train/validation split. Tanh input normalisation. Locked-run: 540 training events + 135 validation events.

- **3.11.2 Four-step PyKAN algorithm.** Following the four-step algorithm of `cho_lee_kim_2025` (see also `noorizadegan_2026`): train → prune → symbolify → affine fine-tune. Architecture matches CPCV-benchmarked KAN.
  - Step 1. Three-phase training. Phase 1: Adam (`lr=1e-3`, `wd=1e-3`), 600 steps, Gaussian noise (std 0.05) clamped to [-1, 1], early stopping. Phase 2a: LBFGS (`lr=0.01`), 20 steps, no regularisation. Phase 2b: LBFGS, 20 steps, L1 + entropy (`λ=0.002`). Grid extension disabled. Dynamic majority-baseline gate after Phase 1 and pre-prune: `val_acc` checked against `max(0.53, val_majority_baseline + 0.01)`; logs warning if missed.
  - Step 2. Pruning. Forward pass populates activations; `model.attribute()` scores edges; `prune(threshold=0.01)` with multi-API fallback. Verify forward pass; revert if broken. Typical compression `[n, 6, 2] → [n, 3, 2]`.
  - Step 3. Symbolification. Library of 14 functions: `x, x², x³, x⁴, exp, log, sqrt, tanh, sin, cos, abs, sgn, arctan, 0`. Per edge: `suggest_symbolic(l, i, j, topk=5)`; winner = highest `R² ≥ SYMBOLIC_R2_THRESHOLD=0.30`. Below threshold: keep spline (partial symbolification). Zero-function special case: always wins on `total_loss` due to zero complexity penalty; brute-force fallback iterates the 13 non-constant candidates, captures R² from PyKAN stdout via regex, keeps best non-constant ≥ 0.30.
  - Step 4. Affine fine-tuning. Each symbolic edge has form `a·f(bx+c) + d`; 30 LBFGS steps at `lr=4e-4` update the four affine scalars without changing the function family. NaN check reverts to pre-fine-tune state.

- **3.11.3 Formula extraction and output.**
  - Two outputs: `logit_bearish` (class 0), `logit_bullish` (class 1). Decision function `= logit_bullish - logit_bearish`. `P(Up) = σ(decision)`.
  - SymPy simplification with 30-second timeout (prevents hang). `nsimplify(tolerance=1e-3)` for rational coefficients.
  - Surviving features = `decision.free_symbols`, generally a subset of the 3 input features (pruning may remove all edges for a feature).
  - Full output (unsimplified logits, SymPy objects for partial-derivative sensitivity, pre/post-symbolic accuracies, symbolification rate, pruned architecture) → Appendix F.

- **3.11.4 Known fragilities and acknowledged limitations.**
  - Two consequential PyKAN behaviours. `abs` symbolises cleanly but yields a SymPy expression non-differentiable at zero; for features whose mean is near zero in tanh-normalised space, partial-derivative sensitivity returns NaN (L6 in 5.4). Brute-force fallback parses PyKAN stdout via regex, fragile to library upgrades.
  - Other fragilities (Appendix F). Missing `sigmoid` in PyKAN's internal library, division by zero at `1/x`, 1-based variable naming, `suggest_symbolic` format variation across versions, `sympy.simplify` indefinite hang.
  - Acknowledged limitations (also 5.4). Fold-specific (different folds may produce different formulas). Small sample (540 events) limits identifiable complexity; motivates the R²=0.30 threshold. Post-symbolic accuracy may fall below pre-symbolic when the symbolic library fits the trained spline less faithfully than the spline fitted the data.
  - **Cite:** `cho_lee_kim_2025`, `liu_kan_2024`, `noorizadegan_2026`.

### Leakage prevention summary table (chapter closer)

| Risk | Mitigation | Where |
|------|-----------|-------|
| FFD `d*` on full data | per-fold training-only estimation | `preprocessing.py` |
| Scaler on full data | RobustScaler on training fold only | `preprocessing.py` |
| Feature selection on full data | multi-model MDA on training fold only | `preprocessing.py` |
| HP tuning on full data | nested Optuna inside training fold (3-fold purged) | `tuning.py` |
| Overlapping TBL labels | three-condition purging | `cv.py` |
| Serial correlation across CV boundary | embargo after test boundary | `cv.py` |
| Calibration on test data | 15% calibration partition of training fold | `calibration.py` |
| On-chain look-ahead | CoinMetrics shifted by 1 day | `external_features.py` |
| NaN imputation across train-test | independent `ffill().bfill()` within each partition; FFD columns drop | `preprocessing.py` |

---

## 4. Results

> Lead with the main result. Numbers from the locked end-to-end run (v6, May 2026). Drop cross-version commentary; report locked-run figures only.

### 4.1 Headline

- No model achieves DSR ≥ 0.95. Top trained DSR is KAN at 0.2470. Buy-and-hold posts median Sharpe 2.2722 (above every trained model) at the cost of -99.98% median max drawdown.
- PBO = 0.657, in the **adversarial regime**: 23 of 35 IS/OOS partitions see the IS-winner underperform the OOS median.
- Leave-one-out PBO: Random Forest is **PBO-neutral** (Δ +0.000); Logistic Regression carries the largest destabilising contribution (Δ -0.286 to PBO 0.371).
- Bootstrap 95% CIs on median Sharpe: 6 of 7 cross zero; only buy-and-hold's (1.8903, 2.4685) stays strictly positive. KAN's CI is (-0.4677, 1.1607).
- 2 of 15 DeLong pairwise AUC tests reach α=0.05: AR Logistic vs XGBoost (p=0.0427) and Random Forest vs XGBoost (p=0.0318). Neither survives Bonferroni (α/15 ≈ 0.0033).

### 4.2 Model Comparison

- **Table.** Model comparison ranked by median Sharpe over 7 CPCV backtest paths, buy-and-hold included. Columns: rank, model, median Sharpe, bootstrap CI, std Sharpe, DSR, median Sortino, median Calmar, mean F1, mean acc, mean AUC, median max DD, median cum ret, median win rate, median profit factor.
- Locked-run figures: Buy-and-Hold (2.2722, CI strictly positive), KAN (0.5588), AR Logistic (0.3806), LSTM (0.1541), XGBoost (0.0493), Random Forest (0.0287), Logistic Regression (-0.2173).
- Sortino reorders the trained models: AR Logistic 0.3894, KAN 0.3211, LSTM 0.1189, XGBoost 0.0259, RF 0.0257, LR -0.1872. Calmar compresses trained models into -0.04 to +0.09 band; KAN leads at 0.0820.
- AUC compression: all mean AUCs in 0.5008 to 0.5204 band. Models nearly indistinguishable at classification level; bet sizing and dual-weighted loss translate tiny edges into Sharpe spread.
- Std Sharpe outlier: Logistic Regression (1.5969) carries the widest CI and contributes most to LOO PBO destabilisation.
- Q1 and Q2 reading: AR Logistic posts a positive median Sharpe with the highest mean AUC; pure-momentum lags deliver competitive financial performance.
- Q3 reading: KAN is rank-1 trained on both median Sharpe and DSR; classification metrics mid-pack.

### 4.3 Classification Metrics

- **Per-split F1 distribution.** Mean F1 ranges 0.4070 (KAN) to 0.4666 (AR Logistic).
- **Pooled AUC.** 0.4942 (XGBoost) to 0.5149 (AR Logistic). XGBoost's pooled AUC below chance for the second consecutive run.
- **DeLong pairwise AUC table.** 15 pairs; 2 significant at α=0.05 (AR Logistic vs XGBoost p=0.0427; Random Forest vs XGBoost p=0.0318), both with XGBoost as lower-AUC half. XGBoost vs KAN approaches significance at p=0.0916.
- **Effect-size disclosure.** AUC range across all six models is 0.021, comparable to seed-to-seed and split-to-split variation within any single model.
- **Multiple-testing correction.** Bonferroni threshold α/15 ≈ 0.0033; neither significant pair clears. Benjamini-Hochberg with FDR=0.05 also rejects. The per-pair significance does not propagate to a family-wise claim.
- **Confusion matrices** aggregated TP/FP/TN/FN per model → Appendix D.

### 4.4 Financial Performance

- **Path-level Sharpe distribution.** Per-model row × 7 paths + median + std + bootstrap CI.
- **DSR computation detail** for KAN (top trained): observed median Sharpe 0.5588, DSR 0.2470 against `n_trials=6`. Full component breakdown → Appendix D.
- **PBO result.** PBO = 0.6571 = 23 of 35 IS/OOS partitions. Joint reading with DSR < 0.95: model selection in this six-model universe is adversarial; the IS-best is the OOS-loser in two thirds of the partitions.
- **Leave-one-out PBO table** (model excluded → 5-model PBO → Δ): Random Forest is PBO-neutral (Δ +0.000); Logistic Regression carries the largest destabilising contribution (Δ -0.286 → 0.371).
- **Equity curves figure.** All 6 models + buy-and-hold overlay; log-scale y-axis. XGBoost, RF, Logistic Regression end below 1.0.
- **Additional metrics table.** Cumulative return, annualised return, max DD, time under water, win rate, profit factor, mean |bet|, % traded. KAN posts second-best max-drawdown (-0.2466) alongside rank-1 trained median Sharpe.

### 4.5 Feature Selection Stability and FFD Stability

- **Feature stability profile.** Of 73 features: 0 stable (> 80%), 2 moderate (50 to 80%): `eth_btc_ratio` 57.1%, `log_returns_lag6` 53.6%. 71 low. 0 never selected.
- **Headline.** No compact feature subset dominates across time periods; the MDA-selected set turns over substantially across the 28 folds.
- **Group breakdown of moderate features.** 1 crypto-macro, 1 lag. 0 TA, 0 macro, 0 on-chain, 0 mathematical.
- **Q2 on-chain finding.** No on-chain feature reaches 50%. The free-lunch hypothesis does not survive multi-model MDA + CPCV.
- **FFD stability.** ATR is the only FFD-treated column. 28 folds × 5 seeds = 140 d* estimates: mean ≈ 0.198, std ≈ 0.085, range [0.050, 0.400]. `std(d*) < 0.1` consistent. No fold required d*=1.0.
- **Figure.** Horizontal bar chart of 73 features by selection frequency, coloured by group.
- **Methodological reading.** Feature-selection turnover high; FFD-d* turnover low. The noise-vs-signal ratio for individual features changes across regimes; ATR's persistence is stable. Validates FFD-only-on-ATR (3.6.1) and multi-model MDA over single-model SFI (3.6.3).

### 4.6 Symbolic Extraction Results (Q4 answer)

- **Fold used.** Split 27 (last fold, deployment-proxy). KAN F1 on this fold: 0.3232.
- **Surviving 3 features.** `eth_btc_ratio` (57%), `log_returns_lag6` (54%), `natgas_ret_30` (39%).
- **Training sample.** 540 train + 135 val. Architecture `[3, 3, 2]` (15 active edges, ~90 spline parameters, ratio 6.0× above the 5× floor).
- **Training diagnostics.** Phase 1 val_acc 0.4815 (gate fails, threshold 0.5693). Phase 2a val_acc 0.4296. Phase 2b val_acc 0.4667 (pre-prune gate fails, threshold 0.5359). Grid extension skipped. Full phase log → Appendix F.
- **Pruning.** No edges pruned (15 of 15 active). Pruned architecture `[[3,0],[3,0],[2,0]]`. Post-prune val_acc 0.4667.
- **Symbolification rate.** 100% (15 of 15 edges clear R² ≥ 0.30). R² distribution: min 0.734, median 0.990, max 1.000. Sin primitive dominates the top.
- **Pre vs post-symbolic accuracy.** Pre 0.4667 (below 50%). Post 0.5185 (Δ = +5.18 pp, but still below 56.85% majority baseline). Affine fine-tuning is doing genuine work; underlying model has not extracted directional signal on this fold.
- **The decision function.** Numbered equation (closed form). Six primitives (`sin`, `cos`, `tanh`, `x²`, `x³`, `x⁴`) across four outer terms with distinct outer wrappers (squared, sin, cos, tanh).
- **Numerical sensitivity table.** Per-σ effect on logit and linearised P(Up) shift for each surviving feature. `log_returns_lag6` largest per-σ shift (-0.0983); `eth_btc_ratio` second (-0.0598); `natgas_ret_30` smallest (+0.0179). All three features have finite, well-defined gradients at the dataset median (no `abs`-primitive non-differentiability).
- **Q2 on-chain.** 0 on-chain features survive extraction. Hardens 4.5 finding.
- **Q2 lag.** `log_returns_lag6` reaches the surviving set, validating the lag-features-in-MDA-pool design choice.
- **Section closer.** "100% symbolification rate on three features; post-symbolic accuracy 51.85% but still below the 56.85% majority baseline. RQ4: closed-form interpretability is achievable, but interpretability does not imply predictive power."

---

## 5. Discussion

> Interpret results. Link findings back to mechanisms in Section 3.

### 5.1 Interpreting the Results (Q1, Q3)

- **Joint DSR / PBO reading.** Top trained DSR 0.2470 (KAN), far below 0.95. PBO 0.657 places model selection in the adversarial regime: the IS-best is the OOS-loser in two of three partitions. Bootstrap evidence: 6 of 7 CIs cross zero; only buy-and-hold strictly positive. No trained model has a statistically significant predictive edge.
- **LOO PBO asymmetry.** Random Forest PBO-neutral; its median Sharpe is second-lowest and its path-Sharpe distribution rarely produces three-path IS wins. Logistic Regression the largest destabiliser (Δ -0.286): widest bootstrap CI; occasional IS-best wins not confirmed OOS.
- **DeLong significance.** 2 of 15 pairs at α=0.05 (both XGBoost as lower-AUC half). XGBoost's pooled AUC 0.4942 is the only trained-model AUC below chance. Neither pair survives Bonferroni; per-pair information is preserved but the family-wise claim is not.
- **Consistency with EMH.** DSR < 0.95 aligns with semi-strong-form EMH for BTC daily direction under leakage-free evaluation. Buy-and-hold's dominance over every trained model strengthens the reading: the asset's overall trajectory carries information that direction-prediction strategies discard. Consistent with `chassot_audrino_2026` (properly fitted baselines are hard to beat). **Cite:** `fama_1970`, `chassot_audrino_2026`.
- **Conservative Predictions framing.** Bet-sizing threshold operationalises selective abstention (`nabar_shroff_2023`): `p ≈ 0.50 → bet ≈ 0`. KAN's mean P(Up) 0.5421 against base rate 0.5685 (within tolerance). Only XGBoost flagged by calibration audit (Δ = -0.0371). The bet-sizing curve operates on calibrated probabilities; the negative DSR is a statement about signal strength, not probability quality.

### 5.2 KAN Performance in Context (Q3, Q4)

- **KAN ranks 1st among trained models** on median Sharpe (0.5588) and DSR (0.2470). Second-best median max drawdown (-0.2466). Highest median win rate (0.5315, statistical tie with AR Logistic at 0.5318).
- **No bootstrap-strict-positive trained model.** All six trained-model CIs cross zero; only buy-and-hold's stays strictly positive.
- **Where KAN's rank-1 trained Sharpe comes from.** Classification metrics mid-pack (mean acc 0.5318, mean AUC 0.5103). The rank-1 Sharpe comes from bet-sizing translation: well-calibrated mean P(Up) (0.5421 vs 0.5685 base rate), top-three win rate, conservative max-drawdown.
- **Contrast with `oad_kasper_2025`.** KASPER reports Sharpe 12.02 on individual stocks with regime detection. This thesis reports KAN median Sharpe 0.56 on BTC direction without a regime layer and under full AFML statistical correction. The gap reflects asset class, target type, and evaluation rigour rather than KAN architectural capacity.
- **Contrast with `cho_lee_kim_2025`.** VIX KAN extracts formulas validated by domain knowledge (mean-reversion, leverage effect). This thesis extracts a three-feature mixed-primitive structure on BTC daily direction where domain knowledge does not directly validate any specific functional form. The pipeline transfers across asset classes; interpretive yield depends on target structure.

### 5.3 Symbolic Extraction as Contribution (Q4)

- **Interpretability is separable from predictive accuracy.** 100% symbolification rate, clean primitives, finite gradients, three economically interpretable features. The closed-form formula provides per-feature symbolic derivatives, term-structure decomposition, and an audit trail from input to probability that a black-box cannot.
- **The +5.18 pp gain in magnitude vs level.** Largest absolute gain across configurations evaluated but operates from a sub-baseline pre-symbolic level (46.67%). The affine fine-tuning step is doing genuine work; the underlying model on this fold has not found signal. The symbolic representation faithfully captures a model that has not.
- **Methodological choice: `n_top_features=3`.** Restricts input to the three most stable features by 28-fold MDA frequency. Trades signal access against formula readability. Produces clean formula (six primitives, no `|·|` non-differentiability) but narrows the input space PyKAN can find signal on.
- **Phase-1 dynamic gate failure.** Val_acc 0.4815 < 0.5693 dynamic threshold. Pre-prune gate also fails. Symbolic extraction can still deliver a usable closed-form formula on a model that did not pass its own gates (structural transparency does not depend on classification accuracy), but the gate failures are honest signals.

### 5.4 Limitations [T-R15]

- **L1 Single asset.** BTC only; cross-asset generalisation unverified.
- **L2 Single timeframe.** Daily bars; intraday data would multiply CUSUM events ~24× and unlock microstructure features.
- **L3 Fold-specific symbolic extraction.** Single deployment fold; per-fold symbolic extraction over all 28 folds would distinguish structural signal from one-off noise.
- **L4 No regime-conditional layer.** KASPER-style regime detection deferred.
- **L5 DeLong family-wise correction.** No Bonferroni in the locked notebook; 2 of 15 significant pairs do not survive correction (acknowledged in 4.3).
- **L6 abs-primitive non-differentiability.** Documented in 3.11.4; resolved in the locked run but the fragility remains in the pipeline.
- **L7 Calibration cross-section.** Five of six models within calibration tolerance; XGBoost flagged. Per-fold calibration diagnostic is the next-step audit.

---

## 6. Conclusion and Future Work

> Do not restate all findings. End with resonance. Target ≤ 10% of textual part.

### Conclusion (no numbered subsection; flow paragraphs)

- **Paragraph 1: the contribution.** Full AFML applied to BTC daily direction, six-model benchmark, first KAN symbolic-formula extraction in this regime, 73-feature universe with multi-model MDA over four families, 5-seed uniform configuration (840 prediction entries).
- **Paragraph 2: the headline.** Under leakage-free evaluation with multiple-testing correction, the literature's 85% to 95% accuracy claims do not survive. Top trained DSR 0.2470 (KAN). PBO 0.657 in the adversarial regime. LOO PBO reveals Random Forest as PBO-neutral and Logistic Regression as the largest destabiliser. 6 of 7 bootstrap CIs cross zero; only buy-and-hold (median Sharpe 2.2722) stays strictly positive. 2 of 15 DeLong pairs significant at α=0.05 but neither survives Bonferroni. Buy-and-hold dominates every trained strategy at the cost of -99.98% max drawdown. The honest answer to Q1: under AFML evaluation, no architecture demonstrates statistically significant predictive edge; KAN is rank-1 trained on Sharpe and DSR but no trained-model CI excludes zero.
- **Paragraph 3: what survives.** The symbolic-extraction pipeline produces a closed-form decision function on three features (`eth_btc_ratio`, `log_returns_lag6`, `natgas_ret_30`) with 100% symbolification rate. Post-symbolic accuracy 51.85% (+5.18 pp over pre-symbolic 46.67%) but still below 56.85% majority baseline. Six primitives (`sin`, `cos`, `tanh`, `x²`, `x³`, `x⁴`) across four outer terms; finite gradients everywhere. Lag-feature finding reproduces across locked configurations. No on-chain feature survives extraction: the Q2 free-lunch hypothesis fails. Interpretability is a separable contribution from predictive accuracy.
- **Closing sentence with resonance.** Template: "X is not a constraint, but a catalyst, for Y." Example: "Methodological honesty is not a constraint on financial machine learning; it is the prerequisite for treating interpretability as a contribution in its own right."

### Future Work (chapter closer; bullet list)

- **MultKAN (KAN 2.0).** Multiplication nodes for multiplicative-interaction discovery. Same symbolic pipeline.
- **Higher-frequency data.** Hourly bars multiply CUSUM events ~24× and unlock microstructure features.
- **Per-fold symbolic extraction.** Run extraction over all 28 folds to distinguish structural signal from one-off noise.
- **Regime-conditional analysis.** Decompose Sharpe by regime (~5 distinct regimes 2014 to 2026).
- **Meta-labelling layer.** Singh and Joubert (2019) provide empirical evidence for meta-labelling efficacy across asset classes. **Cite:** `singh_joubert_2019`.
- **Alternative assets.** ETH, gold for symbolic-formula comparison.
- **Walk-forward vs CPCV.** Compare conclusions under both protocols.

---

## Appendices

- **A. Pipeline architecture diagram** (full flowchart from raw OHLCV through symbolic formula).
- **B. CPCV split details** (group boundaries, train/test timelines for all 28 splits, leakage audit table).
- **C. Hyperparameter search spaces** (model summary table; per-model justifications for tree depth, leaf size, LSTM single-layer, KAN width cap).
- **D. Per-split classification reports** (28 splits × 6 models; DSR component breakdown for top model).
- **E. Full feature list** (complete table of 73 features with descriptions, sources, parameters).
- **F. Symbolic extraction detailed output** (PyKAN training logs, edge-by-edge R², pruned architecture diagrams, unsimplified `logit_bearish` and `logit_bullish` expressions, additional PyKAN fragilities).

---

## Reference list to populate (`mfw_references.bib`)

> Organised to match the four sections of `mfw_references.bib`. All keys here match the .bib exactly.

**I. AFML methodology and core framework**
- `lopez_de_prado_2018`, AFML book.
- `chassot_audrino_2026`, HARd to Beat.
- `slepaczuk_bieganowski_2024`, Supervised Autoencoders with FFD and TBL on crypto.
- `kang_kim_2025`, TBL on Korean equities.
- `singh_joubert_2019`, meta-labelling efficacy.
- `fu_et_al_2024`, GA-driven TBL for crypto pair trading.
- `nabar_shroff_2023`, Conservative Predictions.

**II. KAN architecture and interpretability**
- `liu_kan_2024`, original KAN.
- `liu_kan2_2024`, KAN 2.0 / MultKAN.
- `cho_lee_kim_2025`, VIX KAN (Algorithm 1).
- `oad_kasper_2025`, KASPER.
- `noorizadegan_2026`, Practitioner Guide to KANs.
- `yamak_et_al_2025`, KAN time-series review.

**III. BTC / crypto prediction and DL**
- `mate_confluence_2024`, TA + ML for BTC.
- `omole_enke_2024`, DL for BTC direction.
- `bourday_crypto_dl_2024`, crypto DL comparative analysis.
- `gao_decokan_2025`, DecoKAN.
- `genet_inzirillo_2024`, TKAN.
- `wu_crypto_dl_review_2024`, DL crypto review.

**IV. Foundational baselines, metrics, methods**
- `breiman_2001`, Random Forest.
- `chen_guestrin_2016`, XGBoost.
- `hochreiter_schmidhuber_1997`, LSTM.
- `loshchilov_adamw_2019`, AdamW.
- `platt_1999`, Platt scaling.
- `guo_temperature_2017`, temperature / vector scaling.
- `akiba_optuna_2019`, Optuna.
- `delong_1988`, DeLong AUC test.
- `kolmogorov_1957`, representation theorem.
- `arnold_1958`, Arnold variant.
- `corsi_2009`, HAR.
- `fama_1970`, EMH.
- `garman_klass_1980`, GK volatility.
- `yang_zhang_2000`, YZ volatility.
- `lo_mackinlay_1988`, variance ratio.

> Verify before final compile: `garman_klass_1980`, `yang_zhang_2000`, `lo_mackinlay_1988` exist in the .bib.

---

## Diagnostic checklist before submission

- [ ] Introduction first sentence states the contribution.
- [ ] Each section opens with a claim, not background.
- [ ] Each table caption is self-contained.
- [ ] Two to three significant digits everywhere.
- [ ] No em-dashes, no semicolons.
- [ ] No passive voice; search and destroy "is" / "are" passives.
- [ ] No previews or recalls.
- [ ] No footnotes for parenthetical comments (path-stitching footnote in 3.10.1 is the only methodological exception).
- [ ] Q1↔C1, Q2↔C2, Q3↔C3, Q4↔C4 mapping holds across Introduction, Methodology, Results, Discussion.
- [ ] Each construct (event, label, observation, fold, path) uses the same word throughout.
- [ ] CPCV configuration referenced as N=8, k=2, 28 splits, 7 paths everywhere.
- [ ] All cross-references resolve (Section 2.3.4 from 3.5, Section 2.3.5 from 3.10.3, etc.).
- [ ] No methodology subsection repeats theoretical content already covered in lit review.
- [ ] Conclusion ends with resonance, not a hedge.
- [ ] Reproducibility: a fellow graduate student can reproduce every number from the paper plus the appendices.
