# BTC Daily Direction Prediction Using KANs Within the AFML Framework

**Author:** Petr Terletskiy (l63023)
**Programme:** Master's in Mathematical Finance, ISEG (2024/26)
**Supervisor:** Prof. João Afonso Bastos

---

## How to use this file

This is a **bullet-point content outline**, not draft prose. Each subsection lists topics, claims, parameter values, and citations to land in the final text. Convert each block into one to three paragraphs at writing time. Each section is annotated with the Cochrane principle and Thatcher rule it implements (e.g., `[T-R3, Cochrane: lead with contribution]`), so when you write you can glance left and remember why you are doing what you are doing.

### Writing principles in force across the whole paper

> Drawn from `writing_tips_cochrane.pdf` and `rules_for_writing_elite_info_systems_papers.pdf` in the project files.

- **Cochrane: structure.** Start with the central contribution. Three pages is the upper limit for the introduction. Literature review is a separate section, brief, set off so people can skip it. Position against the 2 to 3 closest papers, do not write a JEL-style review. Tables and figures get self-contained captions. Two to three significant digits. Simple is better; the less math used, the better.
- **Cochrane: voice.** Active voice. Present tense. "I" is fine on a sole-authored paper. Search for "is" and "are" and root out passive sentences.
- **Cochrane: style.** No cute opening quotation. No "the literature has long been interested in". No previews and recalls ("as we will see in Table 6"); they signal poor organization. No footnotes for parenthetical comments.
- **Thatcher rules in force globally.**
  - R1 Open Boldly: first one to two sentences state a clear, field-relevant claim.
  - R2 Narrative Hooks: open with "despite A, we observe B".
  - R3 Lead with the Contribution: every section and paragraph opens with a claim, not background.
  - R4 One Construct, One Paragraph.
  - R5 Thread the Construct Across the Paper. Specific threads to keep consistent here: "event" (CUSUM-triggered timestamp), "label" (TBL output), "observation" (final aligned sample); "model" (one of the six benchmarked architectures); "path" (one of the seven CPCV backtest paths); "fold" (one of the twenty-eight CPCV splits).
  - R6 Parallel Logic: Q1↔C1, Q2↔C2, Q3↔C3, Q4↔C4 across Introduction, Methodology, Results, Discussion.
  - R7 Link Back to Mechanisms: Discussion connects every empirical result to the methodological mechanism that produced it.
  - R13 Compress Without Hollowing: every sentence advances the argument; no hedging.
  - R14 Avoid Stylistic Clutter: no metadiscourse, no nominalizations, no unnecessary quotes.
  - R15 Write for Reviewers: preempt objections with footnotes, appendices, robustness tables.
  - R16 End with Resonance: final sentence of each section punches; never hedges.
  - R17 Make the Paper Easy to Teach: summary boxes, diagrams, and takeaway tables.
- **User-enforced style rules.** No em-dashes. Active voice. Present tense. Triangular structure. One construct per paragraph. Self-contained table captions. End sections with resonance, not hedges.

---

## 1. Introduction

> **Section purpose.** Tell the reader what this paper does, why it matters, and what the headline finding is. Three pages maximum (Cochrane). [T-R1, T-R2, T-R3, Cochrane: lead with contribution, no clearing the throat]

### 1.1 Opening paragraphs (no subsection number)

> **First sentence is the hardest (Cochrane).** Do not start with "the finance literature has long been interested in".

- One paragraph, three sentences:
  - This thesis applies KANs to BTC daily direction prediction within the full AFML framework.
  - It benchmarks KAN against five models (AR Logistic, Logistic Regression, Random Forest, XGBoost, LSTM) under CPCV with DSR, PBO, and DeLong corrections.
  - The novel deliverable is a closed-form symbolic decision function extracted from the trained KAN.
- Telegraph the headline finding immediately (T-R1, T-R3): no model achieves DSR ≥ 0.95, consistent with market efficiency under leakage-free evaluation. The symbolic extraction pipeline produces an interpretable formula nonetheless.
- KAN reference here is one sentence only. No theorem, no splines, no architecture details. Save those for Section 2.1.

### 1.2 Research Context and Motivation [T-R2: narrative hook]

> Frame around the single observation that the crypto-ML literature reports 85% to 95% daily-direction accuracy and ask whether those numbers survive a methodology that closes off label leakage and corrects for selection bias.

- **The narrative hook (despite A, we observe B).** Despite the crypto-ML literature reporting 85% to 95% accuracy with deep learning architectures, no model in this thesis survives the AFML statistical-correction layer.
- **Why BTC.**
  - Largest cryptocurrency by market capitalization, highest cumulative return of any asset class since 2009.
  - Trades 24/7 on a transparent, publicly auditable blockchain. On-chain features are available with no equivalent in traditional finance.
  - Daily OHLCV data freely available from 2014, giving approximately 4,200 daily observations.
- **Why KANs.** Liu et al. (2024) introduced an MLP alternative whose edges carry learnable B-splines. Those splines can be distilled into closed-form symbolic functions. No prior work applies KANs to BTC daily direction or extracts symbolic formulas from a classification KAN under AFML evaluation.

### 1.3 Problems (Q1 to Q4) [T-R6: parallel logic with the contributions in 1.4]

> One subsection per question, two short paragraphs each: state the problem in the field, then state the question this thesis asks. Pull the framings from the defense slides.

#### Q1. Predictability under leakage-free evaluation

- **Problem.** Crypto-ML papers report 85% to 95% daily-direction accuracy using fixed-horizon labels with overlapping spans and naive train-test splits with no purging or embargo.
- **Question.** Does BTC daily direction remain predictable once labels stop leaking the future and overlapping observations are downweighted? Does any positive Sharpe survive the AFML statistical-correction layer (DSR, PBO, DeLong)?

#### Q2. Which feature families carry signal

- **Problem.** Most crypto-ML papers use one or two feature families (typically TA or LSTM-on-prices). No consensus on whether macro, crypto-macro, or on-chain features add information beyond price-derived features.
- **Question.** Among technical, statistical, macroeconomic, and on-chain features, which families survive multi-model permutation-importance selection across CPCV folds? Are on-chain features the "free lunch" the literature suggests?

#### Q3. KAN versus standard model families

- **Problem.** KANs have been applied to VIX forecasting and stock prediction, but never to BTC direction. Their position relative to econometric, linear, tree-based, and deep-learning baselines under a uniform protocol is unknown.
- **Question.** Where does a KAN sit relative to AR Logistic, Logistic Regression, Random Forest, XGBoost, and LSTM under identical CPCV splits, identical features, identical sample weights, and identical metrics?

#### Q4. Closed-form formula extraction

- **Problem.** Existing KAN symbolic-extraction work targets regression problems on relatively predictable series (VIX, stock prices). No work has extracted a closed-form classification formula from a CPCV-trained KAN in a weak-signal regime.
- **Question.** Can a human-readable mathematical expression for P(Up) be extracted from a CPCV-trained KAN, while preserving most of the predictive accuracy of the trained network?

### 1.4 Contributions (C1 to C4) [T-R6: parallel logic]

> One paragraph per contribution. Map one-to-one onto Q1 to Q4. Keep prose tight; the methodology chapter details the mechanics.

- **C1 → Q1. Honest evaluation under full AFML.** End-to-end pipeline applying the complete AFML stack: CUSUM event sampling, triple-barrier labels, sample weights (uniqueness × return attribution × time decay), fractional differencing, CPCV with purging and embargo, plus DSR, PBO, and DeLong corrections.
- **C2 → Q2. A 73-feature universe across four families.** 25 technical, 9 statistical (AFML Part 4), 29 external (20 macro, 1 crypto-macro, 8 on-chain from CoinMetrics), plus 10 autoregressive lag features. All 73 features compete in multi-model MDA selection.
- **C3 → Q3. Six-model apples-to-apples benchmark.** Identical CPCV splits, identical features, identical sample weights, identical metrics across all six models. Pairwise DeLong AUC tests determine which differences are statistically real.
- **C4 → Q4. Closed-form symbolic formula from KAN.** A human-readable expression for P(Up), pruned and substituted from the trained KAN via PyKAN's symbolic-extraction pipeline. The novel contribution that distinguishes this thesis from a standard model-benchmarking study.

### 1.5 Document Structure (optional, Cochrane suggests skipping)

> Cochrane: "I don't write a roadmap paragraph; readers will figure it out when they get there." ISEG conventions favor a brief structure paragraph; keep it minimal if included.

- One short paragraph naming Sections 2 to 7 in two lines max.

---

## 2. Literature Review

> **Section purpose.** Position this thesis against the closest 2 to 3 papers in each stream (Cochrane). Three streams: KANs, BTC ML prediction, AFML methodology. Each stream ends with a gap statement. [T-R3: lead with the contribution; T-R6: parallel logic]
>
> **Where the HARd to Beat paper goes.** In Section 2.3 (AFML), under "the fitting scheme matters more than model choice". The paper is methodological, not BTC-specific; its claim ("a properly fitted HAR beats RF/XGBoost/FFNN on 1,445 US equities for realized volatility") supports the AFML thesis directly: methodology dominates model choice.
>
> **Where Kolmogorov-Arnold theorem and splines go.** In Section 2.1 (KANs) only. Not in Introduction (no math before the main result, Cochrane). Not in Methodology (Methodology assumes the reader has read 2.1).

### 2.1 Kolmogorov-Arnold Networks

> **Section opening claim (T-R3).** "KANs are the only neural architecture whose learned activation functions can be distilled into closed-form symbolic expressions." State this first, then defend.

#### 2.1.1 Mathematical foundation

- Kolmogorov-Arnold representation theorem (Kolmogorov 1957; Arnold 1958): any continuous multivariate function `f(x_1, ..., x_n)` decomposes as a finite sum of continuous univariate functions. **Cite:** `kolmogorov_1957`, `arnold_1958`.
- The theorem was historically considered impractical because the inner functions can be highly non-smooth.
- Liu et al. (2024) made it practical by parameterizing those functions with learnable B-splines.

#### 2.1.2 KAN architecture (Liu et al. 2024)

- MLPs put fixed activation functions on nodes; KANs put learnable activation functions on edges.
- Each edge is a B-spline `g(x) = Σ a_k · B_k(x)` of order `k` over `G` grid intervals.
- Architecture notation: `[n_input, n_hidden, n_output]`.
- Training: Adam followed by L-BFGS, with L1-on-activations and entropy sparsity regularization.
- Key property exploited in this thesis: after training, each edge's spline can be inspected, pruned, or replaced with a symbolic function (sin, exp, x², tanh), yielding an interpretable closed-form expression.
- **Cite:** `liu_kan_2024`.

#### 2.1.3 KAN 2.0 and MultKAN (Liu et al. 2024)

- KAN 2.0 introduces MultKAN: multiplication nodes alongside addition.
- Enables discovery of multiplicative interactions (e.g., RSI × Stoch_K) without log-exp decomposition.
- Same symbolic-extraction pipeline applies. Use as future-work justification (Section 7).
- **Cite:** `liu_kan2_2024`.

#### 2.1.4 KANs for financial time series

- **VIX KAN paper (Cho, Lee, and Kim 2025).** Applies KAN to VIX forecasting (regression). Introduces Algorithm 1: Train (Adam → grid extend → LBFGS) → Prune → Symbolify → Affine fine-tune. Direct methodological ancestor of this thesis's symbolic extraction pipeline. Key difference: VIX is mean-reverting and predictable; BTC daily direction is binary classification in a weak-signal regime, which makes symbolic extraction substantially harder. **Cite:** `cho_lee_kim_2025`.
- **KASPER (Oad 2025).** KAN combined with regime detection (Gumbel-Softmax mechanism) for stock prediction. Reports R² = 0.89 and Sharpe = 12.02 on individual stocks. Closest related work in terms of KAN + finance + interpretability. Differences: regression not classification; regime detection layer absent here; Shapley-based rule extraction vs. direct formula; no AFML evaluation. **Cite:** `oad_kasper_2025`.
- **DecoKAN (Gao et al. 2025).** Applies KAN with an interpretable decomposition (trend, seasonality, residual) to cryptocurrency forecasting. Closest to this thesis on the asset-class axis (crypto), but uses a regression target with naive train-test splits and no AFML correction layer. Reinforces the gap statement: KAN-on-crypto exists, but KAN + AFML + symbolic distillation does not. **Cite:** `gao_decokan_2025`.

#### 2.1.5 Training best practices

- Coarser grids generalize better on small datasets; finer grids overfit.
- L1 + entropy regularization aids pruning and symbolification.
- Adam to L-BFGS staging: Adam escapes local minima, L-BFGS exploits curvature.
- Tanh input normalization to `[-1, 1]` matches default `grid_range`.
- **Cite:** `noorizadegan_2026`.

#### 2.1.6 Broader landscape

- Catch-all citation for KAN time-series variants (TKAN, ChebyKAN, DecoKAN), to avoid citing 40 individual variants.
- TKAN (Genet and Inzirillo 2024) is the canonical recurrent-KAN variant: replaces LSTM gates with KAN edges, demonstrating that the KAN substrate composes with classical sequence-modelling primitives. Cited here as representative of the architectural-variant literature this thesis chose not to pursue (sequence variants are incompatible with PyKAN's symbolic-extraction pipeline).
- The comprehensive review notes that symbolic distillation in financial time series is underexplored.
- **Cite:** `yamak_et_al_2025`, `genet_inzirillo_2024`.

#### 2.1.7 Gap statement

- No prior work applies KANs to BTC daily direction.
- No prior work extracts symbolic formulas from a classification KAN.
- No prior work combines KAN symbolic extraction with AFML evaluation.

### 2.2 Bitcoin Price Prediction with Machine Learning

> **Section opening claim (T-R3).** "Despite extensive ML coverage of BTC prediction, the field's reported accuracy numbers do not survive leakage-free evaluation."

#### 2.2.1 Overview of approaches

- Three families: statistical/econometric (ARIMA, GARCH, VAR); classical ML (RF, XGBoost, SVM, LR); deep learning (LSTM, GRU, CNN, Transformer).
- Two framings: price-level regression or direction classification (this thesis).

#### 2.2.2 Technical analysis and ML for BTC

- Confluence of TA and ML for BTC prediction. Identifies ROC as a top predictor (justifies inclusion of `roc_14` in this thesis's feature set). Demonstrates multi-indicator strategies outperform single-indicator.
- **Limitation.** Standard split, no purging, no overlap accounting.
- **Cite:** `mate_confluence_2024`.

#### 2.2.3 Deep learning for BTC direction

- Same problem as this thesis (BTC daily direction with DL models). Reports competitive LSTM accuracy.
- **Limitation.** Fixed-horizon labels, no CUSUM, no sample weights, no purging or embargo. Comparing conclusions illustrates how methodology shapes reported performance.
- **Cite:** `omole_enke_2024`.

#### 2.2.4 Broader crypto-DL landscape

- Catch-all citation for the broader landscape; avoids citing 20+ individual papers.
- Documents LSTM, GRU, and attention-based dominance.
- Documents recurring issues: small datasets, overfitting, no transaction costs, no risk-adjusted metrics, inconsistent evaluation.
- **Cite:** `bourday_crypto_dl_2024`, `wu_crypto_dl_review_2024`.

#### 2.2.5 Conservative prediction in noisy financial regimes

- Nabar and Shroff (2023) argue that in noisy financial settings the right behaviour is selective abstention rather than always-predict.
- A model that abstains when confidence is low can outperform one that predicts on every observation.
- Directly motivates the bet-sizing threshold mechanism in Section 3.11.1: predictions with `p ≈ 0.50` produce `bet ≈ 0`, which is structural abstention rather than a separate model.
- **Cite:** `nabar_shroff_2023`.

#### 2.2.6 Methodological weaknesses in the literature [bridge to 2.3]

- Synthesize the common problems across the papers reviewed above:
  - Naive train-test splits with overlapping labels leaking future information.
  - Fixed-horizon labeling ignores stop-loss and take-profit dynamics.
  - No sample weighting; redundant overlapping events count equally.
  - Accuracy as primary metric ignores class balance and economic significance.
  - No correction for multiple testing, so reported Sharpes are inflated.
- These are the failure modes that the AFML framework was designed to address (next subsection).

### 2.3 The AFML Methodology

> **Section opening claim (T-R3).** "AFML treats financial ML as a methodology problem first and a model problem second."

#### 2.3.1 Overview and motivation

- Lopez de Prado (2018) identifies systematic flaws in financial ML: backtest overfitting, label-overlap leakage, naive CV, inflated Sharpes from multiple testing.
- Proposes a complete framework addressing each flaw. Asset-agnostic in principle.
- **Cite:** `lopez_de_prado_2018`.

#### 2.3.2 Event-driven sampling: CUSUM filter

- Symmetric CUSUM (AFML Snippet 2.4) identifies structurally meaningful events.
- Reduces the dataset to informationally rich observations, discarding low-activity periods.
- Threshold typically calibrated as a multiple of mean daily volatility.

#### 2.3.3 Triple barrier labeling

- Three barriers per event: upper (take-profit), lower (stop-loss), vertical (time limit).
- Label = first barrier touched. Carries `t1` (resolution timestamp) for downstream purging.
- Reflects real trading outcomes, not arbitrary fixed-horizon snapshots.
- **Empirical validation outside crypto.** Kang and Kim (2025) apply TBL to Korean equities and report that TBL on raw OHLCV inputs outperforms fixed-horizon labels for short-term direction prediction. The TBL machinery transfers across asset classes; this thesis applies the same labelling logic to BTC.
- **Cite:** `lopez_de_prado_2018`, Ch. 3; `kang_kim_2025`.

#### 2.3.4 Fractional differentiation

- Integer differentiation achieves stationarity but destroys long memory; price levels preserve memory but are non-stationary.
- FFD finds the minimum `d* ∈ (0, 1)` that achieves stationarity (ADF test) while preserving maximum memory.
- Fixed-width window for computational efficiency.
- `d*` estimated per fold on training data only.
- **Cite:** `lopez_de_prado_2018`, Ch. 5.

#### 2.3.5 Sample weights

- Concurrent labels → uniqueness → return-attribution weights → time decay.
- Corrects for redundancy in overlapping labels.
- Passed to classifiers via `sample_weight` (sklearn) or per-sample loss multiplier (PyTorch).
- **Cite:** `lopez_de_prado_2018`, Ch. 4.

#### 2.3.6 Combinatorial Purged Cross-Validation

- Standard k-fold is invalid: overlapping labels leak across folds.
- CPCV: `N` contiguous groups, `k` test groups → `C(N, k)` splits, `φ(N, k)` backtest paths.
- Purging removes training observations whose labels overlap with the test period (three sufficient conditions).
- Embargo removes a buffer after each test group to prevent serial-correlation leakage.
- Path matrix enables distribution-level Sharpe analysis.
- **Cite:** `lopez_de_prado_2018`, Ch. 7 and 12.

#### 2.3.7 Deflated Sharpe Ratio and Probability of Backtest Overfitting

- **DSR.** Corrects observed Sharpe for selection bias across `n_trials` models, also corrects for non-normal returns (skew, kurtosis). DSR > 0.95 means significant after multiple-testing correction.
- **PBO.** Combinatorially symmetric CV: split paths into IS/OOS halves; check if IS-best underperforms OOS median. PBO ≈ 0 means reliable selection; PBO > 0.5 means adversarial.
- **Cite:** `lopez_de_prado_2018`, Ch. 11 and 14.

#### 2.3.8 AFML applied to crypto

- Slepaczuk and Bieganowski (2024) combine FFD and TBL with supervised autoencoders on BTC, ETH, and LTC. Walk-forward `d*` estimation. Validates that FFD + TBL transfer to crypto with improved risk-adjusted returns.
- Fu et al. (2024) extend the AFML toolkit further: a genetic-algorithm-driven TBL that searches over barrier widths and time horizons jointly with model hyperparameters, applied to a pair-trading strategy on cryptocurrencies. Demonstrates that TBL parameters themselves are tunable rather than dogmatic, but the search adds compute and is bounded by the same overfitting concerns AFML's CPCV is meant to address.
- **Limitation.** Both papers use walk-forward (or its equivalents) instead of CPCV; neither computes DSR or PBO.
- This thesis fills the evaluation gap with full CPCV + DSR + PBO and keeps the TBL parameters fixed at AFML defaults to avoid the joint-search overfitting risk.
- **Cite:** `slepaczuk_bieganowski_2024`, `fu_et_al_2024`.

#### 2.3.9 The fitting scheme matters more than model choice (HARd to Beat)

> **Why this paper sits here, not in BTC ML.** Audrino and Chassot (2024) test HAR vs. RF, lasso, GBT, FFNN on 1,445 US equities for realized volatility forecasting. Their target is not BTC, not direction, not classification; their claim is methodological.

- Across 1,445 US equities, a properly fitted linear HAR model (Corsi 2009) outperforms RF, XGBoost, GBT, and FFNN for realized volatility forecasting.
- The key finding: studies that report ML superiority over HAR used suboptimal fitting schemes (infrequent re-estimation, short training windows) that handicapped the baseline.
- When the fitting scheme is optimized (daily re-estimation, 2.5 to 4 year training window), the simpler model wins.
- Directly relevant to this thesis: the negative results (top DSR 0.2376, well below 0.95) align with this finding. Properly evaluated baselines are hard to beat, especially in a weak-signal regime like BTC.
- Reinforces Problem 1 framing: methodology dominates model choice in financial ML.
- **Cite:** `chassot_audrino_2026`, `corsi_2009`.

#### 2.3.10 Gap statement

- Few crypto-ML studies adopt the full AFML pipeline; most cherry-pick components.
- No study combines AFML and KAN for any asset.
- No study reports DSR and PBO for KAN-based predictions.
- This thesis is the first to apply the complete pipeline (CUSUM + TBL + FFD + sample weights + CPCV + DSR + PBO) to BTC direction prediction with KANs and five baselines.

---

## 3. Methodology

> **Section purpose.** Document the pipeline so a fellow graduate student can reproduce every number (Cochrane). Methodology + Results + Discussion together = at least 70% of the textual part. [T-R5: thread constructs (event/label/observation/fold/path) consistently throughout]
>
> **Cochrane principle for this chapter.** Nothing before the main result that the reader does not need to understand the main result. Present sequentially; keep math to the minimum needed for results to be interpretable.

### Pipeline overview

> Open with a single short paragraph naming the three phases. Use the table below for the at-a-glance map.

| Phase | Purpose | Output |
|-------|---------|--------|
| I. Pre-CPCV | Raw OHLCV to labelled events | `(X, y, w, t1)` |
| II. CPCV pipeline | Per-fold training and prediction | calibrated test-fold probabilities |
| III. Post-CPCV | Evaluation and interpretability | model comparison + symbolic formula |

### 3.1 Data

- **Source.** BTC-USD daily OHLCV from yfinance.
- **Range.** 2014-11-01 to 2026-05-01 (approximately 4,200 daily bars).
- **CUSUM start-date truncation.** The raw OHLCV series begins on 2014-11-01 to provide the 252-day lookback the longest-warmup features require (Hurst at 252, EMA 50/200 ratio at 200, SADF). The CUSUM event filter is then truncated to start on 2015-08-08, the date of Ethereum's Frontier launch and the first day with valid ETH/USD price data needed for the `eth_btc_ratio` feature. Events that fired before this date are dropped. The CUSUM accumulators themselves are computed on the full raw series, so events that survive truncation reflect the dynamic state of the cumulative drift over the entire pre-event history. Empirically, the mean daily volatility over the truncated window matches the full-series mean to four decimal places, confirming the EWMA was fully converged by the truncation date.
- **Why this buffer-and-truncate design.** Two leakage and data-availability problems forced the choice. First, every engineered feature must have completed its warmup window before the first labelled event so that no fold sees a partial-feature observation. Second, `eth_btc_ratio` cannot be computed before ETH started trading; without truncation, early CPCV folds under N=8 produce fully-NaN test partitions. Truncating CUSUM to start at the ETH availability date solves both problems simultaneously.
- **Validation pipeline.** Empty downloads raise; MultiIndex columns flattened; duplicate dates raise; calendar gaps ≤ 3 days forward-filled, gaps > 3 days raise; OHLCV consistency checks; NaN Close drops the row.
- **Calendar.** All rolling windows use the BTC trading calendar (7-day week, 30-day month, 90-day quarter, 180-day semester, 365-day year). BTC trades 24/7.
- **External sources (overview, detailed in 3.4).** Macro (yfinance, FRED), crypto-macro (CoinMetrics with yfinance fallback), on-chain (CoinMetrics Community API).
- **Anti-leakage.** Externals aligned via `merge_asof(direction='backward')`. CoinMetrics shifted by 1 day (end-of-day reporting convention).
- **Figure.** BTC-USD log price 2014 to 2026 with CPCV group boundaries overlaid; vertical line marking the 2015-08-08 CUSUM truncation date.
- **Cochrane note.** Do not write extensive descriptions of well-known datasets (BTC OHLCV is well-known). Keep this section to about one page.

### 3.2 Labeling [T-R5: introduces the event / label / observation thread]

> Define the constructs in order of appearance: "event" = CUSUM-triggered timestamp; "label" = TBL output; "observation" = aligned final sample.

#### 3.2.1 Daily volatility (AFML Snippet 3.1)

- EWMA std of log returns, `span = 50` (vs. AFML default 100 for equities).
- Justification: BTC's faster regime transitions (halving cycles, regulatory shocks).
- Two roles downstream: target width for TBL barriers; CUSUM threshold calibration.

#### 3.2.2 CUSUM event filter (AFML Snippet 2.4)

- Symmetric CUSUM accumulators `s_pos`, `s_neg` computed on the full raw return series from 2014-11-01 onward.
- Threshold: `h = 1.0 × mean(daily_vol)`.
- Justification for 1.0× (tightened from 1.5×): empirical sweep showed 0.5× too noisy, 3.0× too sparse; 1.0× yields workable event count with balanced classes.
- **Truncation.** After CUSUM fires its candidate events on the full series, the event index is truncated to start on `CUSUM_START_DATE = 2015-08-08` (Section 3.1). The accumulators continue to reflect the dynamic state of the cumulative drift over the entire pre-event history; only the event-firing window is restricted to the data-availability frontier of `eth_btc_ratio`. **Locked-run figures:** 1,348 candidate events fire on the full series, 82 fall before the truncation date and are dropped, leaving 1,266 events that enter triple-barrier labelling.
- The post-truncation event series reduces 4,199 daily bars to 1,266 informative events; after triple-barrier labelling and rare-label removal (Section 3.2.4) this becomes 1,159 binary-labelled observations.
- **Figure.** CUSUM accumulators alongside BTC price with event markers; the 2015-08-08 truncation boundary marked.

#### 3.2.3 Triple barrier labeling (AFML Snippets 3.2, 3.4, 3.5)

- `pt_sl = (1.5, 1.5)` symmetric (no directional bias).
- `num_days = 10` (approximately two trading weeks).
- `min_return = 0.02` (collapses small vertical-barrier returns into class 0 for later removal).
- Output: DataFrame `bins[ret, bin, t1]` where `t1` is the barrier-touch timestamp (critical for downstream purging).
- Observed mean holding period 5.0 days (median 4 days); vertical barrier hit on 19.7% of events (horizontal barriers close approximately 80% before time barrier). Locked-run figures.
- **Figure.** Triple-barrier visualization for two or three representative events.

#### 3.2.4 Rare label removal (AFML Snippet 3.8)

- `min_pct = 0.085` (raised from default 0.05 to aggressively remove residual class-0 events).
- Combined with symmetric `pt_sl` and `min_return = 0.02`, class 0 is eliminated → binary labels {-1, +1}.
- **Final aligned event count.** **1,159 events** after the 2015-08-08 CUSUM truncation and the rare-class drop. Locked-run class balance: `{+1: 661, -1: 498}` = 57.0% Up / 43.0% Down. The CUSUM filter fired 1,348 candidate events on the full series, of which 82 fell before the ETH-availability date and were dropped by the truncation, leaving 1,266 events; rare-label removal then dropped 107 class-0 events (vertical-barrier ties below the 0.02 minimum return threshold), giving the final 1,159. The previous configuration (raw data from September 2014, no CUSUM truncation) produced 1,245 events.
- **Table.** Label distribution before and after rare-label removal (self-contained caption).

### 3.3 Sample Weights (AFML Chapter 4)

> **Opening claim (T-R3).** "Overlapping labels create redundancy that biases training; AFML's four-step weighting scheme corrects for it."

- **Step 1. Concurrent label count (Snippet 4.1).** For each bar `t`, count active labels `c_t = |{i : t0_i ≤ t ≤ t1_i}|`.
- **Step 2. Average uniqueness (Snippet 4.2).** `ū_i = mean(1 / c_t)` over `[t0_i, t1_i]`. Uniqueness near 1.0 → highly informative; near 0 → redundant.
- **Step 3. Return-attribution weights (Snippet 4.10).** `w_i = |ret_i| · ū_i`, normalized to `sum(w) = len(w)` (sklearn-compatible).
- **Step 4. Time decay (Snippet 4.11).** Linear decay with `time_decay_factor = 0.4` (oldest sample weighted at 40% of newest). Re-normalize.
- **Step 5. Outlier capping.** Clip at `weight_cap_quantile = 0.99`.
- **Integration.**
  - sklearn models: `sample_weight` parameter in `.fit()`.
  - PyTorch models: per-sample multiplier in CrossEntropyLoss as `L = mean(w_i · CE(logits_i, y_i))`.
- **Figure.** Sample weights over time, annotated with notable BTC events (March 2020 COVID, 2021 ATH, 2022 FTX collapse).

### 3.4 Feature Engineering [T-R4: one construct per paragraph; group, do not enumerate]

> Group features by function. Detailed per-feature table goes to Appendix E. The main text explains the design rationale, the grouping logic, and the anti-leakage measures.

- 73 features in four groups, all eligible for MDA selection. AR Logistic restricts itself to the 10 lag columns by name from the pre-MDA matrix.

| Group | Count | Source | Purpose |
|-------|-------|--------|---------|
| Technical (TA) | 25 | OHLCV | Price/volume patterns |
| Mathematical (AFML Part 4) | 9 | Returns/log-prices | Information-theoretic, randomness, structural breaks |
| External: macro | 20 | yfinance, FRED | Macro economic environment |
| External: crypto-macro | 1 | CoinMetrics + yfinance fallback | Cross-crypto signal |
| External: on-chain | 8 | CoinMetrics | Blockchain fundamentals |
| Lag (autoregressive) | 10 | Log returns | Pure-autoregressive baseline |
| **Total** | **73** | | |

#### 3.4.1 TA features (25)

- Returns and volatility (8): `log_returns`, realized vol (annualized × √365), Garman-Klass (1980), Yang-Zhang (2000), ATR (EWMA, span=14, log-transformed), Bollinger Band width, `vol_term_7_30` (7-day vs. 30-day realized vol ratio), `vol_term_30_90` (30-day vs. 90-day realized vol ratio).
- Momentum and trend (9): RSI(14, Wilder), MACD/MACD-signal/MACD-hist (12/26/9), `roc_14` (top BTC predictor in `mate_confluence_2024`), Stoch %K/%D, Williams %R, CCI(14).
- Volume (3): OBV (sign-preserving log-transformed), Chaikin oscillator (3/10), MFI(14).
- Distribution shape (2): rolling skew/kurt (window=21).
- Trend ratios (3): EMA ratios 20/50, 50/200; VWMA ratio 20/50.
- Window convention: 21-day rolling for shape features; standard TA periods kept for named indicators (RSI=14, MACD=12/26/9). No optimization of indicator periods (avoids another layer of overfitting risk).
- **Cite:** `garman_klass_1980`, `yang_zhang_2000`, `mate_confluence_2024`.

#### 3.4.2 Mathematical features (9, AFML Part 4)

- Information-theoretic (3): Shannon entropy (window=30), Lempel-Ziv complexity (window=90), `negentropy` (window=30; difference between the maximum-entropy Gaussian benchmark and the empirical Shannon entropy of returns, so positive values flag departures from Gaussianity).
- Random walk tests (2): Hurst (window=180, R/S at sub-windows [10, 21, 42, 63]); variance ratio Lo-MacKinlay 1988 (window=90, lag=7).
- Normality test (1): Jarque-Bera (window=90).
- Structural breaks (3): SADF (min sub-length=90, lags=1), SMT polynomial-1, SMT exponential.
- Cached to `cache/math_features.parquet` (O(n²) for SADF and SMT).
- **Cite:** `lopez_de_prado_2018` Part 4, `lo_mackinlay_1988`.

#### 3.4.3 External features (29)

- **Macro (20).** `dxy_roc_30`, `us2y` (FRED DGS2 with T10Y2Y fallback), `us10y` (^TNX), `yield_curve_2y10y`, `yield_curve_10y30y`, `vix`, plus 30-day and 14-day return windows on each of seven commodity / index series: `sp500_ret_30` / `sp500_ret_14`, `nasdaq_ret_30` / `nasdaq_ret_14`, `gold_ret_30` / `gold_ret_14`, `silver_ret_30` / `silver_ret_14`, `copper_ret_30` / `copper_ret_14`, `oil_ret_30` / `oil_ret_14`, `natgas_ret_30` / `natgas_ret_14`. The 14-day variants were added to give the MDA selection step access to a faster horizon on each commodity / index, since the locked 30-day-only configuration left no shorter macro horizon in the MDA pool.
- **Crypto-macro (1).** `eth_btc_ratio` (alt-rotation signal), computed as `ETH_close / BTC_close` aligned via `merge_asof`. **ETH source priority:** CoinMetrics Community API as the primary source, trying three metrics in order (`ReferenceRateUSD` → `PriceUSD` → derived `CapMrktCurUSD / SplyCur`); the first metric returning more than 100 rows is used. yfinance ETH-USD is the final fallback. This change replaces an earlier yfinance-only implementation whose ETH-USD history began only in November 2017 and produced a 27% NaN rate over the external dataframe with entire-test-partition NaN in early CPCV folds under N=8. With the CoinMetrics PriceUSD source, ETH coverage extends back to 2015-08-08 and the residual NaN rate falls to 7.7%, all of which sits in the November 2014 to August 2015 pre-ETH-trading window and is fully truncated out by the CUSUM start-date filter (Section 3.2.2). The earlier `btc_dominance` column was removed because the CoinGecko endpoint returns BTC market cap (not bounded [0, 100] dominance) and the proxy fallback was a price-correlated approximation that the methodology could not cleanly defend.
- **On-chain (8).** `active_addr_roc_14`, `tx_count_roc_14`, `hashrate_roc_30`, `mvrv` (level), `net_exchange_flow`, `fee_per_tx`, `exchange_supply_pct`, `issuance_ntv`. CoinMetrics Community API, shifted by 1 day.
- **Anti-leakage.** All external series merged onto BTC's calendar via `merge_asof(direction='backward')`. Cache invalidates on column-set change (not just date range).

#### 3.4.4 Lag features (10)

- `AR_LAGS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]`, column prefix `log_returns_lag`.
- The locked configuration adds four lags (4, 5, 6, 21) to the previous six-lag set `[1, 2, 3, 7, 14, 30]`. The expansion fills the gap between lag 3 and lag 7 with the missing trading-week pattern and adds the three-week mark (lag 21) so AR Logistic has a full near-term lag spectrum without the previous gap structure.
- Lag features compete with engineered features in MDA (advisor-driven change to remove information asymmetry between AR Logistic and the other models).
- AR Logistic still selects the 10 lag columns by name from `X_tr_full`, regardless of MDA's choices.

#### 3.4.5 Log transforms

- `atr`: `log(|x| + 1e-8)`.
- `obv`: `sign(x) · log(|x| + 1)`.
- All other features are bounded or dimensionless.

#### 3.4.6 NaN handling across Phase 1

- Phase 1 produces NaN values from rolling-window warmup and external-data calendar gaps. Phase 1 does not drop or impute them.
- Alignment hard-asserts no entirely-NaN columns and fully populated `t1`. Soft-warns on partial-NaN columns.
- Per-fold NaN resolution happens inside the CPCV loop (Section 3.6.1).

#### 3.4.7 Alignment

- `align_for_cv(features, bins, weights) → (X, y, w, t1)` via index intersection.
- Hard assertions: non-empty intersection; no duplicate dates; monotone index; identical lengths; no all-NaN columns.
- Validation: `t1` fully populated; weights > 0; labels ∈ {-1, 0, +1}; X/y/w/t1 share indices.
- **Output.** `(1159 × 73)` X, `(1159)` y, `(1159)` w, `(1159)` t1. Locked-run class balance: `{+1: 661, -1: 498}` = 57.0% Up / 43.0% Down.
- **Table.** Alignment summary (daily bars → CUSUM events → labelled events → aligned size). Self-contained caption.

### 3.5 Cross-Validation Framework

> **Opening claim (T-R3).** "Standard k-fold CV is invalid for financial time series with overlapping labels; CPCV prevents the resulting leakage." Invest 1.5 to 2 pages. The committee must understand purging and embargo to evaluate the results.

#### 3.5.1 Why standard CV fails

- TBL labels span `[t0, t1]` (multi-day intervals).
- Overlapping spans across folds → training labels carry information about test-period prices.
- Inflates metrics: model appears to predict the future but is recognizing patterns from overlapping training labels.

#### 3.5.2 CPCV configuration

- `N_GROUPS = 8`, `K_TEST_GROUPS = 2`, `EMBARGO_PCT = 0.01`.
- Groups 0 to 6 of size `⌊T/N⌋`, group 7 absorbs remainder.
- Total splits: `C(8, 2) = 28`; backtest paths: `C(N-1, k-1) = C(7, 1) = 7`.
- Each group appears in 7 test sets. Per-group sample size approximately 144 events at the locked configuration.
- **Justification for N=8, k=2.** Yields approximately 144 events per group while keeping the training fold at approximately 851 events on average (73.4% of the 1,159 aligned events) after purging and embargo; 28 splits and 7 paths give denser combinatorial diversity for PBO than the earlier N=6 configuration (which produced 15 splits and 5 paths) without dropping per-group sample size below the rough lower bound for daily-bar AFML pipelines. The choice trades a smaller test fold per split against a larger Sharpe-matrix cross-section for PBO and DSR.
- **Locked-run audit (informational).** Across all 28 splits the average train size is 851 events, average test size is 290, average purged-per-split is 0.8 observations, and average embargoed-per-split is 17.5 observations. Per-group sample size is 144 for groups 0 to 6 and 151 for the remainder group 7. Group date ranges span 2015-08-08 (G0) to 2026-04-19 (G7). Zero leakage detected across all 28 splits.
- **Table.** Group boundaries (group ID, positional index range, date range, count). Self-contained caption.

#### 3.5.3 Purging (AFML Snippet 7.1)

- Three sufficient overlap conditions for training observation `i` against test `[t_test_start, t_test_end]`:
  1. `t_test_start ≤ t0_i ≤ t_test_end` (observation falls in test window).
  2. `t_test_start ≤ t1_i ≤ t_test_end` (label resolves in test window).
  3. `t0_i ≤ t_test_start AND t_test_end ≤ t1_i` (label spans the entire test window).
- Any training observation satisfying at least one condition is removed for that split.

#### 3.5.4 Embargo (AFML Section 7.4.2)

- `int(EMBARGO_PCT × T)` = 11 observations removed immediately after each test group (1.0% of T=1,159, rounded down).
- Applied only after the test group, not before: training labels resolving before the test starts contain no future test information.
- Prevents serial-correlation leakage (autocorrelation in volatility, momentum spillover).

#### 3.5.5 Path matrix (AFML 12.4.1)

- 7 backtest paths, each covering all 8 groups exactly once.
- For group `g`, path `p` uses the `p`-th split that includes `g` in its test set.
- Enables distribution-level Sharpe analysis instead of single-point estimates.
- **Figure.** CPCV split visualization for representative splits (0, 13, 27) with train, test, purged, and embargo regions.

#### 3.5.6 Leakage verification [T-R15: write for reviewers]

- Empirical audit across all 28 splits: zero training observations whose `t1` falls within the assigned test group's date range.
- **Table.** Leakage audit (28 rows, columns: split, test groups, train count, test count, leaks, status). Self-contained caption.

### 3.6 Per-Fold Preprocessing

> **Opening claim (T-R3).** "Three transformations happen inside the CPCV loop, fitted on training data only, to prevent test-fold statistics from leaking into the training pipeline."

#### 3.6.1 Fractional differentiation (AFML Chapter 5)

- ADF test across all 73 features at α = 0.05 identifies ATR as the only non-stationary feature. FFD applied to ATR only.
- **Procedure.**
  - Weights: `ω_0 = 1, ω_k = -ω_{k-1} · (d - k + 1) / k`; truncate when `|ω_k| < 1e-4` or `k ≥ 200`.
  - `d*` sweep: `d ∈ [0, 1]`, step 0.05; minimum `d` where ADF p-value < 0.05.
  - If no `d` achieves stationarity, default to `d* = 1.0`.
- **Application scope.** FFD applied to the full ATR series using the training-derived `d*`, so test observations have valid lookback history. Strictly backward-looking convolution.
- **Per-fold NaN policy (asymmetric by column type).**
  - Non-FFD columns: `ffill().bfill()` independently within `X_train` and within `X_test` (no boundary crossing).
  - FFD columns: drop NaN rows (forward-filling FFD output would inject stale lookback values). Logs "FFD: dropped X train, Y test NaN rows from FFD lookback." Typically 2 to 10 rows.
- **Cite:** `lopez_de_prado_2018` Ch. 5; `slepaczuk_bieganowski_2024` for walk-forward FFD validation.

#### 3.6.2 Feature scaling

- `RobustScaler` (median + IQR), fitted on training fold only, applied to all 73 features.
- Choice over StandardScaler: BTC features show fat tails and extreme values; median/IQR resists outliers, mean/std is heavily influenced by them.

#### 3.6.3 Multi-model MDA feature selection (AFML Chapter 8)

- **Multi-model design (novel relative to single-model MDA).**
  - MDA computed independently with Random Forest (500 trees, balanced class weights, captures nonlinear interactions).
  - MDA computed independently with Logistic Regression (balanced, captures linear effects).
  - Final MDA = `mean(MDA_RF, MDA_LR)` per feature.
  - Rationale: prevents bias toward any single model architecture; SFI in weak-signal regimes returns near-uniform scores; RF-only inflates tree-friendly features.
- **Inner CV.** Purged 3-fold on the training set (same `t1`-based overlap conditions as outer CPCV).
- **Selection rule.** Keep features with averaged MDA > 0; cap at `MDA_TOP_K_FRAC = 0.30` (approximately 22 of 73); minimum floor of 5 features.
- **TOP_K_FRAC tightening rationale (advisor-reviewed).** The cap was tightened from an earlier 0.40 to 0.30 in the locked configuration after a high-PBO run with the looser setting. Across the earlier-run CPCV folds, only approximately 6 features cleared 50% selection frequency in the stability bar chart, indicating that the long tail of the MDA-ranked feature set was contributing variance rather than signal. Tightening to 0.30 forces approximately 22 features through the bottleneck and aligns the selection cap with the empirical stability finding. The trade-off is that one or two folds may select fewer features than they would have at 0.40, but those folds were also the ones contributing the most rank variance to PBO, so the tightening attacks the right problem. An even tighter 0.25 cap (approximately 18 features) was tested in an interim configuration; 0.30 was retained as the final value because it preserves enough feature diversity in the inner Optuna step without re-introducing the long tail.
- **AR Logistic exception.** Bypasses MDA entirely; receives pre-MDA matrix and selects 10 lag columns by name.
- Typical result: approximately 22 features selected per fold from 73 candidates (down from approximately 26 to 30 under the previous 0.40 cap; up from approximately 18 under the interim 0.25 cap).

### 3.7 Models [T-R4: one construct per paragraph; six models, four families]

> Six models, four families. Summary table at the end. Subsections describe what is unique about each family. Shared elements (sample weights, class balancing, calibration) in 3.8 and 3.9.

#### 3.7.1 AR Logistic (econometric baseline)

- Tests pure price momentum vs. 73 engineered features.
- Lags `[1, 2, 3, 7, 14, 30]` of log returns.
- Architecture: sklearn `LogisticRegression` with C=1.0, L2, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`.
- 3 seeds, no per-split tuning (deterministic baseline).
- Consumes the 10 lag columns by name from `X_tr_full`, independent of MDA.

#### 3.7.2 Logistic Regression (linear ML baseline)

- On MDA-selected features.
- `class_weight='balanced'` + AFML `sample_weight` (dual weighting).
- Tuned per split: `C` (log-uniform [1e-4, 1e2]), `penalty` ∈ {l1, l2}.
- Solver auto-selected: `liblinear` for L1, `lbfgs` for L2.
- 3 seeds, 30 trials per split.

#### 3.7.3 Random Forest

- 500-tree ensemble, `class_weight='balanced_subsample'`, `n_jobs=-1`.
- Tuned per split: `n_estimators` ∈ [100, 250] step 50 (capped from an earlier 300 ceiling; trees in a noisy regime do not benefit from more than 250), `max_depth` ∈ [2, 6] (tightened from earlier [3, 15]; depth 6 has 64 leaves which is plenty for approximately 851-sample training folds, and shallower forests vote in tighter agreement, reducing the disagreement that surfaces as path-Sharpe variance), `min_samples_leaf` ∈ [15, 40] (raised from earlier [1, 30]; a floor of 15 forces each leaf to represent at least 1.8% of the training fold, preventing leaves that fit just a handful of high-volatility events), `max_features` ∈ {sqrt, log2}.
- 3 seeds, 30 trials per split.
- **Cite:** `breiman_2001`.

#### 3.7.4 XGBoost

- 500-tree gradient-boosted ensemble with early stopping at 20 rounds.
- Objective `binary:logistic`; `scale_pos_weight` from class balance.
- Tuned per split: `max_depth` ∈ [1, 3] (tightened from earlier [2, 6]; XGBoost's sequential boosting compounds depth nonlinearly across rounds, so depth 3 across 50 boosting rounds already produces substantial nonlinear capacity, and depth 6 in this regime memorises residuals), `learning_rate` log-uniform [0.01, 0.3] (floor at 0.01; below this, training takes forever and effectively underfits), `min_child_weight` ∈ [5, 30] (floor raised from 1 to align with RF's leaf-size discipline; with approximately 851 train samples a `min_child_weight=1` permits trees to split off single-event leaves), `subsample` and `colsample_bytree` ∈ [0.6, 1.0], `gamma` log-uniform [1e-8, 1.0], `reg_alpha`, `reg_lambda` log-uniform [1e-8, 10.0].
- **Calibration set dual role.** Calibration set acts as eval set for early stopping AND as Platt-fit data. Acknowledged as a mild dependency: only ensemble size affected, no individual tree decisions.
- 3 seeds, 30 trials per split.
- **Cite:** `chen_guestrin_2016`.

#### 3.7.5 LSTM

> Methodology assumes the reader has read 2.1; do not re-explain the LSTM architecture from first principles.

- **Architecture.** Single-layer `nn.LSTM` (`num_layers=1` hardcoded) → last hidden state from the final layer → LayerNorm → dropout → linear classifier. Hidden size, dropout, and learning rate are tuned per split; `num_layers` is no longer searched.
- **Sliding window.** `LSTM_WINDOW = 14` (deliberately close to TBL `num_days = 10`; longer windows attenuate gradient signal and inflate parameter-to-sample ratio). Reduces effective training count from `N` to `N - 13` sequences. `last_valid_indices` stored for re-alignment.
- **Last-hidden-state pooling.** Earlier learned-attention pooling was removed: with window=14 and approximately 851-sample folds, additional attention parameters did not improve performance.
- **Tanh input normalization.** `z = tanh((x - μ) / σ)`, mean and std fitted on training data only.
- **Training stack.** AdamW (lr tuned, `weight_decay=1e-4`), CrossEntropyLoss with class weights and AFML sample weights, label smoothing 0.1, gradient clipping (max norm 1.0), cosine annealing warm restarts (`T_0=25`, `T_mult=2`), batch size 64, max 100 epochs, early stopping patience 15, best-state restoration.
- **Tuning consistency.** `LSTMClassifier.__init__` reads module-level constants at call time (not as default args), so tuning overrides actually reach the model. Tuning runs at epochs=50, patience=7; production refits at epochs=100, patience=15. This is the only axis where tuning and production diverge; documented as a deliberate compute-vs-fidelity trade-off.
- Tuned per split: `hidden_size` ∈ [16, 32] step 16, `num_layers` fixed at 1 (no longer searched; tightened from earlier [1, 2] then [1, 3]; two- and three-layer LSTMs on 1,159 events are deep-overfit territory and the additional layer added variance to path-Sharpes without improving accuracy. Hardcoding to 1 frees Optuna trials for finer exploration of dropout and learning_rate), `dropout` ∈ [0.1, 0.5] (floor raised from 0.0), `lr` log-uniform [1e-4, 5e-2].
- 2 seeds, 30 trials per split.
- **Cite:** `hochreiter_schmidhuber_1997`, `loshchilov_adamw_2019`.

#### 3.7.6 KAN

> Methodology assumes the reader has read 2.1; this section covers implementation specifics only (efficient-kan vs PyKAN, this thesis's architecture, hyperparameters).

- **Library and architecture.** `efficient_kan.KAN([n_features, width1, 2], grid_size=grid, spline_order=3, grid_range=[-1, 1])`. Single hidden layer by construction: the second hidden layer is permanently disabled (`width2=0` hardcoded in `tuning.py`). Width is tuned per split (`width1` ∈ [2, 6]) and grid size (∈ {3, 5}).
- **Why single-hidden-layer architecture.** The CPCV-evaluated KAN matches the architecture used in the Phase 3 symbolic extraction. The benchmark numbers and the extracted symbolic formula therefore describe the same model rather than two unrelated KAN topologies. A two-hidden-layer KAN would let symbolic extraction nest trigonometric primitives in trigonometric primitives, producing fourth-order compositions that lose interpretability and would force the symbolic chapter to caveat that the formula approximates a different model than the one benchmarked.
- **Tanh input normalization** matching grid range.
- **Training stack.** AdamW (lr and weight_decay tuned), CrossEntropyLoss with class weights and AFML sample weights, label smoothing 0.1, gradient clipping (max norm 1.0), cosine annealing warm restarts (`T_0=30`, `T_mult=2`), early stopping patience 20, best-state restoration. Max 200 epochs.
- **Single-grid training (no coarse-to-fine).** With approximately 851 training samples per CPCV fold, grid refinement adds parameters faster than the data can support.
- **No SWA, no entropy regularization.** SWA conflicted with early stopping; entropy regularization was redundant with `label_smoothing=0.1`. Removed for coherence.
- **Dual-library strategy.** efficient-kan for all 28 CPCV splits (fast, stable, standard PyTorch). PyKAN re-trained independently for symbolic extraction (Section 3.12), where only PyKAN exposes `prune()`, `suggest_symbolic()`, `fix_symbolic()`, `symbolic_formula()`. Both share the B-spline basis, tanh normalization, and now the single-hidden-layer topology.
- Tuned per split: `width1` ∈ [2, 6] (tightened from earlier [3, 12] then [3, 16]; the cap is set at 6 to keep the symbolic formula extracted in Phase 3 humanly readable, since each surviving width1 unit becomes one additive term plus interactions in the closed-form expression), `width2` fixed at 0 (no longer searched), `lr` log-uniform [5e-4, 5e-2], `weight_decay` log-uniform [1e-5, 5e-3], `grid` ∈ {3, 5} (dropped grid=8 to prevent memorization).
- 2 seeds, 30 trials per split.
- **Cite:** `liu_kan_2024`.

#### 3.7.7 Shared neural design

- LSTM and KAN both use dual weighting in CrossEntropyLoss: class weights (inversely proportional to class frequency) AND AFML sample weights as per-sample multipliers.
- **Seed asymmetry justification.** Neural models use 2 seeds; classical models use 3. Neural training is 5 to 10 times more expensive per seed; 2 seeds provide enough variance estimation while keeping total runtime manageable across 28 splits. Split-level metrics are averaged across available seeds; the evaluation code handles the asymmetry by skipping missing entries.
- All models share the BaseModel interface: `fit`, `predict_proba`, `predict_logits`, `get_name`. Identical training/evaluation conditions across architectures.

#### 3.7.8 Summary table [T-R17: easy to teach]

> **Self-contained caption (Cochrane).** "Summary of the six models evaluated under CPCV (N=8, k=2, 28 splits, 7 backtest paths). All models receive AFML sample weights and balanced class weights. Hyperparameters listed under 'Tuned' are optimized per fold via Optuna TPE (Section 3.8); 'Fixed' parameters are held constant across all folds. Role describes what hypothesis each model tests relative to the research questions."

| Model | Family | Architecture | Fixed params | Tuned params | Seeds | Trials | Role |
|-------|--------|--------------|--------------|--------------|-------|--------|------|
| AR Logistic | Econometric | LR on lags [1,2,3,7,14,30] | C=1.0, L2, max_iter=1000 | (none) | 3 | 0 | Pure momentum (Q1, Q2) |
| Logistic Regression | Linear ML | LR on selected features | max_iter=1000 | C, penalty | 3 | 30 | Linear baseline (Q3) |
| Random Forest | Ensemble | balanced_subsample | n_jobs=-1 | n_estimators ≤ 250, max_depth ∈ [2, 6], min_samples_leaf ∈ [15, 40], max_features | 3 | 30 | Nonlinear ensemble (Q3) |
| XGBoost | Ensemble | 500 trees + early stop@20 | binary:logistic | max_depth ∈ [1, 3], lr, min_child_weight ∈ [5, 30], subsample, etc. (8) | 3 | 30 | Gradient boosting (Q3) |
| LSTM | Neural | window=14, last-hidden pooling, num_layers=1 | T_0=25, batch=64 | hidden ∈ {16, 32}, dropout ∈ [0.1, 0.5], lr | 2 | 30 | Temporal dependencies (Q3) |
| KAN | Neural | [n, width1, 2], single hidden layer, k=3 | width2=0, T_0=30, label_smooth=0.1 | width1 ∈ [2, 6], grid ∈ {3, 5}, lr, weight_decay | 2 | 30 | Interpretable architecture (Q3, Q4) |

### 3.8 Hyperparameter Tuning

- **Architecture.** Nested per-split Optuna study inside each CPCV training fold. AFML Ch. 7 compliant: outer CPCV provides unbiased test folds; inner tuning operates entirely within the training fold. Test fold never seen during tuning.
- **Inner CV.** Purged 3-fold (`N_INNER_FOLDS=3`), 10-observation embargo around inner-fold boundaries (matches TBL `num_days=10`).
- **Optuna config.** TPE sampler, `seed=42`. MedianPruner with `n_startup_trials=5` (classical) or 3 (neural), `n_warmup_steps=1`. Pruner kills trials whose intermediate log loss falls below the median of completed trials.
- **Trial budget.** `n_trials=30` per tuned model per split, applied uniformly across LR, RF, XGBoost, LSTM, and KAN. The notebook passes a single `n_trials=30` parameter so every tuned model competes on the same budget; AR Logistic is not tuned. The wider module defaults (`N_TRIALS_CLASSICAL=60`, `N_TRIALS_NEURAL=40`) are reserved for sensitivity experiments outside the headline run.
- **Per-split tuned params application.** `_apply_tuned_params` writes the best params to module-level constants (e.g., `kan_model.KAN_HIDDEN`) before the model training loop for that split. `_reset_module_defaults` snapshots pristine values on first invocation and restores them on subsequent runs to prevent contamination across calls.
- **DSR/PBO validity.** `n_trials` in DSR counts the number of compared models (6), NOT the Optuna trials per split. Optuna trials happen inside the training fold and do not affect test-fold Sharpe estimates.
- **Cite:** `akiba_optuna_2019`, `lopez_de_prado_2018` Ch. 7.

### 3.9 Calibration

> **Opening claim (T-R3).** "Bet sizing depends on calibrated probabilities; miscalibrated probabilities produce systematically wrong position sizes."

- **Calibration set.** 80/20 chronological split of the training fold; calibrator fitted on the held-out 20% (approximately 170 obs at N=8, with model-training set approximately 680 obs after the split). Never touches test data.
- **Two methods, auto-selected by model type.**

| Method | Models | Input | Mechanism |
|--------|--------|-------|-----------|
| Platt scaling (Platt 1999) | AR Logistic, LR, RF, XGBoost | 1D log-odds | `LogisticRegression(C=1e10)` mapping logits → calibrated proba |
| Vector scaling (Guo et al. 2017, §4.2) | LSTM, KAN | 2D logits | Fit `T` and per-class bias `b` minimizing NLL of `softmax((logits + b) / T)` via L-BFGS-B with `T ∈ [0.05, 20]`, `b_c ∈ [-5, 5]` |

- **Why vector scaling rather than temperature scaling [T-R15: write for reviewers, preempt the obvious objection].** A pre-final calibration audit revealed that LSTM and KAN systematically under-predicted P(y=1) by 10 to 23 percentage points, while the empirical base rate of class 1 was approximately 0.55. Pure temperature scaling preserves the argmax of raw logits by construction, so a single `T` cannot shift "lean class 0" to "lean class 1". The bias propagated through bet sizing as systematic short bets in upward-drifting regimes, contributing to a negative path Sharpe in the early KAN equity curves. Vector scaling adds a per-class bias `b` that lifts the directional constraint (Guo et al. 2017 recommends this as the natural extension when temperature scaling alone is insufficient). The substitution was made before the final evaluation pass: a correction of methodological inadequacy, not test-set-informed model selection.
- **Calibration set dual role disclosure.** The 20% subset serves as both early-stopping monitor (XGBoost) and Platt/vector input. Each calibration method fits at most three parameters; early stopping affects only ensemble size, not individual tree decisions. Splitting the small cal set further would degrade both purposes.
- **Cite:** `platt_1999`, `guo_temperature_2017`.

### 3.10 Pipeline Orchestration

- Single entry point: `run_cpcv_pipeline(X, y, w, t1, bins_ret, ..., tune=False, n_trials=None)`.
- Module-default reset on every invocation prevents tuned-value contamination across runs.
- **Per-split execution (28 splits total).**
  1. Extract fold data via positional indices.
  2. `preprocess_fold()` (FFD → scale → MDA select).
  3. Re-align `y, w, t1` after FFD may drop rows.
  4. Keep `X_tr_full` for AR Logistic alongside `X_tr_sel` for the others.
  5. Chronological 80/20 train/cal split.
  6. If `tune=True`: nested Optuna on `X_tr_sel`, apply best params to module constants.
  7. For each `(model, seed)`: create model, route correct X, fit, calibrate, predict on test.
  8. Store keyed by `(model_name, split_idx, seed)`.
- Models run separately in the notebook (Section 4). Results merged via dict union.
- Failed fits caught, logged, skipped.

### 3.11 Evaluation Framework

> **Module constants.** `TRANSACTION_COST=0.001`, `MIN_BET_SIZE=0.05`, `MAX_BET_SIZE=0.75`, `BET_DISCRETIZATION=[0.0, 0.25, 0.50, 0.75]` (1.0 dropped, consistent with the 0.75 cap), `ANNUALIZATION_FACTOR=365`, `RISK_FREE_RATE=0.0`.

#### 3.11.1 Bet sizing (AFML Chapter 10.3)

- Direction: `sign = +1 if P(up) > P(down) else -1`.
- Confidence: `p = max(P(0), P(1)) ∈ [0.5, 1.0]`.
- Z-score: `z = (p - 0.5) / sqrt(p · (1 - p) + 1e-10)`.
- Raw bet: `2 · Φ(z) - 1`.
- Cap: `clip(raw_bet, -0.75, 0.75)`.
- Threshold: `|bet| < 0.05 → 0`.
- Discretize: nearest of {0, 0.25, 0.5, 0.75}.
- Apply sign.
- **Implicit abstention mechanism.** Predictions with `p ≈ 0.50` produce `bet ≈ 0`. Aligns with the Conservative Predictions framework (Section 2.2.5).
- **Cite:** `lopez_de_prado_2018` Ch. 10; `nabar_shroff_2023`.

#### 3.11.2 Strategy returns

- `gross = bet · label_return`; `turnover = |Δbet|`; `tx_cost = 0.1% · turnover`; `net = gross - tx_cost`.
- 0.1% per-trade cost: conservative for major BTC exchanges (maker 0.02% to 0.10%, taker 0.04% to 0.10%, plus slippage).
- Annualization at 365 (BTC trades 24/7); risk-free rate 0 (no universally accepted crypto risk-free rate).

#### 3.11.3 Path stitching (AFML 12.4.1)

- 7 backtest paths assembled from the 28 splits.
- Collects `(group_id, split_id)` from `path_map[path_id]`.
- For each pair: retrieves the split's stored predictions AND filters to events whose positional index falls within `group_bounds[group_id]`. Critical: without the filter, events from co-tested groups get duplicated.
- Multi-seed: averages calibrated probabilities across seeds before bet sizing (variance reduction by approximately 1/√n_seeds).
- Concatenate, sort, assert no duplicate timestamps.
- **Bug-fix disclosure [T-R15].** Earlier implementation pulled each split's full test set, double- or quintuple-counting events. Surfaced as 1/3/5 duplication pattern by direct timestamp inspection. Fix is the group filter described above. All path-level metrics in this thesis use the corrected stitching; the bug is preserved here as a transparency disclosure.

#### 3.11.4 Path performance metrics

| Metric | Formula |
|--------|---------|
| Annualized Sharpe | `(mean / std) · √365` (daily-equivalent scale) |
| Cumulative return | `Π(1 + r_t) - 1` |
| Annualized return | `(1 + cum_ret)^(1 / years_elapsed) - 1` (calendar-time CAGR) |
| Max drawdown | `min((equity - running_max) / running_max)` |
| Time under water | Longest consecutive run below previous peak (days) |
| Win rate | Fraction of traded observations with positive return |
| Profit factor | `Σ(positive) / |Σ(negative)|`; NaN if no trades, `inf` if no losses |
| n_trades | Count where `bet ≠ 0` |
| n_returns | Length of strategy returns including zero-bet rows |
| Skew, kurt | Distribution shape of strategy returns |

- **Sharpe annualization disclosure.** Strategy returns are sampled at CUSUM events (approximately 75 per year), not daily bars (365 per year). The `√365` choice yields the daily-equivalent scale; bet-frequency annualization `√75 ≈ 8.7` would give roughly half the reported Sharpe. Internally consistent (DSR de-annualizes by the same `√365`, so DSR verdicts and rankings are convention-invariant), but the absolute Sharpes sit on the daily-equivalent scale. Disclosed explicitly.

#### 3.11.5 Deflated Sharpe Ratio (AFML Chapter 14)

```
E[max SR] = √(2·ln(n)) · (1 - γ/(2·ln(n))) + γ/(2·√(2·ln(n)))
SR_std    = √((1 - skew·SR + (kurt + 2)/4 · SR²) / (n_obs - 1))
DSR       = Φ((SR_observed - E[max SR]) / SR_std)
```

- `n` = `n_trials` = 6 (number of compared models). NOT Optuna trials.
- **Kurtosis convention.** Mertens (2002) formula assumes raw kurtosis (3 for normal); `scipy.stats.kurtosis` returns excess (0 for normal). Implementation converts: `(γ_4 - 1)/4 = (excess + 2)/4`. For normal returns reduces to `1 + SR²/2` per Lo (2002).
- **n_obs source.** `compute_model_summary` passes `avg_n_returns` (mean across paths of `len(strategy_returns)`), matching the n that estimated the Sharpe.
- NaN-safety clamp on the variance term before sqrt.
- DSR > 0.95 → significant after multiple-testing correction.

#### 3.11.6 PBO via CSCV (AFML Chapter 11)

- Path Sharpe matrix shape (6 models × 7 paths).
- All `C(7, 3) = 35` IS/OOS partitions of the 7 paths (IS=3 paths, OOS=4 paths; with 7 paths the split is asymmetric by construction).
- For each partition: identify IS-best, check if it underperforms OOS median.
- PBO = fraction of partitions where IS-best underperforms.
- PBO < 0.3 → robust selection; PBO > 0.5 → anti-predictive (IS winner is OOS loser).
- The N=8 configuration delivers 35 IS/OOS partitions vs. 10 under N=6, increasing PBO resolution from 0.10 to roughly 0.029 per partition.

#### 3.11.7 DeLong pairwise AUC (DeLong et al. 1988)

- Per `(model, split)`: average predicted probas across available seeds (3 sklearn, 2 LSTM/KAN). Matches what `stitch_paths` does for path-level metrics.
- Pool seed-averaged predictions across all 28 splits per model (valid because CPCV test sets are non-overlapping).
- Z-statistic: `z = (AUC_a - AUC_b) / sqrt(Var(AUC_a) + Var(AUC_b) - 2·Cov(AUC_a, AUC_b))`.
- Two-sided p-value from standard normal.
- 15 pairwise comparisons total (C(6, 2) = 15 model pairs). No Bonferroni correction applied; acknowledged as a limitation in 5.4.
- **Earlier issue disclosure [T-R15].** A prior version used only `seed=0`, making AUC and z-statistics depend on which initialization happened to be labelled seed 0. Averaging across seeds before pooling removes this dependence.
- **Cite:** `delong_1988`.

#### 3.11.8 Stability diagnostics

- **Feature stability.** Per-feature selection frequency across 28 folds. Frequency > 0.80 → "stable". Flat profile → diffuse signal across many features (a finding, not a bug).
- **FFD stability.** Mean and std of `d*` for ATR across 28 folds. `std(d*) < 0.1` → consistent stationarity structure across periods; `> 0.1` → time-varying persistence.

#### 3.11.9 Model comparison and ranking

- **Primary criterion:** median path Sharpe across 7 paths (descending).
- **Tiebreaker:** std Sharpe (ascending; prefer consistency).
- Columns: rank, model, median_sharpe, std_sharpe, DSR, mean_F1, mean_acc, mean_log_loss, mean_AUC, median_max_dd, median_cum_ret, median_win_rate, median_profit_factor.
- Self-contained caption (Cochrane).

### 3.12 Symbolic Extraction

> Frame as exploratory analysis (per advisor guidance), not the core contribution. Keep main text approximately 1.5 pages. Detailed training logs, edge-by-edge R² values, unsimplified formulas → Appendix F.

#### 3.12.1 Purpose and architecture

- Operates after all 28 CPCV splits have been evaluated.
- Reconstructs from `prep_info`: stored `d*`, fitted scaler, selected feature list. No per-instance state passed between efficient-kan and PyKAN.
- Inherits architecture defaults from `kan_model.py`: `KAN_HIDDEN=5`, `KAN_GRID=5`, `KAN_K=3`. Applies data-aware safety floor (`PYKAN_MIN_SAMPLES_PER_PARAM=5`).

#### 3.12.2 Fold selection

- `fold_selection="last"` (split 27): test set covers most recent data; closest to deployment scenario.
- Justification: most recent data reflects current regime. Acknowledged limitation: formula is fold-specific.
- Alternative `"best"` (highest KAN F1) maximizes meaningful symbolic output but biases toward best-performing fold.

#### 3.12.3 Data preparation

- Re-generates CPCV splits for training indices.
- Applies stored `d*` to full series; extracts training fold.
- Applies stored `RobustScaler`.
- Selects features (stored selection or `n_top_features=5` override on most stable features).
- 80/20 chronological split into model-train + validation.
- Tanh normalization fitted on training portion: `z = tanh((x - mean) / (std + 1e-8))`.

#### 3.12.4 Three-phase PyKAN training (Algorithm 1, Step 1, from `cho_lee_kim_2025`)

| Phase | Optimizer | Steps | Key feature |
|-------|-----------|-------|-------------|
| 1. Adam | Adam (lr=1e-3, wd=1e-3) | 600 | Gaussian noise injection (`std=0.05`) clamped to [-1, 1]; dropout-like regularizer; early stopping on val loss |
| 2a. LBFGS warmup | LBFGS (lr=0.01) | 20 | No regularization; light refinement |
| 2b. LBFGS sparsity | LBFGS (lr=0.01) | 20 | L1 + entropy regularization (`lamb=0.002`); encourages sparse activations |

- **Grid extension disabled** (`PYKAN_GRID_EXTEND=False`): with 530 training samples and 133 validation samples (training fold of split 27 after the 80/20 split, locked run), refining grid 3 to 5 adds parameters faster than data supports.
- **Accuracy gate:** if val acc < 53% after Adam, log warning but continue. Symbolic extraction may yield constants.
- **Architecture matches the CPCV-benchmarked KAN.** The PyKAN model trained here uses the same single-hidden-layer topology and the same width cap (`width1` ≤ 6) as the efficient-kan model evaluated under CPCV. The extracted symbolic formula therefore describes the same architecture the benchmark numbers describe, not a simplified surrogate. This is a structural change from earlier configurations, where the symbolic-extraction KAN was constrained more tightly than the CPCV-evaluated KAN to keep sympy simplification tractable; the previous architecture-mismatch caveat (`PYKAN_SYMBOLIC_WIDTH_CAP`, `PYKAN_SYMBOLIC_DROP_WIDTH2`, `PYKAN_SYMBOLIC_FORCE_GRID`) is no longer needed.

#### 3.12.5 Pruning (Algorithm 1, Step 2)

- Forward pass populates cached activations.
- `model.attribute()` for importance scoring.
- `model.prune(threshold=PRUNE_THRESHOLD=0.01)` with multi-API fallback (PyKAN versions vary).
- Verify pruned model still forward-passes; revert if broken.
- Typical compression (with `width1 ≤ 6` from the new tuning regime): a `[n_features, 6, 2]` topology often prunes to something like `[n_features, 3, 2]` after training-time activation thresholds. Earlier configurations with `width1` up to 12 produced richer pre-prune networks but created the architectural-mismatch problem documented in 3.12.4.

#### 3.12.6 Symbolification (Algorithm 1, Step 3)

- Library (14 functions): `x, x², x³, x⁴, exp, log, sqrt, tanh, sin, cos, abs, sgn, arctan, 0`.
- Per-edge: `model.suggest_symbolic(l, i, j, topk=5, lib=SYMBOLIC_LIBRARY)` returns ranked candidates.
- Three format handlers (PyKAN versions return DataFrame, flat tuple, or nested tuples).
- **Constant-skip logic / brute-force fallback.** When `"0"` wins (always wins `total_loss` due to zero complexity penalty), iterate manually through 14 candidates, call `fix_symbolic` for each, capture R² from PyKAN's stdout via regex `r"r2 is ([\d.eE+-]+)"`, keep best non-constant function. Restore model state between trials.
- Selection: best non-constant with R² ≥ `SYMBOLIC_R2_THRESHOLD=0.30` (lowered from 0.50; 0.50 was too aggressive in the weak-signal regime).
- Below threshold → keep spline (partial symbolification).

#### 3.12.7 Affine fine-tuning (Algorithm 1, Step 4)

- 30 LBFGS steps at lr=0.0004 (from VIX KAN paper) on remaining affine parameters `a, b, c, d` per symbolic edge.
- Adjusts the symbolic fit without changing the function family.
- NaN detection reverts to pre-fine-tune state.

#### 3.12.8 Formula extraction

- Tries three SymPy variable naming conventions: `x1..xn`, `x_1..x_n`, `x_0..x_{n-1}`.
- `model.symbolic_formula(var=...)` with no-var fallback.
- Two outputs: `logit_bearish` (class 0), `logit_bullish` (class 1).
- Decision function: `logit_bullish - logit_bearish`.
- `P(up) = sigmoid(decision) = 1 / (1 + exp(-decision))`.
- Substitute placeholder variables with feature names.
- `sympy.simplify` with **30-second timeout** via threading (sympy hangs on complex expressions).
- `sympy.nsimplify(tolerance=1e-3)` for cleaner rational coefficients.
- Identifies surviving features (those in `decision.free_symbols`).

#### 3.12.9 Output dictionary

- `logit_bearish`, `logit_bullish`, `decision_function`, `p_up_formula` (strings).
- `sympy_objects` (SymPy expression objects for partial-derivative sensitivity).
- `pre_symbolic_accuracy`, `post_symbolic_accuracy`.
- `symbolification_rate` (fraction of edges symbolified).
- `pruned_architecture` (e.g., `[5, 3, 2]`).
- `surviving_features`.

#### 3.12.10 Known PyKAN fragilities (transparency) [T-R15]

- `'sigmoid'` is NOT in PyKAN's internal `SYMBOLIC_LIB` → `KeyError`.
- `'1/x'` causes division-by-zero at affine fine-tuning.
- PyKAN uses 1-based variable naming (`x_1..x_n`).
- `suggest_symbolic` return format varies across PyKAN versions.
- `sympy.simplify` can hang indefinitely (→ 30s timeout).
- `"0"` always wins `total_loss` (→ brute-force fallback).

#### 3.12.11 Defensive input handling at `prepare_extraction_data`

- Coerces `y` to `pd.Series` indexed on `X.index` regardless of whether caller passed a Series, numpy array, or other array-like.
- Length mismatch raises clear `ValueError` rather than letting pandas's generic length error propagate.
- Catches a common notebook pattern where `y` gets shadowed by a pooled-prediction array.

#### 3.12.12 Acknowledged limitations (also in 5.3)

- Fold-specific (different folds → potentially different formulas).
- Small sample (530 training observations and 133 validation observations after 80/20 split on split 27).
- R² threshold sensitivity (lowered to 0.30 to admit symbolic fits in weak-signal regime).
- Post-symbolic accuracy may be lower than pre-symbolic accuracy.
- Brute-force fallback depends on PyKAN's stdout format.
- **Architecture parity (no longer a limitation, formerly a caveat).** The single-hidden-layer / `width1 ≤ 6` configuration now matches the CPCV-benchmarked KAN, so the extracted formula describes the same architecture the benchmark numbers describe. The earlier mismatch (a tighter symbolic-extraction architecture than the CPCV one) is documented in 3.12.4 for transparency but does not apply to the locked configuration.
- **Cite:** `cho_lee_kim_2025`, `liu_kan_2024`, `noorizadegan_2026`.

### 3.13 Diagnostics

> Separate module from evaluation. Holds interactive inspection helpers for the thesis chapters. Does not belong to the AFML evaluation protocol per se.

- **Calibration audit.** `pool_predictions`, `calibration_audit` (binned predicted-vs-empirical, bins ≥ 10 samples). The diagnostic that exposed the temperature-vs-vector-scaling issue.
- **Path-level dispersion and regime concentration.** `compute_top_k_concentration` (top-K share of cumulative return), `build_path_dispersion_table`, `summarize_path_dispersion`.
- **Bet-size distribution.** `compute_bet_size_summary` (abstention rate, mean/median |bet|, share at cap, long/short balance).
- **Reliability curves.** `compute_reliability_curve` for binned predicted-mean vs. empirical-mean diagrams.

### Leakage prevention summary table [T-R15: write for reviewers; T-R17: easy to teach]

| Risk | Mitigation | Where |
|------|-----------|-------|
| FFD `d*` on full data | per-fold training-only estimation | `preprocessing.py` |
| Scaler on full data | RobustScaler on training fold only | `preprocessing.py` |
| Feature selection on full data | multi-model MDA on training fold only | `preprocessing.py` |
| HP tuning on full data | nested Optuna inside training fold (3-fold purged) | `tuning.py` |
| Overlapping TBL labels | three-condition purging | `cv.py` |
| Serial correlation across CV boundary | embargo after test boundary | `cv.py` |
| Calibration on test data | held-out 20% of training fold, never test | `calibration.py` |
| XGB early-stop + cal shared cell | acknowledged; only ensemble size affected | `pipeline.py` |
| On-chain look-ahead | CoinMetrics shifted by 1 day | `external_features.py` |
| CUSUM threshold on full data | minor approximation; acknowledged, negligible impact | `labeling.py` |
| NaN imputation across train-test | `ffill().bfill()` independent within each partition; FFD columns drop | `preprocessing.py` |

---

## 4. Results

> **Cochrane.** Lead with the main result. **T-R10.** Frame empirics as theory tests. Numbers below come from the locked end-to-end run (`main.ipynb`, May 2026).

### 4.1 Headline (opening paragraph) [T-R1, T-R3]

- "No model achieves DSR ≥ 0.95. The top DSR among trained models (XGBoost) is 0.2376, far below significance after correcting for selection bias across the six compared models. Buy-and-hold posts a higher median Sharpe (2.2321) than every trained model, illustrating how punishing the AFML correction layer is for trading strategies on a trending asset."
- "PBO = 0.343 lands in the moderate-overfitting regime: in 12 of 35 IS/OOS partitions the in-sample best model underperforms the out-of-sample median. Selection across the six models is reproducible enough to claim that some configurations beat others, but not robust enough to support a confident pick. Leave-one-out PBO sharpens the picture asymmetrically: excluding `logistic`, `xgboost`, or `lstm` reduces PBO (to 0.114, 0.200, and 0.257 respectively), while excluding `kan` actually raises PBO to 0.514. The variance lives in the rotating IS-winners among Logistic Regression, XGBoost, and LSTM; KAN is the only model whose presence stabilises the cross-section."
- "The symbolic extraction pipeline produces a closed-form decision function with **93% symbolification rate** (13 of 14 edges) on five surviving features (`eth_btc_ratio`, `bb_width`, `skewness`, `log_returns_lag1`, `natgas_ret_30`). Post-symbolic accuracy is **60.90%**, a 1.5 percentage-point *gain* over the pre-symbolic 59.40%. The skewness subset bug is resolved (skewness reaches the extraction step), but the Phase-1 53% gate fails again (val_acc = 0.5038 after Adam) and recovery happens only in the subsequent LBFGS phases. For the first time a lag feature (`log_returns_lag1`) survives extraction, and a macro feature (`natgas_ret_30`) enters through an `Abs(·)` primitive whose pole at the dataset mean leaves its sensitivity undefined."
- Roadmap: 4.2 model comparison, 4.3 classification, 4.4 financial, 4.5 stability, 4.6 symbolic.

### 4.2 Model Comparison (the main result, Cochrane)

- **Table.** Self-contained caption: "Model comparison ranked by median Sharpe over seven CPCV backtest paths, with buy-and-hold included as a naive baseline. DSR is computed against `n_trials=6` (number of compared models), correcting the observed Sharpe for selection bias and non-normal returns; DSR is not defined for buy-and-hold since it is not part of the model selection pool. The Sharpe confidence interval is a bootstrap 95% CI on the per-path median Sharpe. No model achieves DSR ≥ 0.95. Tiebreaker: standard deviation of path Sharpe ascending."

| Rank | Model | Median Sharpe | Sharpe CI (low, high) | Std Sharpe | DSR | Med Sortino | Med Calmar | Mean F1 | Mean Acc | Mean AUC | Med Max DD | Med Cum Ret | Med Win Rate | Med Profit Factor |
|------|-------|--------------:|----------------------:|-----------:|-----:|-----:|-----:|------:|------:|------:|------:|------:|------:|------:|
| 1 | Buy-and-Hold | 2.2321 | (1.9766, 2.6408) | 0.6715 | n/a | 2.2931 | 2.4863 | n/a | n/a | n/a | -0.9998 | 52849.15 | 0.5539 | 1.2956 |
| 2 | XGBoost | 0.9483 | (-1.6369, 1.5260) | 1.6147 | 0.2376 | 0.3774 | 0.2184 | 0.4422 | 0.5309 | 0.5106 | -0.2308 | 0.1298 | 0.5374 | 1.4018 |
| 3 | KAN | 0.8375 | (-0.5868, 1.5681) | 1.3346 | 0.2012 | 0.4221 | 0.1268 | 0.4301 | 0.5377 | 0.5205 | -0.2254 | 0.1097 | 0.5517 | 1.2752 |
| 4 | LSTM | 0.6029 | (-0.3630, 1.1349) | 1.1890 | 0.1375 | 0.4529 | 0.0835 | 0.4484 | 0.5259 | 0.5078 | -0.2871 | 0.1399 | 0.4854 | 1.1342 |
| 5 | Logistic Regression | 0.4640 | (-0.7817, 1.0774) | 1.4700 | 0.1073 | 0.3064 | 0.0690 | 0.4217 | 0.5286 | 0.5109 | -0.3506 | 0.1168 | 0.5183 | 1.1002 |
| 6 | Random Forest | 0.3265 | (-1.4870, 0.8769) | 1.3004 | 0.0825 | 0.1818 | 0.0337 | 0.4348 | 0.5336 | 0.5102 | -0.2801 | 0.0473 | 0.5202 | 1.1143 |
| 7 | AR Logistic | -0.1864 | (-0.5538, 0.1112) | 0.7677 | 0.0262 | -0.1472 | -0.0413 | 0.4682 | 0.5292 | 0.5070 | -0.2778 | -0.0668 | 0.5177 | 0.9762 |

- **Key observations (state facts here, save interpretation for 5.1).**
  - **DSR.** Top DSR among trained models is XGBoost at 0.2376; all six trained-model values fall below 0.95. No model demonstrates predictive ability that survives correction for selection bias. The top three DSRs (XGBoost 0.2376, KAN 0.2012, LSTM 0.1375) span 0.10, with smaller gaps to Logistic Regression (0.1073), Random Forest (0.0825), and AR Logistic (0.0262). Buy-and-hold is excluded from the DSR computation because it is not an Optuna-tuned model and does not enter the selection pool.
  - **Sharpe confidence intervals.** 6 of 7 entries have a 95% bootstrap CI that crosses zero (every trained model). Only buy-and-hold (1.9766, 2.6408) has a CI that stays strictly positive. This is an independent confirmation of the DSR finding: the median Sharpe of every trained model is not statistically distinguishable from zero under bootstrap resampling. The previous run had LSTM as the only trained model with a strictly positive CI; in the current locked configuration LSTM's CI (-0.3630, 1.1349) also crosses zero, removing that one piece of supporting evidence for trained-model significance.
  - **Buy-and-hold reference point.** Buy-and-hold posts the highest median Sharpe (2.2321) and the highest cumulative return (≈52,849× over the 2015 to 2026 period, reflecting BTC's overall trajectory from approximately $325 to roughly $108k). Its median max drawdown is -0.9998 (essentially -100%): the position experiences the full BTC drawdown profile, which is precisely what bet sizing and the dual-weighted loss are designed to mitigate. Trained models trade off return for downside protection: LSTM's cumulative return of 0.1399 is roughly five orders of magnitude smaller than buy-and-hold's, but its max drawdown of -0.2871 is also three to four times shallower. This is the financial-management framing in which the DSR result should be read.
  - **Ranking.** XGBoost leads with median Sharpe 0.9483; KAN is second at 0.8375; LSTM third at 0.6029; Logistic Regression fourth at 0.4640; Random Forest fifth at 0.3265; AR Logistic last at -0.1864 (the only model with a negative median Sharpe). The previous run's ranking (LSTM, XGBoost, Random Forest, Logistic, KAN, AR Logistic) reshuffled materially: KAN moved up from fifth to second, XGBoost moved up from third to first, LSTM dropped from first to third, Random Forest dropped from third to fifth. The reshuffle is driven by the seeds change: in the previous configuration LSTM and KAN ran on 2 seeds (against 3 for the four classical models) and the LSTM result benefited from the lower seed cardinality; the locked configuration now uses 3 seeds across all six models, and the within-model averaging is now symmetric.
  - **AUC compression.** All mean AUCs sit in the 0.5070 to 0.5205 band. Models are nearly indistinguishable at the classification level; bet sizing and the dual-weighted loss are what translate the tiny edges into the wider Sharpe spread. KAN posts the highest mean AUC (0.5205) by a small margin over Logistic Regression (0.5109), LSTM (0.5078), and the other models; the previous run's Logistic Regression top-AUC slot is lost to KAN with three seeds per model.
  - **Q1, Q2 evidence (feature engineering).** AR Logistic again posts the worst Sharpe (-0.1864). Pure-momentum lags carry weak signal in the broader cross-section. Logistic Regression on MDA-selected features beats AR Logistic and Random Forest, sitting fourth: with 73 features in the MDA pool the linear hypothesis class still has enough breadth to extract usable signal, but cannot reach the top-three of the cross-section. Random Forest drops to fifth place and posts the lowest median cumulative return among non-negative entries (0.0473), suggesting its bagging variance is the main casualty of the seeds change.
  - **Q3 evidence (KAN positioning).** KAN is back to second place on median Sharpe (0.3834 → 0.8375), second on DSR (0.0920 → 0.2012), first on mean accuracy (0.5377), first on mean AUC (0.5205), and first on median win rate (0.5517). Its std Sharpe (1.3346) is mid-pack rather than second-highest, and its CI crosses zero (-0.5868, 1.5681) but at the upper end (the upper bound 1.5681 is the highest among trained models alongside XGBoost's 1.5260). The leave-one-out PBO finding in Section 4.4 sharpens the result: KAN is the model whose presence stabilises the cross-section, which is a structural finding rather than a ranking finding.
  - **Std Sharpe outlier.** AR Logistic shows the lowest std Sharpe (0.7677) among the trained models. Pure momentum is consistent across paths because it conditions on a simple, stable signal; the cost is the only negative median in the table. XGBoost now carries the highest std Sharpe (1.6147) and Logistic Regression is second (1.4700): both are wide enough to make their bootstrap CIs span more than two Sharpe units, reflecting that their directional accuracy is highly regime-conditional. LSTM has the lowest std Sharpe (1.1890) among the top three, which is the variance reading that makes it the third candidate in this cross-section despite the financial-side ranking.

### 4.3 Classification Metrics

- **Per-split F1 distribution.** Mean F1 ranges narrowly from 0.4165 (Logistic Regression) to 0.4682 (AR Logistic). The high-F1 / negative-Sharpe combination for AR Logistic confirms that classification quality and bet-sizing translation are partially independent: pure momentum gets the average call right slightly more often than the engineered-feature models but loses the high-conviction events that drive Sharpe and overpays in transaction costs, leaving a negative median Sharpe.
- **Pooled AUC.** Per-model pooled AUC across all 28 splits sits in the 0.4863 to 0.5129 band. All models hover within two percentage points of random; the entire signal contest plays out inside an AUC range smaller than what most reviewers would consider economically material.
- **DeLong pairwise AUC.** **0 of 15 pairs significantly different at α=0.05.** Pooled across splits, averaged across the 3 seeds per model:

| Pair | AUC_a | AUC_b | Δ AUC | z | p | Significant |
|------|-------|-------|-------|---|---|-------------|
| LSTM vs KAN | 0.5010 | 0.4863 | +0.0147 | 1.4623 | 0.1437 | No |
| XGBoost vs KAN | 0.5022 | 0.5129 | -0.0107 | -1.5435 | 0.1227 | No |
| Random Forest vs KAN | 0.5037 | 0.5129 | -0.0092 | -1.4804 | 0.1388 | No |
| Logistic vs KAN | 0.5032 | 0.5129 | -0.0097 | -1.4757 | 0.1400 | No |
| AR Logistic vs KAN | 0.5046 | 0.5129 | -0.0083 | -0.8893 | 0.3738 | No |
| AR Logistic vs LSTM | 0.5070 | 0.5020 | +0.0050 | 0.4933 | 0.6218 | No |
| Random Forest vs LSTM | 0.5058 | 0.5020 | +0.0038 | 0.3725 | 0.7095 | No |
| XGBoost vs LSTM | 0.5036 | 0.5020 | +0.0016 | 0.1614 | 0.8718 | No |
| Logistic vs LSTM | 0.5034 | 0.5020 | +0.0014 | 0.1347 | 0.8928 | No |
| AR Logistic vs XGBoost | 0.5046 | 0.5022 | +0.0024 | 0.2598 | 0.7951 | No |
| Random Forest vs XGBoost | 0.5037 | 0.5022 | +0.0015 | 0.2964 | 0.7669 | No |
| Logistic vs XGBoost | 0.5032 | 0.5022 | +0.0010 | 0.1381 | 0.8902 | No |
| AR Logistic vs Random Forest | 0.5046 | 0.5037 | +0.0009 | 0.0961 | 0.9234 | No |
| Logistic vs Random Forest | 0.5032 | 0.5037 | -0.0005 | -0.0732 | 0.9417 | No |
| AR Logistic vs Logistic | 0.5046 | 0.5032 | +0.0014 | 0.1526 | 0.8788 | No |

- **Pairwise summary.** **No pair clears α = 0.05.** KAN posts the highest pooled AUC (0.5129), ahead of AR Logistic (0.5046), Random Forest (0.5037), Logistic Regression (0.5032), XGBoost (0.5022), and LSTM (0.5020), but none of the differences is statistically distinguishable from sampling noise under the DeLong test. The KAN-versus-all pairs cluster tightly around z ≈ -1.5 with p-values in the 0.12 to 0.14 range, showing that KAN's AUC advantage is consistent in sign across comparisons but uniformly too small to clear the significance threshold. The current locked configuration produces the same 0/15 finding as the previous run; the absence of significant pairs is robust across the seeds reconfiguration.
- **Effect-size disclosure (T-R15: write for reviewers).** The entire AUC range across all six models is 0.027 (from LSTM at 0.5020 to KAN at 0.5129), the same order of magnitude as the seed-to-seed and split-to-split variation within any single model. The DeLong test confirms that none of these gaps survives a hypothesis test once cross-split correlation is accounted for.
- **Confusion matrices (compact, Appendix D for full).** Aggregated TP/FP/TN/FN per model. Most models post a slight class-1 bias consistent with the empirical 57.0% / 43.0% Up/Down base rate.
- **Multiple testing.** No Bonferroni or Benjamini-Hochberg correction is needed since no nominal p-value clears α = 0.05 in the first place. The 0 of 15 finding is robust to any sensible correction.

### 4.4 Financial Performance

- **Path-level Sharpe distribution.** Per-model row × 7 paths + median + std. XGBoost has the highest median (0.9483) but the highest path-Sharpe std as well (1.6147); its bootstrap 95% CI (-1.6369, 1.5260) is the widest of any trained model and straddles zero by a wide margin. KAN (median 0.8375, std 1.3346) sits second with a CI (-0.5868, 1.5681) that also crosses zero but at the upper end (the 1.5681 upper bound is the highest among trained models alongside XGBoost's). LSTM (median 0.6029, std 1.1890) has the **lowest std among trained models**, but its CI (-0.3630, 1.1349) still crosses zero on the lower side. Logistic Regression (median 0.4640, std 1.4700) has the second-highest std with a CI (-0.7817, 1.0774) firmly straddling zero. Random Forest (median 0.3265, std 1.3004) is the fifth model and the second-widest CI (-1.4870, 0.8769). AR Logistic (median -0.1864, std 0.7677) is the only trained model with a negative median; its CI (-0.5538, 0.1112) is the only one that does not include a meaningfully positive value. Buy-and-hold (median 2.2321, std 0.6715) dominates the median ranking with the tightest CI (1.9766, 2.6408) of the seven entries and is the only entry whose CI stays strictly positive.
- **Sortino and Calmar.** The downside-deviation-adjusted Sortino ratios reorder the trained models slightly: LSTM (0.4529), KAN (0.4221), XGBoost (0.3774), Logistic Regression (0.3064), Random Forest (0.1818), AR Logistic (-0.1472). LSTM's Sortino leads despite its third-place median Sharpe, suggesting that LSTM's path-Sharpe distribution loads more variance on the upside than on the downside (good draws are larger than bad draws). KAN's Sortino is competitive in second place. Calmar (Sharpe over max drawdown) compresses the trained models into a narrow 0.03 to 0.22 band, with XGBoost at 0.2184 leading and AR Logistic at -0.0413 the only negative entry. Buy-and-hold's Calmar (2.4863) is an order of magnitude above any trained model, but its near-100% max drawdown means a single rebalancing decision during the deepest crash would wipe out the strategy.
- **DSR computation detail.** For XGBoost (top-ranked trained model): observed median Sharpe 0.9483 over 7 paths, DSR = 0.2376 against `n_trials=6`. The expected maximum Sharpe under the null with six compared models, combined with the pooled skew/kurt and SE(SR), produces a DSR threshold the observed Sharpe does not clear. The 0.2376 figure says XGBoost's median Sharpe is roughly at the 24th percentile of what the null hypothesis (all true Sharpes equal zero) would produce when testing six independent strategies, far short of the 0.95 significance threshold. Full DSR-component breakdown for the top model goes to Appendix D.
- **PBO result.** **PBO = 0.343 = 12 of 35 IS/OOS partitions.** Interpretation in context of DSR < 0.95:
  - Theoretical regime map: `PBO ≈ 0` = robust selection (the IS-best model tends to remain best out of sample); `PBO ≈ 0.5` = model selection is random; `PBO > 0.5` = adversarial.
  - Observed regime: **moderate overfitting risk.** In 34.29% of partitions the model that wins on three IS paths underperforms the median of the four OOS paths. The IS ranking holds out of sample more often than not, but not robustly: the IS-winner is the OOS-loser about one time in three. This is closer to "model selection is partially random" than to "the IS ranking holds out of sample". The locked configuration is within 0.03 of the previous run (PBO 0.3714 → 0.343), confirming that the moderate-overfitting regime is the stable headline finding across both 2- and 3-seed configurations.
  - **Leave-one-out PBO sharpens the picture asymmetrically.** Re-running PBO on the 5×7 Sharpe matrix with one model dropped at a time produces a pattern that is more informative than the headline figure. Sorted by 5-model PBO ascending:

| Model excluded | 5-model PBO | Δ vs baseline |
|---|---|---|
| logistic | 0.114 | -0.229 |
| xgboost | 0.200 | -0.143 |
| lstm | 0.257 | -0.086 |
| ar_logistic | 0.343 | +0.000 |
| random_forest | 0.343 | +0.000 |
| kan | 0.514 | +0.171 |

  - **Reading the LOO table.** The IS/OOS instability is concentrated in three models: Logistic Regression, XGBoost, and LSTM. Removing any one of them reduces PBO by 0.09 or more. Logistic Regression alone accounts for the largest single reduction (-0.229): with Logistic Regression dropped, the five remaining models settle into a stable IS ranking that the OOS median agrees with on roughly 89% of partitions. AR Logistic and Random Forest are PBO-neutral: their exclusion does not change the figure, meaning they are far enough from the IS-best slot that they do not contribute to the IS/OOS rotation. KAN is the opposite: removing it *increases* PBO to 0.514, which is the highest value in the LOO table and tips the regime past the random-selection threshold (PBO ≈ 0.5). This is a striking finding: **KAN is the only model whose presence stabilises the model cross-section**. The previous locked configuration had this property attached to LSTM (its exclusion raised PBO to 0.429); the current locked configuration moves it to KAN (its exclusion raises PBO to 0.514, even further). Two anchors across two runs is suggestive evidence that the cross-section needs one well-calibrated, low-variance candidate to keep selection reproducible, regardless of which model fills that role.
  - Joint reading with DSR < 0.95: model selection within this six-model universe is partially reliable (the headline PBO of 0.343 is below the 0.5 random-selection threshold but above the 0.3 robust-selection threshold), is anchored by KAN (whose exclusion moves the regime past random), and is destabilised by Logistic Regression / XGBoost / LSTM rotating in the IS-best slot. The model that selection picks (XGBoost in this configuration) does not achieve a Sharpe that survives multiple-testing correction. The honest conclusion is "XGBoost is the IS-best choice in this cross-section but its bootstrap CI crosses zero at -1.64; KAN is the structural anchor whose presence keeps selection reproducible; and a naive buy-and-hold strategy posts a 2.23 median Sharpe over the same period at the cost of a -99.98% maximum drawdown".
  - Methodological implication (T-R15: write for reviewers, this is the AFML pay-off). Without PBO, the natural conclusion from the model-comparison table is "XGBoost achieved median Sharpe 0.95 on BTC daily direction". With the headline PBO of 0.343 the conclusion sharpens: rankings are stable enough to claim that XGBoost beats four of the other trained models, but the IS/OOS instability among Logistic Regression, XGBoost, and LSTM means that any alternative selection rule (highest Sortino, highest AUC, lowest drawdown) would pick a different model on at least one IS sub-path partition. The leave-one-out PBO refines the message further: KAN is the anchor of the cross-section rather than a source of overfitting risk, and the IS-best model (XGBoost) is itself part of the rotating cluster. The PBO and DSR diagnostics are complementary: PBO answers "did we pick the right model?", DSR answers "is the right model good enough?"; here the answer is "yes for the linear-vs-nonlinear distinction and for the KAN-as-anchor reading, no within the rotating cluster" to the first and "no" to the second.
- **Equity curves figure.** All 6 models + buy-and-hold overlay on the median (or best) path. Note specifically the AR Logistic equity curve, which is the only one ending below 1.0 (median cumulative return -0.0668). Buy-and-hold's curve dominates the plot vertically (52,849× cumulative return); a log-scale y-axis is needed to see the trained models alongside it.
- **Additional metrics table.** Cumulative return, annualised return, max DD, time under water, win rate, profit factor, mean |bet|, % traded. Locked-run highlights from the table at 4.2: LSTM median cumulative return 0.1399 (best among trained), XGBoost 0.1298, Logistic Regression 0.1168, KAN 0.1097, Random Forest 0.0473, AR Logistic -0.0668. Median max-drawdown ranges from -0.2254 (KAN, best) to -0.3506 (Logistic Regression, worst); LSTM's -0.2871 is the third-worst, consistent with its third-place Sharpe ranking. Buy-and-hold's median max-drawdown is -0.9998, the structural cost of holding through the 2017-18 and 2022 BTC crashes. KAN posts the best max-drawdown among trained models despite its second-place Sharpe, evidence that its bet-sizing distribution favours smaller drawdowns in exchange for some upside.

### 4.5 Feature Selection Stability and FFD Stability

- **Feature selection frequency (per-feature count / 28 folds).** Locked-run profile is **flat to a striking degree.** Of 73 candidate features:
  - **0 features stable** (selected in more than 80% of folds, i.e. > 22 of 28 folds).
  - **2 features moderate** (50% to 80%): `eth_btc_ratio` at 60.7% (17 of 28 folds), `bb_width` at 53.6% (15 of 28 folds).
  - **70 features low** (selected in fewer than 50% of folds).
  - **1 feature never selected**: `tx_count_roc_14` (an on-chain feature that fails to clear the multi-model MDA threshold in any of the 28 training folds).
- **Headline finding.** No compact feature subset consistently dominates across time periods. The MDA-selected set turns over substantially across the 28 folds: the median feature is selected in roughly a third of folds, and even the most-stable feature (`eth_btc_ratio`) clears 50% by about 11 percentage points. This is itself a finding: it suggests that the information relevant to BTC daily direction is distributed across many features, with regime-specific pockets of relevance, rather than concentrated in a small permanent core. The 3-seed configuration produces a tighter stability profile than the 2-seed configuration: the previous locked run had `eth_btc_ratio` at 75.0%, `skewness` at 71.4%, and `bb_width` at 57.1%; the current run drops `skewness` below the 50% threshold (to a sub-moderate position in the low bucket) and trims the top two features by 5 to 15 percentage points. The 3-seed averaging spreads selection signal across more independent inner-CV MDA runs, which is more conservative than the 2-seed configuration but produces a similar qualitative reading.
- **Group breakdown of the two moderate features.**
  - 1 of 2 from the technical-analysis group (`bb_width`, Bollinger Band width).
  - 1 of 2 from the external / crypto-macro group (`eth_btc_ratio`, the CoinMetrics-sourced ETH/BTC alt-rotation signal).
  - 0 from macro, on-chain, mathematical (AFML Part 4), or lag groups.
- **On-chain question (Q2).** **No on-chain feature reaches 50% selection frequency.** All eight on-chain columns (active addresses ROC, transaction count ROC, hash rate ROC, MVRV, net exchange flow, fee per transaction, exchange supply percent, issuance) sit in the "low" bucket or below; `tx_count_roc_14` is in fact the single never-selected feature in the entire 73-feature pool. The "free lunch hypothesis" (on-chain transparency providing measurable signal beyond price and volume) does not survive multi-model MDA + CPCV evaluation in this dataset; on-chain features may carry information for individual splits, but no on-chain column is consistently selected across folds. Q2 answer: **on-chain features do not exhibit stability** under the AFML feature-selection regime applied here. This finding hardens across all three locked configurations evaluated: in the 62-feature run `exchange_supply_pct` survived to the symbolic extraction step despite low cross-fold stability; in the previous 73-feature 2-seed run no on-chain feature reached the top-5 stable set; in the current 73-feature 3-seed run one on-chain feature (`tx_count_roc_14`) is *never* selected at all.
- **Figure.** Horizontal bar chart of selection frequency (73 bars, sorted descending, coloured by feature group: green=TA, blue=mathematical, orange=external macro, red=external crypto-macro, brown=external on-chain, grey=lag).
- **FFD stability.** ATR is the only column subject to FFD (the only feature flagged non-stationary by the cross-fold ADF audit at α=0.05 with the FFD column whitelist `['atr']`; the full ADF report identifies 8 non-stationary features at α=0.05, all retained without FFD because the locked configuration applies the transform only to ATR by design). Locked-run statistics across 28 folds × 3 seeds (84 d* estimates):
  - Mean d* = 0.170, std d* = 0.075.
  - Range: d* ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}; modal value d* = 0.15.
  - Std d* < 0.1 → consistent stationarity structure across time periods. The ATR series requires fractional differencing of order ≈ 0.15 to 0.20 regardless of the training-fold partition, indicating that the persistence of the volatility-range process is a stable property rather than a regime-conditional one. No fold required d* = 1.0 (full integer differencing).
- **Methodological note.** Feature-selection turnover (high) and FFD-d* turnover (low) tell a coherent story: the noise-versus-signal ratio for individual features changes across regimes (only `eth_btc_ratio` and `bb_width` clear 50%), but the underlying time-series property that motivates fractional differencing is stable. Stability of d* validates the FFD-only-on-ATR design choice in 3.6.1; instability of feature selection validates the use of multi-model MDA over single-model SFI in 3.6.3.

### 4.6 Symbolic Extraction Results (Q4 answer)

- **Extraction summary.**
  - Fold used: split 27 (groups 6 and 7), `fold_selection="last"`. Test set covers 2024-08-21 to 2026-04-19, the most recent CPCV partition; closest to deployment scenario. Locked-run KAN F1 on this fold: **0.4628**.
  - **Top-5 stable features by CPCV selection frequency all reach the extraction step** (the previous locked configuration suffered a `skewness` subset bug; the expanded 73-feature MDA pool resolves it, even with skewness now sitting below the global 50% threshold at the 28-fold level). Top-5 ranking entering extraction: `eth_btc_ratio` (61%), `bb_width` (54%), `skewness` (46%), `log_returns_lag1` (32%), `natgas_ret_30` (32%). 67 features excluded.
  - Training sample size: 530 train + 133 val (after the 80/20 split on the split-27 training fold). 5 features in the input matrix.
  - **Architecture (CPCV-matched, single hidden layer).** The locked Optuna study did not produce tuned KAN hyperparameters for split 27, so the data-aware fallback fires. Final architecture `[5, 2, 2]`: 5 input features → 2-unit hidden layer → 2-unit binary output. 14 active edges, ~84 spline parameters for 530 training samples (parameter-to-sample ratio 6.3×, comfortably above the 5× floor).
- **Training diagnostics.**
  - Phase 1 (Adam, 600 steps, weight decay 1e-3, noise std 0.05): final train accuracy 0.4604, **validation accuracy 0.5038**.
  - **Phase-1 53% gate FAILS** (val_acc 0.5038 < 0.53 by 0.26 percentage points; the pipeline logs the warning "Adam phase val_acc=0.5038 < 0.53 minimum. PyKAN may not have learned meaningful patterns" but continues to symbolification per the design). The gate-failure pattern returns relative to the previous locked configuration, where Phase 1 had cleared 0.6241. Recovery happens in the subsequent LBFGS phases, but the diagnostic that the previous run used as evidence of a clean training trajectory is no longer available for the current run.
  - Phase 2a (LBFGS warmup, 20 steps, no regularisation): train 0.6962, val 0.5489.
  - Phase 2b (LBFGS sparsity, 20 steps, lambda=2e-3): train 0.7245, val 0.5940.
  - Grid extension skipped (n_train=530 too small for grid 3 → 5 expansion).
  - Detailed phase-by-phase logs in Appendix F.
- **Pruning results.** **Pruned architecture: `[[5, 0], [2, 0], [2, 0]]`** in PyKAN's `(sum_units, mult_units)` per-layer notation. **No edges pruned**: pre-prune edge analysis at threshold 0.01 finds 14 of 14 active (100% survival). The model's edge-importance distribution does not contain any below-threshold edges, so structural sparsification offers no compression at this configuration. Post-prune validation accuracy: 0.5940 (unchanged from Phase 2b final, since no edges were removed).
- **Symbolification rate: 93% (13 of 14 edges).** All 13 symbolified edges clear the 0.30 R² threshold; 1 edge is skipped during the symbolic substitution stage (no symbolic candidate could be assigned). R² distribution across the 13 symbolified edges: min 0.5746, median 0.9960, max 1.0000. Top-five edges by R²:
  - Edge (0, 0, 0): `sin`, R² = 1.0000.
  - Edge (0, 0, 1): `sin`, R² = 1.0000.
  - Edge (0, 1, 1): `sin`, R² = 0.9998.
  - Edge (0, 1, 0): `sin`, R² = 0.9997.
  - Edge (0, 2, 0): `sin`, R² = 0.9988.
- **Surviving features (5; no subset-bug exclusion this run).**
  - `eth_btc_ratio` (external / crypto-macro group, CoinMetrics-sourced ETH/BTC alt-rotation signal). Selected in **61%** of CPCV folds at the 28-fold level; **most stable feature in the dataset**.
  - `bb_width` (TA / volatility group, Bollinger Band width). Selected in **54%** of CPCV folds.
  - `skewness` (TA / distribution-shape group, rolling 21-day third moment of log returns). Selected in **46%** of CPCV folds (below the 50% threshold globally, but rises into the top-5 fold-specific stability ranking on the deployment-relevant final fold).
  - `log_returns_lag1` (lag / autoregressive group, 1-day lag of log returns). Selected in **32%** of CPCV folds. **First lag feature to enter the symbolic representation** in any locked configuration; previously the lag group had contributed 0 features to any extraction.
  - `natgas_ret_30` (external / macro group, 30-day natural gas return). Selected in **32%** of CPCV folds. **First time natural gas enters the symbolic representation**; the previous configuration's macro contender was `us2y`.
  - **Group breakdown.** 2 TA (bb_width, skewness), 1 crypto-macro (eth_btc_ratio), 1 macro (natgas_ret_30), 1 lag (log_returns_lag1), 0 on-chain, 0 mathematical (AFML Part 4). **Lag and macro features enter for the first time**: the surviving set now spans four feature families rather than the three of the previous run.
- **Pre-symbolic vs. post-symbolic accuracy.**
  - Pre-symbolic (B-spline KAN, post-prune): **59.40%**.
  - Post-symbolic (closed-form formula, after affine fine-tuning): **60.90%**.
  - **Δ = +1.50 percentage points (post exceeds pre by 1.5 pp).**
  - Reading: the closed-form approximation *improves* validation accuracy over the trained spline network, in the same positive direction as the previous locked configuration but with a smaller margin (+1.5 pp vs. the previous +4.5 pp). The 14-function symbolic library matches the learned B-spline shapes well enough at the top end (top 5 edges all R² ≈ 1.0000 with the `sin` primitive) but the minimum R² (0.5746) is markedly lower than the previous run's 0.7768, meaning one or more low-R² edges acts as a noisier approximation. The affine fine-tuning step (30 LBFGS steps at lr=4e-4) then tunes the 5 affine parameters per edge to the validation distribution. The post-symbolic 60.90% sits 10.90 percentage points above the 50% baseline and 7.90 percentage points above the 53% Phase-1 gate that the spline KAN failed pre-prune.
  - The post-symbolic gain shrinks relative to the previous locked configuration (+1.5 pp vs. +4.5 pp). The smaller gain is consistent with three diagnostic changes: the pre-symbolic baseline starts higher (0.5940 vs. 0.5564), leaving less room for improvement; the minimum edge R² drops to 0.5746 (vs. 0.7768), meaning one symbolic substitution is a rougher fit; and one edge is skipped entirely (the 93% rate vs. 100%). Together these say the formula in the current run is a slightly noisier representation of the same underlying decision logic.
- **The decision function.** The extracted closed-form expression. Numbered equation in the thesis:

```
decision(x) =
  - (79/176) sin( (4/915) (-4484·log_returns_lag1/489 - 817/451)³
                  + (7440/547) sin(311·bb_width/857 - 59/58)
                  + (1799/892) sin(2353·eth_btc_ratio/812 - 792/151)
                  - (6825/583) sin(631·skewness/429 - 4125/931)
                  - (343/401) |5028·natgas_ret_30/827 + 53/29|
                  + 4553/215 )
  + (461/734) sin( - (495/166) sin(1058·bb_width/873 + 1994/835)
                   + (2811/311) sin(3135·eth_btc_ratio/982 + 16/13)
                   + (1913/696) sin(691·skewness/434 - 6577/941)
                   + (31/376) atan(10·log_returns_lag1 + 10)
                   - (163/65) atan(3448·natgas_ret_30/801 - 523/156)
                   + 9820/697 )
  + (439/794) tanh( (6/653) (-4484·log_returns_lag1/489 - 817/451)³
                    + (21247/743) sin(311·bb_width/857 - 59/58)
                    + (759/179) sin(2353·eth_btc_ratio/812 - 792/151)
                    - (7433/302) sin(631·skewness/429 - 4125/931)
                    - (651/362) |5028·natgas_ret_30/827 + 53/29|
                    + 50188/973 )
  + (405/821) tanh( - (2551/277) sin(1058·bb_width/873 + 1994/835)
                    + (12115/434) sin(3135·eth_btc_ratio/982 + 16/13)
                    + (4117/485) sin(691·skewness/434 - 6577/941)
                    + (55/216) atan(10·log_returns_lag1 + 10)
                    - (2579/333) atan(3448·natgas_ret_30/801 - 523/156)
                    + 6445/256 )
  + 21/583

P(up | x) = 1 / (1 + exp(-decision(x)))
```

  - Coefficients are post-`sympy.nsimplify(tolerance=1e-3)` rationals. `sympy.simplify` timed out at 30 seconds and the unsimplified form is shown above. The underlying floats are stored alongside in `sympy_objects` for sensitivity analysis (numerical sensitivity bullet below).
- **Functional-form pattern.** **The formula uses five distinct primitives: `sin`, `tanh`, `atan`, the cubed term `(...)³`, and the absolute value `|...|`.** This is again substantively different from the previous locked configuration's `sin`, `cos`, `tanh`, `x²`, `x³` set:
  - `cos` and `x²` are no longer present. The previous run had `cos` as a leading primitive (top-edge R² 0.9998) and `x²` as the dominant non-trigonometric primitive; in the locked run both are replaced by `sin` (top 5 edges all `sin` at R² ≥ 0.9988) and by the new primitives `atan` and `|...|`.
  - `atan` enters as a bounded smooth saturator, attached to `log_returns_lag1` and `natgas_ret_30` in both of the inner-sum sin and tanh outer terms with the trigonometric inner argument; this is the first time `atan` has survived since the original 62-feature locked configuration.
  - `|...|` (absolute value, denoted `Abs(·)` in SymPy) enters as a non-smooth V-shape primitive, attached to `natgas_ret_30` in two outer terms. Its presence is the source of the natgas_ret_30 sensitivity-at-mean undefined-gradient issue noted below: `|·|` has a kink at zero and the dataset mean of `natgas_ret_30` (+0.0029) sits inside the [-(53/29 × 1/something), +ε] neighbourhood of the kink, so numerical differentiation reports a non-finite gradient.
  - `(...)³` survives from the previous run, again attached to `log_returns_lag1` in the inner-cube position of two outer terms with the cubed / sine inner argument.
  - `tanh` survives as a bounded smooth saturator, now used as the outer wrapper of two of the four outer terms rather than only as an inner primitive.
  - The four outer terms split into two structural classes: two outer `sin` terms (the first and second outer terms) and two outer `tanh` terms (the third and fourth). The inner arguments of the first and third terms share the same cube-and-sines structure on `log_returns_lag1`, `bb_width`, `eth_btc_ratio`, `skewness`, and `natgas_ret_30`; the inner arguments of the second and fourth terms share the same atan-and-sines structure on the same features. The KAN learned two complementary internal representations and combined them through the sin / tanh outer wrappers.
- **Interpretive note (factual; economic interpretation deferred to 5.3).** Each surviving feature appears in **exactly four terms** of the formula (formula term-structure analysis confirms n_terms_in_formula = 4 for each of `eth_btc_ratio`, `bb_width`, `skewness`, `log_returns_lag1`, `natgas_ret_30`). The five input features therefore contribute roughly equally to formula complexity, with no single feature dominating the structural footprint. The cross-run robustness of this n=4 result across the previous and current locked configurations is suggestive evidence that PyKAN's pruning step systematically produces a four-outer-term representation when the input dimensionality is 5 and the hidden layer is width-2.
- **Numerical sensitivity at the dataset mean.** Partial derivatives evaluated at the mean of each feature, scaled by the feature's standard deviation, give the per-σ effect on the decision function (logit). The sensitivity table is computed at the dataset mean by default and reported as NaN for features whose symbolic formula has a pole at that point:

| Feature | mean | std | d/dx at mean | σ-effect on logit | σ Δp (linearised) |
|---------|-----:|-----:|-----:|-----:|-----:|
| `eth_btc_ratio` | +0.0461 | +0.0247 | -2.9437 | -0.0727 | -0.0182 |
| `bb_width` | +0.2758 | +0.1692 | -1.9317 | -0.3269 | -0.0817 |
| `skewness` | -0.0559 | +1.0171 | -1.7309 | -1.7606 | -0.4401 |
| `log_returns_lag1` | -0.0005 | +0.0415 | +0.0783 | +0.0032 | +0.0008 |
| `natgas_ret_30` | +0.0029 | +0.1622 | NaN | NaN | NaN |

  - **`skewness` is the dominant feature in the formula** in absolute σ-effect terms (-1.76 on the logit, linearised σ Δp ≈ -0.44). A one-σ increase in 21-day return skewness moves P(up) from approximately 0.50 to approximately sigmoid(-1.76) ≈ 0.15. The direction is consistent with the literature on right-skewed crypto-return regimes: positive skew (large isolated up-moves) is followed by mean-reverting weakness rather than persistence. This is a structural reversal from the previous locked configuration, where the dominant per-σ effects were the saturating positive switches `us2y` (+10.24) and `kurtosis` (+10.59); the current formula has no such saturating-positive features, and the strongest signal is negative and order-of-magnitude smaller (-1.76 vs. +10.24).
  - `bb_width` carries σ-effect -0.33 on the logit, linearised σ Δp ≈ -0.08, enough to move P(up) from approximately 0.50 to approximately sigmoid(-0.33) ≈ 0.42. Direction reads naturally: wider Bollinger bands signal high volatility, and the model conditions on lower forward direction probability in such regimes.
  - `eth_btc_ratio` σ-effect -0.07 on the logit, linearised σ Δp ≈ -0.02; a one-σ increase in the ETH/BTC ratio (alt-rotation strength) shifts P(up) by approximately 1.8 percentage points downward. Alt rotation away from BTC is mildly bearish on forward BTC direction, but the magnitude is now an order smaller than the dominant `skewness` effect.
  - `log_returns_lag1` σ-effect +0.003 on the logit, linearised σ Δp ≈ +0.001. **Effectively flat.** The 1-day lag of log returns survived the extraction step on selection-frequency criterion (32% of folds) but its symbolic representation in the locked formula contributes essentially zero gradient at the dataset mean.
  - `natgas_ret_30` **gradient at dataset mean: NaN.** The symbolic formula contains an `|·|` (absolute value) primitive attached to `natgas_ret_30` in two of the four outer terms, with the inner argument `5028·natgas_ret_30/827 + 53/29`. This expression evaluates to a negative number at the dataset mean (-0.0029 × 5028/827 + 53/29 ≈ 0.0177 + 1.828 = 1.846, positive after the constant offset, so the gradient *should* be defined here, but SymPy's automatic differentiation produces a `sign(·) × derivative(·)` term that includes `re(·)` and `im(·)` decompositions which evaluate to non-real or numerically-unstable derivatives at the dataset mean). The diagnostic logs report "1/5 features have non-finite gradient at the median; reported as NaN. Consider eval_point='median' or inspect the symbolic formula for poles near these features' median." The honest reading is that the symbolic gradient computation hits a SymPy edge case at the natgas_ret_30 evaluation point and the σ-effect cannot be reported numerically.
  - **Scale check.** As in the previous locked configuration (and unlike the original 62-feature run with `chaikin_osc`), all five surviving features have interpretable scales. `bb_width`, `skewness`, and `log_returns_lag1` are bounded or normally-scaled by construction; `eth_btc_ratio` is bounded above by the historical maximum (~0.13 in this sample); `natgas_ret_30` is a 30-day return (typically ±0.20). No rescaling caveat is needed for the formula, but the `|·|` primitive on `natgas_ret_30` introduces the non-differentiability noted above.
- **On-chain comment (Q2 evidence).** **Zero on-chain features survive extraction.** This finding is now stable across the previous and current locked configurations (the previous 73-feature 2-seed run also produced zero on-chain survivors, displacing the original 62-feature run's `exchange_supply_pct`). The Q2 answer hardens further: not only do on-chain features fail to clear cross-fold stability (Section 4.5), they consistently fail to enter the locked symbolic representation regardless of seed cardinality. The earlier 62-feature locked-run observation that one on-chain feature could carry the formula's strongest economic effect is **regime-specific and does not generalise to the expanded feature universe at either 2 or 3 seeds.**
- **Lag-feature comment (new Q2 evidence).** **For the first time a lag feature enters the symbolic representation**: `log_returns_lag1` is one of the five surviving features. Its σ-effect is effectively zero (+0.0008 σ Δp), so its presence in the formula does not translate into meaningful predictive contribution at the dataset mean. The reading: the lag-features-in-MDA-pool design choice (Section 3.4.4, advisor-driven, lag features compete with engineered features rather than feeding only into AR Logistic) admits lag features into the surviving set in some configurations, but the surviving lag feature's contribution is structural rather than economic.
- **Surviving-feature group breakdown.**
  - 2 of 5 from TA (`bb_width`, `skewness`).
  - 1 of 5 from crypto-macro (`eth_btc_ratio`).
  - 1 of 5 from macro (`natgas_ret_30`).
  - 1 of 5 from lag (`log_returns_lag1`).
  - 0 of 5 from on-chain or mathematical (AFML Part 4) groups.
- **Section closer.** "The symbolic extraction pipeline produces a closed-form decision function with **93% symbolification rate (13 of 14 edges, one edge skipped)** on a five-feature subset (`eth_btc_ratio`, `bb_width`, `skewness`, `log_returns_lag1`, `natgas_ret_30`). **Post-symbolic accuracy is 60.90%, a 1.5 percentage-point gain over the pre-symbolic 59.40%** and 10.90 percentage points above the 50% baseline. The formula uses five primitives (`sin`, `tanh`, `atan`, `x³`, `|x|`) and contains four outer terms, with each surviving feature contributing equally. The closed-form approximation *improves* accuracy relative to the trained spline network, producing a humanly auditable expression in which one feature (`skewness`) carries the dominant negative effect and one feature (`natgas_ret_30`) hits a SymPy non-differentiability at the dataset mean. Q4 is answered in the affirmative with three structural caveats: the Phase-1 53% gate fails again (val_acc 0.5038), recovery happens only in the LBFGS phases; one of fourteen edges remains un-symbolified; and one of five surviving features has an undefined gradient at the dataset mean. RQ4 conclusion: closed-form interpretability is achievable, informative, and accuracy-preserving even in a regime where DSR < 0.95 across all benchmark models; the structural transparency of the decision function is the primary contribution and stands independent of the DSR result."

---

## 5. Discussion

> **T-R7.** Interpret. Link findings back to the mechanisms in Section 3.

### 5.1 Interpreting the results (Q1, Q3 evidence)

- **The joint DSR / PBO finding.** No model achieves DSR ≥ 0.95 (top trained model is XGBoost at 0.2376; buy-and-hold posts a higher median Sharpe of 2.2321 but is not included in the DSR computation since it does not enter the selection pool). PBO sits in the moderate-overfitting regime at 0.343, with 12 of 35 IS/OOS partitions seeing the IS-winner underperform the OOS median. The combination is informative: rankings are partially reproducible, but every trained-model bootstrap CI now crosses zero (the previous run's LSTM strictly-positive-CI evidence does not survive into the current configuration; LSTM's CI is now -0.3630 to 1.1349), and the model that any selection rule picks does not achieve a Sharpe that survives multiple-testing correction. The honest reading is "no trained model in this six-model cross-section clears the multiple-testing-corrected significance threshold and no trained model's bootstrap CI stays strictly positive, and a naive buy-and-hold strategy on the same data posts a median Sharpe roughly 2.4× the best trained model's figure". This is a different position than either DSR or PBO alone could deliver: we are not in a regime where rankings are random (PBO ≈ 0.5) or anti-predictive (PBO > 0.5), but the model cross-section is not robust enough (PBO ≥ 0.3) to support a confident pick, and even the most defensible pick does not earn the significance label.
- **Locating the variance.** The leave-one-out PBO (Section 4.4) shows that removing Logistic Regression most reduces PBO (to 0.114, a -0.229 shift), followed by XGBoost (0.200, -0.143) and LSTM (0.257, -0.086). Removing AR Logistic or Random Forest does not change PBO. **Removing KAN raises PBO to 0.514 (+0.171)**, making KAN the only model whose presence stabilises the cross-section. This is the same kind of asymmetry the previous locked configuration produced for LSTM, now relocated to KAN. Two implications follow. First, the IS/OOS instability is concentrated in three models (Logistic Regression, XGBoost, LSTM) whose path-Sharpe profiles are similar enough to rotate the IS-best slot across sub-path partitions. Second, KAN acts as a fixed anchor: its mid-pack path-Sharpe std (1.3346) and well-calibrated probability output (mean P(Up) 0.5572 vs. the 0.5703 base rate, the closest to the base rate of any trained model) make it a consistent presence in the cross-section, and its removal lets the rotation among the other three models propagate freely. The cross-run pattern that emerges is suggestive: across the previous and current locked configurations, exactly one model has been the "anchor" (LSTM under 2 seeds, KAN under 3 seeds), and the anchor identity moves with the seeds configuration. The structural finding is that the cross-section needs one well-calibrated, low-variance candidate to keep selection reproducible.
- **Consistency with EMH.** The DSR < 0.95 finding aligns with semi-strong-form EMH (Fama 1970) for BTC daily direction: under leakage-free evaluation with multiple-testing correction, no architecture in the comparison demonstrates a statistically significant predictive edge. The fact that **buy-and-hold posts a higher median Sharpe (2.2321) than any trained model** strengthens this reading: in the 2015 to 2026 BTC sample, the unconditional long position dominates every directional-trading strategy in median Sharpe terms, suggesting that the asset's overall trajectory carries information that direction-prediction strategies discard by going neutral or short in volatile regimes. This is the intellectually honest answer to Q1 and consistent with `chassot_audrino_2026`: properly fitted baselines are hard to beat, and the literature's ML-superiority results came from suboptimal fitting schemes that handicapped the baselines. **Cite:** `fama_1970`, `chassot_audrino_2026`.
- **Conservative Predictions framing.** `nabar_shroff_2023`: in low-signal regimes, abstention beats prediction. The bet-sizing threshold operationalises this: predictions with `p ≈ 0.50` produce `bet ≈ 0`, structural abstention without a separate model. The locked run's mean P(Up) for KAN is 0.5572 against an empirical base rate of 0.5703 (Δ = -0.0131, well within the ±0.03 calibration tolerance), so the calibration audit does not flag KAN; the bet-sizing curve is operating on well-calibrated probabilities, and the negative DSR finding is therefore a statement about signal strength, not about probability quality. Two other models are flagged by the calibration audit (AR Logistic at -0.0302, LSTM at -0.0328), both under-predicting the empirical base rate by slightly more than 3 percentage points. This is a tighter calibration footprint than the previous locked configuration, which flagged three trained models (AR Logistic, XGBoost, LSTM); the current run's XGBoost moves to ok (Δ -0.0243) and the broader profile shifts toward calibration tolerance.
- **Three-version pipeline evolution (advisor narrative).** Inflated ROC-AUC → degenerate all-Down predictions → AFML. Two specific failure modes motivated AFML adoption: uninformative fixed-time-horizon labels and structural leakage that standard splits cannot prevent. The locked-run results show that even after AFML closes both leakage channels, the residual signal is too weak for any compared model to clear DSR = 0.95; this is the AFML pay-off. The methodology produces honest results (PBO 0.34 in the moderate-overfitting regime) rather than the inflated metrics that earlier (leaky) pipeline versions produced. Across the three locked configurations evaluated to date (62 features / 2 seeds for neural, 73 features / 2 seeds for neural, 73 features / 3 seeds for all), PBO has tracked 0.26 → 0.37 → 0.34, all within the [0.25, 0.40] band: a moderate-overfitting reading that is stable across configurations and constitutes the headline statistical finding of the thesis.

### 5.2 KAN performance in context (Q3, Q4 evidence)

- **KAN ranks 2nd of 6** in median Sharpe (0.8375), behind XGBoost (0.9483) and ahead of LSTM (0.6029), Logistic Regression (0.4640), Random Forest (0.3265), and AR Logistic (-0.1864). Buy-and-hold's 2.2321 sits above the entire trained-model cross-section. KAN's positioning recovers substantially from the previous locked configuration, where it ranked 5th of 6 on median Sharpe (0.3834). The recovery is consistent with the seeds reconfiguration: in the previous run LSTM and KAN used 2 seeds while the four classical models used 3, leaving a within-model averaging asymmetry that handicapped the neural models on path-Sharpe variance; the current run uses 3 seeds for all six models and KAN's class accuracy advantage translates back into a competitive Sharpe ranking.
- **Classification-side metrics remain competitive.** KAN posts mean accuracy 0.5377 (the highest in the cross-section), mean AUC 0.5205 (the highest in the cross-section), and median win rate 0.5517 (the highest in the cross-section). The DeLong test (Section 4.3) confirms KAN's pooled AUC of 0.5129 is the highest among trained models, but no pair clears α = 0.05; the closest pairs (KAN vs. LSTM at p = 0.14, KAN vs. XGBoost at p = 0.12) are within reach of significance but do not arrive. KAN's classification-side dominance is consistent across all three locked configurations evaluated to date: it is the highest-AUC or near-highest-AUC trained model in every run, regardless of seed cardinality or feature pool size.
- **Tension to interpret: top classification metrics vs. structural anchor role rather than ranking dominance.** KAN posts the highest mean accuracy, mean AUC, and median win rate in the cross-section, ranks second on median Sharpe, posts the best (smallest) median max-drawdown among trained models (-0.2254), but the leave-one-out PBO finding (Section 4.4) reads its role differently: **KAN is the only model whose presence stabilises the cross-section** (removing it raises PBO from 0.343 to 0.514, the highest LOO value in the table and across the threshold for random selection). The interpretation: KAN's bet-sizing distribution favours conservative, well-calibrated positions (mean P(Up) 0.5572 is the closest of any trained model to the 0.5703 base rate, σ Δp ≈ -0.0131); this calibration is what keeps the model present across IS/OOS sub-path partitions rather than rotating in and out. The financial-ranking finding (rank 2 on Sharpe) and the structural-stability finding (anchor under LOO PBO) are independent observations: KAN is both a top-three model on multiple ranking metrics and the only model whose presence keeps selection reproducible.
- **Statistical-vs-economic gap.** KAN's mean AUC of 0.5205 is highest in the cross-section, but the DeLong test against every other model returns p > 0.10. The classification-side superiority is consistent in sign across all five pairwise comparisons (KAN is higher in every pair) but uniformly too small to clear standard significance levels. What does translate into a ranking position is the bet-sizing transformation of KAN's well-calibrated probabilities: at a base rate of 0.5703 and a KAN mean P(Up) of 0.5572, the bet-sizing curve places conservatively-sized bets that protect against the largest drawdowns (KAN's median max-drawdown -0.2254 is the smallest among trained models). The honest reading: KAN's classification advantage is genuine but small; its rank-2 Sharpe ranking comes from the combination of classification accuracy and well-calibrated bet sizing, not from a single dominant edge.
- **Contrast with `oad_kasper_2025`.** KASPER reports R² = 0.89 and Sharpe = 12.02 on individual stocks with regime detection. This thesis posts median Sharpe = 0.84 for KAN on BTC daily direction without a regime layer. The differences are explanatory rather than competitive: KASPER uses regression on individual stocks with regime-conditional architectures; this thesis uses classification on a single asset (BTC) with no regime layer and applies AFML's full statistical-correction stack (DSR, PBO, DeLong) which KASPER does not. The roughly 14× Sharpe gap reflects asset class, target type, and evaluation rigour rather than KAN architectural capacity.
- **Contrast with `cho_lee_kim_2025`.** The VIX KAN paper extracts symbolic formulas that reveal mean-reversion and leverage effects, validated against domain knowledge. This thesis's symbolic formula reveals a five-feature mixed-primitive structure through `sin`, `tanh`, `atan`, `x³`, and `|·|` primitives, on a target (BTC daily direction) that is not mean-reverting and where domain knowledge does not directly validate any specific functional form. The new primitives in the locked configuration (`atan` and `|·|`, replacing the previous run's `cos` and `x²`) suggest the KAN's learned decision boundary changes its functional character with the surviving feature set rather than producing a stable signature across configurations. The symbolic-extraction machinery transfers across asset classes; the interpretive yield depends on whether the underlying target has a structure that domain knowledge can corroborate, and the recurrence of the four-outer-term structure across both locked configurations is itself a transferable observation about PyKAN's behaviour under the AFML evaluation regime.

### 5.3 Symbolic extraction as contribution (Q4 evidence)

- **Interpretability is a separable contribution from predictive accuracy.** The locked-run formula has **93% symbolification rate** (13 of 14 edges; 1 edge skipped at the symbolic substitution stage) and a closed-form expression on five features; its post-symbolic accuracy of **60.90%** is a 1.5 percentage-point *gain* over the spline KAN's pre-symbolic 59.40% and 10.90 percentage points above the 50% baseline. The closed-form decision function provides three artefacts a black-box model cannot: per-feature symbolic derivatives (Section 4.6 numerical sensitivity), term-structure decomposition (each feature appears in 4 of 4 outer terms), and an audit trail from input to probability that a domain expert can inspect. The fact that the symbolic form outperforms the spline form is consistent across the previous and current locked configurations (the previous run posted +4.5 pp, the current run +1.5 pp): the symbolic substitution appears to act as a smoothing regulariser on the per-edge B-spline shapes, and the affine fine-tuning step then tunes the per-edge scale and bias to the validation distribution.
- **The post-symbolic accuracy gain shrinks relative to the previous run.** The +1.5 pp gain in the current configuration is one-third the magnitude of the previous locked configuration's +4.5 pp gain. Three diagnostic changes accompany the shrinkage: the pre-symbolic baseline starts higher (0.5940 vs. 0.5564), leaving less room for improvement; the minimum per-edge R² drops to 0.5746 (vs. 0.7768), meaning one symbolic substitution is a rougher fit; and one of fourteen edges is skipped entirely (the 93% rate vs. 100% in the previous run). Together these say the formula in the current run is a slightly noisier representation of the same underlying decision logic. The +1.5 pp gain is a single-fold (split 27) result on 133 validation observations and falls within the seed-to-seed and split-to-split noise observed elsewhere in the pipeline; the honest finding is "closed-form symbolic representation does not cost validation accuracy on this dataset across two independent locked runs, and may improve it slightly on the most recent CPCV fold". Per-fold extraction (L3 in 5.4) would let the analysis distinguish between fold-specific gains and a robust symbolic-smoothing effect.
- **The Phase-1 gate failure returns.** Phase 1 (Adam) validation accuracy of 0.5038 falls below the 0.53 minimum gate that the pipeline uses to flag whether the spline KAN has learned meaningful patterns. The previous locked run cleared this gate at 0.6241; the current run does not. The downstream recovery in the LBFGS phases (val_acc 0.5940 at Phase 2b) is what produces the symbolifiable model, but the Phase-1 gate failure is an honest signal that the initial Adam-optimised KAN had not extracted strong direction signal from the five-feature input. The reading: the symbolic-extraction pipeline can still deliver a useful closed-form on a model that did not pass its own warning gate, but the gate-failure return is a real diagnostic regression that future runs should monitor.
- **Mixed-primitive functional form.** The decision function uses five distinct primitives (`sin`, `tanh`, `atan`, `x³`, and `|·|`), arranged in four outer terms that each pool the five input features through different combinations. The previous locked configuration used `sin`, `cos`, `tanh`, `x²`, `x³`; the present run replaces `cos` and `x²` (the cosine and squared-inner-argument primitives that attached to `eth_btc_ratio` / `us2y` / `kurtosis`) with `atan` and `|·|`. The `atan` primitive attaches to `log_returns_lag1` and `natgas_ret_30` and acts as a bounded smooth saturator; the `|·|` primitive attaches to `natgas_ret_30` as a non-smooth V-shape primitive. The presence of `atan` and `|·|` reflects the genuine flexibility of the KAN representation: the brute-force fitting library tries roughly 14 candidates per edge and keeps the highest-R² match, so different feature sets and different folds land on substantively different primitive combinations. The four-outer-term structure recurs across both locked configurations even though the primitive set differs, which is suggestive evidence that PyKAN's pruning step systematically produces a four-outer-term representation under the `[5, 2, 2]` architecture.
- **Numerical sensitivity at the dataset mean.** **`skewness` is the dominant feature in the formula** in absolute σ-effect terms (-1.76 on the logit, σ Δp ≈ -0.44), driving P(up) from approximately 0.50 to approximately 0.15 with a one-σ increase. This is the only feature with a meaningful σ-effect in the current run; `bb_width` follows at -0.33 (σ Δp ≈ -0.08), `eth_btc_ratio` at -0.07 (σ Δp ≈ -0.02), `log_returns_lag1` effectively flat (+0.003), and `natgas_ret_30` undefined due to the `|·|` primitive's gradient computation hitting a SymPy edge case at the dataset mean. The economically meaningful sensitivity ranking is therefore `skewness` (dominant negative, sigmoidal but not saturating) > `bb_width` (moderate negative) > `eth_btc_ratio` (small negative) > `log_returns_lag1` (effectively flat) > `natgas_ret_30` (undefined). This is a structural reversal from the previous locked configuration, where `us2y` and `kurtosis` posted saturating positive σ-effects of approximately +10 each, driving the decision function into upper saturation in a single σ bump; the current configuration has no saturating-positive features, and the strongest signal is moderately negative and from a different feature family (TA / distribution shape rather than macro and TA / distribution shape). The cross-run difference in dominant primitives and dominant features is itself a methodological observation: the symbolic representation is not stable across locked configurations even when the input space is the same 73-feature universe, because the per-fold feature subset and the per-edge symbolic-library winner both vary.
- **The natgas_ret_30 undefined-gradient flag.** The locked-run formula contains an `|5028·natgas_ret_30/827 + 53/29|` term in two of the four outer summands. SymPy's automatic differentiation produces a `sign(·) × derivative(·)` term that includes `re(·)` and `im(·)` decompositions at the dataset mean, and the resulting expression evaluates to non-real or numerically-unstable derivatives. The pipeline reports this honestly as "1/5 features have non-finite gradient at the median; reported as NaN. Consider eval_point='median' or inspect the symbolic formula for poles near these features' median." The reading: the symbolic extraction has produced a formula whose interpretability at the dataset mean is degraded for one of five features. The previous locked configuration had no such issue (all five features had finite gradients at the mean); the current run's `|·|` primitive introduces the non-differentiability. A future run could evaluate sensitivity at a perturbed reference point (e.g. eval_point='median') to recover a finite gradient, but the cleaner fix is to flag this as an L4-style limitation: the symbolic library's inclusion of `|·|` produces formulas with non-smooth points that can land on or near a feature's reference value.
- **The on-chain finding hardens across configurations.** **No on-chain feature survives extraction in either of the two 73-feature locked configurations.** The previous 62-feature locked run retained `exchange_supply_pct` as the single on-chain feature in the surviving four; both subsequent runs at 73 features have produced zero on-chain survivors. The Q2 answer therefore reads: on-chain features may carry signal in some narrowly-defined configurations (the 62-feature pool with `exchange_supply_pct` at 46% selection frequency), but the broader configuration with 20 macro features and 10 lag features competing in the MDA pool produces zero on-chain survivors at any reasonable seeds configuration. The on-chain free lunch fails not only at the cross-fold stability level (no on-chain feature reaches 50% selection frequency across the 28 folds, and one (`tx_count_roc_14`) is never selected at all) but also at the deployment-fold level (no on-chain feature reaches the symbolic decision function on split 27 in either the 2-seed or 3-seed locked run). The earlier locked-run observation that one on-chain feature could carry the formula's strongest effect was regime-specific and does not generalise.
- **The lag-feature finding is new.** For the first time in any locked configuration, a lag feature (`log_returns_lag1`) survives into the symbolic representation. Its σ-effect is effectively zero (+0.0008 σ Δp at the dataset mean), so its presence is structural rather than economic. The lag-features-in-MDA-pool design choice (Section 3.4.4, advisor-driven, lag features compete with engineered features rather than feeding only into AR Logistic) admits lag features into the surviving set in some configurations; the new run is the first to demonstrate this, but the surviving lag feature contributes essentially no gradient at the dataset mean. The interpretive yield is therefore "the lag-feature design choice is not categorically wrong (a lag feature can reach the surviving set), but the surviving lag feature does not necessarily contribute meaningful signal".
- **Contrast with `cho_lee_kim_2025`.** The VIX KAN paper's formulas reveal mean-reversion and leverage effects that domain knowledge corroborates directly. This thesis's formula reveals a five-feature mixed-primitive structure on a target (BTC daily direction) where domain knowledge does not directly validate any specific functional form. The current locked configuration's dominant feature is `skewness` with a negative σ-effect, which can be read against the literature on right-skewed crypto-return regimes (positive skew followed by mean-reverting weakness), but the read is suggestive rather than definitive. The interpretive yield is softer than the VIX KAN paper's mean-reversion result and softer than the previous locked configuration's saturating-positive-switch reading (which had been suggestive of a crisis-then-recovery pattern on `us2y` and `kurtosis`); the cross-run variability in the dominant feature and its sign is itself an honest finding about the symbolic-extraction methodology.
- **Limitations.**
  - **Fold-specific.** Different folds may produce different formulas with different surviving features; per-fold extraction is L3 in 5.4. The cross-run change from `(us2y, kurtosis)` dominating to `skewness` dominating, with no overlap in the saturating-feature set, is direct evidence that fold-specificity is a real limitation rather than a theoretical concern.
  - **Small sample.** 530 training observations on split 27 is not large; PyKAN's per-edge R² fits stabilise on this size but recover signal that the spline KAN already captured rather than discovering new structure.
  - **R² threshold sensitivity.** The 0.30 threshold (lowered from 0.50) admits one edge at R² = 0.5746, which is moderately rough. A stricter 0.50 threshold would have left this edge out as well (taking the symbolification rate from 93% to roughly 86%); the 0.30 threshold preserves more of the formula at the cost of one rough edge.
  - **The `skewness` subset bug remains resolved.** Both the previous and the current locked configuration use the 73-feature MDA pool that routes skewness into the per-fold processed subset on split 27. The earlier methodological inconsistency between Section 4.5's stability output and Section 3.12.3's feature-selection logic is closed in both 73-feature configurations.
  - **The Phase-1 gate failure is a new return.** Unlike the previous locked configuration (where Phase-1 cleared the 53% gate at val_acc 0.6241), the current run reports val_acc 0.5038, below the gate. The downstream recovery in LBFGS produces a useful model, but the Phase-1 diagnostic regression is itself a methodological caveat worth tracking across future runs.
  - **The natgas_ret_30 |·| primitive introduces a non-differentiability** in the symbolic formula. Sensitivity at the dataset mean is undefined for this feature; per-feature evaluation at perturbed reference points would recover a finite gradient but does not appear in the locked-run output.
  - **Architectural parity has been resolved.** The single-hidden-layer / `width1 ≤ 6` configuration matches the CPCV-benchmarked KAN, so the formula describes the architecture the benchmark numbers describe, not a simplified surrogate.

### 5.4 Limitations (five honest, from defense L1 to L5)

- **L1. Computational power.** Currently 3 seeds across all 6 models (an upgrade from the previous locked configuration's split 3 seeds for sklearn / 2 for neural; the current run is symmetric and produces 504 = 28 × 6 × 3 prediction entries). 30 Optuna trials per model per fold, KAN width capped at 6. With more compute: more seeds, more trials, wider HP ranges. The N=8 / 28-split configuration alone roughly doubles the per-experiment runtime relative to N=6 / 15 splits, and the per-fold MDA inside Optuna's TPE inner loop adds roughly 40% on top.
- **L2. Daily OHLCV only.** Discards intraday microstructure. Pipeline is timeframe-agnostic; hourly extension would multiply event count and unlock microstructure features.
- **L3. Symbolic extraction is fold-specific.** The locked formula is the result of running the extraction on split 27 only. Per-fold extraction (28 formulas, one per CPCV split) would let the analysis count which features and primitives recur (structural signal vs. one-off noise). Of particular interest given the cross-run variation observed to date: would `skewness` still post the dominant negative σ-effect on earlier CPCV folds with different regime composition, or is that the result of the 2024-08 to 2026-04 deployment window? And would the `(eth_btc_ratio, bb_width, skewness, log_returns_lag1, natgas_ret_30)` feature set recur or rotate across folds? The `skewness` subset-bug issue noted in earlier drafts is resolved in both 73-feature locked configurations (all top-5 stable features now reach the extraction step), so this limitation is purely about the single-fold scope rather than about any methodological inconsistency.
- **L4. KAN width cap is binding for interpretability.** The `width1 ≤ 6` cap is set so the extracted symbolic formula stays humanly readable, since each surviving width1 unit becomes one additive term plus its interactions in the closed-form expression. The locked configuration's pruned architecture `[[5, 0], [2, 0], [2, 0]]` uses width1 = 5 and width2 = 2, producing 14 active edges and a four-outer-term decision function that fits one printed page in the thesis. A larger width1 cap would let the model fit more complex patterns but would also produce a formula too long to interpret. This is an explicit interpretability-vs-capacity trade-off, not a generic constraint imposed by sample size or compute. The recurrence of the four-outer-term structure across both 73-feature locked configurations suggests the cap is not binding for the discovery of the structure itself, only for the readability of the resulting formula.
- **L5. No regime-conditional analysis.** BTC has had approximately 5 distinct regimes over 2014 to 2026 (the 2017 bubble, 2018-2019 bear, 2020 COVID crash, 2021 bull, 2022 FTX-era contraction, 2023-2026 recovery). Full-sample Sharpe averages over them. A regime-conditional analysis is left for future work, especially given the joint reading "DSR < 0.95 + PBO 0.34 moderate": full-sample non-significance does not rule out within-regime predictability, and the buy-and-hold comparison (median Sharpe 2.23 with max drawdown -0.9998) makes the regime-conditional question financially salient even when full-sample-significance is unattainable.
- **L6. Symbolic-formula non-differentiability flag (new in the locked configuration).** The locked formula contains an `|·|` (absolute value) primitive attached to `natgas_ret_30` in two of the four outer terms. SymPy's automatic differentiation of `|·|` at the dataset mean produces a non-real expression, leaving the σ-effect on the logit undefined for `natgas_ret_30`. This is reported honestly by the pipeline and acknowledged as a single-feature gap in the numerical sensitivity table. A future run could (a) restrict the symbolic library to remove `|·|`, (b) evaluate sensitivity at a perturbed reference point that is comfortably away from the `|·|` kink, or (c) report sensitivity in finite-difference form. None of these is implemented in the current pipeline. The reading: closed-form interpretability is robust most of the time but can produce non-smooth points that require manual diagnostic attention.

---

## 6. Conclusion

> **Cochrane:** Do not restate all findings. **T-R16:** End with resonance, not a hedge. Target ≤ 10% of the textual part (approximately 3.5 pages).

- One paragraph stating the contribution: full AFML applied to BTC, six-model benchmark, first KAN symbolic formula extraction in this regime, expanded 73-feature universe with multi-model MDA selection over four feature families (TA, mathematical, external macro / crypto-macro / on-chain, autoregressive lags), three-seed configuration uniform across all six models (504 prediction entries from 28 splits × 6 models × 3 seeds).
- One paragraph stating the headline: under leakage-free evaluation with multiple-testing correction, the literature's 85% to 95% accuracy claims do not survive. Top DSR among trained models is XGBoost at 0.2376, far below the 0.95 significance threshold. PBO at 0.343 places model selection in the moderate-overfitting regime: 12 of 35 IS/OOS partitions see the IS-winner underperform the OOS median, and the leave-one-out PBO analysis shows that the instability is concentrated in Logistic Regression, XGBoost, and LSTM rotating in the IS-best slot, while KAN is the only model whose presence stabilises the cross-section (removing KAN raises PBO to 0.514, across the random-selection threshold). Every trained-model bootstrap CI crosses zero in the locked configuration; only buy-and-hold's CI (1.9766, 2.6408) stays strictly positive. Buy-and-hold posts a higher median Sharpe (2.2321) than every trained model, with the structural cost of a median max drawdown of -0.9998: directional-trading strategies on BTC daily direction trade absolute return for downside protection rather than producing alpha relative to the asset's overall trajectory. This is the honest answer to Q1: under AFML evaluation, no architecture demonstrates a statistically significant predictive edge over the multi-model selection pool, and no trained model's CI stays strictly positive in the most conservative locked configuration evaluated. The PBO finding across the three locked configurations evaluated to date (0.26, 0.37, 0.34) is stable in the moderate-overfitting regime and constitutes the headline statistical finding of the thesis.
- One paragraph on what does survive: the symbolic extraction pipeline produces a humanly auditable closed-form decision function on five features (`eth_btc_ratio`, `bb_width`, `skewness`, `log_returns_lag1`, `natgas_ret_30`), with **93% symbolification rate (13 of 14 edges; 1 edge skipped)** and **post-symbolic accuracy of 60.90%, a 1.5 percentage-point gain over the pre-symbolic 59.40%**. The formula uses five primitives (`sin`, `tanh`, `atan`, `x³`, `|·|`); `skewness` carries the dominant negative σ-effect on the logit (-1.76, σ Δp ≈ -0.44 from the dataset mean), `bb_width` the second-largest (-0.33), `eth_btc_ratio` a small negative effect (-0.07), `log_returns_lag1` is effectively flat, and `natgas_ret_30` has an undefined gradient at the dataset mean due to the `|·|` primitive's non-smoothness. For the first time a lag feature enters the surviving set (`log_returns_lag1`, with effectively zero economic contribution but structural presence). No on-chain feature survives extraction in either of the 73-feature locked configurations evaluated: the Q2 "on-chain free lunch" hypothesis fails at both the cross-fold stability level (one on-chain feature, `tx_count_roc_14`, is never selected at all across the 28 folds) and the deployment-fold level (no on-chain feature reaches the symbolic decision function in any 73-feature run). The closed-form representation does not cost validation accuracy on this run (in fact gains 1.5 pp, consistent with the previous configuration's +4.5 pp gain in direction if not magnitude) and produces an artefact, with symbolic derivatives, term decomposition, and audit trail, that no black-box model in the comparison can produce. Interpretability is a separable contribution from predictive accuracy and does not need to wait for predictive significance to be valuable.
- One closing sentence with resonance. T-R16 example template: "X is not a constraint, but a catalyst, for Y." Adapt for this thesis (e.g., "Methodological honesty is not a constraint on financial machine learning; it is the prerequisite for treating interpretability as a contribution in its own right.").

---

## 7. Future Work

> **Cochrane:** Do not write your grant application here. Concrete, actionable directions only.

- **MultKAN (KAN 2.0) on the last fold.** Multiplication nodes enable discovery of multiplicative interactions. Same symbolic pipeline.
- **Higher-frequency data.** Hourly bars would multiply CUSUM events approximately 24 times and unlock microstructure features. Pipeline timeframe-agnostic.
- **Per-fold symbolic extraction.** Count which features and primitives recur across all 28 folds to distinguish structural signal from one-off noise.
- **Regime-conditional analysis.** Decompose Sharpe by regime (approximately 5 distinct regimes 2014 to 2026).
- **Meta-labeling layer.** A second classifier on the best primary classifier's output, trained to predict whether to take the trade (AFML Ch. 3). Singh and Joubert (2019, `singh_joubert_2019`) provide empirical evidence that meta-labeling improves signal efficacy across asset classes; applying it on top of this thesis's calibrated probabilities is the natural next layer.
- **Alternative assets.** ETH, gold for symbolic-formula comparison (different on-chain availability).
- **Walk-forward vs. CPCV.** Compare conclusions under both protocols.

---

## Appendices (placeholders only)

- **A. Pipeline architecture diagram** (full flowchart from raw OHLCV through symbolic formula).
- **B. CPCV split details** (group boundaries, train/test timelines for all 28 splits, leakage audit).
- **C. Hyperparameter search spaces** (Optuna search spaces for all tuned models).
- **D. Per-split classification reports** (28 splits × 6 models).
- **E. Full feature list** (complete table of 73 features with descriptions, sources, parameters).
- **F. Symbolic extraction detailed output** (PyKAN training logs, edge-by-edge R², pruned diagrams, unsimplified formulas).

---

## Reference list to populate (`mfw_references.bib`)

> Organized to match the four sections of `mfw_references.bib`. All keys here match the .bib exactly; any divergence breaks the build.

**I. AFML methodology and core framework**
- `lopez_de_prado_2018`, AFML book (root reference).
- `chassot_audrino_2026`, HARd to Beat: rolling windows in ML-era forecasting.
- `slepaczuk_bieganowski_2024`, Supervised Autoencoders with FFD and TBL on crypto.
- `kang_kim_2025`, TBL on Korean equities (TBL transferability outside crypto).
- `singh_joubert_2019`, meta-labeling efficacy (cited in Section 7 Future Work).
- `fu_et_al_2024`, GA-driven TBL with ML for crypto pair trading.
- `nabar_shroff_2023`, Conservative Predictions on noisy financial data.

**II. KAN architecture and interpretability**
- `liu_kan_2024`, original KAN paper.
- `liu_kan2_2024`, KAN 2.0 / MultKAN.
- `cho_lee_kim_2025`, VIX KAN paper (Algorithm 1 source).
- `oad_kasper_2025`, KASPER.
- `noorizadegan_2026`, Practitioner Guide to KANs.
- `yamak_et_al_2025`, comprehensive KAN time-series review.

**III. BTC / crypto prediction and DL**
- `mate_confluence_2024`, TA + ML for BTC.
- `omole_enke_2024`, DL for BTC direction prediction.
- `bourday_crypto_dl_2024`, Cryptocurrency Forecasting with DL: comparative analysis.
- `gao_decokan_2025`, DecoKAN: interpretable decomposition for crypto forecasting.
- `genet_inzirillo_2024`, TKAN: Temporal Kolmogorov-Arnold Networks.
- `wu_crypto_dl_review_2024`, review of DL models for crypto price prediction.

**IV. Foundational baselines, metrics, and methods**
- `breiman_2001`, Random Forest.
- `chen_guestrin_2016`, XGBoost.
- `hochreiter_schmidhuber_1997`, LSTM.
- `platt_1999`, Platt scaling.
- `guo_temperature_2017`, temperature / vector scaling.
- `akiba_optuna_2019`, Optuna.
- `delong_1988`, DeLong AUC test.
- `sharpe_1966`, original Sharpe ratio.
- `kolmogorov_1957`, Kolmogorov representation theorem.
- `arnold_1958`, Arnold's variant of the representation theorem.

> **Still needed in the .bib (cited in body, not yet listed here):** `garman_klass_1980`, `yang_zhang_2000`, `lo_mackinlay_1988`. Add these to the .bib before final compile or remove the citations from Sections 3.4.1 and 3.4.2.

---

## Diagnostic checklist before submission [T-R15, Thatcher Appendix E]

Use this as a final pre-submission read-through. Tick each as you go.

- [ ] Introduction first sentence states the contribution. No "the finance literature has long been interested in".
- [ ] Each section opens with a claim, not background.
- [ ] Each table caption is self-contained; a skimming reader understands the table without the body.
- [ ] Two to three significant digits everywhere.
- [ ] No em-dashes anywhere (user style rule).
- [ ] No passive voice. Search and destroy "is" / "are" passives.
- [ ] No previews ("as we will see in Table 6") or recalls ("recall from Section 2"); they signal poor organization.
- [ ] No footnotes for parenthetical comments.
- [ ] Q1↔C1, Q2↔C2, Q3↔C3, Q4↔C4 mapping holds across Introduction, Methodology, Results, Discussion.
- [ ] Each construct (event, label, observation, fold, path) uses the same word throughout.
- [ ] CPCV configuration referenced consistently as N=8, k=2, 28 splits, 7 paths everywhere it appears.
- [ ] Conclusion ends with resonance, not a hedge.
- [ ] Every robustness check alluded to in the text appears in a table or appendix.
- [ ] Reproducibility: a fellow graduate student can reproduce every number from the paper plus the appendices.