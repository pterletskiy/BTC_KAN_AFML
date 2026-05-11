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
- Directly relevant to this thesis: the negative results (top DSR 0.3925, well below 0.95) align with this finding. Properly evaluated baselines are hard to beat, especially in a weak-signal regime like BTC.
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

- "No model achieves DSR ≥ 0.95. The top DSR (LSTM) is 0.3925, far below significance after correcting for selection bias across the six compared models. Buy-and-hold posts a higher median Sharpe (2.2321) than every trained model, illustrating how punishing the AFML correction layer is for trading strategies on a trending asset."
- "PBO = 0.3714 lands in the moderate-overfitting regime: in 13 of 35 IS/OOS partitions the in-sample best model underperforms the out-of-sample median. Selection across the six models is reproducible enough to claim that some configurations beat others, but not robust enough to support a confident pick. Leave-one-out PBO sharpens the picture asymmetrically: excluding `logistic`, `kan`, or `xgboost` reduces PBO (to 0.143, 0.229, and 0.257 respectively), while excluding `lstm` actually raises PBO to 0.429. The variance lives in the rotating IS-winners among Logistic Regression, KAN, and XGBoost; LSTM is the only model whose presence stabilises the cross-section."
- "The symbolic extraction pipeline produces a closed-form decision function with **100% symbolification rate** (14 of 14 edges) on five surviving features (`eth_btc_ratio`, `skewness`, `bb_width`, `us2y`, `kurtosis`). Post-symbolic accuracy is **60.15%**, a 4.5 percentage-point *gain* over the pre-symbolic 55.64%. The skewness subset bug that affected the earlier locked configuration is resolved (all top-5 stable features now reach the extraction step), and the Phase-1 53% gate is cleared (val_acc = 0.6241 after Adam, vs. 0.4812 previously)."
- Roadmap: 4.2 model comparison, 4.3 classification, 4.4 financial, 4.5 stability, 4.6 symbolic.

### 4.2 Model Comparison (the main result, Cochrane)

- **Table.** Self-contained caption: "Model comparison ranked by median Sharpe over seven CPCV backtest paths, with buy-and-hold included as a naive baseline. DSR is computed against `n_trials=6` (number of compared models), correcting the observed Sharpe for selection bias and non-normal returns; DSR is not defined for buy-and-hold since it is not part of the model selection pool. The Sharpe confidence interval is a bootstrap 95% CI on the per-path median Sharpe. No model achieves DSR ≥ 0.95. Tiebreaker: standard deviation of path Sharpe ascending."

| Rank | Model | Median Sharpe | Sharpe CI (low, high) | Std Sharpe | DSR | Med Sortino | Med Calmar | Mean F1 | Mean Acc | Mean AUC | Med Max DD | Med Cum Ret | Med Win Rate | Med Profit Factor |
|------|-------|--------------:|----------------------:|-----------:|-----:|-----:|-----:|------:|------:|------:|------:|------:|------:|------:|
| 1 | Buy-and-Hold | 2.2321 | (1.9766, 2.6408) | 0.6715 | n/a | 2.2931 | 2.4863 | n/a | n/a | n/a | -0.9998 | 52849.15 | 0.5539 | 1.2956 |
| 2 | LSTM | 1.3377 | (0.4062, 1.5380) | 0.8284 | 0.3925 | 1.0629 | 0.1939 | 0.4367 | 0.5259 | 0.5124 | -0.2256 | 0.3329 | 0.5000 | 1.3055 |
| 3 | XGBoost | 0.9176 | (-0.8493, 1.3837) | 1.2783 | 0.2262 | 0.1899 | 0.1732 | 0.4481 | 0.5336 | 0.5099 | -0.2438 | 0.1417 | 0.5455 | 1.2963 |
| 4 | Random Forest | 0.6919 | (-0.5494, 1.2077) | 1.2019 | 0.1596 | 0.2023 | 0.1213 | 0.4383 | 0.5331 | 0.5096 | -0.1955 | 0.0975 | 0.5000 | 1.2263 |
| 5 | Logistic Regression | 0.5637 | (-0.5423, 1.5779) | 1.5672 | 0.1287 | 0.2230 | 0.1099 | 0.4165 | 0.5332 | 0.5261 | -0.2308 | 0.0544 | 0.5342 | 1.1385 |
| 6 | KAN | 0.3834 | (-0.3750, 2.0819) | 1.4409 | 0.0920 | 0.2332 | 0.0488 | 0.4282 | 0.5415 | 0.5211 | -0.2774 | 0.0530 | 0.5185 | 1.1196 |
| 7 | AR Logistic | -0.1864 | (-0.5538, 0.1112) | 0.7677 | 0.0262 | -0.1472 | -0.0413 | 0.4682 | 0.5292 | 0.5070 | -0.2778 | -0.0668 | 0.5177 | 0.9762 |

- **Key observations (state facts here, save interpretation for 5.1).**
  - **DSR.** Top DSR among trained models is LSTM at 0.3925; all six trained-model values fall below 0.95. No model demonstrates predictive ability that survives correction for selection bias. The top three DSRs (LSTM 0.3925, XGBoost 0.2262, Random Forest 0.1596) span 0.23, with smaller gaps to Logistic Regression (0.1287), KAN (0.0920), and AR Logistic (0.0262). Buy-and-hold is excluded from the DSR computation because it is not an Optuna-tuned model and does not enter the selection pool.
  - **Sharpe confidence intervals.** 5 of 7 entries have a 95% bootstrap CI that crosses zero (XGBoost, Random Forest, Logistic Regression, KAN, AR Logistic). Only buy-and-hold (1.9766, 2.6408) and LSTM (0.4062, 1.5380) have CIs that stay strictly positive. This is an independent confirmation of the DSR finding: the median Sharpe of every trained model except LSTM is not statistically distinguishable from zero under bootstrap resampling.
  - **Buy-and-hold reference point.** Buy-and-hold posts the highest median Sharpe (2.2321) and the highest cumulative return (≈52,849× over the 2015 to 2026 period, reflecting BTC's overall trajectory from approximately $325 to roughly $108k). Its median max drawdown is -0.9998 (essentially -100%): the position experiences the full BTC drawdown profile, which is precisely what bet sizing and the dual-weighted loss are designed to mitigate. Trained models trade off return for downside protection: LSTM's cumulative return of 0.3329 is two orders of magnitude smaller than buy-and-hold's, but its max drawdown of -0.2256 is also four times shallower. This is the financial-management framing in which the DSR result should be read.
  - **Ranking.** LSTM leads with median Sharpe 1.3377; XGBoost is second at 0.9176; Random Forest third at 0.6919; Logistic Regression fourth at 0.5637; KAN fifth at 0.3834; AR Logistic last at -0.1864 (the only model with a negative median Sharpe). The previous run's ranking (LSTM, KAN, RF, XGBoost, AR Logistic, Logistic) reshuffled materially: Logistic Regression moved up from last to fourth, KAN dropped from second to fifth, and AR Logistic moved from fifth to last. The reshuffle is consistent with the expanded feature universe (73 vs. 62) giving Logistic Regression a broader linear basis, while costing KAN the on-chain features that had survived to its previous symbolic representation.
  - **AUC compression.** All mean AUCs sit in the 0.5070 to 0.5261 band. Models are nearly indistinguishable at the classification level; bet sizing and the dual-weighted loss are what translate the tiny edges into the wider Sharpe spread. Logistic Regression now posts the highest mean AUC (0.5261), narrowly beating KAN (0.5211) and LSTM (0.5124); KAN's first-place AUC in the previous locked run is lost on the expanded feature set.
  - **Q1, Q2 evidence (feature engineering).** AR Logistic now posts a negative median Sharpe (-0.1864), the worst in the cross-section. The expanded 10-lag set (lags 4, 5, 6, 21 in addition to the previous 1, 2, 3, 7, 14, 30) does not help; if anything, it makes pure momentum slightly worse, suggesting the additional near-term lag information adds noise rather than signal. Logistic Regression, which receives the MDA-selected feature set, now beats AR Logistic, KAN, and ties XGBoost-and-RF for second-tier status: with 73 features in the MDA pool the linear hypothesis class has enough breadth to extract usable signal where it could not in the 62-feature configuration.
  - **Q3 evidence (KAN positioning).** KAN drops from 2nd to 5th place on median Sharpe (0.9879 → 0.3834) and from 2nd to 5th on DSR (0.2509 → 0.0920). The classification-side metrics remain competitive (mean accuracy 0.5415, second only to XGBoost's 0.5336; mean AUC 0.5211, second only to Logistic Regression's 0.5261), but the Sharpe ranking suggests the bet-sizing transformation is no longer amplifying the classification edge in the same way it did in the previous configuration. KAN's std Sharpe at 1.4409 is the second-highest in the trained-model cross-section, contributing to the wide Sharpe CI (-0.3750 to 2.0819) that straddles zero.
  - **Std Sharpe outlier.** AR Logistic shows the lowest std Sharpe (0.7677) among the trained models. Pure momentum is consistent across paths because it conditions on a simple, stable signal; the cost is the only negative median in the table. Logistic Regression now carries the highest std Sharpe (1.5672), reflecting that its directional accuracy is regime-conditional rather than uniform across paths.

### 4.3 Classification Metrics

- **Per-split F1 distribution.** Mean F1 ranges narrowly from 0.4165 (Logistic Regression) to 0.4682 (AR Logistic). The high-F1 / negative-Sharpe combination for AR Logistic confirms that classification quality and bet-sizing translation are partially independent: pure momentum gets the average call right slightly more often than the engineered-feature models but loses the high-conviction events that drive Sharpe and overpays in transaction costs, leaving a negative median Sharpe.
- **Pooled AUC.** Per-model pooled AUC across all 28 splits sits in the 0.4889 to 0.5151 band. All models hover within two percentage points of random; the entire signal contest plays out inside an AUC range smaller than what most reviewers would consider economically material.
- **DeLong pairwise AUC.** **0 of 15 pairs significantly different at α=0.05.** Pooled across splits, averaged across the 3 sklearn / 2 neural seeds:

| Pair | AUC_a | AUC_b | Δ AUC | z | p | Significant |
|------|-------|-------|-------|---|---|-------------|
| LSTM vs KAN | 0.5026 | 0.4889 | +0.0137 | 1.3785 | 0.1681 | No |
| XGBoost vs KAN | 0.5045 | 0.5151 | -0.0106 | -1.5689 | 0.1167 | No |
| AR Logistic vs KAN | 0.5046 | 0.5151 | -0.0105 | -1.1251 | 0.2605 | No |
| Logistic vs LSTM | 0.5096 | 0.5002 | +0.0094 | 0.9011 | 0.3675 | No |
| Random Forest vs KAN | 0.5064 | 0.5151 | -0.0087 | -1.5091 | 0.1313 | No |
| AR Logistic vs LSTM | 0.5070 | 0.5002 | +0.0068 | 0.6636 | 0.5070 | No |
| Logistic vs XGBoost | 0.5106 | 0.5045 | +0.0062 | 0.8115 | 0.4171 | No |
| AR Logistic vs Logistic | 0.5046 | 0.5106 | -0.0061 | -0.6635 | 0.5070 | No |
| Random Forest vs LSTM | 0.5058 | 0.5002 | +0.0056 | 0.5668 | 0.5708 | No |
| Logistic vs KAN | 0.5106 | 0.5151 | -0.0044 | -0.6724 | 0.5014 | No |
| Logistic vs Random Forest | 0.5106 | 0.5064 | +0.0042 | 0.5924 | 0.5536 | No |
| XGBoost vs LSTM | 0.5042 | 0.5002 | +0.0041 | 0.4112 | 0.6809 | No |
| Random Forest vs XGBoost | 0.5064 | 0.5045 | +0.0019 | 0.3860 | 0.6995 | No |
| AR Logistic vs Random Forest | 0.5046 | 0.5064 | -0.0018 | -0.1938 | 0.8463 | No |
| AR Logistic vs XGBoost | 0.5046 | 0.5045 | +0.0001 | 0.0111 | 0.9912 | No |

- **Pairwise summary.** **No pair clears α = 0.05.** KAN posts the highest pooled AUC (0.5151), narrowly ahead of Logistic Regression (0.5106), Random Forest (0.5064), AR Logistic (0.5046), XGBoost (0.5045), and LSTM (0.5002), but none of the differences is statistically distinguishable from sampling noise under the DeLong test. The expansion of the feature universe from 62 to 73 columns has flattened the classification cross-section: in the previous configuration 5 of 15 pairs were significant; in the locked configuration none are. The five previously-significant pairs all involved the Random Forest column (RF vs. XGBoost, RF vs. LSTM, Logistic vs. RF) or the KAN column (KAN vs. XGBoost, Logistic vs. KAN); the additional features appear to have closed the gap between Random Forest / KAN and the other models on raw classification, leaving the financial differences in Section 4.2 driven entirely by bet sizing rather than by AUC.
- **Effect-size disclosure (T-R15: write for reviewers).** The entire AUC range across all six models is 0.026 (from LSTM at 0.5002 to KAN at 0.5151), the same order of magnitude as the seed-to-seed and split-to-split variation within any single model. The DeLong test confirms that none of these gaps survives a hypothesis test once cross-split correlation is accounted for.
- **Confusion matrices (compact, Appendix D for full).** Aggregated TP/FP/TN/FN per model. Most models post a slight class-1 bias consistent with the empirical 57.0% / 43.0% Up/Down base rate.
- **Multiple testing.** No Bonferroni or Benjamini-Hochberg correction is needed since no nominal p-value clears α = 0.05 in the first place. The 0 of 15 finding is robust to any sensible correction.

### 4.4 Financial Performance

- **Path-level Sharpe distribution.** Per-model row × 7 paths + median + std. LSTM has the highest median (1.3377) **and the lowest std among trained models** (0.8284); its path Sharpes cluster tightly around the median, which is why its bootstrap 95% CI (0.4062, 1.5380) stays strictly positive. XGBoost (median 0.9176, std 1.2783) and Random Forest (median 0.6919, std 1.2019) sit in a middle band with similar dispersion. Logistic Regression (median 0.5637, std 1.5672) has the **highest path-Sharpe std in the cross-section**, with a CI (-0.5423, 1.5779) straddling zero by a wide margin. KAN (median 0.3834, std 1.4409) has the second-highest std and the widest CI (-0.3750, 2.0819), reflecting strong regime sensitivity. AR Logistic (median -0.1864, std 0.7677) is the only trained model with a negative median; its CI (-0.5538, 0.1112) is the only one that does not include a meaningfully positive value. Buy-and-hold (median 2.2321, std 0.6715) dominates the median ranking with the tightest CI (1.9766, 2.6408) of the seven entries.
- **Sortino and Calmar.** The downside-deviation-adjusted Sortino ratios reorder the trained models slightly: LSTM (1.0629), KAN (0.2332), Logistic Regression (0.2230), Random Forest (0.2023), XGBoost (0.1899), AR Logistic (-0.1472). KAN's Sortino exceeds Logistic Regression's even though its Sharpe is lower, suggesting that KAN's path-Sharpe variance is more symmetric (gains and losses scaled together) while Logistic Regression's variance loads more heavily on the upside, leaving downside performance less penalised by Sortino than by Sharpe. Calmar (Sharpe over max drawdown) compresses the trained models into a narrow 0.05 to 0.20 band, with LSTM at 0.1939 leading and AR Logistic at -0.0413 the only negative entry. Buy-and-hold's Calmar (2.4863) is two orders of magnitude above any trained model, but its near-100% max drawdown means a single rebalancing decision during the deepest crash would wipe out the strategy.
- **DSR computation detail.** For LSTM (top-ranked trained model): observed median Sharpe 1.3377 over 7 paths, DSR = 0.3925 against `n_trials=6`. The expected maximum Sharpe under the null with six compared models, combined with the pooled skew/kurt and SE(SR), produces a DSR threshold the observed Sharpe does not clear. The 0.3925 figure says LSTM's median Sharpe is roughly at the 39th percentile of what the null hypothesis (all true Sharpes equal zero) would produce when testing six independent strategies, far short of the 0.95 significance threshold. Full DSR-component breakdown for the top model goes to Appendix D.
- **PBO result.** **PBO = 0.3714 = 13 of 35 IS/OOS partitions.** Interpretation in context of DSR < 0.95:
  - Theoretical regime map: `PBO ≈ 0` = robust selection (the IS-best model tends to remain best out of sample); `PBO ≈ 0.5` = model selection is random; `PBO > 0.5` = adversarial.
  - Observed regime: **moderate overfitting risk.** In 37.14% of partitions the model that wins on three IS paths underperforms the median of the four OOS paths. The IS ranking holds out of sample more often than not, but not robustly: the IS-winner is the OOS-loser about three times out of eight. This is closer to "model selection is partially random" than to "the IS ranking holds out of sample". The locked configuration is a step *worse* than the previous run (PBO 0.2571), driven by the reshuffle in the model cross-section: Logistic Regression's promotion to the top-half of the ranking introduces a model whose path-Sharpe variance is large enough to swap places with the top three nonlinear models across sub-path partitions.
  - **Leave-one-out PBO sharpens the picture asymmetrically.** Re-running PBO on the 5×7 Sharpe matrix with one model dropped at a time produces a pattern that is more informative than the headline figure. Sorted by 5-model PBO ascending:

| Model excluded | 5-model PBO | Δ vs baseline |
|---|---|---|
| logistic | 0.143 | -0.229 |
| kan | 0.229 | -0.143 |
| xgboost | 0.257 | -0.114 |
| ar_logistic | 0.371 | +0.000 |
| random_forest | 0.371 | +0.000 |
| lstm | 0.429 | +0.057 |

  - **Reading the LOO table.** The IS/OOS instability is concentrated in three models: Logistic Regression, KAN, and XGBoost. Removing any one of them reduces PBO by 0.11 or more. Logistic Regression alone accounts for the largest single reduction (-0.229): with Logistic Regression dropped, the five remaining models settle into a stable IS ranking that the OOS median agrees with on roughly 86% of partitions. AR Logistic and Random Forest are PBO-neutral: their exclusion does not change the figure, meaning they are far enough from the IS-best slot that they do not contribute to the IS/OOS rotation. LSTM is the opposite: removing it *increases* PBO to 0.429, which is the highest value in the LOO table and approaches the random-selection regime. This is a striking finding: **LSTM is the only model whose presence stabilises the model cross-section**. Its low path-Sharpe std (0.8284) makes it a consistent IS-winner across sub-path partitions; remove it and the rotation among Logistic Regression, KAN, XGBoost, and Random Forest dominates.
  - Joint reading with DSR < 0.95: model selection within this six-model universe is partially reliable (the headline PBO of 0.371 is below the 0.5 random-selection threshold but well above the 0.3 robust-selection threshold), is anchored by LSTM (whose exclusion moves the regime toward random), and is destabilised by Logistic Regression / KAN / XGBoost rotating in the IS-best slot. The model that selection picks (LSTM in this configuration) does not achieve a Sharpe that survives multiple-testing correction. The honest conclusion is "LSTM is the most stable choice in this cross-section, its bootstrap CI is the only trained-model CI that stays strictly positive, but its 0.39 DSR sits well below the 0.95 multiple-testing-corrected significance threshold and a naive buy-and-hold strategy posts a 2.23 median Sharpe over the same period".
  - Methodological implication (T-R15: write for reviewers, this is the AFML pay-off). Without PBO, the natural conclusion from the model-comparison table is "LSTM achieved median Sharpe 1.34 on BTC daily direction". With the headline PBO of 0.3714 the conclusion sharpens: rankings are stable enough to claim that LSTM beats the other trained models, but the IS/OOS instability among Logistic Regression, KAN, and XGBoost means that any alternative selection rule (highest Sortino, highest AUC, lowest drawdown) would pick a different model on at least one IS sub-path partition. The leave-one-out PBO refines the message further: LSTM is the anchor of the cross-section rather than a source of overfitting risk, reversing the previous run's reading where LSTM was the variance source. This reversal reflects the genuine result of expanding the feature universe: with 73 features, the linear baseline (Logistic Regression) gains enough lift to become a contender in some IS partitions, but its high path-Sharpe std makes the contention unstable. The PBO and DSR diagnostics are complementary: PBO answers "did we pick the right model?", DSR answers "is the right model good enough?"; here the answer is "yes for LSTM, with caveats about the rest of the cross-section" to the first and "no" to the second.
- **Equity curves figure.** All 6 models + buy-and-hold overlay on the median (or best) path. Note specifically the AR Logistic equity curve, which is the only one ending below 1.0 (median cumulative return -0.0668). Buy-and-hold's curve dominates the plot vertically (52,849× cumulative return); a log-scale y-axis is needed to see the trained models alongside it.
- **Additional metrics table.** Cumulative return, annualised return, max DD, time under water, win rate, profit factor, mean |bet|, % traded. Locked-run highlights from the table at 4.2: LSTM median cumulative return 0.3329 (best among trained), XGBoost 0.1417, Random Forest 0.0975, Logistic Regression 0.0544, KAN 0.0530, AR Logistic -0.0668. Median max-drawdown ranges from -0.1955 (Random Forest, best) to -0.2778 (AR Logistic, worst); KAN's -0.2774 is the second-worst, consistent with its high path-variance signature. Buy-and-hold's median max-drawdown is -0.9998, the structural cost of holding through the 2017-18 and 2022 BTC crashes.

### 4.5 Feature Selection Stability and FFD Stability

- **Feature selection frequency (per-feature count / 28 folds).** Locked-run profile is **flat to a striking degree.** Of 73 candidate features:
  - **0 features stable** (selected in more than 80% of folds, i.e. > 22 of 28 folds).
  - **3 features moderate** (50% to 80%): `eth_btc_ratio` at 75.0% (21 of 28 folds), `skewness` at 71.4% (20 of 28 folds), `bb_width` at 57.1% (16 of 28 folds).
  - **70 features low** (selected in fewer than 50% of folds).
  - **0 features never selected**.
- **Headline finding.** No compact feature subset consistently dominates across time periods. The MDA-selected set turns over substantially across the 28 folds: the median feature is selected in roughly a third of folds, and even the most-stable feature (`eth_btc_ratio`) clears 50% by about 25 percentage points. This is itself a finding: it suggests that the information relevant to BTC daily direction is distributed across many features, with regime-specific pockets of relevance, rather than concentrated in a small permanent core. The expansion of the feature universe from 62 to 73 columns lifts the top of the stability distribution, with `eth_btc_ratio` and `skewness` now clearing 70% (compared with 54% and 57% respectively under the smaller pool), suggesting that the additional macro (14-day variants) and lag (lags 4, 5, 6, 21) columns absorb some of the noise that was previously rotating into the MDA-selected set.
- **Group breakdown of the three moderate features.**
  - 2 of 3 from the technical-analysis group (`skewness`, rolling 21-day third moment of log returns; `bb_width`, Bollinger Band width).
  - 1 of 3 from the external / crypto-macro group (`eth_btc_ratio`, the CoinMetrics-sourced ETH/BTC alt-rotation signal).
  - 0 from macro, on-chain, mathematical (AFML Part 4), or lag groups.
- **On-chain question (Q2).** **No on-chain feature reaches 50% selection frequency.** All eight on-chain columns (active addresses ROC, transaction count ROC, hash rate ROC, MVRV, net exchange flow, fee per transaction, exchange supply percent, issuance) sit in the "low" bucket. The "free lunch hypothesis" (on-chain transparency providing measurable signal beyond price and volume) does not survive multi-model MDA + CPCV evaluation in this dataset; on-chain features may carry information for individual splits, but no on-chain column is consistently selected across folds. Q2 answer: **on-chain features do not exhibit stability** under the AFML feature-selection regime applied here. This finding hardens relative to the previous locked configuration: in the smaller 62-feature pool one on-chain column (`exchange_supply_pct`) was retained in the top-5 stable set, but it does not appear in the 73-feature pool's top-5, where it is displaced by `bb_width`, `us2y`, and `kurtosis`.
- **Figure.** Horizontal bar chart of selection frequency (73 bars, sorted descending, coloured by feature group: green=TA, blue=mathematical, orange=external macro, red=external crypto-macro, brown=external on-chain, grey=lag).
- **FFD stability.** ATR is the only column subject to FFD (the only feature flagged non-stationary by the cross-fold ADF audit). Locked-run statistics across 28 folds × 3 seeds (84 d* estimates):
  - Mean d* = 0.170, std d* = 0.075.
  - Range: d* ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}; modal value d* = 0.15.
  - Std d* < 0.1 → consistent stationarity structure across time periods. The ATR series requires fractional differencing of order ≈ 0.15 to 0.20 regardless of the training-fold partition, indicating that the persistence of the volatility-range process is a stable property rather than a regime-conditional one. No fold required d* = 1.0 (full integer differencing).
- **Methodological note.** Feature-selection turnover (high) and FFD-d* turnover (low) tell a coherent story: the noise-versus-signal ratio for individual features changes across regimes (only `eth_btc_ratio`, `skewness`, and `bb_width` clear 50%), but the underlying time-series property that motivates fractional differencing is stable. Stability of d* validates the FFD-only-on-ATR design choice in 3.6.1; instability of feature selection validates the use of multi-model MDA over single-model SFI in 3.6.3.

### 4.6 Symbolic Extraction Results (Q4 answer)

- **Extraction summary.**
  - Fold used: split 27 (groups 6 and 7), `fold_selection="last"`. Test set covers 2024-08-21 to 2026-04-19, the most recent CPCV partition; closest to deployment scenario. Locked-run KAN F1 on this fold: **0.5293**.
  - **Top-5 stable features by CPCV selection frequency all reach the extraction step** (the previous locked configuration suffered a `skewness` subset bug; the expanded 73-feature MDA pool resolves it). Top-5 ranking entering extraction: `eth_btc_ratio` (75%), `skewness` (71%), `bb_width` (57%), `us2y` (43%), `kurtosis` (43%).
  - Training sample size: 530 train + 133 val (after the 80/20 split on the split-27 training fold). 5 features in the input matrix.
  - **Architecture (CPCV-matched, single hidden layer).** The locked Optuna study did not produce tuned KAN hyperparameters for split 27, so the data-aware fallback fires. Final architecture `[5, 2, 2]`: 5 input features → 2-unit hidden layer → 2-unit binary output. 14 active edges, ~84 spline parameters for 530 training samples (parameter-to-sample ratio 6.3×, comfortably above the 5× floor).
- **Training diagnostics.**
  - Phase 1 (Adam, 600 steps, weight decay 1e-3, noise std 0.05): final train accuracy 0.6434, **validation accuracy 0.6241**.
  - **Phase-1 53% gate PASSED.** The pre-prune val_acc of 0.6241 clears the 0.53 threshold by 9.4 percentage points. This reverses the previous locked configuration, in which the gate failed at val_acc 0.4812 (and the pipeline logged a warning but continued to symbolification per the design). The Phase-1 gate is no longer a caveat for this extraction.
  - Phase 2a (LBFGS warmup, 20 steps, no regularisation): train 0.6849, val 0.5865.
  - Phase 2b (LBFGS sparsity, 20 steps, lambda=2e-3): train 0.7075, val 0.5564.
  - Grid extension skipped (n_train=530 too small for grid 3 → 5 expansion).
  - Detailed phase-by-phase logs in Appendix F.
- **Pruning results.** **Pruned architecture: `[[5, 0], [2, 0], [2, 0]]`** in PyKAN's `(sum_units, mult_units)` per-layer notation. **No edges pruned**: pre-prune edge analysis at threshold 0.01 finds 14 of 14 active (100% survival). The model's edge-importance distribution does not contain any below-threshold edges, so structural sparsification offers no compression at this configuration. Post-prune validation accuracy: 0.5564 (unchanged from pre-prune, since no edges were removed).
- **Symbolification rate: 100% (14 of 14 edges).** All edges pass the constant-skip + brute-force fallback at the 0.30 R² threshold. R² distribution across the 14 symbolified edges: min 0.7768, median 0.9540, max 0.9998. Top-five edges by R²:
  - Edge (0, 0, 1): `cos`, R² = 0.9998.
  - Edge (0, 4, 1): `sin`, R² = 0.9995.
  - Edge (0, 2, 0): `tanh`, R² = 0.9988.
  - Edge (0, 0, 0): `x²`, R² = 0.9980.
  - Edge (0, 4, 0): `cos`, R² = 0.9976.
- **Surviving features (5; no subset-bug exclusion this run).**
  - `eth_btc_ratio` (external / crypto-macro group, CoinMetrics-sourced ETH/BTC alt-rotation signal). Selected in **75%** of CPCV folds at the 28-fold level; **most stable feature in the dataset**.
  - `skewness` (TA / distribution-shape group, rolling 21-day third moment of log returns). Selected in **71%** of CPCV folds.
  - `bb_width` (TA / volatility group, Bollinger Band width). Selected in **57%** of CPCV folds.
  - `us2y` (external / macro group, US 2-year Treasury yield from FRED DGS2). Selected in **43%** of CPCV folds. **First macro-rate feature to enter the symbolic representation** in any locked configuration.
  - `kurtosis` (TA / distribution-shape group, rolling 21-day fourth moment of log returns). Selected in **43%** of CPCV folds.
  - **Group breakdown.** 3 TA (skewness, bb_width, kurtosis), 1 crypto-macro (eth_btc_ratio), 1 macro (us2y), 0 on-chain, 0 lag, 0 from any other group. **TA dominates the surviving set** (3 of 5), reversing the previous locked configuration where TA contributed only 1 of 4 (chaikin_osc).
- **Pre-symbolic vs. post-symbolic accuracy.**
  - Pre-symbolic (B-spline KAN, post-prune): **55.64%**.
  - Post-symbolic (closed-form formula, after affine fine-tuning): **60.15%**.
  - **Δ = +4.51 percentage points (post exceeds pre by 4.5 pp).**
  - Reading: the closed-form approximation *improves* validation accuracy over the trained spline network. The 14-function symbolic library matches the learned B-spline shapes well enough (median edge R² = 0.9540) that the symbolic substitution acts as a smoothing regulariser; the affine fine-tuning step (30 LBFGS steps at lr=4e-4) then tunes the 5 affine parameters per edge to the validation distribution. The post-symbolic 60.15% sits 10.2 percentage points above the 50% baseline and 7.2 percentage points above the 53% Phase-1 gate. The formula should therefore be read as **a tightened version of the KAN's decision boundary**, not as a degraded surrogate.
  - This reverses the previous locked configuration where post (52.63%) sat 4.5 pp below pre (57.14%). The reversal is the joint effect of three changes: the Phase-1 gate now passes pre-prune (Adam endpoint at 0.6241 rather than 0.4812 vs. KAN's harder landing in the previous noise-injection trajectory); the symbolification cleanly fits all 14 edges (median R² 0.9540, vs. the previous run where one edge was left as a spline with R² 0.7871); and the affine fine-tuning step lands the post-prune model in a basin that generalises slightly better than the spline endpoint.
- **The decision function.** The extracted closed-form expression. Numbered equation in the thesis:

```
decision(x) =
  - (130/211) sin( (220/933) (305/103 - 6103·eth_btc_ratio/940)²
                   + (15/976) (-195·us2y/23 - 421/953)²
                   - (1361/645) cos(1994·kurtosis/555 - 842/563)
                   + (713/946) cos(1446·skewness/461 - 2507/831)
                   - (1079/434) tanh(779·bb_width/375 - 745/677)
                   - 3644/809 )
  + (353/329) sin( (1/723) (-457·bb_width/53 - 1626/823)³
                   - (11/644) (-9425·skewness/953 - 441/377)²
                   - (915/796) sin(519·kurtosis/146 + 2279/705)
                   + (71/636) sin(1157·us2y/366 + 89/35)
                   + (98/283) cos(2333·eth_btc_ratio/461 + 1774/239)
                   + 6288/931 )
  - (1031/998) sin( (1/677) (-457·bb_width/53 - 1626/823)³
                    - (7/384) (-9425·skewness/953 - 441/377)²
                    - (687/560) sin(519·kurtosis/146 + 2279/705)
                    + (61/512) sin(1157·us2y/366 + 89/35)
                    + (17/46)  cos(2333·eth_btc_ratio/461 + 1774/239)
                    + 719/215 )
  - (994/885) cos( (233/982) (305/103 - 6103·eth_btc_ratio/940)²
                   + (11/711) (-195·us2y/23 - 421/953)²
                   - (1257/592) cos(1994·kurtosis/555 - 842/563)
                   + (135/178) cos(1446·skewness/461 - 2507/831)
                   - (708/283) tanh(779·bb_width/375 - 745/677)
                   + 6359/991 )
  + 1007/546

P(up | x) = 1 / (1 + exp(-decision(x)))
```

  - Coefficients are post-`sympy.nsimplify(tolerance=1e-3)` rationals. `sympy.simplify` timed out at 30 seconds and the unsimplified form is shown above. The underlying floats are stored alongside in `sympy_objects` for sensitivity analysis (numerical sensitivity bullet below).
- **Functional-form pattern.** **The formula uses five distinct primitives: `sin`, `cos`, `tanh`, the squared term `(...)²`, and the cubed term `(...)³`.** This is a substantively different pattern from the previous locked configuration's `sin`, `cos`, `exp`, `atan`, `x²` set:
  - `exp` and `atan` are no longer present (they were the two saturating monotonic primitives in the previous formula, attached to `eth_btc_ratio` and `exchange_supply_pct`).
  - `tanh` enters as a bounded smooth saturator, attached to `bb_width` in both of the outer `sin` terms with the `²`-and-cosine inner argument.
  - `(...)³` enters as a non-monotonic odd-symmetric primitive, attached to `bb_width` in both of the outer `sin` terms with the `³`-and-sine inner argument; this is the first time a cubic primitive has survived the R² threshold in any locked configuration.
  - The `(...)²` primitive attaches to `eth_btc_ratio` and `us2y` in two of the four outer terms and to `skewness` in the other two; the squared transform is the dominant non-trigonometric primitive.
  - The four outer terms split into two structural classes: two outer `sin` terms with the squared / `tanh` inner argument, and two outer `cos`/`sin` terms with the cubed / sine inner argument. The KAN learned two complementary internal representations of the decision boundary, then averaged them through the four-term outer combination.
- **Interpretive note (factual; economic interpretation deferred to 5.3).** Each surviving feature appears in **exactly four terms** of the formula (formula term-structure analysis confirms n_terms_in_formula = 4 for each of `eth_btc_ratio`, `skewness`, `bb_width`, `us2y`, `kurtosis`). The five input features therefore contribute roughly equally to formula complexity, with no single feature dominating the structural footprint.
- **Numerical sensitivity at the dataset mean.** Partial derivatives evaluated at the mean of each feature, scaled by the feature's standard deviation, give the per-σ effect on the decision function (logit):

| Feature | mean | std | d/dx at mean | σ-effect on logit | σ Δp (linearised) |
|---------|-----:|-----:|-----:|-----:|-----:|
| `eth_btc_ratio` | +0.0461 | +0.0247 | -17.6870 | -0.4367 | -0.1092 |
| `skewness` | -0.0559 | +1.0171 | -0.8192 | -0.8333 | -0.2083 |
| `bb_width` | +0.2758 | +0.1692 | -7.3720 | -1.2474 | -0.3119 |
| `us2y` | +2.0931 | +1.4924 | +6.8593 | +10.2365 | +2.5591 |
| `kurtosis` | +2.0727 | +3.2182 | +3.2915 | +10.5927 | +2.6482 |

  - **`us2y` and `kurtosis` dominate the formula in absolute σ-effect terms** (+10.24 and +10.59 respectively on the logit). At p=0.5, a one-σ increase in either feature would (linearised) shift P(up) by approximately +2.56 or +2.65 units, well outside the [0, 1] probability range; the honest reading is that both features drive the model into saturation, lifting P(up) from approximately 0.50 to approximately sigmoid(+10) ≈ 0.99996 in the up direction. The two features therefore act as conditional "strongly bullish" switches in the formula.
  - `bb_width` carries the largest negative σ-effect (-1.25 on the logit, linearised σ Δp ≈ -0.31), enough to drive P(up) from approximately 0.50 to approximately sigmoid(-1.25) ≈ 0.22. Direction reads naturally: wider Bollinger bands signal high volatility, and the model conditions on lower forward direction probability in such regimes.
  - `skewness` carries σ-effect -0.83 (linearised σ Δp ≈ -0.21); a one-σ increase in 21-day return skewness moves P(up) from approximately 0.50 to approximately sigmoid(-0.83) ≈ 0.30. The direction is consistent with the literature on right-skewed crypto-return regimes: positive skew (large isolated up-moves) is followed by mean-reverting weakness rather than persistence.
  - `eth_btc_ratio` σ-effect -0.44 (linearised σ Δp ≈ -0.11). A one-σ increase in the ETH/BTC ratio (alt-rotation strength) shifts P(up) from approximately 0.50 to approximately sigmoid(-0.44) ≈ 0.39; alt rotation away from BTC is mildly bearish on forward BTC direction, but the effect is the smallest meaningful magnitude in the formula.
  - **Scale check.** Unlike the previous locked configuration (where `chaikin_osc` produced a 10¹⁰ scaling artefact), all five surviving features have interpretable scales. The `bb_width`, `skewness`, `us2y`, and `kurtosis` columns are bounded or normally-scaled by construction; `eth_btc_ratio` is bounded above by the historical maximum (~0.13 in this sample). No rescaling caveat is needed for this formula.
- **On-chain comment (Q2 evidence).** **Zero on-chain features survive extraction.** This is a sharpening of the previous locked configuration's finding (where `exchange_supply_pct` was the single on-chain feature in the surviving four). The expanded 73-feature MDA pool reshuffled the top-5 stability set in a way that displaced `exchange_supply_pct` (which fell from 46% to a lower selection frequency once macro 14-day variants and lags 4 / 5 / 6 / 21 competed for the cap). Q2 answer hardens to **on-chain features do not exhibit stability under the AFML feature-selection regime applied here, and do not survive into the locked symbolic representation either.** The earlier locked-run observation that one on-chain feature could carry the formula's strongest economic effect is **regime-specific and does not generalise to the expanded feature universe.**
- **Surviving-feature group breakdown.**
  - 3 of 5 from TA (`skewness`, `bb_width`, `kurtosis`).
  - 1 of 5 from crypto-macro (`eth_btc_ratio`).
  - 1 of 5 from macro (`us2y`).
  - 0 of 5 from on-chain, mathematical (AFML Part 4), or lag groups.
- **Section closer.** "The symbolic extraction pipeline produces a closed-form decision function with **100% symbolification rate (14 of 14 edges)** on a five-feature subset (`eth_btc_ratio`, `skewness`, `bb_width`, `us2y`, `kurtosis`). **Post-symbolic accuracy is 60.15%, a 4.5 percentage-point gain over the pre-symbolic 55.64%** and 10.15 percentage points above the 50% baseline. The formula uses five primitives (`sin`, `cos`, `tanh`, `x²`, `x³`) and contains four outer terms, with each surviving feature contributing equally. The closed-form approximation *improves* accuracy relative to the trained spline network, producing a humanly auditable expression in which two features (`us2y` and `kurtosis`) act as saturating conditional switches and one feature (`bb_width`) carries the largest interpretable negative effect. Q4 is answered in the affirmative with no methodological caveat: the `skewness` subset bug from the previous configuration is resolved, the Phase-1 53% gate is cleared at val_acc = 0.6241 before pruning, and all 14 edges symbolify cleanly. RQ4 conclusion: closed-form interpretability is achievable, informative, and accuracy-preserving (in fact accuracy-enhancing) even in a regime where DSR < 0.95 across all benchmark models; the structural transparency of the decision function is the primary contribution and stands independent of the DSR result."

---

## 5. Discussion

> **T-R7.** Interpret. Link findings back to the mechanisms in Section 3.

### 5.1 Interpreting the results (Q1, Q3 evidence)

- **The joint DSR / PBO finding.** No model achieves DSR ≥ 0.95 (top trained model is LSTM at 0.3925; buy-and-hold posts a higher median Sharpe of 2.2321 but is not included in the DSR computation since it does not enter the selection pool). PBO sits in the moderate-overfitting regime at 0.3714, with 13 of 35 IS/OOS partitions seeing the IS-winner underperform the OOS median. The combination is informative: rankings are partially reproducible, the LSTM bootstrap CI (0.4062, 1.5380) is the only one among trained models that stays strictly positive, but the model that any selection rule picks does not achieve a Sharpe that survives multiple-testing correction. The honest reading is "LSTM is the most stable choice in this six-model cross-section but does not clear the multiple-testing-corrected significance threshold, and a naive buy-and-hold strategy on the same data posts a median Sharpe roughly 1.7× the LSTM figure". This is a different position than either DSR or PBO alone could deliver: we are not in a regime where rankings are random (PBO ≈ 0.5) or anti-predictive (PBO > 0.5), but the model cross-section is not robust enough (PBO ≥ 0.3) to support a confident pick, and even the most defensible pick does not earn the significance label.
- **Locating the variance.** The leave-one-out PBO (Section 4.4) shows that removing Logistic Regression most reduces PBO (to 0.143, a -0.229 shift), followed by KAN (0.229, -0.143) and XGBoost (0.257, -0.114). Removing AR Logistic or Random Forest does not change PBO. **Removing LSTM raises PBO to 0.429 (+0.057)**, making LSTM the only model whose presence stabilises the cross-section. This is an unexpected asymmetry. Two implications follow. First, the IS/OOS instability is concentrated in three models (Logistic Regression, KAN, XGBoost) whose path-Sharpe profiles are similar enough to rotate the IS-best slot across sub-path partitions. Second, LSTM acts as a fixed anchor: with the lowest path-Sharpe std among trained models (0.8284), it is consistently the IS-winner, and its removal lets the rotation among the other three models propagate freely. This reverses the previous locked configuration's reading where LSTM was the variance source (in the smaller 62-feature universe, removing LSTM dropped PBO to 0.000); the reversal is a real result of the 73-feature reshuffle, in which LSTM's wider feature pool sharpens its IS dominance while Logistic Regression's expanded basis introduces it as a destabilising contender.
- **Consistency with EMH.** The DSR < 0.95 finding aligns with semi-strong-form EMH (Fama 1970) for BTC daily direction: under leakage-free evaluation with multiple-testing correction, no architecture in the comparison demonstrates a statistically significant predictive edge. The fact that **buy-and-hold posts a higher median Sharpe (2.2321) than any trained model** strengthens this reading: in the 2015 to 2026 BTC sample, the unconditional long position dominates every directional-trading strategy in median Sharpe terms, suggesting that the asset's overall trajectory carries information that direction-prediction strategies discard by going neutral or short in volatile regimes. This is the intellectually honest answer to Q1 and consistent with `chassot_audrino_2026`: properly fitted baselines are hard to beat, and the literature's ML-superiority results came from suboptimal fitting schemes that handicapped the baselines. **Cite:** `fama_1970`, `chassot_audrino_2026`.
- **Conservative Predictions framing.** `nabar_shroff_2023`: in low-signal regimes, abstention beats prediction. The bet-sizing threshold operationalises this: predictions with `p ≈ 0.50` produce `bet ≈ 0`, structural abstention without a separate model. The locked run's mean P(Up) for KAN is 0.5596 against an empirical base rate of 0.5703 (Δ = -0.0107, well within the ±0.03 calibration tolerance), so the calibration audit does not flag KAN; the bet-sizing curve is operating on well-calibrated probabilities, and the negative DSR finding is therefore a statement about signal strength, not about probability quality. Three other models are now flagged by the calibration audit (AR Logistic at -0.0320, XGBoost at -0.0340, LSTM at -0.0366), all under-predicting the empirical base rate by more than 3 percentage points. This is a deterioration from the previous locked configuration, where only AR Logistic exceeded the tolerance, and is consistent with the broader picture of the expanded feature universe introducing more variability into the trained-model probability distributions.
- **Three-version pipeline evolution (advisor narrative).** Inflated ROC-AUC → degenerate all-Down predictions → AFML. Two specific failure modes motivated AFML adoption: uninformative fixed-time-horizon labels and structural leakage that standard splits cannot prevent. The locked-run results show that even after AFML closes both leakage channels, the residual signal is too weak for any compared model to clear DSR = 0.95; this is the AFML pay-off. The methodology produces honest results (PBO 0.37 in the moderate-overfitting regime) rather than the inflated metrics that earlier (leaky) pipeline versions produced.

### 5.2 KAN performance in context (Q3, Q4 evidence)

- **KAN ranks 5th of 6** in median Sharpe (0.3834), ahead only of AR Logistic (-0.1864) and behind LSTM (1.3377), XGBoost (0.9176), Random Forest (0.6919), and Logistic Regression (0.5637). Buy-and-hold's 2.2321 sits above the entire trained-model cross-section. KAN's positioning shifts substantially from the previous locked configuration, where it ranked 2nd of 6 on median Sharpe (0.9879). The drop is consistent with the broader cross-section reshuffle from the 62-to-73 feature expansion: KAN's previous locked formula relied on `chaikin_osc` and `exchange_supply_pct` (neither of which now survives into the symbolic representation), so the model's expressive advantage on the smaller feature pool was tied to features that the expanded pool de-prioritised.
- **Classification-side metrics remain competitive.** KAN posts mean accuracy 0.5415 (second only to XGBoost's 0.5336 by a hair within rounding, and ahead of Logistic Regression's 0.5332, Random Forest's 0.5331, AR Logistic's 0.5292, LSTM's 0.5259) and mean AUC 0.5211 (second only to Logistic Regression's 0.5261, ahead of all other trained models). The DeLong test confirms KAN's pooled AUC of 0.5151 is the highest in the cross-section, but no pair clears α = 0.05 (Section 4.3). The reading: KAN remains a strong classifier on the expanded feature universe, but classification quality and financial performance are now decoupled.
- **Tension to interpret: classification-side strength vs. last-place trading performance.** KAN posts competitive classification metrics yet ranks 5th of 6 on median Sharpe, with the widest bootstrap CI in the cross-section (-0.3750, 2.0819) and the deepest median max drawdown (-0.2774) among trained models other than AR Logistic. The interpretation: the bet-sizing transformation is no longer amplifying KAN's classification edge into financial outperformance the way it did in the smaller-feature configuration. With the expanded feature pool, KAN's per-split probability distribution acquires more variance (mean P(Up) at 0.5596 is within the calibration tolerance, but path-Sharpe std at 1.4409 is the second-highest among trained models), and the resulting bets are larger in absolute size but more wrong in absolute direction on a per-path basis.
- **Statistical-vs-economic gap.** KAN's mean AUC of 0.5211 is second-best in the cross-section, but the DeLong test against the closest competitor (Logistic Regression at 0.5261) returns p = 0.50. The classification-side superiority is indistinguishable from sampling noise at standard significance levels, and the bet-sizing transformation does not rescue the financial ranking. This is a quieter result than "KAN is a top classifier" and substantially quieter than the previous locked configuration's "KAN ranks 2nd by both classification and Sharpe", but it is what the locked data actually supports.
- **Contrast with `oad_kasper_2025`.** KASPER reports R² = 0.89 and Sharpe = 12.02 on individual stocks with regime detection. This thesis posts median Sharpe = 0.38 for KAN on BTC daily direction without a regime layer. The differences are explanatory rather than competitive: KASPER uses regression on individual stocks with regime-conditional architectures; this thesis uses classification on a single asset (BTC) with no regime layer and applies AFML's full statistical-correction stack (DSR, PBO, DeLong) which KASPER does not. The 30× Sharpe gap reflects asset class, target type, and evaluation rigour rather than KAN architectural capacity.
- **Contrast with `cho_lee_kim_2025`.** The VIX KAN paper extracts symbolic formulas that reveal mean-reversion and leverage effects, validated against domain knowledge. This thesis's symbolic formula reveals a five-feature mixed-primitive structure through `sin`, `cos`, `tanh`, `x²`, `x³` primitives, on a target (BTC daily direction) that is not mean-reverting and where domain knowledge does not directly validate any specific functional form. The presence of `tanh` and `x³` primitives in the locked-run formula (neither of which appeared in the previous locked configuration's `sin`, `cos`, `exp`, `atan`, `x²` set) suggests the KAN's learned decision boundary has more diverse curvature than the smaller-feature run captured. The symbolic-extraction machinery transfers across asset classes; the interpretive yield depends on whether the underlying target has a structure that domain knowledge can corroborate.

### 5.3 Symbolic extraction as contribution (Q4 evidence)

- **Interpretability is a separable contribution from predictive accuracy.** The locked-run formula has **100% symbolification rate** and a closed-form expression on five features; its post-symbolic accuracy of **60.15%** is a 4.5 percentage-point *gain* over the spline KAN's pre-symbolic 55.64% and 10.15 percentage points above the 50% baseline. The closed-form decision function provides three artefacts a black-box model cannot: per-feature symbolic derivatives (Section 4.6 numerical sensitivity), term-structure decomposition (each feature appears in 4 of 4 outer terms), and an audit trail from input to probability that a domain expert can inspect. The fact that the symbolic form outperforms the spline form is itself an interesting finding: it suggests the symbolic substitution acts as a smoothing regulariser on the per-edge B-spline shapes, and the affine fine-tuning step then tunes the per-edge scale and bias to the validation distribution.
- **The post-symbolic accuracy gain is real, but small and fold-specific.** The 14-function symbolic library covers smooth standard primitives, the affine fine-tuning lands the post-prune model in a basin that generalises slightly better than the spline endpoint, and the median per-edge R² of 0.9540 across all 14 edges indicates the library matches the learned shapes faithfully. The 4.5 percentage-point gain should not be over-read: the locked extraction is a single-fold (split 27) result, the validation set is 133 observations, and the gain falls inside the range of seed-to-seed and split-to-split noise observed elsewhere in the pipeline. The honest finding is "closed-form symbolic representation does not cost validation accuracy on this dataset, and may improve it slightly on the most recent CPCV fold". Per-fold extraction (L3 in 5.4) would let the analysis distinguish between fold-specific gains and a robust symbolic-smoothing effect.
- **Mixed-primitive functional form.** The decision function uses five distinct primitives (`sin`, `cos`, `tanh`, `x²`, `x³`), arranged in four outer terms that each pool the five input features through different combinations. The previous locked configuration used `sin`, `cos`, `exp`, `atan`, `x²`; the present run replaces `exp` and `atan` (the two saturating monotonic primitives that attached to `eth_btc_ratio` and `exchange_supply_pct`) with `tanh` and `x³`. The `tanh` primitive attaches to `bb_width` and is the more conventional smooth saturator; the `x³` primitive attaches to `bb_width` as a cubed inner argument inside the two outer `sin` terms and is the first time a cubic primitive has survived the R² threshold in any locked configuration. The presence of `tanh` and `x³` reflects the genuine flexibility of the KAN representation: the library tries roughly 14 candidates per edge and keeps the highest-R² match, so different feature sets and different folds can land on substantively different primitive combinations.
- **Numerical sensitivity at the dataset mean.** **`us2y` and `kurtosis` both post per-σ effects on the logit of approximately +10** (10.24 and 10.59 respectively), driving the decision function into upper saturation in a single σ bump from the mean. Both act as conditional "strongly bullish" switches rather than smoothly graded inputs. `bb_width` carries the largest negative σ-effect (-1.25, linearised σ Δp ≈ -0.31, equivalent to moving P(up) from approximately 0.50 to approximately 0.22), aligning with the conventional reading that wider Bollinger bands signal volatility, which the model conditions on as bearish on forward direction. `skewness` (-0.83, σ Δp ≈ -0.21) and `eth_btc_ratio` (-0.44, σ Δp ≈ -0.11) carry smaller negative effects. The economically meaningful sensitivity ranking is therefore `kurtosis` ≈ `us2y` (saturating positive) > `bb_width` (moderate negative) > `skewness` (modest negative) > `eth_btc_ratio` (smallest meaningful magnitude). Unlike the previous locked configuration (where `chaikin_osc` produced a 10¹⁰ scaling artefact), **no surviving feature in the present run has a scale problem**; all five columns are bounded or normally scaled by construction, so the sensitivity table is interpretable as economic effect throughout.
- **The on-chain finding sharpens.** **No on-chain feature survives extraction** in the locked configuration. The previous locked run retained `exchange_supply_pct` as the single on-chain feature in the surviving set, where it carried the formula's largest economic effect (-4.55 σ-effect on logit, saturating bearish). With the 73-feature pool, `exchange_supply_pct` falls out of the top-5 stable set, displaced by `bb_width`, `us2y`, and `kurtosis`. This is a stronger negative answer to Q2 than the previous configuration provided: the on-chain free lunch fails not only at the cross-fold stability level (no on-chain feature reaches 50% selection frequency across the 28 folds) but also at the deployment-fold level (no on-chain feature reaches the symbolic decision function on split 27). The earlier locked-run observation that one on-chain feature could carry the formula's strongest effect was regime-specific and does not generalise.
- **Contrast with `cho_lee_kim_2025`.** The VIX KAN paper's formulas reveal mean-reversion and leverage effects that domain knowledge corroborates directly. This thesis's formula reveals a five-feature mixed-primitive structure on a target (BTC daily direction) where domain knowledge does not directly validate any specific functional form. **Two of the five surviving features (`us2y`, `kurtosis`) act as saturating switches in the formula**, which is a structural pattern that domain knowledge can begin to engage with: high US 2-year Treasury yields and high return kurtosis (fat-tailed regimes) both pushing the model toward strong "up" predictions is consistent with a "crisis-then-recovery" reading where the model has learned to interpret volatility and rate-tightening signals as buying opportunities. This is a softer interpretive yield than the VIX KAN paper's mean-reversion result, but stronger than the previous locked configuration's mostly-cyclic formula. The symbolic-extraction machinery transfers across asset classes; the interpretive yield depends on whether the underlying target has a structure that domain knowledge can corroborate.
- **Limitations.**
  - **Fold-specific.** Different folds may produce different formulas with different surviving features; per-fold extraction is L3 in 5.4.
  - **Small sample.** 530 training observations on split 27 is not large; PyKAN's per-edge R² fits stabilise on this size but recover signal that the spline KAN already captured rather than discovering new structure.
  - **R² threshold sensitivity.** The 0.30 threshold (lowered from 0.50) admits one edge at R² = 0.7768, which is moderately rough. A stricter 0.50 threshold would have kept the same set of symbolified edges (the minimum R² 0.7768 still clears 0.50), so the 100% symbolification rate is robust to a 0.50 cut on this run; the 0.30 threshold remains as a configurable parameter for runs where the data lands less cleanly on the symbolic library.
  - **The `skewness` subset bug is RESOLVED.** The previous locked configuration's top-5 stability ranking included `skewness`, but the per-fold processed feature subset for split 27 did not contain it; the locked formula therefore answered Q4 conditional on a four-feature subset. With the 73-feature pool, the per-fold processed feature subset on split 27 contains all five top-5 stable features, and the locked formula is now built on the strict top-5 stable subset as the methodology in 3.12.3 prescribes. The earlier methodological inconsistency is closed in the present run.
  - **Architectural parity has been resolved.** The single-hidden-layer / `width1 ≤ 6` configuration matches the CPCV-benchmarked KAN, so the formula describes the architecture the benchmark numbers describe, not a simplified surrogate.

### 5.4 Limitations (five honest, from defense L1 to L5)

- **L1. Computational power.** Currently 2 to 3 seeds, 30 trials, KAN width capped at 6. With more compute: more seeds, more trials, wider HP ranges. The N=8 / 28-split configuration alone roughly doubles the per-experiment runtime relative to N=6 / 15 splits.
- **L2. Daily OHLCV only.** Discards intraday microstructure. Pipeline is timeframe-agnostic; hourly extension would multiply event count and unlock microstructure features.
- **L3. Symbolic extraction is fold-specific.** The locked formula is the result of running the extraction on split 27 only. Per-fold extraction (28 formulas, one per CPCV split) would let the analysis count which features and primitives recur (structural signal vs. one-off noise). Of particular interest: would `us2y` and `kurtosis` still post saturating positive σ-effects on earlier CPCV folds with different regime composition, or are those switches specific to the 2024-08 to 2026-04 deployment window? The `skewness` subset-bug issue noted in earlier drafts is resolved in the locked configuration (all top-5 stable features now reach the extraction step), so this limitation is now purely about the single-fold scope rather than about any methodological inconsistency.
- **L4. KAN width cap is binding for interpretability.** The `width1 ≤ 6` cap is set so the extracted symbolic formula stays humanly readable, since each surviving width1 unit becomes one additive term plus its interactions in the closed-form expression. The locked configuration's pruned architecture `[[5, 0], [2, 0], [2, 0]]` uses width1 = 5 and width2 = 2, producing 14 active edges and a four-outer-term decision function that fits one printed page in the thesis. A larger width1 cap would let the model fit more complex patterns but would also produce a formula too long to interpret. This is an explicit interpretability-vs-capacity trade-off, not a generic constraint imposed by sample size or compute.
- **L5. No regime-conditional analysis.** BTC has had approximately 5 distinct regimes over 2014 to 2026 (the 2017 bubble, 2018-2019 bear, 2020 COVID crash, 2021 bull, 2022 FTX-era contraction, 2023-2026 recovery). Full-sample Sharpe averages over them. A regime-conditional analysis is left for future work, especially given the joint reading "DSR < 0.95 + PBO 0.37 moderate": full-sample non-significance does not rule out within-regime predictability, and the buy-and-hold comparison (median Sharpe 2.23 with max drawdown -0.9998) makes the regime-conditional question financially salient even when full-sample-significance is unattainable.

---

## 6. Conclusion

> **Cochrane:** Do not restate all findings. **T-R16:** End with resonance, not a hedge. Target ≤ 10% of the textual part (approximately 3.5 pages).

- One paragraph stating the contribution: full AFML applied to BTC, six-model benchmark, first KAN symbolic formula extraction in this regime, expanded 73-feature universe with multi-model MDA selection over four feature families (TA, mathematical, external macro / crypto-macro / on-chain, autoregressive lags).
- One paragraph stating the headline: under leakage-free evaluation with multiple-testing correction, the literature's 85% to 95% accuracy claims do not survive. Top DSR among trained models is LSTM at 0.3925, far below the 0.95 significance threshold. PBO at 0.3714 places model selection in the moderate-overfitting regime: 13 of 35 IS/OOS partitions see the IS-winner underperform the OOS median, and the leave-one-out PBO analysis shows that the instability is concentrated in Logistic Regression, KAN, and XGBoost rotating in the IS-best slot, while LSTM is the only model whose presence stabilises the cross-section (removing LSTM raises PBO to 0.429). Buy-and-hold posts a higher median Sharpe (2.2321) than every trained model, with the structural cost of a median max drawdown of -0.9998: directional-trading strategies on BTC daily direction trade absolute return for downside protection rather than producing alpha relative to the asset's overall trajectory. This is the honest answer to Q1: under AFML evaluation, no architecture demonstrates a statistically significant predictive edge over the multi-model selection pool, and the most defensible trained-model pick (LSTM, by being the only model with a strictly positive bootstrap CI) does not clear multiple-testing correction.
- One paragraph on what does survive: the symbolic extraction pipeline produces a humanly auditable closed-form decision function on five features (`eth_btc_ratio`, `skewness`, `bb_width`, `us2y`, `kurtosis`), with **100% symbolification rate (14 of 14 edges)** and **post-symbolic accuracy of 60.15%, a 4.5 percentage-point gain over the pre-symbolic 55.64%**. The formula uses five primitives (`sin`, `cos`, `tanh`, `x²`, `x³`); `us2y` and `kurtosis` act as saturating positive switches on the logit with σ-effects of approximately +10 each, while `bb_width` carries the largest interpretable negative effect (σ-effect -1.25, σ Δp ≈ -0.31 from the dataset mean). No on-chain feature survives extraction in the locked configuration: the Q2 "on-chain free lunch" hypothesis fails at both the cross-fold stability level (no on-chain feature reaches 50% selection frequency across the 28 folds) and the deployment-fold level (no on-chain feature reaches the symbolic decision function on the most recent CPCV partition). The closed-form representation costs nothing in validation accuracy on this run (in fact gains 4.5 pp) and produces an artefact, with symbolic derivatives, term decomposition, and audit trail, that no black-box model in the comparison can produce. Interpretability is a separable contribution from predictive accuracy and does not need to wait for predictive significance to be valuable.
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