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
- Directly relevant to this thesis: the negative results (top DSR 0.2198, well below 0.95) align with this finding. Properly evaluated baselines are hard to beat, especially in a weak-signal regime like BTC.
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
- **Selects features via the `n_top_features` parameter and the `feature_selection_strategy` parameter.** The current locked configuration uses `n_top_features=3` and `feature_selection_strategy="stability"`, ranking features by cross-fold MDA selection frequency over the 28 folds and keeping the top 3. Earlier locked configurations used `n_top_features=5` with the same stability strategy. The parameter is exposed on `run_symbolic_extraction` so the user can trade input-dimensionality (more features ⇒ more potential signal) against formula readability (fewer features ⇒ more compact closed form). The locked-run choice of 3 produces a `[3, 3, 2]` architecture with 15 active edges that fit one printed page; raising to 5 would produce a `[5, 2, 2]` architecture with 14 active edges and a longer formula. The `"stability"` strategy is the configurable default; alternatives (stored per-fold MDA selection, per-fold ranking) are available but not used in the locked run.
- 80/20 chronological split into model-train + validation.
- Tanh normalization fitted on training portion: `z = tanh((x - mean) / (std + 1e-8))`.

#### 3.12.4 Three-phase PyKAN training (Algorithm 1, Step 1, from `cho_lee_kim_2025`)

| Phase | Optimizer | Steps | Key feature |
|-------|-----------|-------|-------------|
| 1. Adam | Adam (lr=1e-3, wd=1e-3) | 600 | Gaussian noise injection (`std=0.05`) clamped to [-1, 1]; dropout-like regularizer; early stopping on val loss |
| 2a. LBFGS warmup | LBFGS (lr=0.01) | 20 | No regularization; light refinement |
| 2b. LBFGS sparsity | LBFGS (lr=0.01) | 20 | L1 + entropy regularization (`lamb=0.002`); encourages sparse activations |

- **Grid extension disabled** (`PYKAN_GRID_EXTEND=False`): with 540 training samples and 135 validation samples (training fold of split 27 after the 80/20 split, locked run), refining grid 3 to 5 adds parameters faster than data supports.
- **Dynamic majority-baseline accuracy gate.** After Phase 1 (Adam), the pipeline checks `val_acc` against `max(PYKAN_MIN_ACCURACY=0.53, val_majority_baseline + 0.01)`. The gate adjusts to the actual class balance in the validation fold rather than assuming the dataset-level base rate, so a fold with 60/40 balance requires `val_acc ≥ 0.61` while a balanced fold requires `val_acc ≥ 0.53`. The previous configuration used a fixed `val_acc ≥ 0.53` threshold; the dynamic gate is a methodology improvement that surfaces sub-baseline learning more reliably. If the gate fails, log a warning but continue. Symbolic extraction may yield constants when the underlying model has not learned signal. A second, structurally similar gate runs at the pre-prune stage using the same dynamic formula.
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
- Small sample (540 training observations and 135 validation observations after 80/20 split on split 27).
- R² threshold sensitivity (lowered to 0.30 to admit symbolic fits in weak-signal regime).
- Post-symbolic accuracy may be lower than pre-symbolic accuracy, and either may fall below the validation-fold majority baseline (the dynamic gate surfaces this case in the warning logs).
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

- "No model achieves DSR ≥ 0.95. The top DSR among trained models (KAN) is 0.2198, far below significance after correcting for selection bias across the six compared models. Buy-and-hold posts a higher median Sharpe (2.2722) than every trained model, illustrating how punishing the AFML correction layer is for trading strategies on a trending asset."
- "PBO = 0.629 lands in the **adversarial regime** (PBO > 0.5): in 22 of 35 IS/OOS partitions the in-sample best model underperforms the out-of-sample median. The IS-winner is the OOS-loser more often than not, which means model selection across this six-model cross-section is not just unreliable but actively misleading on more than half the sub-path partitions. Leave-one-out PBO sharpens the picture: every model's exclusion *reduces* PBO (`ar_logistic` -0.229, `lstm` -0.171, `kan` -0.143, `logistic` -0.114, `xgboost` -0.114, `random_forest` -0.086). No model anchors the cross-section in this configuration. The previous run's KAN-as-anchor finding (where excluding KAN raised PBO past the random-selection threshold) does not reproduce when the PBO baseline itself has moved past that threshold."
- "Despite the adversarial PBO, KAN is the only trained model with a strictly-positive bootstrap 95% CI on its median Sharpe (0.0971, 1.2413). Every other trained-model CI crosses zero. KAN therefore carries two findings simultaneously: it sits at the top of the trained-model financial ranking with bootstrap-strict-positive evidence, and it does not stabilise the cross-section under leave-one-out PBO."
- "The symbolic extraction pipeline produces a closed-form decision function with **100% symbolification rate** (15 of 15 edges) on three surviving features (`eth_btc_ratio`, `log_returns_lag6`, `natgas_ret_30`). Post-symbolic accuracy is **51.85%**, a 5.18 percentage-point *gain* over the pre-symbolic 46.67%, but both numbers sit at or below the 50% baseline: the PyKAN model retrained on the deployment fold did not extract signal that beats coin-flip. The configuration change driving this is `n_top_features=3` (down from 5), which restricts the input to the three most-stable features and forces the architecture to a narrower [3, 3, 2] topology. The Phase-1 majority-baseline gate (`max(0.53, base_rate + 0.01) = 0.5693`) fails at val_acc 0.4815, the pre-prune gate fails at val_acc 0.4667, and recovery in the LBFGS phases does not get past 50%. The Abs(·) non-differentiability flag from the previous configuration is gone: the v5 formula uses sin, cos, tanh, x², x³, and x⁴ as primitives, all smooth at the dataset mean."
- Roadmap: 4.2 model comparison, 4.3 classification, 4.4 financial, 4.5 stability, 4.6 symbolic.

### 4.2 Model Comparison (the main result, Cochrane)

- **Table.** Self-contained caption: "Model comparison ranked by median Sharpe over seven CPCV backtest paths, with buy-and-hold included as a naive baseline. DSR is computed against `n_trials=6` (number of compared models), correcting the observed Sharpe for selection bias and non-normal returns; DSR is not defined for buy-and-hold since it is not part of the model selection pool. The Sharpe confidence interval is a bootstrap 95% CI on the per-path median Sharpe. No model achieves DSR ≥ 0.95. Tiebreaker: standard deviation of path Sharpe ascending."

| Rank | Model | Median Sharpe | Sharpe CI (low, high) | Std Sharpe | DSR | Med Sortino | Med Calmar | Mean F1 | Mean Acc | Mean AUC | Med Max DD | Med Cum Ret | Med Win Rate | Med Profit Factor |
|------|-------|--------------:|----------------------:|-----------:|-----:|-----:|-----:|------:|------:|------:|------:|------:|------:|------:|
| 1 | Buy-and-Hold | 2.2722 | (1.8903, 2.4685) | 0.5768 | n/a | 2.3390 | 2.4111 | n/a | n/a | n/a | -0.9998 | 45896.39 | 0.5559 | 1.3007 |
| 2 | KAN | 0.4778 | (0.0971, 1.2413) | 0.7592 | 0.2198 | 0.2808 | 0.0640 | 0.4109 | 0.5330 | 0.5088 | -0.2563 | 0.0820 | 0.5407 | 1.1032 |
| 3 | AR Logistic | 0.3806 | (-0.0544, 1.2101) | 0.7355 | 0.1896 | 0.3894 | 0.0408 | 0.4666 | 0.5240 | 0.5204 | -0.2739 | 0.0852 | 0.5318 | 1.0687 |
| 4 | Random Forest | 0.2047 | (-0.5289, 0.9257) | 0.7480 | 0.1415 | 0.1839 | 0.0064 | 0.4364 | 0.5252 | 0.5142 | -0.3641 | 0.0199 | 0.5109 | 1.0416 |
| 5 | LSTM | 0.1808 | (-0.6234, 0.8350) | 0.8382 | 0.1354 | 0.1573 | 0.0094 | 0.4140 | 0.5211 | 0.5142 | -0.2481 | 0.0131 | 0.5105 | 1.0343 |
| 6 | XGBoost | 0.0484 | (-1.1278, 1.0358) | 1.1221 | 0.1063 | 0.0363 | -0.0371 | 0.4188 | 0.5144 | 0.4999 | -0.2696 | -0.1590 | 0.5029 | 1.0144 |
| 7 | Logistic Regression | -0.1545 | (-0.5976, 1.9359) | 1.3477 | 0.0707 | -0.1181 | -0.0271 | 0.4281 | 0.5183 | 0.5174 | -0.3512 | -0.0596 | 0.5018 | 0.9703 |

- **Key observations (state facts here, save interpretation for 5.1).**
  - **DSR.** Top DSR among trained models is KAN at 0.2198; all six trained-model values fall below 0.95. No model demonstrates predictive ability that survives correction for selection bias. The top three DSRs (KAN 0.2198, AR Logistic 0.1896, Random Forest 0.1415) span 0.08, with smaller gaps to LSTM (0.1354), XGBoost (0.1063), and Logistic Regression (0.0707). Buy-and-hold is excluded from the DSR computation because it is not an Optuna-tuned model and does not enter the selection pool. KAN's DSR moved from 0.2012 in the previous run to 0.2198 here, a small uptick, while XGBoost's DSR fell from 0.2376 to 0.1063 (a 0.13 drop) and Logistic Regression's fell from 0.1073 to 0.0707.
  - **Sharpe confidence intervals (key finding).** 5 of 7 entries have a 95% bootstrap CI that crosses zero. Two CIs stay strictly positive: buy-and-hold (1.8903, 2.4685) and **KAN (0.0971, 1.2413)**. KAN is the only trained model with bootstrap-strict-positive evidence that its median Sharpe is distinguishable from zero. The previous run had zero trained models with strictly-positive CIs (LSTM's CI had moved into zero-crossing territory under uniform seeds); the current configuration recovers a single trained-model strict-positive at KAN. This is a substantive positive finding for KAN that the headline PBO and DSR results do not capture.
  - **Buy-and-hold reference point.** Buy-and-hold posts the highest median Sharpe (2.2722) and the highest cumulative return (≈45,896× over the 2015 to 2026 period, reflecting BTC's trajectory from approximately $325 to roughly $108k). Its median max drawdown is -0.9998 (essentially -100%): the position experiences the full BTC drawdown profile, which is precisely what bet sizing and the dual-weighted loss are designed to mitigate. Trained models trade off return for downside protection: KAN's cumulative return of 0.0820 is roughly six orders of magnitude smaller than buy-and-hold's, but its max drawdown of -0.2563 is four times shallower. This is the financial-management framing in which the DSR result should be read.
  - **Ranking.** KAN leads the trained models with median Sharpe 0.4778; AR Logistic is third at 0.3806; Random Forest fourth at 0.2047; LSTM fifth at 0.1808; XGBoost sixth at 0.0484; Logistic Regression last at -0.1545 (the only model with a negative median Sharpe). The previous run's ranking (XGBoost, KAN, LSTM, Logistic, RF, AR Logistic) reshuffled materially: KAN moved up from second to first among trained models, AR Logistic moved up from last to third (and from a negative -0.1864 to a positive 0.3806), XGBoost dropped from first to sixth, Logistic Regression dropped from fourth to seventh (and to a negative median). The XGBoost demotion is the largest single change: its median Sharpe fell from 0.9483 to 0.0484, a 0.90 drop, and its DSR fell from 0.2376 to 0.1063.
  - **AUC compression.** All mean AUCs sit in the 0.4999 to 0.5204 band. Models are nearly indistinguishable at the classification level; bet sizing and the dual-weighted loss are what translate the tiny edges into the wider Sharpe spread. AR Logistic now posts the highest mean AUC (0.5204), narrowly ahead of KAN (0.5088) and the rest of the cross-section, but the entire AUC range across the six models is 0.020. The previous run's KAN-top-AUC slot is lost to AR Logistic; the DeLong test (4.3) confirms no AUC difference reaches significance.
  - **Q1, Q2 evidence (feature engineering).** AR Logistic posts a positive median Sharpe (0.3806) for the first time in any locked configuration; pure-momentum lags translate into competitive third-place financial performance in this run. Logistic Regression on MDA-selected features falls to last place and posts a negative median (-0.1545), reversing its previous-run climb to fourth: with 73 features in the MDA pool the linear hypothesis class is now noisy enough on the test paths that the bet-sizing translation actively destroys value relative to an unconditional position. Random Forest drops to fourth and XGBoost to sixth.
  - **Q3 evidence (KAN positioning).** KAN is the top trained model on median Sharpe (0.4778) for the first time, the top trained model on DSR (0.2198), and the only trained model with a strictly-positive bootstrap CI. Mean accuracy (0.5330) is mid-pack, mean AUC (0.5088) is competitive but not first, median win rate 0.5407 is highest among trained models. The bet-sizing distribution that translates calibrated probabilities into positions is doing the work here: KAN's calibration audit Δ of -0.0287 (Section 5.1) sits at the edge of the [WARN] tolerance but is still classed as ok. The leave-one-out PBO finding in Section 4.4 reads differently from the previous run, however: in v5 there is no anchor model, every exclusion reduces PBO, and KAN's exclusion drops PBO from 0.629 to 0.486. KAN is therefore the financially-best trained model in this run but is also part of the rotating IS-winners cluster contributing to the adversarial PBO regime.
  - **Std Sharpe outlier.** The Random Forest std Sharpe (0.7480) is now the lowest among trained models, narrowly ahead of AR Logistic (0.7355) and KAN (0.7592). Logistic Regression carries the highest std Sharpe (1.3477) and XGBoost the second-highest (1.1221), which is why both have CIs spanning more than two Sharpe units and both end below zero on the median. The wide CIs on Logistic Regression and XGBoost contribute to the adversarial PBO regime: these are the models that are rotating into and out of the IS-best slot across sub-path partitions, but their high path-Sharpe variance means the OOS median rarely confirms the IS lead.

### 4.3 Classification Metrics

- **Per-split F1 distribution.** Mean F1 ranges narrowly from 0.4109 (KAN) to 0.4666 (AR Logistic). The high-F1 / low-Sharpe combination for AR Logistic confirms that classification quality and bet-sizing translation are partially independent: pure momentum gets the average call right slightly more often than the engineered-feature models but its bet-sizing translation is more conservative, leaving a competitive but third-place median Sharpe.
- **Pooled AUC.** Per-model pooled AUC across all 28 splits sits in the 0.4954 to 0.5149 band. All models hover within two percentage points of random; the entire signal contest plays out inside an AUC range smaller than what most reviewers would consider economically material. XGBoost's pooled AUC (0.4954) sits below 0.5, the first time in any locked configuration that a trained model has posted a below-chance AUC.
- **DeLong pairwise AUC.** **0 of 15 pairs significantly different at α=0.05.** Pooled across splits, averaged across the 3 seeds per model:

| Pair | AUC_a | AUC_b | Δ AUC | z | p | Significant |
|------|-------|-------|-------|---|---|-------------|
| AR Logistic vs Logistic | 0.5117 | 0.5087 | +0.0030 | 0.3669 | 0.7137 | No |
| AR Logistic vs Random Forest | 0.5117 | 0.5070 | +0.0047 | 0.5482 | 0.5835 | No |
| AR Logistic vs XGBoost | 0.5117 | 0.4954 | +0.0163 | 1.9239 | 0.0544 | No |
| AR Logistic vs LSTM | 0.5149 | 0.5011 | +0.0137 | 1.3449 | 0.1787 | No |
| AR Logistic vs KAN | 0.5117 | 0.5057 | +0.0060 | 0.7715 | 0.4404 | No |
| Logistic vs Random Forest | 0.5087 | 0.5070 | +0.0017 | 0.2743 | 0.7838 | No |
| Logistic vs XGBoost | 0.5087 | 0.4954 | +0.0133 | 1.7250 | 0.0845 | No |
| Logistic vs LSTM | 0.5100 | 0.5011 | +0.0089 | 0.8355 | 0.4034 | No |
| Logistic vs KAN | 0.5087 | 0.5057 | +0.0030 | 0.4816 | 0.6301 | No |
| Random Forest vs XGBoost | 0.5070 | 0.4954 | +0.0116 | 1.9380 | 0.0526 | No |
| Random Forest vs LSTM | 0.5093 | 0.5011 | +0.0082 | 0.7935 | 0.4275 | No |
| Random Forest vs KAN | 0.5070 | 0.5057 | +0.0013 | 0.1925 | 0.8474 | No |
| XGBoost vs LSTM | 0.4978 | 0.5011 | -0.0034 | -0.3352 | 0.7375 | No |
| XGBoost vs KAN | 0.4954 | 0.5057 | -0.0103 | -1.4613 | 0.1439 | No |
| LSTM vs KAN | 0.5065 | 0.5086 | -0.0021 | -0.2036 | 0.8387 | No |

- **Pairwise summary.** **No pair clears α = 0.05.** AR Logistic posts the highest pooled AUC (0.5117 to 0.5149 depending on the comparison's seed-averaging context), narrowly ahead of LSTM (0.5011 to 0.5093), KAN (0.5057 to 0.5086), Logistic Regression (0.5087 to 0.5100), Random Forest (0.5070 to 0.5093), and XGBoost (0.4954 to 0.4978). Two pairs approach significance: Random Forest vs XGBoost at z = 1.9380 (p = 0.0526) and AR Logistic vs XGBoost at z = 1.9239 (p = 0.0544); both just miss the 0.05 threshold. The pattern: both near-significant pairs include XGBoost as the lower-AUC half, consistent with XGBoost's drop in this run. The previous configuration produced the same 0/15 finding; the absence of significant pairs is robust across the seeds and feature-selection-strategy reconfiguration.
- **Effect-size disclosure (T-R15: write for reviewers).** The entire AUC range across all six models is 0.020 (from XGBoost at 0.4954 to AR Logistic at 0.5149), the same order of magnitude as the seed-to-seed and split-to-split variation within any single model. The DeLong test confirms that none of these gaps survives a hypothesis test once cross-split correlation is accounted for.
- **Confusion matrices (compact, Appendix D for full).** Aggregated TP/FP/TN/FN per model. Most models post a slight class-1 bias consistent with the empirical 56.85% / 43.15% Up/Down base rate.
- **Multiple testing.** No Bonferroni or Benjamini-Hochberg correction is needed since no nominal p-value clears α = 0.05 in the first place. The 0 of 15 finding is robust to any sensible correction.

### 4.4 Financial Performance

- **Path-level Sharpe distribution.** Per-model row × 7 paths + median + std. KAN posts the highest median Sharpe among trained models (0.4778) with the second-lowest std (0.7592); its bootstrap 95% CI (0.0971, 1.2413) is the **only trained-model CI that stays strictly positive**. AR Logistic (median 0.3806, std 0.7355) sits second with a CI (-0.0544, 1.2101) that crosses zero very narrowly on the lower side. Random Forest (median 0.2047, std 0.7480) is third, CI (-0.5289, 0.9257). LSTM (median 0.1808, std 0.8382) is fourth, CI (-0.6234, 0.8350). XGBoost (median 0.0484, std 1.1221) is fifth with the second-widest CI (-1.1278, 1.0358). Logistic Regression (median -0.1545, std 1.3477) is the worst trained model with the widest CI (-0.5976, 1.9359), straddling zero by a substantial margin in both directions. Buy-and-hold (median 2.2722, std 0.5768) dominates the median ranking with the tightest CI (1.8903, 2.4685) of the seven entries.
- **Sortino and Calmar.** The downside-deviation-adjusted Sortino ratios reorder the trained models slightly: AR Logistic (0.3894), KAN (0.2808), Random Forest (0.1839), LSTM (0.1573), XGBoost (0.0363), Logistic Regression (-0.1181). AR Logistic's Sortino leads the trained models despite KAN's higher median Sharpe, suggesting AR Logistic's path-Sharpe distribution loads more variance on the upside than on the downside. KAN's Sortino is second. Calmar (Sharpe over max drawdown) compresses the trained models into a narrow -0.04 to +0.07 band, with KAN at 0.0640 leading among trained models. Buy-and-hold's Calmar (2.4111) is two orders of magnitude above any trained model, but its near-100% max drawdown means a single rebalancing decision during the deepest crash would wipe out the strategy.
- **DSR computation detail.** For KAN (top-ranked trained model in this run): observed median Sharpe 0.4778 over 7 paths, DSR = 0.2198 against `n_trials=6`. The expected maximum Sharpe under the null with six compared models, combined with the pooled skew/kurt and SE(SR), produces a DSR threshold the observed Sharpe does not clear. The 0.2198 figure says KAN's median Sharpe is roughly at the 22nd percentile of what the null hypothesis (all true Sharpes equal zero) would produce when testing six independent strategies, far short of the 0.95 significance threshold. Full DSR-component breakdown for the top model goes to Appendix D.
- **PBO result.** **PBO = 0.629 = 22 of 35 IS/OOS partitions.** Interpretation in context of DSR < 0.95:
  - Theoretical regime map: `PBO ≈ 0` = robust selection (the IS-best model tends to remain best out of sample); `PBO ≈ 0.5` = model selection is random; `PBO > 0.5` = adversarial.
  - Observed regime: **adversarial overfitting.** In 62.86% of partitions the model that wins on three IS paths underperforms the median of the four OOS paths. The IS ranking is anti-informative in this run: a strategy that picks the IS-best model and bets on it OOS does worse than the OOS median in nearly two out of three partitions. This is a strong negative finding and a step worse than the previous locked configuration (PBO 0.343 → 0.629, a 0.29 increase). The trajectory of PBO across the three configurations evaluated (0.26 → 0.37 → 0.34 → 0.63) was stable in the moderate-overfitting band across the first three but moves past the random-selection threshold in the current run.
  - **Leave-one-out PBO confirms there is no anchor model in this run.** Re-running PBO on the 5×7 Sharpe matrix with one model dropped at a time produces a pattern where every exclusion *reduces* PBO. Sorted by 5-model PBO ascending:

| Model excluded | 5-model PBO | Δ vs baseline |
|---|---|---|
| ar_logistic | 0.400 | -0.229 |
| lstm | 0.457 | -0.171 |
| kan | 0.486 | -0.143 |
| logistic | 0.514 | -0.114 |
| xgboost | 0.514 | -0.114 |
| random_forest | 0.543 | -0.086 |

  - **Reading the LOO table.** Every model's exclusion *reduces* PBO. There is **no anchor model** in this configuration. AR Logistic's removal reduces PBO most (Δ -0.229 to 0.400), bringing the regime back into the moderate-overfitting band; LSTM's removal reduces it to 0.457; KAN's removal to 0.486 (still in the random-selection band); Logistic and XGBoost each take it to 0.514; Random Forest is the smallest destabiliser at -0.086. The pattern: every model in this six-model cross-section contributes to the IS/OOS rotation in some sub-path partition, and removing any one of them helps reduce PBO. The previous run's finding (KAN as the only anchor whose presence stabilises the cross-section) does not reproduce in the current configuration. Two readings are consistent with this: (a) the cross-section is uniformly destabilising, no model is consistently the IS-best across sub-path partitions, and PBO measures the cost of the rotation; (b) the adversarial baseline regime (PBO > 0.5) is structurally different from the moderate regime in that no single exclusion can pull it back to robust selection, only to the random-selection neighbourhood.
  - Joint reading with DSR < 0.95: model selection within this six-model universe is **adversarial** in the current configuration (PBO 0.629 > 0.5), no model survives multiple-testing correction (top DSR 0.2198 << 0.95), and the IS-best model is the OOS-loser in more than half the partitions. KAN's strictly-positive bootstrap CI (0.0971, 1.2413) provides one piece of independent evidence that KAN's median Sharpe is distinguishable from zero, but this evidence is from path-level bootstrap not multiple-testing-corrected significance. The honest conclusion is "no trained model has predictive ability that survives multiple-testing correction; KAN is the top trained model by median Sharpe and the only one with bootstrap-strict-positive evidence, but model selection in this cross-section is structurally adversarial and the IS-winner is the OOS-loser in most sub-path partitions; a naive buy-and-hold strategy posts a 2.27 median Sharpe over the same period at the cost of a -99.98% maximum drawdown".
  - Methodological implication (T-R15: write for reviewers, this is the AFML pay-off). Without PBO, the natural conclusion from the model-comparison table is "KAN achieved median Sharpe 0.48 on BTC daily direction with a strictly-positive bootstrap CI". With the headline PBO of 0.629 the conclusion sharpens: the IS-best in any sub-path partition is the OOS-loser more often than not, so even a positive bootstrap CI on the headline KAN result must be read against the structural risk that KAN's IS lead does not carry to OOS. The leave-one-out PBO refines the message further: this six-model cross-section has no anchor; the rotation pattern is uniform. The PBO and DSR diagnostics are complementary: PBO answers "did we pick the right model?", DSR answers "is the right model good enough?"; here the answer is "no, the cross-section is adversarial" to the first and "no" to the second.
- **Equity curves figure.** All 6 models + buy-and-hold overlay on the median (or best) path. Note specifically the XGBoost and Logistic Regression equity curves, which are the two ending below 1.0 (median cumulative returns -0.1590 and -0.0596 respectively). Buy-and-hold's curve dominates the plot vertically (45,896× cumulative return); a log-scale y-axis is needed to see the trained models alongside it.
- **Additional metrics table.** Cumulative return, annualised return, max DD, time under water, win rate, profit factor, mean |bet|, % traded. Locked-run highlights from the table at 4.2: AR Logistic median cumulative return 0.0852 (best among trained), KAN 0.0820, Random Forest 0.0199, LSTM 0.0131, Logistic Regression -0.0596, XGBoost -0.1590. Median max-drawdown ranges from -0.2481 (LSTM, best) to -0.3641 (Random Forest, worst); KAN's -0.2563 is the second-best. Buy-and-hold's median max-drawdown is -0.9998, the structural cost of holding through the 2017-18 and 2022 BTC crashes. KAN posts the second-best max-drawdown among trained models alongside its rank-1 trained median Sharpe, the combination consistent with the strictly-positive bootstrap CI finding.

### 4.5 Feature Selection Stability and FFD Stability

- **Feature selection frequency (per-feature count / 28 folds).** Locked-run profile is **flat to a striking degree.** Of 73 candidate features:
  - **0 features stable** (selected in more than 80% of folds, i.e. > 22 of 28 folds).
  - **2 features moderate** (50% to 80%): `eth_btc_ratio` at 57.1% (16 of 28 folds), `log_returns_lag6` at 53.6% (15 of 28 folds).
  - **71 features low** (selected in fewer than 50% of folds).
  - **0 features never selected** (the previous locked configuration had `tx_count_roc_14` as the single never-selected feature; in the current run no feature posts zero selection across the 28 folds).
- **Headline finding.** No compact feature subset consistently dominates across time periods. The MDA-selected set turns over substantially across the 28 folds: the median feature is selected in roughly a third of folds, and even the most-stable feature (`eth_btc_ratio`) clears 50% by about 7 percentage points. This is itself a finding: it suggests that the information relevant to BTC daily direction is distributed across many features, with regime-specific pockets of relevance, rather than concentrated in a small permanent core. The stability profile shifts substantially from the previous locked configuration: `eth_btc_ratio` moves from 60.7% to 57.1% (small drop), `bb_width` drops below the 50% threshold (54% → low bucket), `skewness` also drops below 50% (was 46%, similar position), and `log_returns_lag6` emerges as the new second-most-stable feature at 53.6% (the previous run had `log_returns_lag1` lower in the stability ranking but not in the top-2 moderate set).
- **Group breakdown of the two moderate features.**
  - 1 of 2 from the external / crypto-macro group (`eth_btc_ratio`, the CoinMetrics-sourced ETH/BTC alt-rotation signal).
  - 1 of 2 from the lag / autoregressive group (`log_returns_lag6`, the 6-day lag of log returns). **First time a lag feature appears in the cross-fold moderate-stability set in any locked configuration**.
  - 0 from technical-analysis, macro, on-chain, or mathematical (AFML Part 4) groups.
- **On-chain question (Q2).** **No on-chain feature reaches 50% selection frequency.** All eight on-chain columns (active addresses ROC, transaction count ROC, hash rate ROC, MVRV, net exchange flow, fee per transaction, exchange supply percent, issuance) sit in the "low" bucket. The "free lunch hypothesis" (on-chain transparency providing measurable signal beyond price and volume) does not survive multi-model MDA + CPCV evaluation in this dataset; on-chain features may carry information for individual splits, but no on-chain column is consistently selected across folds. Q2 answer: **on-chain features do not exhibit stability** under the AFML feature-selection regime applied here. This finding hardens across all four locked configurations evaluated: in the 62-feature run `exchange_supply_pct` survived to the symbolic extraction step despite low cross-fold stability; in the previous 73-feature 2-seed run no on-chain feature reached the top-5 stable set; in the previous 73-feature 3-seed run one on-chain feature (`tx_count_roc_14`) was never selected at all; in the current run no on-chain feature is moderate, and the previously-never-selected `tx_count_roc_14` recovers to the low bucket but does not become moderate.
- **TA-feature drop.** The previous locked configuration had `bb_width` and `skewness` (both TA / distribution-shape and volatility) in the moderate / near-moderate set. In the current run, both drop below the 50% threshold. The TA group does not contribute any moderate-stability feature; the moderate set is now {crypto-macro, lag} rather than {crypto-macro, TA}.
- **Figure.** Horizontal bar chart of selection frequency (73 bars, sorted descending, coloured by feature group: green=TA, blue=mathematical, orange=external macro, red=external crypto-macro, brown=external on-chain, grey=lag).
- **FFD stability.** ATR is the only column subject to FFD (the only feature flagged non-stationary by the cross-fold ADF audit at α=0.05 with the FFD column whitelist `['atr']`; the full ADF report identifies 8 non-stationary features at α=0.05, all retained without FFD because the locked configuration applies the transform only to ATR by design). Locked-run statistics across 28 folds × 3 seeds (84 d* estimates):
  - Mean d* = 0.198, std d* = 0.085.
  - Range: d* ∈ [0.050, 0.400]; modal value d* = 0.15 to 0.20.
  - Std d* < 0.1 → consistent stationarity structure across time periods. The ATR series requires fractional differencing of order ≈ 0.20 regardless of the training-fold partition, indicating that the persistence of the volatility-range process is a stable property rather than a regime-conditional one. No fold required d* = 1.0 (full integer differencing). The mean d* shifted from 0.170 (previous run) to 0.198 here, a small upward drift consistent with the slightly larger dataset (1,168 vs. 1,159 aligned events).
- **Methodological note.** Feature-selection turnover (high) and FFD-d* turnover (low) tell a coherent story: the noise-versus-signal ratio for individual features changes across regimes (only `eth_btc_ratio` and `log_returns_lag6` clear 50%), but the underlying time-series property that motivates fractional differencing is stable. Stability of d* validates the FFD-only-on-ATR design choice in 3.6.1; instability of feature selection validates the use of multi-model MDA over single-model SFI in 3.6.3.

### 4.6 Symbolic Extraction Results (Q4 answer)

- **Extraction summary.**
  - Fold used: split 27 (groups 6 and 7), `fold_selection="last"`. Test set covers the most recent CPCV partition; closest to deployment scenario. Locked-run KAN F1 on this fold: **0.3048** (down from 0.4628 in the previous run, a substantial drop in KAN's local performance on the deployment fold).
  - **Methodology change: `n_top_features` reduced from 5 to 3.** The locked notebook call is `run_symbolic_extraction(... n_top_features=3, fold_selection="last", feature_selection_strategy="stability")`. The strategy parameter selects on cross-fold stability rather than per-fold MDA, and the smaller feature count tightens the architecture to a [3, 3, 2] topology that is humanly readable as a four-outer-term formula on three inputs.
  - **Top-3 stable features entering extraction.** `eth_btc_ratio` (57%), `log_returns_lag6` (54%), `natgas_ret_30` (39%). 70 features excluded. The previous run extracted on five features including `bb_width`, `skewness`, and `log_returns_lag1`; the current top-3 set has zero overlap with the previous 4th and 5th features (those were `us2y` and `kurtosis` in the run before, and `log_returns_lag1` / `natgas_ret_30` in the previous 73-feature run). The two top-3 holdovers from the previous run are `eth_btc_ratio` (#1 in both) and `natgas_ret_30` (#5 → #3). The new entrant is `log_returns_lag6`, displacing `log_returns_lag1` as the lag-group representative.
  - Training sample size: **540 train + 135 val** (after the 80/20 split on the split-27 training fold, slightly larger than the previous run's 530 + 133 due to the updated 1,168-sample alignment). 3 features in the input matrix.
  - **Architecture (data-aware fallback).** The locked Optuna study did not produce tuned KAN hyperparameters for split 27, so the data-aware fallback fires. Final architecture `[3, 3, 2]`: 3 input features → 3-unit hidden layer → 2-unit binary output. **15 active edges, ~90 spline parameters for 540 training samples (parameter-to-sample ratio 6.0×, comfortably above the 5× floor).** The hidden width (3) increased from 2 in the previous run, reflecting the data-aware fallback's response to fewer input features.
- **Training diagnostics.**
  - Phase 1 (Adam, 600 steps, weight decay 1e-3, noise std 0.05): final train accuracy 0.5759, **validation accuracy 0.4815**.
  - **Phase-1 dynamic gate FAILS** (val_acc 0.4815 < 0.5693, where the effective threshold is `max(0.53, base_rate + 0.01) = max(0.53, 0.5593 + 0.01) = 0.5693`). The dynamic gate is a methodology improvement: instead of a fixed 0.53 threshold, the gate now adjusts to the actual class balance in the validation fold (`majority_baseline + 0.01 margin, floored at PYKAN_MIN_ACCURACY=0.53`). The pipeline logs the warning "Adam phase val_acc=0.4815 < 0.5693 (majority baseline 0.5593 + 0.01 margin, floored at PYKAN_MIN_ACCURACY=0.53). PyKAN may not have learned meaningful patterns" and continues to symbolification per the design.
  - Phase 2a (LBFGS warmup, 20 steps, no regularisation): train 0.6111, val **0.4296** (the only locked-run configuration where Phase 2a *drops* validation accuracy below Phase 1).
  - Phase 2b (LBFGS sparsity, 20 steps, lambda=2e-3): train 0.6500, val 0.4667.
  - **Pre-prune gate fails too.** The pruning step's pre-prune check (val_acc < 0.5359, where 0.5359 = 0.5259 majority baseline + 0.01 margin) also fails at val_acc 0.4667. The model has not learned patterns that beat the majority baseline by even one percentage point at any of the four training phases.
  - Grid extension skipped (`n_train=540 ≤ 1000`, grid stays at 3).
  - Detailed phase-by-phase logs in Appendix F.
- **Pruning results.** **Pruned architecture: `[[3, 0], [3, 0], [2, 0]]`** in PyKAN's `(sum_units, mult_units)` per-layer notation. **No edges pruned**: pre-prune edge analysis at threshold 0.01 finds 15 of 15 active (100% survival). The model's edge-importance distribution does not contain any below-threshold edges. Post-prune validation accuracy: 0.4667 (unchanged from Phase 2b final, since no edges were removed).
- **Symbolification rate: 100% (15 of 15 edges).** All 15 symbolified edges clear the 0.30 R² threshold; zero edges skipped during symbolic substitution. R² distribution across the 15 symbolified edges: min 0.734, median 0.990, max 1.000. Top-five edges all post R² > 0.99 with the `sin` primitive dominating the top of the ranking.
- **Surviving features (3).**
  - `eth_btc_ratio` (external / crypto-macro group, CoinMetrics-sourced ETH/BTC alt-rotation signal). Selected in **57%** of CPCV folds at the 28-fold level; **most stable feature in the dataset** across all locked configurations to date.
  - `log_returns_lag6` (lag / autoregressive group, 6-day lag of log returns). Selected in **54%** of CPCV folds. **First time `log_returns_lag6` enters the symbolic representation**; the previous run had `log_returns_lag1` as the lag-group entrant at 32% selection frequency. The lag group has now contributed a representative to two consecutive locked configurations (lag1 then lag6), validating the lag-features-in-MDA-pool design choice as producing reproducible (if shifting) lag representation in the symbolic formula.
  - `natgas_ret_30` (external / macro group, 30-day natural gas return). Selected in **39%** of CPCV folds. Returning from the previous run (where it was at 32%). Natural gas is the only macro-group feature to enter symbolic extraction in any of the 73-feature locked configurations.
  - **Group breakdown.** 1 crypto-macro (`eth_btc_ratio`), 1 macro (`natgas_ret_30`), 1 lag (`log_returns_lag6`), 0 TA, 0 on-chain, 0 mathematical (AFML Part 4). **TA features dropped entirely from the surviving set** (was 2 of 5 in the previous run); the surviving set is now distributed across three feature families rather than four.
- **Pre-symbolic vs. post-symbolic accuracy.**
  - Pre-symbolic (B-spline KAN, post-prune): **46.67%** (below 50% baseline).
  - Post-symbolic (closed-form formula, after affine fine-tuning): **51.85%** (barely above 50% baseline).
  - **Δ = +5.18 percentage points (post exceeds pre by 5.2 pp).**
  - **Honest reading: both pre- and post-symbolic accuracies sit at or near the 50% baseline.** The pre-symbolic 46.67% is below chance: the PyKAN model retrained on the deployment fold has not extracted signal that beats the majority-class predictor (which would post 56.85% by predicting class +1 always). The post-symbolic 51.85% is barely above chance and below both the majority baseline (56.85%) and the dynamic Phase-1 gate (56.93%).
  - The +5.18 pp gain *is* larger than the previous run's +1.5 pp gain in magnitude, but the absolute level on which it operates (sub-baseline pre-symbolic accuracy) means the gain represents a recovery from below-chance to barely-above-chance rather than an improvement from a meaningful baseline. The affine fine-tuning step is doing genuine work (it consistently improves over the pre-symbolic spline KAN), but the underlying model on this fold has not found exploitable signal in the three input features.
  - Cross-run trajectory of (pre-symbolic, post-symbolic, Δ): (0.5564, 0.5263, -0.0301) → (0.5564, 0.5940, +0.0376) → (0.5940, 0.6090, +0.0150) → (0.4667, 0.5185, +0.0518). The most recent run has the largest gain magnitude but starts from the lowest baseline; the previous configuration's positive trajectory in absolute terms is not preserved.
- **The decision function.** The extracted closed-form expression. Numbered equation in the thesis:

```
decision(x) =
    + 3·(- 284·eth_btc_ratio/505
         - (-1766·log_returns_lag6/179 - 1529/843)⁴ / 1000
         - 434·tanh(3217·natgas_ret_30/944 + 2141/935) / 663
         + 4511/913)² / 283
    - 351·sin( (797/768 - 3716·natgas_ret_30/745)³ / 109
               - 405·sin(4547·log_returns_lag6/949 + 147/754) / 508
               + 517·cos(2279·eth_btc_ratio/686 + 4631/899) / 687
               + 433/74 ) / 877
    + 197·cos( 10·(797/768 - 3716·natgas_ret_30/745)³ / 847
               - 625·sin(4547·log_returns_lag6/949 + 147/754) / 609
               + 31·cos(2279·eth_btc_ratio/686 + 4631/899) / 32
               + 2734/385 ) / 828
    + 455·tanh( 235·eth_btc_ratio/924
                + 209·tanh(3217·natgas_ret_30/944 + 2141/935) / 706
                - 4610/959 ) / 932
    + 11/19

P(up | x) = 1 / (1 + exp(-decision(x)))
```

  - Coefficients are post-`sympy.nsimplify(tolerance=1e-3)` rationals. The structure has changed substantially from the previous configuration: instead of four outer terms each combining all five features through one of {sin, sin, tanh, tanh} wrappers, the new formula has four outer terms each using a *different* outer function: an outer squared term `3·(...)²/283`, an outer `sin`, an outer `cos`, and an outer `tanh`. The features are distributed unevenly: `natgas_ret_30` appears in all four outer terms (via tanh and via cube primitives), `log_returns_lag6` appears in three (the squared term and the two trigonometric terms), `eth_btc_ratio` appears in all four. The previous run's "each feature in all four outer terms" symmetry does not hold here.
- **Functional-form pattern.** **The formula uses six distinct primitives: `sin`, `cos`, `tanh`, the squared term `(...)²`, the cubed term `(...)³`, and the quartic term `(...)⁴`.** This is again substantively different from the previous configuration:
  - **`atan` and `|...|` are gone.** The previous run had `atan` attached to `log_returns_lag1` and `natgas_ret_30` as a bounded smooth saturator and `|...|` attached to `natgas_ret_30` as a non-smooth V-shape primitive. Neither appears in the v5 formula. **The non-differentiability flag from the previous configuration is resolved**: with `|...|` no longer in the primitive set, no surviving feature posts an undefined gradient at the dataset mean.
  - **`cos` returns.** The previous run replaced cos with sin in every outer-wrapper position; the v5 formula uses both, with `cos` appearing in two positions (an outer cos and an inner cos attached to `eth_btc_ratio`).
  - **`x²` returns.** Used as the outer wrapper of the first outer term, with the inner argument combining `eth_btc_ratio`, `tanh(natgas_ret_30 + ...)`, and a quartic in `log_returns_lag6`.
  - **`x⁴` is new.** The first time a quartic primitive survives the R² threshold in any locked configuration. Attached to `log_returns_lag6` in the first outer term.
  - **`x³` survives** from the previous two runs, attached to `natgas_ret_30` this time (was attached to `log_returns_lag1` in the previous run).
  - **`tanh` survives** as both an outer wrapper (the fourth outer term) and an inner primitive (`tanh(3217·natgas_ret_30/944 + 2141/935)` appears nested inside both the squared first term and the fourth tanh term).
  - The four outer wrappers (squared, sin, cos, tanh) are functionally distinct. The KAN learned four different smooth representations and combined them as a weighted sum. The previous configuration's "two complementary internal representations × two outer wrappers" symmetry does not hold in v5.
- **Interpretive note (factual; economic interpretation deferred to 5.3).** The term-structure is no longer symmetric across features: `eth_btc_ratio` appears in all four outer terms, `natgas_ret_30` in all four (in different roles: tanh nested inside the squared term, x³ inside the sin and cos terms, and tanh again as the outer's inner argument), and `log_returns_lag6` in three (the squared term as a x⁴, and the two sine/cosine inner arguments). The cross-run robustness of the "all features in all four outer terms" pattern from the previous run does not survive the n=3 input dimensionality of v5.
- **Numerical sensitivity at the dataset mean.** Partial derivatives evaluated at the mean of each feature, scaled by the feature's standard deviation, give the per-σ effect on the decision function (logit). All three surviving features now have **finite, well-defined gradients** at the dataset mean (the natgas Abs-primitive non-differentiability from the previous run is gone):

| Feature | mean | std | d/dx at mean | σ-effect on logit | σ Δp (linearised) |
|---------|-----:|-----:|-----:|-----:|-----:|
| `eth_btc_ratio` | TBD | TBD | TBD | TBD | TBD |
| `log_returns_lag6` | TBD | TBD | TBD | TBD | TBD |
| `natgas_ret_30` | TBD | TBD | TBD | TBD | TBD |

  - **Sensitivity table values are TBD pending the sensitivity computation cell run in the locked notebook.** The symbolic decision function and its partial derivatives are reported in the notebook output (Section 6.4 of `main.ipynb`); the σ-scaled effects need to be transcribed once the cell is rerun with the current locked seed configuration. The qualitative pattern from the symbolic form: `natgas_ret_30` enters via tanh and x³ (both bounded primitives at the dataset mean), `log_returns_lag6` enters via x⁴ and the trigonometric inner arguments (the x⁴ term contributes a steep gradient if `log_returns_lag6` is far from the offset `1529/843 ≈ 1.81 / (-1766/179) ≈ -0.183`; at the dataset mean log_returns_lag6 ≈ -0.0005 the x⁴ term contributes a small gradient), `eth_btc_ratio` enters via the cos inner argument and the tanh outer wrapper (both bounded and smooth at the dataset mean).
  - **Scale check.** All three surviving features have interpretable scales. `eth_btc_ratio` is bounded above by the historical maximum (~0.13 in this sample); `log_returns_lag6` is a 6-day log return (bounded around ±0.30); `natgas_ret_30` is a 30-day return (typically ±0.20). No rescaling caveat is needed for the formula.
- **On-chain comment (Q2 evidence).** **Zero on-chain features survive extraction.** This finding is now stable across the previous and current 73-feature locked configurations. The Q2 answer hardens further: not only do on-chain features fail to clear cross-fold stability (Section 4.5), they consistently fail to enter the locked symbolic representation regardless of `n_top_features` configuration. The earlier 62-feature locked-run observation that one on-chain feature could carry the formula's strongest economic effect is **regime-specific and does not generalise to the expanded feature universe.**
- **Lag-feature comment (Q2 evidence).** **A lag feature survives extraction for the second consecutive locked configuration**, but the specific lag has shifted (`log_returns_lag1` → `log_returns_lag6`). The lag-features-in-MDA-pool design choice (Section 3.4.4, advisor-driven, lag features compete with engineered features rather than feeding only into AR Logistic) admits lag features into the surviving set in the 73-feature pool; the shifting lag identity across runs suggests the autoregressive signal in BTC daily direction is not concentrated at any single horizon but rotates across nearby lags depending on the seed and feature-selection-strategy configuration.
- **Surviving-feature group breakdown.**
  - 1 of 3 from crypto-macro (`eth_btc_ratio`).
  - 1 of 3 from macro (`natgas_ret_30`).
  - 1 of 3 from lag (`log_returns_lag6`).
  - 0 of 3 from TA, on-chain, or mathematical (AFML Part 4) groups.
- **Section closer.** "The symbolic extraction pipeline produces a closed-form decision function with **100% symbolification rate (15 of 15 edges)** on a three-feature subset (`eth_btc_ratio`, `log_returns_lag6`, `natgas_ret_30`). **Post-symbolic accuracy is 51.85%, a 5.18 percentage-point gain over the pre-symbolic 46.67%**, but both numbers sit at or below the 50% baseline: the PyKAN model retrained on the deployment fold did not extract signal that beats coin-flip. The formula uses six primitives (`sin`, `cos`, `tanh`, `x²`, `x³`, `x⁴`) and four outer terms with distinct outer wrappers (squared, sin, cos, tanh). The closed-form approximation *improves* accuracy relative to the trained spline network in absolute pp terms but the absolute level remains sub-baseline, the Phase-1 dynamic gate fails (val_acc 0.4815 < 0.5693), and the pre-prune gate also fails. The Abs(·) non-differentiability flag from the previous configuration is resolved: with `|...|` no longer in the primitive set, no surviving feature posts an undefined gradient at the dataset mean. RQ4 conclusion: closed-form interpretability is achievable (100% symbolification rate, clean primitives, finite gradients everywhere), but the locked-run model on the deployment fold has not extracted signal that beats the majority baseline; the symbolic representation faithfully captures a model that has not found the signal."

---

## 5. Discussion

> **T-R7.** Interpret. Link findings back to the mechanisms in Section 3.

### 5.1 Interpreting the results (Q1, Q3 evidence)

- **The joint DSR / PBO finding.** No model achieves DSR ≥ 0.95 (top trained model is KAN at 0.2198; buy-and-hold posts a higher median Sharpe of 2.2722 but is not included in the DSR computation since it does not enter the selection pool). PBO sits in the **adversarial regime at 0.629**, with 22 of 35 IS/OOS partitions seeing the IS-winner underperform the OOS median. The combination is informative: model selection in this six-model cross-section is not just unreliable but actively anti-informative; the IS-best is the OOS-loser in nearly two of three partitions. Despite this, one trained model (KAN) has a strictly-positive bootstrap CI on its median Sharpe (0.0971, 1.2413): KAN therefore carries simultaneous evidence in two opposite directions: bootstrap-strict-positive on the path-level (positive finding) and inside an adversarial PBO regime (structural negative finding). The honest reading is "no trained model clears the multiple-testing-corrected significance threshold; KAN is the only trained model with bootstrap-strict-positive evidence on its median Sharpe; model selection in this cross-section is structurally adversarial; and a naive buy-and-hold strategy on the same data posts a median Sharpe roughly 4.8× the best trained model's figure". This is a stronger negative position than either of the previous two locked configurations could deliver: we have moved past the moderate-overfitting regime (PBO 0.34 → 0.629) into one where the IS ranking is anti-informative, but the bootstrap evidence for KAN keeps a single trained-model positive finding alive on the path-level.
- **PBO trajectory across the four locked configurations.** PBO has tracked 0.26 → 0.37 → 0.34 → 0.629 across the 62-feature, 73-feature 2-seed, 73-feature 3-seed-previous, and current 73-feature 3-seed runs. The first three configurations sit inside the moderate-overfitting band [0.25, 0.40]; the current run breaks past the random-selection threshold (0.5) into the adversarial regime. The configuration changes between the previous run and the current one are limited (1,168 vs. 1,159 events, an updated 2026-05-02 endpoint, and the `n_top_features=3` change in the symbolic-extraction call that does not affect the CPCV results), so the PBO jump must be attributed to the data-refresh-driven model rotation in IS/OOS sub-paths rather than to any methodology change. The implication: small data drifts can move the IS-winner identity across sub-path partitions enough to flip PBO across the random-selection threshold.
- **Locating the variance: no anchor model.** The leave-one-out PBO (Section 4.4) shows that every model's exclusion *reduces* PBO. AR Logistic's removal reduces PBO most (Δ -0.229 to 0.400), bringing the regime back into the moderate-overfitting band; LSTM's removal reduces it to 0.457; KAN's removal to 0.486 (still in the random-selection band); Logistic and XGBoost each take it to 0.514; Random Forest is the smallest destabiliser at -0.086. **There is no anchor model in the current configuration.** The previous run's finding (KAN as the only anchor whose presence stabilises the cross-section) does not reproduce. Two readings are consistent with this: (a) the cross-section is uniformly destabilising, no model is consistently the IS-best across sub-path partitions, and every model's removal reduces the IS/OOS rotation; (b) the adversarial baseline regime (PBO > 0.5) is structurally different from the moderate regime in that no single exclusion can pull it back to robust selection, only to the random-selection neighbourhood. The cross-run pattern that emerges across the four locked configurations is suggestive: in moderate regimes (the first three runs), exactly one model acted as an anchor (LSTM in the 2-seed run, KAN in the previous 3-seed run); in the adversarial regime of the current run, no model anchors and all destabilise.
- **Consistency with EMH.** The DSR < 0.95 finding aligns with semi-strong-form EMH (Fama 1970) for BTC daily direction: under leakage-free evaluation with multiple-testing correction, no architecture in the comparison demonstrates a statistically significant predictive edge. The fact that **buy-and-hold posts a higher median Sharpe (2.2722) than any trained model** strengthens this reading: in the 2015 to 2026 BTC sample, the unconditional long position dominates every directional-trading strategy in median Sharpe terms, suggesting that the asset's overall trajectory carries information that direction-prediction strategies discard by going neutral or short in volatile regimes. The adversarial PBO finding hardens this further: not only is no architecture significantly predictive, but the cross-section's IS rankings are anti-informative about OOS performance. This is the intellectually honest answer to Q1 and consistent with `chassot_audrino_2026`: properly fitted baselines are hard to beat, and the literature's ML-superiority results came from suboptimal fitting schemes that handicapped the baselines. **Cite:** `fama_1970`, `chassot_audrino_2026`.
- **Conservative Predictions framing.** `nabar_shroff_2023`: in low-signal regimes, abstention beats prediction. The bet-sizing threshold operationalises this: predictions with `p ≈ 0.50` produce `bet ≈ 0`, structural abstention without a separate model. The locked run's mean P(Up) for KAN is 0.5398 against an empirical base rate of 0.5685 (Δ = -0.0287, at the edge of the ±0.03 calibration tolerance but classed as ok by the audit), so the calibration audit does not flag KAN. The bet-sizing curve is operating on well-calibrated probabilities, and the negative DSR finding is therefore a statement about signal strength, not about probability quality. Only one model is flagged by the calibration audit in the current configuration (XGBoost at Δ -0.0323, just over the tolerance). This is a tighter calibration footprint than any previous locked configuration; despite the adversarial PBO regime, calibration quality has improved across the cross-section.
- **Three-version pipeline evolution (advisor narrative).** Inflated ROC-AUC → degenerate all-Down predictions → AFML. Two specific failure modes motivated AFML adoption: uninformative fixed-time-horizon labels and structural leakage that standard splits cannot prevent. The locked-run results show that even after AFML closes both leakage channels, the residual signal is too weak for any compared model to clear DSR = 0.95, and the IS rankings in the model cross-section are now anti-informative under PBO; this is the AFML pay-off. The methodology produces honest results (PBO 0.629 in the adversarial regime) rather than the inflated metrics that earlier (leaky) pipeline versions produced. Across the four locked configurations evaluated to date (62 features / 2 seeds for neural, 73 features / 2 seeds for neural, 73 features / 3 seeds previous run, 73 features / 3 seeds current run), PBO has tracked 0.26 → 0.37 → 0.34 → 0.629: stable in the moderate-overfitting band for the first three runs, then jumping into the adversarial regime in the current run. The PBO trajectory itself is now a finding worth reporting: the moderate-overfitting reading is not as stable across data refreshes as the first three runs suggested.

### 5.2 KAN performance in context (Q3, Q4 evidence)

- **KAN ranks 1st among trained models** in median Sharpe (0.4778), ahead of AR Logistic (0.3806), Random Forest (0.2047), LSTM (0.1808), XGBoost (0.0484), and Logistic Regression (-0.1545). Buy-and-hold's 2.2722 sits above the entire trained-model cross-section. KAN's positioning continues to recover relative to early locked configurations: rank-5 in the original 62-feature run, rank-2 in the previous 73-feature 3-seed run, rank-1 in the current run. KAN is also rank-1 on DSR (0.2198) among trained models, the only trained model with a strictly-positive bootstrap CI (0.0971, 1.2413), and posts the second-best median max-drawdown (-0.2563, behind LSTM at -0.2481).
- **The bootstrap-strict-positive finding is the headline KAN result.** Five of six trained-model CIs cross zero; KAN's does not. This is the first locked configuration in which any trained model posts a bootstrap-strict-positive CI on its median Sharpe under uniform 3-seed evaluation, and it is an independent piece of evidence (path-level bootstrap, not multiple-testing correction) that KAN's positive median Sharpe is distinguishable from zero. The previous run had zero trained models with strictly-positive CIs; the current run recovers a single trained-model strict-positive at KAN. This finding must be read against the adversarial PBO baseline (0.629): the cross-section's IS rankings are anti-informative about OOS, so even a positive bootstrap CI on the headline KAN result must be read against the structural risk that KAN's IS lead in any sub-path partition does not carry to OOS. The two findings (KAN strictly-positive bootstrap CI, adversarial cross-section PBO) are independent; both are reported, and the interpretive weight a reader gives each is up to them.
- **Classification-side metrics are no longer dominant.** KAN posts mean accuracy 0.5330 (mid-pack, ahead of XGBoost at 0.5144 but behind AR Logistic at 0.5240 and Random Forest at 0.5252), mean AUC 0.5088 (mid-pack), and median win rate 0.5407 (the highest among trained models). The DeLong test (Section 4.3) confirms no AUC pair clears α = 0.05, with the closest pairs being Random Forest vs. XGBoost (p = 0.0526) and AR Logistic vs. XGBoost (p = 0.0544); both near-significant pairs include XGBoost as the lower-AUC half. AR Logistic now leads on pooled AUC (0.5117 to 0.5149) rather than KAN. The previous configuration's "KAN dominates on classification metrics" reading does not survive into the current run; what survives is KAN's median win-rate lead, which is one classification-side proxy for the bet-sizing translation.
- **Where KAN's rank-1 trained Sharpe comes from.** Given that KAN is mid-pack on AUC and mean accuracy, the rank-1 trained median Sharpe must come from the bet-sizing translation rather than from raw classification accuracy. Three diagnostics support this reading: (a) KAN's mean P(Up) is 0.5398, within tolerance of the 0.5685 empirical base rate at Δ = -0.0287, so the bet-sizing curve operates on calibrated probabilities; (b) KAN's median win rate 0.5407 is the highest among trained models, indicating that when KAN does bet directionally (probability far from 0.5), it wins slightly more often than other models bet directionally; (c) KAN's median max-drawdown -0.2563 is second-best among trained models, reflecting bet-sizing that protects the strategy from the largest path-level moves. The combination of well-calibrated probabilities, slightly-better-than-others directional win rate, and conservative max-drawdown produces the rank-1 trained Sharpe and the strictly-positive bootstrap CI.
- **Tension to interpret: rank-1 trained model is not the structural anchor.** In moderate PBO regimes, the LOO PBO analysis identifies an anchor model whose presence stabilises the cross-section. In the current adversarial regime, no anchor exists: every model's removal reduces PBO. KAN's removal reduces PBO from 0.629 to 0.486 (Δ -0.143), placing KAN in the middle of the LOO-destabiliser ranking. KAN is therefore the financially-best trained model in this run but is also part of the rotating IS-winners cluster that contributes to the adversarial PBO baseline. The previous run's structural-anchor finding for KAN does not reproduce in the current configuration; what survives is the financial-ranking finding (rank-1 on Sharpe, rank-1 on DSR, only strict-positive CI).
- **Statistical-vs-economic gap.** KAN's mean AUC of 0.5088 is mid-pack in the cross-section, and the DeLong test against every other model returns p > 0.14. The classification-side superiority observed in earlier runs has not reproduced. What does translate into a ranking position is the bet-sizing transformation of KAN's well-calibrated probabilities: at a base rate of 0.5685 and a KAN mean P(Up) of 0.5398, the bet-sizing curve places conservatively-sized bets that protect against the largest drawdowns (KAN's median max-drawdown -0.2563 is second-best among trained models, behind LSTM at -0.2481). The honest reading: KAN's classification advantage is no longer present in the current configuration; its rank-1 Sharpe ranking comes from the combination of well-calibrated bet sizing and slightly-better directional win rate, not from a dominant classification edge.
- **Contrast with `oad_kasper_2025`.** KASPER reports R² = 0.89 and Sharpe = 12.02 on individual stocks with regime detection. This thesis posts median Sharpe = 0.48 for KAN on BTC daily direction without a regime layer. The differences are explanatory rather than competitive: KASPER uses regression on individual stocks with regime-conditional architectures; this thesis uses classification on a single asset (BTC) with no regime layer and applies AFML's full statistical-correction stack (DSR, PBO, DeLong) which KASPER does not. The roughly 25× Sharpe gap reflects asset class, target type, and evaluation rigour rather than KAN architectural capacity.
- **Contrast with `cho_lee_kim_2025`.** The VIX KAN paper extracts symbolic formulas that reveal mean-reversion and leverage effects, validated against domain knowledge. This thesis's symbolic formula reveals a three-feature mixed-primitive structure through `sin`, `cos`, `tanh`, `x²`, `x³`, and `x⁴` primitives, on a target (BTC daily direction) that is not mean-reverting and where domain knowledge does not directly validate any specific functional form. The primitive set in the current configuration (the addition of `x⁴` and the return of `cos` and `x²`, with the removal of the previous run's `atan` and `|·|`) suggests the KAN's learned decision boundary changes its functional character with both the surviving feature set and the input dimensionality (5 → 3). The symbolic-extraction machinery transfers across asset classes; the interpretive yield depends on whether the underlying target has a structure that domain knowledge can corroborate. The cross-run variation in primitives suggests the formula's functional signature is regime-conditional rather than stable, and this is itself a methodological observation worth reporting.

### 5.3 Symbolic extraction as contribution (Q4 evidence)

- **Interpretability is a separable contribution from predictive accuracy.** The locked-run formula has **100% symbolification rate** (15 of 15 edges, no skips) and a closed-form expression on three features. Its post-symbolic accuracy of **51.85%** is a 5.18 percentage-point *gain* over the spline KAN's pre-symbolic 46.67%, but both numbers sit at or below the 50% baseline and well below the 56.85% majority baseline. The closed-form decision function provides three artefacts a black-box model cannot: per-feature symbolic derivatives (Section 4.6 numerical sensitivity), term-structure decomposition (each feature appears in at least three of four outer terms), and an audit trail from input to probability that a domain expert can inspect. The fact that the symbolic form outperforms the spline form is consistent across all four locked configurations evaluated to date (-3.0 pp, +3.8 pp, +1.5 pp, +5.2 pp); the symbolic substitution appears to act as a smoothing regulariser on the per-edge B-spline shapes, and the affine fine-tuning step then tunes the per-edge scale and bias to the validation distribution. **What does not survive in the current run is the absolute level**: post-symbolic 51.85% sits below the majority baseline 56.85%, meaning the closed-form formula does not beat the always-predict-class-+1 predictor on this fold.
- **The methodology change (`n_top_features=3`).** The locked notebook call uses `n_top_features=3` (down from 5 in the previous run), restricting the symbolic extraction input to the three most-stable features by cross-fold MDA frequency: `eth_btc_ratio` (57%), `log_returns_lag6` (54%), `natgas_ret_30` (39%). The motivation for the change (as inferable from the codebase: `n_top_features` is exposed as a parameter to `run_symbolic_extraction` and the `feature_selection_strategy="stability"` default ranks by 28-fold selection frequency) is to tighten the formula to a humanly-auditable size on the deployment fold; the trade-off is that the input dimensionality drop forces a narrower [3, 3, 2] architecture and reduces the data PyKAN has to find signal on. The current run is the first locked configuration where the methodology change is made explicit; the previous three runs used the default `n_top_features=5`. The honest reading: the smaller input set produces a cleaner formula (100% sym rate, six clean primitives, no `|·|` non-differentiability) but on this fold also produces a model that does not beat the majority baseline.
- **The post-symbolic accuracy gain in magnitude vs. level.** The +5.18 pp gain in the current configuration is the largest of any locked run in absolute pp terms, but operates from the lowest baseline (pre-symbolic 46.67% vs. previous-run 59.40%, 55.64%, 55.64% for the three prior runs). The reading: the affine fine-tuning step is doing genuine work on the deployment fold, recovering above-chance behaviour from a pre-symbolic representation that was below chance. But the magnitude of the recovery does not get the formula above the majority baseline, so the symbolic representation faithfully captures a model that has not found the signal. Per-fold extraction (L3 in 5.4) would let the analysis distinguish between fold-specific gains and a robust symbolic-smoothing effect; this remains the highest-leverage methodological extension available to the project.
- **The Phase-1 dynamic gate failure.** The Phase 1 (Adam) validation accuracy of 0.4815 falls below the dynamic gate `max(0.53, base_rate + 0.01) = 0.5693` by 8.78 percentage points. The dynamic gate is a methodology improvement over the previous fixed 0.53 threshold: it adjusts to the actual class balance in the validation fold rather than assuming the dataset-level base rate, which makes it sensitive to per-fold imbalance. The pre-prune gate (val_acc < 0.5359, where 0.5359 = 0.5259 majority baseline + 0.01 margin) also fails at val_acc 0.4667. **Both gates fail in the current configuration**, the first locked run where the gate-failure is structural rather than recoverable. The previous run's gate failure (Phase 1) was followed by LBFGS-phase recovery to 0.5940; the current run does not recover above the 50% baseline at any of the four training phases. The reading: the symbolic-extraction pipeline can still deliver a usable closed-form formula on a model that did not pass its own gates (the structural transparency of the decision function does not depend on classification accuracy), but the gate failures are honest signals that the underlying PyKAN model has not extracted directional signal on this fold.
- **Mixed-primitive functional form.** The decision function uses six distinct primitives (`sin`, `cos`, `tanh`, `x²`, `x³`, and `x⁴`), arranged in four outer terms with structurally distinct outer wrappers (squared, sin, cos, tanh). The previous run used `sin`, `tanh`, `atan`, `x³`, and `|·|`; the current run replaces `atan` and `|·|` (which had introduced the natgas non-differentiability flag) with `cos`, `x²`, and `x⁴`. The new primitive set is entirely differentiable; **the previous run's non-differentiability flag is resolved**. The `x⁴` primitive is new across all four locked configurations and attaches to `log_returns_lag6` in the squared outer term. The four-outer-term structure recurs from the previous run, suggesting PyKAN's pruning step produces a four-outer-term representation under both the `[5, 2, 2]` and `[3, 3, 2]` architectures, but the *symmetry* across outer terms is lost in the current configuration: each outer wrapper is a different function (squared, sin, cos, tanh) rather than two pairs (sin × 2, tanh × 2) as in the previous run.
- **Cross-run primitive-set instability.** The primitive set has now changed in every pair of consecutive locked configurations: 62-feature {sin, cos, tanh, x², x³} → 73-feature 2-seed {sin, tanh, atan, x³, |·|} → 73-feature 3-seed previous {sin, tanh, atan, x³, |·|} → current {sin, cos, tanh, x², x³, x⁴}. The cross-run robust subset is {sin, tanh, x³}; the cross-run rotating subset is {cos, atan, |·|, x², x⁴}. The reading: the brute-force fitting library tries roughly 14 candidates per edge and keeps the highest-R² match per fold, so different feature sets and folds land on substantively different primitive combinations. Reporting this cross-run instability is itself a finding about the symbolic-extraction methodology rather than about the underlying BTC direction process: the formula's functional signature is regime- and configuration-conditional, not stable.
- **Numerical sensitivity at the dataset mean.** All three surviving features have finite, well-defined gradients at the dataset mean (the previous run's natgas Abs-primitive non-differentiability is gone). The sensitivity table values are TBD pending the locked-notebook sensitivity-cell rerun (see Section 4.6 numerical sensitivity bullet). The qualitative pattern from the symbolic form: `natgas_ret_30` enters via tanh and x³ (both bounded at the dataset mean), `log_returns_lag6` enters via x⁴ and the trigonometric inner arguments (x⁴ contributes a steep gradient if the inner argument is far from zero; at the dataset mean log_returns_lag6 ≈ -0.0005 the x⁴ term contributes a small gradient), `eth_btc_ratio` enters via the cos inner argument and the tanh outer wrapper (both bounded and smooth at the dataset mean). The locked formula is therefore everywhere-differentiable at the dataset mean; the non-differentiability that previous-run readers would have had to manage is resolved.
- **The on-chain finding hardens further across configurations.** **No on-chain feature survives extraction in any of the three 73-feature locked configurations.** The previous 62-feature locked run retained `exchange_supply_pct` as the single on-chain feature in the surviving four; all three subsequent runs at 73 features have produced zero on-chain survivors. The Q2 answer therefore reads: on-chain features may carry signal in some narrowly-defined configurations (the 62-feature pool with `exchange_supply_pct` at 46% selection frequency), but the broader configuration with 20 macro features and 10 lag features competing in the MDA pool produces zero on-chain survivors at any reasonable seeds configuration and at any `n_top_features` setting. The on-chain free lunch fails at the cross-fold stability level (no on-chain feature reaches 50% selection frequency across the 28 folds in any run) and at the deployment-fold level (no on-chain feature reaches the symbolic decision function on split 27 in any 73-feature run). The earlier locked-run observation that one on-chain feature could carry the formula's strongest effect was regime-specific and does not generalise.
- **The lag-feature finding now reproduces across two configurations.** A lag feature survives into the symbolic representation in both the previous run (`log_returns_lag1` at 32% selection frequency) and the current run (`log_returns_lag6` at 54% selection frequency, moderate stability). The specific lag has rotated (lag1 → lag6) but the lag *group*'s presence in the surviving set is now reproducible. The lag-features-in-MDA-pool design choice (Section 3.4.4, advisor-driven, lag features compete with engineered features rather than feeding only into AR Logistic) admits lag features into the surviving set across configurations; the rotation of the specific lag identity suggests the autoregressive signal in BTC daily direction is not concentrated at any single horizon but rotates across nearby lags depending on the feature-selection-strategy and seed configuration. **This is the cleanest reproducible Q2 finding so far**: lag features are consistently part of the surviving set in the expanded 73-feature universe, even though no specific lag is consistently *the* surviving lag.
- **Contrast with `cho_lee_kim_2025`.** The VIX KAN paper extracts symbolic formulas that reveal mean-reversion and leverage effects, validated against domain knowledge. This thesis's locked formula reveals a three-feature mixed-primitive structure on a target (BTC daily direction) that is not mean-reverting and where domain knowledge does not directly validate any specific functional form. The cross-run variation in the dominant feature and its sign is itself an honest finding about the symbolic-extraction methodology: the formula's economic interpretation is fold- and configuration-conditional, and the current run's formula is faithful to a model that did not extract directional signal on the deployment fold.
- **Limitations specific to symbolic extraction (preview to 5.4).**
  - **Fold-specific.** Different folds may produce different formulas with different surviving features; per-fold extraction is L3 in 5.4. The cross-run change from `(eth_btc_ratio, bb_width, skewness, log_returns_lag1, natgas_ret_30)` to `(eth_btc_ratio, log_returns_lag6, natgas_ret_30)` is direct evidence that fold- and configuration-specificity is a real limitation rather than a theoretical concern.
  - **Small sample.** 540 training observations on split 27 is not large; PyKAN's per-edge R² fits stabilise on this size but recover signal that the spline KAN already captured rather than discovering new structure. With only 3 input features (down from 5), the input dimensionality is now low enough that some signal in the dropped features may have been informative.
  - **R² threshold sensitivity.** The 0.30 threshold (lowered from 0.50) admits the lowest-R² edge at 0.734 in the current run, which is much cleaner than the previous run's 0.5746 minimum. **The 100% symbolification rate is itself a quality improvement** relative to the previous run's 93%, and the R² distribution is tighter at the bottom end.
  - **The dynamic majority-baseline gate is a methodology improvement that now fails in both phases.** The previous fixed-0.53 gate failed at val_acc 0.5038 in the previous run but the model recovered above baseline in subsequent phases. The new dynamic gate (max(0.53, base_rate + 0.01) = 0.5693 in this run) fails by a much wider margin (val_acc 0.4815), and the pre-prune gate (0.5359) also fails. The diagnostic improvement (dynamic threshold) has surfaced a real problem (sub-baseline learning on this fold) that the previous fixed threshold would have masked partially.
  - **The natgas_ret_30 non-differentiability flag is resolved.** The current run's primitive set (sin, cos, tanh, x², x³, x⁴) is entirely differentiable. The previous run's `|·|` primitive that introduced the SymPy `re`/`im` decomposition issue at the dataset mean is no longer in the surviving formula. **Sensitivity at the dataset mean is now finite for every surviving feature.**
  - **Architectural parity has been resolved.** The single-hidden-layer / `width1 ≤ 6` configuration matches the CPCV-benchmarked KAN, so the formula describes the architecture the benchmark numbers describe, not a simplified surrogate. The current run's [3, 3, 2] architecture uses width1 = 3 (down from 5) and width2 = 3 (up from 2), producing 15 active edges and a four-outer-term decision function that fits one printed page in the thesis.

### 5.4 Limitations (six honest, from defense L1 to L5 plus new L6)

- **L1. Computational power.** Currently 3 seeds across all 6 models (uniform, producing 504 = 28 × 6 × 3 prediction entries). 30 Optuna trials per model per fold, KAN width capped at 6. With more compute: more seeds, more trials, wider HP ranges. The N=8 / 28-split configuration alone roughly doubles the per-experiment runtime relative to N=6 / 15 splits, and the per-fold MDA inside Optuna's TPE inner loop adds roughly 40% on top.
- **L2. Daily OHLCV only.** Discards intraday microstructure. Pipeline is timeframe-agnostic; hourly extension would multiply event count and unlock microstructure features.
- **L3. Symbolic extraction is fold-specific and configuration-specific.** The locked formula is the result of running the extraction on split 27 only with `n_top_features=3`. Per-fold extraction (28 formulas, one per CPCV split) would let the analysis count which features and primitives recur (structural signal vs. one-off noise). The cross-run variation observed to date is substantial: surviving feature sets have rotated `{exchange_supply_pct, atr, sadf, oil_ret_30}` → `{eth_btc_ratio, bb_width, skewness, log_returns_lag1, natgas_ret_30}` → `{eth_btc_ratio, log_returns_lag6, natgas_ret_30}` across the three feature-set configurations, with only `eth_btc_ratio` recurring across all three. The primitive set has rotated even more (see Section 5.3). Per-fold extraction is the highest-leverage methodological extension available to the project: it would let the analysis report "X features recurred in N of 28 folds" instead of "X features recurred in the deployment fold only", which is structurally a much stronger claim.
- **L4. KAN width cap is binding for interpretability.** The `width1 ≤ 6` cap is set so the extracted symbolic formula stays humanly readable. The current run's pruned architecture `[[3, 0], [3, 0], [2, 0]]` uses width1 = 3 and width2 = 3, producing 15 active edges and a four-outer-term decision function that fits one printed page in the thesis. The previous run's architecture `[[5, 0], [2, 0], [2, 0]]` used width1 = 5 and width2 = 2 with 14 edges and a different four-outer-term structure. A larger width1 cap would let the model fit more complex patterns but would also produce a formula too long to interpret. This is an explicit interpretability-vs-capacity trade-off, not a generic constraint imposed by sample size or compute. The recurrence of the four-outer-term structure across configurations suggests the cap is not binding for the discovery of the structure itself, only for the readability of the resulting formula.
- **L5. No regime-conditional analysis.** BTC has had approximately 5 distinct regimes over 2014 to 2026 (the 2017 bubble, 2018-2019 bear, 2020 COVID crash, 2021 bull, 2022 FTX-era contraction, 2023-2026 recovery). Full-sample Sharpe averages over them. A regime-conditional analysis is left for future work, especially given the joint reading "DSR < 0.95 + PBO 0.629 adversarial": full-sample non-significance combined with adversarial cross-section ranking does not rule out within-regime predictability, and the buy-and-hold comparison (median Sharpe 2.27 with max drawdown -0.9998) makes the regime-conditional question financially salient even when full-sample-significance is unattainable.
- **L6. Sub-baseline accuracy on the deployment fold.** The locked-run symbolic extraction posts pre-symbolic accuracy 46.67% (below 50% baseline) and post-symbolic accuracy 51.85% (just above 50% baseline but below the 56.85% majority baseline). Both the Phase-1 dynamic gate (0.5693) and the pre-prune gate (0.5359) fail on the deployment fold. The honest reading: the symbolic representation faithfully captures a PyKAN model that has not extracted directional signal on this fold. This is a real limitation: the symbolic-extraction contribution is methodological (closed-form, 100% sym rate, finite gradients everywhere, six clean primitives), not predictive on the deployment fold itself. A future run could (a) revert `n_top_features` to 5 to give PyKAN more data dimensionality on the deployment fold, (b) test alternative fold selections to find one where PyKAN clears its gates, or (c) accept the sub-baseline result and report it as evidence that closed-form extraction works even on weakly-learning models. The previous locked configuration's `|·|`-primitive non-differentiability flag is now resolved; the v6 limitation list trades the non-differentiability flag for the sub-baseline accuracy flag.

---

## 6. Conclusion

> **Cochrane:** Do not restate all findings. **T-R16:** End with resonance, not a hedge. Target ≤ 10% of the textual part (approximately 3.5 pages).

- One paragraph stating the contribution: full AFML applied to BTC, six-model benchmark, first KAN symbolic formula extraction in this regime, expanded 73-feature universe with multi-model MDA selection over four feature families (TA, mathematical, external macro / crypto-macro / on-chain, autoregressive lags), three-seed configuration uniform across all six models (504 prediction entries from 28 splits × 6 models × 3 seeds), and a methodology improvement that replaces the fixed Phase-1 53% gate with a dynamic majority-baseline gate (`max(0.53, base_rate + 0.01)`) and that exposes `n_top_features` as a configurable symbolic-extraction parameter.
- One paragraph stating the headline: under leakage-free evaluation with multiple-testing correction, the literature's 85% to 95% accuracy claims do not survive. Top DSR among trained models is KAN at 0.2198, far below the 0.95 significance threshold. PBO at 0.629 places model selection in the **adversarial regime** (PBO > 0.5): 22 of 35 IS/OOS partitions see the IS-winner underperform the OOS median, and the leave-one-out PBO analysis shows that **every model's exclusion reduces PBO**, so there is no anchor model in this configuration. The cross-section's IS rankings are anti-informative about OOS performance. Despite this adversarial cross-section, KAN is the only trained model with a strictly-positive bootstrap CI on its median Sharpe (0.0971, 1.2413); every other trained-model CI crosses zero, and only buy-and-hold's CI (1.8903, 2.4685) is similarly strict-positive. Buy-and-hold posts a higher median Sharpe (2.2722) than every trained model, with the structural cost of a median max drawdown of -0.9998: directional-trading strategies on BTC daily direction trade absolute return for downside protection rather than producing alpha relative to the asset's overall trajectory. This is the honest answer to Q1: under AFML evaluation, no architecture demonstrates a statistically significant predictive edge over the multi-model selection pool; KAN posts the strongest single-model evidence (rank-1 trained Sharpe, rank-1 trained DSR, only trained model with bootstrap-strict-positive CI), but this evidence is on the path-level and not multiple-testing-corrected. The PBO trajectory across the four locked configurations evaluated to date (0.26 → 0.37 → 0.34 → 0.629) was stable in the moderate-overfitting band across the first three runs and breaks into the adversarial regime in the current run, which is itself a finding about the sensitivity of cross-section stability to small data refreshes.
- One paragraph on what does survive: the symbolic extraction pipeline produces a humanly auditable closed-form decision function on three features (`eth_btc_ratio`, `log_returns_lag6`, `natgas_ret_30`), with **100% symbolification rate (15 of 15 edges, no skips)** and **post-symbolic accuracy of 51.85%, a 5.18 percentage-point gain over the pre-symbolic 46.67%**. The formula uses six primitives (`sin`, `cos`, `tanh`, `x²`, `x³`, `x⁴`) and four outer terms with structurally distinct outer wrappers (squared, sin, cos, tanh). The previous run's `|·|`-primitive non-differentiability flag is resolved: all three surviving features have finite, well-defined gradients at the dataset mean. The lag-feature finding now reproduces across two consecutive 73-feature locked configurations (`log_returns_lag1` then `log_returns_lag6`), validating the lag-features-in-MDA-pool design choice; the specific lag rotates but the lag group's presence in the surviving set does not. No on-chain feature survives extraction in any of the three 73-feature locked configurations evaluated: the Q2 "on-chain free lunch" hypothesis fails at both the cross-fold stability level (no on-chain feature reaches 50% selection frequency across the 28 folds in any run) and the deployment-fold level (no on-chain feature reaches the symbolic decision function on split 27 in any 73-feature run). The closed-form representation does not cost validation accuracy on this run (in fact gains 5.2 pp, the largest gain magnitude across all four locked configurations) but the absolute level remains below the majority baseline (51.85% < 56.85%), meaning the symbolic representation faithfully captures a model that did not extract directional signal on the deployment fold. Interpretability is a separable contribution from predictive accuracy and does not need to wait for predictive significance to be valuable: 100% symbolification rate, six clean primitives, finite gradients everywhere, and a closed-form formula on three economically interpretable features are themselves the contribution.
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