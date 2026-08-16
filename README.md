# Bitcoin Daily Direction Prediction with Kolmogorov-Arnold Networks 📈

> **TL;DR** — Six machine-learning architectures trained on BTC daily direction under a leakage-free protocol. No architecture beats buy-and-hold on the same labelling window under multiple-testing correction, but the KAN yields a readable **closed-form symbolic formula** for `P(Up)` — the novel contribution of the work.

This repository contains the code and thesis material for a **Master's Final Work** (ISEG, Mathematical Finance) on predicting daily Bitcoin price direction under the evaluation protocol from *Advances in Financial Machine Learning* (López de Prado, 2018) and the Kolmogorov-Arnold Network architecture (Liu et al., 2024).

Recent BTC ML studies report daily-direction accuracies in the 80% to 90% range, but their evaluation pipelines rarely apply the full AFML correction stack, which is designed to prevent standard-ML evaluation from over-reporting predictability on financial time series. This thesis re-runs the prediction problem under the complete AFML stack (CUSUM events, triple-barrier labels, sample weighting, fractional differentiation, Combinatorial Purged Cross-Validation, plus DSR, PBO, and DeLong AUC corrections), benchmarks six models on identical splits, and extracts a closed-form symbolic formula from the trained KAN as the primary methodological contribution.

---

## 🎯 Headline Results

| Metric | Value |
|---|---|
| KAN median path Sharpe (rank 1 among trained models) | 0.5588 |
| KAN Deflated Sharpe Ratio | 0.2470 (below the 0.95 significance threshold) |
| Probability of Backtest Overfitting (PBO) | 0.657 (adversarial regime) |
| DeLong pairwise significant AUC differences | 2 of 15 |
| Buy-and-hold median Sharpe on same labelling window | 2.2722 (max drawdown −99.98%) |
| Symbolic extraction: primitive recovery | 100% (15 of 15 edges) |
| Post-symbolic test-fold accuracy | 51.85% (+5.18 pp over the spline KAN) |
| Surviving primitives | `sin`, `cos`, `tanh`, `x²`, `x³`, `x⁴` |
| Surviving features | ETH/BTC ratio, log-returns lag 6, natgas 30-day return |

The headline finding: under leakage-free evaluation with multiple-testing correction, no trained architecture demonstrates a statistically significant predictive edge. The KAN symbolic-extraction contribution, however, stands on its own — the pipeline produces a humanly auditable closed-form decision function that behaves like the trained model.

---

## 🚀 Key Features

- **Leakage-free evaluation:** complete AFML stack from event sampling through the statistical-correction layer.
- **73-feature universe across four families:** 25 technical, 9 mathematical (AFML Part 4), 29 external (20 macro + 1 crypto-macro + 8 on-chain from CoinMetrics), 10 autoregressive lag features.
- **Six-model apples-to-apples benchmark:** AR Logistic, Logistic Regression, Random Forest, XGBoost, LSTM, KAN, all evaluated on identical CPCV splits with identical features, sample weights, and metrics.
- **Closed-form formula extraction:** the trained KAN is distilled into a human-readable symbolic expression for `P(Up)` via a pruning and symbolic-extraction pipeline.

---

## 🎯 Motivation

### Why Bitcoin

- Longest continuous price history in the cryptocurrency asset class (11+ years of daily observations).
- Largest market capitalisation and highest liquidity, which reduces microstructure noise in daily bar data.
- Deepest on-chain data coverage among cryptocurrencies.
- Altcoins exhibit high beta to BTC, so BTC captures the systematic cryptocurrency factor through a single instrument.
- Institutional anchoring from the spot-ETF ecosystem (approximately $58 billion in cumulative net inflows since January 2024), currently unique to BTC among cryptocurrencies.

### Why Kolmogorov-Arnold Networks

- Liu et al. (2024) propose KANs as a multi-layer perceptron alternative whose edges carry learnable univariate activation functions rather than fixed activations on the nodes.
- These edge functions can be distilled into closed-form symbolic expressions, making the trained model as auditable as a linear regression while retaining nonlinear representational power.
- To the author's knowledge, no prior work applies KANs to BTC price direction prediction or extracts symbolic formulas from a classification KAN under the AFML framework. Existing KAN symbolic-extraction work in finance targets regression problems on relatively predictable series (VIX, stock prices, cryptocurrency price levels).

---

## 🧭 Research Questions

**Q1. Feature families.** Most cryptocurrency ML studies rely on one or two feature families, typically technical indicators or price histories, and there is no consensus on whether macroeconomic, crypto-macro, or on-chain features add information beyond price-derived inputs. Which families carry signal under a leakage-free evaluation?

**Q2. Predictability under AFML correction.** Many recent BTC ML studies report daily-direction accuracies of 80 to 90 percent, but these figures rest on pipelines whose train-test splits do not purge concurrent observations, so the reported accuracies may be inflated by an unknown margin. Does BTC price direction remain predictable once labels and features stop leaking the future, under the Deflated Sharpe Ratio (DSR), the Probability of Backtest Overfitting (PBO), and the DeLong AUC test?

**Q3. Closed-form formula extraction.** Existing KAN symbolic-extraction work targets regression problems on relatively predictable series. Can a human-readable expression for `P(Up)` be extracted from a CPCV-trained KAN while preserving most of its predictive accuracy?

**Q4. KAN versus standard model families.** KANs have been applied to VIX forecasting, stock prediction, and cryptocurrency regression, but never to BTC direction classification under a uniform protocol. Where does a KAN sit relative to AR Logistic, Logistic Regression, Random Forest, XGBoost, and LSTM under identical CPCV splits, features, sample weights, and metrics?

---

## 🎯 Contributions

**C1 → Q1. A 73-feature universe spanning four families.** 25 technical features, 9 mathematical features (AFML Part 4), 29 external features (20 macro from FRED, 1 crypto-macro, 8 on-chain from CoinMetrics), and 10 autoregressive lag columns. All 73 features compete in multi-model MDA selection on equal footing, no family privileged a priori.

**C2 → Q2. Honest evaluation under the full AFML stack.** End-to-end pipeline applying CUSUM event sampling, triple-barrier labels, sample weighting (uniqueness × return-attribution × time-decay), fractional differentiation, CPCV with purging and embargo, plus DSR, PBO, and DeLong statistical corrections, in a single reproducible workflow.

**C3 → Q3. Closed-form symbolic formula from the KAN.** A human-readable expression for `P(Up)`, pruned and symbolified from the trained KAN via the four-step pipeline (train, prune, symbolify, fine-tune). The first closed-form classification formula derived from a CPCV-trained KAN in a weak-signal regime, and the primary novel contribution of this work.

**C4 → Q4. Six-model apples-to-apples benchmark.** Identical CPCV splits, features, sample weights, and metrics across all six models. Pairwise DeLong AUC tests flag which cross-section differences reach statistical significance.

---

## 🛠️ Methodology

The pipeline is organised into three sequential phases.

### Phase 1 — Pre-CPCV: data, features, labels, weights

- **Data load (2014-11-01 to 2026-05-09).** BTC-USD daily OHLCV from Yahoo Finance, 20 macroeconomic series (FRED), 1 crypto-macro series (ETH/BTC ratio), 8 on-chain series (CoinMetrics).
- **Feature engineering.** Technical (RSI, MACD, ATR Wilder, Bollinger, EMA/VWMA ratios, ROC, stochastic, OBV, Chaikin, MFI, CCI, Williams %R, and others); mathematical (AFML Part 4: SADF, Shannon and Lempel-Ziv entropy, Hurst exponent, variance ratio, skewness, kurtosis, Jarque-Bera, realised vol, Garman-Klass volatility, SMT poly-1); external (14-day and 30-day macro returns, yield-curve slopes, on-chain MVRV, hash-rate ROC, exchange supply percentage, transaction-count ROC, fee per transaction, active-address ROC); and 10 autoregressive lag features.
- **CUSUM event sampling.** Filters quiet periods; produces 1,168 aligned events from the 4,208 daily bars.
- **Triple-barrier labels.** Parameters `pt_sl = (1.5, 1.5)`, `num_days = 10`, `min_return = 0.02`. Three-class output dropped to binary after removing near-zero-return events.
- **AFML sample weights.** Uniqueness × return-attribution × time-decay, normalised to mean 1, applied end-to-end through training, calibration, and early stopping.

### Phase 2 — CPCV: cross-validated model evaluation

- **CPCV configuration.** `N = 8` groups, `k = 2` test groups per split, producing 28 splits and 7 disjoint backtest paths covering all 1,168 events exactly once.
- **Per-split preprocessing.** Fractional differentiation (FFD) per train fold, RobustScaler fit per train fold, multi-model MDA feature selection (RF + Logistic Regression, per-model z-scoring before averaging, `TOP_K_FRAC = 0.20`).
- **Calibration partition.** Two-way 80/20 split of the training fold. The 80% partition is used for model training; the 20% partition serves a dual role as early-stopping signal (XGBoost, LSTM, KAN) and as the calibrator-fitting set (Platt or vector scaling). Rejected alternative: a dedicated third calibration slice, which would have reduced the model-training partition to approximately 600 events.
- **Six models, identical protocol:**

  | Model | Type | Tuning | Notes |
  |---|---|---|---|
  | AR Logistic | Econometric baseline | None | 10 lag columns by name; bypasses MDA |
  | Logistic Regression | Linear | Optuna | Class-balanced, L1/L2 penalty tuned |
  | Random Forest | Tree ensemble | Optuna | `max_depth ∈ [2, 6]`, `min_samples_leaf ∈ [15, 40]`, `n_estimators ∈ [100, 250]` |
  | XGBoost | Boosted trees | Optuna | `max_depth ∈ [1, 3]`, `min_child_weight ∈ [5, 30]`, `gamma ∈ [10⁻⁸, 1]`, 500 trees with early stopping at 20 rounds |
  | LSTM | RNN | Optuna | `hidden ∈ {16, 32}`, `dropout ∈ [0.1, 0.5]`, window=14, single layer |
  | KAN | Spline network | Optuna | `width₁ ∈ [2, 6]`, `grid ∈ {3, 5}`, single hidden layer, spline order 3 |

- **Calibration.** Platt scaling for the sklearn-style models, vector scaling with `b[0] = 0` pinning for the PyTorch models, both fit on the 20% holdout of Section 3.9.
- **Position sizing.** AFML S-curve from calibrated probabilities, capped at `MAX_BET_SIZE = 0.75`. Per-event returns computed at CUSUM event timestamps.
- **CPCV leakage audit.** Per-split verification of AFML §7.4.1 purging conditions and exact embargo accounting.

### Phase 3 — Post-CPCV: statistical correction and symbolic extraction

- **Path-level metrics.** The 28 splits are stitched into 7 prediction paths. Per-path Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, log-loss, AUC.
- **Statistical correction layer:**
  - **DSR (Deflated Sharpe Ratio, AFML Chapter 14).** Uses the AFML §14.4 Eq 14.13 inverse-CDF form `(1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e))` for `E[max SR]`, with Mertens (2002) skew/kurtosis correction for `σ_SR`. Threshold 0.95.
  - **PBO (Probability of Backtest Overfitting, AFML Chapter 11).** Combinatorially symmetric cross-validation on the path matrix, measuring in-sample-vs-out-of-sample rank consistency of each model in the candidate pool.
  - **DeLong test.** Pairwise AUC comparison across models, accounting for correlated samples through the shared CPCV folds.
- **Leave-one-out PBO.** Diagnostic identifying which individual model drives the multi-model selection-bias signal, supporting the discussion of stable versus unstable contributors to the comparison.
- **Calibration audit.** Per-model mean predicted `P(Up)` versus the empirical base rate (0.5685), flagged if outside ±0.03.
- **Buy-and-hold benchmark.** Evaluated on the same CPCV path structure to contextualise trained-model Sharpe ratios against the passive strategy.
- **Symbolic extraction.** Re-trains a KAN with hyperparameters matching the CPCV-evaluated model on split 27, then runs the four-step pipeline: three-phase training (Adam + LBFGS + L1-plus-entropy sparsity) → pruning → symbolification against a 14-primitive candidate set → LBFGS affine fine-tuning → symbolic-algebra simplification. The output is the closed-form expression for `P(Up)` quoted in the thesis.

---

## 📊 Data Availability

- **BTC-USD OHLCV:** downloaded from Yahoo Finance at runtime (`yfinance`); no key required.
- **Macroeconomic series (FRED):** downloaded via the FRED API; free API key required from [https://fred.stlouisfed.org](https://fred.stlouisfed.org).
- **ETH/BTC ratio:** primary source CoinMetrics Community API (free tier), with Yahoo Finance as fallback.
- **On-chain features (CoinMetrics):** free Community API; rate-limited but sufficient for daily-frequency retrieval.

Cached feature and MDA results in `cache/` are reproducible from the raw sources but are also committed for convenience so the CPCV phase can be run without re-computing Phase 1.

---

## 💻 Tech Stack

- **Python 3.11**, **NumPy**, **Pandas**, **SciPy**
- **scikit-learn** for Logistic Regression, Random Forest, and Platt scaling
- **XGBoost** for gradient-boosted trees
- **PyTorch** for the LSTM and the KAN implementation
- **KAN reference implementation** for the symbolic-extraction pipeline
- **Symbolic algebra package** for closed-form expression manipulation
- **Optuna** for Bayesian hyperparameter tuning with the TPE sampler
- **statsmodels** for ADF testing and OLS utilities
- **Matplotlib** and **Seaborn** for visualisation

External data sources: Yahoo Finance (BTC OHLCV, macro), FRED (macroeconomic series), CoinMetrics (crypto-macro and on-chain).

---

## 📚 Key References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Liu, Z., Wang, Y., Vaidya, S., et al. (2024). *KAN: Kolmogorov-Arnold Networks*. arXiv:2404.19756.
- Bailey, D. H. and López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5).
- Bailey, D. H., Borwein, J., López de Prado, M., and Zhu, Q. J. (2015). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4).
- DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. (1988). Comparing the Areas Under Two or More Correlated ROC Curves. *Biometrics*, 44(3).
- Hudson, R. and Urquhart, A. (2021). Technical Trading and Cryptocurrencies. *Annals of Operations Research*, 297.
- Brock, W., Lakonishok, J., and LeBaron, B. (1992). Simple Technical Trading Rules and the Stochastic Properties of Stock Returns. *Journal of Finance*, 47(5).

---

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

---

## 📖 Citation

If you use this work in academic research, please cite:

```bibtex
@mastersthesis{terletskiy_2026_btc_kan,
  author  = {Terletskiy, Petr},
  title   = {{BTC} Direction Prediction Using {Kolmogorov-Arnold} Networks Within the {AFML} Framework},
  school  = {ISEG -- Lisbon School of Economics and Management, Universidade de Lisboa},
  year    = {2026},
  type    = {Master's Final Work, Mathematical Finance},
  url     = {https://github.com/<your-repo-here>}
}
```

---

**Author:** [Petr Terletskiy](https://www.linkedin.com/in/petr-terletskiy/)
**Context:** Mathematical Finance Master's Final Work (ISEG, defended July 2026)
