# Bitcoin Daily Direction Prediction with Kolmogorov-Arnold Networks 📈

This repository contains a Master's thesis (ISEG, Mathematical Finance) on predicting daily Bitcoin price direction under the leakage-free evaluation protocol from *Advances in Financial Machine Learning* (López de Prado, 2018) and the Kolmogorov-Arnold Network architecture (Liu et al., 2024).

The crypto-ML literature reports 85% to 95% daily-direction accuracy using deep learning architectures trained on fixed-horizon labels with overlapping spans, naive train-test splits, and no statistical correction for multi-trials selection bias. This thesis runs the same prediction problem under the full AFML correction stack (CUSUM events, triple-barrier labels, sample weights, fractional differencing, Combinatorial Purged Cross-Validation, plus DSR, PBO, and DeLong corrections) to test what survives, and extracts a closed-form symbolic formula from the trained KAN as the novel methodological contribution.

## 🚀 Key Features

* **Leakage-free evaluation:** complete AFML stack from event sampling through statistical-correction layer.
* **73-feature universe across four families:** 25 technical, 9 mathematical (AFML Part 4), 29 external (20 macro + 1 crypto-macro + 8 on-chain from CoinMetrics), 10 autoregressive lag features.
* **Six-model apples-to-apples benchmark:** AR Logistic, Logistic Regression, Random Forest, XGBoost, LSTM, KAN, evaluated on identical CPCV splits with identical features, sample weights, and metrics.
* **Closed-form formula extraction:** the trained KAN is distilled into a human-readable SymPy expression for P(Up) via PyKAN's Algorithm 1 pipeline (prune → symbolify → affine fine-tune → simplify).

---

## 🎯 Motivation

### Why Bitcoin

* Largest cryptocurrency by market capitalization.
* Highest cumulative return of any asset class since 2009.
* Trades 24/7 on a transparent, publicly auditable blockchain; on-chain features have no equivalent in traditional finance.
* Daily OHLCV data freely available from 2014, providing roughly 4,200 daily observations.

### Why Kolmogorov-Arnold Networks

* Liu et al. (2024) introduced an MLP alternative whose edges carry learnable B-spline activations.
* Those splines can be distilled into closed-form symbolic primitives, producing interpretable models rather than black-box predictors.
* No prior work applies KANs to BTC daily direction or extracts a symbolic formula from a classification KAN under AFML evaluation; the only prior KAN symbolic-extraction work targets regression problems on relatively predictable series (VIX, stock prices).

---

## 🧭 Research Questions

The four questions map one-to-one onto the four contributions below.

**Q1. Predictability under leakage-free evaluation.** Crypto-ML papers report 85% to 95% accuracy using fixed-horizon labels with overlapping spans and naive train-test splits. Does BTC daily direction remain predictable once labels stop leaking the future and overlapping observations are downweighted? Does any positive Sharpe ratio survive the AFML statistical-correction layer (DSR, PBO, DeLong)?

**Q2. Which feature families carry signal.** Most crypto-ML papers use one or two feature families (typically TA or LSTM-on-prices). There is no consensus on whether macroeconomic, crypto-macro, or on-chain features add information beyond price-derived ones. Among technical, statistical, macroeconomic, crypto-macro, and on-chain features, which families survive multi-model permutation-importance selection across CPCV folds?

**Q3. KAN versus standard model families.** KANs have been applied to VIX forecasting and stock prediction, but never to BTC direction. Where does a KAN sit relative to AR Logistic, Logistic Regression, Random Forest, XGBoost, and LSTM under identical CPCV splits, features, sample weights, and metrics?

**Q4. Closed-form formula extraction.** Existing KAN symbolic-extraction work targets regression problems on relatively predictable series. Can a human-readable mathematical expression for P(Up) be extracted from a CPCV-trained classification KAN while preserving most of the predictive accuracy of the trained network?

---

## 🎯 Contributions

**C1 → Q1. Honest evaluation under the full AFML stack.** End-to-end pipeline applying CUSUM event sampling, triple-barrier labels, sample weights (uniqueness × return-attribution × time-decay), fractional differencing, CPCV with purging and embargo, plus DSR, PBO, and DeLong statistical corrections.

**C2 → Q2. A 73-feature universe spanning four families.** 25 technical features, 9 mathematical features (AFML Part 4), 29 external features (20 macro from FRED, 1 crypto-macro, 8 on-chain from CoinMetrics), and 10 autoregressive lag columns. All 73 features compete in multi-model MDA selection on equal footing.

**C3 → Q3. Six-model apples-to-apples benchmark.** Identical CPCV splits, features, sample weights, and metrics across all six models. Pairwise DeLong AUC tests determine which differences are statistically real rather than due to within-fold sampling noise.

**C4 → Q4. Closed-form symbolic formula from KAN.** A human-readable expression for P(Up), pruned and substituted from the trained KAN via PyKAN's symbolic-extraction pipeline. The novel methodological contribution distinguishing this thesis from a standard model-benchmarking study.

---

## 🛠️ Methodology

The pipeline is organised into three sequential phases.

### Phase 1 — Pre-CPCV: data, features, labels, weights

* **Data load (2014-11-01 to 2026-05-09).** BTC-USD daily OHLCV from Yahoo Finance, 20 macroeconomic series (FRED), 1 crypto-macro series (ETH/BTC ratio), 8 on-chain series (CoinMetrics).
* **Feature engineering.** TA indicators (RSI, MACD, ATR Wilder, Bollinger, ema/vwma ratios, ROC, stochastic, OBV, Chaikin, MFI, CCI, Williams %R, etc.); mathematical features from AFML Part 4 (SADF, Shannon and LZ entropy, Hurst exponent, variance ratio, skewness, kurtosis, Jarque-Bera, realized vol, GK volatility, SMT poly1); external features (14-day / 30-day macro returns, yield-curve slopes, on-chain MVRV, hashrate ROC, exchange supply percentage, transaction-count ROC, fee per tx, active-address ROC); lag features (`log_returns_lag1` through `log_returns_lag30`).
* **CUSUM event sampling.** Filters out quiet periods; produces roughly 1,200 to 1,300 events from the ~4,200 daily bars.
* **Triple-barrier labels.** Parameters `pt_sl = (1.5, 1.5)`, `num_days = 10`, `min_return = 0.02`. Three-class output dropped to binary via `drop_rare(0.085)`.
* **AFML sample weights.** Uniqueness × return-attribution × time-decay, normalized to mean 1, applied end-to-end through training, calibration, and early stopping.

### Phase 2 — CPCV: cross-validated model evaluation

* **CPCV configuration.** `N = 8` groups, `k = 2` test groups per split, producing 28 splits and 7 unique prediction paths.
* **Per-split preprocessing.** Fractional differencing (FFD) per train fold, RobustScaler fit per train fold, multi-model MDA feature selection (RF + Logistic Regression with per-model z-scoring before averaging), `TOP_K_FRAC = 0.20`.
* **Three-way 70/15/15 split inside each training fold.** 70% model-train, 15% validation, 15% calibration. The val partition drives early stopping; the cal partition feeds Platt or vector scaling. The split was widened from an earlier 80/10/10 after a calibration audit revealed systematic P(Up) under-prediction across models.
* **Six models, identical protocol.**

  | Model | Type | Tuning | Notes |
  |-------|------|--------|-------|
  | AR Logistic | Econometric baseline | None | 10 lag columns by name; bypasses MDA |
  | Logistic Regression | Linear | Optuna | Class-balanced, L2 |
  | Random Forest | Tree ensemble | Optuna | Depth ≤ 6, leaf ≥ 15 |
  | XGBoost | Boosted trees | Optuna | Depth ≤ 3, `min_child_weight ∈ [10, 50]`, `gamma ∈ log[0.01, 1.0]` |
  | LSTM | RNN | Optuna | `hidden ∈ {8, 16}`, `dropout ≥ 0.2`, single layer |
  | KAN | Spline network | Optuna | `width1 ∈ [2, 6]`, `grid ∈ {3, 5}`, efficient-kan implementation |

* **Calibration.** Platt scaling for the sklearn-style models, vector scaling with the `b[0] = 0` pin for the PyTorch models, both supporting optional sample-weighted NLL to align calibration with the AFML-weighted training loss.
* **Position sizing.** AFML S-curve from calibrated probabilities, capped at `MAX_BET_SIZE = 0.75`. Per-event returns computed at CUSUM event timestamps.
* **CPCV leakage audit.** Per-split verification of AFML §7.4.1 purging conditions 2 and 3, plus exact embargo accounting.

### Phase 3 — Post-CPCV: statistical correction and symbolic extraction

* **Path-level metrics.** The 28 splits are stitched into 7 prediction paths. Per-path Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, log-loss, AUC.
* **Statistical correction layer.**
  * **DSR (Deflated Sharpe Ratio, AFML Chapter 14).** Uses the exact AFML §14.4 Eq 14.13 inverse-CDF form `(1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e))` for `E[max SR]`; Mertens (2002) skew/kurtosis correction for `σ_SR`. Threshold 0.95.
  * **PBO (Probability of Backtest Overfitting, AFML Chapter 11).** Combinatorially symmetric cross-validation on the path matrix, measuring the in-sample-vs-out-of-sample rank consistency of each model in the candidate pool.
  * **DeLong test.** Pairwise AUC comparison across models, accounting for correlated samples through the shared CPCV folds.
* **Leave-one-out PBO.** Diagnostic identifying which individual model in the pool is driving the multi-model selection-bias signal, supporting the methodology discussion of which models are stable vs unstable contributors to the comparison.
* **Calibration audit.** Per-model mean predicted P(Up) vs the empirical base rate, flagged if outside ±0.03.
* **Buy-and-hold benchmark.** Compared on the same CPCV path structure to contextualise the model Sharpe ratios against the passive strategy of holding BTC throughout each test path.
* **Symbolic extraction.** Re-trains a PyKAN model with hyperparameters matching the CPCV-evaluated efficient-kan on a chosen fold, then runs Algorithm 1 from the VIX-KAN paper: Adam phase → LBFGS warmup → LBFGS sparsity → prune → symbolify → affine fine-tune → SymPy formula extraction. The extracted formula is the closed-form expression for `P(Up)` quoted in the thesis.

---

## 📂 Project Structure

```
thesis/
├── data/                                  # raw OHLCV + macro + on-chain CSVs
├── cache/                                 # computed features, MDA scores, FFD d* per fold
├── src/
│   ├── pre_cpcv/                          # data, features, labels, weights, sampling
│   │   ├── data_loader.py
│   │   ├── features.py
│   │   ├── external_features.py
│   │   ├── labeling.py
│   │   ├── sample_weights.py
│   │   └── pre_cpcv_plots.py
│   ├── cpcv/                              # CPCV core
│   │   ├── cv.py                          # generate_cpcv_splits, purge/embargo
│   │   ├── preprocessing.py               # FFD, scaling, multi-model MDA
│   │   ├── pipeline.py                    # run_cpcv_pipeline orchestration
│   │   ├── tuning.py                      # Optuna search spaces per model
│   │   ├── calibration.py                 # Platt + vector scaling
│   │   ├── models/
│   │   │   ├── base.py
│   │   │   ├── benchmarks.py              # AR Logistic, Logistic Regression
│   │   │   ├── tree_models.py             # RF, XGBoost
│   │   │   ├── lstm_model.py
│   │   │   └── kan_model.py               # efficient-kan
│   │   ├── cpcv_plots.py
│   │   ├── alignment.py
│   │   └── diagnostics.py                 # audit_cpcv_leakage
│   └── post_cpcv/                         # path stitching, statistical tests, symbolic extraction
│       ├── evaluation.py                  # DSR, PBO, DeLong, LOO-PBO
│       ├── path_explorer.py
│       └── symbolic_extraction.py         # PyKAN, Algorithm 1
├── main.ipynb                             # cell-by-cell pipeline driver
├── project_structure.md                   # in-depth methodology documentation
├── src_descriptions.md                    # per-module reference
└── README.md
```

`main.ipynb` is the driver. It runs through Phase 1 (load data, build features, sample CUSUM events, label with triple barriers, compute sample weights), Phase 2 (call `run_cpcv_pipeline` for the six-model evaluation), and Phase 3 (call `analyze_results` for the statistical-correction layer and `run_symbolic_extraction` for the closed-form formula). The two markdown files in the repository root carry the methodology trail and the per-module API reference; they are also intended as context for AI coding assistants operating on the codebase.

---

## 💻 Tech Stack

* **Python 3.11**, **NumPy**, **Pandas**, **SciPy**
* **scikit-learn** for Logistic Regression, Random Forest, and Platt scaling
* **XGBoost** for gradient-boosted trees
* **PyTorch** for the LSTM and the efficient-kan KAN
* **PyKAN** for the symbolic-extraction pipeline (Algorithm 1)
* **SymPy** for closed-form expression manipulation
* **Optuna** for Bayesian hyperparameter tuning with TPE sampler
* **statsmodels** for ADF testing and OLS utilities
* **Matplotlib** and **Seaborn** for visualisation

External data sources: Yahoo Finance (BTC OHLCV, macro), FRED (macro series), CoinMetrics (on-chain).

---

## 📚 Key References

* López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
* Liu, Z., Wang, Y., Vaidya, S., et al. (2024). *KAN: Kolmogorov-Arnold Networks*. arXiv:2404.19756.
* Bailey, D. H. and López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5).
* Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*.

---

**Author:** [Petr Terletskiy](https://www.linkedin.com/in/petr-terletskiy/)
**Context:**  Mathematical Finance Master's Final Work (ISEG)
