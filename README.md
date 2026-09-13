# BTC Direction Prediction Using Kolmogorov-Arnold Networks Within the AFML Framework

**MSc Mathematical Finance thesis** — ISEG, University of Lisbon
**Defended:** 22 July 2026 — **Grade:** 19/20
Petr Terletskiy, supervised by Prof. João Afonso Bastos

This repository is the reproducibility artifact for the thesis. Every
number in Chapter 4 is reproducible from the code and configuration here.

The full thesis PDF is at [`presentation_report/main.pdf`](presentation_report/main.pdf)
(built from the LaTeX sources in `presentation_report/`).

---

## What the thesis does

Applies the Advances in Financial Machine Learning (AFML) methodology of
López de Prado (2018) to daily BTC direction prediction, benchmarking a
Kolmogorov-Arnold Network (KAN) against five baselines (AR Logistic, Logistic
Regression, Random Forest, XGBoost, LSTM) under Combinatorial Purged
Cross-Validation with Deflated Sharpe Ratio, Probability of Backtest
Overfitting, and pairwise DeLong AUC tests. Extracts a closed-form symbolic
approximation of the trained KAN on the most recent CPCV fold.

**Headline finding:** no model achieves statistically significant predictive
performance under leakage-free evaluation with multiple-testing correction,
consistent with the Efficient Market Hypothesis. The KAN ranks first among
trained models by median path Sharpe (0.5588) and DSR (0.2470), but its
confidence interval crosses zero. The symbolic-extraction stage produces a
100%-symbolified closed-form formula on three surviving features (ETH/BTC ratio,
log returns lag 6, 30-day natural gas return) that improves out-of-sample
directional accuracy from 46.67% to 51.85% on split 27.

---

## Repository structure

```
BTC_KAN_AFML/
├── main.ipynb                    # end-to-end pipeline notebook
├── src/
│   ├── pre_cpcv/                 # data loading, features, labelling, weights
│   │   ├── data_loader.py
│   │   ├── features.py
│   │   ├── external_features.py
│   │   ├── labeling.py
│   │   ├── sample_weights.py
│   │   ├── alignment.py
│   │   └── pre_cpcv_plots.py
│   ├── cpcv/                     # CPCV loop, preprocessing, tuning, calibration
│   │   ├── pipeline.py
│   │   ├── cv.py
│   │   ├── preprocessing.py
│   │   ├── tuning.py
│   │   ├── calibration.py
│   │   ├── cpcv_plots.py
│   │   └── models/
│   │       ├── base.py
│   │       ├── benchmarks.py     # AR Logistic, Logistic Regression
│   │       ├── tree_models.py    # Random Forest, XGBoost
│   │       ├── lstm_model.py
│   │       └── kan_model.py
│   └── post_cpcv/                # evaluation, diagnostics, symbolic extraction
│       ├── evaluation.py
│       ├── diagnostics.py
│       ├── path_explorer.py
│       └── symbolic_extraction.py
├── presentation_report/          # thesis LaTeX sources
│   ├── main.tex
│   ├── mfw_references.bib
│   ├── presentation.tex
│   └── chapters/
│       ├── 1_introduction.tex
│       ├── 2_literature_review.tex
│       ├── 3_methodology.tex
│       ├── 4_results.tex
│       ├── 5_discussion.tex
│       └── 6_conclusion.tex
└── backup_old_version/           # earlier iterations, retained for provenance
```

---

## Reproducing Chapter 4

The full pipeline runs end-to-end from `main.ipynb`. Locked configuration
constants that define the thesis's Run E:

| Parameter | Value |
|---|---|
| Data window | 2014-11-01 to 2026-05-09 (`END_DATE = "2026-05-10"`) |
| Total daily bars | 4,208 |
| Binary-labelled events after CUSUM + TBL + zero-class drop | 1,168 |
| Class balance | 56.85% Up / 43.15% Down |
| Feature universe | 73 (25 technical + 9 mathematical + 29 external + 10 lag) |
| CPCV configuration | N = 8 groups, k = 2, 28 splits, 7 paths, 1% embargo |
| Seeds per model per split | 5 |
| Total Optuna trials | 5,600 (28 × 5 × 40) |

Expected runtime: 8 to 14 hours on a laptop CPU for the full pipeline
(intraday feature engineering plus 5,600 Optuna trials plus the CPCV loop).
GPU acceleration for the LSTM and KAN cuts this by roughly a third.

### Environment

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Python 3.11.x is the reference environment. Dependency versions are pinned in
`requirements.txt`.

### External data dependencies

The pipeline pulls three external data sources at runtime:

- **Yahoo Finance** (`yfinance`) for BTC-USD, ETH-USD, and macro asset OHLCV
- **FRED** (`pandas-datareader`) for VIX, DXY, US Treasury yields
- **CoinMetrics Community API** (`coinmetrics-api-client`) for on-chain features

All three are free and require no API keys. Yahoo Finance occasionally revises
historical prices (splits, dividend adjustments); if a rerun produces slightly
different numbers, this is the most likely cause.

---

## Reproducibility note

Full bit-for-bit reproducibility across hardware, OS, Python minor versions,
and PyTorch versions is not guaranteed. Random-seed determinism in PyTorch
holds within a fixed CUDA/CPU backend but not across backends. Yahoo Finance
prices are also occasionally revised. Under a matching environment
(Python 3.11.x, pinned dependencies, same OS), Chapter 4 headline metrics
(Sharpe, DSR, PBO, symbolic-extraction accuracy) should reproduce to within
Monte Carlo noise (±1% relative). Small numerical differences from the
published tables may occur under different hardware or after upstream data
revisions.

---

## Citation

If you use this code or reference the methodology in academic work, please cite:

```bibtex
@mastersthesis{terletskiy2026btckan,
  author  = {Terletskiy, Petr},
  title   = {{BTC} Direction Prediction Using Kolmogorov-Arnold Networks
             Within the {AFML} Framework},
  school  = {ISEG, Lisbon School of Economics and Management,
             University of Lisbon},
  year    = {2026},
  month   = {July},
  type    = {{MSc} Thesis in Mathematical Finance},
  note    = {Defended 22 July 2026, grade 19/20}
}
```

---

## License

Released under the MIT License; see [`LICENSE`](LICENSE).

---

---

## Contact

Petr Terletskiy — [LinkedIn](https://www.linkedin.com/in/pterletskiy/)

For questions about the methodology, the AFML pipeline choices, or the
symbolic-extraction results, opening a GitHub issue on this repo is the
preferred channel; LinkedIn works for anything longer-form.
