# MFW Pipeline: Notebook Roadmap

This orchestrator is structured as a reproducible research report, mapping data ingestion through to symbolic extraction. The strict division between Pre-CV and Inner-CV blocks guarantees zero data leakage.

## Pre-CV Phase (Leakage Risk: None)
*These blocks operate on raw or backward-looking data only. No targets or labels are formed or used.*

**Block 0 — Imports & Environment Setup**
*   **Modules Called**: `src.a_data_loader`, `src.b_features` (Phase 1), plus future imports.
*   **Produces**: Environment configuration, logging, and plotting defaults.
*   **Risk**: None. Purely structural.

**Block 1 — Data Ingestion**
*   **Modules Called**: `src.a_data_loader`
*   **Produces**: Raw dataset `raw_df` combining OHLCV, on-chain, and macro.
*   **Risk**: None. Secondary features are naturally lagged 1-day by the loader.

**Block 2 — Macro EDA (Pre-Split Safe)**
*   **Modules Called**: None (pandas/seaborn only)
*   **Produces**: Price history plots, missing data heatmaps, availability charts, descriptive stats.
*   **Risk**: None. Explores raw data without peeking at labels.

**Block 3 — Basic Feature Removal**
*   **Modules Called**: None (pandas filtering)
*   **Produces**: Filtered `raw_df` dropping zero-variance and >60% missing columns.
*   **Risk**: None. Evaluates only structural integrity, not target correlation.

**Block 4 — Feature Engineering**
*   **Modules Called**: `src.b_features`
*   **Produces**: `features_pre_cv.parquet` and `feature_metadata.json` mapping all rolling/technical features.
*   **Risk**: None. Enforces strict 1-day universal lagging for all non-deterministic features.

**Block 5 — Pre-CV Econometrics**
*   **Modules Called**: `src.c_econometrics`
*   **Produces**: SADF/SMT bubble signals and initial stationarity checks.
*   **Risk**: None. Evaluates raw properties exclusively prior to labeling.

**Block 6 — Triple-Barrier Labeling & Sample Weights**
*   **Modules Called**: `src.d_labels`
*   **Produces**: `y` (labels), `t1` (barrier endpoints), and `w` (sample uniqueness weights).
*   **Risk**: None. Creates the targets mapped to specific timestamps.

**Block 7 — Label EDA & Class Balance Report**
*   **Modules Called**: None (pandas/matplotlib)
*   **Produces**: Label distribution histograms and weight decay checks.
*   **Risk**: None. Analyzes the target structure globally without models.

**Block 8 — CV Fold Construction**
*   **Modules Called**: `src.e_cv`
*   **Produces**: `CombinatorialPurgedKFold` paths safely embargoing evaluation datasets.
*   **Risk**: Low. Safe mapping to prevent leakage during fold traversal.

**Block 9 — Target-Dependent EDA (Inside Fold 0 Only)**
*   **Modules Called**: None (seaborn)
*   **Produces**: Feature-to-target correlations inside the first training fold only.
*   **Risk**: Low. Evaluated strictly inside a training cut to visualize relationships without leaking.

## Inner-CV Wall (Leakage Risk: High)
*All transforms must be fit exclusively on training folds.*

**Block 10 — Inner-CV Loop: FFD, Scaling, Feature Selection, Feature Importance**
*   **Modules Called**: `src.c_econometrics`, `src.f_preproc`
*   **Produces**: Stationary representations, Quantile scaled matrices, SFI scoring.
*   **Risk**: High. `fit` called strictly on train; `transform` applied safely to test.

**Block 11 — Inner-CV Loop: Baseline Training**
*   **Modules Called**: `src.g_models`
*   **Produces**: ARLogistic and Random Forest benchmarks trained per path.
*   **Risk**: High. Weight subsets replicated via Sequential Bootstrapping internally.

**Block 12 — Inner-CV Loop: KAN Training (Phase 1+2)**
*   **Modules Called**: `src.g_models`
*   **Produces**: Optimized PyTorch KAN networks utilizing ReduceLROnPlateau.
*   **Risk**: High. Early stopping explicitly guarded utilizing isolated fold validation cuts.

**Block 13 — Inner-CV Loop: Calibration & Threshold**
*   **Modules Called**: `src.g_models` (sklearn calibration)
*   **Produces**: Isotonic combinations mapping raw logits to empirical probabilities.
*   **Risk**: High. Fit uniquely on out-of-sample validation folds per iteration.

**Block 14 — Inner-CV Loop: Metrics & Registry**
*   **Modules Called**: None (JSON/Dict aggregation)
*   **Produces**: Appended backtest performance arrays mapped mathematically.
*   **Risk**: High. Evaluates absolute correctness natively before aggregation.

## Post-CV Phase (Leakage Risk: None)

**Block 15 — Post-CV: Model Selection & Pruning**
*   **Modules Called**: `src.g_models`
*   **Produces**: Best holistic KAN architecture exported for analytical exploration.
*   **Risk**: None. Model is static post-CV.

**Block 16 — Post-CV: Symbolic Extraction & Fidelity**
*   **Modules Called**: `src.h_kan_math_expression`
*   **Produces**: SymPy logical equations, latex exports, R2 score dictionaries.
*   **Risk**: None. Evaluates properties analytically explicitly.

**Block 17 — Post-CV: Regime Generalization Test**
*   **Modules Called**: None (orchestrator analysis)
*   **Produces**: Output tracking KASPER regime detections across historically unseen shocks.
*   **Risk**: None. Pure evaluation tracking.

**Block 18 — Final Report: DSR, SR Distribution, Tables**
*   **Modules Called**: None (orchestrator final layout)
*   **Produces**: Deflated Sharpe Ratio (DSR) metrics and exhaustive pipeline success bounds.
*   **Risk**: None. Static presentation array logically bound to completion.
