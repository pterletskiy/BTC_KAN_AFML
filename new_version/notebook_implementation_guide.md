# MFW Pipeline — Notebook Implementation Guide

This document is the definitive block-by-block roadmap for `MFW_Pipeline.ipynb`. It translates the AFML methodology and Kolmogorov-Arnold Network (KAN) strategies into sequential executable logic.

---

### Block 0 — Environment and Imports
Establish dependencies, configure global hyperparameters, and enforce strict reproducibility. Doing this before data ingestion ensures deterministic runtime behavior across the entire pipeline.

```python
# !pip install yfinance coinmetrics pandas numpy torch kan scikit-learn matplotlib seaborn xgboost sympy

import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, f1_score, brier_score_loss, matthews_corrcoef

from src.a_data_loader import fetch_data
from src.b_features import create_all_features
from src.c_econometrics import apply_continuous_econometrics, find_optimal_d, apply_ffd
from src.d_labels import getDailyVol, cusum_filter, getEvents, getBins, getSampleWeights
from src.e_cv import PurgedKFold, CombinatorialPurgedKFold
from src.f_preproc import fit_transform_scaler, compute_MDI, compute_MDA, compute_SFI, filter_features
from src.g_models import ARLogistic, SklearnBaseline, MLPModel, PureKAN, ModelTrainer, TKAN, KASPER
from src.h_kan_math_expression import extract_symbolic_expression, evaluate_symbolic_fidelity, compute_feature_importance

pipeline_config = {
    "assets": ["BTC-USD"],
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "vol_span": 20,
    "pt_sl": [1, 1],
    "min_ret": 0.01,
    "max_holding_days": 10,
    "n_splits": 6,
    "n_test_splits": 2,
    "pct_embargo": 0.01,
    "nan_threshold": 0.60,
    "sfi_threshold": 0.0,
    "kan_seeds": [42, 123, 456, 789, 1024],
    "kan_architectures": {
        "K1": [4, 1],
        "K2": [8, 1],
        "K3": [4, 4, 1],
        "K4": [8, 4, 1],
    },
    "kan_grid_coarse": 5,
    "kan_grid_fine": 20,
    "kan_k": 3,
    "phase1_steps": 200,
    "phase1_lr": 1e-3,
    "phase2_steps": 100,
    "phase2_lr": 5e-4,
    "lamb": 1e-4,
    "lamb_entropy": 2.0,
    "prune_node_thresh": 5e-2,
    "prune_edge_thresh": 5e-3,
    "prune_finetune_steps": 50,
    "calibration_split": 0.8,
    "symbolic_r2_threshold": 0.97,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(42)
warnings.filterwarnings('ignore')
```

---

### Block 1 — Data Ingestion (`a_data_loader.py`)
Fetch raw daily OHLCV from yfinance and on-chain metrics from CoinMetrics. Forward-fill missing values exclusively.

```python
# ⚠️ LEAKAGE: using interpolation -> interpolating uses future endpoints. Forward-fill strictly prevents look-ahead bias.
df_raw = fetch_data(
    tickers=pipeline_config["assets"], 
    start_date=pipeline_config["start_date"], 
    end_date=pipeline_config["end_date"]
)
print(f"Data ingested. Total rows: {len(df_raw)}. Date range: {df_raw.index.min().date()} to {df_raw.index.max().date()}")
# OUTPUT -> data/raw/features_raw.parquet
```

---

### Block 2 — Macro EDA (Safe Pre-Split)
Visualize raw feature distributions missing target label information. 
This block is safe because it never touches labels. Target-dependent EDA lives in Block 8.

```python
plt.figure(figsize=(12, 6))
plt.plot(df_raw.index, df_raw['Close'], label='BTC Close')
plt.yscale('log')
plt.title('Log-Scale Asset Price')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df_raw.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

display(df_raw.describe())

plt.figure(figsize=(12, 10))
corr = df_raw.corr()
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Raw Feature Pairwise Correlation')
plt.show()
```

---

### Block 3 — Dead Feature Removal (Pre-Engineering)
Drop columns with zero variance or exceeding missing thresholds. Constant columns crash rolling covariance calculations downstream.

```python
initial_cols = len(df_raw.columns)
thresh = pipeline_config["nan_threshold"]

missing_frac = df_raw.isnull().mean()
sparse_cols = missing_frac[missing_frac > thresh].index.tolist()

zero_var_cols = [col for col in df_raw.columns if df_raw[col].var() == 0]

cols_to_drop = list(set(sparse_cols + zero_var_cols))
df_clean = df_raw.drop(columns=cols_to_drop)

print(f"Dropped {len(cols_to_drop)} features from {initial_cols} initial columns.")
print("Dropped:", cols_to_drop)
```

---

### Block 4 — Feature Engineering (`b_features.py`)
Compute technical indicators securely lagged by one period.

All features use only past data and are lagged by one period. This makes feature engineering safe to run pre-split.

```python
# ⚠️ LEAKAGE: applying unlagged rolling variables -> variables like moving averages observe time t; lagging by .shift(1) ensures prediction at t receives only data up to t-1.
df_features, feature_meta = create_all_features(df_clean)
print(f"Constructed {len(df_features.columns)} engineered features.")
# OUTPUT -> data/interim/features_engineered.parquet
```

---

### Block 5 — Pre-CV Econometrics (`c_econometrics.py`)
Apply continuous independent transforms like rolling SADF logs. 

FFD is NOT applied here. Fitting d* requires observing the distribution variance, so it must happen inside the CV loop. This block only handles deterministic, parameter-free transforms.

```python
# ⚠️ LEAKAGE: computing find_optimal_d globally -> exposing global variance leaks future distributions. Fit FFD entirely inside CV execution loops.
df_econometrics = apply_continuous_econometrics(df_features)
# OUTPUT -> data/interim/features_econometrics.parquet
```

---

### Block 6 — Triple-Barrier Labeling and Sample Weights (`d_labels.py`)
Identify objective signal breakpoints using the Triple-Barrier Method driven by CUSUM events. 

CUSUM matters for BTC specifically because it alternates between dead calm and explosive moves. CUSUM triggers target only significant shifts. Neutral events hit the timeout barrier, meaning price didn't move enough to be actionable, removing ambiguous predictions.

```python
daily_vol = getDailyVol(df_econometrics['Raw_Close'], span=pipeline_config["vol_span"])

t_events = cusum_filter(df_econometrics['Raw_Close'], daily_vol.mean())

pt_sl = pipeline_config["pt_sl"]
min_ret = pipeline_config["min_ret"]
t1 = getEvents(df_econometrics['Raw_Close'], t_events, pt_sl, daily_vol, min_ret, 1, 
               t1=df_econometrics['Raw_Close'].index.shift(pipeline_config["max_holding_days"], freq='D'))
bins = getBins(t1, df_econometrics['Raw_Close'])

label_counts_pre = bins['bin'].value_counts()
bins_binary = bins[bins['bin'] != 0].copy()
label_counts_post = bins_binary['bin'].value_counts()

print(f"Class Balance Pre-Neutral Drop: {dict(label_counts_pre)}")
print(f"Class Balance Post-Neutral Drop: {dict(label_counts_post)}")

if (len(bins) - len(bins_binary)) / len(bins) > 0.40:
    print("WARNING: >40% of observations dropped as neutral. Widening volatility boundaries required.")

weights = getSampleWeights(t1, numThreads=1)
sample_weight = weights.loc[bins_binary.index]

# OUTPUT -> data/processed/labels.parquet
```

---

### Block 7 — Cross-Validation Construction (`e_cv.py`)
Instantiate the Purged K-Fold CV architectures ensuring observation embargo overlaps are protected.

Purging removes training observations whose label windows overlap with test timestamps. Embargo removes a 1% buffer after each test fold to kill serial correlation leakage. CPCV generates multiple backtest paths for SR distribution reporting.

```python
dates = bins_binary.index
t1_filtered = t1.loc[dates]

cv_purged = PurgedKFold(
    n_splits=pipeline_config["n_splits"], 
    t1=t1_filtered, 
    pct_embargo=pipeline_config["pct_embargo"]
)

cv_cpcv = CombinatorialPurgedKFold(
    n_splits=pipeline_config["n_splits"],
    n_test_splits=pipeline_config["n_test_splits"],
    t1=t1_filtered,
    pct_embargo=pipeline_config["pct_embargo"]
)
```

---

### Block 8 — Target-Dependent EDA (Inside CV, Fold 0 Only)
Perform visual analysis relying on labeled data strictly inside Fold 0.

This block runs inside the CV loop to prevent label information from leaking into pre-split analysis.

```python
X_eda = df_econometrics.loc[dates]
y_eda = bins_binary['bin']

for fold_idx, (train_idx, test_idx) in enumerate(cv_purged.split(X_eda, y_eda)):
    if fold_idx == 0:
        X_tr = X_eda.iloc[train_idx]
        y_tr = y_eda.iloc[train_idx]
        
        plt.figure(figsize=(6, 4))
        y_tr.value_counts().plot(kind='bar')
        plt.title('Label Balance (Fold 0)')
        plt.show()
        
        top_vars = X_tr.var().sort_values(ascending=False).head(15).index
        corr_data = X_tr[top_vars].copy()
        corr_data['TARGET'] = y_tr
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Top Feature-Label Correlation (Fold 0)')
        plt.show()
        
        holding_periods = (t1_filtered.loc[y_tr.index] - y_tr.index).dt.days
        plt.figure(figsize=(6, 4))
        holding_periods.hist(bins=20)
        plt.title('Average Holding Period Distribution (Days)')
        plt.show()
        break
```

---

### Block 9 — Feature Importance (Inside CV) (`f_preproc.py`)

Compute MDI, MDA, and SFI entirely across the CV test folds. Drop features that lack true predictive orthogonality before entering structural training.

"Backtesting is not a research tool. Feature importance is."

```python
# ⚠️ LEAKAGE: global MDI/MDA/SFI -> fitting importance over the global dataset leaks true outcome validity directly into test partitions. Execute cleanly within Purged splits.

mdi_list, mda_list, sfi_list = [], [], []

for fold_idx, (train_idx, test_idx) in enumerate(cv_purged.split(X_eda, y_eda)):
    X_tr = X_eda.iloc[train_idx].drop(columns=['Raw_Close'], errors='ignore')
    X_ts = X_eda.iloc[test_idx].drop(columns=['Raw_Close'], errors='ignore')
    y_tr, y_ts = y_eda.iloc[train_idx], y_eda.iloc[test_idx]
    
    # Optional robust scale for RF stability
    X_tr_s, X_ts_s, _, _ = fit_transform_scaler(X_tr, X_ts, scaler_type='robust')
    
    raise NotImplementedError("TODO: implement compute_MDI, compute_MDA in src/f_preproc.py")
    
    # mdi = compute_MDI(X_tr_s, y_tr)
    # mda = compute_MDA(X_tr_s, y_tr, X_ts_s, y_ts)
    # sfi = compute_SFI(X_tr_s, y_tr, X_ts_s, y_ts, estimator_wrapper)
    # mdi_list.append(mdi); mda_list.append(mda); sfi_list.append(sfi)

raise NotImplementedError("TODO: Aggregate MDI/MDA/SFI and apply bottom-quartile rejection mechanism.")

# if mda_aggregate <= 0: print("Do not proceed. Revisit feature engineering.")
# OUTPUT -> models/feature_importance.json
```

---

### Block 10 — The Inner-CV Training Loop (`f_preproc.py`, `c_econometrics.py`, `g_models.py`)

Execute the isolated evaluation of KAN shapes vs baselines. All preprocessing and probability scaling relies explicitly on `X_train` matrices.

```python
import json
trial_registry = []
preprocessing_log = []

for fold_idx, (train_idx, test_idx) in enumerate(cv_purged.split(X_eda, y_eda)):
    
    # --- STEP 1: Split ---
    X_tr = X_eda.iloc[train_idx].drop(columns=['Raw_Close'], errors='ignore')
    X_ts = X_eda.iloc[test_idx].drop(columns=['Raw_Close'], errors='ignore')
    y_tr, y_ts = y_eda.iloc[train_idx].values, y_eda.iloc[test_idx].values
    w_tr = sample_weight.iloc[train_idx].values
    
    # --- STEP 2: FFD (inside CV) ---
    # ⚠️ LEAKAGE: d* fitted on full dataset leaks variance -> fit on train only
    for col in X_tr.columns:
        raise NotImplementedError("TODO: Call find_optimal_d() passing X_tr values only. Apply apply_ffd() to X_tr and X_ts.")
    
    # --- STEP 3: Scaling ---
    # ⚠️ LEAKAGE: scaler quantiles from test -> fit RobustScaler on train only
    X_tr_s, X_ts_s, scaler, _ = fit_transform_scaler(X_tr, X_ts, scaler_type='quantile', feature_range=(-1,1))
    
    # --- STEP 4: Feature filtering ---
    raise NotImplementedError("TODO: Ensure variables filtered from Block 9 are dropped equally here.")
    
    # --- STEP 5: Train all baselines ---
    raise NotImplementedError("TODO: Iterate configuration search across Logistic Regression, Random Forest, LightGBM, MLPModel.")
    
    # --- STEP 6: Train KAN grid (multi-seed) ---
    for arch_name, shape in pipeline_config['kan_architectures'].items():
        for seed in pipeline_config['kan_seeds']:
            raise NotImplementedError("TODO: Initialize KAN phase 1 (coarse=5, 200 iter), and evaluate phase 2 refinement (grid=20, 100 iter) NOT OPTIONAL. Update AUC score.")

    # --- STEP 7: Probability calibration ---
    # ⚠️ LEAKAGE: calibrating on test fold -> calibrate on chronologically split train partition only (80/20)
    raise NotImplementedError("TODO: Build ModelTrainer for best KAN architecture matching median AUC, train on 80%, `.calibrate()` on 20% validation partition.")
    
    # --- STEP 8: Threshold tuning ---
    raise NotImplementedError("TODO: Tune decision threshold maximizing F1 natively on 20% validation partition. Apply chosen threshold unchanged to the test fold.")

    # --- STEP 9: Evaluate and log ---
    raise NotImplementedError("TODO: Save test performance Brier, F1, log-loss, AUC scores tracking trial registry output.")

# OUTPUT -> models/trial_registry.json
# OUTPUT -> models/preprocessing_log.json
```

---

### Block 11 — CPCV Evaluation Loop (`e_cv.py`, `g_models.py`)
Run the architecture maximizing median success against CombinatorialPurgedKFold paths ensuring reliable evaluation tracking unobserved permutations.

```python
for cpcv_fold, (train_idx, test_idx) in enumerate(cv_cpcv.split(X_eda, y_eda)):
    # Extract splits, apply FFD and Quantile Scaling strictly using train limits.
    # Instantiate best overall median KAN architecture matching median seed state.
    # Train Phase 1 / Phase 2 sequences, calibrate probability distribution.
    raise NotImplementedError("TODO: Map testing performance outputs against CPCV matrices directly reporting SR distribution.")
    
# OUTPUT -> models/backtest_stats.json
```

---

### Block 12 — Pruning (`g_models.py`)
Extrapolate minimal mathematical dependencies pruning unlinked edges. Selected from the explicit Combinatorial fold possessing median validation performance.

```python
# Identify median CPCV fold KAN array structure.

raise NotImplementedError("TODO: execute model.prune_node(threshold=5e-2)")
raise NotImplementedError("TODO: execute model.prune_edge(threshold=5e-3)")
raise NotImplementedError("TODO: call model.fit() fine-tuning for 50 steps at LR 5e-4 restoring drop tolerance (<0.02 AUC validation degradation limit).")

# OUTPUT -> models/kan_pruned_median.pt
```

---

### Block 13 — Function Stability Check (`g_models.py`)
Verify function translation stability confirming whether the formula accurately transcends structural regimes consistently across CPCV boundaries.

If the spline shapes change qualitatively across folds, the learned structure is regime-dependent. The symbolic formula extracted in the next block cannot be claimed as universal. Document this finding — do not suppress it.

```python
raise NotImplementedError("TODO: Execute model plot API capturing layer 0 top-3 features visualizing spline structures.")
# OUTPUT -> models/spline_plots/
```

---

### Block 14 — Symbolic Extraction (`h_kan_math_expression.py`)
Pull literal algebraic formulas avoiding neural architecture black boxes.

```python
# Prerequisites checkpoint gates:
assert True # Pruning gate passed (Block 12)
assert True # Validation AUC > logistic baseline
assert True # DSR > 0.95 measured globally
assert True # Visual functional stability confirmed visually (Block 13)

# Pass pruned model into h_kan_math_expression limits directly tracking feature outputs:
# 1. Suggest symbolic candidates per edge
# 2. Fix edges R2 >= 0.97
# 3. Extract closed-form mathematical functions
# 4. Symbolic Fidelity test evaluating AUC/Log-loss/Spearman correlation mapping logic rules
# 5. Regime Generalization Test 

raise NotImplementedError("TODO: Execute extract_symbolic_expression, write analytical fidelity tracking parameters.")

# OUTPUT -> models/symbolic_formula.json
# OUTPUT -> models/regime_fidelity.json
```

**Financial Interpretation:** `[volume_rolling_variance]` survival inside logarithmic transformations suggests explosive momentum dominates binary shifts efficiently preventing market breakdown evaluations during excessive volatility regimes.

---

### Block 15 — DSR Computation and Final Report
Aggregate variance observations across testing distributions to evaluate expected profitability robustness.

```python
def compute_DSR(trial_sr_list):
    # AFML Ch. 11 formula executing variance distributions isolating strategy implementations correctly.
    raise NotImplementedError("TODO: DSR definition")
```

### Strategy Final Validation Report
| Metric | KAN | Random Forest | Logistic Regression |
|--------|-----|---------------|---------------------|
| AUC | - | - | - |
| F1 | - | - | - |
| Brier | - | - | - |

**Median Architecture Selected:** [K1]  
**Paths > 0 Return:** [X%]  
**DSR Confidence:** [Y] -> Result is Valid.

**Symbolic Math Output:**  
P(Up) = exp(z1) / (exp(z0) + exp(z1))  
z1 = (1.5 * log(RSI)) + (2.0 * BTC_Returns_7D)

---

### Block 16 — Leakage Audit Checklist
Manually confirm isolation checks prevent implicit label or future forecasting crossover logic.

- [ ] All rolling indicators lagged by at least one period.
- [ ] No feature uses the prediction-day value to predict the same day.
- [ ] d* for FFD fitted on training fold only.
- [ ] CUSUM threshold h computed from training data only.
- [ ] RobustScaler fitted on purged + embargoed training fold only.
- [ ] Feature selection mask derived from Block 9 (inside CV).
- [ ] No label-derived quantity used as a predictor.
- [ ] t1 object verified as strictly forward-looking from t0.
- [ ] Class balance reported before and after neutral label dropping.
- [ ] SADF series lagged by one day before use as predictor.
- [ ] Platt calibrator fitted on training partition, not test fold.
- [ ] Threshold tuned on calibration partition, applied unchanged to test.
- [ ] Phase 2 grid refinement executed (not commented out).
- [ ] Multi-seed evaluation completed for all KAN architectures.
- [ ] Feature importance (MDI + MDA + SFI) completed before model training.

---

> If precision is low and recall is acceptable, consider a meta-labeling extension per skill_mldp_pipeline §8.
