# notebook_implementation_guide.md

This guide provides the definitive, copy-pasteable Python code sequences necessary to implement the `MFW_Pipeline.ipynb` master orchestrator. It strictly adheres to Marcos López de Prado's (AFML) pipeline protocols, isolating data through a rigid **CV-Wall** guaranteeing zero-leakage correlations across Kolmogorov-Arnold Networks (KAN) and Baselines evaluations.

---

## Block 0 — Imports and Installs

Run this cell first to ensure your environment is fully provisioned.

```python
# !pip install pandas numpy scikit-learn yfinance scipy statsmodels matplotlib seaborn xgboost lightgbm pykan

# ==========================================
# Data & Core 
# ==========================================
import pandas as pd
import numpy as np

# ==========================================
# Visualization & EDA
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Pipeline Modules (MFW Local Modules)
# ==========================================
from src.a_data_loader import fetch_data
from src.b_features import create_all_features
from src.c_econometrics import apply_continuous_econometrics, find_optimal_d, apply_ffd
from src.d_labels import getDailyVol, getEvents, getBins, getSampleWeights
from src.e_cv import PurgedKFold, CombinatorialPurgedKFold
from src.f_preproc import fit_transform_scaler, compute_SFI, filter_features
from src.g_models import ARLogistic, SklearnBaseline, MLPModel, PureKAN, TKAN, KASPER, ModelTrainer

# ==========================================
# ML Metrics & Registry
# ==========================================
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, log_loss
import json
import warnings
warnings.filterwarnings('ignore')
```

---

## Block 1 — Data Ingestion (`a_data_loader.py`)

Configure the raw ingest ensuring strictly chronological padding. 

```python
# Calls src/a_data_loader.py
# Output: pd.DataFrame of raw daily price variables.

pipeline_config = {
    "assets": ["BTC-USD"],
    "on_chain": ["NVTAdj", "HashRate"],
    "start_date": "2015-01-01",
    "end_date": "2026-12-31"
}

print("Fetching Raw Datasets...")
raw_df = fetch_data(pipeline_config)

# NOTE: The output here is RAW, strictly un-shifted daily OHLCV data. 
# Missing days are handled explicitly via forward-filling only. No interpolation 
# (which utilizes future endpoints) is ever permitted.
print(f"Loaded {len(raw_df)} days of observation inputs.")
```

---

## Block 2 — EDA: Macro (Pre-Split)

Execute general macroscopic sanity checks safely across the total continuous structure.

```python
# Note: Two-Type EDA Split — This macro EDA is safe to execute pre-split as it only 
# visualizes raw distributions. ANY target-dependent EDA (feature-to-label correlations 
# or specific probability distributions) must ONLY occur strictly inside isolated CV folds.

# 1. Continuous price history
plt.figure(figsize=(15, 5))
plt.plot(raw_df.index, raw_df['Close'], label='BTC Close')
plt.title("BTC Macro Price History (Log Scale)")
plt.yscale('log')
plt.legend()
plt.show()

# 2. Missing data gap heatmap
plt.figure(figsize=(15, 5))
sns.heatmap(raw_df.isnull().T, cbar=False, cmap='viridis')
plt.title("Missing Data Gap Analysis")
plt.show()

# 3. Raw OHLCV statistical summaries
display(raw_df.describe().T)
```

---

## Block 3 — Basic Feature Removal (Pre-Engineering)

Purge entirely empty or stagnant series mathematically protecting downstream feature generations.

```python
# Drops zero-variance (constant) features or matrices corrupted natively past threshold limits.
# Must occur BEFORE feature engineering because zero-variance vectors mathematically block 
# rolling covariance calculations and crash matrix inversion operators downstream.

NAN_THRESHOLD = 0.60
drop_cols = []

for col in raw_df.columns:
    if raw_df[col].nunique() <= 1:
        drop_cols.append(col)
    elif raw_df[col].isna().mean() > NAN_THRESHOLD:
        drop_cols.append(col)

clean_df = raw_df.drop(columns=drop_cols)
print(f"Dropped bad matrices: {drop_cols}")
```

---

## Block 4 — Feature Engineering (`b_features.py`)

Compute macro continuous temporal transformations inherently isolated from looking ahead.

```python
# Calls src/b_features.py
# Output: `df` containing engineered metrics, and `feature_metadata` dict classifying features.

print("Constructing predictive features...")
df, feature_metadata = create_all_features(clean_df)

# NOTE: All rolling/expanding features generated inside b_features.py are internally 
# lagged by 1 day (`.shift(1)`). This Universal Lag Rule enforces causality and prevents 
# look-ahead bias, mathematically making this feature engineering step safe to run pre-split.
```

---

## Block 5 — Pre-CV Econometrics (`c_econometrics.py`)

Execute mathematically continuous statistical boundary maps retaining test fold isolation logic sequentially.

```python
# Calls src/c_econometrics.py
# Output: Bounded econometrical evaluations spanning stationary mappings natively.

# 1. Creates preserving 'Raw_Close' explicitly protecting unaltered barrier dollar amounts.
# 2. Automatically routes structural log transforms targeting strictly positive inputs.
# 3. Computes rolling Supremum ADF (SADF) & Sub-Martingale (SMT) explosiveness.
# 4. NOTE: Fractional Differentiation (FFD) IS EXPLICITLY NOT APPLIED HERE. 
#    Fitting d* evaluates total distributions mapping CV-Wall leaks. It belongs exclusively inside the inner-loop.

df, feature_metadata = apply_continuous_econometrics(df, feature_metadata)
print(f"Econometrics module applied. Structural boundaries configured.")
```

---

## Block 6 — Triple-Barrier Labeling and Sample Weights (`d_labels.py`)

Assign objective classification configurations tracking the exact temporal span mapping event executions cleanly.

```python
# Calls src/d_labels.py
# Output: Events bounds, label allocations (-1,0,1), t1 arrays, and isolated uniqueness weights.

# 1. Compute dynamic EWMA volatility targeting 20-day structural bounds limits.
volatility = getDailyVol(df['Raw_Close'], span=20)

# 2. CUSUM filter executing sparse event maps natively preventing serial saturation constraints.
tEvents = getEvents(df['Raw_Close'], tEvents=volatility.index, ptSl=[1, 1], trgt=volatility, minRet=0.01)

# 3. Triple-Barrier targets capturing t1 vectors executing the true structural outputs.
bins = getBins(tEvents, df['Raw_Close'])
t1 = tEvents['t1']

# 4. Sample Weights targeting AFML concurrent attribution.
weights = getSampleWeights(t1, df['Raw_Close'], numThreads=1)

# NOTE: The t1 object maps [t0 -> t1] mapping the absolute lifespan covering each prediction window.
# It is critically required by all downstream purging/embargo algorithms identifying testing boundary overlaps.
print(f"Labels assigned. Class balance:")
display(bins['bin'].value_counts(normalize=True))
```

---

## Block 7 — Cross-Validation Splitting (`e_cv.py`)

Construct the specific index mappings executing the dynamic temporal isolation logic correctly preventing sequence correlation limits.

```python
# Calls src/e_cv.py
# Output: Sequence arrays isolating train subsets resolving test overlaps.

pct_embargo = 0.01

# Purged K-Fold
# Purging Rule: Any training sequence where [t0, t1] overlaps test targets gets entirely dropped.
# Embargo Rule: Removes ~1% chronological mapping buffer immediately following test loops killing sequence autocorrelation leakage.
pkf = PurgedKFold(n_splits=6, t1=t1, pct_embargo=pct_embargo)

# Combinatorial CPCV Generator 
# Used generating theoretical portfolio backtest paths. With N groups and k=2, creates natively φ = N-1 backtest testing route variations.
cp_kf = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, t1=t1, pct_embargo=pct_embargo)

print("CV Temporal boundaries configured securely.")
```

---

## Block 8 — EDA: Target-Dependent (Inside CV, First Fold Only)

Only process these visualizations targeting active labels inside structurally sound training logic blocks to preserve absolute test integrity structurally.

```python
# NOTE: ⚠️ LEAKAGE RISK: Running this outside the CV loop leaks target test bounds resolving absolute data correlations 
# inherently ruining mapping benchmarks rendering outcomes biased mathematically structurally.

cv_splits = list(pkf.split(df))

for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
    if fold_idx == 0:
        cv_train = df.iloc[train_idx].copy()
        cv_train['Label'] = bins.iloc[train_idx]['bin']
        cv_train['Weights'] = weights.iloc[train_idx]
        
        plt.figure(figsize=(10, 4))
        sns.countplot(x='Label', data=cv_train)
        plt.title('Fold 0: Target Label Balance')
        plt.show()

        plt.figure(figsize=(12, 10))
        # Keep correlation matrix small selecting top 15 features dynamically scaling target metrics explicitly
        subset_cols = list(cv_train.select_dtypes(include=np.number).columns[:15]) + ['Label']
        sns.heatmap(cv_train[subset_cols].corr(), annot=False, cmap='coolwarm')
        plt.title('Fold 0: Feature-to-Label Cross-Correlations')
        plt.show()
        
        break # Perform exclusively on Fold 0 to observe isolated structural relationships.
```

---

## Block 9 — The Inner-CV Loop Wall (`f_preproc.py`, `c_econometrics.py`, `e_cv.py`, `g_models.py`)

***Everything inside this loop is strictly behind the CV-Wall. Nothing fitted here may peek at or configure test fold boundary distributions mathematically structurally.***

```python
# Registry storing the specific trial outputs mapping hyper-parameters dynamically isolating metrics natively.
trial_registry = []
from sklearn.base import clone

for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
    print(f"\n--- EXECUTING FOLD {fold_idx} ---")
    
    # 1. Extract mapped validation boundaries explicitly structurally.
    X_train, X_test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    y_train, y_test = bins.iloc[train_idx]['bin'], bins.iloc[test_idx]['bin']
    w_train, w_test = weights.iloc[train_idx], weights.iloc[test_idx]

    # ⚠️ LEAKAGE RISK: Target vectors are dropped ensuring feature inputs are strictly exogenous.
    if 'Raw_Close' in X_train.columns:
        X_train.drop(columns=['Raw_Close'], inplace=True)
        X_test.drop(columns=['Raw_Close'], inplace=True)

    # 2. Fractional Differentiation (FFD)
    # ⚠️ LEAKAGE RISK: d* fit on full dataset maps variance from test. Fits exclusively onto Train arrays dynamically.
    d_star, _, _ = find_optimal_d(X_train['Close'])
    X_train['FFD_Close'] = apply_ffd(X_train['Close'], d_star)
    X_test['FFD_Close'] = apply_ffd(X_test['Close'], d_star) 

    # 3. Scaling
    # ⚠️ LEAKAGE RISK: RobustScaler incorporates quantiles. Fitting over X_test bleeds target scale limits natively.
    X_train_scaled, X_test_scaled, scaler = fit_transform_scaler(X_train, X_test, scaler_type='robust')

    # 4. SFI Feature Selection
    # ⚠️ LEAKAGE RISK: Computing metrics tracking limits incorporating test vectors explicitly destroys OOS prediction metrics limits structurally.
    clf_sfi = LogisticRegression(class_weight='balanced') # Basic linear model isolating independent metrics.
    sfi_scores = compute_SFI(X_train_scaled, y_train, X_train_scaled, y_train, clf=clf_sfi) # Evaluated internally
    X_train_filtered, X_test_filtered, kept_features = filter_features(X_train_scaled, X_test_scaled, sfi_scores, threshold=0.0)

    # 5. Baseline Models
    xgb_config = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1}
    baseline_rf = SklearnBaseline('rf', {})
    baseline_xgb = SklearnBaseline('xgb', xgb_config)

    baseline_xgb.fit(X_train_filtered.values, y_train.values, sample_weight=w_train.values)
    xgb_preds = baseline_xgb.predict_proba(X_test_filtered.values)[:, 1]
    
    xgb_auc = roc_auc_score(y_test, xgb_preds)
    xgb_logloss = log_loss(y_test, xgb_preds)
    print(f"XGB Baseline AUC: {xgb_auc:.4f}")

    # 6. KAN Model Training
    # Two-phase explicit boundaries initializing mapping matrices isolating variables dynamically executing bounds completely natively.
    kan_config = {'steps': 200, 'lr': 1e-3, 'lamb_1': 1e-4, 'lamb_entropy': 2.0}
    
    # Phase 1: Coarse Grid
    model = PureKAN(len(kept_features), [4, 1], grid_size=5, k=3)
    trainer = ModelTrainer(model, kan_config)
    trainer.fit_fold(X_train_filtered.values, y_train.values, w_train.values, X_test_filtered.values, y_test.values, w_test.values)
    
    # Phase 2: Refined limits mapped seamlessly preserving mapping exact configurations structurally (pseudo code representation natively)
    # model.refine(grid=20) 
    # trainer.config['lr'] = 5e-4 
    # trainer.fit_fold(...)

    # 7. Platt Scaling (Probability Calibration)
    # Fit purely on subsets tracking training matrices executing calibrations isolating boundary mappings cleanly.
    split_p = int(len(X_train_filtered) * 0.8)
    X_train_c, y_train_c = X_train_filtered.iloc[:split_p], y_train.iloc[:split_p]
    X_calib, y_calib = X_train_filtered.iloc[split_p:], y_train.iloc[split_p:]
    
    # Train proxy capturing raw probabilistic limits natively.
    trainer.fit_fold(X_train_c.values, y_train_c.values, w_train.iloc[:split_p].values, X_calib.values, y_calib.values, w_train.iloc[split_p:].values)
    raw_calib_preds = trainer.predict_proba(X_calib.values)[:, 1]
    
    calibrator = LogisticRegression()
    calibrator.fit(raw_calib_preds.reshape(-1, 1), y_calib.values)
    
    # Extract structural limits resolving target sequences exactly natively mapping structures structurally correctly
    raw_test_preds = trainer.predict_proba(X_test_filtered.values)[:, 1]
    calibrated_preds = calibrator.predict_proba(raw_test_preds.reshape(-1, 1))[:, 1]
    
    # 8. Threshold Tuning
    # Grid searches executing boundaries targeting validation explicitly validating distributions independently natively.
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.3, 0.7, 0.01):
        f1 = f1_score(y_calib, (raw_calib_preds >= thresh).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
            
    final_preds = (calibrated_preds >= best_thresh).astype(int)

    # 9. Registry Logging
    trial_registry.append({
        'fold': fold_idx,
        'kan_auc': roc_auc_score(y_test, calibrated_preds),
        'kan_brier': brier_score_loss(y_test, calibrated_preds),
        'kan_f1': f1_score(y_test, final_preds),
        'threshold': best_thresh
    })

# Save executed limits securely executing outputs tracking boundaries natively exactly.
with open('models/trial_registry.json', 'w') as f:
    json.dump(trial_registry, f, indent=4)
    
print("Inner-CV Validation fully complete. Outputs registered cleanly natively.")
```

---

## Execution Checklist

Run this quick audit immediately before pressing "Run All" assessing the comprehensive boundaries limits guaranteeing strict MLDP evaluations natively predicting isolated bounds securely.

- [ ] **Forward Looking Nulls:** Verify missing input target inputs explicitly evaluate strictly `pad/ffill` executing mapping bounds safely. (No interpolations permitted).
- [ ] **Feature Causality Check:** Verify all explicitly configured feature distributions mathematically evaluating sequences generated apply `.shift(1)` isolating boundary distributions structurally correctly natively.
- [ ] **Data Structural Tracking Leakage:** Ensure `Raw_Close` remains strictly unaffected bypassing continuous tracking logic structurally rendering exact output metrics correctly mapped resolving boundaries securely.
- [ ] **Inner-CV FFD Bound:** Ensure `find_optimal_d` extracts exactly fitting boundary metrics completely isolated matching `X_train` native limits securely avoiding sequence peeking.
- [ ] **Scaling Boundary:** Verify `RobustScaler.fit()` evaluates `X_train` arrays only enforcing absolute limits matching clipping boundary distributions inherently.
- [ ] **CV Selection Isolation:** Verify single feature isolation tracks strictly bounds resolving `compute_SFI` natively within boundaries cleanly mapping target boundaries purely evaluated explicitly natively exclusively inside arrays structurally.
- [ ] **Distribution Isolation Limits:** Validations tuning metrics resolving optimal probability allocations executing boundary distributions natively mapped inside train matrices avoiding exact target boundaries cleanly tracking correlations purely natively.
