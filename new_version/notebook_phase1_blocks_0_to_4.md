# Phase 1: Notebook Blocks 0–4

Paste the following blocks sequentially into `MFW_Pipeline.ipynb`. Ensure each markdown cell is separated strictly from its associated code cell.

---

### [Markdown Cell]
# Block 0 — Imports & Environment Setup
This initialization block configures the environment, establishing plotting defaults, enabling explicit Python logging, and securing all pipeline dependencies. It is executed once globally.

### [Code Cell]
```python
import json
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

# Configure visual defaults
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'figure.figsize': (14, 5), 'figure.dpi': 100})

# Point loggers to stdout for monitoring src/ outputs
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Phase 1 Pipeline Modules
from src.a_data_loader import (
    ASSET_CATALOG,
    DEFAULT_BTC_CONFIG,
    DEFAULT_BTC_FULL_CONFIG,
    load_dataset,
    load_from_config
)
from src.b_features import (
    create_all_features,
    create_lagged_features,
    create_onchain_features,
    create_ta_features,
    drop_warmup_nans,
    filter_correlated_features
)

# Phase 2+ Pipeline Modules
# from src.c_econometrics import apply_ffd, compute_sadf_signal
# from src.d_labels import getEvents, getBins, getSampleTW
# from src.e_cv import CombinatorialPurgedKFold, PurgedKFold
# from src.f_preproc import fit_transform_scaler, filter_features
# from src.g_models import PureKAN, ARLogistic, ModelTrainer
# from src.h_kan_math_expression import extract_symbolic_expression, print_trading_equations

print(f"Environment Initialized.")
print(f"Pandas Version: {pd.__version__}")
print(f"NumPy Version: {np.__version__}")
```

---

### [Markdown Cell]
# Block 1 — Data Ingestion
Fetching raw datasets utilizing the centralized `a_data_loader` protocol. The fetched data represents untransformed daily grids aligned securely to UTC Midnight.

To preserve causal stability natively, the loader naturally lags all secondary features (Macro, On-chain) by 1 day, while the primary OHLCV matrix remains unshifted.

### [Code Cell]
```python
# Fetch raw OHLCV alongside secondary macro/onchain subsets
raw_df = load_from_config(DEFAULT_BTC_FULL_CONFIG)

base_cols = ["Open", "High", "Low", "Close", "Volume"]
onchain_cols = [c for c in raw_df.columns if c not in base_cols and c not in ["DXY", "VIX", "US10Y"]]
macro_cols = [c for c in raw_df.columns if c in ["DXY", "VIX", "US10Y"]]

print("=" * 60)
print(f"Dataset Shape: {raw_df.shape[0]} Rows, {raw_df.shape[1]} Columns")
print(f"Date Range: {raw_df.index.min().date()} to {raw_df.index.max().date()}")
print("-" * 60)
print(f"Primary Target Cols: {base_cols}")
print(f"Macro Cols: {macro_cols}")
print(f"On-Chain Cols: {onchain_cols[:5]} ... ({len(onchain_cols)} total)")
print("=" * 60)

display(raw_df.head(3))
display(raw_df.tail(3))
```

---

### [Markdown Cell]
# Block 2 — Macro EDA (Pre-Split Safe)
Executing structural validations exploring underlying asset properties. This is mathematically safe globally because it examines independent feature distributions strictly avoiding target relationships or label generation. 

Any EDA requiring label boundaries is deferred strictly to Phase 2 (Inside the CV Loop) to avoid data leakage.

### [Code Cell]
```python
# 1. BTC Price History
halvings = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])

plt.figure(figsize=(14, 5))
plt.plot(raw_df.index, raw_df['Close'], color='black', linewidth=1.5, label="BTC-USD Close")
for h in halvings:
    if h in raw_df.index or (h >= raw_df.index.min() and h <= raw_df.index.max()):
        plt.axvline(h, color='red', linestyle='--', alpha=0.7, label=f"Halving ({h.date()})")
plt.yscale('log')
plt.title("BTC-USD Price History (Log Scale) with Halving Epochs")
plt.ylabel("Price (USD)")
plt.legend(loc="upper left")
plt.show()

# 2. Missing Data Heatmap (Transposed)
plt.figure(figsize=(14, 8))
sns.heatmap(raw_df.T.isna(), cmap="viridis", cbar=False, xticklabels=250)
plt.title("Global Missing Data Heatmap")
plt.xlabel("Timeline")
plt.show()

# 3. Data Availability Timeline
first_valid = raw_df.apply(lambda x: x.first_valid_index()).sort_values()
plt.figure(figsize=(10, 8))
first_valid.plot(kind='barh', color='steelblue')
plt.title("Feature Availability Timeline (First Non-NaN Date)")
plt.xlabel("Date of inception")
plt.show()

# 4. Raw Distributions Summary
summary_df = raw_df.describe().T
summary_df['nan_fraction'] = raw_df.isna().mean()
summary_df['is_zero_variance'] = (summary_df['min'] == summary_df['max'])

print("\n--- Features Flagged for Instability ---")
flagged_nans = summary_df[summary_df['nan_fraction'] > 0.5]
flagged_vars = summary_df[summary_df['is_zero_variance']]

if not flagged_nans.empty:
    print(f"\nExceeds 50% NaNs:\n{flagged_nans['nan_fraction'].round(3)}")
if not flagged_vars.empty:
    print(f"\nZero Variance:\n{flagged_vars.index.tolist()}")

display(summary_df.head(10))
```

---

### [Markdown Cell]
# Block 3 — Basic Feature Removal
Eliminating mathematically broken columns before extensive mathematical feature engineering occurs. Dropping structures exhibiting zero-variance or unrecoverable NaN blocks prevents volatile downstream `ValueError` crashes during rolling window aggregation.

Note: Feature Selection (measuring predictive importance) is strictly executed later inside the Inner-CV loop. This is purely a pre-processing validation.

### [Code Cell]
```python
orig_shape = raw_df.shape
nan_fractions = raw_df.isna().mean()

# Identify drops structurally
cols_to_drop = set()

# 1. Missing data > 60%
high_nans = nan_fractions[nan_fractions > 0.60].index
cols_to_drop.update(high_nans)

# 2. Zero Variance
zero_var = [col for col in raw_df.columns if raw_df[col].nunique() <= 1]
cols_to_drop.update(zero_var)

# Execute drops safely
clean_df = raw_df.drop(columns=list(cols_to_drop))

print(f"Dropped {len(cols_to_drop)} columns total.")
print(f"High-NaN Drops: {list(high_nans)}")
print(f"Zero-Variance Drops: {list(zero_var)}")
print(f"Shape transformed: {orig_shape} -> {clean_df.shape}")

print("\nRemaining Top 10 NaNs by count:")
print(clean_df.isna().sum().sort_values(ascending=False).head(10))
```

---

### [Markdown Cell]
# Block 4 — Feature Engineering
Generating predictive structures from clean pricing/on-chain matrices.

`b_features.py` applies a Universal Lag Rule: all rolling/expanding features are shifted by 1 day (`.shift(1)`) to isolate and prevent look-ahead bias. Cyclical metrics and deterministic event counters remain unshifted because they are universally known in advance. This makes the feature generation fully Pre-CV safe.

The `raw_level` variables (e.g. EMAs, ATR) are intrinsically non-stationary and will undergo Fractional Differentiation (FFD) exclusively inside the CV splits.

### [Code Cell]
```python
# Instantiate robust features
feat_df, feature_metadata = create_all_features(
    clean_df, 
    include_ta=True, 
    include_onchain=True, 
    drop_correlated=False # Preserved intentionally until CV evaluation
)

# Parse output metadata
raw_level_feats = [col for col, tag in feature_metadata.items() if tag == "raw_level"]
tags_series = pd.Series(feature_metadata)

print("=" * 60)
print(f"Total Features Generated: {len(feature_metadata)}")
print("=" * 60)
print(tags_series.value_counts())
print("\n[WARNING] The following `raw_level` features REQUIRE downstream FFD:")
print(raw_level_feats[:5] + ["..."])

# Matrix presentation
meta_df = pd.DataFrame(list(feature_metadata.items()), columns=["Feature", "Tag"]).sort_values("Tag")
display(meta_df.head(10))

# Visualize initial warmup matrices
plt.figure(figsize=(14, 4))
plt.hist(feat_df[list(feature_metadata.keys())].isna().sum(), bins=30, color='darkorange')
plt.title("Distribution of NaN counts per feature (Rolling Window Warmup)")
plt.xlabel("Number of Leading NaNs")
plt.ylabel("Feature Count")
plt.show()

# Purge Warmup Vectors
feat_df_purged, rows_dropped = drop_warmup_nans(feat_df, list(feature_metadata.keys()))
print(f"\nDropped {rows_dropped} warm-up rows.")
print(f"Usable Date Range: {feat_df_purged.index.min().date()} to {feat_df_purged.index.max().date()}")

# Save intermediate outputs for downstream persistence
interim_dir = Path("data/interim")
interim_dir.mkdir(parents=True, exist_ok=True)

feat_df_purged.to_parquet(interim_dir / "features_pre_cv.parquet")
with open(interim_dir / "feature_metadata.json", "w") as f:
    json.dump(feature_metadata, f, indent=4)
    
display(feat_df_purged.head(3))
```
