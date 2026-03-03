# SKILL: Econometrics & Data Preprocessing Standards

## Context
You are acting as a rigorous Econometrics and Machine Learning Research Assistant for a Master's Thesis in Mathematical Finance. This project predicts financial asset direction (e.g., Bitcoin) using on-chain data, OHLCV data, and Kolmogorov-Arnold Networks (KAN). 

## Trigger
Apply these rules strictly whenever creating, refactoring, or modifying files related to data preprocessing, feature engineering, or econometric testing (specifically `src/econometrics.py` and `src/preprocessing.py`).

## Core Directives

### 1. Absolute Prevention of Data Leakage (The Golden Rule)
- **Train/Test Separation:** Never apply global transformations before splitting the dataset temporally. 
- **Scaling/Normalizing:** Any scaler (e.g., `StandardScaler`, `MinMaxScaler`) or statistical transformation MUST be `fit` ONLY on the `X_train` partition. The `X_val` and `X_test` partitions must only be `transformed` using the parameters learned from the training set.
- **Validation Pipeline:** The function `preprocess_evaluation_set()` must strictly enforce this forward-looking paradigm. You must never look ahead in the time series.

### 2. Time-Series Stationarity
- Financial time-series data for Machine Learning must be stationary.
- Preserve all logic related to the Augmented Dickey-Fuller (ADF) and KPSS tests.
- When refactoring `frac_diff_ffd` (Fractional Differencing), ensure that the Pandas DataFrame/Series datetime indices are perfectly preserved. Dropping the time index during differencing is a fatal error.

### 3. Feature Selection & Dimensionality Reduction
- **Correlation Filtering:** The function `remove_highly_correlated_features()` must evaluate correlations based on the training data only.
- **Importance Metrics:** Mutual Information and Random Forest Feature Importance calculations must only be executed on the `X_train` and `y_train` sets to prevent look-ahead bias.

### 4. Code Quality & Imports
- Ensure Type Hinting (`-> pd.DataFrame`, `: pd.Series`) is used for all econometric functions.
- Keep dependencies explicit. Do not import `matplotlib` or `seaborn` in `econometrics.py` or `preprocessing.py` unless absolutely necessary for a specific plotting function; otherwise, return the data and let the master Jupyter Notebook handle the plotting.