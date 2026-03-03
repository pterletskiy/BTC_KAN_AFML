# SKILL: KAN Architecture & Machine Learning Evaluation Protocol

## Context
You are acting as a Lead Machine Learning Engineer for a Master's Thesis in Mathematical Finance. The project predicts the price direction of financial assets (Bitcoin, Equities) using baseline models (Logistic Regression, XGBoost, Random Forest, LSTM) and advanced Kolmogorov-Arnold Networks (KAN). 

## Trigger
Apply these rules strictly whenever creating, refactoring, or modifying files related to model architectures, training loops, hyperparameter tuning, and evaluation (specifically `src/models.py`).

## Core Directives

### 1. Absolute Reproducibility (The Seed Rule)
- Financial machine learning is highly sensitive to random initialization.
- Every training function (e.g., `fine_tune_robust_pykan`, `fine_tune_xgboost`) must accept a `random_seed` parameter (default 42).
- The `set_seed(seed)` function must aggressively lock stochasticity across `numpy`, `random`, `torch` (including `torch.backends.cudnn.deterministic = True`), and any Sklearn/XGBoost constructors.

### 2. Time-Series Validation Restraints
- **No Shuffling:** Never use `train_test_split(..., shuffle=True)` or standard `KFold`.
- When refactoring the evaluation pipelines, strictly preserve the chronological order of the data. Walk-forward validation or strict chronological train/val/test splits must be enforced.

### 3. Decoupling of Models and Data
- Model training functions must be entirely agnostic to data fetching and econometric scaling. 
- A function like `fine_tune_robust_pykan()` must only accept raw matrices/tensors (`X_train`, `y_train`, `X_val`, `y_val`). It should never call `create_ta_features` or `StandardScaler` internally. The data passed to `models.py` is assumed to be fully stationary and scaled by `preprocessing.py`.

### 4. Required Evaluation Metrics
- Accuracy alone is fundamentally flawed for financial time-series due to regime imbalances.
- Every evaluation function must output, at minimum:
  - Directional Accuracy
  - Macro F1-Score (to account for class imbalances in UP/DOWN days)
  - Confusion Matrix components
- For PyKAN specifically, ensure the mathematical expression retrieval (Symbolic regression capabilities of KAN) is preserved and returned as part of the model evaluation dictionary.