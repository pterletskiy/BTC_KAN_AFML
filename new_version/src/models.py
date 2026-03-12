# models.py — Machine Learning models for the MFW Asset Direction Predictor.

import itertools
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kan import KAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    MaxPooling1D)

from tensorflow.keras.models import Sequential

import xgboost as xgb

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Absolute Reproducibility (The Seed Rule)
# ═══════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42) -> None:
    """Aggressively lock all sources of stochasticity.

    Covers: ``random``, ``numpy``, ``torch`` (CPU + CUDA + cuDNN),
    ``tensorflow``, and environment variables.

    Parameters
    ----------
    seed : int
        Random seed (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TensorFlow
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass  # older TF versions

    logger.info("Random seed locked: %d", seed)


# ═══════════════════════════════════════════════════════════════════════════
# Standardized Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None,
                   label_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute standardized evaluation metrics for a binary classifier.

    Every model function in this module returns metrics through this
    function to ensure consistency.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted class labels.
    y_pred_proba : array-like, optional
        Predicted probabilities for the positive class (for ROC-AUC).
    label_names : list of str, optional
        Names for ``[class_0, class_1]``.  Default ``['Down (0)', 'Up (1)']``.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``f1_macro``, ``roc_auc`` (if proba given),
        ``confusion_matrix``, ``classification_report``.
    """
    if label_names is None:
        label_names = ["Down (0)", "Up (1)"]

    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report_dict": classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True,
        ),
        "classification_report_text": classification_report(
            y_true, y_pred, target_names=label_names,
        ),
    }

    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics

def print_evaluation(metrics: Dict[str, Any], model_name: str = "Model",
                     best_params: Optional[Dict[str, Any]] = None) -> None:
    """Pretty-print evaluation results.

    Shows: header, best hyperparameters, F1 + AUC, sklearn classification report.
    """
    print("\n" + "=" * 60)
    print(f"  {model_name} — Evaluation Results")
    print("=" * 60)

    if best_params:
        print("\n  Best Hyperparameters:")
        for k, v in best_params.items():
            print(f"    {k}: {v}")

    print(f"\n  Macro F1-Score : {metrics['f1_macro']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")

    print(f"\n{metrics['classification_report_text']}")


def plot_evaluation(metrics: Dict[str, Any], model_name: str = "Model",
                    feature_importances: Optional[pd.Series] = None,
                    top_n: int = 10, figsize: Tuple[int, int] = (14, 5)) -> None:
    """Plot confusion matrix and feature importance side by side.

    Parameters
    ----------
    metrics : dict
        Output of :func:`evaluate_model`.
    model_name : str
        Title prefix.
    feature_importances : pd.Series, optional
        Feature importance / coefficient series (top *top_n* shown).
    top_n : int
        Number of features to show in the importance plot.
    figsize : tuple
        Figure size.
    """
    has_imp = feature_importances is not None and len(feature_importances) > 0
    ncols = 2 if has_imp else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    # --- Confusion Matrix ---
    cm = metrics["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
    axes[0].set_title(f"{model_name} — Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # --- Feature Importance (top N) ---
    if has_imp:
        top = feature_importances.abs().sort_values(ascending=False).head(top_n)
        top = top.sort_values(ascending=True)  # horizontal bar: largest at top
        top.plot.barh(ax=axes[1], color="steelblue")
        axes[1].set_title(f"{model_name} — Top {top_n} Feature Importances")
        axes[1].set_xlabel("Importance")

    plt.tight_layout()
    plt.show()


def select_best_per_category(results: Dict[str, Dict[str, Any]],
                             benchmark: str = "AR Logistic",
                             ml_models: Optional[List[str]] = None,
                             dl_models: Optional[List[str]] = None,
                             kan_name: str = "KAN",
                             metric: str = "roc_auc") -> Dict[str, Dict[str, Any]]:
    """Pick the best model from each category by a validation metric.

    Categories: Benchmark (AR), Best ML, Best DL, KAN.

    Parameters
    ----------
    results : dict
        ``{model_name: result_dict}`` from the training stage.
    benchmark, kan_name : str
        Names of the benchmark and KAN entries.
    ml_models, dl_models : list of str, optional
        Model names belonging to ML / DL categories.
    metric : str
        Validation metric key inside ``val_metrics`` to compare.

    Returns
    -------
    dict
        ``{category_label: result_dict}`` for the 4 selected models.
    """
    if ml_models is None:
        ml_models = ["Logistic Regression", "Random Forest", "XGBoost"]
    if dl_models is None:
        dl_models = ["1D-CNN", "LSTM"]

    def _best(names: List[str]) -> Tuple[str, Dict[str, Any]]:
        candidates = {n: results[n] for n in names if n in results}
        best_name = max(candidates, key=lambda n: candidates[n]["val_metrics"].get(metric, 0))
        return best_name, candidates[best_name]

    selected: Dict[str, Dict[str, Any]] = {}

    # Benchmark
    if benchmark in results:
        selected[f"Benchmark ({benchmark})"] = results[benchmark]

    # Best ML
    ml_name, ml_result = _best(ml_models)
    selected[f"Best ML ({ml_name})"] = ml_result

    # Best DL
    dl_name, dl_result = _best(dl_models)
    selected[f"Best DL ({dl_name})"] = dl_result

    # KAN
    if kan_name in results:
        selected[f"KAN"] = results[kan_name]

    return selected

# ═══════════════════════════════════════════════════════════════════════════
# Utility — Sequence creation for LSTM / CNN
# ═══════════════════════════════════════════════════════════════════════════
def create_sequences(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                     time_steps: int = 14) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Slide a window across 2D data to create 3D tensors for RNNs/CNNs.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : pd.Series or np.ndarray
        Target vector.
    time_steps : int
        Lookback window size (default 14 — standard 2-week cycle).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray or None]
        - ``X_3d`` of shape ``(samples, time_steps, features)``
        - ``y_seq`` of shape ``(samples,)``
        - ``dates`` array (if X has a DatetimeIndex, else ``None``)
    """
    has_index = hasattr(X, "index")
    X_vals = X.values if hasattr(X, "values") else X
    y_vals = y.values if hasattr(y, "values") else y

    Xs, ys, dates = [], [], []
    for i in range(len(X_vals) - time_steps):
        Xs.append(X_vals[i : i + time_steps])
        ys.append(y_vals[i + time_steps])
        if has_index:
            dates.append(X.index[i + time_steps])

    return (
        np.array(Xs),
        np.array(ys),
        np.array(dates) if dates else None,
    )

# ═══════════════════════════════════════════════════════════════════════════
# Baseline 0 — Autoregressive (AR) Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════
def fine_tune_ar_logistic(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                          random_seed: int = 42, cv_splits: int = 5,
                          param_grid: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
    """Tune an Autoregressive Logistic Regression via ``TimeSeriesSplit``.

    This is the **ultimate baseline**: ``Y = logistic(X_t, X_{t-1}, …)``.
    The input data is expected to already contain lagged columns produced
    by :func:`features.create_lagged_features`.

    Parameters
    ----------
    X_train, y_train : training data (must include lagged columns).
    X_val, y_val     : validation data (must include lagged columns).
    random_seed      : seed for reproducibility (§1).
    cv_splits        : number of ``TimeSeriesSplit`` folds.
    param_grid       : custom hyperparameter grid (optional).

    Returns
    -------
    dict
        ``model``, ``best_params``, ``val_metrics``, ``feature_importances``.
    """
    set_seed(random_seed)

    if param_grid is None:
        param_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.001, 0.01, 0.1, 1, 10],
            "solver": ["liblinear"],
        }

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    lr = LogisticRegression(
        max_iter=2000, random_state=random_seed, class_weight="balanced",
    )

    print("--- Starting AR Logistic Regression Time-Series Grid Search ---")
    gs = GridSearchCV(
        estimator=lr, param_grid=param_grid,
        scoring="roc_auc", cv=tscv, verbose=1, n_jobs=-1,
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    val_metrics = evaluate_model(y_val, y_pred, y_proba)
    print_evaluation(val_metrics, "AR Logistic Regression (Validation)")

    # Feature importance — coefficients
    importances = pd.Series(
        best_model.coef_[0], index=X_train.columns, name="coefficient",
    )

    return {
        "model": best_model,
        "best_params": gs.best_params_,
        "val_metrics": val_metrics,
        "feature_importances": importances,
    }

# ═══════════════════════════════════════════════════════════════════════════
# Baseline 1 — Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════
def fine_tune_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                                  random_seed: int = 42, cv_splits: int = 5,
                                  param_grid: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
    """Tune Logistic Regression via ``TimeSeriesSplit`` grid search.

    Parameters
    ----------
    X_train, y_train : training data (fully preprocessed).
    X_val, y_val     : validation data (fully preprocessed).
    random_seed      : seed for reproducibility (§1).
    cv_splits        : number of ``TimeSeriesSplit`` folds.
    param_grid       : custom hyperparameter grid (optional).

    Returns
    -------
    dict
        ``model``, ``best_params``, ``val_metrics``, ``feature_importances``.
    """
    set_seed(random_seed)

    if param_grid is None:
        param_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear"],
        }

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    lr = LogisticRegression(
        max_iter=2000, random_state=random_seed, class_weight="balanced",
    )

    print("--- Starting Logistic Regression Time-Series Grid Search ---")
    gs = GridSearchCV(
        estimator=lr, param_grid=param_grid,
        scoring="roc_auc", cv=tscv, verbose=1, n_jobs=-1,
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    val_metrics = evaluate_model(y_val, y_pred, y_proba)
    print_evaluation(val_metrics, "Logistic Regression (Validation)")

    # Feature importance — coefficients
    importances = pd.Series(
        best_model.coef_[0], index=X_train.columns, name="coefficient",
    )

    return {
        "model": best_model,
        "best_params": gs.best_params_,
        "val_metrics": val_metrics,
        "feature_importances": importances,
    }

# ═══════════════════════════════════════════════════════════════════════════
# Baseline 2 — XGBoost
# ═══════════════════════════════════════════════════════════════════════════
def fine_tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                      random_seed: int = 42, cv_splits: int = 5,
                      param_grid: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
    """Tune XGBoost via ``TimeSeriesSplit`` grid search.

    Returns
    -------
    dict
        ``model``, ``best_params``, ``val_metrics``, ``feature_importances``.
    """
    set_seed(random_seed)

    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    clf = xgb.XGBClassifier(
        random_state=random_seed, eval_metric="logloss", use_label_encoder=False,
    )

    print("--- Starting XGBoost Time-Series Grid Search ---")
    gs = GridSearchCV(
        estimator=clf, param_grid=param_grid,
        scoring="roc_auc", cv=tscv, verbose=1, n_jobs=-1,
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    val_metrics = evaluate_model(y_val, y_pred, y_proba)
    print_evaluation(val_metrics, "XGBoost (Validation)")

    importances = pd.Series(
        best_model.feature_importances_, index=X_train.columns, name="importance",
    )

    return {
        "model": best_model,
        "best_params": gs.best_params_,
        "val_metrics": val_metrics,
        "feature_importances": importances,
    }

# ═══════════════════════════════════════════════════════════════════════════
# Baseline 3 — Random Forest
# ═══════════════════════════════════════════════════════════════════════════
def fine_tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                            random_seed: int = 42, cv_splits: int = 5,
                            param_grid: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
    """Tune Random Forest via ``TimeSeriesSplit`` grid search.

    Returns
    -------
    dict
        ``model``, ``best_params``, ``val_metrics``, ``feature_importances``.
    """
    set_seed(random_seed)

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15],
            "min_samples_leaf": [1, 5, 10],
            "max_features": ["sqrt", "log2"],
        }

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    clf = RandomForestClassifier(
        random_state=random_seed, class_weight="balanced", n_jobs=-1,
    )

    print("--- Starting Random Forest Time-Series Grid Search ---")
    gs = GridSearchCV(
        estimator=clf, param_grid=param_grid,
        scoring="roc_auc", cv=tscv, verbose=1, n_jobs=-1,
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    val_metrics = evaluate_model(y_val, y_pred, y_proba)
    print_evaluation(val_metrics, "Random Forest (Validation)")

    importances = pd.Series(
        best_model.feature_importances_, index=X_train.columns, name="importance",
    )

    return {
        "model": best_model,
        "best_params": gs.best_params_,
        "val_metrics": val_metrics,
        "feature_importances": importances,
    }

# ═══════════════════════════════════════════════════════════════════════════
# Deep Learning 1 — LSTM
# ═══════════════════════════════════════════════════════════════════════════
def fine_tune_lstm(X_train_3d: np.ndarray, y_train: np.ndarray, X_val_3d: np.ndarray, y_val: np.ndarray,
                   random_seed: int = 42, epochs: int = 100, batch_size: int = 32,
                   patience: int = 15, learning_rate: float = 0.001) -> Dict[str, Any]:
    """Build, train, and evaluate an LSTM model.

    Expects 3D input tensors from :func:`create_sequences`.

    Parameters
    ----------
    X_train_3d, y_train : training data (3D arrays).
    X_val_3d, y_val     : validation data (3D arrays).
    random_seed         : seed for reproducibility (§1).
    epochs              : maximum training epochs.
    batch_size          : batch size.
    patience            : EarlyStopping patience.
    learning_rate       : Adam learning rate.

    Returns
    -------
    dict
        ``model``, ``best_params``, ``val_metrics``, ``history``.
    """
    set_seed(random_seed)

    # Class weights
    weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train,
    )
    class_weights = {0: weights[0], 1: weights[1]}

    # Architecture
    n_timesteps = X_train_3d.shape[1]
    n_features = X_train_3d.shape[2]

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(n_timesteps, n_features)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_auc", mode="max", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=5, verbose=1,
        ),
    ]

    print("--- Training LSTM Model ---")
    history = model.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks, verbose=1,
    )

    # Evaluate
    y_proba = model.predict(X_val_3d).squeeze()
    y_pred = np.where(y_proba > 0.5, 1, 0)
    val_metrics = evaluate_model(y_val, y_pred, y_proba)
    print_evaluation(val_metrics, "LSTM (Validation)")

    return {
        "model": model,
        "best_params": {
            "epochs_trained": len(history.history["loss"]),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
        "val_metrics": val_metrics,
        "history": history.history,
        "feature_importances": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Deep Learning 2 — 1D-CNN
# ═══════════════════════════════════════════════════════════════════════════
def fine_tune_cnn(X_train_3d: np.ndarray, y_train: np.ndarray, X_val_3d: np.ndarray, y_val: np.ndarray,
                  random_seed: int = 42, epochs: int = 100, batch_size: int = 32,
                  patience: int = 15, learning_rate: float = 0.001) -> Dict[str, Any]:
    """Build, train, and evaluate a 1D-CNN model.

    Expects 3D input tensors from :func:`create_sequences`.

    Returns
    -------
    dict
        ``model``, ``best_params``, ``val_metrics``, ``history``.
    """
    set_seed(random_seed)

    weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train,
    )
    class_weights = {0: weights[0], 1: weights[1]}

    n_timesteps = X_train_3d.shape[1]
    n_features = X_train_3d.shape[2]

    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu",
               input_shape=(n_timesteps, n_features)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(32, kernel_size=3, activation="relu"),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_auc", mode="max", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=5, verbose=1,
        ),
    ]

    print("--- Training 1D-CNN Model ---")
    history = model.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks, verbose=1,
    )

    y_proba = model.predict(X_val_3d).squeeze()
    y_pred = np.where(y_proba > 0.5, 1, 0)
    val_metrics = evaluate_model(y_val, y_pred, y_proba)
    print_evaluation(val_metrics, "1D-CNN (Validation)")

    return {
        "model": model,
        "best_params": {
            "epochs_trained": len(history.history["loss"]),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
        "val_metrics": val_metrics,
        "history": history.history,
        "feature_importances": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PyKAN (Kolmogorov-Arnold Network)
# ═══════════════════════════════════════════════════════════════════════════
def fine_tune_kan(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                  random_seed: int = 42, cv_splits: int = 3, cv_steps: int = 50, final_steps: int = 200,
                  param_grid: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
    """Tune and train a PyKAN model via ``TimeSeriesSplit`` cross-validation.

    The best hyperparameters are found through CV, then the final model
    is retrained on the full training set.

    Parameters
    ----------
    X_train, y_train : training data (2D, fully preprocessed).
    X_val, y_val     : validation data (2D, fully preprocessed).
    random_seed      : seed for reproducibility (§1).
    cv_splits        : number of ``TimeSeriesSplit`` folds for CV.
    cv_steps         : training steps per fold during CV search.
    final_steps      : training steps for the final model.
    param_grid       : custom hyperparameter grid (optional).

    Returns
    -------
    dict
        ``model``, ``best_params``, ``val_metrics``, ``feature_importances``.
    """
    set_seed(random_seed)

    if param_grid is None:
        param_grid = {
            "hidden_nodes": [5, 6, 7],
            "grid_size": [2, 3, 4],
            "lr": [0.01, 0.025, 0.05],
        }

    print("--- Starting Robust PyKAN Grid Search ---")
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    n_features = X_train.shape[1]
    best_auc = 0.0
    best_params: Dict[str, Any] = {}

    for params in combinations:
        print(f"Testing params: {params} …")
        fold_aucs: List[float] = []

        for train_idx, test_idx in tscv.split(X_train):
            X_fold_tr = X_train.iloc[train_idx]
            X_fold_te = X_train.iloc[test_idx]
            y_fold_tr = y_train.iloc[train_idx]
            y_fold_te = y_train.iloc[test_idx]

            # Per-fold class weights
            fold_w = compute_class_weight(
                "balanced", classes=np.unique(y_fold_tr), y=y_fold_tr,
            )
            loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(fold_w, dtype=torch.float32),
            )

            fold_data = {
                "train_input": torch.tensor(X_fold_tr.values, dtype=torch.float32),
                "train_label": torch.tensor(y_fold_tr.values, dtype=torch.long),
                "test_input": torch.tensor(X_fold_te.values, dtype=torch.float32),
                "test_label": torch.tensor(y_fold_te.values, dtype=torch.long),
            }

            fold_model = KAN(
                width=[n_features, params["hidden_nodes"], 2],
                grid=params["grid_size"], k=3, seed=random_seed,
            )
            fold_model.fit(
                fold_data, opt="Adam", steps=cv_steps,
                lr=params["lr"], loss_fn=loss_fn,
            )

            with torch.no_grad():
                raw = fold_model(fold_data["test_input"])
                proba = torch.softmax(raw, dim=1)[:, 1].numpy()
                fold_aucs.append(roc_auc_score(y_fold_te, proba))

        mean_auc = np.mean(fold_aucs)
        print(f"  → Mean CV ROC-AUC: {mean_auc:.4f}\n")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    # --- Retrain best model on full training set ---
    print("=" * 50)
    print(f"Best Hyperparameters: {best_params}")
    print("\n--- Retraining Best KAN on Full Training Set ---")

    final_w = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train,
    )
    final_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(final_w, dtype=torch.float32),
    )

    final_data = {
        "train_input": torch.tensor(X_train.values, dtype=torch.float32),
        "train_label": torch.tensor(y_train.values, dtype=torch.long),
        "test_input": torch.tensor(X_val.values, dtype=torch.float32),
        "test_label": torch.tensor(y_val.values, dtype=torch.long),
    }

    best_kan = KAN(
        width=[n_features, best_params["hidden_nodes"], 2],
        grid=best_params["grid_size"], k=3, seed=random_seed,
    )
    best_kan.fit(
        final_data, opt="Adam", steps=final_steps,
        lr=best_params["lr"], loss_fn=final_loss,
    )

    # Evaluate on validation
    with torch.no_grad():
        val_raw = best_kan(final_data["test_input"])
        y_proba = torch.softmax(val_raw, dim=1)[:, 1].numpy()
        y_pred = torch.argmax(val_raw, dim=1).numpy()

    val_metrics = evaluate_model(y_val.values, y_pred, y_proba)
    print_evaluation(val_metrics, "PyKAN (Validation)")

    return {
        "model": best_kan,
        "best_params": best_params,
        "val_metrics": val_metrics,
        "feature_importances": None,
    }
