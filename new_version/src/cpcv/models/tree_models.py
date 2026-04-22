"""
8.3) Tree-Based
====================
Random Forest and XGBoost classifiers wrapped in the BaseModel interface.
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.cpcv.models.base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS = 500
RF_MAX_FEATURES = "sqrt"
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 5

XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.7
XGB_COLSAMPLE_BYTREE = 0.7
XGB_MIN_CHILD_WEIGHT = 5
XGB_GAMMA = 1.0
XGB_REG_ALPHA = 0.1
XGB_REG_LAMBDA = 2.0
XGB_EARLY_STOPPING_ROUNDS = 20


# =====================================================================
# Random Forest
# =====================================================================
class RandomForestModel(BaseModel):
    """Sklearn Random Forest with balanced subsampling."""

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.model = None

    def fit(self, X_train, y_train, sample_weight=None, X_val=None, y_val=None) -> None:
        X = X_train.values if hasattr(X_train, "values") else X_train
        y = y_train.values if hasattr(y_train, "values") else y_train
        w = sample_weight.values if hasattr(sample_weight, "values") else sample_weight

        self.model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_features=RF_MAX_FEATURES,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.seed,
        )
        self.model.fit(X, y, sample_weight=w)

        logger.info(
            "RandomForest fitted: %d samples, %d features, %d trees.",
            X.shape[0], X.shape[1], RF_N_ESTIMATORS,
        )

    def predict_proba(self, X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else X
        return self.model.predict_proba(X_arr)

    def predict(self, X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else X
        return self.model.predict(X_arr)

    def predict_logits(self, X) -> np.ndarray:
        """Convert probabilities to log-odds for calibration."""
        proba = self.predict_proba(X)
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        logits = np.log(proba[:, 1] / proba[:, 0])
        return logits.reshape(-1, 1)

    def get_name(self) -> str:
        return "Random_Forest"


# =====================================================================
# XGBoost
# =====================================================================
class XGBoostModel(BaseModel):
    """XGBoost gradient-boosted trees with optional early stopping."""

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.model = None

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_val=None,
        y_val=None,) -> None:
        X = X_train.values if hasattr(X_train, "values") else X_train
        y = y_train.values if hasattr(y_train, "values") else y_train
        w = sample_weight.values if hasattr(sample_weight, "values") else sample_weight

        init_params = {
            "n_estimators": XGB_N_ESTIMATORS,
            "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            "min_child_weight": XGB_MIN_CHILD_WEIGHT,
            "gamma": XGB_GAMMA,
            "reg_alpha": XGB_REG_ALPHA,
            "reg_lambda": XGB_REG_LAMBDA,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": self.seed,
        }

        if X_val is not None and y_val is not None:
            init_params["early_stopping_rounds"] = XGB_EARLY_STOPPING_ROUNDS

        self.model = XGBClassifier(**init_params)

        fit_params = {"X": X, "y": y, "sample_weight": w, "verbose": False}

        if X_val is not None and y_val is not None:
            X_v = X_val.values if hasattr(X_val, "values") else X_val
            y_v = y_val.values if hasattr(y_val, "values") else y_val
            fit_params["eval_set"] = [(X_v, y_v)]

            self.model.fit(**fit_params)

            best_iter = self.model.best_iteration
            logger.info(
                "XGBoost fitted: %d samples, early stopped at iteration %d/%d.",
                X.shape[0], best_iter, XGB_N_ESTIMATORS,
            )
        else:
            self.model.fit(**fit_params)
            logger.info(
                "XGBoost fitted: %d samples, %d rounds (no early stopping).",
                X.shape[0], XGB_N_ESTIMATORS,
            )

    def predict_proba(self, X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else X
        return self.model.predict_proba(X_arr)

    def predict(self, X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else X
        return self.model.predict(X_arr)

    def predict_logits(self, X) -> np.ndarray:
        """Convert probabilities to log-odds for calibration."""
        proba = self.predict_proba(X)
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        logits = np.log(proba[:, 1] / proba[:, 0])
        return logits.reshape(-1, 1)

    def get_name(self) -> str:
        return "XGBoost"
