"""
8.2) Benchmarks
====================
Two baseline models for the CPCV pipeline:
  - ARLogistic: econometric baseline using autoregressive lagged returns
  - LogisticRegressionModel: linear baseline on the full selected feature set

These establish the minimum bar that more complex models (trees, LSTM, KAN)
must beat to justify their additional complexity.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.cpcv.models.base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
AR_LAGS = [1, 2, 3, 5, 10, 21]
LOGISTIC_C = 1.0
LOGISTIC_PENALTY = "l2"
LOGISTIC_MAX_ITER = 1000


# =====================================================================
# AR Logistic — Econometric Baseline
# =====================================================================
class ARLogistic(BaseModel):
    """Autoregressive logistic regression on lagged log returns.

    Deliberately constructs its own features (ignoring the selected
    feature set) to isolate the purely autoregressive signal. This tests
    whether the additional features from Steps 6–7 add value beyond
    simple price momentum.

    Pipeline note: must receive X with a 'log_ret' column (pre-feature-
    selection), not the feature-selected subset.
    """

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.model = None
        self.ar_lags = AR_LAGS
        self._lag_columns = [f"lag_{lag}" for lag in self.ar_lags]

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_val=None,
        y_val=None,
    ) -> None:
        X_lagged, y_aligned, w_aligned = self._build_lag_features(
            X_train, y_train, sample_weight
        )

        self.model = LogisticRegression(
            C=LOGISTIC_C,
            penalty=LOGISTIC_PENALTY,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=LOGISTIC_MAX_ITER,
            random_state=self.seed,
        )
        self.model.fit(X_lagged, y_aligned, sample_weight=w_aligned)

        logger.info(
            "ARLogistic fitted on %d samples with lags %s.",
            len(y_aligned), self.ar_lags,
        )

    def predict_proba(self, X) -> np.ndarray:
        X_lagged = self._build_lag_features_predict(X)
        return self.model.predict_proba(X_lagged)

    def predict(self, X) -> np.ndarray:
        X_lagged = self._build_lag_features_predict(X)
        return self.model.predict(X_lagged)

    def predict_logits(self, X) -> np.ndarray:
        """Return log-odds (logits) for calibration."""
        proba = self.predict_proba(X)
        logits = np.log(proba[:, 1] / (proba[:, 0] + 1e-10))
        return logits.reshape(-1, 1)

    def get_name(self) -> str:
        return "AR_Logistic"

    # ── internal helpers ──────────────────────────────────────────────
    def _extract_log_ret(self, X) -> pd.Series:
        """Extract log returns from X, handling both DataFrame and ndarray."""
        if isinstance(X, pd.DataFrame):
            if "log_returns" in X.columns:
                return X["log_returns"]
            if "log_ret" in X.columns:
                return X["log_ret"]
            raise ValueError(
                "ARLogistic requires a 'log_returns' column in X. "
                "Pass the pre-feature-selection DataFrame."
            )
        raise ValueError(
            "ARLogistic requires a DataFrame with a 'log_returns' column, "
            f"got {type(X).__name__}."
        )

    def _build_lag_features(self, X, y, w=None):
        """Build lagged features and drop NaN rows, aligning y and w."""
        log_ret = self._extract_log_ret(X)

        lagged = pd.DataFrame(index=log_ret.index)
        for lag in self.ar_lags:
            lagged[f"lag_{lag}"] = log_ret.shift(lag)

        # drop NaN rows from longest lag
        valid_mask = lagged.notna().all(axis=1)
        lagged = lagged.loc[valid_mask]

        # align y and w
        common = lagged.index.intersection(y.index if hasattr(y, "index") else lagged.index)
        lagged = lagged.loc[common]

        if hasattr(y, "loc"):
            y_aligned = y.loc[common].values
        else:
            # positional fallback
            y_aligned = y[valid_mask.values]

        w_aligned = None
        if w is not None:
            if hasattr(w, "loc"):
                w_aligned = w.loc[common].values
            else:
                w_aligned = w[valid_mask.values]

        n_dropped = len(valid_mask) - valid_mask.sum()
        if n_dropped > 0:
            logger.info("ARLogistic: dropped %d NaN rows from lag construction.", n_dropped)

        return lagged.values, y_aligned, w_aligned

    def _build_lag_features_predict(self, X) -> np.ndarray:
        """Build lagged features for prediction (no NaN dropping)."""
        log_ret = self._extract_log_ret(X)

        lagged = pd.DataFrame(index=log_ret.index)
        for lag in self.ar_lags:
            lagged[f"lag_{lag}"] = log_ret.shift(lag)

        # for prediction, forward-fill any remaining NaN at the edges
        lagged = lagged.ffill().bfill()

        return lagged.values


# =====================================================================
# Logistic Regression — Linear ML Baseline
# =====================================================================
class LogisticRegressionModel(BaseModel):
    """Standard logistic regression on the full selected feature set.

    Simple linear baseline that all more complex models must outperform.
    """

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.model = None

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_val=None,
        y_val=None,
    ) -> None:
        # convert to numpy if DataFrame
        X = X_train.values if hasattr(X_train, "values") else X_train
        y = y_train.values if hasattr(y_train, "values") else y_train
        w = sample_weight.values if hasattr(sample_weight, "values") else sample_weight

        self.model = LogisticRegression(
            C=LOGISTIC_C,
            penalty=LOGISTIC_PENALTY,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=LOGISTIC_MAX_ITER,
            random_state=self.seed,
        )
        self.model.fit(X, y, sample_weight=w)

        logger.info(
            "LogisticRegression fitted on %d samples, %d features.",
            X.shape[0], X.shape[1],
        )

    def predict_proba(self, X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else X
        return self.model.predict_proba(X_arr)

    def predict(self, X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else X
        return self.model.predict(X_arr)

    def predict_logits(self, X) -> np.ndarray:
        """Return raw log-odds from sklearn's decision_function."""
        X_arr = X.values if hasattr(X, "values") else X
        logits = self.model.decision_function(X_arr)
        return logits.reshape(-1, 1)

    def get_name(self) -> str:
        return "Logistic_Regression"