"""
8.2) Benchmarks
====================
Two baseline models for the CPCV pipeline:
  - ARLogistic: econometric baseline using autoregressive lagged returns
  - LogisticRegressionModel: linear baseline on the full selected feature set

These establish the minimum bar that more complex models (trees, LSTM, KAN)
must beat to justify their additional complexity.

AR Logistic consumes precomputed lag columns from
``pre_cpcv.features.compute_lag_features``. Lag features sit alongside
TA / math / external features in X and are eligible for MDA selection
for the other models; AR Logistic itself selects its six lag columns
by name from the pre-MDA feature matrix via the pipeline's
``X_tr_full`` route, independently of MDA's choices.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.cpcv.models.base import BaseModel
from src.pre_cpcv.features import AR_LAGS, lag_column_names

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# AR_LAGS lives in src.pre_cpcv.features so the precompute step and the
# model that consumes the columns share a single source of truth.
LOGISTIC_C = 1.0
LOGISTIC_PENALTY = "l2"
LOGISTIC_MAX_ITER = 1000


# =====================================================================
# AR Logistic — Econometric Baseline
# =====================================================================
class ARLogistic(BaseModel):
    """Autoregressive logistic regression on lagged log returns.

    Deliberately ignores the engineered feature set and uses only
    precomputed log-return lags, isolating the purely autoregressive
    signal as a baseline. A more complex model that cannot beat AR
    Logistic has not learned anything beyond price autocorrelation.

    Pipeline contract
    -----------------
    X must contain the lag columns produced by
    ``pre_cpcv.features.compute_lag_features``
    (``log_returns_lag1``, ..., ``log_returns_lag30``). The CPCV
    pipeline routes ``X_tr_full`` (all pre-selection columns) to AR
    Logistic so the lag columns are present even when MDA selection
    drops them from ``selected``. Other models receive the MDA-selected
    subset, which may or may not include lag columns depending on
    permutation importance for that fold.
    """

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.model = None
        self.ar_lags = AR_LAGS
        self._lag_columns = lag_column_names(self.ar_lags)

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_val=None,
        y_val=None,
    ) -> None:
        X_lagged, y_aligned, w_aligned = self._select_lag_features(
            X_train, y_train, sample_weight, drop_nan=True,
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
        X_lagged, _, _ = self._select_lag_features(
            X, None, None, drop_nan=False,
        )
        return self.model.predict_proba(X_lagged)

    def predict(self, X) -> np.ndarray:
        X_lagged, _, _ = self._select_lag_features(
            X, None, None, drop_nan=False,
        )
        return self.model.predict(X_lagged)

    def predict_logits(self, X) -> np.ndarray:
        """Return log-odds (logits) for calibration.

        Uses a symmetric clip matching the tree models (XGBoost, RF) so
        that all CPCV-pipeline classifiers compute logits via the same
        numerical-stability convention.
        """
        proba = self.predict_proba(X)
        proba = np.clip(proba, 1e-10, 1.0 - 1e-10)
        logits = np.log(proba[:, 1] / proba[:, 0])
        return logits.reshape(-1, 1)

    def get_name(self) -> str:
        return "AR_Logistic"

    # ── internal helpers ──────────────────────────────────────────────
    def _select_lag_features(self, X, y, w, drop_nan: bool):
        """Select the precomputed lag columns from X.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix containing the precomputed lag columns.
        y, w : pd.Series or array-like or None
            Labels and sample weights (training-time only).
        drop_nan : bool
            If True (training), drop rows with any NaN in the lag
            columns and align ``y``, ``w`` to the surviving rows.
            If False (prediction), require the lag columns to be
            already populated and raise on any NaN; reordering or
            dropping rows at predict time would misalign the output
            predictions with the test fold's timestamps.

        Returns
        -------
        np.ndarray
            Lag feature matrix.
        np.ndarray or None
            Aligned labels (None if ``y`` was None).
        np.ndarray or None
            Aligned sample weights (None if ``w`` was None).
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "ARLogistic requires a DataFrame containing the precomputed "
                f"lag columns ({self._lag_columns}); got {type(X).__name__}."
            )

        missing = [c for c in self._lag_columns if c not in X.columns]
        if missing:
            raise ValueError(
                f"ARLogistic: missing lag column(s) {missing}. "
                "Did you call pre_cpcv.features.compute_lag_features() "
                "and concatenate the result into the feature matrix?"
            )

        lagged = X[self._lag_columns]

        if drop_nan:
            valid_mask = lagged.notna().all(axis=1)
            n_dropped = int((~valid_mask).sum())
            if n_dropped > 0:
                logger.info(
                    "ARLogistic: dropped %d row(s) with NaN lag features "
                    "from training fold.",
                    n_dropped,
                )
            lagged = lagged.loc[valid_mask]

            y_aligned = None
            w_aligned = None
            if y is not None:
                y_aligned = (
                    y.loc[lagged.index].values
                    if hasattr(y, "loc")
                    else np.asarray(y)[valid_mask.values]
                )
            if w is not None:
                w_aligned = (
                    w.loc[lagged.index].values
                    if hasattr(w, "loc")
                    else np.asarray(w)[valid_mask.values]
                )
            return lagged.values, y_aligned, w_aligned

        # prediction path: lag columns must already be valid
        if lagged.isna().any().any():
            n_nan_rows = int(lagged.isna().any(axis=1).sum())
            raise ValueError(
                f"ARLogistic.predict: {n_nan_rows} row(s) have NaN lag "
                "features. Lag features should be populated for every "
                "aligned event because they are precomputed on the full "
                "daily series. Recompute via "
                "pre_cpcv.features.compute_lag_features(df_raw)."
            )
        return lagged.values, None, None


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
        X = X_train.values if hasattr(X_train, "values") else X_train
        y = y_train.values if hasattr(y_train, "values") else y_train
        w = sample_weight.values if hasattr(sample_weight, "values") else sample_weight

        solver = "liblinear" if LOGISTIC_PENALTY == "l1" else "lbfgs"
        self.model = LogisticRegression(
            C=LOGISTIC_C,
            penalty=LOGISTIC_PENALTY,
            class_weight="balanced",
            solver=solver,
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