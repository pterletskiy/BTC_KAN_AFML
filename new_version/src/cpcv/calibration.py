"""
10) Calibration
===================
Calibrate raw model probabilities so that predicted confidence levels
correspond to empirical accuracy. Essential because downstream bet sizing
converts probabilities directly into position sizes via De Prado's
S-curve, so poorly calibrated probabilities produce systematically wrong
bet sizes.

Two methods:
  - Platt scaling (Platt 1999): for sklearn models (logistic, RF, XGBoost,
    AR). Fits a sigmoid mapping from raw logits to calibrated probabilities.
  - Temperature scaling (Guo et al. 2017): for PyTorch models (LSTM, KAN).
    Fits a single scalar T > 0 that minimizes NLL of softmax(logits / T).
"""

import logging
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
CALIBRATION_METHOD_SKLEARN = "platt"
CALIBRATION_METHOD_PYTORCH = "temperature"

# Models that use temperature scaling (PyTorch-based, 2D logits)
_TEMPERATURE_MODELS = {"LSTM", "KAN"}


# =====================================================================
# Platt Scaling
# =====================================================================
def fit_platt_scaling(
    logits: np.ndarray, y_true: np.ndarray
) -> LogisticRegression:
    """Fit a sigmoid mapping from raw logits to calibrated probabilities.

    Parameters
    ----------
    logits : np.ndarray
        1D array of log-odds or decision function values.
    y_true : np.ndarray
        True binary labels (0 or 1).

    Returns
    -------
    LogisticRegression
        Fitted model. Calibrate via
        ``platt_model.predict_proba(new_logits.reshape(-1, 1))``.
    """
    logits_2d = logits.reshape(-1, 1)

    platt = LogisticRegression(
        C=1e10,        # no regularization, purely data-driven sigmoid
        solver="lbfgs",
        max_iter=1000,
    )
    platt.fit(logits_2d, y_true)

    logger.info(
        "Platt scaling fitted: coef=%.4f, intercept=%.4f.",
        platt.coef_[0, 0], platt.intercept_[0],
    )
    return platt


# =====================================================================
# Temperature Scaling (Guo et al. 2017)
# =====================================================================
def fit_temperature_scaling(
    logits: np.ndarray, y_true: np.ndarray,
) -> float:
    """Learn a single scalar T that minimizes NLL of softmax(logits / T).

    Standard temperature scaling as introduced in Guo et al. (2017),
    "On Calibration of Modern Neural Networks". Treats all samples
    equally; no per-sample weighting.

    Parameters
    ----------
    logits : np.ndarray
        Shape (n_samples, n_classes), raw pre-softmax logits.
    y_true : np.ndarray
        True labels (0-indexed integers).

    Returns
    -------
    float
        Optimal temperature T > 0. Calibrate via
        ``softmax(new_logits / T, axis=1)``.
    """
    def nll(T):
        scaled = logits / T
        log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
        correct_log_probs = log_probs[np.arange(len(y_true)), y_true]
        return -np.mean(correct_log_probs)

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    optimal_T = result.x

    logger.info(
        "Temperature scaling fitted: T=%.4f (NLL=%.4f).",
        optimal_T, result.fun,
    )
    return optimal_T


# =====================================================================
# Unified Calibrator
# =====================================================================
class Calibrator:
    """Unified calibration wrapper for both sklearn and PyTorch models.

    Usage
    -----
    >>> cal = Calibrator()
    >>> cal.fit(model, X_cal, y_cal)
    >>> raw_logits = model.predict_logits(X_new)
    >>> calibrated_proba = cal.calibrate(raw_logits)
    """

    def __init__(self):
        self.method = None
        self.platt_model = None
        self.temperature = None
        self.fitted = False

    def fit(self, model, X_cal, y_cal) -> None:
        """Fit calibration on a held-out calibration set.

        Parameters
        ----------
        model : BaseModel
            Trained model with ``predict_logits`` and ``get_name`` methods.
        X_cal : np.ndarray or pd.DataFrame
            Calibration features.
        y_cal : np.ndarray or pd.Series
            True labels (0-indexed).
        """
        y = y_cal.values if hasattr(y_cal, "values") else np.asarray(y_cal)
        model_name = model.get_name()

        if model_name in _TEMPERATURE_MODELS:
            self.method = CALIBRATION_METHOD_PYTORCH
            logits = model.predict_logits(X_cal)     # (n, n_classes)
            self.temperature = fit_temperature_scaling(logits, y)
        else:
            self.method = CALIBRATION_METHOD_SKLEARN
            logits = model.predict_logits(X_cal)
            # convert 2D logits (n, n_classes) to 1D log-odds for Platt
            if logits.ndim == 2 and logits.shape[1] >= 2:
                logits = logits[:, 1] - logits[:, 0]
            logits = logits.ravel()
            self.platt_model = fit_platt_scaling(logits, y)

        self.fitted = True
        logger.info("Calibrator fitted: method=%s, model=%s.", self.method, model_name)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply fitted calibration to raw logits.

        Parameters
        ----------
        logits : np.ndarray
            For Platt: shape (n,) or (n, 1), raw log-odds.
            For temperature: shape (n, n_classes), raw pre-softmax logits.

        Returns
        -------
        np.ndarray
            Calibrated probabilities, shape (n_samples, n_classes),
            rows sum to 1.
        """
        if not self.fitted:
            raise RuntimeError(
                "Calibrator has not been fitted. Call .fit() first."
            )

        if self.method == CALIBRATION_METHOD_SKLEARN:
            # convert 2D logits (n, n_classes) to 1D log-odds for Platt
            if logits.ndim == 2 and logits.shape[1] >= 2:
                logits = logits[:, 1] - logits[:, 0]
            logits_2d = logits.reshape(-1, 1)
            return self.platt_model.predict_proba(logits_2d)

        elif self.method == CALIBRATION_METHOD_PYTORCH:
            scaled = logits / self.temperature
            return softmax(scaled, axis=1)

        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def fit_from_logits(self, logits, y_cal, method="temperature"):
        """Fit calibration directly from pre-computed logits (for LSTM alignment)."""
        y = y_cal.values if hasattr(y_cal, "values") else np.asarray(y_cal)

        if method == "temperature":
            self.method = CALIBRATION_METHOD_PYTORCH
            self.temperature = fit_temperature_scaling(logits, y)
        else:
            self.method = CALIBRATION_METHOD_SKLEARN
            self.platt_model = fit_platt_scaling(logits.ravel(), y)

        self.fitted = True
        logger.info("Calibrator fitted from pre-computed logits: method=%s.", self.method)

    def __repr__(self) -> str:
        if not self.fitted:
            return "Calibrator(fitted=False)"
        if self.method == CALIBRATION_METHOD_SKLEARN:
            return (
                f"Calibrator(method=platt, "
                f"coef={self.platt_model.coef_[0, 0]:.4f}, "
                f"intercept={self.platt_model.intercept_[0]:.4f})"
            )
        return f"Calibrator(method=temperature, T={self.temperature:.4f})"