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
  - Vector scaling (Guo et al. 2017): for PyTorch models (LSTM, KAN). Fits
    a temperature T plus a per-class bias vector b that minimise the NLL of
    softmax((logits + b) / T). Adding the bias term lets the calibrator
    correct directional miscalibration that standard temperature scaling
    cannot. Pure temperature scaling preserves the argmax of the raw
    logits; on this dataset that property left a systematic short-bias
    in PyTorch model outputs that bet-sizing then converted into negative
    drift. Vector scaling is the natural extension recommended in the
    same Guo et al. (2017) paper for cases where temperature alone is
    insufficient.

"""

import logging
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
CALIBRATION_METHOD_SKLEARN = "platt"
CALIBRATION_METHOD_PYTORCH = "vector"     # default for LSTM / KAN
CALIBRATION_METHOD_TEMPERATURE = "temperature"  # available for opt-in / tests

# Models that use vector scaling (PyTorch-based, 2D logits)
_PYTORCH_MODELS = {"LSTM", "KAN"}

# Vector-scaling optimiser bounds
_VECTOR_T_BOUNDS = (0.05, 20.0)
_VECTOR_BIAS_BOUNDS = (-5.0, 5.0)


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

    logger.debug(
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

    logger.debug(
        "Temperature scaling fitted: T=%.4f (NLL=%.4f).",
        optimal_T, result.fun,
    )
    return optimal_T


# =====================================================================
# Vector Scaling (Guo et al. 2017, Section 4.2)
# =====================================================================
def fit_vector_scaling(
    logits: np.ndarray, y_true: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Learn a temperature T and per-class bias b minimising the NLL of
    ``softmax((logits + b) / T)``.

    Vector scaling extends temperature scaling with a per-class additive
    bias, allowing it to correct directional miscalibration as well as
    sharpness. The parameterisation has one redundant degree of freedom
    in the bias (softmax is invariant to a constant shift across all
    classes); the optimiser handles this without issue and the resulting
    calibrated probabilities are unambiguous.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(n_samples, n_classes)``, raw pre-softmax logits.
    y_true : np.ndarray
        True labels (0-indexed integers).

    Returns
    -------
    T : float
        Optimal temperature, ``T > 0``.
    b : np.ndarray
        Per-class bias vector, shape ``(n_classes,)``.

    Notes
    -----
    Calibrate new logits via
    ``softmax((new_logits + b) / T, axis=1)``.
    """
    n_classes = logits.shape[1]

    def neg_log_likelihood(params: np.ndarray) -> float:
        T = params[0]
        b = params[1:]
        if T <= 0:
            return 1e10
        scaled = (logits + b) / T
        log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
        correct_log_probs = log_probs[np.arange(len(y_true)), y_true]
        return -float(np.mean(correct_log_probs))

    x0 = np.zeros(1 + n_classes)
    x0[0] = 1.0
    bounds = [_VECTOR_T_BOUNDS] + [_VECTOR_BIAS_BOUNDS] * n_classes

    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    optimal_T = float(result.x[0])
    optimal_b = result.x[1:].astype(float)

    logger.debug(
        "Vector scaling fitted: T=%.4f, b=%s (NLL=%.4f).",
        optimal_T,
        np.array2string(optimal_b, precision=4),
        result.fun,
    )
    return optimal_T, optimal_b


# =====================================================================
# Unified Calibrator
# =====================================================================
class Calibrator:
    """Unified calibration wrapper for both sklearn and PyTorch models.

    PyTorch models (LSTM, KAN) are calibrated with vector scaling by
    default. sklearn-compatible models use Platt scaling. Temperature
    scaling is supported as an opt-in method for backward compatibility
    and unit tests but is not the default for any model.

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
        self.temperature = None     # used by both vector and temperature
        self.bias = None            # used only by vector scaling
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

        if model_name in _PYTORCH_MODELS:
            self.method = CALIBRATION_METHOD_PYTORCH
            logits = model.predict_logits(X_cal)     # (n, n_classes)
            self.temperature, self.bias = fit_vector_scaling(logits, y)
        else:
            self.method = CALIBRATION_METHOD_SKLEARN
            logits = model.predict_logits(X_cal)
            # convert 2D logits (n, n_classes) to 1D log-odds for Platt
            if logits.ndim == 2 and logits.shape[1] >= 2:
                logits = logits[:, 1] - logits[:, 0]
            logits = logits.ravel()
            self.platt_model = fit_platt_scaling(logits, y)

        self.fitted = True
        logger.debug("Calibrator fitted: method=%s, model=%s.", self.method, model_name)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply fitted calibration to raw logits.

        Parameters
        ----------
        logits : np.ndarray
            For Platt: shape (n,) or (n, 1), raw log-odds.
            For vector / temperature: shape (n, n_classes), raw pre-softmax
            logits.

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

        if self.method == CALIBRATION_METHOD_PYTORCH:
            scaled = (logits + self.bias) / self.temperature
            return softmax(scaled, axis=1)

        if self.method == CALIBRATION_METHOD_TEMPERATURE:
            scaled = logits / self.temperature
            return softmax(scaled, axis=1)

        raise ValueError(f"Unknown calibration method: {self.method}")

    def fit_from_logits(self, logits, y_cal, method=CALIBRATION_METHOD_PYTORCH):
        """Fit calibration directly from pre-computed logits.

        Used by the LSTM training loop where index alignment requires
        pre-computed logits rather than re-running ``predict_logits``
        through the model's interface.

        Parameters
        ----------
        method : str
            One of ``"vector"`` (default), ``"temperature"``, or ``"platt"``.
        """
        y = y_cal.values if hasattr(y_cal, "values") else np.asarray(y_cal)

        if method == CALIBRATION_METHOD_PYTORCH:
            self.method = CALIBRATION_METHOD_PYTORCH
            self.temperature, self.bias = fit_vector_scaling(logits, y)
        elif method == CALIBRATION_METHOD_TEMPERATURE:
            self.method = CALIBRATION_METHOD_TEMPERATURE
            self.temperature = fit_temperature_scaling(logits, y)
        elif method == CALIBRATION_METHOD_SKLEARN:
            self.method = CALIBRATION_METHOD_SKLEARN
            self.platt_model = fit_platt_scaling(logits.ravel(), y)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self.fitted = True
        logger.debug("Calibrator fitted from pre-computed logits: method=%s.", self.method)

    def __repr__(self) -> str:
        if not self.fitted:
            return "Calibrator(fitted=False)"
        if self.method == CALIBRATION_METHOD_SKLEARN:
            return (
                f"Calibrator(method=platt, "
                f"coef={self.platt_model.coef_[0, 0]:.4f}, "
                f"intercept={self.platt_model.intercept_[0]:.4f})"
            )
        if self.method == CALIBRATION_METHOD_PYTORCH:
            b_str = np.array2string(self.bias, precision=4)
            return f"Calibrator(method=vector, T={self.temperature:.4f}, b={b_str})"
        return f"Calibrator(method=temperature, T={self.temperature:.4f})"