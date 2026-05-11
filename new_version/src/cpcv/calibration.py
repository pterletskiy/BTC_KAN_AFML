"""
12) Calibration
===================
Calibrate raw model probabilities so predicted confidence corresponds to
empirical accuracy. Essential because downstream bet sizing converts
probabilities into position sizes via López de Prado's S-curve; poorly
calibrated probabilities translate directly into systematically wrong
bet sizes.

Two methods, routed automatically by model family:

  - Platt scaling (Platt 1999): sklearn models (Logistic, RF, XGBoost, AR).
    Fits a sigmoid mapping from raw logits to calibrated probabilities.
  - Vector scaling (Guo et al. 2017 §4.2): PyTorch models (LSTM, KAN).
    Fits a temperature T plus a per-class bias vector b minimising the
    NLL of ``softmax((logits + b) / T)``. The bias term lets the calibrator
    correct directional miscalibration that pure temperature scaling
    cannot, because pure temperature scaling preserves the argmax of the
    raw logits. On this dataset that property left a systematic short-bias
    in PyTorch model outputs which bet sizing then converted into negative
    drift; vector scaling fixes it.
"""

import logging
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# --- Module-level constants -------------------------------------------------
CALIBRATION_METHOD_SKLEARN = "platt"
CALIBRATION_METHOD_PYTORCH = "vector"     # default for LSTM / KAN
CALIBRATION_METHOD_TEMPERATURE = "temperature"  # available for opt-in / tests

# Models routed to vector scaling rather than Platt.
_PYTORCH_MODELS = {"LSTM", "KAN"}

# Vector-scaling optimiser bounds; intentionally generous on temperature so the
# bounded LBFGS does not pin the optimum at an edge.
_VECTOR_T_BOUNDS = (0.05, 20.0)
_VECTOR_BIAS_BOUNDS = (-5.0, 5.0)


# --- 1. Platt Scaling -------------------------------------------------------
# Sigmoid fit from raw log-odds to calibrated probability; the standard 1D method.
def fit_platt_scaling(
    logits: np.ndarray, y_true: np.ndarray
) -> LogisticRegression:
    """Fit a sigmoid ``σ(a·logit + b)`` mapping raw log-odds to calibrated probabilities.

    Returns the fitted LogisticRegression; calibrate via
    ``platt_model.predict_proba(new_logits.reshape(-1, 1))``.

    Note on sample weights: the training side uses AFML sample weights end-to-end,
    but Platt is fit unweighted by design. Calibration estimates the conditional
    ``P(class | logit)`` of the underlying data-generating process, which is a
    property of the data distribution rather than of the AFML training signal.
    Fitting Platt with AFML weights would tilt the sigmoid toward high-weight
    samples instead of toward the empirical class frequency at each logit, which
    is the opposite of what calibration should measure.
    """
    logits_2d = logits.reshape(-1, 1)

    # C=1e10 disables regularisation so the fit is purely data-driven.
    platt = LogisticRegression(
        C=1e10,
        solver="lbfgs",
        max_iter=1000,
    )
    platt.fit(logits_2d, y_true)

    logger.debug(
        "Platt scaling fitted: coef=%.4f, intercept=%.4f.",
        platt.coef_[0, 0], platt.intercept_[0],
    )
    return platt


# --- 2. Temperature Scaling (Guo et al. 2017) ------------------------------
# Single-scalar variant kept for opt-in / unit-test use; not the default for any model.
def fit_temperature_scaling(
    logits: np.ndarray, y_true: np.ndarray,
) -> float:
    """Learn the scalar T that minimises the NLL of ``softmax(logits / T)``.

    Returns the optimal T > 0; calibrate via ``softmax(new_logits / T, axis=1)``.
    """
    # NLL of softmax(logits / T) expressed in log-sum-exp form for numerical stability.
    def nll(T):
        scaled = logits / T
        log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
        correct_log_probs = log_probs[np.arange(len(y_true)), y_true]
        return -np.mean(correct_log_probs)

    result = minimize_scalar(nll, bounds=_VECTOR_T_BOUNDS, method="bounded")
    optimal_T = result.x

    logger.debug(
        "Temperature scaling fitted: T=%.4f (NLL=%.4f).",
        optimal_T, result.fun,
    )
    return optimal_T


# --- 3. Vector Scaling (Guo et al. 2017 §4.2) ------------------------------
# Temperature + per-class bias; extends temperature scaling to correct directional miscalibration.
def fit_vector_scaling(
    logits: np.ndarray, y_true: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Learn ``T`` and bias vector ``b`` minimising the NLL of ``softmax((logits + b) / T)``.

    Softmax is invariant under a constant shift across all biases, so the raw
    parameterisation ``[T, b_0, b_1, ..., b_{C-1}]`` has one redundant degree of
    freedom: ``(T, b)`` and ``(T, b + c·1)`` produce identical calibrated
    probabilities. To make ``(T, b)`` uniquely identifiable, ``b_0`` is pinned to
    zero and only ``T`` plus ``b_1, ..., b_{C-1}`` are optimised. The returned
    ``b`` is the full vector with ``b[0] = 0``. Calibrate new logits via
    ``softmax((new_logits + b) / T, axis=1)``.
    """
    n_classes = logits.shape[1]

    # NLL with reduced parameter vector ``[T, b_1, ..., b_{C-1}]``; b_0 is implicitly zero.
    def neg_log_likelihood(params: np.ndarray) -> float:
        T = params[0]
        # Reconstruct the full bias vector with b_0 = 0 pinned in.
        b = np.concatenate([[0.0], params[1:]])
        scaled = (logits + b) / T
        log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
        correct_log_probs = log_probs[np.arange(len(y_true)), y_true]
        return -float(np.mean(correct_log_probs))

    # Initialise at T=1, b_1..b_{C-1}=0 (the identity calibrator with b_0 pinned at 0).
    x0 = np.zeros(n_classes)
    x0[0] = 1.0
    bounds = [_VECTOR_T_BOUNDS] + [_VECTOR_BIAS_BOUNDS] * (n_classes - 1)

    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    optimal_T = float(result.x[0])
    # Return the full bias vector including the pinned b_0 = 0.
    optimal_b = np.concatenate([[0.0], result.x[1:]]).astype(float)

    logger.debug(
        "Vector scaling fitted: T=%.4f, b=%s (NLL=%.4f).",
        optimal_T,
        np.array2string(optimal_b, precision=4),
        result.fun,
    )
    return optimal_T, optimal_b


# --- 4. Unified Calibrator --------------------------------------------------
# Routes sklearn models to Platt and PyTorch models to vector scaling; same interface either way.
class Calibrator:
    """Auto-routing calibration wrapper for both sklearn and PyTorch models.

    PyTorch models (LSTM, KAN) get vector scaling; sklearn-compatible models get
    Platt. Temperature scaling is supported as an opt-in via ``fit_from_logits``
    but is not the default for any model.

    Usage::

        cal = Calibrator()
        cal.fit(model, X_cal, y_cal)
        cal_proba = cal.calibrate(model.predict_logits(X_new))
    """

    def __init__(self):
        self.method = None
        self.platt_model = None
        self.temperature = None     # used by both vector and temperature
        self.bias = None            # used only by vector scaling
        self.fitted = False

    # Fit by inspecting the model: PyTorch → vector scaling, sklearn → Platt.
    def fit(self, model, X_cal, y_cal) -> None:
        """Fit calibration on a held-out calibration set; routes by model name."""
        y = y_cal.values if hasattr(y_cal, "values") else np.asarray(y_cal)
        model_name = model.get_name()

        if model_name in _PYTORCH_MODELS:
            self.method = CALIBRATION_METHOD_PYTORCH
            logits = model.predict_logits(X_cal)     # (n, n_classes)
            self.temperature, self.bias = fit_vector_scaling(logits, y)
        else:
            self.method = CALIBRATION_METHOD_SKLEARN
            logits = model.predict_logits(X_cal)
            # Platt needs 1D log-odds; collapse the per-class logits into class-1-vs-class-0 difference.
            if logits.ndim == 2 and logits.shape[1] >= 2:
                logits = logits[:, 1] - logits[:, 0]
            logits = logits.ravel()
            self.platt_model = fit_platt_scaling(logits, y)

        self.fitted = True
        logger.debug("Calibrator fitted: method=%s, model=%s.", self.method, model_name)

    # Apply the fitted calibration map to fresh raw logits.
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities ``(n_samples, n_classes)`` from raw logits.

        Platt accepts ``(n,)`` or ``(n, 1)`` log-odds; vector and temperature accept
        ``(n, n_classes)`` raw pre-softmax logits.
        """
        if not self.fitted:
            raise RuntimeError(
                "Calibrator has not been fitted. Call .fit() first."
            )

        # NaN logits would propagate silently through softmax to NaN probabilities,
        # which then yield argmax=0 and crash downstream metrics with length mismatch.
        # Warn at the boundary so the source of the NaN is obvious in the log.
        if np.isnan(logits).any():
            n_nan_rows = int(np.isnan(logits).any(axis=-1 if logits.ndim == 2 else 0).sum())
            logger.warning(
                "Calibrator.calibrate received logits with NaN in %d row(s); "
                "calibrated output rows will be NaN. The caller should slice NaN "
                "rows out before passing (typical cause: LSTM warm-up window).",
                n_nan_rows,
            )

        # Platt branch: collapse 2D logits into 1D log-odds if needed, then sigmoid via predict_proba.
        if self.method == CALIBRATION_METHOD_SKLEARN:
            if logits.ndim == 2 and logits.shape[1] >= 2:
                logits = logits[:, 1] - logits[:, 0]
            logits_2d = logits.reshape(-1, 1)
            return self.platt_model.predict_proba(logits_2d)

        # Vector branch: (logits + b) / T then softmax.
        if self.method == CALIBRATION_METHOD_PYTORCH:
            scaled = (logits + self.bias) / self.temperature
            return softmax(scaled, axis=1)

        # Temperature-only branch: logits / T then softmax.
        if self.method == CALIBRATION_METHOD_TEMPERATURE:
            scaled = logits / self.temperature
            return softmax(scaled, axis=1)

        raise ValueError(f"Unknown calibration method: {self.method}")

    # Alternate entry point: fit from pre-computed logits, bypassing the model interface.
    def fit_from_logits(self, logits, y_cal, method=CALIBRATION_METHOD_PYTORCH):
        """Fit calibration directly from pre-computed logits.

        Used by the LSTM training loop, where the windowing alignment requires
        pre-computed logits rather than re-running ``predict_logits`` through the
        model interface. ``method`` ∈ ``{"vector", "temperature", "platt"}``.
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

    # Compact one-line representation for logging.
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