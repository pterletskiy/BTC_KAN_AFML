"""
10.1) Abstract Base Class
=============================
Uniform interface that every model in the CPCV pipeline implements, so the
runner can iterate over heterogeneous classifiers without knowing their
internals.

Label convention: the pipeline maps original labels {-1, +1} to {0, 1} before
passing them to models, so every implementation works with standard 0-indexed
classes. The evaluation layer maps back to {-1, +1} for economic interpretation.
"""

from abc import ABC, abstractmethod

import numpy as np


# Contract every CPCV model must satisfy: fit, predict_proba, predict_logits, get_name,
# plus a default predict implemented via argmax.
class BaseModel(ABC):
    """All models in the pipeline implement this interface."""

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.seed = seed

    # Fit on the training fold. X_val/y_val/sample_weight_val are used by models that
    # support early stopping (neural nets, XGBoost); ignored by others.
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> None:
        """Train the model.

        ``sample_weight`` carries the AFML Chapter 4 weights (return-attribution
        and time-decay). ``X_val``, ``y_val``, and ``sample_weight_val`` are
        optional and used only by models that support early stopping (neural
        nets, XGBoost); ``sample_weight_val`` keeps the early-stopping criterion
        weighted on the same basis as the training loss.
        """
        pass

    # Class-probability vector, shape (n_samples, n_classes); the primary output for downstream calibration.
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n_samples, n_classes)."""
        pass

    # Raw logits for downstream calibration. Shape convention is heterogeneous and intentional:
    # sklearn-family models (Logistic, RF, XGB) return (n_samples, 1) log-odds for Platt scaling;
    # PyTorch-family models (KAN, LSTM) return (n_samples, n_classes) pre-softmax logits for
    # vector scaling. The calibration layer dispatches on this shape.
    @abstractmethod
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Return raw logits for downstream calibration.

        Shape convention:
          - sklearn-family models (Logistic, RF, XGB) return ``(n_samples, 1)``
            log-odds intended for Platt scaling.
          - PyTorch-family models (KAN, LSTM) return ``(n_samples, n_classes)``
            pre-softmax logits intended for vector scaling.

        Models that drop rows during inference (e.g. sequence-window models like
        LSTM) pad the dropped positions with NaN so the output row count always
        matches the input row count.
        """
        pass

    # Hard-label fallback derived from predict_proba; models can override for efficiency.
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard labels via argmax of ``predict_proba``."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    # Human-readable name used in logs and aggregated result tables.
    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable model name for logging/comparison."""
        pass