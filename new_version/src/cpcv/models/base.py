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


# Contract every CPCV model must satisfy: fit, predict_proba, get_name, plus a default predict.
class BaseModel(ABC):
    """All models in the pipeline implement this interface."""

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.seed = seed

    # Fit on the training fold; X_val/y_val are used by models that support early stopping.
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Train the model.

        ``sample_weight`` carries the AFML Chapter 4 weights (return-attribution
        and time-decay). ``X_val`` / ``y_val`` are optional and used only by
        models that support early stopping (neural nets, XGBoost).
        """
        pass

    # Class-probability vector, shape (n_samples, n_classes); the primary output for downstream calibration.
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n_samples, n_classes)."""
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