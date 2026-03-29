"""
8.1) Abstract Base Class
=============================
Define a uniform interface that every model must implement, so the CPCV
pipeline can iterate over models without knowing their internals.

Labels convention: the pipeline maps original labels {-1, +1} to {0, 1}
before passing to models, so all models work with standard 0-indexed
classes. The evaluation module maps back to {-1, +1} for economic
interpretation.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """All models in the pipeline implement this interface."""

    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.seed = seed

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

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training features and labels (0-indexed classes).
        sample_weight : np.ndarray, optional
            Per-sample weights from AFML Chapter 4.
        X_val, y_val : np.ndarray, optional
            Validation set for early stopping (neural nets).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n_samples, n_classes)."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard labels via argmax of predict_proba."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable model name for logging/comparison."""
        pass