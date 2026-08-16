"""
8.0) Registry & Factory
=============================
Provide a single registry and factory function so the pipeline can
create model instances by name without importing individual classes.

Models not yet implemented are skipped gracefully, so the pipeline
works as soon as at least one model file exists.
"""

import logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {}

# ── Benchmarks (always available) ─────────────────────────────────────
from .benchmarks import ARLogistic, LogisticRegressionModel

MODEL_REGISTRY["ar_logistic"] = ARLogistic
MODEL_REGISTRY["logistic"] = LogisticRegressionModel

# ── Tree models ───────────────────────────────────────────────────────
try:
    from .tree_models import RandomForestModel, XGBoostModel
    MODEL_REGISTRY["random_forest"] = RandomForestModel
    MODEL_REGISTRY["xgboost"] = XGBoostModel
except ImportError:
    logger.debug("tree_models not available yet, skipping.")

# ── LSTM ──────────────────────────────────────────────────────────────
try:
    from .lstm_model import LSTMModel
    MODEL_REGISTRY["lstm"] = LSTMModel
except ImportError:
    logger.debug("lstm_model not available yet, skipping.")

# ── KAN ───────────────────────────────────────────────────────────────
try:
    from .kan_model import KANModel
    MODEL_REGISTRY["kan"] = KANModel
except ImportError:
    logger.debug("kan_model not available yet, skipping.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_model(
    name: str, n_features: int, n_classes: int = 2, seed: int = 42
):
    """Factory function. Returns an instance of the requested model.

    Parameters
    ----------
    name : str
        Key in MODEL_REGISTRY (e.g., 'logistic', 'kan').
    n_features : int
        Number of input features.
    n_classes : int
        Number of output classes.
    seed : int
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](
        n_features=n_features, n_classes=n_classes, seed=seed
    )


def list_models() -> list[str]:
    """Return list of currently available model name strings."""
    return list(MODEL_REGISTRY.keys())