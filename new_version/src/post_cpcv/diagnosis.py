"""
12.3) Diagnostics
==================
Interactive inspection helpers for examining trained model predictions
after ``run_cpcv_pipeline`` and ``analyze_results`` have produced their
outputs. These helpers operate on the raw predictions dict and never
touch the canonical event-aligned X / y / w / t1 series. Use freely
from notebooks without polluting global state.
"""

import numpy as np


def pool_predictions(model_name, results, n_seeds=2, n_splits=15):
    """Pool calibrated P(y=1) and true labels across splits and seeds.

    Parameters
    ----------
    model_name : str
        One of the model keys used in ``results["predictions"]``.
    results : dict
        Output of ``run_cpcv_pipeline``.
    n_seeds : int, default 2
        Number of seeds the model was trained with. Iterates 0..n_seeds-1.
    n_splits : int, default 15
        Number of CPCV splits. Iterates 0..n_splits-1.

    Returns
    -------
    proba_pool : np.ndarray
        Concatenated calibrated P(class=1) across all (split, seed)
        combinations present in ``results["predictions"]``.
    y_pool : np.ndarray
        Concatenated ground-truth labels (0 or 1) aligned with proba_pool.
    """
    proba_pool, y_pool = [], []
    for split_idx in range(n_splits):
        for seed in range(n_seeds):
            key = (model_name, split_idx, seed)
            if key in results["predictions"]:
                proba_pool.append(results["predictions"][key]["cal_proba"][:, 1])
                y_pool.append(results["predictions"][key]["y_true"])

    proba_pool = np.concatenate(proba_pool) if proba_pool else np.array([])
    y_pool = np.concatenate(y_pool) if y_pool else np.array([])
    return proba_pool, y_pool


def calibration_audit(model_name, results, n_seeds=2, n_splits=15, n_bins=10):
    """Print a calibration table comparing predicted P(y=1) to empirical
    P(y=1) within fixed-width probability bins.

    Useful for diagnosing systematic directional bias (mean predicted
    well below empirical), miscalibrated sharpness (high-confidence bins
    not matching their predicted level), or distributional collapse
    (most predictions piled in [0.4, 0.5)).

    Parameters
    ----------
    n_bins : int, default 10
        Number of equal-width bins on [0, 1].
    """
    proba_pool, y_pool = pool_predictions(model_name, results, n_seeds, n_splits)

    if len(proba_pool) == 0:
        print(f"{model_name}: no predictions found.")
        return

    print(f"{model_name}: mean P̂(y=1) = {proba_pool.mean():.4f}, "
          f"empirical = {y_pool.mean():.4f}")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(proba_pool, bin_edges) - 1

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 10:
            print(f"  P̂ ∈ [{bin_edges[b]:.1f}, {bin_edges[b+1]:.1f}): "
                  f"n={mask.sum()}, empirical = {y_pool[mask].mean():.3f}")