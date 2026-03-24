"""
src/6_preproc.py
----------------
Distribution-dependent preprocessing executed STRICTLY INSIDE the CV loop.
All transformations fit on train folds only. No data leakage permitted.

Pipeline position: Called by MFW_Pipeline.ipynb after 5_cv.py fold generation.
Downstream consumer: 7_models.py (KAN architecture).

References:
  - AFML Ch. 8, Sec. 8.4.1 — Single Feature Importance (SFI)
  - AFML Ch. 9, Sec. 9.4   — Scoring and Hyper-parameter Tuning (neg_log_loss)
  - AFML Ch. 6, Sec. 6.3   — Bagging Classifiers (sample weights)
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.preprocessing import MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


def fit_transform_scaler(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    scaler_type: str = 'robust', 
    feature_range: tuple = (-1, 1)
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fit a scaler exclusively on training data and transform both folds safely.
    
    Robust bounds scaling resolving extreme outliers specifically calibrating the fixed domains
    [−1, 1] necessary for stable mapping bounds running KAN B-spline architecture grids.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training fold chronological observation mapping feature columns explicitly.
    X_test : pd.DataFrame
        Testing fold isolating forward predicting horizons.
    scaler_type : str, optional
        Either 'robust' mapping median interquartile scale bounded locally, 
        or 'minmax' applying strict scale maps (default 'robust').
    feature_range : tuple, optional
        Fixed bounded tuple limits mapping inputs inherently (default (-1, 1)).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, object]
        Processed (X_train_scaled, X_test_scaled) tracking explicit target inputs and the fitted scaler object.

    Raises
    ------
    ValueError
        If `scaler_type` input is unrecognized.

    Notes
    -----
    Strict isolation constraints guarantee .fit() is ONLY executed targeting X_train preserving pipeline integrity.
    """
    if scaler_type not in ['robust', 'minmax']:
        raise ValueError(f"scaler_type must be either 'robust' or 'minmax'. Got: '{scaler_type}'")
        
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:  # 'minmax'
        scaler = MinMaxScaler(feature_range=feature_range)
        
    # FIT strictly on X_train. NEVER on X_test.
    scaler.fit(X_train_scaled)
    
    X_train_scaled.loc[:, :] = scaler.transform(X_train_scaled)
    X_test_scaled.loc[:, :] = scaler.transform(X_test_scaled)
    
    # Enforce hard boundaries strictly ensuring gradient alignment stability across KAN nodes.
    if scaler_type == 'robust':
        X_train_scaled = X_train_scaled.clip(lower=feature_range[0], upper=feature_range[1])
        X_test_scaled = X_test_scaled.clip(lower=feature_range[0], upper=feature_range[1])

    actual_min = X_train_scaled.min().min()
    actual_max = X_train_scaled.max().max()
    
    logger.info(
        "[SCALING] Executed type='%s' on %d features. Achieved Train Min/Max bounds: [%.2f, %.2f]",
        scaler_type, X_train_scaled.shape[1], actual_min, actual_max
    )
    
    return X_train_scaled, X_test_scaled, scaler


def compute_SFI(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    clf, 
    sample_weight_train: pd.Series = None, 
    scoring: str = 'neg_log_loss'
) -> pd.DataFrame:
    """Implement the Single Feature Importance (SFI) out-of-sample mapping performance indicator.
    
    Executes MLDP Ch 8 isolation tracking. Trains individual estimators mapping one variable specifically
    determining pure predictive validity avoiding multi-collinear substitutions.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target labels natively associated scaling weights tracking MLDP.
    X_test : pd.DataFrame
        Testing target bounds.
    y_test : pd.Series
        Testing objective targets explicitly maintaining scoring index vectors natively.
    clf : sklearn.base.BaseEstimator
        Classification estimator configured to establish target maps.
    sample_weight_train : pd.Series, optional
        Average uniqueness bounds executing target sizes properly scaling estimator predictions.
    scoring : str, optional
        Scikit score tracking target boundary (default 'neg_log_loss'). Neg_log_loss establishes
        AFML Ch. 9 preferred objective evaluating confidence bounds mapping drawdown constraints dynamically.

    Returns
    -------
    pd.DataFrame
        Target indicator evaluating boundaries mapping scores indexing single ['feature', 'sfi_score'].

    Notes
    -----
    Implements SFI as described in AFML Ch. 8, Section 8.4.1.
    """
    sfi_scores = []
    scorer = check_scoring(clf, scoring=scoring)
    
    features = list(X_train.columns)
    
    for i, f in enumerate(features):
        X_train_single = X_train[[f]].values
        X_test_single = X_test[[f]].values
        
        # Clone ensures no warm-start accumulation across iteration loops
        clf_f = clone(clf)
        
        # Explicit evaluation passing uniqueness indices 
        if sample_weight_train is not None:
            try:
                clf_f.fit(X_train_single, y_train, sample_weight=sample_weight_train.values)
            except TypeError as e:
                if i == 0:  # Log warning only on the first feature executing
                    logger.warning("[SFI] Estimator fails capturing sample_weight configurations effectively. Omitting weights: %s", e)
                clf_f.fit(X_train_single, y_train)
        else:
            clf_f.fit(X_train_single, y_train)
            
        # Map metric evaluation out-of-sample determining genuine forecasting capability
        score = scorer(clf_f, X_test_single, y_test)
        sfi_scores.append((f, score))
        
        if (i + 1) % 10 == 0:
            logger.debug("[SFI Progress] Feature %d/%d ('%s') score: %.4f", i + 1, len(features), f, score)
            
    # Organize structural output ordering
    sfi_df = pd.DataFrame(sfi_scores, columns=['feature', 'sfi_score'])
    sfi_df = sfi_df.sort_values(by='sfi_score', ascending=False)
    
    return sfi_df


def filter_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    sfi_scores: pd.DataFrame, 
    threshold: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Dynamically drops features exhibiting purely catastrophic out-of-sample predictions tracking Single Feature Importance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Original input temporal logic mapping observation evaluations natively.
    X_test : pd.DataFrame
        Original boundaries matching test subsets directly.
    sfi_scores : pd.DataFrame
        Scoring structural boundaries containing metrics evaluated via native single runs.
    threshold : float, optional
        Threshold target pruning features generating inferior OOS output structures explicitly dropping noise targets (default 0.0).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, list]
        Purged datasets explicitly generating the identical exact tracking arrays alongside native lists defining surviving element references.

    Raises
    ------
    ValueError
        Catastrophic selection pruning entirely eliminating all feature target nodes.

    Notes
    -----
    Operates strictly as isolated pipeline mechanism retaining temporal integrity matrices supporting 5_cv.py boundaries.
    """
    kept_subset = sfi_scores[sfi_scores['sfi_score'] >= threshold]['feature'].tolist()
    dropped_subset = sfi_scores[sfi_scores['sfi_score'] < threshold]['feature'].tolist()
    
    # Check catastrophic boundaries mapping total feature isolation
    if not kept_subset:
        raise ValueError(
            f"Terminal filtering collapse: All features fell below SFI threshold ({threshold}). "
            "Please lower the tolerance limit or re-examine upstream feature implementations isolating predictive capabilities natively."
        )
        
    # Re-align explicitly matching original index column sequence targeting stable arrays 
    kept_ordered = [col for col in X_train.columns if col in kept_subset]
    
    # Pure native isolation avoiding index resetting boundaries
    X_train_filtered = X_train[kept_ordered]
    X_test_filtered = X_test[kept_ordered]
    
    logger.info(
        "[FILTERING] SFI cutoff processed (%.3f). Target mapping: %d Initial -> %d Kept. Dropped %d. Purged Targets: %s",
        threshold, X_train.shape[1], len(kept_ordered), len(dropped_subset), dropped_subset
    )
    
    return X_train_filtered, X_test_filtered, kept_ordered
