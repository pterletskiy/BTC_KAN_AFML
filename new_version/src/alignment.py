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
from typing import Tuple, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer

logger = logging.getLogger(__name__)


def fit_transform_scaler(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    scaler_type: str = 'robust', 
    feature_range: Tuple[float, float] = (-1.0, 1.0)
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, List[str]]:
    """
    Fit a scaler exclusively on training data and transform both folds safely.
    
    Robust bounds scaling, min-max scaling, or quantile transformation mapping inputs 
    to a strictly fixed domain (e.g., [-1, 1]) necessary for KAN B-spline grids.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training fold features, shape (n_train, n_features).
    X_test : pd.DataFrame
        Testing fold features, shape (n_test, n_features).
    scaler_type : str, optional
        Either 'robust' (median interquartile), 'minmax' (strict bounds), 
        or 'quantile' (uniform marginals). Default is 'robust'.
    feature_range : tuple, optional
        Target fixed boundary range. Default is (-1.0, 1.0).

    Returns
    -------
    X_train_scaled : pd.DataFrame
        Scaled training features.
    X_test_scaled : pd.DataFrame
        Scaled testing features.
    scaler : object
        Fitted scikit-learn scaler object.
    dropped_cols : list
        List of feature column names dropped due to zero variance, NaN, or Inf.
        (Note for backward compatibility: unpack first 3 or 4 elements accordingly).

    Raises
    ------
    ValueError
        If inputs are empty DataFrames, or `scaler_type` is unrecognized.
    """
    if X_train.empty or X_train.shape[1] == 0:
        raise ValueError("X_train must be a non-empty DataFrame.")
    if X_test.empty or X_test.shape[1] == 0:
        raise ValueError("X_test must be a non-empty DataFrame.")
        
    if scaler_type not in ['robust', 'minmax', 'quantile']:
        raise ValueError(f"scaler_type must be 'robust', 'minmax', or 'quantile'. Got: '{scaler_type}'")
        
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    else:  # 'quantile'
        scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
        
    # FIT strictly on X_train. NEVER on X_test.
    scaler.fit(X_train_scaled)
    
    X_train_scaled.loc[:, :] = scaler.transform(X_train_scaled)
    X_test_scaled.loc[:, :] = scaler.transform(X_test_scaled)
    
    # 1. Map quantile output [0, 1] to feature_range linearly
    if scaler_type == 'quantile':
        span = feature_range[1] - feature_range[0]
        X_train_scaled = X_train_scaled * span + feature_range[0]
        X_test_scaled = X_test_scaled * span + feature_range[0]
    
    # Enforce hard boundaries strictly ensuring gradient alignment stability.
    if scaler_type in ['robust', 'quantile']:
        X_train_scaled = X_train_scaled.clip(lower=feature_range[0], upper=feature_range[1])
        X_test_scaled = X_test_scaled.clip(lower=feature_range[0], upper=feature_range[1])

    # 2. Check for zero variance, NaNs, or Infs
    invalid_mask_train = X_train_scaled.isna() | np.isinf(X_train_scaled)
    vars_train = X_train_scaled.var(axis=0, skipna=True)
    
    dropped_cols = []
    for col in X_train_scaled.columns:
        if invalid_mask_train[col].any() or vars_train[col] == 0.0 or pd.isna(vars_train[col]):
            dropped_cols.append(col)
            
    if dropped_cols:
        logger.warning(
            "[SCALING] Dropping %d feature(s) due to zero-variance, NaN, or Inf post-scaling: %s", 
            len(dropped_cols), dropped_cols
        )
        X_train_scaled = X_train_scaled.drop(columns=dropped_cols)
        X_test_scaled = X_test_scaled.drop(columns=dropped_cols, errors='ignore')

    if not X_train_scaled.empty:
        actual_min = X_train_scaled.min().min()
        actual_max = X_train_scaled.max().max()
        logger.info(
            "[SCALING] Executed type='%s' on %d features. Achieved Train Min/Max bounds: [%.2f, %.2f]",
            scaler_type, X_train_scaled.shape[1], actual_min, actual_max
        )
    else:
        logger.error("[SCALING] All features were dropped due to zero variance/NaN distributions!")
    
    return X_train_scaled, X_test_scaled, scaler, dropped_cols


def compute_SFI(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    clf: Any, 
    sample_weight_train: Optional[pd.Series] = None, 
    sample_weight_test: Optional[pd.Series] = None,
    scoring: str = 'neg_log_loss'
) -> pd.DataFrame:
    """
    Implement the Single Feature Importance (SFI) out-of-sample mapping performance indicator.
    
    Trains individual estimators exclusively on one feature to determine 
    pure predictive validity, avoiding multi-collinear substitutions (AFML Ch. 8).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training fold features, shape (n_train, n_features).
    y_train : pd.Series
        Training target labels.
    X_test : pd.DataFrame
        Testing fold features, shape (n_test, n_features).
    y_test : pd.Series
        Testing objective targets explicitly maintaining scoring index vectors.
    clf : sklearn.base.BaseEstimator
        Classification estimator to establish target maps.
    sample_weight_train : pd.Series, optional
        Sample weights applied during .fit() (AFML Ch. 6, Sec 6.3).
    sample_weight_test : pd.Series, optional
        Sample weights applied directly to the scoring metric evaluation.
    scoring : str, optional
        Scikit-learn score matching boundary constraints. Default 'neg_log_loss'.

    Returns
    -------
    pd.DataFrame
        DataFrame ranking individual features by OOS score ['feature', 'sfi_score'].

    Raises
    ------
    ValueError
        If inputs are empty DataFrames.
    """
    if X_train.empty or X_train.shape[1] == 0:
        raise ValueError("X_train must be a non-empty DataFrame.")
    if X_test.empty or X_test.shape[1] == 0:
        raise ValueError("X_test must be a non-empty DataFrame.")
        
    sfi_scores = []
    scorer = check_scoring(clf, scoring=scoring)
    
    features = list(X_train.columns)
    
    for i, f in enumerate(features):
        X_train_single = X_train[[f]].values
        X_test_single = X_test[[f]].values
        
        # Clone ensures no warm-start accumulation across iteration loops
        clf_f = clone(clf)
        
        # Train strictly with weights if passed
        if sample_weight_train is not None:
            try:
                clf_f.fit(X_train_single, y_train, sample_weight=sample_weight_train.values)
            except TypeError as e:
                if i == 0: 
                    logger.warning(
                        "[SFI] Estimator fails capturing sample_weight configurations effectively. "
                        "Omitting train weights: %s", e
                    )
                clf_f.fit(X_train_single, y_train)
        else:
            clf_f.fit(X_train_single, y_train)
            
        # Map metric evaluation out-of-sample determining genuine forecasting capability
        if sample_weight_test is not None:
            try:
                score = scorer(clf_f, X_test_single, y_test, sample_weight=sample_weight_test.values)
            except TypeError as e:
                if i == 0:
                    logger.warning(
                        "[SFI] Scorer metric rejects kwargs for sample_weight_test. "
                        "Omitting test weights: %s", e
                    )
                score = scorer(clf_f, X_test_single, y_test)
        else:
            score = scorer(clf_f, X_test_single, y_test)
            
        sfi_scores.append({'feature': f, 'sfi_score': score})
        
        if (i + 1) % 10 == 0:
            logger.debug("[SFI Progress] Feature %d/%d ('%s') score: %.4f", i + 1, len(features), f, score)
            
    # Organize structural output ordering
    sfi_df = pd.DataFrame(sfi_scores)
    sfi_df = sfi_df.sort_values(by='sfi_score', ascending=False)
    
    return sfi_df


def filter_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    sfi_scores: pd.DataFrame, 
    threshold: float = 0.0,
    baseline: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, list, pd.DataFrame]:
    """
    Dynamically drop features exhibiting purely catastrophic out-of-sample combinations.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training fold features safely matching scaled properties.
    X_test : pd.DataFrame
        Test fold features matching X_train properties exactly.
    sfi_scores : pd.DataFrame
        Scoring ranking boundaries evaluated via single feature execution.
    threshold : float, optional
        Fallback threshold explicitly dropping features beneath value (default 0.0).
    baseline : float, optional
        A baseline out-of-sample score (e.g., from a dummy classifier). 
        If provided, takes precedence over the static `threshold`.

    Returns
    -------
    X_train_filtered : pd.DataFrame
        Purged training fold retaining surviving features cleanly natively.
    X_test_filtered : pd.DataFrame
        Purged test fold retaining surviving features cleanly natively.
    kept_ordered : list
        List specifying exact tracking keys surviving the filter arrays.
    sfi_kept : pd.DataFrame
        The strictly paired subset of `sfi_scores` corresponding to the surviving features.
        (Note for backward compatibility: unpack first 3 or 4 elements accordingly).

    Raises
    ------
    ValueError
        If inputs are empty, or selection collapse eliminates all target features permanently!
    """
    if X_train.empty or X_train.shape[1] == 0:
        raise ValueError("X_train must be a non-empty DataFrame.")
    if X_test.empty or X_test.shape[1] == 0:
        raise ValueError("X_test must be a non-empty DataFrame.")
        
    cutoff = baseline if baseline is not None else threshold
    
    if baseline is None:
        logger.warning(
            "[FILTERING] No baseline score provided. Defaulting to strict threshold=%s. "
            "For metrics like neg_log_loss, threshold=0.0 is often semantically wrong "
            "and should be calibrated against a random classifier explicitly natively.", threshold
        )
        
    kept_subset = sfi_scores[sfi_scores['sfi_score'] >= cutoff]['feature'].tolist()
    dropped_subset = sfi_scores[sfi_scores['sfi_score'] < cutoff]['feature'].tolist()
    
    if not kept_subset:
        raise ValueError(
            f"Terminal filtering collapse: All features fell below cutoff ({cutoff}). "
            "Please explicitly lower the tolerance limit or re-examine upstream implementations."
        )
        
    kept_ordered = [col for col in X_train.columns if col in kept_subset]
    
    X_train_filtered = X_train[kept_ordered]
    X_test_filtered = X_test[kept_ordered]
    sfi_kept = sfi_scores[sfi_scores['feature'].isin(kept_ordered)].copy()
    
    logger.info(
        "[FILTERING] SFI cutoff processed (%.3f). Target mapping: %d Initial -> %d Kept. Dropped %d. Purged Targets: %s",
        cutoff, X_train.shape[1], len(kept_ordered), len(dropped_subset), dropped_subset
    )
    
    return X_train_filtered, X_test_filtered, kept_ordered, sfi_kept
