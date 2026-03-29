"""
10) Pipeline
================
Master orchestration function that ties together split generation,
preprocessing, model training, calibration, and prediction storage
across all splits × models × seeds.

Single entry point: ``run_cpcv_pipeline()`` called from the notebook.
"""

import logging
import time

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score

from src.cpcv.cv import (
    build_path_matrix,
    generate_cpcv_splits,
    get_split_info,
)
from src.cpcv.preprocessing import preprocess_fold
from src.cpcv.calibration import Calibrator
from src.cpcv.models import create_model, list_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_cpcv_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    t1: pd.Series,
    bins_ret: pd.Series,
    n_groups: int = 6,
    k: int = 2,
    embargo_pct: float = 0.01,
    n_seeds: int = 3,
    models: list[str] | None = None,
    price_columns: list[str] | None = None,
) -> dict:
    """Run the full CPCV evaluation across all models and all splits.

    Parameters
    ----------
    X : pd.DataFrame
        Aligned feature matrix (from Step 9).
    y : pd.Series
        Labels, values in {-1, +1}.
    w : pd.Series
        Sample weights.
    t1 : pd.Series
        Barrier touch timestamps (for purging).
    bins_ret : pd.Series
        Return at barrier touch (``bins['ret']``), needed for financial
        performance computation downstream.
    n_groups : int
        Number of CPCV groups.
    k : int
        Number of test groups per split.
    embargo_pct : float
        Embargo fraction.
    n_seeds : int
        Number of random seeds per model per fold.
    models : list[str], optional
        Model names to evaluate. Defaults to all registered models.
    price_columns : list[str], optional
        Columns requiring FFD. Defaults to empty (no FFD applied).
        Specify based on ADF results from EDA step 8.4.

    Returns
    -------
    dict
        Contains 'predictions', 'split_info', 'path_map', 'n_paths',
        'n_splits', 'models', 'n_seeds'. The predictions dict is keyed
        by (model_name, split_idx, seed) tuples.
    """
    pipeline_start = time.time()

    # ── defaults ──────────────────────────────────────────────────────
    if models is None:
        models = list_models()
    if price_columns is None:
        price_columns = []  # user must specify from ADF results (EDA step 8.4)

    print("=" * 60)
    print("CPCV Pipeline")
    print("=" * 60)
    print(f"  Models:         {models}")
    print(f"  Seeds/fold:     {n_seeds}")
    print(f"  Groups (N):     {n_groups}")
    print(f"  Test groups (k):{k}")
    print(f"  Embargo:        {embargo_pct*100:.1f}%")
    print(f"  Price columns:  {price_columns}")
    print(f"  Samples:        {len(X)}")
    print(f"  Features:       {X.shape[1]}")
    print("=" * 60)

    # ── map labels: {-1, +1} → {0, 1} ────────────────────────────────
    y_mapped = ((y + 1) // 2).astype(int)

    # ── splits and paths ──────────────────────────────────────────────
    splits = generate_cpcv_splits(X, t1, n_groups, k, embargo_pct)
    n_paths, path_map = build_path_matrix(n_groups, k)
    split_info = get_split_info(X, t1, n_groups, k, embargo_pct)

    # ── storage ───────────────────────────────────────────────────────
    all_predictions = {}

    total_tasks = len(splits) * len(models) * n_seeds
    task_counter = 0

    # ── main loop ─────────────────────────────────────────────────────
    for split_idx, (train_idx, test_idx) in enumerate(splits):
        split_start = time.time()
        print(f"\n{'─'*60}")
        print(f"Split {split_idx + 1}/{len(splits)}")
        print(f"{'─'*60}")

        # ── extract fold data ─────────────────────────────────────────
        y_tr = y_mapped.iloc[train_idx]
        y_te = y_mapped.iloc[test_idx]
        w_tr = w.iloc[train_idx]
        t1_tr = t1.iloc[train_idx]
        ret_te = bins_ret.iloc[test_idx]

        # ── preprocessing (shared across all models this fold) ────────
        X_tr_proc, X_te_proc, selected, prep_info = preprocess_fold(
            X, train_idx, test_idx, y_tr, w_tr, t1_tr, price_columns
        )

        # re-align y, w, t1, ret after FFD may have dropped NaN rows
        y_tr = y_tr.loc[X_tr_proc.index]
        w_tr = w_tr.loc[X_tr_proc.index]
        t1_tr = t1_tr.loc[X_tr_proc.index]

        # keep pre-selection X for AR Logistic (needs log_ret column)
        X_tr_full = X_tr_proc.copy()
        X_te_full = X_te_proc.copy()

        # apply feature selection
        X_tr_sel = X_tr_proc[selected]
        X_te_sel = X_te_proc[selected]

        # ── calibration split: 80% train, 20% calibration ────────────
        cal_boundary = int(len(X_tr_sel) * 0.8)

        X_model = X_tr_sel.iloc[:cal_boundary]
        X_cal = X_tr_sel.iloc[cal_boundary:]
        X_model_full = X_tr_full.iloc[:cal_boundary]
        X_cal_full = X_tr_full.iloc[cal_boundary:]
        y_model = y_tr.iloc[:cal_boundary]
        y_cal = y_tr.iloc[cal_boundary:]
        w_model = w_tr.iloc[:cal_boundary]

        print(
            f"  Preprocessing: {len(selected)} features selected, "
            f"train={len(X_model)} + cal={len(X_cal)}, "
            f"test={len(X_te_sel)}"
        )

        # ── model × seed loop ────────────────────────────────────────
        for model_name in models:
            for seed in range(n_seeds):
                task_counter += 1
                task_start = time.time()

                # create model
                if model_name == "ar_logistic":
                    n_feat = X_tr_full.shape[1]
                else:
                    n_feat = len(selected)

                model = create_model(
                    model_name, n_features=n_feat, seed=seed
                )

                # select correct X depending on model type
                if model_name == "ar_logistic":
                    X_fit = X_model_full
                    X_c = X_cal_full
                    X_predict = X_te_full
                else:
                    X_fit = X_model
                    X_c = X_cal
                    X_predict = X_te_sel

                # train
                try:
                    model.fit(
                        X_fit, y_model,
                        sample_weight=w_model,
                        X_val=X_c, y_val=y_cal,
                    )
                except Exception as e:
                    logger.error(
                        "Model %s (seed=%d, split=%d) failed to train: %s",
                        model_name, seed, split_idx, e,
                    )
                    continue

                # calibrate (handle LSTM index mismatch on calibration set)
                try:
                    calibrator = Calibrator()
                    if model_name == "lstm" and hasattr(model, "last_valid_indices"):
                        # get logits and align y_cal to valid indices
                        cal_logits = model.predict_logits(X_c)
                        cal_valid_idx = model.last_valid_indices
                        y_cal_aligned = y_cal.iloc[cal_valid_idx]
                        calibrator.fit_from_logits(cal_logits, y_cal_aligned, method="temperature")
                    else:
                        calibrator.fit(model, X_c, y_cal)
                        
                except Exception as e:
                    logger.warning(
                        "Calibration failed for %s (seed=%d, split=%d): %s. "
                        "Using uncalibrated probabilities.",
                        model_name, seed, split_idx, e,
                    )
                    calibrator = None

                # predict on test fold
                raw_logits = model.predict_logits(X_predict)

                if calibrator is not None:
                    cal_proba = calibrator.calibrate(raw_logits)
                else:
                    cal_proba = model.predict_proba(X_predict)

                y_pred = np.argmax(cal_proba, axis=1)

                # handle LSTM index mismatch
                if model_name == "lstm" and hasattr(model, "last_valid_indices"):
                    valid_idx = model.last_valid_indices
                    test_timestamps = X_te_proc.index[valid_idx]
                    y_true_aligned = y_te.reindex(test_timestamps).values
                    ret_aligned = ret_te.reindex(test_timestamps).values
                else:
                    test_timestamps = X_te_proc.index
                    y_true_aligned = y_te.reindex(test_timestamps).values
                    ret_aligned = ret_te.reindex(test_timestamps).values

                f1 = f1_score(y_true_aligned, y_pred, average="macro")
                try:
                    auc = roc_auc_score(y_true_aligned, cal_proba[:, 1])
                except ValueError:
                    auc = float("nan")  # single class in test fold

                # store
                store_key = (model_name, split_idx, seed)
                all_predictions[store_key] = {
                    "y_true": y_true_aligned,
                    "y_pred": y_pred,
                    "cal_proba": cal_proba,
                    "f1_macro": f1,
                    "roc_auc": auc,
                    "timestamps": test_timestamps,
                    "test_idx": test_idx,
                    "ret": ret_aligned,
                    "split_idx": split_idx,
                    "seed": seed,
                    "prep_info": prep_info,
                    "calibrator": repr(calibrator) if calibrator else "none",
                }
                
                elapsed = time.time() - task_start
                print(
                    f"  [{task_counter:>3d}/{total_tasks}] "
                    f"{model_name:>20s} (seed={seed}) "
                    f"F1={f1:.3f}  AUC={auc:.3f} "
                    f"({elapsed:.1f}s)"
                )

        split_elapsed = time.time() - split_start
        print(f"  Split {split_idx + 1} completed in {split_elapsed:.1f}s")

    # ── summary ───────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    n_successful = len(all_predictions)
    n_failed = total_tasks - n_successful

    print(f"\n{'='*60}")
    print(f"CPCV Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Total time:     {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print(f"  Tasks:          {n_successful} succeeded, {n_failed} failed")
    print(f"  Results keys:   {n_successful} (model, split, seed) entries")
    print(f"  Backtest paths: {n_paths}")
    print(f"{'='*60}")

    if n_failed > 0:
        logger.warning("%d model fits failed during pipeline execution.", n_failed)

    return {
        "predictions": all_predictions,
        "split_info": split_info,
        "path_map": path_map,
        "n_paths": n_paths,
        "n_splits": len(splits),
        "models": models,
        "n_seeds": n_seeds,
    }