"""
11) Pipeline
================
Master orchestration function that ties together split generation,
preprocessing, model training, calibration, and prediction storage
across all splits × models × seeds.

Hyperparameter tuning (optional, nested): for each of the 15 CPCV
splits, Optuna TPE with Purged K-Fold CV tunes hyperparameters using
only the training data of that split. Each split gets its own optimal
parameters, correctly simulating how an algorithm would adapt to new
data in live trading. This nested architecture ensures DSR and PBO
remain valid (López de Prado, 2018, Ch. 7 & 12).

Single entry point: ``run_cpcv_pipeline()`` called from the notebook.
"""

import logging
import time

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, log_loss, roc_auc_score

from src.cpcv.cv import (
    build_path_matrix,
    generate_cpcv_splits,
    get_split_info,
)
from src.cpcv.preprocessing import preprocess_fold
from src.cpcv.calibration import Calibrator
from src.cpcv.models import create_model, list_models

logger = logging.getLogger(__name__)


# =====================================================================
# Pristine module-default capture and reset
# =====================================================================
# ``_apply_tuned_params`` mutates module-level constants on the model
# files (LOGISTIC_C, RF_N_ESTIMATORS, KAN_HIDDEN, ...). Without a reset,
# a second ``run_cpcv_pipeline`` call inherits the previous run's tuned
# values: e.g. running with tuning, then re-running without tuning,
# gives the second run's "untuned" models the first run's tuned widths.
# ``_reset_module_defaults`` snapshots the original constants the first
# time it is invoked and restores them on every subsequent call.
_PRISTINE_DEFAULTS: dict[str, dict[str, object]] = {}

# Names per module mirror the keys ``_apply_tuned_params`` writes to.
_TRACKED_CONSTANTS = {
    "benchmarks":  ["LOGISTIC_C", "LOGISTIC_PENALTY"],
    "tree_models": [
        "RF_N_ESTIMATORS", "RF_MAX_DEPTH", "RF_MIN_SAMPLES_LEAF", "RF_MAX_FEATURES",
        "XGB_MAX_DEPTH", "XGB_LEARNING_RATE", "XGB_MIN_CHILD_WEIGHT",
        "XGB_SUBSAMPLE", "XGB_COLSAMPLE_BYTREE", "XGB_GAMMA",
        "XGB_REG_ALPHA", "XGB_REG_LAMBDA",
    ],
    "lstm_model":  ["LSTM_HIDDEN_SIZE", "LSTM_NUM_LAYERS", "LSTM_DROPOUT", "LSTM_LR"],
    "kan_model":   ["KAN_HIDDEN", "KAN_HIDDEN2", "KAN_GRID", "KAN_LR", "KAN_WEIGHT_DECAY"],
}


def _model_module(name: str):
    """Resolve a tracked-module name to the live module object."""
    import src.cpcv.models.benchmarks as bench_mod
    import src.cpcv.models.tree_models as tree_mod
    import src.cpcv.models.lstm_model as lstm_mod
    import src.cpcv.models.kan_model as kan_mod
    return {
        "benchmarks": bench_mod,
        "tree_models": tree_mod,
        "lstm_model": lstm_mod,
        "kan_model": kan_mod,
    }[name]


def _reset_module_defaults() -> None:
    """Restore tracked module-level constants to their pristine values.

    On the first call, snapshots the current values (which are the
    pristine import-time defaults, since this runs before any
    ``_apply_tuned_params`` mutation). On every subsequent call,
    restores those snapshots, undoing any tuning-driven mutation
    from a prior ``run_cpcv_pipeline`` invocation.
    """
    first_call = not _PRISTINE_DEFAULTS

    for mod_name, attr_names in _TRACKED_CONSTANTS.items():
        mod = _model_module(mod_name)

        if first_call:
            _PRISTINE_DEFAULTS[mod_name] = {
                a: getattr(mod, a) for a in attr_names if hasattr(mod, a)
            }
        else:
            for a, v in _PRISTINE_DEFAULTS[mod_name].items():
                setattr(mod, a, v)

    if not first_call:
        logger.debug("Reset tracked module-level constants to pristine defaults.")


# =====================================================================
# Apply tuned hyperparameters to module-level constants
# =====================================================================
def _apply_tuned_params(tuning_results: dict) -> dict:
    """Override module-level constants with tuned hyperparameters.

    Parameters
    ----------
    tuning_results : dict
        Output of tune_all_models(). Keys are model names, values are
        dicts with "best_params".

    Returns
    -------
    dict
        Summary of applied parameters for logging.
    """
    applied = {}

    # ── Logistic Regression ──────────────────────────────────────────
    if "logistic" in tuning_results:
        params = tuning_results["logistic"].get("best_params", {})
        if params:
            import src.cpcv.models.benchmarks as bench_mod
            if "C" in params:
                bench_mod.LOGISTIC_C = params["C"]
            if "penalty" in params:
                bench_mod.LOGISTIC_PENALTY = params["penalty"]
            applied["logistic"] = params

    # ── Random Forest ────────────────────────────────────────────────
    if "random_forest" in tuning_results:
        params = tuning_results["random_forest"].get("best_params", {})
        if params:
            import src.cpcv.models.tree_models as tree_mod
            if "n_estimators" in params:
                tree_mod.RF_N_ESTIMATORS = params["n_estimators"]
            if "max_depth" in params:
                v = params["max_depth"]
                tree_mod.RF_MAX_DEPTH = int(v) if v is not None and v == v else None
            if "min_samples_leaf" in params:
                tree_mod.RF_MIN_SAMPLES_LEAF = params["min_samples_leaf"]
            if "max_features" in params:
                tree_mod.RF_MAX_FEATURES = params["max_features"]
            applied["random_forest"] = params

    # ── XGBoost ──────────────────────────────────────────────────────
    if "xgboost" in tuning_results:
        params = tuning_results["xgboost"].get("best_params", {})
        if params:
            import src.cpcv.models.tree_models as tree_mod
            if "max_depth" in params:
                tree_mod.XGB_MAX_DEPTH = int(params["max_depth"])
            if "learning_rate" in params:
                tree_mod.XGB_LEARNING_RATE = params["learning_rate"]
            if "min_child_weight" in params:
                tree_mod.XGB_MIN_CHILD_WEIGHT = int(params["min_child_weight"])
            if "subsample" in params:
                tree_mod.XGB_SUBSAMPLE = params["subsample"]
            if "colsample_bytree" in params:
                tree_mod.XGB_COLSAMPLE_BYTREE = params["colsample_bytree"]
            if "gamma" in params:
                tree_mod.XGB_GAMMA = params["gamma"]
            if "reg_alpha" in params:
                tree_mod.XGB_REG_ALPHA = params["reg_alpha"]
            if "reg_lambda" in params:
                tree_mod.XGB_REG_LAMBDA = params["reg_lambda"]
            applied["xgboost"] = params

    # ── LSTM ──────────────────────────────────────────────────────────
    if "lstm" in tuning_results:
        params = tuning_results["lstm"].get("best_params", {})
        if params:
            import src.cpcv.models.lstm_model as lstm_mod
            if "hidden_size" in params:
                lstm_mod.LSTM_HIDDEN_SIZE = params["hidden_size"]
            if "num_layers" in params:
                lstm_mod.LSTM_NUM_LAYERS = params["num_layers"]
            if "dropout" in params:
                lstm_mod.LSTM_DROPOUT = params["dropout"]
            if "learning_rate" in params:
                lstm_mod.LSTM_LR = params["learning_rate"]
            applied["lstm"] = params

    # ── KAN ────────────────────────────────────────────────────────────
    if "kan" in tuning_results:
        params = tuning_results["kan"].get("best_params", {})
        if params:
            import src.cpcv.models.kan_model as kan_mod
            if "width1" in params:
                kan_mod.KAN_HIDDEN = int(params["width1"])
            if "width2" in params:
                kan_mod.KAN_HIDDEN2 = int(params["width2"])
            if "grid" in params:
                kan_mod.KAN_GRID = int(params["grid"])
            if "lr" in params:
                kan_mod.KAN_LR = float(params["lr"])
            if "weight_decay" in params:
                kan_mod.KAN_WEIGHT_DECAY = float(params["weight_decay"])
            applied["kan"] = params

    return applied



# =====================================================================
# Public API
# =====================================================================
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
    ffd_columns: list[str] | None = None,
    top_k_frac: float | None = None,
    tune: bool = False,
    tune_models: list[str] | None = None,
    n_trials: int | None = None,
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
    ffd_columns : list[str], optional
        Columns requiring FFD. Defaults to empty (no FFD applied).
    top_k_frac : float, optional
        Fraction of features to keep via MDA selection.
    tune : bool
        If True, run nested hyperparameter tuning inside each CPCV split
        using Optuna TPE + Purged K-Fold CV on the training data only.
        Each split gets its own optimal parameters.
    tune_models : list[str], optional
        Which models to tune. Defaults to ["logistic", "random_forest",
        "xgboost"]. Add "lstm" and/or "kan" for full tuning (slower).
    n_trials : int, optional
        Number of Optuna trials per model per split. Overrides the
        defaults in tuning.py (60 for classical, 40 for neural).
        Lower values (e.g. 20–30) speed up tuning significantly.

    Returns
    -------
    dict
        Contains 'predictions', 'split_info', 'path_map', 'n_paths',
        'n_splits', 'models', 'n_seeds', and optionally 'tuning_results'.
    """
    pipeline_start = time.time()

    # Restore pristine module constants. Prevents one run's tuned
    # hyperparameters from leaking into a later run that does not
    # tune those models.
    _reset_module_defaults()

    # ── defaults ──────────────────────────────────────────────────────
    if models is None:
        models = list_models()
    if ffd_columns is None:
        ffd_columns = []
    if tune_models is None:
        tune_models = ["logistic", "random_forest", "xgboost"]

    print("=" * 60)
    print("CPCV Pipeline")
    print("=" * 60)
    print(f"  Models:         {models}")
    print(f"  Seeds/fold:     {n_seeds}")
    print(f"  Groups (N):     {n_groups}")
    print(f"  Test groups (k):{k}")
    print(f"  Embargo:        {embargo_pct*100:.1f}%")
    print(f"  FFD columns:    {ffd_columns}")
    print(f"  Samples:        {len(X)}")
    print(f"  Features:       {X.shape[1]}")
    print(f"  Tuning:         {'NESTED (per-split) — ' + str(tune_models) + (' [' + str(n_trials) + ' trials]' if n_trials else '') if tune else 'OFF'}")
    print("=" * 60)

    # ── map labels: {-1, +1} → {0, 1} ────────────────────────────────
    y_mapped = ((y + 1) // 2).astype(int)

    # ── splits and paths ──────────────────────────────────────────────
    splits = generate_cpcv_splits(X, t1, n_groups, k, embargo_pct)
    n_paths, path_map = build_path_matrix(n_groups, k)
    split_info = get_split_info(X, t1, n_groups, k, embargo_pct)

    # ── storage ───────────────────────────────────────────────────────
    all_predictions = {}
    all_tuning_results = {}  # per-split tuning results
    total_tasks = len(splits) * len(models) * n_seeds
    task_counter = 0

    # ── main CPCV loop ────────────────────────────────────────────────
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
        needs_selection = not all(m == "ar_logistic" for m in models)
        X_tr_proc, X_te_proc, selected, prep_info = preprocess_fold(
            X, train_idx, test_idx, y_tr, w_tr, t1_tr, ffd_columns, top_k_frac,
            skip_selection=not needs_selection,
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

        if needs_selection:
            print(
                f"  Preprocessing: {len(selected)} features selected, "
                f"train={len(X_model)} + cal={len(X_cal)}, "
                f"test={len(X_te_sel)}"
            )
        else:
            from src.cpcv.models.benchmarks import AR_LAGS
            print(
                f"  Preprocessing: FFD + scaling only (no feature selection), "
                f"train={len(X_model)} + cal={len(X_cal)}, "
                f"test={len(X_te_full)}"
            )
            print(f"  AR Logistic lags: {AR_LAGS}")

        # ── per-split hyperparameter tuning (nested) ─────────────────
        # Tune on the FULL training fold (X_tr_sel), not the 80% model
        # portion. The inner purged K-fold handles its own train/val
        # splitting. This is the correct nested CV architecture per
        # López de Prado (2018, Ch. 7).
        if tune:
            from src.cpcv.tuning import tune_all_models

            tune_start = time.time()
            print(f"\n  Tuning (Optuna TPE + Purged K-Fold) on training fold...")

            split_tune_results = tune_all_models(
                X_tr_sel, y_tr, w_tr,
                n_features=len(selected),
                models=[m for m in tune_models if m != "ar_logistic"],
                seed=0,
                verbose=True,
                n_trials=n_trials,
            )

            # apply tuned params to module-level constants
            # (KAN widths are written to kan_mod.KAN_HIDDEN / KAN_HIDDEN2,
            # which KANModel.__init__ reads at call time)
            applied = _apply_tuned_params(split_tune_results)

            tune_elapsed = time.time() - tune_start
            print(f"  Tuning done in {tune_elapsed:.1f}s")
            for m, p in applied.items():
                print(f"    {m}: {p}")

            all_tuning_results[split_idx] = split_tune_results

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
                        cal_logits = model.predict_logits(X_c)
                        cal_valid_idx = model.last_valid_indices
                        y_cal_aligned = y_cal.iloc[cal_valid_idx]
                        calibrator.fit_from_logits(cal_logits, y_cal_aligned, method="vector")
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
                    auc = float("nan")
                try:
                    ll = log_loss(y_true_aligned, cal_proba)
                except ValueError:
                    ll = float("nan")

                # store
                store_key = (model_name, split_idx, seed)
                all_predictions[store_key] = {
                    "y_true": y_true_aligned,
                    "y_pred": y_pred,
                    "cal_proba": cal_proba,
                    "f1_macro": f1,
                    "roc_auc": auc,
                    "log_loss": ll,
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
                    f"F1={f1:.3f}  AUC={auc:.3f}  LL={ll:.4f} "
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
    if tune and all_tuning_results:
        print(f"  Tuning:         nested (per-split), {len(all_tuning_results)} "
              f"splits tuned, models={tune_models}")
    print(f"{'='*60}")

    if n_failed > 0:
        logger.warning("%d model fits failed during pipeline execution.", n_failed)

    result = {
        "predictions": all_predictions,
        "split_info": split_info,
        "path_map": path_map,
        "n_paths": n_paths,
        "n_splits": len(splits),
        "models": models,
        "n_seeds": n_seeds,
    }

    if tune and all_tuning_results:
        result["tuning_results"] = all_tuning_results

    return result