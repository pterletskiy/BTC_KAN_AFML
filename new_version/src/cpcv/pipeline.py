"""
13) Pipeline
================
Master orchestrator that ties together split generation, preprocessing, model
training, calibration, and prediction storage across all splits × models × seeds.

Nested tuning (optional): for each outer CPCV split, Optuna TPE + purged K-Fold
CV tunes hyperparameters on the training data only. Each split gets its own
optimal parameters, which keeps DSR and PBO valid (AFML Ch. 7 & 12).

Single entry point: ``run_cpcv_pipeline()``.
"""

import logging
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, log_loss, roc_auc_score
from tqdm import tqdm

from src.cpcv.cv import build_path_matrix, generate_cpcv_splits, get_split_info
from src.cpcv.preprocessing import preprocess_fold
from src.cpcv.calibration import Calibrator
from src.cpcv.models import create_model, list_models

logger = logging.getLogger(__name__)


# --- 1. Warning capture for clean tqdm output ------------------------------
# WARNING-level records break the tqdm bar by writing above it; buffer them and replay after the run.
class _WarningBuffer(logging.Handler):
    """Capture WARNING+ log records into a list for deferred replay after the tqdm bar finishes."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []
        self.setFormatter(logging.Formatter(
            "%(name)s  %(levelname)s  %(message)s",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


# --- 2. Pristine module-default capture and reset --------------------------
# ``_apply_tuned_params`` mutates module-level constants on each model file;
# without a reset, a second run inherits the first run's tuned values.
_PRISTINE_DEFAULTS: dict[str, dict[str, object]] = {}

# Per-module constants that ``_apply_tuned_params`` writes to.
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


# Resolve a tracked-module name string to the live module object for read/write of constants.
def _model_module(name: str):
    """Return the live module object for one of the four tracked model files."""
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


# Snapshot import-time defaults on first call, restore them on every subsequent call.
def _reset_module_defaults() -> None:
    """Restore tracked module-level constants to their pristine import-time values.

    First call snapshots the current values (which are pristine, since this runs before
    any ``_apply_tuned_params`` mutation). Later calls restore the snapshot, undoing any
    tuning-driven mutation from a prior pipeline invocation.
    """
    first_call = not _PRISTINE_DEFAULTS

    for mod_name, attr_names in _TRACKED_CONSTANTS.items():
        mod = _model_module(mod_name)

        if first_call:
            # Capture once: subsequent restores point back to these values.
            _PRISTINE_DEFAULTS[mod_name] = {
                a: getattr(mod, a) for a in attr_names if hasattr(mod, a)
            }
        else:
            for a, v in _PRISTINE_DEFAULTS[mod_name].items():
                setattr(mod, a, v)

    if not first_call:
        logger.debug("Reset tracked module-level constants to pristine defaults.")


# --- 3. Apply tuned hyperparameters to module-level constants --------------
# Write the Optuna best_params into each model file's module-level constants so model __init__s pick them up.
def _apply_tuned_params(tuning_results: dict) -> dict:
    """Override module-level constants with the best params from Optuna; return a per-model summary."""
    applied = {}

    # Logistic Regression: C and penalty live in benchmarks.py.
    if "logistic" in tuning_results:
        params = tuning_results["logistic"].get("best_params", {})
        if params:
            import src.cpcv.models.benchmarks as bench_mod
            if "C" in params:
                bench_mod.LOGISTIC_C = params["C"]
            if "penalty" in params:
                bench_mod.LOGISTIC_PENALTY = params["penalty"]
            applied["logistic"] = params

    # Random Forest: 4 constants in tree_models.py.
    if "random_forest" in tuning_results:
        params = tuning_results["random_forest"].get("best_params", {})
        if params:
            import src.cpcv.models.tree_models as tree_mod
            if "n_estimators" in params:
                tree_mod.RF_N_ESTIMATORS = params["n_estimators"]
            if "max_depth" in params:
                v = params["max_depth"]
                # NaN-safe coercion so None / NaN values fall through cleanly.
                tree_mod.RF_MAX_DEPTH = int(v) if v is not None and v == v else None
            if "min_samples_leaf" in params:
                tree_mod.RF_MIN_SAMPLES_LEAF = params["min_samples_leaf"]
            if "max_features" in params:
                tree_mod.RF_MAX_FEATURES = params["max_features"]
            applied["random_forest"] = params

    # XGBoost: 8 constants in tree_models.py.
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

    # LSTM: 4 constants in lstm_model.py.
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

    # KAN: 5 constants in kan_model.py; KANModel.__init__ reads them at call time.
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


# --- 4. Public API: run_cpcv_pipeline --------------------------------------
# The single entry point used by the notebook: builds splits, loops over models × seeds, returns predictions.
def run_cpcv_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    t1: pd.Series,
    bins_ret: pd.Series,
    n_groups: int = 8,
    k: int = 2,
    embargo_pct: float = 0.01,
    n_seeds: int = 3,
    models: list[str] | None = None,
    ffd_columns: list[str] | None = None,
    top_k_frac: float | None = None,
    tune: bool = False,
    tune_models: list[str] | None = None,
    n_trials: int | None = None,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    path_map: dict | None = None,
    n_paths: int | None = None,
    split_info: dict | None = None,
) -> dict:
    """Run the full CPCV evaluation across all models × splits × seeds.

    Pre-computed ``splits``, ``path_map``, ``n_paths``, and ``split_info`` from
    the notebook's CV cell are preferred over recomputing internally: it avoids
    duplicate log lines and ensures the configuration printed by the CV summary
    matches what the pipeline trains on. ``ffd_columns`` defaults to empty,
    meaning no fractional differentiation unless requested.

    When ``tune=True``, Optuna runs INSIDE each outer split on the training
    fold only, and the resulting best params are written into the model modules'
    constants via ``_apply_tuned_params``. ``tune_models`` defaults to the three
    classical models; add ``"lstm"`` or ``"kan"`` to enable neural tuning.

    Returns a dict with ``predictions``, ``split_info``, ``path_map``, ``n_paths``,
    ``n_splits``, ``models``, ``n_seeds``, and optionally ``tuning_results``.
    """
    pipeline_start = time.time()

    # Bulletproof warning suppression: notebook-level filters can be overridden mid-run by sklearn
    # internals, so we reset the filter list and install a single catch-all rule here.
    warnings.simplefilter("ignore")

    # Install a buffer that captures WARNING+ records during the run; replay after the bar finishes.
    # The FileHandler set up in the imports cell is left alone, so the log file records everything live.
    warning_buffer = _WarningBuffer()
    root_logger = logging.getLogger()
    root_logger.addHandler(warning_buffer)
    suppressed_handlers: list[tuple[logging.Handler, int]] = []
    for h in list(root_logger.handlers):
        if h is warning_buffer:
            continue
        if isinstance(h, logging.FileHandler):
            continue  # leave the file handler alone
        if isinstance(h, logging.StreamHandler):
            suppressed_handlers.append((h, h.level))
            h.setLevel(logging.ERROR)

    # Restore pristine module constants so a previous run's tuned values do not leak into this run.
    _reset_module_defaults()

    # Resolve defaults for the optional list arguments.
    if models is None:
        models = list_models()
    if ffd_columns is None:
        ffd_columns = []
    if tune_models is None:
        tune_models = ["logistic", "random_forest", "xgboost"]

    # Use precomputed splits/path matrix from the notebook when provided; otherwise compute internally.
    if splits is None:
        splits = generate_cpcv_splits(X, t1, n_groups, k, embargo_pct)
    if path_map is None or n_paths is None:
        n_paths, path_map = build_path_matrix(n_groups, k)
    if split_info is None:
        split_info = get_split_info(
            X, t1, n_groups=n_groups, k=k, embargo_pct=embargo_pct,
            splits=splits, path_map=path_map, n_paths=n_paths,
            print_summary=False,
        )
    # If splits were passed in, re-read N / k / embargo_pct from split_info so the header below
    # reports what the pipeline actually trains on rather than the function defaults.
    n_groups = split_info["n_groups"]
    k = split_info["k"]
    embargo_pct = split_info["embargo_pct"]

    logger.info("=" * 60)
    logger.info("CPCV Pipeline")
    logger.info("=" * 60)
    logger.info("  Models:         %s", models)
    logger.info("  Seeds/fold:     %d", n_seeds)
    logger.info("  Groups (N):     %d", n_groups)
    logger.info("  Test groups (k):%d", k)
    logger.info("  Embargo:        %.1f%%", embargo_pct * 100)
    logger.info("  FFD columns:    %s", ffd_columns)
    logger.info("  Samples:        %d", len(X))
    logger.info("  Features:       %d", X.shape[1])
    logger.info("  Splits:         %d, Paths: %d", len(splits), n_paths)
    logger.info(
        "  Tuning:         %s",
        ('NESTED (per-split) — ' + str(tune_models)
         + (' [' + str(n_trials) + ' trials]' if n_trials else ''))
        if tune else 'OFF',
    )
    logger.info("=" * 60)

    # Map labels from {-1, +1} to {0, 1} for sklearn / PyTorch compatibility.
    y_mapped = ((y + 1) // 2).astype(int)

    # Result accumulator: keyed by (model_name, split_idx, seed).
    all_predictions = {}
    all_tuning_results = {}
    total_tasks = len(splits) * len(models) * n_seeds
    task_counter = 0

    # Outer CPCV loop. Per-split detail is routed through Python's logging to the FileHandler;
    # the notebook cell shows only this self-updating tqdm bar plus a final summary line, which
    # keeps the saved notebook small enough to avoid the "array buffer allocation failed" error.
    pbar_label = (
        f"CPCV ({', '.join(models)})" if len(models) <= 3
        else f"CPCV ({len(models)} models)"
    )
    pbar = tqdm(
        enumerate(splits),
        total=len(splits),
        desc=pbar_label,
        unit="split",
    )
    # Track the most recent task's metrics so the tqdm postfix reflects what's happening right now.
    last_f1 = float("nan")
    last_auc = float("nan")
    for split_idx, (train_idx, test_idx) in pbar:
        split_start = time.time()
        pbar.set_postfix_str(
            f"split {split_idx + 1}/{len(splits)} "
            f"| last F1={last_f1:.3f} AUC={last_auc:.3f}"
        )

        # Extract this fold's labels, weights, barrier times, and barrier-touch returns.
        y_tr = y_mapped.iloc[train_idx]
        y_te = y_mapped.iloc[test_idx]
        w_tr = w.iloc[train_idx]
        t1_tr = t1.iloc[train_idx]
        ret_te = bins_ret.iloc[test_idx]

        # Preprocessing (FFD → scaling → MDA selection) shared across all models this fold.
        needs_selection = not all(m == "ar_logistic" for m in models)
        X_tr_proc, X_te_proc, selected, prep_info = preprocess_fold(
            X, train_idx, test_idx, y_tr, w_tr, t1_tr, ffd_columns, top_k_frac,
            skip_selection=not needs_selection,
            split_idx=split_idx,
            n_splits=len(splits),
        )

        # FFD may drop the lookback-head NaN rows; re-align y / w / t1 to the surviving training index.
        y_tr = y_tr.loc[X_tr_proc.index]
        w_tr = w_tr.loc[X_tr_proc.index]
        t1_tr = t1_tr.loc[X_tr_proc.index]

        # Keep the pre-selection matrix so AR Logistic can grab its lag columns by name regardless of MDA.
        X_tr_full = X_tr_proc.copy()
        X_te_full = X_te_proc.copy()

        # MDA-selected feature subset for non-AR models.
        X_tr_sel = X_tr_proc[selected]
        X_te_sel = X_te_proc[selected]

        # Calibration split: 80% train / 20% calibration. The calibration slice is held out from
        # model fitting and consumed only by the Calibrator (Platt or vector scaling).
        cal_boundary = int(len(X_tr_sel) * 0.8)

        X_model = X_tr_sel.iloc[:cal_boundary]
        X_cal = X_tr_sel.iloc[cal_boundary:]
        X_model_full = X_tr_full.iloc[:cal_boundary]
        X_cal_full = X_tr_full.iloc[cal_boundary:]
        y_model = y_tr.iloc[:cal_boundary]
        y_cal = y_tr.iloc[cal_boundary:]
        w_model = w_tr.iloc[:cal_boundary]

        if needs_selection:
            logger.info(
                "  Preprocessing: %d features selected, "
                "train=%d + cal=%d, test=%d",
                len(selected), len(X_model), len(X_cal), len(X_te_sel),
            )
        else:
            from src.cpcv.models.benchmarks import AR_LAGS
            logger.info(
                "  Preprocessing: FFD + scaling only (no feature selection), "
                "train=%d + cal=%d, test=%d",
                len(X_model), len(X_cal), len(X_te_full),
            )
            logger.info("  AR Logistic lags: %s", AR_LAGS)

        # Per-split nested tuning. Runs on the FULL training fold (X_tr_sel), not the 80% model
        # portion, because the inner purged K-fold handles its own train/val split. This is the
        # correct nested-CV architecture per AFML Ch. 7.
        if tune:
            from src.cpcv.tuning import tune_all_models

            tune_start = time.time()

            split_tune_results = tune_all_models(
                X_tr_sel, y_tr, w_tr,
                n_features=len(selected),
                models=[m for m in tune_models if m != "ar_logistic"],
                seed=0,
                verbose=True,
                n_trials=n_trials,
            )

            # Write the tuned values into the model modules so subsequent model creation picks them up.
            # KAN widths flow through KAN_HIDDEN / KAN_HIDDEN2, which KANModel.__init__ reads at call time.
            applied = _apply_tuned_params(split_tune_results)

            _ = time.time() - tune_start
            for m, p in applied.items():
                logger.info("    applied %s: %s", m, p)

            all_tuning_results[split_idx] = split_tune_results

        # Inner loop over models × seeds. Each task builds a fresh model, fits, calibrates, predicts.
        for model_name in models:
            for seed in range(n_seeds):
                task_counter += 1
                task_start = time.time()

                # AR Logistic takes the pre-selection matrix (needs lag columns); others take MDA subset.
                if model_name == "ar_logistic":
                    n_feat = X_tr_full.shape[1]
                else:
                    n_feat = len(selected)

                model = create_model(
                    model_name, n_features=n_feat, seed=seed
                )

                # Route the correct X tensors to the model depending on its category.
                if model_name == "ar_logistic":
                    X_fit = X_model_full
                    X_c = X_cal_full
                    X_predict = X_te_full
                else:
                    X_fit = X_model
                    X_c = X_cal
                    X_predict = X_te_sel

                # Training: any per-task failure is logged but does not abort the run.
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

                # Calibration. LSTM windowing produces logits offset by ``window-1`` from the calibration
                # set head, so we go through fit_from_logits with the LSTM's stored valid_indices to align.
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
                    # Fallback path: a failed calibration falls back to uncalibrated probabilities below.
                    logger.warning(
                        "Calibration failed for %s (seed=%d, split=%d): %s. "
                        "Using uncalibrated probabilities.",
                        model_name, seed, split_idx, e,
                    )
                    calibrator = None

                # Prediction on the test fold, calibrated if the calibrator was successfully fitted.
                raw_logits = model.predict_logits(X_predict)

                if calibrator is not None:
                    cal_proba = calibrator.calibrate(raw_logits)
                else:
                    cal_proba = model.predict_proba(X_predict)

                y_pred = np.argmax(cal_proba, axis=1)

                # LSTM offset handling: align the predicted timestamps to the (window-1)-shifted index.
                if model_name == "lstm" and hasattr(model, "last_valid_indices"):
                    valid_idx = model.last_valid_indices
                    test_timestamps = X_te_proc.index[valid_idx]
                    y_true_aligned = y_te.reindex(test_timestamps).values
                    ret_aligned = ret_te.reindex(test_timestamps).values
                else:
                    test_timestamps = X_te_proc.index
                    y_true_aligned = y_te.reindex(test_timestamps).values
                    ret_aligned = ret_te.reindex(test_timestamps).values

                # Per-task evaluation metrics; wrap AUC and log-loss in try/except since they can fail
                # on single-class predictions or degenerate probability vectors.
                f1 = f1_score(y_true_aligned, y_pred, average="macro")
                try:
                    auc = roc_auc_score(y_true_aligned, cal_proba[:, 1])
                except ValueError:
                    auc = float("nan")
                try:
                    ll = log_loss(y_true_aligned, cal_proba)
                except ValueError:
                    ll = float("nan")

                # Store everything downstream evaluation needs: predictions, probabilities, metrics, metadata.
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
                logger.info(
                    "  [%3d/%d] %20s (seed=%d) F1=%.3f  AUC=%.3f  LL=%.4f (%.1fs)",
                    task_counter, total_tasks, model_name, seed,
                    f1, auc, ll, elapsed,
                )
                # Update postfix stats so the tqdm bar reflects what just finished.
                last_f1 = f1
                last_auc = auc

        # Per-split elapsed is captured by tqdm's it/s + ETA display; no explicit per-split log line.
        _ = time.time() - split_start

    # --- Run-level summary --------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    n_successful = len(all_predictions)
    n_failed = total_tasks - n_successful

    # Detailed summary lives in the log file; the cell shows a single completion notice.
    logger.info("=" * 60)
    logger.info("CPCV Pipeline Complete")
    logger.info("=" * 60)
    logger.info("  Total time:     %.1fs (%.1fm)", total_elapsed, total_elapsed / 60)
    logger.info("  Tasks:          %d succeeded, %d failed", n_successful, n_failed)
    logger.info("  Results keys:   %d (model, split, seed) entries", n_successful)
    logger.info("  Backtest paths: %d", n_paths)
    if tune and all_tuning_results:
        logger.info(
            "  Tuning:         nested (per-split), %d splits tuned, models=%s",
            len(all_tuning_results), tune_models,
        )
    logger.info("=" * 60)

    # One concise visible line in the cell.
    summary_msg = (
        f"\n✅ CPCV complete: {n_successful}/{total_tasks} tasks succeeded "
        f"in {total_elapsed/60:.1f}m."
    )
    if n_failed:
        summary_msg += f" ({n_failed} failed)"
    print(summary_msg)

    if n_failed > 0:
        logger.warning("%d model fits failed during pipeline execution.", n_failed)

    # Replay deferred WARNING-level messages AFTER the completion notice, so the tqdm bar stayed
    # clean during the run and warnings still surface at the end.
    if warning_buffer.records:
        print(
            f"\n⚠ {len(warning_buffer.records)} "
            f"warning{'s' if len(warning_buffer.records) != 1 else ''} "
            f"during pipeline run:"
        )
        for line in warning_buffer.records:
            print(f"   {line}")

    # Restore the original console handler levels and remove the warning buffer so subsequent
    # notebook code sees the pre-pipeline logger configuration.
    for h, level in suppressed_handlers:
        h.setLevel(level)
    root_logger.removeHandler(warning_buffer)

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