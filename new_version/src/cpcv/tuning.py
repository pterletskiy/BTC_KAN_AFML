"""
11) Hyperparameter Tuning — Optuna TPE + Purged K-Fold CV
============================================================
Bayesian hyperparameter optimisation with Optuna's TPE sampler and the
Median Pruner, evaluated on a purged K-Fold (K=3, embargo=10 obs) inner CV
that respects AFML's label-overlap constraints (AFML Ch. 7).

Runs INSIDE each outer CPCV fold using only training data:
  - Inner folds: purged K-fold on the CPCV training set
  - Best params selected by lowest mean log loss across inner folds
  - The outer test fold is never seen during tuning, so DSR / PBO remain valid

Every ``tune_*()`` function returns the same shape:
  ``{"best_params": {...}, "best_log_loss": float, "results_df": DataFrame}``
"""

import copy
import logging
import random as python_random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError("Optuna is required for tuning. Install: pip install optuna")

logger = logging.getLogger(__name__)

# --- Purged K-Fold configuration -------------------------------------------
N_INNER_FOLDS = 3
PURGE_EMBARGO = 10              # embargo length in observations (matches TBL num_days=10)

# --- Optuna configuration ---------------------------------------------------
N_TRIALS_CLASSICAL = 30         # trials for Logistic, RF, XGBoost
N_TRIALS_NEURAL = 30            # trials for LSTM, KAN (more expensive per trial)
OPTUNA_SEED = 42
OPTUNA_VERBOSITY = optuna.logging.WARNING


# --- 1. Purged K-Fold Split Generator --------------------------------------
# Time-ordered contiguous-block splits with AFML label-overlap purging (when t1 is supplied)
# and a one-sided post-val embargo. Matches the outer CPCV's methodology in cv._purge_train.
def _purged_kfold_splits(X_train, y_train, w_train=None, t1_train=None,
                         n_folds=N_INNER_FOLDS, embargo=PURGE_EMBARGO):
    """Generate ``[(X_tr, y_tr, w_tr, X_val, y_val, w_val), ...]`` with label-overlap purging.

    When ``t1_train`` is supplied and ``X_train`` carries a timestamp index, training
    rows are purged using the three AFML §7.4.1 label-overlap conditions:
      1. Training row starts inside the validation window.
      2. Training label resolves inside the validation window.
      3. Training label horizon straddles the entire validation window.
    A one-sided post-val embargo of ``embargo`` rows is then applied to bar nearby
    training rows from leaking the lookahead into the val set.

    When ``t1_train`` is not supplied (or X_train has no index), the function falls
    back to symmetric positional purging of ±``embargo`` rows around the val block.
    """
    if hasattr(X_train, "values"):
        X_arr = X_train.values
        X_index = X_train.index if hasattr(X_train, "index") else None
    else:
        X_arr = np.asarray(X_train)
        X_index = None

    y_arr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
    w_arr = None
    if w_train is not None:
        w_arr = w_train.values if hasattr(w_train, "values") else np.asarray(w_train)

    use_label_overlap = t1_train is not None and X_index is not None
    if use_label_overlap:
        t1_arr = t1_train.values if hasattr(t1_train, "values") else np.asarray(t1_train)

    n = len(X_arr)
    fold_size = n // n_folds
    splits = []

    for fold_idx in range(n_folds):
        val_start = fold_idx * fold_size
        val_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n

        # Initial candidate train set: everything outside the contiguous val block.
        candidate_train = set(range(0, val_start)) | set(range(val_end, n))

        if use_label_overlap:
            # AFML §7.4.1 label-overlap purging on t1 against the val timestamps.
            t_val_start = X_index[val_start]
            t_val_end = X_index[val_end - 1]

            purged = set()
            for i in candidate_train:
                t_i_start = X_index[i]
                t_i_end = t1_arr[i]
                if pd.isna(t_i_end):
                    continue
                # Three AFML conditions — same as cv._purge_train, but for a single val block.
                if t_val_start <= t_i_start <= t_val_end:
                    purged.add(i)
                elif t_val_start <= t_i_end <= t_val_end:
                    purged.add(i)
                elif t_i_start <= t_val_start and t_val_end <= t_i_end:
                    purged.add(i)
            candidate_train -= purged

            # One-sided post-val embargo (AFML §7.4.2): bar a fixed window after the val block.
            embargo_range = set(range(val_end, min(val_end + embargo, n)))
            candidate_train -= embargo_range
        else:
            # Positional fallback when t1 is unavailable: symmetric ±embargo around val block.
            purge_start = max(0, val_start - embargo)
            purge_end = min(n, val_end + embargo)
            candidate_train -= set(range(purge_start, purge_end))

        train_indices = np.array(sorted(candidate_train), dtype=np.int64)
        val_indices = np.arange(val_start, val_end)

        if len(train_indices) == 0 or len(val_indices) == 0:
            continue

        X_tr = X_arr[train_indices]
        y_tr = y_arr[train_indices]
        w_tr = w_arr[train_indices] if w_arr is not None else None
        X_val = X_arr[val_indices]
        y_val = y_arr[val_indices]
        w_val = w_arr[val_indices] if w_arr is not None else None

        splits.append((X_tr, y_tr, w_tr, X_val, y_val, w_val))

    return splits


# --- 2. Generic split evaluator with pruning support -----------------------
# Run a model_fn across all inner splits, reporting intermediate scores for Optuna pruning.
def _evaluate_on_splits(splits, model_fn, trial=None):
    """Return ``(mean_log_loss, mean_accuracy)`` across all inner splits; prunes via ``trial.report``.

    Both metrics are sample-weighted by ``w_val`` when supplied, so the tuning
    objective matches the production training loss (also sample-weighted).
    """
    split_losses = []
    split_accs = []

    for fold_idx, (X_tr, y_tr, w_tr, X_val, y_val, w_val) in enumerate(splits):
        model = model_fn(X_tr, y_tr, w_tr)
        proba = model.predict_proba(X_val)
        ll = log_loss(y_val, proba, sample_weight=w_val)
        acc = accuracy_score(y_val, model.predict(X_val), sample_weight=w_val)

        split_losses.append(ll)
        split_accs.append(acc)

        # Mid-trial pruning: Optuna's MedianPruner can terminate clearly bad configs after each fold.
        if trial is not None:
            trial.report(np.mean(split_losses), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return np.mean(split_losses), np.mean(split_accs)


# --- 3. Logistic Regression tuner ------------------------------------------
# Search C (log-uniform) and penalty (l1 vs l2); solver auto-chosen.
def tune_logistic(X_train, y_train, w_train=None, t1_train=None,
                  seed=42, verbose=True, n_trials=None):
    """Tune Logistic Regression: ``C`` log-uniform [1e-4, 1e2], ``penalty`` ∈ {l1, l2}."""
    splits = _purged_kfold_splits(X_train, y_train, w_train, t1_train)
    _n = n_trials if n_trials is not None else N_TRIALS_CLASSICAL

    if verbose:
        logger.info(
            "  [tuning] logistic: %d purged K-fold splits, %d Optuna trials (TPE)",
            len(splits), _n,
        )

    all_results = []

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "liblinear" if penalty == "l1" else "lbfgs"

        # Build a fresh classifier per fold so refit costs are reflected in the trial timing.
        def model_fn(X_tr, y_tr, w_tr):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LogisticRegression(
                    C=C, penalty=penalty, solver=solver,
                    class_weight="balanced", max_iter=1000,
                    random_state=seed,
                )
                m.fit(X_tr, y_tr, sample_weight=w_tr)
            return m

        avg_ll, avg_acc = _evaluate_on_splits(splits, model_fn, trial)

        all_results.append({
            "C": C, "penalty": penalty,
            "accuracy": avg_acc, "log_loss": avg_ll,
        })
        return avg_ll

    optuna.logging.set_verbosity(OPTUNA_VERBOSITY)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=OPTUNA_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=_n, show_progress_bar=False)

    df = pd.DataFrame(all_results).sort_values("log_loss", ignore_index=True)
    best = study.best_params

    if verbose:
        _print_study_summary(study, df, "logistic")

    return {
        "best_params": {"C": best["C"], "penalty": best["penalty"]},
        "best_log_loss": study.best_value,
        "results_df": df,
    }


# --- 4. Random Forest tuner ------------------------------------------------
# Search depth, leaf size, n_estimators (stepped), and max_features.
def tune_random_forest(X_train, y_train, w_train=None, t1_train=None,
                       seed=42, verbose=True, n_trials=None):
    """Tune Random Forest: ``n_estimators`` ∈ [100, 250] step 50, ``max_depth`` ∈ [2, 6],
    ``min_samples_leaf`` ∈ [15, 40], ``max_features`` ∈ {sqrt, log2}."""
    splits = _purged_kfold_splits(X_train, y_train, w_train, t1_train)
    _n = n_trials if n_trials is not None else N_TRIALS_CLASSICAL

    if verbose:
        logger.info(
            "  [tuning] random_forest: %d purged K-fold splits, %d Optuna trials (TPE)",
            len(splits), _n,
        )

    all_results = []

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 250, step=50)
        max_depth = trial.suggest_int("max_depth", 2, 6)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 15, 40)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

        def model_fn(X_tr, y_tr, w_tr):
            m = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            return m

        avg_ll, avg_acc = _evaluate_on_splits(splits, model_fn, trial)

        all_results.append({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "accuracy": avg_acc, "log_loss": avg_ll,
        })
        return avg_ll

    optuna.logging.set_verbosity(OPTUNA_VERBOSITY)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=OPTUNA_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=_n, show_progress_bar=False)

    df = pd.DataFrame(all_results).sort_values("log_loss", ignore_index=True)
    best = study.best_params

    if verbose:
        _print_study_summary(study, df, "random_forest")

    return {
        "best_params": {
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "min_samples_leaf": best["min_samples_leaf"],
            "max_features": best["max_features"],
        },
        "best_log_loss": study.best_value,
        "results_df": df,
    }


# --- 5. XGBoost tuner ------------------------------------------------------
# Search depth, learning rate, child weight, subsampling, gamma, L1/L2; early stop on val.
def tune_xgboost(X_train, y_train, w_train=None, t1_train=None,
                 seed=42, verbose=True, n_trials=None):
    """Tune XGBoost across 8 parameters; n_estimators fixed at 500 with 20-round early stopping."""
    splits = _purged_kfold_splits(X_train, y_train, w_train, t1_train)
    _n = n_trials if n_trials is not None else N_TRIALS_CLASSICAL

    if verbose:
        logger.info(
            "  [tuning] xgboost: %d purged K-fold splits, %d Optuna trials (TPE)",
            len(splits), _n,
        )

    all_results = []

    def objective(trial):
        # 8-parameter search; ranges chosen to span the regularised end of the parameter space.
        max_depth = trial.suggest_int("max_depth", 1, 3)
        lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 5, 30)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)

        split_losses = []
        split_accs = []

        for fold_idx, (X_tr, y_tr, w_tr, X_val, y_val, w_val) in enumerate(splits):
            # scale_pos_weight balances class imbalance natively inside XGBoost.
            n_pos = (y_tr == 1).sum()
            n_neg = (y_tr == 0).sum()
            spw = n_neg / max(n_pos, 1)

            m = XGBClassifier(
                n_estimators=500,
                max_depth=max_depth,
                learning_rate=lr,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                scale_pos_weight=spw,
                objective="binary:logistic",
                eval_metric="logloss",
                early_stopping_rounds=20,
                random_state=seed,
            )
            # Weighted eval_set: early stopping decides on the same weighted log-loss objective
            # the training side optimises; previously XGB's eval_metric was unweighted while
            # training was weighted, producing an asymmetric stopping rule.
            fit_kwargs = dict(
                X=X_tr, y=y_tr, sample_weight=w_tr,
                eval_set=[(X_val, y_val)], verbose=False,
            )
            if w_val is not None:
                fit_kwargs["sample_weight_eval_set"] = [w_val]
            m.fit(**fit_kwargs)

            proba = m.predict_proba(X_val)
            ll = log_loss(y_val, proba, sample_weight=w_val)
            acc = accuracy_score(y_val, m.predict(X_val), sample_weight=w_val)

            split_losses.append(ll)
            split_accs.append(acc)

            # Mid-trial pruning after each fold.
            trial.report(np.mean(split_losses), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        avg_ll = np.mean(split_losses)
        avg_acc = np.mean(split_accs)

        all_results.append({
            "max_depth": max_depth,
            "learning_rate": lr,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "accuracy": avg_acc,
            "log_loss": avg_ll,
        })
        return avg_ll

    optuna.logging.set_verbosity(OPTUNA_VERBOSITY)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=OPTUNA_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=_n, show_progress_bar=False)

    df = pd.DataFrame(all_results).sort_values("log_loss", ignore_index=True)
    best = study.best_params

    if verbose:
        _print_study_summary(study, df, "xgboost")

    return {
        "best_params": {
            "max_depth": best["max_depth"],
            "learning_rate": best["learning_rate"],
            "min_child_weight": best["min_child_weight"],
            "subsample": best["subsample"],
            "colsample_bytree": best.get("colsample_bytree", 0.8),
            "gamma": best.get("gamma", 0.0),
            "reg_alpha": best.get("reg_alpha", 0.0),
            "reg_lambda": best.get("reg_lambda", 1.0),
        },
        "best_log_loss": study.best_value,
        "results_df": df,
    }


# --- 6. LSTM tuner ---------------------------------------------------------
# Locked narrow ranges from sensitivity testing; see project_structure.md for the methodology trail.
def tune_lstm(X_train, y_train, w_train=None, t1_train=None, n_features=None,
              seed=42, verbose=True, n_trials=None):
    """Tune LSTM: ``hidden_size`` ∈ {16, 32}, ``dropout`` ∈ [0.1, 0.5], ``lr`` log-uniform [1e-4, 5e-2].

    ``num_layers`` is hardcoded at 1 (multi-layer overfits on ~1,250 events). Window=14
    and batch_size=64 are fixed. Tuning uses epochs=50 / patience=7 to bound per-trial
    cost; the production fit re-runs at the full epochs=100 / patience=15 budget.

    Search ranges are locked from a wider-search sensitivity test that produced worse
    out-of-sample Sharpe and was rejected.
    """
    from torch.utils.data import TensorDataset, DataLoader
    from src.cpcv.models.lstm_model import LSTMClassifier, create_sequences

    splits = _purged_kfold_splits(X_train, y_train, w_train, t1_train)

    if n_features is None:
        n_features = splits[0][0].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match production lstm_model.py constants, but use a shorter early-stop budget for tuning.
    window = 14
    batch_size = 64
    epochs = 50
    patience = 7

    # Pre-build all sliding-window sequences once so each Optuna trial only repeats the training step.
    # w_val is sequenced too so the val loss can be sample-weighted symmetrically with training.
    seq_splits = []
    for X_tr, y_tr, w_tr, X_val, y_val, w_val in splits:
        X_seq, y_seq, w_seq, _ = create_sequences(X_tr, y_tr, w_tr, window=window)
        X_val_seq, y_val_seq, w_val_seq, _ = create_sequences(
            X_val, y_val, w_val, window=window,
        )

        if len(X_seq) == 0 or len(X_val_seq) == 0:
            continue

        seq_splits.append({
            "X_seq_t": torch.tensor(X_seq, dtype=torch.float32).to(device),
            "y_seq_t": torch.tensor(y_seq, dtype=torch.long).to(device),
            "w_seq_t": (
                torch.tensor(w_seq, dtype=torch.float32).to(device)
                if w_seq is not None
                else torch.ones(len(y_seq), dtype=torch.float32).to(device)
            ),
            "X_val_t": torch.tensor(X_val_seq, dtype=torch.float32).to(device),
            "y_val_t": torch.tensor(y_val_seq, dtype=torch.long).to(device),
            "w_val_t": (
                torch.tensor(w_val_seq, dtype=torch.float32).to(device)
                if w_val_seq is not None
                else torch.ones(len(y_val_seq), dtype=torch.float32).to(device)
            ),
            "y_val_np": y_val_seq,
            "y_seq_np": y_seq,
            "w_val_np": w_val_seq,
        })

    if not seq_splits:
        return {"best_params": {}, "best_log_loss": np.nan, "results_df": pd.DataFrame()}

    _n = n_trials if n_trials is not None else N_TRIALS_NEURAL

    if verbose:
        logger.info(
            "  [tuning] lstm: %d purged K-fold splits, %d Optuna trials (TPE)",
            len(seq_splits), _n,
        )

    all_results = []

    def objective(trial):
        # Locked narrow ranges; num_layers hardcoded so Optuna spends trials on dropout / lr instead.
        hidden_size = trial.suggest_int("hidden_size", 16, 32, step=16)
        num_layers = 1
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)

        split_losses = []

        for fold_idx, s in enumerate(seq_splits):
            try:
                # Seed every RNG so an identical (hidden, dropout, lr) reproduces the same val loss.
                torch.manual_seed(seed)
                np.random.seed(seed)
                python_random.seed(seed)
                # cuDNN determinism: pin the GPU LSTM kernel to its deterministic path so the
                # tuner is reproducible across runs on the same hardware.
                if torch.cuda.is_available():
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

                # Class-frequency-inverse weighting inside the loss.
                cc = np.bincount(s["y_seq_np"], minlength=2)
                cw = 1.0 / (cc + 1e-8)
                cw = cw / cw.sum() * 2
                cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

                net = LSTMClassifier(
                    n_features=n_features, n_classes=2,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                ).to(device)

                # Same loss + optimiser stack as the production training loop.
                # Both train and val use reduction="none" so the per-sample CE is multiplied
                # by AFML sample weights before averaging — symmetric weighting on both sides.
                criterion = nn.CrossEntropyLoss(
                    weight=cw_t, reduction="none",
                    label_smoothing=0.1,
                )
                criterion_val = nn.CrossEntropyLoss(
                    weight=cw_t, reduction="none",
                    label_smoothing=0.1,
                )
                optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=25, T_mult=2, eta_min=1e-5,
                )

                train_ds = TensorDataset(
                    s["X_seq_t"], s["y_seq_t"], s["w_seq_t"]
                )
                train_dl = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True
                )

                best_val_loss = float("inf")
                best_state = None
                patience_counter = 0

                # Training loop with early stopping on weighted val loss.
                for epoch in range(epochs):
                    net.train()
                    for X_b, y_b, w_b in train_dl:
                        optimizer.zero_grad()
                        logits = net(X_b)
                        per_sample = criterion(logits, y_b)
                        loss = (per_sample * w_b).mean()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        optimizer.step()

                    scheduler.step()

                    with torch.no_grad():
                        val_logits = net(s["X_val_t"])
                        val_per_sample = criterion_val(val_logits, s["y_val_t"])
                        val_loss = (val_per_sample * s["w_val_t"]).mean().item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(net.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        break

                if best_state is not None:
                    net.load_state_dict(best_state)

                # Final scoring on val using the best-weights snapshot.
                net.eval()
                with torch.no_grad():
                    logits = net(s["X_val_t"])
                    proba = torch.softmax(logits, dim=1).cpu().numpy()

                # Weighted log_loss matches the weighted val loss used for early stopping.
                ll = log_loss(s["y_val_np"], proba, sample_weight=s["w_val_np"])
                split_losses.append(ll)

                # Free GPU memory between folds to avoid OOM on long tuning runs.
                del net, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                trial.report(np.mean(split_losses), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                # Per-fold failure should not kill the trial; record and continue.
                logger.debug("LSTM tuning fold failed: %s", e)
                continue

        if not split_losses:
            return float("inf")

        avg_ll = np.mean(split_losses)
        all_results.append({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": lr,
            "log_loss": avg_ll,
        })
        return avg_ll

    optuna.logging.set_verbosity(OPTUNA_VERBOSITY)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=OPTUNA_SEED),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=_n, show_progress_bar=False)

    if not all_results:
        return {"best_params": {}, "best_log_loss": np.nan, "results_df": pd.DataFrame()}

    df = pd.DataFrame(all_results).sort_values("log_loss", ignore_index=True)
    best = study.best_params

    if verbose:
        _print_study_summary(study, df, "lstm")

    return {
        "best_params": {
            "hidden_size": best["hidden_size"],
            "num_layers": 1,
            "dropout": best["dropout"],
            "learning_rate": best["learning_rate"],
        },
        "best_log_loss": study.best_value,
        "results_df": df,
    }


# --- 7. KAN tuner (efficient-kan + AdamW) -----------------------------------
# Locked narrow ranges from sensitivity testing; see project_structure.md for the methodology trail.
def tune_kan(X_train, y_train, w_train=None, t1_train=None, n_features=None,
             seed=42, verbose=True, n_trials=None):
    """Tune KAN: ``width1`` ∈ [2, 6], ``grid`` ∈ {3, 5}, ``lr`` log-uniform [5e-4, 5e-2],
    ``weight_decay`` log-uniform [1e-5, 5e-3].

    ``width2`` is hardcoded at 0 (single hidden layer keeps the symbolic formula tractable;
    nested compositions stress sympy and lose interpretability). B-spline order ``k=3``,
    epochs=200, patience=20, full-batch training, tanh input normalisation.

    Search ranges are locked from a wider-search sensitivity test that produced worse
    out-of-sample Sharpe and was rejected.
    """
    from efficient_kan import KAN

    splits = _purged_kfold_splits(X_train, y_train, w_train, t1_train)

    if n_features is None:
        n_features = splits[0][0].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed training-loop hyperparameters; only the four searched params vary across trials.
    k = 3
    epochs = 200
    patience = 20

    _n = n_trials if n_trials is not None else N_TRIALS_NEURAL

    if verbose:
        logger.info(
            "  [tuning] kan (efficient-kan + AdamW): %d purged K-fold splits, %d Optuna trials (TPE)",
            len(splits), _n,
        )

    all_results = []

    def objective(trial):
        # Locked ranges; width2 hardcoded to keep symbolic extraction tractable.
        width1 = trial.suggest_int("width1", 2, 6)
        width2 = 0
        lr = trial.suggest_float("lr", 5e-4, 5e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
        grid = trial.suggest_categorical("grid", [3, 5])

        # Single hidden layer architecture: [features, width1, 2].
        widths = [n_features, width1, 2]

        split_losses = []

        for fold_idx, (X_tr, y_tr, w_tr, X_val, y_val, w_val) in enumerate(splits):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)
                python_random.seed(seed)
                # cuDNN determinism inside per-fold seed block; KAN itself doesn't use cuDNN
                # but the surrounding PyTorch ops do, so the flag set is the same as for LSTM.
                if torch.cuda.is_available():
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

                X_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
                y_t = torch.tensor(y_tr, dtype=torch.long).to(device)
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

                if w_tr is not None:
                    w_t = torch.tensor(w_tr, dtype=torch.float32).to(device)
                else:
                    w_t = torch.ones(len(y_t), dtype=torch.float32).to(device)
                # w_val tensor for the weighted val loss; falls back to uniform ones.
                if w_val is not None:
                    w_val_t = torch.tensor(w_val, dtype=torch.float32).to(device)
                else:
                    w_val_t = torch.ones(len(y_val_t), dtype=torch.float32).to(device)

                # Tanh normalisation: matches the production KAN training stack.
                input_mean = X_t.mean(dim=0)
                input_std = X_t.std(dim=0) + 1e-8
                X_t = torch.tanh((X_t - input_mean) / input_std)
                X_val_t = torch.tanh((X_val_t - input_mean) / input_std)

                # Class weights inside the cross-entropy loss.
                cc = np.bincount(y_tr, minlength=2)
                cw = 1.0 / (cc + 1e-8)
                cw = cw / cw.sum() * 2
                cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

                criterion_train = nn.CrossEntropyLoss(
                    weight=cw_t, reduction="none",
                    label_smoothing=0.1,
                )
                # Same reduction=none on val so val loss can be weighted by w_val_t.
                criterion_val = nn.CrossEntropyLoss(
                    weight=cw_t, reduction="none",
                    label_smoothing=0.1,
                )

                model = KAN(
                    layers_hidden=widths,
                    grid_size=grid,
                    spline_order=k,
                    grid_range=[-1, 1],
                ).to(device)

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=wd,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=30, T_mult=2, eta_min=1e-5,
                )

                best_val_loss = float("inf")
                best_state = None
                patience_counter = 0

                # Full-batch training step + per-epoch weighted val score for early stopping.
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()

                    logits = model(X_t)
                    per_sample = criterion_train(logits, y_t)
                    loss = (per_sample * w_t).mean()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    model.eval()
                    with torch.no_grad():
                        val_logits = model(X_val_t)
                        val_per_sample = criterion_val(val_logits, y_val_t)
                        val_loss = (val_per_sample * w_val_t).mean().item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        break

                if best_state is not None:
                    model.load_state_dict(best_state)

                model.eval()
                with torch.no_grad():
                    logits = model(X_val_t)
                    proba = torch.softmax(logits, dim=1).cpu().numpy()

                # Weighted log_loss matches the weighted val loss used for early stopping.
                ll = log_loss(y_val, proba, sample_weight=w_val)
                split_losses.append(ll)

                # Free GPU memory between folds.
                del model, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                trial.report(np.mean(split_losses), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.debug("KAN tuning fold failed: %s", e)
                continue

        if not split_losses:
            return float("inf")

        avg_ll = np.mean(split_losses)

        all_results.append({
            "width1": width1,
            "width2": width2,
            "architecture": "x".join(str(w) for w in widths),
            "grid": grid,
            "lr": lr,
            "weight_decay": wd,
            "log_loss": avg_ll,
        })
        return avg_ll

    optuna.logging.set_verbosity(OPTUNA_VERBOSITY)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=OPTUNA_SEED),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=_n, show_progress_bar=False)

    if not all_results:
        return {"best_params": {}, "best_log_loss": np.nan, "results_df": pd.DataFrame()}

    df = pd.DataFrame(all_results).sort_values("log_loss", ignore_index=True)
    best = study.best_params

    if verbose:
        _print_study_summary(study, df, "kan")

    return {
        "best_params": {
            "width1": best["width1"],
            "width2": 0,
            "grid": best["grid"],
            "lr": best["lr"],
            "weight_decay": best["weight_decay"],
        },
        "best_log_loss": study.best_value,
        "results_df": df,
    }


# --- 8. Orchestrator -------------------------------------------------------
# Dispatch table: run the per-model tuner for each requested model, return a combined results dict.
def tune_all_models(
    X_train, y_train, w_train=None, t1_train=None, n_features=None,
    models=None, seed=42, verbose=True, n_trials=None,
):
    """Run Optuna TPE tuning for every requested model and return ``{model: result_dict}``.

    ``models`` defaults to ``["logistic", "random_forest", "xgboost"]``; pass
    ``["lstm"]`` or ``["kan"]`` to add neural tuning. ``t1_train`` is the label
    end-time series used by the inner CV for AFML §7.4.1 label-overlap purging;
    when None, the inner CV falls back to symmetric positional purging.
    """
    if models is None:
        models = ["logistic", "random_forest", "xgboost"]

    if n_features is None:
        n_features = (
            X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])
        )

    # Lambda-wrapped dispatch keeps the call site flat.
    dispatch = {
        "logistic": lambda: tune_logistic(
            X_train, y_train, w_train, t1_train, seed, verbose, n_trials
        ),
        "random_forest": lambda: tune_random_forest(
            X_train, y_train, w_train, t1_train, seed, verbose, n_trials
        ),
        "xgboost": lambda: tune_xgboost(
            X_train, y_train, w_train, t1_train, seed, verbose, n_trials
        ),
        "lstm": lambda: tune_lstm(
            X_train, y_train, w_train, t1_train, n_features, seed, verbose, n_trials
        ),
        "kan": lambda: tune_kan(
            X_train, y_train, w_train, t1_train, n_features, seed, verbose, n_trials
        ),
    }

    all_results = {}
    total_start = time.time()

    if verbose:
        logger.info(
            "Tuning (%d model%s) — Optuna TPE + Purged K-Fold "
            "(inner: %d folds, embargo=%d obs)",
            len(models), 's' if len(models) != 1 else '',
            N_INNER_FOLDS, PURGE_EMBARGO,
        )

    for model_name in models:
        if model_name not in dispatch:
            logger.warning("No tuning function for '%s'. Skipping.", model_name)
            continue

        t0 = time.time()
        result = dispatch[model_name]()
        elapsed = time.time() - t0

        all_results[model_name] = result

        if verbose and result["best_params"]:
            logger.info(
                "  [tuning] %s done in %.1fs — best log_loss=%.4f",
                model_name, elapsed, result['best_log_loss'],
            )

    elapsed_total = time.time() - total_start

    if verbose:
        logger.info("Tuning total: %.1fs", elapsed_total)

    return all_results


# --- 9. Helpers -------------------------------------------------------------
# One-line study summary; the full per-trial table lives in the returned results_df.
def _print_study_summary(study, df, model_name):
    """Log a single-line summary: best params + completed/pruned trial counts."""
    n_complete = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])

    logger.info(
        "    %s: %d completed, %d pruned (of %d total) — "
        "best log_loss=%.4f, params=%s",
        model_name, n_complete, n_pruned, len(study.trials),
        study.best_value, study.best_params,
    )