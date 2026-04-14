"""
10.2) Hyperparameter Tuning — Optuna TPE + Purged K-Fold CV
============================================================
Bayesian hyperparameter optimization using Optuna's Tree-structured
Parzen Estimator (TPE) with Median Pruner and Purged K-Fold CV.

Inner CV: Purged K-Fold (K=5) with an embargo gap of PURGE_EMBARGO
observations between training and validation folds to prevent label
leakage from overlapping Triple Barrier labels.

Optimization: Optuna TPE explores the hyperparameter space adaptively,
concentrating trials in promising regions. The Median Pruner terminates
underperforming trials after evaluating each fold, cutting computation
on clearly bad configurations.

Each tune_*() function returns the same interface as before:
  {"best_params": {...}, "best_log_loss": float, "results_df": DataFrame}

This runs INSIDE each outer CPCV fold using only training data:
  - Inner folds: purged K-fold on the CPCV training set
  - Best params selected by lowest mean log loss across inner folds
  - Test fold is never seen during tuning (DSR/PBO remain valid)

References:
  - Bergstra et al. (2011), Algorithms for Hyper-Parameter Optimization
  - López de Prado (2018), AFML Ch. 7 (Purged Cross-Validation)
  - Akiba et al. (2019), Optuna: A Next-generation Hyperparameter
    Optimization Framework
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
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError(
        "Optuna is required for tuning. Install: pip install optuna"
    )

logger = logging.getLogger(__name__)

# Purged K-Fold configuration
N_INNER_FOLDS = 3               # number of inner CV folds
PURGE_EMBARGO = 10              # observations to purge between train/val
                                # (matches TBL num_days=10)

# Optuna configuration
N_TRIALS_CLASSICAL = 60         # trials for Logistic, RF, XGBoost
N_TRIALS_NEURAL = 40            # trials for LSTM, KAN (more expensive)
OPTUNA_SEED = 42
OPTUNA_VERBOSITY = optuna.logging.WARNING


# =====================================================================
# Purged K-Fold Split Generator
# =====================================================================
def _purged_kfold_splits(X_train, y_train, w_train=None,
                         n_folds=N_INNER_FOLDS, embargo=PURGE_EMBARGO):
    """Generate time-ordered K-Fold splits with purging and embargo.

    Each fold is a contiguous time block. The training set for each
    fold excludes observations within `embargo` positions of the
    validation fold boundaries to prevent leakage from overlapping
    Triple Barrier labels.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
    y_train : array-like, shape (n_samples,)
    w_train : array-like or None, shape (n_samples,)
    n_folds : int
    embargo : int
        Number of observations to remove between train and val.

    Returns
    -------
    list of (X_tr, y_tr, w_tr, X_val, y_val) tuples
    """
    X = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    y = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    w = None
    if w_train is not None:
        w = w_train.values if hasattr(w_train, "values") else np.array(w_train)

    n = len(X)
    fold_size = n // n_folds
    splits = []

    for fold_idx in range(n_folds):
        val_start = fold_idx * fold_size
        val_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n

        # purge: remove embargo observations around val boundaries
        purge_start = max(0, val_start - embargo)
        purge_end = min(n, val_end + embargo)

        # training indices: everything outside purged zone
        train_mask = np.ones(n, dtype=bool)
        train_mask[purge_start:purge_end] = False

        # validation indices
        val_indices = np.arange(val_start, val_end)
        train_indices = np.where(train_mask)[0]

        if len(train_indices) == 0 or len(val_indices) == 0:
            continue

        X_tr = X[train_indices]
        y_tr = y[train_indices]
        w_tr = w[train_indices] if w is not None else None
        X_val = X[val_indices]
        y_val = y[val_indices]

        splits.append((X_tr, y_tr, w_tr, X_val, y_val))

    return splits


# =====================================================================
# Evaluate a model on all purged K-Fold splits (with pruning support)
# =====================================================================
def _evaluate_on_splits(splits, model_fn, trial=None):
    """Evaluate a model factory on all inner splits.

    Parameters
    ----------
    splits : list of (X_tr, y_tr, w_tr, X_val, y_val)
    model_fn : callable(X_tr, y_tr, w_tr) -> model with predict_proba
    trial : optuna.Trial or None
        If provided, reports intermediate values for pruning.

    Returns
    -------
    float : mean log loss across splits
    float : mean accuracy across splits
    """
    split_losses = []
    split_accs = []

    for fold_idx, (X_tr, y_tr, w_tr, X_val, y_val) in enumerate(splits):
        model = model_fn(X_tr, y_tr, w_tr)
        proba = model.predict_proba(X_val)
        ll = log_loss(y_val, proba)
        acc = (model.predict(X_val) == y_val).mean()

        split_losses.append(ll)
        split_accs.append(acc)

        # report intermediate result for pruning
        if trial is not None:
            trial.report(np.mean(split_losses), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return np.mean(split_losses), np.mean(split_accs)


# =====================================================================
# Logistic Regression
# =====================================================================
def tune_logistic(X_train, y_train, w_train=None, seed=42, verbose=True, n_trials=None):
    """Tune Logistic Regression via Optuna TPE + Purged K-Fold.

    Search space:
        C:       log-uniform [1e-4, 1e2]
        penalty: categorical {l1, l2}
    """
    splits = _purged_kfold_splits(X_train, y_train, w_train)
    _n = n_trials if n_trials is not None else N_TRIALS_CLASSICAL

    if verbose:
        print(
            f"    [tuning] logistic: {len(splits)} purged K-fold splits, "
            f"{_n} Optuna trials (TPE)"
        )

    all_results = []

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "liblinear" if penalty == "l1" else "lbfgs"

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


# =====================================================================
# Random Forest
# =====================================================================
def tune_random_forest(X_train, y_train, w_train=None, seed=42, verbose=True, n_trials=None):
    """Tune Random Forest via Optuna TPE + Purged K-Fold.

    Search space:
        n_estimators:     int [100, 300] step 50
        max_depth:        int [3, 20]
        min_samples_leaf: int [1, 30]
        max_features:     categorical {sqrt, log2}
    """
    splits = _purged_kfold_splits(X_train, y_train, w_train)
    _n = n_trials if n_trials is not None else N_TRIALS_CLASSICAL

    if verbose:
        print(
            f"    [tuning] random_forest: {len(splits)} purged K-fold splits, "
            f"{_n} Optuna trials (TPE)"
        )

    all_results = []

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 300, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)
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


# =====================================================================
# XGBoost
# =====================================================================
def tune_xgboost(X_train, y_train, w_train=None, seed=42, verbose=True, n_trials=None):
    """Tune XGBoost via Optuna TPE + Purged K-Fold.

    Search space:
        max_depth:        int [2, 6]
        learning_rate:    log-uniform [0.01, 0.3]
        min_child_weight: int [1, 30]
        subsample:        uniform [0.6, 1.0]
        colsample_bytree: uniform [0.6, 1.0]
        gamma:            log-uniform [1e-8, 1.0]
        reg_alpha:        log-uniform [1e-8, 10.0]
        reg_lambda:       log-uniform [1e-8, 10.0]

    n_estimators fixed at 500 with early stopping (20 rounds).
    """
    splits = _purged_kfold_splits(X_train, y_train, w_train)
    _n = n_trials if n_trials is not None else N_TRIALS_CLASSICAL

    if verbose:
        print(
            f"    [tuning] xgboost: {len(splits)} purged K-fold splits, "
            f"{_n} Optuna trials (TPE)"
        )

    all_results = []

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 2, 6)
        lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 30)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)

        split_losses = []
        split_accs = []

        for fold_idx, (X_tr, y_tr, w_tr, X_val, y_val) in enumerate(splits):
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
            m.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_val, y_val)], verbose=False,
            )
            proba = m.predict_proba(X_val)
            ll = log_loss(y_val, proba)
            acc = (m.predict(X_val) == y_val).mean()

            split_losses.append(ll)
            split_accs.append(acc)

            # report for pruning
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


# =====================================================================
# LSTM
# =====================================================================
def tune_lstm(X_train, y_train, w_train=None, n_features=None,
              seed=42, verbose=True, n_trials=None):
    """Tune LSTM via Optuna TPE + Purged K-Fold.

    Search space:
        hidden_size:   int [16, 64] step 16
        num_layers:    int [1, 3]
        dropout:       uniform [0.1, 0.5]
        learning_rate: log-uniform [1e-4, 5e-2]
    """
    from torch.utils.data import TensorDataset, DataLoader
    from src.cpcv.models.lstm_model import LSTMClassifier, create_sequences

    splits = _purged_kfold_splits(X_train, y_train, w_train)

    if n_features is None:
        n_features = splits[0][0].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    window = 21
    batch_size = 64
    epochs = 100
    patience = 10

    # pre-build sequences for all splits
    seq_splits = []
    for X_tr, y_tr, w_tr, X_val, y_val in splits:
        X_seq, y_seq, w_seq, _ = create_sequences(X_tr, y_tr, w_tr, window=window)
        X_val_seq, y_val_seq, _, _ = create_sequences(X_val, y_val, window=window)

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
            "y_val_np": y_val_seq,
            "y_seq_np": y_seq,
        })

    if not seq_splits:
        return {"best_params": {}, "best_log_loss": np.nan, "results_df": pd.DataFrame()}

    _n = n_trials if n_trials is not None else N_TRIALS_NEURAL

    if verbose:
        print(
            f"    [tuning] lstm: {len(seq_splits)} purged K-fold splits, "
            f"{_n} Optuna trials (TPE)"
        )

    all_results = []

    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 16, 64, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)

        split_losses = []

        for fold_idx, s in enumerate(seq_splits):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)
                python_random.seed(seed)

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

                criterion = nn.CrossEntropyLoss(weight=cw_t, reduction="none")
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, factor=0.5
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

                for epoch in range(epochs):
                    net.train()
                    for X_b, y_b, w_b in train_dl:
                        optimizer.zero_grad()
                        logits = net(X_b)
                        per_sample = criterion(logits, y_b)
                        loss = (per_sample * w_b).mean()
                        loss.backward()
                        optimizer.step()

                    net.eval()
                    with torch.no_grad():
                        val_logits = net(s["X_val_t"])
                        val_loss = nn.CrossEntropyLoss(
                            weight=cw_t
                        )(val_logits, s["y_val_t"]).item()

                    scheduler.step(val_loss)

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

                net.eval()
                with torch.no_grad():
                    logits = net(s["X_val_t"])
                    proba = torch.softmax(logits, dim=1).cpu().numpy()

                ll = log_loss(s["y_val_np"], proba)
                split_losses.append(ll)

                del net, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # report for pruning
                trial.report(np.mean(split_losses), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
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
            "num_layers": best["num_layers"],
            "dropout": best["dropout"],
            "learning_rate": best["learning_rate"],
        },
        "best_log_loss": study.best_value,
        "results_df": df,
    }


# =====================================================================
# KAN (efficient-kan + AdamW)
# =====================================================================
def tune_kan(X_train, y_train, w_train=None, n_features=None,
             seed=42, verbose=True, n_trials=None):
    """Tune KAN via Optuna TPE + Purged K-Fold.

    Search space:
        width1:       int [3, 12]          (1st hidden layer)
        width2:       int [0, 10]          (2nd hidden; 0 = skip)
        lr:           log-uniform [1e-3, 0.1]
        weight_decay: log-uniform [1e-5, 1e-2]
        grid:         categorical {3, 5}

    Fixed: k=3, epochs=200, patience=20, full-batch, tanh normalization.
    """
    from efficient_kan import KAN

    splits = _purged_kfold_splits(X_train, y_train, w_train)

    if n_features is None:
        n_features = splits[0][0].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = 3
    epochs = 200
    patience = 20

    _n = n_trials if n_trials is not None else N_TRIALS_NEURAL

    if verbose:
        print(
            f"    [tuning] kan (efficient-kan + AdamW): "
            f"{len(splits)} purged K-fold splits, "
            f"{_n} Optuna trials (TPE)"
        )

    all_results = []

    def objective(trial):
        width1 = trial.suggest_int("width1", 3, 12)
        width2 = trial.suggest_int("width2", 0, 10)
        lr = trial.suggest_float("lr", 1e-3, 0.1, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        grid = trial.suggest_categorical("grid", [3, 5])

        if width2 == 0:
            widths = [n_features, width1, 2]
        else:
            widths = [n_features, width1, width2, 2]

        split_losses = []

        for fold_idx, (X_tr, y_tr, w_tr, X_val, y_val) in enumerate(splits):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)
                python_random.seed(seed)

                X_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
                y_t = torch.tensor(y_tr, dtype=torch.long).to(device)
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

                if w_tr is not None:
                    w_t = torch.tensor(w_tr, dtype=torch.float32).to(device)
                else:
                    w_t = torch.ones(len(y_t), dtype=torch.float32).to(device)

                # tanh normalization
                input_mean = X_t.mean(dim=0)
                input_std = X_t.std(dim=0) + 1e-8
                X_t = torch.tanh((X_t - input_mean) / input_std)
                X_val_t = torch.tanh((X_val_t - input_mean) / input_std)

                # class weights
                cc = np.bincount(y_tr, minlength=2)
                cw = 1.0 / (cc + 1e-8)
                cw = cw / cw.sum() * 2
                cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

                criterion_train = nn.CrossEntropyLoss(
                    weight=cw_t, reduction="none"
                )
                criterion_val = nn.CrossEntropyLoss(weight=cw_t)

                model = KAN(
                    layers_hidden=widths,
                    grid_size=grid,
                    spline_order=k,
                    grid_range=[-1, 1],
                ).to(device)

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=wd,
                )

                best_val_loss = float("inf")
                best_state = None
                patience_counter = 0

                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()

                    logits = model(X_t)
                    per_sample = criterion_train(logits, y_t)
                    loss = (per_sample * w_t).mean()

                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_loss = criterion_val(
                            model(X_val_t), y_val_t
                        ).item()

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

                ll = log_loss(y_val, proba)
                split_losses.append(ll)

                del model, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # report for pruning
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
            "width2": best["width2"],
            "grid": best["grid"],
            "lr": best["lr"],
            "weight_decay": best["weight_decay"],
        },
        "best_log_loss": study.best_value,
        "results_df": df,
    }


# =====================================================================
# Tune all models
# =====================================================================
def tune_all_models(
    X_train, y_train, w_train=None, n_features=None,
    models=None, seed=42, verbose=True, n_trials=None,
):
    """Run Optuna TPE hyperparameter tuning for all specified models.

    Parameters
    ----------
    models : list[str], optional
        Models to tune. Defaults to ["logistic", "random_forest", "xgboost"].
        Add "lstm" and/or "kan" for neural model tuning.
    n_trials : int, optional
        Number of Optuna trials per model. If None, uses defaults
        (60 for classical, 40 for neural).

    Returns
    -------
    dict[str, dict]
        {model_name: {"best_params": {...}, "best_log_loss": float,
                       "results_df": DataFrame}}
    """
    if models is None:
        models = ["logistic", "random_forest", "xgboost"]

    if n_features is None:
        n_features = (
            X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])
        )

    dispatch = {
        "logistic": lambda: tune_logistic(X_train, y_train, w_train, seed, verbose, n_trials),
        "random_forest": lambda: tune_random_forest(X_train, y_train, w_train, seed, verbose, n_trials),
        "xgboost": lambda: tune_xgboost(X_train, y_train, w_train, seed, verbose, n_trials),
        "lstm": lambda: tune_lstm(X_train, y_train, w_train, n_features, seed, verbose, n_trials),
        "kan": lambda: tune_kan(X_train, y_train, w_train, n_features, seed, verbose, n_trials),
    }

    all_results = {}
    total_start = time.time()

    if verbose:
        print(f"\n  {'='*60}")
        print(f"  Hyperparameter Tuning (Optuna TPE + Purged K-Fold CV)")
        print(f"  Inner CV: {N_INNER_FOLDS} folds, "
              f"embargo={PURGE_EMBARGO} observations")
        print(f"  {'='*60}")

    for model_name in models:
        if model_name not in dispatch:
            logger.warning("No tuning function for '%s'. Skipping.", model_name)
            continue

        t0 = time.time()
        result = dispatch[model_name]()
        elapsed = time.time() - t0

        all_results[model_name] = result

        if verbose and result["best_params"]:
            print(
                f"\n    [tuning] {model_name} done in {elapsed:.1f}s"
                f" — best log_loss={result['best_log_loss']:.4f}"
            )

    elapsed_total = time.time() - total_start

    if verbose:
        print(f"\n  {'='*60}")
        print(f"  Tuning Summary ({elapsed_total:.1f}s total)")
        print(f"  {'─'*60}")
        for name, res in all_results.items():
            if res["best_params"]:
                print(
                    f"  {name:<16} "
                    f"log_loss={res['best_log_loss']:.4f}  "
                    f"params={res['best_params']}"
                )
        print(f"  {'='*60}")

    return all_results


# =====================================================================
# Helpers
# =====================================================================
def _print_study_summary(study, df, model_name):
    """Print Optuna study summary."""
    n_complete = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])

    print(
        f"\n      {model_name}: {n_complete} completed, "
        f"{n_pruned} pruned (of {len(study.trials)} total)"
    )
    print(f"      Best log_loss: {study.best_value:.4f}")
    print(f"      Best params: {study.best_params}")

    if len(df) >= 2:
        print(f"\n      Top 5 {model_name} configurations:")
        print(f"      {'─'*70}")
        for i, row in df.head(5).iterrows():
            params = " ".join(
                f"{c}={row[c]:.4g}" if isinstance(row[c], float) else f"{c}={row[c]}"
                for c in df.columns if c not in ["accuracy", "log_loss"]
            )
            ll_str = f"log_loss={row['log_loss']:.4f}"
            acc_str = (
                f"acc={row['accuracy']:.3f}"
                if "accuracy" in row and not np.isnan(row.get("accuracy", np.nan))
                else ""
            )
            print(f"      {i+1}. {params}  {ll_str} {acc_str}")
