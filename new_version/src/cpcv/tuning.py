"""
10.2) Hyperparameter Tuning
========================
Brute-force grid search for each model, evaluated on a chronological
inner validation split within the CPCV training fold.

Uses log loss (binary cross-entropy) as the tuning metric.

The tuning happens INSIDE each outer CPCV fold using only training data:
  - Inner split: 75% tune-train / 25% tune-val (chronological)
  - Best params selected by lowest log loss on tune-val
  - Test fold is never seen during tuning (DSR/PBO remain valid)

Grid sizes per model:
  - Logistic Regression:  12 combinations  (~5 seconds)
  - Random Forest:        96 combinations  (~3 minutes)
  - XGBoost:             144 combinations  (~3 minutes)
  - LSTM:                 81 combinations  (~30 minutes)
  - KAN:                 108 combinations  (~5 minutes)
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

logger = logging.getLogger(__name__)

# Inner validation split fraction (chronological)
TUNE_VAL_FRAC = 0.25


# =====================================================================
# Logistic Regression — 12 combinations
# =====================================================================
def tune_logistic(X_train, y_train, w_train=None, seed=42, verbose=True):
    """Tune Logistic Regression: C × penalty.

    Grid: C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}
          penalty ∈ {l1, l2}
    """
    X, y, w, X_val, y_val = _prepare_split(X_train, y_train, w_train)

    if verbose:
        print(f"    [tuning] logistic: {len(X)} train, {len(X_val)} val, 12 combinations")

    results = []

    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for penalty in ["l1", "l2"]:
            solver = "liblinear" if penalty == "l1" else "lbfgs"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LogisticRegression(
                    C=C, penalty=penalty, solver=solver,
                    class_weight="balanced", max_iter=1000,
                    random_state=seed,
                )
                m.fit(X, y, sample_weight=w)
                proba = m.predict_proba(X_val)
                ll = log_loss(y_val, proba)
                acc = (m.predict(X_val) == y_val).mean()

                results.append({
                    "C": C, "penalty": penalty,
                    "accuracy": acc, "log_loss": ll,
                })

                if verbose:
                    print(
                        f"      C={C:<8} penalty={penalty:<4} "
                        f"acc={acc:.3f} log_loss={ll:.4f}"
                    )

    df = pd.DataFrame(results).sort_values("log_loss", ignore_index=True)
    best = df.iloc[0]

    if verbose:
        _print_top5(df, "logistic")

    return {
        "best_params": {"C": best["C"], "penalty": best["penalty"]},
        "best_log_loss": best["log_loss"],
        "results_df": df,
    }


# =====================================================================
# Random Forest — 96 combinations
# =====================================================================
def tune_random_forest(X_train, y_train, w_train=None, seed=42, verbose=True):
    """Tune Random Forest: n_estimators × max_depth × min_samples_leaf × max_features.

    Grid: n_estimators ∈ {100, 300, 500}
          max_depth ∈ {5, 10, 20, None}
          min_samples_leaf ∈ {1, 5, 10, 20}
          max_features ∈ {sqrt, log2}
    """
    X, y, w, X_val, y_val = _prepare_split(X_train, y_train, w_train)

    if verbose:
        print(f"    [tuning] random_forest: {len(X)} train, {len(X_val)} val, 96 combinations")

    results = []
    count = 0

    for n_estimators in [100, 300, 500]:
        for max_depth in [5, 10, 20, None]:
            for min_samples_leaf in [1, 5, 10, 20]:
                for max_features in ["sqrt", "log2"]:
                    count += 1

                    m = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    )
                    m.fit(X, y, sample_weight=w)
                    proba = m.predict_proba(X_val)
                    ll = log_loss(y_val, proba)
                    acc = (m.predict(X_val) == y_val).mean()

                    results.append({
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,
                        "accuracy": acc,
                        "log_loss": ll,
                    })

                    if verbose:
                        print(
                            f"      [{count:>3}] "
                            f"n_est={n_estimators:<5} "
                            f"max_d={str(max_depth):<5} "
                            f"min_leaf={min_samples_leaf:<4} "
                            f"max_feat={max_features:<5} "
                            f"acc={acc:.3f} log_loss={ll:.4f}"
                        )

    df = pd.DataFrame(results).sort_values("log_loss", ignore_index=True)
    best = df.iloc[0]

    if verbose:
        _print_top5(df, "random_forest")

    return {
        "best_params": {
            "n_estimators": int(best["n_estimators"]),
            "max_depth": best["max_depth"],
            "min_samples_leaf": int(best["min_samples_leaf"]),
            "max_features": best["max_features"],
        },
        "best_log_loss": best["log_loss"],
        "results_df": df,
    }


# =====================================================================
# XGBoost — 144 combinations
# =====================================================================
def tune_xgboost(X_train, y_train, w_train=None, seed=42, verbose=True):
    """Tune XGBoost: max_depth × learning_rate × min_child_weight × subsample.

    Grid: max_depth ∈ {3, 5, 7, 10}
          learning_rate ∈ {0.01, 0.05, 0.1}
          min_child_weight ∈ {1, 5, 10, 20}
          subsample ∈ {0.7, 0.8, 1.0}

    n_estimators fixed at 500 with early stopping (20 rounds).
    scale_pos_weight computed from class distribution.
    """
    X, y, w, X_val, y_val = _prepare_split(X_train, y_train, w_train)

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    if verbose:
        print(f"    [tuning] xgboost: {len(X)} train, {len(X_val)} val, 144 combinations")

    results = []
    count = 0

    for max_depth in [3, 5, 7, 10]:
        for lr in [0.01, 0.05, 0.1]:
            for min_child_weight in [1, 5, 10, 20]:
                for subsample in [0.7, 0.8, 1.0]:
                    count += 1

                    m = XGBClassifier(
                        n_estimators=500,
                        max_depth=max_depth,
                        learning_rate=lr,
                        min_child_weight=min_child_weight,
                        subsample=subsample,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        early_stopping_rounds=20,
                        random_state=seed,
                    )
                    m.fit(
                        X, y, sample_weight=w,
                        eval_set=[(X_val, y_val)], verbose=False,
                    )
                    proba = m.predict_proba(X_val)
                    ll = log_loss(y_val, proba)
                    acc = (m.predict(X_val) == y_val).mean()

                    results.append({
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "min_child_weight": min_child_weight,
                        "subsample": subsample,
                        "best_iteration": m.best_iteration,
                        "accuracy": acc,
                        "log_loss": ll,
                    })

                    if verbose:
                        print(
                            f"      [{count:>3}] "
                            f"max_d={max_depth:<3} "
                            f"lr={lr:<5} "
                            f"min_cw={min_child_weight:<4} "
                            f"sub={subsample:<4} "
                            f"iter={m.best_iteration:<4} "
                            f"acc={acc:.3f} log_loss={ll:.4f}"
                        )

    df = pd.DataFrame(results).sort_values("log_loss", ignore_index=True)
    best = df.iloc[0]

    if verbose:
        _print_top5(df, "xgboost")

    return {
        "best_params": {
            "max_depth": int(best["max_depth"]),
            "learning_rate": best["learning_rate"],
            "min_child_weight": int(best["min_child_weight"]),
            "subsample": best["subsample"],
        },
        "best_log_loss": best["log_loss"],
        "results_df": df,
    }


# =====================================================================
# LSTM — 81 combinations
# =====================================================================
def tune_lstm(X_train, y_train, w_train=None, n_features=None,
              seed=42, verbose=True):
    """Tune LSTM: hidden_size × num_layers × dropout × learning_rate.

    Grid: hidden_size ∈ {32, 64, 128}
          num_layers ∈ {1, 2, 3}
          dropout ∈ {0.1, 0.2, 0.3}
          learning_rate ∈ {1e-4, 1e-3, 1e-2}

    Window, batch_size, and epochs fixed. Early stopping on val loss.
    """
    from torch.utils.data import TensorDataset, DataLoader
    from src.cpcv.models.lstm_model import LSTMClassifier, create_sequences

    X, y, w, X_val, y_val = _prepare_split(X_train, y_train, w_train)

    if n_features is None:
        n_features = X.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    window = 21
    batch_size = 64
    epochs = 100
    patience = 10

    X_seq, y_seq, w_seq, _ = create_sequences(X, y, w, window=window)
    X_val_seq, y_val_seq, _, _ = create_sequences(X_val, y_val, window=window)

    X_seq_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_seq_t = torch.tensor(y_seq, dtype=torch.long).to(device)
    w_seq_t = (
        torch.tensor(w_seq, dtype=torch.float32).to(device)
        if w_seq is not None
        else torch.ones(len(y_seq), dtype=torch.float32).to(device)
    )
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.long).to(device)

    class_counts = np.bincount(y_seq, minlength=2)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * 2
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if verbose:
        print(
            f"    [tuning] lstm: {len(X_seq)} sequences train, "
            f"{len(X_val_seq)} val, 81 combinations"
        )

    results = []
    count = 0

    for hidden_size in [32, 64, 128]:
        for num_layers in [1, 2, 3]:
            for dropout in [0.1, 0.2, 0.3]:
                for lr in [1e-4, 1e-3, 1e-2]:
                    count += 1

                    try:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        python_random.seed(seed)

                        net = LSTMClassifier(
                            n_features=n_features, n_classes=2,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                        ).to(device)

                        criterion = nn.CrossEntropyLoss(
                            weight=class_weights_t, reduction="none"
                        )
                        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, patience=5, factor=0.5
                        )

                        train_ds = TensorDataset(X_seq_t, y_seq_t, w_seq_t)
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
                                val_logits = net(X_val_t)
                                val_loss = nn.CrossEntropyLoss(
                                    weight=class_weights_t
                                )(val_logits, y_val_t).item()

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
                            logits = net(X_val_t)
                            proba = torch.softmax(logits, dim=1).cpu().numpy()

                        ll = log_loss(y_val_seq, proba)
                        acc = (logits.argmax(dim=1).cpu().numpy() == y_val_seq).mean()
                        stopped_at = epoch + 1

                        results.append({
                            "hidden_size": hidden_size,
                            "num_layers": num_layers,
                            "dropout": dropout,
                            "learning_rate": lr,
                            "stopped_epoch": stopped_at,
                            "accuracy": acc,
                            "log_loss": ll,
                        })

                        if verbose:
                            print(
                                f"      [{count:>3}] "
                                f"hidden={hidden_size:<4} "
                                f"layers={num_layers} "
                                f"drop={dropout:<4} "
                                f"lr={lr:<6} "
                                f"epoch={stopped_at:<4} "
                                f"acc={acc:.3f} log_loss={ll:.4f}"
                            )

                    except Exception as e:
                        if verbose:
                            print(
                                f"      [{count:>3}] FAILED: "
                                f"hidden={hidden_size} layers={num_layers} "
                                f"drop={dropout} lr={lr} — {e}"
                            )
                        logger.debug("LSTM tuning failed: %s", e)
                        continue

    if not results:
        return {"best_params": {}, "best_log_loss": np.nan, "results_df": pd.DataFrame()}

    df = pd.DataFrame(results).sort_values("log_loss", ignore_index=True)
    best = df.iloc[0]

    if verbose:
        _print_top5(df, "lstm")

    return {
        "best_params": {
            "hidden_size": int(best["hidden_size"]),
            "num_layers": int(best["num_layers"]),
            "dropout": best["dropout"],
            "learning_rate": best["learning_rate"],
        },
        "best_log_loss": best["log_loss"],
        "results_df": df,
    }


# =====================================================================
# KAN — 108 combinations (efficient-kan + AdamW)
# =====================================================================
def tune_kan(X_train, y_train, w_train=None, n_features=None,
             seed=42, verbose=True):
    """Tune KAN using efficient-kan with AdamW (matches kan_model.py).

    Grid: width1 ∈ {5, 10, 15}             (1st hidden layer)
          width2 ∈ {0, 5, 10}              (2nd hidden; 0 = skip)
          lr ∈ {1e-3, 5e-3, 1e-2, 5e-2}    (AdamW learning rate)
          weight_decay ∈ {1e-4, 1e-3, 1e-2} (L2 regularization)

    Fixed: grid=5, k=3, epochs=200, patience=20, full-batch training.
    Tanh normalization applied to inputs (same as kan_model.py).

    Total: 9 × 4 × 3 = 108 combinations (~5 min per fold).
    """
    from efficient_kan import KAN

    X, y, w, X_val, y_val = _prepare_split(X_train, y_train, w_train)

    if n_features is None:
        n_features = X.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── tensors ───────────────────────────────────────────────────────
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    if w is not None:
        w_t = torch.tensor(w, dtype=torch.float32).to(device)
    else:
        w_t = torch.ones(len(y_t), dtype=torch.float32).to(device)

    # ── tanh normalization (same as kan_model.py) ─────────────────────
    input_mean = X_t.mean(dim=0)
    input_std = X_t.std(dim=0) + 1e-8
    X_t = torch.tanh((X_t - input_mean) / input_std)
    X_val_t = torch.tanh((X_val_t - input_mean) / input_std)

    # ── class weights ─────────────────────────────────────────────────
    class_counts = np.bincount(y, minlength=2)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * 2
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion_train = nn.CrossEntropyLoss(weight=class_weights_t, reduction="none")
    criterion_val = nn.CrossEntropyLoss(weight=class_weights_t)

    # ── fixed hyperparameters ─────────────────────────────────────────
    grid = 5
    k = 3
    epochs = 200
    patience = 20

    # ── grid values ───────────────────────────────────────────────────
    widths1 = [5, 10, 15]
    widths2 = [0, 5, 10]
    lrs = [1e-3, 5e-3, 1e-2, 5e-2]
    weight_decays = [1e-4, 1e-3, 1e-2]

    total = len(widths1) * len(widths2) * len(lrs) * len(weight_decays)

    if verbose:
        print(
            f"    [tuning] kan (efficient-kan + AdamW): "
            f"{len(X)} train, {len(X_val)} val, {total} combinations"
        )

    results = []
    count = 0

    for width1 in widths1:
        for width2 in widths2:
            for lr in lrs:
                for wd in weight_decays:
                    count += 1

                    if width2 == 0:
                        widths = [n_features, width1, 2]
                    else:
                        widths = [n_features, width1, width2, 2]

                    arch_str = "x".join(str(w) for w in widths)

                    try:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        python_random.seed(seed)

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

                            # validation
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
                        acc = (logits.argmax(dim=1).cpu().numpy() == y_val).mean()
                        stopped_at = epoch + 1

                        results.append({
                            "width1": width1,
                            "width2": width2,
                            "architecture": arch_str,
                            "grid": grid,
                            "lr": lr,
                            "weight_decay": wd,
                            "stopped_epoch": stopped_at,
                            "accuracy": acc,
                            "log_loss": ll,
                        })

                        if verbose:
                            print(
                                f"      [{count:>3}/{total}] "
                                f"arch={arch_str:<16} "
                                f"lr={lr:<6} "
                                f"wd={wd:<6} "
                                f"epoch={stopped_at:<4} "
                                f"acc={acc:.3f} "
                                f"log_loss={ll:.4f}"
                            )

                        del model, optimizer
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        if verbose and count <= 5:
                            print(
                                f"      [{count:>3}/{total}] "
                                f"FAILED: {type(e).__name__}: {e}"
                            )
                        logger.debug(
                            "KAN tuning (%s, lr=%.4f, wd=%.4f): %s",
                            arch_str, lr, wd, e,
                        )
                        continue

    if not results:
        return {"best_params": {}, "best_log_loss": np.nan, "results_df": pd.DataFrame()}

    df = pd.DataFrame(results).sort_values("log_loss", ignore_index=True)
    best = df.iloc[0]

    if verbose:
        _print_top5(df, "kan")

    return {
        "best_params": {
            "width1": int(best["width1"]),
            "width2": int(best["width2"]),
            "grid": grid,
            "lr": best["lr"],
            "weight_decay": best["weight_decay"],
        },
        "best_log_loss": best["log_loss"],
        "results_df": df,
    }


# =====================================================================
# Tune all models
# =====================================================================
def tune_all_models(
    X_train, y_train, w_train=None, n_features=None,
    models=None, seed=42, verbose=True,
):
    """Run hyperparameter tuning for all specified models.

    Parameters
    ----------
    models : list[str], optional
        Models to tune. Defaults to ["logistic", "random_forest", "xgboost"].
        Add "lstm" and/or "kan" for full tuning (much slower).

    Returns
    -------
    dict[str, dict]
        {model_name: {"best_params": {...}, "best_log_loss": float, "results_df": DataFrame}}
    """
    if models is None:
        models = ["logistic", "random_forest", "xgboost"]

    if n_features is None:
        n_features = (
            X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])
        )

    dispatch = {
        "logistic": lambda: tune_logistic(X_train, y_train, w_train, seed, verbose),
        "random_forest": lambda: tune_random_forest(X_train, y_train, w_train, seed, verbose),
        "xgboost": lambda: tune_xgboost(X_train, y_train, w_train, seed, verbose),
        "lstm": lambda: tune_lstm(X_train, y_train, w_train, n_features, seed, verbose),
        "kan": lambda: tune_kan(X_train, y_train, w_train, n_features, seed, verbose),
    }

    all_results = {}
    total_start = time.time()

    if verbose:
        print(f"\n  {'='*60}")
        print(f"  Hyperparameter Tuning")
        print(f"  {'='*60}")

    for model_name in models:
        if model_name not in dispatch:
            logger.warning("No tuning grid for '%s'. Skipping.", model_name)
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
def _prepare_split(X_train, y_train, w_train=None):
    """Chronological 75/25 inner split, converting to numpy."""
    X = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    y = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    w = None
    if w_train is not None:
        w = w_train.values if hasattr(w_train, "values") else np.array(w_train)

    split = int(len(X) * (1 - TUNE_VAL_FRAC))
    X_tune, X_val = X[:split], X[split:]
    y_tune, y_val = y[:split], y[split:]
    w_tune = w[:split] if w is not None else None

    return X_tune, y_tune, w_tune, X_val, y_val


def _print_top5(df, model_name):
    """Print the top 5 configurations."""
    print(f"\n      Top 5 {model_name} configurations:")
    print(f"      {'─'*70}")
    cols = [c for c in df.columns if c not in ["accuracy", "log_loss"]]
    for i, row in df.head(5).iterrows():
        params = " ".join(f"{c}={row[c]}" for c in cols)
        print(
            f"      {i+1}. {params}  "
            f"acc={row['accuracy']:.3f} log_loss={row['log_loss']:.4f}"
        )