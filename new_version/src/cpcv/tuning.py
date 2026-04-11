"""
10.2) Hyperparameter Tuning — Walk-Forward Nested Time-Series CV
================================================================
Walk-forward validation for hyperparameter tuning, following Ślepaczuk
& Bieganowski (2024): the training set initially expands then caps at
a fixed window length (MAX_TRAIN_PERIODS), shifting forward thereafter.

Inner CV structure (N_PERIODS=6, MAX_TRAIN_PERIODS=3):
  Split 1: train [P1]         → val [P2]
  Split 2: train [P1,P2]      → val [P3]
  Split 3: train [P1,P2,P3]   → val [P4]   (cap reached)
  Split 4: train [P2,P3,P4]   → val [P5]   (sliding)
  Split 5: train [P3,P4,P5]   → val [P6]   (sliding)

Each hyperparameter combination is evaluated across all inner splits
and selected by lowest average log loss.

The tuning happens INSIDE each outer CPCV fold using only training data:
  - Inner splits: walk-forward as above
  - Best params selected by lowest mean log loss across splits
  - Test fold is never seen during tuning (DSR/PBO remain valid)

Grid sizes per model:
  - Logistic Regression:  12 combinations  (~10 seconds)
  - Random Forest:        36 combinations  (~3 minutes)
  - XGBoost:              72 combinations  (~3 minutes)
  - LSTM:                 36 combinations  (~30 minutes)
  - KAN:                  27 combinations  (~5 minutes)
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

# Walk-forward inner CV configuration
N_PERIODS = 6               # number of inner periods
MAX_TRAIN_PERIODS = 3        # cap training window at this many periods


# =====================================================================
# Walk-forward split generator
# =====================================================================
def _prepare_walkforward_splits(X_train, y_train, w_train=None):
    """Generate walk-forward inner CV splits.

    Divides the training data into N_PERIODS equal-sized periods,
    then generates splits where:
      - Training expands from 1 period up to MAX_TRAIN_PERIODS
      - After reaching the cap, training window slides forward
      - Validation is always the next period

    Returns list of (X_tr, y_tr, w_tr, X_val, y_val) tuples.
    """
    X = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    y = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    w = None
    if w_train is not None:
        w = w_train.values if hasattr(w_train, "values") else np.array(w_train)

    n = len(X)
    period_size = n // N_PERIODS
    splits = []

    for val_period in range(1, N_PERIODS):
        # training periods: expand up to MAX_TRAIN_PERIODS, then slide
        train_end = val_period
        train_start = max(0, train_end - MAX_TRAIN_PERIODS)

        tr_start_idx = train_start * period_size
        tr_end_idx = train_end * period_size
        val_start_idx = val_period * period_size
        val_end_idx = (
            (val_period + 1) * period_size
            if val_period < N_PERIODS - 1
            else n
        )

        X_tr = X[tr_start_idx:tr_end_idx]
        y_tr = y[tr_start_idx:tr_end_idx]
        w_tr = w[tr_start_idx:tr_end_idx] if w is not None else None
        X_val = X[val_start_idx:val_end_idx]
        y_val = y[val_start_idx:val_end_idx]

        splits.append((X_tr, y_tr, w_tr, X_val, y_val))

    return splits


# =====================================================================
# Logistic Regression — 12 combinations
# =====================================================================
def tune_logistic(X_train, y_train, w_train=None, seed=42, verbose=True):
    """Tune Logistic Regression: C × penalty.

    Grid: C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}
          penalty ∈ {l1, l2}
    """
    splits = _prepare_walkforward_splits(X_train, y_train, w_train)
    n_splits = len(splits)
    total = 12

    if verbose:
        print(
            f"    [tuning] logistic: {n_splits} WF splits, "
            f"{total} combinations"
        )

    results = []
    count = 0

    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for penalty in ["l1", "l2"]:
            count += 1
            solver = "liblinear" if penalty == "l1" else "lbfgs"

            split_losses = []
            split_accs = []

            for X, y, w, X_val, y_val in splits:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = LogisticRegression(
                        C=C, penalty=penalty, solver=solver,
                        class_weight="balanced", max_iter=1000,
                        random_state=seed,
                    )
                    m.fit(X, y, sample_weight=w)
                    proba = m.predict_proba(X_val)
                    split_losses.append(log_loss(y_val, proba))
                    split_accs.append((m.predict(X_val) == y_val).mean())

            avg_ll = np.mean(split_losses)
            avg_acc = np.mean(split_accs)

            results.append({
                "C": C, "penalty": penalty,
                "accuracy": avg_acc, "log_loss": avg_ll,
            })

            if verbose:
                print(
                    f"      [{count:>3}/{total}] "
                    f"C={C:<8} penalty={penalty:<4} "
                    f"acc={avg_acc:.3f} log_loss={avg_ll:.4f}"
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
# Random Forest — 36 combinations
# =====================================================================
def tune_random_forest(X_train, y_train, w_train=None, seed=42, verbose=True):
    """Tune Random Forest: n_estimators × max_depth × min_samples_leaf.

    Grid: n_estimators ∈ {100, 300, 500}
          max_depth ∈ {5, 10, 20}
          min_samples_leaf ∈ {1, 5, 10, 20}
          max_features = sqrt (fixed)
    """
    splits = _prepare_walkforward_splits(X_train, y_train, w_train)
    n_splits = len(splits)
    total = 36

    if verbose:
        print(
            f"    [tuning] random_forest: {n_splits} WF splits, "
            f"{total} combinations"
        )

    results = []
    count = 0

    for n_estimators in [100, 300, 500]:
        for max_depth in [5, 10, 20]:
            for min_samples_leaf in [1, 5, 10, 20]:
                count += 1

                split_losses = []
                split_accs = []

                for X, y, w, X_val, y_val in splits:
                    m = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    )
                    m.fit(X, y, sample_weight=w)
                    proba = m.predict_proba(X_val)
                    split_losses.append(log_loss(y_val, proba))
                    split_accs.append((m.predict(X_val) == y_val).mean())

                avg_ll = np.mean(split_losses)
                avg_acc = np.mean(split_accs)

                results.append({
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": "sqrt",
                    "accuracy": avg_acc,
                    "log_loss": avg_ll,
                })

                if verbose:
                    print(
                        f"      [{count:>3}/{total}] "
                        f"n_est={n_estimators:<5} "
                        f"max_d={max_depth:<5} "
                        f"min_leaf={min_samples_leaf:<4} "
                        f"acc={avg_acc:.3f} log_loss={avg_ll:.4f}"
                    )

    df = pd.DataFrame(results).sort_values("log_loss", ignore_index=True)
    best = df.iloc[0]

    if verbose:
        _print_top5(df, "random_forest")

    return {
        "best_params": {
            "n_estimators": int(best["n_estimators"]),
            "max_depth": int(best["max_depth"]),
            "min_samples_leaf": int(best["min_samples_leaf"]),
            "max_features": best["max_features"],
        },
        "best_log_loss": best["log_loss"],
        "results_df": df,
    }


# =====================================================================
# XGBoost — 72 combinations
# =====================================================================
def tune_xgboost(X_train, y_train, w_train=None, seed=42, verbose=True):
    """Tune XGBoost: max_depth × learning_rate × min_child_weight × subsample.

    Grid: max_depth ∈ {3, 5, 7}
          learning_rate ∈ {0.05, 0.1}
          min_child_weight ∈ {1, 5, 10, 20}
          subsample ∈ {0.7, 0.8, 1.0}

    n_estimators fixed at 500 with early stopping (20 rounds).
    """
    splits = _prepare_walkforward_splits(X_train, y_train, w_train)
    n_splits = len(splits)
    total = 72

    if verbose:
        print(
            f"    [tuning] xgboost: {n_splits} WF splits, "
            f"{total} combinations"
        )

    results = []
    count = 0

    for max_depth in [3, 5, 7]:
        for lr in [0.05, 0.1]:
            for min_child_weight in [1, 5, 10, 20]:
                for subsample in [0.7, 0.8, 1.0]:
                    count += 1

                    split_losses = []
                    split_accs = []

                    for X, y, w, X_val, y_val in splits:
                        n_pos = (y == 1).sum()
                        n_neg = (y == 0).sum()
                        spw = n_neg / max(n_pos, 1)

                        m = XGBClassifier(
                            n_estimators=500,
                            max_depth=max_depth,
                            learning_rate=lr,
                            min_child_weight=min_child_weight,
                            subsample=subsample,
                            colsample_bytree=0.8,
                            scale_pos_weight=spw,
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
                        split_losses.append(log_loss(y_val, proba))
                        split_accs.append((m.predict(X_val) == y_val).mean())

                    avg_ll = np.mean(split_losses)
                    avg_acc = np.mean(split_accs)

                    results.append({
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "min_child_weight": min_child_weight,
                        "subsample": subsample,
                        "accuracy": avg_acc,
                        "log_loss": avg_ll,
                    })

                    if verbose:
                        print(
                            f"      [{count:>3}/{total}] "
                            f"max_d={max_depth:<3} "
                            f"lr={lr:<5} "
                            f"min_cw={min_child_weight:<4} "
                            f"sub={subsample:<4} "
                            f"acc={avg_acc:.3f} log_loss={avg_ll:.4f}"
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
# LSTM — 36 combinations
# =====================================================================
def tune_lstm(X_train, y_train, w_train=None, n_features=None,
              seed=42, verbose=True):
    """Tune LSTM: hidden_size × num_layers × dropout × learning_rate.

    Grid: hidden_size ∈ {32, 64, 128}
          num_layers ∈ {1, 2}
          dropout ∈ {0.1, 0.2, 0.3}
          learning_rate ∈ {1e-3, 1e-2}

    Window, batch_size, and epochs fixed. Early stopping on val loss.
    """
    from torch.utils.data import TensorDataset, DataLoader
    from src.cpcv.models.lstm_model import LSTMClassifier, create_sequences

    splits = _prepare_walkforward_splits(X_train, y_train, w_train)
    n_splits = len(splits)

    if n_features is None:
        n_features = splits[0][0].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    window = 21
    batch_size = 64
    epochs = 100
    patience = 10

    # pre-build sequences for all splits
    seq_splits = []
    for X, y, w, X_val, y_val in splits:
        X_seq, y_seq, w_seq, _ = create_sequences(X, y, w, window=window)
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

    total = 36

    if verbose:
        print(
            f"    [tuning] lstm: {len(seq_splits)} WF splits, "
            f"{total} combinations"
        )

    results = []
    count = 0

    for hidden_size in [32, 64, 128]:
        for num_layers in [1, 2]:
            for dropout in [0.1, 0.2, 0.3]:
                for lr in [1e-3, 1e-2]:
                    count += 1

                    split_losses = []
                    split_accs = []

                    for s in seq_splits:
                        try:
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            python_random.seed(seed)

                            # class weights per split
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

                            criterion = nn.CrossEntropyLoss(
                                weight=cw_t, reduction="none"
                            )
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

                            split_losses.append(log_loss(s["y_val_np"], proba))
                            split_accs.append(
                                (logits.argmax(dim=1).cpu().numpy() == s["y_val_np"]).mean()
                            )

                        except Exception as e:
                            logger.debug("LSTM tuning split failed: %s", e)
                            continue

                    if not split_losses:
                        continue

                    avg_ll = np.mean(split_losses)
                    avg_acc = np.mean(split_accs)
                    stopped_at = epoch + 1

                    results.append({
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "learning_rate": lr,
                        "stopped_epoch": stopped_at,
                        "accuracy": avg_acc,
                        "log_loss": avg_ll,
                    })

                    if verbose:
                        print(
                            f"      [{count:>3}/{total}] "
                            f"hidden={hidden_size:<4} "
                            f"layers={num_layers} "
                            f"drop={dropout:<4} "
                            f"lr={lr:<6} "
                            f"acc={avg_acc:.3f} log_loss={avg_ll:.4f}"
                        )

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
# KAN — 27 combinations (efficient-kan + AdamW)
# =====================================================================
def tune_kan(X_train, y_train, w_train=None, n_features=None,
             seed=42, verbose=True):
    """Tune KAN using efficient-kan with AdamW (matches kan_model.py).

    Grid: width1 ∈ {5, 10, 15}             (1st hidden layer)
          width2 ∈ {0, 5, 10}              (2nd hidden; 0 = skip)
          lr ∈ {5e-3, 1e-2, 5e-2}          (AdamW learning rate)
          weight_decay = 1e-4 (fixed)

    Fixed: grid=5, k=3, epochs=200, patience=20, full-batch training.
    Tanh normalization applied to inputs (same as kan_model.py).

    Total: 9 × 3 = 27 combinations.
    """
    from efficient_kan import KAN

    splits = _prepare_walkforward_splits(X_train, y_train, w_train)
    n_splits = len(splits)

    if n_features is None:
        n_features = splits[0][0].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fixed hyperparameters
    grid = 5
    k = 3
    epochs = 200
    patience = 20
    wd = 1e-4

    # grid values
    widths1 = [5, 10, 15]
    widths2 = [0, 5, 10]
    lrs = [5e-3, 1e-2, 5e-2]

    total = len(widths1) * len(widths2) * len(lrs)

    if verbose:
        print(
            f"    [tuning] kan (efficient-kan + AdamW, WF-CV): "
            f"{n_splits} WF splits, {total} combinations"
        )

    results = []
    count = 0

    for width1 in widths1:
        for width2 in widths2:
            for lr in lrs:
                count += 1

                if width2 == 0:
                    widths = [n_features, width1, 2]
                else:
                    widths = [n_features, width1, width2, 2]

                arch_str = "x".join(str(w) for w in widths)

                split_losses = []
                split_accs = []

                for X, y, w, X_val, y_val in splits:
                    try:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        python_random.seed(seed)

                        # tensors
                        X_t = torch.tensor(X, dtype=torch.float32).to(device)
                        y_t = torch.tensor(y, dtype=torch.long).to(device)
                        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

                        if w is not None:
                            w_t = torch.tensor(w, dtype=torch.float32).to(device)
                        else:
                            w_t = torch.ones(len(y_t), dtype=torch.float32).to(device)

                        # tanh normalization
                        input_mean = X_t.mean(dim=0)
                        input_std = X_t.std(dim=0) + 1e-8
                        X_t = torch.tanh((X_t - input_mean) / input_std)
                        X_val_t = torch.tanh((X_val_t - input_mean) / input_std)

                        # class weights
                        cc = np.bincount(y, minlength=2)
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

                        split_losses.append(log_loss(y_val, proba))
                        split_accs.append(
                            (logits.argmax(dim=1).cpu().numpy() == y_val).mean()
                        )

                        del model, optimizer
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logger.debug(
                            "KAN tuning split (%s, lr=%.4f): %s",
                            arch_str, lr, e,
                        )
                        continue

                if not split_losses:
                    continue

                avg_ll = np.mean(split_losses)
                avg_acc = np.mean(split_accs)

                results.append({
                    "width1": width1,
                    "width2": width2,
                    "architecture": arch_str,
                    "grid": grid,
                    "lr": lr,
                    "weight_decay": wd,
                    "accuracy": avg_acc,
                    "log_loss": avg_ll,
                })

                if verbose:
                    print(
                        f"      [{count:>3}/{total}] "
                        f"arch={arch_str:<16} "
                        f"lr={lr:<6} "
                        f"acc={avg_acc:.3f} "
                        f"log_loss={avg_ll:.4f}"
                    )

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
            "weight_decay": wd,
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
    """Run walk-forward hyperparameter tuning for all specified models.

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
        print(f"  Hyperparameter Tuning (Walk-Forward CV)")
        print(f"  Inner CV: {N_PERIODS} periods, "
              f"max {MAX_TRAIN_PERIODS} training periods")
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