"""
12) Symbolic Extraction
================================
Re-train a PyKAN model on the best CPCV fold's preprocessed data,
prune dead edges/nodes, replace B-spline activations with closed-form
symbolic functions (Algorithm 1 from VIX KAN paper / KASPER framework),
fine-tune affine parameters, and extract human-readable mathematical
expressions for P(price goes up).

This is an interpretability analysis tool, not a production model.
"""

import copy
import logging
import os

import numpy as np
import pandas as pd
import sympy
import torch
import torch.nn as nn

from src.cpcv.cv import generate_cpcv_splits
from src.cpcv.preprocessing import apply_ffd, scale_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# PyKAN architecture
PYKAN_GRID = 3
PYKAN_K = 3

# Training
PYKAN_LR = 0.04
PYKAN_TRAIN_STEPS = 200
PYKAN_LAMB = 0.01
PYKAN_LAMB_L1 = 1.0
PYKAN_LAMB_ENTROPY = 2.0
PYKAN_LR_DECAY_PATIENCE = 5
PYKAN_LR_DECAY_FACTOR = 0.1
PYKAN_EARLY_STOP_PATIENCE = 10

# Pruning
PYKAN_PRUNE_THRESHOLD = 0.01

# Symbolification
SYMBOLIC_LIBRARY = ["x", "x^2", "x^3", "exp", "log", "sqrt", "tanh", "sin", "abs", "0"]
SYMBOLIC_R2_THRESHOLD = 0.5
SYMBOLIC_TOPK = 5
SYMBOLIC_WEIGHT_SIMPLE = 0.8

# Affine fine-tuning
AFFINE_FINETUNE_STEPS = 30
AFFINE_LR = 0.0004

# Cache
CACHE_DIR = "cache/"


# =====================================================================
# 1. Select best extraction fold
# =====================================================================
def select_extraction_fold(cpcv_results: dict) -> tuple[int, dict]:
    """Identify the CPCV fold where KAN achieved best F1 macro."""
    predictions = cpcv_results["predictions"]
    n_splits = cpcv_results["n_splits"]
    n_seeds = cpcv_results["n_seeds"]

    split_f1s = {}
    split_prep = {}

    for split_idx in range(n_splits):
        seed_f1s = []
        prep_info = None
        for seed in range(n_seeds):
            key = ("kan", split_idx, seed)
            if key not in predictions:
                continue
            pred = predictions[key]
            f1 = pred.get("f1_macro", np.nan)
            if not np.isnan(f1):
                seed_f1s.append(f1)
            if prep_info is None:
                prep_info = pred.get("prep_info", {})

        if seed_f1s:
            split_f1s[split_idx] = np.mean(seed_f1s)
            split_prep[split_idx] = prep_info

    if not split_f1s:
        # fallback: use any available model's prep_info
        logger.warning("No KAN predictions found. Using first available split.")
        for key, pred in predictions.items():
            return key[1], pred.get("prep_info", {})

    best_split = max(split_f1s, key=split_f1s.get)
    logger.info(
        "Best KAN fold: split %d (F1=%.4f). All folds: %s",
        best_split, split_f1s[best_split],
        {k: f"{v:.4f}" for k, v in split_f1s.items()},
    )

    return best_split, split_prep[best_split]


# =====================================================================
# 2. Prepare extraction data
# =====================================================================
def prepare_extraction_data(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    t1: pd.Series,
    cpcv_results: dict,
    best_split_idx: int,
    prep_info: dict,
) -> tuple[dict, list[str]]:
    """Reconstruct preprocessed data for the extraction fold in PyKAN format."""
    # re-generate splits to get indices
    splits = generate_cpcv_splits(X, t1)
    train_idx, test_idx = splits[best_split_idx]

    # map labels
    y_mapped = ((y + 1) // 2).astype(int)

    # get stored preprocessing parameters
    ffd_info = prep_info.get("ffd", {})
    scaler = prep_info.get("scaler", None)
    selected_features = prep_info.get("selected_features", list(X.columns))

    # apply FFD to full series, then extract fold
    X_transformed = X.copy()
    for col, d_star in ffd_info.items():
        if col in X_transformed.columns:
            X_transformed[col] = apply_ffd(X_transformed[col], d_star)

    X_train = X_transformed.iloc[train_idx].copy()
    X_test = X_transformed.iloc[test_idx].copy()

    # drop NaN rows
    train_mask = X_train.notna().all(axis=1)
    test_mask = X_test.notna().all(axis=1)
    X_train = X_train.loc[train_mask]
    X_test = X_test.loc[test_mask]

    # align labels
    y_train = y_mapped.loc[X_train.index]
    y_test = y_mapped.loc[X_test.index]

    # scale
    if scaler is not None:
        X_train = pd.DataFrame(
            scaler.transform(X_train), index=X_train.index, columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), index=X_test.index, columns=X_test.columns
        )

    # select features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # 80/20 calibration split from training
    cal_boundary = int(len(X_train) * 0.8)
    X_model = X_train.iloc[:cal_boundary]
    X_val = X_train.iloc[cal_boundary:]
    y_model = y_train.iloc[:cal_boundary]
    y_val = y_train.iloc[cal_boundary:]

    # convert to PyKAN format
    dataset = {
        "train_input": torch.tensor(X_model.values, dtype=torch.float32),
        "train_label": torch.tensor(y_model.values, dtype=torch.float32),
        "test_input": torch.tensor(X_val.values, dtype=torch.float32),
        "test_label": torch.tensor(y_val.values, dtype=torch.float32),
    }

    feature_names = list(selected_features)
    logger.info(
        "Extraction data prepared: %d train, %d val, %d features.",
        len(X_model), len(X_val), len(feature_names),
    )

    return dataset, feature_names


# =====================================================================
# 3. Re-train PyKAN
# =====================================================================
def retrain_pykan(dataset: dict, n_features: int, n_classes: int = 2):
    """Train a PyKAN model with small architecture for symbolification."""
    from kan import KAN

    width = [n_features, 2 * n_features, n_classes]

    model = KAN(width=width, grid=PYKAN_GRID, k=PYKAN_K, seed=42)

    # try PyKAN's built-in fit first
    try:
        results = model.fit(
            dataset,
            opt="LBFGS",
            lr=PYKAN_LR,
            steps=PYKAN_TRAIN_STEPS,
            lamb=PYKAN_LAMB,
            lamb_l1=PYKAN_LAMB_L1,
            lamb_entropy=PYKAN_LAMB_ENTROPY,
            loss_fn=nn.CrossEntropyLoss(),
        )
        logger.info("PyKAN trained via model.fit(): %d steps.", PYKAN_TRAIN_STEPS)

    except (TypeError, AttributeError) as e:
        logger.info("PyKAN model.fit() does not support loss_fn: %s. Using custom loop.", e)
        model = _custom_pykan_train(model, dataset, width)

    # compute pre-symbolification accuracy
    with torch.no_grad():
        val_pred = model(dataset["test_input"])
        val_acc = (val_pred.argmax(dim=1) == dataset["test_label"].long()).float().mean().item()

    model._pre_symbolic_accuracy = val_acc
    logger.info("PyKAN pre-symbolic validation accuracy: %.4f", val_acc)

    return model


def _custom_pykan_train(model, dataset: dict, width: list) -> object:
    """Custom LBFGS training loop for PyKAN when model.fit() doesn't support CrossEntropyLoss."""
    optimizer = torch.optim.LBFGS(model.parameters(), lr=PYKAN_LR, max_iter=20)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    lr_patience_counter = 0
    current_lr = PYKAN_LR

    for step in range(PYKAN_TRAIN_STEPS):
        model.train()

        def closure():
            optimizer.zero_grad()
            pred = model(dataset["train_input"])
            loss = loss_fn(pred, dataset["train_label"].long())
            # add regularization if available
            try:
                reg = PYKAN_LAMB * (
                    PYKAN_LAMB_L1 * model.regularization_loss(regularize_activation=1.0)
                    + PYKAN_LAMB_ENTROPY * model.regularization_loss(regularize_entropy=1.0)
                )
                loss = loss + reg
            except (AttributeError, TypeError):
                pass
            loss.backward()
            return loss

        train_loss = optimizer.step(closure)

        # validation every 10 steps
        if (step + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(dataset["test_input"])
                val_loss = loss_fn(val_pred, dataset["test_label"].long()).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                lr_patience_counter = 0
            else:
                patience_counter += 1
                lr_patience_counter += 1

            # LR decay
            if lr_patience_counter >= PYKAN_LR_DECAY_PATIENCE:
                current_lr *= PYKAN_LR_DECAY_FACTOR
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
                lr_patience_counter = 0
                logger.info("Step %d: LR reduced to %.6f", step + 1, current_lr)

            # early stopping
            if patience_counter >= PYKAN_EARLY_STOP_PATIENCE:
                logger.info("Step %d: early stopping (best val loss: %.4f)", step + 1, best_val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    logger.info("Custom PyKAN training complete: %d steps, best val loss: %.4f", step + 1, best_val_loss)

    return model


# =====================================================================
# 4. Prune network
# =====================================================================
def prune_network(model, dataset: dict):
    """Prune dead edges and nodes from the trained PyKAN model."""
    # forward pass to populate cached activations
    _ = model(dataset["train_input"])

    # get original architecture
    try:
        original_width = list(model.width)
    except AttributeError:
        original_width = ["unknown"]

    # compute attribution scores
    try:
        model.attribute()
    except (AttributeError, Exception) as e:
        logger.warning("model.attribute() failed: %s. Proceeding without attribution.", e)

    # prune
    try:
        model = model.prune(threshold=PYKAN_PRUNE_THRESHOLD)
    except (AttributeError, Exception) as e:
        logger.warning("model.prune() failed: %s. Returning unpruned model.", e)
        return model

    # get pruned architecture
    try:
        pruned_width = list(model.width)
    except AttributeError:
        pruned_width = ["unknown"]

    logger.info("Pruned: %s → %s", original_width, pruned_width)

    # save visualization
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        fig = model.plot(mask=True, beta=10)
        if hasattr(fig, "savefig"):
            fig.savefig(os.path.join(CACHE_DIR, "kan_pruned_network.png"), dpi=150, bbox_inches="tight")
        else:
            import matplotlib.pyplot as plt
            plt.savefig(os.path.join(CACHE_DIR, "kan_pruned_network.png"), dpi=150, bbox_inches="tight")
            plt.close()
        logger.info("Pruned network visualization saved.")
    except Exception as e:
        logger.warning("Could not save pruned network plot: %s", e)

    return model


# =====================================================================
# 5. Symbolify network
# =====================================================================
def symbolify_network(model, dataset: dict):
    """Replace B-spline activations with closed-form symbolic functions."""
    # refresh cached activations
    _ = model(dataset["train_input"])

    # pre-symbolification accuracy
    with torch.no_grad():
        pre_pred = model(dataset["test_input"])
        pre_acc = (pre_pred.argmax(dim=1) == dataset["test_label"].long()).float().mean().item()

    # save state for rollback if fine-tuning produces NaN
    pre_finetune_state = copy.deepcopy(model.state_dict())

    total_edges = 0
    symbolified_edges = 0

    # iterate over layers and edges
    try:
        for l in range(len(model.width) - 1):
            n_in = model.width[l]
            n_out = model.width[l + 1]

            # handle PyKAN width format (can be int or list)
            if isinstance(n_in, (list, tuple)):
                n_in = n_in[0] if len(n_in) > 0 else 0
            if isinstance(n_out, (list, tuple)):
                n_out = n_out[0] if len(n_out) > 0 else 0

            for i in range(n_in):
                for j in range(n_out):
                    total_edges += 1

                    try:
                        suggestions = model.suggest_symbolic(
                            l, i, j, topk=SYMBOLIC_TOPK, lib=SYMBOLIC_LIBRARY
                        )
                    except Exception as e:
                        logger.debug("suggest_symbolic(%d,%d,%d) failed: %s", l, i, j, e)
                        continue

                    if not suggestions or len(suggestions) == 0:
                        continue

                    # extract best suggestion
                    best_fn = suggestions[0][0]
                    best_r2 = suggestions[0][2] if len(suggestions[0]) > 2 else 0.0

                    if best_r2 >= SYMBOLIC_R2_THRESHOLD:
                        try:
                            model.fix_symbolic(l, i, j, best_fn)
                            symbolified_edges += 1
                            logger.info(
                                "Edge (%d,%d,%d): %s (R²=%.4f)", l, i, j, best_fn, best_r2
                            )
                        except Exception as e:
                            logger.warning(
                                "fix_symbolic(%d,%d,%d) failed: %s", l, i, j, e
                            )
                    else:
                        logger.info(
                            "Edge (%d,%d,%d): best R²=%.4f < %.2f, keeping spline.",
                            l, i, j, best_r2, SYMBOLIC_R2_THRESHOLD,
                        )

    except Exception as e:
        logger.warning("Symbolification loop encountered error: %s", e)

    sym_rate = symbolified_edges / max(total_edges, 1)
    logger.info(
        "Symbolification: %d/%d edges (%.1f%%)",
        symbolified_edges, total_edges, sym_rate * 100,
    )

    # affine fine-tuning (Step 4 of Algorithm 1)
    print(f"  [symbolic] Fine-tuning affine parameters ({AFFINE_FINETUNE_STEPS} steps)...")
    try:
        model.fit(
            dataset,
            opt="LBFGS",
            lr=AFFINE_LR,
            steps=AFFINE_FINETUNE_STEPS,
            loss_fn=nn.CrossEntropyLoss(),
        )
    except (TypeError, AttributeError):
        # custom fine-tuning loop
        try:
            optimizer = torch.optim.LBFGS(model.parameters(), lr=AFFINE_LR, max_iter=10)
            loss_fn = nn.CrossEntropyLoss()

            for step in range(AFFINE_FINETUNE_STEPS):
                def closure():
                    optimizer.zero_grad()
                    pred = model(dataset["train_input"])
                    loss = loss_fn(pred, dataset["train_label"].long())
                    if torch.isnan(loss):
                        raise ValueError("NaN loss during affine fine-tuning")
                    loss.backward()
                    return loss

                try:
                    optimizer.step(closure)
                except ValueError:
                    logger.warning(
                        "NaN loss at affine step %d. Reverting to pre-finetune state.", step
                    )
                    model.load_state_dict(pre_finetune_state)
                    break

        except Exception as e:
            logger.warning("Affine fine-tuning failed: %s. Reverting.", e)
            model.load_state_dict(pre_finetune_state)

    # save post-symbolification visualization
    try:
        fig = model.plot(mask=True, beta=10)
        if hasattr(fig, "savefig"):
            fig.savefig(os.path.join(CACHE_DIR, "kan_symbolified_network.png"), dpi=150, bbox_inches="tight")
        else:
            import matplotlib.pyplot as plt
            plt.savefig(os.path.join(CACHE_DIR, "kan_symbolified_network.png"), dpi=150, bbox_inches="tight")
            plt.close()
        logger.info("Symbolified network visualization saved.")
    except Exception as e:
        logger.warning("Could not save symbolified network plot: %s", e)

    # store metadata on the model
    model._symbolification_rate = sym_rate
    model._pre_symbolic_accuracy = pre_acc

    return model


# =====================================================================
# 6. Extract formulas
# =====================================================================
def extract_formulas(model, feature_names: list[str]) -> dict:
    """Extract closed-form mathematical expressions from the symbolified model."""
    # get symbolic formulas from PyKAN
    try:
        formulas = model.symbolic_formula()
    except Exception as e:
        logger.error("model.symbolic_formula() failed: %s", e)
        return _empty_formulas(model, feature_names)

    # extract raw SymPy expressions
    try:
        if isinstance(formulas, (list, tuple)) and len(formulas) > 0:
            if isinstance(formulas[0], (list, tuple)) and len(formulas[0]) >= 2:
                logit_bearish = formulas[0][0]
                logit_bullish = formulas[0][1]
            else:
                logit_bearish = formulas[0]
                logit_bullish = formulas[1] if len(formulas) > 1 else sympy.Integer(0)
        else:
            logger.warning("Unexpected formula format: %s", type(formulas))
            return _empty_formulas(model, feature_names)
    except Exception as e:
        logger.error("Formula extraction failed: %s", e)
        return _empty_formulas(model, feature_names)

    # substitute variable names
    for i, name in enumerate(feature_names):
        old_var = sympy.Symbol(f"x_{i}")
        new_var = sympy.Symbol(name)
        try:
            logit_bearish = logit_bearish.subs(old_var, new_var)
            logit_bullish = logit_bullish.subs(old_var, new_var)
        except (AttributeError, TypeError):
            pass

    # decision function = logit_bullish - logit_bearish
    try:
        decision_function = sympy.simplify(logit_bullish - logit_bearish)
    except Exception:
        decision_function = logit_bullish - logit_bearish

    # simplify expressions
    try:
        logit_bearish = sympy.nsimplify(logit_bearish, tolerance=1e-4)
        logit_bullish = sympy.nsimplify(logit_bullish, tolerance=1e-4)
        decision_function = sympy.nsimplify(decision_function, tolerance=1e-4)
    except Exception:
        pass

    # compute post-symbolification accuracy
    pre_acc = getattr(model, "_pre_symbolic_accuracy", np.nan)
    sym_rate = getattr(model, "_symbolification_rate", np.nan)

    try:
        with torch.no_grad():
            # need dataset for this, use stored test data
            post_pred = model(model._test_input) if hasattr(model, "_test_input") else None
            if post_pred is not None:
                post_acc = (post_pred.argmax(dim=1) == model._test_label.long()).float().mean().item()
            else:
                post_acc = np.nan
    except Exception:
        post_acc = np.nan

    # identify surviving features (those appearing in the decision function)
    surviving = []
    for name in feature_names:
        if sympy.Symbol(name) in decision_function.free_symbols:
            surviving.append(name)

    # pruned architecture
    try:
        pruned_arch = list(model.width)
    except AttributeError:
        pruned_arch = ["unknown"]

    result = {
        "logit_bearish": str(logit_bearish),
        "logit_bullish": str(logit_bullish),
        "decision_function": str(decision_function),
        "p_up_formula": f"1 / (1 + exp(-({decision_function})))",
        "sympy_objects": {
            "bearish": logit_bearish,
            "bullish": logit_bullish,
            "decision": decision_function,
        },
        "pre_symbolic_accuracy": pre_acc,
        "post_symbolic_accuracy": post_acc,
        "symbolification_rate": sym_rate,
        "pruned_architecture": pruned_arch,
        "surviving_features": surviving,
    }

    logger.info(
        "Formulas extracted. Decision function: %s", str(decision_function)[:200]
    )

    return result


def _empty_formulas(model, feature_names: list[str]) -> dict:
    """Return empty formula dict when extraction fails."""
    return {
        "logit_bearish": "extraction_failed",
        "logit_bullish": "extraction_failed",
        "decision_function": "extraction_failed",
        "p_up_formula": "extraction_failed",
        "sympy_objects": {"bearish": None, "bullish": None, "decision": None},
        "pre_symbolic_accuracy": getattr(model, "_pre_symbolic_accuracy", np.nan),
        "post_symbolic_accuracy": np.nan,
        "symbolification_rate": getattr(model, "_symbolification_rate", np.nan),
        "pruned_architecture": [],
        "surviving_features": [],
    }


# =====================================================================
# Top-level orchestration
# =====================================================================
def run_symbolic_extraction(
    cpcv_results: dict,
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    t1: pd.Series,
) -> dict:
    """Chain the entire symbolic extraction pipeline.

    Called from the notebook after CPCV results are available.
    """
    print("=" * 60)
    print("Symbolic Extraction Pipeline")
    print("=" * 60)

    # 1. find best KAN fold
    best_split_idx, prep_info = select_extraction_fold(cpcv_results)
    print(f"  Selected fold {best_split_idx} for symbolic extraction")

    # 2. prepare data
    feature_names = prep_info.get("selected_features", list(X.columns))
    dataset, feature_names = prepare_extraction_data(
        X, y, w, t1, cpcv_results, best_split_idx, prep_info
    )
    print(
        f"  Extraction data: {dataset['train_input'].shape[0]} train, "
        f"{dataset['test_input'].shape[0]} val, {len(feature_names)} features"
    )

    # 3. re-train PyKAN
    print("\n  Re-training PyKAN model...")
    model = retrain_pykan(dataset, n_features=len(feature_names))

    # store test data on model for post-symbolic accuracy computation
    model._test_input = dataset["test_input"]
    model._test_label = dataset["test_label"]

    # 4. prune
    print("\n  Pruning network...")
    model = prune_network(model, dataset)

    # 5. symbolify
    print("\n  Symbolifying activation functions...")
    model = symbolify_network(model, dataset)

    # 6. extract formulas
    print("\n  Extracting symbolic formulas...")
    formulas = extract_formulas(model, feature_names)

    # summary
    print(f"\n{'='*60}")
    print("Symbolic Extraction Results")
    print(f"{'='*60}")
    print(f"  Pre-symbolic accuracy:  {formulas['pre_symbolic_accuracy']:.4f}")
    print(f"  Post-symbolic accuracy: {formulas['post_symbolic_accuracy']:.4f}")
    print(f"  Symbolification rate:   {formulas['symbolification_rate']:.1%}")
    print(f"  Surviving features:     {formulas['surviving_features']}")
    print(f"  Pruned architecture:    {formulas['pruned_architecture']}")
    print(f"\n  Decision function:")
    print(f"    {formulas['decision_function']}")
    print(f"\n  P(up) = {formulas['p_up_formula']}")
    print(f"{'='*60}")

    return formulas