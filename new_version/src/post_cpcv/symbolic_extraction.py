"""
Post-CPCV — Symbolic Extraction
================================
Take the best CPCV fold, retrain a PyKAN model (same architecture and
parameters as kan_model.py), then apply Algorithm 1 from the VIX KAN paper:
  1. Train with L1 + entropy regularization
  2. Prune low-importance edges and nodes
  3. Symbolify activation functions with closed-form candidates
  4. Fine-tune affine parameters

Since kan_model.py now uses PyKAN directly, this module uses the exact
same library and architecture, ensuring consistency between the prediction
model and the interpretability analysis.
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
from src.cpcv.preprocessing import apply_ffd
from src.cpcv.models.kan_model import (
    KAN_GRID, KAN_K, KAN_LR, KAN_TRAIN_STEPS,
    KAN_LAMB, KAN_LAMB_L1, KAN_LAMB_ENTROPY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants specific to symbolic extraction (beyond training)
# ---------------------------------------------------------------------------
PRUNE_THRESHOLD = 0.01

SYMBOLIC_LIBRARY = [
    "x", "x^2", "x^3", "exp", "log", "sqrt", "tanh", "sin", "abs", "0",
]
SYMBOLIC_R2_THRESHOLD = 0.5
SYMBOLIC_TOPK = 5

AFFINE_FINETUNE_STEPS = 30
AFFINE_LR = 0.0004          # from VIX paper

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
        logger.warning("No KAN predictions found. Using first available split.")
        for key, pred in predictions.items():
            return key[1], pred.get("prep_info", {})

    best_split = max(split_f1s, key=split_f1s.get)
    logger.info(
        "Best KAN fold: split %d (F1=%.4f). All: %s",
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
    splits = generate_cpcv_splits(X, t1)
    train_idx, _ = splits[best_split_idx]

    y_mapped = ((y + 1) // 2).astype(int)

    ffd_info = prep_info.get("ffd", {})
    scaler = prep_info.get("scaler", None)
    selected_features = prep_info.get("selected_features", list(X.columns))

    # apply FFD to full series then extract training fold
    X_transformed = X.copy()
    for col, d_star in ffd_info.items():
        if col in X_transformed.columns:
            X_transformed[col] = apply_ffd(X_transformed[col], d_star)

    X_train = X_transformed.iloc[train_idx].copy()
    X_train = X_train.loc[X_train.notna().all(axis=1)]
    y_train = y_mapped.loc[X_train.index]

    # scale using the stored scaler (same transform as pipeline)
    if scaler is not None:
        X_train = pd.DataFrame(
            scaler.transform(X_train),
            index=X_train.index, columns=X_train.columns,
        )

    X_train = X_train[selected_features]

    # 80/20 split (same as pipeline)
    cal_boundary = int(len(X_train) * 0.8)
    X_model = X_train.iloc[:cal_boundary]
    X_val = X_train.iloc[cal_boundary:]
    y_model = y_train.iloc[:cal_boundary]
    y_val = y_train.iloc[cal_boundary:]

    dataset = {
        "train_input": torch.tensor(X_model.values, dtype=torch.float32),
        "train_label": torch.tensor(y_model.values, dtype=torch.float32),
        "test_input": torch.tensor(X_val.values, dtype=torch.float32),
        "test_label": torch.tensor(y_val.values, dtype=torch.float32),
    }

    feature_names = list(selected_features)
    logger.info(
        "Extraction data: %d train, %d val, %d features.",
        len(X_model), len(X_val), len(feature_names),
    )
    return dataset, feature_names


# =====================================================================
# 3. Train PyKAN (same architecture as kan_model.py)
# =====================================================================
def train_pykan(dataset: dict, n_features: int, n_classes: int = 2):
    """Train a PyKAN model using the same config as kan_model.py."""
    from kan import KAN

    width = [n_features, 2 * n_features, n_classes]
    model = KAN(width=width, grid=KAN_GRID, k=KAN_K, seed=42)

    # try native fit with CrossEntropyLoss (needs Long labels)
    try:
        dataset_fit = {
            "train_input": dataset["train_input"],
            "train_label": dataset["train_label"].long(),
            "test_input": dataset["test_input"],
            "test_label": dataset["test_label"].long(),
        }
        model.fit(
            dataset_fit,
            opt="LBFGS",
            lr=KAN_LR,
            steps=KAN_TRAIN_STEPS,
            lamb=KAN_LAMB,
            lamb_l1=KAN_LAMB_L1,
            lamb_entropy=KAN_LAMB_ENTROPY,
            loss_fn=nn.CrossEntropyLoss(),
            update_grid=False,
        )
        logger.info("PyKAN trained via model.fit(): %d steps.", KAN_TRAIN_STEPS)

    except (TypeError, AttributeError, RuntimeError, Exception) as e:
        logger.info("model.fit() failed (%s). Using custom loop.", e)
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=KAN_LR, max_iter=20,
            line_search_fn="strong_wolfe",
        )
        loss_fn = nn.CrossEntropyLoss()

        for step in range(KAN_TRAIN_STEPS):
            model.train()

            def closure():
                optimizer.zero_grad()
                pred = model(dataset["train_input"])
                loss = loss_fn(pred, dataset["train_label"].long())
                try:
                    reg = KAN_LAMB * (
                        KAN_LAMB_L1 * model.regularization_loss(regularize_activation=1.0)
                        + KAN_LAMB_ENTROPY * model.regularization_loss(regularize_entropy=1.0)
                    )
                    loss = loss + reg
                except (AttributeError, TypeError):
                    pass
                loss.backward()
                return loss

            optimizer.step(closure)

        model.eval()

    # store pre-symbolic accuracy
    with torch.no_grad():
        pred = model(dataset["test_input"])
        val_acc = (pred.argmax(dim=1) == dataset["test_label"].long()).float().mean().item()

    model._pre_symbolic_accuracy = val_acc
    logger.info("PyKAN trained. Val accuracy: %.4f, width: %s", val_acc, width)
    return model


# =====================================================================
# 4. Prune (Algorithm 1, Step 2)
# =====================================================================
def prune_network(model, dataset: dict):
    """Prune dead edges and nodes."""
    _ = model(dataset["train_input"])

    try:
        original_width = list(model.width)
    except AttributeError:
        original_width = ["unknown"]

    try:
        model.attribute()
    except Exception as e:
        logger.warning("model.attribute() failed: %s", e)

    # save state before pruning in case it produces an empty layer
    pre_prune_state = copy.deepcopy(model.state_dict())
    pre_prune_model = model

    # try different PyKAN prune APIs (varies by version)
    pruned = False
    try:
        model = model.prune(threshold=PRUNE_THRESHOLD)
        pruned = True
    except TypeError:
        try:
            model = model.prune(node_th=PRUNE_THRESHOLD, edge_th=PRUNE_THRESHOLD)
            pruned = True
        except TypeError:
            try:
                model = model.prune()
                pruned = True
            except Exception as e:
                logger.warning("model.prune() failed: %s. Returning unpruned.", e)
                return pre_prune_model
    except Exception as e:
        logger.warning("model.prune() failed: %s. Returning unpruned.", e)
        return pre_prune_model

    # verify pruned model can still do a forward pass
    if pruned:
        try:
            _ = model(dataset["train_input"])
        except (RuntimeError, Exception) as e:
            logger.warning(
                "Pruned model forward pass failed (%s). "
                "Pruning removed too many nodes. Returning unpruned model.", e
            )
            return pre_prune_model

    try:
        pruned_width = list(model.width)
    except AttributeError:
        pruned_width = ["unknown"]

    logger.info("Pruned: %s → %s", original_width, pruned_width)
    print(f"  Architecture: {original_width} → {pruned_width}")

    _save_plot(model, "kan_pruned_network.png")
    return model


# =====================================================================
# 5. Symbolify (Algorithm 1, Step 3 + 4)
# =====================================================================
def symbolify_network(model, dataset: dict):
    """Replace B-spline activations with symbolic functions, then fine-tune."""
    # refresh cached activations
    _ = model(dataset["train_input"])

    # pre-symbolification accuracy
    with torch.no_grad():
        pre_pred = model(dataset["test_input"])
        pre_acc = (pre_pred.argmax(dim=1) == dataset["test_label"].long()).float().mean().item()

    pre_state = copy.deepcopy(model.state_dict())

    total_edges = 0
    symbolified_edges = 0

    # ── Step 3: replace splines with symbolic functions ───────────────
    try:
        for l in range(len(model.width) - 1):
            n_in = model.width[l]
            n_out = model.width[l + 1]

            # PyKAN width can be int or list
            if isinstance(n_in, (list, tuple)):
                n_in = n_in[0] if n_in else 0
            if isinstance(n_out, (list, tuple)):
                n_out = n_out[0] if n_out else 0

            for i in range(n_in):
                for j in range(n_out):
                    total_edges += 1
                    try:
                        suggestions = model.suggest_symbolic(
                            l, i, j, topk=SYMBOLIC_TOPK, lib=SYMBOLIC_LIBRARY,
                        )
                    except Exception as e:
                        logger.debug("suggest_symbolic(%d,%d,%d) failed: %s", l, i, j, e)
                        continue

                    if suggestions is None:
                        continue

                    # ── parse suggestions (handles both DataFrame and list) ──
                    best_fn = None
                    best_r2 = 0.0

                    # newer PyKAN returns a DataFrame
                    if hasattr(suggestions, "iloc"):
                        if len(suggestions) == 0:
                            continue
                        best_fn = str(suggestions.iloc[0, 0])
                        r2_col = [
                            c for c in suggestions.columns
                            if "r2" in c.lower() and "loss" not in c.lower()
                        ]
                        if r2_col:
                            best_r2 = float(suggestions.iloc[0][r2_col[0]])
                    # older PyKAN returns a list of tuples
                    elif isinstance(suggestions, (list, tuple)):
                        if len(suggestions) == 0:
                            continue
                        best_fn = suggestions[0][0]
                        best_r2 = float(suggestions[0][2]) if len(suggestions[0]) > 2 else 0.0
                    else:
                        continue

                    if best_fn is None:
                        continue

                    # ── apply symbolic replacement if R² is good enough ──
                    if best_r2 >= SYMBOLIC_R2_THRESHOLD:
                        try:
                            model.fix_symbolic(l, i, j, best_fn)
                            symbolified_edges += 1
                            logger.info(
                                "Edge (%d,%d,%d): %s (R²=%.4f)",
                                l, i, j, best_fn, best_r2,
                            )
                        except Exception as e:
                            logger.warning(
                                "fix_symbolic(%d,%d,%d) failed: %s",
                                l, i, j, e,
                            )
                    else:
                        logger.info(
                            "Edge (%d,%d,%d): best R²=%.4f < %.2f, keeping spline.",
                            l, i, j, best_r2, SYMBOLIC_R2_THRESHOLD,
                        )

    except Exception as e:
        logger.warning("Symbolification loop error: %s", e)

    sym_rate = symbolified_edges / max(total_edges, 1)
    print(f"  Symbolified: {symbolified_edges}/{total_edges} edges ({sym_rate:.0%})")

    # ── Step 4: fine-tune affine parameters ───────────────────────────
    print(f"  Fine-tuning affine parameters ({AFFINE_FINETUNE_STEPS} steps)...")
    try:
        dataset_fit = {
            "train_input": dataset["train_input"],
            "train_label": dataset["train_label"].long(),
            "test_input": dataset["test_input"],
            "test_label": dataset["test_label"].long(),
        }
        model.fit(
            dataset_fit, opt="LBFGS", lr=AFFINE_LR,
            steps=AFFINE_FINETUNE_STEPS,
            loss_fn=nn.CrossEntropyLoss(),
            update_grid=False,  # grid update breaks after symbolification
        )
    except (TypeError, AttributeError, RuntimeError):
        try:
            optimizer = torch.optim.LBFGS(model.parameters(), lr=AFFINE_LR, max_iter=10)
            loss_fn = nn.CrossEntropyLoss()
            for step in range(AFFINE_FINETUNE_STEPS):
                def closure():
                    optimizer.zero_grad()
                    pred = model(dataset["train_input"])
                    loss = loss_fn(pred, dataset["train_label"].long())
                    if torch.isnan(loss):
                        raise ValueError("NaN loss")
                    loss.backward()
                    return loss
                try:
                    optimizer.step(closure)
                except ValueError:
                    logger.warning("NaN at affine step %d. Reverting.", step)
                    model.load_state_dict(pre_state)
                    break
        except Exception as e:
            logger.warning("Affine fine-tuning failed: %s. Reverting.", e)
            model.load_state_dict(pre_state)

    # save visualization
    _save_plot(model, "kan_symbolified_network.png")

    model._symbolification_rate = sym_rate
    model._pre_symbolic_accuracy = pre_acc
    return model


# =====================================================================
# 6. Extract formulas
# =====================================================================
def extract_formulas(model, dataset: dict, feature_names: list[str]) -> dict:
    """Extract closed-form expressions from the symbolified PyKAN model."""
    # get symbolic formulas
    try:
        formulas = model.symbolic_formula()
    except Exception as e:
        logger.error("model.symbolic_formula() failed: %s", e)
        return _empty_result(model, feature_names)

    # parse SymPy expressions
    try:
        if isinstance(formulas, (list, tuple)) and len(formulas) > 0:
            expr_list = formulas[0] if isinstance(formulas[0], (list, tuple)) else formulas
            logit_bearish = expr_list[0]
            logit_bullish = expr_list[1] if len(expr_list) > 1 else sympy.Integer(0)
        else:
            return _empty_result(model, feature_names)
    except Exception as e:
        logger.error("Formula parsing failed: %s", e)
        return _empty_result(model, feature_names)

    # substitute x_0, x_1, ... with actual feature names
    for i, name in enumerate(feature_names):
        old = sympy.Symbol(f"x_{i}")
        new = sympy.Symbol(name)
        try:
            logit_bearish = logit_bearish.subs(old, new)
            logit_bullish = logit_bullish.subs(old, new)
        except (AttributeError, TypeError):
            pass

    # decision function
    try:
        decision = sympy.simplify(logit_bullish - logit_bearish)
    except Exception:
        decision = logit_bullish - logit_bearish

    # simplify with rational approximation
    try:
        logit_bearish = sympy.nsimplify(logit_bearish, tolerance=1e-4)
        logit_bullish = sympy.nsimplify(logit_bullish, tolerance=1e-4)
        decision = sympy.nsimplify(decision, tolerance=1e-4)
    except Exception:
        pass

    # post-symbolic accuracy
    pre_acc = getattr(model, "_pre_symbolic_accuracy", np.nan)
    sym_rate = getattr(model, "_symbolification_rate", np.nan)

    try:
        with torch.no_grad():
            post_pred = model(dataset["test_input"])
            post_acc = (
                (post_pred.argmax(dim=1) == dataset["test_label"].long())
                .float().mean().item()
            )
    except Exception:
        post_acc = np.nan

    # surviving features
    surviving = [
        name for name in feature_names
        if sympy.Symbol(name) in decision.free_symbols
    ]

    # pruned architecture
    try:
        pruned_arch = list(model.width)
    except AttributeError:
        pruned_arch = ["unknown"]

    return {
        "logit_bearish": str(logit_bearish),
        "logit_bullish": str(logit_bullish),
        "decision_function": str(decision),
        "p_up_formula": f"1 / (1 + exp(-({decision})))",
        "sympy_objects": {
            "bearish": logit_bearish,
            "bullish": logit_bullish,
            "decision": decision,
        },
        "pre_symbolic_accuracy": pre_acc,
        "post_symbolic_accuracy": post_acc,
        "symbolification_rate": sym_rate,
        "pruned_architecture": pruned_arch,
        "surviving_features": surviving,
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
    """Run the full symbolic extraction pipeline (Algorithm 1).

    Called from the notebook after CPCV results are available.
    """
    print("=" * 60)
    print("Symbolic Extraction (VIX KAN Paper, Algorithm 1)")
    print("=" * 60)

    # 1. select best fold
    best_split, prep_info = select_extraction_fold(cpcv_results)
    print(f"  Best fold: split {best_split}")

    # 2. prepare data
    dataset, feature_names = prepare_extraction_data(
        X, y, w, t1, cpcv_results, best_split, prep_info,
    )
    print(
        f"  Data: {dataset['train_input'].shape[0]} train, "
        f"{dataset['test_input'].shape[0]} val, "
        f"{len(feature_names)} features: {feature_names}"
    )

    # 3. train (Step 1)
    print("\n  Step 1: Training PyKAN...")
    model = train_pykan(dataset, n_features=len(feature_names))

    # 4. prune (Step 2)
    print("\n  Step 2: Pruning...")
    model = prune_network(model, dataset)

    # 5. symbolify + fine-tune (Steps 3 + 4)
    print("\n  Steps 3+4: Symbolification + affine fine-tuning...")
    model = symbolify_network(model, dataset)

    # 6. extract formulas
    print("\n  Extracting formulas...")
    result = extract_formulas(model, dataset, feature_names)

    # summary
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Pre-symbolic accuracy:  {result['pre_symbolic_accuracy']:.4f}")
    print(f"  Post-symbolic accuracy: {result['post_symbolic_accuracy']:.4f}")
    print(f"  Symbolification rate:   {result['symbolification_rate']:.0%}")
    print(f"  Surviving features:     {result['surviving_features']}")
    print(f"  Pruned architecture:    {result['pruned_architecture']}")
    print(f"\n  Decision function:")
    print(f"    {result['decision_function']}")
    print(f"\n  P(up) = {result['p_up_formula']}")
    print(f"{'='*60}")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_plot(model, filename: str) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            model.plot(mask=True, beta=10)
        except TypeError:
            model.plot(beta=10)
        import matplotlib.pyplot as plt
        plt.savefig(
            os.path.join(CACHE_DIR, filename),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        logger.info("Saved %s", filename)
    except Exception as e:
        logger.warning("Could not save %s: %s", filename, e)


def _empty_result(model, feature_names: list[str]) -> dict:
    """Fallback when formula extraction fails."""
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