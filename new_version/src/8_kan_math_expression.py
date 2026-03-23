# kan_math_expression.py — KAN symbolic regression / equation extraction.
# **Experimental module** for extracting interpretable mathematical formulas
# from a trained PyKAN model.  Kept separate from ``models.py`` to allow
# independent experimentation without affecting the production pipeline.

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)

def evaluate_symbolic_fidelity(pruned_model: KAN, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """Evaluate predicting power in pure symbolic mode to prove formula fidelity."""
    # Force symbolic mode
    pruned_model.use_symbolic(True)
    
    with torch.no_grad():
        test_input = torch.tensor(X_val.values, dtype=torch.float32)
        raw_out = pruned_model(test_input)
        proba = torch.softmax(raw_out, dim=1)[:, 1].numpy()
        pred = torch.argmax(raw_out, dim=1).numpy()
        
    acc = accuracy_score(y_val.values, pred)
    try:
        auc = roc_auc_score(y_val.values, proba)
    except ValueError:
        auc = float('nan')
        
    logger.info("Symbolic Fidelity: Accuracy = %.4f, ROC-AUC = %.4f", acc, auc)
    return {"accuracy": acc, "roc_auc": auc}


def extract_symbolic_expression(kan_model: KAN, feature_names: List[str], lib: Optional[List[str]] = None) -> Dict[str, Any]:
    """Prune a KAN and extract exact symbolic mathematical equations.
    
    Extracts the analytical affine parameters (a, b, c, d) representing:
    c * f(a*x + b) + d
    """
    if lib is None:
        lib = ["x", "x^2", "x^3", "log", "exp", "tanh", "abs", "sqrt"]

    # 1. Prune
    logger.info("Pruning KAN model …")
    pruned = kan_model.prune()

    # 2. Auto-symbolic fitting
    logger.info("Fitting symbolic functions from lib=%s …", lib)
    pruned.auto_symbolic(lib=lib)

    expressions: Dict[str, Dict[str, str]] = {}
    
    num_layers = len(pruned.symbolic_fun)
    prev_layer_nodes = feature_names.copy()
    
    for layer_idx in range(num_layers):
        layer = pruned.symbolic_fun[layer_idx]
        n_out = len(layer.funs_name)
        n_in = len(layer.funs_name[0])
        
        current_layer_nodes = []
        layer_eqs = {}
        
        for out_node in range(n_out):
            terms = []
            for in_node in range(n_in):
                fn = layer.funs_name[out_node][in_node]
                
                if fn in ("0", "0.0"):
                    continue
                    
                # R2 Quality Check
                r2 = layer.r2[out_node, in_node].item()
                if r2 < 0.90:
                    logger.warning(
                        "Poor symbolic fit at layer %d, edge (%d -> %d): '%s' with R^2 = %.4f", 
                        layer_idx, in_node, out_node, fn, r2
                    )
                    
                # Extract analytical mathematically real affine coefficients
                # c * f(a*x + b) + d
                a, b, c, d = layer.affine[out_node, in_node].tolist()
                
                inner_var = prev_layer_nodes[in_node]
                inner_term = f"({a:.4f} * [{inner_var}] + {b:.4f})"
                
                if fn == "x":
                    f_val = inner_term
                elif fn == "x^2":
                    f_val = f"({inner_term})^2"
                elif fn == "x^3":
                    f_val = f"({inner_term})^3"
                else:
                    f_val = f"{fn}{inner_term}"
                    
                term_str = f"({c:.4f} * {f_val} + {d:.4f})"
                terms.append(term_str)
                
            node_name = f"z_{out_node}" if layer_idx == num_layers - 1 else f"H_{layer_idx}_{out_node}"
            current_layer_nodes.append(node_name)
            
            if terms:
                layer_eqs[node_name] = " + \n      ".join(terms)
            else:
                layer_eqs[node_name] = "0.0000"
                
        expressions[f"layer_{layer_idx}"] = layer_eqs
        prev_layer_nodes = current_layer_nodes
        
    final_eq = "P(Up) = exp(z_1) / (exp(z_0) + exp(z_1))"

    feat_map = {i: f for i, f in enumerate(feature_names)}

    result = {
        "pruned_model": pruned,
        "expressions": expressions,
        "final_equation": final_eq,
        "feature_map": feat_map,
    }

    logger.info("Symbolic extraction complete (%d layers parsed)", num_layers)
    return result


def print_trading_equations(expr: Dict[str, Any]) -> None:
    """Pretty-print the rigorously extracted mathematical KAN equations."""
    print("\n" + "=" * 80)
    print(" EXTRACTED MATHEMATICAL TRADING EQUATIONS")
    print("=" * 80)

    for layer_name, layer_eqs in expr["expressions"].items():
        print(f"\n--- {layer_name.upper()} ---")
        for node_name, equation in layer_eqs.items():
            print(f"{node_name} = {equation}\n")

    print(f"\n--- FINAL PREDICTION (Softmax) ---")
    print(expr["final_equation"])

    print("\n" + "-" * 40)
    print("Input Node Mapping (Top → Bottom):")
    for idx, feat in expr["feature_map"].items():
        print(f"  Node {idx}: {feat}")

    print("=" * 80)
