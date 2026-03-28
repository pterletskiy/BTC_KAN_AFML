"""
src/8_kan_math_expression.py
----------------------------
KAN symbolic regression and robust mathematical equation extraction.
Extracts interpretable mathematical formulas from trained PyKAN architectures
using SymPy algebraic simplification.

References:
  - PyKAN Algorithm 1: Pruning, Symbolification, and Affine Fine-Tuning.
"""

import copy
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sympy
import torch
from kan import KAN
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)


def evaluate_symbolic_fidelity(pruned_model: KAN, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """Evaluate predicting power in pure symbolic mode across N-class Softmax topologies."""
    device = next(pruned_model.parameters()).device if hasattr(pruned_model, 'parameters') else torch.device('cpu')
    
    # Save original symbolic state safely avoiding destructive mutations
    original_state = getattr(pruned_model, 'symbolic_enabled', False)
    pruned_model.use_symbolic(True)
    
    try:
        with torch.no_grad():
            test_input = torch.tensor(X_val.values, dtype=torch.float32).to(device)
            raw_out = pruned_model(test_input)
            
            # Dynamic multi-class Softmax evaluation
            n_classes = raw_out.shape[1]
            proba = torch.softmax(raw_out, dim=1).cpu().numpy()
            pred = torch.argmax(raw_out, dim=1).cpu().numpy()
            
        y_true = y_val.values
        acc = accuracy_score(y_true, pred)
        
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, proba[:, 1])
            else:
                auc = roc_auc_score(y_true, proba, multi_class='ovr', average='macro')
        except ValueError:
            auc = float('nan')
            
        logger.info("Symbolic Fidelity: Accuracy = %.4f, ROC-AUC = %.4f", acc, auc)
        return {"accuracy": acc, "roc_auc": auc}
        
    finally:
        pruned_model.use_symbolic(original_state)


def _round_sympy_expr(expr: sympy.Expr, threshold: float = 1e-4) -> sympy.Expr:
    """Traverses a SymPy AST zeroing out negligible numerical floating coefficients."""
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Float):
            if abs(float(node)) < threshold:
                expr = expr.subs(node, 0)
            else:
                expr = expr.subs(node, round(float(node), 4))
    return expr


def extract_symbolic_expression(
    kan_model: KAN, 
    feature_names: List[str], 
    lib: Optional[List[str]] = None,
    finetune_epochs: int = 30,
    finetune_lr: float = 0.0004,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    r2_threshold: float = 0.90,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """Prune a KAN, apply symbolic regression, optionally fine-tune, and extract simplified SymPy equations."""
    if lib is None:
        lib = ["x", "x^2", "x^3", "log", "exp", "tanh", "abs", "sqrt"]

    # Protect the caller's model reference natively explicitly correctly magically smoothly formally structurally
    device = next(kan_model.parameters()).device if hasattr(kan_model, 'parameters') else torch.device('cpu')
    logger.info("Cloning and Pruning KAN model …")
    pruned = copy.deepcopy(kan_model).to(device)
    pruned = pruned.prune()

    logger.info("Fitting symbolic functions from lib=%s …", lib)
    pruned.auto_symbolic(lib=lib)

    # Post-Symbolification Fine-Tuning explicitly resolving affine coefficient alignments
    if X_train is not None and y_train is not None:
        logger.info("Fine-tuning affine parameters for %d epochs at LR=%.5f...", finetune_epochs, finetune_lr)
        dataset = {
            'train_input': torch.tensor(X_train.values, dtype=torch.float32).to(device),
            'train_label': torch.tensor(y_train.values, dtype=torch.long).to(device),
            'test_input': torch.tensor(X_train.values, dtype=torch.float32).to(device),
            'test_label': torch.tensor(y_train.values, dtype=torch.long).to(device)
        }
        try:
            # PyKAN native trainer targeting strictly continuous affine structural alignments
            pruned.fit(dataset, opt="Adam", lr=finetune_lr, steps=finetune_epochs)
        except Exception as e:
            logger.warning("PyKAN fine-tuning iteration failed. Model may be strictly locked: %s", e)
    else:
        logger.warning("No PyTorch dataset provided. Post-symbolic affine fine-tuning SKIPPED.")

    expressions: Dict[str, Dict[str, str]] = {}
    sympy_expressions: Dict[str, Dict[str, str]] = {}
    r2_scores: Dict[str, float] = {}
    
    try:
        num_layers = len(pruned.symbolic_fun)
    except AttributeError as e:
        raise AttributeError(f"PyKAN structural 'symbolic_fun' missing. Version incompatibility: {e}")

    prev_layer_nodes_sym = [sympy.Symbol(f) for f in feature_names]
    
    for layer_idx in range(num_layers):
        try:
            layer = pruned.symbolic_fun[layer_idx]
            funs_name = layer.funs_name
            affine = layer.affine
            r2_mat = layer.r2
        except AttributeError as e:
            raise AttributeError(f"PyKAN affine attribute missing. Expected 'funs_name', 'affine', 'r2': {e}")
            
        n_out = len(funs_name)
        n_in = len(funs_name[0])
        
        current_layer_nodes_sym = []
        layer_eqs_str = {}
        layer_eqs_sym = {}
        
        for out_node in range(n_out):
            terms_sym = []
            for in_node in range(n_in):
                fn = funs_name[out_node][in_node]
                
                if fn in ("0", "0.0"):
                    continue
                    
                r2 = r2_mat[out_node, in_node].item()
                r2_scores[f"L{layer_idx}_{in_node}->{out_node}"] = r2
                
                if r2 < r2_threshold:
                    msg = f"Poor symbolic fit at layer {layer_idx}, edge ({in_node} -> {out_node}): '{fn}' with R^2 = {r2:.4f}"
                    if strict_mode:
                        raise ValueError(f"Strict Mode R2 Rejection: {msg}")
                    logger.warning(msg)
                    
                a, b, c, d = affine[out_node, in_node].tolist()
                
                inner_sym = prev_layer_nodes_sym[in_node]
                inner_term_sym = a * inner_sym + b
                
                if fn == "x":
                    f_val_sym = inner_term_sym
                elif fn == "x^2":
                    f_val_sym = inner_term_sym**2
                elif fn == "x^3":
                    f_val_sym = inner_term_sym**3
                elif fn == "log":
                    f_val_sym = sympy.log(inner_term_sym)
                elif fn == "exp":
                    f_val_sym = sympy.exp(inner_term_sym)
                elif fn == "tanh":
                    f_val_sym = sympy.tanh(inner_term_sym)
                elif fn == "abs":
                    f_val_sym = sympy.Abs(inner_term_sym)
                elif fn == "sqrt":
                    f_val_sym = sympy.sqrt(inner_term_sym)
                else:
                    f_val_sym = sympy.Function(fn)(inner_term_sym)
                    
                term_sym = c * f_val_sym + d
                terms_sym.append(term_sym)
                
            node_name = f"z_{out_node}" if layer_idx == num_layers - 1 else f"H_{layer_idx}_{out_node}"
            
            if terms_sym:
                raw_expr = sum(terms_sym)
                simplified_expr = sympy.simplify(raw_expr)
                simplified_expr = _round_sympy_expr(simplified_expr)
            else:
                simplified_expr = sympy.sympify(0.0)
                
            current_layer_nodes_sym.append(simplified_expr)
            layer_eqs_sym[node_name] = str(simplified_expr)
            layer_eqs_str[node_name] = str(simplified_expr)

        expressions[f"layer_{layer_idx}"] = layer_eqs_str
        sympy_expressions[f"layer_{layer_idx}"] = layer_eqs_sym
        prev_layer_nodes_sym = current_layer_nodes_sym

    # Dynamic Multi-Class Softmax explicitly mapping arbitrary classification bounds
    n_out_final = len(pruned.symbolic_fun[-1].funs_name)
    denom = " + ".join([f"exp(z_{k})" for k in range(n_out_final)])
    
    if n_out_final == 2:
        final_eq = f"P(Class_1) = exp(z_1) / ({denom})"
    else:
        final_eq = "\n".join([f"P(Class_{k}) = exp(z_{k}) / ({denom})" for k in range(n_out_final)])

    feat_map = {i: f for i, f in enumerate(feature_names)}

    result = {
        "pruned_model": pruned,
        "expressions": sympy_expressions,
        "final_equation": final_eq,
        "feature_map": feat_map,
        "r2_scores": r2_scores,
        "lib": lib
    }

    logger.info("Symbolic extraction complete (%d layers parsed).", num_layers)
    return result


def compute_feature_importance(expr: Dict[str, Any], feature_names: List[str]) -> Dict[str, float]:
    """Calculate aggregated absolute feature mapping importance recursively identifying critical drivers."""
    if "pruned_model" not in expr:
        return {}
        
    try:
        layer0 = expr["pruned_model"].symbolic_fun[0]
        affine0 = layer0.affine  # [n_out, n_in, 4] where a=0, b=1, c=2, d=3
        
        importance = {}
        # Importance proxy for the first layer natively mapping: |c * a| representing explicit amplitude variance slopes
        for in_node, feat in enumerate(feature_names):
            total_impact = 0.0
            for out_node in range(len(layer0.funs_name)):
                a = affine0[out_node, in_node, 0].item()
                c = affine0[out_node, in_node, 2].item()
                total_impact += abs(a * c)
            importance[feat] = total_impact
            
        # Sort structurally highest to lowest specifically ranking
        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    except AttributeError:
        logger.warning("Could not compute importance. Affine PyKAN attributes missing natively.")
        return {}


def export_latex(expr: Dict[str, Any]) -> str:
    """Exports globally configured mathematically correct analytical SymPy strings into LaTeX formats."""
    latex_out = "\\begin{align*}\n"
    for layer_name, layer_eqs in expr["expressions"].items():
        latex_out += f"\\text{{--- {layer_name.upper()} ---}} \\\\\n"
        for node_name, equation in layer_eqs.items():
            sym_eq = sympy.sympify(equation)
            latex_str = sympy.latex(sym_eq)
            latex_out += f"{node_name} &= {latex_str} \\\\\n"
    
    latex_out += "\\text{--- FINAL PREDICTION ---} \\\\\n"
    # Hand-parse the pseudo-softmax equations into LaTeX purely for structural research parsing
    for line in expr["final_equation"].split('\n'):
        if line.strip():
            left, right = line.split("=")
            left = left.strip().replace("P(Class_", "P(\\text{Class}_{").replace(")", "})")
            latex_out += f"{left} &= \\frac{{{right.split('/')[0].strip()}}}{{{right.split('/')[1].strip()}}} \\\\\n"
            
    latex_out += "\\end{align*}"
    return latex_out


def save_expressions(expr: Dict[str, Any], path: str) -> None:
    """Serializes explicit symbolic outputs mapping formulas and mappings into structural JSON boundaries natively."""
    out_dict = {
        "timestamp": datetime.now().isoformat(),
        "expressions": expr["expressions"],
        "final_equation": expr["final_equation"],
        "feature_map": expr["feature_map"],
        "r2_scores": expr.get("r2_scores", {}),
        "lib": expr.get("lib", [])
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=4)
    logger.info("Saved symbolic expressions formally to %s", path)


def load_expressions(path: str) -> Dict[str, Any]:
    """Retrieves standard extracted structural mathematical arrays seamlessly via JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info("Loaded symbolic expressions strictly from %s", path)
    return data


def print_trading_equations(expr: Dict[str, Any]) -> str:
    """Pretty-print the rigorously extracted mathematical KAN equations formally via standard log frameworks."""
    out = []
    out.append("\n" + "=" * 80)
    out.append(" EXTRACTED MATHEMATICAL TRADING EQUATIONS")
    out.append("=" * 80)

    for layer_name, layer_eqs in expr["expressions"].items():
        out.append(f"\n--- {layer_name.upper()} ---")
        for node_name, equation in layer_eqs.items():
            out.append(f"{node_name} = {equation}\n")

    out.append("\n--- FINAL PREDICTION (Softmax) ---")
    out.append(expr["final_equation"])

    out.append("\n" + "-" * 40)
    out.append("Input Node Mapping (Top -> Bottom):")
    for idx, feat in expr["feature_map"].items():
        out.append(f"  Node {idx}: {feat}")
    out.append("=" * 80)
    
    final_str = "\n".join(out)
    logger.info(final_str)
    return final_str
