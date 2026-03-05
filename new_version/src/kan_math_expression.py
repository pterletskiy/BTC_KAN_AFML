"""
kan_math_expression.py — KAN symbolic regression / equation extraction.

**Experimental module** for extracting interpretable mathematical formulas
from a trained PyKAN model.  Kept separate from ``models.py`` to allow
independent experimentation without affecting the production pipeline.

Usage from the Master Notebook::

    from src.kan_math_expression import extract_symbolic_expression, print_trading_equations

    expr = extract_symbolic_expression(
        kan_model=pruned_or_trained_kan,
        feature_names=top_10_features,
        lib=['x', 'x^2', 'x^3', 'sin', 'exp'],
    )
    print_trading_equations(expr)
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from kan import KAN

logger = logging.getLogger(__name__)


def extract_symbolic_expression(
    kan_model: KAN,
    feature_names: List[str],
    lib: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Prune a KAN and extract symbolic mathematical expressions.

    Steps:
      1. Prune the network (remove near-zero activation edges).
      2. Fit symbolic functions from ``lib`` to the remaining edges.
      3. Parse the first hidden layer to build human-readable equations
         for each hidden node.

    Parameters
    ----------
    kan_model : KAN
        A **trained** PyKAN model.
    feature_names : list of str
        Names of the input features (in order).
    lib : list of str, optional
        Allowed symbolic function library.
        Default: ``['x', 'x^2', 'x^3', 'sin', 'exp']``.

    Returns
    -------
    dict
        Keys:

        - ``pruned_model`` — the pruned KAN.
        - ``hidden_node_equations`` — dict mapping ``"H_0"``, ``"H_1"``, …
          to their formula strings.
        - ``final_equation`` — the output-layer formula string.
        - ``feature_map`` — mapping of node index → feature name.
        - ``raw_funs_name`` — raw function names from the symbolic layer
          (for advanced inspection).
    """
    if lib is None:
        lib = ["x", "x^2", "x^3", "sin", "exp"]

    # 1. Prune
    logger.info("Pruning KAN model …")
    pruned = kan_model.prune()

    # 2. Auto-symbolic fitting
    logger.info("Fitting symbolic functions from lib=%s …", lib)
    pruned.auto_symbolic(lib=lib)

    # 3. Parse hidden-layer equations
    layer = pruned.symbolic_fun[0]
    n_inputs = len(feature_names)

    # Determine number of hidden nodes from funs_name (rows = hidden nodes)
    n_hidden = len(layer.funs_name)

    hidden_eqs: Dict[str, str] = {}
    weight_idx = 1

    for h in range(n_hidden):
        terms: List[str] = []
        for i in range(n_inputs):
            fn = layer.funs_name[h][i]
            feat = feature_names[i]

            if fn in ("0", "0.0"):
                continue

            if fn == "x":
                terms.append(f"w_{weight_idx} * [{feat}]")
            elif fn == "x^2":
                terms.append(f"w_{weight_idx} * ([{feat}])^2")
            elif fn == "x^3":
                terms.append(f"w_{weight_idx} * ([{feat}])^3")
            else:
                terms.append(f"{fn}(w_{weight_idx} * [{feat}])")
            weight_idx += 1

        if terms:
            eq = " + \n      ".join(terms) + f" + b_{h}"
        else:
            eq = "0"

        hidden_eqs[f"H_{h}"] = eq

    # 4. Output layer equation
    out_terms = " + ".join(
        f"W_out{h} * H_{h}" for h in range(n_hidden)
    )
    final_eq = f"P(Up) = Sigmoid( {out_terms} + B_out )"

    # 5. Feature map
    feat_map = {i: f for i, f in enumerate(feature_names)}

    result = {
        "pruned_model": pruned,
        "hidden_node_equations": hidden_eqs,
        "final_equation": final_eq,
        "feature_map": feat_map,
        "raw_funs_name": {
            h: list(layer.funs_name[h]) for h in range(n_hidden)
        },
    }

    logger.info("Symbolic extraction complete (%d hidden nodes)", n_hidden)
    return result


def print_trading_equations(expr: Dict[str, Any]) -> None:
    """Pretty-print the extracted KAN trading equations.

    Parameters
    ----------
    expr : dict
        Output of :func:`extract_symbolic_expression`.
    """
    print("\n" + "=" * 80)
    print(" EXTRACTED MATHEMATICAL TRADING EQUATIONS")
    print("=" * 80)

    for node_name, equation in expr["hidden_node_equations"].items():
        step = int(node_name.split("_")[1]) + 1
        print(f"\n--- STEP {step}: Calculate {node_name} ---")
        print(f"{node_name} = {equation}")

    print(f"\n--- FINAL PREDICTION ---")
    print(expr["final_equation"])

    print("\n" + "-" * 40)
    print("Input Node Mapping (Top → Bottom):")
    for idx, feat in expr["feature_map"].items():
        print(f"  Node {idx}: {feat}")

    print("=" * 80)
