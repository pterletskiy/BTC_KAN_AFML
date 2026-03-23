---
name: skill_kan_modeling
description: >
  Governs KAN (Kolmogorov-Arnold Network) architecture, training, pruning,
  probability calibration, symbolic extraction, and baseline comparison for
  binary Bitcoin direction classification on daily time-bar data.
  Use this skill whenever working on: KAN architecture selection, spline grid
  refinement, pykan/efficient-kan training loops, pruning protocols, symbolic
  math extraction (suggest_symbolic / auto_symbolic), KAN vs baseline comparison,
  meta-labeling with KAN as primary or secondary model, or feeding KAN
  calibrated probabilities into the AFML bet sizing formula.
  Always read skill_MLdP_pipeline.md first — this skill depends on it.
---

# KAN Time-Series Modeling Skill
## Binary Direction Classification for Daily BTC Data

## 0. Scope and Dependencies

This skill governs **modeling only**. It assumes `skill_MLdP_pipeline.md`
has already produced:
- Leak-free feature matrix with FFD-transformed continuous series.
- TBM labels with `t0`, `t1`, and uniqueness-based sample weights.
- Purged/embargoed fold indices (Purged K-Fold or CPCV).
- Feature importance rankings (MDI, MDA, SFI) — required before training.
- `models/trial_registry.json` — must be updated after every configuration tested.

**Library:** default to `pykan` (`pip install pykan`). If training speed
becomes prohibitive, switch to `efficient-kan` and document the change.
All API references in this skill assume `pykan` conventions.

---

## 1. Modeling Objective

- Task: **binary classification** — predict `P(label = +1)`.
- Output: calibrated probability, not a raw score.
- Primary objective: out-of-sample discrimination and calibration.
- The symbolic formula extracted at the end is the thesis contribution;
  classification performance is its prerequisite.

---

## 2. Architecture Selection

### 2.1 Compact First Principle

Daily BTC has ~2,000–3,000 usable training observations after CUSUM
filtering. Overparameterization is lethal. Start small.

**Default candidate grid for daily tabular data:**

| ID | Shape | Grid points | Notes |
|---|---|---|---|
| K1 | `[F, 4, 1]` | 5 | Baseline: 1 hidden layer, 4 nodes |
| K2 | `[F, 8, 1]` | 5 | Wider hidden layer |
| K3 | `[F, 4, 4, 1]` | 5 | 2 hidden layers |
| K4 | `[F, 8, 4, 1]` | 5 | Wider 2-layer |

Where `F` = number of features after importance-based dropping.

Search this grid in full. Do not add architectures mid-search without
registering the new configuration in `trial_registry.json`.

### 2.2 Input Format

- Input: 2D tensor of shape `(N_events, F)` — one row per CUSUM event.
- Do not construct rolling windows unless using a sequence-aware KAN
  variant. Rolling-window reshaping is not the default.
- Features must already be scaled (`RobustScaler` from the inner CV fold).

### 2.3 Spline Order

- Default: `k=3` (cubic B-splines). Sufficient for smooth activation
  functions and symbolic approximation.
- Try `k=2` if training is unstable. Document any change.

---

## 3. Training Protocol — Two-Phase (Grid Coarsening → Refinement)

KANs are trained in **two phases**. Skipping phase 2 degrades symbolic
extraction quality significantly.

### Phase 1 — Coarse Grid Training

```python
model = KAN(width=[F, 4, 1], grid=5, k=3, seed=42)
model.train(
    dataset,
    opt="Adam",
    lr=1e-3,
    steps=200,
    lamb=1e-4,          # L1 sparsity on activations
    lamb_entropy=2.0,   # Entropy regularization — encourages simple functions
    loss_fn=bce_with_weights,
    metrics=[auc_metric],
)
```

### Phase 2 — Grid Refinement

After phase 1 converges, refine the spline grid without reinitializing weights:

```python
model = model.refine(grid=20)   # increase grid resolution
model.train(
    dataset,
    opt="Adam",
    lr=5e-4,
    steps=100,
    lamb=1e-4,
    lamb_entropy=2.0,
)
```

Grid refinement allows the splines to capture finer functional detail
before symbolic extraction. Use `grid=20` as default; increase to `grid=50`
only if symbolic candidates are still imprecise.

### 3.1 Loss Function

Use binary cross-entropy with sample weights from the preprocessing skill:

```python
def bce_with_weights(pred, target, weights):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    return (bce * weights).mean()
```

Never use MSE. If class imbalance is severe after CUSUM filtering,
apply `pos_weight = n_neg / n_pos` inside `BCEWithLogitsLoss`.

### 3.2 Early Stopping

Monitor validation AUC. Stop if AUC does not improve for `patience=20`
steps (phase 1) or `patience=10` steps (phase 2).
Save the checkpoint with highest validation AUC, not lowest loss.

### 3.3 Regularization Priority

Apply in this order before increasing architecture size:

1. L1 on spline activations (`lamb`): start at `1e-4`.
2. Entropy regularization (`lamb_entropy`): start at `2.0`.
3. Early stopping with generous patience.
4. Weight decay on the optimizer (`1e-5`).

If after all four the model still overfits, reduce architecture width,
not regularization.

---

## 4. Imbalance Handling

- Always report class distribution of TBM labels before training.
- If `|n_+1 - n_-1| / N > 0.15`, apply `pos_weight` or sample weights.
- Sample weights from `skill_MLdP_pipeline.md §4` are passed directly
  to the loss function. Do not recompute them here.
- Threshold tuning: grid search on validation fold only, step size 0.01.
  Report the chosen threshold. Apply it unchanged to the test fold.

---

## 5. Baseline Models

Every KAN result requires comparison against baselines on **identical**
features, labels, splits, and evaluation protocol.

### 5.1 Required Baselines

| Baseline | Library | Notes |
|---|---|---|
| Logistic Regression | `sklearn` | L2 regularized, calibrated with Platt |
| Random Forest | `sklearn` | 500 trees, `class_weight='balanced'` |
| LightGBM | `lightgbm` | Binary objective, `is_unbalance=True` |
| MLP (2-layer) | `torch` | Same input/output shape as KAN, BCEWithLogitsLoss |

**Why no LSTM/1D-CNN:** daily tabular data does not have a natural sequence
structure after CUSUM event sampling. Sequence models would require
arbitrary window construction and introduce additional hyperparameters
that make comparison unfair.

### 5.2 Hyperparameter Budget Rule

Each baseline gets at most **one** hyperparameter search round with
`n_iter=20` randomized search on the validation fold.
Results are logged to `trial_registry.json` under the baseline name.

---

## 6. Probability Calibration

Raw KAN sigmoid outputs are not guaranteed to be calibrated.

### 6.1 When to Calibrate

Calibrate if the Brier skill score on the validation fold is worse
than the logistic regression baseline, or if the reliability diagram
shows systematic over- or under-confidence.

### 6.2 Method

Use **Platt scaling** (logistic regression on raw logits) as the default.
Fit the calibrator on a held-out calibration partition of the training
fold (20% of training fold, never the test fold).

```python
from sklearn.calibration import CalibratedClassifierCV
# or manually: fit LogisticRegression on (raw_logits, y_val_calib)
```

### 6.3 Calibration Evaluation

Report: Brier score, ECE (15-bin), reliability diagram.
The calibrated model is the one fed into the bet sizing formula
from `skill_MLdP_pipeline.md §9`.

---

## 7. Pruning Protocol

Pruning must happen **before** symbolic extraction.
Pruning on an uncalibrated or unstable model is wasted effort.

### 7.1 Two-Stage Pruning

**Stage 1 — Node pruning:**

```python
model.prune_node(threshold=5e-2)   # remove nodes with low attribution
```

**Stage 2 — Edge pruning:**

```python
model.prune_edge(threshold=5e-3)   # remove splines with near-zero magnitude
```

After pruning, retrain for a short fine-tuning pass (50 steps, low LR)
to restore any performance loss from pruning.

### 7.2 Pruning Gate

Accept the pruned model only if its validation AUC drops by less than
`0.02` relative to the pre-pruned model. If the drop exceeds this,
loosen the pruning threshold by 50% and retry.

### 7.3 Function Stability Check

Train the same architecture on each of the CPCV training folds.
Plot the learned spline functions for the top-3 features across folds.
If the functions change shape qualitatively across folds, the model is
regime-dependent. Document this — do not suppress it.

---

## 8. Symbolic Extraction (Primary Thesis Contribution)

Symbolic extraction is the methodological core of this thesis.
Treat it with the same rigor applied to the classification validation.

### 8.1 Prerequisites (all must pass before extraction)

- [ ] Model passes pruning gate (§7.2).
- [ ] Validation AUC ≥ logistic regression baseline.
- [ ] DSR > 0.95 on at least one CPCV path (from `skill_MLdP_pipeline.md §10`).
- [ ] Function stability confirmed (§7.3).

### 8.2 Extraction Workflow (pykan)

**Step 1 — Suggest candidates per edge:**

```python
model.suggest_symbolic(
    lib=['x', 'x^2', 'x^3', 'exp', 'log', 'tanh', 'sin', 'abs', 'sqrt'],
    a_range=(-5, 5), b_range=(-5, 5),
    verbose=True
)
```

This outputs a ranked list of symbolic candidates per spline edge,
scored by R² fit to the learned spline.

**Step 2 — Fix candidates with R² > 0.97:**

```python
model.fix_symbolic(layer_id, node_in, node_out, fn_name)
```

Only fix edges where the top candidate has `R² ≥ 0.97`.
For edges below this threshold, keep the spline — do not force a
symbolic assignment on a poor fit.

**Step 3 — Auto-symbolify remaining edges (optional):**

```python
model.auto_symbolic(lib=['x', 'x^2', 'exp', 'log', 'tanh', 'abs'])
```

Use only after manual review of Step 1 candidates.

**Step 4 — Extract the formula:**

```python
formula = model.symbolic_formula()
```

### 8.3 Symbolic Fidelity Tests (Required)

After extraction, run all three fidelity checks:

| Test | Pass Criterion |
|---|---|
| AUC parity | AUC(symbolic) ≥ AUC(KAN) − 0.03 |
| Log-loss parity | LogLoss(symbolic) ≤ LogLoss(KAN) + 0.05 |
| Rank correlation | Spearman ρ(symbolic probs, KAN probs) ≥ 0.95 |

If any test fails, do not claim the formula as equivalent.
Report the gap and label the formula as an approximation.

### 8.4 Regime Generalization Test (Thesis Differentiator)

This is what separates a rigorous thesis from a demonstration.

1. Extract the symbolic formula on the full training period.
2. Split the test period into sub-regimes by SADF signal:
   - **Explosive regime:** SADF > critical value (bubble / strong trend).
   - **Stable regime:** SADF ≤ critical value (mean-reverting / quiet).
3. Evaluate the symbolic formula's AUC separately on each sub-regime.
4. Compare to the full-period AUC.

If the formula degrades sharply in one regime, it has captured a regime-
specific pattern, not a universal structure. Report this finding — it is
scientifically valid and interesting.

### 8.5 Formula Documentation

For the extracted formula, report:
- The full symbolic expression.
- Which input features appear in it (and which were pruned away).
- The financial interpretation of each retained function (e.g., "log of
  lagged volume" suggests the model uses volume acceleration as a signal).
- The regime generalization results from §8.4.

---

## 9. Meta-Labeling Integration

KAN can serve as either the **primary** model (predicts direction) or
the **secondary** model (filters false positives from a simpler primary).

### 9.1 KAN as Primary Model

- Symmetric TBM barriers, side labels `{+1, -1}`.
- Optimize for recall; accept lower precision.
- Pass false positive rate and F1 to the secondary model stage.

### 9.2 KAN as Secondary (Meta-Labeling) Model

- Meta-labels: `{1=true positive, 0=false positive}` from primary's calls.
- Use `class_weight='balanced'` — false positives typically dominate.
- Score with F1. AUC is a secondary check.
- Output: `P(true positive)` → fed into bet sizing.

The decision of which role to assign should be based on which setup
achieves higher F1 on the validation fold. Document both attempts.

---

## 10. CPCV Integration

Each CPCV training/test fold combination is a separate model fit.

**Per CPCV path:**
1. Train KAN on the fold's training observations.
2. Evaluate on the path's test observations.
3. Record: AUC, F1, Brier, annualized SR of the implied strategy.
4. Write to `trial_registry.json`.

**Aggregate across paths:**
- Report distribution of SR across paths (mean, std, min, max, fraction > 0).
- Report distribution of AUC across paths.
- Use SR distribution to compute DSR per `skill_MLdP_pipeline.md §10.2`.

A KAN result is only reportable as the thesis final result if DSR > 0.95.

---

## 11. Validation and Selection Criteria

### 11.1 Primary Metrics (in priority order)

1. AUC (discrimination ability)
2. F1 (precision/recall balance under imbalance)
3. Brier score (calibration)
4. DSR (statistical reliability after multiple trials)

### 11.2 Secondary Metrics

- MCC (balanced accuracy for imbalanced classes)
- Log-loss
- Precision @ chosen threshold
- Recall @ chosen threshold

### 11.3 Model Selection Gate

Accept a KAN configuration as the final model only if:
- Validation AUC ≥ LightGBM baseline AUC.
- DSR > 0.95 across CPCV paths.
- Brier score ≤ logistic regression baseline Brier score.

If no KAN configuration passes all three, report the best KAN honestly
alongside the best baseline and discuss the gap in the thesis.

---

## 12. Output Artefacts

| File | Content |
|---|---|
| `models/kan_*.pt` | Trained KAN checkpoint per fold |
| `models/kan_pruned_*.pt` | Pruned KAN checkpoint |
| `models/calibrator_*.pkl` | Platt scaler per fold |
| `models/symbolic_formula.json` | Extracted formula, R², fidelity test results |
| `models/trial_registry.json` | All configs tested + SR for DSR (shared with MLdP skill) |
| `models/baseline_*.pkl` | Baseline model checkpoints |
| `models/backtest_stats.json` | SR distribution across CPCV paths (shared with MLdP skill) |
| `models/spline_plots/` | Edge function plots per fold for stability check |
| `models/regime_fidelity.json` | Symbolic formula AUC per SADF regime |

---

## 13. Forbidden Actions

| Action | Reason |
|---|---|
| Skip grid refinement (Phase 2) | Degrades symbolic extraction quality |
| Extract symbols before pruning | Redundant edges corrupt the formula |
| Fix symbolic edges with R² < 0.97 | Forces a false simplification |
| Compare KAN to baselines on different feature sets | Invalidates the comparison |
| Report SR without DSR | Cannot distinguish skill from trial count |
| Use MSE or Huber as primary loss | Wrong objective for binary classification |
| Calibrate on test set | Look-ahead bias |
| Skip regime generalization test | Incomplete thesis methodology |
| Report single-fold AUC as final result | Ignores variability across CPCV paths |
| Use LSTM/1D-CNN baselines on daily events | Requires arbitrary window construction |

---

## 14. Default Workflow (Sequential Steps)

```
1.  Receive: features, labels, sample_weights, fold_indices
             from skill_MLdP_pipeline.md
2.  Confirm: feature importance rankings available
3.  Drop:    bottom-quartile features (MDI+MDA+SFI consensus)
4.  Loop over CPCV fold combinations:
    a.  Scale features (RobustScaler on training fold only)
    b.  Train baseline grid (LogReg, RF, LightGBM, MLP)
    c.  Train KAN grid [K1–K4] — Phase 1 (coarse grid)
    d.  Grid refinement — Phase 2 (grid=20)
    e.  Select best KAN by validation AUC
    f.  Calibrate probabilities (Platt on held-out train partition)
    g.  Tune decision threshold on validation fold
    h.  Record SR, AUC, F1, Brier → trial_registry.json
5.  Select best KAN across all folds by median validation AUC
6.  Prune (node then edge), fine-tune 50 steps
7.  Verify function stability across folds (spline plots)
8.  Attempt symbolic extraction (§8.2)
9.  Run symbolic fidelity tests (§8.3)
10. Run regime generalization test (§8.4)
11. Compute DSR from trial_registry.json
12. Report: SR distribution, DSR, AUC, symbolic formula
```