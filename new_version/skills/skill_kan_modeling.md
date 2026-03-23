# KAN Time-Series Modeling Skill
## Binary Direction Classification for Financial Data

## 0. Scope
This skill governs **modeling only** for binary market-direction prediction on preprocessed financial time series.
It assumes the upstream preprocessing skill has already produced:
* leak-free features,
* leakage-safe binary labels,
* training/validation/test splits,
* optional sample weights,
* and any required stationarity transforms.

This skill does **not** handle raw feature engineering, labeling, fractional differentiation, purging/embargo logic, or data leakage prevention at the preprocessing stage. Those belong to the preprocessing skill file.

---

## 1. Modeling Objective
* The task is **binary classification**.
* Predict the probability that the target direction is **up** rather than down.
* The model output must be interpreted as a **class probability, not as a regression value**.
* The main objective is to optimize **out-of-sample discrimination and calibration, not raw regression error**.
* The final decision threshold must be tuned on validation data only, never on the test set.

---

## 2. KAN Architecture for Sequential Financial Data

### 2.1 Compact First Principle
* Use **small and parsimonious KAN architectures** by default.
* Financial time series are noisy and low signal-to-noise, so the first candidate architecture must be shallow and narrow.
* Start from a small grid of architectures rather than one fixed shape.
* Prefer the smallest architecture that is stable under validation and does not collapse under class imbalance.

### 2.2 Allowed Architecture Search
* Search over a small set of compact KANs.
* Example starting points may include:
  * a shallow KAN for tabular lagged features,
  * a compact KAN with 1 to 3 hidden layers,
  * a few narrow width choices.
* The agent must not assume that deeper or wider is better.
* The agent must report the selected architecture and justify it using validation performance and stability.

### 2.3 Input Format
* Use the preprocessed feature matrix provided by the preprocessing skill.
* If the input is tabular lagged features, treat it as a binary classification problem on one feature vector per observation.
* If the input is sequential windows, preserve the window structure only if the model variant explicitly supports sequence input.
* Do not reshape data arbitrarily; the input structure must be consistent with the chosen model family.

### 2.4 KAN Layer Behavior
* Prefer KAN variants that expose learnable edge functions and remain interpretable.
* Use spline-based or equivalent KAN implementations when available.
* Keep the model simple enough that learned functions can be visualized and inspected after training.
* If a variant introduces extra architectural components, they must be documented and compared fairly.

---

## 3. Training Objective and Optimization

### 3.1 Classification Loss
* Use **binary cross-entropy with logits** as the default loss for binary direction prediction.
* If the classes are imbalanced, use **class weighting** or **focal loss**.
* Do not use MSE or Huber loss as the primary objective for a binary direction classifier.
* If calibration is planned, train the base classifier first, then calibrate probabilities afterward.

### 3.2 Optimization
* Use a standard gradient-based optimizer such as Adam or AdamW.
* Track training and validation loss, but make model selection based on validation classification metrics, not loss alone.
* Apply early stopping when validation performance stops improving for a predefined patience window.

### 3.3 Regularization
* Regularize aggressively, because financial data are noisy and labels are often weak.
* Acceptable regularizers include:
  * weight decay,
  * sparsity penalties,
  * pruning,
  * dropout if supported by the implementation,
  * early stopping.
* Prefer simpler regularization before increasing architectural complexity.
* If the KAN implementation supports edge/function pruning, use it after training to simplify the model.

---

## 4. Imbalance Handling

### 4.1 Class Balance Awareness
* Always inspect the post-labeling class distribution before training.
* Report the share of up, down, and any neutral observations if neutral labels are retained upstream.
* Do not train a binary classifier without checking whether the positive class is rare or dominant.

### 4.2 Loss Weighting
* If the class distribution is imbalanced, apply:
  * class weights, or
  * focal loss, or
  * sample weights passed from the preprocessing stage.
* Sample weights must be applied only as they were produced upstream and must not be recomputed inside the model skill unless explicitly documented.

### 4.3 Thresholding
* Do not assume 0.5 is always the best decision threshold.
* Tune the classification threshold on the validation fold only.
* The chosen threshold must be reported and reused unchanged on the test fold.

---

## 5. Probability Calibration

### 5.1 Calibration Requirement
* The model must output probabilities that can be interpreted as estimated likelihoods of an upward move.
* Raw model outputs are not automatically valid probabilities unless the implementation explicitly guarantees calibration.

### 5.2 Calibration Methods
* Apply post-hoc calibration on validation data only when needed.
* Acceptable methods include:
  * Platt scaling,
  * isotonic regression,
  * or another held-out calibration method.
* Calibration must be fitted only on training or validation data designated for calibration, never on the test set.

### 5.3 Calibration Evaluation
* Evaluate calibration using:
  * Brier score,
  * calibration curves,
  * reliability diagrams,
  * and, if available, expected calibration error.
* Prefer the calibrated probability model when decision thresholds and trading rules depend on confidence.

---

## 6. Validation and Model Selection

### 6.1 Leak-Free Validation
* Evaluation must use the preprocessed leakage-safe split strategy provided by the upstream skill.
* Do not fit the model on data that contains future information relative to the evaluation fold.
* Do not use random cross-validation.

### 6.2 Selection Criteria
* Primary selection metrics:
  * AUC,
  * balanced accuracy,
  * F1,
  * MCC,
  * and calibration quality.
* Secondary selection metrics:
  * precision,
  * recall,
  * log loss,
  * Brier score.
* If the thesis later includes trading results, those should be evaluated separately from classification metrics.

### 6.3 Robustness
* Use multiple evaluation folds if available.
* Report both mean performance and variability across folds.
* Prefer the model that is stable, not just the model with the highest single-fold score.

---

## 7. Baseline Models

### 7.1 Baseline Comparison Requirement
* Every KAN result must be compared against strong baselines.
* KAN is not meaningful unless compared to simpler and more established models on the same preprocessed data.

### 7.2 Recommended Baselines
* The modeling skill should support at least the following baseline families:
  * AR-Logistic,
  * plain Logistic Regression,
  * Random Forest,
  * XGBoost or LightGBM,
  * 1D-CNN,
  * LSTM.

### 7.3 Fair Comparison Rule
* All baselines must use the same features, the same labels, the same split logic, and the same evaluation protocol.
* Hyperparameter tuning must be limited and documented.
* Do not give KAN a different preprocessing path than the baselines.

---

## 8. Pruning and Interpretability

### 8.1 Pruning
* After training, prune unimportant nodes and edges if the KAN implementation supports it.
* Prefer simpler pruned models when performance remains comparable.
* Pruning should reduce complexity without materially degrading validation performance.

### 8.2 Visual Inspection
* Always inspect learned edge functions when available.
* Plot the most important learned functions and compare them with financial intuition.
* Interpretability is a secondary contribution only if the predictive performance remains credible.

### 8.3 Function Stability
* If learned functions vary wildly across folds, the model is likely unstable.
* Prefer models whose learned functions are qualitatively similar across validation splits.

---

## 9. Optional Symbolic Extraction

### 9.1 When to Use
* Symbolic extraction is optional and should be treated as an interpretability extension, not a training requirement.
* Attempt symbolic extraction only after the classifier has shown stable validation performance.

### 9.2 Pruning Before Extraction
* Prune the model before attempting symbolic extraction.
* Remove weak or redundant edges first.
* Do not attempt to symbolify a highly redundant or unstable network.

### 9.3 Fidelity Check
* If a symbolic formula is extracted, validate that it preserves the predictive behavior of the trained KAN.
* Compare the symbolic approximation to the original KAN using held-out classification performance and probability ranking behavior.
* Use metrics appropriate to classification, such as AUC, log loss, and calibration quality.
* If a regression-style fidelity metric is used internally by the implementation, it is only a secondary check and not the main success criterion for this thesis.

---

## 10. Output Standards
The modeling pipeline must output:
* the trained KAN model,
* training and validation curves,
* calibrated probabilities if calibration was applied,
* chosen threshold,
* fold-level metrics,
* final test metrics,
* pruning summary,
* interpretability plots if available,
* and a reproducible record of hyperparameters.

---

## 11. Forbidden Actions
* Do not treat binary direction prediction as a regression problem.
* Do not optimize primarily for MSE or Huber loss.
* Do not calibrate on the test set.
* Do not tune thresholds on the test set.
* Do not compare KAN to baselines with different feature sets or split logic.
* Do not force symbolic extraction before the classifier is stable.
* Do not use oversized architectures by default.
* Do not assume the raw KAN output is already a valid probability.
* Do not bypass the preprocessing skill file by redoing feature engineering here.

---

## 12. Recommended Default Workflow
The agent should follow this order:
1. Load leak-free preprocessed data from the preprocessing skill.
2. Train a compact baseline KAN classifier.
3. Compare against logistic, tree-based, and sequence baselines.
4. Apply class weighting or focal loss if needed.
5. Calibrate probabilities on validation data if calibration is poor.
6. Tune the decision threshold on validation only.
7. Prune the best stable KAN.
8. Optionally attempt symbolic extraction.
9. Report fold-wise and final test metrics.

---

## 13. Thesis Alignment
This modeling skill must be used to support a thesis on **binary Bitcoin direction prediction**. Therefore:
* the target is directional,
* the output is probabilistic,
* the evaluation is classification-oriented,
* and the modeling conclusions must remain consistent with the preprocessing skill and the thesis methodology.
