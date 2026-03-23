# López de Prado Preprocessing Skill

## 0. Scope
This skill governs data retrieval, feature preprocessing, stationarity transforms, path-dependent labeling, and leakage-safe evaluation design for financial time series. It does **not** define model architecture, benchmark selection, or KAN modeling. All modeling logic belongs to a separate skill file.

---

## 1. Chronological Processing Rule
* All transformations must preserve temporal order. No operation may use future information relative to the observation being transformed. 
* Any rolling or expanding statistic must be computed using only past data and must be explicitly lagged when necessary to avoid leakage.

---

## 2. Continuous Feature Transformations
These transformations apply only to **raw continuous predictors** before labeling and before cross-validation.

### 2.1 Log Transformations
* Apply log transforms only to strictly positive, scale-skewed variables such as price, volume, or price range proxies.
* Never log-transform bounded indicators, signed oscillators, binary flags, one-hot variables, labels, or already standardized series.
* If a feature can be zero or negative, do not log-transform it unless it has first been safely shifted and that shift is documented.

### 2.2 Optional Explosiveness Diagnostics
* Supremum ADF, right-tail unit-root tests, or similar explosiveness diagnostics may be run as diagnostic checks on selected raw price series.
* These tests are not mandatory for every feature.
* They must never be used to leak future information into preprocessing decisions.

### 2.3 Fractional Differentiation
* Use Fixed-Width Window Fractional Differentiation (FFD) as the default stationarity transform for continuous price-like series.
* The objective is to achieve stationarity while preserving the maximum possible memory of the original series.
* Select the minimum fractional order $d^*$ that makes the transformed training series pass the ADF test at the chosen threshold.
* Search over a reasonable fractional grid, for example $d \in [0,1]$ in small steps.
* Fit the choice of $d^*$ on the training data only, then apply the same transform to validation and test data.
* Never apply standard integer differencing if a smaller fractional order already achieves stationarity with adequate memory retention.
* Always drop or mask the warm-up region created by the fractional filter.

---

## 3. Triple-Barrier Labeling
* Labels must be created using a path-dependent barrier method.
* Never use fixed-time-horizon labeling as the primary label construction method.
* Define an upper barrier, a lower barrier, and a vertical barrier for each event.
* Set horizontal barrier widths as a function of a rolling volatility estimate computed from past data only.
* For each event, record:
  * $t_0$: the event start time.
  * $t_1$: the time of the first barrier hit.
* Assign the label according to the first barrier touched:
  * **1** if the upper barrier is hit first.
  * **-1** if the lower barrier is hit first.
  * **0** if the vertical barrier is hit first, representing a timeout / neutral event.
* The implementation must inspect the chronological price path between $t_0$ and $t_1$ to determine the first touch exactly.
* Any neutral events may later be dropped or kept, but the label assignment logic must remain explicit and deterministic.

---

## 4. Cross-Validation Rule
* Use leakage-safe finance-aware validation only.
* Standard random K-Fold is forbidden.
* Standard `TimeSeriesSplit` is forbidden unless it is explicitly wrapped in a purged and embargoed design.
* Use Purged K-Fold or Combinatorial Purged Cross-Validation as the default evaluation strategy.
* **Purging rule:** remove from the training set any observation whose information window overlaps the test window.
* **Embargo rule:** remove a small buffer immediately after the test window to prevent leakage from serial dependence and overlapping outcomes.

---

## 5. Inner-CV Fit Wall
* Any transformation that learns from the data distribution must be fitted only inside the cross-validation loop.
* Fit the scaler only on the purged and embargoed training fold.
* Fit feature selection only on the training fold.
* Fit PCA only on the training fold.
* Fit imputation logic only on the training fold if imputation is distribution-dependent.
* Fit threshold calibration only on the training fold or a separate calibration fold, never on the full dataset.
* Apply the fitted transformer to both train and test folds after fitting.

---

## 6. Feature Leakage Checks
Before model training, validate the following:
* All rolling indicators are lagged correctly.
* No feature uses values from the prediction day when predicting the next day.
* No scaler, selector, or volatility estimator has seen validation or test data.
* No label-dependent transformation has been applied to predictors.
* The class balance after labeling is reported.
* The fraction of neutral / timeout labels is reported.

---

## 7. Output Standards
The preprocessing pipeline must output:
* A clean feature matrix.
* A label vector.
* A mask for dropped warm-up rows.
* A training-only fitted preprocessing state.
* A reproducible record of all feature definitions, window lengths, and transform parameters.

---

## 8. Forbidden Actions
* Do not preprocess the full dataset before splitting.
* Do not compute rolling features without explicit lagging.
* Do not fit scalers or selectors on the full sample.
* Do not apply any transformation to labels as if they were predictors.
* Do not blur the distinction between preprocessing and modeling.
