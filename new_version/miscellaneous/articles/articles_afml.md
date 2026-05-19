# Article record: HARd to beat: The overlooked impact of rolling windows in the era of machine learning

## Metadata
- **BibTeX key**: chassot_audrino_2026
- **Authors**: Jonathan Chassot, Francesco Audrino
- **Year**: 2026 (Published online: 2025)
- **Title**: HARd to beat: The overlooked impact of rolling windows in the era of machine learning
- **Journal/Conference**: International Journal of Forecasting (Volume 42, 2026, Pages 330-343)
- **DOI/URL**: https://doi.org/10.1016/j.ijforecast.2025.06.003
- **Keywords**: Forecasting practice, HAR, Machine learning, Realized volatility, Volatility forecasting

## 1. Theoretical foundation

**Key theorems**: [not found]

**Key mathematical definitions:**

- **Asset Price Evolution**: $dp(t) = \mu(t)dt + \sigma(t)dW(t)$, where $p(t)$ is the logarithm of the price, $\mu(t)$ is the drift, $\sigma(t)$ is the volatility, and $W(t)$ is a standard Brownian motion.
- **Integrated Variance (IV)**: The theoretical measure of volatility over period $(t-h, t]$ is $IV(t,h) = \int_{t-h}^{t}\sigma^{2}(s)ds$.
- **Logarithmic Realized Volatility (RV)**: The daily estimator used to measure IV in practice from high-frequency intraday data: $RV_{t}^{(d)} = \log \left( \sum_{i=0}^{m-1} r_{t-\Delta i}^{2} \right)$, where $r_{t-\Delta i}$ denotes the intraday logarithmic return over $\Delta$ minutes, and $m$ represents the number of intervals in a trading day.

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: The paper does not propose a new architecture but conducts a rigorous comparative analysis of the Heterogeneous Autoregressive (HAR) model against Machine Learning (ML) models (Lasso, Random Forests, Gradient Boosted Trees, Feedforward Neural Networks) with a specific focus on optimal fitting schemes (rolling windows and re-estimation frequency).

**Core mathematical formulation:**

- **Canonical HAR Model**: $RV_{i,t+1}^{(d)} = c + \beta^{(d)}RV_{i,t}^{(d)} + \beta^{(w)}RV_{i,t}^{(w)} + \beta^{(m)}RV_{i,t}^{(m)} + \epsilon_{i,t+1}$, where $RV_{i,t}^{(w)} = \frac{1}{5}\sum_{h=1}^{5}RV_{i,t-h+1}^{(d)}$ and $RV_{i,t}^{(m)} = \frac{1}{22}\sum_{h=1}^{22}RV_{i,t-h+1}^{(d)}$.
- **HAR-VIX Model**: $RV_{i,t+1}^{(d)} = c + \beta^{(d)}RV_{i,t}^{(d)} + \beta^{(w)}RV_{i,t}^{(w)} + \beta^{(m)}RV_{i,t}^{(m)} + \beta^{(v)}VIX_{t} + \epsilon_{i,t+1}$.
- **Lasso Optimization**: $\hat{\beta} = \arg \min_{\beta} ||y - X\beta||_{2}^{2} + \lambda ||\beta||_{1}$.

**Algorithm pseudocode**: [not found]

**Architecture details:**

- **HAR models**: Estimated using Ordinary Least Squares (OLS) and Weighted Least Squares (WLS). Both individual and pooled stock estimations are evaluated.
- **Lasso**: Standard linear model with $L_1$ penalty.
- **Random Forests (RF)**: Ensemble bagging method using bootstrap sampling and random feature subset selection for uncorrelated decision trees.
- **Gradient Boosted Trees (GBT)**: Sequential tree building where each new tree fits the residuals of the previous ensemble.
- **Feedforward Neural Network (FFNN)**: Multilayer perceptrons with fully connected layers (hyperparameter specifics relegated to Appendix D).

**Innovative components**: The study highlights that the perceived superiority of ML models in recent literature is an artifact of utilizing suboptimal fitting schemes for the baseline HAR model. By applying a meticulously chosen rolling window (630 days) and a daily re-estimation frequency (stride=1), the authors demonstrate that linear HAR models perform better than or on par with computationally heavy ML models.

## 3. Data

**Dataset name(s)**: CRSP U.S. Stock Database / NYSE Trade and Quote (TAQ) database.

**Data source**: Proprietary (CRSP and TAQ databases).

**Data size:**

- Time period: January 1, 2015, to October 31, 2023 (9 years).
- Number of assets: 1445 U.S. stocks consistently present throughout the 9 years (filtered down from an initial 6061 queried stocks). Subsets analyzed include 27 DJIA constituents and 69 Nasdaq-100 constituents.
- Frequency: 5-minute intraday intervals ($\Delta = 5$).

**Data preprocessing:**

- Processed following the methodology of Barndorff-Nielsen et al. (2009).
- Evaluated on a constrained information set using only past RV series (daily, weekly, monthly, plus 100 lags for ML models) and the VIX index.

**Train/validation/test split:**

- **ML Models (Static Window)**: Training period: Jan 2016 - Dec 2020 (~64%); Validation period: Jan 2021 - Dec 2021 (~13%); Testing period: Jan 2022 - Nov 2023 (~23%).
- **HAR Models (Rolling Window)**: Training duration of exactly 630 days (approx. 3.5 years), rolling forward with a stride length of 1 day, evaluated on the exact same testing period (Jan 2022 - Nov 2023).

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against**: HAR (OLS), HAR (OLS, pooled), HAR (WLS), HAR (WLS, pooled). These are compared mutually against Lasso, Lasso (pooled), Random Forests, Random Forests (pooled), Gradient Boosted Trees, Gradient Boosted Trees (pooled), Feedforward Neural Network, and Feedforward Neural Network (pooled). All evaluated with and without VIX.

**Evaluation metrics:**

- **Mean Squared Error (MSE)**: $MSE_{i} = \frac{1}{T}\sum_{t=1}^{T}(RV_{i,t}^{(d)} - \hat{RV}_{i,t}^{(d)})^{2}$.
- **Quasi-Likelihood (QLIKE)**: $QLIKE_{i} = \frac{1}{T}\sum_{t=1}^{T} \left( \frac{\exp(RV_{i,t}^{(d)})}{\exp(\hat{RV}_{i,t}^{(d)})} - (RV_{i,t}^{(d)} - \hat{RV}_{i,t}^{(d)}) - 1 \right)$.
- **Realized Utility (RU)**: $RU_{t} = \frac{SR^{2}}{\gamma} \cdot \left( \sqrt{\frac{\exp(RV_{t+1})}{\exp(\hat{RV}_{t+1})}} - \frac{1}{2} \cdot \frac{\exp(RV_{t+1})}{\exp(\hat{RV}_{t+1})} \right)$.

**Hyperparameters:**

- **HAR**: Training window = 630 days, Stride = 1 day.
- **Realized Utility**: Risk aversion parameter $\gamma = 2$, constant Sharpe ratio $SR = 40\%$. Maximum achievable utility is $SR^2 / (2\gamma) = 4\%$. Transaction costs modeled using a 9-month rolling median of the bid-ask spread.
- **ML Models**: Hyperparameters tuned individually for each stock on the validation set. (Specific grids located in Appendix D, out of text scope, but explicitly utilized for optimization).

**Hardware/software environment**: [not found explicitly, but noted the heavy computational burden requiring average runtimes like 1.77s for HAR vs 25s for FFNN per stock setup without rolling].

**Number of runs / seeds**: Run independently across 1445 stocks.

**Statistical tests performed**: Model Confidence Set (MCS) procedure (Hansen, Lunde, & Nason, 2011) at a 95% confidence level.

## 5. Results

**Quantitative results:**

- **MCS Inclusion % (Table 3, 95% confidence level), No VIX | VIX:**
  - HAR (OLS): MSE (78.1% | 84.6%), QLIKE (27.2% | 43.5%)
  - HAR (WLS): MSE (87.1% | 96.9%), QLIKE (74.1% | 88.2%)
  - Lasso: MSE (70.5% | 91.5%), QLIKE (23.4% | 76.4%)
  - Lasso (pooled): MSE (77.1% | 94.0%), QLIKE (17.7% | 73.9%)
  - Random Forests (pooled): MSE (78.8% | 88.7%), QLIKE (40.7% | 64.7%)
  - FFNN (pooled): MSE (78.8% | 86.4%), QLIKE (44.3% | 68.5%)
- **MSE Means (Table 4), No VIX | VIX:**
  - HAR (OLS): 101.370 | 102.114
  - HAR (WLS): 99.565 | 99.953
  - Lasso (pooled): 101.763 | 102.509
  - Random Forests (pooled): 101.605 | 102.063
  - GBT (pooled): 102.930 | 104.357
  - FFNN: 906.084 | 434.199
  - FFNN (pooled): 103.910 | 105.320
- **QLIKE Means (Table 5), No VIX | VIX:**
  - HAR (OLS): 0.330 | 0.329
  - HAR (WLS): 0.311 | 0.310
  - Lasso: 0.334 | 0.315
  - Random Forests (pooled): 0.327 | 0.320
  - FFNN (pooled): 0.328 | 0.322
- **Realized Utility Means without Transaction Costs (Table 6), No VIX | VIX:**
  - HAR (OLS): 3.158% | 3.156%
  - HAR (WLS): 3.224% | 3.223%
  - Lasso: 3.146% | 3.211%
  - Random Forests (pooled): 3.163% | 3.186%
  - FFNN (pooled): 3.165% | 3.184%
- **Realized Utility Means with Transaction Costs (Table 7), No VIX | VIX:**
  - HAR (OLS): 2.369% | 2.445%
  - HAR (WLS): 2.436% | 2.518%
  - Lasso: 2.396% | 2.505%
  - Random Forests (pooled): 2.521% | 2.536%
  - FFNN (pooled): 2.482% | 2.381%

**Qualitative results:**

- **Fig. 1 (Heatmap)**: Shows visually that the HAR model's performance is highly sensitive to the stride (re-estimation frequency). If the stride deviates from 1 day (e.g., 5 or 10 days), the Root Mean Squared Prediction Error severely deteriorates. The optimal training window sits roughly between 400 and 800 days (approx. 2.5 to 4 years).
- **Fig. 2 (Cumulative Squared Error Difference)**: Plots CSED relative to HAR-VIX (OLS). Shows that HAR-VIX (OLS) outperforms pooled random forests for the majority of the testing period, with the ML models only beginning to catch up at the very end of the sample (July 2023).

**Ablation studies:**

- **VIX Inclusion**: Across almost all models, inclusion of the VIX consistently lowered the Cumulative Squared Error Difference, indicating substantial predictive value. This value was most pronounced in non-pooled models.
- **Market Capitalization Segregation (Tables 8 & 9)**: For the MSE, the pooled Random Forests model was particularly effective for micro/small-cap stocks (lowest market cap quantiles), whereas the HAR-VIX models outperformed as market capitalization rose. For QLIKE, HAR-WLS+VIX demonstrated a clear edge in the smallest quantiles, while the pooled Lasso matched its performance at the largest quantiles.
- **Estimation Method (OLS vs WLS)**: WLS significantly improved HAR performance on the asymmetric QLIKE metric compared to OLS.

**Comparison with baselines:**

- HAR models estimated via WLS with VIX achieved the highest MCS inclusion rates (88.2% for QLIKE, 96.9% for MSE) and yielded the lowest average QLIKE error (0.310).
- ML models completely failed to systematically outperform the carefully fitted HAR models on the metrics utilized, particularly QLIKE and Realized Utility, effectively contradicting earlier literature that claimed ML supremacy.
- Pooled Random Forests (101.605) did outperform the HAR-OLS (101.370) and HAR-WLS (99.565) in standard literature configurations when VIX was ignored, but still failed to beat HAR-WLS overall.

## 6. Direct quotations

- "In this study, we argue that the reality is more subtle and, despite their appeal, ML methods are not a panacea for volatility forecasting. Specifically, our findings demonstrate that the effectiveness of the HAR model is highly sensitive to the choice of fitting scheme." (p. 331)
- "As we demonstrate in this study, these discrepancies can easily be explained by the sensitivity of the HAR model's performance to the choice of fitting scheme, i.e., training window length and re-estimation frequency." (p. 331)
- "A salient observation from our analysis is the relative underperformance of non-pooled ML models compared to pooled ML models and HAR models, especially when the latter are estimated using WLS." (p. 337)
- "These results highlight once more the difficulty of more sophisticated models based on ML to consistently outperform optimized HAR models for specific parts of the stocks' cross-section, such as, for example, middle or small-cap stocks." (p. 340)
- "The additional model complexity and associated computational burden of traditional ML algorithms may negatively impact forecasting performance, especially when the information set is limited. In contrast, the HAR model's linear structure makes it a lightweight, interpretable, and robust forecasting tool, particularly when the fitting scheme is meticulously selected." (p. 340-341)

## 7. Authors' conclusions

The authors conclude that despite the paradigm shift toward machine learning in financial forecasting, the linear, highly interpretable Heterogeneous Autoregressive (HAR) model remains exceptionally robust and "HARd to beat" for realized volatility forecasting when implemented with a strictly optimized fitting scheme. By utilizing an unprecedented dataset of 1445 U.S. stocks and ensuring daily re-estimation over a ~2.5 to 4-year rolling window (specifically 630 days), the HAR model (particularly when estimated with Weighted Least Squares and incorporating the VIX) consistently matches or outperforms computationally expensive ML models (Lasso, RF, GBT, FFNN) on MSE, QLIKE, and Realized Utility. The study attributes previous literature's claims of ML superiority to their use of suboptimal, computationally sluggish fitting schemes (like expanding windows with long strides) applied unfairly to the HAR baseline.

## 8. Limitations stated by authors

- The study's information set is deliberately constrained strictly to historical Realized Volatility series and the VIX index. It does not evaluate scenarios where ML models could leverage high-dimensional external predictors (e.g., firm characteristics, macroeconomic variables, or sentiment data).
- The analysis does not specifically address or separate the individual forecasting complications arising from intraday jumps or microstructure noise; it estimates total Integrated Variance contributions, including jumps.
- ML models were evaluated using a static training window rather than a rolling window due to prohibitive computational costs (an estimated 460 times longer runtime if re-estimated daily with hyperparameter tuning).

## 9. Future work suggested by authors

- Enlarging the information set by including additional features (such as sentiment indicators or comprehensive firm characteristics) to determine if ML methods definitively outperform HAR under high-dimensional data regimes.

## Additional critical details

- **Computational Efficiency Discrepancy**: The authors highlight the severe computational cost barrier of ML models. While HAR (OLS) takes an average of 1.77 seconds to roll forward 460 re-estimations per stock, an FFNN takes 24.53 seconds just to train statically once (including hyperparameter tuning). To roll an FFNN dynamically 460 times would make the evaluation mathematically correct but computationally impossible for a 1445-stock universe.
- **Pooled Estimation Findings**: While pooling data across all stocks generally improved the performance of ML algorithms, the linear HAR model did not benefit from pooled estimation, with its single-stock versions outperforming pooled variants across the board.

---
**Extraction date**: 2026-05-19
**PDF source**: hard_to_beatrolling_windows_in_ml.pdf

---

# Article record: Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data: Evidence from Korean Markets

## Metadata
- **BibTeX key**: kang_kim_2025
- **Authors**: Sungwoo Kang, Jong-Kook Kim
- **Year**: 2025
- **Title**: Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data: Evidence from Korean Markets
- **Journal/Conference**: arXiv preprint arXiv:2504.02249v2 [cs.CE]
- **DOI/URL**: [not found]
- **Keywords**: Stock price prediction, Triple barrier labeling, OHLCV, Deep learning, LSTM, Korean markets

## 1. Theoretical foundation

**Key theorems**: [not found]

**Key mathematical definitions:**

- **Triple Barrier Labeling (TBL)**: A method for generating target labels for financial time series prediction based on the movement of a stock's price relative to three barriers: a take-profit level (upper barrier), a stop-loss level (lower barrier), and a time limit (vertical barrier).

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: Long Short-Term Memory (LSTM) network trained exclusively on raw OHLCV data and labeled via optimized Triple Barrier Labeling.

**Core mathematical formulation**: [not found]

**Algorithm pseudocode**: [not found]

**Architecture details:**

- **LSTM**: 4 layers, hidden size = 8. Total parameters: 2,235. (Grid search evaluated hidden sizes from 4 to 64 and depths from 1 to 4; dropout fixed at 0).
- **TCN**: 4 layers, hidden size c = 8, filters = 8. Total parameters: 1,667.
- **CNN**: 4 layers, hidden size c = 8, filters = 9. Total parameters: 2,339.
- **ResNet**: 4 layers, hidden size = 8, filters = 6. Total parameters: 2,735.

**Innovative components:**

- Demonstrating that raw OHLCV data without explicit feature engineering (i.e., no technical indicators) can match the performance of heavily engineered ML baselines when processed by an LSTM.
- Optimizing TBL parameters specifically for the Korean stock market (KOSPI and KOSDAQ) using high and low prices for barriers instead of just closing prices.

## 3. Data

**Dataset name(s)**: KOSPI and KOSDAQ listed tickers.

**Data source**: Publicly scraped from www.finance.naver.com.

**Data size**: Daily OHLCV data from January 2, 2006, to December 31, 2024. Extracting sequences of length 100 yielded 8,566,617 instances. Instance shape is (5, 100).

**Data preprocessing:**

- Missing values were forward-filled.
- Sequences extracted using a rolling window method.
- Triple Barrier Labeling parameters were grid-searched. The optimal setup (yielding balanced classes) was a 29-day prediction horizon with a 9% take-profit/stop-loss threshold.
- TBL used low and high prices for stop-loss and take-profit evaluations to reduce intraday uncertainty. If both barriers were hit on the same day, the label defaulted to "time limit" (no move).

**Train/validation/test split**: Data split chronologically into 6 parts (4 for training, 1 for validation, 1 for testing).

- Training: Jan 1, 2006 to Mar 31, 2020.
- Validation: Mar 31, 2020 to Sep 29, 2022.
- Testing: Sep 29, 2022 to Dec 31, 2024.

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against:**

- **Traditional ML (via PyCaret)**: CatBoost Classifier, Extreme Gradient Boosting (XGBoost), Light Gradient Boosting (LightGBM), Gradient Boosting Classifier, Random Forest Classifier, Extra Trees Classifier, Ada Boost Classifier, Decision Tree Classifier, Logistic Regression, Linear Discriminant Analysis, Ridge Classifier, K Neighbors Classifier, Naive Bayes, SVM, and Quadratic Discriminant Analysis.
- **Naive baseline**: Dummy Classifier (predicts most frequent class).
- **Deep Learning baselines**: ResNet, Temporal Convolutional Networks (TCN), CNN.
- **Literature baseline**: Phuoc et al. (LSTM with technical indicators).

**Evaluation metrics:**

- Macro-averaged F1 Score
- Accuracy
- AUC (Area Under the Curve)
- MCC (Matthew's Correlation Coefficient)

**Hyperparameters:**

- **LSTM Grid Search**: Window lengths $\in$ [5, 20, 50, 100, 200, 300] days; Hidden sizes $\in$ [4, 8, 16, 32, 64] units. Dropout = 0.
- **XGBoost optimization**: 70 iterations of tuning via 5-fold time-series cross-validation. Selected top 12 engineered features while removing multicollinear indicators (threshold: 0.9).

**Hardware/software environment**: PyCaret framework utilized for implementing and evaluating traditional machine learning models.

**Number of runs / seeds**: [not found]

**Statistical tests performed**: Correlation coefficient calculated between validation and test F1 scores (0.793).

## 5. Results

**Quantitative results:**

- **Optimal TBL Label Distribution**: Time Limit (36.16%), Stop Loss (28.95%), Take Profit (34.89%).
- **DL Architecture Comparison (F1 Score)**: LSTM (0.4312), TCN (0.4219), ResNet (0.3975), CNN (0.3045).
- **Feature Selection F1 Scores (using LSTM)**: Close price only (0.4170), Close + Volume (0.4297), Full OHLCV (0.4312).
- **Model Comparison Metrics:**
  - LSTM (raw OHLCV): Accuracy = 0.4328, AUC = 0.6249, F1 = 0.4312.
  - XGBoost (technical indicators): Accuracy = 0.4311, AUC = 0.6247, F1 = 0.4316.
  - Dummy Classifier: Accuracy = 0.3539, AUC = 0.5000, F1 = 0.1852.
  - Phuoc et al. adapted model: F1 = 0.3290.

**Qualitative results:**

- Increasing the hidden size beyond 8 (e.g., to 32 or 64) provided diminishing returns and did not significantly improve performance.
- Very short temporal windows (5 or 20 days) lacked enough context to make accurate predictions, while overly long windows (200 or 300 days) offered no additional benefit. The optimal balance was achieved with a 100-day window.

**Ablation studies:**

- **Input Feature Ablation**: Tested Close-only, Close+Volume, and full OHLCV. Adding Volume to Close price improved the F1 score by 1.27%. Adding Open, High, and Low prices provided marginal but positive additional gains, confirming the value of the full OHLCV set.

**Comparison with baselines:**

- A simple LSTM network trained solely on raw OHLCV matched the predictive performance of a heavily optimized XGBoost model utilizing complex technical indicators across all tracked metrics (F1, Accuracy, AUC).
- The LSTM architecture decisively outperformed all other deep learning architectures tested (TCN, ResNet, CNN) given equivalent parameter budgets.
- The proposed method outperformed the adapted baseline from Phuoc et al. by a wide margin (F1 of 0.4312 vs. 0.3290).

## 6. Direct quotations

- "Our key contribution is demonstrating that a simple LSTM network trained on raw OHLCV data alone can match the performance of sophisticated ML models that incorporate technical indicators, challenging the conventional wisdom that technical indicators are essential for effective stock price prediction." (p. 1)
- "The optimal configuration was found to be a hidden size of 8 and a window length of 100, achieving an F1 score of 0.4312." (p. 4)
- "The findings indicate that adding volume data improves the F1 score by 1.27%, while the remaining OHLCV features (open, high, low) together contribute only minimal additional performance gains." (p. 4)
- "The results show that our LSTM model achieves comparable performance to XGBoost across all metrics, with both models significantly outperforming the Dummy Classifier baseline." (p. 5)
- "This suggests that the field of financial forecasting may benefit from reconsidering the necessity of complex feature engineering in the era of deep learning." (p. 7)

## 7. Authors' conclusions

The authors conclude that extensive feature engineering via technical indicators may be unnecessary in the era of deep learning for stock price forecasting. A simple LSTM network trained purely on raw OHLCV data can match the performance of heavily optimized traditional machine learning models (like XGBoost) that use extensive engineered features. For the Korean market, optimizing Triple Barrier Labeling parameters is essential to generate balanced targets, which they achieved using a 29-day window and 9% barriers. Additionally, the study highlights a critical relationship between window size and model capacity; optimal performance was observed with a 100-day window and a minimal hidden size of 8, demonstrating that massive model complexity is not inherently superior. Finally, the full suite of OHLCV data proved superior to reduced subsets like Close-only or Close+Volume.

## 8. Limitations stated by authors

- **Market Specificity**: The findings are empirically derived solely from Korean stock market data (KOSPI/KOSDAQ) and may not generalize to markets with distinct trading behaviors.
- **Feature Expansion**: The study restricts itself to OHLCV data. Incorporating alternative data such as news sentiment, macroeconomic indicators, or level-2 order book data could yield further performance enhancements.
- **Trading Strategy Integration**: The study evaluates predictive accuracy but does not backtest these predictions within a realistic trading strategy that accounts for transaction costs and market impact.
- **Explainability**: Deep learning models operate as "black boxes", lacking the interpretability provided by traditional indicator-based rules.

## 9. Future work suggested by authors

- Explore more advanced sequence modeling architectures, such as attention mechanisms or transformer networks.
- Develop methods to enhance the interpretability and explainability of deep learning models in financial applications to improve trust.
- Train predictive models selectively on subsets of stocks that exhibit higher sensitivity to price movements, rather than broad market indices.

## Additional critical details

- **Technical Indicator Baseline Detail**: The XGBoost baseline was granted access to a rich set of normalized technical features prior to selection, including Ichimoku Cloud Indicators, RSI, Stochastic RSI, CCI, MFI, MACD, ADX, EMA (5, 20, 60, 120, and 240 days), ATR, Bollinger Bands, OBV, and CMF. The fact that raw OHLCV matched this highlights the implicit feature extraction capability of LSTMs.

---
**Extraction date**: 2026-05-19
**PDF source**: arXiv:2504.02249

---

# Article record: Supervised Autoencoders with Fractionally Differentiated Features and Triple Barrier Labelling Enhance Predictions on Noisy Data

## Metadata
- **BibTeX key**: slepaczuk_bieganowski_2024
- **Authors**: Robert Ślepaczuk, Bartosz Bieganowski
- **Year**: 2024
- **Title**: Supervised Autoencoders with Fractionally Differentiated Features and Triple Barrier Labelling Enhance Predictions on Noisy Data
- **Journal/Conference**: Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF-24) / arXiv
- **DOI/URL**: arXiv:2411.12753v2 [q-fin.TR]
- **Keywords**: machine learning, algorithmic investment strategy, supervised autoencoders, financial time series, trading strategy, risk-adjusted return

## 1. Theoretical foundation

**Key theorems**: [not found]

**Key mathematical definitions:**

- **Fractional Differentiation Backshift Operator**: $B^k X_t = X_{t-k}$, where the difference between current and last feature value is $(1-B)X_t$.
- **Binomial Series Expansion**: $(1+x)^d = \sum_{k=0}^{\infty} \binom{d}{k} x^k$, allowing differentiation to be generalized to non-integer orders.
- **Fractional Differencing Weights**: $w_k = (-1)^k \prod_{i=0}^{k-1} \frac{d-i}{k!}$, where $d$ is the fractional degree.

**Key lemmas or propositions:**

- **Optimal Fractional Order ($d$)**: If a time series $X_t$ is stationary, $d=0$. If it contains a unit root, $d<1$. If it has explosive behaviors, $d>1$. A state of $0 < d < 1$ indicates the series is "slightly non-stationary", allowing retention of memory while passing stationarity tests (like the Augmented Dickey-Fuller test).

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: Supervised Autoencoder - Multi-Layer Perceptron (SAE-MLP) network applied within an Algorithmic Investment Strategy (AIS).

**Core mathematical formulation:**

- **Triple Barrier Labeling (TBL)**:
  - $Y_{true} = 1$, if $\max(S_t,\dots,S_{t+n}) \ge S_t \cdot (1+\lambda)$
  - $Y_{true} = -1$, if $\min(S_t,\dots,S_{t+n}) \le S_t \cdot (1-\lambda)$
  - $Y_{true} = 0$, otherwise.
- **Performance/Return Estimation**:
  - Directly Correct Count ($DCC$): Number of times $Y_{pred} \neq 0$ and $Y_{pred} = Y_{true}$.
  - Directly Incorrect Count ($DIC$): Number of times $Y_{pred} \neq 0$ and $Y_{pred} \neq Y_{true}$.
  - Base return formula: $\mathfrak{p} = (1+\lambda)^{DCC} \cdot (1-\lambda)^{DIC}$
- **TBL-Optimized Performance Metric**: $\mathfrak{p}^* = (1+\lambda)^{DCC} \cdot (1-\lambda)^{DIC} \cdot (1-\lambda)^{\frac{TEC}{\delta}}$, where $TEC$ is the "timed exit count" (barrier hit by time limit instead of profit/loss). $\delta$ is a discouragement factor (set to 20) ensuring timed exits are penalized but at a lower magnitude than direct incorrect classifications.

**Algorithm pseudocode:**

```
Algorithm 1: Fractional Feature Differentiation in Walk-Forward Validation:
1. Set range of possible values for d (0 to 1).
2. Set significance level for ADF test (e.g., 1%).
3. Initiate dictionary associating features with optimal d.
4-13. Loop over segment pairs (train, test) and features. Apply differencing
      to train segment, calculate ADF p-value. Choose lowest d with p-value
      < significance level. Apply optimal d differencing to train and test
      sets, then train model and evaluate.

Algorithm 2: Triple-Barrier Labeling - Simple implementation:
1-20. Iterate over price indices. Calculate profit_target (1+lambda) and
      stop_loss_target (1-lambda). Iterate up to time_barrier_idx. If price
      hits target, label 1; if hits stop-loss, label -1. If loop reaches
      time barrier without hitting thresholds, label 0.
```

**Architecture details:**

- **Supervised Autoencoder (SAE)**: Contains an Encoder (compresses input into a "code" or bottleneck) and a Decoder (reconstructs input from the code).
- **Supervised combination**: Encoded values (the bottleneck) are concatenated with the original input data. This concatenated array is used to train a supervised Multi-Layer Perceptron (MLP) on the TBL labels. The loss/performance metric combines unsupervised reconstruction accuracy and supervised prediction accuracy.

**Innovative components:**

- Using fractional differentiation specifically combined with a walk-forward optimization framework to maintain maximum memory while achieving stationarity dynamically.
- Creation of a novel, return-maximizing optimization metric ($\mathfrak{p}^*$) that integrates a custom penalty ($\delta$) for "timed-out" trades, optimizing specifically for TBL dynamics rather than standard classification accuracy.

## 3. Data

**Dataset name(s)**: Binance Cryptocurrency OHLC Data and FirstRateData Macro Futures.

**Data source**: Public/Commercial APIs (Binance for crypto; FirstRateData and FRED for macro features).

**Data size**: Time period spans from January 1, 2016, to April 30, 2022. Frequency is 1-minute bars. Traded assets: Bitcoin (BTC), Ethereum (ETH), Litecoin (LTC).

**Data preprocessing:**

- **Feature Engineering**: Six macroeconomic features added (Initial Claims/ICSA, WTI Crude Oil, Henry Hub Natural Gas, Corn, Gold, Copper, Aluminum).
- **Fractional Differencing**: Applied dynamically per walk-forward window to each feature until ADF stationarity (p < 0.01) is achieved.

**Train/validation/test split:**

- **In-Sample (Train)**: Pre-2020 data (January 1, 2016 to December 31, 2019).
- **Out-of-Sample (Test)**: January 1, 2020 to April 30, 2022.
- **Validation Scheme**: Walk-forward validation with overlapping windows of constant or expanding size (train on first part of window, validate on the second, shifting forward).

**Data augmentation techniques**: Data was augmented by introducing Gaussian noise to features based on a fraction of their historical volatility (e.g., standard deviation of noise = noise ratio * historical volatility). This exposes the model to a wider variety of realistic market scenarios without breaking time-series sequence integrity.

## 4. Experimental setup

**Baseline models compared against**: Passive Buy-and-Hold strategy.

**Evaluation metrics:**

- Total Return
- Annualized Return
- Annualized Standard Deviation
- Information Ratio (IR)
- Maximum Drawdown (MDD)
- Maximum Loss Duration (MLD)
- Starred Information Ratio (IR*)

**Hyperparameters:**

- Model intervals (bar configurations): 10-minute, 20-minute, and 30-minute intervals.
- Timed exit discouragement magnitude ($\delta$): 20.
- Transaction costs assumed: 0.05% per trade opening/closing (total 0.1% round trip for direction change).

**Hardware/software environment**: Implemented in Python 3.10 using the NumPy package.

**Number of runs / seeds**: [not found]

**Statistical tests performed**: Augmented Dickey-Fuller (ADF) test to ensure feature stationarity.

## 5. Results

**Quantitative results:**

- **Bitcoin (BTC) Strategies:**
  - 10min: Total Return = 163.05%, Ann Return = 36.08%, Ann Std = 35.52%, IR = 1.01, Max Drawdown = 39.74%, MLD = 0.56, IR* = 0.92
  - 20min: Total Return = 306.47%, Ann Return = 51.48%, Ann Std = 28.51%, IR = 1.81, Max Drawdown = 25.86%, MLD = 0.47, IR* = 3.59
  - 30min: Total Return = 233.92%, Ann Return = 42.91%, Ann Std = 21.11%, IR = 2.03, Max Drawdown = 14.02%, MLD = 0.42, IR* = 6.22
  - Buy and hold: Total Return = 406.02%, Ann Return = 61.63%, Ann Std = 63.31%, IR = 0.97, Max Drawdown = 53.30%, MLD = 0.51, IR* = 1.13
- **Ethereum (ETH) Strategies:**
  - 10min: Total Return = 1611.79%, Ann Return = 131.87%, Ann Std = 69.25%, IR = 1.9, Max Drawdown = 54.25%, MLD = 0.47, IR* = 4.63
  - 20min: Total Return = 1658.93%, Ann Return = 133.75%, Ann Std = 50.63%, IR = 2.64, Max Drawdown = 37.17%, MLD = 0.41, IR* = 9.51
  - 30min: Total Return = 1151.59%, Ann Return = 111.34%, Ann Std = 42.51%, IR = 2.62, Max Drawdown = 35.68%, MLD = 0.41, IR* = 8.17
  - Buy and hold: Total Return = 1929.01%, Ann Return = 143.85%, Ann Std = 82.85%, IR = 1.74, Max Drawdown = 62.17%, MLD = 0.47, IR* = 4.02
- **Litecoin (LTC) Strategies:**
  - 10min: Total Return = 60.77%, Ann Return = 15.10%, Ann Std = 48.32%, IR = 0.31, Max Drawdown = 46.21%, MLD = 0.97, IR* = 0.1
  - 20min: Total Return = 356.54%, Ann Return = 56.78%, Ann Std = 37.96%, IR = 1.49, Max Drawdown = 30.86%, MLD = 0.65, IR* = 2.75
  - 30min: Total Return = 382.49%, Ann Return = 59.37%, Ann Std = 28.09%, IR = 2.11, Max Drawdown = 25.88%, MLD = 0.48, IR* = 4.85
  - Buy and hold: Total Return = 118.92%, Ann Return = 26.11%, Ann Std = 88.61%, IR = 0.29, Max Drawdown = 75.21%, MLD = 0.97, IR* = 0.1
- **1:1:1 Portfolio (Equal Weight):**
  - 10min: Total Return = 372.71%, Ann Std = 45.78%, IR = 1.28, Max Drawdown = 40.40%, IR* = 1.84
  - 20min: Total Return = 619.80%, Ann Std = 34.84%, IR = 2.28, Max Drawdown = 27.83%, IR* = 6.5
  - 30min: Total Return = 501.63%, Ann Std = 27.37%, IR = 2.56, Max Drawdown = 23.55%, IR* = 7.63
  - Buy and hold: Total Return = 595.24%, Ann Std = 72.61%, IR = 1.07, Max Drawdown = 59.46%, IR* = 1.39

**Qualitative results:**

- A correlation heatmap (Figure 13) confirms that the returns of the various interval strategies (10, 20, 30 min) and the spot returns are highly correlated, but the SAE models smooth out severe drawdowns.
- Across all assets, the 30-minute interval consistently yielded the highest or second-highest risk-adjusted returns (IR and IR*) and lowest maximum drawdowns, indicating it effectively captured trends while filtering noise better than the 10-minute strategy.

**Ablation studies**: [not found]

**Comparison with baselines:**

- For BTC and ETH, Buy-and-Hold produced higher absolute total returns due to the test period being a historic bull market. However, SAE strategies (especially 20min and 30min) massively outperformed Buy-and-Hold on all risk-adjusted metrics, reducing Max Drawdown by ~40% for BTC and ~25% for ETH, and posting much higher Information Ratios.
- For LTC and the Equal Weight Portfolio, the 20min and 30min SAE strategies outperformed the Buy-and-Hold baseline in both absolute Total Return and risk-adjusted metrics.

## 6. Direct quotations

- "Financial time series are notorious for their low signal-to-noise ratios, largely due to arbitrage forces in the market." (p. 2)
- "To maximize the trading strategy performance, we must introduce an error preference mechanism. After all, missing a profitable trade (type 0 error) will have a lesser effect on our portfolio than entering an incorrect trade (type 1 error)." (p. 3)
- "Our research has demonstrated that the application of Supervised Autoencoder de-noising in combination with Triple Barrier labeling significantly improves algorithmic trading strategies in comparison to buy and hold approaches." (p. 7)
- "While the buy-and-hold strategy yielded the highest total return (406.02%) and annualized return (61.63%), it also exhibited the highest volatility with an annualized standard deviation of 63.31% and a maximum drawdown of 53.30%." (p. 5)
- "Despite the correlation of returns between strategies, combining them into a portfolio outperforms the buy-end-hold equal-weight portfolio." (p. 7)

## 7. Authors' conclusions

The authors conclude that implementing Supervised Autoencoder (SAE) models, augmented with fractionally differentiated macroeconomic features and volatility-based noise injection, yields highly effective algorithmic trading strategies when paired with Triple Barrier Labeling (TBL). The introduction of a novel optimization metric tailored for TBL significantly enhanced the models' ability to prefer safe trades over uncertain timed-exits. In out-of-sample testing from 2020 to 2022 across Bitcoin, Ethereum, Litecoin, and a diversified equal-weight portfolio, the SAE models (specifically the 20-minute and 30-minute intervals) vastly outperformed passive buy-and-hold strategies in terms of risk-adjusted returns (Information Ratios) and maximum drawdowns. Although the sheer bullish momentum of the test period allowed buy-and-hold to secure higher absolute returns for BTC and ETH, the SAE models protected the portfolio from devastating volatility, proving the necessity of these advanced machine learning models for risk-managed active asset management.

## 8. Limitations stated by authors

- The findings are based on historical data; predictive power for future performance must be considered with caution given the volatile nature of markets.
- The research does not account for execution slippage, which could diminish net returns (particularly in illiquid markets or with large capital deployments).
- The strategy assumes that stop-loss and take-profit levels generated by TBL are executed immediately and perfectly, which is not always realistic in live markets.
- The test set (2020-2022) occurred during a highly bullish macroeconomic period for cryptocurrencies, which naturally skews long-biased or hold-biased results.

## 9. Future work suggested by authors

- Investigating other types of data noise and their specific impacts on strategy performance.
- Integrating execution slippage into the model to simulate more realistic net returns.
- Exploring different architectures for the autoencoder or the integration of other deep learning techniques.
- Testing the impact of these methods on other types of financial time series data beyond cryptocurrencies (e.g., the S&P 500 index, which exhibits consistent long-term trends).

---
**Extraction date**: 2026-05-19
**PDF source**: Supervised_Autoencoders_with_TBL_and_FFD.pdf

---

# Article record: Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling Method and Machine Learning Approach for Pair Trading Strategy in Cryptocurrency Markets

## Metadata
- **BibTeX key**: fu_et_al_2024
- **Authors**: Ning Fu, Mingu Kang, Joongi Hong, Suntae Kim
- **Year**: 2024
- **Title**: Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling Method and Machine Learning Approach for Pair Trading Strategy in Cryptocurrency Markets
- **Journal/Conference**: Mathematics (MDPI)
- **DOI/URL**: https://doi.org/10.3390/math12050780
- **Keywords**: pair trading; triple barrier labeling method; cryptocurrency; genetic algorithm; AdaBoost classifier

## 1. Theoretical foundation

**Key theorems:**

- **Mean Reversion Theory**: The foundational theory for pair trading, stating that the spread between highly correlated assets tends to revert to its historical mean after diverging.

**Key mathematical definitions:**

- **Pearson Correlation Coefficient ($r$)**: Measures linear correlation between two cryptocurrencies. $r = \frac{\sum_{t=1}^n (Crypto\_A_t - \overline{Crypto\_A})(Crypto\_B_t - \overline{Crypto\_B})}{\sqrt{\sum_{t=1}^n (Crypto\_A_t - \overline{Crypto\_A})^2 \sum_{t=1}^n (Crypto\_B_t - \overline{Crypto\_B})^2}}$
- **Cointegration Test Regression**: $Crypto\_B_t = \beta_0 + \beta_1 Crypto\_A_t + \epsilon_t$, where $\epsilon_t$ is the residual error term to be tested for a unit root.
- **Augmented Dickey-Fuller (ADF) Test**: $\Delta\epsilon_t = \alpha + \beta t + \gamma\epsilon_{t-1} + \delta_1\Delta\epsilon_{t-1} + ... + \delta_{p-1}\Delta\epsilon_{t-p+1} + \epsilon_t$. If $\gamma = 0$, the sequence is non-stationary. If $\gamma \neq 0$ (and p-value < 0.05), the null hypothesis is rejected, implying the residual is stationary and the assets are cointegrated.

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: Triple Barrier Labeling Method applied to Z-spread (TBM for Z-spread), optimized via Genetic Algorithm (GA), and predicted using an AdaBoost Classifier (incorporating Extreme Gradient Boosting).

**Core mathematical formulation:**

- **Pairs Spread**: $Spread = \left(\frac{Crypto\_A_T - Crypto\_A_{T-1}}{Crypto\_A_{T-1}}\right) - \left(\frac{Crypto\_B_T - Crypto\_B_{T-1}}{Crypto\_B_{T-1}}\right)$
- **Z-spread**: $Z\text{-}spread = \frac{Spread - \mu_{spread(Slide\_Window=55)}}{\delta_{spread(Slide\_Window=55)}}$
- **Genetic Algorithm (GA) Selection**: $P_{t+1}^{sel} = Select(P_t, f) = \{c' | c' \text{ is chosen from } P_t \text{ based on fitness } f(c)\}$
- **GA Crossover**: $(c''_1, c''_2) = cross(c'_1, c'_2) = (c'_1 \otimes c'_2, c'_2 \otimes c'_1)$
- **GA Mutation**: $c'''_1 = Mutate(c''_1) = c''_1 \oplus \mu$
- **GA Fitness Function**: $f(c) = w_p Profit(c) + w_{mdd} MDD(c)$

**Algorithm pseudocode:**

```
Framework (Stage 1: Label Generation):
1. Select crypto pairs via correlation and cointegration tests.
2. Calculate Pairs Spread and corresponding Z-score.
3. Generate labels via TBM for Pair Trading Strategy:
   - Upper threshold -> 1: Buy Crypto_B
   - Lower threshold -> -1: Buy Crypto_A
   - Max holding period/vertical barrier -> 0: Sell All.
4. Run GA Optimization: Initialize random population -> Selection -> Crossover ->
   Mutation -> Apply parameters to TBM-Zspread -> Calculate Fitness (Profit & MDD)
   -> Repeat until criteria met.

Framework (Stage 2: Model Trading):
1. Extract Optimized Signals (HRHP and LRLP) and training data.
2. Train Machine Learning Classifier.
3. Predict on testing dataset.
4. Backtest to generate final Profit and MDD metrics.
```

**Architecture details:**

- **AdaBoost Classifier (Final model)**: The study utilized Extreme Gradient Boosting (XGBoost) as the foundational classifier and amalgamated it with the AdaBoost Classifier to elevate classification accuracy over standalone or deep learning architectures.

**Innovative components:**

- Applying the Triple Barrier Labeling Method (TBL) strictly to the Z-spread between two assets, rather than single asset price paths.
- Using a Genetic Algorithm to explicitly optimize the TBL hyperparameters (Upper threshold, Lower threshold, Max holding period) into two distinct strategic label sets: High Risk and High Profit (HRHP) and Low Risk and Low Profit (LRLP).
- Replacing the standard regression of price spreads with an AdaBoost classification task predicting these optimized discrete trading signals, demonstrating that simple ML classifiers mitigate overfitting on volatile crypto data better than Deep Learning models (LSTM, CNN).

## 3. Data

**Dataset name(s)**: Cryptocurrency Market Data (Yahoo Finance).

**Data source**: Publicly sourced using Yahoo Finance Python Package (yfinance version 0.2.27).

**Data size:**

- Time period: 9 November 2017 to 1 December 2023.
- Frequency: Daily.
- Selected pair: Bitcoin (BTC) and Ethereum (ETH).

**Data preprocessing:**

- Calculated daily change rates (DCR) for all coins to compute Pearson correlation and ADF cointegration tests.
- Feature engineering: Features generated include high, low, close, adj close, volume, transaction, and Z-spread.
- Rolling window calculation: Spread mean ($\mu$) and standard deviation ($\delta$) computed using a 55-day sliding window.

**Train/validation/test split:**

- **Training set (and GA label optimization)**: 9 November 2017 to 31 August 2022.
- **Testing set (out-of-sample backtesting)**: 1 September 2022 to 1 December 2023.

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against:**

- AI Regression Model (Osifo Ernest et al. methodology baseline).
- Pair Trading Strategy (Traditional).
- Buy-and-Hold Bitcoin.
- Buy-and-Hold Ethereum.
- Various standalone classifiers evaluated: Multilayer Perceptron, LSTM, CNN+LSTM+Attention, Ridge, Support Vector Classification, LightGBM, SGD, Decision Tree, Gaussian Process.

**Evaluation metrics:**

- **Profit (Times)**: The multiple of the initial investment achieved.
- **Maximum Drawdown (MDD)**: The maximum observed loss from a peak to a trough, measured as a percentage.
- **Sharpe Ratio**: Measure of risk-adjusted return.
- **Classifier Metrics**: Accuracy, Recall, and Area Under Curve (AUC). (Rank Graduation Accuracy (RGA) was discussed but discarded as unsuitable for these time-series state shifts).

**Hyperparameters:**

- **GA Search Space**: Upper Threshold (1-300%), Lower Threshold (1-300%), Max Holding Period (1-10 days).
- **HRHP Labels Fitness Weights**: $w_p = 0.70$ (Profit), $w_{mdd} = 0.30$ (MDD).
- **LRLP Labels Fitness Weights**: $w_p = 0.30$ (Profit), $w_{mdd} = 0.70$ (MDD).
- **Optimized parameters applied post-GA (HRHP)**: Upper Threshold = 17.3043%, Lower Threshold = 11.7092%, Max Holding Period = 2 days.
- **Optimized parameters applied post-GA (LRLP)**: Upper Threshold = 66.2016%, Lower Threshold = 22.0194%, Max Holding Period = 2 days.

**Hardware/software environment**: Python using yfinance package.

**Number of runs / seeds**: Addressed conceptually via "Multiple Runs of GA" to average out randomness during optimization, but exact seed quantities were not reported.

**Statistical tests performed**: Augmented Dickey-Fuller (ADF) test (for cointegration and spread stationarity), Pearson correlation test.

## 5. Results

**Quantitative results:**

- **Correlation & Cointegration**: BTC and ETH pair showed ADF p-value = 0.0 and Pearson correlation = 0.8128.
- **Label Evaluation (Training period max theoretical boundaries):**
  - HRHP Label: Profit 386.9837x, MDD -0.5516
  - LRLP Label: Profit 46.9574x, MDD -0.4337
  - Buy-and-Hold BTC: Profit 3.1537x, MDD -0.7185
  - Buy-and-Hold ETH: Profit 4.9373x, MDD -0.7935
  - Traditional Pair Trading: Profit 20.7737x, MDD -0.5358
- **Classifier Performance Evaluation:**
  - AdaBoost: Accuracy 0.7903, Recall 0.6621, AUC 0.6409
  - AdaBoost + Bayesian Data Mining: Accuracy 0.7732, Recall 0.6192, AUC 0.6335
  - XGBoost: Accuracy 0.7208, Recall 0.5597, AUC 0.6299
  - Decision Tree: Accuracy 0.6922, Recall 0.5987, AUC 0.6167
  - LightGBM: Accuracy 0.6518, Recall 0.5460, AUC 0.5956
  - LSTM: Accuracy 0.6392, Recall 0.5500, AUC 0.5408
  - MLP: Accuracy 0.6216, Recall 0.5473, AUC 0.5458
  - CNN+LSTM+Attention: Accuracy 0.6216, Recall 0.5417, AUC 0.5477
- **Model Trading Prediction Results (Test period: Sep 1, 2022 - Dec 1, 2023):**
  - HRHP-trained model: Profit 2.0754x, MDD -0.1611, Sharpe Ratio 1.8491
  - LRLP-trained model: Profit 1.7034x, MDD -0.0691, Sharpe Ratio 2.0016
  - AI regression model (Baseline): Profit 1.4499x, MDD -0.2047, Sharpe Ratio 0.9216
  - Pair trading strategy (Baseline): Profit 1.3702x, MDD -0.2579, Sharpe Ratio 0.8476
  - Buy-and-hold BTC: Profit 1.8737x, MDD -0.2942, Sharpe Ratio 1.3073
  - Buy-and-hold ETH: Profit 1.2940x, MDD -0.3806, Sharpe Ratio 0.6517

**Qualitative results:**

- Visual analysis proved that Z-spread accurately captures fluctuation deviations that the raw unnormalized spread misses.
- Deep learning models (LSTM, CNN+LSTM+Attention, MLP) severely underperformed ensemble tree models (AdaBoost, XGBoost). The complexity of deep learning caused overfitting on the highly volatile and noisy cryptocurrency data.
- HRHP strategy executes a much higher frequency of trades compared to LRLP (due to narrower barrier thresholds), leading to higher absolute returns but larger drawdowns.
- The LRLP strategy executes fewer trades but possesses a higher win rate, giving it the highest risk-adjusted return (Sharpe Ratio of 2.0016).

**Ablation studies:**

- Tested AdaBoost combined with Bayesian data mining and Markov chain Monte Carlo (MCMC) techniques for feature selection/transformation. The performance degraded slightly (Accuracy 0.7732 vs 0.7903), demonstrating that Bayesian data mining methods are not suitable for highly volatile crypto data.

**Comparison with baselines:**

- The HRHP-trained ML model achieved a profit of 2.0754x, beating the traditional pair trading baseline (1.3702x) by 51.42%.
- The LRLP-trained ML model achieved an MDD of -0.0691, representing an enormous 73.24% reduction in drawdown risk compared to traditional pair trading (-0.2579).
- Both HRHP and LRLP methods dominated the AI Regression baseline across all metrics (Profit, MDD, Sharpe).

## 6. Direct quotations

- "In pursuit of high-performance labels to elevate the precision of classification models, this study advanced the Triple Barrier Labeling Method for enhanced compatibility with pair trading strategies." (p. 1)
- "In this adaptation, single asset price sequences were transformed into spreads between paired assets, and the parameters of the Triple Barrier Labeling Method were adjusted." (p. 2)
- "By choosing relatively simple machine learning classifiers over deep learning, we can mitigate overfitting and, consequently, improve prediction accuracy when confronted with the high volatility and noise inherent in cryptocurrency data." (p. 2)
- "During experimentation, it was discovered that the Extreme Gradient Boosting Classifier outperformed alternative classification models, achieving accuracy and recall rates of 0.7208 and 0.5597, respectively. As a result, we selected the Extreme Gradient Boosting Classifier as the foundational classifier and amalgamated it with the AdaBoost Classifier to elevate the accuracy and recall rates to 0.7903 and 0.6621, correspondingly." (p. 15-16)
- "Specifically, the model trained with HRHP labels demonstrated high profitability with a profit of 2.0754, significantly surpassing the AI regression analysis model's 1.4499 and the traditional pair trading strategy's 1.3702 profit. Conversely, the LRLP model exhibited remarkable risk control, with its MDD at -0.0691, better than the AI regression analysis model's -0.2047 and the traditional pair trading strategy's -0.2579 MDD." (p. 17)

## 7. Authors' conclusions

The authors conclude that adapting the Triple Barrier Labeling Method from single-asset prices to Z-spreads creates exceptionally high-quality signals for cryptocurrency pair trading. By deploying a Genetic Algorithm with dual fitness functions, they successfully generated two targeted behavioral datasets: High Risk High Profit (HRHP) and Low Risk Low Profit (LRLP). The study decisively concludes that training an AdaBoost classifier on these optimized, discrete labels is far superior to standard AI regression methods and traditional pair trading formulas. This approach mitigates overfitting inherent in noisy crypto data (a weakness exposed in deep learning baselines) and delivers state-of-the-art results out-of-sample, offering traders a strategy that either drastically amplifies returns (HRHP) or significantly compresses drawdown risk (LRLP) relative to the market benchmarks.

## 8. Limitations stated by authors

- **Data Frequency Limitation**: The study utilized daily sampled datasets to regulate trade frequency. This choice potentially leads to information loss intraday, which could impact the accuracy of the final analysis.
- **Genetic Algorithm Randomness**: GAs inherently rely on random selection for population initialization, crossover, and mutation, meaning exact parameters can fluctuate slightly between runs, posing a potential threat to the precise reproducibility of the optimization outputs.

## 9. Future work suggested by authors

- Exploring the application of the proposed method on datasets with smaller, higher-frequency time intervals (e.g., 1-hour or 5-minute data) to address the information loss inherent in daily sampling.
- Broadening the selection of classification models to include advanced sequence models like Temporal Convolutional Networks (TCN), WaveNet, and TimesNet to potentially push profitability and risk management even further.

## Additional critical details

- **Nonlinear Compounding Alert**: The authors specifically noted a critical phenomenon when analyzing results: simply calculating the cumulative returns of ETH and BTC separately to determine which coin drove strategy profits is mathematically invalid. Executing hedging pair trades exhibits complex, nonlinear compound interest effects that render isolated asset-return calculations inadequate for measuring portfolio performance.

---
**Extraction date**: 2026-05-19
**PDF source**: https://doi.org/10.3390/math12050780

---

# Article record: Does Meta-Labeling Add to Signal Efficacy?

## Metadata
- **BibTeX key**: singh_joubert_2019
- **Authors**: A. Singh, J. Joubert
- **Year**: 2019 (Inferred from text references to 2019 and Github repository dates)
- **Title**: Does Meta-Labeling Add to Signal Efficacy?
- **Journal/Conference**: [not found]
- **DOI/URL**: [not found]
- **Keywords**: Machine Learning, Meta-Labeling, CUSUM Filter, Triple Barrier Labeling, Trend-following, Mean-Reversion

## 1. Theoretical foundation

**Key theorems**: [not found]

**Key mathematical definitions:**

- **CUSUM (Cumulative Sum Control Chart) Filter**: A technique to detect shifts in the mean of a process away from a target value. For a locally stationary process $\{y_t\}$, the cumulative sum $S_t$ is defined as: $S_t = \max\{0, S_{t-1} + y_t - E_{t-1}[y_t]\}$.
- **Symmetric CUSUM filter** for detecting both upward and downward shifts:
  - $S_t^+ = \max\{0, S_{t-1}^+ + y_t - E_{t-1}[y_t]\}$, with $S_0^+ = 0$.
  - $S_t^- = \min\{0, S_{t-1}^- + y_t - E_{t-1}[y_t]\}$, with $S_0^- = 0$. (Note: The original text contained a typo Smin 0, interpreted here as min).
  - $S_t = \max\{S_t^+, -S_t^-\}$.
- **Geometric Brownian Motion**: Assumed as the underlying model for stock prices in the Triple Barrier method, dictating that prices follow a random walk with drift and variance.

**Key lemmas or propositions:**

- **Empirical Rule**: Assuming log-normal distribution of stock prices, 99.7% of data lies within 3 standard deviations, 95% within 2 standard deviations, and 68% within 1 standard deviation.

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: Implementation of Meta-Labeling utilizing the mlfinlab open-source Python package applied to Trend-Following and Bollinger Band Mean-Reverting strategies.

**Core mathematical formulation:**

- **Triple Barrier Labeling**: Sets an upper horizontal barrier (take profit), a lower horizontal barrier (stop loss), and a vertical barrier (time limit). The horizontal barriers are determined by the daily standard deviation of log returns multiplied by a user-defined multiple. If the upper barrier is hit first, it returns 1; if the lower is hit, it returns -1; if the vertical is hit first, it returns 0 (which is also treated as a -1 in a binary 0,1 scheme).
- **Meta-Labeling**: A primary model sets the side (-1, 0, 1), and a secondary ML model learns the size of the bet (1 to trade, 0 to pass).

**Algorithm pseudocode:**

```
1. Compute long/short signals for the strategy (Primary model).
2. Get time stamps of the events using a CUSUM filter and point estimate of volatility.
3. Determine events when one of the three exit points (profit taking, stop-loss,
   vertical barrier) occur to get trade direction (1 or -1).
4. Determine the bet size (1 or 0) using meta-labeling.
5. Tune hyperparameters (max_depth, n_estimators) of Random Forest using grid
   search and cross-validation.
6. Train the machine-learning algorithm with engineered features. Iterate until
   in-sample results are acceptable.
7. Evaluate the performance of the meta-model on out-of-sample data.
8. Evaluate the performance of the Primary model.
9. Compare the performance of the meta-model against the primary model.
```

**Architecture details:**

- **Primary Model 1 (Trend Following)**: A 20-bar (fast) and 50-bar (slow) Simple Moving Average (SMA) crossover strategy.
- **Primary Model 2 (Mean Reversion)**: Bollinger Bands set at 1.5 standard deviations around the average closing price.
- **Secondary Model (Meta-Labeling)**: Random Forest classifier.

**Innovative components:**

- The creation of mlfinlab, an open-source Python package that democratizes Dr. Marcos Lopez de Prado's advanced financial machine learning concepts (event-based sampling, Triple Barrier labeling, CUSUM filtering, and Meta-Labeling).
- Providing empirical proof that meta-labeling effectively filters out false positives from baseline technical strategies, thereby boosting Sharpe ratios and reducing maximum drawdowns.

## 3. Data

**Dataset name(s)**: S&P 500 E-mini futures tick data. MNIST dataset (used strictly for a non-financial toy example).

**Data source**: Tick Data LLC (proprietary, purchased for approximately $750).

**Data size**: Time period from 01 July 2011 to 13 February 2019 at tick-level frequency (millisecond timestamps).

**Data preprocessing:**

- **Sampling**: Converted raw tick data into volume bars, tick bars, and dollar bars to recover partial statistical normality.
- **Event Extraction**: Applied CUSUM filter with a point-in-time volatility threshold to sample key events and prevent whipsaws.
- **Class Balancing**: Utilized up-sampling (sampling with replacement) on the training data to balance classes, as there were significantly more "0" (do not trade) labels than "1" (trade) labels.
- **Feature Engineering**: Engineered features include 14-day RSI, volatility (50-day, 30-day, 15-day), moving averages (7-day, 15-day), 1 to 5-day auto-correlation, and 1 to 5-day momentum/returns. Features were lagged to prevent look-ahead bias.

**Train/validation/test split:**

- **Toy Example (MNIST)**: 90% train, 10% validation.
- **Mean Reverting Strategy (Out-of-sample Test)**: 2018-01-04 to 2019-01-28.
- **Trend Following Strategy (Out-of-sample Test)**: 2018-01-18 to 2019-01-31.

**Data augmentation techniques**: Up-sampling (sampling with replacement) was used to correct class imbalances in the training set.

## 4. Experimental setup

**Baseline models compared against**: The Primary Models (Trend-following and Mean-reverting strategies executed without the secondary meta-labeling model).

**Evaluation metrics:**

- **Precision**: True Positives / (True Positives + False Positives).
- **Recall (True Positive Rate)**: True Positives / (True Positives + False Negatives).
- **Accuracy**: (True Positives + True Negatives) / Total observations.
- **F1-score**: Harmonic average between precision and recall.
- **Strategy Metrics**: Annual return, Cumulative returns, Annual volatility, Sharpe ratio, Calmar ratio, Max drawdown, Daily value at risk.

**Hyperparameters:**

- **Trend Following**: 20-bar (fast) and 50-bar (slow) Simple Moving Averages.
- **Mean Reversion**: Bollinger bands at 1.5 standard deviations.
- **Random Forest**: Hyperparameters tuned included n_estimators, max_features, and min_samples_leaf using Grid Search.

**Hardware/software environment**: Python, Github, Travis CI. Libraries used include Scikit-learn (GridSearchCV) and Keras (for logistic regression in the toy example).

**Number of runs / seeds**: The random state was kept constant during grid search and cross-validation for reproducibility.

**Statistical tests performed**: Jarque-Bera test (to test for normality of sampled bars) and Auto-Correlation Function (ACF) tests.

## 5. Results

**Quantitative results:**

- **Statistical Properties of Bars (Jarque-Bera Test)**: Time (1782853), Tick (2898186), Volume (337591), Dollar (143045), proving dollar bars are closest to normality.
- **Mean Reverting Strategy (Validation Data):**
  - Primary Model: Accuracy 20.7%, Precision (Class 1) 0.21, Recall 1.00, F1 0.34.
  - Meta Model: Accuracy 77.36%, Precision (Class 1) 0.39, Recall 0.16, F1 0.23.
- **Mean Reverting Strategy (Out-of-Sample Data):**
  - Primary Model: Accuracy 16.78%, Precision (Class 1) 0.17, Recall 1.00.
  - Meta Model: Accuracy 63.11%, Precision (Class 1) 0.20, Recall 0.41.
  - Primary Strategy Metrics: Annual return (17.7%), Volatility (95.0%), Sharpe (0.65), Max Drawdown (-61.9%).
  - Meta Model Strategy Metrics: Annual return (35.3%), Volatility (56.7%), Sharpe (0.82), Max Drawdown (-36.8%).
- **Trend Following Strategy (Validation Data):**
  - Primary Model: Accuracy 37.0%, Precision (Class 1) 0.37, Recall 1.00.
  - Meta Model: Accuracy 56.43%, Precision (Class 1) 0.42, Recall 0.44.
- **Trend Following Strategy (Out-of-Sample Data):**
  - Primary Model: Accuracy 48.14%, Precision (Class 1) 0.48, Recall 1.00.
  - Meta Model: Accuracy 55.33%, Precision (Class 1) 0.54, Recall 0.45.
  - Primary Strategy Metrics: Annual return (310.2%), Volatility (121.0%), Sharpe (1.66), Max Drawdown (-61.7%).
  - Meta Model Strategy Metrics: Annual return (182.2%), Volatility (68.1%), Sharpe (1.82), Max Drawdown (-22.4%).
- **MNIST Toy Example (Out-of-Sample)**: Accuracy improved from 0.8933 (Base) to 0.9569 (Meta). False positives were reduced from 192 to 35.

**Qualitative results:**

- Dollar bars successfully exhibited the lowest auto-correlation and closest recovery to a normal distribution compared to time, tick, and volume bars.
- Meta-labeling effectively identifies false positives generated by the primary model and reclassifies them as true negatives, directly boosting precision and accuracy.

**Ablation studies**: [not found]

**Comparison with baselines:**

- For the Mean Reverting strategy, the Meta Model vastly outperformed the Primary Model in out-of-sample testing, increasing the Sharpe ratio from 0.65 to 0.82 and reducing the Max Drawdown from -61.9% to -36.8%.
- For the Trend Following strategy, while the Meta Model had a lower absolute Annual Return (182.2% vs 310.2%), it significantly outperformed the Primary Model on a risk-adjusted basis, increasing the Sharpe ratio from 1.66 to 1.82 and reducing the Max Drawdown from -61.7% to -22.4%.

## 6. Direct quotations

- "Our results confirm the fact that a combination of event-based sampling, triple-barrier method and meta-labeling improves the performance of the strategies." (p. 1)
- "Second, we show that by using tick data and converting into event based sampling methods such as volume, dollar or tick leads to better statistical properties of the data and that in turn helps machine learning algorithms learn and predict." (p. 2)
- "The ACF of the bars show that dollar bars have the lowest auto-correlation among all others." (p. 5)
- "We can see that in the confusion matrix, that the false positives from the primary model, are now being correctly identified as true negatives with the help of meta-labeling. This leads to a boost in performance metrics. Meta-labeling works as advertised!" (p. 37)
- "This shows that the meta-model adds a lot of value to the out-of-sample performance. All the metrics have improved across the board." (p. 20)

## 7. Authors' conclusions

The authors conclude that implementing the advanced financial machine learning concepts outlined by Dr. Marcos Lopez de Prado, specifically event-based sampling, CUSUM filtering, Triple Barrier Labeling, and Meta-Labeling, materially improves quantitative trading strategies. They demonstrated that converting standard time-series data into dollar bars partially recovers normality and removes autocorrelation, establishing a superior foundation for machine learning. Most critically, their empirical backtesting of both mean-reverting and trend-following strategies proved that secondary meta-labeling models effectively filter out false positive signals generated by primary technical models. This decoupling of bet side from bet size consistently leads to higher precision, significantly better risk-adjusted returns (Sharpe ratios), and heavily reduced maximum drawdowns out-of-sample.

## 8. Limitations stated by authors

- High-quality tick data is costly and generally not readily available for independent research, presenting a barrier to entry.
- Meta-labeling requires a competent primary algorithm that already possesses good in-sample performance; if the primary algorithm is poor, meta-labeling can only limit downside risk but cannot manufacture alpha.
- The strategies evaluated were basic technical analysis models, which inherently do not provide the best signals; a primary model with stronger predictive power is necessary for deeper insights.
- As the quantitative finance community becomes increasingly familiar with these advanced techniques (meta-labeling, robust backtesting), their democratization is likely to exert downward pressure on available alpha in the market.

## 9. Future work suggested by authors

- Implement cross-validation best practices for more robust model evaluation.
- Integrate proper position sizing (bet sizing) and rigorous risk management into the strategies to provide a more realistic performance assessment.
- Build unit-tests for the mlfinlab library functions and construct a comprehensive "feature zoo".
- Redesign the trend-following strategy utilizing newly engineered features and superior primary models.
- Test the established framework and strategies on alternative datasets, such as the Euro STOXX index tick data.

## Additional critical details

- A major contribution of this project was the initial development and open-sourcing of the mlfinlab Python package, which translates Lopez de Prado's theories into usable code and has subsequently generated significant interest and collaboration within the quantitative finance community.

---
**Extraction date**: 2026-05-19
**PDF source**: [not found]