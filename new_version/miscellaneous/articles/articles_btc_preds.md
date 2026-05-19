# Article record: Next step for bitcoin: Confluence of technical indicators and machine learning

## Metadata
- **BibTeX key**: mate_confluence_2024
- **Authors**: Domicián Máté, Hassan Raza, Ishtiaq Ahmad, Sándor Kovács
- **Year**: 2024
- **Title**: Next step for bitcoin: Confluence of technical indicators and machine learning
- **Journal/Conference**: Journal of International Studies, Vol. 17, No. 3
- **DOI/URL**: 10.14254/2071-8330.2024/17-3/4
- **Keywords**: bitcoin forecasting, comparative analysis, cryptocurrency market trends, machine learning algorithms, predictive model evaluation

## 1. Theoretical foundation

**Key theorems**: [not found]

**Key mathematical definitions**: [not found]

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: Artificial Neural Network (ANN), Support Vector Machines (SVM), Convolutional Neural Network and Long Short-Term Memory (CNN-LSTM), and Random Forest (RF).

**Core mathematical formulation:**

- ANN weighted sum: $Z_i = \sum_j W_{i,j}x_j + b_i$
- ANN hidden layer (ReLU): $\alpha_i = \text{Max}(0, z_j)$
- ANN output layer (Sigmoid): $h(x) = \frac{1}{1 + e^{-z}}$
- SVM optimal hyperplane: $w \cdot x + b = 0$
- SVM Margin: $\frac{1}{||w||}$
- SVM kernel transformation: $w \cdot \Phi(x) + b = 0$
- RF decision tree: $T_i = f(D_i, F_i)$
- RF tree prediction: $P_i(x) = T_i(x)$
- RF classification (majority voting): $P(x) = \text{mode}\{P_1(x), P_2(x), ..., P_n(x)\}$
- RF regression: $P(x) = \frac{1}{n}\sum_{i=1}^n P_i(x)$

**Technical Indicators (Table 1):**

- $\%K = \frac{(close\_Price - lowest\_low)}{(highest\_high - lowest\_low)} * 100$
- $\%D = (1/n)\sum_{i=0}^{n-1}\Sigma\%K_{i-1}$
- $ROC = \frac{(cp - cp_n)}{cp_n} * 100$
- $\%R = \frac{(highest\_high - close)}{(highest\_high - low)} * 100$
- $Momentum = close - close\_4$
- $Disparity\_5 = \frac{close}{MA5} * 100$
- $Disparity\_14 = \frac{close}{MA14} * 100$
- $OSCP = MA5 - MA10$
- $CCI = \frac{(typical\_price - MA)}{(0.015 \times mean\_deviation)}$
- $RSI = 100 - [\frac{100}{(1 + relative\_strength)}]$
- $PP = \frac{(high + low + close)}{\alpha}$ (where $\alpha=3$)
- $S1 = (PP * 2) - high$
- $S2 = PP - (high - low)$
- $R1 = (PP * 2) - low$
- $R2 = PP + (high - low)$
- $MA_t = EMA_{t-1} + \alpha * (Price_t - EMA_{t-1})$
- $WMA = \frac{\omega_1 P_t + \omega_2 P_{t-1} + \dots + \omega_n P_{t-n}}{\omega_1 + \omega_2 + \dots + \omega_n}$
- $Upper\_Band = SMA + (std\_multiplier * STD)$
- $LB = SMA - (std\_multiplier * STD)$
- $MACD = Short\_EMA - Long\_EMA$
- $Signal\_Line = \text{EMA}(MACD\_Histogram, Signal\_Window)$
- $ATR = \frac{ATR_{prev}(n-1) + TR}{n}$
- $OBV = OBV_{prev} + volume$ (if close > close$_{prev}$); $0$ (if equal); $-volume$ (if close < close$_{prev}$)
- $Chaikin\_Oscillator = CO$ (3 day EMA of ADL) - (10 day EMA of ADL)
- $MFI = 100 - \frac{100}{(1 + Money\_Flow\_Ratio)}$
- $Day\_of\_Week\_Anomaly = \text{close.groupby}(Day\_of\_Week)\text{.trans(mean)}$
- $Week\_of\_Month\_Anomaly = \text{close.groupby}(Week\_of\_month)\text{.trans(mean)}$

**Algorithm pseudocode**: [not found]

**Architecture details:**

- **ANN**: 3 layers. Input layer with 27 features. Hidden layer with 32 neurons using ReLU activation. Output layer with 1 neuron using Sigmoid activation.
- **SVM**: Uses a Radial Basis Function (RBF) kernel.
- **CNN-LSTM**: First LSTM layer has 50 units. Second LSTM layer has 50 units. Followed by a densely connected layer with a single unit using Sigmoid activation.

**Innovative components**: The study's novelty lies in integrating 27 specific technical indicators, heavily focusing on temporal and spatial trends via momentum, ROC, and %K features, with multiple advanced ML architectures (ANN, SVM, CNN-LSTM, RF) in a side-by-side comparative framework for predicting highly volatile cryptocurrency directional movements.

## 3. Data

**Dataset name(s)**: Historical Bitcoin Data

**Data source**: Yahoo Finance API

**Data size**: Spans from September 17, 2014, to October 1, 2023. Daily frequency.

**Data preprocessing:**

- Missing data was eliminated.
- Target formulation: Added a binary "direction" column (1 for upward price movement relative to threshold, 0 for downward movement).
- Temporal alignment: Historical data was shifted forward by one day to prevent lookahead bias when predicting the target.
- Normalization: Min-Max scaling was applied to rescale all feature values to a uniform range between 0 and 1.

**Train/validation/test split**: 80% Training set, 20% Testing set.

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against**: Standard ARIMA (mentioned historically), ANN, SVM, CNN-LSTM, and RF.

**Evaluation metrics:**

- **Accuracy**: Proportion of correctly classified directional movements.
- **Precision**: True Positives / (True Positives + False Positives).
- **Recall**: True Positives / (True Positives + False Negatives).
- **F1-score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Tabulation of True Positives, False Positives, True Negatives, False Negatives.

**Hyperparameters:**

- **ANN**: Optimizer = Adam, Loss Function = Binary cross-entropy, Epochs = 50, Batch Size = 64.
- **CNN-LSTM**: Optimizer = Adam, Loss Function = Binary cross-entropy, Epochs = 10, Batch Size = 32.
- **Random Forest**: Parameters like number of trees ($n$), max depth ($d$), and features per split ($m$) are tuned via cross-validation $\text{argmin}_{n,d,m} \text{CV\_error}(n,d,m)$.

**Hardware/software environment**: Python 3.10.4, using the NumPy, Pandas, TensorFlow, and Scikit-learn libraries.

**Number of runs / seeds**: [not found]

**Statistical tests performed**: [not found]

## 5. Results

**Quantitative results:**

- **ANN Classification Results (Support: Class 0=291, Class 1=330):**
  - Accuracy: 0.81
  - Class 0: Precision=0.78, Recall=0.81, F1=0.80
  - Class 1: Precision=0.83, Recall=0.80, F1=0.81
  - Confusion Matrix: TN=237, FP=54, FN=67, TP=263
- **SVM Classification Results (Support: Class 0=291, Class 1=330):**
  - Accuracy: 0.82
  - Class 0: Precision=0.78, Recall=0.85, F1=0.82
  - Class 1: Precision=0.86, Recall=0.79, F1=0.82
  - Confusion Matrix: TN=248, FP=43, FN=69, TP=261
- **CNN-LSTM Classification Results (Support: Class 0=291, Class 1=330):**
  - Accuracy: 0.80
  - Class 0: Precision=0.76, Recall=0.84, F1=0.80
  - Class 1: Precision=0.84, Recall=0.77, F1=0.81
  - Confusion Matrix: TN=244, FP=47, FN=75, TP=255
- **Random Forest Classification Results (Support: Class 0=291, Class 1=330):**
  - Accuracy: 0.81
  - Class 0: Precision=0.80, Recall=0.80, F1=0.80
  - Class 1: Precision=0.82, Recall=0.83, F1=0.83
  - Confusion Matrix: TN=233, FP=58, FN=57, TP=273

**Qualitative results:**

- Feature importance was visually analyzed via bar charts for each model.
- **ANN Top Features**: %R (0.22014), %D (0.16778), %K (0.15491), Chaikin Oscillator (0.14894).
- **SVM Top Features**: %D (7.36412), %R (5.04162), %K (5.02149).
- **CNN-LSTM Top Features**: %K and ROC (0.3349), Momentum (0.3140), Disparity_5 (0.2641).
- **RF Top Features**: Chaikin Oscillator (0.2435), %R (0.1518), Disparity_5 (0.0700).

**Ablation studies**: [not found]

**Comparison with baselines**: All tested ML models achieved significantly lower error rates than traditional statistical models, improving prediction accuracy by an estimated 20-30% over standard ARIMA models. Among the ML models, SVM achieved the highest testing accuracy (82%), closely followed by RF and ANN (81%), and CNN-LSTM (80%).

## 6. Direct quotations

- "The results showed that ANN and SVM achieve a significant prediction accuracy of 81% and 82%, respectively, which is higher than the results of traditional models such as standard ARIMA." (p. 69)
- "The unique contribution of the current work was the use of temporal and spatial trends via momentum, ROC, and %K features, which demonstrated a holistic approach to cryptocurrency market forecasting." (p. 70)
- "In contrast, machine learning models such as ANN, SVM, CNN-LSTM, and RF can capture complex patterns and relationships within the data, resulting in more accurate predictions. Machine learning models typically achieve significantly lower error rates, often improving prediction accuracy by 20-30% over standard ARIMA models in practical applications..." (p. 89)
- "The comparative lens employed here seeks to uncover the nuances of each model's predictive power, thereby providing a holistic understanding of their effectiveness" (p. 70)
- "These models are highly applicable in real-world trading scenarios, where accurate predictions can lead to significant financial gains. For example, the ANN's reliance on the Williams Percent Range (%R) and the SVM's reliance on the %D moving average underscored the critical role of these technical indicators in guiding trading strategies." (p. 89)

## 7. Authors' conclusions

The authors conclude that machine learning techniques significantly outperform traditional statistical regression methodologies (such as ARIMA) in navigating the extreme volatility and non-linearity of the Bitcoin market. Through comparative analysis, ANN, SVM, CNN-LSTM, and Random Forest models all demonstrated robust predictive capabilities, achieving accuracy rates between 80% and 82%. A critical component of this success was the heavy reliance on data mining and the extraction of 27 specific technical indicators. The feature importance metrics revealed that indicators like Williams %R, Stochastic Oscillators (%K and %D), ROC, and the Chaikin Oscillator act as primary drivers for predictive accuracy. Ultimately, integrating these sophisticated ML algorithms with well-engineered technical indicators provides traders and financial experts with highly reliable tools for anticipating market direction, improving investment strategies, and managing risk.

## 8. Limitations stated by authors

- Historical Bitcoin data quality and reliability can negatively impact the accuracy of machine learning models.
- Sudden, extreme price fluctuations and rapid market shifts present major predictive challenges.
- Deep learning architectures (like CNN-LSTM and large neural networks) demand substantial computational resources.
- Continual regulatory changes in the cryptocurrency space can alter market dynamics, requiring the models to be constantly updated and adapted.

## 9. Future work suggested by authors

- Exploring the interplay of these models (e.g., creating ensembles or hybrids) to push predictive accuracy even higher.
- Refining feature extraction methods.
- Expanding the forecasting dataset to include other cryptocurrencies beyond Bitcoin.
- Incorporating alternative data streams, such as social media sentiment analysis, to extract deeper market insights and improve model robustness.

## Additional critical details

- **Discrepancy in Text vs. Tables**: In the text (page 84), the authors state "Both models [ANN and SVM] achieved the same overall accuracy of 85%". However, Table 4 explicitly records the SVM accuracy as 0.82 (82%), which matches the abstract and conclusion of the paper.
- The target output is formulated strictly as a binary classification problem (1 = upward movement, 0 = downward movement) rather than a continuous price regression.

---
**Extraction date**: 2026-05-19
**PDF source**: Next_step_for_bitcoin__Confluence_of_TA_and_ML.pdf

---

# Article record: Review of deep learning models for crypto price prediction: implementation and evaluation

## Metadata
- **BibTeX key**: wu_crypto_dl_review_2024
- **Authors**: Jingyang Wu, Xinyi Zhang, Fangyixuan Huang, Haochen Zhou, Rohtiash Chandra
- **Year**: 2024
- **Title**: Review of deep learning models for crypto price prediction: implementation and evaluation
- **Journal/Conference**: arXiv preprint arXiv:2405.11431v2 [cs.LG]
- **DOI/URL**: arXiv:2405.11431v2
- **Keywords**: Cryptocurrency, Price prediction, Deep learning, Long short-term memory, Convolutional neural networks, Transformer

## 1. Theoretical foundation

**Key theorems**: [not found]

**Key mathematical definitions**: [not found]

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: The study evaluates and compares seven deep learning architectures: Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), Bidirectional LSTM (Bi-LSTM), Gated Recurrent Unit (GRU), 1-Dimensional Convolutional Neural Network (1D-CNN), Temporal Convolutional Network (TCN), and Transformer.

**Core mathematical formulation:**

- **RNN**: $h_t = \sigma(W_{hx} x_t + W_{hh} h_{t-1} + b_h)$ and $y_t = W_{yh} h_t + b_y$
- **LSTM**:
  - Forget gate: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
  - Input gate: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
  - Cell state update: $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$
  - New cell state: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
  - Output gate: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
  - Hidden state: $h_t = o_t * \tanh(C_t)$
- **GRU**:
  - Update gate: $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$
  - Reset gate: $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$
  - Candidate hidden state: $\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t] + b)$
  - Final hidden state: $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$
- **CNN (1D)**:
  - Convolution operation: $(f * g)(t) = \sum_{k=1}^K f(k) g(t - k \cdot d)$ (where $d$ is the dilation factor)
- **Transformer**:
  - Self-attention: $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V$
  - Multi-head attention: $MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O$, where $head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)$
  - Positional Encoding: $PE_{(pos, 2i)} = \sin(\frac{pos}{10000^{2i/d_{model}}})$ and $PE_{(pos, 2i+1)} = \cos(\frac{pos}{10000^{2i/d_{model}}})$

**Algorithm pseudocode**: [not found]

**Architecture details:**

- The framework implements standard versions of RNN, LSTM, Bi-LSTM, and GRU for sequential modeling.
- 1D-CNN utilizes causal convolutions to prevent information leakage from future to past.
- TCN employs dilated causal convolutions with residual blocks to capture long-term dependencies efficiently.
- Transformer relies entirely on multi-head self-attention mechanisms without recurrence, requiring positional encodings to inject sequence order.

**Innovative components**: The paper provides a comprehensive, head-to-head empirical evaluation of seven distinct DL architectures applied to both univariate and multivariate cryptocurrency forecasting, explicitly partitioning data into "pre-COVID-19" and "COVID-19" periods to assess model robustness under varying market volatility.

## 3. Data

**Dataset name(s)**: Bitcoin (BTC), Ethereum (ETH), Litecoin (LTC), Ripple (XRP).

**Data source**: Yahoo Finance.

**Data size:**

- Overall time period: August 2015 to August 2023.
- Pre-COVID-19 subset: August 2015 to January 2020.
- COVID-19 subset: January 2020 to August 2023.
- Frequency: Daily data.
- Features: 6 features (Open, High, Low, Close, Adjusted Close, Volume).

**Data preprocessing:**

- Normalization: Min-Max scaling to the range (0, 1) using $X_{nom} = \frac{X - X_{min}}{X_{max} - X_{min}}$.
- Sequence structuring: A sliding window technique was utilized to format the time series.
- Look-back window (input sequence length): 60 days.
- Prediction horizons: 1-day, 3-day, and 7-day ahead.

**Train/validation/test split**: 70% Training, 10% Validation, 20% Testing (chronologically split to prevent look-ahead bias).

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against**: The study mutually compares RNN, LSTM, Bi-LSTM, GRU, 1D-CNN, TCN, and Transformer (none is explicitly designated as the sole baseline; they are benchmarked against each other).

**Evaluation metrics:**

- Root Mean Squared Error (RMSE): $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$.
- Mean Absolute Error (MAE): $MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$.
- Mean Absolute Percentage Error (MAPE): $MAPE = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|$.
- R-squared ($R^2$): $R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$.

**Hyperparameters:**

- Hidden units tested: 32, 64, 128 (best performing were generally 64 or 128 depending on the model/dataset).
- Batch size: 64.
- Epochs: 100.
- Early stopping: Patience set to 10 epochs.
- Optimizer: Adam.
- Learning rate: Tested 0.01, 0.001, and 0.0001; 0.001 was selected as optimal.
- Regularization: Dropout rate of 0.2.
- Activation functions: Tanh for recurrent layers, ReLU for CNN layers.
- Loss function: Mean Squared Error (MSE).
- Transformer parameters: Number of attention heads = 4, Number of encoder layers = 2.

**Hardware/software environment**: Python programming language using Keras and TensorFlow 2.12.0; executed on Kaggle Notebooks equipped with dual Tesla T4 GPUs.

**Number of runs / seeds**: 10 independent runs for each experimental setting to compute mean and standard deviation.

**Statistical tests performed**: [not found]

## 5. Results

**Quantitative results:**

- **Pre-COVID-19 (Univariate, 1-day ahead), Best Models:**
  - BTC: TCN (RMSE: $0.0210 \pm 0.0003$, MAE: $0.0142 \pm 0.0001$, $R^2$: $0.9702 \pm 0.0004$).
  - ETH: TCN (RMSE: $0.0381 \pm 0.0004$, MAE: $0.0163 \pm 0.0003$, $R^2$: $0.9575 \pm 0.0010$).
  - LTC: Bi-LSTM (RMSE: $0.0392 \pm 0.0003$, $R^2$: $0.9328 \pm 0.0011$).
  - XRP: Bi-LSTM (RMSE: $0.0366 \pm 0.0005$, $R^2$: $0.9403 \pm 0.0014$).
- **COVID-19 (Univariate, 1-day ahead), Best Models:**
  - BTC: TCN (RMSE: $0.0249 \pm 0.0001$, MAE: $0.0177 \pm 0.0002$, $R^2$: $0.9855 \pm 0.0001$).
  - ETH: TCN (RMSE: $0.0270 \pm 0.0002$, MAE: $0.0183 \pm 0.0003$, $R^2$: $0.9846 \pm 0.0002$).
  - LTC: TCN (RMSE: $0.0354 \pm 0.0002$, MAE: $0.0219 \pm 0.0003$, $R^2$: $0.9443 \pm 0.0005$).
  - XRP: TCN (RMSE: $0.0315 \pm 0.0001$, MAE: $0.0197 \pm 0.0002$, $R^2$: $0.9634 \pm 0.0003$).
- **Pre-COVID-19 (Multivariate, 1-day ahead), Best Models:**
  - BTC: Bi-LSTM (RMSE: $0.0219 \pm 0.0016$, MAE: $0.0146 \pm 0.0012$, $R^2$: $0.9678 \pm 0.0044$).
  - ETH: TCN (RMSE: $0.0401 \pm 0.0028$, $R^2$: $0.9526 \pm 0.0069$).
  - LTC: TCN (RMSE: $0.0393 \pm 0.0006$, $R^2$: $0.9323 \pm 0.0016$).
  - XRP: GRU (RMSE: $0.0375 \pm 0.0020$, $R^2$: $0.9367 \pm 0.0076$).
- **COVID-19 (Multivariate, 1-day ahead), Best Models:**
  - BTC: 1D-CNN (RMSE: $0.0246 \pm 0.0013$, MAE: $0.0177 \pm 0.0010$, $R^2$: $0.9859 \pm 0.0016$).
  - ETH: 1D-CNN (RMSE: $0.0279 \pm 0.0005$, $R^2$: $0.9835 \pm 0.0006$).
  - LTC: 1D-CNN (RMSE: $0.0355 \pm 0.0005$, $R^2$: $0.9443 \pm 0.0017$).
  - XRP: 1D-CNN (RMSE: $0.0322 \pm 0.0003$, $R^2$: $0.9616 \pm 0.0009$).
- **Transformer Performance (Example: Pre-COVID Univariate BTC, 1-day):** Transformer yielded an RMSE of $0.0324 \pm 0.0069$ and $R^2$ of $0.9290 \pm 0.0298$, significantly worse than TCN and showing much higher standard deviation (instability).

**Qualitative results:**

- The Transformer model exhibited the poorest performance and highest instability across all datasets, struggling to capture the volatility of cryptocurrency prices and often forecasting a "flat line" that missed extreme price fluctuations entirely.
- TCN captured actual trends highly effectively, closely matching extreme peaks and troughs during periods of high volatility.
- Univariate models generally outperformed multivariate models for 1-day ahead predictions, indicating that auxiliary features like trade volume may introduce noise rather than predictive signal for short horizons.
- Predictive accuracy degraded severely as forecasting horizons expanded from 1-day to 3-day and 7-day windows due to the accumulation of compounding errors.

**Ablation studies:**

- **Dataset Split**: Partitioned data into Pre-COVID-19 (lower volatility) and COVID-19 (higher volatility) to test model robustness. Found that TCN and 1D-CNN handled the high volatility of the COVID-19 period better than RNN variants.
- **Feature Input**: Compared Univariate vs. Multivariate inputs. Found that Univariate inputs led to lower error metrics overall for 1-step predictions.
- **Prediction Horizon**: Compared 1-day, 3-day, and 7-day ahead forecasts. Found significant performance degradation in multi-step forecasting across all models.

**Comparison with baselines:**

- TCN outperformed all RNN, LSTM, GRU, and Transformer variants for 1-day univariate tasks across the board (e.g., lower RMSE and higher $R^2$).
- 1D-CNN outperformed recurrent models in the multivariate COVID-19 dataset, showcasing superior feature extraction from multiple variables during volatile periods.
- Transformer was consistently outperformed by all convolutional and recurrent models.

## 6. Direct quotations

- "We evaluate univariate and multivariate approaches for multi-step ahead predicting of cryptocurrencies close-price." (p. 1)
- "The results demonstrate that the Temporal Convolutional Network (TCN) consistently outperforms other models in the univariate dataset, establishing it as the superior choice." (p. 2)
- "Our results demonstrate that TCN, 1D-CNN, LSTM, GRU, and Bi-LSTM all exhibit better prediction performance for univariate models when forecasting 1 day ahead." (p. 18)
- "The Transformer framework... exhibits the poorest performance across all selected evaluation criteria, accompanied by larger standard deviations, highlighting its instability." (p. 19)
- "Generally, prediction error increases with longer prediction horizons... the models exhibit greater stability in their single-step predictions." (p. 25)

## 7. Authors' conclusions

The authors conclude that convolutional architectures, specifically the Temporal Convolutional Network (TCN) and 1D-CNN, are highly effective for cryptocurrency price prediction. TCN emerged as the superior model for univariate, single-step forecasting because its dilated causal convolutions efficiently capture both local trends and long-term dependencies without suffering from vanishing gradients. Meanwhile, the 1D-CNN proved most robust for multivariate forecasting during periods of extreme market volatility (such as the COVID-19 era). Conversely, the standard Transformer architecture yielded the worst performance and highest instability, struggling to adapt to the noisy and highly volatile nature of financial time series data. Additionally, the study found that univariate models generally outperformed multivariate models for short-term predictions, suggesting that adding multiple market features (like volume) introduces noise rather than signal. Finally, all models suffered significant accuracy degradation as the forecasting horizon extended from 1 to 3 and 7 days.

## 8. Limitations stated by authors

- The study's multivariate analysis relied solely on historical price and volume data, which introduced noise; it did not incorporate external factors such as macroeconomic indices, social media sentiment, or on-chain metrics.
- Long-term predictions (3-day and 7-day ahead) suffered heavily from compounding errors, making them unreliable for practical trading applications.
- The poor performance of the Transformer model suggests that the standard self-attention mechanism may not be suitable for noisy financial time series without customized modifications.

## 9. Future work suggested by authors

- Integrating external features, such as social media sentiment analysis, global economic indicators, and blockchain network data (e.g., hash rate) to improve multivariate forecasting accuracy.
- Exploring hybrid architectures (such as CNN-LSTM) to simultaneously leverage the spatial feature extraction capabilities of CNNs and the temporal sequence memory of LSTMs.
- Implementing advanced data augmentation and denoising techniques (such as wavelet transforms) to filter out market noise before feeding the data into deep learning models.
- Investigating specialized Transformer variants explicitly tailored for time series forecasting (such as Informer, Autoformer, or PatchTST) to overcome the limitations observed with the standard Transformer architecture.

---
**Extraction date**: 2026-05-19
**PDF source**: arXiv:2405.11431

---

# Article record: Deep learning for Bitcoin price direction prediction: models and trading strategies empirically compared

## Metadata
- **BibTeX key**: omole_enke_2024
- **Authors**: Oluwadamilare Omole, David Enke
- **Year**: 2024
- **Title**: Deep learning for Bitcoin price direction prediction: models and trading strategies empirically compared
- **Journal/Conference**: Financial Innovation, 10:117
- **DOI/URL**: https://doi.org/10.1186/s40854-024-00643-1
- **Keywords**: Backtesting, Bitcoin, Cryptocurrency, Deep learning, Feature selection, On-chain data

## 1. Theoretical foundation

**Key theorems:**

- **Universal Approximation Theorem**: Mentioned conceptually as the basis for deep learning implementations.

**Key mathematical definitions:**

- **Class 1 (Upward Movement)**: Scenario where the price of Bitcoin shows an increase from day $t$ to day $t+1$ ($y_{t+1} > y_t$), encoded as 1.
- **Class 0 (Downward/Stable Movement)**: Scenario where the Bitcoin price either decreases or stays the same from day $t$ to day $t+1$ ($y_{t+1} \le y_t$), encoded as 0.

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: The study evaluates CNN-LSTM (Convolutional Neural Network - Long Short-Term Memory), LSTNet (Long- and Short-Term Time-Series Network), and TCN (Temporal Convolutional Network) combined with feature selection methods (Boruta, GA, LightGBM).

**Core mathematical formulation:**

- MACD line: $MACDline = EMA_{12-day} - EMA_{26-day}$ (Eq. 6)
- Signal line: $Signalline = EMA_{9-day}(MACDline)$ (Eq. 7)
- Rate of Return (ROR): $ROR = \frac{\text{End Portfolio} - \text{Starting Portfolio}}{\text{Starting Portfolio}} \times 100$ (Eq. 8)
- Maximum Drawdown (MDD): $MDD = \frac{\text{Trough Value} - \text{Peak Value}}{\text{Peak Value}}$ (Eq. 9)
- Sharpe Ratio: Computed using Expected Return, Risk Free Rate, and Excess Return/Volatility Return% variables (Eq. 10 / Eq. 12 formatting in text).
- Market Exposure: $\text{Market Exposure} = \frac{\text{Trade Duration}}{\text{Total Period}}$ (Eq. 11)

**Algorithm pseudocode (Boruta Feature Selection Algorithm):**

```
1. Duplicate original dataset to create a "shadow dataset."
2. Shuffle shadow features randomly to eliminate correlation with the target.
3. Concatenate original and shadow datasets; train a Random Forest (RF) classifier.
   Calculate and record Z-scores.
4. Determine the Maximum Z-score among shadow attributes (MZSA).
5. Perform a two-sided test on MZSA. Features significantly less critical than MZSA
   are removed; features significantly more important are retained.
6. Discard shadow features and iterate until a predefined stopping criterion is reached.
```

**Architecture details:**

- **CNN-LSTM**:
  - CNN component: 1D Conv layer (filter size=64, kernel size=3, ReLU) $\rightarrow$ Batch Normalization $\rightarrow$ Average Pooling 1D (pool size=1) $\rightarrow$ Dropout (rate=0.5).
  - LSTM component: LSTM layer (128 units, TanH) $\rightarrow$ Batch Normalization $\rightarrow$ Dropout (0.5) $\rightarrow$ LSTM layer (80 units, TanH) $\rightarrow$ Batch Normalization $\rightarrow$ Dropout (0.5).
  - Output component: Dense layer (1 unit, Sigmoid).
- **LSTNet**:
  - CNN component: Two 1D CNN layers (filters=128, 64; kernel=3) $\rightarrow$ three time-distributed dense layers (units=64, 32, 1) $\rightarrow$ global average pooling.
  - LSTM component: Two LSTM layers (units=256, 128) $\rightarrow$ three time-distributed dense layers (units=64, 32, 1) $\rightarrow$ global average pooling.
  - AR component: Two layers (units=64, 32) $\rightarrow$ dense layer (1 unit) $\rightarrow$ global average pooling.
  - Fusion: Outputs added $\rightarrow$ dense layer (64 units) $\rightarrow$ dropout (0.5) $\rightarrow$ dense layer (32 units) $\rightarrow$ dropout (0.5) $\rightarrow$ output dense layer (1 unit, Sigmoid).
- **TCN**:
  - Layer 1: 64 filters, kernel size=3, dilations=1, 2, 4, 8, ReLU $\rightarrow$ Dropout (0.5). Output is 3D.
  - Layer 2: 64 filters, kernel size=3, dilations=1, 2, 4, 8, 16, ReLU $\rightarrow$ Dropout (0.5). Output is 2D.
  - Output: Dense layer (1 unit, Sigmoid).

**Innovative components**: The integration of 87 specific blockchain on-chain data metrics, rigorously reduced via wrapper/embedded feature selection algorithms (Boruta, GA, LightGBM), subsequently fed into complex hybrid deep learning architectures (CNN-LSTM, LSTNet, TCN) and validated via simulated trading strategy backtesting.

## 3. Data

**Dataset name(s)**: Bitcoin On-Chain and Price Data.

**Data source**: Glassnode (2023).

**Data size**: 3,665 daily observations (February 6, 2013, to February 18, 2023). 87 distinct features (including price).

**Data preprocessing:**

- Missing completely at random (MCAR): Listwise deletion.
- Missing not at random (MNAR): Regression imputation.
- Transformation: Min-max standardization applied after the train/test split to prevent data leakage.
- Feature Selection Subsets generated: Boruta (26 features), Genetic Algorithm (42 features), LightGBM (43 features), All Data (87 features), Univariate (1 feature - price only).

**Train/validation/test split**: 80% Training, 20% Testing (chronological split, no shuffling).

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against**: ARIMA (for predictive modeling accuracy), MACD (for trading strategy profitability baseline).

**Evaluation metrics:**

- Accuracy: $\frac{TP+TN}{TP+TN+FP+FN}$ (Eq. 1)
- Precision: $\frac{TP}{TP+FP}$ (Eq. 2)
- Recall: $\frac{TP}{TP+FN}$ (Eq. 3)
- F1-score: $\frac{2*(Precision*Recall)}{Precision+Recall}$ (Eq. 4)
- MCC: $\frac{TP*TN - FP*FN}{\sqrt{(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)}}$ (Eq. 5)
- AUC-ROC

**Hyperparameters:**

- Explored via random search: Learning rate, batch size, epoch count, dropout coefficient, LSTM units, CNN filter dimensions, convolutional kernel dimensions, and activation functions.
- Fixed configurations: Seeds varied from 0 to 10. Lookback window sizes varied: 3, 5, 7, 14, and 30 days.

**Hardware/software environment**: Python (implied by "Python's random number generator").

**Number of runs / seeds**: Evaluated across 11 different seed values (0 through 10) and 5 different window sizes.

**Statistical tests performed**: Wilcoxon signed-rank test (pairwise comparisons), Friedman test (multiple comparisons).

## 5. Results

**Quantitative results:**

- **Max Accuracy by Feature Selection + Model:**
  - Boruta + CNN-LSTM: 0.8244 (Mean $\pm$ SD across seeds: $0.7439 \pm 0.050$, across windows: $0.7934 \pm 0.030$)
  - LightGBM + CNN-LSTM: 0.7662
  - AllData + LSTNet: 0.7449
  - GA + CNN-LSTM: 0.7431
  - Boruta + TCN: 0.7194
  - AllData + TCN: 0.7194
  - Boruta + ARIMA: 0.5416
  - Univariate + CNN-LSTM: 0.5099
  - GA + ARIMA: 0.5061
- **Detailed Metrics for Best Model (Boruta + CNN-LSTM):**
  - Accuracy: 0.8244
  - Recall: 0.8078
  - Precision: 0.8309
  - F-1 Score: 0.8192
  - AUC-ROC: 0.8242
  - MCC: 0.6489
- **Statistical Tests:**
  - Friedman test (All models): Statistic = 178, P-value = 7.202e-28 (Reject null).
  - Wilcoxon test (Boruta+CNN-LSTM vs LightGBM+CNN-LSTM): Statistic = 13, P-value = 0.0830 (Accept null). All other pairwise comparisons against Boruta+CNN-LSTM rejected the null hypothesis (p < 0.05).
- **Backtesting / Trading Simulation Results (Boruta + CNN-LSTM over 729 test days):**
  - Long-only buy-and-sell: Annual Return = 437.2461%, Sharpe = 0.1312, MDD = -0.1131, Market Exposure = 0.4787, Trades = 203.
  - Short-only buy-and-sell: Annual Return = 1084.1600%, Sharpe = 0.1814, MDD = -0.0582, Market Exposure = 0.5213, Trades = 204.
  - Long and short: Annual Return = 6653.7497%, Sharpe = 1.8583, MDD = -0.0704, Market Exposure = 0.5583, Trades = 406.
  - Long-only with price protection: Annual Return = 8.6255%, MDD = -0.5761, Trades = 25.
  - Baseline (Long and short MACD): Annual Return = -55.5873%, Sharpe = 0.0032, MDD = -0.7968, Trades = 65.

**Qualitative results:**

- The CNN-LSTM model struggled to accurately differentiate noise from signal without feature selection, underperforming on the "AllData" set but excelling on the feature-reduced sets.
- LSTNet consistently underperformed on test data despite excellent training performance, indicating extreme overfitting and architectural over-complexity.
- TCN proved highly effective at distinguishing noise, performing best when given the raw, unselected dataset ("AllData"), but suffering from underfitting when applied to the reduced feature sets.

**Ablation studies:**

- Tested various lookback window sizes (3, 5, 7, 14, 30 days). LSTNet peaked at window=5 and declined. TCN initially improved but subsequently declined. CNN-LSTM showed no linear trend with window size but remained superior across sets.
- Tested different seed initializations (0 to 10). Performance varied significantly based on seed, proving the necessity of evaluating deep learning models across multiple weight initializations.

**Comparison with baselines:**

- Deep learning models dramatically outperformed the ARIMA benchmark (max accuracy 0.5416).
- The DL-driven "long and short" trading strategy generated 6,653% annualized returns compared to the traditional MACD strategy which yielded -55.58% returns over the same period.

## 6. Direct quotations

- "The novelty in this research lies in combining deep learning (DL) models, input types, and output types that have not been sufficiently explored in the literature. The inputs for the model were Bitcoin price and a large set of on-chain data." (p. 3)
- "Extensive experimentation revealed that the combination of Boruta feature selection and the CNN-LSTM model consistently outperformed other configurations, attaining an accuracy of 82.44%." (p. 21)
- "LSTNet model consistently underperformed on test data despite exhibiting excellent training performance. This indicates that the model is overly complex and prone to overfitting." (p. 18)
- "The 'long-and-short' strategy generated a 6653% return while maintaining a manageable MDD of 0.0704." (p. 21)
- "With more cryptocurrency price modeling (in this case, Bitcoin pricing), price transparency should increase along with efficiency in these historically volatile markets. Thus, proper modeling and better price transparency should make it possible for securities exchanges and regulators to offer and approve new security offerings, such as a spot Bitcoin ETF..." (p. 21-22)

## 7. Authors' conclusions

The authors conclude that deep learning methodologies, when properly combined with aggressive feature selection, are highly effective at forecasting the directional movement of Bitcoin prices using on-chain data. The Boruta feature selection algorithm coupled with the CNN-LSTM architecture emerged as the superior predictive framework, achieving an accuracy of 82.44%. The study demonstrates that supplying raw, high-dimensional on-chain data directly to models without dimensionality reduction generally degrades predictive power (as seen with LSTNet overfitting). Furthermore, translating these high-accuracy forecasts into automated trading strategies proved exceptionally profitable; a combined long-and-short strategy yielded an annualized return exceeding 6,600% with manageable drawdowns, vastly outperforming traditional MACD benchmark strategies. Ultimately, robust predictive modeling using blockchain-native (on-chain) data can increase market transparency and efficiency, paving the way for institutional adoption and the approval of regulated financial products like spot ETFs.

## 8. Limitations stated by authors

- The study exclusively utilized on-chain metrics and historical price data; it omitted other potentially highly predictive features such as social sentiment data and traditional technical analysis indicators.
- The prediction time frame was relatively short-term, meaning the models' robustness in capturing macro-level, long-term secular market trends remains unverified.

## 9. Future work suggested by authors

- Incorporating additional data types, particularly sentiment data and technical analysis indicators, to ascertain whether they yield synergistic improvements in prediction accuracy.
- Investigating model performance under varying macroeconomic conditions to gain a comprehensive understanding of their effectiveness at capturing long-term market trends.

## Additional critical details

- **Data Ground Truth Simulation**: To contextualize the 6,653% return, the authors ran a theoretical 100% accurate simulation (the "ground truth" strategy) over the exact same period, which yielded an astronomical 20,453% return for the long-and-short strategy, establishing the absolute ceiling for trading profitability in that window.
- **Trading Simulation Parameters**: The backtest assumed a starting capital of $1,000, 100% compounded capital reinvestment after each trade, a strict 0.5% transaction cost per trade, and a 30% tax rate applied to each realized profit. Slippage and liquidity constraints were excluded.

---
**Extraction date**: 2026-05-19
**PDF source**: DL_for_BTC_price_direction_prediction.pdf

---

# Article record: Cryptocurrency Forecasting Using Deep Learning Models: A Comparative Analysis

## Metadata
- **BibTeX key**: bourday_crypto_2024
- **Authors**: Rachid Bourday, Issam Aatouchi, Mounir Ait Kerroum, Ali Zaaouat
- **Year**: 2024
- **Title**: Cryptocurrency Forecasting Using Deep Learning Models: A Comparative Analysis
- **Journal/Conference**: HighTech and Innovation Journal, Vol. 5, No. 4
- **DOI/URL**: 10.28991/HIJ-2024-05-04-013
- **Keywords**: Cryptocurrency; Deep Learning; Machine Learning; Transformers; Bitcoin Price

## 1. Theoretical foundation

**Key theorems**: [not found]

**Key mathematical definitions**: [not found]

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: A hybrid pipeline utilizing a Transformer as a Feature Extractor coupled with four distinct prediction models: Transformer-LSTM, Transformer-ANN, Transformer-XGBoost, and Transformer-SVR.

**Core mathematical formulation:**

- Scaled Dot-Product Attention: $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
- Multi-Head Attention: $MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$, where $head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$
- Position-wise Feed-Forward Network: $FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$
- LSTM Cell state: $c_t = f_t * c_{t-1} + i_t * \tilde{c}_t$
- LSTM Hidden state: $h_t = o_t * \tanh(c_t)$
- SVR Objective: Minimize $\frac{1}{2} ||w||^2$ subject to $|y_i - \langle w, x_i \rangle - b| \le \varepsilon$
- XGBoost Objective function: $obj = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$
- Min-Max Normalization: $X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$

**Algorithm pseudocode:**

```
Step 1: Raw data extraction (from Yahoo Finance API).
Step 2: Calculate and append Technical Indicators (SMA, EMA, RSI, MACD, OBV, BB).
Step 3: Scaling using Min-Max Scaler (range 0 to 1).
Step 4: Windowing (Sliding Window Approach, Sequence Length = 60 days).
Step 5: Train/Test Split (70/30).
Step 6: Transformer Model processes data (Encoder only, Self-attention) to extract features.
Step 7: Flatten/Reshape Output of the Transformer.
Step 8: Pass embeddings into Prediction Models (ANN, LSTM, SVR, XGBoost).
Step 9: Next Day Price Prediction (t+1).
Step 10: Compare Performance Metrics (MSE, RMSE, MAE, R^2).
```

**Architecture details:**

- **Transformer Extractor**: Relies exclusively on the encoder portion. Uses 4 Attention Heads. It does not output a prediction but rather an enriched multi-dimensional feature embedding capturing long-range dependencies. The output is flattened before being passed downstream.
- **LSTM downstream**: Uses input gate ($i_t$), output gate ($o_t$), and forget gate ($f_t$) to process the Transformer embeddings.
- **ANN downstream**: A standard fully connected feedforward multi-layer perceptron.
- **XGBoost downstream**: Sequence of decision trees.
- **SVR downstream**: Support Vector Machine adapted for regression, utilizing a Radial Basis Function (RBF) kernel.

**Innovative components**: Using the Transformer strictly as an upstream feature extractor in financial forecasting, rather than an end-to-end sequence predictor, and systematically comparing how well classical ML (SVR, XGBoost) and DL (LSTM, ANN) models perform when fed these Transformer-derived embeddings.

## 3. Data

**Dataset name(s)**: Bitcoin (BTC-USD) historical price dataset.

**Data source**: Public (Yahoo Finance API).

**Data size**: Daily frequency. Date range from August 1, 2016, to August 1, 2023 (7 years). Total of 2,557 observations.

**Data preprocessing:**

- **Cleaning**: Missing values were dropped (dropna).
- **Feature Engineering**: 9 technical indicators were computed and added to the 6 base features (Open, High, Low, Close, Adj Close, Volume). The indicators are: 20-day and 50-day Simple Moving Average (SMA), 20-day and 50-day Exponential Moving Average (EMA), 14-day Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), On-Balance Volume (OBV), and 20-day Bollinger Bands (BB, upper and lower). Total features = 15.
- **Normalization**: Min-Max Scaler applied to map all feature values to the [0, 1] interval.
- **Windowing**: A sliding window technique was used with a sequence length of 60 days to predict the value at $t+1$.

**Train/validation/test split**: 70% Training, 30% Testing (chronological split).

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against**: The four models (Transformer-LSTM, Transformer-ANN, Transformer-XGBoost, Transformer-SVR) are mutually compared against one another.

**Evaluation metrics:**

- Mean Squared Error (MSE): $MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$
- Root Mean Squared Error (RMSE): $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$
- Mean Absolute Error (MAE): $MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$
- R-squared ($R^2$): $R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$

**Hyperparameters:**

- **Transformer-LSTM**: Optimizer = Adam, Loss Function = MSE, Batch Size = 32, Epochs = 50.
- **Transformer-ANN**: Optimizer = Adam, Loss Function = MSE, Batch Size = 32, Epochs = 50.
- **Transformer-XGBoost**: Objective = reg:squarederror, Number of Estimators = 100, Learning Rate = 0.1, Max Depth = 3.
- **Transformer-SVR**: Kernel = RBF, C (Regularization Parameter) = 1.0, Epsilon = 0.1.
- **Transformer parameters**: Sequence Length = 60, Features = 15, Number of Attention Heads = 4.

**Hardware/software environment**: Python 3.10.12, TensorFlow 2.15.0, Keras 2.15.0, Scikit-learn 1.2.2, XGBoost 2.0.3, Pandas, Numpy.

**Number of runs / seeds**: [not found]

**Statistical tests performed**: [not found]

## 5. Results

**Quantitative results:**

- Transformer-LSTM: MSE = 0.000305, RMSE = 0.017462, MAE = 0.012586, $R^2$ = 0.985923
- Transformer-ANN: MSE = 0.000674, RMSE = 0.025958, MAE = 0.018698, $R^2$ = 0.968856
- Transformer-XGBoost: MSE = 0.000725, RMSE = 0.026917, MAE = 0.018804, $R^2$ = 0.966508
- Transformer-SVR: MSE = 0.003310, RMSE = 0.057530, MAE = 0.046890, $R^2$ = 0.847113

**Qualitative results:**

- Visual tracking plots of True vs. Predicted prices on the test set demonstrate that Transformer-LSTM accurately captures sharp volatility spikes, including the 2021 bull run and the subsequent 2022 market crash, with minimal lag.
- Transformer-SVR consistently exhibited the highest deviations, underestimating peaks and failing to quickly adapt to sharp downward trend reversals.
- Transformer-XGBoost and Transformer-ANN tracked prices closely but showed slightly higher variance at extreme volatility points compared to the LSTM configuration.

**Ablation studies**: [not found]

**Comparison with baselines:**

- The Transformer-LSTM hybrid decisively outperformed all other models, achieving the highest $R^2$ score (0.9859) and the lowest error across all three error metrics (MSE, RMSE, MAE).
- The Transformer-SVR model was conclusively the worst performer, with an $R^2$ of 0.8471, confirming that SVR algorithms struggle significantly when mapping the highly complex, flattened embeddings generated by Transformers compared to sequential networks like LSTM.

## 6. Direct quotations

- "This study is the first to explore the effectiveness of transformer-based architectures, particularly focusing on feature extraction, in complex financial market predictions." (p. 1)
- "Using a sliding window strategy with a predetermined window size... Each window acts as an input sequence for the model. For instance, in our experimental setup, a window size of 60 days corresponds to approximately two months of trading data..." (p. 6)
- "The results of our analysis reveal that the Transformer-LSTM model outperforms the other models, demonstrating superior predictive accuracy." (p. 1)
- "The results underscore the Transformer-LSTM model's capability to generalize well to unseen data, capturing the intricate dynamics and volatility of the cryptocurrency market." (p. 10)
- "The Transformer-SVR model showed the least favorable performance, highlighting the challenges of using Support Vector Regression in capturing the complexities of Bitcoin price dynamics." (p. 11)

## 7. Authors' conclusions

The authors conclude that combining a Transformer-based feature extraction mechanism with deep learning predictive models is a highly robust approach for forecasting the highly volatile Bitcoin market. The study demonstrates that the hybrid Transformer-LSTM architecture is definitively the superior model, achieving an exceptionally high predictive accuracy ($R^2 = 0.9859$) and the lowest error rates among the evaluated methods. This indicates that the Transformer's multi-head attention mechanism effectively decodes complex, long-range dependencies from raw price data and technical indicators, while the downstream LSTM efficiently models the sequential nature of those embeddings. While Transformer-ANN and Transformer-XGBoost proved competent, Transformer-SVR struggled significantly, validating that deep sequential networks handle non-linear financial embeddings much better than traditional regression techniques.

## 8. Limitations stated by authors

- The models rely exclusively on historical price data and derived technical indicators. This approach inherently fails to capture market movements driven by sudden exogenous events, such as regulatory crackdowns, macroeconomic shifts, or geopolitical news.
- Qualitative factors, such as social media sentiment and investor psychology, which heavily influence cryptocurrency markets, were not included in the feature set.

## 9. Future work suggested by authors

- Incorporating sentiment analysis (from news articles, Twitter, or Reddit), macroeconomic indicators, and regulatory news into the predictive models to capture external market influences.
- Exploring predictions over longer-term horizons to assess the robustness of the models beyond next-day (t+1) forecasting.
- Investigating ensemble methods that combine the strengths of various architectures (e.g., merging LSTM and XGBoost outputs) for improved forecasting reliability.
- Testing the proposed hybrid architectures on other major cryptocurrencies and traditional financial assets.

## Additional critical details

- The paper explicitly notes that while XGBoost is typically a dominant algorithm for tabular regression, its performance here (while strong at $R^2 = 0.9665$) was capped by the fact that the input was a flattened sequence of Transformer embeddings, an environment where recurrent neural networks (like LSTM) naturally have a structural advantage.

---
**Extraction date**: 2026-05-19
**PDF source**: Cryptocurrency_Forecasting_Using_DL_Models.pdf

---

# Article record: Conservative Predictions on Noisy Financial Data

## Metadata
- **BibTeX key**: nabar_shroff_2023
- **Authors**: Omkar Nabar, Gautam Shroff
- **Year**: 2023
- **Title**: Conservative Predictions on Noisy Financial Data
- **Journal/Conference**: 4th ACM International Conference on AI in Finance (ICAIF '23)
- **DOI/URL**: https://doi.org/10.1145/3604237.3626859
- **Keywords**: Cascading models, Differentiable Decision Trees, Noisy Financial Data, Conservative Predictions, Algorithmic Trading

## 1. Theoretical foundation

**Key theorems**: [not found]

**Key mathematical definitions:**

- **Gini Impurity**: Used to calculate the imbalance in the probability distribution of classes to measure prediction confidence. Lower Gini impurity corresponds to greater imbalance and thus higher model confidence. $GiniImpurity = 1 - \sum_{c \in classes} p_c^2$.
- **Path Probability ($P_i$)**: In Differentiable Decision Trees (DDT), the probability of reaching an inner node. It is multiplied by the branching probability $p_i$ for the right branch and $1-p_i$ for the left branch.

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: Cascading Models using Data Pruning (applied to Differentiable Decision Trees and Multi-Layer Perceptrons).

**Core mathematical formulation:**

- Sigmoid branch probability with Temperature Scaling ($T$): $p_i = \sigma(T(x w_i + b_i))$
- Leaf probability distribution: $p = \sum_{l \in leaves} P_l Q_l$ (where $P_l$ is the path probability and $Q_l$ is the learned distribution of the leaf node).
- Class probability (Softmax): $P_c = \frac{e^{p_c}}{\sum_{c' \in classes} e^{p_{c'}}}$
- Inner node split balance ($\alpha$): $\alpha = \frac{\sum_{x} P_i(x) p_i(x)}{\sum_{x} P_i(x)}$
- Frosst and Hinton Regularization Penalty: $P = -\lambda \sum_{i \in InnerNodes} 0.5 \log(\alpha_i) + 0.5 \log(1 - \alpha_i)$, where $\lambda$ varies as $2^{-d}$ ($d$ is the depth of the node).

**Algorithm pseudocode:**

```
Algorithm 1: Cascading models using data pruning (training)
Ensure: MaxImpurity in (0, 1) and Levels >= 0

1. Initialize Unpruned empty set to contain predictions and un-pruned data-points.
2. Loop level from 1 to Levels:
     Initialize empty set D'
     Train a fresh model on the data-set D
     For d in D:
       p = model.forward(d)
       If GiniIndex(p) > MaxImpurity:
         Append d to D'
       Else: Append (p, d) to Unpruned
     D = D'

Algorithm 2: Inference using Cascading Models
Ensure: MaxImpurity in (0, 1) and Levels >= 0

1. Initialize Unpruned empty set.
2. n = Models.length
3. For d in D:
     For i = 1 to n:
       model = Models[i]
       p = model.forward(d)
       If GiniIndex(p) <= MaxImpurity:
         Append (p, d) to Unpruned
         break
```

**Architecture details:**

- **Differentiable Decision Trees (DDT)**: Initialized as a balanced binary tree of a fixed depth. Inner nodes utilize a scaled sigmoid activation to make "fuzzy" splits. Leaf nodes contain a learned probability distribution.
- **Multi-Layer Perceptrons (MLP)**: Standard feed-forward networks used as the base classifier.
- **Cascading Framework**: The architecture sequentially links 3 models (Levels = 3). Each subsequent model is trained exclusively on the data points where the preceding models exhibited a Gini Impurity greater than the threshold (0.5).

**Innovative components**: The adaptation of noise-combatting rule-learning concepts (like CN2) to modern differentiable models (DDTs and MLPs) by enforcing abstention. The model autonomously prunes data it is uncertain about during both training and inference, yielding actionable, high-confidence predictions strictly at the extreme ends of the distribution.

## 3. Data

**Dataset name(s)**: Indian Equity Market Data, Synthetic Sine-Wave Data.

**Data source**: Real equity price data from the Indian equity market; synthetic data generated algorithmically.

**Data size:**

- Market Data: 4,522 points (Experiment 1), 55,450 points (Experiment 2).
- Synthetic Data: 1,571 points (Experiment 1 at noise level [0.075, 0.05]), 7,729 points (Experiment 2 at noise level [0.075, 0.05]). Sampled at 375 one-minute ticks per synthetic 'day', yielding 75 five-minute candles.

**Data preprocessing:**

- **Market**: 5-minute OHLCV candles. Daily normalized by the close price of the first candle (base = 1). Volume normalized by historical average 5-minute volume computed from a prior year. Augmented with an 'era' attribute representing the day.
- **Feature Engineering**: Standard technical indicators (moving averages, RSI, MACD, Bollinger-bands). Logical/temporal features: Open-Close diff, High-Low diff, 10/20 window MA diffs, Slopes of Close (3/5/10 windows). "Change-length" feature counting consecutive monotonic increasing/decreasing candles.
- **Target**: 10-candle return (difference between normalized close at $t$ and $t+50$).
- **Discretization**: All features and targets discretized into 5 equally populated bins. Thresholds computed from a strictly out-of-sample prior year (market) or independent sample (synthetic) to prevent look-ahead bias.
- **Synthetic Formulation**: $f(t) = 1 + \epsilon(t) \sum_{k=0}^{K-1} (1 + \epsilon_{p[k]}) \sin(2\pi \dots)$. Base noise $\epsilon(t) \sim \mathcal{N}(0, \sigma)$, peak noise $\epsilon_{p[k]} \sim \mathcal{N}(0, \sigma_p)$.

**Train/validation/test split:**

- Experiment 1: 80% train, 20% test (sampled from the same set of eras).
- Experiment 2: Train set contains all points from a set of eras; Test set contains all points from a completely disjoint set of eras.
- Experiment 3: 80% train, 20% test strictly from a single era.
- Experiment 4: Train on 100% of a single era, test on 100% of a different single era.

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against**: Standard (Base) Differentiable Decision Trees (DDT) and Standard (Base) Multi-Layer Perceptrons (MLP) without cascading/pruning.

**Evaluation metrics:**

- **Accuracy**: Computed strictly on the 'un-pruned' points (predictions passing the Gini threshold).
- **Support**: The fraction of data on which the model makes a confident prediction.
- **Average Utility**: $\frac{\sum(Gain - Loss)}{\# \text{ of predictions of class 0 or 4}}$. Scoring logic: Predict 0 & Truth 0 = +2. Predict 0 & Truth 1 = +1. Predict 0 & Truth 3/4 = -1/-2. (Reverse logic for Predict 4). Predict on Truth 2 = 0.
- **Downside-risk adjusted return (DRAR)**: $\frac{\sum(Gain - Loss)}{\sum Loss}$.
- **Traded Sharpe ratio**: Ratio of utility to the standard deviation of returns, measured exclusively across trades actually recommended (classes 0 or 4).

**Hyperparameters:**

- Max admissible impurity: 0.5
- Cascade Levels: 3
- DDT: depth = 6, epochs = 500. (Alternative test: depth = 4, epochs = 1000).
- MLP: hidden dimensions = (128, 64, 32), epochs = 1000. (Alternative test: dimensions = (256, 128, 64, 32), epochs = 1000).

**Hardware/software environment**: [not found]

**Number of runs / seeds**: K-fold cross-validation utilized to determine optimal hyperparameters.

**Statistical tests performed**: [not found]

## 5. Results

**Quantitative results:**

- **Exp 1 Synthetic Data (DDT, Noise [0.075, 0.05])**: Base Train Acc 0.546, Base Test Acc 0.457. Cascaded Train Acc 0.81, Cascaded Test Acc 0.738. Cascaded Support Train 0.237, Support Test 0.23.
- **Exp 2 Synthetic Data (DDT, Noise [0.075, 0.05])**: Base Train Acc 0.528, Base Test Acc 0.446. Cascaded Train Acc 0.824, Cascaded Test Acc 0.697. Cascaded Support Train 0.245, Support Test 0.158.
- **Exp 1 Market Data Utility (Table 7):**
  - DDT Base: Avg Utility 0.50, Gain-Loss 1590-683, DRAR 1.33, Traded Sharpe 0.38
  - DDT Cascaded: Avg Utility 1.29, Gain-Loss 175-25, DRAR 6.00, Traded Sharpe 1.07
  - MLP Base: Avg Utility 0.79, Gain-Loss 1227-327, DRAR 2.75, Traded Sharpe 0.63
  - MLP Cascaded: Avg Utility 1.46, Gain-Loss 107-9, DRAR 10.89, Traded Sharpe 1.46
- **Exp 2 Market Data Utility (Table 7):**
  - DDT Base: Avg Utility 0.33, DRAR 0.81, Traded Sharpe 0.26
  - DDT Cascaded: Avg Utility 0.72, DRAR 2.14, Traded Sharpe 0.54
  - MLP Base: Avg Utility 0.47, DRAR 1.28, Traded Sharpe 0.37
  - MLP Cascaded: Avg Utility 0.50, DRAR 1.18, Traded Sharpe 0.36

**Qualitative results:**

- Confusion matrices (e.g., Figures 1-4) vividly demonstrate that cascading models concentrate their un-pruned predictions at the extreme bins (0 and 4) of the target distribution. This effectively filters out the "placid" middle states of the market, yielding highly actionable long/short trade signals.

**Ablation studies:**

- **Levels of Cascading**: Increasing the cascading levels to 3 improved the support by 50% or more compared to using only a single level.
- **Cross-Noise Application**: Training on clean synthetic data and testing on noisy data resulted in failure. Models overfit the clean data, emitting low impurity everywhere, thus breaking the cascading/pruning mechanism entirely.

**Comparison with baselines:**

- Cascaded models consistently achieved higher test accuracy and substantially superior risk-adjusted returns (DRAR and Traded Sharpe) compared to base models, directly trading off coverage (Support) for precision.
- Cascaded DDTs generally outperformed cascaded MLPs in DRAR and Traded Sharpe ratio, especially in the out-of-distribution (Experiment 2) tests.

## 6. Direct quotations

- "We apply a similar approach, where a model abstains from making a prediction on data points that it is uncertain on. During training, a cascade of such models are learned in sequence... with each model being trained only on data on which the previous model(s) were uncertain." (p. 1)
- "Note that in a financial market scenario, it is preferable to have a model with 70% or even 60% accuracy on say 20% or even 10% of the data, on which it makes confident predictions, and abstains on the balance, as this serves to minimise risk in any decisions taken based on the model's predictions." (p. 2)
- "The lower the gini impurity, greater the imbalance in the probability distribution of the classes. We set a value to be the maximum admissible impurity in the predictions, and the model is confident on the ones with lower impurity." (p. 2)
- "Lastly, the test-time confusion matrices revealed that the predictions made by the cascade of models i.e., predictions on the un-pruned data, lie primarily at the extremes of the target distribution." (p. 4)
- "As it happens, confident and accurate predictions at the extremes are exactly what is desired for making trading decisions, i.e., buy (long) for a prediction of 4 and sell (short) for 0." (p. 5)

## 7. Authors' conclusions

The authors conclude that applying traditional, "always-predict" deep learning models to highly noisy financial data introduces unacceptable risk. By adapting a rule-based philosophy into a differentiable cascade framework, where models evaluate their Gini impurity and abstain from predicting on uncertain samples, they effectively increase accuracy and isolate actionable signals. The framework specifically learns to predict only the extreme, actionable market movements (classes 0 and 4). While coverage (support) drops, the utility per trade, downside-risk-adjusted returns (DRAR), and Traded Sharpe ratios drastically improve compared to standard baselines. Differentiable Decision Trees (DDT) operating within this cascaded framework proved particularly adept at yielding the highest risk-adjusted trading metrics.

## 8. Limitations stated by authors

- All evaluated models (base and cascaded) experienced notable performance degradation when tested on temporal distributions (eras) different from their training data.
- The cascaded models often yield very low support (coverage) when operating on real market data across different eras, meaning they abstain from trading decisions on the vast majority of time steps.
- The mechanism fails if trained on clean data and applied to noisy data, as the model overfits the clean domain and fails to appropriately assign high impurity to noisy test samples.

## 9. Future work suggested by authors

- Integrating meta-learning or continual learning techniques with the proposed data-pruning cascade to improve generalization across shifting temporal distributions (eras).
- Investigating train-time pruning methodologies (as opposed to post-training sequence cascading) to further refine calibration error and maximize the fraction of high-confidence predictions.

## Additional critical details

- **Discretization Strategy**: The target variable is an overlapping 10-candle return window computed at 5-minute intervals. The 5-bin discretization explicitly establishes a classification framework where bin 0 represents a severe price drop (ideal short opportunity) and bin 4 represents a severe price jump (ideal long opportunity).
- **Utility Metric Design**: The utility metric explicitly penalizes "wrong extreme" predictions more heavily (e.g., -1 or -2) than it rewards correctly catching milder moves, perfectly simulating the severe financial penalty of taking a leveraged trade in the wrong direction.

---
**Extraction date**: 2026-05-19
**PDF source**: Conservative_Predictions_on_Noisy_Financial_Data.pdf

---

# Article record: Analysis of Macro Factors Influencing Cryptocurrency Price Fluctuations

## Metadata
- **BibTeX key**: song_macro_2025
- **Authors**: Yangxi Song
- **Year**: 2025
- **Title**: Analysis of Macro Factors Influencing Cryptocurrency Price Fluctuations
- **Journal/Conference**: Proceedings of ICFTBA 2025 Symposium: Data-Driven Decision Making in Business and Economics
- **DOI/URL**: 10.54254/2754-1169/2025.BL28702
- **Keywords**: cryptocurrency, macroeconomic indicators, Pearson correlation coefficient, ARIMA model prediction

## 1. Theoretical foundation

**Key theorems**: [not found]

**Key mathematical definitions:**

- **Pearson Correlation Coefficient**: Evaluates the linear relationship between two continuous variables. The correlation coefficient $r$ ranges from -1 to 1. The formulation is: $r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}$
- **Autoregressive Integrated Moving Average (ARIMA) Model**: A time-series prediction method structured around an autoregressive term ($p$), a differencing degree ($d$), and a moving average term ($q$). It estimates future data points by analyzing historical data patterns and errors.

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: Pearson Correlation Analysis integrated with an Autoregressive Integrated Moving Average (ARIMA) forecasting model.

**Core mathematical formulation:**

- **Pearson Correlation ($r$)**: $r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}$

**Algorithm pseudocode**: [not found]

**Architecture details:**

- The ARIMA model is characterized by three core parameters: $(p, d, q)$.
- First-order differencing ($d=1$) was applied to achieve sequence stationarity prior to modeling.
- Three specific parameter combinations were tested: ARIMA(2,1,0), ARIMA(1,1,1), and ARIMA(0,1,2).

**Innovative components**: The study explores the sequential extraction of macroeconomic factors (M2, CPI, PPI, DXY) using Pearson correlation analysis as a filtering step before applying a traditional linear time-series model (ARIMA) to predict highly volatile cryptocurrency prices.

## 3. Data

**Dataset name(s)**: Bitcoin Closing Prices, M2 Money Supply, Consumer Price Index (CPI), Producer Price Index (PPI), US Dollar Index (DXY).

**Data source**: Publicly sourced. Cryptocurrency prices were retrieved from CoinGecko. Macroeconomic data were retrieved from the Federal Reserve Bank of St. Louis (FRED).

**Data size**: The data spans a 7-year period from March 2017 to March 2024. The frequency is monthly, resulting in exactly 85 data points.

**Data preprocessing:**

- Target variables and predictive features were chronologically aligned by month.
- Time-series data underwent stationarity checks. Because the raw data was non-stationary, first-order differencing was applied before ARIMA modeling.

**Train/validation/test split**: The text references splitting the data into a "training set" and a "test set" for evaluation, but exact partition sizes or ratios were not explicitly stated.

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against**: No external algorithms were used as baselines. Internal model comparisons were conducted between three distinct ARIMA configurations: ARIMA(2,1,0), ARIMA(1,1,1), and ARIMA(0,1,2).

**Evaluation metrics:**

- R-squared ($R^2$)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- **Akaike Information Criterion (AIC)**: Used for model selection; lower values indicate better models.
- **Bayesian Information Criterion (BIC)**: Used for model selection; imposes a heavier penalty for model complexity.
- **Ljung-Box $Q$-statistic**: Evaluates whether model residuals resemble white noise (lack of autocorrelation).

**Hyperparameters:**

- Differencing degree ($d$): 1
- Evaluated AR ($p$) and MA ($q$) ranges: $p \in \{0, 1, 2\}$, $q \in \{0, 1, 2\}$.

**Hardware/software environment**: [not found]

**Number of runs / seeds**: [not found]

**Statistical tests performed:**

- Pearson correlation significance test (p-values derived for $r$).
- Ljung-Box test for residual autocorrelation.

## 5. Results

**Quantitative results:**

- **Pearson Correlation with Bitcoin Price:**
  - M2 Money Supply: $r = 0.812$ ($p < 0.01$)
  - Consumer Price Index (CPI): $r = 0.771$ ($p < 0.01$)
  - Producer Price Index (PPI): $r = 0.589$ ($p < 0.01$)
  - US Dollar Index (DXY): $r = 0.222$ ($p = 0.04$)
- **ARIMA Model Evaluation Metrics (Table 2):**
  - ARIMA(2,1,0): $R^2$: 0.941, MAE: 3848.455, RMSE: 6223.771, AIC: 1740.098, BIC: 1747.362, Ljung-Box $Q(18)$: 18.232 ($p=0.440$).
  - ARIMA(1,1,1): $R^2$: 0.941, MAE: 3843.518, RMSE: 6204.606, AIC: 1739.584, BIC: 1746.848, Ljung-Box $Q(18)$: 17.513 ($p=0.488$).
  - ARIMA(0,1,2): $R^2$: 0.941, MAE: 3844.757, RMSE: 6207.240, AIC: 1739.654, BIC: 1746.918, Ljung-Box $Q(18)$: 17.842 ($p=0.466$).
- **ARIMA(1,1,1) Parameter Estimates (Table 3):**
  - Constant: 494.394 ($t=0.551$, $p=0.582$)
  - AR(1) coefficient: -0.428 ($t=-1.026$, $p=0.305$)
  - MA(1) coefficient: 0.560 ($t=1.458$, $p=0.145$)

**Qualitative results:**

- Ljung-Box tests for all three ARIMA models yielded p-values greater than 0.05, demonstrating that the residual errors approximate white noise and confirming that the models adequately captured the linear information in the data.
- Time-series plot visualizations (Figures 1-4) confirmed similar overarching historical trends between Bitcoin and the selected macro variables, particularly the M2 supply.
- When forecasting out-of-sample, the ARIMA model's prediction curve showed a profound divergence from the actual testing values, capturing almost none of the sharp price spikes characteristic of Bitcoin.

**Ablation studies**: [not found]

**Comparison with baselines:**

- ARIMA(1,1,1) was selected as the optimal model structure strictly because it achieved the minimum AIC (1739.584) and BIC (1746.848) among the candidates.
- Despite being the "best" linear fit, the model suffered from an exceptionally high RMSE (6204.606), demonstrating a near-total failure to accurately forecast extreme cryptocurrency fluctuations.

## 6. Direct quotations

- "The prediction performance of the ARIMA model is poor, and the research method still needs improvement. The purpose of this study is to quantify macro monetary policy, reflect the impact of cryptocurrency price fluctuations, and construct a more stable asset framework." (p. 1)
- "The increase in the M2 money supply provides more liquidity to the market, and investors hold sufficient funds... the total amount of Bitcoin is capped at 21 million, determining its scarcity. A large amount of fiat currency will result in investors chasing scarce assets with more currency, leading to the rise of cryptocurrency." (p. 4)
- "Through comparison, the AIC and BIC values of the ARIMA (1, 1, 1) model are relatively minimal, while the variation of the $R^2$ values are minor. Consequently, ARIMA (1, 1, 1) was selected as the final evaluation model." (p. 5)
- "Based on actual values, the model error is quite significant, and the fluctuation curve does not align well, indicating poor predictive performance of the model." (p. 6)
- "The ARIMA model is suitable for linear analysis. However, given the high volatility of Bitcoin, analyzing it strictly within a linear framework is inappropriate, resulting in significant predictive deviations in the ARIMA model." (p. 6)

## 7. Authors' conclusions

The author concludes that macroeconomic policies and indicators play a critical role in shaping cryptocurrency prices. Specifically, the M2 money supply exerts a highly significant positive correlation with Bitcoin prices, primarily due to fiat currency dilution increasing the demand for scarce, deflationary assets. The CPI and PPI also display strong positive correlations, whereas the US dollar index exhibits a comparatively weak correlation. However, despite identifying these strong statistical relationships, the study concludes that traditional linear time-series models, such as ARIMA, are fundamentally ill-equipped to forecast cryptocurrency markets. The extreme non-linear volatility of Bitcoin caused the ARIMA model to suffer massive predictive deviations and high error rates. Ultimately, researchers and investors must abandon strictly linear frameworks in favor of advanced, non-linear algorithms when building asset allocation and risk management models for cryptocurrencies.

## 8. Limitations stated by authors

- The utilization of monthly data over a restricted 7-year timeframe resulted in a small sample size (only 85 observations), which significantly reduced the statistical rigor and significance of the model's predictions.
- The methodologies employed (Pearson correlation and ARIMA) are strictly linear analytical techniques. They inherently fail to capture the complex, non-linear, and highly volatile behavior that characterizes cryptocurrency assets.
- Cryptocurrency prices are subjected to diverse external shocks beyond the four macroeconomic indicators analyzed, including geopolitical conflicts, regulatory changes, and institutional events (e.g., exchange bankruptcies), none of which were captured by the model.

## 9. Future work suggested by authors

- Expanding the predictive framework by incorporating a broader array of variables and adopting higher-frequency data (such as daily or minute-level intervals) to substantially increase the sample size and characterization accuracy.
- Directly integrating specific market events, black swan occurrences, and evolving regulatory policy shifts into future predictive models.
- Utilizing advanced, non-linear deep learning architectures, such as Artificial Neural Networks (ANN) or Long Short-Term Memory (LSTM) networks, to successfully model the erratic volatility of the cryptocurrency market.

## Additional critical details

- **ARIMA Sub-Parameter Insignificance**: When analyzing the optimal ARIMA(1,1,1) model, none of the specific coefficient estimates (Constant, AR(1), or MA(1)) were statistically significant at a standard $\alpha = 0.05$ level (all p-values were $>0.10$). This further reinforces the author's conclusion that the linear parameters failed to capture meaningful predictive signals for Bitcoin.

---
**Extraction date**: 2026-05-19
**PDF source**: 10.54254/2754-1169/2025.BL28702