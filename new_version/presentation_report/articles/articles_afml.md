# Article record: HARd to beat: The overlooked impact of rolling windows in the era of machine learning

## Metadata
- **BibTeX key**: chassot_audrino_2026
- **Authors**: Jonathan Chassot and Francesco Audrino
- **Year**: 2026
- **Title**: HARd to beat: The overlooked impact of rolling windows in the era of machine learning
- **Journal/Publication**: International Journal of Forecasting, 42, 330-343
- **DOI/URL**: [not provided]
- **Keywords from article**: Forecasting practice, HAR, Machine learning, Realized volatility, Volatility forecasting

## 1. Study objective(s)
To investigate the predictive abilities of the heterogeneous autoregressive (HAR) model compared to advanced machine learning (ML) techniques for realized volatility forecasting, by specifically focusing on the often-overlooked impact of fitting schemes, namely the training window size and the re-estimation frequency.

## 2. Research question / Hypothesis
Do complex machine learning models genuinely outperform the simple linear HAR model in forecasting realized volatility, or is their perceived superiority merely a result of comparing them against a HAR baseline that utilizes a suboptimal fitting scheme (training window and re-estimation frequency)?

## 3. Methodology
- **Study design**: Empirical comparative quantitative analysis evaluating volatility forecasting performance across different models and rolling-window fitting schemes.
- **Sample / corpus**: An unprecedentedly large and comprehensive dataset consisting of 1,445 individual stocks from the US equity market.
- **Instruments / materials**: The traditional Heterogeneous Autoregressive (HAR) model and several Machine Learning algorithms (Random Forest, Gradient Boosting Regression Trees, Neural Networks, and LASSO). The primary predictors were past Realized Volatility (RV) and the VIX index.
- **Procedures**:
  1. Computed the daily realized volatility for the 1,445 stocks using high-frequency intraday data.
  2. Defined various fitting schemes by systematically altering the training window size (e.g., varying the number of days used for training) and the re-estimation frequency (how often the model parameters are updated).
  3. Trained both the baseline HAR models and the ML models across these different schemes. The ML models underwent extensive hyperparameter tuning using time-series cross-validation.
  4. Generated 1-day ahead out-of-sample volatility forecasts for all models and configurations.
  5. Evaluated and ranked the forecasts using statistical loss functions and economic utility metrics.
- **Data analysis**: QLIKE (the primary robust loss function for volatility), Mean Squared Error (MSE), and Realized Utility (assessing economic value in a mean-variance portfolio allocation framework). Statistical significance of outperformance was likely assessed using Diebold-Mariano and Model Confidence Set (MCS) tests.

## 4. Main results (only facts from the article)
- The predictive performance of the HAR model is highly sensitive to the chosen fitting scheme. Properly tuning the training window size and re-estimation frequency significantly improves its forecasting accuracy.
- Despite undergoing extensive hyperparameter tuning, the advanced machine learning models (RF, GBRT, NN, Lasso) failed to surpass the predictive accuracy of the simple linear HAR model when a refined fitting approach was utilized for the HAR benchmark.
- The HAR model consistently outperformed its ML counterparts across both statistical loss metrics (QLIKE, MSE) and economic metrics (Realized Utility) when both classes of models relied solely on realized volatility and VIX as predictors.
- The HAR model achieves this superior or equivalent predictive performance with drastically lower computational costs and much higher interpretability compared to the "black-box" ML models.

## 5. Relevant direct quotations (max 3)
- "Despite extensive hyperparameter tuning, ML models fail to surpass the linear benchmark set by HAR when utilizing a refined fitting approach for the latter." (p. 330)
- "We assess performance using QLIKE, MSE, and realized utility metrics, finding that HAR consistently outperforms its ML counterparts when both rely solely on realized volatility and VIX as predictors." (p. 330)
- "Our results underscore the importance of a correctly specified fitting scheme." (p. 330)

## 6. Authors' conclusions
The authors conclude that the perceived dominance of machine learning models in volatility forecasting is frequently an artifact of poor baseline configurations. When the simple, linear HAR model is implemented with a carefully optimized fitting scheme (appropriately sized rolling windows and re-estimation frequencies), it establishes a benchmark that is incredibly "HARd to beat." Because HAR provides superior or matched forecasting accuracy alongside high interpretability and negligible computational cost, it remains the most practical and effective model for predicting realized volatility using standard predictors, cautioning researchers against blindly favoring complex ML algorithms without properly optimizing their linear baselines.

## 7. Limitations acknowledged by the authors
The study specifically compares models relying solely on standard predictors (past realized volatility and the VIX). It leaves open the possibility that ML models might only show their true superiority when exposed to massive, high-dimensional alternative datasets (such as order book data or textual sentiment) where linear models struggle.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: Does Meta-Labeling Add to Signal Efficacy?

## Metadata
- **BibTeX key**: singh_joubert_2019
- **Authors**: A. Singh, J. Joubert
- **Year**: 2019
- **Title**: Does Meta-Labeling Add to Signal Efficacy?
- **Journal/Publication**: SSRN Electronic Journal
- **DOI/URL**: [not provided]
- **Keywords from article**: Machine Learning, Meta-Labeling, CUSUM Filter, Triple Barrier Labeling, Trend-following, Mean-Reversion

## 1. Study objective(s)
To create an open-source Python package (mlfinlab) based on Marcos Lopez de Prado's framework in Advances in Financial Machine Learning, and to empirically test whether combining event-based sampling, the triple-barrier method, and meta-labeling improves the performance of trend-following and mean-reverting trading strategies.

## 2. Research question / Hypothesis
Does the application of a secondary machine learning model (meta-labeling), combined with event-based sampling and the triple-barrier method, improve the performance and signal efficacy of primary trading strategies?

## 3. Methodology
- **Study design**: Empirical quantitative backtesting of financial trading strategies using machine learning classification metrics.
- **Sample / corpus**: High-frequency tick data (E-mini S&P 500 futures).
- **Instruments / materials**: Python, the mlfinlab package (developed by the authors), Random Forest algorithm (for the secondary meta-labeling model), Symmetric CUSUM filter, and the Triple-Barrier Method.
- **Procedures**:
  1. Developed the mlfinlab package to construct standard financial data structures (e.g., dollar bars, volume bars).
  2. Applied a Symmetric CUSUM filter to sample events based on volatility thresholds.
  3. Labeled the sampled events using the Triple-Barrier Method (setting upper profit-taking, lower stop-loss, and vertical expiration barriers).
  4. Developed primary trading models (e.g., Trend Following and Mean-Reverting Bollinger Band strategies) to generate initial trading signals.
  5. Applied meta-labeling by training a secondary model (Random Forest) on the primary model's signals to predict whether the primary signal would be successful (True) or unsuccessful (False).
  6. Combined the predictions: a trade is only executed if the primary model generates a signal and the secondary model predicts the signal is highly probable to be correct.
- **Data analysis**: Evaluation of out-of-sample data using classification performance metrics, specifically accuracy, precision, recall, F1-score, and confusion matrices.

## 4. Main results (only facts from the article)
- The application of meta-labeling successfully identified false positives generated by the primary model and correctly reclassified them as true negatives.
- The base primary model achieved an accuracy of 0.8933 on the out-of-sample data, with 192 false positives.
- By applying meta-labeling, the overall performance metrics (precision, recall, and F1-score) experienced a boost because the secondary model filtered out unprofitable trades before execution.

## 5. Relevant direct quotations (max 3)
- "Our results confirm the fact that a combination of event-based sampling, triple-barrier method and meta-labeling improves the performance of the strategies." (p. 1)
- "We can see that in the confusion matrix, that the false positives from the primary model, are now being correctly identified as true negatives with the help of meta-labeling. This leads to a boost in performance metrics. Meta-labeling works as advertised!" (p. 37)
- "In this project we create a open-source python package (mlfinlab) that is based on the work of Dr. Marcos Lopez de Prado in his book Advances in Financial Machine Learning." (p. 1)

## 6. Authors' conclusions
The authors conclude that implementing the advanced financial machine learning techniques proposed by Lopez de Prado genuinely enhances the efficacy of trading signals. Specifically, meta-labeling acts as a highly effective secondary filter that reduces the number of false positives generated by primary trading strategies (like trend-following or mean-reversion). This reduction in false positives directly leads to improved performance metrics on out-of-sample data. Furthermore, the development of their mlfinlab package provides a robust, open-source foundation for researchers to continue building quantitative investment strategies using these methodologies.

## 7. Limitations acknowledged by the authors
[not provided]

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data: Evidence from Korean Markets

## Metadata
- **BibTeX key**: kang_kim_2025
- **Authors**: Sungwoo Kang and Jong-Kook Kim
- **Year**: 2025
- **Title**: Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data: Evidence from Korean Markets
- **Journal/Publication**: arXiv preprint arXiv:2504.02249v2 [cs.CE]
- **DOI/URL**: https://arxiv.org/abs/2504.02249
- **Keywords from article**: [not provided]

## 1. Study objective(s)
To demonstrate that deep learning models trained exclusively on raw OHLCV (Open-High-Low-Close-Volume) data can achieve predictive performance comparable to traditional machine learning models that rely on extensively engineered technical indicators.

## 2. Research question / Hypothesis
Can a simple Long Short-Term Memory (LSTM) network utilizing only raw OHLCV data match the stock price prediction performance of sophisticated traditional machine learning models (like XGBoost) that incorporate complex technical indicators?

## 3. Methodology
- **Study design**: Empirical quantitative comparative analysis of predictive financial models.
- **Sample / corpus**: A dataset of Korean market stocks spanning from 2006 to 2024.
- **Instruments / materials**: Long Short-Term Memory (LSTM) network, traditional machine learning models (specifically XGBoost), and the Triple Barrier Labeling (TBL) method.
- **Procedures**:
  1. Collected historical OHLCV data for Korean stocks from 2006 to 2024.
  2. Applied the Triple Barrier Labeling (TBL) method to create discrete target labels.
  3. Optimized the TBL parameters to achieve a balanced distribution of labels across the dataset, specifically testing different window sizes and barrier percentages.
  4. Trained a simple LSTM network using strictly raw OHLCV data (without any technical indicators).
  5. Trained baseline traditional ML models (XGBoost) using a feature set that included technical indicators.
  6. Evaluated and compared the predictive performance of the LSTM against the baseline models.
  7. Investigated the relationship between the optimal input window size and the LSTM model's hidden size configuration.
- **Data analysis**: Performance benchmarking between deep learning and traditional ML models, alongside hyperparameter optimization for both the labeling mechanism (TBL) and the neural network architecture.

## 4. Main results (only facts from the article)
- LSTM networks trained strictly on raw OHLCV data without any technical indicators achieved predictive performance similar to traditional machine learning models like XGBoost that utilized technical indicators.
- The optimization of the Triple Barrier Labeling parameters for the Korean stock dataset resulted in balanced label proportions when using a 29-day window and 9% barriers.
- The study identified that the optimal input window size for the LSTM model varies depending on the model's hidden size.
- The specific network configuration that yielded the best results was an input window size of 100 combined with a hidden size of 8.

## 5. Relevant direct quotations (max 3)
- "This paper demonstrates that deep learning models trained on raw OHLCV (open-high-low-close-volume) data can achieve comparable performance to traditional machine learning (ML) models using technical indicators for stock price prediction in Korean markets." (p. 1)
- "...we optimize the triple barrier labeling parameters to achieve balanced label proportions with a 29-day window and 9% barriers." (p. 1)
- "Furthermore, we identify that the optimal window size varies with model hidden size, with a configuration of window size 100 and hidden size 8 yielding the bes[t]..." (p. 1)

## 6. Authors' conclusions
The authors conclude that extensive feature engineering and the calculation of complex technical indicators may be unnecessary when forecasting stock prices with deep learning architectures. By properly optimizing the Triple Barrier Labeling method to ensure balanced dataset labels, a simple LSTM network fed only with raw OHLCV data is highly capable of extracting the necessary market patterns. This approach successfully matches the performance of sophisticated, feature-heavy traditional machine learning models like XGBoost, highlighting the capacity of LSTMs to autonomously learn predictive representations directly from raw financial time-series data.

## 7. Limitations acknowledged by the authors
[not provided]

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: Supervised Autoencoders with Fractionally Differentiated Features and Triple Barrier Labelling Enhance Predictions on Noisy Data

## Metadata
- **BibTeX key**: slepaczuk_bieganowski_2024
- **Authors**: Robert Ślepaczuk and Bartosz Bieganowski
- **Year**: 2024
- **Title**: Supervised Autoencoders with Fractionally Differentiated Features and Triple Barrier Labelling Enhance Predictions on Noisy Data
- **Journal/Publication**: Proceedings of ICAIF-24, November 14-16, 2024, New York City, NY / arXiv preprint
- **DOI/URL**: https://doi.org/10.48550/arXiv.2411.12753
- **Keywords from article**: machine learning, algorithmic investment strategy, supervised autoencoders, financial time series, trading strategy, risk-adjusted return

## 1. Study objective(s)
To investigate how financial time series forecasting can be enhanced using neural networks, specifically supervised autoencoders (SAE), to improve algorithmic investment strategy performance. It specifically examines the impact of data noise augmentation and triple barrier labeling on risk-adjusted returns (such as Sharpe and Information Ratios).

## 2. Research question / Hypothesis
- RQ1: Does data augmentation and denoising via autoencoders combined with triple barrier labeling (TBL) produce better risk-reward metrics than buy-and-hold?
- RQ2: Does a portfolio of SAE-MLP strategies outperform a similar portfolio of buy-and-hold cryptocurrencies in terms of risk-reward metrics?

## 3. Methodology
- **Study design**: Empirical quantitative backtesting of algorithmic trading strategies using walk-forward validation.
- **Sample / corpus**: High-frequency 1-minute OHLC price data for Bitcoin, Ethereum, and Litecoin from Binance. The data spans from January 1, 2016, to April 30, 2022, with the out-of-sample period running from January 1, 2020, to April 30, 2022. The dataset also incorporates six external macroeconomic and commodity features (Initial Claims, WTI Crude Oil, Henry Hub Natural Gas, Corn, Gold, Copper, and Aluminum).
- **Instruments / materials**: Supervised Autoencoder, Multi-Layer Perceptron (SAE-MLP) network. Triple Barrier Labeling (TBL). Fixed-width window fractional differentiation (FFD). Python 3.10 with the NumPy package.
- **Procedures**:
  1. Collected 1-minute historical data and applied fractional differencing to the features using a walk-forward approach to ensure stationarity while preserving series memory.
  2. Applied Triple Barrier Labeling (TBL) to classify trading positions into long (1), short (-1), or no position (0).
  3. Augmented the training data by injecting Gaussian noise scaled to the historical volatility of the features to prevent model overfitting.
  4. Trained the SAE-MLP network using walk-forward validation (overlapping expanding/shifting windows) to generate predictions.
  5. Backtested a simulated buy-sell algorithmic trading strategy based on model predictions, factoring in a 0.05% transaction cost.
  6. Evaluated the strategies for each asset individually and as a 1:1:1 equally weighted portfolio across 10-minute, 20-minute, and 30-minute intervals.
- **Data analysis**: Evaluation of strategy performance using Total Return, Annualized Return, Annualized Standard Deviation, Maximum Drawdown, Information Ratio, and a custom TBL-optimized metric.

## 4. Main results (only facts from the article)
- For Bitcoin, the 30-minute SAE strategy achieved the highest Information Ratio (2.03) and the lowest maximum drawdown (14.02%), demonstrating vastly superior risk-adjusted efficiency compared to buy-and-hold, which suffered a 53.30% drawdown.
- For Ethereum, the 20-minute SAE strategy provided the optimal balance, achieving an Information Ratio of 2.64 and a total return of 1658.93%, significantly mitigating the extreme risks associated with the buy-and-hold benchmark.
- For Litecoin, the 30-minute SAE strategy massively outperformed the buy-and-hold method in absolute terms (382.49% vs. 118.92% total return) and achieved the best risk-adjusted metrics with an Information Ratio of 2.11.
- In the diversified 1:1:1 equally weighted portfolio, all SAE strategies outperformed the buy-and-hold approach on risk-adjusted metrics. The 30-minute portfolio strategy was the most effective, producing an Information Ratio of 2.56 and limiting maximum drawdown to 23.55%.
- The combination of supervised autoencoder denoising and Triple Barrier Labeling proved highly effective, answering both research questions affirmatively regarding risk-reward outperformance.

## 5. Relevant direct quotations (max 3)
- "This paper investigates the enhancement of financial time series forecasting with the use of neural networks through supervised autoencoders (SAE), to improve investment strategy performance." (p. 1)
- "While the buy-and-hold strategy yielded the highest total return (406.02%) and annualized return (61.63%), it also exhibited the highest volatility with an annualized standard deviation of 63.31% and a maximum drawdown of 53.30%." (p. 5)
- "Our research has demonstrated that the application of Supervised Autoencoder de-noising in combination with Triple Barrier labeling significantly improves algorithmic trading strategies in comparison to buy and hold approaches." (p. 7)

## 6. Authors' conclusions
The authors conclude that integrating Supervised Autoencoders (SAE) with the Triple Barrier Labeling (TBL) method significantly enhances the performance of algorithmic trading strategies in the cryptocurrency market. While a passive buy-and-hold strategy can occasionally yield higher absolute returns during massive bull markets, the SAE-MLP models provide vastly superior risk-adjusted returns by drastically lowering portfolio volatility and protecting against severe maximum drawdowns. Consequently, they argue that institutional investors and asset managers must adopt these advanced machine-learning frameworks to remain competitive, manage systemic risks effectively, and modernize financial market practices.

## 7. Limitations acknowledged by the authors
- The results are based purely on historical data, meaning the predictive power in future, volatile markets should be treated with caution.
- The backtesting engine did not account for slippage (assuming instant execution at the last close price), which could diminish net returns in real-world or illiquid markets.
- The study unrealistically assumes that the stop-loss and take-profit barriers determined by the Triple Barrier Labeling are executed perfectly and immediately.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling Method and Machine Learning Approach for Pair Trading Strategy in Cryptocurrency Markets

## Metadata
- **BibTeX key**: fu_et_al_2024
- **Authors**: Ning Fu, Mingu Kang, Joongi Hong and Suntae Kim
- **Year**: 2024
- **Title**: Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling Method and Machine Learning Approach for Pair Trading Strategy in Cryptocurrency Markets
- **Journal/Publication**: Mathematics
- **DOI/URL**: https://doi.org/10.3390/math12050780
- **Keywords from article**: pair trading; triple barrier labeling method; cryptocurrency; genetic algorithm; AdaBoost classifier

## 1. Study objective(s)
To integrate pair trading strategies with an adapted Triple Barrier Labeling (TBL) Method to generate precise trading signals. Furthermore, the study aims to utilize a Genetic Algorithm (GA) to optimize these labels for distinct risk-reward styles, specifically High Risk and High Profit (HRHP) and Low Risk and Low Profit (LRLP), and subsequently train an AdaBoost classifier to predict these optimal trading behaviors in the cryptocurrency market.

## 2. Research question / Hypothesis
- Research Question 1: How can the labels generated by Triple Barrier Labeling for pair trading be evaluated in terms of profit and the MDD?
- Research Question 2: How to choose a classifier for the predictive model and evaluate its performance in comparison to other models?
- Research Question 3: What is the predictive performance of models trained by using label types obtained through genetic optimization, precisely the label types of HRHP or LRLP?

## 3. Methodology
- **Study design**: Empirical quantitative backtesting and predictive modeling applying genetic algorithms and machine learning classifiers to financial time series data.
- **Sample / corpus**: Daily cryptocurrency market data sourced from Yahoo Finance. The training and label generation phase utilized data from 9 November 2017 to 31 August 2022. The out-of-sample testing and model evaluation phase utilized data from 1 September 2022 to 1 December 2023.
- **Instruments / materials**: Genetic Algorithm (GA) for hyperparameter optimization; Augmented Dickey-Fuller (ADF) and Pearson correlation tests; modified Triple Barrier Labeling Method; an AdaBoost Classifier coupled with an Extreme Gradient Boosting Classifier.
- **Procedures**:
  1. Selected optimal cryptocurrency pairs by calculating Pearson correlation and conducting ADF cointegration tests across various digital assets.
  2. Calculated the spread in daily price changes between the selected pairs and standardized this spread into a dynamic Z-spread using a 55-day sliding window.
  3. Redefined the Triple Barrier Labeling Method parameters (Upper Threshold, Lower Threshold, Max Holding Period) to take the Z-spread as the input rather than single asset prices, generating buy/sell/hold labels.
  4. Applied a Genetic Algorithm using a combined fitness function of profit and Maximum Drawdown (MDD) to optimize the barrier parameters into two sets: HRHP (70% profit weight / 30% MDD) and LRLP (30% profit weight / 70% MDD).
  5. Trained multiple machine learning and deep learning classification models using the optimized labels alongside a comprehensive feature set (high, low, close, volume, Z-spread).
  6. Evaluated the selected AdaBoost Classifier by predicting signals on the testing dataset and running a simulated backtest to calculate out-of-sample Profit, MDD, and Sharpe Ratios.
- **Data analysis**: Cointegration testing (ADF), Pearson Correlation, AUC (Area Under the ROC Curve) for classifier selection, and financial metrics including simulated trading profit, Maximum Drawdown (MDD), and Sharpe Ratio for final model evaluation.

## 4. Main results (only facts from the article)
- BTC and ETH were selected as the optimal trading pair, demonstrating perfect cointegration (ADF p-value = 0.0) and the highest Pearson correlation coefficient (0.8128) among all tested cryptocurrencies.
- The AdaBoost Classifier combined with Extreme Gradient Boosting outperformed all tested deep learning models (such as CNN+LSTM) and other machine learning algorithms, achieving an accuracy of 0.7903, recall of 0.6621, and an AUC of 0.6409.
- In the out-of-sample backtest, the model trained with HRHP (High Risk and High Profit) labels achieved an exceptional profit multiple of 2.0754, drastically outperforming the AI regression baseline (1.4499) and the traditional pair trading strategy (1.3702), though it incurred a higher MDD of -0.1611.
- The model trained with LRLP (Low Risk and Low Profit) labels demonstrated superior downside risk control with an MDD of only -0.0691, while still delivering a profit of 1.7034. It achieved the highest overall risk-adjusted performance with a Sharpe Ratio of 2.0016.
- Both HRHP and LRLP models significantly outperformed the traditional cryptocurrency buy-and-hold strategies (which suffered from severe MDDs of -0.2942 for BTC and -0.3806 for ETH) and the traditional statistical pair trading strategy across profit and MDD metrics.

## 5. Relevant direct quotations (max 3)
- "In pursuit of high-performance labels to elevate the precision of classification models, this study advanced the Triple Barrier Labeling Method for enhanced compatibility with pair trading strategies."
- "These empirical findings further reinforce the conclusion that machine learning models excel over deep learning models in effectively managing data characterized by high volatility and noise, a phenomenon observed within the cryptocurrency market context..."
- "The experimental results demonstrate that the two distinct trading models trained by using our proposed method outperform traditional pair trading strategies and the buy-and-hold strategy for individual cryptocurrencies in terms of profit and MDD metrics."

## 6. Authors' conclusions
The authors conclude that adapting the Triple Barrier Labeling Method to utilize the Z-spread of paired assets, rather than single asset prices, creates a highly effective mechanism for generating pair trading signals in the volatile cryptocurrency market. By optimizing these barrier parameters using a Genetic Algorithm, traders can generate specific labels tailored to their preferred trading style (either High Risk/High Profit or Low Risk/Low Profit). Furthermore, training an AdaBoost Classifier on these optimized labels yields state-of-the-art predictive performance, significantly outperforming both traditional statistical pair trading strategies and long-term buy-and-hold strategies in terms of absolute profitability and downside risk mitigation (MDD).

## 7. Limitations acknowledged by the authors
- The inherent randomness of the Genetic Algorithm (GA) during selection, crossover, and mutation operations means that individual runs might produce slightly different optimization outcomes, posing a potential threat to research robustness.
- The study relied on daily sampled data to regulate trade frequency, which may lead to the loss of granular information and potentially impact the analysis accuracy compared to higher-frequency intraday data.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---