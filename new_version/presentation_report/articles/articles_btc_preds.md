# Article record: Review of Deep Learning Models for Crypto Price Prediction: Implementation and Evaluation

## Metadata
- **BibTeX key**: wu_crypto_dl_review_2024
- **Authors**: Jingyang Wu, Xinyi Zhang, Fangyixuan Huang, Haochen Zhou, and Rohtiash Chandra
- **Year**: 2024
- **Title**: Review of Deep Learning Models for Crypto Price Prediction: Implementation and Evaluation
- **Journal/Publication**: arXiv preprint arXiv:2405.11431
- **DOI/URL**: https://arxiv.org/abs/2405.11431
- **Keywords from article**: cryptocurrency, deep learning, time series prediciton

## 1. Study objective(s)
To comprehensively review the literature on deep learning for cryptocurrency forecasting and to empirically evaluate novel deep learning models (variants of LSTM, CNN, and Transformer) for multi-step ahead cryptocurrency price prediction. The study specifically aims to compare univariate and multivariate approaches and to investigate the impact of the COVID-19 pandemic's extreme market volatility on prediction accuracy.

## 2. Research question / Hypothesis
Which deep learning architectures (LSTM variants, CNNs, or Transformers) provide the best multi-step ahead prediction accuracy for highly volatile cryptocurrency prices? Furthermore, how do multivariate models (incorporating external features like gold and correlated cryptos) compare to univariate models, and how does the high volatility induced by the COVID-19 pandemic impact the models' predictive capabilities?

## 3. Methodology
- **Study design**: Literature review combined with a comparative experimental, quantitative evaluation of deep learning models for time-series forecasting.
- **Sample / corpus**: Daily historical data for four cryptocurrencies: Bitcoin (BTC), Ethereum (ETH), Dogecoin (DOGE), and Litecoin (LTC). External data included the daily stock price of Gold from the London Bullion Market (LBMA). Data spans from 2013/2015 to April 2024, yielding up to 3,991 daily observations per asset.
- **Instruments / materials**: Six deep learning models: LSTM, Bidirectional LSTM (BD-LSTM), Encoder-Decoder LSTM (ED-LSTM), Convolutional LSTM (Conv-LSTM), CNN, and Transformer. Baseline models included Multilayer Perceptron (MLP) and ARIMA. Optimization was performed using the Adam optimizer, and data was normalized using a Min-Max scaler.
- **Procedures**:
  1. Data extraction and pre-processing, including linear interpolation to fill missing gold prices on weekends/holidays, and min-max scaling.
  2. Data splitting into two distinct experimental sets: Dataset 1 (training on pre-COVID-19 data) and Dataset 2 (training on data including the COVID-19 period), split using a 70:30 train-test ratio.
  3. Hyperparameter selection based on literature, trial runs, and minimum RMSE on the training set (optimizing hidden units, learning rates, etc.).
  4. Pre-experiment data analysis, specifically calculating the monthly volatility of the target assets and mapping feature correlations using Pearson coefficients.
  5. Experiment 1: Evaluating the models on Dataset 1, comparing Univariate and standard Multivariate strategies.
  6. Experiment 2: Evaluating the models on Dataset 2 using the Univariate and an enhanced "Multivariate*" strategy (which adds the price of the most highly correlated alternative cryptocurrency) to test robustness against COVID-19 volatility.
- **Data analysis**: Root Mean Square Error (RMSE) used as the primary performance criterion. Models were run 30 independent times to compute the mean RMSE and 95% confidence intervals for multi-step prediction horizons (1 to 5 days ahead). Pearson correlation was utilized for feature selection.

## 4. Main results (only facts from the article)
- The Convolutional LSTM (Conv-LSTM) model utilizing the enhanced multivariate strategy consistently provided the best prediction accuracy across both pre-COVID-19 and COVID-19 experimental settings.
- Multivariate deep learning models exhibited notably better performance and stability in forecasting the four cryptocurrencies when compared to univariate models.
- The enhanced "Multivariate*" strategy, which included the crypto's open, high, low, close prices, Gold prices, and the close price of the most correlated alternative cryptocurrency (e.g., adding Ethereum data when predicting Bitcoin), significantly outperformed standard univariate and multivariate baselines.
- Prediction accuracy for all models naturally deteriorated (RMSE increased) as the multi-step prediction horizon expanded from 1 day to 5 days ahead.
- Training models on the COVID-19 dataset (Experiment 2) resulted in worse overall accuracy (e.g., BTC test mean RMSE dropped from ~0.02 to ~0.03) compared to pre-COVID-19 data, proving that high historical volatility weakens prediction precision.
- Dogecoin yielded the worst prediction accuracy (highest RMSE) among the four assets, heavily attributed to its extreme and sudden price fluctuations driven by social media rather than steady market trends.
- The Transformer model delivered surprisingly unsatisfactory results (ranking lowest or second lowest in many tests), which the researchers attributed to the lack of sufficient training data size required for attention-heavy models.

## 5. Relevant direct quotations (max 3)
- "Our results show that the convolutional LSTM with a multivariate approach provides the best prediction accuracy in two major experimental settings." (p. 1)
- "Our findings indicate that combining the closing prices of the highest-correlated cryptocurrencies into the multivariate model leads to a significant enhancement in the model's accuracy in forecasting." (p. 30)
- "Our experimental results show that utilising a training data set with high volatility weakens the precision of our predictions." (p. 30)

## 6. Authors' conclusions
The authors conclude that deep learning methodologies, particularly the Convolutional LSTM architecture, are highly effective tools for cryptocurrency time-series forecasting, significantly outperforming traditional statistical models like ARIMA and standard MLPs. They assert that relying purely on univariate historical data is insufficient for this asset class; instead, multi-step predictions are drastically improved by adopting a multivariate strategy that incorporates external, highly correlated features, specifically the price of gold and closely related cryptocurrencies. Ultimately, they conclude that while deep learning captures the complex, non-linear dynamics of crypto markets well, extreme market disruptions (such as the COVID-19 pandemic) introduce levels of volatility that inherently degrade the predictive capabilities and robustness of all tested models.

## 7. Limitations acknowledged by the authors
- Lack of uncertainty quantification in the point-forecasts, which could be remedied in the future by adopting Bayesian deep learning methods (like MCMC sampling or variational inference).
- The study relied exclusively on unimodal (numerical time-series) data. It did not capture qualitative market drivers such as news media sentiment or social media text, which heavily influence cryptocurrency markets.
- The limited size of the available historical dataset constrained the performance of data-hungry models, particularly the Transformer architecture.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: Next step for bitcoin: Confluence of technical indicators and machine learning

## Metadata
- **BibTeX key**: mate_confluence_2024
- **Authors**: Domicián Máté, Hassan Raza, Ishtiaq Ahmad, and Sándor Kovács
- **Year**: 2024
- **Title**: Next step for bitcoin: Confluence of technical indicators and machine learning
- **Journal/Publication**: Journal of International Studies, 17(3), 68-94
- **DOI/URL**: 10.14254/2071-8330.2024/17-3/4
- **Keywords from article**: bitcoin forecasting, comparative analysis, cryptocurrency market trends, machine learning algorithms, predictive model evaluation

## 1. Study objective(s)
To evaluate and compare the effectiveness of four advanced machine learning models (ANN, CNN-LSTM, SVM, and Random Forest) combined with 27 specific technical indicators in predicting the daily directional price movements of Bitcoin.

## 2. Research question / Hypothesis
How do different machine learning techniques (ANN, SVM, CNN-LSTM, and RF) compare in their predictive power for Bitcoin's one-day directional movement when utilizing a comprehensive suite of 27 technical indicators (including temporal and spatial trends like momentum, ROC, and %K)?

## 3. Methodology
- **Study design**: Comparative quantitative analysis and predictive modeling using historical time-series data and technical indicators for binary classification (predicting upward or downward trends).
- **Sample / corpus**: Historical Bitcoin stock data (OHLCV) obtained from the Yahoo Finance API, spanning September 17, 2014, to October 1, 2023.
- **Instruments / materials**: Python 3.10.4 utilizing the numpy, pandas, tensorflow, and sklearn packages. Four ML models were tested: Artificial Neural Network (ANN), Support Vector Machine (SVM), Convolutional Neural Network combined with Long Short-Term Memory (CNN-LSTM), and Random Forest (RF).
- **Procedures**:
  1. Extracted historical data and removed missing values to clean the dataset.
  2. Calculated 27 technical indicators (e.g., Stochastic Oscillator %K and %D, ROC, MACD, Bollinger Bands, RSI, OBV, Chaikin Oscillator, etc.).
  3. Created a binary "direction" target variable representing the next day's movement (1 for up, 0 for down) by temporally shifting historical data forward by one day to prevent look-ahead bias.
  4. Normalized all feature values to a range between 0 and 1 using Min-Max scaling.
  5. Split the dataset sequentially into 80% for training and 20% for testing.
  6. Trained and optimized the models (e.g., ANN trained for 50 epochs with a batch size of 64 using the Adam optimizer and binary cross-entropy loss).
  7. Evaluated model performance on the test set and extracted feature importance weights to determine which indicators drove the predictions.
- **Data analysis**: Binary classification evaluation metrics, specifically confusion matrices, accuracy, precision, recall, and F1-scores. Feature importance was quantified by analyzing the average absolute weights assigned by the models to the input variables.

## 4. Main results (only facts from the article)
- The Artificial Neural Network (ANN) and Random Forest (RF) models both achieved a strong overall prediction accuracy of 81%.
- The Support Vector Machine (SVM) achieved the highest comparative performance with an overall accuracy of 82% and an F1-score of 0.82 for both positive and negative classifications.
- The CNN-LSTM model demonstrated an accuracy rate of 80%, providing competitive performance but falling slightly behind the ANN, SVM, and RF models.
- Feature importance varied by model architecture. The ANN relied most heavily on %R, %D, %K, and the Chaikin Oscillator; the SVM prioritized %D, %K, and %R; the CNN-LSTM was heavily influenced by %K, ROC, and Momentum; and the RF model was predominantly driven by the Chaikin Oscillator.
- The implemented machine learning models consistently outperformed traditional statistical regression (such as standard ARIMA), often improving prediction accuracy by 20% to 30% in practical applications.

## 5. Relevant direct quotations (max 3)
- "The results showed that ANN and SVM achieve a significant prediction accuracy of 81% and 82%, respectively, which is higher than the results of traditional models such as standard ARIMA." (p. 68-69)
- "The unique contribution of the current work was the use of temporal and spatial trends via momentum, ROC, and %K features, which demonstrated a holistic approach to cryptocurrency market forecasting." (p. 70)
- "Machine learning models typically achieve significantly lower error rates, often improving prediction accuracy by 20-30% over standard ARIMA models in practical applications..." (p. 89)

## 6. Authors' conclusions
The authors conclude that advanced machine learning algorithms (ANN, SVM, CNN-LSTM, and RF) are highly robust and significantly more efficient than traditional regression models at navigating the non-linear volatility of the Bitcoin market. By combining these models with a broad spectrum of 27 technical indicators, particularly highlighting the predictive power of oscillators like %R, %K, and the Chaikin Oscillator, the study proves that algorithmic forecasting can achieve over 80% accuracy. Ultimately, they assert that integrating these data-driven machine learning frameworks into trading strategies can empower investors to make highly informed decisions, improve profitability, and effectively manage financial risks within the cryptocurrency landscape.

## 7. Limitations acknowledged by the authors
- The quality and reliability of historical Bitcoin data can negatively affect the accuracy of the models.
- Sudden price fluctuations and rapid market changes present significant forecasting challenges.
- Deep learning architectures (like Neural Networks) demand substantial computational resources.
- Unexpected regulatory changes in the cryptocurrency sector can disrupt model performance, requiring the models to be continuously adapted.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: Deep learning for Bitcoin price direction prediction: models and trading strategies empirically compared

## Metadata
- **BibTeX key**: omole_enke_2024
- **Authors**: Oluwadamilare Omole and David Enke
- **Year**: 2024
- **Title**: Deep learning for Bitcoin price direction prediction: models and trading strategies empirically compared
- **Journal/Publication**: Financial Innovation
- **DOI/URL**: https://doi.org/10.1186/s40854-024-00643-1
- **Keywords from article**: Backtesting, Bitcoin, Cryptocurrency, Deep learning, Feature selection, On-chain data

## 1. Study objective(s)
To apply deep learning models to predict Bitcoin price directions and assess the subsequent profitability of trading strategies based on those predictions. Specifically, the study compares the performance of CNN-LSTM, LSTNet, and TCN models against an ARIMA benchmark using on-chain data, applying feature-selection methods (Boruta, GA, and LightGBM) to handle high dimensionality.

## 2. Research question / Hypothesis
What predictive improvements can be achieved by applying feature-selection methods to large on-chain datasets for Bitcoin? How do deep learning models (CNN-LSTM, LSTNet, TCN) comprehensively compare against each other and an ARIMA benchmark for price direction prediction? Finally, how profitable is the best-performing model when evaluated using different algorithmic trading strategies?

## 3. Methodology
- **Study design**: Empirical comparative analysis of machine learning/deep learning models for binary classification, followed by financial backtesting of trading strategies.
- **Sample / corpus**: 87 distinct on-chain metrics plus price data sourced from Glassnode. The data spans from February 6, 2013, to February 18, 2023, comprising 3,665 days of data.
- **Instruments / materials**: Three deep learning models (CNN-LSTM, LSTNet, TCN) and one statistical benchmark (ARIMA). Three feature selection methods: Boruta, Genetic Algorithm (GA), and LightGBM.
- **Procedures**:
  1. Data preprocessing involving listwise deletion for data Missing Completely at Random (MCAR) and regression imputation for data Missing Not at Random (MNAR).
  2. Formulating the problem as binary classification, encoding '1' for a price increase the next day and '0' for a decrease or no change.
  3. Splitting data chronologically into 80% training and 20% testing sets without shuffling, followed by standardization.
  4. Performing feature selection using Boruta, GA, and LightGBM to create three subset datasets, alongside an all-features dataset and a univariate price dataset.
  5. Tuning hyperparameters via random search (adjusting learning rate, batch size, epoch count, LSTM units, CNN filters, etc.).
  6. Conducting a parameter study to assess model robustness by varying seed values and rolling window sizes (3, 5, 7, 14, and 30 days).
  7. Applying Wilcoxon signed-rank and Friedman tests to assess the statistical significance of performance differences.
  8. Backtesting three trading strategies (buy-and-sell, buy-and-sell with price protection, and an MACD baseline) across long, short, and long/short positions, factoring in a 30% tax rate and 0.5% transaction costs.
- **Data analysis**: Accuracy, Precision, Recall, F1-score, AUC-ROC, and Matthews correlation coefficient (MCC) for classification evaluation. Rate of Return (ROR), Sharpe ratio, Maximum Drawdown (MDD), Market exposure, and Volatility for backtesting.

## 4. Main results (only facts from the article)
- The combination of Boruta feature selection and the CNN-LSTM model consistently outperformed all other configurations, achieving an overall maximum accuracy of 82.44%.
- The ARIMA benchmark combined with Genetic Algorithm (GA) feature selection was the weakest performer, yielding an overall maximum accuracy of just 50.61%.
- Statistical testing (Wilcoxon signed-rank and Friedman tests) confirmed that the Boruta + CNN-LSTM model was significantly better than the other models.
- In the backtesting simulation, the "long-and-short" buy-and-sell strategy utilizing the Boruta + CNN-LSTM predictions generated an extraordinary annual return of 6653.75% with a Sharpe ratio of 1.8583 and a manageable Maximum Drawdown (MDD) of -0.0704.
- Adding "price protection" mechanisms to the trading strategies (to avoid selling at a loss) significantly reduced annual returns (e.g., dropping the long-and-short strategy to 35.53%) because it limited opportunities for profit during market rebounds.

## 5. Relevant direct quotations (max 3)
- "Results indicate that combining Boruta feature selection with the CNN-LSTM model consistently outperforms other combinations, achieving an accuracy of 82.44%." (p. 1)
- "The long-and-short buy-and-sell investment approach generated an extraordinary annual return of 6654% when informed by higher-accuracy price-direction predictions." (p. 1)
- "This result underscored the importance of feature selection in enhancing DL model performance..." (p. 21)

## 6. Authors' conclusions
The authors conclude that implementing appropriate feature-selection methods, specifically Boruta, significantly enhances the performance of deep learning models like CNN-LSTM for predicting short-term Bitcoin price movements using on-chain data. They establish that a highly accurate predictive model (82.44% accuracy) translates into immense financial viability in a controlled backtesting environment, particularly when utilizing a combined long-and-short trading strategy that vastly outperforms traditional MACD momentum trading. The authors suggest that leveraging unique on-chain data with predictive models provides greater price transparency, which could help securities exchanges and regulators approve new offerings like spot Bitcoin ETFs, ultimately fostering greater market efficiency and risk management.

## 7. Limitations acknowledged by the authors
- The study exclusively utilized on-chain and price data, ignoring the potential predictive value of sentiment data and technical analysis indicators.
- The prediction time frame examined was relatively short, meaning the models' effectiveness at capturing long-term trends under varying market conditions remains unexplored.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: Cryptocurrency Forecasting Using Deep Learning Models: A Comparative Analysis

## Metadata
- **BibTeX key**: bourday_crypto_dl_2024
- **Authors**: Rachid Bourday, Issam Aatouchi, Mounir Ait Kerroum, and Ali Zaaouat
- **Year**: 2024
- **Title**: Cryptocurrency Forecasting Using Deep Learning Models: A Comparative Analysis
- **Journal/Publication**: HighTech and Innovation Journal, Vol. 5, No. 4
- **DOI/URL**: http://dx.doi.org/10.28991/HIJ-2024-05-04-013
- **Keywords from article**: Forecasting; Deep Learning; Machine learning; Transformer; LSTM; XGBoost.

## 1. Study objective(s)
To provide a comprehensive and comparative analysis of novel hybrid deep learning models, specifically Transformer encoders combined with XGBoost, ANN, LSTM, and SVR, to predict short and medium-term Bitcoin prices. The study aims to explore the effectiveness of Transformer-based architectures for complex feature extraction in highly volatile financial markets.

## 2. Research question / Hypothesis
How effectively can hybrid models that utilize a Transformer encoder for feature extraction, combined with distinct predictive algorithms (XGBoost, LSTM, SVR, ANN), forecast Bitcoin prices across 7, 30, and 90-day timeframes, and which combination yields the highest predictive accuracy?

## 3. Methodology
- **Study design**: Quantitative comparative analysis evaluating four hybrid Transformer-based forecasting models across short to medium-term horizons.
- **Sample / corpus**: Historical daily Bitcoin prices and 20 blockchain features (e.g., transaction volume, block size, difficulty, hashrate, active addresses, Twitter/Google trends data) collected from January 1, 2016, to December 31, 2023, totaling 2,922 days.
- **Instruments / materials**: A custom Python web scraper for data collection from bitinfocharts.com. Four hybrid deep learning architectures: Transformer-ANN, Transformer-LSTM, Transformer-SVR, and Transformer-XGBoost.
- **Procedures**:
  1. Scraped 8 years of daily historical data and 20 blockchain features.
  2. Handled missing values using interpolation data cleaning methods.
  3. Performed feature selection using Variance Inflation Factor (VIF) to detect and remove features contributing to high multicollinearity (excluding features with VIF > 5, such as Hashrate and Difficulty).
  4. Applied power transformations and data normalization to align feature scales and address data skewness.
  5. Split the dataset sequentially into a 90% training set and a 10% testing/validation set.
  6. Used a Transformer Encoder to process the preprocessed data, utilizing multi-head self-attention to extract complex long-range temporal dependencies.
  7. Fed the encoded features into four different prediction models (XGBoost, LSTM, ANN, SVR) to generate final predictions for 7, 30, and 90-day horizons.
- **Data analysis**: Model performance was evaluated and compared using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Squared Error (MSE).

## 4. Main results (only facts from the article)
- The Transformer combined with the XGBoost model consistently outperformed all other tested hybrid models across all metrics, achieving the highest overall accuracy with a Mean Absolute Error (MAE) of 0.011 and a Root Mean Squared Error (RMSE) of 0.018.
- For short-term (7-day) forecasting, Transformer + XGBoost achieved a near-zero MSE of 0.0002, significantly outperforming Transformer + SVR (MSE 0.002) and Transformer + LSTM (MSE 0.030).
- The Transformer combined with SVR ranked as the second-best model (overall MAE 0.021, RMSE 0.028), proving more effective than the ANN and LSTM variants but falling short of XGBoost's precision.
- The Transformer combined with LSTM demonstrated the worst predictive accuracy among the four hybrids, yielding the highest error values overall (MAE 0.067, RMSE 0.10) and struggling particularly on shorter timeframes.
- In the 90-day forecast horizon, while XGBoost maintained its superiority, the ANN and LSTM models showed improved accuracy compared to their 7-day and 30-day performance, suggesting they may be better suited for longer-term predictions rather than short-term oscillations.

## 5. Relevant direct quotations (max 3)
- "Our experiments demonstrate that the Transformer with the XGBoost model outperforms the baseline models, achieving a Mean Absolute Error (MAE) of 0.011 and a Root Mean Squared Error (RMSE) of 0.018." (p. 1055)
- "The Transformer with XGBoost model presents the efficient performance across all metrics, indicating high accuracy and reliability. In contrast, the Transformer with LSTM indicates the least accuracy, having the highest values in both metrics." (p. 1062)
- "These models do not currently assess the intensity of market sentiment reflected in text-based data, which is crucial for understanding the volatile cryptocurrency market." (p. 1064)

## 6. Authors' conclusions
The authors conclude that hybrid deep learning models that pair the advanced feature-extraction capabilities of a Transformer Encoder with robust predictive algorithms are highly effective at managing the complexities and non-linear volatility of the cryptocurrency market. Specifically, integrating a Transformer with XGBoost provides the most reliable and accurate forecasts for short to medium-term Bitcoin prices, significantly outperforming traditional neural network architectures like LSTM and ANN in this specific setup. Ultimately, the study proves that Transformer-based feature extraction enhances predictive accuracy, offering a powerful tool for investors, though the authors note that future iterations must incorporate text-based sentiment analysis to capture the full spectrum of market dynamics.

## 7. Limitations acknowledged by the authors
- The models do not integrate dynamic sentiment analysis from text-based sources (like social media), limiting their ability to assess the intensity of market emotions that heavily drive crypto volatility.
- The study relied on a limited number of feature selection methods (only VIF); exploring other techniques could yield further improvements.
- The research focused exclusively on Bitcoin and did not attempt to forecast other major digital currencies like Ethereum or Ripple.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: Conservative Predictions on Noisy Financial Data

## Metadata
- **BibTeX key**: nabar_shroff_2023
- **Authors**: Omkar Nabar and Gautam Shroff
- **Year**: 2023
- **Title**: Conservative Predictions on Noisy Financial Data
- **Journal/Publication**: 4th ACM International Conference on AI in Finance (ICAIF '23)
- **DOI/URL**: https://doi.org/10.1145/3604237.3626859
- **Keywords from article**: [not provided]

## 1. Study objective(s)
To mitigate the impact of feature and label noise in financial predictions by developing a cascading machine-learning approach where models abstain from making predictions when uncertain, thereby trading overall prediction volume (support) for higher utility and reduced downside risk.

## 2. Research question / Hypothesis
Can a cascade of models, specifically Differentiable Decision Trees (DDT) or Multi-Layer Perceptrons (MLP), trained sequentially on increasingly uncertain subsets of noisy financial data improve the utility (average gain per trade) and downside-risk-adjusted returns compared to a single base model?

## 3. Methodology
- **Study design**: Experimental quantitative evaluation comparing base models versus cascaded models on classification tasks across various controlled noise levels and out-of-distribution temporal splits.
- **Sample / corpus**: Two main datasets: (1) Real Indian equity market data (5-minute candles, normalized, ranging from 4,522 to 55,450 points depending on the experiment) and (2) Synthetic sine-wave data simulating mean-reverting markets with parameterized base and peak noise levels (ranging from 1,571 to 7,729 points).
- **Instruments / materials**: Differentiable Decision Trees (DDT) of depth 4 and 6, and Multi-Layer Perceptrons (MLP) with hidden dimensions (128,64,32) and (256,128,64,32). Features include standard OHLCV, 5-bin discretized technical indicators (RSI, MACD, Bollinger-bands), and logical/temporal slope features.
- **Procedures**:
  1. Preprocessed the data by normalizing OHLCV data and discretizing all features and target returns into 5 bins using percentile-based thresholds computed from prior historical data.
  2. Trained an initial base classification model on the training dataset.
  3. Evaluated the model's prediction confidence using the Gini Impurity index of its output class probabilities.
  4. Implemented a 3-level cascade training: data points where the model was uncertain (Gini Impurity > 0.5) were "pruned" (no prediction made) and passed forward to train the next model in the sequence.
  5. Evaluated the models using 4 core experiments: train/test on the same set of eras (days), train/test on different eras, train/test on a single era, and tests with mismatched noise levels between train and test sets.
- **Data analysis**: Evaluation using standard Accuracy and Support (the fraction of data the model confidently predicts). Financial viability was evaluated using custom metrics: Average Utility (average gain/loss per extreme prediction), Downside-Risk-Adjusted Return (DRAR), and Traded Sharpe Ratio.

## 4. Main results (only facts from the article)
- Cascaded models consistently achieved significantly higher Average Utility and Downside-Risk-Adjusted Return (DRAR) than base models, despite making predictions on a much smaller fraction of the data (lower support).
- In Experiment 1 on synthetic data, the Cascaded DDT improved the Average Utility from 1.11 to 1.69 and achieved a massive DRAR increase from 4.31 to 74.75 compared to the base DDT.
- In Experiment 1 on real market data, the Cascaded DDT improved Average Utility from 0.50 to 1.29, while the Cascaded MLP showed the highest DRAR improvement (from 2.75 to 10.89).
- The cascaded models predominantly made predictions at the extreme ends of the target distribution (classes 0 and 4), which are highly actionable for executing long or short algorithmic trading positions.
- When training on noisy synthetic data and testing on clean data, cascading models offered significant accuracy improvements over base models; however, when trained on clean data and tested on noisy data, the cascading models failed to improve performance due to the base model overfitting the clean training data.

## 5. Relevant direct quotations (max 3)
- "We apply a similar approach, where a model abstains from making a prediction on data points that it is uncertain on. During training, a cascade of such models are learned in sequence... with each model being trained only on data on which the previous model(s) were uncertain." (p. 1)
- "...in a financial market scenario, it is preferable to have a model with 70% or even 60% accuracy on say 20% or even 10% of the data, on which it makes confident predictions, and abstains on the balance, as this serves to minimise risk in any decisions taken..." (p. 2)
- "We observe that most of the data points on which the cascaded models make a prediction, i.e., the un-pruned data, are at the extremes of the target distribution, i.e., predicting classes 0 or 4... confident and accurate predictions at the extremes are exactly what is desired for making trading decisions..." (p. 5)

## 6. Authors' conclusions
The authors conclude that utilizing a cascade of models that purposefully abstain from predicting uncertain instances is a highly effective strategy for navigating noisy financial data. While this approach inevitably sacrifices prediction volume (support), it substantially increases the reliability, utility, and profitability of the predictions that are actually made. They emphasize that cascaded Differentiable Decision Trees (DDTs), in particular, yield superior risk-adjusted returns and actionable trading signals at the market extremes compared to standard MLPs. Ultimately, minimizing risk by abstaining from uncertain trades protects traders against severe drawdowns and margin calls, making conservative prediction models practically superior for leveraged algorithmic trading strategies.

## 7. Limitations acknowledged by the authors
- Performance of all evaluated models degrades when tested on distributions (eras) different from those they are trained on, highlighting vulnerability to market regime shifts.
- Cascaded models fail to improve performance when trained on perfectly clean data and tested on noisy data, as the base model overfits the clean data and fails to trigger the pruning mechanism properly.
- Small dataset sizes (e.g., training on a single era) lead to severe overfitting, neutralizing the benefits of the cascading mechanism.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---