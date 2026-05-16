# Article record: KAN: Kolmogorov-Arnold Networks

## Metadata
- **BibTeX key**: liu_kan_2024
- **Authors**: Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark
- **Year**: 2024
- **Title**: KAN: Kolmogorov-Arnold Networks
- **Journal/Publication**: arXiv preprint arXiv:2404.19756
- **DOI/URL**: [not provided]
- **Keywords from article**: [not provided]

## 1. Study objective(s)
To propose Kolmogorov-Arnold Networks (KANs) as a promising architectural alternative to Multi-Layer Perceptrons (MLPs), demonstrating that placing learnable 1D spline functions on network edges rather than fixed activation functions on nodes significantly improves both predictive accuracy and human interpretability for AI and science tasks.

## 2. Research question / Hypothesis
Do Kolmogorov-Arnold Networks, which parameterize all weight parameters as univariate spline functions based on the Kolmogorov-Arnold representation theorem, outperform traditional MLPs in terms of neural scaling laws, function fitting accuracy, and explicit interpretability (symbolic extraction) in scientific tasks?

## 3. Methodology
- **Study design**: Theoretical architectural proposal combined with empirical comparative evaluation of KANs against MLPs across mathematical, physical, and synthetic data tasks.
- **Sample / corpus**: A variety of datasets were used, including synthetic datasets for fitting special mathematical functions (e.g., Bessel, Jacobi), the Feynman dataset (consisting of 100 physical equations), a knot theory dataset (with 17 topological invariants), and a condensed matter physics dataset (for Anderson localization).
- **Instruments / materials**: Kolmogorov-Arnold Networks (built with B-spline activation functions), Multi-Layer Perceptrons (MLPs), L-BFGS optimizer, and the PyTorch framework.
- **Procedures**:
  1. Designed the KAN architecture by generalizing the exact Kolmogorov-Arnold representation theorem to arbitrary network widths and depths, using matrices of learnable 1D spline functions on the edges.
  2. Implemented structural improvements including grid extension (to avoid spline boundary issues), grid regularization, and sparsification techniques (using L1 regularization and entropy penalties) to enhance both accuracy and interpretability.
  3. Conducted extensive synthetic data experiments to compare the function-fitting accuracy and neural scaling laws of KANs versus MLPs.
  4. Tested the interpretability and symbolic extraction capabilities of KANs on the Feynman dataset by pruning the networks down to minimal sub-graphs and identifying the exact mathematical formulas.
  5. Applied KANs to authentic scientific discovery tasks, specifically predicting topological invariants in knot theory and identifying the mobility edge in Anderson localization.
- **Data analysis**: Evaluation using Root Mean Square Error (RMSE) loss, parameter count comparisons, computation of neural scaling exponents ($\alpha$), and evaluating the success rate of symbolic extraction (symbolification).

## 4. Main results (only facts from the article)
- KANs consistently exhibited much faster neural scaling laws than MLPs (e.g., $N^{-4}$ vs $N^{-1}$), allowing them to achieve higher predictive accuracy with significantly fewer parameters in function fitting tasks.
- Through sparsification and pruning, trained KANs could be "symbolified" to extract exact, closed-form mathematical formulas, successfully rediscovering known physical laws from the Feynman dataset.
- KANs demonstrated their ability to act as "collaborators" in scientific discovery by uncovering a new nonlinear relation among topological invariants in knot theory and extracting the exact mobility edge formula for Anderson localization.
- KANs were shown to naturally mitigate catastrophic forgetting compared to MLPs; because B-splines are locally supported, new data points only update local spline coefficients without destroying globally learned representations.

## 5. Relevant direct quotations (max 3)
- "While MLPs have fixed activation functions on nodes ("neurons"), KANs have learnable activation functions on edges ("weights"). KANs have no linear weights at all, every weight parameter is replaced by a univariate function parametrized as a spline." (p. 1)
- "We show that KANs possess faster scaling laws than MLPs, and can be pruned down to individual functions, bridging the gap between machine learning and human interpretability." (p. 31)
- "In summary, we have shown that KANs are promising alternatives to MLPs for scientific tasks, both because of their higher accuracy (faster scaling) and higher interpretability." (p. 33)

## 6. Authors' conclusions
The authors conclude that Kolmogorov-Arnold Networks represent a fundamental architectural paradigm shift from standard MLPs, offering a highly compelling alternative for AI and scientific tasks where accuracy and interpretability are critical. By replacing static nodes with learnable 1D B-splines on the network edges, KANs achieve faster neural scaling laws and allow for the precise mathematical symbolification of learned representations. The authors assert that despite current computational inefficiencies related to hardware optimization, the mathematical elegance and empirical success of KANs in data-fitting and knowledge discovery prove their massive potential as an essential tool for future AI-assisted scientific research.

## 7. Limitations acknowledged by the authors
- The primary limitation is slow training speed; KANs are currently about 10 times slower than MLPs with the same number of parameters because each weight is an independent spline function, making it difficult to fully exploit batch matrix multiplication optimizations in modern GPUs.
- The current evaluation of KANs is restricted to relatively small-scale AI+Science tasks; their scalability to massive machine learning tasks (such as large language modeling) remains untested and potentially challenging.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: KAN 2.0: Kolmogorov-Arnold Networks Meet Science

## Metadata
- **BibTeX key**: liu_kan2_2024
- **Authors**: Ziming Liu, Pingchuan Ma, Yixuan Wang, Wojciech Matusik, and Max Tegmark
- **Year**: 2024
- **Title**: KAN 2.0: Kolmogorov-Arnold Networks Meet Science
- **Journal/Publication**: arXiv preprint arXiv:2408.10205
- **DOI/URL**: [not provided]
- **Keywords from article**: [not provided]

## 1. Study objective(s)
To propose a unified framework bridging connectionist AI and symbolic science by seamlessly synergizing Kolmogorov-Arnold Networks (KANs) with scientific discovery processes. The study aims to introduce major new functionalities (MultKAN, kanpiler, and tree converter) to facilitate a bidirectional integration: incorporating scientific knowledge into KANs and extracting scientific insights from them.

## 2. Research question / Hypothesis
How can Kolmogorov-Arnold Networks be architecturally extended and equipped with specific software tools to seamlessly incorporate prior scientific knowledge, discover hidden modular structures, and extract transparent symbolic formulas (such as physical laws) directly from data?

## 3. Methodology
- **Study design**: Methodological and architectural extension of KANs, followed by empirical validation on physics and mathematical tasks.
- **Sample / corpus**: Various synthetic and physical datasets simulating fundamental physical phenomena, including conserved quantities, Lagrangians, symmetries, and constitutive laws.
- **Instruments / materials**: MultKAN architecture (KANs with multiplication nodes), kanpiler (a compiler translating analytical formulas into KAN architectures), and tree converter (a tool converting networks into tree graphs). Implementation within the pykan library.
- **Procedures**:
  1. Developed MultKAN by introducing multiplication nodes into the standard KAN architecture to natively handle the multiplicative interactions common in physical laws.
  2. Created the kanpiler to allow researchers to compile known scientific equations directly into the KAN's structure, effectively injecting explicit inductive biases into the network prior to training.
  3. Designed the tree converter to transform trained neural network topologies into explicit parse trees to facilitate structural and modular analysis.
  4. Tested the bidirectional framework (Science $\rightarrow$ KAN and KAN $\rightarrow$ Science) on tasks discovering features, modular structures, and symbolic formulas across various physical examples.
- **Data analysis**: Symbolic regression, network pruning, modularity analysis via tree graphs, and evaluating the exact symbolic recovery of fundamental physical equations.

## 4. Main results (only facts from the article)
- The introduction of MultKAN (KANs with multiplication nodes) significantly improved the network's ability to model variable interactions and extract exact symbolic formulas compared to standard additive KANs.
- The kanpiler successfully demonstrated the ability to embed prior analytical knowledge (mathematical formulas) directly into KAN architectures, establishing a reliable "science-to-KAN" pipeline that guides model training.
- The tree converter effectively revealed the hidden modular structures of trained networks, aiding in the interpretability of complex interactions.
- The unified framework proved highly effective at autonomously discovering diverse physical laws, including conserved quantities, Lagrangians, and symmetries, directly from data by extracting human-readable formulas.

## 5. Relevant direct quotations (max 3)
- "To bridge the two worlds, we propose a framework to seamlessly synergize Kolmogorov-Arnold Networks (KANs) and science. The framework highlights KANs' usage for three aspects of scientific discovery: identifying relevant features, revealing modular structures, and discovering symbolic formulas." (p. 1)
- "We highlight major new functionalities in pykan: (1) MultKAN: KANs with multiplication nodes. (2) kanpiler: a KAN compiler that compiles symbolic formulas into KANs. (3) tree converter: convert KANs (or any neural networks) to tree graphs." (p. 1)

## 6. Authors' conclusions
The authors conclude that KAN 2.0 represents a major step toward unifying the connectionist capabilities of modern AI with the symbolic requirements of rigorous science. By introducing multiplication nodes and dedicated compilation/conversion tools, the framework allows researchers to seamlessly inject prior physical knowledge into networks and reliably extract transparent, symbolic formulas from them. They assert that KANs are uniquely suited for AI-driven scientific discovery, functioning as collaborative tools that identify features, reveal structural modularity, and derive mathematical laws across diverse physical and scientific domains.

## 7. Limitations acknowledged by the authors
- Training KANs and extracting complex symbolic formulas remains computationally intensive compared to standard MLPs.
- The process of symbolification and pruning heavily relies on careful hyperparameter tuning and human intervention, meaning the formula extraction is not yet fully automated.
- The scalability of extracting clean symbolic formulas from highly noisy, high-dimensional real-world data requires further exploration, as current demonstrations focus primarily on fundamental physics tasks.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: DecoKAN: Interpretable Decomposition for Forecasting Cryptocurrency Market Dynamics

## Metadata
- **BibTeX key**: gao_decokan_2025
- **Authors**: Yuan Gao, Zhenguo Dong, Xuelong Wang, Zhiqiang Wang, Yong Zhang, and Shaofan Wang
- **Year**: 2025
- **Title**: DecoKAN: Interpretable Decomposition for Forecasting Cryptocurrency Market Dynamics
- **Journal/Publication**: arXiv preprint arXiv:2512.20028v1 [cs.LG]
- **DOI/URL**: [not provided]
- **Keywords from article**: Cryptocurrency, Time Series Forecasting, Kolmogorov-Arnold Networks, Wavelet Transform

## 1. Study objective(s)
To propose DecoKAN, an interpretable forecasting framework that synergistically integrates multi-level Discrete Wavelet Transform (DWT) for hierarchical signal decomposition with Kolmogorov-Arnold Network (KAN) mixers for transparent, nonlinear modeling of complex cryptocurrency market dynamics.

## 2. Research question / Hypothesis
Can a forecasting framework combining Discrete Wavelet Transform (DWT) with Kolmogorov-Arnold Networks (KANs) accurately decouple and predict the composite dynamics (long-term trends and high-frequency speculative noise) of cryptocurrency markets while providing intrinsic, mathematically explicit interpretability?

## 3. Methodology
- **Study design**: Methodological proposal and empirical quantitative evaluation against state-of-the-art baselines on multivariate time series forecasting.
- **Sample / corpus**: Four general ETT benchmark datasets (ETTh1, ETTh2, ETTm1, ETTm2) and three real-world cryptocurrency datasets (Bitcoin/BTC, Ethereum/ETH, Monero/XMR) sourced from Coin Metrics.
- **Instruments / materials**: Python/PyTorch framework, multi-level Discrete Wavelet Transform (using 'db4' wavelet), KAN mixers (incorporating learnable B-spline activations), and Reversible Instance Normalization (RevIN).
- **Procedures**:
  1. Normalized input multivariate time series using RevIN to mitigate distribution shifts.
  2. Decomposed the normalized series into distinct approximation (low-frequency) and detail (high-frequency) coefficient series using multi-level DWT.
  3. Processed each coefficient series independently through dedicated KAN Resolution Branches containing Temporal and Feature KAN Mixers to learn frequency-specific patterns.
  4. Synthesized the predicted coefficients back into the time domain using Inverse DWT (IDWT) and denormalized the final output.
  5. Evaluated model accuracy (MSE and MAE) against various MLP, Transformer, CNN, and Graph-based baselines across multiple prediction horizons.
  6. Applied a three-stage interpretability pipeline (Sparsification via L1/entropy regularization, Pruning based on L2 norm threshold, and Symbolification) to extract explicit mathematical formulas from the trained KAN mixers.
- **Data analysis**: Mean Squared Error (MSE) and Mean Absolute Error (MAE) for forecasting accuracy; computational efficiency analysis (Parameters, GFLOPs, Training/Inference time); symbolic regression analysis ($R^2$ scores) for interpretability.

## 4. Main results (only facts from the article)
- DecoKAN achieved state-of-the-art long-term forecasting performance, recording the lowest average MSE on all tested real-world cryptocurrency datasets (BTC, ETH, XMR) across all prediction horizons compared to baseline models like WPMixer and TimeFilter.
- Compared to the strong WPMixer baseline, DecoKAN reduced the average MSE by approximately 15.0% on BTC, 29.1% on ETH, and 15.1% on XMR.
- An ablation study revealed a functional separation based on dataset types: the Feature KAN mixer was critical for capturing instantaneous cross-variate dependencies in volatile crypto data, while the Temporal KAN mixer was more important for highly autocorrelated general datasets like ETTh2.
- Interpretability analysis (pruning) showed that the Approximation Branch acts as a dense, robust structural backbone (only 4.79% of edges pruned), whereas the Detail Branch learns highly sparse representations (76.28% pruned) to effectively model transient fluctuations.
- The symbolification process successfully extracted explicit mathematical formulas (predominantly polynomials and trigonometric functions) from the Detail Branch with high confidence ($R^2 > 0.99$), explicitly explaining the high-frequency market volatility.
- DecoKAN demonstrated highly efficient theoretical computation (e.g., 0.0073 GFLOPs at T=96) and competitive inference speeds, despite requiring longer training times due to unoptimized B-spline operations.

## 5. Relevant direct quotations (max 3)
- "To overcome these limitations, we propose DecoKAN, an interpretable forecasting framework that integrates multi-level Discrete Wavelet Transform (DWT) for decoupling ing and hierarchical signal decomposition with Kolmogorov-Arnold Network (KAN) mixers for transparent and interpretable nonlinear modeling." (p. 1)
- "Unlike traditional time series (e.g., electricity load) dominated by strong temporal autocorrelation, cryptocurrency prices are often driven by high-frequency shocks and the instantaneous interplay between system variables... Consequently, the performance of DecoKAN here relies predominantly on the Feature KAN component's ability to capture these complex cross-variate dependencies..." (p. 9)
- "Interpretability analysis revealed a functional separation within the model: the Detail Branch captures explicit symbolic relations for high-frequency fluctuations, while the Approximation Branch acts as a robust structural backbone for the forecasting task." (p. 11)

## 6. Authors' conclusions
The authors conclude that DecoKAN successfully bridges the gap between high predictive accuracy and intrinsic model transparency for complex multivariate time series. By utilizing a multi-resolution wavelet decomposition (DWT) combined with Kolmogorov-Arnold Networks (KANs), the framework effectively separates slow-moving trends from high-frequency market noise. Furthermore, the intrinsic interpretability of the KAN mixers allows the extraction of explicit mathematical laws governing cryptocurrency dynamics. They assert that this combination of decoupled signal processing and symbolic transparency makes DecoKAN highly suitable for trustworthy financial decision-making, algorithmic trading, and risk assessment, despite current hardware-related training inefficiencies.

## 7. Limitations acknowledged by the authors
- Slower training times compared to standard MLP or Transformer architectures due to the computationally intensive nature of B-spline operations in KANs, which currently lack optimized GPU parallelization.
- The symbolification process relies on a pre-defined, finite set of candidate mathematical functions to fit the curves.
- Interpreting the extracted symbolic graphs and formulas still requires significant human domain expertise.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: TKAN: Temporal Kolmogorov-Arnold Networks

## Metadata
- **BibTeX key**: genet_inzirillo_2024
- **Authors**: Rémi Genet and Hugo Inzirillo
- **Year**: 2024
- **Title**: TKAN: Temporal Kolmogorov-Arnold Networks
- **Journal/Publication**: arXiv preprint arXiv:2405.07344v4
- **DOI/URL**: https://arxiv.org/abs/2405.07344
- **Keywords from article**: [not provided]

## 1. Study objective(s)
To propose and evaluate a novel neural network architecture called Temporal Kolmogorov-Arnold Networks (TKAN), which integrates the memory management mechanisms of Long Short-Term Memory (LSTM) networks with the learnable spline activations of Kolmogorov-Arnold Networks (KANs) for enhanced multi-step time series forecasting.

## 2. Research question / Hypothesis
Can the integration of KAN-based layers into an LSTM-like recurrent architecture (forming Recurring Kolmogorov-Arnold Networks or RKANs) significantly improve prediction accuracy and stability for multi-step time series forecasting compared to traditional RNNs, GRUs, and LSTMs?

## 3. Methodology
- **Study design**: Architectural proposal and empirical quantitative comparative analysis on synthetic and real-world financial time series data.
- **Sample / corpus**: Two distinct datasets were utilized: (1) A synthetic Sine Wave dataset simulating regular, predictable periodic sequences, and (2) A real-world historical Bitcoin (BTC) market dataset utilizing 1-hour timeframe OHLCV data to test complex, noisy financial patterns.
- **Instruments / materials**: The proposed TKAN model (built using RKAN layers), alongside established baseline models including standard LSTM, GRU, RNN, and standard KAN. Implementation was carried out using Python and standard deep learning frameworks.
- **Procedures**:
  1. Designed the foundational RKAN layer by substituting the standard fully connected dense layers within traditional LSTM gating mechanisms (the forget, input, output, and cell state gates) with KANLinear layers (which utilize B-splines instead of fixed activations).
  2. Conducted initial baseline evaluations using the generated sine wave dataset to verify the model's fundamental ability to capture and reproduce sequential patterns across multiple future steps.
  3. Processed and sequenced the historical Bitcoin dataset for a multi-step forecasting task.
  4. Trained the TKAN, LSTM, and GRU models on the Bitcoin data under identical configurations.
  5. Evaluated and compared the models on out-of-sample data across multiple prediction steps to assess long-term sequence generation and error accumulation.
- **Data analysis**: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared ($R^2$) scores were utilized to quantify forecasting accuracy, stability, and goodness-of-fit over extended prediction horizons.

## 4. Main results (only facts from the article)
- On the synthetic sine wave dataset, TKAN significantly outperformed traditional LSTM and GRU models in multi-step predictions, accurately maintaining the wave structure and amplitude over long horizons where other models degraded or flatlined.
- On the real-world Bitcoin dataset, TKAN consistently achieved lower Mean Squared Error (MSE) and Mean Absolute Error (MAE) compared to both LSTM and GRU across multiple out-of-sample prediction horizons.
- TKAN demonstrated superior stability and reduced error accumulation over long periods (multi-step forecasting), effectively addressing a common weakness of standard recurrent networks which often struggle with long-term predictive degradation.
- The experiments empirically proved that the core idea of the original KAN paper (using learnable activation functions on edges) is highly relevant and transferable to sequential time-series analysis when combined with memory management.

## 5. Relevant direct quotations (max 3)
- "Inspired by the Kolmogorov-Arnold Networks (KANs), a promising alternative to Multi-Layer Perceptrons (MLPs), we propose a new neural network architecture inspired by KAN and LSTM, called 'Temporal Kolomogorov-Arnold Networks' (TKANs)." (p. 1)
- "By substituting the dense layers in traditional RNNs or LSTMs with KAN layers, TKANs benefit from the exact representational power of KANs, allowing for a more accurate and efficient processing of sequential data." (p. 3)
- "While not specifically interesting for short-term predictions, it especially demonstrates an ability to largely outperform other models when it comes to multi-step predictions." (p. 6)

## 6. Authors' conclusions
The authors conclude that embedding KAN layers into recurrent architectures (specifically replacing LSTM dense gates with KANLinear layers to form RKANs) creates a highly robust model for time series analysis. TKAN successfully leverages the high-accuracy function approximation of B-splines while preserving the vital memory management required for processing sequential data. They emphasize that while TKAN may not be strictly necessary for very simple or short-term single-step predictions, it drastically outperforms standard LSTMs and GRUs in multi-step, long-horizon forecasting tasks due to its superior stability and reduced error accumulation over time.

## 7. Limitations acknowledged by the authors
- TKAN architectures involve higher computational complexity during training due to the evaluation of B-splines in every gate, making them computationally heavier and slower to train than standard LSTMs.
- The model does not provide significant predictive advantages for very short-term (e.g., single-step) predictions, making it overkill for simpler forecasting tasks.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: A Practitioner's Guide to Kolmogorov-Arnold Networks

## Metadata
- **BibTeX key**: noorizadegan_2026
- **Authors**: Amir Noorizadegan, Sifan Wang, Leevan Ling, and Juan P. Dominguez-Morales
- **Year**: 2026
- **Title**: A Practitioner's Guide to Kolmogorov-Arnold Networks
- **Journal/Publication**: arXiv preprint arXiv:2510.25781
- **DOI/URL**: [not provided]
- **Keywords from article**: [not provided]

## 1. Study objective(s)
To provide a systematic and comprehensive overview of the rapidly expanding Kolmogorov-Arnold Network (KAN) literature, clarifying their theoretical foundations, and to offer a highly practical "Choose-Your-KAN" guide to help researchers select and optimize these models.

## 2. Research question / Hypothesis
How do various Kolmogorov-Arnold Network architectures relate to traditional MLPs and classical Kolmogorov Superposition Theory, and what are the best practices for systematically designing, selecting, and optimizing KANs based on specific problem requirements and basis functions?

## 3. Methodology
- **Study design**: Comprehensive literature review and practitioner's methodological guide/taxonomy.
- **Sample / corpus**: The rapidly expanding body of recent literature proposing and evaluating variants of Kolmogorov-Arnold Networks (KANs).
- **Instruments / materials**: Theoretical analysis of the Kolmogorov Superposition Theorem (KST), Multi-Layer Perceptrons (MLPs), classical kernel methods, and various basis functions (e.g., B-splines, Wavelets, Chebyshev polynomials).
- **Procedures**:
  1. Clarified the theoretical relationships between KANs, exact KST, MLPs, and classical kernel methods.
  2. Categorized and analyzed KAN variants based on their central design axis: the choice of basis functions on the network edges.
  3. Summarized recent literature advances regarding model accuracy, computational efficiency, regularization techniques, and convergence properties.
  4. Developed a step-by-step "Choose-Your-KAN" algorithmic guide (decision tree) to direct practical implementation based on data properties and computational constraints.
- **Data analysis**: Qualitative synthesis, taxonomic classification of neural network architectures, and development of practical heuristic frameworks.

## 4. Main results (only facts from the article)
- Modern KANs are practically "inspired by" rather than strictly "dictated by" the exact Kolmogorov Superposition Theorem, allowing them to bypass the pathological roughness of exact KST representations to achieve empirical machine learning utility.
- The choice of basis function (e.g., B-splines, Chebyshev polynomials, Radial Basis Functions, Wavelets) serves as the central design axis, fundamentally dictating the network's accuracy, efficiency, and degree of interpretability.
- While standard KANs suffer from computational inefficiency due to the cost of evaluating independent splines per edge, the review identifies that newer variants (such as FastKAN, ReLU-KAN, and implementations using JAX/CUDA kernels) significantly mitigate this speed bottleneck.
- The study established a comprehensive 6-step framework for practitioners, providing explicit rules such as: using Wavelet KANs for multiscale/heterogeneous problems, applying residual-adaptive sampling for localized errors, and prioritizing ReLU-KAN or FastKAN when computational speed is the primary objective.

## 5. Relevant direct quotations (max 3)
- "Kolmogorov-Arnold Networks (KANs), whose design is inspired rather than dictated by the Kolmogorov superposition theorem, have emerged as a structured alternative to MLPs." (p. 1)
- "The review is organized around three core themes: (i) clarifying the relationships between KANs and Kolmogorov superposition theory (KST), MLPs, and classical kernel methods; (ii) analyzing basis functions as a central design axis..." (p. 1)
- "if problem is multiscale or spatially heterogeneous then Use Wavelet KAN or FBKAN... if speed is the primary objective then Prefer ReLU-KAN, HRKAN, FastKAN, LeanKAN" (p. 21)

## 6. Authors' conclusions
The authors conclude that KANs offer a highly structured, expressive, and interpretable alternative to traditional MLPs, effectively bridging the gap between exact mathematical approximation theory and empirical deep learning. By understanding the theoretical underpinnings and treating the choice of basis functions and grid resolutions as configurable design axes, practitioners can tailor KANs to solve complex, domain-specific problems. They emphasize that while standard KANs face recognized computational and memory challenges, the rapidly growing ecosystem of specialized variants and hardware optimizations makes these networks increasingly viable for both high-speed applications and rigorous scientific discovery.

## 7. Limitations acknowledged by the authors
- The primary limitation highlighted across the KAN literature is their computational overhead and slower training speeds compared to highly optimized MLPs, primarily due to the necessity of evaluating complex, independent basis functions on every network connection.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: Kolmogorov-Arnold networks for time series forecasting: a comprehensive review

## Metadata
- **BibTeX key**: yamak_et_al_2025
- **Authors**: Peter Tettey Yamak, Yujian Li, Ting Zhang, and Muhammad Salman Pathan
- **Year**: 2025
- **Title**: Kolmogorov-Arnold networks for time series forecasting: a comprehensive review
- **Journal/Publication**: Cluster Computing
- **DOI/URL**: 10.1007/s10586-025-05574-9
- **Keywords from article**: Kolmogorov-Arnold Networks, Time series forecasting, B-splines, Explainable AI, Nonlinear dynamics, Deep learning

## 1. Study objective(s)
To present a comprehensive review of Kolmogorov-Arnold Networks (KANs) applied to time series forecasting (TSF), analyzing how they leverage the Kolmogorov-Arnold theorem to decompose temporal dependencies into interpretable, spline-parameterized univariate functions, while evaluating their empirical performance, architectural innovations, and current limitations.

## 2. Research question / Hypothesis
How do Kolmogorov-Arnold Networks (KANs) transform time series forecasting by replacing fixed node activations with learnable spline functions on edges, and how do they compare against state-of-the-art models like Transformers and MLPs in terms of predictive accuracy, nonlinear modeling, and intrinsic interpretability?

## 3. Methodology
- **Study design**: Systematic and comprehensive literature review of neural network architectures in time series forecasting.
- **Sample / corpus**: The rapidly expanding body of literature proposing, modifying, and applying KANs to time series forecasting tasks across various domains (finance, IoT, hydrology, etc.).
- **Instruments / materials**: Theoretical analysis of the Kolmogorov-Arnold representation theorem, Multi-Layer Perceptrons (MLPs), Transformer architectures, and KAN variants (including B-spline, wavelet, and polynomial-based implementations).
- **Procedures**:
  1. Detailed the mathematical foundations distinguishing KANs from traditional MLPs (learnable edge functions vs. fixed node activations).
  2. Categorized the recent architectural innovations in KANs, including adaptive grid refinement, gated residual mechanisms, and hybrid attention integration.
  3. Synthesized the empirical performance of KAN-based models across univariate, multivariate, and high-frequency time series forecasting tasks.
  4. Analyzed the mechanisms through which KANs provide "unparalleled transparency," specifically focusing on feature attribution and explicit symbolic representation.
  5. Identified the critical computational bottlenecks and scalability limitations facing current KAN implementations.
- **Data analysis**: Qualitative synthesis, taxonomic classification of network architectures, and comparative evaluation of empirical forecasting metrics (such as MSE) reported in the primary literature.

## 4. Main results (only facts from the article)
- KAN-based models frequently achieve state-of-the-art performance in time series forecasting, outperforming sophisticated Transformer models with up to 98% lower Mean Squared Error (MSE) on specific benchmarks.
- The architecture of KANs, utilizing spline-parameterized univariate functions on the network edges, is highly effective at modeling complex, nonlinear dynamics across diverse data regimes, including univariate, multivariate, and high-frequency data.
- KANs provide unparalleled transparency and interpretability in time series forecasting by allowing for explicit feature attribution and the potential decomposition of the network into interpretable mathematical functions.
- The framework has seen rapid domain adaptation, with highly successful applications spanning cryptocurrency volatility prediction, hydrological forecasting, and IoT-enabled anomaly detection.
- Despite their predictive and interpretative superiority, standard KAN implementations suffer from significant computational overhead during spline optimization and face challenges scaling in high-dimensional settings compared to the highly optimized matrix multiplications of MLPs.

## 5. Relevant direct quotations (max 3)
- "This study presents Kolmogorov-Arnold Networks (KANs) for time series forecasting (TSF) as a transformative paradigm, leveraging the Kolmogorov-Arnold theorem to decompose multivariate temporal dependencies into interpretable, spline-parameterized univariate functions." (p. 1)
- "Through architectural innovations like adaptive grid refinement, gated residual mechanisms, and hybrid attention integration, KAN-based models achieve state-of-the-art performance, outperforming transformers with up to 98% lower MSE on benchmarks while offering unparalleled transparency in feature attribution." (p. 1)
- "However, challenges persist, including computational overhead in spline optimization and scalability in high-dimensional settings." (p. 1)

## 6. Authors' conclusions
The authors conclude that Kolmogorov-Arnold Networks represent a highly effective and transformative paradigm for time series forecasting. By successfully translating the Kolmogorov-Arnold representation theorem into a deep learning architecture, KANs bridge the longstanding gap between high predictive power and model interpretability. They excel at capturing complex nonlinear dynamics in fields ranging from cryptocurrency to IoT. However, the authors caution that the widespread adoption of KANs is currently bottlenecked by the computational intensity of spline optimization and the difficulties of scaling these networks to massive, high-dimensional datasets, which remain critical areas for future research.

## 7. Limitations acknowledged by the authors
- Significant computational overhead required for spline optimization during the training phase.
- Scalability challenges when applied to extremely high-dimensional datasets.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: Forecasting VIX using interpretable Kolmogorov-Arnold networks

## Metadata
- **BibTeX key**: cho_lee_kim_2025
- **Authors**: So-Yoon Cho, Sungchul Lee, and Hyun-Gyoon Kim
- **Year**: 2025
- **Title**: Forecasting VIX using interpretable Kolmogorov-Arnold networks
- **Journal/Publication**: arXiv preprint arXiv:2502.00980v1
- **DOI/URL**: https://arxiv.org/abs/2502.00980
- **Keywords from article**: Kolmogorov-Arnold network; Interpretable neural network; Time-series forecasting; Volatility; VIX

## 1. Study objective(s)
To employ Kolmogorov-Arnold Networks (KANs) for forecasting the CBOE Volatility Index (VIX) to overcome the black-box limitations of traditional multi-layer perceptrons (MLPs) by providing an interpretable, closed-form symbolic expression of the forecast.

## 2. Research question / Hypothesis
Can KANs achieve competitive forecasting performance for the VIX while utilizing significantly fewer parameters than MLPs/LSTMs and providing explicit, interpretable mathematical formulas that capture key volatility characteristics like mean reversion and the leverage effect?

## 3. Methodology
- **Study design**: Empirical comparative quantitative analysis evaluating time-series forecasting accuracy and model interpretability.
- **Sample / corpus**: Daily CBOE VIX data from January 2004 to December 2023, alongside S&P 500 daily excess returns and 1-month U.S. Treasury bill yields.
- **Instruments / materials**: A parsimonious 2-layer KAN with 2 hidden nodes; traditional statistical models (ARMA, ARIMA, HAR, Forward filling); and neural network baselines (MLP, LSTM).
- **Procedures**:
  1. Preprocessed the historical VIX data into three distinct datasets designed to capture short-term lags, non-uniform lags, and aggregated multi-horizon averages (weekly, monthly, quarterly).
  2. Split the data chronologically into training, validation, and testing sets across three different period configurations to ensure robustness.
  3. Trained the KAN models using the L-BFGS optimizer, subsequently pruning unimportant edges and nodes based on low activation impact.
  4. Performed symbolification by replacing the trained B-spline-based activation functions with explicit symbolic functions (predominantly linear) and fine-tuning the affine parameters.
  5. Evaluated and compared forecasting accuracy against baseline models using statistical tests (Mincer-Zarnowitz, Durbin-Watson) and error metrics.
  6. Augmented the KAN model with S&P 500 excess returns to test its ability to dynamically capture the leverage effect.
- **Data analysis**: Evaluation utilizing Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), R-squared ($R^2$), and Quasi-Likelihood (QLIKE).

## 4. Main results (only facts from the article)
- The trained B-spline activation functions naturally converged to nearly linear shapes, indicating that the tested explanatory variables primarily exert a linear influence on VIX forecasting.
- The symbolification process successfully extracted parsimonious closed-form mathematical expressions, slightly improving the $R^2$ metric in almost all cases by mitigating slight over-fitting inherent in the splines.
- KAN explicitly captured the mean-reverting property of the VIX, revealing a dominant dependence on the previous day's value ($V_{t-1}$ coefficient typically > 0.8) and strong predictability from aggregated historical averages.
- When augmented with S&P 500 excess returns, the KAN model explicitly captured the leverage effect, yielding downward-sloping activation functions that correctly associated lower stock returns with a higher predicted VIX.
- KAN achieved forecasting accuracy (MSE, MAE, etc.) highly competitive with or superior to deep learning benchmarks (MLP and LSTM) while utilizing approximately 300 to 1,000 times fewer parameters (e.g., ~60-72 parameters for KAN vs. ~63,000 for LSTM).

## 5. Relevant direct quotations (max 3)
- "Based on a parsimonious architecture with symbolic functions, KAN expresses a forecast of the VIX as a closed-form in terms of explanatory variables, and provide interpretable insights into key characteristics of the VIX, including mean reversion and the leverage effect." (p. 1)
- "Whereas these statistical models presuppose linearity, KAN identifies such structures directly from data, imposing no prior assumptions on its functional form." (p. 6)
- "Remarkably, KAN's parsimonious design consistently achieves performance on par with these more parameter-intensive MLP-based networks, underscoring its robustness and stability in noisy financial time-series data." (p. 14)

## 6. Authors' conclusions
The authors conclude that Kolmogorov-Arnold Networks provide a highly robust, interpretable, and parsimonious alternative to traditional black-box deep learning models (like MLPs and LSTMs) for financial time-series forecasting. By automatically pruning unimportant features and utilizing symbolification, KAN generates closed-form mathematical expressions that accurately describe the inherent characteristics of the VIX, such as volatility clustering, mean reversion, and the leverage effect. Ultimately, KANs achieve state-of-the-art predictive accuracy with vastly fewer parameters, supplying financial decision-makers with the transparency required for trustworthy risk management.

## 7. Limitations acknowledged by the authors
[not provided] (The authors do not explicitly outline a dedicated limitations section for their methodology, though they acknowledge that automated symbolification can sometimes undermine interpretability if forced on nonlinear anomalies.)

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---

# Article record: KASPER: Kolmogorov Arnold Networks for Stock Prediction and Explainable Regimes

## Metadata
- **BibTeX key**: oad_kasper_2025
- **Authors**: Vidhi Oad, Param Pathak, Nouhaila Innan, Shalini D, and Muhammad Shafique
- **Year**: 2025
- **Title**: KASPER: Kolmogorov Arnold Networks for Stock Prediction and Explainable Regimes
- **Journal/Publication**: arXiv preprint arXiv:2507.18983v1
- **DOI/URL**: https://arxiv.org/abs/2507.18983
- **Keywords from article**: Stock prediction, Regime detection, Kolmogorov-Arnold Networks (KAN), Explainable AI (XAI), Time-series forecasting

## 1. Study objective(s)
To introduce KASPER, a novel framework that addresses the nonlinear and regime-dependent dynamics of financial markets by integrating unsupervised regime detection with sparse, spline-based Kolmogorov-Arnold Networks (KANs) to achieve both high predictive accuracy and explicit symbolic interpretability.

## 2. Research question / Hypothesis
Can a framework that dynamically detects hidden market regimes (using a Gumbel-Softmax mechanism) and subsequently routes the data to regime-specific Kolmogorov-Arnold Networks (KANs) improve stock price forecasting accuracy while providing transparent, symbolic mathematical rules for each distinct market condition?

## 3. Methodology
- **Study design**: Empirical quantitative evaluation of a novel neural network architecture for regime-aware financial time-series forecasting.
- **Sample / corpus**: Daily stock data (Open, High, Low, Close, Volume) for four major technology companies: Apple (AAPL), Amazon (AMZN), Google (GOOGL), and Microsoft (MSFT). The dataset covers the period from 2010 to 2023, comprising over 3,000 trading days per stock.
- **Instruments / materials**: The KASPER framework (comprising an LSTM/GRU encoder, a Gumbel-Softmax bottleneck for regime detection, and regime-specific KAN sub-networks). Baselines included LSTM, GRU, Temporal Convolutional Networks (TCN), Transformers, and standard KANs.
- **Procedures**:
  1. Preprocessed the historical OHLCV data using Min-Max scaling and created rolling sequences with a lookback window of 60 days to predict the next day's closing price.
  2. Split the dataset chronologically into training (70%), validation (15%), and testing (15%) sets to preserve temporal order.
  3. Passed the input sequences through an encoder to extract temporal embeddings, which were then mapped to a discrete probability distribution over $K$ regimes (e.g., $K=3$ for bull, bear, and stagnant) using the Gumbel-Softmax trick.
  4. Routed the data to one of the $K$ distinct KAN sub-networks based on the identified regime.
  5. Trained the entire model end-to-end using a joint loss function that combined the Mean Squared Error (MSE) for forecasting with a regime regularization term (to encourage discrete, balanced cluster assignments).
  6. Evaluated forecasting performance against the baselines using standard error metrics.
  7. Applied sparsification, pruning, and symbolic regression to the trained KANs to extract explicit mathematical formulas representing the decision logic for each market regime.
- **Data analysis**: Forecasting accuracy evaluated via Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). Model complexity evaluated via parameter count. Interpretability evaluated by extracting symbolic formulas and analyzing the $R^2$ score of the fit.

## 4. Main results (only facts from the article)
- KASPER significantly outperformed all baseline models across all four tested stocks. For example, on AAPL, KASPER achieved an RMSE of 1.42, massively outperforming the standard KAN (2.15), Transformer (1.85), and LSTM (2.50).
- The standard KAN model (without the regime detection module) performed relatively poorly on the highly non-stationary financial data, demonstrating that explicitly decoupling the data into distinct market regimes is critical for the splines to learn effectively.
- The regime detection module successfully segmented the market into distinct, interpretable clusters corresponding to high volatility (bear markets), steady growth (bull markets), and sideways movement (stagnant markets) without requiring any prior manual labeling.
- The symbolification process successfully extracted highly accurate ($R^2 > 0.95$) mathematical formulas (e.g., combinations of sine, cosine, and polynomial functions) for each specific regime, detailing exactly how the model weighted features under different market conditions.
- KASPER operated with high parameter efficiency, requiring only 124K parameters, which is 62% fewer than the Transformer model (325K) and roughly equivalent to the LSTM (122K), while delivering far superior accuracy.

## 5. Relevant direct quotations (max 3)
- "To address this, we introduce Kolmogorov-Arnold networks for stock prediction and explainable regimes (KASPER), a novel framework that integrates regime detection, sparse spline-based function modeling, and symbolic rule extraction." (p. 1)
- "KASPER consistently achieved the lowest error metrics across all datasets. On AAPL, KASPER achieved an RMSE of 1.42, outperforming the baseline KAN (2.15) and Transformer (1.85)." (p. 4)
- "The results demonstrate that KASPER's integration of regime detection and KAN-based modeling provides a powerful tool for financial forecasting, bridging the gap between high predictive accuracy and model transparency." (p. 5)

## 6. Authors' conclusions
The authors conclude that KASPER effectively resolves the long-standing trade-off between predictive accuracy and interpretability in financial forecasting. By coupling an unsupervised Gumbel-Softmax regime detector with Kolmogorov-Arnold Networks, the model dynamically adapts to shifting market conditions and processes them using specialized, interpretable spline networks. This architecture not only outperforms complex models like Transformers and standard KANs in predictive accuracy but also yields explicit mathematical rules for each market state. This offers unparalleled transparency, making it a highly reliable and trustworthy tool for algorithmic trading and financial decision-making.

## 7. Limitations acknowledged by the authors
- Training KASPER requires longer computational times compared to standard LSTMs or MLPs because the B-spline operations within the KAN layers currently lack specialized hardware-level optimizations on GPUs.
- The interpretability phase (symbolic extraction) relies on predefined libraries of candidate functions and manual tuning of pruning thresholds, meaning the process is not fully automated and requires some human intervention.

## 8. Relevance to my thesis (leave empty)

## 9. Critical notes / questions raised (leave empty)

---