# Article record: KAN: Kolmogorov-Arnold Networks

## Metadata
- **BibTeX key**: liu_kan_2024
- **Authors**: Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark
- **Year**: 2024 (ArXiv v5: 9 Feb 2025)
- **Title**: KAN: Kolmogorov-Arnold Networks
- **Journal/Conference**: Accepted by International Conference on Learning Representations (ICLR) 2025 (arXiv preprint arXiv:2404.19756)
- **DOI/URL**: arXiv:2404.19756v5 [cs.LG]
- **Keywords**: [not found]

## 1. Theoretical foundation

**Key theorems:**

- **Kolmogorov-Arnold Representation Theorem**: If $f$ is a multivariate continuous function on a bounded domain, it can be written as a finite composition of continuous functions of a single variable and addition. Specifically, for a smooth $f:[0,1]^n \rightarrow \mathbb{R}$: $f(x) = f(x_1,...,x_n) = \sum_{q=1}^{2n+1} \Phi_q (\sum_{p=1}^n \phi_{q,p}(x_p))$, where $\phi_{q,p}:[0,1] \rightarrow \mathbb{R}$ and $\Phi_q: \mathbb{R} \rightarrow \mathbb{R}$.
- **Theorem 2.1 (Approximation theory, KAT)**: Let $x=(x_1,x_2,...,x_n)$. Suppose that a function $f(x)$ admits a representation $f = (\Phi_{L-1} \circ \Phi_{L-2} \circ ... \circ \Phi_1 \circ \Phi_0)x$, where each $\Phi_{l,i,j}$ is $(k+1)$-times continuously differentiable. Then there exists a constant $C$ depending on $f$ and its representation such that there exist $k$-th order B-spline functions $\Phi_{l,i,j}^G$ such that for any $0 \le m \le k$ we have the bound: $||f - (\Phi_{L-1}^G \circ \Phi_{L-2}^G \circ ... \circ \Phi_1^G \circ \Phi_0^G)x||_{C^m} \le C G^{-k-1+m}$.

**Key mathematical definitions:**

- **KAN layer**: A matrix of 1D functions $\Phi = \{\phi_{q,p}\}$, where $p=1,2,...,n_{in}$ and $q=1,2,...,n_{out}$.
- **$C^m$-norm**: Measures the magnitude of derivatives up to order $m$: $||g||_{C^m} = \max_{|\beta| \le m} \sup_{x \in [0,1]^n} |D^\beta g(x)|$.

**Key lemmas or propositions:**

- KANs possess a theoretical neural scaling exponent of $\alpha = k+1$. For cubic splines ($k=3$), $\alpha = 4$, yielding the scaling law $l \propto N^{-4}$.

**Relevant mathematical proofs (summary):**

- **Proof of Theorem 2.1**: Relies on classical 1D B-spline theory. Since $\Phi_{l,i,j}$ are continuous, they can be uniformly bounded on a bounded domain. 1D B-spline approximations $\Phi_{l,i,j}^G$ can bound the error at each layer by $C G^{-k-1+m}$. The residue $R_l$ for each layer is similarly bounded, and the total network error is the sum of layer-wise residues $R_{L-1} + R_{L-2} + ... + R_0$, which maintains the asymptotic rate $C G^{-k-1+m}$ completely independent of the input dimension $n$, proving it theoretically beats the curse of dimensionality.

## 2. Proposed method / architecture / model

**Name of method/model**: Kolmogorov-Arnold Networks (KANs). Also discusses Learnable Activation Networks (LANs) in the Appendix.

**Core mathematical formulation:**

- Node activation computation: $x_{l+1, j} = \sum_{i=1}^{n_l} \tilde{x}_{l,j,i} = \sum_{i=1}^{n_l} \phi_{l,j,i}(x_{l,i})$.
- Matrix form of a layer: $x_{l+1} = \Phi_l x_l$, where $\Phi_l$ is the function matrix.
- Full KAN network: $KAN(x) = (\Phi_{L-1} \circ \Phi_{L-2} \circ ... \circ \Phi_1 \circ \Phi_0)x$.
- Residual activation function on edges: $\phi(x) = w_b b(x) + w_s \text{spline}(x)$.
- Basis function: $b(x) = \text{silu}(x) = x/(1+e^{-x})$.
- Spline parameterization: $\text{spline}(x) = \sum_i c_i B_i(x)$.
- Sparsification Loss (L1): $|\phi|_1 \equiv \frac{1}{N_p} \sum_{s=1}^{N_p} |\phi(x^{(s)})|$ and layer norm $|\Phi|_1 \equiv \sum_{i=1}^{n_{in}} \sum_{j=1}^{n_{out}} |\phi_{i,j}|_1$.
- Sparsification Loss (Entropy): $S(\Phi) = -\sum_{i=1}^{n_{in}} \sum_{j=1}^{n_{out}} \frac{|\phi_{i,j}|_1}{|\Phi|_1} \log(\frac{|\phi_{i,j}|_1}{|\Phi|_1})$.
- Total Objective: $l_{total} = l_{pred} + \lambda(\mu_1 \sum_{l=0}^{L-1} |\Phi_l|_1 + \mu_2 \sum_{l=0}^{L-1} S(\Phi_l))$.
- Pruning Scores: Incoming score $I_{l,i} = \max_k (|\phi_{l-1, i, k}|_1)$, Outgoing score $O_{l,i} = \max_j (|\phi_{l+1, j, i}|_1)$.
- Grid Extension (Least Squares): $\{c_j^\prime\} = \arg\min_{\{c_j^\prime\}} \mathbb{E}_{x \sim p(x)} (\sum_{j=0}^{G_2+k-1} c_j^\prime B_j^\prime(x) - \sum_{i=0}^{G_1+k-1} c_i B_i(x))^2$.

**Algorithm pseudocode (Interactive Symbolification Pipeline):**

```
1. Train KAN with sparsification regularization (L1 + Entropy).
2. Prune unimportant nodes (score < theta).
3. Auto/Manual Symbolic Snapping (suggest/fix functions like sin, exp, x^2 via fix_symbolic or auto_symbolic).
4. Retrain affine parameters to machine precision.
5. Output final symbolic formula (using sympy).
```

**Architecture details:**

- KANs completely eliminate linear weight matrices. All parameters are stored inside 1D spline functions located on the network edges.
- Nodes perform simple addition (no non-linearities applied at the node).
- Initialization: $w_s = 1$, $c_i \sim \mathcal{N}(0, 0.1^2)$, $w_b$ initialized via Xavier.
- Number of parameters for width $N$, depth $L$, grid intervals $G$, spline order $k$: $O(N^2 L(G+k)) \sim O(N^2 L G)$.
- Grid points dynamically updated on the fly based on input activation distributions to keep splines within bounds.

**Innovative components:**

- Shifting learnable activations from nodes to edges.
- Overcoming the Curse of Dimensionality (COD) by relying strictly on 1D univariate spline approximations combined with learned compositional routing.
- Grid extension: fine-graining the spline grids mid-training (using least squares) to increase accuracy without retraining the model from scratch.
- Human-interactive symbolic regression: allowing continuous gradient-descent search to find representations, followed by discrete affine-snapping to exact algebraic formulas.

## 3. Data

**Dataset name(s):**

- Synthetic Toy Datasets (5 mathematical functions).
- Special Functions Dataset (15 math/physics functions, e.g., Bessel, Jacobian elliptic, Spherical Harmonics).
- Feynman Dataset (Feynman_no_units).
- Partial Differential Equation (Poisson equation setup).
- Continual Learning Toy Dataset (1D regression with 5 Gaussian peaks).
- Knot Theory Dataset (Supervised & Unsupervised topological invariants).
- Anderson Localization Datasets (Mosaic Model [MM], Generalized Aubry-André Model [GAAM], Modified Aubry-André Model [MAAM]).

**Data source**: Synthetic (generated natively for functions, PDEs, and Hamiltonians); Public (Feynman equations, Knot dataset from DeepMind).

**Data size:**

- Function fitting: $N_p=1000$ training samples.
- PDE solving: $n_i = 10000$ interior points, $n_b = 800$ boundary points.

**Data preprocessing:**

- Unsupervised Knot Theory: Feature corruption generated by random permutation of each feature across the entire training set to create negative samples ($y=0$), keeping true samples as positive ($y=1$).

**Train/validation/test split**: [not found - exact splits not explicitly detailed, only references to "train" and "test" sets/losses].

**Data augmentation techniques**: Negative sampling via random feature permutation for contrastive-style unsupervised learning.

## 4. Experimental setup

**Baseline models compared against:**

- Multi-Layer Perceptrons (MLPs) with depths $\{2,3,4,5,6\}$, widths $\{5, 100\}$, and activations {Tanh, ReLU, SiLU}.
- DeepMind's MLP (4 layers, width 300) for Knot signature prediction.
- Learnable Activation Networks (LANs) (evaluated in Appendix B).
- SIREN (Implicit neural representation with periodic activation functions, Appendix B).

**Evaluation metrics:**

- RMSE (Root Mean Square Error) for regression tasks.
- Test Loss / Training Loss.
- $L^2$ error squared and $H^1$ error squared for PDE solving.
- Accuracy (%) for classification (e.g., Knot signature, Anderson mobility edge).
- Parameter Count ($N$) and theoretical GFLOPs scaling.

**Hyperparameters:**

- Spline order: $k=3$ (cubic splines).
- Spline noise initialization: $\sigma = 0.1$.
- Grid points $G$: Swept across $\{3, 5, 10, 20, 50, 100, 200, 500, 1000\}$; typically updated/extended every 200 L-BFGS steps.
- Optimizer: L-BFGS (for KANs and baseline MLPs in function fitting). Adam (used for image fitting in Appendix B, batch size 4096, LR $10^{-3}$ then $10^{-4}$).
- Sparsification penalties: $\mu_1=1, \mu_2=1$, with $\lambda = 10^{-2}$ or $10^{-3}$.
- Pruning threshold: $\theta = 10^{-2}$.
- PDE PINN hyperparameter: $\alpha = 0.01$.

**Hardware/software environment**: PyTorch (implemented as pykan), SymPy (for symbolic formula extraction).

**Number of runs / seeds:**

- 3 random seeds per hyperparameter combination for Pareto frontier evaluations.
- 100 random seeds (0 to 99) for Unsupervised Knot Theory relation discovery.

**Statistical tests performed**: [not found]

## 5. Results

**Quantitative results:**

- **Neural Scaling Laws**: KANs achieved $\alpha \approx 4$ (test $RMSE \propto G^{-4}$ when evaluated via sqrt median squared loss), matching the theoretical optimum. MLPs hit a plateau and scaled at $\approx N^{-1}$.
- **PDE Solving (Poisson)**: A 2-Layer width-10 KAN ($10^2$ params) achieved $10^{-7}$ MSE. A 4-Layer width-100 MLP ($10^4$ params) achieved $10^{-5}$ MSE. (KAN 100x more accurate with 100x fewer parameters).
- **Knot Theory Signature Classification (Table 3)**:
  - DeepMind MLP: $3 \times 10^5$ params, 78.0% accuracy.
  - KAN [17, 1, 14]: $2 \times 10^2$ params, 81.6% accuracy.
  - Ablation on importance: $\mu_r$ alone yielded 65.0% accuracy; $\lambda$ alone 77.8%; $\mu_i$ alone 43.8%.
- **Knot Theory Symbolic Formulas (Table 4)**:
  - Formula A (Human DM): 83.1% acc.
  - Formula D (KAN [3,2,1]): 84.0% acc.
  - Formula E (KAN Pade approx): 82.8% acc.
  - Formula F (KAN/KAN division trick on just $\mu_r, \lambda$): 77.8% acc.
- **Anderson Localization GAAM (Table 5)**:
  - Theory: $\alpha E + 2\lambda - 2 = 0$ (99.2% acc).
  - KAN auto formula: $1.62E^2 + ... + 45.13\lambda - 54.45 = 0$ (99.0% acc).
- **Anderson Localization MAAM (Table 5)**:
  - Theory: $E + \exp(p) - \lambda \cosh(p) = 0$ (98.6% acc).
  - KAN auto: 97.1% acc.
  - KAN manual step 4A (snap p to cosh): 96.6% acc.
- **Special Functions Pareto Frontier**: KAN consistently outperforms MLPs across all 15 functions. Examples:
  - ellipj: Minimal KAN [2,2,1] RMSE $7.29 \times 10^{-3}$; Best KAN [2,2,1,1,1] $1.33 \times 10^{-4}$; MLP $6.48 \times 10^{-4}$.
  - jv: Minimal KAN [2,2,1] RMSE $4.92 \times 10^{-3}$; Best KAN [2,3,1,1] $1.64 \times 10^{-5}$; MLP $5.32 \times 10^{-3}$.
  - sph_harm_m_0_n_1: Minimal KAN [2,1,1] $2.21 \times 10^{-7}$; Best KAN [2,1,1] $2.21 \times 10^{-7}$; MLP $1.25 \times 10^{-4}$.
- **Feynman Dataset**: KAN auto-discovered formulas frequently required fewer layers than human-constructed KANs. E.g., relativistic velocity ($f(u,v)=\frac{u+v}{1+uv}$) human-constructed required 5 layers; KAN auto-discovered a 2-layer [2,1,1] shape by implicitly discovering the "rapidity" trick ($\tanh(\text{arctanh } u + \text{arctanh } v)$).

**Qualitative results:**

- **Continual Learning**: On the 5 Gaussian peak task, KANs perfectly avoided catastrophic forgetting. Because splines are local, new data updates only local coefficients. MLPs (using global activations) exhibited severe catastrophic forgetting, destroying previous peaks when learning new ones.
- **Unsupervised Discovery**: KANs implicitly discovered three relations in knot invariants without human prompting: (1) Signature dependence on $\mu_r$ and $\lambda$; (2) Cusp volume definition $V = \mu_r \lambda$; (3) Injectivity radius bounds ($2r \ge g_r$).
- **Visual Interpretability**: Adjusting visual edge transparency proportional to $\tanh(3 |\phi|_1)$ allows users to immediately "see" which variables matter and exactly what algebraic shape the neural connection has taken.

**Ablation studies:**

- **Hyperparameters (Appendix C)**:
  - Entropy regularization is absolutely critical; L1 alone fails to remove duplicate/redundant spline functions.
  - Grid size $G$: If $G$ is too small (e.g., $G=2$), individual splines lack expressivity, forcing the KAN to compensate by utilizing redundant pathways (ensembling), which destroys interpretability.
- **Learnable Activation Networks (LANs, Appendix B)**: Tested architectures where nodes have learnable splines, but edges remain linear weights. While LANs achieved higher PSNR than standard MLPs (e.g., fitting SIREN images), they proved far less interpretable than KANs because the linear weight matrices retained too many unconstrained degrees of freedom.

**Comparison with baselines:**

- KANs saturated the $N^{-4}$ scaling law, beating MLPs ($N^{-1}$) consistently across all synthetic, physics, and mathematical tasks.
- KANs achieved higher test accuracy with dramatically fewer parameters (e.g., 200 params vs 300,000 params on Knot theory dataset).

## 6. Direct quotations

- "Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs)." (p. 1)
- "While MLPs have fixed activation functions on nodes ("neurons"), KANs have learnable activation functions on edges ("weights"). KANs have no linear weights at all every weight parameter is replaced by a univariate function parametrized as a spline." (p. 1)
- "We know that asymptotically, provided that the assumption in Theorem 2.1 holds, KANs with finite grid size can approximate the function well with a residue rate independent of the dimension, hence beating curse of dimensionality!" (p. 8)
- "KANs have local plasticity and can avoid catastrophic forgetting by leveraging the locality of splines." (p. 19)
- "It is remarkable that KANs can be both more accurate and much more parameter efficient than MLPs at the same time." (p. 23)

## 7. Authors' conclusions

The authors propose Kolmogorov-Arnold Networks (KANs) as a fundamental architectural paradigm shift away from Multi-Layer Perceptrons. By eliminating fixed node activations and linear weight matrices in favor of placing learnable 1D B-splines on the network edges, KANs directly exploit the compositional structure of data to overcome the curse of dimensionality. This structural change results in highly favorable neural scaling laws (scaling at $N^{-4}$) and allows KANs to achieve equal or superior accuracy to MLPs while using orders of magnitude fewer parameters. Furthermore, KANs are intrinsically interpretable. Through a pipeline of sparsification, pruning, and symbolic snapping, humans can visually interact with the network to extract exact closed-form algebraic formulas governing physical and mathematical systems. The authors conclude that while current implementations face computational speed bottlenecks, KANs serve as an ideal foundation model for "AI + Science", shifting the paradigm from black-box automated oracles toward transparent, collaborative tools that assist scientists in discovering natural laws.

## 8. Limitations stated by authors

- **Slow training speeds**: KANs are typically 10x slower to train than MLPs with the same number of parameters. Because every edge evaluates a distinct spline function, KANs cannot currently leverage the highly optimized batched matrix multiplication operations that make MLPs fast on modern GPUs.
- **Theoretical gaps**: The empirical success of "deep" KANs relies on representations that stretch beyond the original depth-2 Kolmogorov-Arnold theorem. The mathematical notion of a "Kolmogorov-Arnold depth" is not yet formally characterized.
- **Suboptimal unsupervised relation discovery**: The current unsupervised mode relies heavily on random seeds to find different variable relations. It lacks a systematic, controlled method to guarantee the discovery of a complete set of relations.
- **Optimization hurdles with fine grids**: When the grid size $G$ is extended to a very large level (comparable to the number of data points), training loss can suddenly spike or cease to optimize, likely due to bad loss landscapes or boundary effects.

## 9. Future work suggested by authors

- Investigating the exact dependence of the approximation constant $C$ (from Theorem 2.1) on the input dimension $n$.
- Exploring the capability of KANs to scale to massive, standard machine learning tasks like language modeling (e.g., creating "kansformers").
- Integrating KANs with state-of-the-art continual learning techniques to test if spline locality prevents catastrophic forgetting in highly complex, high-dimensional realistic setups.
- Developing systematic, non-random methods for fully uncovering all structural relations in unsupervised datasets.
- Replacing MLPs with KANs in established PDE and operator learning frameworks, such as Physics-Informed Neural Networks (PINNs), Fourier Neural Operators (FNOs), and DeepONets.
- Improving computational efficiency by introducing "multi-head" activations (where groups of edges share the same spline parameters to enable batching) or by implementing adaptive grid strategies and domain-dependent multiscale basis functions.

## Additional critical details

- **Initialization trick**: Spline coefficients $c_i$ are drawn from a normal distribution $\mathcal{N}(0, \sigma^2)$ with a very small variance ($\sigma=0.1$) to ensure the activation function starts almost flat (close to zero). The baseline transformation $b(x)$ relies on the silu function scaled by $w_b$ (initialized via Xavier), ensuring stability early in training.
- **Interpolation Threshold**: The authors experimentally observed a "U-shape" test loss curve during grid extension, corresponding to the bias-variance tradeoff. They conjecture that the optimal test loss occurs exactly at the interpolation threshold, where the total number of spline parameters matches the total number of training data points (e.g., $15G \approx 1000 \rightarrow G \approx 67$).
- **Visual Scaling**: The visualization tool provided in pykan renders the network transparently based on connection importance. Specifically, transparency scales with $\tanh(\beta A_{l,i,j})$ where $\beta=3$ and $A_{l,i,j}$ is the L1 norm of the activation, cleanly fading out inactive splines.

---
**Extraction date**: 2026-05-19
**PDF source**: KAN.pdf

---

# Article record: KAN 2.0: Kolmogorov-Arnold Networks Meet Science

## Metadata
- **BibTeX key**: liu_kan2_2024
- **Authors**: Ziming Liu, Pingchuan Ma, Yixuan Wang, Wojciech Matusik, Max Tegmark
- **Year**: 2024
- **Title**: KAN 2.0: Kolmogorov-Arnold Networks Meet Science
- **Journal/Conference**: arXiv preprint arXiv:2408.10205v1 [cs.LG]
- **DOI/URL**: arXiv:2408.10205v1
- **Keywords**: [not found]

## 1. Theoretical foundation

**Key theorems:**

- **Kolmogorov-Arnold Representation Theorem (KART)**: Any continuous high-dimensional function can be decomposed into a finite composition of univariate continuous functions and additions: $f(x) = f(x_1, ..., x_n) = \sum_{q=1}^{2n+1} \Phi_q (\sum_{p=1}^n \phi_{q,p}(x_p))$.

**Key mathematical definitions:**

- **Additive Separability**: A function $f$ is additively separable if $f(x_1, x_2, ... x_n) = g(x_1, ..., x_k) + h(x_{k+1}, ..., x_n)$. It is detected by verifying that the Hessian matrix elements $\frac{\partial^2 f}{\partial x_i \partial x_j} = 0$ for all $1 \le i \le k$ and $k+1 \le j \le n$.
- **Multiplicative Separability**: Defined by applying the logarithm to convert to additive separability. Detected by verifying the block structure of $H_{ij} \equiv \frac{\partial^2 \log|f|}{\partial x_i \partial x_j} = 0$.
- **Generalized Separability**: A function $f$ has generalized separability if $f(x_1, x_2, ... x_n) = F(g(x_1, ..., x_k) + h(x_{k+1}, ..., x_n))$. Detected by evaluating $\frac{\partial f / \partial x_i}{\partial f / \partial x_j} = g_{x_i}(x_1, ..., x_k) \times \frac{1}{h_{x_j}(x_{k+1}, ..., x_n)}$.
- **Generalized Symmetry**: A function $f$ has generalized symmetry if $f(x_1, x_2, ..., x_n) = g(h(x_1, ..., x_k), x_{k+1}, ..., x_n)$. Detected by verifying that the direction of the gradient $\hat{\nabla_y f} \equiv \frac{\nabla_y f}{|\nabla_y f|}$ is independent of $z$, such that $\nabla_z(\hat{\nabla_y f}) = 0$.

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: MultKAN (Multiplicative Kolmogorov-Arnold Networks), kanpiler (KAN Compiler), and tree converter.

**Core mathematical formulation:**

- MultKAN Layer Transformation: $M_l(z_l) = \text{concatenate}(z_l[:n_{l+1}^a], z_l[n_{l+1}^a::2] \odot z_l[n_{l+1}^a+1::2]) \in \mathbb{R}^{n_{l+1}^a+n_{l+1}^m}$, where $\odot$ is element-wise multiplication.
- MultKAN Layer: $\Psi_l \equiv M_l \circ \Phi_l$.
- Whole MultKAN Network: $MultKAN(x) = (\Psi_L \circ \Psi_{L-1} \circ ... \circ \Psi_1 \circ \Psi_0)x$.
- Attribution/Importance Score (Backward Computation):
  - $B_{l-1,i,j} = A_{l,j} \frac{E_{l,i,j}}{\sum_{k=0}^{n_l-1} E_{l,k,j}}$
  - $A_{l-1,i} = \sum_{j=0}^{n_l-1} B_{l-1,i,j}$ for $l=L, L-1, ..., 1$, initialized with $A_{L,i}=1$.

**Algorithm pseudocode:**

```
kanpiler compilation:
  1. Parse symbolic formula into an expression tree (nodes represent expressions, edges denote operations/functions).
  2. Modify tree to align with KAN graph structure (move leaf nodes to input layer via dummy edges, add dummy subnodes/nodes).
  3. Combine variables in the first layer to convert the tree into a valid KAN graph.

Symbolic Extraction Pipeline:
  1. Train a general network and probe its modularity.
  2. Build the identified modular structure into a new KAN (inductive bias).
  3. Initialize connections sparsely (sparse_init=True).
  4. Train, prune irrelevant edges via attribution scores, and extract symbolic formulas.
  5. Use Hypothesis Testing (model.rewind()) to revert and branch different topological/symbolic assumptions.
```

**Architecture details:**

- The MultKAN width is defined by two arrays: addition width $n^a \equiv [n_0^a, n_1^a, ..., n_L^a]$ and multiplication width $n^m \equiv [n_0^m, n_1^m, ..., n_L^m]$.
- Multiplication nodes perform multiplication on $k$ subnodes from the previous layer (default $k=2$).
- Zero trainable parameters exist in the multiplication layers $M_l$, allowing standard $l_1$ and entropy regularizations to be seamlessly applied.

**Innovative components:**

- The explicit inclusion of multiplicative nodes to natively model multiplicative physical laws (which standard additive KANs struggle to model cleanly).
- Introduction of the kanpiler to translate pure symbolic mathematical expressions into initialized neural network graphs with frozen/trainable parameters.
- Transition from forward-calculated L1 function norms to backward-calculated attribution scores to accurately prune networks based on global feature contribution.

## 3. Data

**Dataset name(s):**

- Feynman dataset (120 physical equations).
- Synthetic relational datasets: Multitask sparse parity, Hierarchical majority voting.
- Single pendulum Lagrangian dataset.
- Relativistic mass in a uniform field Lagrangian dataset.
- Schwarzschild black hole metric transformation dataset.
- Neo-Hookean material constitutive law dataset.

**Data source**: Synthetic datasets generated programmatically from known physical/mathematical equations.

**Data size**: [not found - explicit integer sample sizes for datasets are not detailed in the text]

**Data preprocessing:**

- For the Neo-Hookean material task, the deformation tensor matrix elements $F_{ij}$ were independently sampled from a uniform distribution $U[\delta_{ij} - w, \delta_{ij} + w]$ with $w=0.2$.

**Train/validation/test split**: [not found]

**Data augmentation techniques:**

- Feature augmentation via auxiliary variables: Manually augmenting the input data array by injecting physical combinations of base inputs (e.g., adding $\beta \equiv v/c$ and $\gamma \equiv 1/\sqrt{1 - (v/c)^2}$ for relativistic mass; adding the determinant of $F$ for the constitutive laws).

## 4. Experimental setup

**Baseline models compared against**: Standard Multi-Layer Perceptrons (MLPs).

**Evaluation metrics:**

- Root Mean Square Error (RMSE) for training and test loss.
- Minkowski Loss.
- $R^2$ (R-squared fitness) for symbolic regression matching.

**Hyperparameters:**

- Elasticity parameters for Neo-Hookean materials: Young's modulus $Y=1$, Poisson ratio $\nu=0.2$, yielding Lamé parameters $\mu = 5/12 \approx 0.42$ and $\lambda = 5/18 \approx 0.28$.

**Hardware/software environment:**

- Implementation in Python via the pykan library (v0.2.x).
- Supported on both CPU and GPU (e.g., training a [4,100,100,100,1] KAN for 100 steps takes 20s on CPU vs <1s on GPU).
- SymPy used for symbolic regression and expression tree parsing.

**Number of runs / seeds**: Explicit variations over random seeds were used to test multiple hypotheses (e.g., Seed 0 vs. Seed 1 for relativistic mass testing).

**Statistical tests performed**: Hypothesis testing within network branches using model.rewind() to compare test RMSE between structurally different network versions.

## 5. Results

**Quantitative results:**

- **Relativistic Mass (with auxiliary variables)**: Seed 0 achieved $10^{-6}$ loss; Seed 1 achieved $10^{-4}$ loss.
- **Schwarzschild Black Hole Symmetry**:
  - MLP baseline achieved a smooth global solution with $10^{-8}$ loss.
  - Standard KAN got stuck in a domain-wall local minimum with a $10^{-3}$ loss spike.
  - Supervised KAN (fine-tuned using the MLP as a teacher with 1000 grid intervals) achieved $10^{-15}$ loss (machine precision).
- **Constitutive Law ($P_{11}$)**:
  - Step 1 (Compile linear): Loss $10^{-2}$.
  - Step 3 (Perturb & Train): Loss $6\times10^{-3}$.
  - Step 5 (Add determinant auxiliary & Train): Loss $2\times10^{-4}$.
  - Step 6 (Symbolify formula): Loss $3\times10^{-11}$.
- **Constitutive Law ($P_{12}$)**:
  - With prior knowledge compilation: Symbolification resulted in loss $7\times10^{-3}$.
  - Without prior knowledge (random init): Symbolification resulted in loss $6\times10^{-9}$.

**Qualitative results:**

- **Anatomical Modularity**: The auto_swap method successfully sorted neural topologies. For the Multitask sparse parity, auto_swap successfully identified clear visual modules for both KANs and MLPs. For Hierarchical majority voting, KANs revealed clear structure, but MLPs failed to show global modular structure even after swapping.
- **Feature Attribution vs L1 Norm**: On $y = \exp(\sin(\pi x_1) + x_2^2)$ and symmetric equations, the backward-computed attribution score properly pruned irrelevant paths and accurately reflected equal variable importance, whereas the forward L1 norm produced messy, incorrect hierarchical feature importance.
- **Lagrangian Discovery**: For a relativistic mass in a uniform field, KAN symbolic regression successfully isolated $V(q) = q$. For the kinetic component, $x^{1/2}$ provided a slightly better empirical fit ($R^2=0.999949$) than the theoretical truth $x^{-1/2}$ ($R^2=0.999168$), highlighting the necessity of human physical priors (divergence at $v \to c$) to select the correct symbolic function over raw data-fitting.

**Ablation studies:**

- **Prior Knowledge Inductive Bias**: Tested the effect of injecting the linear constitutive law as a structural prior versus purely random initialization on predicting the $P_{12}$ tensor. Random initialization converged to the exact Neo-Hookean mathematical form ($6\times10^{-9}$ loss). Incorporating the prior trapped the network in a sub-optimal local minimum with minor correction terms ($7\times10^{-3}$ loss), showing that incorrect structural priors can be harmful.

**Comparison with baselines:**

- On the transformation to translation-invariant spatial sections (Schwarzschild metric), KANs eventually out-scaled MLPs in sheer precision ($10^{-15}$ vs $10^{-8}$), proving their theoretical capability to reach machine precision.
- MultKAN accurately modeled $f(x,y)=xy$ using exactly 1 multiplication node, solving the structural ambiguity where a standard KAN required 2 complex addition nodes.

## 6. Direct quotations

- "To bridge the two worlds, we propose a framework to seamlessly synergize Kolmogorov-Arnold Networks (KANs) and science. The framework highlights KANs' usage for three aspects of scientific discovery: identifying relevant features, revealing modular structures, and discovering symbolic formulas." (p. 1)
- "This implies that addition is the only true multivariate operation, while other multivariate operations (including multiplication) can be expressed as additions combined with univariate functions." (p. 3)
- "Comparing the two scenarios, one with and one without prior knowledge, reveals a surprising outcome: in this example, prior knowledge appears harmful, possibly because the linear constitutive law is probably near a (bad) local minimum which is hard for the model to escape." (p. 19)
- "In Figure 14 (a) visualizes the position of software 1.0, software 2.0, and KANs on the interpretability-learnability plane, illustrating how KANs can balance the trade-offs between these two paradigms." (p. 21)
- "We hope that KANs will evolve into universal foundation models for physical discoveries." (p. 20)

## 7. Authors' conclusions

The authors conclude that "KAN 2.0" successfully bridges the paradigm gap between connectionist AI (Software 2.0) and symbolic scientific computation (Software 1.0). By introducing Multiplicative Kolmogorov-Arnold Networks (MultKANs), the kanpiler compiler, and the tree converter, the framework allows for bidirectional synergy: scientists can encode known physical rules directly into network topologies as inductive biases, and conversely extract previously unknown modular structures and symbolic physical laws directly from raw data. Through extensive empirical validations on conserved quantities, Lagrangians, and constitutive laws, the authors demonstrate that KANs are not merely black-box predictive oracles, but highly interactive and interpretable tools capable of facilitating curiosity-driven scientific discoveries.

## 8. Limitations stated by authors

- **Efficiency**: KANs still lag significantly behind MLPs in computational efficiency and runtime, particularly at larger scales.
- **Interpretability Scaling**: The high degree of interpretability demonstrated is largely confined to small-scale networks. As networks scale up, managing and deciphering the combined output of numerous 1D functions becomes overwhelmingly difficult.
- **Local Minima from Priors**: Injecting prior physical knowledge into the network can sometimes trap the model in bad local minima, preventing it from discovering the true underlying formula.
- **Symbolic Regression Fragility**: Symbolic regression is sensitive to noise; algorithms may prioritize slightly over-fitted incorrect mathematical functions (e.g., $x^{1/2}$) over the true physical laws (e.g., $x^{-1/2}$), necessitating human oversight and domain knowledge.
- **Smoothness Requirements**: Modularity detection methods rely on evaluating second-order derivatives (Hessians). Adversarial or non-smooth data can easily break these modularity checks.

## 9. Future work suggested by authors

- Applying the KAN 2.0 framework to larger-scale problems to test the limits of its scalability.
- Extending the methodology to other scientific disciplines beyond physics (e.g., chemistry, biology).
- Developing more advanced interpretability methods that can push the current Pareto Frontiers of interpretability versus scale.

## Additional critical details

- **Software Optimization Mechanisms**: The pykan library was updated to dramatically improve efficiency via three specific mechanisms:
  1. Memory usage for spline evaluations was reduced from $O(LN^2G)$ to $O(LNG)$ by avoiding unnecessary input expansions.
  2. The highly unoptimized "symbolic branch" of the KAN layer was disabled by default to save processing time.
  3. Intermediate activations are now only cached when explicitly requested via model.speed(), preventing excessive memory bloat during standard training.
- **Tree Converter Hierarchy**: To extract modular structures from trained networks, the authors implemented a recursive application of generalized symmetry detection. It tests combinations of variable groups starting from $k=2$ all the way to $k=n$, generating explicit hierarchical parsing trees.

---
**Extraction date**: 2026-05-19
**PDF source**: arXiv:2408.10205

---

# Article record: A Practitioner's Guide to Kolmogorov-Arnold Networks

## Metadata
- **BibTeX key**: noorizadegan_2026
- **Authors**: Amir Noorizadegan, Sifan Wang, Leevan Ling, Juan P. Dominguez-Morales
- **Year**: 2026
- **Title**: A Practitioner's Guide to Kolmogorov-Arnold Networks
- **Journal/Conference**: arXiv preprint
- **DOI/URL**: arXiv:2510.25781v3 [cs.LG] / https://github.com/AmirNoori68/kan-review
- **Keywords**: Kolmogorov-Arnold Networks, KAN, Multi-Layer Perceptrons, Kolmogorov Superposition Theorem, Machine Learning, Deep Learning, B-splines

## 1. Theoretical foundation

**Key theorems:**

- **Kolmogorov Superposition Theorem (KST) (1957)**: Any multivariate continuous function $f:[0,1]^d \to \mathbb{R}$ can be represented as a finite composition of continuous functions of a single variable and addition: $f(\mathbf{x}) = \sum_{q=1}^{2d+1} \Phi_q \left( \sum_{p=1}^d \phi_{q,p}(x_p) \right)$, where $\phi_{q,p} : [0,1] \to \mathbb{R}$ and $\Phi_q : \mathbb{R} \to \mathbb{R}$.

**Key mathematical definitions:**

- **KAN Layer**: $\mathbf{x}_{l+1} = \Phi_l(\mathbf{x}_l)$, where $\Phi_l$ is a matrix of 1D functions $\phi_{l,i,j}$.
- **KAN Network**: $KAN(\mathbf{x}) = (\Phi_{L-1} \circ \dots \circ \Phi_1 \circ \Phi_0)(\mathbf{x})$.

**Key lemmas or propositions:**

- **Vitushkin's Construction & Girosi and Poggio's Critique (1989)**: Shows that the exact inner functions $\phi_{q,p}$ from KST are highly non-smooth (fractal-like and nowhere differentiable), rendering exact KST practically irrelevant for smooth function approximation in machine learning. Modern KANs bypass this by using smooth basis functions and deeper, narrower architectures, making them "KST-inspired" rather than strict implementations.

**Relevant mathematical proofs (summary)**: [not found - the paper is a review and methodological guide, focusing on synthesizing existing mathematical theory rather than proposing new proofs].

## 2. Proposed method / architecture / model

**Name of method/model**: The paper proposes a taxonomy and a "Choose-Your-KAN" algorithmic framework for practitioners.

**Core mathematical formulation:**

- Edge activation in standard B-spline KAN: $\phi(x) = w \left( b(x) + \text{spline}(x) \right)$, where $b(x)$ is a basis function (e.g., SiLU, $x/(1+e^{-x})$) and $\text{spline}(x) = \sum_{i=1}^G c_i B_i(x)$, with $c_i$ being learnable coefficients and $B_i(x)$ being B-spline basis functions over a grid of size $G$.

**Algorithm pseudocode (Algorithm 1: Choose Your KAN: A Practitioner's Guide):**

```
Step 1: Preprocess data (Normalize features to [-1, 1] or [0, 1] to prevent basis saturation).
Step 2: Select the core basis function based on problem type:
  - If data is smooth/global -> Use Chebyshev or Polynomial KANs.
  - If data is multiscale/heterogeneous -> Use Wavelet KAN or FBKAN.
  - If high-dimensional -> Use KKAN, SPIKAN, or KAF.
Step 3: Configure grid and basis resolution (Use coarse grids initially, refine adaptively).
Step 4: Add physics-informed constraints if applicable (Construct PINN/PIKAN loss).
Step 5: Choose optimization strategy (Warm up with Adam, refine with L-BFGS).
Step 6: Optimize for speed vs. accuracy:
  - If speed is primary -> Prefer ReLU-KAN, HRKAN, FastKAN, LeanKAN.
  - Else -> Prefer B-spline, Chebyshev/Jacobi, Wavelet KANs with adaptive grids.
```

**Architecture details:**

- In KANs, nodes simply sum incoming signals without applying non-linearities.
- All non-linearities (activation functions) are parameterized as 1D functions on the edges (weights).
- The review details the classification of edge functions into: Spline-based (B-splines, NURBS), Polynomial-based (Chebyshev, Legendre, Jacobi), Radial Basis Functions (FastKAN, Gaussian RBFs), Frequency/Wavelet-based (Fourier, Wav-KAN), and MLP-based approximations (ReLU-KAN).

**Innovative components:**

- The paper provides the first comprehensive taxonomic framework that treats the choice of basis function as a configurable design axis, unifying dozens of disparate KAN variants (FastKAN, Wav-KAN, ChebyKAN) into a single decision-tree methodology for practitioners.

## 3. Data

**Dataset name(s)**: [not found - this is a comprehensive literature review and practitioner's guide. It does not introduce new datasets or execute primary empirical studies.]

**Data source**: [not found]

**Data size**: [not found]

**Data preprocessing:**

- **Normalization**: Strongly emphasizes that input data must be normalized (e.g., via min-max scaling to $[-1, 1]$) because localized basis functions (like B-splines) and domain-restricted polynomials (like Chebyshev) will fail or saturate if inputs fall outside their defined grids.

**Train/validation/test split**: [not found]

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against**: Conceptually compares KANs against Multi-Layer Perceptrons (MLPs) and Classical Kernel Methods.

**Evaluation metrics**: Mentions that the reviewed literature evaluates models using RMSE, Parameter Count, and Inference Time (GFLOPs).

**Hyperparameters:**

- **Optimizers**: Recommends Adam/RAdam for initial warm-up, followed by L-BFGS or quasi-Newton methods for fine-tuning splines.
- **Grid Size ($G$)**: Recommends starting with coarse grids and adaptively refining (grid extension) rather than starting with a dense grid.
- **Regularization**: Mentions $L_1$ and entropy regularization used in KANs for sparsification.

**Hardware/software environment**: Highlights standard bottlenecks on GPUs. Recommends libraries with custom CUDA kernels or JAX implementations (e.g., efficient-kan) to mitigate the slow evaluation of independent edge splines.

**Number of runs / seeds**: [not found]

**Statistical tests performed**: [not found]

## 5. Results

**Quantitative results:**

- As a review paper, it cites the scaling laws established by Liu et al. (2024), specifically that KANs can achieve an empirical neural scaling law of $O(N^{-4})$ compared to the standard MLP scaling of $O(N^{-1})$. No independent quantitative experimental tables are presented.

**Qualitative results:**

- **Accuracy & Scaling**: KANs consistently demonstrate the ability to beat the curse of dimensionality by leveraging 1D univariate approximations, allowing smaller networks to outperform larger MLPs in function fitting.
- **Interpretability**: By applying sparsification and pruning, KANs can be "symbolified" (extracting exact mathematical formulas from the trained splines), making them highly transparent.
- **Catastrophic Forgetting**: Because B-splines have local support, KANs naturally mitigate catastrophic forgetting in continual learning; new data updates only local coefficients without destroying global representations.

**Ablation studies**: [not found]

**Comparison with baselines:**

- Synthesizes that KANs generally require far fewer parameters than MLPs to achieve equivalent accuracy. However, standard KANs are currently computationally slower (e.g., 10x slower training times) and memory-heavy due to the inability to leverage highly optimized dense matrix multiplications.

## 6. Direct quotations

- "Kolmogorov-Arnold Networks (KANs), whose design is inspired rather than dictated by the Kolmogorov superposition theorem, have emerged as a structured alternative to MLPs." (p. 1)
- "The review is organized around three core themes: (i) clarifying the relationships between KANs and Kolmogorov superposition theory (KST), MLPs, and classical kernel methods; (ii) analyzing basis functions as a central design axis..." (p. 1)
- "Unlike MLPs, where activations operate globally across the domain, B-splines have local support... this locality inherently mitigates catastrophic forgetting in continual learning." (p. 6)
- "While KANs exhibit remarkable theoretical and empirical scaling properties, their standard implementation involves evaluating independent splines per edge, leading to significant computational overhead compared to the highly optimized dense matrix multiplications of MLPs." (p. 11)
- "Step 6: Optimize for speed or accuracy. If speed is the primary objective then Prefer ReLU-KAN, HRKAN, FastKAN, LeanKAN. Enable JAX/CUDA kernels..." (p. 21)

## 7. Authors' conclusions

The authors conclude that KANs represent a highly structured, expressive, and interpretable alternative to traditional MLPs, effectively bridging the gap between exact mathematical approximation theory (KST) and empirical deep learning. By understanding the theoretical underpinnings and treating the choice of basis functions (e.g., B-splines, Chebyshev polynomials, Wavelets) and grid resolutions as configurable design axes, practitioners can systematically tailor KANs to solve complex, domain-specific problems. They emphasize that while standard KANs face recognized computational speed and memory challenges, the rapidly growing ecosystem of specialized variants and hardware optimizations makes these networks increasingly viable for both high-speed applications and rigorous scientific discovery.

## 8. Limitations stated by authors

- **Computational Overhead**: Evaluating independent complex functions on every network connection prevents the use of highly optimized batched matrix multiplication, making KANs significantly slower to train and infer than MLPs on standard GPU hardware.
- **Memory Consumption**: Storing grid parameters and basis activations per edge drastically increases the memory footprint, scaling at $O(N^2 G)$.
- **High-Dimensional Scaling**: While theoretical KST avoids the curse of dimensionality, practical KAN implementations struggle with very high-dimensional inputs (e.g., $d > 100$) due to the explosive, quadratic growth in the number of edge functions.

## 9. Future work suggested by authors

- Developing highly optimized, fused CUDA kernels specifically designed for heterogeneous basis functions on edges to close the speed gap with MLPs.
- Exploring adaptive network topologies that dynamically grow, prune, or re-allocate nodes and edges during the training process based on local error metrics.
- Integrating KAN layers with foundational architectures (e.g., transformers to create "KANsformers") to enable scalability for massive natural language processing and computer vision tasks.
- Formulating rigorous approximation bounds and theoretical guarantees for "deep" KANs (architectures with depth $L > 2$), which extend beyond the original KST proofs.

## Additional critical details

- **Taxonomy of Basis Functions**: The paper introduces a critical classification system for KAN variants based on their edge functions:
  - **Spline-based (B-splines, NURBS)**: Localized, allows grid extension, excellent for symbolic extraction, but computationally slow.
  - **Polynomial-based (Chebyshev, Jacobi, Legendre)**: Computationally fast, global support, but prone to Runge's phenomenon at domain edges.
  - **Radial Basis Functions (RBFs) (FastKAN)**: Highly localized, memory-efficient.
  - **Frequency/Wavelet-based (Fourier, Wav-KAN)**: Ideal for multiscale, highly oscillatory, or periodic data.
  - **MLP/Piecewise Linear (ReLU-KAN)**: Extremely fast, leveraging standard ReLU hardware optimization to mimic spline behavior without the overhead.

---
**Extraction date**: 2026-05-19
**PDF source**: Practitioner_Guide_to_KANs.pdf

---

# Article record: TKAN: Temporal Kolmogorov-Arnold Networks

## Metadata
- **BibTeX key**: genet_inzirillo_2024
- **Authors**: Rémi Genet, Hugo Inzirillo
- **Year**: 2024 (ArXiv v4 submitted July 2025 according to PDF header, but BibTeX indicates 2024)
- **Title**: TKAN: Temporal Kolmogorov-Arnold Networks
- **Journal/Conference**: arXiv preprint arXiv:2405.07344v4 [cs.LG]
- **DOI/URL**: https://arxiv.org/abs/2405.07344
- **Keywords**: [not found]

## 1. Theoretical foundation

**Key theorems:**

- **Kolmogorov-Arnold Representation Theorem (1957)**: States that any multivariate continuous function can be represented as a finite composition of continuous functions of a single variable and the operation of addition.

**Key mathematical definitions:**

- **KAN Formulation**: The network is based on the theorem, formalized as $f(x) = \sum_{q=1}^{2n+1} \Phi_q (\sum_{p=1}^{n} \phi_{q,p}(x_p))$, replacing fixed node activations with learnable spline functions on edges.
- **Standard LSTM Formulation**: Defined by equations controlling the forget gate $f_t$, input gate $i_t$, cell state $C_t$, output gate $o_t$, and hidden state $h_t$ using linear weight matrices $W$, biases $b$, and fixed activation functions $\sigma$ and $\tanh$.

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: Temporal Kolmogorov-Arnold Networks (TKAN), composed of Recurring Kolmogorov-Arnold Networks (RKAN) Layers.

**Core mathematical formulation:**

The RKAN layer replaces the standard dense linear layers of an LSTM with KANLinear layers. The new formulation is:

- Forget Gate: $f_t = \sigma(KANLinear_f([h_{t-1}, x_t]))$
- Input Gate: $i_t = \sigma(KANLinear_i([h_{t-1}, x_t]))$
- Cell State Update: $\tilde{C}_t = \tanh(KANLinear_C([h_{t-1}, x_t]))$
- Output Gate: $o_t = \sigma(KANLinear_o([h_{t-1}, x_t]))$
- Final Cell State: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
- Final Hidden State: $h_t = o_t \odot \tanh(C_t)$

**Algorithm pseudocode**: [not found]

**Architecture details:**

- The architecture integrates memory management for sequential data by maintaining the gating mechanism of LSTMs while utilizing KAN's B-spline-based learnable activations on the edges inside the gates.
- Inputs (e.g., of shape (batch_size, timesteps, features)) are passed into the RKAN layer.
- To generate multi-step forecasts, the output of the final timestep from the RKAN layers is fed into subsequent KAN or Dense layers to produce the desired multi-step output sequence.

**Innovative components:**

- Standard KANs lack internal memory states, making them ill-suited for complex sequential time-series tracking. Standard RNNs/LSTMs are limited by fixed node activations. TKAN solves both by embedding KANLinear layers directly into the recurring gating logic of the LSTM, combining sequence memory with highly accurate, learnable spline approximations.

## 3. Data

**Dataset name(s):**

- Synthetic Sine Wave Dataset.
- Historical Bitcoin (BTC) Market Dataset.

**Data source:**

- Synthetic data generated for benchmarking.
- Public historical market data.

**Data size:**

- Sine Wave: 10,000 data points.
- BTC Market Data: From January 1, 2021, to January 1, 2024; sampled at a 1-hour frequency.

**Data preprocessing:**

- BTC dataset utilizes OHLCV (Open, High, Low, Close, Volume) features.
- Normalized using a MinMax scaler.
- Sequenced using a sliding window: lookback of 40 intervals to predict the next 10 intervals.

**Train/validation/test split:**

- Sine Wave: Split into a training set and an independent test set.
- BTC Market Data: 80% Training set, 20% Testing set.

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against**: SimpleRNN, GRU, LSTM, and standard KAN.

**Evaluation metrics:**

- $R^2$ Score (Coefficient of Determination).
- Mean Absolute Error (MAE).
- Mean Squared Error (MSE).

**Hyperparameters:**

- **Sine Wave experiment**: Lookback = 10 steps, Output = 15 steps. Hidden units = 50. Adam optimizer, learning rate = $10^{-3}$. Epochs = 10. Batch size = 32. TKAN specific: spline order = 3, grid size = 3.
- **BTC experiment**: Lookback = 40 steps, Output = 10 steps. Hidden units = 100. Adam optimizer, learning rate = $10^{-3}$. Epochs = 10. Batch size = 32. TKAN specific: spline order = 3, grid size = 3.

**Hardware/software environment**: Implemented in Python. Code available at github.com/remigenet/TKAN and via pip install tkan.

**Number of runs / seeds**: The BTC experiment was run 10 separate times for all models to ensure statistical significance.

**Statistical tests performed**: Calculated the averages and standard deviations of out-of-sample metrics across the 10 runs.

## 5. Results

**Quantitative results:**

- **BTC Out-of-Sample Performance (Averaged over 10 runs):**
  - TKAN: MSE = 0.016 (Std: 0.003) | MAE = 0.091 (Std: 0.009)
  - LSTM: MSE = 0.019 (Std: 0.004) | MAE = 0.100 (Std: 0.007)
  - GRU: MSE = 0.021 (Std: 0.006) | MAE = 0.103 (Std: 0.013)
- **Model Complexity (BTC setup with 100 hidden units):**
  - LSTM/GRU: ~60,800 parameters.
  - TKAN: ~120,400 parameters.

**Qualitative results:**

- **Sine Wave**: Figure 3 demonstrates that standard models (LSTM, GRU) successfully fit 1-step predictions but suffer severe degradation (flatlining) on multi-step predictions (T+15). TKAN perfectly maintains the periodic structure, amplitude, and phase across the entire multi-step horizon.
- **BTC Data**: Figure 4 visualizes prediction horizons on real market data. TKAN consistently adapts to the complex non-linear sequence better than baselines over the 10 predicted steps.

**Ablation studies:**

- Compared the sequence generation of a purely dense KAN against the memory-equipped TKAN on the sine wave task. The dense KAN failed to track long-term sequential dependencies, proving that memory management (LSTM gating) combined with KAN is required for time-series tasks.

**Comparison with baselines:**

- On the real-world BTC dataset, TKAN outperformed the LSTM baseline by reducing the MSE by approximately 15.7% (from 0.019 to 0.016) and the MAE by 9% (from 0.100 to 0.091).

## 6. Direct quotations

- "Inspired by the Kolmogorov-Arnold Networks (KANs), a promising alternative to Multi-Layer Perceptrons (MLPs), we propose a new neural network architecture inspired by KAN and LSTM, called 'Temporal Kolomogorov-Arnold Networks' (TKANs)." (p. 1)
- "TKANs combine the strength of both networks. They are composed of Recurring Kolmogorov-Arnold Networks (RKANs) Layers embedding memory management." (p. 1)
- "By substituting the dense layers in traditional RNNs or LSTMs with KAN layers, TKANs benefit from the exact representational power of KANs, allowing for a more accurate and efficient processing of sequential data." (p. 3)
- "While not specifically interesting for short-term predictions, it especially demonstrates an ability to largely outperform other models when it comes to multi-step predictions." (p. 6)
- "Our experiments show that it is usable and more stable than GRU and LSTM on real historical market data." (p. 6)

## 7. Authors' conclusions

The authors conclude that Temporal Kolmogorov-Arnold Networks (TKAN) successfully merge the highly accurate, B-spline-based function approximation of KANs with the essential memory management of LSTMs. By integrating KANLinear layers directly into recurrent gating mechanisms (RKANs), the architecture processes sequential data efficiently. While TKAN is not strictly necessary for simple short-term single-step predictions, it drastically outperforms standard LSTMs and GRUs in multi-step forecasting tasks by maintaining long-term stability and significantly reducing error accumulation. The results on both synthetic and real-world financial datasets (Bitcoin) confirm that TKANs are a highly robust and relevant architectural advancement for complex temporal analysis.

## 8. Limitations stated by authors

- **Computational Complexity**: TKANs are more computationally intensive during training compared to standard LSTMs because evaluating B-splines at every recurrent gate increases the parameter count and the mathematical operations per step (e.g., ~120k parameters vs. ~60k parameters for an LSTM with the same hidden size).
- **Short-Term Utility**: The model does not offer significant predictive advantages for very short-term (e.g., single-step) predictions, making it more suited exclusively for multi-step tasks.

## 9. Future work suggested by authors

- Opens new pathways to improve the capacity to calibrate highly accurate time-series models over multiple steps, addressing one of the hardest challenges in temporal analysis.

## Additional critical details

- The paper explicitly notes that TKAN is available as an open-source Python package, accessible via pip install tkan, indicating its readiness for direct practitioner implementation.

---
**Extraction date**: 2026-05-19
**PDF source**: arXiv:2405.07344

---

# Article record: DecoKAN: Interpretable Decomposition for Forecasting Cryptocurrency Market Dynamics

## Metadata
- **BibTeX key**: gao_decokan_2025
- **Authors**: Yuan Gao, Zhenguo Dong, Xuelong Wang, Zhiqiang Wang, Yong Zhang, Shaofan Wang
- **Year**: 2025
- **Title**: DecoKAN: Interpretable Decomposition for Forecasting Cryptocurrency Market Dynamics
- **Journal/Conference**: arXiv preprint arXiv:2512.20028v1 [cs.LG]
- **DOI/URL**: arXiv:2512.20028v1
- **Keywords**: Multivariate time series forecasting, Cryptocurrency, Kolmogorov-Arnold Network, Discrete wavelet transform, Interpretability

## 1. Theoretical foundation

**Key theorems:**

- **Kolmogorov-Arnold Representation Theorem (KART)**: Any multivariate continuous function $f:[0,1]^n \to \mathbb{R}$ can be represented as a finite composition of continuous functions of a single variable and the operation of addition. Mathematically: $f(x) = f(x_1, \dots, x_n) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^n \phi_{q,p}(x_p) \right)$, where $\phi_{q,p} : [0,1] \to \mathbb{R}$ and $\Phi_q : \mathbb{R} \to \mathbb{R}$ are continuous 1D functions.

**Key mathematical definitions:**

- **Discrete Wavelet Transform (DWT)**: Decomposes a signal sequence $x = \{x_1, x_2, \dots, x_L\}$ into approximate (low-frequency) and detail (high-frequency) coefficients using a low-pass filter $l$ and a high-pass filter $h$. For level $j$:
  - $a^{(j)}_k = \sum_{m} l_{m-2k} a^{(j-1)}_m$
  - $d^{(j)}_k = \sum_{m} h_{m-2k} a^{(j-1)}_m$
- **Inverse Discrete Wavelet Transform (IDWT)**: Reconstructs the signal from its coefficients: $a^{(j-1)}_m = \sum_k (l_{m-2k} a^{(j)}_k + h_{m-2k} d^{(j)}_k)$.
- **Reversible Instance Normalization (RevIN)**: $\hat{X}_{l,c} = \frac{X_{l,c} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$, where $\mu_c = \frac{1}{L}\sum_{l=1}^L X_{l,c}$ and $\sigma_c^2 = \frac{1}{L}\sum_{l=1}^L (X_{l,c} - \mu_c)^2$.

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: DecoKAN (Interpretable Decomposition for Forecasting Cryptocurrency Market Dynamics)

**Core mathematical formulation:**

- **KAN Edge Activation**: $\phi(x) = w_b b(x) + w_s \text{spline}(x)$, where $b(x) = \text{SiLU}(x) = \frac{x}{1+e^{-x}}$ and $\text{spline}(x) = \sum_{i=0}^{G+k-1} c_i B_i(x)$. $B_i(x)$ are B-spline basis functions of order $k$ over $G$ grid intervals, and $c_i$ are learnable control points.
- **Resolution Branch Transformation**: Given a normalized input matrix $\hat{\mathcal{X}} \in \mathbb{R}^{L \times C}$, DWT decouples it into sub-series $\hat{\mathcal{X}}_i$. Each is processed by:
  - **Temporal KAN Mixer**: $\mathcal{Y}_T = \text{TemporalKAN}(\hat{\mathcal{X}}_i)$, applied across the time dimension $L$ to capture temporal autocorrelations.
  - **Feature KAN Mixer**: $\mathcal{Y}_F = \text{FeatureKAN}(\mathcal{Y}_T)$, applied across the feature dimension $C$ to capture instantaneous cross-variate dependencies.
- **Sparsification Penalty (L1 Norm)**: $|\phi|_1 \equiv \frac{1}{N_p} \sum_{s=1}^{N_p} |\phi(x^{(s)})|$ and layer-wise $L_1$ norm $|\Phi|_1 \equiv \sum_{i=1}^{n_{in}} \sum_{j=1}^{n_{out}} |\phi_{i,j}|_1$.
- **Sparsification Penalty (Entropy)**: $S(\Phi) \equiv -\sum_{i=1}^{n_{in}} \sum_{j=1}^{n_{out}} \frac{|\phi_{i,j}|_1}{|\Phi|_1} \log \left( \frac{|\phi_{i,j}|_1}{|\Phi|_1} \right)$.
- **Overall Loss Function**: $\mathcal{L}_{total} = \text{MSE}(\hat{Y}, Y) + \lambda \left( \gamma_1 \sum_{l=0}^{L-1} |\Phi_l|_1 + \gamma_2 \sum_{l=0}^{L-1} S(\Phi_l) \right)$, where $\lambda$ scales the regularization, and $\gamma_1, \gamma_2$ balance L1 and Entropy.

**Algorithm pseudocode**: [not found]

**Architecture details:**

- **RevIN Layer**: Normalizes the $L \times C$ input sequence to mitigate non-stationary distribution shifts.
- **Multi-level DWT Module**: Uses the Daubechies 4 ('db4') wavelet to decompose the input sequence into one Approximation branch (low frequency) and one or more Detail branches (high frequency).
- **KAN Mixers (Resolution Branches)**: Each decoupled coefficient series is routed into independent branches. Each branch contains a Temporal KAN (mixing $L \to T$) followed by a Feature KAN (mixing $C \to C$).
- **IDWT Reconstruction**: The predicted coefficients are synthesized back into the time domain sequence of length $T$.
- **Inverse RevIN Layer**: Denormalizes the $T \times C$ predictions back to the original data scale.

**Innovative components:**

- Synergistic integration of signal processing (DWT) with KANs to isolate high-frequency crypto shocks from macro trends.
- Using dual KAN mixers (Temporal and Feature) in a multi-resolution framework to achieve competitive forecasting accuracy while remaining intrinsically transparent and allowing for symbolic extraction (unlike black-box MLPs or Transformers).

## 3. Data

**Dataset name(s):**

- Standard Benchmarks: ETTh1, ETTh2, ETTm1, ETTm2.
- Cryptocurrency Datasets: Bitcoin (BTC), Ethereum (ETH), Monero (XMR).

**Data source**: ETT datasets from standard public repositories; Cryptocurrency datasets sourced from the Coin Metrics API.

**Data size:**

- ETTh1/ETTh2: 24 months, hourly frequency (7 features).
- ETTm1/ETTm2: 24 months, 15-minute frequency (7 features).
- BTC: 120 months (2013-05-01 to 2024-04-30), daily frequency (18 features).
- ETH: 100 months (2016-01-01 to 2024-04-30), daily frequency (18 features).
- XMR: 112 months (2015-01-01 to 2024-04-30), daily frequency (18 features).

**Data preprocessing**: Zero-mean normalization applied locally via the RevIN module. Missing values (NANs) filled using forward fill (ffill).

**Train/validation/test split:**

- ETT datasets: Chronological split of 7:1:2 (Train/Validation/Test).
- Crypto datasets: Chronological split of 8:1:1 (Train/Validation/Test).

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against:**

- Transformer-based: PatchTST, Crossformer, iTransformer, FEDformer, Informer, Autoformer.
- MLP-based: WPMixer, DLinear, TimeMixer.
- CNN-based: TimesNet.
- Graph-based: TimeFilter.

**Evaluation metrics:**

- Mean Squared Error (MSE).
- Mean Absolute Error (MAE).

**Hyperparameters:**

- Look-back window length ($L$): 96.
- Prediction horizons ($T$): $\{96, 192, 336, 720\}$.
- Optimizer: Adam.
- Loss function for training: MSE.
- Batch size: 32 (default), scaled up to 128 for robustness analysis.
- Learning rate: Grid searched across $\{0.0001, 0.0005, 0.001, 0.005\}$.
- Wavelet basis: 'db4'. Wavelet decomposition levels ($J$): 1.
- KAN Spline Order ($k$): 3.
- KAN Grid Size ($G$): 3.
- Regularization Weights: $\gamma_1, \gamma_2 \in [0.01, 0.1]$, Overall scale $\lambda \in \{0.001, 0.01, 0.1\}$.
- Pruning threshold ($L_2$ norm): $10^{-4}$.

**Hardware/software environment**: PyTorch 2.1.2 framework, executed on a single NVIDIA RTX 3090 GPU (24GB).

**Number of runs / seeds**: [not found]

**Statistical tests performed**: [not found]

## 5. Results

**Quantitative results:**

- **Average performance across all 4 horizons (T=96, 192, 336, 720):**
  - BTC: DecoKAN (MSE: 0.085, MAE: 0.170) | WPMixer (0.100, 0.187) | PatchTST (0.102, 0.198) | TimesNet (0.111, 0.210) | TimeFilter (0.105, 0.198).
  - ETH: DecoKAN (MSE: 0.056, MAE: 0.138) | WPMixer (0.079, 0.169) | PatchTST (0.089, 0.187) | TimeFilter (0.089, 0.187).
  - XMR: DecoKAN (MSE: 0.090, MAE: 0.182) | WPMixer (0.106, 0.203) | TimeFilter (0.106, 0.206).
- **Specific Horizon Performance (BTC):**
  - T=96: DecoKAN (MSE: 0.063, MAE: 0.145)
  - T=192: DecoKAN (MSE: 0.076, MAE: 0.162)
  - T=336: DecoKAN (MSE: 0.089, MAE: 0.174)
  - T=720: DecoKAN (MSE: 0.111, MAE: 0.197)
- **ETT Benchmark (Summary)**: DecoKAN achieved the best or second-best results in 21 out of 32 experimental settings across the four ETT datasets, demonstrating strong general-purpose forecasting utility.
- **Computational Efficiency (T=96):**
  - DecoKAN: Parameters = 11.66M, GFLOPs = 0.0073.
  - PatchTST: Parameters = 453.71M, GFLOPs = 0.8407.
  - WPMixer: Parameters = 11.83M, GFLOPs = 0.0142.

**Qualitative results:**

- The model successfully extracts explicit, human-readable formulas representing high-frequency dynamics. In an explicit case study predicting the target variable $y_1$, the symbolified formula was derived with high confidence ($R^2 > 0.99$): $y_1 = -0.5\sin(6.31x) + 0.3\cos(2.1x) + 0.12x^2 - 0.05x + 0.01$.
- Visualization of the network topologies pre- and post-pruning shows that the Approximation Branch acts as a dense backbone (only 4.79% edges pruned), whereas the Detail Branch captures transient shocks and forms a highly sparse, interpretable network (76.28% edges pruned).

**Ablation studies:**

- **w/o Temporal KAN (w/o T)**: Significant performance degradation primarily on standard ETT datasets (e.g., ETTh2), validating the module's role in capturing high temporal autocorrelation.
- **w/o Feature KAN (w/o F)**: Massive performance drop on Cryptocurrency datasets (e.g., BTC average MSE jumped from 0.085 to 0.158), proving that instantaneous cross-variate relationships (feature dependencies) overwhelmingly drive crypto volatility.
- **w/o DWT**: Removing the wavelet transform led to consistent performance degradation across all datasets, confirming the essential value of frequency-based signal decoupling.
- **Wavelet Basis ('haar' vs 'db4')**: Substituting 'db4' for the simpler 'haar' wavelet slightly increased MSE across all horizons on ETTh2, confirming 'db4's superior capability in capturing continuous trends.

**Comparison with baselines:**

- DecoKAN outperformed the strongest baseline (WPMixer) by reducing the average MSE by 15.0% on BTC, 29.1% on ETH, and 15.1% on XMR.
- Highly parameter-efficient, utilizing $38\times$ fewer parameters and requiring $115\times$ fewer GFLOPs compared to leading Transformer models (PatchTST).

## 6. Direct quotations

- "To overcome these limitations, we propose DecoKAN, an interpretable forecasting framework that integrates multi-level Discrete Wavelet Transform (DWT) for decoupling and hierarchical signal decomposition with Kolmogorov-Arnold Network (KAN) mixers for transparent and interpretable nonlinear modeling." (p. 1)
- "Unlike traditional time series (e.g., electricity load) dominated by strong temporal autocorrelation, cryptocurrency prices are often driven by high-frequency shocks and the instantaneous interplay between system variables... Consequently, the performance of DecoKAN here relies predominantly on the Feature KAN component's ability to capture these complex cross-variate dependencies." (p. 9)
- "Interpretability analysis revealed a functional separation within the model: the Detail Branch captures explicit symbolic relations for high-frequency fluctuations, while the Approximation Branch acts as a robust structural backbone for the forecasting task." (p. 11)
- "Compared to the strongest baseline model (WPMixer), DecoKAN reduces the average MSE by approximately 15.0% on the BTC dataset, 29.1% on ETH, and 15.1% on XMR." (p. 6)
- "Despite KAN's theoretical parameter efficiency, B-spline operations lack hardware-level optimization in current GPU architectures compared to the highly optimized matrix multiplications in MLPs and Transformers, resulting in longer training times per epoch." (p. 11)

## 7. Authors' conclusions

The authors conclude that DecoKAN successfully bridges the gap between state-of-the-art predictive accuracy and intrinsic model interpretability in the context of complex multivariate time series. By integrating Discrete Wavelet Transform (DWT) with Kolmogorov-Arnold Network (KAN) mixers, the framework effectively decouples long-term trends from high-frequency speculative noise, making it exceptionally well-suited for cryptocurrency market dynamics. Through an automated interpretability pipeline involving L1/entropy sparsification and L2-norm pruning, DecoKAN can extract precise, closed-form mathematical formulas representing the modeled dynamics with high fitness ($R^2 > 0.99$). The study establishes that while Transformers and traditional MLPs operate as 'black boxes', DecoKAN provides trustworthy, transparent financial insights without sacrificing, and often exceeding, empirical forecasting performance.

## 8. Limitations stated by authors

- **Computational Overhead**: B-spline parameterizations currently lack optimized parallelization on GPU hardware (unlike dense matrix multiplications in MLPs), resulting in noticeably longer training times per epoch.
- **Symbolification Constraints**: The symbolic extraction process relies on fitting a pre-defined finite set of candidate mathematical functions, potentially restricting its ability to uncover entirely novel or highly unconventional functional forms.
- **Human Expertise Requirement**: Translating the extracted mathematical formulas and graphical topological structures into actionable financial or physical theories still requires significant human domain expertise.

## 9. Future work suggested by authors

- Engineering efficient, hardware-aware implementations of KAN operations (such as fused CUDA kernels) to drastically accelerate the evaluation of B-splines and reduce training times.
- Broadening the application of the DecoKAN framework to discover physical and economic laws in other complex, real-world dynamical systems beyond cryptocurrencies.

## Additional critical details

- **Cryptocurrency Dataset Features used (18 total)**: CapMrktCurUSD, FeeTotUSD, IssTotUSD, PriceUSD, TxTfrValAdjUSD, VtyDayRet30d, VtyDayRet60d, TxCnt, TxTfrValUSD, TxTfrCnt, SplyCur, SplyFF, DiffMean, DiffLast, HashRate, BlkCnt, BlkSizeMean, BlkSizeByte.
- The interpretability pipeline strictly follows three sequential stages:
  1. Sparsification during training via L1 and entropy regularizations.
  2. Pruning post-training by dropping edges with $L_2$ activation norms below $10^{-4}$.
  3. Symbolification via mapping surviving splines to explicit symbolic expressions.

---
**Extraction date**: 2026-05-19
**PDF source**: arXiv:2512.20028

---

# Article record: Kolmogorov-Arnold networks for time series forecasting: a comprehensive review

## Metadata
- **BibTeX key**: yamak_et_al_2025
- **Authors**: Peter Tettey Yamak, Yujian Li, Ting Zhang, Muhammad Salman Pathan
- **Year**: 2025
- **Title**: Kolmogorov-Arnold networks for time series forecasting: a comprehensive review
- **Journal/Conference**: Cluster Computing
- **DOI/URL**: https://doi.org/10.1007/s10586-025-05574-9
- **Keywords**: Kolmogorov-Arnold networks, Time-series forecasting, Interpretability, Adaptive architectures

## 1. Theoretical foundation

**Key theorems:**

- **Kolmogorov-Arnold Representation Theorem**: Any continuous multivariate function $f(x): [0,1]^n \rightarrow \mathbb{R}$ can be expressed as a finite composition of univariate functions: $f(x) = \sum_{q=1}^{2n+1} \Phi_q (\sum_{p=1}^n \phi_{q,p}(x_p))$ where $\Phi_q$ (outer functions) and $\phi_{q,p}$ (inner functions) are continuous univariate functions.

**Key mathematical definitions:**

- **Spline Parameterization on Edges**: Each edge function $g_{ij}(x)$ replaces fixed node activations and is parameterized as $g_{ij}(x) = \sum_{k=0}^K a_{ijk} B_k(x)$, where $B_k(x)$ are B-spline basis functions of polynomial order $k$ (typically cubic, $k=3$) and $a_{ijk}$ are learnable control coefficients.
- **Dynamic Grid Extension**: Spline resolution is refined iteratively during training by inserting new knots to adapt to nonstationary data: $G_{new} = G \cup \{x_j | |\sum_{k=0}^K a_{ijk} B_k(x) - y| > \epsilon\}$, where $\epsilon$ is an error tolerance.
- **Deep KAN Architecture**: Generalized from the 2-layer theorem into deeper arbitrary architectures: $y = KAN(x) = (\Phi_L \circ \Phi_{L-1} \circ \dots \circ \Phi_1)(x)$.

**Key lemmas or propositions:**

- **Spline Convergence Guarantee**: Under the Lipschitz continuity constraint of splines $(|B_k(x) - B_k(y)| \le L|x-y|)$, stochastic gradient descent for KANs converges at a rate of $\mathcal{O}(1/\sqrt{T})$, matching theoretical guarantees for standard MLPs.

**Relevant mathematical proofs (summary)**: [not found - paper is a review/survey]

## 2. Proposed method / architecture / model

**Name of method/model**: As a review, the paper classifies state-of-the-art KAN models into five distinct architectural frameworks: (1) Core Spline-Optimized Models, (2) Recurrent KAN Models (RKAN), (3) Gated Residual KAN Models (GRKAN), (4) Attention-Enhanced KAN Models, and (5) Cross-Architecture Hybrid Models.

**Core mathematical formulation:**

- **KAN Composite Loss Function**: $L = \frac{1}{N}\sum_{t=1}^N (y_t - \hat{y}_t)^2 + \lambda \sum_{i,j,k} (\int [g_{ij}''(x)]^2 dx)$ combining MSE with a curvature penalty to avoid overfitting.
- **RKAN Gating Mechanism (Eqs. 16-18)**: Replaces linear layers in LSTMs with KANs. $f_t = \sigma(RKAN_{forget}(x_t, h_{t-1}))$, $i_t = \sigma(RKAN_{input}(x_t, h_{t-1}))$, $c_t = f_t \odot c_{t-1} + i_t \odot \tanh(RKAN_{cell}(x_t, h_{t-1}))$, $o_t = \sigma(RKAN_{output}(x_t, h_{t-1}))$, $h_t = o_t \odot \tanh(c_t)$.
- **GRKAN (Gated Residual KAN)**: Nested KAN transformation $\eta_1 = KAN(\varphi_{\eta_1}(\cdot), \eta_2)$ modulated via Gated Linear Units (GLUs): $GLU_\omega(\gamma) = \sigma(W_{4,\omega}\gamma + b_{4,\omega}) \odot (W_{5,\omega}\gamma + b_{5,\omega})$. Final output is $LayerNorm(x + GLU_\omega(\eta_2))$.
- **HiPPO-KAN**: Uses High-order Polynomial Projection Operators to encode sequence $u_{1:L}$ into fixed latent dimension $c^{(L)} = \text{hippo}_z(u_{1:L})$, processed by KANs as $c^{(L+1)} = KAN(c^{(L)})$.

**Algorithm pseudocode**: [not found]

**Architecture details:**

- KANs remove fixed activation functions at the nodes and place learnable, 1D B-spline functions along the edges/weights.
- **m-TKAN**: Incorporates a Markov regime-switching matrix into RKAN layers dynamically estimated via $P_t = \text{softmax}(W_z \cdot RKAN(v_{t-1})) + b_z$.
- **TabKANet**: Unifies categorical features with numerical features using a KAN embedding module with Batch Normalization: $\tilde{x}_{num} = BN(KAN(x_{num}))$.

**Innovative components:**

- KANs achieve feature disentanglement and direct visual interpretability, tracking gradient magnitudes along specific spline edges to isolate feature contributions dynamically without needing post-hoc methods like SHAP.
- Gated/Attention-hybrid KANs demonstrate how classical Kolmogorov-Arnold decomposition can successfully merge with sequence memory (LSTMs) and spatial projection (HiPPO) for high-frequency non-stationary datasets.

## 3. Data

**Dataset name(s)**: ETTh1, ETTh2, ETTm1, ETTm2, Weather (WTH), Electricity Consuming Load (ECL), Influenza-like illness (ILI), Traffic, and Financial datasets (S&P 500, DJIA, multiple cryptocurrencies: BTC, ETH, ADA, XMR, EOS, MATIC, TRX, FTM, BNB, XLM, ENJ, CHZ, BUSD, ATOM, LINK, ETC, XRP, BCH, LTC), Chiller Energy.

**Data source**: Recognized public benchmark datasets.

**Data size**: Hourly (ETTh1, ETTh2, WTH), Minutely (ETTm1, ETTm2), Half-hourly (ECL), 5-minute (Traffic), Weekly (ILI), Daily (Crypto/Stock).

**Data preprocessing**: [not found - the review aggregates various studies' data configurations]

**Train/validation/test split**: [not found]

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against**: DLinear, SCINet, PatchTST, Time-LLM, TimesNet.

**Evaluation metrics:**

- MAE (Mean Absolute Error): $\frac{1}{N}\sum_{t=1}^N|y_t - \hat{y}_t|$
- MSE (Mean Squared Error): $\frac{1}{N}\sum_{t=1}^N(y_t - \hat{y}_t)^2$
- RMSE (Root Mean Square Error): $\sqrt{\frac{1}{N}\sum_{t=1}^N(y_t - \hat{y}_t)^2}$
- MAPE (Mean Absolute Percentage Error): $\frac{100\%}{N}\sum_{t=1}^N |\frac{y_t - \hat{y}_t}{y_t}|$
- $R^2$ (Coefficient of Determination): $1 - \frac{\sum_{t=1}^N(y_t - \hat{y}_t)^2}{\sum_{t=1}^N(y_t - \bar{y}_t)^2}$

**Hyperparameters**: In the unified test, the forecasting horizon was $T=96$ steps. Spline order ($k=3$), grid size ($G=5$), optimized via Adam with cosine annealing or L-BFGS for fine-tuning splines. L1 regularization used for pruning.

**Hardware/software environment**: [not found]

**Number of runs / seeds**: The cross-validation performance bounds were generated across 5 separate simulation runs.

**Statistical tests performed**: 95% Confidence Intervals calculated for MAE and MSE across cross-validation runs.

## 5. Results

**Quantitative results:**

- **ETTh1 Benchmark (Forecast horizon 96):**
  - SigKAN: MSE = 0.0027, MAE = 0.0415.
  - TKAT: MSE = 0.1523, MAE = 2.1839. (Note: Table 10 provides alternate text values displaying TKAT MSE=2.1682 / MAE=7.2272, signaling severe instability).
  - DLinear: MSE = 0.3827, MAE = 0.3931.
  - SCINet: MSE = 0.4393, MAE = 0.4171.
  - Time-LLM: MSE = 0.3661, MAE = 0.3763.
  - RMoK: MSE = 0.3252, MAE = 0.3613.
- **ETTh2 Benchmark (Forecast horizon 96):**
  - PatchTST: MSE = 0.1758.
  - TimesNet: MSE = 0.1883.
- **Cross-Validation Stability (5 runs):**
  - SigKAN exhibited extremely robust stability, with 95% CI widths below 0.02 for MAE, and actual MAE ranging between 0.0338–0.0712 across all datasets.
  - TKAT exhibited catastrophic instability with massive variance: ETTh1 MSE peaked at $9.24 \pm 2.24$ (CI: 6.46-12.02); ETTm2 MSE hit $17.48 \pm 3.50$.
  - RMoK demonstrated deterministically tight stability with standard deviations below 0.001 across runs, though with moderate accuracy.
- **Computational Efficiency (Iterations/sec) & Parameters:**
  - SigKAN: 95.33 it/s (38.00K parameters).
  - DLinear: 95.29 it/s (9.30K parameters).
  - TimesNet: 89.30 it/s (11.95K parameters).
  - SCINet: 65.12 it/s (10.11K parameters).
  - PatchTST: 40.82 it/s (5.80K parameters).
  - RMoK: 35.17 it/s (130.00K parameters).
  - TKAT: 9.44 it/s (14.22K parameters).
  - Time-LLM: 5.04 it/s (975.83K parameters).

**Qualitative results:**

- **Loss Curve Analysis**: Visual examination of SigKAN's training curves exposes an overfitting risk: validation loss plateaus around epoch 20 and slightly diverges upward, meaning the model's capacity to perfectly fit nonlinearities requires aggressive early stopping to maintain its test set superiority.
- **Interpretability**: KANs uniquely map temporal causality via spline heatmaps without post-hoc methods. By adjusting visual transparency via $\tanh(\beta A_{l,i,j})$, human operators can directly "see" the exact features and lag intervals influencing predictions, vital for domains like medical diagnostics and smart grids.
- **Catastrophic Underfitting**: TKAT's dismal results expose the dangers of hybridizing KANs with attention transformers; the combination failed to converge efficiently and heavily underfitted.

**Ablation studies**: [not found - the paper aggregates diverse models]

**Comparison with baselines**: SigKAN comprehensively outperformed DLinear, PatchTST, TimesNet, and Time-LLM on the ETTh1 benchmark, boasting up to 98% lower MSE while effectively matching the computational speed of the highly efficient DLinear model.

## 6. Direct quotations

- "This study presents Kolmogorov-Arnold Networks (KANs) for time series forecasting (TSF) as a transformative paradigm, leveraging the Kolmogorov-Arnold theorem to decompose multivariate temporal dependencies into interpretable, spline-parameterized univariate functions." (p. 1)
- "Through architectural innovations like adaptive grid refinement, gated residual mechanisms, and hybrid attention integration, KAN-based models achieve state-of-the-art performance, outperforming transformers with up to 98% lower MSE on benchmarks while offering unparalleled transparency in feature attribution." (p. 1)
- "SigKAN emerges as a standout among KAN-based models, achieving an MSE of 0.0027 and an MAE of 0.0415 on the ETTh1 dataset, far surpassing TKAT's MSE of 0.1523 and MAE of 2.1839, and even outpacing transformer-based models like PatchTST." (p. 25)
- "KANs face significant computational bottlenecks due to their reliance on spline-parameterized activations... the quadratic growth of parameters in RKAN layers, coupled with dynamic grid refinement, leads to memory-intensive training, often exceeding GPU capacities for large-scale applications." (p. 32)
- "As the demand for interpretable, scalable, and efficient forecasting grows, KANs stand at the forefront of a new era in time series analysis, one where mathematical transparency and empirical performance converge to empower decision-making across industries." (p. 35)

## 7. Authors' conclusions

The authors conclude that Kolmogorov-Arnold Networks (KANs) represent a foundational paradigm shift for Time Series Forecasting. By directly replacing fixed node activations with learnable, spline-based edge parameters, KANs elegantly bridge the gap between high-capacity "black-box" models (like Transformers and LLMs) and mathematically interpretable statistical methods. Variants like SigKAN and GRKAN consistently match or beat state-of-the-art forecasting benchmarks, achieving massive reductions in error metrics (MSE/MAE) while retaining processing efficiency comparable to linear frameworks. However, realizing the technology's full potential requires overcoming severe scalability bottlenecks and data-quality sensitivities inherent to B-spline approximation. Ultimately, the authors foresee KANs, particularly when hybridized with Graph Neural Networks and adaptive regularization, revolutionizing decision-making in sensitive domains like precision healthcare, algorithmic finance, and sustainable energy grid optimization.

## 8. Limitations stated by authors

- **Computational Overhead and Scalability**: High-dimensional datasets scale poorly because B-spline evaluations lack hardware-level optimization on current GPUs, unlike dense matrix multiplications used in MLPs.
- **Sequential Latency**: Spline activations compute sequentially, eliminating parallelization advantages, which renders real-time tracking (e.g., high-frequency crypto trading) difficult due to latency and error propagation in multi-step predictions.
- **Data Sensitivity & Stationarity Assumption**: Spline models heavily overfit to noisy or abrupt distributional shifts. High-frequency noise distorts B-spline smoothness, rendering standard KANs vulnerable to adversarial perturbations or sensor drift.
- **Interpretability Dilution**: In complex hybrids (like Attention-Enhanced TKAT), the core transparency of the KAN edges becomes muddled by layered attention weights, undermining the model's primary explanatory advantage.
- **Standardization Deficits**: Wide variations in implementation libraries, spline order ($k$), grid interval ($G$) tuning, and disparate baseline evaluations prevent reliable cross-study benchmarking.

## 9. Future work suggested by authors

- **Hardware-Aware Acceleration**: Developing quantized spline activations for FPGA/edge devices, and utilizing Neuromorphic or Photonic KAN architectures to achieve multi-fold reductions in energy consumption and computational latency.
- **Meta-Learning & Adaptive Regularization**: Employing meta-learned spline initialization (inspired by Few-Shot learning) to skip expensive iterative optimizations, combined with domain-adaptive unified loss functions blending shape-metrics with Dynamic Time Warping (DTW).
- **Spatial-Temporal Extension via GNNs**: Fusing KAN layers with Graph Neural Networks (GKANs) to solve complex multivariate spatial dependencies, replacing MLPs in message-passing frameworks for enhanced feature extraction.
- **Federated and Bayesian Implementations**: Creating Federated KANs for privacy-preserving clinical time series, and Bayesian KANs using Gaussian Process priors over spline coefficients to establish rigorous predictive uncertainty quantification.

## Additional critical details

- To objectively weigh the existing literature, the researchers evaluated the included papers utilizing a customized 4-point PRISMA-compliant rubric encompassing (1) Clarity of Methodology, (2) Reproducibility, (3) Evaluation Rigor, and (4) Handling of Limitations. The resulting synthesis scored the C-KAN paper the highest (3.5/4), validating its open-source rigor, while penalizing purely theoretical or proprietary models.
- An initial search space of 203 documents across IEEE, ACM, Scopus, Web of Science, and arXiv was systematically filtered down to exactly 13 foundational KAN papers utilizing the PRISMA workflow, ensuring deep focus specifically on rigorous time-series implementations.

---
**Extraction date**: 2026-05-19
**PDF source**: Article_KANs_for_time_series_forecasting_compreensive_review.pdf

---

# Article record: Forecasting VIX using interpretable Kolmogorov-Arnold networks

## Metadata
- **BibTeX key**: cho_lee_kim_2025
- **Authors**: So-Yoon Cho, Sungchul Lee, Hyun-Gyoon Kim
- **Year**: 2025
- **Title**: Forecasting VIX using interpretable Kolmogorov-Arnold networks
- **Journal/Conference**: arXiv preprint arXiv:2502.00980v1 [cs.LG]
- **DOI/URL**: https://arxiv.org/abs/2502.00980v1
- **Keywords**: Kolmogorov-Arnold network; Interpretable neural network; Time-series forecasting; Volatility; VIX

## 1. Theoretical foundation

**Key theorems:**

- **Kolmogorov-Arnold Representation Theorem**: Any continuous multivariate function $f$ on a bounded domain can be expressed as a finite composition of continuous functions of a single variable and addition. Mathematically: $f(x) = f(x_1, \dots, x_n) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^n \phi_{q,p}(x_p) \right)$, where $\phi_{q,p}: [0,1] \rightarrow \mathbb{R}$ and $\Phi_q: \mathbb{R} \rightarrow \mathbb{R}$ are continuous functions.
- **Universal Approximation Theorem (Mentioned for baselines)**: States that a multi-layer perceptron (MLP) with a single hidden layer and a non-linear activation function can approximate any continuous function on a compact subset of $\mathbb{R}^n$.

**Key mathematical definitions:**

- **VIX (CBOE Volatility Index)**: Represents the market's expectation of 30-day forward-looking volatility, derived from S&P 500 index options.
- **B-spline functions**: Piecewise polynomial functions defined over a grid of intervals, used in KANs for smooth curve approximation.

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: Interpretable Kolmogorov-Arnold Networks (KAN) for VIX forecasting.

**Core mathematical formulation:**

- **KAN Layer**: $x_{l+1} = \Phi_l x_l$, where $\Phi_l$ is a matrix of 1D functions $\phi_{l,j,i}$.
- **Edge Activation Function**: $\phi(x) = w \left( b(x) + \text{spline}(x) \right)$.
- **Basis function**: $b(x) = \text{silu}(x) = \frac{x}{1+e^{-x}}$.
- **Spline function**: $\text{spline}(x) = \sum_{i} c_i B_i(x)$, where $c_i$ are trainable control points and $B_i(x)$ are B-spline basis functions.
- **Sparsification Regularization (L1 norm)**: $|\phi|_1 = \frac{1}{N} \sum_{s=1}^N |\phi(x^{(s)})|$ and layer norm $|\Phi|_1 = \sum_{i,j} |\phi_{i,j}|_1$.
- **Entropy Regularization**: $S(\Phi) = -\sum_{i,j} \frac{|\phi_{i,j}|_1}{|\Phi|_1} \log \left( \frac{|\phi_{i,j}|_1}{|\Phi|_1} \right)$.
- **Overall Objective Function**: $Loss = L_{pred} + \lambda \left( \mu_1 \sum_{l=0}^{L-1} |\Phi_l|_1 + \mu_2 \sum_{l=0}^{L-1} S(\Phi_l) \right)$, where $L_{pred}$ is the prediction loss (e.g., MSE).
- **Symbolification (Affine mapping)**: $y \approx c \cdot f(a x + b) + d$, where $f$ is a selected symbolic function (e.g., linear, sine, exponential) and $(a,b,c,d)$ are fine-tuned affine parameters.

**Algorithm pseudocode:**

```
1. Train the KAN model with sparsity-inducing regularization.
2. Prune unimportant nodes and edges based on the L1 norms of their activation functions.
3. Set a list of candidate symbolic functions (e.g., x, x^2, sin(x), e^x).
4. Suggest the best-matching symbolic function for each edge using a grid search based on R^2.
5. Train the affine parameters (a,b,c,d) to fit the selected symbolic function to the data.
```

**Architecture details:**

- Highly parsimonious network structures tested: [n, 1], [n, 2, 1], [n, 3, 1], and deeper ones like [n, 2, 2, 1].
- The model fundamentally relies on a 2-layer structure with small hidden dimensions (e.g., 2 hidden nodes). Total parameters are explicitly constrained to ~60–72 parameters after pruning and symbolification.

**Innovative components:**

- Application of KANs to financial volatility forecasting to extract explicit, closed-form mathematical equations directly from the data.
- Bridging predictive accuracy and theoretical transparency by demonstrating that the extracted equations organically capture established financial stylized facts (mean reversion and the leverage effect) without prior structural assumptions.

## 3. Data

**Dataset name(s)**: CBOE Volatility Index (VIX), S&P 500 excess returns, 1-month U.S. Treasury bill yields.

**Data source**: Public data sourced from CBOE (for VIX) and CRSP (for S&P 500 and Treasury bills).

**Data size**: Daily frequency. Total of 5,033 observations spanning January 2, 2004, to December 29, 2023.

**Data preprocessing:**

- Input features were configured into three distinct experimental settings to capture different memory structures:
  - **Exp I (Short-term lags)**: $\{V_{t-1}, V_{t-2}, V_{t-3}, V_{t-4}, V_{t-5}\}$.
  - **Exp II (Heterogeneous lags)**: $\{V_{t-1}, V_{t-5}, V_{t-10}, V_{t-22}, V_{t-66}\}$.
  - **Exp III (Moving averages/HAR-style)**: $\{V_{t-1}, V_{t-1:t-5}, V_{t-1:t-22}, V_{t-1:t-66}\}$.
- For modeling the leverage effect, the previous day's S&P 500 excess return $R_{t-1}^e$ was added to the input set.

**Train/validation/test split:**

- Full Period (2004-2023): Train (2004-2015), Val (2016-2019), Test (2020-2023).
- Sub-period I (2004-2013): Train (2004-2009), Val (2010-2011), Test (2012-2013).
- Sub-period II (2014-2023): Train (2014-2019), Val (2020-2021), Test (2022-2023).

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against:**

- Random Walk / Forward-filling (RW)
- ARMA (Autoregressive Moving Average)
- ARIMA (Autoregressive Integrated Moving Average)
- HAR (Heterogeneous Autoregressive model)
- MLP (Multi-Layer Perceptron)
- LSTM (Long Short-Term Memory)

**Evaluation metrics:**

- MSE (Mean Squared Error): $\frac{1}{n} \sum_{t=1}^n (V_t - \hat{V}_t)^2$
- MAE (Mean Absolute Error): $\frac{1}{n} \sum_{t=1}^n |V_t - \hat{V}_t|$
- MAPE (Mean Absolute Percentage Error): $\frac{1}{n} \sum_{t=1}^n \left| \frac{V_t - \hat{V}_t}{V_t} \right|$
- $R^2$ (R-squared): $1 - \frac{\sum (V_t - \hat{V}_t)^2}{\sum (V_t - \bar{V}_t)^2}$
- QLIKE (Quasi-Likelihood): $\frac{1}{n} \sum_{t=1}^n \left( \frac{V_t}{\hat{V}_t} - \log\left(\frac{V_t}{\hat{V}_t}\right) - 1 \right)$

**Hyperparameters:**

- **KAN**: Optimizer = L-BFGS, Grid size $G \in [5, 10]$, Spline order $k=3$, Sparsity penalty $\lambda \in \{0.01, 0.05\}$.
- **MLP/LSTM**: Optimizer = Adam, Epochs = 1000, Batch size = 32, Learning rate = $10^{-3}$, Dropout = 0.2, Early stopping patience = 50.
- **MLP Architecture**: 4 hidden layers with sizes (128, 64, 32, 16). Total parameters: ~11,000.
- **LSTM Architecture**: 2 LSTM layers (128, 64) followed by 2 dense layers (32, 16). Total parameters: ~63,000.

**Hardware/software environment**: [not found]

**Number of runs / seeds**: [not found]

**Statistical tests performed:**

- **Mincer-Zarnowitz (MZ) regression test**: Evaluates forecast unbiasedness via $V_t = \alpha + \beta \hat{V}_t + e_t$. Tests hypotheses $H_0: \alpha=0$ and $H_0: \beta=1$.
- **Durbin-Watson (DW) test**: Tests for first-order serial correlation in the forecast errors. Values near 2 indicate no autocorrelation.

## 5. Results

**Quantitative results:**

- **Experiment III (HAR-style inputs), Full Period (2004-2023) Test Set:**
  - RW: MSE=3.992, MAE=1.354, MAPE=0.060, $R^2$=0.909, QLIKE=0.003
  - HAR: MSE=3.619, MAE=1.332, MAPE=0.058, $R^2$=0.917, QLIKE=0.003
  - MLP: MSE=3.617, MAE=1.328, MAPE=0.059, $R^2$=0.917, QLIKE=0.003
  - LSTM: MSE=3.879, MAE=1.341, MAPE=0.060, $R^2$=0.911, QLIKE=0.003
  - KAN: MSE=3.585, MAE=1.323, MAPE=0.058, $R^2$=0.918, QLIKE=0.003
  - Symbolic KAN: MSE=3.582, MAE=1.324, MAPE=0.058, $R^2$=0.918, QLIKE=0.003
- **Mincer-Zarnowitz (MZ) Test (Exp III, Full Period for Sym-KAN)**: $\alpha = -0.061$ (p-val=0.444), $\beta = 1.002$ (p-val=0.000). The model successfully passed unbiasedness tests.
- **Durbin-Watson (DW) Test (Exp III, Full Period for Sym-KAN)**: DW statistic = 2.015, indicating no serial correlation in errors.
- **Model Complexity**: Sym-KAN requires only ~60-72 parameters, compared to ~11,000 for MLP and ~63,000 for LSTM.

**Qualitative results:**

- Visualizations of the trained B-splines showed that the activation functions converged to nearly linear shapes, indicating that VIX dynamics relative to the chosen explanatory variables are fundamentally linear.
- Extraction of computational graphs post-pruning yielded highly interpretable paths, heavily weighting the $V_{t-1}$ node.
- **Leverage Effect Visualization**: When adding the S&P 500 excess return ($R_{t-1}^e$) to the inputs, the visual activation function for this specific feature was distinctly downward sloping, explicitly capturing the asymmetric financial phenomenon where negative returns correspond to volatility spikes.

**Ablation studies:**

- **Network Depth and Width (Appendix A)**: Tested deeper and wider KAN architectures (e.g., from [n, 2, 1] up to [n, 10, 1] and [n, 2, 2, 2, 1]). Increasing depth/width yielded negligible improvements in raw testing performance and occasionally degraded performance after symbolification due to overfitting. The parsimonious [n, 2, 1] or [n, 1] structures proved optimal.

**Comparison with baselines:**

- Sym-KAN consistently outperformed both MLP and LSTM in almost all sub-periods and configurations, despite utilizing nearly 1,000 times fewer parameters than LSTM.
- In Exp III (Full Period), Sym-KAN achieved the lowest MSE (3.582) compared to MLP (3.617) and LSTM (3.879), reducing error by 0.9% and 7.6% respectively.

## 6. Direct quotations

- "Based on a parsimonious architecture with symbolic functions, KAN expresses a forecast of the VIX as a closed-form in terms of explanatory variables, and provide interpretable insights into key characteristics of the VIX, including mean reversion and the leverage effect." (p. 1)
- "Whereas these statistical models presuppose linearity, KAN identifies such structures directly from data, imposing no prior assumptions on its functional form." (p. 6)
- "Remarkably, KAN's parsimonious design consistently achieves performance on par with these more parameter-intensive MLP-based networks, underscoring its robustness and stability in noisy financial time-series data." (p. 14)
- "This finding suggests that the relationships between the VIX and the explanatory variables used in our experiments are fundamentally linear, a structure that KAN uncovers autonomously without prior specification." (p. 16)
- "By addressing the black-box limitations of traditional deep learning, KAN bridges the gap between predictive accuracy and theoretical transparency, offering a valuable tool for practitioners and researchers in finance." (p. 19)

## 7. Authors' conclusions

The authors conclude that Kolmogorov-Arnold Networks (KANs) serve as a highly effective, interpretable, and parsimonious alternative to traditional black-box neural networks (like MLPs and LSTMs) for financial time-series forecasting. By replacing fixed activation functions with learnable B-splines and subsequently extracting explicit closed-form mathematical equations via symbolification, KANs inherently prune irrelevant noise and prevent overfitting. The extracted formulas successfully and autonomously validate established financial stylized facts, explicitly demonstrating the mean-reverting property of the VIX (through heavy weighting of the first lag) and the leverage effect (through a downward-sloping activation relative to S&P 500 returns). Ultimately, KAN achieves out-of-sample predictive accuracy that matches or exceeds heavily parameterized deep learning baselines while using a fraction of the computational parameters, making it an invaluable tool for transparent financial risk management.

## 8. Limitations stated by authors

[not found] (The authors do not include an explicit "Limitations" section, though they briefly note in the appendix that forcing symbolification on highly non-linear, overfitted splines can occasionally degrade out-of-sample performance.)

## 9. Future work suggested by authors

(The paper concludes with the broad potential of KANs as a valuable tool for practitioners, but does not explicitly outline specific future research steps for the authors to undertake.)

## Additional critical details

- **Extracted Formula Example**: In Exp I (Full Period), the symbolification process reduced the entire network to a specific closed-form equation approximating $\hat{V}_t = 0.86 V_{t-1} + 0.04 V_{t-2} + 0.05 V_{t-3} - 0.03 V_{t-4} + 0.06 V_{t-5} + 0.40$, perfectly reflecting the dominant role of the first lag in volatility persistence.
- **Two-Stage Optimization**: Training KANs involves a dual-stage process: first optimizing the B-splines using L-BFGS to capture continuous patterns, then snapping these curves to symbolic functions and retraining purely the affine transformation parameters (a, b, c, d) to lock in the final mathematical expression.

---
**Extraction date**: 2026-05-19
**PDF source**: VIX_Forecasting_with_KANs.pdf

---

# Article record: KASPER: Kolmogorov Arnold Networks for Stock Prediction and Explainable Regimes

## Metadata
- **BibTeX key**: oad_kasper_2025
- **Authors**: Vidhi Oad, Param Pathak, Nouhaila Innan, Shalini D, Muhammad Shafique
- **Year**: 2025
- **Title**: KASPER: Kolmogorov Arnold Networks for Stock Prediction and Explainable Regimes
- **Journal/Conference**: arXiv preprint
- **DOI/URL**: arXiv:2507.18983v1 [cs.LG]
- **Keywords**: Kolmogorov Arnold Networks, Stock Prediction, Explainable Regimes, Sparse Spline Approximation, Symbolic Rule Extraction

## 1. Theoretical foundation

**Key theorems:**

- **Kolmogorov-Arnold Representation Theorem**: States that any multivariate continuous function can be decomposed into a finite sum of univariate functions. KANs use this theorem to approximate high-dimensional mappings via learnable spline-based transformations on edges.

**Key mathematical definitions:**

- **Markov Regime-Switching Models**: $y_t = \mu_{z_t} + \epsilon_t$, $P(z_t=j|z_{t-1}=i) = A_{ij}$, where $y_t$ is the observed return, $z_t$ is the latent regime indicator, and $A$ is the transition matrix.
- **Regime-dependent returns**: $r_t | z_t=k \sim \mathcal{N}(\mu_k, \sigma_k^2)$.
- **Logistic Transition Function (VLSTAR)**: $G_t(s_t; \gamma, c) = [1 + \exp(-\gamma(s_t-c))]^{-1}$, where $s_t$ triggers regime shifts and $\gamma$ controls transition smoothness.

**Key lemmas or propositions**: [not found]

**Relevant mathematical proofs (summary)**: [not found]

## 2. Proposed method / architecture / model

**Name of method/model**: KASPER (Kolmogorov-Arnold networks for Stock Prediction and Explainable Regimes)

**Core mathematical formulation:**

- **Input State Matrix**: $S_{t,i} = [\ln(\frac{p_{t-i+1}}{p_{t-i}}), \ln(\frac{f_{t-i+1}^{High}}{f_{t-i}^{High}}), \ln(\frac{f_{t-i+1}^{Low}}{f_{t-i}^{Low}}), \dots]$
- **Spline Activation Function**: $f(x) = L(x) + C(x)$
  - **Linear Component**: $L(x) = \sum_{i=0}^{N_{max}-1} \tanh(w_i)\{ReLU[(x_{norm}-k_i) - (x_{norm}-k_{i+1})]\}$
  - **Cubic Component**: $C(x) = \sum_{i=0}^{N_{obj}-1} \sigma(v_i) x_{norm}^3$
- **Gumbel Softmax (Regime Classification)**: $P_t^{(i)} = \frac{\exp(f_i(S_t)/\tau)}{\sum_{j=1}^k \exp(f_j(S_t)/\tau)}$
- **Contrastive Loss**: $\mathcal{L}_{contrastive} = \mathbb{E}[||z_i - z_j||^2 \cdot y_{ij}]$ (where $y_{ij}=1$ if samples share the same regime, 0 otherwise).
- **Orthogonality Regularization**: $\mathcal{L}_{orth} = \mathbb{E}[||W_r W_r^T - I||_F^2]$
- **Forecasting Model**: $\hat{y}_t^{(i)} = \sum_{j=1}^a w_j^{(i)} \phi_j^{(i)}(S_t)$
- **Spline Basis Function**: $\phi_j^{(i)}(S_t) = \sum_{k=1}^K \beta_{j,k}^{(i)} B_k(S_t; \xi^{(i)})$
- **Sparsity Enforcement ($l_1$)**: $w_j^{(i)} \propto ReLU(|w_j^{(i)}| - \theta^{(i)})$
- **Composite Loss Function**: $\mathcal{L} = \mathcal{L}_{Huber} + \lambda \sum |w_j^{(i)}| + \lambda_{contr} \mathcal{L}_{contrastive} + \lambda_{orth} \mathcal{L}_{orth} + \lambda_b \mathcal{L}_{balance}$
- **Shapley Value Estimation**: $\hat{\phi}_j = \frac{1}{N}\sum_{i=1}^N [f(S_i \cup \{j\}) - f(S_i)]$
- **Temporal Weighting Scheme**: $\tilde{\phi}_j = \sum_{t=1}^T w_t \phi_j^t$ where $w_t = \frac{\gamma^{T-t}}{\sum_{t=1}^T \gamma^{T-t}}$
- **Rule Extraction**: $\mathcal{R}_k = \{ \arg\max_{j \in F} |\tilde{\phi}_j^{(k)}| \mid j=1,2,3 \}$ yielding rules like $X_{j_1} + X_{j_2} + X_{j_3} \rightarrow Y_k$.

**Algorithm pseudocode (Algorithm 1: KASPER):**

```
1. Preprocessing & Feature Engineering: Load data, forward-fill missing values.
   Extract 15 features, compute target y_t = (C_{t+1} - C_t)/C_t.
   Split data (70/15/15), apply SelectKBest, and standardize.
2. Architecture: Define Spline Activation.
   Create Layer 1 (Regime Detection) using Gumbel-Softmax.
   Initialize Layer 2 (Regime Adaptive Forecasting) with regime-specific basis functions.
3. Training: Iterate through epochs. Compute predictions, regime probabilities, and embeddings.
   Calculate composite loss (L_pred + lambda_c * L_contrastive + ...).
   Backpropagate via AdamW. Apply early stopping.
4. Rule Extraction: For each regime, compute Monte Carlo Shapley values
   and select the top-3 contributing features.
5. Evaluation: Compute financial metrics (Sharpe Ratio, Max Drawdown, etc.) on the test set.
```

**Architecture details:**

- Two main KAN layers.
- Number of Regimes ($K$): 3.
- Hidden Dimension: 64.
- Spline Configuration: Hybrid linear (3) and cubic (2) splines with quantile-based knot initialization.
- Activation Functions: Spline Activation and GELU.

**Innovative components:**

- Integration of an unsupervised Gumbel-Softmax regime detector with sparse, spline-based KAN forecasting layers to adapt to volatile market conditions.
- Application of Monte Carlo Shapley values equipped with a temporal weighting scheme ($\gamma$) to extract explicit, regime-specific symbolic rules (top-3 drivers per regime).
- Implementation of orthogonality constraints to prevent feature collapse across regimes, paired with contrastive regularization to maximize inter-regime separation.

## 3. Data

**Dataset name(s)**: Yahoo Finance Dataset

**Data source**: Kaggle/Yahoo Finance

**Data size**: Time range: 2018-2023. Frequency: Daily.

**Data preprocessing:**

- Forward-filling of missing values.
- Target formulation: $y_t = \frac{C_{t+1} - C_t}{C_t}$ (next-day percentage return).
- 15 engineered features extracted (lags, rolling statistics, volatility, volume, momentum, temporal dummies).
- Feature selection: Reduced to 8 features using SelectKBest with f_regression.
- Scaling: StandardScaler applied to both features and the target.
- Strict closed-left windows used for rolling calculations to prevent lookahead bias.

**Train/validation/test split**: 70% Training, 15% Validation, 15% Testing (Temporal split).

**Data augmentation techniques**: [not found]

## 4. Experimental setup

**Baseline models compared against:**

- RF Monte Carlo
- Single Layer LSTM
- LSTM + KAN
- AE-LSTM DRL (Autoencoder-LSTM + Deep Reinforcement Learning)
- VLSTAR
- AGNES
- DQN

**Evaluation metrics:**

- **Statistical**: $R^2$, MAE, MSE, RMSE.
- **Financial**:
  - Sharpe Ratio: $\frac{\mu_r - r_f}{\sigma_r}$ (annualized using $\sqrt{252}$).
  - Max Drawdown (MDD): $\min_t(\frac{V_t}{\max_{s\le t}V_s} - 1)$.
  - Cumulative Return: $\prod_{t=1}^N (1+r_t) - 1$.
  - Win Rate: $\frac{N_+}{N} \times 100$.
  - Direction Accuracy: $\frac{1}{N} \sum_{t=1}^N \mathcal{H}[\text{sign}(\hat{y}_t) = \text{sign}(y_t)] \times 100$.
  - Profit Factor.

**Hyperparameters:**

- Optimizer: AdamW (learning rate = 0.001, weight_decay = 1e-5).
- Batch Size: 32.
- Epochs: 100 (Early stopping patience = 15).
- Loss Function: Huber loss with composite regularization.
- Regularization Weights: Contrastive $\lambda_c=0.01$, Sparsity $\lambda_s=0.001$, Orthogonality $\lambda_o=0.01$, Regime Balance $\lambda_b=0.05$.
- Learning Rate Scheduler: ReduceLROnPlateau (factor=0.7, patience=7).
- Gradient Clipping: 0.5.

**Hardware/software environment:**

- Hardware: AMD Ryzen 5 processor, 8 GB RAM, 128 GB SSD, AMD Radeon graphics card.
- Software: Python, PyTorch, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn. GPU acceleration via CUDA.

**Number of runs / seeds**: Results averaged over 5 walk-forward runs.

**Statistical tests performed**: Standard deviations were recorded across the 5 runs to assess statistical consistency/robustness.

## 5. Results

**Quantitative results:**

- **Overall Performance:**
  - $R^2$: $0.8953 \pm 0.0030$
  - MSE: 0.0001
  - RMSE: 0.0046
  - MAE: 0.0033
  - Sharpe Ratio: 12.02
  - Max Drawdown: -0.09%
  - Cumulative Returns: 2.76%
  - Win Rate: 83.17%
  - Profit Factor: 1.53
- **Aggregated Walk-Forward Analysis (Mean $\pm$ Std. Dev.):**
  - Direction Accuracy: 80.94% $\pm$ 7.27%
  - Sharpe Ratio: 11.24 $\pm$ 2.78
  - Max Drawdown: -0.10% $\pm$ 0.01%
  - Cumulative Returns: 2.66% $\pm$ 0.44%
  - Win Rate: 80.94% $\pm$ 7.27%
  - Total Trades: 700
  - Profitable Trades: 566
  - Average Return: 0.0069%
  - Average Win: 0.0102%
  - Average Loss: -0.0070%
  - Profit Factor: 1.45

**Qualitative results:**

- **Regime Distribution**: Regime 0 (Neutral High Confidence) dominated assignments (18.7% of samples). Bearish trends were strongly captured in Regime 2 (9.2%). Low-confidence classifications across all states were extremely rare (< 1.5%), demonstrating high model decisiveness.
- **Feature Attribution:**
  - Regime 0 (Stable): Driven by OC_spread (88.9%) and price_velocity (3.9%).
  - Regime 1 (Transitional/Momentum): OC_spread (83.6%), momentum_state (7.0%), HL_spread (4.9%).
  - Regime 2 (Strong Trend): Extreme dominance of OC_spread (89.4%) and price_velocity (5.4%), with range-based indicators (HL_spread at 1.0%) losing importance.
- **Drawdown Analysis**: Outlier market crashes caused by unexpected news forced sudden standard deviation band widening, but the strategy rebounded instantly; maximum drawdown never exceeded an extreme low of -0.09%.

**Ablation studies**: [not found]

**Comparison with baselines:**

- KASPER vastly outperformed all baselines.
- vs. AE-LSTM DRL: KASPER achieved a Sharpe Ratio of 12.02 compared to 1.85, and reduced Max Drawdown from a devastating -28.11% down to -0.09%.
- vs. Single Layer LSTM: KASPER improved $R^2$ from 0.79 to 0.89 and reduced MSE from 0.00057 to 0.0001.
- vs. RF Monte Carlo: KASPER improved $R^2$ from 0.78 to 0.89.

## 6. Direct quotations

- "To address this, we introduce Kolmogorov-Arnold networks for stock prediction and explainable regimes (KASPER), a novel framework that integrates regime detection, sparse spline-based function modeling, and symbolic rule extraction." (p. 1)
- "Unlike transformer models that mix features and cause high variability, KASPER controls volatility and avoids bad investments during market changes." (p. 7)
- "The evaluation on Yahoo Finance data reveals strong performance, with an $R^2$ of 0.8953, a Sharpe Ratio of 12.02, and an MSE of 0.0001." (p. 10)
- "KASPER addresses these limitations through its orthogonality-constrained regime classification and Monte Carlo Shapley-based interpretability, enabling it to achieve superior performance metrics and provide actionable insights into regime-specific market drivers." (p. 10)
- "The hybrid spline activation functions enable KASPER to model both linear and nonlinear relationships without overfitting, maintaining strong performance during volatile periods where other models falter." (p. 10)

## 7. Authors' conclusions

The KASPER framework successfully resolves the persistent challenge of accurately predicting non-stationary financial markets by integrating unsupervised adaptive regime modeling with sparse Kolmogorov-Arnold Networks. By deploying Gumbel-Softmax regime classification guided by orthogonality constraints and contrastive loss, the model effectively separates distinct market phases (bullish, bearish, neutral) without manual labeling. Furthermore, the use of Monte Carlo Shapley values provides explicit, human-readable rules describing the dominant driving features for each isolated regime. Evaluated on the Yahoo Finance dataset, KASPER achieved state-of-the-art predictive accuracy ($R^2$ = 0.89) alongside an exceptional risk-adjusted profile (Sharpe Ratio = 12.02, Max Drawdown = -0.09%), vastly outperforming complex deep learning baselines. Ultimately, KASPER successfully bridges the gap between black-box accuracy and the critical need for interpretability in real-world financial decision-making.

## 8. Limitations stated by authors

[not found]

## 9. Future work suggested by authors

[not found]

## Additional critical details

- **Temporal Forward-Filling**: To enforce strict temporal causality and prevent lookahead bias in backtesting, the authors explicitly utilized closed-left windows during feature creation and shifted the closing target by -1 timestep.
- **Dynamic Loss Penalty Calibration**: The loss function heavily relies on balancing four distinct penalties. The regime balance penalty ($\lambda_b=0.05$) explicitly prevents the Gumbel Softmax layer from experiencing "regime collapse" (where the model lazily classifies all data into a single regime).

---
**Extraction date**: 2026-05-19
**PDF source**: KASPER__KANs_for_Stock_Prediction_and_Explainable_Regimes.pdf