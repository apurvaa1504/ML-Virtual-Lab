AIM = """**Aim:** Implement the Gradient Descent algorithm on a given dataset and analyze convergence behavior,
learning-rate effects, and optimization performance."""

THEORY_GRADIENT = r"""
### 1.1 What is a Gradient?
For a differentiable function $J(\theta_1,\dots,\theta_n)$, the **gradient** is the vector of partial derivatives

$$\nabla J(\theta)=\left[\frac{\partial J}{\partial\theta_1},\dots,\frac{\partial J}{\partial\theta_n}\right]^T$$

It points in the direction of steepest **increase** of $J$; its negative is the direction of steepest decrease.
"""
THEORY_GD = r"""
### 1.2 Gradient Descent
Gradient Descent is an iterative first-order optimization algorithm. Starting from $\theta^{(0)}$:

$$\theta^{(t+1)}=\theta^{(t)}-\alpha\,\nabla J\!\left(\theta^{(t)}\right)$$

where $\alpha$ is the **learning rate** (step size).

**Objective used in this experiment** (no regression/classification model, the label `y` is unused):

$$J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}\lVert\theta-X_i\rVert^2,\qquad
\nabla J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(\theta-X_i)=\theta-\bar X$$

The minimizer is $\theta^*=\bar X$ (the feature means; $0$ after standardization).
"""
THEORY_CONV = r"""
### 1.3 Convergence behavior, learning-rate effects and optimization performance
Substituting the gradient into the update rule gives a closed form:

$$\theta^{(t)}-\theta^*=(1-\alpha)^t\,(\theta^{(0)}-\theta^*)$$

| Learning rate | Behavior |
|---|---|
| $0<\alpha<1$ | Monotonic, smooth convergence; small $\alpha$ is slow |
| $\alpha=1$ | Converges in a single step (for this objective) |
| $1<\alpha<2$ | Overshoots the minimum, oscillates, still converges |
| $\alpha\ge 2$ | Diverges ($\lvert 1-\alpha\rvert\ge 1$) |

**Convergence criterion:** stop when $\lvert J_t-J_{t-1}\rvert<\epsilon$.

**Performance measures:** initial objective, final objective, iterations to converge, learning rate, parameter change per step.
"""
DATASET = """**Bank Marketing Dataset** (UCI Machine Learning Repository, `bank-full.csv`, 45,211 rows, `;` separated).
Only the numerical attributes are used: `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`.
`pdays = -1` (not previously contacted) is replaced by `0`; features are standardized to zero mean and unit variance."""

PROCEDURE = [
    "Open **Simulation** and load the dataset (bundled file, upload, or fallback).",
    "Inspect shape, dtypes, missing values; confirm preprocessing and standardization.",
    "Run Gradient Descent with `lr = 0.1`, `iterations = 100`; record initial/final parameters and final objective.",
    "Plot objective vs iterations and judge: converges? how fast? smooth or unstable?",
    "Inspect the convergence of each parameter.",
    "Compare learning rates 0.001 to 1.0 (try 1.5 and 2.1 to see oscillation and divergence).",
    "Vary the number of iterations (10 to 500) and decide whether more iterations help.",
    "Apply the criterion `|J_t - J_(t-1)| < tolerance`, fill the result table, and write your conclusion.",
]

PRETEST = [
    ("The gradient of a function points toward…", ["Steepest decrease", "Steepest increase", "Zero change", "The minimum always"], 1, "The gradient is the direction of steepest ascent; GD moves opposite to it."),
    ("In θ ← θ − α∇J(θ), α is called…", ["Momentum", "Regularizer", "Learning rate", "Bias"], 2, "α controls the step size."),
    ("Why standardize features before GD?", ["To make labels binary", "To give comparable scales", "To remove the gradient", "To add noise"], 1, "Comparable scales give well-conditioned, stable optimization."),
    ("A very small learning rate typically causes…", ["Divergence", "Slow convergence", "Instant convergence", "Random behavior"], 1, "Tiny steps need many iterations."),
    ("Gradient Descent is a ___ order method.", ["Zeroth", "First", "Second", "Third"], 1, "It uses only first derivatives."),
]
POSTTEST = [
    ("For J(θ)=1/(2m)Σ‖θ−Xᵢ‖², the minimizer is…", ["θ = 0 always", "The mean of Xᵢ", "The median of Xᵢ", "The max of Xᵢ"], 1, "∇J = θ − mean(X) = 0 at θ = mean."),
    ("For this objective, α = 1.0 converges in…", ["One step", "100 steps", "Never", "Exactly 10 steps"], 0, "(1−α)^t = 0 for α = 1."),
    ("For which α does GD diverge here?", ["0.5", "1.0", "1.9", "2.1"], 3, "Divergence when |1−α| ≥ 1, i.e. α ≥ 2."),
    ("With 1 < α < 2 the objective curve…", ["Is monotone and slow", "Oscillates in θ but still converges", "Diverges", "Stays constant"], 1, "1−α is negative, so θ alternates sides while shrinking."),
    ("Converged criterion used in this lab:", ["J_t < 0", "|J_t − J_(t−1)| < ε", "θ = 0", "Gradient > 1"], 1, "A small change in objective between iterations signals convergence."),
]
REFERENCES = [
    "UCI Machine Learning Repository: Bank Marketing Dataset - https://archive.ics.uci.edu/dataset/222/bank+marketing",
    "S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.",
    "I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016 (Ch. 4, 8).",
    "S. Ruder, *An overview of gradient descent optimization algorithms*, arXiv:1609.04747.",
]
