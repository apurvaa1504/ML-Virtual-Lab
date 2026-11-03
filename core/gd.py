import numpy as np
import pandas as pd

FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
LR_LIST = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
ITER_LIST = [10, 25, 50, 100, 200, 500]


def synthetic_bank(n=45211, seed=42):
    """Fallback dataset with the same numeric columns/distribution shape as bank-full.csv."""
    r = np.random.default_rng(seed)
    pd_ = np.where(r.random(n) < 0.82, -1, r.integers(1, 871, n))
    prev = np.where(pd_ == -1, 0, r.poisson(2.5, n) + 1)
    return pd.DataFrame({
        "age": np.clip(r.normal(41, 10.6, n), 18, 95).astype(int),
        "balance": (r.lognormal(6.5, 1.3, n) - 600).astype(int),
        "day": r.integers(1, 32, n),
        "duration": r.gamma(1.6, 160, n).astype(int),
        "campaign": r.geometric(0.35, n),
        "pdays": pd_,
        "previous": prev,
    })


def load_bank(uploaded=None, path="data/bank-full.csv"):
    import os
    if uploaded is not None:
        return pd.read_csv(uploaded, sep=";"), "Uploaded file"
    if os.path.exists(path):
        return pd.read_csv(path, sep=";"), f"{path} (UCI Bank Marketing)"
    return synthetic_bank(), "Synthetic stand-in (place bank-full.csv in data/ to use the real UCI data)"


def preprocess(df):
    X = df[FEATURES].copy()
    X["pdays"] = X["pdays"].replace(-1, 0)
    X = X.to_numpy(dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, mean, std


def objective_function(theta, X):
    m = X.shape[0]
    return float((1 / (2 * m)) * np.sum((theta - X) ** 2))


def calculate_gradient(theta, X):
    return (1 / X.shape[0]) * np.sum(theta - X, axis=0)


def gradient_descent(X, learning_rate, iterations, theta0=None):
    theta = np.zeros(X.shape[1]) if theta0 is None else np.array(theta0, float)
    costs, thetas = [], []
    with np.errstate(all="ignore"):
        for _ in range(iterations):
            theta = theta - learning_rate * calculate_gradient(theta, X)
            costs.append(objective_function(theta, X))
            thetas.append(theta.copy())
    return theta, costs, np.array(thetas)


def iters_to_converge(costs, tol=1e-6):
    c = np.asarray(costs, float)
    if len(c) < 2 or not np.all(np.isfinite(c)):
        return None
    hit = np.where(np.abs(np.diff(c)) < tol)[0]
    return int(hit[0] + 2) if len(hit) else None


def classify(lr, costs, tol=1e-6):
    c = np.asarray(costs, float)
    if not np.all(np.isfinite(c)) or lr >= 2:
        return False, "Diverges"
    n = iters_to_converge(costs, tol)
    if n is None:
        return False, "Slow - not converged within budget"
    tag = "Oscillating, then stable" if lr > 1 else "Stable"
    return True, f"{tag} (converged in {n} iterations)"
