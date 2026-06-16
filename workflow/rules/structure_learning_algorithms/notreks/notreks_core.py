from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


Pair = Tuple[int, int]


@dataclass(frozen=True)
class NotreksConfig:
    algorithm_id: str
    function_class: str
    score: str
    dag_seq: str
    dag_reg: float
    dag_s: float
    trek_seq: str
    trek_reg: float
    regularizer: str
    regularizer_scale: float
    independence_test: str
    independence_alpha: float
    independence_correction: str
    seed: int
    max_iter: int
    lr: float
    path_steps: int
    mu_init: float
    mu_factor: float
    warm_iter: int
    tol: float
    threshold: float
    timeout: Optional[float]
    init: str = "zero"
    checkpoint: int = 1000


def _soft_threshold(values: np.ndarray, scale: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - scale, 0.0)


def _fit_target(X: np.ndarray, target: int, regularizer: str, regularizer_scale: float) -> np.ndarray:
    n, d = X.shape
    parents = [i for i in range(d) if i != target]
    design = X[:, parents]
    y = X[:, target]

    intercept = np.ones((n, 1), dtype=float)
    design_with_intercept = np.hstack([intercept, design])

    if regularizer == "l2" and regularizer_scale > 0:
        penalty = np.eye(design_with_intercept.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = design_with_intercept.T @ design_with_intercept + regularizer_scale * penalty
        rhs = design_with_intercept.T @ y
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    else:
        beta, *_ = np.linalg.lstsq(design_with_intercept, y, rcond=None)

    coefs = beta[1:]
    if regularizer == "l1" and regularizer_scale > 0:
        coefs = _soft_threshold(coefs, regularizer_scale)

    result = np.zeros(d, dtype=float)
    result[parents] = coefs
    return result


def fit_linear_baseline(
    X: np.ndarray,
    *,
    score: str,
    regularizer: str,
    regularizer_scale: float,
    independence_pairs: Sequence[Pair],
    rng: np.random.Generator,
) -> np.ndarray:
    if score not in {"least_squares", "gaussian_likelihood"}:
        raise ValueError(f"Unsupported score: {score}")
    if regularizer not in {"none", "l1", "l2"}:
        raise ValueError(f"Unsupported regularizer: {regularizer}")

    del rng
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input data must be a two-dimensional array")

    _, d = X.shape
    W = np.zeros((d, d), dtype=float)
    for target in range(d):
        W[:, target] = _fit_target(X, target, regularizer, regularizer_scale)

    np.fill_diagonal(W, 0.0)
    for i, j in independence_pairs:
        W[i, j] = 0.0
        W[j, i] = 0.0
    return W


def threshold_adjacency(W: np.ndarray, threshold: float) -> np.ndarray:
    adjmat = (np.abs(W) > float(threshold)).astype(int)
    np.fill_diagonal(adjmat, 0)
    return adjmat
