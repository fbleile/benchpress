from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla


try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


Pair = Tuple[int, int]


@dataclass(frozen=True)
class OptimizerDiagnostics:
    converged: bool
    iterations: int
    objective: float
    score: float
    dag_penalty: float
    trek_penalty: float
    regularizer: float
    backend: str


def _least_squares_value_grad(X: np.ndarray, W: np.ndarray) -> Tuple[float, np.ndarray]:
    n = X.shape[0]
    residual = X @ W - X
    value = 0.5 * float(np.sum(residual * residual)) / float(n)
    grad = X.T @ residual / float(n)
    return value, grad


def _dag_logdet_value_grad(W: np.ndarray, s: float) -> Tuple[float, np.ndarray]:
    d = W.shape[0]
    M = float(s) * np.eye(d) - W * W
    sign, logdet = la.slogdet(M)
    if sign <= 0:
        return float("inf"), np.full_like(W, np.nan)
    try:
        Minv = sla.inv(M)
    except la.LinAlgError:
        return float("inf"), np.full_like(W, np.nan)
    value = -float(logdet) + d * float(np.log(s))
    grad = 2.0 * W * Minv.T
    return value, grad


def _trek_exp_value_grad(W: np.ndarray, pairs: Sequence[Pair]) -> Tuple[float, np.ndarray]:
    if len(pairs) == 0:
        return 0.0, np.zeros_like(W)
    grad = np.zeros_like(W)
    value = 0.0
    for i, j in pairs:
        value += W[i, j] * W[i, j] + W[j, i] * W[j, i]
        grad[i, j] += 2.0 * W[i, j]
        grad[j, i] += 2.0 * W[j, i]
    return float(value), grad


def _regularizer_value_grad(W: np.ndarray, regularizer: str, eps: float = 1e-8) -> Tuple[float, np.ndarray]:
    if regularizer == "none":
        return 0.0, np.zeros_like(W)
    if regularizer == "l1":
        smooth_abs = np.sqrt(W * W + eps)
        return float(np.sum(smooth_abs)), W / smooth_abs
    if regularizer == "l2":
        return float(np.sum(W * W)), 2.0 * W
    raise NotImplementedError(f"regularizer='{regularizer}' is not supported by the optimizer")


def _objective_value_grad(W: np.ndarray, X: np.ndarray, cfg, independence_pairs: Sequence[Pair]):
    if cfg.score != "least_squares":
        raise NotImplementedError("The optimizer currently supports only score='least_squares'")
    if cfg.dag_seq not in {"none", "logdet"}:
        raise NotImplementedError("The optimizer currently supports only dag_seq in {'none', 'logdet'}")
    if cfg.trek_seq not in {"none", "exp"}:
        raise NotImplementedError("The optimizer currently supports only trek_seq in {'none', 'exp'}")

    score, grad = _least_squares_value_grad(X, W)

    dag_value = 0.0
    if cfg.dag_seq == "logdet" and cfg.dag_reg > 0:
        dag_value, dag_grad = _dag_logdet_value_grad(W, cfg.dag_s)
        if not np.isfinite(dag_value) or not np.all(np.isfinite(dag_grad)):
            return float("inf"), grad, score, dag_value, 0.0, 0.0
        grad = grad + cfg.dag_reg * dag_grad

    trek_value = 0.0
    if cfg.trek_seq == "exp" and cfg.trek_reg > 0:
        trek_value, trek_grad = _trek_exp_value_grad(W, independence_pairs)
        grad = grad + cfg.trek_reg * trek_grad

    reg_value, reg_grad = _regularizer_value_grad(W, cfg.regularizer)
    grad = grad + cfg.regularizer_scale * reg_grad

    np.fill_diagonal(grad, 0.0)
    objective = (
        score
        + cfg.dag_reg * dag_value
        + cfg.trek_reg * trek_value
        + cfg.regularizer_scale * reg_value
    )
    return float(objective), grad, score, dag_value, trek_value, reg_value


def fit_notreks_optimizer(
    X: np.ndarray,
    cfg,
    independence_pairs: Sequence[Pair],
    *,
    W_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, OptimizerDiagnostics]:
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    W = np.zeros((d, d), dtype=float) if W_init is None else np.asarray(W_init, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    init_obj, *_ = _objective_value_grad(W, X, cfg, independence_pairs)
    if not np.isfinite(init_obj):
        W = np.zeros((d, d), dtype=float)

    # The schema does not expose a learning-rate knob yet. This conservative
    # value is intentionally fixed until the optimizer interface is stabilized.
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    previous = None
    converged = False
    last_parts = (float("inf"), float("inf"), float("inf"), float("inf"))

    for iteration in range(1, cfg.max_iter + 1):
        objective, grad, score, dag_value, trek_value, reg_value = _objective_value_grad(
            W, X, cfg, independence_pairs
        )
        if not np.isfinite(objective):
            break
        last_parts = (score, dag_value, trek_value, reg_value)

        if previous is not None:
            denom = max(abs(previous), 1.0)
            if abs(previous - objective) / denom <= cfg.tol:
                converged = True
                break
        previous = objective

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        step = (m / (1.0 - beta1**iteration)) / (np.sqrt(v / (1.0 - beta2**iteration)) + 1e-8)

        step_size = lr
        accepted = False
        for _ in range(20):
            candidate = W - step_size * step
            np.fill_diagonal(candidate, 0.0)
            cand_obj, *_ = _objective_value_grad(candidate, X, cfg, independence_pairs)
            if np.isfinite(cand_obj) and cand_obj <= objective + 1e-10:
                W = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break

    final_obj, _, score, dag_value, trek_value, reg_value = _objective_value_grad(W, X, cfg, independence_pairs)
    if np.isfinite(final_obj):
        last_parts = (score, dag_value, trek_value, reg_value)
    else:
        final_obj = previous if previous is not None else float("inf")
        score, dag_value, trek_value, reg_value = last_parts

    diagnostics = OptimizerDiagnostics(
        converged=converged,
        iterations=iteration,
        objective=float(final_obj),
        score=float(score),
        dag_penalty=float(cfg.dag_reg * dag_value),
        trek_penalty=float(cfg.trek_reg * trek_value),
        regularizer=float(cfg.regularizer_scale * reg_value),
        backend="numpy/scipy",
    )
    return W, diagnostics
