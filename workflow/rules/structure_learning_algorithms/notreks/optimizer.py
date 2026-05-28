from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla


try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jsla

    jax.config.update("jax_enable_x64", True)

    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


Pair = Tuple[int, int]
L1_SMOOTH_EPS = 1e-8
DOMAIN_TOL = 1e-12
TREK_INV_EPS = 1e-8


@dataclass(frozen=True)
class OptimizerDiagnostics:
    converged: bool
    iterations_total: int
    path_steps_completed: int
    final_mu: float
    objective: float
    score: float
    dag_penalty: float
    trek_penalty: float
    regularizer: float
    backend: str
    stages: List[Dict[str, float]] = field(default_factory=list)


def _least_squares_value_grad(X: np.ndarray, W: np.ndarray) -> Tuple[float, np.ndarray]:
    n = X.shape[0]
    residual = X @ W - X
    value = 0.5 * float(np.sum(residual * residual)) / float(n)
    grad = X.T @ residual / float(n)
    return value, grad


def _gaussian_likelihood_value_grad(X: np.ndarray, W: np.ndarray) -> Tuple[float, np.ndarray]:
    n, d = X.shape
    residual = X - X @ W
    sigma2 = np.mean(residual * residual, axis=0) + 1e-12
    value = 0.5 * float(np.sum(np.log(sigma2)))
    grad = np.zeros((d, d), dtype=float)
    for j in range(d):
        grad[:, j] = -(X.T @ residual[:, j]) / (float(n) * sigma2[j])
    return value, grad


def _score_value_grad(X: np.ndarray, W: np.ndarray, score: str) -> Tuple[float, np.ndarray]:
    if score == "least_squares":
        return _least_squares_value_grad(X, W)
    if score == "gaussian_likelihood":
        return _gaussian_likelihood_value_grad(X, W)
    raise NotImplementedError(f"score='{score}' is not supported by the optimizer")


def _logdet_domain_inverse(W: np.ndarray, s: float) -> Tuple[bool, Optional[np.ndarray], float]:
    d = W.shape[0]
    M = float(s) * np.eye(d) - W * W
    sign, logdet = la.slogdet(M)
    if sign <= 0:
        return False, None, float("inf")
    try:
        Minv = sla.inv(M)
    except la.LinAlgError:
        return False, None, float("inf")
    if np.min(Minv) < -DOMAIN_TOL:
        return False, None, float("inf")
    return True, Minv, float(logdet)


def _dag_value_grad(W: np.ndarray, seq: str, s: float) -> Tuple[float, np.ndarray]:
    d = W.shape[0]
    if seq == "none":
        return 0.0, np.zeros_like(W)
    if seq == "logdet":
        ok, Minv, logdet = _logdet_domain_inverse(W, s)
        if not ok or Minv is None:
            return float("inf"), np.full_like(W, np.nan)
        value = -logdet + d * float(np.log(s))
        grad = 2.0 * W * Minv.T
        return float(value), grad
    if seq == "exp":
        A = W * W
        E = sla.expm(A)
        value = float(np.trace(E) - d)
        grad = 2.0 * W * E.T
        return value, grad
    if seq in {"log", "inv"}:
        raise NotImplementedError(f"dag_seq='{seq}' is not implemented in the optimizer")
    raise NotImplementedError(f"dag_seq='{seq}' is not supported by the optimizer")


def _series_I_minus_log_I_minus_W_jax(A, K: int):
    d = A.shape[0]
    F = jnp.eye(d, dtype=A.dtype)
    power = A
    for k in range(1, int(K) + 1):
        F = F + power / float(k)
        power = power @ A
    return F


def _trek_value_jax(W, pairs, seq: str, K_log: int, eps_inv: float):
    d = W.shape[0]
    A = W * W
    if seq == "exp":
        F = jsla.expm(A)
    elif seq == "inv":
        I = jnp.eye(d, dtype=W.dtype)
        F = jnp.linalg.solve(I - A + float(eps_inv) * I, I)
    elif seq == "log":
        F = _series_I_minus_log_I_minus_W_jax(A, K_log)
    else:
        raise NotImplementedError(f"trek_seq='{seq}' is not implemented in the optimizer")

    H = F.T @ F
    rows = pairs[:, 0]
    cols = pairs[:, 1]
    return jnp.sum(0.5 * (H[rows, cols] + H[cols, rows]))


def _trek_value_grad_jax(W: np.ndarray, seq: str, pairs: Sequence[Pair]) -> Tuple[float, np.ndarray]:
    pairs_np = np.asarray(pairs, dtype=np.int64)
    if pairs_np.size == 0:
        return 0.0, np.zeros_like(W)
    if pairs_np.ndim != 2 or pairs_np.shape[1] != 2:
        raise ValueError("independence_pairs must have shape (m, 2)")

    K_log = 2 * W.shape[0]
    W_jax = jnp.asarray(W, dtype=jnp.float64)
    pairs_jax = jnp.asarray(pairs_np, dtype=jnp.int32)

    try:
        value, grad = jax.value_and_grad(_trek_value_jax)(
            W_jax,
            pairs_jax,
            seq,
            K_log,
            TREK_INV_EPS,
        )
    except Exception:
        return float("inf"), np.full_like(W, np.nan)

    value_np = float(np.asarray(value))
    grad_np = np.asarray(grad, dtype=float)
    if not np.isfinite(value_np) or not np.all(np.isfinite(grad_np)):
        return float("inf"), np.full_like(W, np.nan)
    return value_np, grad_np


def _trek_value_grad(W: np.ndarray, seq: str, pairs: Sequence[Pair]) -> Tuple[float, np.ndarray]:
    if seq == "none" or len(pairs) == 0:
        return 0.0, np.zeros_like(W)
    if seq not in {"exp", "inv", "log"}:
        raise NotImplementedError(f"trek_seq='{seq}' is not implemented in the optimizer")
    if not JAX_AVAILABLE:
        raise ImportError(
            f"trek_seq='{seq}' requires JAX for matrix-function gradients. "
            "Install with: pip install jax jaxlib"
        )
    return _trek_value_grad_jax(W, seq, pairs)


def _regularizer_value_grad(W: np.ndarray, regularizer: str) -> Tuple[float, np.ndarray]:
    if regularizer == "none":
        return 0.0, np.zeros_like(W)
    if regularizer == "l1":
        smooth_abs = np.sqrt(W * W + L1_SMOOTH_EPS)
        return float(np.sum(smooth_abs)), W / smooth_abs
    if regularizer == "l2":
        return float(np.sum(W * W)), 2.0 * W
    raise NotImplementedError(f"regularizer='{regularizer}' is not supported by the optimizer")


def _stage_objective_value_grad(
    W: np.ndarray,
    X: np.ndarray,
    cfg,
    independence_pairs: Sequence[Pair],
    *,
    mu: float,
    s: float,
):
    score_value, score_grad = _score_value_grad(X, W, cfg.score)
    reg_value, reg_grad = _regularizer_value_grad(W, cfg.regularizer)
    dag_value, dag_grad = _dag_value_grad(W, cfg.dag_seq, s)
    trek_value, trek_grad = _trek_value_grad(W, cfg.trek_seq, independence_pairs)

    if (
        not np.isfinite(dag_value)
        or not np.all(np.isfinite(dag_grad))
        or not np.isfinite(trek_value)
        or not np.all(np.isfinite(trek_grad))
    ):
        return float("inf"), np.full_like(W, np.nan), score_value, dag_value, trek_value, reg_value

    grad = (
        float(mu) * (score_grad + cfg.regularizer_scale * reg_grad)
        + cfg.dag_reg * dag_grad
        + cfg.trek_reg * trek_grad
    )
    np.fill_diagonal(grad, 0.0)

    objective = (
        float(mu) * (score_value + cfg.regularizer_scale * reg_value)
        + cfg.dag_reg * dag_value
        + cfg.trek_reg * trek_value
    )
    return float(objective), grad, score_value, dag_value, trek_value, reg_value


def _choose_s(stage: int, cfg) -> float:
    if cfg.dag_seq != "logdet":
        return float(cfg.dag_s)
    return max(float(cfg.dag_s) - 0.1 * float(stage), 0.5)


def minimize_stage(
    *,
    W: np.ndarray,
    X: np.ndarray,
    cfg,
    independence_pairs: Sequence[Pair],
    mu: float,
    s: float,
    max_iter: int,
    lr: float,
) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    current = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(current, 0.0)

    if cfg.dag_seq == "logdet":
        ok, _, _ = _logdet_domain_inverse(current, s)
        if not ok:
            current = np.zeros_like(current)

    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(current)
    v = np.zeros_like(current)
    previous = None
    converged = False
    objective = float("inf")
    parts = (float("inf"), float("inf"), float("inf"), float("inf"))

    for iteration in range(1, int(max_iter) + 1):
        objective, grad, score, dag_value, trek_value, reg_value = _stage_objective_value_grad(
            current, X, cfg, independence_pairs, mu=mu, s=s
        )
        if not np.isfinite(objective) or not np.all(np.isfinite(grad)):
            return current, False, {
                "iterations": iteration,
                "objective": float("inf"),
                "success": False,
                "converged": False,
            }
        parts = (score, dag_value, trek_value, reg_value)

        if previous is not None:
            denom = max(abs(previous), 1.0)
            if abs(previous - objective) / denom <= cfg.tol:
                converged = True
                break
        previous = objective

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        step = (m / (1.0 - beta1**iteration)) / (np.sqrt(v / (1.0 - beta2**iteration)) + 1e-8)

        step_size = float(lr)
        accepted = False
        for _ in range(25):
            candidate = current - step_size * step
            np.fill_diagonal(candidate, 0.0)
            cand_obj, *_ = _stage_objective_value_grad(
                candidate, X, cfg, independence_pairs, mu=mu, s=s
            )
            if np.isfinite(cand_obj) and cand_obj <= objective + 1e-10:
                current = candidate
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            return current, False, {
                "iterations": iteration,
                "objective": float(objective),
                "success": False,
                "converged": converged,
            }

    score, dag_value, trek_value, reg_value = parts
    stage_diag = {
        "iterations": int(iteration),
        "objective": float(objective),
        "score": float(score),
        "dag_penalty": float(cfg.dag_reg * dag_value),
        "trek_penalty": float(cfg.trek_reg * trek_value),
        "regularizer": float(cfg.regularizer_scale * reg_value),
        "success": True,
        "converged": bool(converged),
    }
    return current, True, stage_diag


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

    mu = float(cfg.mu_init)
    lr = float(cfg.lr)
    stages: List[Dict[str, float]] = []
    iterations_total = 0
    path_steps_completed = 0
    converged = False

    for stage in range(int(cfg.path_steps)):
        s_stage = _choose_s(stage, cfg)
        inner_iter = int(cfg.max_iter) if stage == int(cfg.path_steps) - 1 else int(cfg.warm_iter)
        stage_lr = lr
        stage_s = s_stage
        success = False
        stage_diag: Dict[str, float] = {}

        for retry in range(6):
            W_candidate, success, stage_diag = minimize_stage(
                W=W,
                X=X,
                cfg=cfg,
                independence_pairs=independence_pairs,
                mu=mu,
                s=stage_s,
                max_iter=inner_iter,
                lr=stage_lr,
            )
            if success:
                W = W_candidate
                break
            stage_lr *= 0.5
            if cfg.dag_seq == "logdet":
                stage_s += 0.1
            stage_diag["retry"] = retry + 1

        iterations_total += int(stage_diag.get("iterations", 0))
        stage_diag.update({
            "stage": int(stage),
            "mu": float(mu),
            "s": float(stage_s),
            "lr": float(stage_lr),
            "success": bool(success),
        })
        stages.append(stage_diag)

        if not success:
            break

        path_steps_completed += 1
        converged = bool(stage_diag.get("converged", False))
        mu *= float(cfg.mu_factor)

    final_s = _choose_s(max(path_steps_completed - 1, 0), cfg)
    objective, _, score, dag_value, trek_value, reg_value = _stage_objective_value_grad(
        W, X, cfg, independence_pairs, mu=max(mu, 1e-300), s=final_s
    )

    diagnostics = OptimizerDiagnostics(
        converged=bool(converged and path_steps_completed == int(cfg.path_steps)),
        iterations_total=int(iterations_total),
        path_steps_completed=int(path_steps_completed),
        final_mu=float(mu),
        objective=float(objective),
        score=float(score),
        dag_penalty=float(cfg.dag_reg * dag_value),
        trek_penalty=float(cfg.trek_reg * trek_value),
        regularizer=float(cfg.regularizer_scale * reg_value),
        backend="numpy/scipy+jax-available" if JAX_AVAILABLE else "numpy/scipy",
        stages=stages,
    )
    return W, diagnostics
