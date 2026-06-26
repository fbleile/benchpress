from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jsla

    jax.config.update("jax_enable_x64", True)

    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


Pair = Tuple[int, int]
DOMAIN_TOL = 1e-12
TREK_INV_EPS = 1e-8
W_STAT_THRESHOLDS = (0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5)
POWER_ITERATION_STEPS = 5
POWER_ITERATION_EPS = 1e-12
SCC_THRESHOLD = 1e-8


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
    raw_score: float
    raw_dag_penalty: float
    raw_trek_penalty: float
    raw_regularizer: float
    scaled_score: float
    scaled_dag_penalty: float
    scaled_trek_penalty: float
    scaled_regularizer: float
    grad_norm: float
    grad_score_norm: float
    grad_dag_norm: float
    grad_trek_norm: float
    grad_regularizer_norm: float
    n_samples: int
    n_variables: int
    num_independence_pairs: int
    backend: str
    stages: List[Dict[str, float]] = field(default_factory=list)


def _least_squares_value_grad(X: np.ndarray, W: np.ndarray) -> Tuple[float, np.ndarray]:
    X = X - np.mean(X, axis=0, keepdims=True)
    n = X.shape[0]
    residual = X @ W - X
    value = 0.5 * float(np.sum(residual * residual)) / float(n)
    grad = (X.T @ residual) / float(n)
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


def _dag_value_grad(
    W: np.ndarray,
    seq: str,
    s: float,
    *,
    power_iter_steps: int = POWER_ITERATION_STEPS,
    scc_threshold: float = SCC_THRESHOLD,
) -> Tuple[float, np.ndarray]:
    d = W.shape[0]
    if str(seq) in {"None", "none", "null", ""}:
        seq = "none"
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
    if seq == "scc_power_iteration":
        return _scc_power_iteration_value_grad(
            W,
            power_iter_steps=power_iter_steps,
            scc_threshold=scc_threshold,
        )
    if seq in {"power_iteration", "spectral_radius"}:
        raise NotImplementedError(
            f"Unknown dag_seq='{seq}'. Use dag_seq='scc_power_iteration'."
        )
    if seq in {"log", "inv"}:
        raise NotImplementedError(f"dag_seq='{seq}' is not implemented in the optimizer")
    raise NotImplementedError(f"dag_seq='{seq}' is not supported by the optimizer")


def _normalize_numpy_vector(vector: np.ndarray) -> np.ndarray:
    return vector / float(np.sqrt(np.sum(vector * vector) + POWER_ITERATION_EPS))


def _scc_power_iteration_gradient_proxy(
    W: np.ndarray,
    *,
    power_iter_steps: int,
    scc_threshold: float,
) -> np.ndarray:
    """Return a detached Perron-gradient proxy, blockwise on nontrivial SCCs."""

    if power_iter_steps < 1:
        raise ValueError("power_iter_steps must be positive")
    if scc_threshold < 0:
        raise ValueError("scc_threshold must be non-negative")

    d = W.shape[0]
    A = W * W
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    support = A > float(scc_threshold)
    n_components, labels = connected_components(
        csr_matrix(support),
        directed=True,
        connection="strong",
        return_labels=True,
    )
    G = np.zeros_like(W, dtype=float)
    for component in range(int(n_components)):
        indices = np.flatnonzero(labels == component)
        if indices.size <= 1:
            continue
        block = A[np.ix_(indices, indices)]
        size = block.shape[0]
        u = np.ones(size, dtype=float) / np.sqrt(float(size))
        v = np.ones(size, dtype=float) / np.sqrt(float(size))
        for _ in range(int(power_iter_steps)):
            u = _normalize_numpy_vector(block.T @ u + POWER_ITERATION_EPS)
            v = _normalize_numpy_vector(block @ v + POWER_ITERATION_EPS)
        denom = float(np.dot(u, v) + POWER_ITERATION_EPS)
        G_block = np.outer(u, v) / denom
        G[np.ix_(indices, indices)] = G_block
    np.fill_diagonal(G, 0.0)
    return G


def _scc_power_iteration_surrogate_jax(W, G):
    d = W.shape[0]
    offdiag = 1.0 - jnp.eye(d, dtype=W.dtype)
    A = W * W * offdiag
    return jnp.sum(jax.lax.stop_gradient(G) * A)


def _scc_power_iteration_value_grad(
    W: np.ndarray,
    *,
    power_iter_steps: int = POWER_ITERATION_STEPS,
    scc_threshold: float = SCC_THRESHOLD,
) -> Tuple[float, np.ndarray]:
    if not JAX_AVAILABLE:
        raise ImportError(
            "dag_seq='scc_power_iteration' requires JAX for power-iteration gradients. "
            "Install with: pip install jax jaxlib"
        )
    W_jax = jnp.asarray(W, dtype=jnp.float64)
    G = _scc_power_iteration_gradient_proxy(
        np.asarray(W, dtype=float),
        power_iter_steps=int(power_iter_steps),
        scc_threshold=float(scc_threshold),
    )
    if not np.any(G):
        return 0.0, np.zeros_like(W)
    G_jax = jnp.asarray(G, dtype=jnp.float64)
    try:
        value, grad = jax.value_and_grad(_scc_power_iteration_surrogate_jax)(W_jax, G_jax)
    except Exception:
        return float("inf"), np.full_like(W, np.nan)
    value_np = float(np.asarray(value))
    grad_np = np.asarray(grad, dtype=float)
    if not np.isfinite(value_np) or not np.all(np.isfinite(grad_np)):
        return float("inf"), np.full_like(W, np.nan)
    return value_np, grad_np


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
    return jnp.sum(H[rows, cols])


def _trek_value_grad_jax(W: np.ndarray, seq: str, pairs: Sequence[Pair]) -> Tuple[float, np.ndarray]:
    pairs_np = np.asarray(pairs, dtype=np.int64)
    if pairs_np.size == 0:
        return 0.0, np.zeros_like(W)
    if pairs_np.ndim != 2 or pairs_np.shape[1] != 2:
        raise ValueError("independence_pairs must have shape (m, 2)")
    if np.any(pairs_np < 0) or np.any(pairs_np >= W.shape[0]):
        raise ValueError(f"independence_pairs contain indices outside [0, {W.shape[0] - 1}]")
    if np.any(pairs_np[:, 0] == pairs_np[:, 1]):
        raise ValueError("independence_pairs must not contain diagonal/self pairs")

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
    if seq == "none":
        return 0.0, np.zeros_like(W)
    pairs_np = np.asarray(pairs, dtype=np.int64)
    if pairs_np.size == 0:
        return 0.0, np.zeros_like(W)
    if pairs_np.ndim != 2 or pairs_np.shape[1] != 2:
        raise ValueError("independence_pairs must have shape (m, 2)")
    if np.any(pairs_np < 0) or np.any(pairs_np >= W.shape[0]):
        raise ValueError(f"independence_pairs contain indices outside [0, {W.shape[0] - 1}]")
    if np.any(pairs_np[:, 0] == pairs_np[:, 1]):
        raise ValueError("independence_pairs must not contain diagonal/self pairs")
    if seq not in {"exp", "inv", "log"}:
        raise NotImplementedError(f"trek_seq='{seq}' is not implemented in the optimizer")
    if not JAX_AVAILABLE:
        raise ImportError(
            f"trek_seq='{seq}' requires JAX for matrix-function gradients. "
            "Install with: pip install jax jaxlib"
        )
    return _trek_value_grad_jax(W, seq, pairs_np)


def _w_stats(W: np.ndarray) -> Dict[str, float]:
    vals = np.abs(W[~np.eye(W.shape[0], dtype=bool)])
    nonzero = vals[vals > 0.0]
    stats = {
        "max_abs_w": float(np.max(vals)) if vals.size else 0.0,
        "median_nonzero_abs_w": float(np.median(nonzero)) if nonzero.size else 0.0,
    }
    for threshold in W_STAT_THRESHOLDS:
        stats[f"n_abs_w_gt_{threshold:g}"] = int(np.sum(vals > threshold))
    return stats


def _regularizer_value_grad(W: np.ndarray, regularizer: str) -> Tuple[float, np.ndarray]:
    if regularizer == "none":
        return 0.0, np.zeros_like(W)
    if regularizer == "l1":
        return float(np.sum(np.abs(W))), np.sign(W)
    if regularizer == "l2":
        return float(np.sum(W * W)), 2.0 * W
    raise NotImplementedError(f"regularizer='{regularizer}' is not supported by the optimizer")


def _trek_penalty_scale(cfg, mu: float) -> float:
    mode = getattr(cfg, "trek_penalty_mu_mode", "hard_outside_mu")
    if mode == "hard_outside_mu":
        return float(cfg.trek_reg)
    if mode == "soft_inside_mu":
        return float(mu) * float(cfg.trek_reg)
    raise ValueError(
        "trek_penalty_mu_mode must be one of {'hard_outside_mu', 'soft_inside_mu'}"
    )


def _stage_objective_value_grad(
    W: np.ndarray,
    X: np.ndarray,
    cfg,
    independence_pairs: Sequence[Pair],
    *,
    mu: float,
    s: float,
):
    power_iter_steps = int(getattr(cfg, "power_iter_steps", POWER_ITERATION_STEPS))
    scc_threshold = float(getattr(cfg, "scc_threshold", SCC_THRESHOLD))
    score_value, score_grad = _score_value_grad(X, W, cfg.score)
    reg_value, reg_grad = _regularizer_value_grad(W, cfg.regularizer)
    dag_value, dag_grad = _dag_value_grad(
        W,
        cfg.dag_seq,
        s,
        power_iter_steps=power_iter_steps,
        scc_threshold=scc_threshold,
    )
    trek_value, trek_grad = _trek_value_grad(W, cfg.trek_seq, independence_pairs)

    if (
        not np.isfinite(dag_value)
        or not np.all(np.isfinite(dag_grad))
        or not np.isfinite(trek_value)
        or not np.all(np.isfinite(trek_grad))
    ):
        return float("inf"), np.full_like(W, np.nan), score_value, dag_value, trek_value, reg_value

    trek_scale = _trek_penalty_scale(cfg, mu)
    grad = float(mu) * (score_grad + cfg.regularizer_scale * reg_grad) + trek_scale * trek_grad + cfg.dag_reg * dag_grad
    np.fill_diagonal(grad, 0.0)

    objective = (
        float(mu) * (score_value + cfg.regularizer_scale * reg_value)
        + trek_scale * trek_value
        + cfg.dag_reg * dag_value
    )
    return float(objective), grad, score_value, dag_value, trek_value, reg_value


def _choose_s(stage: int, cfg) -> float:
    if cfg.dag_seq != "logdet":
        return float(cfg.dag_s)
    s_path = getattr(cfg, "s_path", None)
    if s_path is not None:
        if len(s_path) == 0:
            raise ValueError("cfg.s_path must not be empty")
        return float(s_path[min(stage, len(s_path) - 1)])
    return max(float(cfg.dag_s) - 0.1 * float(stage), 0.6)


def _stage_diagnostics(
    W: np.ndarray,
    X: np.ndarray,
    cfg,
    independence_pairs: Sequence[Pair],
    *,
    mu: float,
    s: float,
) -> Dict[str, float]:
    n, d = X.shape
    power_iter_steps = int(getattr(cfg, "power_iter_steps", POWER_ITERATION_STEPS))
    scc_threshold = float(getattr(cfg, "scc_threshold", SCC_THRESHOLD))
    score_value, score_grad = _score_value_grad(X, W, cfg.score)
    reg_value, reg_grad = _regularizer_value_grad(W, cfg.regularizer)
    dag_value, dag_grad = _dag_value_grad(
        W,
        cfg.dag_seq,
        s,
        power_iter_steps=power_iter_steps,
        scc_threshold=scc_threshold,
    )
    trek_value, trek_grad = _trek_value_grad(W, cfg.trek_seq, independence_pairs)
    trek_scale = _trek_penalty_scale(cfg, mu)
    total_grad = float(mu) * (score_grad + cfg.regularizer_scale * reg_grad) + trek_scale * trek_grad + cfg.dag_reg * dag_grad
    np.fill_diagonal(total_grad, 0.0)
    return {
        "score": float(score_value),
        "dag_penalty": float(cfg.dag_reg * dag_value),
        "trek_penalty": float(trek_scale * trek_value),
        "regularizer": float(float(mu) * cfg.regularizer_scale * reg_value),
        "raw_score": float(score_value),
        "raw_dag_penalty": float(dag_value),
        "raw_trek_penalty": float(trek_value),
        "raw_regularizer": float(reg_value),
        "scaled_score": float(float(mu) * score_value),
        "scaled_dag_penalty": float(cfg.dag_reg * dag_value),
        "scaled_trek_penalty": float(trek_scale * trek_value),
        "scaled_regularizer": float(float(mu) * cfg.regularizer_scale * reg_value),
        "grad_norm": float(np.linalg.norm(total_grad)),
        "grad_score_norm": float(np.linalg.norm(float(mu) * score_grad)),
        "grad_dag_norm": float(np.linalg.norm(cfg.dag_reg * dag_grad)),
        "grad_trek_norm": float(np.linalg.norm(trek_scale * trek_grad)),
        "grad_regularizer_norm": float(np.linalg.norm(float(mu) * cfg.regularizer_scale * reg_grad)),
        "n_samples": int(n),
        "n_variables": int(d),
        "num_independence_pairs": int(len(independence_pairs)),
        "least_squares": float(score_value) if cfg.score == "least_squares" else float("nan"),
        "gaussian_likelihood": float(score_value) if cfg.score == "gaussian_likelihood" else float("nan"),
        "l1": float(reg_value) if cfg.regularizer == "l1" else 0.0,
        "l2": float(reg_value) if cfg.regularizer == "l2" else 0.0,
        "regularizer_contribution": float(cfg.regularizer_scale * reg_value),
        "dag_contribution": float(cfg.dag_reg * dag_value),
        "trek_contribution": float(cfg.trek_reg * trek_value),
        **_w_stats(W),
    }


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

    beta1 = float(getattr(cfg, "beta1", 0.99))
    beta2 = float(getattr(cfg, "beta2", 0.999))
    m = np.zeros_like(current)
    v = np.zeros_like(current)
    previous = None
    converged = False
    objective = float("inf")
    parts = (float("inf"), float("inf"), float("inf"), float("inf"))
    checkpoint_every = max(int(getattr(cfg, "checkpoint", 1000)), 1)
    checkpoints: List[Dict[str, float]] = []
    line_search_rejections_total = 0
    final_accepted_step_size = 0.0

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
        rejected_this_iter = 0
        accepted_step_size = 0.0
        for _ in range(25):
            candidate = current - step_size * step
            np.fill_diagonal(candidate, 0.0)
            cand_obj, *_ = _stage_objective_value_grad(
                candidate, X, cfg, independence_pairs, mu=mu, s=s
            )
            if np.isfinite(cand_obj) and cand_obj <= objective + 1e-10:
                current = candidate
                accepted = True
                accepted_step_size = float(step_size)
                break
            rejected_this_iter += 1
            step_size *= 0.5
        line_search_rejections_total += rejected_this_iter
        final_accepted_step_size = accepted_step_size

        if not accepted:
            return current, False, {
                "iterations": iteration,
                "objective": float(objective),
                "success": False,
                "converged": converged,
                "line_search_rejections": int(line_search_rejections_total),
                "final_accepted_step_size": float(final_accepted_step_size),
                "checkpoints": checkpoints,
            }

        if iteration % checkpoint_every == 0 or iteration == int(max_iter):
            checkpoint = {
                "iteration": int(iteration),
                "checkpoint": int(checkpoint_every),
                "objective": float(objective),
                "line_search_rejections": int(line_search_rejections_total),
                "final_accepted_step_size": float(final_accepted_step_size),
            }
            checkpoint.update(_stage_diagnostics(current, X, cfg, independence_pairs, mu=mu, s=s))
            checkpoints.append(checkpoint)

    diagnostics = _stage_diagnostics(
        current, X, cfg, independence_pairs, mu=mu, s=s
    )
    if not checkpoints or int(checkpoints[-1].get("iteration", -1)) != int(iteration):
        checkpoint = {
            "iteration": int(iteration),
            "checkpoint": int(checkpoint_every),
            "objective": float(objective),
            "line_search_rejections": int(line_search_rejections_total),
            "final_accepted_step_size": float(final_accepted_step_size),
        }
        checkpoint.update(diagnostics)
        checkpoints.append(checkpoint)
    stage_diag = {
        "iterations": int(iteration),
        "objective": float(objective),
        "success": True,
        "converged": bool(converged),
        "line_search_rejections": int(line_search_rejections_total),
        "final_accepted_step_size": float(final_accepted_step_size),
        "checkpoints": checkpoints,
    }
    stage_diag.update(diagnostics)
    return current, True, stage_diag


def fit_notreks_optimizer(
    X: np.ndarray,
    cfg,
    independence_pairs: Sequence[Pair],
    *,
    W_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, OptimizerDiagnostics]:
    X = np.asarray(X, dtype=float)
    if cfg.score == "least_squares":
        X = X - np.mean(X, axis=0, keepdims=True)
    d = X.shape[1]
    W = np.zeros((d, d), dtype=float) if W_init is None else np.asarray(W_init, dtype=float).copy()
    np.fill_diagonal(W, 0.0)

    mu = float(cfg.mu_init)
    lr = float(cfg.lr)
    stages: List[Dict[str, float]] = []
    iterations_total = 0
    path_steps_completed = 0
    converged = False
    last_mu_used = float(mu)

    for stage in range(int(cfg.path_steps)):
        s_stage = _choose_s(stage, cfg)
        inner_iter = int(cfg.max_iter)
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

        for checkpoint in stage_diag.get("checkpoints", []):
            checkpoint.update({
                "stage": int(stage),
                "mu": float(mu),
                "s": float(stage_s),
                "lr": float(stage_lr),
                "success": bool(success),
                "initialization": getattr(cfg, "init", "zero"),
            })

        iterations_total += int(stage_diag.get("iterations", 0))
        stage_diag.update({
            "stage": int(stage),
            "mu": float(mu),
            "s": float(stage_s),
            "lr": float(stage_lr),
            "success": bool(success),
            "initialization": getattr(cfg, "init", "zero"),
        })
        stages.append(stage_diag)

        if not success:
            break

        path_steps_completed += 1
        last_mu_used = float(mu)
        converged = bool(stage_diag.get("converged", False))
        mu *= float(cfg.mu_factor)

    final_s = _choose_s(max(path_steps_completed - 1, 0), cfg)
    objective, _, score, dag_value, trek_value, reg_value = _stage_objective_value_grad(
        W, X, cfg, independence_pairs, mu=max(last_mu_used, 1e-300), s=final_s
    )
    final_diag = _stage_diagnostics(
        W, X, cfg, independence_pairs, mu=max(last_mu_used, 1e-300), s=final_s
    )

    diagnostics = OptimizerDiagnostics(
        converged=bool(converged and path_steps_completed == int(cfg.path_steps)),
        iterations_total=int(iterations_total),
        path_steps_completed=int(path_steps_completed),
        final_mu=float(last_mu_used),
        objective=float(objective),
        score=float(score),
        dag_penalty=float(final_diag["scaled_dag_penalty"]),
        trek_penalty=float(final_diag["scaled_trek_penalty"]),
        regularizer=float(final_diag["scaled_regularizer"]),
        raw_score=float(final_diag["raw_score"]),
        raw_dag_penalty=float(final_diag["raw_dag_penalty"]),
        raw_trek_penalty=float(final_diag["raw_trek_penalty"]),
        raw_regularizer=float(final_diag["raw_regularizer"]),
        scaled_score=float(final_diag["scaled_score"]),
        scaled_dag_penalty=float(final_diag["scaled_dag_penalty"]),
        scaled_trek_penalty=float(final_diag["scaled_trek_penalty"]),
        scaled_regularizer=float(final_diag["scaled_regularizer"]),
        grad_norm=float(final_diag["grad_norm"]),
        grad_score_norm=float(final_diag["grad_score_norm"]),
        grad_dag_norm=float(final_diag["grad_dag_norm"]),
        grad_trek_norm=float(final_diag["grad_trek_norm"]),
        grad_regularizer_norm=float(final_diag["grad_regularizer_norm"]),
        n_samples=int(final_diag["n_samples"]),
        n_variables=int(final_diag["n_variables"]),
        num_independence_pairs=int(final_diag["num_independence_pairs"]),
        backend="numpy/scipy+jax-available" if JAX_AVAILABLE else "numpy/scipy",
        stages=stages,
    )
    return W, diagnostics
