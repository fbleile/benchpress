from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
import platform
import time
from typing import Literal

import numpy as np
import scipy.linalg as sla

from .graph_utils import cyclic_scc_sizes, is_dag, postprocess_graph


TerminationReason = Literal[
    "converged", "max_runtime", "max_iterations", "domain_failure",
    "numerical_failure", "completed_schedule",
]


@dataclass
class Checkpoint:
    elapsed_seconds: float
    stage: int
    iteration: int
    objective: float | None
    score: float | None
    acyclicity_value: float | None
    domain_margin: float | None
    gradient_or_prox_residual: float | None
    edge_count_raw: int
    raw_thresholded_is_dag: bool
    cyclic_scc_sizes: list[int]
    precision: str
    current_weighted_adjacency_raw: list[list[float]] = field(default_factory=list)
    best_projected_dag: list[list[int]] = field(default_factory=list)


@dataclass
class DagmaAnytimeResult:
    weighted_adjacency_raw: np.ndarray
    weighted_adjacency_best: np.ndarray
    adjacency_thresholded_raw: np.ndarray
    adjacency_projected_dag: np.ndarray
    elapsed_seconds: float
    termination_reason: TerminationReason
    timed_out: bool
    converged: bool
    current_stage: int
    iterations_total: int
    iterations_by_stage: list[int]
    objective: float | None
    score: float | None
    acyclicity_value: float | None
    domain_margin: float | None
    gradient_or_prox_residual: float | None
    edge_count_raw: int
    edge_count_projected: int
    raw_thresholded_is_dag: bool
    precision_policy: str
    dtype_by_component: dict[str, str]
    factorization_count: int
    domain_backtrack_count: int
    checkpoint_history: list[Checkpoint] = field(default_factory=list)
    timeout_overshoot_seconds: float = 0.0
    projected_removed_edges: list[tuple[int, int, float]] = field(default_factory=list)
    method: str = "dagma_vanilla_anytime"

    def to_jsonable(self) -> dict:
        d = asdict(self)
        for key in [
            "weighted_adjacency_raw", "weighted_adjacency_best",
            "adjacency_thresholded_raw", "adjacency_projected_dag",
        ]:
            d[key] = np.asarray(d[key]).tolist()
        return d


def _now_ns() -> int:
    return time.perf_counter_ns()


def fused_inverse_logdet(A: np.ndarray):
    """Return A^{-T}, log|det(A)|, sign, and M-matrix-domain validity.

    One LAPACK LU factorization is reused for determinant information and the
    inverse. This is the implementation used by ``dagma_fused64_exact``.
    """
    matrix = np.array(A, dtype=np.float64, order="F", copy=True)
    getrf = sla.get_lapack_funcs("getrf", (matrix,))
    getri = sla.get_lapack_funcs("getri", (matrix,))
    lu, pivots, info = getrf(matrix, overwrite_a=True)
    if info != 0:
        return np.empty_like(matrix).T, -np.inf, 0.0, False
    diagonal = np.diag(lu)
    if np.any(diagonal == 0) or not np.all(np.isfinite(diagonal)):
        return np.empty_like(matrix).T, -np.inf, 0.0, False
    swaps = int(np.count_nonzero(pivots != np.arange(len(pivots))))
    sign = float(
        np.prod(np.sign(diagonal)) * (-1.0 if swaps % 2 else 1.0))
    logabsdet = float(np.sum(np.log(np.abs(diagonal))))
    inverse, info = getri(lu, pivots, overwrite_lu=True)
    if info != 0 or not np.all(np.isfinite(inverse)):
        return np.empty_like(matrix).T, logabsdet, sign, False
    valid = bool(sign > 0 and np.all(inverse >= -1e-12))
    return inverse.T, logabsdet, sign, valid


def gaussian_score_from_cov(W: np.ndarray, cov: np.ndarray) -> float:
    dif = np.eye(W.shape[0], dtype=W.dtype) - W
    return float(0.5 * np.trace(dif.T @ (cov @ dif)))


def bic_candidate_score(W: np.ndarray, cov: np.ndarray, n: int, threshold: float) -> float:
    A = postprocess_graph(W, threshold).projected_dag
    W_masked = np.asarray(W) * A
    rss = 2.0 * gaussian_score_from_cov(W_masked, cov)
    edges = int(A.sum())
    return float(n * np.log(max(rss / W.shape[0], 1e-16)) + math.log(max(n, 2)) * edges)


class LinearDagmaKernel:
    def __init__(
        self, cov: np.ndarray, lambda1: float,
        precision_policy: str = "float64", *, use_fused_logdet: bool = False,
    ):
        self.precision_policy = precision_policy
        self.use_fused_logdet = bool(use_fused_logdet)
        dtype = np.float32 if precision_policy == "float32" else np.float64
        self.cov_opt = np.asarray(cov, dtype=np.float32 if precision_policy in {"float32", "mixed"} else np.float64)
        self.cov_eval = np.asarray(cov, dtype=np.float64)
        self.lambda1 = float(lambda1)
        self.d = cov.shape[0]
        self.Id_opt = np.eye(self.d, dtype=dtype)
        self.Id64 = np.eye(self.d, dtype=np.float64)
        self.factorization_count = 0
        self.domain_backtrack_count = 0

    def score(self, W: np.ndarray) -> tuple[float, np.ndarray]:
        if self.precision_policy == "mixed":
            W32 = np.asarray(W, dtype=np.float32)
            dif = np.eye(self.d, dtype=np.float32) - W32
            rhs = self.cov_opt @ dif
            return float(0.5 * np.trace(dif.T @ rhs)), np.asarray(-rhs, dtype=W.dtype)
        dif = self.Id_opt.astype(W.dtype, copy=False) - W
        rhs = self.cov_opt.astype(W.dtype, copy=False) @ dif
        return float(0.5 * np.trace(dif.T @ rhs)), -rhs

    def h(self, W: np.ndarray, s: float) -> tuple[float, np.ndarray, float]:
        if self.precision_policy == "mixed":
            W_domain = np.asarray(W, dtype=np.float64)
            Id = self.Id64
        else:
            W_domain = W
            Id = np.eye(self.d, dtype=W.dtype)
        M = s * Id - W_domain * W_domain
        if self.use_fused_logdet:
            inverse_transpose, logdet, sign, valid = fused_inverse_logdet(M)
            self.factorization_count += 1
            if sign <= 0 or not np.isfinite(logdet) or not valid:
                raise FloatingPointError("invalid log-det M-matrix domain")
            h = -float(logdet) + self.d * math.log(float(s))
            margin = float(np.min(np.linalg.eigvalsh(M)))
            return h, np.asarray(
                2.0 * W_domain * inverse_transpose, dtype=W.dtype), margin
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0 or not np.isfinite(logdet):
            raise FloatingPointError("invalid logdet sign")
        invM = sla.inv(M)
        self.factorization_count += 1
        if np.any(invM < -1e-12):
            raise FloatingPointError("inverse violates M-matrix domain")
        h = -float(logdet) + self.d * math.log(float(s))
        G_h = 2.0 * W_domain * invM.T
        margin = float(np.min(np.linalg.eigvalsh(M)))
        return h, np.asarray(G_h, dtype=W.dtype), margin

    def objective(self, W: np.ndarray, mu: float, s: float) -> tuple[float, float, float, float]:
        score, _ = self.score(W)
        h, _, margin = self.h(W, s)
        return mu * (score + self.lambda1 * float(np.abs(W).sum())) + h, score, h, margin


def _adam_update(m, v, grad, iteration, beta_1, beta_2, fast_bias=False):
    m = m * beta_1 + (1.0 - beta_1) * grad
    v = v * beta_2 + (1.0 - beta_2) * (grad ** 2)
    if fast_bias:
        # Caller supplies powers through iteration-compatible exponentiation fallback for reproducibility.
        pass
    m_hat = m / (1.0 - beta_1 ** iteration)
    v_hat = v / (1.0 - beta_2 ** iteration)
    return m, v, m_hat / (np.sqrt(v_hat) + 1e-8)


def _soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def fit_linear_dagma_anytime(
    X: np.ndarray,
    *,
    method: str = "dagma_vanilla_anytime",
    lambda1: float = 0.03,
    w_threshold: float = 0.3,
    T: int = 5,
    mu_init: float = 1.0,
    mu_factor: float = 0.1,
    s: list[float] | tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6),
    warm_iter: int = 30000,
    max_iter: int = 60000,
    lr: float = 0.0003,
    checkpoint: int = 1000,
    beta_1: float = 0.99,
    beta_2: float = 0.999,
    max_runtime_seconds: float | None = None,
    snapshot_times_seconds: list[float] | None = None,
    random_seed: int = 0,
    precision_policy: str | None = None,
    checkpoint_interval: int | None = None,
    zero_diagonal: bool = False,
    proximal_l1: bool = False,
    adaptive_precision: bool = False,
    initial_W: np.ndarray | None = None,
) -> DagmaAnytimeResult:
    del random_seed
    start_ns = _now_ns()
    deadline_ns = None if max_runtime_seconds is None else start_ns + int(max_runtime_seconds * 1e9)
    snapshots = sorted(float(t) for t in (snapshot_times_seconds or []))
    next_snapshot = 0
    X0 = np.asarray(X)
    if X0.ndim != 2:
        raise ValueError("X must be a two-dimensional data matrix")
    if precision_policy is None:
        precision_policy = {
            "dagma_float32": "float32",
            "dagma_mixed": "mixed",
        }.get(method, "float64")
    dtype = np.float32 if precision_policy == "float32" else np.float64
    Xc = np.asarray(X0, dtype=dtype if precision_policy != "mixed" else np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    n, d = Xc.shape
    cov = (Xc.T @ Xc) / float(n)
    kernel = LinearDagmaKernel(
        cov, lambda1, precision_policy,
        use_fused_logdet=method in {"dagma_fast64", "dagma_fused64_exact"})
    if initial_W is None:
        W = np.zeros((d, d), dtype=dtype)
    else:
        W = np.asarray(initial_W, dtype=dtype).copy()
        if W.shape != (d, d):
            raise ValueError(f"initial_W must have shape {(d, d)}")
        np.fill_diagonal(W, 0.0)
    best_raw = W.copy()
    best_raw_stage_key: tuple[int, float] | None = None
    best_dag = postprocess_graph(W, w_threshold)
    best_dag_score = bic_candidate_score(W, kernel.cov_eval, n, w_threshold)
    history: list[Checkpoint] = []
    iterations_by_stage: list[int] = []
    iterations_total = 0
    termination: TerminationReason = "completed_schedule"
    converged = False
    gradient_residual = None
    objective = score = hval = margin = None
    diag_mask = np.ones((d, d), dtype=dtype)
    np.fill_diagonal(diag_mask, 0.0)
    schedule = list(float(x) for x in s)
    if len(schedule) < int(T):
        schedule += [schedule[-1]] * (int(T) - len(schedule))
    mus = [float(mu_init) * float(mu_factor) ** stage for stage in range(int(T))]
    check_every = max(1, int(checkpoint_interval or checkpoint))
    timer_check_every = 10 if d <= 50 else 5

    def elapsed() -> float:
        return (_now_ns() - start_ns) / 1e9

    def record(stage: int, iteration: int) -> None:
        nonlocal best_dag, best_dag_score, best_raw, best_raw_stage_key
        nonlocal objective, score, hval, margin, gradient_residual, next_snapshot
        try:
            objective, score, hval, margin = kernel.objective(W, mus[stage], schedule[stage])
            key = (stage, -objective)
            if best_raw_stage_key is None or key > best_raw_stage_key:
                best_raw_stage_key = key
                best_raw = W.copy()
        except Exception:
            objective = score = hval = margin = None
        pp = postprocess_graph(W, w_threshold)
        cand_score = bic_candidate_score(pp.projected_dag * W, kernel.cov_eval, n, w_threshold)
        if cand_score < best_dag_score:
            best_dag_score = cand_score
            best_dag = pp
        cp = Checkpoint(
            elapsed_seconds=elapsed(), stage=stage + 1, iteration=iteration,
            objective=None if objective is None else float(objective),
            score=None if score is None else float(score),
            acyclicity_value=None if hval is None else float(hval),
            domain_margin=None if margin is None else float(margin),
            gradient_or_prox_residual=None if gradient_residual is None else float(gradient_residual),
            edge_count_raw=int(pp.thresholded_raw.sum()),
            raw_thresholded_is_dag=bool(pp.raw_is_dag),
            cyclic_scc_sizes=cyclic_scc_sizes(pp.thresholded_raw),
            precision=precision_policy,
            current_weighted_adjacency_raw=np.asarray(W, dtype=float).tolist(),
            best_projected_dag=np.asarray(best_dag.projected_dag, dtype=int).tolist(),
        )
        history.append(cp)
        while next_snapshot < len(snapshots) and cp.elapsed_seconds >= snapshots[next_snapshot]:
            next_snapshot += 1

    try:
        for stage, mu in enumerate(mus):
            max_stage_iter = int(max_iter if stage == len(mus) - 1 else warm_iter)
            m = np.zeros_like(W)
            v = np.zeros_like(W)
            obj_prev = 1e16
            stage_iters = 0
            for iteration in range(1, max_stage_iter + 1):
                if deadline_ns is not None and iteration % timer_check_every == 0 and _now_ns() >= deadline_ns:
                    termination = "max_runtime"
                    break
                try:
                    _, G_score = kernel.score(W)
                    hval_step, G_h, margin_step = kernel.h(W, schedule[stage])
                    G_smooth = mu * G_score + G_h
                    if proximal_l1 or method in {"dagma_prox_adaptive", "dagma_hybrid"}:
                        m, v, adam_grad = _adam_update(m, v, G_smooth, iteration, beta_1, beta_2)
                        W_next = _soft_threshold(W - lr * adam_grad, lr * mu * lambda1)
                    else:
                        Gobj = G_smooth + mu * lambda1 * np.sign(W)
                        m, v, adam_grad = _adam_update(m, v, Gobj, iteration, beta_1, beta_2)
                        W_next = W - lr * adam_grad
                    if zero_diagonal or method in {"dagma_prox_adaptive", "dagma_hybrid"}:
                        W_next *= diag_mask
                    # Domain-safe rejection/backtracking for non-vanilla variants.
                    if method != "dagma_vanilla_anytime":
                        trial_lr = lr
                        while True:
                            try:
                                kernel.h(W_next, schedule[stage])
                                break
                            except Exception:
                                kernel.domain_backtrack_count += 1
                                trial_lr *= 0.5
                                if trial_lr <= 1e-16:
                                    raise
                                if proximal_l1 or method in {"dagma_prox_adaptive", "dagma_hybrid"}:
                                    W_next = _soft_threshold(W - trial_lr * adam_grad, trial_lr * mu * lambda1)
                                else:
                                    W_next = W - trial_lr * adam_grad
                                if zero_diagonal or method in {"dagma_prox_adaptive", "dagma_hybrid"}:
                                    W_next *= diag_mask
                    W = W_next.astype(dtype, copy=False)
                    gradient_residual = float(np.linalg.norm(adam_grad))
                except FloatingPointError:
                    if adaptive_precision and precision_policy == "mixed":
                        precision_policy = "float64"
                    termination = "domain_failure"
                    break
                except Exception:
                    termination = "numerical_failure"
                    break
                iterations_total += 1
                stage_iters += 1
                if iteration % check_every == 0 or iteration == max_stage_iter:
                    try:
                        obj_new, sc, hv, mg = kernel.objective(W, mu, schedule[stage])
                        record(stage, iteration)
                        rel = abs((obj_prev - obj_new) / obj_prev)
                        if method in {"dagma_prox_adaptive", "dagma_hybrid"}:
                            if rel <= 1e-6 or gradient_residual <= 1e-6:
                                converged = True
                                termination = "converged"
                                break
                            pp = postprocess_graph(W, w_threshold)
                            if pp.raw_is_dag and int(pp.thresholded_raw.sum()) > 0 and hv <= 1e-8:
                                converged = True
                                termination = "converged"
                                break
                        else:
                            if rel <= 1e-6:
                                converged = True
                                break
                        obj_prev = obj_new
                    except Exception:
                        termination = "numerical_failure"
                        break
            iterations_by_stage.append(stage_iters)
            if termination in {"max_runtime", "domain_failure", "numerical_failure"}:
                break
        else:
            termination = "completed_schedule"
    finally:
        if not history:
            record(0, 0)
    final_pp = postprocess_graph(W, w_threshold)
    elapsed_seconds = elapsed()
    overshoot = 0.0
    timed_out = termination == "max_runtime"
    if timed_out and max_runtime_seconds is not None:
        overshoot = max(0.0, elapsed_seconds - float(max_runtime_seconds))
    return DagmaAnytimeResult(
        weighted_adjacency_raw=np.asarray(W, dtype=float),
        weighted_adjacency_best=np.asarray(best_raw, dtype=float),
        adjacency_thresholded_raw=final_pp.thresholded_raw,
        adjacency_projected_dag=final_pp.projected_dag,
        elapsed_seconds=float(elapsed_seconds),
        termination_reason=termination,
        timed_out=timed_out,
        converged=bool(converged or termination == "completed_schedule"),
        current_stage=len(iterations_by_stage),
        iterations_total=int(iterations_total),
        iterations_by_stage=iterations_by_stage,
        objective=objective,
        score=score,
        acyclicity_value=hval,
        domain_margin=margin,
        gradient_or_prox_residual=gradient_residual,
        edge_count_raw=int(final_pp.thresholded_raw.sum()),
        edge_count_projected=int(final_pp.projected_dag.sum()),
        raw_thresholded_is_dag=bool(final_pp.raw_is_dag),
        precision_policy=precision_policy,
        dtype_by_component={
            "covariance": str(kernel.cov_opt.dtype),
            "W": str(W.dtype),
            "adam_state": str(dtype),
            "logdet": "float64" if precision_policy == "mixed" else str(dtype),
        },
        factorization_count=int(kernel.factorization_count),
        domain_backtrack_count=int(kernel.domain_backtrack_count),
        checkpoint_history=history,
        timeout_overshoot_seconds=float(overshoot),
        projected_removed_edges=final_pp.removed_edges,
        method=method,
    )


def hardware_metadata() -> dict[str, str | int | None]:
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }


def write_result_json(result: DagmaAnytimeResult, path: str, extra: dict | None = None) -> None:
    payload = result.to_jsonable()
    payload["hardware"] = hardware_metadata()
    if extra:
        payload["extra"] = extra
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
