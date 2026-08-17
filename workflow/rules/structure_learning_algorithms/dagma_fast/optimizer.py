"""Generic continuation optimizer with the exact fused float64 DAGMA penalty."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import warnings

import numpy as np

from .penalties import LogDetDagPenalty


ALIASES = {
    "dagma_fast64": "dagma_fast",
    "dagma_fused64_exact": "dagma_fast",
}


@dataclass(frozen=True)
class DagmaFastConfig:
    method: str = "dagma_fast"
    lambda1: float = 0.03
    T: int = 5
    mu_init: float = 1.0
    mu_factor: float = 0.1
    s: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6)
    warm_iter: int = 30000
    max_iter: int = 60000
    lr: float = 0.0003
    beta_1: float = 0.99
    beta_2: float = 0.999
    checkpoint: int = 1000
    optimizer_tol: float = 1e-6
    max_runtime_seconds: float | None = None


@dataclass
class DagmaFastResult:
    weighted_adjacency: np.ndarray
    objective: float
    data_loss: float
    dag_penalty: float
    structural_penalties: list[float]
    iterations: int
    iterations_by_stage: list[int]
    termination_reason: str
    runtime_seconds: float
    factorization_count: int
    lambda1_resolved: float
    diagnostics: dict = field(default_factory=dict)


def _adam(m, v, gradient, iteration, beta_1, beta_2):
    m = beta_1 * m + (1 - beta_1) * gradient
    v = beta_2 * v + (1 - beta_2) * gradient * gradient
    step = (
        (m / (1 - beta_1 ** iteration))
        / (np.sqrt(v / (1 - beta_2 ** iteration)) + 1e-8))
    return m, v, step


def fit_weighted_adjacency(
    objective,
    optimizer_config=DagmaFastConfig(),
    initialization=None,
    *,
    dag_penalty=None,
    structural_penalties=(),
):
    config = optimizer_config
    method = ALIASES.get(config.method, config.method)
    if method != config.method:
        warnings.warn(
            f"{config.method} is deprecated; use dagma_fast",
            DeprecationWarning, stacklevel=2)
    if method != "dagma_fast":
        raise ValueError(f"unknown optimizer method: {config.method}")
    d = int(objective.dimension)
    W = (
        np.zeros((d, d), dtype=np.float64) if initialization is None
        else np.asarray(initialization, dtype=np.float64).copy())
    if W.shape != (d, d):
        raise ValueError(f"initialization must have shape {(d, d)}")
    np.fill_diagonal(W, 0.0)
    mask = np.ones_like(W)
    np.fill_diagonal(mask, 0.0)
    penalty = dag_penalty or LogDetDagPenalty(d)
    schedule = list(config.s)
    if len(schedule) < config.T:
        schedule.extend([schedule[-1]] * (config.T - len(schedule)))
    mus = [
        config.mu_init * config.mu_factor ** stage
        for stage in range(config.T)]
    started = time.perf_counter()
    total = 0
    by_stage = []
    termination = "completed_schedule"
    data_value = dag_value = full_value = float("nan")
    structural_values = []
    for stage, mu in enumerate(mus):
        m = np.zeros_like(W)
        v = np.zeros_like(W)
        previous = float("inf")
        stage_iterations = (
            config.max_iter if stage == config.T - 1
            else config.warm_iter)
        completed = 0
        for iteration in range(1, stage_iterations + 1):
            if (
                config.max_runtime_seconds is not None
                and time.perf_counter() - started >= config.max_runtime_seconds
            ):
                termination = "max_runtime"
                break
            data_value, data_gradient = objective.data_loss_value_and_grad(W)
            dag_value, dag_gradient = penalty.value_and_grad(
                W, schedule[stage])
            structural_values = []
            structural_gradient = np.zeros_like(W)
            for component in structural_penalties:
                value, gradient = component.value_and_grad(W)
                structural_values.append(float(value))
                structural_gradient += np.asarray(gradient, dtype=np.float64)
            gradient = (
                mu * (data_gradient + config.lambda1 * np.sign(W))
                + dag_gradient + structural_gradient)
            m, v, step = _adam(
                m, v, gradient, iteration, config.beta_1, config.beta_2)
            trial = config.lr
            while True:
                candidate = (W - trial * step) * mask
                if penalty.is_valid(candidate, schedule[stage]):
                    break
                trial *= 0.5
                if trial <= 1e-16:
                    termination = "domain_failure"
                    break
            if termination == "domain_failure":
                break
            W = candidate
            completed += 1
            total += 1
            if iteration % config.checkpoint == 0 or iteration == stage_iterations:
                full_value = (
                    mu * (data_value + config.lambda1 * np.abs(W).sum())
                    + dag_value + sum(structural_values))
                relative = (
                    float("inf") if not np.isfinite(previous)
                    else abs(previous - full_value) / max(
                        abs(previous), 1.0))
                if relative <= config.optimizer_tol:
                    termination = "converged"
                    break
                previous = full_value
        by_stage.append(completed)
        if termination in {"max_runtime", "domain_failure"}:
            break
    np.fill_diagonal(W, 0.0)
    return DagmaFastResult(
        W, float(full_value), float(data_value), float(dag_value),
        structural_values, total, by_stage, termination,
        time.perf_counter() - started,
        int(getattr(penalty, "factorization_count", 0)),
        float(config.lambda1), {
            "method": "dagma_fast",
            "lambda1_resolved": float(config.lambda1),
            "dimension": d,
            "structural_penalty_count": len(structural_penalties),
        })
