"""Experimental fixed-order continuous parent-block optimization.

This is not the production FLOP+NOTREKS path.  It optimizes one child's
allowed parent block at a time while holding every other weighted entry fixed.
The order guarantees acyclicity; the final binary graph is still checked by
the exact NOTREKS certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize

from workflow.rules.structure_learning_algorithms.dagma.shared import notreks_value_grad


@dataclass(frozen=True)
class ContinuousBlockConfig:
    lambda1: float = 0.03
    trek_weight: float = 1.0
    threshold: float = 0.30
    maxiter: int = 100
    seed: int = 9137
    hard_feasible_support: bool = True
    smooth_l1_epsilon: float = 1e-6


@dataclass
class ContinuousBlockResult:
    adjacency: np.ndarray
    weights: np.ndarray
    runtime_seconds: float
    blocks_optimized: int
    repair_deletions: int
    notreks_violation_count: int
    termination_reason: str


def _reachability(graph: np.ndarray) -> np.ndarray:
    reach = np.asarray(graph, dtype=bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(len(reach)):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def _violations(graph: np.ndarray, pairs) -> int:
    reach = _reachability(graph)
    return sum(bool(np.any(reach[:, i] & reach[:, j])) for i, j in pairs)


def _repair(graph: np.ndarray, pairs, X: np.ndarray) -> tuple[np.ndarray, int]:
    graph = np.asarray(graph, dtype=np.uint8).copy()
    deletions = 0
    while _violations(graph, pairs):
        current = _score_graph(X, graph, 0.0)
        candidates = []
        for i, j in zip(*np.nonzero(graph)):
            candidate = graph.copy()
            candidate[i, j] = 0
            candidates.append((_violations(candidate, pairs),
                               _score_graph(X, candidate, 0.0), int(i), int(j), candidate))
        _, _, _, _, graph = min(candidates, key=lambda x: x[:4])
        deletions += 1
    return graph, deletions


def _feasible_addition(graph: np.ndarray, parent: int, child: int, pairs) -> bool:
    """Exact monotone feasibility test for one proposed edge."""
    if graph[parent, child]:
        return True
    candidate = graph.copy()
    candidate[parent, child] = 1
    return _violations(candidate, pairs) == 0


def _project_weighted_graph(W: np.ndarray, order: np.ndarray, pairs, threshold: float):
    """Greedily threshold by weight while enforcing exact feasibility.

    This is a projection, not a score search: weights determine proposal order,
    while the certificate decides which edges may enter.  It prevents the
    continuous solver from accumulating a dense infeasible support and then
    deleting many useful edges in one final repair pass.
    """
    d = W.shape[0]
    graph = np.zeros((d, d), dtype=np.uint8)
    candidates = [(float(abs(W[i, j])), i, j)
                  for i in range(d) for j in range(d)
                  if i != j and abs(W[i, j]) >= threshold]
    position = {int(node): k for k, node in enumerate(order)}
    candidates.sort(key=lambda x: (-x[0], position.get(x[1], x[1]), position.get(x[2], x[2])))
    for _, parent, child in candidates:
        if position[parent] < position[child] and _feasible_addition(graph, parent, child, pairs):
            graph[parent, child] = 1
    return graph


def _score_graph(X: np.ndarray, graph: np.ndarray, lambda1: float) -> float:
    residual = X - X @ graph
    return float(0.5 * np.mean(residual * residual) + lambda1 * np.abs(graph).sum())


def optimize_parent_block(
    X: np.ndarray,
    W: np.ndarray,
    child: int,
    parents: np.ndarray,
    pairs,
    config: ContinuousBlockConfig,
) -> np.ndarray:
    """Warm-start L-BFGS for one child, with all other support fixed."""
    parents = np.asarray(parents, dtype=int)
    if not len(parents):
        return W
    base = np.asarray(W, dtype=float).copy()
    if config.hard_feasible_support:
        current = (np.abs(base) >= config.threshold).astype(np.uint8)
        np.fill_diagonal(current, 0)
        parents = np.asarray([p for p in parents if _feasible_addition(current, int(p), int(child), pairs)], dtype=int)
        if not len(parents):
            return W
    x0 = base[parents, child].copy()
    n = len(X)

    def value_grad(beta):
        candidate = base.copy()
        candidate[parents, child] = beta
        residual = X - X @ candidate
        value = 0.5 * float(np.sum(residual * residual)) / n
        smooth_abs = np.sqrt(beta * beta + config.smooth_l1_epsilon)
        value += config.lambda1 * float(smooth_abs.sum())
        trek, trek_grad = notreks_value_grad(candidate, pairs, "inv", inverse_epsilon=0.0)
        value += config.trek_weight * trek
        grad = -(X[:, parents].T @ residual[:, child]) / n
        grad += config.lambda1 * beta / smooth_abs
        grad += config.trek_weight * trek_grad[parents, child]
        return value, grad

    result = minimize(value_grad, x0, jac=True, method="L-BFGS-B",
                      options={"maxiter": config.maxiter, "ftol": 1e-12})
    base[parents, child] = result.x
    return base


def fit_fixed_order_continuous(
    X: np.ndarray,
    no_trek_pairs,
    order: np.ndarray,
    config: ContinuousBlockConfig = ContinuousBlockConfig(),
) -> ContinuousBlockResult:
    """Optimize all parent blocks for one fixed order, then certify exactly."""
    started = time.perf_counter()
    X = np.asarray(X, dtype=float)
    order = np.asarray(order, dtype=int)
    d = X.shape[1]
    W = np.zeros((d, d), dtype=float)
    for position, child in enumerate(order):
        parents = order[:position]
        W = optimize_parent_block(X, W, int(child), parents, no_trek_pairs, config)
        # Threshold after each inner block.  In hard-support mode, project by
        # descending weight and exact feasibility rather than repairing a dense
        # infeasible graph at the end.
        if config.hard_feasible_support:
            A = _project_weighted_graph(W, order, no_trek_pairs, config.threshold)
            W[A == 0] = 0.0
        else:
            W[np.abs(W) < config.threshold] = 0.0
    A = (_project_weighted_graph(W, order, no_trek_pairs, config.threshold)
         if config.hard_feasible_support else (np.abs(W) >= config.threshold).astype(np.uint8))
    np.fill_diagonal(A, 0)
    A, deletions = _repair(A, no_trek_pairs, X) if not config.hard_feasible_support else (A, 0)
    return ContinuousBlockResult(
        adjacency=A, weights=W, runtime_seconds=time.perf_counter() - started,
        blocks_optimized=d, repair_deletions=deletions,
        notreks_violation_count=_violations(A, no_trek_pairs),
        termination_reason="completed",
    )
