"""Deletion-only normalized projection of a continuous DAGMA support."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    is_dag,
)
from workflow.rules.structure_learning_algorithms.dagma.shared import (
    NoTreksKernel,
)


@dataclass
class ProjectionResult:
    adjacency: np.ndarray
    diagnostics: dict


def _reachability(graph: np.ndarray) -> np.ndarray:
    reach = np.asarray(graph, dtype=bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(len(reach)):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def _dag_penalty(W: np.ndarray) -> float:
    matrix = np.eye(len(W)) - W * W
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(logdet):
        return float("inf")
    return float(-logdet)


def _notreks_penalty(W: np.ndarray, pairs, kernel=None) -> float:
    if kernel is None and not pairs:
        return 0.0
    if kernel is None:
        kernel = NoTreksKernel.from_pairs(pairs, len(W))
    value, _ = kernel.value_grad(
        W, "inv", inverse_epsilon=1e-8)
    return float(value)


def _scale_epsilon(value: float) -> float:
    scale = max(1.0, abs(float(value))) if np.isfinite(value) else 1.0
    return 1e-12 * scale


def _violating_edges(graph: np.ndarray, pairs) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return cycle and concrete-path NOTREKS candidate edges."""
    reach = _reachability(graph)
    cycle_edges = {
        (int(u), int(v))
        for u, v in zip(*np.nonzero(graph))
        if reach[v, u]
    }
    notreks_edges: set[tuple[int, int]] = set()
    for left, right in pairs:
        common = np.flatnonzero(reach[:, left] & reach[:, right])
        if not len(common):
            continue
        for u, v in zip(*np.nonzero(graph)):
            if any(reach[c, u] and (reach[v, left] or reach[v, right])
                   for c in common):
                notreks_edges.add((int(u), int(v)))
    return cycle_edges, notreks_edges


def normalized_greedy_projection(
    W,
    notreks_pairs: Sequence[tuple[int, int]] = (),
    *,
    dag_constraint: bool = True,
    notreks_constraint: bool = False,
    epsilon: float | None = None,
    weight_epsilon: float | None = None,
) -> ProjectionResult:
    """Project support by deleting edges using fixed normalized penalties."""
    original = np.asarray(W, dtype=np.float64)
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        raise ValueError("W must be a square matrix")
    if not np.all(np.isfinite(original)):
        raise ValueError("W must contain only finite values")
    pairs = tuple(tuple(sorted((int(i), int(j)))) for i, j in notreks_pairs)
    graph = (original != 0).astype(np.uint8)
    np.fill_diagonal(graph, 0)
    graph[np.diag_indices_from(graph)] = 0
    # Construct this once.  Rebuilding and validating the kernel for every
    # deletion candidate was needlessly expensive.
    nt_kernel = (NoTreksKernel.from_pairs(pairs, len(original))
                 if pairs else None)
    penalty_cache: dict[bytes, tuple[float, float]] = {}

    def penalties(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        key = matrix.tobytes()
        if key not in penalty_cache:
            penalty_cache[key] = (
                _dag_penalty(matrix),
                _notreks_penalty(matrix, pairs, nt_kernel),
            )
        return penalty_cache[key]

    dag_initial, nt_initial = penalties(original)
    eps_d = _scale_epsilon(dag_initial) if epsilon is None else float(epsilon)
    eps_i = _scale_epsilon(nt_initial) if epsilon is None else float(epsilon)
    eps_w = (np.finfo(float).eps * max(1.0, float(np.max(np.abs(original))))
             if weight_epsilon is None else float(weight_epsilon))
    if eps_d <= 0 or eps_i <= 0 or eps_w <= 0:
        raise ValueError("projection epsilons must be positive")
    use_dag = bool(dag_constraint and not is_dag(graph))
    use_nt = bool(notreks_constraint and pairs
                  and common_ancestor_violations(graph, pairs))
    c_d = dag_initial + eps_d if use_dag else None
    c_i = nt_initial + eps_i if use_nt else None

    def normalized_penalty(matrix):
        dag_value, nt_value = penalties(matrix)
        value = 0.0
        if c_d is not None:
            value += dag_value / c_d
        if c_i is not None:
            value += nt_value / c_i
        return float(value)

    started = perf_counter()
    current = original.copy()
    trajectory = []
    fallback_used = False
    discarded_squared_weight = 0.0
    while True:
        dag_ok = (not dag_constraint) or is_dag(graph)
        nt_count = (common_ancestor_violations(graph, pairs)
                    if notreks_constraint else 0)
        if dag_ok and nt_count == 0:
            break
        cycle_edges, notreks_edges = _violating_edges(graph, pairs)
        candidates = sorted(cycle_edges | notreks_edges)
        if not candidates:
            # This should only be reachable for malformed numerical input.
            candidates = sorted((int(u), int(v))
                                for u, v in zip(*np.nonzero(graph)))
        before_dag, before_nt = penalties(current)
        before_penalty = normalized_penalty(current)
        scored = []
        for u, v in candidates:
            proposal = current.copy()
            proposal[u, v] = 0.0
            reduction = before_penalty - normalized_penalty(proposal)
            if not np.isfinite(reduction):
                reduction = 0.0
            reduction = max(0.0, float(reduction))
            score = reduction / (current[u, v] ** 2 + eps_w)
            scored.append((-score, u, v, reduction, proposal))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        if all(item[3] <= eps_d + eps_i for item in scored):
            fallback_used = True
            selected = min(
                scored, key=lambda item: (abs(current[item[1], item[2]]),
                                          item[1], item[2]))
        else:
            selected = scored[0]
        selected_score, u, v, reduction, proposal = selected
        weight = float(current[u, v])
        after_dag, after_nt = penalties(proposal)
        trajectory.append({
            "step": len(trajectory) + 1,
            "deleted_edge": [u, v],
            "deleted_weight": weight,
            "candidate_count": len(candidates),
            "dag_penalty_before": before_dag,
            "dag_penalty_after": after_dag,
            "notreks_penalty_before": before_nt,
            "notreks_penalty_after": after_nt,
            "normalized_penalty_reduction": float(reduction),
            "R": float(-selected_score),
            "cycle_violations_after": int(not is_dag(proposal)),
            "notreks_violations_after": int(
                common_ancestor_violations(proposal, pairs)
                if pairs else 0),
            "fallback": bool(fallback_used and reduction <= eps_d + eps_i),
        })
        current = proposal
        graph = (current != 0).astype(np.uint8)
        discarded_squared_weight += weight * weight

    projected = (current != 0).astype(np.uint8)
    diagnostics = {
        "policy": "normalized_greedy_projection",
        "dag_penalty_initial": float(dag_initial),
        "notreks_penalty_initial": float(nt_initial),
        "dag_normalizer": c_d,
        "notreks_normalizer": c_i,
        "epsilon_penalty_dag": eps_d,
        "epsilon_penalty_notreks": eps_i,
        "epsilon_weight": eps_w,
        "projection_steps": len(trajectory),
        "projection_trajectory": trajectory,
        "projection_distance_squared": float(
            np.sum((original - projected) ** 2)),
        "discarded_squared_weight": float(discarded_squared_weight),
        "edges_removed": int(np.count_nonzero(original) - projected.sum()),
        "final_edges": int(projected.sum()),
        "fallback_used": bool(fallback_used),
        "runtime_seconds": perf_counter() - started,
        "penalty_cache_entries": len(penalty_cache),
        "dag_violations_after": int(not is_dag(projected)),
        "notreks_violations_after": int(
            common_ancestor_violations(projected, pairs)
            if pairs else 0),
    }
    if diagnostics["dag_violations_after"] or diagnostics["notreks_violations_after"]:
        raise RuntimeError("normalized projection failed to reach feasibility")
    return ProjectionResult(projected, diagnostics)
