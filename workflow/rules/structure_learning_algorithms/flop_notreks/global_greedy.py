"""FLOP-style order reinsertion with a hard-NOTREKS global greedy inner loop.

This diagnostic/paper comparator reuses FLOP's outer idea, but deliberately
scores complete graphs and routes every edge move through exact discrete DAG
and NOTREKS feasibility. It does not use graph truth or a decomposable-score
shortcut for selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np

from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.feasibility import (
    CompositeFeasibility,
    DagConstraint,
    NoTreksConstraint,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.scores import (
    GaussianBICGraphScore,
)


@dataclass(frozen=True)
class GlobalGreedyConfig:
    restarts: int = 8
    max_sweeps: int = 4
    lambda_bic: float = 2.0
    seed: int = 1729


@dataclass
class GlobalGreedyResult:
    adjacency: np.ndarray
    score: float
    order: tuple[int, ...]
    restart: int
    runtime_seconds: float
    graph_score_evaluations: int
    feasibility_rejections: int
    accepted_edge_moves: int
    accepted_reinsertions: int
    termination_reason: str
    notreks_violation_count: int


def _project_to_order(graph: np.ndarray, order: Sequence[int]) -> np.ndarray:
    position = np.empty(len(order), dtype=int)
    position[np.asarray(order, dtype=int)] = np.arange(len(order))
    result = np.asarray(graph, dtype=np.uint8).copy()
    result[position[:, None] >= position[None, :]] = 0
    np.fill_diagonal(result, 0)
    return result


def _reinsert(order: Sequence[int], node: int, position: int) -> list[int]:
    candidate = list(order)
    candidate.remove(node)
    candidate.insert(position, node)
    return candidate


def fit_global_greedy_notreks(
    X: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]],
    config: GlobalGreedyConfig = GlobalGreedyConfig(),
) -> GlobalGreedyResult:
    """Fit a hard-feasible graph without consulting graph truth."""
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array")
    if config.restarts < 1 or config.max_sweeps < 1:
        raise ValueError("restarts and max_sweeps must be positive")
    d = data.shape[1]
    constraints = CompositeFeasibility((
        DagConstraint(), NoTreksConstraint(no_trek_pairs)))
    scorer = GaussianBICGraphScore(data, lambda_bic=config.lambda_bic)
    score_cache: dict[bytes, float] = {}
    score_evaluations = feasibility_rejections = accepted_moves = 0

    def score(graph: np.ndarray) -> float:
        nonlocal score_evaluations
        key = np.asarray(graph, dtype=np.uint8).tobytes()
        if key not in score_cache:
            score_cache[key] = scorer.score_graph(graph)
            score_evaluations += 1
        return score_cache[key]

    def improve(graph: np.ndarray, order: Sequence[int]):
        nonlocal feasibility_rejections, accepted_moves
        current = _project_to_order(graph, order)
        current_score = score(current)
        position = np.empty(d, dtype=int)
        position[np.asarray(order, dtype=int)] = np.arange(d)
        edges = [
            (i, j) for i in range(d) for j in range(d)
            if position[i] < position[j]]
        while True:
            best = (current_score, current.tobytes(), None)
            for i, j in edges:
                proposal = current.copy()
                proposal[i, j] ^= 1
                if not constraints.is_feasible(proposal):
                    feasibility_rejections += 1
                    continue
                value = score(proposal)
                candidate = (value, proposal.tobytes(), proposal)
                if candidate[:2] < best[:2]:
                    best = candidate
            if best[2] is None or best[0] >= current_score - 1e-10:
                return current, current_score
            current, current_score = best[2], best[0]
            accepted_moves += 1

    rng = np.random.default_rng(config.seed)
    started = perf_counter()
    winner = None
    accepted_reinsertions = 0
    for restart in range(config.restarts):
        order = list(map(int, rng.permutation(d)))
        graph, current = improve(np.zeros((d, d), dtype=np.uint8), order)
        termination = "maximum_sweeps"
        for _ in range(config.max_sweeps):
            before = current
            for node in order.copy():
                old_position = order.index(node)
                best = (current, tuple(order), graph, order)
                for new_position in range(d):
                    if new_position == old_position:
                        continue
                    candidate_order = _reinsert(order, node, new_position)
                    candidate_graph, candidate_score = improve(
                        graph, candidate_order)
                    key = (candidate_score, tuple(candidate_order))
                    if key < best[:2]:
                        best = (
                            candidate_score, tuple(candidate_order),
                            candidate_graph, candidate_order)
                if best[0] < current - 1e-10:
                    current, graph, order = best[0], best[2], best[3]
                    accepted_reinsertions += 1
            if before - current <= 1e-10:
                termination = "joint_plateau"
                break
        candidate = (current, graph.tobytes(), restart, graph, tuple(order), termination)
        if winner is None or candidate[:3] < winner[:3]:
            winner = candidate
    assert winner is not None
    summary = constraints.violation_summary(winner[3])
    if not summary["feasible"]:
        raise RuntimeError("global greedy returned an infeasible graph")
    return GlobalGreedyResult(
        adjacency=winner[3], score=float(winner[0]), order=winner[4],
        restart=int(winner[2]), runtime_seconds=perf_counter() - started,
        graph_score_evaluations=score_evaluations,
        feasibility_rejections=feasibility_rejections,
        accepted_edge_moves=accepted_moves,
        accepted_reinsertions=accepted_reinsertions,
        termination_reason=winner[5],
        notreks_violation_count=int(summary["notreks_violation_count"]),
    )
