"""FLOP-style order reinsertion with a hard-NOTREKS global greedy inner loop.

This diagnostic/paper comparator reuses FLOP's outer idea and scores complete
graphs under exact discrete DAG and NOTREKS feasibility. Gaussian BIC is
decomposable, so the implementation caches exact node-local scores while
retaining the same graph-level acceptance objective and deterministic ties.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import scipy.linalg as sla

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
    local_score_cache_misses: int
    full_feasibility_evaluations: int
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


class _LocalGaussianBIC:
    """Exact cached local scores matching the graph-level BIC convention."""

    def __init__(self, data: np.ndarray, lambda_bic: float):
        self.data = data - data.mean(axis=0, keepdims=True)
        self.n = len(data)
        self.penalty = float(lambda_bic) * np.log(self.n)
        self.cache: dict[tuple[int, tuple[int, ...]], float] = {}
        self.cache_misses = 0

    def score(self, child: int, parents: Sequence[int]) -> float:
        key = (int(child), tuple(sorted(map(int, parents))))
        if key not in self.cache:
            parent_array = np.asarray(key[1], dtype=int)
            if len(parent_array):
                beta, *_ = sla.lstsq(
                    self.data[:, parent_array], self.data[:, child],
                    lapack_driver="gelsy")
                residual = (self.data[:, child]
                            - self.data[:, parent_array] @ beta)
            else:
                residual = self.data[:, child]
            variance = max(
                float(residual @ residual) / self.n, 1e-8)
            self.cache[key] = (
                self.n * np.log(variance) + len(parent_array) * self.penalty)
            self.cache_misses += 1
        return self.cache[key]

    def graph_score(self, graph: np.ndarray) -> float:
        return float(sum(
            self.score(child, np.flatnonzero(graph[:, child]))
            for child in range(graph.shape[0])))


def _transitive_reach(graph: np.ndarray) -> np.ndarray:
    reach = np.asarray(graph, dtype=bool).copy()
    np.fill_diagonal(reach, True)
    for node in range(len(reach)):
        reach |= reach[:, [node]] & reach[[node], :]
    return reach


def _addition_violates_notreks(
    reach: np.ndarray,
    source: int,
    target: int,
    pairs: np.ndarray,
) -> bool:
    """Check an order-valid edge addition from the current feasible graph."""
    if not len(pairs):
        return False
    new_ancestors = reach[:, source]
    for left, right in pairs:
        left_affected = bool(reach[target, left])
        right_affected = bool(reach[target, right])
        if left_affected and right_affected:
            return True
        if left_affected and np.any(new_ancestors & reach[:, right]):
            return True
        if right_affected and np.any(new_ancestors & reach[:, left]):
            return True
    return False


def _invalid_notreks_additions(
    reach: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """Return exact invalid ``source -> target`` additions for a feasible DAG."""
    d = reach.shape[0]
    invalid = np.zeros((d, d), dtype=bool)
    if not len(pairs):
        return invalid
    # common[i, j] is true exactly when i and j currently share an ancestor.
    common = (reach.T.astype(np.uint16) @ reach.astype(np.uint16)) > 0
    for left, right in pairs:
        left_affected = reach[:, left]
        right_affected = reach[:, right]
        invalid |= np.outer(common[:, right], left_affected)
        invalid |= np.outer(common[:, left], right_affected)
        invalid[:, left_affected & right_affected] = True
    return invalid


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
    local_scorer = _LocalGaussianBIC(data, config.lambda_bic)
    canonical_pairs = np.asarray(sorted({
        tuple(sorted(map(int, pair))) for pair in no_trek_pairs
    }), dtype=int).reshape(-1, 2)
    score_evaluations = feasibility_rejections = accepted_moves = 0
    full_feasibility_evaluations = 0

    def improve(graph: np.ndarray, order: Sequence[int]):
        nonlocal score_evaluations, feasibility_rejections, accepted_moves
        current = _project_to_order(graph, order)
        current_score = local_scorer.graph_score(current)
        position = np.empty(d, dtype=int)
        position[np.asarray(order, dtype=int)] = np.arange(d)
        edges = [
            (i, j) for i in range(d) for j in range(d)
            if position[i] < position[j]]
        while True:
            reach = _transitive_reach(current)
            invalid_additions = _invalid_notreks_additions(
                reach, canonical_pairs)
            best = (current_score, current.tobytes(), None)
            for i, j in edges:
                adding = not bool(current[i, j])
                if adding and invalid_additions[i, j]:
                    feasibility_rejections += 1
                    continue
                old_parents = np.flatnonzero(current[:, j])
                if adding:
                    new_parents = np.append(old_parents, i)
                else:
                    new_parents = old_parents[old_parents != i]
                value = (current_score
                         - local_scorer.score(j, old_parents)
                         + local_scorer.score(j, new_parents))
                score_evaluations += 1
                proposal = current.copy()
                proposal[i, j] ^= 1
                candidate = (value, proposal.tobytes(), proposal)
                if candidate[:2] < best[:2]:
                    best = candidate
            if best[2] is None or best[0] >= current_score - 1e-10:
                return current, current_score
            current, current_score = best[2], best[0]
            current_score = local_scorer.graph_score(current)
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
    full_feasibility_evaluations += 1
    summary = constraints.violation_summary(winner[3])
    if not summary["feasible"]:
        raise RuntimeError("global greedy returned an infeasible graph")
    exact_score = scorer.score_graph(winner[3])
    if not np.isclose(exact_score, winner[0], rtol=1e-10, atol=1e-8):
        raise RuntimeError("incremental and graph-level Gaussian BIC disagree")
    return GlobalGreedyResult(
        adjacency=winner[3], score=float(exact_score), order=winner[4],
        restart=int(winner[2]), runtime_seconds=perf_counter() - started,
        graph_score_evaluations=score_evaluations,
        local_score_cache_misses=local_scorer.cache_misses,
        full_feasibility_evaluations=full_feasibility_evaluations,
        feasibility_rejections=feasibility_rejections,
        accepted_edge_moves=accepted_moves,
        accepted_reinsertions=accepted_reinsertions,
        termination_reason=winner[5],
        notreks_violation_count=int(summary["notreks_violation_count"]),
    )
