"""Hard-DAG global greedy search with a continuous NOTREKS score."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    fixed_order_bic_search,
    gaussian_bic,
)
from workflow.rules.structure_learning_algorithms.dagma.shared import (
    notreks_value_grad,
)


@dataclass(frozen=True)
class PenaltyGreedyConfig:
    restarts: int = 1
    max_sweeps: int = 8
    max_reinsertions: int = 8
    lambda_bic: float = 2.0
    notreks_weight: float = 1.0
    seed: int = 1729


@dataclass
class PenaltyGreedyResult:
    adjacency: np.ndarray
    bic: float
    notreks: float
    objective: float
    violations: int
    runtime_seconds: float
    graph_evaluations: int


def _project(graph, order):
    position = np.empty(len(order), dtype=int)
    position[np.asarray(order)] = np.arange(len(order))
    result = np.asarray(graph, dtype=np.uint8).copy()
    result[position[:, None] >= position[None, :]] = 0
    np.fill_diagonal(result, 0)
    return result


def _evaluate(X, graph, pairs, config):
    bic, coefficients = gaussian_bic(
        X, graph, lambda_bic=config.lambda_bic)
    notreks, _ = notreks_value_grad(
        coefficients, pairs, "inv", inverse_epsilon=1e-8)
    return (float(bic), float(notreks),
            float(bic + config.notreks_weight * notreks))


def fit_penalty_global_greedy(
    X: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    config: PenaltyGreedyConfig = PenaltyGreedyConfig(),
) -> PenaltyGreedyResult:
    data = np.asarray(X, dtype=float)
    if config.restarts < 1 or config.max_sweeps < 1:
        raise ValueError("restarts and max_sweeps must be positive")
    rng = np.random.default_rng(config.seed)
    started = perf_counter()
    evaluations = 0
    best = None

    def evaluate(graph):
        nonlocal evaluations
        evaluations += 1
        return _evaluate(data, graph, pairs, config)

    def improve(graph, order):
        current = evaluate(graph)
        for _ in range(config.max_sweeps):
            position = np.empty(len(order), dtype=int)
            position[np.asarray(order)] = np.arange(len(order))
            candidates = []
            for source in range(len(order)):
                for target in range(len(order)):
                    if source == target or position[source] >= position[target]:
                        continue
                    proposal = graph.copy()
                    proposal[source, target] ^= 1
                    value = evaluate(proposal)
                    candidates.append((value, proposal))
            if not candidates:
                break
            value, proposal = min(
                candidates, key=lambda item: (item[0][2], item[1].tobytes()))
            if value[2] >= current[2] - 1e-10:
                break
            graph, current = proposal, value
        return graph, current

    for _ in range(config.restarts):
        order = list(map(int, rng.permutation(data.shape[1])))
        graph = fixed_order_bic_search(
            data, order, lambda_bic=config.lambda_bic).astype(np.uint8)
        graph, current = improve(graph, order)
        for _ in range(config.max_reinsertions):
            best_move = (current[2], graph, order, current)
            for node in order:
                for position in range(len(order)):
                    proposed_order = list(order)
                    proposed_order.remove(node)
                    proposed_order.insert(position, node)
                    projected = _project(graph, proposed_order)
                    candidate, value = improve(projected, proposed_order)
                    if value[2] < best_move[0] - 1e-10:
                        best_move = (value[2], candidate,
                                     proposed_order, value)
            if best_move[0] >= current[2] - 1e-10:
                break
            _, graph, order, current = best_move
        if best is None or current[2] < best[0][2]:
            best = (current, graph.copy())

    assert best is not None
    value, graph = best
    return PenaltyGreedyResult(
        adjacency=graph, bic=value[0], notreks=value[1],
        objective=value[2],
        violations=common_ancestor_violations(graph, pairs),
        runtime_seconds=perf_counter() - started,
        graph_evaluations=evaluations)
