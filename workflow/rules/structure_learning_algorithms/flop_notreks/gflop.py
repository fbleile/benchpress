"""Generic gFLOP: FLOP order search around a pluggable global backend.

The engine knows only orders, scalar merits, dense edge strengths, and hard
candidate callbacks.  It never calls a nodewise/decomposable score routine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol, Sequence

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    is_dag, topological_order,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.exact_solver import (
    canonical_pairs, no_trek_violations,
)


def order_mask(order: Sequence[int]) -> np.ndarray:
    """Complete forward-triangular mask in parent->child orientation."""
    order = tuple(map(int, order))
    if sorted(order) != list(range(len(order))):
        raise ValueError("order must be a permutation of 0..d-1")
    position = np.empty(len(order), dtype=int)
    position[np.asarray(order)] = np.arange(len(order))
    return (position[:, None] < position[None, :]).astype(np.uint8)


@dataclass
class BackendResult:
    state: Any
    edge_strengths: np.ndarray
    base_score: float
    regularizer: float
    notreks_penalty: float
    diagnostics: dict = field(default_factory=dict)

    def merit(self, penalty_weight: float) -> float:
        return float(self.base_score + self.regularizer
                     + penalty_weight * self.notreks_penalty)


@dataclass
class HardCandidate:
    adjacency: np.ndarray
    target_score: float
    diagnostics: dict = field(default_factory=dict)


class GlobalOrderBackend(Protocol):
    dimension: int

    def initialize(self, order: Sequence[int], incumbent: Any | None = None) -> Any: ...
    def transport(self, state: Any, old_order: Sequence[int],
                  new_order: Sequence[int]) -> Any: ...
    def optimize(self, order: Sequence[int], state: Any, *,
                 penalty_weight: float, budget: int) -> BackendResult: ...
    def hard_candidates(self, result: BackendResult, order: Sequence[int], *,
                        use_notreks: bool, budget_seconds: float) -> list[HardCandidate]: ...


@dataclass(frozen=True)
class GFlopConfig:
    continuation_weights: tuple[float, ...] = (0., .1, 1., 10.)
    sweeps_per_stage: int = 1
    screening_budget: int = 50
    refinement_budget: int = 300
    refinement_top_k: int = 2
    finalist_postselection_seconds: float = .05
    postselection_seconds: float = 1.
    archive_size: int = 8
    minimum_improvement: float = 1e-9
    seed: int = 1729
    use_notreks: bool = True
    return_if_initial_feasible: bool = False


@dataclass
class GFlopResult:
    adjacency: np.ndarray
    target_score: float
    order: tuple[int, ...]
    edge_strengths: np.ndarray
    runtime: float
    accepted_order_moves: int
    continuation_trace: list[dict]
    protected_incumbent_updates: int
    screening_seconds: float
    refinement_seconds: float
    postselection_seconds: float
    no_trek_violations: int
    backend_name: str
    use_notreks: bool


def _reinsert(order, node, landing):
    result = list(order)
    result.remove(node)
    result.insert(landing, node)
    return tuple(result)


def _candidate_key(candidate: HardCandidate):
    graph = np.asarray(candidate.adjacency, dtype=np.uint8)
    return candidate.target_score, int(graph.sum()), graph.tobytes()


def fit_gflop(
    backend: GlobalOrderBackend,
    pairs: Sequence[tuple[int, int]],
    config: GFlopConfig = GFlopConfig(),
    *,
    initial_order: Sequence[int] | None = None,
    initial_state: Any | None = None,
    feasible_initializations: Sequence[HardCandidate] = (),
) -> GFlopResult:
    """Run score-independent order search and return a certified hard DAG."""
    started = perf_counter()
    d = int(backend.dimension)
    pairs = canonical_pairs(d, pairs) if config.use_notreks else ()
    if not config.continuation_weights or any(
            value < 0 for value in config.continuation_weights):
        raise ValueError("continuation weights must be nonempty and nonnegative")
    if min(config.screening_budget, config.refinement_budget) < 1:
        raise ValueError("optimizer budgets must be positive")
    order = tuple(range(d)) if initial_order is None else tuple(map(int, initial_order))
    order_mask(order)  # validate
    state = backend.initialize(order, initial_state)
    protected = []
    updates = 0

    def register(items):
        nonlocal updates
        for item in items:
            graph = np.asarray(item.adjacency, dtype=np.uint8)
            if not is_dag(graph) or no_trek_violations(graph, pairs):
                continue
            protected.append(HardCandidate(graph.copy(), float(item.target_score),
                                           dict(item.diagnostics)))
            protected.sort(key=_candidate_key)
            del protected[config.archive_size:]
            updates += 1

    register(feasible_initializations)
    if config.return_if_initial_feasible and protected:
        best = protected[0]
        return GFlopResult(
            best.adjacency, best.target_score, order, np.zeros((d, d)),
            perf_counter() - started, 0, [], updates, 0., 0., 0., 0,
            type(backend).__name__, config.use_notreks)

    trace, archive = [], []
    accepted = 0
    screening_seconds = refinement_seconds = postselection_seconds = 0.
    current = None

    def quick_candidates(result, candidate_order):
        callback = getattr(backend, "quick_hard_candidates", None)
        if callback is None:
            return backend.hard_candidates(
                result, candidate_order, use_notreks=config.use_notreks,
                budget_seconds=config.finalist_postselection_seconds)
        return callback(
            result, candidate_order, use_notreks=config.use_notreks)

    for stage, penalty_weight in enumerate(config.continuation_weights):
        stage_weight = float(penalty_weight if config.use_notreks else 0.)
        refined_started = perf_counter()
        current = backend.optimize(
            order, state, penalty_weight=stage_weight,
            budget=config.refinement_budget)
        refinement_seconds += perf_counter() - refined_started
        state = current.state
        archive.append((current, order))
        hard_started = perf_counter()
        register(quick_candidates(current, order))
        postselection_seconds += perf_counter() - hard_started
        stage_accepted = 0
        for sweep in range(config.sweeps_per_stage):
            improved = False
            for node in tuple(order):
                proposals = []
                screened_started = perf_counter()
                for landing in range(d):
                    candidate_order = _reinsert(order, node, landing)
                    if candidate_order == order:
                        continue
                    transported = backend.transport(
                        current.state, order, candidate_order)
                    result = backend.optimize(
                        candidate_order, transported,
                        penalty_weight=stage_weight,
                        budget=config.screening_budget)
                    proposals.append((result.merit(stage_weight), candidate_order, result))
                proposals.sort(key=lambda item: (item[0], item[1]))
                # Give the current order the same screening allocation as
                # every candidate before equal-budget refinement.
                current_screened = backend.optimize(
                    order, current.state, penalty_weight=stage_weight,
                    budget=config.screening_budget)
                screening_seconds += perf_counter() - screened_started
                finalists = [(order, current_screened)] + [
                    (candidate_order, result)
                    for _, candidate_order, result in proposals[:config.refinement_top_k]]
                refined = []
                refined_started = perf_counter()
                for candidate_order, result in finalists:
                    polished = backend.optimize(
                        candidate_order, result.state,
                        penalty_weight=stage_weight,
                        budget=config.refinement_budget)
                    refined.append((polished.merit(stage_weight), candidate_order, polished))
                refinement_seconds += perf_counter() - refined_started
                refined.sort(key=lambda item: (item[0], item[1]))
                hard_started = perf_counter()
                for _, finalist_order, finalist in refined:
                    register(quick_candidates(finalist, finalist_order))
                postselection_seconds += perf_counter() - hard_started
                best_merit, best_order, best_result = refined[0]
                current_entry = next(item for item in refined if item[1] == order)
                if (best_order != order
                        and best_merit < current_entry[0] - config.minimum_improvement):
                    order, current, state = best_order, best_result, best_result.state
                    accepted += 1
                    stage_accepted += 1
                    improved = True
                    archive.append((current, order))
                    hard_started = perf_counter()
                    register(quick_candidates(current, order))
                    postselection_seconds += perf_counter() - hard_started
                else:
                    current = current_entry[2]
                    state = current.state
            if not improved:
                break
        trace.append({
            "stage": stage, "penalty_weight": stage_weight,
            "base_score": current.base_score,
            "regularizer": current.regularizer,
            "notreks_penalty": current.notreks_penalty,
            "merit": current.merit(stage_weight),
            "accepted_order_moves": stage_accepted,
            "hard_feasible_incumbents": len(protected),
            "backend_diagnostics": dict(current.diagnostics),
        })

    # Run the full budgeted hard selector on the strongest dense states.
    post_started = perf_counter()
    for result, candidate_order in sorted(
            archive, key=lambda item: item[0].merit(
                config.continuation_weights[-1] if config.use_notreks else 0.))[
                    :config.archive_size]:
        register(backend.hard_candidates(
            result, candidate_order, use_notreks=config.use_notreks,
            budget_seconds=config.postselection_seconds))
    postselection_seconds += perf_counter() - post_started
    if not protected:
        raise RuntimeError("gFLOP found no independently certified hard candidate")
    best = protected[0]
    violations = len(no_trek_violations(best.adjacency, pairs))
    if not is_dag(best.adjacency) or violations:
        raise RuntimeError("protected gFLOP incumbent failed independent verification")
    return GFlopResult(
        adjacency=best.adjacency, target_score=best.target_score,
        order=tuple(topological_order(best.adjacency)),
        edge_strengths=np.asarray(current.edge_strengths).copy(),
        runtime=perf_counter() - started, accepted_order_moves=accepted,
        continuation_trace=trace, protected_incumbent_updates=updates,
        screening_seconds=screening_seconds,
        refinement_seconds=refinement_seconds,
        postselection_seconds=postselection_seconds,
        no_trek_violations=violations,
        backend_name=type(backend).__name__, use_notreks=config.use_notreks)
