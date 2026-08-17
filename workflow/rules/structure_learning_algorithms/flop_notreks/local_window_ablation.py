"""Local-window ablation: order reinsertion plus parent-family blocks.

The block solve is exact only over the declared candidate families, proposed
order, selected block, and frozen outside graph.  Feasibility is always
checked in the complete graph.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from time import perf_counter
from typing import Sequence

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic, is_dag, topological_order,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.exact_solver import (
    ancestor_bitsets, canonical_pairs, no_trek_violations,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.flop_support import (
    build_flop_union_support,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.global_greedy import (
    _LocalGaussianBIC,
)


@dataclass(frozen=True)
class LocalWindowConfig:
    block_size: int = 4
    sweeps: int = 1
    initial_flop_runs: int = 4
    seed: int = 1729
    seed_stride: int = 7919
    lambda_bic: float = 2.0
    candidate_max_parents: int = 3
    max_families_per_node: int = 24
    block_time_limit_seconds: float = 2.0
    overall_time_limit_seconds: float = 60.0
    max_block_calls: int = 1000
    minimum_improvement: float = 1e-9


@dataclass
class BlockResult:
    adjacency: np.ndarray
    score: float
    lower_bound: float
    absolute_gap: float
    certified: bool
    timed_out: bool
    combinations_total: int
    combinations_evaluated: int
    feasible_combinations: int
    notreks_rejections: int
    changed_parent_sets: int
    runtime: float
    block: tuple[int, ...]
    family_counts: tuple[int, ...]


@dataclass
class LocalWindowResult:
    adjacency: np.ndarray
    order: tuple[int, ...]
    score: float
    runtime: float
    no_trek_violations: int
    block_size: int
    number_of_block_calls: int
    certified_block_solves: int
    timed_out_block_solves: int
    maximum_block_gap: float
    accepted_block_improvements: int
    jointly_changed_parent_sets: int
    outer_search_seconds: float
    block_solving_seconds: float
    initial_solution: str
    support_allowed_arcs: int
    source_cuts: int
    trek_cuts: int
    termination_reason: str
    block_diagnostics: list[dict]


def _reinsert(order: Sequence[int], node: int, position: int) -> list[int]:
    candidate = list(order)
    candidate.remove(node)
    candidate.insert(position, node)
    return candidate


def _delete_backward_edges(adjacency: np.ndarray, order: Sequence[int]):
    position = np.empty(len(order), dtype=int)
    position[np.asarray(order, dtype=int)] = np.arange(len(order))
    result = np.asarray(adjacency, dtype=np.uint8).copy()
    deleted = []
    for parent, child in np.argwhere(result):
        if position[parent] >= position[child]:
            result[parent, child] = 0
            deleted.append((int(parent), int(child)))
    return result, deleted


def _score_graph(local_scorer, adjacency):
    return float(sum(local_scorer.score(
        child, np.flatnonzero(adjacency[:, child]))
        for child in range(len(adjacency))))


def _greedy_flop_family(child, predecessors, local_scorer, max_parents):
    """Deterministic FLOP-style grow/shrink local parent fit."""
    parents = ()
    while len(parents) < max_parents:
        current = local_scorer.score(child, parents)
        choices = []
        for parent in predecessors:
            if parent not in parents:
                candidate = tuple(sorted((*parents, parent)))
                choices.append((local_scorer.score(child, candidate), candidate))
        if not choices:
            break
        best_score, best = min(choices)
        if best_score > current:
            break
        parents = best
    while parents:
        current = local_scorer.score(child, parents)
        choices = [(local_scorer.score(
            child, tuple(node for node in parents if node != removed)), removed)
                   for removed in parents]
        best_score, removed = min(choices)
        if best_score > current:
            break
        parents = tuple(node for node in parents if node != removed)
    return parents


def _candidate_families(child, order, baseline, local_scorer,
                        allowed_arcs, observed_parent_sets, config):
    position = {node: index for index, node in enumerate(order)}
    # gFLOP is not hard-restricted to the FLOP-union superstructure.  The
    # union contributes observed families below, while single-add/replacement
    # candidates may use every predecessor in the proposed order.
    predecessors = list(order[:position[child]])
    baseline = tuple(sorted(map(int, baseline)))
    families = {baseline, ()}
    for parent in baseline:
        families.add(tuple(node for node in baseline if node != parent))
    for parent in predecessors:
        if parent not in baseline:
            families.add(tuple(sorted((*baseline, parent))))
    for removed in baseline:
        retained = tuple(node for node in baseline if node != removed)
        for added in predecessors:
            if added not in retained:
                families.add(tuple(sorted((*retained, added))))
    observed_valid = []
    for parents in observed_parent_sets:
        parents = tuple(sorted(map(int, parents)))
        if all(parent in predecessors for parent in parents):
            families.add(parents)
            if parents not in observed_valid:
                observed_valid.append(parents)
    best_flop_family = _greedy_flop_family(
        child, predecessors, local_scorer, config.candidate_max_parents)
    families.add(best_flop_family)
    mandatory = [baseline]
    if baseline != ():
        mandatory.append(())
    mandatory.extend(parents for parents in observed_valid
                     if parents not in mandatory)
    if best_flop_family not in mandatory:
        mandatory.append(best_flop_family)
    ranked = sorted(families,
                    key=lambda parents: (local_scorer.score(child, parents), parents))
    retained = list(mandatory)
    retained.extend(parents for parents in ranked if parents not in mandatory)
    return tuple(retained[:max(config.max_families_per_node, len(mandatory))])


def solve_parent_family_block(
    data: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    baseline_graph: np.ndarray,
    order: Sequence[int],
    block: Sequence[int],
    families_by_node: dict[int, Sequence[tuple[int, ...]]],
    *,
    lambda_bic: float = 2.0,
    time_limit_seconds: float = 2.0,
) -> BlockResult:
    """Globally enumerate the declared family product with full-graph checks."""
    started = perf_counter()
    graph = np.asarray(baseline_graph, dtype=np.uint8)
    if not is_dag(graph) or no_trek_violations(graph, pairs):
        raise ValueError("baseline block graph must be hard feasible")
    position = {node: index for index, node in enumerate(order)}
    nodes = tuple(map(int, block))
    families = []
    for child in nodes:
        baseline = tuple(map(int, np.flatnonzero(graph[:, child])))
        declared = {tuple(sorted(map(int, parents)))
                    for parents in families_by_node[child]}
        declared.add(baseline)
        if any(any(position[parent] >= position[child] for parent in parents)
               for parents in declared):
            raise ValueError("declared family contains a non-predecessor parent")
        families.append(tuple(sorted(declared)))
    scorer = _LocalGaussianBIC(np.asarray(data, dtype=float), lambda_bic)
    outside_score = sum(scorer.score(child, np.flatnonzero(graph[:, child]))
                        for child in range(len(graph)) if child not in nodes)
    lower_bound = float(outside_score + sum(
        min(scorer.score(child, parents) for parents in child_families)
        for child, child_families in zip(nodes, families)))
    baseline_score = _score_graph(scorer, graph)
    best = (baseline_score, graph.tobytes(), graph.copy())
    evaluated = feasible = 0
    timed_out = False
    total = int(np.prod([len(child_families) for child_families in families]))
    for assignment in product(*families):
        if perf_counter() - started >= time_limit_seconds:
            timed_out = True
            break
        candidate = graph.copy()
        for child, parents in zip(nodes, assignment):
            candidate[:, child] = 0
            candidate[list(parents), child] = 1
        evaluated += 1
        # Fixed order gives acyclicity; this assertion guards interface errors.
        if not is_dag(candidate):
            raise AssertionError("fixed-order block candidate became cyclic")
        if no_trek_violations(candidate, pairs):
            continue
        feasible += 1
        score = float(outside_score + sum(
            scorer.score(child, parents)
            for child, parents in zip(nodes, assignment)))
        key = (score, candidate.tobytes(), candidate)
        if key[:2] < best[:2]:
            best = key
    changed = sum(
        tuple(np.flatnonzero(best[2][:, child]))
        != tuple(np.flatnonzero(graph[:, child])) for child in nodes)
    certified = not timed_out
    gap = 0.0 if certified else max(0.0, best[0] - lower_bound)
    return BlockResult(
        adjacency=best[2], score=best[0], lower_bound=lower_bound,
        absolute_gap=gap, certified=certified, timed_out=timed_out,
        combinations_total=total, combinations_evaluated=evaluated,
        feasible_combinations=feasible,
        notreks_rejections=evaluated - feasible,
        changed_parent_sets=changed,
        runtime=perf_counter() - started, block=nodes,
        family_counts=tuple(map(len, families)))


def _select_block(node, deleted, graph, order, local_scorer, allowed,
                  observed, pairs, config):
    selected = [int(node)]
    selected.extend(child for _, child in deleted if child not in selected)
    gains = []
    for child in range(len(graph)):
        baseline = tuple(np.flatnonzero(graph[:, child]))
        families = _candidate_families(
            child, order, baseline, local_scorer, allowed,
            observed.get(child, ()), config)
        best = min(local_scorer.score(child, parents) for parents in families)
        gains.append((best - local_scorer.score(child, baseline), child))
    selected.extend(child for _, child in sorted(gains)
                    if child not in selected)
    constrained_degree = [0] * len(graph)
    for left, right in pairs:
        constrained_degree[left] += 1
        constrained_degree[right] += 1
    selected.extend(child for child in sorted(
        range(len(graph)), key=lambda child: (-constrained_degree[child], child))
        if child not in selected)
    return tuple(selected[:min(config.block_size, len(graph))])


def fit_local_window_ablation(
    data: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    config: LocalWindowConfig = LocalWindowConfig(),
) -> LocalWindowResult:
    """Run the low-dimensional gFLOP large-neighborhood prototype."""
    started = perf_counter()
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data must be two-dimensional")
    if (config.block_size < 1 or config.sweeps < 1
            or config.max_block_calls < 1
            or config.overall_time_limit_seconds <= 0):
        raise ValueError("block size, sweeps, call limit, and time limit must be positive")
    pairs = canonical_pairs(X.shape[1], pairs)
    support = build_flop_union_support(
        X, pairs, runs=config.initial_flop_runs, seed=config.seed,
        seed_stride=config.seed_stride, lambda_bic=config.lambda_bic)
    # Match the vanilla comparator's multi-start search as an additional warm
    # start.  The independent support runs remain useful for support/families,
    # but need not select the same solution as one multi-start FLOP call.
    import flopsearch
    _, vanilla_diagnostics = flopsearch.flop_notreks(
        X, float(config.lambda_bic), [],
        restarts=max(0, config.initial_flop_runs - 1), seed=config.seed,
        max_signature_rounds=0, search_version="fixed_signature_a",
        return_diagnostics=True)
    vanilla_graph = np.zeros((X.shape[1], X.shape[1]), dtype=np.uint8)
    for parent, child in vanilla_diagnostics["selected_dag_edges"]:
        vanilla_graph[int(parent), int(child)] = 1
    feasible_initializers = []
    if support.best_feasible_dag is not None:
        feasible_initializers.append((
            gaussian_bic(X, support.best_feasible_dag,
                         lambda_bic=config.lambda_bic)[0],
            "best_feasible_independent_seeded_flop",
            support.best_feasible_dag))
    if not no_trek_violations(vanilla_graph, pairs):
        feasible_initializers.append((
            gaussian_bic(X, vanilla_graph,
                         lambda_bic=config.lambda_bic)[0],
            "feasible_vanilla_multistart_flop", vanilla_graph))
    if not feasible_initializers:
        graph = np.zeros((X.shape[1], X.shape[1]), dtype=np.uint8)
        initial_solution = "empty"
    else:
        _, initial_solution, initial_graph = min(
            feasible_initializers, key=lambda item: (item[0], item[1]))
        graph = initial_graph.copy()
    order = topological_order(graph)
    local_scorer = _LocalGaussianBIC(X, config.lambda_bic)
    score = _score_graph(local_scorer, graph)
    observed = {child: [] for child in range(X.shape[1])}
    for dag in support.selected_dags:
        for child in range(X.shape[1]):
            observed[child].append(tuple(np.flatnonzero(dag[:, child])))
    for child in range(X.shape[1]):
        observed[child].append(tuple(np.flatnonzero(vanilla_graph[:, child])))
    diagnostics = []
    block_seconds = 0.0
    accepted = jointly_changed = 0
    stop = False
    termination_reason = "sweep_limit"
    for sweep in range(config.sweeps):
        sweep_improved = False
        for node in list(order):
            old_position = order.index(node)
            for new_position in range(len(order)):
                if len(diagnostics) >= config.max_block_calls:
                    termination_reason, stop = "block_call_limit", True
                    break
                if perf_counter() - started >= config.overall_time_limit_seconds:
                    termination_reason, stop = "time_limit", True
                    break
                if new_position == old_position:
                    continue
                proposed_order = _reinsert(order, node, new_position)
                safe_graph, deleted = _delete_backward_edges(graph, proposed_order)
                if no_trek_violations(safe_graph, pairs):
                    raise AssertionError("edge deletion broke NOTREKS feasibility")
                block = _select_block(
                    node, deleted, safe_graph, proposed_order, local_scorer,
                    support.allowed_arcs, observed, pairs, config)
                family_map = {}
                for child in block:
                    family_map[child] = _candidate_families(
                        child, proposed_order,
                        np.flatnonzero(safe_graph[:, child]), local_scorer,
                        support.allowed_arcs, observed[child], config)
                result = solve_parent_family_block(
                    X, pairs, safe_graph, proposed_order, block, family_map,
                    lambda_bic=config.lambda_bic,
                    time_limit_seconds=config.block_time_limit_seconds)
                block_seconds += result.runtime
                row = asdict(result)
                row.pop("adjacency")
                row.update({"sweep": int(sweep), "moved_node": int(node),
                            "old_position": old_position,
                            "new_position": new_position,
                            "deleted_order_edges": deleted, "accepted": False})
                if result.score < score - config.minimum_improvement:
                    graph, order, score = (
                        result.adjacency, proposed_order, result.score)
                    row["accepted"] = True
                    accepted += 1
                    jointly_changed += result.changed_parent_sets
                    sweep_improved = True
                    diagnostics.append(row)
                    break
                diagnostics.append(row)
            if stop:
                break
            if sweep_improved:
                break
        if stop:
            break
        if not sweep_improved:
            termination_reason = "local_optimum"
            break
    runtime = perf_counter() - started
    violations = len(no_trek_violations(graph, pairs))
    if not is_dag(graph) or violations:
        raise RuntimeError("gFLOP returned an independently infeasible DAG")
    return LocalWindowResult(
        adjacency=graph, order=tuple(map(int, order)), score=float(score), runtime=runtime,
        no_trek_violations=violations, block_size=config.block_size,
        number_of_block_calls=len(diagnostics),
        certified_block_solves=sum(row["certified"] for row in diagnostics),
        timed_out_block_solves=sum(row["timed_out"] for row in diagnostics),
        maximum_block_gap=max((row["absolute_gap"] for row in diagnostics), default=0.),
        accepted_block_improvements=accepted,
        jointly_changed_parent_sets=jointly_changed,
        outer_search_seconds=runtime - block_seconds,
        block_solving_seconds=block_seconds,
        initial_solution=initial_solution,
        support_allowed_arcs=int(support.allowed_arcs.sum()),
        source_cuts=0, trek_cuts=0,
        termination_reason=termination_reason,
        block_diagnostics=diagnostics)
