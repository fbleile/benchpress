"""Canonical weighted-to-graph policies."""

from __future__ import annotations

import time
import warnings

import numpy as np

from .candidate_generation import DEFAULT_THRESHOLDS, generate_candidates
from .local_search import optimize_graph_locally
from .types import (
    GraphCandidate,
    PostprocessingBudget,
    PostprocessingResult,
    WeightedGraphEstimate,
)


POLICY_ALIASES = {
    "threshold_grid_bic_greedy": "threshold_grid_score_search",
    "PS1_joint_feasible_greedy_score": "threshold_grid_score_search",
    "PS2_joint_feasible_local_search": "weighted_feasible_local_search",
    "PS5_fixed_threshold_joint_feasible": "fixed_threshold",
}


def _canonical(policy):
    if policy in POLICY_ALIASES:
        warnings.warn(
            f"{policy} is deprecated; use {POLICY_ALIASES[policy]}",
            DeprecationWarning, stacklevel=3)
        return POLICY_ALIASES[policy]
    return policy


def _choose(candidates):
    return min(candidates, key=lambda item: (
        float(item.score), int(item.graph.sum()), item.graph.tobytes()))


def threshold_grid_score_search(
    estimate, data, score, constraints, budget,
    *, thresholds=DEFAULT_THRESHOLDS, local_search=True,
):
    started = time.perf_counter()
    starts = generate_candidates(
        estimate, constraints, thresholds=thresholds)
    evaluated = []
    for candidate in starts:
        initial_score = score.score_graph(candidate.graph, data)
        graph, final_score, diagnostics = (
            optimize_graph_locally(
                candidate.graph, estimate, data, score, constraints, budget)
            if local_search else
            (candidate.graph, initial_score, {
                "termination_reason": "candidate_scoring_only",
                "moves_evaluated": 0,
                "moves_accepted": 0,
                "moves_rejected_infeasible": 0,
            }))
        evaluated.append(GraphCandidate(
            graph, candidate.construction, candidate.threshold,
            final_score, constraints.is_feasible(graph), {
                **candidate.diagnostics,
                **diagnostics,
                "initial_score": initial_score,
            }))
    selected = _choose(evaluated)
    return PostprocessingResult(
        selected.graph.copy(), float(selected.score),
        "threshold_grid_score_search", 0, evaluated, {
            "thresholds_considered": list(thresholds),
            "candidate_count": len(evaluated),
            "runtime_seconds": time.perf_counter() - started,
            **constraints.violation_summary(selected.graph),
        })


def weighted_feasible_local_search(
    estimate, data, score, constraints, budget,
    *, thresholds=DEFAULT_THRESHOLDS,
):
    result = threshold_grid_score_search(
        estimate, data, score, constraints, budget,
        thresholds=thresholds, local_search=True)
    result.policy = "weighted_feasible_local_search"
    return result


def fixed_threshold(
    estimate, data, score, constraints, budget, *, threshold=0.30,
):
    return threshold_grid_score_search(
        estimate, data, score, constraints, budget,
        thresholds=(float(threshold),), local_search=False)


def postprocess_weighted_graph(
    estimate,
    data,
    score,
    constraints,
    policy="threshold_grid_score_search",
    budget=PostprocessingBudget(),
    **policy_options,
):
    if not isinstance(estimate, WeightedGraphEstimate):
        estimate = WeightedGraphEstimate(np.asarray(estimate, dtype=float))
    name = _canonical(policy)
    registry = {
        "threshold_grid_score_search": threshold_grid_score_search,
        "weighted_feasible_local_search": weighted_feasible_local_search,
        "adaptive_feasible_local_search": weighted_feasible_local_search,
        "fixed_threshold": fixed_threshold,
    }
    if name not in registry:
        raise ValueError(f"unknown weighted-graph policy: {policy}")
    return registry[name](
        estimate, data, score, constraints, budget, **policy_options)


def postprocess_weighted_graphs(
    estimates, data, score, constraints,
    policy="threshold_grid_score_search",
    budget=PostprocessingBudget(),
    **policy_options,
):
    results = [
        postprocess_weighted_graph(
            estimate, data, score, constraints, policy, budget,
            **policy_options)
        for estimate in estimates]
    selected_index, selected = min(enumerate(results), key=lambda item: (
        item[1].score, int(item[1].graph.sum()),
        item[1].graph.tobytes(), item[0]))
    selected.source_estimate_index = selected_index
    selected.diagnostics["weighted_estimate_count"] = len(results)
    return selected
