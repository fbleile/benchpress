"""Budgeted, weight-informed, score-accepted graph local search."""

from __future__ import annotations

import time

import numpy as np


def _key(score, graph):
    return (float(score), int(graph.sum()), graph.tobytes())


def optimize_graph_locally(
    initial_graph,
    estimate,
    data,
    score,
    constraints,
    budget,
):
    started = time.perf_counter()
    graph = np.asarray(initial_graph, dtype=int).copy()
    best_score = score.score_graph(graph, data)
    evaluated = accepted = rejected = iterations = 0
    weights = np.abs(estimate.weights)
    all_edges = [
        (float(weights[i, j]), int(i), int(j))
        for i in range(len(graph)) for j in range(len(graph))
        if i != j and weights[i, j] > 0]
    while iterations < budget.max_local_search_iterations:
        if time.perf_counter() - started >= budget.max_seconds:
            termination = "max_seconds"
            break
        proposals = []
        # Weakest retained edges are tried for deletion first.
        for _, i, j in sorted(
                (edge for edge in all_edges if graph[edge[1], edge[2]])):
            proposal = graph.copy()
            proposal[i, j] = 0
            proposals.append(("delete", i, j, proposal))
        # Strongest excluded weighted edges are tried for addition.
        for _, i, j in sorted(
                (edge for edge in all_edges if not graph[edge[1], edge[2]]),
                reverse=True):
            proposal = graph.copy()
            proposal[i, j] = 1
            proposals.append(("add", i, j, proposal))
        best_move = None
        for move, i, j, proposal in proposals:
            if evaluated >= budget.max_candidate_evaluations:
                termination = "max_candidate_evaluations"
                break
            if not constraints.is_feasible(proposal):
                rejected += 1
                continue
            candidate_score = score.score_graph(proposal, data)
            evaluated += 1
            candidate_key = _key(candidate_score, proposal)
            if candidate_key < _key(best_score, graph) and (
                    best_move is None or candidate_key < best_move[0]):
                best_move = (candidate_key, move, i, j, proposal)
        else:
            termination = "local_optimum" if best_move is None else "improved"
        if best_move is None:
            break
        _, _, _, _, graph = best_move
        best_score = best_move[0][0]
        accepted += 1
        iterations += 1
    else:
        termination = "max_local_search_iterations"
    return graph, best_score, {
        "moves_evaluated": evaluated,
        "moves_rejected_infeasible": rejected,
        "moves_accepted": accepted,
        "local_search_iterations": iterations,
        "termination_reason": termination,
        "runtime_seconds": time.perf_counter() - started,
    }
