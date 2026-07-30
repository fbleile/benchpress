"""Weight-informed candidate generation."""

from __future__ import annotations

import numpy as np

from .types import GraphCandidate, WeightedGraphEstimate


DEFAULT_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.20, 0.30)


def _project_by_weakest_edge(graph, weights, constraints):
    projected = np.asarray(graph, dtype=int).copy()
    removed = []
    while not constraints.is_feasible(projected) and projected.any():
        edges = [
            (abs(float(weights[i, j])), int(i), int(j))
            for i, j in zip(*np.nonzero(projected))]
        _, i, j = min(edges)
        projected[i, j] = 0
        removed.append((i, j))
    return projected, removed


def generate_candidates(
    estimate: WeightedGraphEstimate,
    constraints,
    *,
    thresholds=DEFAULT_THRESHOLDS,
):
    weights = estimate.weights
    candidates = []
    seen = set()
    for threshold in sorted(set(float(x) for x in thresholds)):
        raw = (np.abs(weights) >= threshold).astype(int)
        np.fill_diagonal(raw, 0)
        graph, removed = _project_by_weakest_edge(raw, weights, constraints)
        key = graph.tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(GraphCandidate(
            graph=graph,
            construction="absolute_weight_threshold",
            threshold=threshold,
            feasible=constraints.is_feasible(graph),
            diagnostics={
                "raw_edges": int(raw.sum()),
                "removed_for_feasibility": removed,
            }))
    if not candidates:
        empty = np.zeros_like(weights, dtype=int)
        candidates.append(GraphCandidate(
            empty, "empty_fallback", None, feasible=True))
    return candidates
