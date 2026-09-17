"""Common graph-level postselection for FLOP and DAGMA candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.end_flop_prune import (
    end_flop_prune,
)
from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    gaussian_bic,
    is_dag,
)


@dataclass
class SharedPostselectionResult:
    adjacency: np.ndarray
    coefficients: np.ndarray
    bic: float
    input_edges: int
    masked_edges: int
    repaired_edges: int
    final_edges: int
    direct_mask_applied: bool
    notreks_applied: bool
    violations_before: int
    violations_after: int
    edges_deleted: int


def _direct_mask(graph: np.ndarray,
                 pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    result = np.asarray(graph, dtype=np.uint8).copy()
    for left, right in pairs:
        result[int(left), int(right)] = 0
        result[int(right), int(left)] = 0
    np.fill_diagonal(result, 0)
    return result


def _repair_notreks(data: np.ndarray, graph: np.ndarray,
                    pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Delete weakest edges until the exact common-ancestor constraint holds."""
    result = np.asarray(graph, dtype=np.uint8).copy()
    while common_ancestor_violations(result, pairs):
        _, coefficients = gaussian_bic(data, result, lambda_bic=2.)
        before = common_ancestor_violations(result, pairs)
        candidates = []
        for source, target in zip(*np.nonzero(result)):
            proposal = result.copy()
            proposal[source, target] = 0
            reduction = before - common_ancestor_violations(proposal, pairs)
            candidates.append((
                -int(reduction), abs(float(coefficients[source, target])),
                int(source), int(target)))
        if not candidates:
            raise RuntimeError("cannot repair NOTREKS candidate graph")
        _, _, source, target = min(candidates)
        result[source, target] = 0
    return result


def shared_graph_postselection(
    X: np.ndarray,
    candidate_dag: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]] = (),
    *,
    direct_mask: bool = False,
    notreks: bool = False,
    lambda_bic: float = 2.0,
) -> SharedPostselectionResult:
    """Apply identical graph-level postselection to any candidate DAG.

    The candidate is first optionally direct-masked, then optionally repaired
    for the full common-ancestor constraint, and finally passed through the
    canonical deletion-only BIC parent shrinker and OLS refit.
    """
    data = np.asarray(X, dtype=np.float64)
    graph = (np.asarray(candidate_dag) != 0).astype(np.uint8)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("candidate_dag must be square")
    if graph.shape[1] != data.shape[1] or not is_dag(graph):
        raise ValueError("candidate_dag must be a DAG matching X")
    pairs = tuple((int(i), int(j)) for i, j in no_trek_pairs)
    input_edges = int(graph.sum())
    violations_before = common_ancestor_violations(graph, pairs)
    if direct_mask:
        graph = _direct_mask(graph, pairs)
    masked_edges = int(graph.sum())
    if notreks:
        graph = _repair_notreks(data, graph, pairs)
    repaired_edges = int(graph.sum())
    if not is_dag(graph) or (notreks and common_ancestor_violations(graph, pairs)):
        raise RuntimeError("shared postselection produced an infeasible graph")
    final, coefficients, diagnostics = end_flop_prune(
        data, graph, lambda_bic=lambda_bic)
    violations_after = common_ancestor_violations(final, pairs)
    if notreks and violations_after:
        raise RuntimeError("BIC parent shrink violated NOTREKS feasibility")
    bic, refit = gaussian_bic(data, final, lambda_bic=lambda_bic)
    return SharedPostselectionResult(
        adjacency=final,
        coefficients=refit,
        bic=float(bic),
        input_edges=input_edges,
        masked_edges=masked_edges,
        repaired_edges=repaired_edges,
        final_edges=int(final.sum()),
        direct_mask_applied=bool(direct_mask),
        notreks_applied=bool(notreks),
        violations_before=int(violations_before),
        violations_after=int(violations_after),
        edges_deleted=int(input_edges - final.sum()),
    )
