"""Seeded, symmetrized FLOP-union superstructures for DAGMA-NOTREKS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    gaussian_bic,
    is_dag,
)


@dataclass(frozen=True)
class FlopUnionSupport:
    allowed_arcs: np.ndarray
    selected_dags: tuple[np.ndarray, ...]
    best_feasible_dag: np.ndarray | None
    best_feasible_coefficients: np.ndarray | None
    diagnostics: dict


def _selected_dag(diagnostics: dict, dimension: int) -> np.ndarray:
    adjacency = np.zeros((dimension, dimension), dtype=np.uint8)
    for parent, child in diagnostics["selected_dag_edges"]:
        adjacency[int(parent), int(child)] = 1
    return adjacency


def build_flop_union_support(
    data: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]],
    *,
    runs: int,
    seed: int,
    seed_stride: int = 7919,
    lambda_bic: float = 2.0,
) -> FlopUnionSupport:
    """Return a symmetrized union; every selected adjacency allows both arcs."""
    if runs < 1:
        raise ValueError("FLOP support runs must be positive")
    if seed_stride < 1:
        raise ValueError("FLOP support seed_stride must be positive")
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("data must be two-dimensional")
    import flopsearch

    dimension = X.shape[1]
    union_skeleton = np.zeros((dimension, dimension), dtype=np.uint8)
    dags = []
    rows = []
    feasible = []
    for index in range(runs):
        algorithm_seed = (int(seed) + index * int(seed_stride)) % (2**64)
        _, flop_diagnostics = flopsearch.flop_notreks(
            X, float(lambda_bic), [], restarts=0, seed=algorithm_seed,
            max_signature_rounds=0, search_version="fixed_signature_a",
            return_diagnostics=True)
        dag = _selected_dag(flop_diagnostics, dimension)
        if not is_dag(dag):
            raise RuntimeError("seeded FLOP support run returned a cyclic DAG")
        skeleton = (dag | dag.T).astype(np.uint8)
        union_skeleton |= skeleton
        violations = common_ancestor_violations(dag, no_trek_pairs)
        score, coefficients = gaussian_bic(X, dag, lambda_bic=lambda_bic)
        dags.append(dag)
        rows.append({
            "run": index, "algorithm_seed": algorithm_seed,
            "selected_edges": int(dag.sum()), "exact_bic": float(score),
            "no_trek_violations": int(violations),
            "selected_dag_edges": [tuple(map(int, edge))
                                   for edge in np.argwhere(dag)],
        })
        if violations == 0:
            feasible.append((float(score), int(dag.sum()), index, dag, coefficients))
    allowed = union_skeleton.copy()
    np.fill_diagonal(allowed, 0)
    best = min(feasible, key=lambda item: item[:3]) if feasible else None
    possible = dimension * (dimension - 1)
    diagnostics = {
        "support_mode": "flop_union_support",
        "flop_support_runs": runs,
        "flop_support_seed": int(seed),
        "flop_support_seed_stride": int(seed_stride),
        "support_is_symmetrized": True,
        "allowed_directed_arcs": int(allowed.sum()),
        "support_density": float(allowed.sum() / possible) if possible else 0.0,
        "unique_directed_dags": len({dag.tobytes() for dag in dags}),
        "unique_skeletons": len({(dag | dag.T).tobytes() for dag in dags}),
        "feasible_flop_initializers": len(feasible),
        "selected_feasible_initializer_run": None if best is None else best[2],
        "runs": rows,
    }
    return FlopUnionSupport(
        allowed_arcs=allowed,
        selected_dags=tuple(dags),
        best_feasible_dag=None if best is None else best[3].copy(),
        best_feasible_coefficients=None if best is None else best[4].copy(),
        diagnostics=diagnostics,
    )


def excluded_edges(allowed_arcs: np.ndarray) -> list[tuple[int, int]]:
    allowed = np.asarray(allowed_arcs, dtype=bool)
    if allowed.ndim != 2 or allowed.shape[0] != allowed.shape[1]:
        raise ValueError("allowed_arcs must be square")
    return [(row, column) for row in range(len(allowed))
            for column in range(len(allowed))
            if row != column and not allowed[row, column]]
