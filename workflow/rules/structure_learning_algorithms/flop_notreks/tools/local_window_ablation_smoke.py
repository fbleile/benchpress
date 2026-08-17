#!/usr/bin/env python3
"""Real low-dimensional smoke for the preserved local-window ablation."""
import json
from pathlib import Path

import flopsearch
import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic, is_dag,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.exact_solver import (
    no_trek_violations,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.local_window_ablation import (
    LocalWindowConfig, fit_local_window_ablation,
)

OUT = Path("results/dagma_notreks_oracle/local_window_ablation_smoke")


def case(seed=21):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 6))
    truth = np.zeros((6, 6), dtype=np.uint8)
    for parent, child, weight in ((0, 1, .55), (1, 2, .5),
                                  (3, 4, .55), (4, 5, .5)):
        truth[parent, child] = 1
        X[:, child] += weight * X[:, parent]
    pairs = [(left, right) for left in range(3) for right in range(3, 6)]
    return X, truth, pairs


def selected_dag(diagnostics, dimension):
    adjacency = np.zeros((dimension, dimension), dtype=np.uint8)
    for parent, child in diagnostics["selected_dag_edges"]:
        adjacency[parent, child] = 1
    return adjacency


def metrics(name, adjacency, truth, X, pairs, runtime, extra=None):
    skeleton = (adjacency | adjacency.T).astype(bool)
    true_skeleton = (truth | truth.T).astype(bool)
    upper = np.triu(np.ones_like(truth, dtype=bool), 1)
    tp = int(np.sum(skeleton & true_skeleton & upper))
    fp = int(np.sum(skeleton & ~true_skeleton & upper))
    fn = int(np.sum(~skeleton & true_skeleton & upper))
    bic, _ = gaussian_bic(X, adjacency, lambda_bic=2.)
    return {
        "method": name, "graph_seed": 21, "algorithm_seed": 21,
        "knowledge_seed": 0, "knowledge_fraction": 1.,
        "representation": "DAG", "runtime": runtime,
        "gaussian_bic": bic, "edge_count": int(adjacency.sum()),
        "directed_shd": int(np.sum(adjacency != truth)),
        "skeleton_shd": fp + fn,
        "skeleton_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "is_dag": is_dag(adjacency),
        "no_trek_violations": len(no_trek_violations(adjacency, pairs)),
        **(extra or {}),
    }


def main():
    X, truth, pairs = case()
    X = (X - X.mean(0)) / X.std(0)
    rows = []
    for name, version, supplied in (
            ("vanilla_flop", "fixed_signature_a", []),
            ("global_greedy_hybrid", "global_greedy_hybrid", pairs)):
        import time
        started = time.perf_counter()
        _, diagnostics = flopsearch.flop_notreks(
            X, 2., supplied, restarts=1, seed=21,
            max_signature_rounds=2, search_version=version,
            return_diagnostics=True)
        adjacency = selected_dag(diagnostics, len(truth))
        rows.append(metrics(name, adjacency, truth, X, pairs,
                            time.perf_counter() - started))
    result = fit_local_window_ablation(
        X, pairs, LocalWindowConfig(
            block_size=4, sweeps=2, initial_flop_runs=2, seed=21,
            candidate_max_parents=2, max_families_per_node=16,
            block_time_limit_seconds=1.))
    rows.append(metrics(
        "local_window_ablation", result.adjacency, truth, X, pairs, result.runtime,
        {key: value for key, value in result.__dict__.items()
         if key not in {"adjacency", "block_diagnostics"}}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "block_diagnostics.json").write_text(
        json.dumps(result.block_diagnostics, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
