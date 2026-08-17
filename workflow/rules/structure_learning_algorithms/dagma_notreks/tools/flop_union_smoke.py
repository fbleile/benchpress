#!/usr/bin/env python3
"""Small real numerical smoke test for masked FLOP-union DAGMA-NOTREKS."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations, is_dag,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    ProductionConfig, run_production_pipeline,
)

OUT = Path("results/dagma_notreks_oracle/flop_union_dagma_smoke")


def case(seed=4401):
    rng = np.random.default_rng(seed)
    n, d = 400, 8
    adjacency = np.zeros((d, d), dtype=np.uint8)
    for parent, child in ((0, 1), (0, 2), (1, 3),
                          (4, 5), (4, 6), (6, 7)):
        adjacency[parent, child] = 1
    weights = adjacency * rng.choice([-1., 1.], (d, d)) * .75
    X = np.zeros((n, d))
    noise = rng.normal(size=(n, d))
    for node in range(d):
        X[:, node] = noise[:, node] + X @ weights[:, node]
    pairs = [(left, right) for left in range(4) for right in range(4, 8)]
    return X, adjacency, pairs


def metrics(adjacency, truth, pairs):
    skeleton = (adjacency | adjacency.T).astype(bool)
    true_skeleton = (truth | truth.T).astype(bool)
    upper = np.triu(np.ones_like(truth, dtype=bool), 1)
    tp = int(np.sum(skeleton & true_skeleton & upper))
    fp = int(np.sum(skeleton & ~true_skeleton & upper))
    fn = int(np.sum(~skeleton & true_skeleton & upper))
    return {
        "directed_shd": int(np.sum(adjacency != truth)),
        "skeleton_shd": fp + fn,
        "skeleton_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "edge_count": int(adjacency.sum()),
        "is_dag": is_dag(adjacency),
        "no_trek_violations": common_ancestor_violations(adjacency, pairs),
    }


def main():
    X, truth, pairs = case()
    base = ProductionConfig(
        restarts=2, seed=7711, T=2, s=(1., .9), warm_iter=150,
        max_iter=200, checkpoint=50,
        postselection_policy="PS1_joint_feasible_greedy_score",
        max_search_seconds=.25, max_expanded_nodes=200, max_queue_size=200)
    rows = []
    for mode in ("unrestricted", "flop_union_support"):
        config = replace(base, support_mode=mode, flop_support_runs=2)
        selected, restarts = run_production_pipeline(X, pairs, config)
        row = {
            "method": ("dagma_notreks" if mode == "unrestricted"
                       else "dagma_notreks_flop_union_support"),
            "graph_seed": 4401, "algorithm_seed": config.seed,
            "knowledge_seed": 0, "knowledge_fraction": 1.0,
            "runtime": (sum(result.runtime for result in restarts)
                        + selected.support.get("support_build_seconds", 0.)),
            "selection_score": selected.exact_bic,
            "gaussian_bic": selected.gaussian_bic,
            "representation": "DAG", **metrics(selected.adjacency, truth, pairs),
            "support": selected.support,
        }
        if mode == "flop_union_support":
            allowed = selected.support["allowed_directed_arcs"]
            row["support_size"] = allowed
            row["support_density"] = selected.support["support_density"]
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
