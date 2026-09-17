"""d=20 smoke test for shared graph-level postselection."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import flopsearch

from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    ProductionConfig, production_candidate_graph, run_production_pipeline,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.shared_postselection import (
    shared_graph_postselection,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import generate


def metrics(A, truth):
    estimated = np.asarray(A, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    return {
        "edges": int(estimated.sum()),
        "true_edges": int(truth.sum()),
        "true_positive": int(np.sum(estimated & truth)),
        "false_positive": int(np.sum(estimated & ~truth)),
        "false_negative": int(np.sum(~estimated & truth)),
        "directed_shd": int(np.sum(estimated != truth)),
    }


def direct_mask_edges(pairs):
    return sorted({(int(u), int(v)) for u, v in pairs} |
                  {(int(v), int(u)) for u, v in pairs})


def flop_candidate(X, pairs, knowledge, seed):
    forbidden = (direct_mask_edges(pairs)
                 if knowledge in {"direct_mask", "full_notreks_direct_mask"}
                 else [])
    if knowledge in {"none", "direct_mask", "full_notreks_direct_mask"}:
        _, diagnostic = flopsearch.flop_notreks(
            X, 2., (pairs if knowledge == "full_notreks_direct_mask" else []),
            forbidden_edges=forbidden, restarts=1, seed=seed,
            search_version="global_greedy_rust", return_diagnostics=True)
    elif knowledge == "full_notreks":
        _, diagnostic = flopsearch.flop_notreks(
            X, 2., pairs, restarts=1, seed=seed,
            max_signature_rounds=8, search_version="global_greedy_rust",
            return_diagnostics=True)
    else:
        raise ValueError(knowledge)
    graph = np.zeros((X.shape[1], X.shape[1]), dtype=np.uint8)
    for source, target in diagnostic["selected_dag_edges"]:
        graph[int(source), int(target)] = 1
    return graph


def dagma_candidate(X, pairs, knowledge, seed):
    use_notreks = knowledge in {"full_notreks", "full_notreks_direct_mask"}
    edge_mask = None
    if knowledge in {"direct_mask", "full_notreks_direct_mask"}:
        edge_mask = np.ones((X.shape[1], X.shape[1]), dtype=float)
        for u, v in direct_mask_edges(pairs):
            edge_mask[u, v] = 0.0
        np.fill_diagonal(edge_mask, 0.0)
    result, _ = run_production_pipeline(
        X, pairs if use_notreks else [], ProductionConfig(
            restarts=5, seed=seed, edge_mask=edge_mask))
    candidate, _ = production_candidate_graph(
        result.weighted_adjacency, [], screening_floor=.01)
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4001)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, truth, pairs = generate(args.seed)
    candidates = {}
    for method, candidate_fn in (("flop", flop_candidate),
                                 ("dagma", dagma_candidate)):
      for knowledge in ("none", "direct_mask", "full_notreks",
                        "full_notreks_direct_mask"):
        started = time.perf_counter()
        candidates[(method, knowledge)] = candidate_fn(
            X, pairs, knowledge, args.seed)
        print(method, knowledge, "candidate_seconds",
              time.perf_counter() - started, flush=True)

    rows = []
    for (method, knowledge), candidate in candidates.items():
        for label, notreks in (("ordinary", False),
                               ("notreks_postselection", True)):
            started = time.perf_counter()
            result = shared_graph_postselection(
                X, candidate, pairs, direct_mask=False,
                notreks=notreks, lambda_bic=2.)
            rows.append({
                "method": method,
                "knowledge": knowledge,
                "variant": label,
                **metrics(result.adjacency, truth),
                "input_edges": result.input_edges,
                "masked_edges": result.masked_edges,
                "repaired_edges": result.repaired_edges,
                "edges_deleted": result.edges_deleted,
                "violations_before": result.violations_before,
                "violations_after": result.violations_after,
                "postselection_seconds": time.perf_counter() - started,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "results.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
