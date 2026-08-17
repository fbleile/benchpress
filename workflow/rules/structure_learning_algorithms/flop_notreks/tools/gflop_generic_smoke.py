#!/usr/bin/env python3
"""Small real d=20 smoke for generic gFLOP and preserved baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import flopsearch
import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic, is_dag, topological_order,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.exact_solver import (
    no_trek_violations,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.gflop import (
    GFlopConfig, HardCandidate, fit_gflop,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.gflop_dagma_backend import (
    DagmaBackendConfig, DagmaOrderBackend,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.local_window_ablation import (
    LocalWindowConfig, fit_local_window_ablation,
)

OUT = Path("results/dagma_notreks_oracle/gflop_generic_d20_smoke")


def case(seed=8401, knowledge_seed=9401, n=500, d=20):
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    truth = np.zeros((d, d), dtype=np.uint8)
    for left in range(d):
        for right in range(left + 1, d):
            if rng.random() < 2.5 / (d - 1):
                truth[order[left], order[right]] = 1
    weights = truth * rng.uniform(.5, .9, truth.shape) * rng.choice([-1, 1], truth.shape)
    X = np.zeros((n, d))
    noise = rng.normal(size=X.shape)
    for node in order:
        X[:, node] = noise[:, node] + X @ weights[:, node]
    X = (X - X.mean(0)) / X.std(0)
    reach = truth.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for node in range(d):
        reach |= reach[:, [node]] & reach[[node], :]
    all_pairs = [(a, b) for a in range(d) for b in range(a + 1, d)
                 if not np.any(reach[:, a] & reach[:, b])]
    chosen = np.random.default_rng(knowledge_seed).permutation(len(all_pairs))[
        :round(.25 * len(all_pairs))]
    return X, truth, sorted(all_pairs[index] for index in chosen)


def adjacency(diagnostics, d):
    graph = np.zeros((d, d), dtype=np.uint8)
    for parent, child in diagnostics["selected_dag_edges"]:
        graph[int(parent), int(child)] = 1
    return graph


def metrics(graph, truth, pairs):
    skeleton = (graph | graph.T).astype(bool)
    target = (truth | truth.T).astype(bool)
    upper = np.triu(np.ones_like(truth, dtype=bool), 1)
    tp = int(np.sum(skeleton & target & upper))
    fp = int(np.sum(skeleton & ~target & upper))
    fn = int(np.sum(~skeleton & target & upper))
    return {"edge_count": int(graph.sum()), "skeleton_shd": fp + fn,
            "skeleton_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "directed_shd": int(np.sum(graph != truth)),
            "violations": len(no_trek_violations(graph, pairs)),
            "is_dag": bool(is_dag(graph))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, default=8401)
    parser.add_argument("--algorithm-seed", type=int, default=7401)
    parser.add_argument("--knowledge-seed", type=int, default=9401)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    X, truth, pairs = case(
        args.data_seed, args.knowledge_seed, args.sample_size, args.dimension)
    common = {"data_seed": args.data_seed,
              "algorithm_seed": args.algorithm_seed,
              "knowledge_seed": args.knowledge_seed,
              "dimension": args.dimension, "sample_size": args.sample_size,
              "knowledge_fraction": .25, "supplied_pairs": len(pairs)}
    rows = []
    vanilla_graph = vanilla_coefficients = None
    for method, version, supplied in (
            ("vanilla_flop", "fixed_signature_a", []),
            ("global_greedy_parallel", "global_greedy_parallel", pairs),
            ("global_greedy_hybrid", "global_greedy_hybrid", pairs)):
        started = perf_counter()
        _, diagnostics = flopsearch.flop_notreks(
            X, 2., supplied, restarts=1, seed=args.algorithm_seed,
            max_signature_rounds=2, search_version=version,
            return_diagnostics=True)
        graph = adjacency(diagnostics, X.shape[1])
        bic, coefficients = gaussian_bic(X, graph, lambda_bic=2.)
        if method == "vanilla_flop":
            vanilla_graph, vanilla_coefficients = graph, coefficients
        rows.append({**common, "method": method, "runtime": perf_counter() - started,
                     "base_score": float(bic), "regularizer": np.nan,
                     "notreks_penalty": np.nan, "accepted_order_moves": np.nan,
                     "continuation_trace": [], "screening_seconds": np.nan,
                     "refinement_seconds": np.nan,
                     "postselection_seconds": np.nan,
                     **metrics(graph, truth, pairs)})

    started = perf_counter()
    window = fit_local_window_ablation(
        X, pairs, LocalWindowConfig(
            block_size=4, sweeps=1, initial_flop_runs=2,
            seed=args.algorithm_seed,
            max_block_calls=4, overall_time_limit_seconds=10.,
            block_time_limit_seconds=.5, max_families_per_node=12))
    rows.append({**common, "method": "local_window_ablation", "runtime": perf_counter() - started,
                 "base_score": window.score, "regularizer": np.nan,
                 "notreks_penalty": np.nan,
                 "accepted_order_moves": window.accepted_block_improvements,
                 "continuation_trace": [],
                 "screening_seconds": window.outer_search_seconds,
                 "refinement_seconds": window.block_solving_seconds,
                 "postselection_seconds": 0.,
                 **metrics(window.adjacency, truth, pairs)})

    for use_notreks in (False, True):
        backend = DagmaOrderBackend(
            X, pairs if use_notreks else (), DagmaBackendConfig(
                seed=args.algorithm_seed, learning_rate=.001, checkpoint=5))
        initial_score = backend.scorer.score(vanilla_graph)
        feasible = ([HardCandidate(vanilla_graph, initial_score,
                                   {"source": "vanilla_flop"})]
                    if not use_notreks or not no_trek_violations(
                        vanilla_graph, pairs) else [])
        result = fit_gflop(
            backend, pairs, GFlopConfig(
                continuation_weights=((0., .5, 5., 25.) if use_notreks else (0.,)),
                sweeps_per_stage=1, screening_budget=40,
                refinement_budget=300, refinement_top_k=2,
                finalist_postselection_seconds=.01,
                postselection_seconds=.5, archive_size=6,
                seed=args.algorithm_seed, use_notreks=use_notreks),
            initial_order=topological_order(vanilla_graph),
            initial_state=vanilla_coefficients,
            feasible_initializations=feasible)
        final_stage = result.continuation_trace[-1]
        rows.append({**common,
            "method": ("generic_gflop_notreks" if use_notreks
                       else "generic_gflop_unconstrained"),
            "runtime": result.runtime, "base_score": final_stage["base_score"],
            "hard_target_score": result.target_score,
            "regularizer": final_stage["regularizer"],
            "notreks_penalty": final_stage["notreks_penalty"],
            "accepted_order_moves": result.accepted_order_moves,
            "continuation_trace": result.continuation_trace,
            "screening_seconds": result.screening_seconds,
            "refinement_seconds": result.refinement_seconds,
            "postselection_seconds": result.postselection_seconds,
            **metrics(result.adjacency, truth, pairs)})

    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = OUT / "results.csv"
    if args.append and csv_path.exists():
        frame = pd.concat([pd.read_csv(csv_path), frame], ignore_index=True)
        frame = frame.drop_duplicates(
            ["data_seed", "algorithm_seed", "knowledge_seed", "method"],
            keep="last")
    frame.to_csv(csv_path, index=False)
    (OUT / "results.json").write_text(
        frame.to_json(orient="records", indent=2) + "\n")
    print(frame.drop(columns="continuation_trace").to_string(index=False))


if __name__ == "__main__":
    main()
