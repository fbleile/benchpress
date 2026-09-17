"""Controlled fixed-order comparison of local toggle and grow-shrink fitting."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    gaussian_bic,
    is_dag,
    topological_order,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import generate
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import (
    cpdag_shd,
    select_knowledge,
    vanilla_flop_candidate,
)


def score(X, graph):
    return float(gaussian_bic(X, graph, lambda_bic=2.0)[0])


def _feasible_addition(graph, parent, child, pairs):
    proposal = graph.copy()
    proposal[parent, child] = 1
    return common_ancestor_violations(proposal, pairs) == 0


def local_toggle(X, order, pairs, passes):
    d = X.shape[1]
    position = {node: i for i, node in enumerate(order)}
    graph = np.zeros((d, d), dtype=np.uint8)
    current = score(X, graph)
    accepted = 0
    for _ in range(max(1, passes)):
        changed = False
        for child in order:
            best = None
            for parent in order[:position[child]]:
                adding = not graph[parent, child]
                if adding and not _feasible_addition(graph, parent, child, pairs):
                    continue
                proposal = graph.copy()
                proposal[parent, child] = int(adding)
                value = score(X, proposal)
                if best is None or value < best[0]:
                    best = (value, parent, adding)
            if best is not None and best[0] < current - 1e-10:
                _, parent, adding = best
                graph[parent, child] = int(adding)
                current = best[0]
                accepted += 1
                changed = True
        if not changed:
            break
    return graph, current, accepted


def grow_shrink(X, order, pairs, passes):
    d = X.shape[1]
    position = {node: i for i, node in enumerate(order)}
    graph = np.zeros((d, d), dtype=np.uint8)
    current = score(X, graph)
    accepted = 0
    for _ in range(max(1, passes)):
        changed = False
        for child in order:
            while True:
                best = None
                for parent in order[:position[child]]:
                    if graph[parent, child]:
                        continue
                    if not _feasible_addition(graph, parent, child, pairs):
                        continue
                    proposal = graph.copy()
                    proposal[parent, child] = 1
                    value = score(X, proposal)
                    if best is None or value < best[0]:
                        best = (value, parent)
                if best is None or best[0] >= current - 1e-10:
                    break
                graph[best[1], child] = 1
                current = best[0]
                accepted += 1
                changed = True
            while True:
                best = None
                for parent in np.flatnonzero(graph[:, child]):
                    proposal = graph.copy()
                    proposal[parent, child] = 0
                    value = score(X, proposal)
                    if best is None or value < best[0]:
                        best = (value, int(parent))
                if best is None or best[0] >= current - 1e-10:
                    break
                graph[best[1], child] = 0
                current = best[0]
                accepted += 1
                changed = True
        if not changed:
            break
    return graph, current, accepted


def skeleton_shd(truth, estimate):
    truth = truth.astype(bool) | truth.astype(bool).T
    estimate = estimate.astype(bool) | estimate.astype(bool).T
    upper = np.triu(np.ones_like(truth, dtype=bool), 1)
    return int(np.sum(truth[upper] != estimate[upper]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[4001, 4002, 4003, 4004, 4005])
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--graph-type", default="er2")
    parser.add_argument("--knowledge-fraction", type=float, default=0.25)
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed in enumerate(args.seeds, 1):
        X, truth, all_pairs = generate(
            seed, d=args.d, n=args.n, graph_type=args.graph_type,
            scm="linear", noise="gaussian")
        pairs = select_knowledge(all_pairs, args.knowledge_fraction, seed)
        vanilla, _ = vanilla_flop_candidate(X, seed, 1)
        order = topological_order(vanilla)
        for name, fit in (("fixed_order_toggle", local_toggle),
                          ("fixed_order_grow_shrink", grow_shrink)):
            started = time.perf_counter()
            graph, bic, accepted = fit(X, order, pairs, args.passes)
            runtime = time.perf_counter() - started
            rows.append({
                "seed": seed, "method": name, "passes": args.passes,
                "runtime": runtime, "bic": bic,
                "edges": int(graph.sum()), "accepted_toggles": accepted,
                "cpdag_shd": cpdag_shd(truth, graph),
                "skeleton_shd": skeleton_shd(truth, graph),
                "violations": common_ancestor_violations(graph, pairs),
                "dag": is_dag(graph),
            })
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output_dir / "per_seed.csv", index=False)
        print(f"[{index}/{len(args.seeds)}] seed={seed}", flush=True)
        print(frame[frame.seed == seed].to_string(index=False), flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("method").agg(
        cpdag_shd_mean=("cpdag_shd", "mean"),
        cpdag_shd_std=("cpdag_shd", "std"),
        skeleton_shd_mean=("skeleton_shd", "mean"),
        runtime_mean=("runtime", "mean"),
        bic_mean=("bic", "mean"),
        violations_max=("violations", "max"),
        edges_mean=("edges", "mean"),
    ).reset_index()
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
