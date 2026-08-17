#!/usr/bin/env python3
"""Measure multi-seed FLOP support unions as DAGMA superstructures."""
import argparse
from pathlib import Path

import flopsearch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results/dagma_notreks_oracle/flop_support_union"
TAG = "flop_notreks_dense_d50_budget"


def selected_dag(diagnostics, p):
    adjacency = np.zeros((p, p), dtype=np.uint8)
    for parent, child in diagnostics["selected_dag_edges"]:
        adjacency[parent, child] = 1
    return adjacency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seeds", nargs="*", type=int,
                        default=list(range(5001, 5011)))
    parser.add_argument("--algorithm-seeds", type=int, default=32)
    args = parser.parse_args()
    rows = []
    for data_seed in args.data_seeds:
        data = pd.read_csv(ROOT / f"resources/data/mydatasets/{TAG}/s{data_seed}.csv").to_numpy(float)
        truth = pd.read_csv(ROOT / f"resources/adjmat/myadjmats/{TAG}/g{data_seed}.csv").to_numpy(np.uint8)
        data = (data - data.mean(0)) / data.std(0)
        true_skeleton = (truth | truth.T).astype(bool)
        upper = np.triu(np.ones_like(truth, dtype=bool), 1)
        union = np.zeros_like(truth)
        unique_dags = set()
        unique_skeletons = set()
        for index in range(1, args.algorithm_seeds + 1):
            seed = 10007 + 7919 * index
            _, diagnostics = flopsearch.flop_notreks(
                data, 2., [], restarts=0, seed=seed,
                max_signature_rounds=0, search_version="fixed_signature_a",
                return_diagnostics=True)
            dag = selected_dag(diagnostics, len(truth))
            union |= dag
            unique_dags.add(dag.tobytes())
            unique_skeletons.add((dag | dag.T).tobytes())
            if index not in {1, 2, 4, 8, 16, 32, args.algorithm_seeds}:
                continue
            skeleton = (union | union.T).astype(bool)
            tp = int(np.sum(skeleton & true_skeleton & upper))
            fp = int(np.sum(skeleton & ~true_skeleton & upper))
            fn = int(np.sum(~skeleton & true_skeleton & upper))
            directed_tp = int(np.sum(union & truth))
            directed_fp = int(np.sum(union & ~truth.astype(bool)))
            rows.append({
                "data_seed": data_seed, "flop_runs": index,
                "true_edges": int(truth.sum()),
                "unique_dags_seen": len(unique_dags),
                "unique_skeletons_seen": len(unique_skeletons),
                "union_directed_arcs": int(union.sum()),
                "dagma_allowed_directed_arcs": int(2 * np.sum(skeleton & upper)),
                "dagma_mask_density": float(np.sum(skeleton & ~np.eye(len(truth), dtype=bool))
                                            / (len(truth) * (len(truth) - 1))),
                "skeleton_recall": tp / max(1, tp + fn),
                "skeleton_precision": tp / max(1, tp + fp),
                "missing_true_skeleton_edges": fn,
                "extra_skeleton_edges": fp,
                "directed_recall": directed_tp / max(1, int(truth.sum())),
                "directed_false_arcs": directed_fp,
            })
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "per_dataset.csv", index=False)
    summary = raw.groupby("flop_runs", as_index=False).agg(
        datasets=("data_seed", "size"),
        mean_skeleton_recall=("skeleton_recall", "mean"),
        minimum_skeleton_recall=("skeleton_recall", "min"),
        mean_skeleton_precision=("skeleton_precision", "mean"),
        mean_missing_true_edges=("missing_true_skeleton_edges", "mean"),
        maximum_missing_true_edges=("missing_true_skeleton_edges", "max"),
        mean_extra_edges=("extra_skeleton_edges", "mean"),
        mean_allowed_arcs=("dagma_allowed_directed_arcs", "mean"),
        mean_mask_density=("dagma_mask_density", "mean"),
        mean_directed_recall=("directed_recall", "mean"),
        mean_unique_dags_seen=("unique_dags_seen", "mean"),
        mean_unique_skeletons_seen=("unique_skeletons_seen", "mean"))
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
