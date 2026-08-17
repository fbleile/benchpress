#!/usr/bin/env python3
"""Multi-seed benchmark for optimized and strategy-enhanced global greedy."""
import argparse
import json
import time
from pathlib import Path

import flopsearch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results/dagma_notreks_oracle/global_greedy_strategy"


def no_trek_pairs(adjacency):
    p = adjacency.shape[0]
    reach = adjacency.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(p):
        reach |= reach[:, [k]] & reach[[k], :]
    return [(i, j) for i in range(p) for j in range(i + 1, p)
            if not np.any(reach[:, i] & reach[:, j])]


def synthetic_case(p, seed, n=1000):
    rng = np.random.default_rng(seed)
    order = rng.permutation(p)
    adjacency = np.zeros((p, p), dtype=np.uint8)
    probability = min(1., 3. / max(1, p - 1))
    for left in range(p):
        for right in range(left + 1, p):
            if rng.random() < probability:
                adjacency[order[left], order[right]] = 1
    weights = adjacency * rng.uniform(.45, .9, size=(p, p)) * rng.choice([-1, 1], (p, p))
    data = np.zeros((n, p))
    noise = rng.normal(size=(n, p))
    for node in order:
        data[:, node] = noise[:, node] + data @ weights[:, node]
    return (data - data.mean(0)) / data.std(0), adjacency


def disk_case(seed):
    tag = "flop_notreks_dense_d50_budget"
    data = pd.read_csv(ROOT / f"resources/data/mydatasets/{tag}/s{seed}.csv").to_numpy(float)
    truth = pd.read_csv(ROOT / f"resources/adjmat/myadjmats/{tag}/g{seed}.csv").to_numpy(np.uint8)
    return (data - data.mean(0)) / data.std(0), truth


def metrics(estimate, truth):
    est_skel = (estimate | estimate.T).astype(bool)
    true_skel = (truth | truth.T).astype(bool)
    upper = np.triu(np.ones_like(truth, dtype=bool), 1)
    tp = int(np.sum(est_skel & true_skel & upper))
    fp = int(np.sum(est_skel & ~true_skel & upper))
    fn = int(np.sum(~est_skel & true_skel & upper))
    return {"directed_shd": int(np.sum(estimate != truth)),
            "skeleton_shd": fp + fn,
            "skeleton_f1": 2 * tp / max(1, 2 * tp + fp + fn)}


def violation_count(adjacency, pairs):
    reach = adjacency.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for node in range(len(reach)):
        reach |= reach[:, [node]] & reach[[node], :]
    return sum(bool(np.any(reach[:, left] & reach[:, right]))
               for left, right in pairs)


def run_case(dimension, data_seed, algorithm_seed, restarts, knowledge_fraction,
             max_sweeps):
    if dimension == 50 and data_seed in range(5001, 5011):
        data, truth = disk_case(data_seed)
    else:
        data, truth = synthetic_case(dimension, data_seed)
    pairs = no_trek_pairs(truth)
    pair_rng = np.random.default_rng(data_seed * 1009 + 17)
    pair_rng.shuffle(pairs)
    pairs = sorted(pairs[:round(len(pairs) * knowledge_fraction)])
    rows = []
    specifications = (
        ("flop_vanilla_seeded", "fixed_signature_a", []),
        ("global_greedy_parallel", "global_greedy_parallel", pairs),
        ("global_greedy_hybrid", "global_greedy_hybrid", pairs),
    )
    for strategy, search_version, supplied_pairs in specifications:
        started = time.perf_counter()
        _, diagnostics = flopsearch.flop_notreks(
            data, 2., supplied_pairs, restarts=restarts - 1,
            seed=algorithm_seed,
            max_signature_rounds=(0 if not supplied_pairs else max_sweeps),
            search_version=search_version,
            return_diagnostics=True)
        runtime = time.perf_counter() - started
        estimate = np.zeros_like(truth)
        for parent, child in diagnostics["selected_dag_edges"]:
            estimate[parent, child] = 1
        rows.append({"dimension": dimension, "data_seed": data_seed,
                     "algorithm_seed": algorithm_seed, "strategy": strategy,
                     "restarts_requested": restarts,
                     "knowledge_fraction": knowledge_fraction,
                     "pair_count": len(pairs), "runtime": runtime,
                     "independent_no_trek_violations":
                         violation_count(estimate, pairs),
                     **metrics(estimate, truth), **diagnostics})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d50-seeds", nargs="*", type=int,
                        default=[5001, 5002, 5003, 5004])
    parser.add_argument("--algorithm-seeds", nargs="*", type=int,
                        default=[7001, 8017])
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--higher-dimensions", nargs="*", type=int,
                        default=[75, 100])
    parser.add_argument("--higher-seeds", nargs="*", type=int,
                        default=[6101, 6203])
    parser.add_argument("--knowledge-fraction", type=float, default=.25)
    parser.add_argument("--max-sweeps", type=int, default=4)
    parser.add_argument("--append", action="store_true",
                        help="merge with previous per-run rows")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    previous = OUT / "per_run.csv"
    if args.summarize_only:
        if not previous.exists():
            raise FileNotFoundError(previous)
        raw = pd.read_csv(previous)
    else:
        for data_seed in args.d50_seeds:
            for algorithm_seed in args.algorithm_seeds:
                rows += run_case(50, data_seed, algorithm_seed, args.restarts, 1.,
                                 args.max_sweeps)
        for dimension in args.higher_dimensions:
            for data_seed in args.higher_seeds:
                for algorithm_seed in args.algorithm_seeds:
                    rows += run_case(dimension, data_seed, algorithm_seed,
                                     min(2, args.restarts), args.knowledge_fraction,
                                     args.max_sweeps)
        raw = pd.DataFrame(rows)
    if args.append and previous.exists() and not args.summarize_only:
        raw = pd.concat([pd.read_csv(previous), raw], ignore_index=True)
        raw = raw.drop_duplicates(
            ["dimension", "data_seed", "algorithm_seed", "strategy",
             "restarts_requested", "knowledge_fraction"], keep="last")
    raw.to_csv(OUT / "per_run.csv", index=False)
    summary = raw.groupby(["dimension", "strategy"], as_index=False).agg(
        runs=("data_seed", "size"), mean_runtime=("runtime", "mean"),
        mean_directed_shd=("directed_shd", "mean"),
        mean_skeleton_shd=("skeleton_shd", "mean"),
        mean_skeleton_f1=("skeleton_f1", "mean"),
        mean_bic=("final_bic", "mean"),
        mean_restarts_completed=("restarts_completed", "mean"),
        total_violations=("independent_no_trek_violations", "sum"))
    summary.to_csv(OUT / "summary.csv", index=False)
    diversity = []
    for data_seed in sorted(raw.loc[raw.dimension == 50, "data_seed"].unique()):
        _, truth = disk_case(int(data_seed))
        diversity.append({"data_seed": int(data_seed),
                          "true_edges": int(truth.sum()),
                          "maximum_indegree": int(truth.sum(0).max()),
                          "true_sources": int(np.sum(truth.sum(0) == 0)),
                          "no_trek_pairs": len(no_trek_pairs(truth))})
    pd.DataFrame(diversity).to_csv(OUT / "dataset_diversity.csv", index=False)
    paired_rows = []
    wide = raw.pivot(index=["dimension", "data_seed", "algorithm_seed"],
                     columns="strategy",
                     values=["skeleton_shd", "directed_shd", "final_bic"])
    for metric in ("skeleton_shd", "directed_shd", "final_bic"):
        for reference in ("flop_vanilla_seeded", "global_greedy_parallel"):
            delta = (wide[metric]["global_greedy_hybrid"]
                     - wide[metric][reference]).dropna()
            tolerance = 1e-7 if metric == "final_bic" else 0
            paired_rows.append({
                "metric": metric, "hybrid_compared_with": reference,
                "pairs": len(delta), "mean_hybrid_minus_reference": delta.mean(),
                "hybrid_wins": int((delta < -tolerance).sum()),
                "ties": int((abs(delta) <= tolerance).sum()),
                "hybrid_losses": int((delta > tolerance).sum())})
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_comparisons.csv", index=False)
    per_dataset = raw.groupby(["data_seed", "strategy"], as_index=False).agg(
        runs=("algorithm_seed", "size"), runtime=("runtime", "mean"),
        directed_shd=("directed_shd", "mean"),
        skeleton_shd=("skeleton_shd", "mean"),
        skeleton_f1=("skeleton_f1", "mean"), bic=("final_bic", "mean"),
        violations=("independent_no_trek_violations", "mean"))
    per_dataset.to_csv(OUT / "per_dataset.csv", index=False)
    (OUT / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
