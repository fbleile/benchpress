#!/usr/bin/env python3
"""Small paired FLOP/DAGMA/NOTREKS benchmark with honest wall times."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import flopsearch
import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations, gaussian_bic, is_dag,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    ProductionConfig, run_production_pipeline,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.local_window_ablation import (
    LocalWindowConfig, fit_local_window_ablation,
)

ROOT = Path(__file__).resolve().parents[5]
TAG = "flop_notreks_dense_d50_budget"


def disk_case(seed: int):
    data = pd.read_csv(
        ROOT / f"resources/data/mydatasets/{TAG}/s{seed}.csv").to_numpy(float)
    truth = pd.read_csv(
        ROOT / f"resources/adjmat/myadjmats/{TAG}/g{seed}.csv").to_numpy(np.uint8)
    data = (data - data.mean(0)) / data.std(0)
    return data, truth


def synthetic_case(dimension: int, seed: int, sample_size: int = 1000):
    """The established higher-dimensional generator used by prior results."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(dimension)
    truth = np.zeros((dimension, dimension), dtype=np.uint8)
    probability = min(1., 3. / max(1, dimension - 1))
    for left in range(dimension):
        for right in range(left + 1, dimension):
            if rng.random() < probability:
                truth[order[left], order[right]] = 1
    weights = (truth * rng.uniform(.45, .9, size=truth.shape)
               * rng.choice([-1, 1], truth.shape))
    data = np.zeros((sample_size, dimension))
    noise = rng.normal(size=data.shape)
    for node in order:
        data[:, node] = noise[:, node] + data @ weights[:, node]
    data = (data - data.mean(0)) / data.std(0)
    return data, truth


def benchmark_case(dimension: int, seed: int):
    if dimension == 50 and 5001 <= seed <= 5010:
        return disk_case(seed)
    return synthetic_case(dimension, seed)


def oracle_pairs(adjacency: np.ndarray):
    reach = adjacency.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for node in range(len(reach)):
        reach |= reach[:, [node]] & reach[[node], :]
    return [(left, right) for left in range(len(reach))
            for right in range(left + 1, len(reach))
            if not np.any(reach[:, left] & reach[:, right])]


def selected_dag(diagnostics: dict, dimension: int):
    adjacency = np.zeros((dimension, dimension), dtype=np.uint8)
    for parent, child in diagnostics["selected_dag_edges"]:
        adjacency[int(parent), int(child)] = 1
    return adjacency


def graph_metrics(adjacency, truth, pairs, data):
    skeleton = (adjacency | adjacency.T).astype(bool)
    true_skeleton = (truth | truth.T).astype(bool)
    upper = np.triu(np.ones_like(truth, dtype=bool), 1)
    tp = int(np.sum(skeleton & true_skeleton & upper))
    fp = int(np.sum(skeleton & ~true_skeleton & upper))
    fn = int(np.sum(~skeleton & true_skeleton & upper))
    bic, _ = gaussian_bic(data, adjacency, lambda_bic=2.)
    return {
        "gaussian_bic": float(bic), "edge_count": int(adjacency.sum()),
        "skeleton_shd": fp + fn,
        "skeleton_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "directed_shd": int(np.sum(adjacency != truth)),
        "is_dag": bool(is_dag(adjacency)),
        "no_trek_violations": int(common_ancestor_violations(adjacency, pairs)),
        "representation": "DAG",
    }


def support_missing_truth(support: dict, truth: np.ndarray):
    allowed = np.zeros_like(truth, dtype=bool)
    for run in support.get("runs", []):
        for parent, child in run.get("selected_dag_edges", []):
            allowed[parent, child] = allowed[child, parent] = True
    true_skeleton = (truth | truth.T).astype(bool)
    upper = np.triu(np.ones_like(truth, dtype=bool), 1)
    return int(np.sum(true_skeleton & ~allowed & upper))


def run_flop(data, truth, pairs, seed, restarts, version, method,
             time_budget=None):
    started = perf_counter()
    search_budget = ({"timeout": float(time_budget)} if time_budget is not None
                     else {"restarts": restarts - 1})
    _, diagnostics = flopsearch.flop_notreks(
        data, 2., [] if method == "vanilla_flop" else pairs,
        seed=seed, max_signature_rounds=4, **search_budget,
        search_version=version, return_diagnostics=True)
    runtime = perf_counter() - started
    adjacency = selected_dag(diagnostics, data.shape[1])
    return {"method": method, "runtime": runtime,
            "algorithm_work": (f"{time_budget:g}-second FLOP budget"
                               if time_budget is not None else
                               f"{restarts} seeded FLOP starts"),
            **graph_metrics(adjacency, truth, pairs, data)}


def run_dagma(data, truth, pairs, config, method):
    fit_pairs = [] if method == "vanilla_dagma" else pairs
    started = perf_counter()
    selected, restart_rows = run_production_pipeline(data, fit_pairs, config)
    runtime = perf_counter() - started
    row = {"method": method, "runtime": runtime,
           "optimizer_runtime": float(sum(item.runtime for item in restart_rows)),
           "support_build_seconds": float(
               selected.support.get("support_build_seconds", 0.)),
           "algorithm_work": (
               f"{config.restarts} DAGMA restarts; T={config.T}; "
               f"warm/max={config.warm_iter}/{config.max_iter}"),
           **graph_metrics(selected.adjacency, truth, pairs, data)}
    if config.support_mode == "flop_union_support":
        row.update({
            "support_runs": config.flop_support_runs,
            "support_size": selected.support["allowed_directed_arcs"],
            "support_density": selected.support["support_density"],
            "missing_true_support_edges": support_missing_truth(
                selected.support, truth),
        })
    return row


def run_case(dimension, data_seed, algorithm_seed, knowledge_seed, fraction,
             flop_restarts, skip_unrestricted=False, flop_time_budget=None):
    data, truth = benchmark_case(dimension, data_seed)
    all_pairs = oracle_pairs(truth)
    rng = np.random.default_rng(knowledge_seed)
    chosen = rng.permutation(len(all_pairs))[:round(fraction * len(all_pairs))]
    pairs = sorted(all_pairs[index] for index in chosen)
    common = {"dimension": dimension, "data_seed": data_seed,
              "graph_seed": data_seed, "algorithm_seed": algorithm_seed,
              "knowledge_seed": knowledge_seed, "knowledge_fraction": fraction,
              "supplied_pair_count": len(pairs)}
    rows = []
    for method, version in (("vanilla_flop", "fixed_signature_a"),
                            ("global_greedy_hybrid", "global_greedy_hybrid")):
        rows.append({**common, **run_flop(
            data, truth, pairs, algorithm_seed, flop_restarts, version, method,
            flop_time_budget)})

    base = ProductionConfig(
        restarts=2, seed=algorithm_seed, T=2, s=(1., .9),
        warm_iter=300, max_iter=500, checkpoint=100,
        postselection_policy="PS1_joint_feasible_greedy_score",
        max_search_seconds=1., max_expanded_nodes=500, max_queue_size=500,
        support_initialization="zero_then_best_feasible_flop")
    dagma_specs = [] if skip_unrestricted else [
        ("vanilla_dagma", replace(
            base, notreks_constraint_active=False, support_mode="unrestricted")),
        ("unrestricted_dagma_notreks", base),
    ]
    dagma_specs.extend((f"flop_union_{runs}_dagma_notreks", replace(
        base, support_mode="flop_union_support", flop_support_runs=runs))
                       for runs in (1, 2, 4))
    for method, config in dagma_specs:
        rows.append({**common, **run_dagma(data, truth, pairs, config, method)})

    started = perf_counter()
    gflop = fit_local_window_ablation(data, pairs, LocalWindowConfig(
        block_size=4, sweeps=1, initial_flop_runs=flop_restarts,
        seed=algorithm_seed, block_time_limit_seconds=1.,
        overall_time_limit_seconds=20., max_block_calls=12,
        max_families_per_node=16))
    runtime = perf_counter() - started
    rows.append({**common, "method": "local_window_ablation_k4",
                 "runtime": runtime,
                 "algorithm_work": (
                     f"{flop_restarts} FLOP starts; <=12 blocks; <=20 seconds"),
                 **graph_metrics(gflop.adjacency, truth, pairs, data),
                 **{key: value for key, value in asdict(gflop).items()
                    if key not in {"adjacency", "order", "block_diagnostics",
                                   "runtime", "score", "no_trek_violations"}}})
    return rows


def run_flop_only_case(dimension, data_seed, algorithm_seed, knowledge_seed,
                       fraction, flop_restarts, flop_time_budget=None):
    data, truth = benchmark_case(dimension, data_seed)
    all_pairs = oracle_pairs(truth)
    rng = np.random.default_rng(knowledge_seed)
    chosen = rng.permutation(len(all_pairs))[:round(fraction * len(all_pairs))]
    pairs = sorted(all_pairs[index] for index in chosen)
    common = {"dimension": dimension, "data_seed": data_seed,
              "graph_seed": data_seed, "algorithm_seed": algorithm_seed,
              "knowledge_seed": knowledge_seed, "knowledge_fraction": fraction,
              "supplied_pair_count": len(pairs)}
    rows = []
    for method, version in (("vanilla_flop", "fixed_signature_a"),
                            ("global_greedy_hybrid", "global_greedy_hybrid")):
        row = {**common, **run_flop(
            data, truth, pairs, algorithm_seed, flop_restarts, version, method,
            flop_time_budget)}
        rows.append(row)
        print(pd.DataFrame([row])[[
            "data_seed", "method", "runtime", "gaussian_bic", "edge_count",
            "skeleton_shd", "skeleton_f1", "directed_shd",
            "no_trek_violations"]].to_string(index=False), flush=True)
    return rows


def run_additional_flop_family_case(
        dimension, data_seed, algorithm_seed, knowledge_seed, fraction,
        flop_restarts):
    """Missing non-DAGMA arms for the best-quality FLOP-family benchmark."""
    data, truth = benchmark_case(dimension, data_seed)
    all_pairs = oracle_pairs(truth)
    rng = np.random.default_rng(knowledge_seed)
    chosen = rng.permutation(len(all_pairs))[:round(fraction * len(all_pairs))]
    pairs = sorted(all_pairs[index] for index in chosen)
    common = {"dimension": dimension, "data_seed": data_seed,
              "graph_seed": data_seed, "algorithm_seed": algorithm_seed,
              "knowledge_seed": knowledge_seed, "knowledge_fraction": fraction,
              "supplied_pair_count": len(pairs)}
    rows = [{**common, **run_flop(
        data, truth, pairs, algorithm_seed, flop_restarts,
        "global_greedy_parallel", "global_greedy_parallel")}]
    print(pd.DataFrame(rows)[[
        "data_seed", "method", "runtime", "gaussian_bic", "edge_count",
        "skeleton_shd", "skeleton_f1", "directed_shd",
        "no_trek_violations"]].to_string(index=False), flush=True)
    started = perf_counter()
    gflop = fit_local_window_ablation(data, pairs, LocalWindowConfig(
        block_size=4, sweeps=1, initial_flop_runs=flop_restarts,
        seed=algorithm_seed, block_time_limit_seconds=2.,
        overall_time_limit_seconds=60., max_block_calls=12,
        max_families_per_node=24))
    row = {**common, "method": "local_window_ablation_k4",
           "runtime": perf_counter() - started,
           "algorithm_work": (
               f"{flop_restarts} FLOP starts; <=12 exact-family blocks"),
           **graph_metrics(gflop.adjacency, truth, pairs, data),
           **{key: value for key, value in asdict(gflop).items()
              if key not in {"adjacency", "order", "block_diagnostics",
                             "runtime", "score", "no_trek_violations"}}}
    rows.append(row)
    print(pd.DataFrame([row])[[
        "data_seed", "method", "runtime", "gaussian_bic", "edge_count",
        "skeleton_shd", "skeleton_f1", "directed_shd",
        "no_trek_violations"]].to_string(index=False), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=50)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[5001])
    parser.add_argument("--algorithm-seed", type=int, default=7001)
    parser.add_argument("--knowledge-seed", type=int, default=9101)
    parser.add_argument("--knowledge-fraction", type=float, default=.25)
    parser.add_argument("--flop-restarts", type=int, default=4)
    parser.add_argument("--flop-time-budget", type=float)
    parser.add_argument("--skip-unrestricted", action="store_true")
    parser.add_argument("--flop-only", action="store_true")
    parser.add_argument("--additional-flop-family", action="store_true")
    parser.add_argument("--benchmark-label", default="")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    rows = []
    for seed in args.data_seeds:
        if args.additional_flop_family:
            rows.extend(run_additional_flop_family_case(
                args.dimension, seed, args.algorithm_seed, args.knowledge_seed,
                args.knowledge_fraction, args.flop_restarts))
        elif args.flop_only:
            runner = run_flop_only_case
            rows.extend(runner(
                args.dimension, seed, args.algorithm_seed, args.knowledge_seed,
                args.knowledge_fraction, args.flop_restarts,
                args.flop_time_budget))
        else:
            runner = run_case
            rows.extend(runner(
                args.dimension, seed, args.algorithm_seed, args.knowledge_seed,
                args.knowledge_fraction, args.flop_restarts,
                args.skip_unrestricted, args.flop_time_budget))
    suffix = f"_{args.benchmark_label}" if args.benchmark_label else ""
    OUT = ROOT / (f"results/dagma_notreks_oracle/"
                  f"fair_d{args.dimension}_flop_dagma{suffix}")
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    output = OUT / "per_run.csv"
    if args.append and output.exists():
        frame = pd.concat([pd.read_csv(output), frame], ignore_index=True)
        frame = frame.drop_duplicates(
            ["data_seed", "algorithm_seed", "knowledge_seed",
             "knowledge_fraction", "method"], keep="last")
    frame.to_csv(output, index=False)
    settings = vars(args).copy()
    settings["evaluated_data_seeds"] = sorted(
        map(int, frame["data_seed"].unique()))
    (OUT / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(frame[["data_seed", "method", "runtime", "gaussian_bic",
                 "edge_count", "skeleton_shd", "skeleton_f1", "directed_shd",
                 "no_trek_violations"]].to_string(index=False))


if __name__ == "__main__":
    main()
