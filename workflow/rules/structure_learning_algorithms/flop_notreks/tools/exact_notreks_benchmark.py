#!/usr/bin/env python3
"""Resumable exact-NOTREKS ablation on repository instances."""
import argparse
import json
import time
from pathlib import Path

import flopsearch
import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag
from workflow.rules.structure_learning_algorithms.flop_notreks.exact_solver import (
    ExactConfig, fit_exact_notreks, no_trek_violations, result_dict,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.tools.global_greedy_strategy_benchmark import (
    metrics, no_trek_pairs,
)

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results/dagma_notreks_oracle/exact_notreks_d50"


def selected_dag(diagnostics, p):
    adjacency = np.zeros((p, p), dtype=np.uint8)
    for parent, child in diagnostics["selected_dag_edges"]:
        adjacency[parent, child] = 1
    return adjacency


def save_row(name, row, adjacency):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(adjacency).to_csv(OUT / f"{name}_adjacency.csv", index=False)
    (OUT / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, default=5001)
    parser.add_argument("--algorithm-seed", type=int, default=7001)
    parser.add_argument("--max-indegree", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=60.)
    parser.add_argument("--flop-restarts", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tag = "flop_notreks_dense_d50_budget"
    data = pd.read_csv(ROOT / f"resources/data/mydatasets/{tag}/s{args.data_seed}.csv").to_numpy(float)
    truth = pd.read_csv(ROOT / f"resources/adjmat/myadjmats/{tag}/g{args.data_seed}.csv").to_numpy(np.uint8)
    data = (data - data.mean(0)) / data.std(0)
    pairs = no_trek_pairs(truth)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # Official vanilla FLOP is included for structural-quality/runtime
    # comparison.  Its Python API returns a CPDAG, not the selected DAG/BIC.
    name = "flop_vanilla"
    cache = OUT / f"{name}.json"
    if cache.exists() and not args.force:
        rows.append(json.loads(cache.read_text()))
    else:
        started = time.perf_counter()
        cpdag = convert_flop_cpdag(
            flopsearch.flop(data, 2., restarts=args.flop_restarts), len(truth))
        row = {"method": name, "runtime_seconds": time.perf_counter() - started,
               "data_seed": args.data_seed, "algorithm_seed": None,
               "objective": None, "dual_bound": None, "mip_gap": None,
               "status": "heuristic", "zero_gap_certificate": False,
               "notreks_verified": False, **metrics(cpdag, truth)}
        save_row(name, row, cpdag)
        rows.append(row)

    # Seeded empty-constraint FLOP exposes its selected DAG and decomposable
    # BIC, allowing the same independent NOTREKS verification as exact runs.
    name = "flop_vanilla_seeded_dag"
    cache = OUT / f"{name}.json"
    if cache.exists() and not args.force:
        rows.append(json.loads(cache.read_text()))
    else:
        started = time.perf_counter()
        _, diagnostics = flopsearch.flop_notreks(
            data, 2., [], restarts=args.flop_restarts - 1,
            seed=args.algorithm_seed, max_signature_rounds=0,
            search_version="fixed_signature_a", return_diagnostics=True)
        vanilla_dag = selected_dag(diagnostics, len(truth))
        violations = no_trek_violations(vanilla_dag, pairs)
        row = {"method": name,
               "runtime_seconds": time.perf_counter() - started,
               "data_seed": args.data_seed,
               "algorithm_seed": args.algorithm_seed,
               "objective": diagnostics["final_bic"],
               "dual_bound": None, "mip_gap": None,
               "status": "heuristic", "zero_gap_certificate": False,
               "notreks_verified": not violations,
               "independent_no_trek_violations": len(violations),
               **metrics(vanilla_dag, truth)}
        save_row(name, row, vanilla_dag)
        rows.append(row)

    # A strong feasible incumbent is reused both as comparator and MIP start.
    name = "global_greedy_hybrid"
    cache = OUT / f"{name}.json"
    if cache.exists() and not args.force:
        row = json.loads(cache.read_text())
        warm = pd.read_csv(OUT / f"{name}_adjacency.csv").to_numpy(np.uint8)
    else:
        started = time.perf_counter()
        _, diagnostics = flopsearch.flop_notreks(
            data, 2., pairs, restarts=1, seed=args.algorithm_seed,
            max_signature_rounds=2, search_version="global_greedy_hybrid",
            return_diagnostics=True)
        warm = selected_dag(diagnostics, len(truth))
        row = {"method": name, "runtime_seconds": time.perf_counter() - started,
               "data_seed": args.data_seed, "algorithm_seed": args.algorithm_seed,
               "objective": diagnostics["final_bic"], "dual_bound": None,
               "mip_gap": None, "status": "heuristic",
               "zero_gap_certificate": False,
               "notreks_verified": not no_trek_violations(warm, pairs),
               **metrics(warm, truth)}
        save_row(name, row, warm)
    rows.append(row)

    for variant in ("ordinary", "forbidden_arcs", "source_cut", "lazy_trek"):
        name = f"branch_cut_{variant}_k{args.max_indegree}"
        cache = OUT / f"{name}.json"
        legacy_cache = OUT / f"exact_{variant}_k{args.max_indegree}.json"
        if not cache.exists() and legacy_cache.exists() and not args.force:
            row = json.loads(legacy_cache.read_text())
            row["method"] = name
            cache.write_text(json.dumps(row, indent=2) + "\n")
        if cache.exists() and not args.force:
            rows.append(json.loads(cache.read_text()))
            continue
        result = fit_exact_notreks(
            data, pairs,
            ExactConfig(max_indegree=args.max_indegree,
                        time_limit_seconds=args.time_limit,
                        variant=variant, random_seed=args.algorithm_seed),
            warm_start=warm)
        # Do not trust the callback diagnostics alone: verify from the saved
        # adjacency with an independent graph traversal implementation.
        violations = no_trek_violations(result.adjacency, pairs)
        if result.dag_verified is False:
            raise RuntimeError(f"{variant} returned a cyclic incumbent")
        if variant == "lazy_trek" and violations:
            raise RuntimeError("lazy-trek incumbent failed independent verification")
        row = {"method": name, "data_seed": args.data_seed,
               "algorithm_seed": args.algorithm_seed,
               **result_dict(result), **metrics(result.adjacency, truth),
               "independent_no_trek_violations": len(violations)}
        save_row(name, row, result.adjacency)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "summary.csv", index=False)
    (OUT / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
