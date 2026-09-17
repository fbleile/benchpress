"""Compare hard global greedy with and without graph-only NOTREKS evaluation."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import flopsearch

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    gaussian_bic,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dimension in args.dimensions:
      for seed in args.seeds:
        X, truth, pairs = generate(seed, d=dimension)
        configs = ["hard_global", "hard_global_binary_penalty"]
        outputs = {}
        for method in configs:
            def run_once():
                if method == "hard_global":
                    return flopsearch.flop_notreks(
                        X, 2., pairs, restarts=args.restarts, seed=seed,
                        max_signature_rounds=args.sweeps, return_dag=True,
                        return_diagnostics=True)
                return flopsearch.global_greedy_binary_penalty(
                    X, 2., pairs, restarts=args.restarts,
                    max_sweeps=args.sweeps, seed=seed,
                    return_diagnostics=True)

            timings = []
            A, diagnostics = run_once()
            for _ in range(args.timing_repeats):
                started = time.perf_counter()
                repeated_A, _ = run_once()
                timings.append(time.perf_counter() - started)
                if not np.array_equal(np.asarray(A), np.asarray(repeated_A)):
                    raise RuntimeError(f"{method} is nondeterministic for seed {seed}")
            runtime = float(np.median(timings))
            A = np.asarray(A, dtype=np.uint8)
            outputs[method] = A.copy()
            bic, _ = gaussian_bic(X, A, lambda_bic=2.)
            truth_bool = truth != 0
            estimate_bool = A != 0
            violations = int(common_ancestor_violations(A, pairs))
            rows.append({
                "seed": seed,
                "dimension": dimension,
                "method": method,
                "bic": float(bic),
                "graph_penalty": violations,
                "objective_reported": float(diagnostics["final_bic"]),
                "edges": int(A.sum()),
                "true_edges": int(truth_bool.sum()),
                "true_positive": int(np.sum(estimate_bool & truth_bool)),
                "false_positive": int(np.sum(estimate_bool & ~truth_bool)),
                "false_negative": int(np.sum(~estimate_bool & truth_bool)),
                "directed_shd": int(np.sum(estimate_bool != truth_bool)),
                "notreks_violations": violations,
                "runtime": runtime,
                "runtime_min": float(np.min(timings)),
                "runtime_max": float(np.max(timings)),
                "search_restarts": args.restarts,
                "search_sweeps": args.sweeps,
            })
        if not np.array_equal(outputs[configs[0]], outputs[configs[1]]):
            raise RuntimeError(f"graph mismatch for seed {seed}")
        print(pd.DataFrame(rows[-len(configs):]).to_string(index=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_seed.csv", index=False)
    aggregate = frame.groupby(["dimension", "method"]).agg({
        "bic": ["mean", "std"],
        "graph_penalty": ["mean", "std"],
        "edges": ["mean", "std"],
        "true_edges": ["mean", "std"],
        "true_positive": ["mean", "std"],
        "false_positive": ["mean", "std"],
        "false_negative": ["mean", "std"],
        "directed_shd": ["mean", "std"],
        "notreks_violations": ["max", "mean"],
        "runtime": ["mean", "std"],
    })
    aggregate.to_csv(args.output_dir / "aggregate.csv")
    print(aggregate.to_string())


if __name__ == "__main__":
    main()
