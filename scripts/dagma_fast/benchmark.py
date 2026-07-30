"""Small/resumable DAGMA-fast policy comparison.

Defaults are intentionally a smoke workload. Increase iterations explicitly
for scientific benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma_fast import (
    DagmaFastConfig,
    LinearL2Objective,
    fit_weighted_adjacency,
    resolve_lambda1,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing import (
    CompositeFeasibility,
    GaussianBICGraphScore,
    PostprocessingBudget,
    WeightedGraphEstimate,
    postprocess_weighted_graph,
)


def simulate(d, n, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    adjacency = np.zeros((d, d), dtype=int)
    probability = min(1.0, 4.0 / max(d - 1, 1))
    for left in range(d):
        for right in range(left + 1, d):
            if rng.random() < probability:
                adjacency[order[left], order[right]] = 1
    coefficients = adjacency * rng.uniform(.5, 1.5, (d, d))
    X = np.zeros((n, d))
    noise = rng.normal(size=(n, d))
    for node in order:
        X[:, node] = noise[:, node] + X @ coefficients[:, node]
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    return X, adjacency


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    result_path = output / "results.csv"
    if args.resume and result_path.exists():
        rows = pd.read_csv(result_path).to_dict("records")
    done = {
        (int(row["seed"]), str(row["rounding_policy"])) for row in rows}
    hashes = {}
    for seed in args.seeds:
        X, truth = simulate(args.dimension, args.n, seed)
        hashes[str(seed)] = hashlib.sha256(
            np.ascontiguousarray(X).view(np.uint8)).hexdigest()
        resolved = resolve_lambda1(
            args.lambda_policy, n=args.n, d=args.dimension,
            fixed=args.lambda1)
        optimized = fit_weighted_adjacency(
            LinearL2Objective(X),
            DagmaFastConfig(
                lambda1=resolved.value, T=args.T,
                warm_iter=args.warm_iter, max_iter=args.max_iter,
                checkpoint=max(1, min(args.max_iter, 1000)),
                max_runtime_seconds=args.max_runtime_seconds))
        for policy in args.rounding_policies:
            if (seed, policy) in done:
                continue
            rounded = postprocess_weighted_graph(
                WeightedGraphEstimate(optimized.weighted_adjacency),
                X, GaussianBICGraphScore(X), CompositeFeasibility(),
                policy, PostprocessingBudget(max_seconds=args.rounding_seconds))
            rows.append({
                "seed": seed,
                "dimension": args.dimension,
                "n": args.n,
                "lambda_policy": resolved.policy,
                "lambda1_resolved": resolved.value,
                "rounding_policy": rounded.policy,
                "graph_score": rounded.score,
                "edges": int(rounded.graph.sum()),
                "DAG_valid": rounded.diagnostics["DAG_valid"],
                "optimizer_runtime": optimized.runtime_seconds,
                "rounding_runtime": rounded.diagnostics["runtime_seconds"],
                "support_difference_diagnostic": int(
                    np.sum(rounded.graph != truth)),
            })
            pd.DataFrame(rows).to_csv(result_path, index=False)
    (output / "configuration.json").write_text(
        json.dumps(vars(args), indent=2) + "\n")
    (output / "dataset_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dimension", type=int, default=5)
    result.add_argument("--n", type=int, default=100)
    result.add_argument("--seeds", nargs="+", type=int, default=[1701])
    result.add_argument(
        "--lambda-policy",
        choices=("fixed", "sqrt_log_d_over_n"), default="fixed")
    result.add_argument("--lambda1", type=float, default=.03)
    result.add_argument(
        "--rounding-policies", nargs="+",
        choices=(
            "threshold_grid_score_search",
            "weighted_feasible_local_search",
            "fixed_threshold"),
        default=["threshold_grid_score_search"])
    result.add_argument("--T", type=int, default=2)
    result.add_argument("--warm-iter", type=int, default=5)
    result.add_argument("--max-iter", type=int, default=8)
    result.add_argument("--max-runtime-seconds", type=float, default=5)
    result.add_argument("--rounding-seconds", type=float, default=.1)
    result.add_argument(
        "--output-dir", default="results/dagma_fast/smoke")
    result.add_argument("--resume", action="store_true")
    return result


def main(argv=None):
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
