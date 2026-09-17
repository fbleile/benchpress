#!/usr/bin/env python3
"""Canonical command-line entry point for the production pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.knowledge import (
    load_sidecar,
    named_pairs_to_indices,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    ProductionConfig,
    run_production_pipeline,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical log-det DAGMA-NOTREKS pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run on a CSV data set and sidecar")
    run.add_argument("--data", required=True, type=Path)
    run.add_argument("--knowledge", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--restarts", type=int, default=5)
    run.add_argument("--seed", type=int, default=1729)
    run.add_argument(
        "--initialization-mode",
        choices=("empty_random", "empty_feasible_random"),
        default="empty_random",
        help="use empty plus random weights, or empty plus supplied-NOTREKS-feasible random DAGs")
    run.add_argument("--initialization-edge-probability", type=float, default=0.15)
    run.add_argument(
        "--proximal-l1", action="store_true",
        help="use proximal-gradient soft-thresholding for the L1 term")
    run.add_argument(
        "--postselection-policy",
        choices=("feasible_parent_shrink",
                 "normalized_greedy_projection_refit",
                 "normalized_greedy_projection_refit_shrink"),
        default="feasible_parent_shrink")
    run.add_argument("--warm-iter", type=int, default=30000)
    run.add_argument("--max-iter", type=int, default=60000)
    run.add_argument("--lambda-policy", default="fixed_0.03")
    run.add_argument("--regularizer-type", choices=("L1", "L2"), default="L1")
    run.add_argument(
        "--constraint-regime",
        choices=("DAG_only", "DAG_NOTREKS_one_correct",
                 "DAG_NOTREKS_several_correct",
                 "DAG_NOTREKS_one_percent"))

    smoke = subparsers.add_parser(
        "smoke", help="run a small deterministic end-to-end validation")
    smoke.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dagma_notreks_oracle/production_smoke"))
    smoke.add_argument("--seed", type=int, default=1729)
    return parser


def _load_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if len(header) != data.shape[1]:
        raise ValueError("CSV header and data column counts differ")
    return data, header


def _smoke_data(seed: int):
    rng = np.random.default_rng(seed)
    n = 120
    noise = rng.normal(size=(n, 5))
    data = noise.copy()
    data[:, 1] += 0.8 * data[:, 0]
    data[:, 3] += 0.7 * data[:, 2]
    data[:, 4] += 0.6 * data[:, 3]
    # The two disconnected components have no common ancestors.
    return data, [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]


def _write_result(output_dir: Path, selected, restarts, config, pairs=()):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "weighted_adjacency.npy", selected.weighted_adjacency)
    np.save(output_dir / "adjacency.npy", selected.adjacency)
    np.save(output_dir / "coefficients.npy", selected.coefficients)
    payload = {
        "method": "dagma_notreks",
        "dag_penalty": "logdet",
        "notreks_function": config.trek_function,
        "notreks_kernel": config.trek_kernel,
        "lambda1": config.lambda1,
        "lambda_policy": config.lambda_policy,
        "lambda1_effective": selected.lambda1_effective,
        "regularizer_type": config.regularizer_type,
        "trek_weight": config.trek_weight,
        "restarts": config.restarts,
        "initialization_mode": config.initialization_mode,
        "initialization_edge_probability": config.initialization_edge_probability,
        "proximal_l1": config.proximal_l1,
        "dagma_postselection_policy": config.dagma_postselection_policy,
        "screening_floor": config.screening_floor,
        "postselection_policy": "feasible_parent_shrink",
        "selection": "feasibility_first_parent_shrink_refit",
        "selected_restart": selected.restart,
        "selected_bic": selected.exact_bic,
        "feasibility_threshold": selected.feasibility_threshold,
        "candidate_threshold": selected.candidate_threshold,
        "candidate_edges": selected.candidate_edges,
        "final_edges": selected.final_edges,
        "oracle_violations": selected.oracle_violations,
        "number_of_notreks_pairs": len(pairs),
        "constraint_regime": config.constraint_regime,
        **selected.standardization,
        "postselection": selected.postselection,
        "restart_diagnostics": [
            {
                "restart": result.restart,
                "initialization": result.initialization,
                "initial_edges": result.initial_edges,
                "exact_bic": result.exact_bic,
                "candidate_threshold": result.candidate_threshold,
                "candidate_edges": result.candidate_edges,
                "final_edges": result.final_edges,
                "runtime": result.runtime,
            }
            for result in restarts
        ],
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "smoke":
        data, pairs = _smoke_data(args.seed)
        config = ProductionConfig(
            restarts=1, seed=args.seed, warm_iter=25, max_iter=40,
            checkpoint=10)
        output_dir = args.output_dir
    else:
        data, node_names = _load_csv(args.data)
        payload = load_sidecar(args.knowledge, node_names)
        pairs = named_pairs_to_indices(payload, node_names)
        config = ProductionConfig(
            restarts=args.restarts, seed=args.seed,
            initialization_mode=args.initialization_mode,
            initialization_edge_probability=args.initialization_edge_probability,
            proximal_l1=args.proximal_l1,
            dagma_postselection_policy=args.postselection_policy,
            warm_iter=args.warm_iter, max_iter=args.max_iter,
            lambda_policy=args.lambda_policy,
            regularizer_type=args.regularizer_type,
            constraint_regime=args.constraint_regime)
        output_dir = args.output_dir
    selected, restarts = run_production_pipeline(data, pairs, config)
    payload = _write_result(output_dir, selected, restarts, config, pairs)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
