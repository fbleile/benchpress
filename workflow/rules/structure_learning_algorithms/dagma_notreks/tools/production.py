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
    run.add_argument("--warm-iter", type=int, default=30000)
    run.add_argument("--max-iter", type=int, default=60000)
    run.add_argument("--lambda-policy", default="fixed_0.03")
    run.add_argument("--regularizer-type", choices=("L1", "L2"), default="L1")
    run.add_argument(
        "--postselection-policy",
        choices=(
            "PS1_joint_feasible_greedy_score",
            "PS2_joint_feasible_local_search",
            "PS3_joint_feasible_budgeted_search",
            "PS4_joint_violation_repair",
            "PS5_fixed_threshold_joint_feasible",
            "REF_threshold_grid_scc_bic_infeasible"),
        default="PS1_joint_feasible_greedy_score")
    run.add_argument(
        "--candidate-edge-pool",
        choices=("threshold_grid", "fixed_threshold",
                 "low_threshold_supergraph", "union_threshold_supports"),
        default="threshold_grid")
    run.add_argument("--threshold-grid", nargs="+", type=float,
                     default=[.01, .03, .05, .10, .20, .30])
    run.add_argument("--fixed-threshold", type=float, default=.30)
    run.add_argument("--max-search-seconds", type=float, default=1.)
    run.add_argument("--max-expanded-nodes", type=int, default=1000)
    run.add_argument("--max-queue-size", type=int, default=1000)
    run.add_argument("--max-ambiguous-edges", type=int, default=20)
    run.add_argument("--max-indegree", type=int)
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
        "screening_floor": config.screening_floor,
        "postselection_policy": config.postselection_policy,
        "candidate_edge_pool": config.candidate_edge_pool,
        "threshold_grid": list(config.threshold_grid),
        "fixed_threshold": config.fixed_threshold,
        "selection": "configured_model_refit_score",
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
            checkpoint=10,
            postselection_policy="PS1_joint_feasible_greedy_score")
        output_dir = args.output_dir
    else:
        data, node_names = _load_csv(args.data)
        payload = load_sidecar(args.knowledge, node_names)
        pairs = named_pairs_to_indices(payload, node_names)
        config = ProductionConfig(
            restarts=args.restarts, seed=args.seed,
            warm_iter=args.warm_iter, max_iter=args.max_iter,
            lambda_policy=args.lambda_policy,
            regularizer_type=args.regularizer_type,
            postselection_policy=args.postselection_policy,
            candidate_edge_pool=args.candidate_edge_pool,
            threshold_grid=tuple(args.threshold_grid),
            fixed_threshold=args.fixed_threshold,
            max_search_seconds=args.max_search_seconds,
            max_expanded_nodes=args.max_expanded_nodes,
            max_queue_size=args.max_queue_size,
            max_ambiguous_edges=args.max_ambiguous_edges,
            max_indegree=args.max_indegree,
            constraint_regime=args.constraint_regime)
        output_dir = args.output_dir
    selected, restarts = run_production_pipeline(data, pairs, config)
    payload = _write_result(output_dir, selected, restarts, config, pairs)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
