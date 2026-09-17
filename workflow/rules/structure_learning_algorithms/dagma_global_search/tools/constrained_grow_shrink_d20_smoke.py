"""Smoke comparison for constrained fixed-order grow-shrink FLOP fitting."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import flopsearch

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    gaussian_bic,
    is_dag,
    topological_order,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import generate
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import (
    cpdag_shd,
    flop_notreks_candidate,
    flop_notreks_order_guided_candidate,
    select_knowledge,
    trek_violation_mass,
    vanilla_flop_candidate,
)


def constrained_grow_shrink(X, order, pairs, *, seed, lambda_bic=2.0):
    """FLOP's from-scratch grow-shrink parent fitting with no-trek checks.

    Nodes are processed in ``order``.  Within each node, FLOP shuffles the
    admissible nonparents, accepts every non-worsening addition in that pass,
    then shuffles the current parents and accepts every non-worsening removal.
    The only modification is the pre-score feasibility check for additions.
    """
    d = X.shape[1]
    graph = np.zeros((d, d), dtype=np.uint8)
    position = {node: i for i, node in enumerate(order)}
    current = float(gaussian_bic(X, graph, lambda_bic=lambda_bic)[0])
    accepted = 0
    rng = np.random.default_rng(seed ^ 0x454E44464C4F5000)

    for child in order:
        nonparents = list(order[:position[child]])
        while True:
            changed = False
            shuffled = list(rng.permutation(nonparents))
            for parent in shuffled:
                if graph[parent, child]:
                    continue
                proposal = graph.copy()
                proposal[parent, child] = 1
                # NOTREKS is checked before the score/refit computation.
                if common_ancestor_violations(proposal, pairs):
                    continue
                value = float(gaussian_bic(
                    X, proposal, lambda_bic=lambda_bic)[0])
                if value <= current + 1e-10:
                    graph[parent, child] = 1
                    current = value
                    nonparents.remove(parent)
                    accepted += 1
                    changed = True
            if not changed:
                break

        while True:
            changed = False
            parents = list(np.flatnonzero(graph[:, child]))
            for parent in rng.permutation(parents):
                proposal = graph.copy()
                proposal[parent, child] = 0
                value = float(gaussian_bic(
                    X, proposal, lambda_bic=lambda_bic)[0])
                if value <= current + 1e-10:
                    graph[parent, child] = 0
                    current = value
                    nonparents.append(int(parent))
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
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--flop-sweeps", type=int, default=16)
    parser.add_argument("--flop-local-passes", type=int, default=8)
    parser.add_argument("--order-guided-repair-candidates", type=int, default=2)
    parser.add_argument("--order-guided-coverage-starts", type=int, default=1)
    parser.add_argument("--order-guided-lex-fraction", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed in enumerate(args.seeds, 1):
        X, truth, all_pairs = generate(
            seed, d=args.d, n=args.n, graph_type=args.graph_type,
            scm="linear", noise="gaussian")
        pairs = select_knowledge(all_pairs, args.knowledge_fraction, seed)

        vanilla_started = time.perf_counter()
        vanilla, vanilla_diag = vanilla_flop_candidate(X, seed, args.attempts)
        vanilla_runtime = time.perf_counter() - vanilla_started

        started = time.perf_counter()
        grow, grow_diag = flopsearch.flop_notreks(
            X, 2.0, pairs,
            restarts=args.attempts,
            seed=seed,
            max_signature_rounds=20,
            search_version="alternating_full_refit_b",
            return_dag=True,
            return_diagnostics=True)
        grow = np.asarray(grow, dtype=np.uint8)
        grow_bic = float(gaussian_bic(X, grow, lambda_bic=2.0)[0])
        grow_runtime = time.perf_counter() - started

        started = time.perf_counter()
        local, local_diag = flop_notreks_candidate(
            X, pairs, seed, args.attempts, args.flop_sweeps,
            search_version="local_greedy_rust",
            local_greedy_passes=args.flop_local_passes)
        local_runtime = time.perf_counter() - started

        started = time.perf_counter()
        guided, guided_diag = flop_notreks_order_guided_candidate(
            X, pairs, seed, args.attempts, args.flop_sweeps,
            local_greedy_passes=args.flop_local_passes,
            lex_fraction=args.order_guided_lex_fraction,
            repair_candidates=args.order_guided_repair_candidates,
            coverage_starts=args.order_guided_coverage_starts)
        guided_runtime = time.perf_counter() - started

        rows.extend([
            {
                "seed": seed, "method": "vanilla_flop",
                "runtime": vanilla_runtime,
                "bic": float(gaussian_bic(X, vanilla, lambda_bic=2.0)[0]),
                "edges": int(vanilla.sum()), "accepted_toggles": np.nan,
                "cpdag_shd": cpdag_shd(truth, vanilla),
                "skeleton_shd": skeleton_shd(truth, vanilla),
                "violations": common_ancestor_violations(vanilla, pairs),
                "trek_violation_mass": trek_violation_mass(vanilla, pairs),
                "dag": is_dag(vanilla),
                "attempts": vanilla_diag["optimizer_restarts"],
            },
            {
                "seed": seed, "method": "constrained_grow_shrink",
                "runtime": grow_runtime, "bic": grow_bic,
                "edges": int(grow.sum()),
                "accepted_toggles": np.nan,
                "cpdag_shd": cpdag_shd(truth, grow),
                "skeleton_shd": skeleton_shd(truth, grow),
                "violations": common_ancestor_violations(grow, pairs),
                "trek_violation_mass": trek_violation_mass(grow, pairs),
                "dag": is_dag(grow),
                "attempts": grow_diag["restarts_completed"],
            },
            {
                "seed": seed, "method": "flop_nt_local",
                "runtime": local_runtime,
                "bic": float(gaussian_bic(X, local, lambda_bic=2.0)[0]),
                "edges": int(local.sum()), "accepted_toggles": np.nan,
                "cpdag_shd": cpdag_shd(truth, local),
                "skeleton_shd": skeleton_shd(truth, local),
                "violations": common_ancestor_violations(local, pairs),
                "trek_violation_mass": trek_violation_mass(local, pairs),
                "dag": is_dag(local),
                "attempts": local_diag["optimizer_restarts"],
            },
            {
                "seed": seed, "method": "flop_nt_order_guided",
                "runtime": guided_runtime,
                "bic": float(gaussian_bic(X, guided, lambda_bic=2.0)[0]),
                "edges": int(guided.sum()), "accepted_toggles": np.nan,
                "cpdag_shd": cpdag_shd(truth, guided),
                "skeleton_shd": skeleton_shd(truth, guided),
                "violations": common_ancestor_violations(guided, pairs),
                "trek_violation_mass": trek_violation_mass(guided, pairs),
                "raw_trek_violations": int(
                    common_ancestor_violations(
                        guided_diag["candidate_graph"], pairs)),
                "raw_trek_violation_mass": trek_violation_mass(
                    guided_diag["candidate_graph"], pairs),
                "dag": is_dag(guided),
                "attempts": guided_diag["optimizer_restarts"],
                "raw_feasible": guided_diag[
                    "order_guided_raw_feasible_count"],
                "repaired_winner": int(
                    guided_diag["order_guided_winner"] == "repaired"),
                "violation_trajectory": json.dumps(
                    guided_diag["order_guided_violation_trajectory"]),
            },
        ])
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output_dir / "per_seed.csv", index=False)
        print(f"[{index}/{len(args.seeds)}] seed={seed}", flush=True)
        print(frame[frame.seed == seed].to_string(index=False), flush=True)

    frame = pd.DataFrame(rows)
    frame.groupby("method").agg(
        cpdag_shd_mean=("cpdag_shd", "mean"),
        cpdag_shd_std=("cpdag_shd", "std"),
        skeleton_shd_mean=("skeleton_shd", "mean"),
        runtime_mean=("runtime", "mean"),
        violations_max=("violations", "max"),
        trek_violation_mass_mean=("trek_violation_mass", "mean"),
        trek_violation_mass_max=("trek_violation_mass", "max"),
        raw_trek_violations_max=("raw_trek_violations", "max"),
        raw_trek_violation_mass_max=("raw_trek_violation_mass", "max"),
        edges_mean=("edges", "mean"),
        raw_feasible_mean=("raw_feasible", "mean"),
        repaired_winner_rate=("repaired_winner", "mean"),
    ).reset_index().to_csv(args.output_dir / "summary.csv", index=False)
    print(frame.groupby("method").agg(
        cpdag_shd_mean=("cpdag_shd", "mean"),
        cpdag_shd_std=("cpdag_shd", "std"),
        skeleton_shd_mean=("skeleton_shd", "mean"),
        runtime_mean=("runtime", "mean"),
        violations_max=("violations", "max"),
        trek_violation_mass_mean=("trek_violation_mass", "mean"),
        trek_violation_mass_max=("trek_violation_mass", "max"),
        raw_trek_violations_max=("raw_trek_violations", "max"),
        raw_trek_violation_mass_max=("raw_trek_violation_mass", "max"),
        edges_mean=("edges", "mean"),
        raw_feasible_mean=("raw_feasible", "mean"),
        repaired_winner_rate=("repaired_winner", "mean"),
    ).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
