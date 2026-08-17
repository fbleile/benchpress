#!/usr/bin/env python3
"""Paired linear-Gaussian FLOP/DAGMA-NOTREKS benchmark.

This is intentionally terminal-runnable and uses full production DAGMA
defaults unless the caller explicitly overrides the restart count/iterations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import flopsearch
import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic, is_dag,
)
from workflow.rules.structure_learning_algorithms.dagma_fast import (
    DagmaFastConfig, LinearL2Objective, fit_weighted_adjacency,
)
from workflow.rules.structure_learning_algorithms.dagma.shared import (
    deterministic_initial_adjacency,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.postselection import (
    LinearCandidateScorer, PostselectionConfig, select_postselection_candidate,
)
from workflow.rules.structure_learning_algorithms.notreks import NoTreksPenalty


def make_case(dimension, sample_size, graph_seed, graph_family="er",
              noise_scale_spread=.2):
    rng = np.random.default_rng(graph_seed)
    order = rng.permutation(dimension)
    truth = np.zeros((dimension, dimension), dtype=np.uint8)
    probability = min(1., 3.0 / max(1, dimension - 1))
    earlier_degree = np.zeros(dimension, dtype=float)
    for left in range(dimension):
        for right in range(left + 1, dimension):
            if graph_family == "hub":
                edge_probability = .65 if left < min(3, dimension) else .015
            elif graph_family == "chain_fork":
                edge_probability = .05
                if right == left + 1 or left == 0:
                    edge_probability = .8
            elif graph_family == "preferential":
                attractiveness = 1. + earlier_degree[left]
                denom = 1. + earlier_degree[:right].sum()
                edge_probability = min(.9, 2.5 * attractiveness / denom)
            else:
                edge_probability = probability
            if rng.random() < edge_probability:
                truth[order[left], order[right]] = 1
                earlier_degree[left] += 1
    weights = truth * rng.uniform(.45, .9, truth.shape)
    weights *= rng.choice([-1., 1.], truth.shape)
    noise_scales = rng.uniform(
        1. - float(noise_scale_spread),
        1. + float(noise_scale_spread), dimension)
    X = np.zeros((sample_size, dimension))
    noise = rng.normal(size=X.shape) * noise_scales
    for node in order:
        X[:, node] = noise[:, node] + X @ weights[:, node]
    # The benchmark contract is standardized data, even though the SCM noise
    # variances are deliberately heterogeneous before standardization.
    X = (X - X.mean(0)) / X.std(0, ddof=0)
    return X, truth, noise_scales


def no_trek_pairs(truth):
    reach = truth.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for node in range(len(reach)):
        reach |= reach[:, [node]] & reach[[node], :]
    return [(left, right) for left in range(len(reach))
            for right in range(left + 1, len(reach))
            if not np.any(reach[:, left] & reach[:, right])]


def supplied_pairs(truth, fraction, knowledge_seed):
    pairs = no_trek_pairs(truth)
    rng = np.random.default_rng(knowledge_seed)
    chosen = rng.permutation(len(pairs))[:round(float(fraction) * len(pairs))]
    return sorted(pairs[index] for index in chosen)


def dag_from_diagnostics(diagnostics, dimension):
    graph = np.zeros((dimension, dimension), dtype=np.uint8)
    for parent, child in diagnostics["selected_dag_edges"]:
        graph[int(parent), int(child)] = 1
    return graph


def violation_count(graph, pairs):
    reach = np.asarray(graph, dtype=bool).copy()
    np.fill_diagonal(reach, True)
    for node in range(len(reach)):
        reach |= reach[:, [node]] & reach[[node], :]
    return int(sum(np.any(reach[:, left] & reach[:, right])
                   for left, right in pairs))


def metrics(graph, truth, pairs, X):
    skeleton = (graph | graph.T).astype(bool)
    target = (truth | truth.T).astype(bool)
    upper = np.triu(np.ones_like(truth, dtype=bool), 1)
    tp = int(np.sum(skeleton & target & upper))
    fp = int(np.sum(skeleton & ~target & upper))
    fn = int(np.sum(~skeleton & target & upper))
    bic, _ = gaussian_bic(X, graph, lambda_bic=2.)
    return {
        "gaussian_bic": float(bic), "edge_count": int(graph.sum()),
        "skeleton_shd": fp + fn,
        "skeleton_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "directed_shd": int(np.sum(graph != truth)),
        "is_dag": bool(is_dag(graph)),
        "notreks_violations": violation_count(graph, pairs),
        "representation": "DAG",
    }


def flop_run(X, truth, pairs, seed, strategy, restarts, sweeps):
    started = perf_counter()
    vanilla_graph = None
    vanilla_bic = None
    if strategy == "global_greedy_hybrid":
        # The hybrid's protected incumbent is a seeded vanilla FLOP DAG.  Run
        # that seed explicitly here as well, so the benchmark enforces the
        # promised invariant whenever the vanilla graph is hard-feasible.
        _, vanilla_diagnostics = flopsearch.flop_notreks(
            X, 2., [], restarts=restarts - 1, seed=seed,
            max_signature_rounds=sweeps, search_version="fixed_signature_a",
            return_diagnostics=True)
        vanilla_graph = dag_from_diagnostics(vanilla_diagnostics, len(X[0]))
        vanilla_bic, _ = gaussian_bic(X, vanilla_graph, lambda_bic=2.)
    _, diagnostics = flopsearch.flop_notreks(
        X, 2., [] if strategy == "vanilla_flop" else pairs,
        restarts=restarts - 1, seed=seed, max_signature_rounds=sweeps,
        search_version=("fixed_signature_a" if strategy == "vanilla_flop"
                        else "global_greedy_hybrid"),
        return_diagnostics=True)
    graph = dag_from_diagnostics(diagnostics, len(X[0]))
    candidate_bic, _ = gaussian_bic(X, graph, lambda_bic=2.)
    if (vanilla_graph is not None
            and violation_count(vanilla_graph, pairs) == 0
            and vanilla_bic <= candidate_bic):
        graph = vanilla_graph
    return {
        "method": strategy, "runtime": perf_counter() - started,
        "optimizer_restarts": restarts,
        **metrics(graph, truth, pairs, X),
    }


def dagma_run(X, truth, pairs, seed, args, method):
    fit_pairs = pairs if method == "dagma_notreks" else []
    started = perf_counter()
    objective = LinearL2Objective(X)
    best = None
    for restart in range(args.dagma_restarts):
        components = []
        if fit_pairs:
            components.append(NoTreksPenalty(
                fit_pairs, X.shape[1], weight=args.notreks_weight,
                function="inv", kernel="fast"))
        result = fit_weighted_adjacency(
            objective,
            DagmaFastConfig(
                lambda1=args.dagma_lambda1,
                T=5,
                warm_iter=args.dagma_warm_iter,
                max_iter=args.dagma_max_iter,
                optimizer_tol=args.dagma_tol),
            initialization=deterministic_initial_adjacency(
                X.shape[1], seed + restart),
            structural_penalties=components)
        scorer = LinearCandidateScorer(
            X, regularizer_type="L1",
            regularizer_weight=args.dagma_lambda1)
        postselection = select_postselection_candidate(
            result.weighted_adjacency,
            scorer=scorer,
            config=PostselectionConfig(
                policy="PS5_fixed_threshold_joint_feasible",
                fixed_threshold=args.dagma_threshold,
                notreks_constraint_active=bool(fit_pairs)),
            model_class="linear_dagma", notreks_pairs=fit_pairs)
        candidate = (postselection.candidate_score, restart, result,
                     postselection)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    _, _, result, postselection = best
    graph = postselection.adjacency
    return {
        "method": method, "runtime": perf_counter() - started,
        "optimizer_restarts": args.dagma_restarts,
        "optimizer_seconds": result.runtime_seconds,
        "optimizer_iterations": result.iterations,
        "notreks_weight": args.notreks_weight if fit_pairs else 0.,
        "dagma_lambda1": args.dagma_lambda1,
        **metrics(graph, truth, pairs, X),
    }


def run_case(args, graph_seed, algorithm_seed, knowledge_fraction):
    X, truth, noise_scales = make_case(
        args.dimension, args.sample_size, graph_seed, args.graph_family,
        args.noise_scale_spread)
    pairs = supplied_pairs(truth, knowledge_fraction, args.knowledge_seed)
    base = {
        "dimension": args.dimension, "sample_size": args.sample_size,
        "graph_seed": graph_seed, "algorithm_seed": algorithm_seed,
        "knowledge_seed": args.knowledge_seed,
        "knowledge_fraction": knowledge_fraction,
        "supplied_pair_count": len(pairs),
        "noise_scale_min": float(noise_scales.min()),
        "noise_scale_max": float(noise_scales.max()),
        "true_edge_count": int(truth.sum()),
        "graph_family": args.graph_family,
    }
    rows = []
    # Vanilla methods are the zero-knowledge baseline.  At positive supplied
    # knowledge fractions only the corresponding NOTREKS-aware methods are
    # benchmarked; vanilla methods cannot consume those pairs.
    flop_methods = (("vanilla_flop",) if knowledge_fraction == 0
                    else ("global_greedy_hybrid",))
    for strategy in flop_methods:
        rows.append({**base, **flop_run(
            X, truth, pairs, algorithm_seed, strategy, args.flop_restarts,
            args.flop_sweeps)})
    if knowledge_fraction == 0:
        rows.append({**base, **dagma_run(
            X, truth, pairs, algorithm_seed, args, "dagma")})
    else:
        rows.append({**base, **dagma_run(
            X, truth, pairs, algorithm_seed, args, "dagma_notreks")})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--graph-family", choices=(
        "er", "hub", "chain_fork", "preferential"), default="er")
    parser.add_argument("--noise-scale-spread", type=float, default=.2)
    parser.add_argument("--graph-seeds", nargs="+", type=int,
                        default=[2001])
    parser.add_argument("--algorithm-seeds", nargs="+", type=int,
                        default=[7001])
    parser.add_argument("--knowledge-fractions", nargs="+", type=float,
                        default=[0., .25, .75])
    parser.add_argument("--knowledge-seed", type=int, default=9101)
    parser.add_argument("--flop-restarts", type=int, default=4)
    parser.add_argument("--flop-sweeps", type=int, default=100,
                        help="FLOP signature/order sweeps per restart")
    parser.add_argument("--dagma-restarts", type=int, default=5)
    parser.add_argument("--dagma-warm-iter", type=int, default=30000)
    parser.add_argument("--dagma-max-iter", type=int, default=60000)
    parser.add_argument("--dagma-tol", type=float, default=1e-6,
                        help="fast-DAGMA checkpoint convergence tolerance")
    parser.add_argument("--dagma-lambda1", type=float, default=0.10,
                        help="L1 strength for fast DAGMA and support scoring")
    parser.add_argument("--dagma-threshold", type=float, default=0.30,
                        help="fixed DAGMA support threshold before feasibility repair")
    parser.add_argument("--notreks-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dagma_notreks_oracle/linear_flop_dagma"))
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    rows = []
    for graph_seed in args.graph_seeds:
        for algorithm_seed in args.algorithm_seeds:
            for fraction in args.knowledge_fractions:
                rows.extend(run_case(args, graph_seed, algorithm_seed, fraction))
                print(pd.DataFrame(rows[-4:]).to_string(index=False), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "per_run.csv"
    frame = pd.DataFrame(rows)
    if args.append and path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
        frame = frame.drop_duplicates(
            ["dimension", "graph_seed", "algorithm_seed",
             "knowledge_seed", "knowledge_fraction", "graph_family", "method"],
            keep="last")
    frame.to_csv(path, index=False)
    (args.output_dir / "settings.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n")
    print(frame.groupby(["method", "knowledge_fraction"])[[
        "gaussian_bic", "skeleton_shd", "directed_shd",
        "notreks_violations", "runtime"]].mean().to_string())


if __name__ == "__main__":
    main()
