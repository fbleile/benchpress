"""Small fair-comparison pilot for d=20 oracle data."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import flopsearch

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import gaussian_bic
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import ProductionConfig, run_production_pipeline
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import generate


def score(X, truth, method, A, runtime, **extra):
    bic, _ = gaussian_bic(X, A, lambda_bic=2.)
    truth = np.asarray(truth, dtype=bool)
    estimated = np.asarray(A, dtype=bool)
    true_positive = int(np.sum(estimated & truth))
    false_positive = int(np.sum(estimated & ~truth))
    false_negative = int(np.sum(~estimated & truth))
    return {"method": method, "bic": float(bic), "edges": int(A.sum()),
            "true_edges": int(truth.sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "directed_shd": int(np.sum(A != truth)), "runtime": float(runtime), **extra}


def rust_hard(X, pairs, seed, sweeps, attempts):
    # The Rust API adds the initial attempt to the configured `restarts`
    # value, so pass attempts-1 to match DAGMA's actual restart count.
    _, diag = flopsearch.flop_notreks(
        X, 2., pairs, restarts=max(0, attempts - 1), seed=seed,
        max_signature_rounds=sweeps, search_version="global_greedy_rust",
        return_diagnostics=True)
    A = np.zeros((X.shape[1], X.shape[1]), dtype=np.uint8)
    for u, v in diag["selected_dag_edges"]:
        A[int(u), int(v)] = 1
    return A, diag


def dagma(X, pairs, seed, notreks, attempts):
    cfg = ProductionConfig(restarts=attempts, seed=seed)
    best, _ = run_production_pipeline(X, pairs if notreks else [], cfg)
    return best.adjacency, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sweeps", type=int, default=16)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        X, truth, pairs = generate(seed)
        t = time.perf_counter()
        A, _ = rust_hard(X, pairs, seed, args.sweeps, args.restarts)
        rows.append(score(X, truth, "flop_notreks_rust_hard", A,
                          time.perf_counter() - t, sweeps=args.sweeps,
                          restarts=args.restarts))

        for notreks, label in ((False, "dagma_bic_grid"),
                               (True, "dagma_notreks_bic_grid")):
            t = time.perf_counter()
            A, result = dagma(X, pairs, seed, notreks, args.restarts)
            rows.append(score(
                X, truth, label, A, time.perf_counter() - t,
                restarts=args.restarts,
                candidate_edges=result.candidate_edges,
                selected_threshold=result.candidate_threshold,
                final_edges=result.final_edges))
        print(pd.DataFrame(rows[-3:]).to_string(index=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_seed.csv", index=False)
    frame.groupby("method").agg({
        "bic": ["mean", "std", "median"],
        "edges": ["mean", "std"],
        "directed_shd": ["mean", "std"],
        "runtime": ["mean", "std"],
        "true_edges": ["mean", "std"],
    }).to_csv(args.output_dir / "aggregate.csv")
    print(frame.groupby("method")[
        ["bic", "edges", "true_edges", "true_positive", "false_positive",
         "false_negative", "directed_shd", "runtime"]].mean().to_string())


if __name__ == "__main__":
    main()
