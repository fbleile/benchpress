"""Compare vanilla DAGMA under legacy defaults and common BIC postselection."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import gaussian_bic
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    ProductionConfig, run_production_pipeline,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import generate


def row(X, truth, method, result, runtime):
    estimated = np.asarray(result.adjacency, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    bic, _ = gaussian_bic(X, estimated, lambda_bic=2.)
    return {
        "method": method,
        "bic": float(bic),
        "candidate_edges": int(result.candidate_edges),
        "final_edges": int(result.final_edges),
        "true_edges": int(truth.sum()),
        "true_positive": int(np.sum(estimated & truth)),
        "false_positive": int(np.sum(estimated & ~truth)),
        "false_negative": int(np.sum(~estimated & truth)),
        "directed_shd": int(np.sum(estimated != truth)),
        "selected_threshold": float(result.candidate_threshold),
        "runtime": float(runtime),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        X, truth, pairs = generate(seed)
        configs = [
            ("dagma_legacy_default", [], ProductionConfig(
                restarts=5, seed=seed)),
            ("dagma_common_bic_grid", [], ProductionConfig(
                restarts=5, seed=seed)),
            ("dagma_notreks_legacy_default", pairs, ProductionConfig(
                restarts=5, seed=seed)),
            ("dagma_notreks_common_bic_grid", pairs, ProductionConfig(
                restarts=5, seed=seed)),
        ]
        for method, no_trek_pairs, config in configs:
            started = time.perf_counter()
            result, _ = run_production_pipeline(X, no_trek_pairs, config)
            rows.append(row(X, truth, method, result,
                             time.perf_counter() - started))
        print(pd.DataFrame(rows[-2:]).to_string(index=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_seed.csv", index=False)
    frame.groupby("method").agg({
        "bic": ["mean", "std"],
        "candidate_edges": ["mean", "std"],
        "final_edges": ["mean", "std"],
        "true_edges": ["mean", "std"],
        "true_positive": ["mean", "std"],
        "false_positive": ["mean", "std"],
        "false_negative": ["mean", "std"],
        "directed_shd": ["mean", "std"],
        "runtime": ["mean", "std"],
    }).to_csv(args.output_dir / "aggregate.csv")
    print(frame.groupby("method")[[
        "bic", "candidate_edges", "final_edges", "true_edges",
        "true_positive", "false_positive", "false_negative",
        "directed_shd", "runtime"]].mean().to_string())


if __name__ == "__main__":
    main()
