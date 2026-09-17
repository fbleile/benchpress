"""Small predeclared Rust soft-NOTREKS weight sensitivity study."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import flopsearch

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations, gaussian_bic,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[30., 100., 300.])
    parser.add_argument("--redesigned", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        X, truth, pairs = generate(seed)
        methods = [("hard", None)] + [("soft", weight) for weight in args.weights]
        for method, weight in methods:
            started = time.perf_counter()
            if method == "hard":
                A, diagnostics = flopsearch.flop_notreks(
                    X, 2., pairs, restarts=1, seed=seed,
                    max_signature_rounds=8,
                    search_version="global_greedy_rust",
                    return_dag=True,
                    return_diagnostics=True)
                support_evaluations = np.nan
            else:
                if args.redesigned:
                    A, diagnostics = flopsearch.soft_global_greedy_redesigned(
                        X, pairs, restarts=8, max_sweeps=8,
                        lambda_bic=2., lazy_top_k=16, seed=seed)
                else:
                    A, diagnostics = flopsearch.soft_global_greedy(
                        X, pairs, restarts=8, max_sweeps=8,
                        lambda_bic=2., soft_notreks_weight=weight,
                        lazy_top_k=16, seed=seed)
                support_evaluations = diagnostics["support_evaluations"]
            A = np.asarray(A, dtype=np.uint8)
            bic, _ = gaussian_bic(X, A, lambda_bic=2.)
            rows.append({
                "seed": seed,
                "method": method,
                "soft_weight": weight,
                "bic": float(bic),
                "edges": int(A.sum()),
                "true_edges": int(truth.sum()),
                "true_positive": int(np.sum((A != 0) & (truth != 0))),
                "false_positive": int(np.sum((A != 0) & (truth == 0))),
                "false_negative": int(np.sum((A == 0) & (truth != 0))),
                "directed_shd": int(np.sum(A != truth)),
                "notreks_violations": int(common_ancestor_violations(A, pairs)),
                "support_evaluations": support_evaluations,
                "runtime": time.perf_counter() - started,
            })
        print(pd.DataFrame(rows[-len(methods):]).to_string(index=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_seed.csv", index=False)
    aggregate = frame.groupby(["method", "soft_weight"], dropna=False).agg({
        "bic": ["mean", "std"],
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
