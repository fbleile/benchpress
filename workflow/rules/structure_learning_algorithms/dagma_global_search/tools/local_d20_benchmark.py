"""Direct d=20 standard-regime oracle benchmark; no Snakemake required."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import flopsearch

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import gaussian_bic
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import ProductionConfig, run_production_pipeline
from workflow.rules.structure_learning_algorithms.flop_notreks.global_greedy import GlobalGreedyConfig, fit_global_greedy_notreks
from workflow.rules.structure_learning_algorithms.flop_soft_notreks import SoftGreedyConfig, fit_soft_notreks


def generate(seed: int, d: int = 20, n: int = 250,
             graph_type: str = "er2", scm: str = "linear",
             noise: str = "gaussian", return_raw: bool = False):
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    truth = np.zeros((d, d), dtype=np.uint8)
    weights = np.zeros((d, d), dtype=float)
    if graph_type.startswith("er"):
        expected_degree = float(graph_type[2:])
        edge_probability = expected_degree / max(1, d - 1)
        for i in range(d):
            for j in range(i + 1, d):
                u, v = int(order[i]), int(order[j])
                if rng.random() < edge_probability:
                    truth[u, v] = 1
    elif graph_type == "chain":
        for i in range(d - 1):
            truth[int(order[i]), int(order[i + 1])] = 1
    elif graph_type == "hub":
        hub = int(order[0])
        for node in order[1:]:
            truth[hub, int(node)] = 1
    else:
        raise ValueError("graph_type must be er<degree>, chain, or hub")
    weights[truth != 0] = rng.uniform(.5, 1.0, size=int(truth.sum()))
    weights[truth != 0] *= rng.choice([-1., 1.], size=int(truth.sum()))
    if noise == "gaussian":
        # Unequal-variance Gaussian SCM.  Draw the innovation variances
        # independently of the graph, node depth, and edge weights:
        # log(sigma_j^2) ~ Unif(-log 2, log 2).
        log_variances = rng.uniform(-np.log(2.0), np.log(2.0), size=d)
        innovation_scales = np.exp(0.5 * log_variances)
        X = rng.normal(size=(n, d)) * innovation_scales
    elif noise == "laplace":
        X = rng.laplace(size=(n, d)) / np.sqrt(2.0)
    elif noise == "uniform":
        X = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(n, d))
    else:
        raise ValueError("noise must be gaussian, laplace, or uniform")
    for node in order:
        parents = np.flatnonzero(truth[:, node])
        if len(parents):
            signal = X[:, parents] @ weights[parents, node]
            if scm == "linear":
                X[:, node] += signal
            elif scm == "nonlinear":
                X[:, node] += np.tanh(signal)
            else:
                raise ValueError("scm must be linear or nonlinear")
    raw_X = X.copy()
    X = (X - X.mean(0)) / X.std(0, ddof=0)
    reach = truth.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(d):
        reach |= reach[:, [k]] & reach[[k], :]
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)
             if not np.any(reach[:, i] & reach[:, j])]
    return (X, truth, pairs, raw_X) if return_raw else (X, truth, pairs)


def row(X, truth, method, A, runtime, **extra):
    bic, _ = gaussian_bic(X, A, lambda_bic=2.)
    return {"method": method, "bic": float(bic), "edges": int(A.sum()),
            "directed_shd": int(np.sum(A != truth)), "runtime": float(runtime), **extra}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(4001, 4016)))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/d20_standard_oracle_local"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        X, truth, pairs = generate(seed)
        t = time.perf_counter()
        _, diag = flopsearch.flop_notreks(
            X, 2., [], restarts=1, seed=seed,
            search_version="global_greedy_rust", return_diagnostics=True)
        A = np.zeros_like(truth)
        for u, v in diag["selected_dag_edges"]:
            A[int(u), int(v)] = 1
        rows.append(row(X, truth, "vanilla_flop", A, time.perf_counter() - t))

        t = time.perf_counter()
        hard = fit_global_greedy_notreks(
            X, pairs, GlobalGreedyConfig(seed=seed, restarts=8,
                                         max_sweeps=8, inner_backend="python"))
        rows.append(row(X, truth, "flop_notreks_global_greedy",
                        hard.adjacency, time.perf_counter() - t))

        t = time.perf_counter()
        soft = fit_soft_notreks(
            X, pairs, SoftGreedyConfig(seed=seed, restarts=8,
                                       max_sweeps=16, lazy_top_k=16))
        rows.append(row(X, truth, "flop_soft_notreks", soft.adjacency,
                        time.perf_counter() - t, raw_bic=soft.raw_bic,
                        raw_notreks=soft.raw_notreks,
                        support_evaluations=soft.support_evaluations))

        t = time.perf_counter()
        vanilla, _ = run_production_pipeline(X, [], ProductionConfig())
        rows.append(row(X, truth, "vanilla_dagma", vanilla.adjacency,
                        time.perf_counter() - t))

        t = time.perf_counter()
        constrained, _ = run_production_pipeline(X, pairs, ProductionConfig())
        rows.append(row(X, truth, "dagma_notreks", constrained.adjacency,
                        time.perf_counter() - t))
        print(pd.DataFrame(rows[-5:]).to_string(index=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_seed.csv", index=False)
    frame.groupby("method").agg({"bic": ["mean", "std", "median"],
                                  "directed_shd": ["mean", "std"],
                                  "runtime": ["mean", "std"]}).to_csv(
                                      args.output_dir / "aggregate.csv")


if __name__ == "__main__":
    main()
