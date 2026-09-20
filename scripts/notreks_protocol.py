#!/usr/bin/env python3
"""Small, reproducible executable slice of the NOTREKS experiment protocol.

This runner deliberately keeps the protocol artifact layer separate from the
legacy factorial benchmark.  It is intended for bounded pilots first; the
same artifacts are suitable for expanding the grid later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from io import StringIO
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.notreks_benchmark_pipeline import method_run
from scripts.notreks_protocol_registry import REGISTRY, select_registry, scaled_replicates
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import (
    flop_notreks_candidate, metrics,
)

MASTER_SEED = 20260917
METHODS = ("flop", "flop_notreks", "dagma", "dagma_notreks")
METHOD_LABELS = {
    "flop": "vanilla_flop",
    "flop_notreks": "flop_notreks",
    "dagma": "vanilla_dagma",
    "dagma_notreks": "dagma_notreks",
}


def derive_seed(namespace: str, *parts: object, master: int = MASTER_SEED) -> int:
    payload = json.dumps([namespace, master, *parts], separators=(",", ":"),
                         sort_keys=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def graph_pairs(truth: np.ndarray) -> list[tuple[int, int]]:
    reach = truth.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(truth.shape[0]):
        reach |= reach[:, [k]] & reach[[k], :]
    ancestors = reach.T
    return [(i, j) for i in range(truth.shape[0]) for j in range(i + 1, truth.shape[0])
            if not np.any(ancestors[i] & ancestors[j])]


def make_graph(d: int, family: str, density: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    graph = np.zeros((d, d), dtype=np.uint8)
    if family == "er":
        p = density / max(1, d - 1)
        for i in range(d):
            for j in range(i + 1, d):
                if rng.random() < p:
                    graph[order[i], order[j]] = 1
    elif family == "ws":
        # Use the repository's existing pcalg randDAG implementation.  Its
        # `d` argument is the mean total neighbourhood degree, matching the
        # ER-k naming used by this protocol.
        del rng, order, graph
        expression = (
            'source("resources/binarydatagen/generate_DAG.R"); '
            'args <- commandArgs(trailingOnly=TRUE); '
            # R uses signed 32-bit seeds; protocol hashes are uint32 values.
            'set.seed(as.integer(as.numeric(args[[1]]) %% 2147483647)); '
            'a <- randDAGMaxParents(n=as.integer(args[[2]]), '
            'd=as.integer(args[[3]]), method="watts", '
            'par1=NULL, par2=NULL, DAG=TRUE, max_parents=NULL); '
            'write.table(a, file="", sep=",", row.names=FALSE, '
            'col.names=FALSE, quote=FALSE)'
        )
        try:
            result = subprocess.run(
                ["Rscript", "--vanilla", "-e", expression,
                 str(int(seed)), str(int(d)), str(int(density))],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "pcalg randDAG(method='watts') failed: "
                f"{exc.stderr.strip()}"
            ) from exc
        graph = np.loadtxt(StringIO(result.stdout), delimiter=",")
        graph = (np.asarray(graph) != 0).astype(np.uint8)
        if graph.shape != (d, d):
            raise ValueError(f"pcalg watts returned shape {graph.shape}, expected {(d, d)}")
        np.fill_diagonal(graph, 0)
        return graph
    else:
        raise ValueError("family must be er or ws")
    return graph


def make_data(truth: np.ndarray, n: int, graph_seed: int, data_seed: int):
    d = truth.shape[0]
    order = _topological_order(truth)
    rng = np.random.default_rng(data_seed)
    weights = np.zeros_like(truth, dtype=float)
    count = int(truth.sum())
    weights[truth.astype(bool)] = rng.uniform(.5, 1.0, size=count)
    weights[truth.astype(bool)] *= rng.choice([-1.0, 1.0], size=count)
    # Unequal-variance Gaussian SCM: log(sigma_j^2) is iid uniform on
    # [-log(2), log(2)], independently of graph structure and depth.
    log_variances = rng.uniform(-np.log(2.0), np.log(2.0), size=d)
    innovation_scales = np.exp(0.5 * log_variances)
    x = rng.normal(size=(n, d)) * innovation_scales
    for node in order:
        parents = np.flatnonzero(truth[:, node])
        if len(parents):
            x[:, node] += x[:, parents] @ weights[parents, node]
    mean, std = x.mean(0), x.std(0, ddof=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (x - mean) / std, weights


def _topological_order(graph: np.ndarray) -> list[int]:
    indegree = graph.sum(0).astype(int).tolist()
    remaining = set(range(graph.shape[0]))
    result = []
    while remaining:
        ready = sorted(v for v in remaining if indegree[v] == 0)
        if not ready:
            raise ValueError("graph is not acyclic")
        for v in ready:
            remaining.remove(v); result.append(v)
            for child in np.flatnonzero(graph[v]):
                indegree[int(child)] -= 1
    return result


def select_pairs(all_pairs, q: float, seed: int):
    if not all_pairs:
        return []
    if q == 1.0:
        return [tuple(p) for p in all_pairs]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(all_pairs))
    count = max(1, int(round(q * len(all_pairs))))
    return [tuple(all_pairs[int(i)]) for i in order[:count]]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def protocol_method_run(name, x, pairs, seed, args):
    """Dispatch the protocol's declared methods without legacy aliases.

    ``flop_notreks`` is the FLOP-like implementation selected as the standard
    NOTREKS method.  ``local_greedy_rust`` remains available in the legacy
    host runner as an explicitly named comparison, but is not used here.
    """
    if name == "flop_notreks":
        started = time.perf_counter()
        candidate, diag = flop_notreks_candidate(
            x, pairs, seed, args.attempts, args.flop_sweeps,
            search_version="flop_like", local_greedy_passes=8)
        return candidate, diag, time.perf_counter() - started
    mapped = {"dagma_notreks": "dagma-pstrek", "dagma": "dagma",
              "flop": "flop"}[name]
    return method_run(
        mapped, x, pairs, seed, args.attempts, args.flop_sweeps,
        args.dagma_stages, args.dagma_warm_iter, args.dagma_max_iter,
        args.trek_weight)


def run(args):
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")
    out = args.output_root.resolve()
    specs = select_registry(args.experiments, args.fraction)
    executable = [spec for spec in specs if spec.derives_from is None]
    deferred = [spec.name for spec in specs if spec.derives_from is not None]
    if [spec.name for spec in executable] != ["main"]:
        raise NotImplementedError(
            "protocol experiment handlers are not yet enabled for: "
            + ", ".join(spec.name for spec in executable if spec.name != "main"))
    for part in ("graph_bank", "data", "knowledge", "raw", "analysis"):
        (out / part).mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.max_wall_hours * 3600.0
    ledger = []
    rows = []
    graph_audit = []
    # The 1% pilot is intentionally one graph cell but exercises both graph
    # construction and all information levels. Larger fractions expand this
    # deterministic list without changing any seed already assigned.
    candidates = list(REGISTRY["main"].cells)
    count = max(1, int(np.ceil(args.fraction * len(candidates))))
    candidates = candidates[:count]
    n_values = [args.n]
    if args.fraction >= .25 and args.n == 500:
        n_values = [100, 500, 2000]
    graph_reps = scaled_replicates(REGISTRY["main"], args.fraction)
    q_values = list(REGISTRY["main"].q_values)
    for cell_index, (d, family, density) in enumerate(candidates):
        for replicate in range(graph_reps):
            graph_seed = derive_seed("graph", d, family, density, replicate,
                                    master=args.master_seed)
            truth = make_graph(d, family, density, graph_seed)
            all_pairs = graph_pairs(truth)
            if len(all_pairs) < 4:
                raise RuntimeError(f"ineligible graph: {d=} {family=} {density=} "
                                   f"{replicate=} has |I|={len(all_pairs)}")
            graph_id = f"g{cell_index:02d}_{family}{density}_r{replicate:02d}"
            np.savez_compressed(out / "graph_bank" / f"{graph_id}.npz",
                                adjacency=truth, graph_seed=graph_seed,
                                no_trek_pairs=np.asarray(all_pairs, dtype=int))
            graph_audit.append({"graph_id": graph_id, "d": d, "family": family,
                                "density": density, "replicate": replicate,
                                "graph_seed": graph_seed, "edges": int(truth.sum()),
                                "no_trek_pairs": len(all_pairs), "eligible": True})
            for n in n_values:
                data_seed = derive_seed("data", graph_id, n, master=args.master_seed)
                x, weights = make_data(truth, n, graph_seed, data_seed)
                data_id = f"{graph_id}_n{n}"
                np.savez_compressed(out / "data" / f"{data_id}.npz",
                                    X=x, truth=truth, weights=weights,
                                    data_seed=data_seed)
                vanilla_cache = {}
                for q in q_values:
                    rounds = REGISTRY["main"].q25_rounds if q == .25 else 1
                    for round_id in range(rounds):
                        knowledge_seed = derive_seed("knowledge", graph_id, n,
                                                     q, round_id,
                                                     master=args.master_seed)
                        pairs = select_pairs(all_pairs, q, knowledge_seed)
                        prior_id = f"{data_id}_q{str(q).replace('.', 'p')}_r{round_id}"
                        write_json(out / "knowledge" / f"{prior_id}.json", {
                            "graph_id": graph_id, "data_id": data_id,
                            "q": q, "round": round_id,
                            "knowledge_seed": knowledge_seed,
                            "pairs": [list(p) for p in pairs],
                        })
                        for method in METHODS:
                            if time.time() >= deadline:
                                raise TimeoutError(
                                    f"protocol pilot reached --max-wall-hours={args.max_wall_hours}")
                            method_seed = derive_seed("method", data_id, q,
                                                      round_id, method,
                                                      master=args.master_seed)
                            ledger.append({"graph_id": graph_id, "data_id": data_id,
                                           "prior_id": prior_id, "method": method,
                                           "graph_seed": graph_seed,
                                           "data_seed": data_seed,
                                           "knowledge_seed": knowledge_seed,
                                           "method_seed": method_seed,
                                           "attempts": args.attempts})
                            cache_key = (data_id, method)
                            t0 = time.perf_counter()
                            if method in {"flop", "dagma"} and cache_key in vanilla_cache:
                                candidate, diag, runtime = vanilla_cache[cache_key]
                            else:
                                candidate, diag, runtime = protocol_method_run(
                                    method, x, pairs, method_seed, args)
                                if method in {"flop", "dagma"}:
                                    vanilla_cache[cache_key] = (candidate, diag, runtime)
                            row = metrics(
                                x, truth, pairs, METHOD_LABELS[method],
                                diag["candidate_graph"], candidate, diag["cpdag"],
                                runtime, args.attempts,
                                diag.get("optimizer_restarts", args.attempts))
                            row.update({"protocol_version": "pilot-20260917-v1",
                                        "experiment_id": "main",
                                        "graph_id": graph_id, "data_id": data_id,
                                        "instance_id": data_id,
                                        "prior_id": prior_id, "d": d, "n": n,
                                        "er_degree": density,
                                        "graph_family": family, "graph_density": density,
                                        "graph_replicate": replicate, "q": q,
                                        "knowledge_fraction": q,
                                        "knowledge_round": round_id,
                                        "graph_seed": graph_seed, "data_seed": data_seed,
                                        "knowledge_seed": knowledge_seed,
                                        "method_seed": method_seed,
                                        "data_model": "linear_gaussian_scm",
                                        "discovery_score": "gaussian_bic",
                                        "knowledge_pairs": len(pairs),
                                        "solver_status": "ok",
                                        "truth_bic": float("nan"),
                                        "bic_gap_to_truth": float("nan"),
                                        "wall_seconds": time.perf_counter() - t0})
                            rows.append(row)
                            print(f"{graph_id} n={n} q={q} round={round_id} "
                                  f"{method} SHD={row['SHD_cpdag']} "
                                  f"runtime={row['candidate_runtime']:.2f}s", flush=True)
    pd.DataFrame(graph_audit).to_csv(out / "graph_audit.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "seed_ledger.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "raw" / "results.csv", index=False)
    summary = frame.groupby("method", dropna=False).agg(
        runs=("method", "size"), SHD_cpdag_mean=("SHD_cpdag", "mean"),
        SHD_cpdag_std=("SHD_cpdag", "std"), runtime_mean=("candidate_runtime", "mean"),
        violations_max=("violations_after", "max")).reset_index()
    summary.to_csv(out / "analysis" / "summary.csv", index=False)
    # Keep figure generation part of the executable chain.  Import locally so
    # the solver can still be used in minimal environments without pyplot.
    from scripts.notreks_protocol_analysis import analyze
    analyze(out)
    manifest = {
        "protocol_version": "pilot-20260917-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": args.master_seed, "fraction": args.fraction,
        "requested_experiments": [spec.name for spec in specs],
        "deferred_derived_experiments": deferred,
        "graph_cells": candidates, "graph_replicates": graph_reps,
        "n_values": n_values, "q_values": q_values,
        "methods": list(METHODS), "attempts": args.attempts,
        "flop_sweeps": args.flop_sweeps, "dagma_stages": args.dagma_stages,
        "platform": platform.platform(), "python": sys.version,
        "elapsed_seconds": time.time() - started,
        "known_scope": "bounded pilot; expands deterministically with fraction",
    }
    write_json(out / "manifest.json", manifest)
    print("\nPilot summary:")
    print(summary.to_string(index=False))
    print(f"Artifacts: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, default=.01)
    parser.add_argument("--experiments", nargs="+", default=["main"],
                        choices=[*REGISTRY, "all"],
                        help="protocol experiments; handlers are enabled incrementally")
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--flop-sweeps", type=int, default=16)
    parser.add_argument("--dagma-stages", type=int, default=5)
    parser.add_argument("--dagma-warm-iter", type=int, default=3000)
    parser.add_argument("--dagma-max-iter", type=int, default=6000)
    parser.add_argument("--trek-weight", type=float, default=1.0)
    parser.add_argument("--max-wall-hours", type=float, default=10.0,
                        help="hard wall-clock guard; partial artifacts are retained")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.list_experiments:
        for name, spec in REGISTRY.items():
            print(f"{name}: {spec.description}; rows at 100%%={__import__('scripts.notreks_protocol_registry', fromlist=['planned_solver_rows']).planned_solver_rows(spec, 1.0)}")
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --list-experiments is used")
    run(args)


if __name__ == "__main__":
    main()
