#!/usr/bin/env python3
"""Run the registered NOTREKS protocol experiments with shared artifacts."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.notreks_protocol import (
    derive_seed, graph_pairs, make_data, make_graph, select_pairs, write_json,
)
from scripts.notreks_protocol_registry import REGISTRY, select_registry, scaled_replicates
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import (
    flop_notreks_candidate, gaussian_bic, metrics,
    order_parent_postselection_from_candidate,
)
from scripts.notreks_benchmark_pipeline import method_run


def _job_key(row):
    strategy = row.get("knowledge_strategy", "random")
    if not isinstance(strategy, str) or not strategy:
        strategy = "random"
    return (row.get("experiment_id"), row.get("data_id"),
            float(row.get("knowledge_fraction")),
            int(row.get("knowledge_round")),
            strategy, row.get("method"))


def _replicate_count(spec, args):
    if args.graph_replicates is not None:
        return args.graph_replicates
    return scaled_replicates(spec, args.fraction)


def _knowledge_round_count(spec, q, args):
    # q=0 is the single vanilla job and q=1 is the complete set; only q=.25
    # has multiple independent subset draws.
    if q in {0.0, 1.0}:
        return 1
    if q == .25 and args.knowledge_rounds is not None:
        return args.knowledge_rounds
    return spec.q25_rounds if q == .25 else 1


def _knowledge_strategy_list(spec, q):
    # Structural subset constructions are part of the prior-structure study.
    # q=0 and q=1 have deterministic knowledge sets.
    return spec.knowledge_strategies if q == .25 else ("random",)


def _attempts_for(spec, method, args):
    return int(args.attempts) if args.attempts is not None else spec.attempts_for(method)


def _methods_for_job(spec, d, q):
    """Return methods that are meaningful for this knowledge level.

    SortnRegress does not consume NOTREKS information, so it is run once at
    q=1 as a knowledge-independent baseline rather than duplicated for every
    prior subset.
    """
    methods = spec.methods_for(d)
    vanilla = {"flop", "dagma", "var_sortnregress", "r2_sortnregress"}
    if q == 0.0:
        return tuple(m for m in methods if m in vanilla)
    return tuple(m for m in methods if m not in vanilla)


def print_design_and_objective(specs, args):
    print("\nProtocol design:", flush=True)
    print(f"  master seed: {args.master_seed} (all graph/data/knowledge/method "
          "seeds are deterministic derivations)", flush=True)
    print(f"  graph seeds per cell: "
          f"{args.graph_replicates if args.graph_replicates is not None else 'registry × fraction'}",
          flush=True)
    if args.attempts is None:
        print("  solver restarts: FLOP-family=20; DAGMA-family=2 (registry defaults)",
              flush=True)
    else:
        print(f"  solver restarts override (--attempts): {args.attempts}", flush=True)
    print(f"  independent NOTREKS-set rounds at q=.25: "
          f"{args.knowledge_rounds if args.knowledge_rounds is not None else 'registry default 5'}; "
          "q=0 is vanilla; q=1 uses one deterministic complete set",
          flush=True)
    print(f"  workers: {args.workers}; FLOP sweeps: {args.flop_sweeps}; "
          f"DAGMA stages: {args.dagma_stages}; DAGMA warm/max iterations: "
          f"{args.dagma_warm_iter}/{args.dagma_max_iter}", flush=True)
    print("  cells:", flush=True)
    for spec in specs:
        if spec.derives_from is not None:
            continue
        reps = (_replicate_count(spec, args))
        parts = []
        for q in spec.q_values:
            parts.append(f"q={q:g}: {_knowledge_round_count(spec, q, args)} set(s)")
        rows = sum(
            reps * len(spec.n_values)
            * sum(_knowledge_round_count(spec, q, args)
                  * len(_knowledge_strategy_list(spec, q)) for q in spec.q_values)
            * len(spec.methods_for(d))
            for d, _, _ in spec.cells
        )
        print(f"    {spec.name}: {len(spec.cells)} graph cells × {reps} graph seeds × "
              f"n={spec.n_values} × {', '.join(parts)} × "
              f"dimension-specific methods = {rows} solver rows", flush=True)

    print("\nDAGMA objective (standard square-map production path):", flush=True)
    print("  J_{mu,s}(W) = mu * [Q_n(W) + lambda_1 ||W||_1] "
          "+ h_DAG,s(W) + w_NT * R_I,s(W)", flush=True)
    print("  Q_n(W) = 0.5 * tr[(I-W)^T Sigma_hat (I-W)] on standardized data",
          flush=True)
    print("  lambda_1 = 0.03; DAG penalty weight = 1; "
          "w_NT = args.trek_weight = " + str(args.trek_weight), flush=True)
    print("  h_DAG,s(W) = -log det(s I - W∘W) + d log s", flush=True)
    print("  R_I,s(W) = [2/(d-1)] * sum_{(i,j) in I} "
          "[(s I - W∘W)^(-T)(s I - W∘W)^(-1)]_{ij}", flush=True)
    print("  The factor 2/(d-1) is the kernel's outer-sum scaling; "
          "there is no division by |I|.", flush=True)
    print("  mu schedule = (1, 0.3, 0.1, 0.01, 0.001); "
          "s schedule = (1.1, 1.0, 0.9, 0.8, 0.7)", flush=True)
    print("  NOTREKS is absent when the method has no supplied pairs; for "
          "DAGMA+NOTREKS, the displayed R_I,s term is active.", flush=True)


def _checkpoint(rows, path):
    """Atomically persist the current solver ledger/results checkpoint."""
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(
            ["experiment_id", "data_id", "knowledge_fraction",
             "knowledge_round", "knowledge_strategy", "method"], keep="last")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def dispatch(name, x, pairs, seed, args, attempts):
    if name in {"var_sortnregress", "r2_sortnregress"}:
        # These baselines are implemented in the benchmark toolchain, but
        # are not part of the legacy method_run dispatcher.  Wire them here
        # explicitly so the protocol cannot silently fall through to the
        # legacy unknown-method error.
        from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.sortnregress import (
            sortnregress,
        )
        started = time.perf_counter()
        kind = "variance" if name == "var_sortnregress" else "r2"
        candidate, diag = sortnregress(x, kind=kind)
        return candidate, diag, time.perf_counter() - started
    if name == "flop_notreks":
        start = time.perf_counter()
        candidate, diag = flop_notreks_candidate(
            x, pairs, seed, attempts, args.flop_sweeps,
            search_version="flop_like", local_greedy_passes=8)
        return candidate, diag, time.perf_counter() - start
    aliases = {"dagma_notreks": "dagma-pstrek", "dagma_notreks_tcc": "dagma_notreks_tcc",
               "dagma": "dagma",
               "flop": "flop", "flop-nt-edge-mask": "flop-nt-edge-mask",
               "flop-nt-post": "flop-nt-post", "dagma-nt-edge-mask": "dagma-nt-edge-mask",
               "dagma-nt-post": "dagma-nt-post",
               "var_sortnregress": "var_sortnregress",
               "r2_sortnregress": "r2_sortnregress"}
    return method_run(aliases[name], x, pairs, seed, attempts,
                      args.flop_sweeps, args.dagma_stages,
                      args.dagma_warm_iter, args.dagma_max_iter,
                      args.trek_weight)


def prior_properties(d, pairs, candidate=None):
    edges = {tuple(sorted(map(int, pair))) for pair in pairs}
    degree = np.zeros(d, dtype=int)
    adj = [set() for _ in range(d)]
    for a, b in edges:
        degree[a] += 1; degree[b] += 1; adj[a].add(b); adj[b].add(a)
    seen, components = set(), []
    for start in range(d):
        if start in seen: continue
        stack, comp = [start], []
        while stack:
            node = stack.pop()
            if node in seen: continue
            seen.add(node); comp.append(node); stack.extend(adj[node] - seen)
        components.append(comp)
    h = np.zeros((d, d), dtype=bool)
    for a, b in edges: h[a, b] = h[b, a] = True
    clique = 1 if d else 0
    for a in range(d):
        clique = max(clique, 1 + sum(all(h[b, c] for c in range(d) if c != b and c in adj[a])
                                     for b in adj[a]))
    alignment = np.nan
    if candidate is not None and edges:
        reach = np.asarray(candidate, dtype=bool).copy()
        for k in range(d): reach |= reach[:, [k]] & reach[[k], :]
        bad = sum(bool(reach[:, a] @ reach[:, b]) for a, b in edges)
        alignment = bad / len(edges)
    return {"prior_edge_density": len(edges) / max(1, d * (d - 1) / 2),
            "prior_degree_mean": float(degree.mean()),
            "prior_degree_std": float(degree.std()),
            "prior_degree_max": int(degree.max(initial=0)),
            "prior_isolated_nodes": int(np.sum(degree == 0)),
            "prior_endpoint_coverage": int(np.sum(degree > 0)) / max(1, d),
            "prior_connected_components": len(components),
            "prior_largest_component": max(map(len, components), default=0),
            "maximum_clique_size": clique, "chromatic_number_exact": np.nan,
            "trek_error_alignment": alignment}


def run(args):
    specs = select_registry(args.experiments, args.fraction)
    executable = [s for s in specs if s.derives_from is None
                  and s.dataset == "synthetic_linear_gaussian"]
    external = [s for s in specs if s.derives_from is None
                and s.dataset != "synthetic_linear_gaussian"]
    if external:
        print("Skipping non-synthetic registry datasets in this runner: "
              + ", ".join(s.name for s in external)
              + ". Use scripts/causalassembly_benchmark.py for causalAssembly.",
              flush=True)
    out = args.output_root.resolve()
    for part in ("graph_bank", "data", "knowledge", "raw", "analysis"):
        (out / part).mkdir(parents=True, exist_ok=True)
    started = time.time()
    result_path = out / "raw" / "results.csv"
    if result_path.exists():
        rows = pd.read_csv(result_path).to_dict("records")
        for row in rows:
            row.setdefault("knowledge_strategy", "random")
            if not isinstance(row.get("knowledge_strategy"), str):
                row["knowledge_strategy"] = "random"
        print(f"resuming {len(rows)} checkpointed solver rows from {result_path}",
              flush=True)
    else:
        rows = []
    successful = {
        _job_key(row) for row in rows if row.get("solver_status") == "ok"
    }
    ledger, audits = [], []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    for spec in executable:
        for cell_index, (d, family, density) in enumerate(spec.cells):
            reps = _replicate_count(spec, args)
            for replicate in range(reps):
                # Some dense/Watts--Strogatz DAGs legitimately have no no-trek
                # pairs: every node pair shares an ancestor.  Keep those
                # cells in the registry with an empty information set rather
                # than changing the graph distribution or aborting the run.
                seed_attempt = 0
                graph_seed = derive_seed("graph", spec.name, d, family,
                                         density, replicate, seed_attempt,
                                         master=args.master_seed)
                truth = make_graph(d, family, density, graph_seed)
                all_pairs = graph_pairs(truth)
                graph_id = f"{spec.name}_g{cell_index:02d}_{family}{density}_r{replicate:02d}"
                np.savez_compressed(out / "graph_bank" / f"{graph_id}.npz",
                                    adjacency=truth, graph_seed=graph_seed,
                                    no_trek_pairs=np.asarray(all_pairs, dtype=int))
                audits.append({"experiment": spec.name, "graph_id": graph_id,
                               "d": d, "family": family, "density": density,
                               "replicate": replicate, "graph_seed": graph_seed,
                               "seed_attempt": seed_attempt,
                               "edges": int(truth.sum()),
                               "no_trek_pairs": len(all_pairs),
                               "eligible_information": bool(all_pairs)})
                for n in spec.n_values:
                    data_seed = derive_seed("data", graph_id, n, master=args.master_seed)
                    x, weights = make_data(truth, n, graph_seed, data_seed)
                    data_id = f"{graph_id}_n{n}"
                    np.savez_compressed(out / "data" / f"{data_id}.npz", X=x,
                                        truth=truth, weights=weights, data_seed=data_seed)
                    cache = {}
                    for q in spec.q_values:
                        rounds = _knowledge_round_count(spec, q, args)
                        for strategy in _knowledge_strategy_list(spec, q):
                          for round_id in range(rounds):
                            knowledge_seed = derive_seed("knowledge", graph_id, n,
                                                         q, strategy, round_id,
                                                         master=args.master_seed)
                            pairs = select_pairs(all_pairs, q, knowledge_seed, strategy)
                            methods_for_job = _methods_for_job(spec, d, q)
                            if args.methods:
                                methods_for_job = tuple(
                                    method for method in methods_for_job
                                    if method in args.methods)
                            prior_id = f"{data_id}_q{q:g}_{strategy}_r{round_id}"
                            write_json(out / "knowledge" / f"{prior_id}.json", {
                                "experiment": spec.name, "graph_id": graph_id,
                                "data_id": data_id, "q": q, "round": round_id,
                                "knowledge_strategy": strategy,
                                "knowledge_seed": knowledge_seed,
                                "pairs": [list(p) for p in pairs]})
                            expected = {
                                (spec.name, data_id, float(q), round_id, strategy, method)
                                for method in methods_for_job
                            }
                            if expected <= successful:
                                continue
                            pending = {}
                            cached = {}
                            methods_to_run = []
                            for method in methods_for_job:
                                if (spec.name, data_id, float(q), round_id, strategy, method) in successful:
                                    continue
                                if time.time() >= started + args.max_wall_hours * 3600:
                                    raise TimeoutError("protocol all-experiment wall-clock guard reached")
                                method_seed = derive_seed("method", spec.name, data_id,
                                                          q, strategy, round_id, method,
                                                          master=args.master_seed)
                                ledger.append({"experiment": spec.name, "graph_id": graph_id,
                                               "data_id": data_id, "prior_id": prior_id,
                                               "method": method, "graph_seed": graph_seed,
                                               "data_seed": data_seed,
                                               "knowledge_seed": knowledge_seed,
                                               "knowledge_strategy": strategy,
                                               "method_seed": method_seed,
                                               "attempts": _attempts_for(spec, method, args)})
                                methods_to_run.append(method)
                                key = (data_id, method)
                                base_method = "flop" if method.startswith("flop") else "dagma"
                                if method in {"flop-nt-post", "dagma-nt-post"} and (data_id, base_method) in cache:
                                    base_candidate, base_diag, base_runtime = cache[(data_id, base_method)]
                                    started_post = time.perf_counter()
                                    post_candidate, post_diag = order_parent_postselection_from_candidate(
                                        x, base_candidate, pairs)
                                    cached[method] = (
                                        post_candidate,
                                        {**post_diag,
                                         "candidate_graph": base_candidate.copy(),
                                         "optimizer_restarts": base_diag.get("optimizer_restarts", _attempts_for(spec, method, args)),
                                         "base_solver_runtime": base_runtime},
                                        time.perf_counter() - started_post)
                                elif method in {"flop", "dagma"} and key in cache:
                                    cached[method] = cache[key]
                                else:
                                    pending[method] = pool.submit(
                                        dispatch, method, x, pairs, method_seed, args,
                                        _attempts_for(spec, method, args))
                            results = dict(cached)
                            for method, future in pending.items():
                                try:
                                    result = future.result()
                                    results[method] = result
                                    if method in {"flop", "dagma"}:
                                        cache[(data_id, method)] = result
                                except Exception as exc:  # preserve the rest of the run
                                    results[method] = exc
                            for method in methods_to_run:
                                result = results.get(method)
                                method_seed = derive_seed("method", spec.name, data_id,
                                                          q, strategy, round_id, method,
                                                          master=args.master_seed)
                                if isinstance(result, Exception):
                                    row = {"protocol_version": "20260917-v2",
                                           "experiment_id": spec.name,
                                           "graph_id": graph_id, "data_id": data_id,
                                           "instance_id": data_id, "prior_id": prior_id,
                                           "d": d, "n": n, "er_degree": density,
                                           "graph_family": family, "graph_density": density,
                                           "graph_replicate": replicate, "q": q,
                                           "knowledge_fraction": q, "knowledge_round": round_id,
                                           "graph_seed": graph_seed, "data_seed": data_seed,
                                           "knowledge_seed": knowledge_seed, "method_seed": method_seed,
                                           "knowledge_strategy": strategy,
                                           "method": method, "solver_status": "failed",
                                           "solver_error": repr(result),
                                           "knowledge_pairs": len(pairs),
                                           "data_model": "unequal_variance_linear_gaussian_scm",
                                           "discovery_score": "profiled_unequal_variance_gaussian",
                                           "candidate_runtime": np.nan, "SHD_cpdag": np.nan,
                                           "F1_skel": np.nan, "violations_after": np.nan}
                                    print(f"{spec.name} {graph_id} n={n} q={q} "
                                          f"{method} FAILED: {result!r}", flush=True)
                                else:
                                    candidate, diag, runtime = result
                                    row = metrics(x, truth, pairs, method,
                                                  diag["candidate_graph"], candidate, diag["cpdag"],
                                                  runtime, _attempts_for(spec, method, args),
                                                  diag.get("optimizer_restarts",
                                                           _attempts_for(spec, method, args)))
                                    row.update({"protocol_version": "20260917-v2",
                                                "experiment_id": spec.name,
                                                "graph_id": graph_id, "data_id": data_id,
                                                "instance_id": data_id, "prior_id": prior_id,
                                                "d": d, "n": n, "er_degree": density,
                                                "graph_family": family, "graph_density": density,
                                                "graph_replicate": replicate, "q": q,
                                                "knowledge_fraction": q, "knowledge_round": round_id,
                                                "graph_seed": graph_seed, "data_seed": data_seed,
                                                "knowledge_seed": knowledge_seed, "method_seed": method_seed,
                                                "knowledge_strategy": strategy,
                                                "solver_status": "ok", "knowledge_pairs": len(pairs),
                                                "data_model": "unequal_variance_linear_gaussian_scm",
                                                "discovery_score": "profiled_unequal_variance_gaussian"})
                                    if spec.name == "prior-structure":
                                        row.update(prior_properties(d, pairs, diag["candidate_graph"]))
                                    successful.add(_job_key(row))
                                    print(f"{spec.name} {graph_id} n={n} q={q} "
                                          f"{method} SHD={row['SHD_cpdag']}", flush=True)
                                rows.append(row)
                                _checkpoint(rows, result_path)
                          # end round
                        # end strategy
    frame = pd.DataFrame(rows).drop_duplicates(
        ["experiment_id", "data_id", "knowledge_fraction",
         "knowledge_round", "knowledge_strategy", "method"], keep="last")
    # Compute the truth-support reference after resumption as well, so old
    # checkpoints gain the same BIC-gap fields as newly completed jobs.
    if not frame.empty:
        truth_bic = {}
        for data_id in frame.data_id.dropna().unique():
            data_path = out / "data" / f"{data_id}.npz"
            if data_path.exists():
                artifact = np.load(data_path)
                truth_bic[data_id] = float(
                    gaussian_bic(artifact["X"], artifact["truth"], lambda_bic=2.0)[0])
        frame["truth_bic"] = frame.data_id.map(truth_bic)
        frame["bic_gap_to_truth"] = frame["final_bic"] - frame["truth_bic"]
        frame.to_csv(result_path, index=False)
    pool.shutdown(wait=True)
    frame.to_csv(out / "raw" / "results.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "seed_ledger.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "graph_audit.csv", index=False)
    if frame.empty:
        summary = pd.DataFrame(columns=[
            "experiment_id", "method", "runs", "SHD_cpdag_mean",
            "SHD_cpdag_std", "runtime_mean", "violations_max"])
    else:
        summary = frame.groupby(["experiment_id", "method"], as_index=False).agg(
            runs=("method", "size"), SHD_cpdag_mean=("SHD_cpdag", "mean"),
            SHD_cpdag_std=("SHD_cpdag", "std"), runtime_mean=("candidate_runtime", "mean"),
            violations_max=("violations_after", "max"))
    summary.to_csv(out / "analysis" / "summary.csv", index=False)
    if not args.skip_figures:
        from scripts.notreks_production_figures import write_protocol_figures
        write_protocol_figures(frame, out / "analysis" / "figures", out)
    write_json(out / "manifest.json", {"protocol_version": "20260917-v2",
        "master_seed": args.master_seed, "fraction": args.fraction,
        "experiments": [s.name for s in specs],
        "planned_solver_rows": sum(len(s.cells) * _replicate_count(s, args) *
                                    len(s.n_values) * sum(_knowledge_round_count(s, q, args) for q in s.q_values) *
                                    sum(len(s.methods_for(d)) for d, _, _ in s.cells) / max(1, len(s.cells))
                                    * sum(len(_knowledge_strategy_list(s, q)) * _knowledge_round_count(s, q, args)
                                          for q in s.q_values)
                                    for s in executable),
        "derived_experiments": [s.name for s in specs if s.derives_from],
        "attempts_override": args.attempts,
        "attempts_by_family": {family: attempts for family, attempts in
                                {family: spec.attempts_for(family)
                                 for spec in specs for family, _ in spec.attempts_by_family}.items()},
        "workers": args.workers,
        "graph_replicates_override": args.graph_replicates,
        "knowledge_rounds_override": args.knowledge_rounds,
        "elapsed_seconds": time.time() - started})
    print("\nSummary:\n" + summary.to_string(index=False))
    print(f"Artifacts: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+", choices=[*REGISTRY, "all"], default=["all"])
    p.add_argument("--methods", nargs="+", default=None,
                   help="run only these registered method identifiers")
    p.add_argument("--fraction", type=float, default=.1)
    p.add_argument("--master-seed", type=int, default=20260917)
    p.add_argument("--graph-replicates", type=int, default=None,
                   help="explicit graph seeds per registry cell; overrides --fraction")
    p.add_argument("--knowledge-rounds", type=int, default=None,
                   help="explicit independent NOTREKS subset draws at q=.25; q=1 remains one full set")
    p.add_argument("--attempts", "--restarts", dest="attempts", type=int,
                   default=None,
                   help="override registry solver restarts for every method")
    p.add_argument("--flop-sweeps", type=int, default=16)
    p.add_argument("--dagma-stages", type=int, default=5)
    p.add_argument("--dagma-warm-iter", type=int, default=3000)
    p.add_argument("--dagma-max-iter", type=int, default=6000)
    p.add_argument("--trek-weight", type=float, default=1.0)
    p.add_argument("--max-wall-hours", type=float, default=8.0)
    p.add_argument("--workers", type=int, default=1,
                   help="number of concurrent solver jobs per data/prior cell")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--skip-figures", action="store_true",
                   help="do not generate publication figures; use the separate analysis step")
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be at least 1")
    if args.attempts is not None and args.attempts < 1:
        p.error("--attempts/--restarts must be at least 1")
    if args.graph_replicates is not None and args.graph_replicates < 1:
        p.error("--graph-replicates must be at least 1")
    if args.knowledge_rounds is not None and args.knowledge_rounds < 1:
        p.error("--knowledge-rounds must be at least 1")
    specs = select_registry(args.experiments, args.fraction)
    print_design_and_objective(specs, args)
    run(args)


if __name__ == "__main__": main()
