#!/usr/bin/env python3
"""Resumable dense d=50 compute-budget experiment using production kernels."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import flopsearch
import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.knowledge import (
    named_pairs_to_indices, write_oracle_sidecar,
)
from workflow.rules.structure_learning_algorithms.dagma.shared import (
    SharedDagmaLinear, deterministic_initial_adjacency,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.tools.local_smoke import (
    _edge_count, _metrics, _violations,
)
from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag
from workflow.rules.structure_learning_algorithms.flop_notreks.adapter import (
    selected_dag_from_diagnostics,
)
from workflow.rules.structure_learning_algorithms.pc_mi_oracle.pc import (
    MIOracleFisherZ, stable_pc,
)

ROOT = Path(__file__).resolve().parents[5]
TAG = "flop_notreks_dense_d50_budget"
OUT = ROOT / "results/dagma_notreks_oracle" / TAG
REPORT = OUT / "report"
SEEDS = range(5001, 5011)


def save_graph(A, names, path):
    pd.DataFrame(A, columns=names).to_csv(path, index=False)


def evaluate(seed, method, graph, true_path, names, runtime, extra=None):
    case = OUT / f"seed_{seed}"
    path = case / f"{method}_adjmat.csv"
    save_graph(graph, names, path)
    metrics = _metrics(true_path, path, case / f"{method}_metrics.csv")
    return {"seed": seed, "method": method, "runtime": runtime,
            "estimated_edges": _edge_count(graph), **metrics, **(extra or {})}


def objective(model, W, nt_weight):
    last = model.stage_diagnostics[-1]
    mu = last["mu"]
    return (mu * (model.score_final + model.lambda1 * np.abs(W).sum())
            + model.h_final + nt_weight * last["raw_notreks_value"]
            * (2 / (W.shape[0] - 1)))


def run_dagma(seed, X, pairs, true_path, names, notreks):
    method = "dagma_notreks_multistart" if notreks else "dagma_multistart"
    case = OUT / f"seed_{seed}"
    cached = case / f"{method}_row.json"
    if cached.exists():
        return json.loads(cached.read_text())
    kwargs = dict(lambda1=.03, w_threshold=0., T=5, mu_init=1.,
                  mu_factor=.1, s=[1., .9, .8, .7, .6], warm_iter=30000,
                  max_iter=60000, lr=.0003, checkpoint=1000,
                  beta_1=.99, beta_2=.999)
    best = None
    restart_rows = []
    started_all = time.perf_counter()
    for restart in range(5):
        model = SharedDagmaLinear("l2")
        initial = None if restart == 0 else deterministic_initial_adjacency(
            X.shape[1], seed * 1009 + restart, .05)
        started = time.perf_counter()
        W = model.fit(
            X.copy(), initial_W=initial, no_trek_pairs=pairs if notreks else (),
            trek_function="inv", trek_weight=10. if notreks else 0., **kwargs)
        runtime = time.perf_counter() - started
        obj = objective(model, W, 10. if notreks else 0.)
        item = {"seed": seed, "method": method, "restart_index": restart,
                "initialization_type": "zero" if restart == 0 else "random",
                "initial_spectral_radius": 0. if initial is None else float(
                    max(abs(np.linalg.eigvals(initial * initial)))),
                "runtime": runtime, "iterations": sum(
                    s["iterations_performed"] for s in model.stage_diagnostics),
                "final_objective": obj, "final_score": model.score_final,
                "final_h": model.h_final,
                "final_notreks": model.stage_diagnostics[-1]["raw_notreks_value"],
                "selected": False, "failure_reason": ""}
        restart_rows.append(item)
        if best is None or (obj, restart) < (best[0], best[1]):
            best = (obj, restart, W.copy(), model)
    best[3]
    restart_rows[best[1]]["selected"] = True
    pd.DataFrame(restart_rows).to_csv(case / f"{method}_restarts.csv", index=False)
    W = best[2]
    A = (np.abs(W) >= .2).astype(int)
    np.fill_diagonal(A, 0)
    row = evaluate(seed, method, A, true_path, names,
                   time.perf_counter() - started_all, {
                       "best_restart_index": best[1],
                       "best_objective": best[0],
                       "num_oracle_violations": _violations(A, pairs),
                       "violation_graph_representation": "thresholded selected DAG",
                       "restarts_requested": 5, "restarts_completed": 5})
    np.save(case / f"{method}_weighted.npy", W)
    cached.write_text(json.dumps(row, indent=2) + "\n")
    return row


def run(seed):
    case = OUT / f"seed_{seed}"
    case.mkdir(parents=True, exist_ok=True)
    data_path = ROOT / f"resources/data/mydatasets/{TAG}/s{seed}.csv"
    true_path = ROOT / f"resources/adjmat/myadjmats/{TAG}/g{seed}.csv"
    frame = pd.read_csv(data_path)
    names = list(frame.columns)
    X = frame.to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    true = pd.read_csv(true_path).to_numpy(int)
    sidecar = write_oracle_sidecar(true, names, case / "no_trek_pairs.json")
    pairs = named_pairs_to_indices(sidecar, names)
    constrained_targets = len({v for pair in pairs for v in pair})
    dataset = {
        "seed": seed, "graph_hash": hashlib.sha256(true_path.read_bytes()).hexdigest(),
        "num_true_edges": int(true.sum()), "num_oracle_mi_pairs": len(pairs),
        "oracle_mi_pair_fraction": len(pairs) / 1225,
        "num_distinct_constrained_targets": constrained_targets,
    }
    (case / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    rows = []
    specs = [
        ("flop_seeded_empty", [], 64, 1, 0, 0., 0, "alternating_full_refit_b"),
        ("fixed_signature_a", pairs, 64, 1, 0, 3., 6, "fixed_signature_a"),
        ("flop_notreks_b_medium", pairs, 64, 64, 16, 3., 6,
         "alternating_full_refit_b"),
        ("flop_notreks_b_large", pairs, 128, 96, 32, 3.5, 7,
         "alternating_full_refit_b"),
    ]
    for method, supplied, restarts, top, explore, mean, maximum, version in specs:
        cached = case / f"{method}_row.json"
        if cached.exists():
            rows.append(json.loads(cached.read_text()))
            continue
        started = time.perf_counter()
        raw, diag = flopsearch.flop_notreks(
            X, 2., supplied, restarts=restarts - 1, seed=99173 + seed,
            signature_top_k=top, signature_exploration_k=explore,
            max_signature_rounds=0 if method == "flop_seeded_empty" else
            (0 if version == "fixed_signature_a" else
             (500 if method.endswith("medium") else 1000)),
            initial_signature_mean_size=mean, initial_signature_max_size=maximum,
            search_version=version, return_diagnostics=True)
        runtime = time.perf_counter() - started
        cpdag = convert_flop_cpdag(raw, 50)
        dag = selected_dag_from_diagnostics(diag, 50)
        violations = _violations(dag, pairs)
        if supplied and violations:
            raise RuntimeError(f"{seed} {method}: {violations} hard-constraint violations")
        row = evaluate(seed, method, cpdag, true_path, names, runtime, {
            **diag, "num_oracle_violations": violations,
            "violation_graph_representation": "selected DAG", **dataset})
        cached.write_text(json.dumps(row, indent=2) + "\n")
        (case / f"{method}_diagnostics.json").write_text(
            json.dumps(diag, indent=2) + "\n")
        rows.append(row)
    cached = case / "flop_official_r64_row.json"
    if cached.exists():
        rows.append(json.loads(cached.read_text()))
    else:
        started = time.perf_counter()
        raw = flopsearch.flop(X, 2., restarts=64)
        row = evaluate(seed, "flop_official_r64", convert_flop_cpdag(raw, 50),
                       true_path, names, time.perf_counter() - started, {
                           **dataset, "num_oracle_violations": None,
                           "violation_graph_representation": "unavailable"})
        cached.write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
    cached = case / "pc_mi_oracle_row.json"
    if cached.exists():
        rows.append(json.loads(cached.read_text()))
    else:
        dispatch = MIOracleFisherZ(X.copy(), pairs, .05)
        started = time.perf_counter()
        graph, counts = stable_pc(X.copy(), dispatch, 48)
        row = evaluate(seed, "pc_mi_oracle", graph, true_path, names,
                       time.perf_counter() - started, {
                           **dataset, **counts.__dict__,
                           "num_oracle_violations": None,
                           "violation_graph_representation": "unavailable (CPDAG)"})
        cached.write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
    rows += [run_dagma(seed, X, pairs, true_path, names, False),
             run_dagma(seed, X, pairs, true_path, names, True)]
    return rows


def summarize(rows):
    REPORT.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    datasets = pd.DataFrame([json.loads(path.read_text())
                             for path in sorted(OUT.glob("seed_*/dataset.json"))])
    for column in datasets.columns:
        if column != "seed":
            mapped = raw["seed"].map(datasets.set_index("seed")[column])
            raw[column] = raw[column].fillna(mapped) if column in raw else mapped
    raw.to_csv(REPORT / "per_run.csv", index=False)
    numeric = ["SHD_pattern", "SHD_cpdag", "F1_skel", "F1_pattern",
               "estimated_edges", "runtime", "num_oracle_mi_pairs",
               "num_oracle_violations", "oracle_mi_pair_fraction"]
    agg = raw.groupby("method").agg(**{
        f"mean_{c}": (c, "mean") for c in numeric
    }, sample_sd_SHD_pattern=("SHD_pattern", "std"),
       median_SHD_pattern=("SHD_pattern", "median"),
       successful_datasets=("seed", "count")).reset_index()
    agg.to_csv(REPORT / "algorithm_summary.csv", index=False)
    datasets.to_csv(REPORT / "dataset_summary.csv", index=False)
    datasets.to_csv(REPORT / "oracle_pair_summary.csv", index=False)
    restarts = []
    for path in OUT.glob("seed_*/*_restarts.csv"):
        restarts.append(pd.read_csv(path))
    pd.concat(restarts, ignore_index=True).to_csv(
        REPORT / "restart_diagnostics.csv", index=False) if restarts else None
    trajectories = raw[raw.method.str.contains("flop_notreks")][[
        c for c in ["seed", "method", "best_restart_index", "final_bic",
                    "number_of_signature_rounds",
                    "number_of_accepted_promotions",
                    "number_of_post_promotion_order_blocks",
                    "number_of_canonical_compressions",
                    "fraction_of_candidate_parent_relations_pruned",
                    "termination_reason"] if c in raw]]
    trajectories.to_csv(REPORT / "selected_restart_trajectories.csv", index=False)
    medium = raw[raw.method == "flop_notreks_b_medium"]
    large = raw[raw.method == "flop_notreks_b_large"]
    ml = medium.merge(large, on="seed", suffixes=("_medium", "_large"))
    if len(ml):
        ml["delta_SHD_pattern"] = ml.SHD_pattern_large - ml.SHD_pattern_medium
        ml["delta_SHD_cpdag"] = ml.SHD_cpdag_large - ml.SHD_cpdag_medium
        ml["runtime_ratio"] = ml.runtime_large / ml.runtime_medium
        ml["delta_edge_count"] = ml.estimated_edges_large - ml.estimated_edges_medium
    ml.to_csv(REPORT / "version_b_medium_vs_large.csv", index=False)
    comparisons = []
    rng = np.random.default_rng(20260725)
    for first, second in [
        ("flop_notreks_b_medium", "flop_notreks_b_large"),
        ("flop_seeded_empty", "flop_notreks_b_large"),
        ("fixed_signature_a", "flop_notreks_b_large"),
        ("flop_notreks_b_large", "dagma_notreks_multistart")]:
        paired = raw[raw.method == first].merge(
            raw[raw.method == second], on="seed", suffixes=("_first", "_second"))
        delta = (paired.SHD_pattern_second - paired.SHD_pattern_first).to_numpy()
        boot = np.array([rng.choice(delta, len(delta)).mean()
                         for _ in range(10000)]) if len(delta) else np.array([np.nan])
        comparisons.append({
            "first": first, "second": second, "n": len(delta),
            "mean_delta_SHD_pattern": np.mean(delta),
            "bootstrap_95_low": np.nanpercentile(boot, 2.5),
            "bootstrap_95_high": np.nanpercentile(boot, 97.5),
            "wins_second": int(np.sum(delta < 0)), "ties": int(np.sum(delta == 0)),
            "losses_second": int(np.sum(delta > 0)),
            "mean_delta_F1_pattern": (
                paired.F1_pattern_second - paired.F1_pattern_first).mean(),
            "mean_delta_edges": (
                paired.estimated_edges_second - paired.estimated_edges_first).mean(),
            "mean_delta_violations": (
                paired.num_oracle_violations_second -
                paired.num_oracle_violations_first).mean(),
            "runtime_ratio_second_over_first": (
                paired.runtime_second / paired.runtime_first).mean(),
        })
    pd.DataFrame(comparisons).to_csv(
        REPORT / "paired_method_comparisons.csv", index=False)
    threshold_rows = []
    for seed in datasets.seed:
        true_path = ROOT / f"resources/adjmat/myadjmats/{TAG}/g{seed}.csv"
        names = list(pd.read_csv(true_path).columns)
        pairs = named_pairs_to_indices(json.loads(
            (OUT / f"seed_{seed}/no_trek_pairs.json").read_text()), names)
        for method in ("dagma_multistart", "dagma_notreks_multistart"):
            W = np.load(OUT / f"seed_{seed}/{method}_weighted.npy")
            for threshold in (.1, .2, .3):
                A = (np.abs(W) >= threshold).astype(int)
                np.fill_diagonal(A, 0)
                path = OUT / f"seed_{seed}/{method}_threshold_{threshold}.csv"
                save_graph(A, names, path)
                metric = _metrics(true_path, path, OUT / f"seed_{seed}/"
                                  f"{method}_threshold_{threshold}_metrics.csv")
                threshold_rows.append({
                    "seed": seed, "method": method, "threshold": threshold,
                    "estimated_edges": _edge_count(A),
                    "num_oracle_violations": _violations(A, pairs), **metric})
    pd.DataFrame(threshold_rows).to_csv(
        REPORT / "threshold_diagnostics.csv", index=False)
    (REPORT / "report.md").write_text(
        "# Dense d=50 FLOP-NOTREKS compute-budget diagnostic\n\n"
        + "```csv\n" + agg.to_csv(index=False) + "```\n")
    (REPORT / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "flopsearch": "0.3.0 research fork",
    }, indent=2) + "\n")
    (REPORT / "command_log.txt").write_text(
        "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 "
        "PYTHONPATH=. .venv-local-smoke/bin/python "
        "workflow/rules/structure_learning_algorithms/flop_notreks/tools/"
        "dense_budget.py --seeds 5001 5002 5003 5004 5005 5006 5007 5008 5009 5010\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        rows.extend(run(seed))
        existing = []
        for path in OUT.glob("seed_*/*_row.json"):
            existing.append(json.loads(path.read_text()))
        summarize(existing)
        print(f"completed seed {seed}", flush=True)


if __name__ == "__main__":
    main()
