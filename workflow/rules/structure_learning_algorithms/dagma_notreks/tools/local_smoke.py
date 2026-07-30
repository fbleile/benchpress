#!/usr/bin/env python3
"""Container-free smoke runner using production algorithms and Benchpress R metrics."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.knowledge import (
    load_sidecar, named_pairs_to_indices, write_oracle_sidecar,
)
from workflow.rules.structure_learning_algorithms.dagma.shared import (
    SharedDagmaLinear, notreks_value_grad,
)
from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag
from workflow.rules.structure_learning_algorithms.flop_notreks.adapter import (
    count_no_trek_violations, selected_dag_from_diagnostics,
)
from workflow.rules.structure_learning_algorithms.pc_mi_oracle.pc import (
    MIOracleFisherZ, stable_pc,
)


ROOT = Path(__file__).resolve().parents[5]
R_EVAL = ROOT / "workflow/rules/evaluation/benchmarks/run_summarise.R"


def _metrics(true_path, estimated_path, output_path):
    subprocess.run([
        "Rscript", str(R_EVAL), "--adjmat_true", str(true_path),
        "--adjmat_est", str(estimated_path), "--filename", str(output_path),
    ], check=True, capture_output=True, text=True)
    row = pd.read_csv(output_path).iloc[0].to_dict()
    for suffix in ("skel", "pattern"):
        tp, fp, fn = (float(row[f"TP_{suffix}"]), float(row[f"FP_{suffix}"]),
                      float(row[f"FN_{suffix}"]))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        row[f"precision_{suffix}"] = precision
        row[f"recall_{suffix}"] = recall
        row[f"F1_{suffix}"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    row["SHD_skel"] = float(row["FP_skel"]) + float(row["FN_skel"])
    return row


def _reachability(A):
    reach = np.asarray(A, dtype=bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(len(A)):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def _violations(A, pairs):
    reach = _reachability(A)
    return sum(bool(np.any(reach[:, i] & reach[:, j])) for i, j in pairs)


def _edge_count(A):
    skeleton = np.asarray(A, dtype=bool) | np.asarray(A, dtype=bool).T
    return int(np.triu(skeleton, 1).sum())


def _write_estimate(A, names, directory, algorithm_id):
    path = directory / f"{algorithm_id}_adjmat.csv"
    pd.DataFrame(A, columns=names).to_csv(path, index=False)
    return path


def _flop_worker(data_path, output_path, lambda_bic, restarts):
    import flopsearch
    X = pd.read_csv(data_path).to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0, ddof=0)
    raw = flopsearch.flop(X, lambda_bic, restarts=restarts)
    np.save(output_path, np.asarray(raw))


def _flop_notreks_worker(data_path, sidecar_path, output_path, diagnostics_path,
                          lambda_bic, restarts, seed, top_k, rounds):
    import flopsearch
    frame = pd.read_csv(data_path)
    X = frame.to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0, ddof=0)
    payload = load_sidecar(sidecar_path, list(frame.columns))
    pairs = named_pairs_to_indices(payload, list(frame.columns))
    raw, diagnostics = flopsearch.flop_notreks(
        X, lambda_bic, pairs, restarts=restarts, seed=seed,
        signature_top_k=top_k, max_signature_rounds=rounds,
        search_version="alternating_full_refit_b",
        return_diagnostics=True)
    np.save(output_path, np.asarray(raw))
    Path(diagnostics_path).write_text(json.dumps(diagnostics, indent=2) + "\n")


def _base_row(seed, family, algorithm_id, runtime, metrics):
    return {
        "seed": seed, "data": f"s{seed}", "algorithm": family, "id": algorithm_id,
        "runtime": runtime, "time": runtime, "successful": True, **metrics,
    }


def run(tag):
    out = ROOT / f"results/dagma_notreks_oracle/{tag}"
    joint = ROOT / f"results/output/{tag}/benchmarks/dagma_notreks_oracle/{tag}/joint_benchmarks.csv"
    config_path = ROOT / f"configs/dagma_notreks_oracle/expanded/{tag}_config.json"
    grid_path = ROOT / f"configs/dagma_notreks_oracle/grids/{tag}.json"
    config, grid = json.loads(config_path.read_text()), json.loads(grid_path.read_text())
    algorithms = config["resources"]["structure_learning_algorithms"]
    out.mkdir(parents=True, exist_ok=True)
    joint.parent.mkdir(parents=True, exist_ok=True)
    input_tag = grid.get("reuse_inputs_from", tag)
    shared_true_path = ROOT / f"resources/adjmat/myadjmats/g{input_tag}.csv"
    rows, equality, stages, threshold_rows, dataset_metadata = [], [], [], [], []
    flop_status = []

    print(json.dumps({
        "number_of_datasets": len(grid["data"]["seeds"]), **grid["data"],
        "dagma": algorithms["dagma"][0],
        "trek_weights": [x["trek_weight"] for x in algorithms["dagma_notreks"]],
        "trek_functions": sorted({x["trek_function"] for x in algorithms["dagma_notreks"]}),
        "pc": algorithms["pc_mi_oracle"][0], "flop": algorithms["flop"][0],
        "flop_notreks": algorithms.get("flop_notreks", [None])[0],
    }, indent=2))

    for seed in grid["data"]["seeds"]:
        true_path = (
            ROOT / f"resources/adjmat/myadjmats/{input_tag}/g{seed}.csv"
            if grid["data"].get("independent_instances", False)
            else shared_true_path
        )
        true = pd.read_csv(true_path).to_numpy(dtype=int)
        dataset_dir = out / f"seed_{seed}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        data_path = ROOT / f"resources/data/mydatasets/{input_tag}/s{seed}.csv"
        frame = pd.read_csv(data_path)
        names, X = list(frame.columns), frame.to_numpy(dtype=float)
        sidecar_path = dataset_dir / "no_trek_pairs.json"
        payload = write_oracle_sidecar(true, names, sidecar_path)
        pairs = named_pairs_to_indices(payload, names)
        dataset_metadata.append({
            "seed": int(seed),
            "graph_hash": hashlib.sha256(true_path.read_bytes()).hexdigest(),
            "number_of_true_edges": int(np.sum(true)),
            "number_of_no_trek_pairs": len(pairs),
            "fraction_pairs_constrained": len(pairs) /
                (len(true) * (len(true) - 1) / 2),
        })

        ordinary = algorithms["dagma"][0].copy()
        for key in ("id", "timeout", "loss_type"):
            ordinary.pop(key, None)
        ordinary["s"] = [float(x) for x in ordinary["s"].split(",")]
        threshold = float(ordinary.pop("w_threshold"))

        vanilla_model = SharedDagmaLinear("l2")
        vanilla_model.profile_components = True
        start = time.perf_counter()
        vanilla_W = vanilla_model.fit(X.copy(), w_threshold=0.0, **ordinary)
        vanilla_runtime = time.perf_counter() - start
        vanilla_A = (np.abs(vanilla_W) >= threshold).astype(int)
        np.fill_diagonal(vanilla_A, 0)
        vanilla_path = _write_estimate(vanilla_A, names, dataset_dir, "dagma")
        metric_path = dataset_dir / "dagma_metrics.csv"
        row = _base_row(seed, "dagma", "dagma", vanilla_runtime,
                        _metrics(true_path, vanilla_path, metric_path))
        row.update({
            "trek_weight": np.nan, "trek_function": "none",
            "number_of_no_trek_pairs": len(pairs),
            "raw_notreks_penalty_before_threshold": np.nan,
            "scaled_notreks_penalty_before_threshold": np.nan,
            "raw_notreks_penalty_after_threshold": np.nan,
            "number_of_oracle_pairs_violated_after_threshold": _violations(vanilla_A, pairs),
            "fraction_of_oracle_pairs_violated_after_threshold":
                _violations(vanilla_A, pairs) / len(pairs) if pairs else 0.0,
            "dagma_score": float(vanilla_model.score_final),
            "dagma_h_value": float(vanilla_model.h_final),
            "optimizer_success": True,
            "optimizer_convergence": "not exposed by dagma 1.1.1; configured max_iter completed or tolerance stop",
            "optimizer_iterations": sum(
                x["iterations_performed"] for x in vanilla_model.stage_diagnostics),
            "maximum_inverse_condition_number": 0.0,
            "inverse_solve_failures": 0, "inverse_stage_retries": 0,
        })
        rows.append(row)
        np.save(dataset_dir / "dagma_weighted.npy", vanilla_W)
        for item in vanilla_model.stage_diagnostics:
            stages.append({"tag": tag, "d": len(names), "n": len(X), "seed": seed,
                           "algorithm": "dagma", "id": "dagma", "trek_weight": np.nan, **item})
        for diagnostic_threshold in (0.05, 0.1, 0.2, 0.3):
            diagnostic_A = (np.abs(vanilla_W) >= diagnostic_threshold).astype(int)
            np.fill_diagonal(diagnostic_A, 0)
            diagnostic_path = _write_estimate(
                diagnostic_A, names, dataset_dir, f"dagma_threshold_{diagnostic_threshold}")
            diagnostic_metrics = _metrics(
                true_path, diagnostic_path,
                dataset_dir / f"dagma_threshold_{diagnostic_threshold}_metrics.csv")
            threshold_rows.append({
                "tag": tag, "d": len(names), "n": len(X), "seed": seed,
                "algorithm": "dagma", "id": "dagma", "trek_weight": np.nan,
                "threshold": diagnostic_threshold, "estimated_edges": _edge_count(diagnostic_A),
                "oracle_violations": _violations(diagnostic_A, pairs), **diagnostic_metrics})

        for nt in algorithms["dagma_notreks"]:
            model = SharedDagmaLinear("l2")
            model.profile_components = True
            start = time.perf_counter()
            W = model.fit(
                X.copy(), w_threshold=0.0, no_trek_pairs=pairs,
                trek_weight=float(nt["trek_weight"]), trek_function=nt["trek_function"],
                trek_log_terms=int(nt["trek_log_terms"]),
                trek_inverse_epsilon=float(nt["trek_inverse_epsilon"]), **ordinary)
            runtime = time.perf_counter() - start
            A = (np.abs(W) >= threshold).astype(int)
            np.fill_diagonal(A, 0)
            before, _ = notreks_value_grad(
                W, pairs, nt["trek_function"], log_terms=int(nt["trek_log_terms"]),
                inverse_epsilon=float(nt["trek_inverse_epsilon"]))
            after, _ = notreks_value_grad(
                A.astype(float), pairs, nt["trek_function"],
                log_terms=int(nt["trek_log_terms"]),
                inverse_epsilon=float(nt["trek_inverse_epsilon"]))
            scale = 2 / (len(A) - 1) if len(A) > 1 else 0
            violations = _violations(A, pairs)
            estimate_path = _write_estimate(A, names, dataset_dir, nt["id"])
            metric_path = dataset_dir / f"{nt['id']}_metrics.csv"
            row = _base_row(seed, "dagma_notreks", nt["id"], runtime,
                            _metrics(true_path, estimate_path, metric_path))
            row.update({
                "trek_weight": float(nt["trek_weight"]), "trek_function": nt["trek_function"],
                "number_of_no_trek_pairs": len(pairs),
                "raw_notreks_penalty_before_threshold": before / scale if scale else 0.0,
                "scaled_notreks_penalty_before_threshold": before,
                "raw_notreks_penalty_after_threshold": after / scale if scale else 0.0,
                "number_of_oracle_pairs_violated_after_threshold": violations,
                "fraction_of_oracle_pairs_violated_after_threshold":
                    violations / len(pairs) if pairs else 0.0,
                "dagma_score": float(model.score_final),
                "dagma_h_value": float(model.h_final),
                "optimizer_success": True,
                "optimizer_convergence": "not exposed by dagma 1.1.1; configured max_iter completed or tolerance stop",
                "optimizer_iterations": sum(
                    x["iterations_performed"] for x in model.stage_diagnostics),
                "maximum_inverse_condition_number":
                    model._trek_kernel.maximum_condition_number,
                "inverse_solve_failures": model._trek_kernel.solve_failures,
                "inverse_stage_retries": sum(
                    x["inverse_stage_retries"] for x in model.stage_diagnostics),
            })
            rows.append(row)
            np.save(dataset_dir / f"{nt['id']}_weighted.npy", W)
            for item in model.stage_diagnostics:
                stages.append({"tag": tag, "d": len(names), "n": len(X), "seed": seed,
                               "algorithm": "dagma_notreks", "id": nt["id"],
                               "trek_weight": float(nt["trek_weight"]), **item})
            for diagnostic_threshold in (0.05, 0.1, 0.2, 0.3):
                diagnostic_A = (np.abs(W) >= diagnostic_threshold).astype(int)
                np.fill_diagonal(diagnostic_A, 0)
                diagnostic_path = _write_estimate(
                    diagnostic_A, names, dataset_dir,
                    f"{nt['id']}_threshold_{diagnostic_threshold}")
                diagnostic_metrics = _metrics(
                    true_path, diagnostic_path,
                    dataset_dir / f"{nt['id']}_threshold_{diagnostic_threshold}_metrics.csv")
                threshold_rows.append({
                    "tag": tag, "d": len(names), "n": len(X), "seed": seed,
                    "algorithm": "dagma_notreks", "id": nt["id"],
                    "trek_weight": float(nt["trek_weight"]),
                    "threshold": diagnostic_threshold, "estimated_edges": _edge_count(diagnostic_A),
                    "oracle_violations": _violations(diagnostic_A, pairs), **diagnostic_metrics})
            if float(nt["trek_weight"]) == 0:
                equality.append({
                    "seed": seed,
                    "weighted_exact": bool(np.array_equal(vanilla_W, W)),
                    "weighted_max_abs_difference": float(np.max(np.abs(vanilla_W - W))),
                    "maximum_stage_weighted_difference": float(max(
                        np.max(np.abs(a - b)) for a, b in zip(
                            vanilla_model.stage_adjacencies, model.stage_adjacencies))),
                    "thresholded_exact": bool(np.array_equal(vanilla_A, A)),
                    "score_difference": float(abs(
                        vanilla_model.score_final - model.score_final)),
                    "h_difference": float(abs(vanilla_model.h_final - model.h_final)),
                    "stage_iterations_equal": [
                        x["iterations_performed"] for x in vanilla_model.stage_diagnostics
                    ] == [
                        x["iterations_performed"] for x in model.stage_diagnostics
                    ],
                })

        pc = algorithms["pc_mi_oracle"][0]
        dispatcher = MIOracleFisherZ(X.copy(), pairs, float(pc["alpha"]))
        start = time.perf_counter()
        pc_A, counts = stable_pc(X.copy(), dispatcher, int(pc["max_cond_set"]))
        pc_runtime = time.perf_counter() - start
        pc_path = _write_estimate(pc_A, names, dataset_dir, pc["id"])
        row = _base_row(seed, "pc_mi_oracle", pc["id"], pc_runtime,
                        _metrics(true_path, pc_path, dataset_dir / "pc_mi_oracle_metrics.csv"))
        row.update({
            "trek_weight": np.nan, "number_of_no_trek_pairs": len(pairs),
            "number_of_oracle_pairs_violated_after_threshold": _violations(pc_A, pairs),
            "fraction_of_oracle_pairs_violated_after_threshold":
                _violations(pc_A, pairs) / len(pairs) if pairs else 0.0,
            "number_of_oracle_marginal_queries": counts.oracle_marginal_queries,
            "number_of_oracle_independent_answers": counts.oracle_independent_answers,
            "number_of_oracle_dependent_answers": counts.oracle_dependent_answers,
            "number_of_statistical_conditional_tests": counts.statistical_conditional_tests,
        })
        rows.append(row)

        flop = algorithms["flop"][0]
        flop_raw = dataset_dir / "flop_raw.npy"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--flop-worker",
            str(data_path), str(flop_raw), str(flop["lambda_bic"]), str(flop["restarts"]),
        ]
        start = time.perf_counter()
        try:
            flop_timeout = int(grid.get("local_overrides", {}).get("flop_timeout", 60))
            proc = subprocess.run(command, timeout=flop_timeout, capture_output=True, text=True)
            flop_runtime = time.perf_counter() - start
            if proc.returncode:
                raise RuntimeError(proc.stderr.strip() or f"exit {proc.returncode}")
            flop_A = convert_flop_cpdag(np.load(flop_raw), len(names))
            flop_path = _write_estimate(flop_A, names, dataset_dir, "flop")
            row = _base_row(seed, "flop", "flop", flop_runtime,
                            _metrics(true_path, flop_path, dataset_dir / "flop_metrics.csv"))
            row.update({"trek_weight": np.nan,
                        "number_of_oracle_pairs_violated_after_threshold": _violations(flop_A, pairs)})
            rows.append(row)
            flop_status.append({"seed": seed, "status": "completed", "runtime": flop_runtime,
                                "command": command})
        except subprocess.TimeoutExpired:
            flop_status.append({"seed": seed, "status": f"timeout_{flop_timeout}s",
                                "runtime": float(flop_timeout),
                                "command": command})

        if "flop_notreks" in algorithms:
            fn = algorithms["flop_notreks"][0]
            fn_raw = dataset_dir / "flop_notreks_raw.npy"
            fn_diagnostics_path = dataset_dir / "flop_notreks_diagnostics.json"
            algorithm_seed = (int(seed) + int(fn["algorithm_seed"])) % (2**64)
            command = [
                sys.executable, str(Path(__file__).resolve()), "--flop-notreks-worker",
                str(data_path), str(sidecar_path), str(fn_raw),
                str(fn_diagnostics_path), str(fn["lambda_bic"]), str(fn["restarts"]),
                str(algorithm_seed), str(fn["signature_top_k"]),
                str(fn["max_signature_rounds"]),
            ]
            start = time.perf_counter()
            try:
                fn_timeout = int(fn.get("timeout") or 120)
                proc = subprocess.run(
                    command, timeout=fn_timeout, capture_output=True, text=True)
                fn_runtime = time.perf_counter() - start
                if proc.returncode:
                    raise RuntimeError(proc.stderr.strip() or f"exit {proc.returncode}")
                raw = np.load(fn_raw)
                fn_A = convert_flop_cpdag(raw, len(names))
                diagnostics = json.loads(fn_diagnostics_path.read_text())
                selected_dag = selected_dag_from_diagnostics(diagnostics, len(names))
                violations = count_no_trek_violations(selected_dag, pairs)
                if violations or diagnostics["final_no_trek_violation_count"] != 0:
                    raise RuntimeError(
                        f"FLOP-NOTREKS selected DAG has {violations} violations")
                fn_path = _write_estimate(fn_A, names, dataset_dir, fn["id"])
                row = _base_row(
                    seed, "flop_notreks", fn["id"], fn_runtime,
                    _metrics(true_path, fn_path,
                             dataset_dir / "flop_notreks_metrics.csv"))
                row.update({
                    "trek_weight": np.nan,
                    "number_of_no_trek_pairs": len(pairs),
                    "number_of_oracle_pairs_violated_after_threshold": violations,
                    **diagnostics,
                })
                rows.append(row)
                flop_status.append({
                    "seed": seed, "family": "flop_notreks",
                    "status": "completed", "runtime": fn_runtime, "command": command})
            except subprocess.TimeoutExpired:
                flop_status.append({
                    "seed": seed, "family": "flop_notreks",
                    "status": f"timeout_{fn_timeout}s", "runtime": float(fn_timeout),
                    "command": command})

    if not all(x["weighted_exact"] and x["thresholded_exact"] for x in equality):
        raise RuntimeError(f"weight-zero equivalence failed: {equality}")
    results = pd.DataFrame(rows)
    results.insert(0, "tag", tag)
    results.insert(1, "d", int(grid["data"]["d"]))
    results.insert(2, "n", int(grid["data"]["n"]))
    results["estimated_edges"] = results.apply(
        lambda x: _edge_count(pd.read_csv(
            out / f"seed_{int(x.seed)}" / f"{x.id}_adjmat.csv").to_numpy()), axis=1)
    results["weighted_max_absolute_edge"] = results.apply(
        lambda x: float(np.abs(np.load(out / f"seed_{int(x.seed)}" /
            (f"{x.id}_weighted.npy" if x.algorithm == "dagma_notreks" else "dagma_weighted.npy"))).max())
        if x.algorithm in {"dagma", "dagma_notreks"} else np.nan, axis=1)
    results = results.merge(pd.DataFrame(dataset_metadata), on="seed", how="left")
    results.to_csv(out / "raw_per_run.csv", index=False)
    results.to_csv(joint, index=False)
    pd.DataFrame(stages).to_csv(out / "stage_diagnostics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out / "threshold_diagnostics.csv", index=False)
    pd.DataFrame(dataset_metadata).to_csv(out / "dataset_manifest.csv", index=False)
    total_pairs = len(true) * (len(true) - 1) // 2
    graph_summary = {
        "tag": tag, "d": len(true), "total_unordered_node_pairs": total_pairs,
        "number_of_no_trek_oracle_pairs": len(pairs),
        "fraction_no_trek": len(pairs) / total_pairs if total_pairs else 0.0,
        "true_edge_count": _edge_count(true), "true_cpdag_edge_count": _edge_count(true),
        "oracle_source": "structural inclusive-ancestor intersection from true DAG",
    }
    (out / "oracle_graph_summary.json").write_text(json.dumps(graph_summary, indent=2) + "\n")
    (out / "weight_zero_equivalence.json").write_text(json.dumps(equality, indent=2) + "\n")
    (out / "flop_status.json").write_text(json.dumps(flop_status, indent=2) + "\n")
    print(f"raw_results={out / 'raw_per_run.csv'}")
    print(f"joint_benchmarks={joint}")
    print(f"weight_zero_equivalence={json.dumps(equality)}")
    print(f"flop_status={json.dumps(flop_status)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flop-worker", nargs=4, metavar=("DATA", "OUTPUT", "LAMBDA", "RESTARTS"))
    parser.add_argument("--flop-notreks-worker", nargs=9)
    parser.add_argument("--tag", default="oracle_smoke")
    args = parser.parse_args()
    if args.flop_worker:
        data, output, penalty, restarts = args.flop_worker
        _flop_worker(data, output, float(penalty), int(restarts))
    elif args.flop_notreks_worker:
        data, sidecar, output, diagnostics, penalty, restarts, seed, top_k, rounds = \
            args.flop_notreks_worker
        _flop_notreks_worker(
            data, sidecar, output, diagnostics, float(penalty), int(restarts),
            int(seed), int(top_k), int(rounds))
    else:
        run(args.tag)


if __name__ == "__main__":
    main()
