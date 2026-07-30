#!/usr/bin/env python3
"""Benchmark FLOP-NOTREKS on the exact saved oracle_local_* inputs."""

import json
import subprocess
import time
from pathlib import Path

import flopsearch
import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.knowledge import (
    load_sidecar, named_pairs_to_indices,
)
from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag
from workflow.rules.structure_learning_algorithms.flop_notreks.adapter import (
    count_no_trek_violations, selected_dag_from_diagnostics,
)


ROOT = Path(__file__).resolve().parents[5]
OUTPUT = ROOT / "results/dagma_notreks_oracle/flop_notreks_local_benchmark"
R_EVAL = ROOT / "workflow/rules/evaluation/benchmarks/run_summarise.R"
CASES = [
    ("oracle_local_d20_inv", "oracle_local_d20", 20, 1001),
    ("oracle_local_d20_inv", "oracle_local_d20", 20, 1002),
    ("oracle_local_d50_inv", "oracle_local_d50", 50, 1001),
]


def metrics(true_path, estimate_path, output_path):
    subprocess.run([
        "Rscript", str(R_EVAL), "--adjmat_true", str(true_path),
        "--adjmat_est", str(estimate_path), "--filename", str(output_path),
    ], check=True, capture_output=True, text=True)
    row = pd.read_csv(output_path).iloc[0].to_dict()
    for suffix in ("skel", "pattern"):
        tp, fp, fn = (float(row[f"TP_{suffix}"]), float(row[f"FP_{suffix}"]),
                      float(row[f"FN_{suffix}"]))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        row[f"F1_{suffix}"] = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0)
    return row


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, input_tag, dimension, seed in CASES:
        source = ROOT / "results/dagma_notreks_oracle" / tag
        case_output = OUTPUT / f"d{dimension}_seed{seed}"
        case_output.mkdir(parents=True, exist_ok=True)
        data_path = ROOT / f"resources/data/mydatasets/{input_tag}/s{seed}.csv"
        true_path = ROOT / f"resources/adjmat/myadjmats/g{input_tag}.csv"
        sidecar_path = source / f"seed_{seed}/no_trek_pairs.json"
        frame = pd.read_csv(data_path)
        data = frame.to_numpy(float)
        data = (data - data.mean(0)) / data.std(0, ddof=0)
        pairs = named_pairs_to_indices(
            load_sidecar(sidecar_path, list(frame.columns)), list(frame.columns))
        for search_version in ("fixed_signature_a", "alternating_full_refit_b"):
            started = time.perf_counter()
            cpdag_raw, diagnostics = flopsearch.flop_notreks(
                data, 2.0, pairs, restarts=1, seed=seed,
                signature_top_k=5, max_signature_rounds=20,
                search_version=search_version, return_diagnostics=True)
            runtime = time.perf_counter() - started
            cpdag = convert_flop_cpdag(cpdag_raw, dimension)
            dag = selected_dag_from_diagnostics(diagnostics, dimension)
            violations = count_no_trek_violations(dag, pairs)
            if violations:
                raise RuntimeError(
                    f"d={dimension} seed={seed} {search_version}: "
                    f"{violations} violations")
            prefix = f"flop_notreks_{search_version}"
            cpdag_path = case_output / f"{prefix}_adjmat.csv"
            dag_path = case_output / f"{prefix}_selected_dag.csv"
            pd.DataFrame(cpdag, columns=frame.columns).to_csv(
                cpdag_path, index=False)
            pd.DataFrame(dag, columns=frame.columns).to_csv(
                dag_path, index=False)
            evaluated = metrics(
                true_path, cpdag_path, case_output / f"{prefix}_metrics.csv")
            row = {
                "d": dimension, "seed": seed, "method": prefix,
                "runtime": runtime, "BIC": diagnostics["final_bic"],
                "SHD_pattern": evaluated["SHD_pattern"],
                "SHD_cpdag": evaluated.get("SHD_cpdag"),
                "F1_skel": evaluated["F1_skel"],
                "F1_pattern": evaluated["F1_pattern"],
                "selected_DAG_edges": int(dag.sum()),
                "no_trek_violations": violations,
                **diagnostics,
            }
            rows.append(row)
            (case_output / f"{prefix}_diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2) + "\n")

        raw = pd.read_csv(source / "raw_per_run.csv")
        threshold = pd.read_csv(source / "threshold_diagnostics.csv")
        vanilla_started = time.perf_counter()
        # The released ``flop`` API has no seed.  For a reproducible diagnostic,
        # use the fork's tested empty-constraint path, which performs the same
        # unpruned FLOP search with an injected RNG seed.
        vanilla_raw = flopsearch.flop_notreks(
            data, 2.0, [], restarts=1, seed=seed)
        vanilla_runtime = time.perf_counter() - vanilla_started
        vanilla_cpdag = convert_flop_cpdag(vanilla_raw, dimension)
        vanilla_path = case_output / "flop_adjmat.csv"
        pd.DataFrame(vanilla_cpdag, columns=frame.columns).to_csv(
            vanilla_path, index=False)
        vanilla_metrics = metrics(
            true_path, vanilla_path, case_output / "flop_metrics.csv")
        rows.append({
            "d": dimension, "seed": seed, "method": "flop",
            "runtime": vanilla_runtime, "BIC": np.nan,
            "SHD_pattern": vanilla_metrics["SHD_pattern"],
            "SHD_cpdag": vanilla_metrics.get("SHD_cpdag"),
            "F1_skel": vanilla_metrics["F1_skel"],
            "F1_pattern": vanilla_metrics["F1_pattern"],
            "selected_DAG_edges": np.nan, "no_trek_violations": np.nan,
        })
        for method in ("pc_mi_oracle",):
            base = raw[(raw.seed == seed) & (raw.algorithm == method)].iloc[0]
            rows.append({
                "d": dimension, "seed": seed, "method": method,
                "runtime": base.runtime, "BIC": np.nan,
                "SHD_pattern": base.SHD_pattern, "SHD_cpdag": base.SHD_cpdag,
                "F1_skel": base.F1_skel, "F1_pattern": base.F1_pattern,
                "selected_DAG_edges": np.nan,
                "no_trek_violations":
                    base.number_of_oracle_pairs_violated_after_threshold,
            })
        for method, weight in (("dagma", np.nan), ("dagma_notreks", 1.0)):
            subset = threshold[
                (threshold.seed == seed) & (threshold.algorithm == method)
                & np.isclose(threshold.threshold, .2)]
            if method == "dagma_notreks":
                subset = subset[np.isclose(subset.trek_weight, weight)]
            base = subset.iloc[0]
            runtime_row = raw[
                (raw.seed == seed) & (raw.algorithm == method)]
            if method == "dagma_notreks":
                runtime_row = runtime_row[np.isclose(runtime_row.trek_weight, weight)]
            rows.append({
                "d": dimension, "seed": seed,
                "method": "dagma_notreks_w1" if method == "dagma_notreks" else method,
                "runtime": runtime_row.iloc[0].runtime, "BIC": np.nan,
                "SHD_pattern": base.SHD_pattern, "SHD_cpdag": base.SHD_cpdag,
                "F1_skel": base.F1_skel, "F1_pattern": base.F1_pattern,
                "selected_DAG_edges": base.estimated_edges,
                "no_trek_violations": base.oracle_violations,
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "per_run.csv", index=False)
    summary = table.groupby(["d", "method"], dropna=False).agg(
        SHD_pattern=("SHD_pattern", "mean"), SHD_cpdag=("SHD_cpdag", "mean"),
        F1_skel=("F1_skel", "mean"), F1_pattern=("F1_pattern", "mean"),
        runtime=("runtime", "mean"), no_trek_violations=("no_trek_violations", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    taxonomy = pd.DataFrame([
        {
            "method": "flop", "signatures": "none",
            "signature_search": "none", "cone_promotion": "none",
            "proposal_evaluation": "ordinary incremental FLOP",
        },
        {
            "method": "fixed_signature_a", "signatures": "fixed",
            "signature_search": "no", "cone_promotion": "no",
            "proposal_evaluation": "ordinary constrained FLOP",
        },
        {
            "method": "alternating_full_refit_b", "signatures": "variable",
            "signature_search": "yes", "cone_promotion": "yes",
            "proposal_evaluation": "complete constrained refit",
        },
        {
            "method": "incremental_c", "signatures": "variable",
            "signature_search": "yes", "cone_promotion": "yes",
            "proposal_evaluation": "reserved: exact affected-node update",
        },
        {
            "method": "hybrid_bc", "signatures": "variable",
            "signature_search": "yes", "cone_promotion": "yes",
            "proposal_evaluation": "reserved: incremental or full refit",
        },
    ])
    taxonomy.to_csv(OUTPUT / "version_taxonomy.csv", index=False)
    (OUTPUT / "report.md").write_text(
        "# FLOP-NOTREKS version benchmark\n\n"
        "Exact saved inputs; hard constraints are checked on each selected DAG.\n\n"
        "## Version taxonomy\n\n```\n"
        + taxonomy.to_string(index=False)
        + "\n```\n\n## Aggregate results\n\n```\n"
        + summary.to_string(index=False)
        + "\n```\n\nVersion C and Hybrid B/C are reserved and were not run.\n"
    )
    print(table.to_string(index=False))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
