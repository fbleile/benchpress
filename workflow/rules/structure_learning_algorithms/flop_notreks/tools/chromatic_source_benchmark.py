#!/usr/bin/env python3
"""Paired FLOP comparison for the chromatic source-prefix proposal."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import flopsearch
import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.knowledge import no_trek_pairs_from_dag
from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag
from workflow.rules.structure_learning_algorithms.flop_notreks.adapter import (
    count_no_trek_violations,
    selected_dag_from_diagnostics,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.chromatic_sources import (
    fit_chromatic_source_flop,
    fit_chromatic_source_notreks,
    fit_chromatic_source_fixed_notreks,
)


ROOT = Path(__file__).resolve().parents[5]
R_EVAL = ROOT / "workflow/rules/evaluation/benchmarks/run_summarise.R"


def evaluate(true_path: Path, estimate_path: Path, output_path: Path) -> dict:
    subprocess.run(
        ["Rscript", str(R_EVAL), "--adjmat_true", str(true_path),
         "--adjmat_est", str(estimate_path), "--filename", str(output_path)],
        check=True, capture_output=True, text=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001])
    parser.add_argument("--restarts", type=int, default=0)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument(
        "--methods", nargs="+",
        choices=["flop", "global", "global_cached", "global_parallel", "fixed", "incremental", "chromatic", "combined", "fixed_combined"],
        default=["flop", "global", "global_cached", "global_parallel", "fixed", "incremental", "chromatic", "combined", "fixed_combined"],
        help="Run only the selected methods; useful for extending one comparator's budget.")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/dagma_notreks_oracle/chromatic_source_benchmark")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_tag = f"oracle_local_d{args.dimension}"
    data_dir = ROOT / "resources/data/mydatasets" / input_tag
    true_path = ROOT / "resources/adjmat/myadjmats" / f"g{input_tag}.csv"
    true_frame = pd.read_csv(true_path)
    names = list(true_frame.columns)
    named_pairs = no_trek_pairs_from_dag(true_frame.to_numpy(), names)
    index = {name: i for i, name in enumerate(names)}
    pairs = [(index[left], index[right]) for left, right in named_pairs]
    rows = []

    for seed in args.seeds:
        frame = pd.read_csv(data_dir / f"s{seed}.csv")
        data = frame.to_numpy(float)
        data = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
        case_dir = args.output / f"d{args.dimension}_seed{seed}"
        case_dir.mkdir(parents=True, exist_ok=True)

        available_calls = {
            "flop": ("flop", lambda: (flopsearch.flop(
                data, 2.0, restarts=args.restarts), None)),
            "global": ("flop_notreks_global_greedy", lambda: flopsearch.flop_notreks(
                data, 2.0, pairs, restarts=args.restarts, seed=seed,
                max_signature_rounds=args.max_sweeps,
                search_version="global_greedy_rust", return_diagnostics=True)),
            "global_cached": ("flop_notreks_global_greedy_cached", lambda: flopsearch.flop_notreks(
                data, 2.0, pairs, restarts=args.restarts, seed=seed,
                max_signature_rounds=args.max_sweeps,
                search_version="global_greedy_cached", return_diagnostics=True)),
            "global_parallel": ("flop_notreks_global_greedy_parallel", lambda: flopsearch.flop_notreks(
                data, 2.0, pairs, restarts=args.restarts, seed=seed,
                max_signature_rounds=args.max_sweeps,
                search_version="global_greedy_parallel", return_diagnostics=True)),
            "fixed": ("flop_notreks_fixed_signature", lambda: flopsearch.flop_notreks(
                data, 2.0, pairs, restarts=args.restarts, seed=seed,
                max_signature_rounds=args.max_sweeps,
                search_version="fixed_signature_a", return_diagnostics=True)),
            "incremental": ("flop_notreks_incremental_promotion", lambda: flopsearch.flop_notreks(
                data, 2.0, pairs, restarts=args.restarts, seed=seed,
                max_signature_rounds=args.max_sweeps,
                search_version="incremental_promotion_d", return_diagnostics=True)),
            "chromatic": ("flop_chromatic_source_prefix", lambda: fit_chromatic_source_flop(
                data, pairs, restarts=args.restarts, seed=seed)),
            "combined": ("flop_notreks_chromatic_sources", lambda: fit_chromatic_source_notreks(
                data, pairs, restarts=args.restarts, seed=seed,
                max_sweeps=args.max_sweeps)),
            "fixed_combined": (
                "flop_notreks_fixed_signature_chromatic_sources",
                lambda: fit_chromatic_source_fixed_notreks(
                    data, pairs, restarts=args.restarts, seed=seed)),
        }
        calls = [available_calls[name] for name in args.methods]
        for method, call in calls:
            started = time.perf_counter()
            raw, diagnostics = call()
            runtime = time.perf_counter() - started
            cpdag = convert_flop_cpdag(raw, args.dimension)
            estimate_path = case_dir / f"{method}_adjmat.csv"
            pd.DataFrame(cpdag, columns=names).to_csv(estimate_path, index=False)
            metrics = evaluate(
                true_path, estimate_path, case_dir / f"{method}_metrics.csv")
            row = {
                "d": args.dimension, "seed": seed, "method": method,
                "runtime": runtime, "SHD_pattern": metrics["SHD_pattern"],
                "SHD_cpdag": metrics.get("SHD_cpdag"),
                "F1_skel": metrics["F1_skel"],
                "F1_pattern": metrics["F1_pattern"],
                "number_of_no_trek_pairs": len(pairs),
                "restarts": args.restarts,
                "max_sweeps": args.max_sweeps,
            }
            if diagnostics is not None:
                diagnostics = dict(diagnostics)
                dag = selected_dag_from_diagnostics(diagnostics, args.dimension)
                row.update({
                    "BIC": diagnostics.get("final_bic"),
                    "selected_DAG_edges": int(dag.sum()),
                    "no_trek_violations": count_no_trek_violations(dag, pairs),
                    "source_prefix": diagnostics.get("source_prefix"),
                    "chromatic_exact": diagnostics.get("chromatic_exact"),
                    "chromatic_clique_lower_bound": diagnostics.get(
                        "chromatic_clique_lower_bound"),
                    "chromatic_coloring_upper_bound": diagnostics.get(
                        "chromatic_coloring_upper_bound"),
                })
                (case_dir / f"{method}_diagnostics.json").write_text(
                    json.dumps(diagnostics, indent=2) + "\n")
            rows.append(row)

    table = pd.DataFrame(rows)
    if "no_trek_violations" not in table:
        table["no_trek_violations"] = np.nan
    table.to_csv(args.output / "per_run.csv", index=False)
    summary = table.groupby("method", dropna=False).agg(
        runs=("seed", "count"), SHD_pattern=("SHD_pattern", "mean"),
        SHD_cpdag=("SHD_cpdag", "mean"), F1_skel=("F1_skel", "mean"),
        F1_pattern=("F1_pattern", "mean"), runtime=("runtime", "mean"),
        no_trek_violations=("no_trek_violations", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "report.md").write_text(
        "# Chromatic source-prefix FLOP benchmark\n\n"
        "Paired oracle inputs and the full oracle no-trek graph H.\n\n``\n"
        + summary.to_string(index=False) + "\n``\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
