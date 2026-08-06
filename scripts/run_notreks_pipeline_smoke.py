#!/usr/bin/env python3
"""Container-free one-seed execution of the complete four-method pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.notreks_benchmark import analyse, compile_configs
from scripts.tune_notreks_smoke import dataset
from workflow.rules.structure_learning_algorithms.dagma.shared import SharedDagmaLinear
from workflow.rules.structure_learning_algorithms.dagma.knowledge import no_trek_pairs_from_dag
from workflow.rules.structure_learning_algorithms.dagma_notreks.tools.local_smoke import _metrics
from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag
from workflow.rules.structure_learning_algorithms.flop_notreks.adapter import (
    count_no_trek_violations,
)


def run(output_dir: Path) -> None:
    import flopsearch

    root = Path(__file__).resolve().parents[1]
    calibration = json.loads((root / "configs/notreks_benchmark/calibrations/"
                               "d20_n200_oracle25_v1.json").read_text())
    spec = root / "configs/notreks_benchmark/pipeline_smoke_v1.json"
    compile_configs(spec, output_dir / "benchpress_config", smoke=False)
    data, truth, pairs = dataset(9201, d=20, n=200)
    names = [f"X{index + 1}" for index in range(data.shape[1])]
    number_of_oracle_pairs = len(no_trek_pairs_from_dag(truth, names))
    true_path = output_dir / "true_graph.csv"
    data_path = output_dir / "data.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(truth, columns=names).to_csv(true_path, index=False)
    pd.DataFrame(data, columns=names).to_csv(data_path, index=False)
    estimates = {}
    runtime = {}

    started = time.perf_counter()
    raw = flopsearch.flop(data, 2.0, restarts=16)
    estimates["flop"] = convert_flop_cpdag(np.asarray(raw), data.shape[1])
    runtime["flop"] = time.perf_counter() - started

    started = time.perf_counter()
    raw, diagnostics = flopsearch.flop_notreks(
        data, 2.0, pairs,
        restarts=int(calibration["flop_notreks_restarts"]),
        max_signature_rounds=int(calibration["flop_notreks_max_sweeps"]),
        seed=int(calibration["algorithm_seed"]) + 9201,
        search_version="global_greedy_rust", return_diagnostics=True,
        return_dag=True)
    estimates["flop_notreks"] = np.zeros((data.shape[1], data.shape[1]), dtype=np.uint8)
    for parent, child in diagnostics["selected_dag_edges"]:
        estimates["flop_notreks"][int(parent), int(child)] = 1
    runtime["flop_notreks"] = time.perf_counter() - started

    fit = dict(lambda1=.03, w_threshold=0., T=5,
               s=(1., .9, .8, .7, .6), warm_iter=3000,
               max_iter=6000, checkpoint=500)
    started = time.perf_counter()
    weighted = SharedDagmaLinear("l2").fit(
        data.copy(), no_trek_pairs=[], trek_weight=0., **fit)
    estimates["dagma"] = (np.abs(weighted) >= .30).astype(np.uint8)
    np.fill_diagonal(estimates["dagma"], 0)
    runtime["dagma"] = time.perf_counter() - started

    started = time.perf_counter()
    weighted_nt = SharedDagmaLinear("l2").fit(
        data.copy(), no_trek_pairs=pairs,
        trek_weight=float(calibration["dagma_trek_weight"]),
        trek_function="inv", trek_kernel="fast", **fit)
    estimates["dagma_notreks"] = (
        np.abs(weighted_nt) >= .30).astype(np.uint8)
    np.fill_diagonal(estimates["dagma_notreks"], 0)
    runtime["dagma_notreks"] = time.perf_counter() - started

    rows = []
    for method, adjacency in estimates.items():
        estimate_path = output_dir / f"{method}.csv"
        metric_path = output_dir / f"{method}_metrics.csv"
        pd.DataFrame(adjacency, columns=names).to_csv(estimate_path, index=False)
        metrics = _metrics(true_path, estimate_path, metric_path)
        # Evaluate the same supplied no-trek knowledge for every method.  The
        # vanilla arms do not use the pairs during optimization, but their
        # post-hoc violation count is scientifically important rather than
        # missing data.
        violations = count_no_trek_violations(adjacency, pairs)
        rows.append({
            "scenario": "pipeline-smoke-v1", "model": "linear_gaussian",
            "graph": "er2", "d": 20, "n": 200,
            "knowledge_fraction": .25, "seed": 9201,
            "id": method, "algorithm": method, "time": runtime[method],
            "num_oracle_mi_pairs": number_of_oracle_pairs,
            "num_supplied_mi_pairs": len(pairs),
            "num_mi_violations": violations,
            "mi_violation_fraction": violations / len(pairs) if pairs else 0.0,
            **metrics,
        })
    results = output_dir / "all_runs.csv"
    pd.DataFrame(rows).to_csv(results, index=False)
    analyse(results, output_dir / "analysis")
    print(pd.DataFrame(rows)[[
        "id", "SHD_cpdag", "SHD_pattern", "F1_pattern", "time",
        "num_mi_violations"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
