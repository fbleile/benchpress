#!/usr/bin/env python3
"""One-time, truth-free NOTREKS hyperparameter calibration smoke."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import gaussian_bic
from workflow.rules.structure_learning_algorithms.dagma.knowledge import no_trek_pairs_from_dag
from workflow.rules.structure_learning_algorithms.dagma.shared import SharedDagmaLinear
from workflow.rules.structure_learning_algorithms.flop_notreks.adapter import (
    count_no_trek_violations, selected_dag_from_diagnostics,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.global_greedy import (
    GlobalGreedyConfig, fit_global_greedy_notreks,
)
from workflow.rules.structure_learning_algorithms.notreks import subsample_no_trek_pairs


def dataset(seed: int, d: int = 20, n: int = 200):
    sequence = np.random.SeedSequence(seed)
    graph_seed, coefficient_seed, noise_seed = sequence.spawn(3)
    rng = np.random.default_rng(graph_seed)
    order = rng.permutation(d)
    adjacency = np.zeros((d, d), dtype=np.uint8)
    for left in range(d):
        for right in range(left + 1, d):
            if rng.random() < 2.0 / (d - 1):
                adjacency[order[left], order[right]] = 1
    coefficients = adjacency * np.random.default_rng(coefficient_seed).uniform(
        0.5, 1.0, size=(d, d)) * np.random.default_rng(
            coefficient_seed).choice((-1, 1), size=(d, d))
    noise = np.random.default_rng(noise_seed).normal(size=(n, d))
    data = noise @ np.linalg.inv(np.eye(d) - coefficients)
    data = (data - data.mean(0)) / data.std(0, ddof=0)
    names = [f"X{index + 1}" for index in range(d)]
    named = no_trek_pairs_from_dag(adjacency, names)
    pairs = [(names.index(left), names.index(right)) for left, right in named]
    return data, adjacency, subsample_no_trek_pairs(pairs, 0.25, 271828)


def postselect(data, weighted, pairs):
    """Apply the same fixed threshold used by both benchmark DAGMA arms."""
    graph = (np.abs(np.asarray(weighted)) >= .30).astype(np.uint8)
    np.fill_diagonal(graph, 0)
    bic, _ = gaussian_bic(data, graph, lambda_bic=2.0)
    return graph, float(bic), None


def truth_metrics(truth, estimate):
    truth = np.asarray(truth, dtype=bool)
    estimate = np.asarray(estimate, dtype=bool)
    tp = int(np.sum(truth & estimate))
    fp = int(np.sum(~truth & estimate))
    fn = int(np.sum(truth & ~estimate))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "shd_pattern": fp + fn,
        "f1_pattern": (2 * precision * recall / (precision + recall)
                       if precision + recall else 0.0),
    }


def run(output_dir: Path, write_defaults: Path | None = None):
    import flopsearch

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    dagma_weights = (0.3, 3.0, 10.0)
    flop_settings = ((1, 1), (2, 1), (2, 2))
    for seed in (9101, 9102, 9103, 9104, 9105):
        data, truth, pairs = dataset(seed)
        model = SharedDagmaLinear("l2")
        started = time.perf_counter()
        weighted = model.fit(
            data.copy(), lambda1=.03, w_threshold=0., T=5,
            s=(1., .9, .8, .7, .6), warm_iter=3000, max_iter=6000,
            checkpoint=500,
            no_trek_pairs=[], trek_weight=0.)
        graph, bic, _ = postselect(data, weighted, [])
        rows.append({"seed": seed, "family": "dagma", "setting": "default",
                     "bic": bic, "violations": count_no_trek_violations(graph, pairs),
                     "edges": int(graph.sum()),
                     "runtime": time.perf_counter() - started,
                     **truth_metrics(truth, graph)})
        for weight in dagma_weights:
            model = SharedDagmaLinear("l2")
            started = time.perf_counter()
            weighted = model.fit(
                data.copy(), lambda1=.03, w_threshold=0., T=5,
                s=(1., .9, .8, .7, .6), warm_iter=3000, max_iter=6000,
                checkpoint=500,
                no_trek_pairs=pairs, trek_weight=weight, trek_function="inv",
                trek_kernel="fast")
            graph, bic, _ = postselect(data, weighted, pairs)
            rows.append({"seed": seed, "family": "dagma_notreks",
                         "setting": f"weight={weight:g}", "trek_weight": weight,
                         "bic": bic,
                         "violations": count_no_trek_violations(graph, pairs),
                         "edges": int(graph.sum()),
                         "runtime": time.perf_counter() - started,
                         **truth_metrics(truth, graph)})
        started = time.perf_counter()
        raw_flop = flopsearch.flop(data, 2.0, restarts=2)
        from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag
        flop_graph = convert_flop_cpdag(np.asarray(raw_flop), data.shape[1])
        rows.append({"seed": seed, "family": "flop", "setting": "restarts=2",
                     "runtime": time.perf_counter() - started,
                     "edges": int(np.sum(flop_graph != 0)),
                     **truth_metrics(truth, flop_graph)})
        for restarts, sweeps in flop_settings:
            started = time.perf_counter()
            result = fit_global_greedy_notreks(
                data, pairs, GlobalGreedyConfig(
                    restarts=restarts, max_sweeps=sweeps,
                    lambda_bic=2.0, seed=seed + 9137))
            graph = result.adjacency
            bic, _ = gaussian_bic(data, graph, lambda_bic=2.0)
            rows.append({"seed": seed, "family": "flop_notreks",
                         "setting": f"restarts={restarts},sweeps={sweeps}",
                         "restarts": restarts, "max_sweeps": sweeps,
                         "bic": bic,
                         "violations": count_no_trek_violations(graph, pairs),
                         "edges": int(graph.sum()),
                         "runtime": time.perf_counter() - started,
                         **truth_metrics(truth, graph)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_run.csv", index=False)
    eligible = frame[(frame.family.str.endswith("notreks")) & (frame.violations == 0)]
    summary = eligible.groupby(["family", "setting"], dropna=False).agg(
        mean_bic=("bic", "mean"), median_bic=("bic", "median"),
        mean_edges=("edges", "mean"), mean_runtime=("runtime", "mean"),
        successful_seeds=("seed", "nunique")).reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)
    dagma_eligible = eligible[eligible.family == "dagma_notreks"]
    if dagma_eligible.groupby("seed").edges.max().eq(0).any():
        raise RuntimeError(
            "DAGMA-NOTREKS calibration is structurally degenerate: at least "
            "one seed has no nonempty candidate for any tested weight")
    dagma_summary = summary[summary.family == "dagma_notreks"].copy()
    dagma_summary["strength"] = dagma_summary.setting.str.split("=").str[1].astype(float)
    best_bic = dagma_summary.mean_bic.min()
    # Numerical ties within one BIC unit are scientifically indistinguishable
    # in this deliberately tiny smoke; prefer the weaker structural penalty.
    dagma_choice = dagma_summary[
        dagma_summary.mean_bic <= best_bic + 1.0].sort_values(
            ["successful_seeds", "strength", "mean_edges"],
            ascending=[False, True, True]).iloc[0]
    flop_choice = summary[summary.family == "flop_notreks"].sort_values(
        ["successful_seeds", "mean_bic", "median_bic", "mean_runtime"],
        ascending=[False, True, True, True]).iloc[0]
    chosen = {
        "selection_rule": "lowest mean fixed-threshold Gaussian BIC among zero-violation runs",
        "dagma_trek_weight": float(dagma_choice.setting.split("=")[1]),
        "flop_notreks_restarts": int(
            flop_choice.setting.split(",")[0].split("=")[1]),
        "flop_notreks_max_sweeps": int(
            flop_choice.setting.split(",")[1].split("=")[1]),
    }
    (output_dir / "chosen.json").write_text(json.dumps(chosen, indent=2) + "\n")
    if write_defaults:
        defaults = json.loads(write_defaults.read_text())
        defaults.update(chosen)
        defaults.pop("selection_rule", None)
        defaults["calibration_status"] = "frozen_from_d20_smoke_9101_9105"
        write_defaults.write_text(json.dumps(defaults, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("chosen", json.dumps(chosen, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-defaults", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.write_defaults)


if __name__ == "__main__":
    main()
