"""Benchpress runner for canonical DAGMA-fast plus generic postprocessing."""

import json
import time

import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma_fast import (
    DagmaFastConfig,
    LinearL2Objective,
    fit_weighted_adjacency,
    resolve_lambda1,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing import (
    CompositeFeasibility,
    GaussianBICGraphScore,
    PostprocessingBudget,
    WeightedGraphEstimate,
    postprocess_weighted_graph,
)


def value(name, default):
    item = snakemake.wildcards.get(name, default)
    return default if str(item) in {"", "None", "null"} else item


started = time.perf_counter()
frame = pd.read_csv(snakemake.input["data"])
data = frame.to_numpy(dtype=float)
n, d = data.shape
resolved = resolve_lambda1(
    str(value("lambda_policy", "fixed")),
    n=n, d=d, fixed=float(value("lambda1", 0.03)))
optimized = fit_weighted_adjacency(
    LinearL2Objective(data),
    DagmaFastConfig(
        lambda1=resolved.value,
        T=int(value("T", 5)),
        warm_iter=int(value("warm_iter", 30000)),
        max_iter=int(value("max_iter", 60000)),
        lr=float(value("lr", 0.0003))))
policy = str(value("rounding_policy", "threshold_grid_score_search"))
rounded = postprocess_weighted_graph(
    WeightedGraphEstimate(
        optimized.weighted_adjacency,
        tuple(frame.columns),
        {"source_method": "dagma_fast",
         "continuous_objective": optimized.objective,
         "lambda1": resolved.value}),
    data,
    GaussianBICGraphScore(data),
    CompositeFeasibility(),
    policy,
    PostprocessingBudget(
        max_seconds=float(value("rounding_max_seconds", 1.0)),
        max_candidate_evaluations=int(
            value("rounding_max_candidate_evaluations", 1000)),
        max_local_search_iterations=int(
            value("rounding_max_iterations", 100))))

pd.DataFrame(rounded.graph, columns=frame.columns).to_csv(
    snakemake.output["adjmat"], index=False)
with open(snakemake.output["time"], "w") as handle:
    handle.write(str(time.perf_counter() - started))
with open(snakemake.output["ntests"], "w") as handle:
    handle.write("None")
if "diagnostics" in snakemake.output:
    with open(snakemake.output["diagnostics"], "w") as handle:
        json.dump({
            "method": "dagma_fast",
            "lambda_policy": resolved.policy,
            "lambda1_resolved": resolved.value,
            "n": n,
            "d": d,
            "optimizer": optimized.diagnostics,
            "rounding_policy": rounded.policy,
            "rounding": rounded.diagnostics,
        }, handle, indent=2)
