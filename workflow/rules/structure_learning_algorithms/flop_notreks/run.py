import json
import time
from dataclasses import asdict

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
from workflow.rules.structure_learning_algorithms.flop_notreks.global_greedy import (
    GlobalGreedyConfig, fit_global_greedy_notreks,
)
from workflow.rules.structure_learning_algorithms.notreks import subsample_no_trek_pairs


df = pd.read_csv(snakemake.input["data"])
X = df.to_numpy(dtype=float)
means, stds = X.mean(0), X.std(0, ddof=0)
if np.any(stds == 0):
    raise ValueError("FLOP-NOTREKS cannot standardize a constant column")
X = (X - means) / stds

knowledge = snakemake.input.get("knowledge")
if knowledge:
    payload = load_sidecar(str(knowledge), list(df.columns))
    pairs = named_pairs_to_indices(payload, list(df.columns))
else:
    pairs = []
number_of_oracle_pairs = len(pairs)
pairs = subsample_no_trek_pairs(
    pairs,
    float(snakemake.wildcards.get("knowledge_fraction", 1.0)),
    int(snakemake.wildcards.get("knowledge_seed", 0))
    + int(snakemake.wildcards.get("seed", 0)),
)

strategy = str(snakemake.wildcards.get(
    "search_strategy", "signature_alternating"))
kwargs = {
    "seed": (int(snakemake.wildcards["seed"]) + int(snakemake.wildcards["algorithm_seed"]))
            % (2**64),
    "signature_top_k": int(snakemake.wildcards["signature_top_k"]),
    "max_signature_rounds": int(snakemake.wildcards["max_signature_rounds"]),
    "search_version": str(snakemake.wildcards["search_version"]),
    "return_diagnostics": True,
}
restarts = snakemake.wildcards["restarts"]
search_timeout = snakemake.wildcards["search_timeout"]
if str(restarts) not in {"None", "null"}:
    kwargs["restarts"] = int(restarts)
elif str(search_timeout) not in {"None", "null"}:
    kwargs["timeout"] = float(search_timeout)
else:
    raise ValueError("FLOP-NOTREKS requires restarts or search_timeout")

start = time.perf_counter()
if strategy == "global_greedy":
    if str(restarts) in {"None", "null"}:
        raise ValueError("global-greedy FLOP-NOTREKS requires restart count")
    result = fit_global_greedy_notreks(
        X, pairs, GlobalGreedyConfig(
            restarts=int(restarts),
            max_sweeps=int(snakemake.wildcards.get("max_sweeps", 4)),
            lambda_bic=float(snakemake.wildcards["lambda_bic"]),
            seed=kwargs["seed"]))
    selected_dag = result.adjacency
    A = selected_dag.copy()
    diagnostics = asdict(result)
    diagnostics.pop("adjacency", None)
    diagnostics["search_strategy"] = strategy
else:
    raw, diagnostics = flopsearch.flop_notreks(
        X, float(snakemake.wildcards["lambda_bic"]), pairs, **kwargs)
    raw = np.asarray(raw)
    selected_dag = selected_dag_from_diagnostics(diagnostics, X.shape[1])
    A = convert_flop_cpdag(raw, X.shape[1])
    diagnostics["search_strategy"] = strategy
elapsed = time.perf_counter() - start
violations = count_no_trek_violations(selected_dag, pairs)
if violations:
    raise RuntimeError(f"selected FLOP-NOTREKS DAG has {violations} no-trek violations")
if (strategy != "global_greedy"
        and int(diagnostics["final_no_trek_violation_count"]) != violations):
    raise RuntimeError("Rust and Benchpress no-trek violation diagnostics disagree")

pd.DataFrame(A.astype(int), columns=df.columns).to_csv(
    snakemake.output["adjmat"], index=False)
diagnostics.update({
    "method_id": str(snakemake.wildcards.get("id", "flop_notreks")),
    "seed": int(snakemake.wildcards.get("seed", 0)),
    "runtime": elapsed,
    "number_of_oracle_pairs": number_of_oracle_pairs,
    "knowledge_fraction": float(
        snakemake.wildcards.get("knowledge_fraction", 1.0)),
    "number_of_supplied_constraints": len(pairs),
    "final_no_trek_violation_count": violations,
    "fraction_of_oracle_pairs_violated_after_threshold": (
        violations / len(pairs) if pairs else 0.0),
    "selected_dag_edge_count": int(selected_dag.sum()),
})
with open(snakemake.output["diagnostics"], "w", encoding="utf-8") as handle:
    json.dump(diagnostics, handle, indent=2)
    handle.write("\n")
with open(snakemake.output["time"], "w", encoding="utf-8") as handle:
    handle.write(str(elapsed))
with open(snakemake.output["ntests"], "w", encoding="utf-8") as handle:
    handle.write("None")
