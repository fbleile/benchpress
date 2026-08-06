import json
import time

try:
    import flopsearch
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "FLOP-NOTREKS requires the Linux flopsearch package in the active "
        "Python environment. In host mode run "
        "bash scripts/install_flopsearch_host.sh before starting the farm."
    ) from exc
import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.knowledge import (
    load_sidecar, named_pairs_to_indices,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.adapter import (
    count_no_trek_violations,
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

strategy = "global_greedy_rust"
seed = (int(snakemake.wildcards["seed"]) + int(snakemake.wildcards["algorithm_seed"])) % (2**64)
restarts = snakemake.wildcards["restarts"]
if str(restarts) in {"None", "null"}:
    raise ValueError("FLOP-NOTREKS requires a restart count")

start = time.perf_counter()
raw, diagnostics = flopsearch.flop_notreks(
    X, float(snakemake.wildcards["lambda_bic"]), pairs,
    restarts=int(restarts),
    seed=seed,
    max_signature_rounds=int(snakemake.wildcards.get("max_sweeps", 4)),
    search_version="global_greedy_rust",
    return_diagnostics=True,
    return_dag=False,
)
selected_dag = np.zeros((X.shape[1], X.shape[1]), dtype=np.uint8)
for parent, child in diagnostics["selected_dag_edges"]:
    selected_dag[int(parent), int(child)] = 1
A = np.asarray(raw).astype(np.uint8)
diagnostics["search_strategy"] = strategy
elapsed = time.perf_counter() - start
violations = count_no_trek_violations(selected_dag, pairs)
if violations:
    raise RuntimeError(f"selected FLOP-NOTREKS DAG has {violations} no-trek violations")
if int(diagnostics["final_no_trek_violation_count"]) != violations:
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
