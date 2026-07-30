import json
import time

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
raw, diagnostics = flopsearch.flop_notreks(
    X, float(snakemake.wildcards["lambda_bic"]), pairs, **kwargs)
elapsed = time.perf_counter() - start
raw = np.asarray(raw)
selected_dag = selected_dag_from_diagnostics(diagnostics, X.shape[1])
violations = count_no_trek_violations(selected_dag, pairs)
if violations:
    raise RuntimeError(f"selected FLOP-NOTREKS DAG has {violations} no-trek violations")
if int(diagnostics["final_no_trek_violation_count"]) != violations:
    raise RuntimeError("Rust and Benchpress no-trek violation diagnostics disagree")

A = convert_flop_cpdag(raw, X.shape[1])
pd.DataFrame(A.astype(int), columns=df.columns).to_csv(
    snakemake.output["adjmat"], index=False)
diagnostics.update({
    "runtime": elapsed,
    "number_of_supplied_constraints": len(pairs),
    "final_no_trek_violation_count": violations,
    "selected_dag_edge_count": int(selected_dag.sum()),
})
with open(snakemake.output["diagnostics"], "w", encoding="utf-8") as handle:
    json.dump(diagnostics, handle, indent=2)
    handle.write("\n")
with open(snakemake.output["time"], "w", encoding="utf-8") as handle:
    handle.write(str(elapsed))
with open(snakemake.output["ntests"], "w", encoding="utf-8") as handle:
    handle.write("None")
