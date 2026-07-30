import json
import time

import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.knowledge import load_sidecar, named_pairs_to_indices
from workflow.rules.structure_learning_algorithms.pc_mi_oracle.pc import MIOracleFisherZ, stable_pc

df = pd.read_csv(snakemake.input["data"])
source = str(snakemake.wildcards["knowledge_source"])
if source == "none":
    pairs = []
else:
    payload = load_sidecar(snakemake.input["knowledge"], list(df.columns))
    pairs = named_pairs_to_indices(payload, list(df.columns))
alpha = float(snakemake.wildcards["alpha"])
test = MIOracleFisherZ(df.to_numpy(), pairs, alpha)
start = time.perf_counter()
A, counts = stable_pc(df.to_numpy(), test, int(snakemake.wildcards["max_cond_set"]))
elapsed = time.perf_counter() - start
pd.DataFrame(A, columns=df.columns).to_csv(snakemake.output["adjmat"], index=False)
with open(snakemake.output["time"], "w") as handle:
    handle.write(str(elapsed))
with open(snakemake.output["ntests"], "w") as handle:
    handle.write(str(counts.statistical_conditional_tests))
with open(snakemake.output["diagnostics"], "w") as handle:
    json.dump({
        "number_of_oracle_marginal_queries": counts.oracle_marginal_queries,
        "number_of_oracle_independent_answers": counts.oracle_independent_answers,
        "number_of_oracle_dependent_answers": counts.oracle_dependent_answers,
        "number_of_statistical_conditional_tests": counts.statistical_conditional_tests,
        "alpha": alpha, "pc_variant": "stable",
    }, handle, indent=2)
