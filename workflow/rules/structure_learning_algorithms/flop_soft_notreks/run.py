import json
import pandas as pd
from workflow.rules.structure_learning_algorithms.dagma.knowledge import load_sidecar, named_pairs_to_indices
from workflow.rules.structure_learning_algorithms.flop_soft_notreks import SoftGreedyConfig, fit_soft_notreks

def value(name, default):
    x = snakemake.wildcards.get(name, default)
    return default if str(x) in {"", "None", "null"} else x

frame = pd.read_csv(snakemake.input["data"])
pairs = []
if snakemake.input.get("knowledge"):
    pairs = named_pairs_to_indices(load_sidecar(str(snakemake.input["knowledge"]), list(frame.columns)), list(frame.columns))
cfg = SoftGreedyConfig(
    restarts=int(value("restarts", 8)), max_sweeps=int(value("max_sweeps", 12)),
    lazy_top_k=int(value("lazy_top_k", 12)),
    soft_notreks_weight=float(value("soft_notreks_weight", 100.0)),
    lambda_bic=float(value("lambda_bic", 2.0)),
    seed=int(value("random_seed", value("seed", 0))))
result = fit_soft_notreks(frame.to_numpy(float), pairs, cfg)
pd.DataFrame(result.adjacency, columns=frame.columns).to_csv(snakemake.output["adjmat"], index=False)
open(snakemake.output["time"], "w").write(str(result.runtime_seconds))
open(snakemake.output["ntests"], "w").write(str(result.support_evaluations))
if "diagnostics" in snakemake.output:
    with open(snakemake.output["diagnostics"], "w") as handle:
        json.dump({"bic": result.bic, "notreks": result.notreks, "score": result.score,
                   "raw_bic": result.raw_bic, "raw_notreks": result.raw_notreks,
                   "support_evaluations": result.support_evaluations,
                   "cache_hits": result.cache_hits, "distinct_supports": result.distinct_supports}, handle, indent=2)
