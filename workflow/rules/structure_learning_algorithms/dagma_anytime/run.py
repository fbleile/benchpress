import json
import os

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma_anytime.solver import (
    fit_linear_dagma_anytime,
    write_result_json,
)


def _value(name, default):
    value = snakemake.wildcards.get(name, default)
    return default if str(value) in {"None", "null", ""} else value


def _bool(name, default=False):
    return str(_value(name, str(default).lower())).lower() == "true"


def _float_list(name, default=None):
    value = _value(name, default)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value)
    if text in {"None", "null", ""}:
        return None
    return [float(x) for x in (json.loads(text) if text.startswith("[") else text.split(","))]


blas_threads = _value("blas_threads", None)
if blas_threads is not None:
    for env_name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[env_name] = str(int(blas_threads))

df = pd.read_csv(snakemake.input["data"])
method = str(_value("solver_variant", _value("id", "dagma_vanilla_anytime")))
snapshots = _float_list("snapshot_times_seconds", None)
result = fit_linear_dagma_anytime(
    df.to_numpy(dtype=float),
    method=method,
    lambda1=float(_value("lambda1", 0.03)),
    w_threshold=float(_value("w_threshold", 0.3)),
    T=int(_value("T", 5)),
    mu_init=float(_value("mu_init", 1.0)),
    mu_factor=float(_value("mu_factor", 0.1)),
    s=[float(x) for x in str(_value("s", "1.0,0.9,0.8,0.7,0.6")).split(",")],
    warm_iter=int(_value("warm_iter", 30000)),
    max_iter=int(_value("max_iter", 60000)),
    lr=float(_value("lr", 0.0003)),
    checkpoint=int(_value("checkpoint", 1000)),
    beta_1=float(_value("beta_1", 0.99)),
    beta_2=float(_value("beta_2", 0.999)),
    max_runtime_seconds=None if _value("max_runtime_seconds", None) is None else float(_value("max_runtime_seconds", None)),
    snapshot_times_seconds=snapshots,
    random_seed=int(_value("random_seed", 0)),
    precision_policy=None if _value("precision_policy", None) is None else str(_value("precision_policy", None)),
    checkpoint_interval=None if _value("checkpoint_interval", None) is None else int(_value("checkpoint_interval", None)),
    zero_diagonal=_bool("zero_diagonal", False),
    proximal_l1=_bool("proximal_l1", False),
    adaptive_precision=_bool("adaptive_precision", False),
)

pd.DataFrame(result.adjacency_thresholded_raw.astype(int), columns=df.columns).to_csv(
    snakemake.output["adjmat"], index=False
)
with open(snakemake.output["time"], "w") as handle:
    handle.write(str(result.elapsed_seconds))
with open(snakemake.output["ntests"], "w") as handle:
    handle.write("None")
if "diagnostics" in snakemake.output:
    write_result_json(result, snakemake.output["diagnostics"], extra={"columns": list(df.columns)})
