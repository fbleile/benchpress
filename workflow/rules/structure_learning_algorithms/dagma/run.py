import time
import json

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.shared import SharedDagmaLinear
from workflow.rules.structure_learning_algorithms.dagma.inverse_structural import lambda1_sqrt_logd_over_n


def _value(name, default):
    value = snakemake.wildcards.get(name, default)
    return default if str(value) in {"None", "null", ""} else value


df = pd.read_csv(snakemake.input["data"])
raw_mu_schedule = _value("mu_schedule", None)
mu_schedule = None
if raw_mu_schedule is not None:
    if isinstance(raw_mu_schedule, (list, tuple)):
        mu_schedule = [float(x) for x in raw_mu_schedule]
    else:
        text = str(raw_mu_schedule)
        mu_schedule = [float(x) for x in (
            json.loads(text) if text.startswith("[") else text.split(","))]
lambda1_scaling = str(_value("lambda1_scaling", "fixed"))
if lambda1_scaling == "fixed":
    lambda1 = float(_value("lambda1", .03))
elif lambda1_scaling == "sqrt_logd_over_n":
    lambda1 = lambda1_sqrt_logd_over_n(
        len(df), len(df.columns),
        reference=float(_value("lambda1_reference", .03)),
        reference_d=int(_value("lambda1_reference_d", 50)),
        reference_n=int(_value("lambda1_reference_n", 1000)))
else:
    raise ValueError("lambda1_scaling must be fixed or sqrt_logd_over_n")
kwargs = dict(
    lambda1=lambda1,
    w_threshold=float(_value("w_threshold", .3)),
    T=int(_value("T", 5)),
    mu_init=float(_value("mu_init", 1.0)),
    mu_factor=float(_value("mu_factor", .1)),
    s=[float(x) for x in str(_value("s", "1.0,0.9,0.8,0.7,0.6")).split(",")],
    warm_iter=int(_value("warm_iter", 30000)),
    max_iter=int(_value("max_iter", 60000)),
    lr=float(_value("lr", .0003)),
    checkpoint=int(_value("checkpoint", 1000)),
    beta_1=float(_value("beta_1", .99)),
    beta_2=float(_value("beta_2", .999)),
    dag_penalty_weight=float(_value("dag_penalty_weight", 1.0)),
    mu_schedule=mu_schedule,
    terminal_zero_stage=str(_value(
        "terminal_zero_stage", "false")).lower() == "true",
    zero_block_iterations=int(_value("zero_block_iterations", 10000)),
    maximum_zero_iterations=int(_value("maximum_zero_iterations", 300000)),
    h_tolerance=float(_value("h_tolerance", 1e-12)),
    notreks_tolerance=float(_value("notreks_tolerance", 1e-12)),
    feasibility_threshold_tolerance=float(_value(
        "feasibility_threshold_tolerance", 1e-6)),
    gradient_tolerance=float(_value("gradient_tolerance", 1e-8)),
)
start = time.perf_counter()
model = SharedDagmaLinear(str(_value("loss_type", "l2")))
W = model.fit(df.to_numpy(dtype=float, copy=True), **kwargs)
elapsed = time.perf_counter() - start
pd.DataFrame((W != 0).astype(int), columns=df.columns).to_csv(snakemake.output["adjmat"], index=False)
with open(snakemake.output["time"], "w") as handle:
    handle.write(str(elapsed))
with open(snakemake.output["ntests"], "w") as handle:
    handle.write("None")
