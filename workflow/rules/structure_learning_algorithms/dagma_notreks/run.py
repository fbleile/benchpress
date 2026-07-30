import json
import time

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.knowledge import load_sidecar, named_pairs_to_indices
from workflow.rules.structure_learning_algorithms.dagma.shared import SharedDagmaLinear, notreks_value_grad
from workflow.rules.structure_learning_algorithms.dagma.inverse_structural import lambda1_sqrt_logd_over_n
from workflow.rules.structure_learning_algorithms.dagma_notreks.postselection import (
    LinearCandidateScorer, PostselectionConfig, lambda_policy,
    select_postselection_candidate, standardize_training_data,
)


def value(name, default):
    x = snakemake.wildcards.get(name, default)
    return default if str(x) in {"None", "null", ""} else x


df = pd.read_csv(snakemake.input["data"])
standardize_data = str(value("standardize_data", "true")).lower() == "true"
data_means = df.mean(axis=0)
data_stds = df.std(axis=0, ddof=0)
if standardize_data:
    standardized, means, safe_stds = standardize_training_data(
        df.to_numpy(dtype=float), ddof=0, std_floor=1e-12)
    df = pd.DataFrame(standardized, columns=df.columns)
else:
    raise ValueError("configured standardized run reached optimizer unstandardized")
raw_mu_schedule = value("mu_schedule", None)
mu_schedule = None
if raw_mu_schedule is not None:
    if isinstance(raw_mu_schedule, (list, tuple)):
        mu_schedule = [float(x) for x in raw_mu_schedule]
    else:
        text = str(raw_mu_schedule)
        mu_schedule = [float(x) for x in (
            json.loads(text) if text.startswith("[") else text.split(","))]
source = str(value("knowledge_source", "none"))
if source == "none":
    payload, pairs = None, []
else:
    payload = load_sidecar(snakemake.input["knowledge"], list(df.columns))
    pairs = named_pairs_to_indices(payload, list(df.columns))
threshold = float(value("w_threshold", .3))
lambda_policy_name = value("lambda_policy", None)
lambda1_scaling = str(value("lambda1_scaling", "fixed"))
if lambda_policy_name is not None:
    _, lambda1_requested, lambda1_multiplier, lambda1 = lambda_policy(
        str(lambda_policy_name), len(df.columns), len(df))
elif lambda1_scaling == "fixed":
    lambda1 = float(value("lambda1", .03))
    lambda1_requested, lambda1_multiplier = lambda1, 1.0
elif lambda1_scaling == "sqrt_logd_over_n":
    lambda1 = lambda1_sqrt_logd_over_n(
        len(df), len(df.columns),
        reference=float(value("lambda1_reference", .03)),
        reference_d=int(value("lambda1_reference_d", 50)),
        reference_n=int(value("lambda1_reference_n", 1000)))
    lambda1_requested, lambda1_multiplier = None, 1.0
else:
    raise ValueError("lambda1_scaling must be fixed or sqrt_logd_over_n")
fit_args = dict(
    lambda1=lambda1, w_threshold=0.0,
    T=int(value("T", 5)), mu_init=float(value("mu_init", 1.0)),
    mu_factor=float(value("mu_factor", .1)),
    s=[float(x) for x in str(value("s", "1.0,0.9,0.8,0.7,0.6")).split(",")],
    warm_iter=int(value("warm_iter", 30000)), max_iter=int(value("max_iter", 60000)),
    lr=float(value("lr", .0003)), checkpoint=int(value("checkpoint", 1000)),
    beta_1=float(value("beta_1", .99)), beta_2=float(value("beta_2", .999)),
    dag_penalty_weight=float(value("dag_penalty_weight", 1.0)),
    dag_constraint=str(value("dag_constraint", "logdet")),
    gamma_inv=float(value("gamma_inv", 1.0)),
    mu_schedule=mu_schedule,
    terminal_zero_stage=str(value(
        "terminal_zero_stage", "false")).lower() == "true",
    zero_block_iterations=int(value("zero_block_iterations", 10000)),
    maximum_zero_iterations=int(value("maximum_zero_iterations", 300000)),
    h_tolerance=float(value("h_tolerance", 1e-12)),
    notreks_tolerance=float(value("notreks_tolerance", 1e-12)),
    feasibility_threshold_tolerance=float(value(
        "feasibility_threshold_tolerance", 1e-6)),
    gradient_tolerance=float(value("gradient_tolerance", 1e-8)),
    no_trek_pairs=pairs, trek_weight=float(value("trek_weight", .1)),
    trek_function=str(value("trek_function", "inv")),
    trek_kernel=str(value("trek_kernel", "notreks_reference")),
    trek_log_terms=int(value("trek_log_terms", 2 * len(df.columns))),
    trek_inverse_epsilon=float(value("trek_inverse_epsilon", 1e-8)),
)
model = SharedDagmaLinear(str(value("loss_type", "l2")))
start = time.perf_counter()
W = model.fit(df.to_numpy(dtype=float), **fit_args)
elapsed = time.perf_counter() - start
raw_scaled, _ = notreks_value_grad(
    W, pairs, fit_args["trek_function"], log_terms=fit_args["trek_log_terms"],
    inverse_epsilon=fit_args["trek_inverse_epsilon"])
postselection_policy = str(value(
    "postselection_policy", "PS1_joint_feasible_greedy_score"))
threshold_grid = value("threshold_grid", [.01, .03, .05, .10, .20, .30])
if isinstance(threshold_grid, str):
    threshold_grid = json.loads(threshold_grid) if threshold_grid.startswith(
        "[") else [float(item) for item in threshold_grid.split(",")]
scorer = LinearCandidateScorer(
    df.to_numpy(dtype=float),
    regularizer_type=str(value("regularizer_type", "L1")),
    regularizer_weight=lambda1)
postselection = select_postselection_candidate(
    W, scorer=scorer,
    config=PostselectionConfig(
        policy=postselection_policy,
        candidate_edge_pool=str(value(
            "candidate_edge_pool", "threshold_grid")),
        threshold_grid=tuple(float(item) for item in threshold_grid),
        fixed_threshold=float(value("fixed_threshold", threshold)),
        max_search_seconds=float(value("max_search_seconds", 1.0)),
        max_expanded_nodes=int(value("max_expanded_nodes", 1000)),
        max_queue_size=int(value("max_queue_size", 1000)),
        max_ambiguous_edges=int(value("max_ambiguous_edges", 20)),
        max_indegree=(
            None if value("max_indegree", None) is None
            else int(value("max_indegree", None))),
        dag_constraint_active=str(value(
            "dag_constraint_active", "true")).lower() == "true",
        notreks_constraint_active=(
            str(value("notreks_constraint_active", "true")).lower() == "true"
            and bool(pairs))),
    model_class="linear_dagma", notreks_pairs=pairs)
A = postselection.adjacency
after_scaled, _ = notreks_value_grad(A.astype(float), pairs, fit_args["trek_function"],
                                     log_terms=fit_args["trek_log_terms"],
                                     inverse_epsilon=fit_args["trek_inverse_epsilon"])
reach = A.astype(bool)
np.fill_diagonal(reach, True)
for k in range(len(A)):
    reach |= reach[:, [k]] & reach[[k], :]
violations = sum(bool(np.any(reach[:, i] & reach[:, j])) for i, j in pairs)
score, _ = model._score(W)
h, _ = model._h(W, float(fit_args["s"][-1]))
scale = 2.0 / (len(A) - 1) if len(A) > 1 else 0.0
diagnostics = {
    "trek_weight": fit_args["trek_weight"], "trek_function": fit_args["trek_function"],
    "trek_kernel": fit_args["trek_kernel"],
    "data_standardized": standardize_data,
    "data_standardised": standardize_data,
    "standardisation_ddof": 0,
    "standardisation_std_floor": 1e-12,
    "training_means": means.tolist(),
    "training_standard_deviations": safe_stds.tolist(),
    "n": len(df), "d": len(df.columns),
    "lambda_policy": str(lambda_policy_name or lambda1_scaling),
    "lambda1_requested": lambda1_requested,
    "lambda1_multiplier": lambda1_multiplier,
    "lambda1_effective": lambda1,
    "regularizer_type": scorer.regularizer_type,
    "postselection": postselection.to_row(),
    "input_column_mean_max_abs_before_standardization": float(np.max(np.abs(data_means.to_numpy(dtype=float)))),
    "input_column_std_min_before_standardization": float(np.min(data_stds.to_numpy(dtype=float))),
    "input_column_std_max_before_standardization": float(np.max(data_stds.to_numpy(dtype=float))),
    "number_of_no_trek_pairs": len(pairs),
    "raw_notreks_penalty_before_threshold": raw_scaled / scale if scale else 0.0,
    "scaled_notreks_penalty_before_threshold": raw_scaled,
    "raw_notreks_penalty_after_threshold": after_scaled / scale if scale else 0.0,
    "number_of_oracle_pairs_violated_after_threshold": violations,
    "fraction_of_oracle_pairs_violated_after_threshold": violations / len(pairs) if pairs else 0.0,
    "dagma_score": score, "dagma_h_value": h, "knowledge_source": source,
}
pd.DataFrame(A, columns=df.columns).to_csv(snakemake.output["adjmat"], index=False)
with open(snakemake.output["time"], "w") as handle:
    handle.write(str(elapsed))
with open(snakemake.output["ntests"], "w") as handle:
    handle.write("None")
with open(snakemake.output["diagnostics"], "w") as handle:
    json.dump(diagnostics, handle, indent=2)
