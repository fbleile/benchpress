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
from workflow.rules.structure_learning_algorithms.flop_notreks.chromatic_sources import (
    fit_chromatic_source_flop,
    fit_chromatic_source_notreks,
    fit_chromatic_source_fixed_notreks,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.local_window_ablation import (
    LocalWindowConfig, fit_local_window_ablation,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.gflop import (
    GFlopConfig, HardCandidate, fit_gflop,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.gflop_dagma_backend import (
    DagmaBackendConfig, DagmaOrderBackend,
)
from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic, topological_order,
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
    "search_strategy", snakemake.wildcards.get("search_version", "global_greedy_rust")))
seed = (int(snakemake.wildcards["seed"]) + int(snakemake.wildcards["algorithm_seed"])) % (2**64)
restarts = snakemake.wildcards["restarts"]
if str(restarts) in {"None", "null"}:
    raise ValueError("FLOP-NOTREKS requires a restart count")

start = time.perf_counter()
verification_pairs = pairs
if strategy in {"global_greedy_rust", "global_greedy_cached", "global_greedy_parallel", "global_greedy_hybrid"}:
    raw, diagnostics = flopsearch.flop_notreks(
        X, float(snakemake.wildcards["lambda_bic"]), pairs,
        restarts=int(restarts),
        seed=seed,
        max_signature_rounds=int(snakemake.wildcards.get("max_sweeps", 4)),
        search_version=strategy,
        return_diagnostics=True,
        return_dag=False,
    )
elif strategy == "incremental_promotion_d":
    raw, diagnostics = flopsearch.flop_notreks(
        X, float(snakemake.wildcards["lambda_bic"]), pairs,
        restarts=int(restarts),
        seed=seed,
        max_signature_rounds=int(snakemake.wildcards.get("max_sweeps", 100)),
        search_version="incremental_promotion_d",
        return_diagnostics=True,
        return_dag=False,
    )
elif strategy == "chromatic_source_prefix_flop":
    raw, diagnostics = fit_chromatic_source_flop(
        X,
        pairs,
        lambda_bic=float(snakemake.wildcards["lambda_bic"]),
        restarts=int(restarts),
        seed=seed,
        max_chromatic_search_nodes=int(
            snakemake.wildcards.get("max_chromatic_search_nodes", 2_000_000)),
    )
elif strategy == "global_greedy_chromatic_sources":
    raw, diagnostics = fit_chromatic_source_notreks(
        X,
        pairs,
        lambda_bic=float(snakemake.wildcards["lambda_bic"]),
        restarts=int(restarts),
        seed=seed,
        max_sweeps=int(snakemake.wildcards.get("max_sweeps", 4)),
        max_chromatic_search_nodes=int(
            snakemake.wildcards.get("max_chromatic_search_nodes", 2_000_000)),
    )
elif strategy == "fixed_signature_chromatic_sources":
    raw, diagnostics = fit_chromatic_source_fixed_notreks(
        X,
        pairs,
        lambda_bic=float(snakemake.wildcards["lambda_bic"]),
        restarts=int(restarts),
        seed=seed,
        max_chromatic_search_nodes=int(
            snakemake.wildcards.get("max_chromatic_search_nodes", 2_000_000)),
    )
elif strategy == "local_window_ablation":
    result = fit_local_window_ablation(
        X, pairs, LocalWindowConfig(
            block_size=int(snakemake.wildcards.get("block_size", 4)),
            sweeps=int(snakemake.wildcards.get("max_sweeps", 1)),
            initial_flop_runs=int(snakemake.wildcards.get(
                "initial_flop_runs", max(1, int(restarts) + 1))),
            seed=seed,
            lambda_bic=float(snakemake.wildcards["lambda_bic"]),
            candidate_max_parents=int(snakemake.wildcards.get(
                "candidate_max_parents", 3)),
            max_families_per_node=int(snakemake.wildcards.get(
                "max_families_per_node", 24)),
            block_time_limit_seconds=float(snakemake.wildcards.get(
                "block_time_limit_seconds", 2.0)),
            overall_time_limit_seconds=float(snakemake.wildcards.get(
                "overall_time_limit_seconds", 60.0)),
            max_block_calls=int(snakemake.wildcards.get(
                "max_block_calls", 1000))))
    raw = result.adjacency
    diagnostics = {
        key: value for key, value in result.__dict__.items()
        if key not in {"adjacency", "block_diagnostics"}
    }
    diagnostics["block_diagnostics"] = result.block_diagnostics
    diagnostics["selected_dag_edges"] = [
        (int(parent), int(child)) for parent, child in np.argwhere(result.adjacency)]
    diagnostics["selected_dag_edge_count"] = int(result.adjacency.sum())
    diagnostics["final_bic"] = result.score
    diagnostics["final_no_trek_violation_count"] = result.no_trek_violations
elif strategy == "gflop":
    use_notreks = bool(snakemake.wildcards.get("use_notreks", True))
    verification_pairs = pairs if use_notreks else []
    backend = DagmaOrderBackend(
        X, pairs if use_notreks else (), DagmaBackendConfig(
            lambda1=float(snakemake.wildcards.get("gflop_lambda1", .03)),
            learning_rate=float(snakemake.wildcards.get("gflop_lr", .0003)),
            seed=seed))
    initial_order = tuple(range(X.shape[1]))
    initial_state = None
    feasible_initializations = []
    if str(snakemake.wildcards.get("gflop_initialization", "flop")) == "flop":
        _, flop_diagnostics = flopsearch.flop_notreks(
            X, float(snakemake.wildcards["lambda_bic"]), [],
            restarts=int(restarts), seed=seed, max_signature_rounds=0,
            search_version="fixed_signature_a", return_diagnostics=True)
        flop_graph = np.zeros((X.shape[1], X.shape[1]), dtype=np.uint8)
        for parent, child in flop_diagnostics["selected_dag_edges"]:
            flop_graph[int(parent), int(child)] = 1
        initial_order = tuple(topological_order(flop_graph))
        flop_score, initial_state = gaussian_bic(
            X, flop_graph, lambda_bic=float(snakemake.wildcards["lambda_bic"]))
        if count_no_trek_violations(flop_graph, pairs if use_notreks else ()) == 0:
            feasible_initializations.append(HardCandidate(
                flop_graph, float(backend.scorer.score(flop_graph)),
                {"source": "vanilla_flop", "gaussian_bic": float(flop_score)}))
    continuation = tuple(float(value) for value in snakemake.wildcards.get(
        "gflop_continuation_weights", [0., .1, 1., 10.]))
    result = fit_gflop(
        backend, pairs, GFlopConfig(
            continuation_weights=continuation,
            sweeps_per_stage=int(snakemake.wildcards.get("max_sweeps", 1)),
            screening_budget=int(snakemake.wildcards.get(
                "gflop_screening_budget", 50)),
            refinement_budget=int(snakemake.wildcards.get(
                "gflop_refinement_budget", 300)),
            refinement_top_k=int(snakemake.wildcards.get(
                "gflop_refinement_top_k", 2)),
            finalist_postselection_seconds=float(snakemake.wildcards.get(
                "gflop_finalist_postselection_seconds", .05)),
            postselection_seconds=float(snakemake.wildcards.get(
                "gflop_postselection_seconds", 1.)),
            seed=seed, use_notreks=use_notreks,
            return_if_initial_feasible=bool(snakemake.wildcards.get(
                "gflop_return_if_initial_feasible", False))),
        initial_order=initial_order, initial_state=initial_state,
        feasible_initializations=feasible_initializations)
    raw = result.adjacency
    diagnostics = {
        key: value for key, value in result.__dict__.items()
        if key not in {"adjacency", "edge_strengths"}}
    diagnostics["selected_dag_edges"] = [
        (int(parent), int(child)) for parent, child in np.argwhere(result.adjacency)]
    diagnostics["selected_dag_edge_count"] = int(result.adjacency.sum())
    diagnostics["final_bic"] = result.target_score
    diagnostics["final_no_trek_violation_count"] = result.no_trek_violations
else:
    raise ValueError(f"unsupported FLOP-NOTREKS search strategy {strategy!r}")
selected_dag = np.zeros((X.shape[1], X.shape[1]), dtype=np.uint8)
for parent, child in diagnostics["selected_dag_edges"]:
    selected_dag[int(parent), int(child)] = 1
A = np.asarray(raw).astype(np.uint8)
diagnostics["search_strategy"] = strategy
elapsed = time.perf_counter() - start
violations = count_no_trek_violations(selected_dag, verification_pairs)
hard_feasible_strategies = {
    "global_greedy_rust",
    "global_greedy_cached",
    "global_greedy_parallel",
    "global_greedy_hybrid",
    "global_greedy_chromatic_sources",
    "fixed_signature_chromatic_sources",
    "incremental_promotion_d",
    "gflop",
    "local_window_ablation",
}
if strategy in hard_feasible_strategies and violations:
    raise RuntimeError(f"selected FLOP-NOTREKS DAG has {violations} no-trek violations")
if (strategy in hard_feasible_strategies
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
