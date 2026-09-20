"""Canonical DAGMA-NOTREKS production pipeline.

Continuous optimisation, structural screening, deletion-only parent shrinking,
and model selection intentionally remain separate steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Sequence

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.end_flop_prune import (
    end_flop_prune,
)
from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    gaussian_bic,
    is_dag,
)
from workflow.rules.structure_learning_algorithms.dagma.shared import (
    SharedDagmaLinear,
    deterministic_initial_adjacency,
)
from workflow.rules.structure_learning_algorithms.dagma.structural import (
    feasibility_thresholds,
    support_at_threshold,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.core import (
    StandardizationMetadata,
    lambda_policy,
    standardize_training_data,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.normalized_projection import (
    normalized_greedy_projection,
)


@dataclass(frozen=True)
class ProductionConfig:
    """Stable production settings; diagnostic runners may override these."""

    lambda1: float = 0.03
    # Experimental vanilla-DAGMA policies may supply a target vector or an
    # off-diagonal penalty matrix. Production configurations leave this None.
    lambda1_penalty: np.ndarray | None = None
    # The constrained objective is calibrated on the same standardized scale
    # as vanilla DAGMA.  The pooled coefficient grid selected 200 as the
    # default; callers can still override it explicitly.
    trek_weight: float = 200.0
    trek_function: str = "inv"
    trek_kernel: str = "fast"
    adjacency_map: str = "square"
    adjacency_map_tau: float = 1.0
    notreks_resolvent_normalization: bool = False
    # The kernel's internal 2/(d-1) factor is converted to the continuation-
    # aware total normalization s^2/(d-1) by an outer s^2/2 multiplier.
    notreks_stage_scaling: str = "s2_over_d_minus_1"
    restarts: int = 5
    seed: int = 1729
    initialization_scale: float = 0.05
    initialization_mode: str = "empty_random"
    initialization_edge_probability: float = 0.15
    proximal_l1: bool = False
    record_trajectory: bool = False
    screening_floor: float = 0.01
    lambda_bic: float = 2.0
    # Production uses the profiled unequal-variance Gaussian likelihood.
    loss_type: str = "gaussian_profile"
    T: int = 5
    mu_init: float = 1.0
    mu_factor: float = 0.1
    s: tuple[float, ...] = (1.1, 1.0, 0.9, 0.8, 0.7)
    warm_iter: int = 30000
    max_iter: int = 60000
    lr: float = 0.0003
    checkpoint: int = 1000
    beta_1: float = 0.99
    beta_2: float = 0.999
    standardize_data: bool = True
    standardization_ddof: int = 0
    standardization_std_floor: float = 1e-12
    lambda_policy: str = "fixed_0.03"
    regularizer_type: str = "L1"
    dag_constraint_active: bool = True
    notreks_constraint_active: bool = True
    constraint_regime: str | None = None
    edge_mask: np.ndarray | None = None
    dagma_postselection_policy: str = "feasible_parent_shrink"
    mu_schedule: tuple[float, ...] | None = (1.0, 0.3, 0.1, 0.01, 0.001)


@dataclass
class RestartResult:
    restart: int
    weighted_adjacency: np.ndarray
    candidate_graph: np.ndarray
    adjacency: np.ndarray
    coefficients: np.ndarray
    feasibility_threshold: float
    candidate_threshold: float
    exact_bic: float
    runtime: float
    candidate_edges: int
    final_edges: int
    oracle_violations: int
    stage_diagnostics: list[dict]
    postselection: dict | None = None
    lambda1_effective: float | None = None
    standardization: dict | None = None
    initialization: str = "empty"
    initial_edges: int = 0
    stage_adjacencies: list[np.ndarray] | None = None
    trajectory_times: list[float] | None = None
    selected_stage: int | None = None
    checkpoint_candidates: list["RestartResult"] | None = None


def production_candidate_graph(
    weighted_adjacency: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]],
    *,
    screening_floor: float = 0.01,
    notreks_active: bool = True,
) -> tuple[np.ndarray, dict]:
    """Return a DAG support at the requested structural-feasibility level."""
    active_pairs = tuple(no_trek_pairs) if notreks_active else ()
    feasibility = feasibility_thresholds(weighted_adjacency, active_pairs)
    threshold = max(float(feasibility["tau_feas"]), float(screening_floor))
    # Preserve the historical P3 convention: the finite feasibility threshold
    # removes its tied group with strict ``>``, while the fixed screening floor
    # uses the established fixed-threshold ``>=`` convention.
    if float(feasibility["tau_feas"]) >= float(screening_floor):
        candidate = support_at_threshold(weighted_adjacency, threshold)
    else:
        candidate = (np.abs(np.asarray(weighted_adjacency)) >= threshold).astype(int)
        np.fill_diagonal(candidate, 0)
    if not is_dag(candidate):
        raise RuntimeError("production candidate graph is cyclic")
    if notreks_active and common_ancestor_violations(candidate, active_pairs):
        raise RuntimeError("production candidate graph violates no-trek knowledge")
    return candidate, {
        "feasibility_threshold": float(feasibility["tau_feas"]),
        "candidate_threshold": threshold,
        "screening_floor": float(screening_floor),
    }


def threshold_bic_search(
    X: np.ndarray,
    weighted_adjacency: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]],
    *,
    lambda_bic: float = 2.0,
    notreks_active: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Search raw-weight thresholds, then choose the best feasible refit.

    Feasibility is monotone under threshold increases: raising the threshold
    only deletes edges.  We therefore binary-search the boundary and evaluate
    every distinct feasible support above it, because refitted BIC itself is
    not monotone in the threshold.
    """
    weights = np.abs(np.asarray(weighted_adjacency, dtype=np.float64)).copy()
    np.fill_diagonal(weights, 0.0)
    nonzero = np.unique(weights[weights > 0.0])
    if nonzero.size:
        thresholds = np.concatenate((
            [np.nextafter(float(nonzero.max()), np.inf)],
            nonzero[::-1]))
    else:
        thresholds = np.array([0.0], dtype=np.float64)
    pairs = tuple(no_trek_pairs) if notreks_active else ()

    def candidate(threshold: float) -> np.ndarray:
        graph = (weights >= float(threshold)).astype(np.uint8)
        np.fill_diagonal(graph, 0)
        return graph

    def feasible(threshold: float) -> bool:
        graph = candidate(threshold)
        return is_dag(graph) and not common_ancestor_violations(graph, pairs)

    # thresholds are descending: feasibility is true for a prefix and can
    # only become false as more lower-weight edges are admitted.
    lo, hi = 0, len(thresholds) - 1
    if not feasible(float(thresholds[lo])):
        raise RuntimeError("no feasible threshold support exists")
    last_feasible = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if feasible(float(thresholds[mid])):
            last_feasible = mid
            lo = mid + 1
        else:
            hi = mid - 1

    records = []
    best = None
    for index in range(last_feasible + 1):
        threshold = float(thresholds[index])
        graph = candidate(threshold)
        if not is_dag(graph) or common_ancestor_violations(graph, pairs):
            raise RuntimeError("threshold feasibility monotonicity failed")
        bic, coefficients = gaussian_bic(X, graph, lambda_bic=lambda_bic)
        record = {
            "threshold": threshold,
            "edges": int(graph.sum()),
            "bic": float(bic),
            "feasible": True,
        }
        records.append(record)
        key = (float(bic), -int(graph.sum()), threshold)
        if best is None or key < best[0]:
            best = (key, graph, coefficients, record)
    if best is None:
        raise RuntimeError("threshold search did not evaluate a support")
    _, graph, coefficients, winner = best
    return graph, coefficients, {
        "candidate_graph": graph.copy(),
        "candidate_edges": int(graph.sum()),
        "final_edges": int(graph.sum()),
        "candidate_threshold": float(winner["threshold"]),
        "feasibility_threshold": float(winner["threshold"]),
        "screening_floor": 0.0,
        "postprocessed_bic": float(winner["bic"]),
        "edges_deleted": int(np.count_nonzero(weights) - graph.sum()),
        "threshold_search": {
            "method": "binary_boundary_then_feasible_bic_scan",
            "threshold_count": int(len(thresholds)),
            "feasible_threshold_count": int(last_feasible + 1),
            "records": records,
        },
    }


def postprocess_weighted_adjacency(
    X: np.ndarray,
    weighted_adjacency: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]],
    *,
    screening_floor: float = 0.01,
    lambda_bic: float = 2.0,
    notreks_active: bool = True,
    postselection_policy: str = "feasible_parent_shrink",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply production screening and deletion-only fixed-order shrinking."""
    if postselection_policy in {
            "normalized_greedy_projection_refit",
            "normalized_greedy_projection_refit_shrink"}:
        projected = normalized_greedy_projection(
            weighted_adjacency, no_trek_pairs,
            dag_constraint=True, notreks_constraint=notreks_active)
        adjacency = projected.adjacency
        coefficients = gaussian_bic(
            X, adjacency, lambda_bic=lambda_bic)[1]
        exact_bic, _ = gaussian_bic(
            X, adjacency, lambda_bic=lambda_bic)
        diagnostics = {
            "candidate_graph": adjacency.copy(),
            "candidate_edges": int(adjacency.sum()),
            "candidate_threshold": 0.0,
            "feasibility_threshold": 0.0,
            "final_edges": int(adjacency.sum()),
            "edges_deleted": projected.diagnostics["edges_removed"],
            "postprocessed_bic": float(exact_bic),
            "fixed_order_parent_shrink": None,
            "normalized_projection": projected.diagnostics,
        }
        if postselection_policy.endswith("_shrink"):
            adjacency, coefficients, shrink = end_flop_prune(
                X, adjacency, lambda_bic=lambda_bic)
            exact_bic, coefficients = gaussian_bic(
                X, adjacency, lambda_bic=lambda_bic)
            diagnostics["fixed_order_parent_shrink"] = shrink
            diagnostics["final_edges"] = int(adjacency.sum())
            diagnostics["edges_deleted"] = (
                projected.diagnostics["edges_removed"]
                + int(projected.adjacency.sum() - adjacency.sum()))
            diagnostics["postprocessed_bic"] = float(exact_bic)
        if not is_dag(adjacency) or (
                notreks_active and common_ancestor_violations(
                    adjacency, no_trek_pairs)):
            raise RuntimeError("normalized projection returned infeasible graph")
        return adjacency, coefficients, diagnostics
    if postselection_policy == "threshold_bic_search":
        adjacency, coefficients, diagnostics = threshold_bic_search(
            X, weighted_adjacency, no_trek_pairs,
            lambda_bic=lambda_bic, notreks_active=notreks_active)
        if not is_dag(adjacency) or (
                notreks_active and common_ancestor_violations(
                    adjacency, no_trek_pairs)):
            raise RuntimeError("threshold search returned infeasible graph")
        return adjacency, coefficients, diagnostics
    candidate, diagnostics = production_candidate_graph(
        weighted_adjacency, no_trek_pairs, screening_floor=screening_floor,
        notreks_active=notreks_active)
    adjacency, coefficients, shrink_diagnostics = end_flop_prune(
        X, candidate, lambda_bic=lambda_bic)
    if np.any((adjacency != 0) & (candidate == 0)):
        raise RuntimeError("fixed-order parent shrink introduced an edge")
    if notreks_active and common_ancestor_violations(adjacency, no_trek_pairs):
        raise RuntimeError("edge deletion did not preserve no-trek feasibility")
    exact_bic, refit = gaussian_bic(
        X, adjacency, lambda_bic=lambda_bic)
    # Use the canonical OLS refit returned by the graph-level scorer.
    diagnostics.update({
        "candidate_graph": candidate.copy(),
        "candidate_edges": int(candidate.sum()),
        "final_edges": int(adjacency.sum()),
        "edges_deleted": int(candidate.sum() - adjacency.sum()),
        "postprocessed_bic": float(exact_bic),
        "fixed_order_parent_shrink": shrink_diagnostics,
    })
    return adjacency, refit, diagnostics


def _feasible_random_initial_adjacency(
        d: int, pairs: Sequence[tuple[int, int]], seed: int, scale: float,
        edge_probability: float) -> np.ndarray:
    """Build a data-independent DAG feasible for the supplied NOTREKS set."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    graph = np.zeros((d, d), dtype=np.uint8)
    possible = [(int(order[i]), int(order[j]))
                for i in range(d) for j in range(i + 1, d)]
    rng.shuffle(possible)
    for parent, child in possible:
        if rng.random() > edge_probability:
            continue
        graph[parent, child] = 1
        if common_ancestor_violations(graph, pairs):
            graph[parent, child] = 0
    weights = np.zeros((d, d), dtype=np.float64)
    weights[graph != 0] = rng.normal(scale=scale, size=int(graph.sum()))
    return weights


def _initial_adjacency(config: ProductionConfig, d: int, restart: int,
                       pairs: Sequence[tuple[int, int]]):
    if restart == 0:
        return np.zeros((d, d), dtype=np.float64), "empty", 0
    if config.initialization_mode == "empty_feasible_random":
        result = _feasible_random_initial_adjacency(
            d, pairs, config.seed + restart, config.initialization_scale,
            config.initialization_edge_probability)
        return result, "feasible_random_dag", int(np.count_nonzero(result))
    result = deterministic_initial_adjacency(
        d, config.seed + restart, config.initialization_scale, config.s[0])
    return result, "random_weight_matrix", 0


def run_production_pipeline(
    X: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]],
    config: ProductionConfig = ProductionConfig(),
) -> tuple[RestartResult, list[RestartResult]]:
    """Run five deterministic restarts and select postprocessed minimum BIC."""
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array")
    if config.restarts < 1:
        raise ValueError("restarts must be positive")
    if config.initialization_mode not in {"empty_random", "empty_feasible_random"}:
        raise ValueError("unsupported DAGMA initialization mode")
    if not 0.0 <= config.initialization_edge_probability <= 1.0:
        raise ValueError("initialization edge probability must lie in [0, 1]")
    if not config.standardize_data:
        raise ValueError(
            "production standardized run reached optimizer without "
            "standardization")
    data, means, stds = standardize_training_data(
        data, ddof=config.standardization_ddof,
        std_floor=config.standardization_std_floor)
    standardization = StandardizationMetadata(
        True, config.standardization_ddof,
        config.standardization_std_floor, tuple(means), tuple(stds),
        len(data), data.shape[1])
    _, requested, multiplier, effective_lambda = lambda_policy(
        config.lambda_policy, data.shape[1], len(data))
    # A literal custom lambda remains available through fixed_0.03 only when
    # explicitly equal; registry policies otherwise determine the value.
    if config.lambda_policy == "fixed_0.03" and config.lambda1 != .03:
        effective_lambda = float(config.lambda1)
    if config.lambda1_penalty is not None:
        effective_lambda = np.asarray(config.lambda1_penalty, dtype=float)
    results: list[RestartResult] = []
    for restart in range(config.restarts):
        started = perf_counter()
        model = SharedDagmaLinear(loss_type=config.loss_type, verbose=False)
        initial_W, initialization, initial_edges = _initial_adjacency(
            config, data.shape[1], restart, no_trek_pairs)
        model.fit(
            data.copy(),
            no_trek_pairs=no_trek_pairs,
            trek_weight=config.trek_weight,
            trek_function=config.trek_function,
            trek_kernel=config.trek_kernel,
            adjacency_map=config.adjacency_map,
            adjacency_map_tau=config.adjacency_map_tau,
            diagnostic_edge_threshold=config.screening_floor,
            notreks_resolvent_normalization=(
                config.notreks_resolvent_normalization),
            notreks_stage_scaling=config.notreks_stage_scaling,
            proximal_l1=config.proximal_l1,
            initial_W=initial_W,
            lambda1=effective_lambda,
            w_threshold=0.0,
            T=config.T,
            mu_init=config.mu_init,
            mu_factor=config.mu_factor,
            s=config.s,
            warm_iter=config.warm_iter,
            max_iter=config.max_iter,
            lr=config.lr,
            checkpoint=config.checkpoint,
            beta_1=config.beta_1,
            beta_2=config.beta_2,
            edge_mask=config.edge_mask,
            mu_schedule=config.mu_schedule,
        )
        weighted = np.asarray(model.W_est, dtype=np.float64).copy()
        adjacency, coefficients, diagnostics = postprocess_weighted_adjacency(
            data, weighted, no_trek_pairs,
            screening_floor=config.screening_floor,
            lambda_bic=config.lambda_bic,
            notreks_active=config.notreks_constraint_active,
            postselection_policy=config.dagma_postselection_policy)
        postselection_row = {
            "postselection_policy": config.dagma_postselection_policy,
            "candidate_score": diagnostics["postprocessed_bic"],
            "feasible": True,
        }
        if not is_dag(adjacency):
            raise RuntimeError("postselection returned a cyclic graph")
        if (config.notreks_constraint_active
                and common_ancestor_violations(adjacency, no_trek_pairs)):
            raise RuntimeError(
                "postselection returned a graph violating no-trek knowledge")
        result = RestartResult(
            restart=restart,
            weighted_adjacency=weighted,
            candidate_graph=diagnostics["candidate_graph"],
            adjacency=adjacency,
            coefficients=coefficients,
            feasibility_threshold=diagnostics["feasibility_threshold"],
            candidate_threshold=diagnostics["candidate_threshold"],
            exact_bic=diagnostics["postprocessed_bic"],
            runtime=perf_counter() - started,
            candidate_edges=diagnostics["candidate_edges"],
            final_edges=diagnostics["final_edges"],
            oracle_violations=common_ancestor_violations(
                adjacency, no_trek_pairs),
            stage_diagnostics=list(model.stage_diagnostics),
            postselection=postselection_row,
            lambda1_effective=(
                float(effective_lambda)
                if np.asarray(effective_lambda).ndim == 0
                else float(np.mean(effective_lambda))),
            standardization=asdict(standardization),
            initialization=initialization,
            initial_edges=initial_edges,
            stage_adjacencies=[matrix.copy()
                              for matrix in getattr(
                                  model, "stage_adjacencies", [])],
            trajectory_times=(
                [0.0] + [float(item["elapsed_seconds"])
                 for item in getattr(model, "trajectory_metadata", [])]
                if config.record_trajectory else None),
        )
        if config.dagma_postselection_policy == "best_projected_checkpoint":
            checkpoint_candidates = []
            for stage_index, stage_matrix in enumerate(model.stage_adjacencies, 1):
                stage_adjacency, stage_coefficients, stage_diag = (
                    postprocess_weighted_adjacency(
                        data, stage_matrix, no_trek_pairs,
                        screening_floor=config.screening_floor,
                        lambda_bic=config.lambda_bic,
                        notreks_active=config.notreks_constraint_active,
                        postselection_policy="feasible_parent_shrink"))
                checkpoint_candidates.append(replace(
                    result,
                    weighted_adjacency=np.asarray(stage_matrix).copy(),
                    candidate_graph=stage_diag["candidate_graph"],
                    adjacency=stage_adjacency,
                    coefficients=stage_coefficients,
                    feasibility_threshold=stage_diag["feasibility_threshold"],
                    candidate_threshold=stage_diag["candidate_threshold"],
                    exact_bic=stage_diag["postprocessed_bic"],
                    candidate_edges=stage_diag["candidate_edges"],
                    final_edges=stage_diag["final_edges"],
                    oracle_violations=common_ancestor_violations(
                        stage_adjacency, no_trek_pairs),
                    postselection={
                        "postselection_policy": "feasible_parent_shrink",
                        "candidate_score": stage_diag["postprocessed_bic"],
                        "feasible": True,
                        "selected_stage": stage_index,
                    },
                    selected_stage=stage_index,
                    checkpoint_candidates=None,
                ))
            result = replace(result, checkpoint_candidates=checkpoint_candidates)
        results.append(result)
    checkpoint_results = [
        candidate
        for result in results
        for candidate in (result.checkpoint_candidates or [])]
    selection_pool = checkpoint_results if (
        config.dagma_postselection_policy == "best_projected_checkpoint") \
        else results
    selected = min(selection_pool, key=lambda result: (
        result.exact_bic,
        result.final_edges,
        -result.candidate_threshold,
        result.restart,
    ))
    return selected, results
