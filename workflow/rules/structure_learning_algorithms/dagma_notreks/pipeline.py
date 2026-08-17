"""Canonical DAGMA-NOTREKS production pipeline.

Continuous optimisation, structural screening, deletion-only parent shrinking,
and model selection intentionally remain separate steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from workflow.rules.structure_learning_algorithms.dagma_notreks.postselection import (
    LinearCandidateScorer,
    PostselectionConfig,
    StandardizationMetadata,
    lambda_policy,
    select_postselection_candidate,
    standardize_training_data,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.flop_support import (
    build_flop_union_support,
    excluded_edges,
)


@dataclass(frozen=True)
class ProductionConfig:
    """Stable production settings; diagnostic runners may override these."""

    lambda1: float = 0.03
    # The constrained objective is calibrated on the same standardized scale
    # as vanilla DAGMA.  A weight of 10 overwhelms the data term and caused
    # the NOTREKS variant to collapse on dense graphs; 1 is the production
    # baseline used by the benchmark calibration.
    trek_weight: float = 1.0
    trek_function: str = "inv"
    trek_kernel: str = "fast"
    restarts: int = 5
    seed: int = 1729
    initialization_scale: float = 0.05
    screening_floor: float = 0.01
    lambda_bic: float = 2.0
    loss_type: str = "l2"
    T: int = 5
    mu_init: float = 1.0
    mu_factor: float = 0.1
    s: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6)
    warm_iter: int = 30000
    max_iter: int = 60000
    lr: float = 0.0003
    # Relative objective tolerance used by the Adam inner loop.  Production
    # callers may set this to zero when they explicitly want the full budget.
    optimizer_tol: float = 1e-6
    checkpoint: int = 1000
    beta_1: float = 0.99
    beta_2: float = 0.999
    standardize_data: bool = True
    standardization_ddof: int = 0
    standardization_std_floor: float = 1e-12
    lambda_policy: str = "fixed_0.03"
    regularizer_type: str = "L1"
    # Internal compatibility default for existing Python callers.  The public
    # production CLI explicitly defaults to PS1.
    postselection_policy: str = "legacy_fixed_order_parent_shrink"
    candidate_edge_pool: str = "threshold_grid"
    threshold_grid: tuple[float, ...] = (
        0.01, 0.03, 0.05, 0.10, 0.20, 0.30)
    fixed_threshold: float = 0.30
    max_search_seconds: float = 1.0
    max_expanded_nodes: int = 1000
    max_queue_size: int = 1000
    max_ambiguous_edges: int = 20
    max_indegree: int | None = None
    dag_constraint_active: bool = True
    notreks_constraint_active: bool = True
    constraint_regime: str | None = None
    support_mode: str = "unrestricted"
    flop_support_runs: int = 2
    flop_support_seed_stride: int = 7919
    support_initialization: str = "zero_then_best_feasible_flop"


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
    gaussian_bic: float
    runtime: float
    candidate_edges: int
    final_edges: int
    oracle_violations: int
    stage_diagnostics: list[dict]
    postselection: dict | None = None
    lambda1_effective: float | None = None
    standardization: dict | None = None
    support: dict | None = None


def production_candidate_graph(
    weighted_adjacency: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]],
    *,
    screening_floor: float = 0.01,
) -> tuple[np.ndarray, dict]:
    """Return support at max(minimum joint-feasibility threshold, floor)."""
    feasibility = feasibility_thresholds(weighted_adjacency, no_trek_pairs)
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
    if common_ancestor_violations(candidate, no_trek_pairs):
        raise RuntimeError("production candidate graph violates no-trek knowledge")
    return candidate, {
        "feasibility_threshold": float(feasibility["tau_feas"]),
        "candidate_threshold": threshold,
        "screening_floor": float(screening_floor),
    }


def postprocess_weighted_adjacency(
    X: np.ndarray,
    weighted_adjacency: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]],
    *,
    screening_floor: float = 0.01,
    lambda_bic: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply production screening and deletion-only fixed-order shrinking."""
    candidate, diagnostics = production_candidate_graph(
        weighted_adjacency, no_trek_pairs, screening_floor=screening_floor)
    adjacency, coefficients, shrink_diagnostics = end_flop_prune(
        X, candidate, lambda_bic=lambda_bic)
    if np.any((adjacency != 0) & (candidate == 0)):
        raise RuntimeError("fixed-order parent shrink introduced an edge")
    if common_ancestor_violations(adjacency, no_trek_pairs):
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


def _initial_adjacency(config: ProductionConfig, d: int, restart: int,
                       allowed_arcs: np.ndarray,
                       feasible_flop_coefficients: np.ndarray | None):
    if restart == 0:
        return np.zeros((d, d), dtype=np.float64)
    if config.support_initialization == "zero":
        return np.zeros((d, d), dtype=np.float64)
    if (restart == 1
            and config.support_initialization == "zero_then_best_feasible_flop"
            and feasible_flop_coefficients is not None):
        return np.asarray(feasible_flop_coefficients, dtype=np.float64).copy()
    initial = deterministic_initial_adjacency(
        d, config.seed + restart, config.initialization_scale, config.s[0])
    return initial * allowed_arcs


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
    if config.support_mode not in {"unrestricted", "flop_union_support"}:
        raise ValueError("support_mode must be unrestricted or flop_union_support")
    if config.support_initialization not in {
            "zero", "zero_then_best_feasible_flop", "zero_then_masked_random"}:
        raise ValueError("unsupported support_initialization")
    allowed_arcs = np.ones((data.shape[1], data.shape[1]), dtype=np.uint8)
    np.fill_diagonal(allowed_arcs, 0)
    support_diagnostics = {
        "support_mode": "unrestricted",
        "allowed_directed_arcs": int(allowed_arcs.sum()),
        "support_density": 1.0,
        "support_is_symmetrized": False,
        "support_build_seconds": 0.0,
    }
    feasible_flop_coefficients = None
    if config.support_mode == "flop_union_support":
        support_started = perf_counter()
        support = build_flop_union_support(
            data, no_trek_pairs, runs=config.flop_support_runs,
            seed=config.seed, seed_stride=config.flop_support_seed_stride,
            lambda_bic=config.lambda_bic)
        allowed_arcs = support.allowed_arcs
        feasible_flop_coefficients = support.best_feasible_coefficients
        support_diagnostics = support.diagnostics
        support_diagnostics["support_build_seconds"] = (
            perf_counter() - support_started)
    exclusions = excluded_edges(allowed_arcs)
    results: list[RestartResult] = []
    for restart in range(config.restarts):
        started = perf_counter()
        model = SharedDagmaLinear(loss_type=config.loss_type, verbose=False)
        model.fit(
            data.copy(),
            no_trek_pairs=no_trek_pairs,
            trek_weight=config.trek_weight,
            trek_function=config.trek_function,
            trek_kernel=config.trek_kernel,
            initial_W=_initial_adjacency(
                config, data.shape[1], restart, allowed_arcs,
                feasible_flop_coefficients),
            lambda1=effective_lambda,
            w_threshold=0.0,
            T=config.T,
            mu_init=config.mu_init,
            mu_factor=config.mu_factor,
            s=config.s,
            warm_iter=config.warm_iter,
            max_iter=config.max_iter,
            lr=config.lr,
            tol=config.optimizer_tol,
            checkpoint=config.checkpoint,
            beta_1=config.beta_1,
            beta_2=config.beta_2,
            exclude_edges=exclusions,
        )
        weighted = np.asarray(model.W_est, dtype=np.float64).copy()
        outside_support = ((allowed_arcs == 0)
                           & ~np.eye(data.shape[1], dtype=bool))
        if np.any((weighted != 0) & outside_support):
            raise RuntimeError("DAGMA optimizer escaped the FLOP support mask")
        if config.postselection_policy == "legacy_fixed_order_parent_shrink":
            adjacency, coefficients, diagnostics = (
                postprocess_weighted_adjacency(
                    data, weighted, no_trek_pairs,
                    screening_floor=config.screening_floor,
                    lambda_bic=config.lambda_bic))
            postselection_row = {
                "postselection_policy":
                    "legacy_fixed_order_parent_shrink",
                "candidate_score": diagnostics["postprocessed_bic"],
                "feasible": True,
            }
        else:
            scorer = LinearCandidateScorer(
                data, regularizer_type=config.regularizer_type,
                regularizer_weight=effective_lambda)
            postselection = select_postselection_candidate(
                weighted, scorer=scorer,
                config=PostselectionConfig(
                    policy=config.postselection_policy,
                    candidate_edge_pool=config.candidate_edge_pool,
                    threshold_grid=config.threshold_grid,
                    fixed_threshold=config.fixed_threshold,
                    max_search_seconds=config.max_search_seconds,
                    max_expanded_nodes=config.max_expanded_nodes,
                    max_queue_size=config.max_queue_size,
                    max_ambiguous_edges=config.max_ambiguous_edges,
                    max_indegree=config.max_indegree,
                    dag_constraint_active=config.dag_constraint_active,
                    notreks_constraint_active=(
                        config.notreks_constraint_active
                        and bool(no_trek_pairs))),
                model_class="linear_dagma", notreks_pairs=no_trek_pairs)
            adjacency = postselection.adjacency
            coefficients = scorer.coefficients(adjacency)
            diagnostics = {
                "candidate_graph": adjacency.copy(),
                "candidate_edges": postselection.predicted_edges,
                "final_edges": postselection.predicted_edges,
                "postprocessed_bic": postselection.candidate_score,
                "candidate_threshold": postselection.selected_threshold,
                "feasibility_threshold": postselection.selected_threshold,
            }
            postselection_row = postselection.to_row()
        if np.any((adjacency != 0) & (allowed_arcs == 0)):
            raise RuntimeError("postselection introduced an edge outside the FLOP support")
        gaussian_bic_score, _ = gaussian_bic(
            data, adjacency, lambda_bic=config.lambda_bic)
        results.append(RestartResult(
            restart=restart,
            weighted_adjacency=weighted,
            candidate_graph=diagnostics["candidate_graph"],
            adjacency=adjacency,
            coefficients=coefficients,
            feasibility_threshold=diagnostics["feasibility_threshold"],
            candidate_threshold=diagnostics["candidate_threshold"],
            exact_bic=diagnostics["postprocessed_bic"],
            gaussian_bic=float(gaussian_bic_score),
            runtime=perf_counter() - started,
            candidate_edges=diagnostics["candidate_edges"],
            final_edges=diagnostics["final_edges"],
            oracle_violations=common_ancestor_violations(
                adjacency, no_trek_pairs),
            stage_diagnostics=list(model.stage_diagnostics),
            postselection=postselection_row,
            lambda1_effective=effective_lambda,
            standardization=asdict(standardization),
            support=dict(support_diagnostics),
        ))
    selected = min(results, key=lambda result: (
        result.exact_bic,
        result.final_edges,
        -result.candidate_threshold,
        result.restart,
    ))
    return selected, results
