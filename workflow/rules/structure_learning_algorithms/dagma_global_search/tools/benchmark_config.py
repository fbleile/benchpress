"""Validated configuration for the production NOTREKS benchmarks.

The current benchmark deliberately covers one model regime: acyclic,
linear, equal-variance Gaussian SCMs.  Other data generators remain useful
for future work, but are not silently presented as supported solver regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SUPPORTED_METHODS = (
    "flop",
    "flop_notreks_edge_mask",
    "flop_parent_shrink",
    "flop_edge_mask_parent_shrink",
    "flop_notreks_postselection",
    "flop_notreks_order_postselection",
    "flop_notreks",
    "flop_notreks_local",
    "flop_notreks_active_exact",
    "flop_notreks_active_reinsert",
    "flop_notreks_greedy",
    "flop_notreks_active_exact_edge_mask",
    "flop_notreks_local_adaptive",
    "flop_notreks_order_guided",
    "flop_notreks_trekcut",
    "flop_notreks_prefix_feasible",
    "flop_notreks_trek_dominance",
    "flop_notreks_adaptive_soft_completion",
    "flop_notreks_source_signature",
    "flop_notreks_random_order",
    "dagma",
    "dagma_edge_mask",
    "dagma_postselection",
    "dagma_edge_mask_postselection",
    "dagma_notreks",
    "dagma_notreks_edge_mask",
    "dagma_proximal",
    "dagma_notreks_proximal",
    "dagma_normalized_projection",
    "dagma_normalized_projection_shrink",
    "dagma_notreks_normalized_projection",
    "dagma_notreks_normalized_projection_shrink",
    "dagma_proximal_normalized_projection",
    "dagma_proximal_normalized_projection_shrink",
    "dagma_notreks_proximal_normalized_projection",
    "dagma_notreks_proximal_normalized_projection_shrink",
    "dagma_best_projected_checkpoint",
    "dagma_notreks_best_projected_checkpoint",
    "dagma_stable",
    "dagma_notreks_stable",
    "dagma_notreks_normalized",
    "dagma_threshold_bic_search",
    "dagma_coeff_old",
    "dagma_coeff_calibrated",
    "dagma_coeff_learned",
    "dagma_coeff_density_adaptive",
    "dagma_coeff_density_continuous",
    "dagma_notreks_coeff_old",
    "dagma_notreks_coeff_calibrated",
    "dagma_notreks_coeff_learned",
    "dagma_notreks_coeff_density_adaptive",
    "dagma_notreks_s2_over_i",
    "dagma_notreks_s2_over_i_calibrated",
    "var_sortnregress",
    "r2_sortnregress",
)

DEFAULT_METHODS = (
    "flop",
    "flop_notreks",
    "dagma",
    "dagma_notreks",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    d: int = 20
    n: int | None = None
    graph_type: str = "er2"
    scm: str = "linear"
    noise: str = "gaussian"
    equal_variance: bool = False
    knowledge_fraction: float = 0.25
    corrupted_knowledge_fraction: float = 0.0
    attempts: int = 5
    flop_sweeps: int = 16
    flop_lambda_bic: float = 2.0
    flop_search_version: str = "local_greedy_rust"
    flop_local_passes: int = 8
    flop_order_guided_lex_fraction: float = 0.5
    flop_order_guided_repair_candidates: int = 2
    flop_order_guided_coverage_starts: int = 1
    flop_trekcut_oracle_budget: int = 32
    flop_trekcut_refinement_passes: int = 4
    flop_prefix_beam_width: int = 1
    flop_source_signature_polish_outputs: int = 1
    dagma_stages: int = 5
    dagma_warm_iter: int = 30000
    dagma_max_iter: int = 60000
    dagma_trek_weight: float = 1.0
    dagma_initialization_mode: str = "empty_random"
    dagma_initialization_edge_probability: float = 0.15
    dagma_map_tau: float = 1.0
    dagma_record_trajectory: bool = False
    dagma_mu_schedule: tuple[float, ...] | None = (1.0, 0.3, 0.1, 0.01, 0.001)
    dagma_s_schedule: tuple[float, ...] | None = (1.1, 1.0, 0.9, 0.8, 0.7)

    @property
    def effective_n(self) -> int:
        return 10 * self.d if self.n is None else self.n

    def validate(self, methods: Iterable[str]) -> None:
        methods = tuple(methods)
        unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
        if unknown:
            raise ValueError(f"unknown benchmark methods: {unknown}")
        if self.d < 2 or self.effective_n < 1:
            raise ValueError("d must be at least 2 and n must be positive")
        if self.scm != "linear" or self.noise != "gaussian":
            raise ValueError(
                "the production benchmark currently supports only "
                "linear equal-variance Gaussian SCMs")
        if not 0.0 <= self.knowledge_fraction <= 1.0:
            raise ValueError("knowledge_fraction must lie in [0, 1]")
        if not 0.0 <= self.corrupted_knowledge_fraction <= 1.0:
            raise ValueError(
                "corrupted_knowledge_fraction must lie in [0, 1]")
        if self.attempts < 1 or self.flop_sweeps < 1:
            raise ValueError("attempts and flop_sweeps must be positive")
        if self.flop_local_passes < 1:
            raise ValueError("flop_local_passes must be positive")
        if self.dagma_initialization_mode not in {
                "empty_random", "empty_feasible_random"}:
            raise ValueError("unsupported DAGMA initialization mode")
        if not 0.0 <= self.dagma_initialization_edge_probability <= 1.0:
            raise ValueError(
                "DAGMA initialization edge probability must lie in [0, 1]")
        for name, schedule in (("mu", self.dagma_mu_schedule),
                               ("s", self.dagma_s_schedule)):
            if schedule is not None and (
                    not schedule or any(float(value) <= 0 for value in schedule)):
                raise ValueError(f"DAGMA {name} schedule must be positive")
        if not 0.0 <= self.flop_order_guided_lex_fraction <= 1.0:
            raise ValueError("order-guided lex fraction must lie in [0, 1]")
        if self.flop_order_guided_repair_candidates < 0:
            raise ValueError("order-guided repair candidates must be nonnegative")
        if self.flop_order_guided_coverage_starts < 0:
            raise ValueError("order-guided coverage starts must be nonnegative")
        if self.flop_trekcut_oracle_budget < 1:
            raise ValueError("TrekCut oracle budget must be positive")
        if self.flop_trekcut_refinement_passes < 1:
            raise ValueError("TrekCut refinement passes must be positive")
        if self.flop_prefix_beam_width not in {1, 4}:
            raise ValueError("prefix beam width must be 1 or 4")
        if self.flop_source_signature_polish_outputs < 1:
            raise ValueError("source-signature polish outputs must be positive")
        if self.flop_search_version not in {
            "global_greedy_rust", "global_greedy_rust_optimized",
            "local_greedy_rust",
        }:
            raise ValueError("unsupported FLOP search version")
