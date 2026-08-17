"""Linear DAGMA/PSTrek backend for the generic gFLOP order engine."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.shared import SharedDagmaLinear
from workflow.rules.structure_learning_algorithms.dagma.structural import (
    feasibility_thresholds, support_at_threshold,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.postselection import (
    LinearCandidateScorer, PostselectionConfig, select_postselection_candidate,
    standardize_training_data, lambda_policy,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.gflop import (
    BackendResult, HardCandidate, order_mask,
)


@dataclass(frozen=True)
class DagmaBackendConfig:
    lambda_policy: str = "fixed_0.03"
    lambda1: float = .03
    loss_type: str = "l2"
    regularizer_type: str = "L1"
    learning_rate: float = .0003
    checkpoint: int = 100
    beta_1: float = .99
    beta_2: float = .999
    trek_function: str = "inv"
    trek_kernel: str = "fast"
    initialization_scale: float = .02
    seed: int = 1729
    threshold_grid: tuple[float, ...] = (.01, .03, .05, .1, .2, .3)
    postselection_max_nodes: int = 1000
    postselection_max_queue: int = 1000


@dataclass
class DagmaOptimizerState:
    weights: np.ndarray
    adam_m: np.ndarray | float = 0.
    adam_v: np.ndarray | float = 0.
    optimizer_step: int = 0


class DagmaOrderBackend:
    """Full-matrix fixed-order optimizer; no acyclicity penalty is used."""

    def __init__(self, data, pairs=(), config=DagmaBackendConfig()):
        self.data, _, _ = standardize_training_data(np.asarray(data, dtype=float))
        self.dimension = self.data.shape[1]
        self.pairs = tuple((int(a), int(b)) for a, b in pairs)
        self.config = config
        _, _, _, effective_lambda = lambda_policy(
            config.lambda_policy, self.dimension, len(self.data))
        if config.lambda_policy == "fixed_0.03" and config.lambda1 != .03:
            effective_lambda = float(config.lambda1)
        self.lambda1 = float(effective_lambda)
        self.scorer = LinearCandidateScorer(
            self.data, regularizer_type=config.regularizer_type,
            regularizer_weight=self.lambda1)
        self.optimize_calls = 0

    def initialize(self, order: Sequence[int], incumbent=None):
        mask = order_mask(order)
        if incumbent is not None:
            if isinstance(incumbent, DagmaOptimizerState):
                state = incumbent
                weights = np.asarray(state.weights, dtype=float).copy()
                adam_m = np.asarray(state.adam_m, dtype=float) * mask
                adam_v = np.asarray(state.adam_v, dtype=float) * mask
                step = state.optimizer_step
            else:
                weights = np.asarray(incumbent, dtype=float).copy()
                adam_m = adam_v = 0.
                step = 0
            if weights.shape != mask.shape:
                raise ValueError("initial backend state has the wrong shape")
            return DagmaOptimizerState(weights * mask, adam_m, adam_v, step)
        rng = np.random.default_rng(self.config.seed)
        return DagmaOptimizerState(rng.normal(
            scale=self.config.initialization_scale, size=mask.shape) * mask)

    def transport(self, state, old_order, new_order):
        del old_order
        state = self.initialize(new_order, state)
        return state

    def optimize(self, order, state, *, penalty_weight, budget):
        started = perf_counter()
        mask = order_mask(order)
        state = self.initialize(order, state)
        exclusions = [tuple(map(int, edge)) for edge in np.argwhere(mask == 0)
                      if edge[0] != edge[1]]
        model = SharedDagmaLinear(self.config.loss_type, verbose=False)
        model.preserve_optimizer_state = True
        model.opt_m = np.asarray(state.adam_m, dtype=float).copy()
        model.opt_v = np.asarray(state.adam_v, dtype=float).copy()
        model.optimizer_step = int(state.optimizer_step)
        model.fit(
            self.data.copy(), initial_W=state.weights * mask,
            no_trek_pairs=self.pairs, trek_weight=float(penalty_weight),
            trek_function=self.config.trek_function,
            trek_kernel=self.config.trek_kernel,
            dag_penalty_weight=0., lambda1=self.lambda1,
            w_threshold=0., T=1, mu_init=1., mu_factor=1., s=(1.,),
            warm_iter=int(budget), max_iter=int(budget),
            lr=self.config.learning_rate, checkpoint=max(1, min(
                self.config.checkpoint, int(budget))),
            beta_1=self.config.beta_1, beta_2=self.config.beta_2,
            exclude_edges=exclusions)
        weights = np.asarray(model.W_est, dtype=float) * mask
        base_score = float(model._score(weights)[0])
        regularizer = float(self.lambda1 * np.abs(weights).sum())
        penalty = float(model._trek_kernel.value_grad(
            weights, self.config.trek_function,
            log_terms=model.trek_log_terms,
            inverse_epsilon=model.trek_inverse_epsilon)[0])
        self.optimize_calls += 1
        return BackendResult(
            state=DagmaOptimizerState(
                weights, np.asarray(model.opt_m).copy(),
                np.asarray(model.opt_v).copy(), int(model.optimizer_step)),
            edge_strengths=np.abs(weights),
            base_score=base_score, regularizer=regularizer,
            notreks_penalty=penalty,
            diagnostics={
                "runtime": perf_counter() - started,
                "iterations": int(budget), "acyclicity_penalty_weight": 0.,
                "optimizer_step": int(model.optimizer_step),
                "stopped_by_tolerance": bool(
                    model.stage_diagnostics[-1]["stopped_by_tolerance"]),
                "iterations_performed": int(
                    model.stage_diagnostics[-1]["iterations_performed"]),
                "allowed_edges": int(mask.sum()),
                "maximum_forbidden_weight": float(
                    np.abs(weights[mask == 0]).max(initial=0.)),
            })

    def hard_candidates(self, result, order, *, use_notreks, budget_seconds):
        del order
        strengths = np.asarray(result.edge_strengths, dtype=float)
        items = []
        if use_notreks and self.pairs:
            threshold = float(feasibility_thresholds(
                strengths, self.pairs)["tau_feas"])
            graph = support_at_threshold(strengths, threshold)
            score = self.scorer.score(graph)
            items.append(HardCandidate(
                graph.astype(np.uint8), score,
                {"policy": "smallest_feasible_threshold",
                 "threshold": threshold}))
        selection = select_postselection_candidate(
            strengths, scorer=self.scorer,
            config=PostselectionConfig(
                policy="PS3_joint_feasible_budgeted_search",
                candidate_edge_pool="threshold_grid",
                threshold_grid=self.config.threshold_grid,
                max_search_seconds=float(budget_seconds),
                max_expanded_nodes=self.config.postselection_max_nodes,
                max_queue_size=self.config.postselection_max_queue,
                dag_constraint_active=True,
                notreks_constraint_active=bool(use_notreks and self.pairs)),
            model_class="linear_dagma", notreks_pairs=(
                self.pairs if use_notreks else ()))
        items.append(HardCandidate(
            selection.adjacency.astype(np.uint8),
            float(selection.candidate_score), selection.to_row()))
        return items

    def quick_hard_candidates(self, result, order, *, use_notreks):
        """Threshold/refit only; reserve branch-and-bound for termination."""
        del order
        strengths = np.asarray(result.edge_strengths, dtype=float)
        if use_notreks and self.pairs:
            threshold = float(feasibility_thresholds(
                strengths, self.pairs)["tau_feas"])
            graph = support_at_threshold(strengths, threshold)
            return [HardCandidate(
                graph.astype(np.uint8), self.scorer.score(graph),
                {"policy": "smallest_feasible_threshold",
                 "threshold": threshold, "quick": True})]
        candidates = []
        for threshold in self.config.threshold_grid:
            graph = (strengths >= threshold).astype(np.uint8)
            np.fill_diagonal(graph, 0)
            candidates.append(HardCandidate(
                graph, self.scorer.score(graph),
                {"policy": "threshold_grid_refit", "threshold": threshold,
                 "quick": True}))
        return [min(candidates, key=lambda item: (
            item.target_score, int(item.adjacency.sum()),
            item.adjacency.tobytes()))]
