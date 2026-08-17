import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import is_dag
from workflow.rules.structure_learning_algorithms.flop_notreks.gflop import (
    BackendResult, GFlopConfig, HardCandidate, fit_gflop, order_mask,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.gflop_dagma_backend import (
    DagmaBackendConfig, DagmaOptimizerState, DagmaOrderBackend,
)


class CoupledBackend:
    """Nondecomposable toy: two edges are valuable only jointly."""
    dimension = 3

    def __init__(self):
        self.calls = 0

    def local_score(self, *args, **kwargs):
        raise AssertionError("generic gFLOP must never call a local score")

    def initialize(self, order, incumbent=None):
        return np.zeros((3, 3)) if incumbent is None else np.array(incumbent, copy=True)

    def transport(self, state, old_order, new_order):
        del old_order
        return np.asarray(state) * order_mask(new_order)

    def optimize(self, order, state, *, penalty_weight, budget):
        del budget
        self.calls += 1
        W = np.asarray(state, dtype=float).copy() * order_mask(order)
        mask = order_mask(order)
        # This is deliberately global/nondecomposable: the coupled pair enters
        # together, although either edge alone would have positive cost.
        if mask[0, 1] and mask[0, 2]:
            W[0, 1] = W[0, 2] = 1.
        base = float(W.sum() - 5 * (W[0, 1] != 0 and W[0, 2] != 0))
        violation_proxy = float(abs(W[0, 1] * W[0, 2]))
        if penalty_weight >= 10:
            W[0, 2] = 0.
            base = float(W.sum())
            violation_proxy = 0.
        return BackendResult(W, abs(W), base, 0., violation_proxy,
                             {"global_coupling": True})

    def hard_candidates(self, result, order, *, use_notreks, budget_seconds):
        del order, budget_seconds
        graph = (result.edge_strengths > .5).astype(np.uint8)
        if use_notreks:
            graph[0, 2] = 0
        return [HardCandidate(graph, float(graph.sum()))]


def small_config(**changes):
    values = dict(continuation_weights=(0., 10.), sweeps_per_stage=1,
                  screening_budget=1, refinement_budget=1,
                  refinement_top_k=1, postselection_seconds=.01,
                  archive_size=3, seed=7)
    values.update(changes)
    return GFlopConfig(**values)


def test_complete_order_mask_is_dense_forward_and_acyclic():
    for order in ((0, 1, 2, 3), (2, 0, 3, 1)):
        mask = order_mask(order)
        assert mask.sum() == 6
        assert is_dag(mask)


def test_absent_edges_can_enter_and_score_can_be_nondecomposable():
    backend = CoupledBackend()
    result = fit_gflop(backend, [], small_config(use_notreks=False),
                       initial_state=np.zeros((3, 3)))
    assert result.edge_strengths[0, 1] > 0
    assert result.edge_strengths[0, 2] > 0
    assert backend.calls > 0


def test_global_refit_crosses_local_window_counterexample():
    # With only child 1 active, the one-edge score is +1, so a window update
    # deletes 0->1. A full refit can simultaneously use 0->1 and 0->2 and
    # obtains their coupled score 2 - 5 = -3.
    local_window_state = np.zeros((3, 3))
    local_window_state[0, 1] = 1
    if local_window_state.sum() > 0:
        local_window_state[0, 1] = 0
    assert not local_window_state.any()
    result = fit_gflop(CoupledBackend(), [], small_config(use_notreks=False))
    assert result.continuation_trace[-1]["base_score"] == -3.


def test_temporary_infeasibility_then_hard_feasible_result():
    result = fit_gflop(CoupledBackend(), [(1, 2)], small_config())
    assert result.continuation_trace[0]["notreks_penalty"] > 0
    assert result.continuation_trace[-1]["notreks_penalty"] == 0
    assert result.no_trek_violations == 0


def test_protected_incumbent_is_never_worsened():
    incumbent = np.zeros((3, 3), dtype=np.uint8)
    result = fit_gflop(
        CoupledBackend(), [(1, 2)], small_config(),
        feasible_initializations=[HardCandidate(incumbent, -100.)])
    assert result.target_score == -100.
    assert not result.adjacency.any()


def test_deterministic_and_unconstrained_modes():
    first = fit_gflop(CoupledBackend(), [(1, 2)], small_config(use_notreks=False))
    second = fit_gflop(CoupledBackend(), [(1, 2)], small_config(use_notreks=False))
    np.testing.assert_array_equal(first.adjacency, second.adjacency)
    assert first.continuation_trace == second.continuation_trace
    assert first.use_notreks is False


def test_dagma_backend_has_no_hidden_initializer_support_mask():
    rng = np.random.default_rng(44)
    data = rng.normal(size=(250, 3))
    data[:, 1] += 1.5 * data[:, 0]
    backend = DagmaOrderBackend(
        data, [], DagmaBackendConfig(
            learning_rate=.003, checkpoint=20, seed=4))
    state = backend.initialize((0, 1, 2), np.zeros((3, 3)))
    optimized = backend.optimize(
        (0, 1, 2), state, penalty_weight=0., budget=100)
    assert optimized.edge_strengths[0, 1] > .1
    assert optimized.diagnostics["acyclicity_penalty_weight"] == 0.
    assert optimized.diagnostics["allowed_edges"] == 3
    assert optimized.diagnostics["maximum_forbidden_weight"] == 0.
    assert isinstance(optimized.state, DagmaOptimizerState)
    assert optimized.state.optimizer_step == 100
    transported = backend.transport(
        optimized.state, (0, 1, 2), (2, 0, 1))
    assert transported.optimizer_step == optimized.state.optimizer_step
    assert not np.any(transported.weights[order_mask((2, 0, 1)) == 0])
    assert not np.any(transported.adam_m[order_mask((2, 0, 1)) == 0])
