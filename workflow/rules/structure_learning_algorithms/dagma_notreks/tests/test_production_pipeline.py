import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    is_dag,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    ProductionConfig,
    _feasible_random_initial_adjacency,
    production_candidate_graph,
    run_production_pipeline,
)
from workflow.rules.structure_learning_algorithms.dagma.shared import soft_threshold


def test_soft_thresholding_has_exact_zeros_and_preserves_signs():
    values = np.array([-2.0, -0.5, 0.0, 0.25, 3.0])
    result = soft_threshold(values, 0.5)
    assert np.array_equal(result, np.array([-1.5, -0.0, 0.0, 0.0, 2.5]))


def test_production_defaults_are_explicit():
    config = ProductionConfig()
    assert config.lambda1 == 0.03
    assert config.restarts == 5
    assert config.trek_function == "inv"
    assert config.trek_weight == 200
    assert config.screening_floor == 0.01
    assert not hasattr(config, "postselection_policy")


def test_candidate_uses_feasibility_threshold_or_floor():
    weighted = np.zeros((4, 4))
    weighted[0, 1] = 0.5
    weighted[1, 0] = 0.005  # removed by the fixed screening floor
    weighted[2, 3] = 0.4
    candidate, diagnostics = production_candidate_graph(
        weighted, [(0, 2)], screening_floor=0.01)
    assert diagnostics["candidate_threshold"] >= 0.01
    assert is_dag(candidate)
    assert common_ancestor_violations(candidate, [(0, 2)]) == 0


def test_fixed_screening_floor_preserves_historical_inclusive_semantics():
    weighted = np.zeros((3, 3))
    weighted[0, 1] = 0.01
    candidate, diagnostics = production_candidate_graph(
        weighted, [], screening_floor=0.01)
    assert diagnostics["feasibility_threshold"] == 0
    assert candidate[0, 1] == 1


def test_inactive_notreks_screening_does_not_use_supplied_pairs():
    weighted = np.zeros((3, 3))
    weighted[0, 2] = 0.8
    weighted[2, 1] = 0.7
    candidate, diagnostics = production_candidate_graph(
        weighted, [(0, 1)], screening_floor=0.01,
        notreks_active=False)
    assert diagnostics["feasibility_threshold"] == 0.0
    assert candidate[0, 2] == 1
    assert candidate[2, 1] == 1
    assert common_ancestor_violations(candidate, [(0, 1)]) == 1


def test_restart_selection_uses_postprocessed_bic(monkeypatch):
    import workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline as pipeline

    bics = iter([3.0, 1.0, 2.0])

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.stage_diagnostics = []

        def fit(self, X, **kwargs):
            self.W_est = np.zeros((X.shape[1], X.shape[1]))

    def fake_postprocess(X, W, pairs, **kwargs):
        bic = next(bics)
        graph = np.zeros_like(W, dtype=int)
        return graph, W.copy(), {
            "feasibility_threshold": 0.0,
            "candidate_threshold": 0.01,
            "postprocessed_bic": bic,
            "candidate_graph": graph.copy(),
            "candidate_edges": 0,
            "final_edges": 0,
        }

    monkeypatch.setattr(pipeline, "SharedDagmaLinear", FakeModel)
    monkeypatch.setattr(pipeline, "postprocess_weighted_adjacency",
                        fake_postprocess)
    selected, results = run_production_pipeline(
        np.zeros((5, 3)), [], ProductionConfig(restarts=3))
    assert len(results) == 3
    assert selected.restart == 1


def test_feasible_random_initialization_is_deterministic_and_feasible():
    first = _feasible_random_initial_adjacency(
        5, [(0, 1)], seed=123, scale=0.05, edge_probability=0.8)
    second = _feasible_random_initial_adjacency(
        5, [(0, 1)], seed=123, scale=0.05, edge_probability=0.8)
    assert np.array_equal(first, second)
    assert is_dag(first != 0)
    assert common_ancestor_violations(first != 0, [(0, 1)]) == 0
    assert np.all(np.diag(first) == 0)
