import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    is_dag,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    ProductionConfig,
    production_candidate_graph,
    run_production_pipeline,
)


def test_production_defaults_are_explicit():
    config = ProductionConfig()
    assert config.lambda1 == 0.03
    assert config.restarts == 5
    assert config.trek_function == "inv"
    assert config.trek_weight == 1
    assert config.screening_floor == 0.01


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
