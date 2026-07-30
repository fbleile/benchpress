import numpy as np

from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing import (
    CallableGraphScore,
    CompositeFeasibility,
    DagConstraint,
    NoTreksConstraint,
    PostprocessingBudget,
    WeightedGraphEstimate,
    postprocess_weighted_graph,
    postprocess_weighted_graphs,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.candidate_generation import (
    generate_candidates,
)


def edge_distance_score(target):
    return CallableGraphScore(
        lambda graph, _: float(np.sum(np.abs(graph - target))))


def test_identical_weights_ignore_source_method_metadata():
    W = np.zeros((4, 4))
    W[0, 1], W[1, 2], W[0, 3] = .8, .7, .2
    score = edge_distance_score((np.abs(W) >= .1).astype(int))
    constraint = CompositeFeasibility()
    outputs = [
        postprocess_weighted_graph(
            WeightedGraphEstimate(W, metadata={"source_method": method}),
            None, score, constraint,
            budget=PostprocessingBudget(max_candidate_evaluations=20))
        for method in ("dagma", "notears", "mock_nonlinear")]
    for output in outputs[1:]:
        np.testing.assert_array_equal(output.graph, outputs[0].graph)
        assert output.score == outputs[0].score


def test_weight_magnitude_orders_candidates_but_score_controls_acceptance():
    W = np.zeros((3, 3))
    W[0, 1], W[1, 2], W[0, 2] = .9, .8, .2
    target = np.zeros((3, 3), dtype=int)
    target[0, 2] = 1
    result = postprocess_weighted_graph(
        WeightedGraphEstimate(W), None, edge_distance_score(target),
        CompositeFeasibility(), "weighted_feasible_local_search",
        PostprocessingBudget(
            max_seconds=2, max_candidate_evaluations=100,
            max_local_search_iterations=10))
    np.testing.assert_array_equal(result.graph, target)
    assert result.diagnostics["feasible"]


def test_dag_and_mock_notreks_constraints_are_enforced():
    W = np.zeros((4, 4))
    W[0, 2], W[1, 2], W[2, 3], W[3, 0] = .9, .8, .7, .1
    constraints = CompositeFeasibility(
        (DagConstraint(), NoTreksConstraint(((0, 1),))))
    result = postprocess_weighted_graph(
        W, None, edge_distance_score(np.zeros((4, 4))),
        constraints, "threshold_grid_score_search",
        PostprocessingBudget(max_candidate_evaluations=50))
    assert constraints.is_feasible(result.graph)
    assert result.diagnostics["notreks_violation_count"] == 0


def test_budget_and_deterministic_tie_breaking():
    W = np.full((4, 4), .2)
    np.fill_diagonal(W, 0)
    kwargs = dict(
        data=None,
        score=edge_distance_score(np.zeros((4, 4))),
        constraints=CompositeFeasibility(),
        policy="weighted_feasible_local_search",
        budget=PostprocessingBudget(
            max_seconds=1, max_candidate_evaluations=3,
            max_local_search_iterations=2))
    first = postprocess_weighted_graph(W, **kwargs)
    second = postprocess_weighted_graph(W, **kwargs)
    np.testing.assert_array_equal(first.graph, second.graph)
    assert all(
        candidate.diagnostics.get("moves_evaluated", 0) <= 3
        for candidate in first.candidates)


def test_multiple_weighted_estimates_use_truth_free_score():
    target = np.zeros((3, 3), dtype=int)
    target[0, 1] = 1
    weak = np.zeros((3, 3))
    strong = weak.copy()
    strong[0, 1] = .8
    result = postprocess_weighted_graphs(
        [WeightedGraphEstimate(weak), WeightedGraphEstimate(strong)],
        None, edge_distance_score(target), CompositeFeasibility(),
        budget=PostprocessingBudget(max_candidate_evaluations=20))
    np.testing.assert_array_equal(result.graph, target)
    assert result.source_estimate_index == 1


def test_candidate_generation_uses_threshold_sequence():
    W = np.zeros((3, 3))
    W[0, 1], W[1, 2] = .8, .04
    candidates = generate_candidates(
        WeightedGraphEstimate(W), CompositeFeasibility(),
        thresholds=(.01, .1))
    assert [candidate.graph.sum() for candidate in candidates] == [2, 1]
