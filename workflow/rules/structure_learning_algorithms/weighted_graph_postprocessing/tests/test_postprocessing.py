import numpy as np
import pytest

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import is_dag

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
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.local_search import (
    optimize_graph_locally,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.core import (
    FeasibilityChecker,
    PostselectionConfig,
    REFERENCE_POLICY,
    select_postselection_candidate,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.normalized_projection import (
    normalized_greedy_projection,
)


def test_normalized_projection_keeps_feasible_support_unchanged():
    W = np.zeros((3, 3))
    W[0, 1], W[1, 2] = .2, -.3
    result = normalized_greedy_projection(W)
    np.testing.assert_array_equal(result.adjacency, (W != 0).astype(np.uint8))
    assert result.diagnostics["projection_steps"] == 0


def test_normalized_projection_deletes_cycle_edges_only():
    W = np.zeros((3, 3))
    W[0, 1], W[1, 2], W[2, 0] = .9, .8, .7
    result = normalized_greedy_projection(W)
    assert is_dag(result.adjacency)
    assert np.all((result.adjacency != 0) <= (W != 0))
    assert result.diagnostics["edges_removed"] >= 1


def test_normalized_projection_repairs_notreks_and_cycle_together():
    W = np.zeros((4, 4))
    W[0, 2], W[1, 2], W[2, 3], W[3, 0] = .9, .8, .7, .6
    result = normalized_greedy_projection(
        W, [(0, 1)], dag_constraint=True, notreks_constraint=True)
    assert is_dag(result.adjacency)
    from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import common_ancestor_violations
    assert common_ancestor_violations(result.adjacency, [(0, 1)]) == 0
    assert np.all((result.adjacency != 0) <= (W != 0))


def test_normalized_projection_ties_are_deterministic_and_fallback_progresses():
    W = np.full((3, 3), .2)
    np.fill_diagonal(W, 0)
    first = normalized_greedy_projection(W)
    second = normalized_greedy_projection(W)
    np.testing.assert_array_equal(first.adjacency, second.adjacency)
    assert is_dag(first.adjacency)


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


def test_local_search_checks_initial_feasibility_before_scoring():
    graph = np.zeros((3, 3), dtype=int)
    graph[2, 0] = graph[2, 1] = 1
    constraints = CompositeFeasibility((
        DagConstraint(), NoTreksConstraint(((0, 1),))))

    class ScoreThatMustNotRun:
        def score_graph(self, *_args):
            raise AssertionError("infeasible graph was scored")

    with pytest.raises(ValueError, match="feasible initial graph"):
        optimize_graph_locally(
            graph, WeightedGraphEstimate(graph), None,
            ScoreThatMustNotRun(), constraints, PostprocessingBudget())


def test_reference_postselection_checks_feasibility_before_bic(monkeypatch):
    import workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.core as core

    X = np.random.default_rng(81).normal(size=(40, 3))
    W = np.zeros((3, 3))
    W[2, 0], W[2, 1] = .9, .8
    checker = FeasibilityChecker(
        d=3, dag_constraint_active=True, notreks_constraint_active=True,
        notreks_pairs=((0, 1),))
    original = core.gaussian_bic
    scored = []

    def checked_bic(data, adjacency, lambda_bic):
        assert checker.check(adjacency).feasible
        scored.append(np.asarray(adjacency).copy())
        return original(data, adjacency, lambda_bic=lambda_bic)

    monkeypatch.setattr(core, "gaussian_bic", checked_bic)
    result = select_postselection_candidate(
        W,
        scorer=type("ReferenceScorer", (), {"X": X})(),
        config=PostselectionConfig(
            policy=REFERENCE_POLICY, threshold_grid=(.1,),
            notreks_constraint_active=True),
        model_class="linear_dagma", notreks_pairs=((0, 1),))
    assert result.feasible
    assert scored
