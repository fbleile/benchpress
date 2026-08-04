import numpy as np

from workflow.rules.structure_learning_algorithms.flop_notreks.global_greedy import (
    GlobalGreedyConfig,
    _LocalGaussianBIC,
    _addition_violates_notreks,
    _invalid_notreks_additions,
    _transitive_reach,
    fit_global_greedy_notreks,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.feasibility import (
    NoTreksConstraint,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.scores import (
    GaussianBICGraphScore,
)


def test_global_greedy_is_deterministic_and_hard_feasible():
    rng = np.random.default_rng(51)
    X = rng.normal(size=(100, 5))
    X[:, 2] += 0.9 * X[:, 0]
    X[:, 3] += 0.8 * X[:, 1]
    settings = GlobalGreedyConfig(restarts=2, max_sweeps=2, seed=7)
    first = fit_global_greedy_notreks(X, [(2, 3)], settings)
    second = fit_global_greedy_notreks(X, [(2, 3)], settings)
    np.testing.assert_array_equal(first.adjacency, second.adjacency)
    assert first.score == second.score
    assert first.notreks_violation_count == 0
    assert first.termination_reason in {"joint_plateau", "maximum_sweeps"}


def test_empty_constraints_are_vanilla_global_greedy_mode():
    X = np.random.default_rng(52).normal(size=(60, 4))
    result = fit_global_greedy_notreks(
        X, [], GlobalGreedyConfig(restarts=1, max_sweeps=1, seed=9))
    assert result.notreks_violation_count == 0
    assert np.all(np.diag(result.adjacency) == 0)


def test_cached_local_bic_matches_full_graph_score():
    rng = np.random.default_rng(53)
    X = rng.normal(size=(90, 7))
    local = _LocalGaussianBIC(X, lambda_bic=2.0)
    full = GaussianBICGraphScore(X, lambda_bic=2.0)
    for _ in range(30):
        graph = np.triu((rng.random((7, 7)) < .3).astype(np.uint8), 1)
        np.testing.assert_allclose(
            local.graph_score(graph), full.score_graph(graph),
            rtol=1e-11, atol=1e-9)


def test_incremental_implementation_matches_preoptimization_reference():
    rng = np.random.default_rng(204)
    X = rng.normal(size=(120, 6))
    X[:, 2] += .8 * X[:, 0]
    X[:, 4] += .7 * X[:, 1] - .4 * X[:, 3]
    result = fit_global_greedy_notreks(
        X, [(2, 4)],
        GlobalGreedyConfig(restarts=2, max_sweeps=2, seed=33))
    expected = np.asarray([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(result.adjacency, expected)
    np.testing.assert_allclose(result.score, 17.349924658949348,
                               rtol=1e-12, atol=1e-12)
    assert result.full_feasibility_evaluations == 1


def test_incremental_notreks_addition_check_matches_exact_checker():
    rng = np.random.default_rng(407)
    pairs = np.asarray([(0, 3), (1, 4)], dtype=int)
    checker = NoTreksConstraint(pairs)
    checked = 0
    for _ in range(100):
        graph = np.triu((rng.random((5, 5)) < .18).astype(np.uint8), 1)
        if not checker.is_feasible(graph):
            continue
        reach = _transitive_reach(graph)
        invalid = _invalid_notreks_additions(reach, pairs)
        for source in range(5):
            for target in range(source + 1, 5):
                if graph[source, target]:
                    continue
                proposal = graph.copy()
                proposal[source, target] = 1
                fast = _addition_violates_notreks(
                    reach, source, target, pairs)
                assert fast == (not checker.is_feasible(proposal))
                assert invalid[source, target] == fast
                checked += 1
    assert checked > 100
