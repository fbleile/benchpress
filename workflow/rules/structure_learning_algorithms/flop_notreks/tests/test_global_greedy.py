import numpy as np

from workflow.rules.structure_learning_algorithms.flop_notreks.global_greedy import (
    GlobalGreedyConfig,
    fit_global_greedy_notreks,
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
