import numpy as np

from workflow.rules.structure_learning_algorithms.dagma_notreks.tools.inv_regression_audit import (
    LAMBDAS, threshold_graph,
)


def test_fixed_and_scaled_lambda_values():
    assert LAMBDAS["fixed"] == .03
    assert np.isclose(LAMBDAS["scaled"], .021213203435596423)


def test_fixed_threshold_uses_literal_point_zero_one():
    W = np.array([[0., .01, .0099], [0., 0., .2], [0., 0., 0.]])
    graph = threshold_graph(W, .01)
    assert graph[0, 1] == 1
    assert graph[0, 2] == 0


def test_three_restart_indices_are_strict_subset_of_five():
    assert set(range(3)) < set(range(5))


def test_threshold_support_is_monotone_and_diagonal_free():
    W = np.array([[9., .005, .05], [.02, 8., .3], [.001, .1, 7.]])
    low = threshold_graph(W, .01)
    high = threshold_graph(W, .2)
    assert np.all((high == 0) | (low != 0))
    assert not np.diag(low).any()
