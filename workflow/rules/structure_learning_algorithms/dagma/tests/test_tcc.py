import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.shared import tcc_value_grad


def test_tcc_is_zero_without_a_trek_and_positive_for_direct_trek():
    empty = np.zeros((3, 3))
    value, gradient = tcc_value_grad(empty, [(0, 1)], coupling=1.0)
    assert value == 0.0
    assert np.all(np.isfinite(gradient))

    direct = empty.copy()
    direct[0, 1] = 0.8
    value, gradient = tcc_value_grad(direct, [(0, 1)], coupling=1.0)
    assert value > 0.0
    assert np.all(np.isfinite(gradient))


def test_tcc_uses_separate_pair_auxiliary_graphs():
    W = np.zeros((4, 4))
    W[0, 2] = 0.7
    W[1, 2] = 0.6
    value, gradient = tcc_value_grad(W, [(0, 1), (1, 3)], coupling=1.0)
    assert value >= 0.0
    assert gradient.shape == W.shape
    assert np.all(np.isfinite(gradient))
