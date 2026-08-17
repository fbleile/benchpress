import numpy as np

from workflow.rules.structure_learning_algorithms.dagma_fast.penalties import (
    LogDetDagPenalty,
)
from workflow.rules.structure_learning_algorithms.notreks import NoTreksPenalty


def finite_difference(fun, W, eps=1e-6):
    out = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if i == j:
                continue
            plus, minus = W.copy(), W.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            out[i, j] = (fun(plus) - fun(minus)) / (2 * eps)
    return out


def test_phi_log_dag_gradient():
    W = np.array([[0., .04, -.03], [.02, 0., .01], [.01, -.02, 0.]])
    penalty = LogDetDagPenalty(3, adjacency_mapping="phi_log")
    value, gradient = penalty.value_and_grad(W)
    numeric = finite_difference(lambda x: penalty.value_and_grad(x)[0], W)
    assert np.isfinite(value)
    assert np.allclose(gradient, numeric, atol=2e-5, rtol=2e-4)


def test_phi_log_notreks_gradient():
    W = np.array([[0., .04, -.03], [.02, 0., .01], [.01, -.02, 0.]])
    penalty = NoTreksPenalty(
        [(0, 1)], 3, kernel="fast", function="inv",
        adjacency_mapping="phi_log")
    value, gradient = penalty.value_and_grad(W)
    numeric = finite_difference(lambda x: penalty.value_and_grad(x)[0], W)
    assert np.isfinite(value)
    assert np.allclose(gradient, numeric, atol=2e-5, rtol=2e-4)
