import numpy as np
import pytest
import torch

from workflow.rules.structure_learning_algorithms.dagma.shared import (
    NoTreksKernel,
    notreks_value_grad,
)


def _torch_exp(W, pairs):
    tensor = torch.tensor(W, dtype=torch.float64, requires_grad=True)
    E = torch.matrix_exp(tensor * tensor)
    H = E.T @ E
    scale = 2.0 / (W.shape[0] - 1) if W.shape[0] > 1 else 0.0
    value = scale * sum(H[i, j] for i, j in pairs) if pairs else H.sum() * 0
    value.backward()
    return float(value.detach()), tensor.grad.detach().numpy()


def _finite_difference(W, pairs, function, step=1e-6):
    gradient = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            plus, minus = W.copy(), W.copy()
            plus[i, j] += step
            minus[i, j] -= step
            gradient[i, j] = (
                notreks_value_grad(plus, pairs, function)[0]
                - notreks_value_grad(minus, pairs, function)[0]
            ) / (2 * step)
    return gradient


@pytest.mark.parametrize("d", [4, 10, 20])
@pytest.mark.parametrize("pair_kind", ["empty", "one", "multiple", "dense"])
@pytest.mark.parametrize("matrix_kind", ["zero", "small", "dagma_domain"])
def test_exp_analytic_matches_torch(d, pair_kind, matrix_kind):
    rng = np.random.default_rng(9000 + d)
    W = rng.normal(scale=.03 if matrix_kind == "small" else .12, size=(d, d))
    np.fill_diagonal(W, 0)
    if matrix_kind == "zero":
        W.fill(0)
    all_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    pairs = {
        "empty": [],
        "one": all_pairs[:1],
        "multiple": all_pairs[::max(1, len(all_pairs) // 5)][:5],
        "dense": all_pairs,
    }[pair_kind]
    analytic = NoTreksKernel.from_pairs(pairs, d).value_grad(W, "exp")
    reference = _torch_exp(W, pairs)
    np.testing.assert_allclose(analytic[0], reference[0], rtol=1e-10, atol=3e-12)
    np.testing.assert_allclose(analytic[1], reference[1], rtol=1e-9, atol=3e-11)


@pytest.mark.parametrize("function", ["exp", "inv"])
def test_direct_kernels_match_central_finite_difference(function):
    rng = np.random.default_rng(41)
    W = rng.normal(scale=.08, size=(4, 4))
    np.fill_diagonal(W, 0)
    pairs = [(0, 1), (0, 3), (2, 3)]
    _, gradient = notreks_value_grad(W, pairs, function, inverse_epsilon=0)
    numerical = _finite_difference(W, pairs, function)
    np.testing.assert_allclose(gradient, numerical, rtol=3e-7, atol=2e-9)


def test_inv_reports_near_singular_system():
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        notreks_value_grad(W, [(0, 1)], "inv", inverse_epsilon=0)
