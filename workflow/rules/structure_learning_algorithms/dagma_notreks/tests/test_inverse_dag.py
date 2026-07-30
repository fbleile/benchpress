import numpy as np
import pytest

from workflow.rules.structure_learning_algorithms.dagma.inverse_structural import (
    InverseStructuralKernel, inverse_dag_value_grad,
    lambda1_sqrt_logd_over_n,
)
from workflow.rules.structure_learning_algorithms.dagma.shared import (
    NoTreksKernel,
)


def finite_difference(function, W, step=1e-6):
    gradient = np.zeros_like(W)
    for i in range(len(W)):
        for j in range(len(W)):
            plus, minus = W.copy(), W.copy()
            plus[i, j] += step
            minus[i, j] -= step
            gradient[i, j] = (function(plus) - function(minus)) / (2 * step)
    return gradient


@pytest.mark.parametrize("d", [3, 5])
def test_inverse_dag_gradient_and_one_shared_solve(d):
    rng = np.random.default_rng(120 + d)
    W = rng.normal(scale=.07, size=(d, d))
    np.fill_diagonal(W, 0.)
    mask = np.zeros((d, d))
    mask[0, 1] = mask[1, 0] = 1.
    kernel = InverseStructuralKernel.from_pair_mask(mask, 2 / (d - 1))
    result = kernel.evaluate(W)
    numerical = finite_difference(
        lambda value: inverse_dag_value_grad(value)[0], W)
    assert np.max(np.abs(result.inverse_dag_gradient - numerical)) < 2e-7
    assert kernel.matrix_factorizations == 1
    assert kernel.matrix_inverse_or_solve_calls == 1
    assert kernel.matrix_multiplications == 5


def test_inverse_dag_zero_on_dags_positive_on_cycles_and_diagonal_gradient():
    dag = np.zeros((4, 4))
    dag[0, 1] = .4
    dag[1, 2] = -.3
    value, gradient = inverse_dag_value_grad(dag)
    assert abs(value) < 1e-12
    assert np.allclose(np.diag(gradient), 0.)
    cycle = dag.copy()
    cycle[2, 0] = .2
    assert inverse_dag_value_grad(cycle)[0] > 0.


def test_shared_notreks_matches_existing_inverse_kernel():
    rng = np.random.default_rng(77)
    W = rng.normal(scale=.05, size=(6, 6))
    np.fill_diagonal(W, 0.)
    old = NoTreksKernel.from_pairs([(0, 1), (2, 5), (3, 4)], 6)
    expected_value, expected_gradient = old.value_grad(W, "inv")
    shared = InverseStructuralKernel.from_pair_mask(
        old.pair_mask, old.scale, 1e-8).evaluate(W)
    assert shared.notreks_value == pytest.approx(
        expected_value, rel=2e-14, abs=2e-14)
    assert np.allclose(
        shared.notreks_gradient, expected_gradient, rtol=2e-13, atol=2e-13)


def test_inverse_domain_rejection():
    W = np.array([[0., 1.1], [1.1, 0.]])
    with pytest.raises(np.linalg.LinAlgError, match="outside its domain"):
        inverse_dag_value_grad(W)


def test_inverse_dag_matches_torch_autodiff():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(881)
    W = rng.normal(scale=.06, size=(4, 4))
    np.fill_diagonal(W, 0.)
    value, gradient = inverse_dag_value_grad(W)
    tensor = torch.tensor(W, dtype=torch.float64, requires_grad=True)
    alpha = 1.0 + 1e-8
    inverse = torch.linalg.solve(
        alpha * torch.eye(4, dtype=torch.float64) - tensor * tensor,
        torch.eye(4, dtype=torch.float64))
    reference = alpha * torch.trace(inverse) - 4
    reference.backward()
    assert value == pytest.approx(reference.item(), abs=2e-14)
    assert np.allclose(gradient, tensor.grad.numpy(), rtol=2e-12, atol=2e-12)


def test_lambda1_reference_and_scaling():
    assert lambda1_sqrt_logd_over_n(1000, 50) == pytest.approx(.03)
    assert lambda1_sqrt_logd_over_n(500, 50) == pytest.approx(.03 * np.sqrt(2))
    assert lambda1_sqrt_logd_over_n(2000, 50) == pytest.approx(.03 / np.sqrt(2))
    assert lambda1_sqrt_logd_over_n(1000, 100) > .03
