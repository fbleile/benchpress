import numpy as np
import pytest
import scipy.linalg as sla

from workflow.rules.structure_learning_algorithms.dagma.shared import (
    SharedDagmaLinear,
    notreks_value_grad,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.kernels import (
    KERNEL_REGISTRY,
    make_notreks_kernel,
    notreks_value_grad_kernel,
)


def _finite_difference(function, W, step=1e-6):
    gradient = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            plus = W.copy()
            minus = W.copy()
            plus[i, j] += step
            minus[i, j] -= step
            gradient[i, j] = (function(plus) - function(minus)) / (2 * step)
    return gradient


def test_registry_contains_recoverable_reference_and_new_kernels():
    assert "notreks_reference" in KERNEL_REGISTRY
    assert "dense_current" in KERNEL_REGISTRY
    assert "selected_inv" in KERNEL_REGISTRY
    assert "poly_selected_walk" in KERNEL_REGISTRY


def test_every_kernel_uses_w_squared_argument():
    rng = np.random.default_rng(11)
    W = rng.normal(scale=0.05, size=(5, 5))
    np.fill_diagonal(W, 0.0)
    pairs = [(0, 1), (2, 4)]
    for kernel_name, function in [
        ("dense_current", "exp"),
        ("dense_current", "inv"),
        ("selected_inv", "inv"),
        ("poly_selected_walk", "walk"),
        ("poly_selected_exp", "poly_exp"),
    ]:
        value, gradient, _ = notreks_value_grad_kernel(
            W, pairs, function, kernel_name=kernel_name, inverse_epsilon=1e-8)
        flipped_value, flipped_gradient, _ = notreks_value_grad_kernel(
            -W, pairs, function, kernel_name=kernel_name, inverse_epsilon=1e-8)
        assert value == pytest.approx(flipped_value, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(gradient, -flipped_gradient, rtol=1e-11, atol=1e-12)


def test_selected_inverse_matches_dense_reference_value_and_gradient():
    rng = np.random.default_rng(12)
    W = rng.normal(scale=0.04, size=(8, 8))
    np.fill_diagonal(W, 0.0)
    pairs = [(0, 1), (0, 7), (3, 6)]
    expected = notreks_value_grad(W, pairs, "inv", inverse_epsilon=1e-8)
    result = make_notreks_kernel("selected_inv", pairs, 8).value_and_grad(
        W, function="inv", inverse_epsilon=1e-8)
    assert result.penalty_value == pytest.approx(expected[0], rel=2e-12, abs=2e-14)
    np.testing.assert_allclose(result.gradient_W, expected[1], rtol=2e-11, atol=2e-12)
    assert result.diagnostics.full_matrix_materialized is False
    assert result.diagnostics.selected_column_count < W.shape[0]


def test_selected_inverse_can_reuse_a_validated_resolvent():
    rng = np.random.default_rng(121)
    W = rng.normal(scale=0.04, size=(8, 8))
    np.fill_diagonal(W, 0.0)
    pairs = [(0, 1), (0, 7), (3, 6)]
    kernel = make_notreks_kernel("selected_inv", pairs, 8)
    F = sla.solve(np.eye(8) - W * W, np.eye(8))
    expected_value, expected_gradient = kernel.value_grad(
        W, "inv", inverse_epsilon=0.0)
    value, gradient = kernel.value_grad_from_resolvent(W, F)
    assert value == pytest.approx(expected_value, rel=1e-12, abs=1e-14)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-11, atol=1e-13)


def test_selected_inverse_directional_derivative():
    rng = np.random.default_rng(13)
    W = rng.normal(scale=0.03, size=(6, 6))
    np.fill_diagonal(W, 0.0)
    direction = rng.normal(size=W.shape)
    np.fill_diagonal(direction, 0.0)
    pairs = [(0, 5), (1, 4)]
    kernel = make_notreks_kernel("selected_inv", pairs, 6)
    value, gradient = kernel.value_grad(W, "inv", inverse_epsilon=1e-8)
    eps = 1e-6
    forward = kernel.value_grad(W + eps * direction, "inv", inverse_epsilon=1e-8)[0]
    backward = kernel.value_grad(W - eps * direction, "inv", inverse_epsilon=1e-8)[0]
    numerical = (forward - backward) / (2 * eps)
    analytic = float(np.sum(gradient * direction))
    assert value >= 0.0
    assert analytic == pytest.approx(numerical, rel=3e-6, abs=3e-8)


def test_selected_polynomial_matches_finite_difference():
    rng = np.random.default_rng(14)
    W = rng.normal(scale=0.04, size=(5, 5))
    np.fill_diagonal(W, 0.0)
    pairs = [(0, 1), (3, 4)]
    kernel = make_notreks_kernel("poly_selected_walk", pairs, 5)
    value, gradient = kernel.value_grad(W, "walk", log_terms=4)
    numerical = _finite_difference(
        lambda Z: kernel.value_grad(Z, "walk", log_terms=4)[0], W)
    assert value >= 0.0
    np.testing.assert_allclose(gradient, numerical, rtol=3e-6, atol=3e-8)


def test_duplicate_and_reversed_pairs_are_rejected_for_unordered_semantics():
    with pytest.raises(ValueError, match="duplicate"):
        make_notreks_kernel("dense_current", [(0, 1), (1, 0)], 3)


def test_inverse_domain_failure_is_reported_explicitly():
    W = np.array([[0.0, 1.1], [1.1, 0.0]])
    kernel = make_notreks_kernel("selected_inv", [(0, 1)], 2)
    with pytest.raises(np.linalg.LinAlgError, match="outside domain"):
        kernel.value_grad(W, "inv", inverse_epsilon=0.0)
    assert kernel.diagnostics()["domain_valid"] is False


def test_zero_notreks_weight_reproduces_reference_kernel_fit():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(80, 4))
    settings = dict(
        no_trek_pairs=[(0, 1)], trek_weight=0.0, trek_function="inv",
        T=1, warm_iter=5, max_iter=5, checkpoint=5, w_threshold=0.0,
        lambda1=0.01, lr=0.0003)
    reference = SharedDagmaLinear("l2").fit(
        X.copy(), trek_kernel="notreks_reference", **settings)
    selected = SharedDagmaLinear("l2").fit(
        X.copy(), trek_kernel="selected_inv", **settings)
    np.testing.assert_allclose(selected, reference, rtol=0.0, atol=0.0)


def test_dagma_fit_uses_the_shared_resolvent_at_matching_shift():
    rng = np.random.default_rng(16)
    X = rng.normal(size=(80, 4))
    settings = dict(
        no_trek_pairs=[(0, 1)], trek_weight=0.2, trek_function="inv",
        trek_inverse_epsilon=0.0, trek_kernel="selected_inv", T=1,
        warm_iter=5, max_iter=5, checkpoint=5, w_threshold=0.0,
        lambda1=0.01, lr=0.0003, s=(1.0,))
    model = SharedDagmaLinear("l2")
    model.fit(X.copy(), **settings)
    assert model.stage_diagnostics[-1]["shared_resolvent_iterations"] > 0
