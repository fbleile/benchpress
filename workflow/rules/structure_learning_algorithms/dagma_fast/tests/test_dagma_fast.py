import numpy as np
import pytest

from workflow.rules.structure_learning_algorithms.dagma_fast import (
    CallableDagPenalty,
    CallableObjective,
    DagmaFastConfig,
    LinearL2Objective,
    fit_weighted_adjacency,
    resolve_lambda1,
)
from workflow.rules.structure_learning_algorithms.dagma_anytime.solver import (
    fit_linear_dagma_anytime,
)


def small_config(**changes):
    values = dict(
        T=2, s=(1.0, 0.9), warm_iter=3, max_iter=5,
        checkpoint=2, max_runtime_seconds=5)
    values.update(changes)
    return DagmaFastConfig(**values)


def test_linear_optimizer_is_deterministic_and_masks_diagonal():
    X = np.random.default_rng(2).normal(size=(40, 4))
    objective = LinearL2Objective(X)
    first = fit_weighted_adjacency(objective, small_config())
    second = fit_weighted_adjacency(objective, small_config())
    np.testing.assert_allclose(
        first.weighted_adjacency, second.weighted_adjacency)
    np.testing.assert_array_equal(np.diag(first.weighted_adjacency), 0)
    assert np.isfinite(first.objective)
    assert first.factorization_count >= first.iterations


def test_mock_nonlinear_objective_requires_no_optimizer_change():
    def value_grad(W):
        target = np.zeros_like(W)
        target[0, 1] = 0.1
        difference = W - target
        return float(np.sum(difference ** 4)), 4 * difference ** 3

    result = fit_weighted_adjacency(
        CallableObjective(3, value_grad), small_config())
    assert result.weighted_adjacency.shape == (3, 3)
    assert np.all(np.diag(result.weighted_adjacency) == 0)


def test_mock_alternative_dag_and_multiple_structural_penalties():
    dag = CallableDagPenalty(
        lambda W, s: (float(np.sum(W * W)), 2 * W))

    class Structural:
        def __init__(self, scale):
            self.scale = scale

        def value_and_grad(self, W):
            return (
                float(self.scale * np.sum(W * W)),
                2 * self.scale * W)

    objective = CallableObjective(
        3, lambda W: (float(np.sum((W - .01) ** 2)), 2 * (W - .01)))
    result = fit_weighted_adjacency(
        objective, small_config(), dag_penalty=dag,
        structural_penalties=(Structural(.1), Structural(.2)))
    assert len(result.structural_penalties) == 2
    assert result.diagnostics["structural_penalty_count"] == 2


def test_lambda_registry_records_resolution_and_is_independent():
    fixed = resolve_lambda1("fixed", n=1000, d=50, fixed=.03)
    scaled = resolve_lambda1(
        "sqrt_log_d_over_n", n=2000, d=50)
    assert fixed.value == .03
    assert scaled.value == pytest.approx(.03 / np.sqrt(2))
    assert (scaled.n, scaled.d) == (2000, 50)


def test_deprecated_backend_alias_is_numerically_identical():
    X = np.random.default_rng(3).normal(size=(30, 3))
    objective = LinearL2Objective(X)
    canonical = fit_weighted_adjacency(objective, small_config())
    with pytest.warns(DeprecationWarning):
        alias = fit_weighted_adjacency(
            objective, small_config(method="dagma_fused64_exact"))
    np.testing.assert_allclose(
        canonical.weighted_adjacency, alias.weighted_adjacency)


def test_fast_matches_retained_fused64_reference():
    X = np.random.default_rng(8).normal(size=(40, 4))
    config = small_config()
    canonical = fit_weighted_adjacency(
        LinearL2Objective(X), config).weighted_adjacency
    reference = fit_linear_dagma_anytime(
        X, method="dagma_fused64_exact", lambda1=.03,
        w_threshold=0, T=config.T, s=config.s,
        warm_iter=config.warm_iter, max_iter=config.max_iter,
        checkpoint=config.checkpoint, zero_diagonal=True,
    ).weighted_adjacency_raw
    np.testing.assert_allclose(canonical, reference, atol=1e-15, rtol=1e-13)
