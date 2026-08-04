import numpy as np

from workflow.rules.structure_learning_algorithms.dagma_fast.objective import (
    LinearL2Objective,
)
from workflow.rules.structure_learning_algorithms.dagma_fast.optimizer import (
    DagmaFastConfig,
    fit_weighted_adjacency,
)
from workflow.rules.structure_learning_algorithms.notreks import (
    FAST_KERNEL,
    NoTreksPenalty,
    make_notreks_kernel,
)


def test_fast_alias_is_selected_inverse_and_matches_reference():
    rng = np.random.default_rng(41)
    W = rng.normal(scale=0.03, size=(6, 6))
    np.fill_diagonal(W, 0.0)
    pairs = [(0, 4), (1, 5)]
    fast = make_notreks_kernel(FAST_KERNEL, pairs, 6).value_and_grad(
        W, function="inv")
    reference = make_notreks_kernel(
        "notreks_reference", pairs, 6).value_and_grad(W, function="inv")
    np.testing.assert_allclose(
        fast.penalty_value, reference.penalty_value, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(
        fast.gradient_W, reference.gradient_W, rtol=2e-11, atol=2e-12)
    assert fast.diagnostics.name == "selected_inv"


def test_vanilla_dagma_and_optional_notreks_use_same_optimizer():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 4))
    config = DagmaFastConfig(
        T=1, warm_iter=4, max_iter=4, checkpoint=4, lambda1=0.01)
    vanilla = fit_weighted_adjacency(LinearL2Objective(X), config)
    zero_weight = fit_weighted_adjacency(
        LinearL2Objective(X), config,
        structural_penalties=[NoTreksPenalty(
            [(0, 1)], 4, weight=0.0)])
    np.testing.assert_allclose(
        zero_weight.weighted_adjacency, vanilla.weighted_adjacency,
        rtol=0.0, atol=0.0)
    assert vanilla.diagnostics["structural_penalty_count"] == 0
    assert zero_weight.diagnostics["structural_penalty_count"] == 1


def test_nonzero_notreks_is_an_optional_component():
    penalty = NoTreksPenalty([(0, 2)], 3, weight=7.0)
    W = np.array([[0.0, 0.1, 0.1], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]])
    value, gradient = penalty.value_and_grad(W)
    assert value >= 0.0
    assert gradient.shape == W.shape
    assert penalty.diagnostics()["kernel"] == "fast"
