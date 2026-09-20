import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.shared import (
    SharedDagmaLinear,
    dagma_adjacency_pullback,
    make_dagma_adjacency,
    stable_abs_row_adjacency,
)


MAPS = ("square", "abs", "pseudo_huber", "entrywise_capped_square",
        "row_capped_square", "row_capped_abs")
from workflow.rules.structure_learning_algorithms.notreks import (
    make_notreks_kernel,
)


def test_stable_map_is_nonnegative_diagonal_free_and_support_preserving():
    rng = np.random.default_rng(7)
    W = rng.normal(size=(6, 6))
    np.fill_diagonal(W, 0.0)
    A = stable_abs_row_adjacency(W, 0.7)
    assert np.all(A >= 0)
    assert np.all(np.diag(A) == 0)
    assert np.array_equal(A != 0, W != 0)
    assert np.all(A.sum(axis=1) < 0.7)
    assert max(abs(np.linalg.eigvals(A))) < 0.7


def test_all_adjacency_maps_are_equivariant_finite_and_have_finite_gradients():
    rng = np.random.default_rng(17)
    W = rng.normal(size=(6, 6))
    np.fill_diagonal(W, 0.0)
    permutation = np.array([2, 4, 0, 3, 1, 5])
    P = np.eye(6)[permutation]
    for name in MAPS:
        A = make_dagma_adjacency(W, name, 0.7, 1.0)
        assert np.all(np.isfinite(A)) and np.all(A >= 0)
        assert np.all(np.diag(A) == 0)
        assert np.array_equal(A != 0, W != 0)
        np.testing.assert_allclose(
            make_dagma_adjacency(P.T @ W @ P, name, 0.7, 1.0),
            P.T @ A @ P)
        grad = dagma_adjacency_pullback(
            W, np.ones_like(W), name, 0.7, 1.0)
        assert np.all(np.isfinite(grad))


def test_domain_safe_maps_have_bounded_rows_and_spectral_radius():
    rng = np.random.default_rng(18)
    W = rng.normal(size=(8, 8)) * 100.0
    np.fill_diagonal(W, 0.0)
    for name in ("entrywise_capped_square", "row_capped_square",
                 "row_capped_abs"):
        A = make_dagma_adjacency(W, name, 0.7)
        assert np.all(A.sum(axis=1) < 0.7)
        assert max(abs(np.linalg.eigvals(A))) < 0.7


def test_map_local_scaling_orders():
    rng = np.random.default_rng(19)
    W = rng.normal(size=(5, 5))
    np.fill_diagonal(W, 0.0)
    for name, expected_power in (("square", 2), ("abs", 1),
                                 ("pseudo_huber", 2),
                                 ("row_capped_square", 2),
                                 ("row_capped_abs", 1)):
        t = 1e-4
        A = make_dagma_adjacency(t * W, name, 0.8)
        target = t**expected_power * (
            W * W if expected_power == 2 else np.abs(W))
        assert np.linalg.norm(A - target) / max(np.linalg.norm(target), 1e-300) < 1e-3


def test_stable_map_is_locally_linear_and_permutation_equivariant():
    rng = np.random.default_rng(8)
    W = rng.normal(size=(5, 5))
    np.fill_diagonal(W, 0.0)
    s = 0.8
    t = 1e-7
    ratio = np.linalg.norm(
        stable_abs_row_adjacency(t * W, s) - t * np.abs(W)) / (
            t * np.linalg.norm(W))
    assert ratio < 1e-6
    permutation = np.array([2, 4, 0, 3, 1])
    P = np.eye(5)[permutation]
    np.testing.assert_allclose(
        stable_abs_row_adjacency(P.T @ W @ P, s),
        P.T @ stable_abs_row_adjacency(W, s) @ P)


def test_stable_dagma_zero_set_and_cycle_penalty():
    model = SharedDagmaLinear("l2")
    model.d = 3
    model.Id = np.eye(3)
    model.adjacency_map = "stable_abs_row"
    model.adjacency_map_tau = 1.0
    model.stable_s_star = 0.7
    dag = np.zeros((3, 3))
    dag[0, 1], dag[1, 2] = 0.4, 0.3
    cycle = dag.copy()
    cycle[2, 0] = 0.2
    assert abs(model._h(dag, 0.7)[0]) < 1e-12
    assert model._h(cycle, 0.7)[0] > 0


def test_stable_notreks_zero_and_positive_trek_penalties():
    kernel = make_notreks_kernel("fast", [(0, 1)], 3)
    empty = np.zeros((3, 3))
    A = stable_abs_row_adjacency(empty, 0.7)
    R = np.eye(3)
    value, _ = kernel.value_grad_from_resolvent_adjacency(A, R)
    assert value == 0.0
    common_parent = empty.copy()
    common_parent[2, 0] = common_parent[2, 1] = 0.4
    A = stable_abs_row_adjacency(common_parent, 0.7)
    R = np.linalg.inv(np.eye(3) - A)
    value, gradient = kernel.value_grad_from_resolvent_adjacency(A, R)
    assert value > 0
    assert np.all(np.isfinite(gradient))
