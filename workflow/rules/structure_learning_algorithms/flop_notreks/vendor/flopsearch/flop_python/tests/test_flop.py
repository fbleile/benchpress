import flopsearch

import numpy as np
import pytest
from scipy import linalg


def test_path():
    p = 10
    W = np.diag(np.ones(p - 1), 1)
    X = np.random.randn(10000, p).dot(linalg.inv(np.eye(p) - W))
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    G = flopsearch.flop(X_std, 2.0, restarts=50)
    assert np.all(np.diag(G, k=1) == 2)
    assert np.all(np.diag(G, k=-1) == 2)


def _has_common_ancestor(dag, i, j):
    def ancestors(node):
        result, stack = {node}, [node]
        while stack:
            child = stack.pop()
            for parent in np.flatnonzero(dag[:, child]):
                parent = int(parent)
                if parent not in result:
                    result.add(parent)
                    stack.append(parent)
        return result
    return bool(ancestors(i) & ancestors(j))


def test_flop_notreks_inputs_outputs_and_reproducibility():
    data = np.random.default_rng(42).normal(size=(200, 5))
    kwargs = dict(restarts=0, seed=19, signature_top_k=3,
                  max_signature_rounds=3, return_dag=True,
                  return_diagnostics=True)
    first, d1 = flopsearch.flop_notreks(data, 2.0, [(0, 1)], **kwargs)
    second, d2 = flopsearch.flop_notreks(
        data, 2.0, np.asarray([[1, 0], [0, 1]], dtype=np.int64), **kwargs)
    assert np.array_equal(first, second)
    assert d1["final_bic"] == d2["final_bic"]
    assert d1["search_version"] == "global_greedy_rust"
    assert d1["final_no_trek_violation_count"] == d2["final_no_trek_violation_count"] == 0
    assert d1["final_no_trek_violation_count"] == 0
    assert not _has_common_ancestor(first, 0, 1)
    cpdag = flopsearch.flop_notreks(
        data, 2.0, [(0, 1)], restarts=0, seed=19)
    assert cpdag.shape == (5, 5)
    assert set(np.unique(cpdag)) <= {0, 1, 2}


def test_flop_notreks_empty_and_invalid_inputs():
    data = np.random.default_rng(3).normal(size=(100, 4))
    _, diagnostics = flopsearch.flop_notreks(
        data, 2.0, [], restarts=0, seed=1, return_diagnostics=True)
    assert diagnostics["number_of_candidate_parent_relations_pruned"] == 0
    for pairs in ([(0, 0)], [(0, 4)], [[0, 1, 2]]):
        with pytest.raises((ValueError, RuntimeError)):
            flopsearch.flop_notreks(data, 2.0, pairs, restarts=0)


def test_legacy_search_versions_are_not_public_options():
    data = np.random.default_rng(8).normal(size=(100, 4))
    for version in ("fixed_signature_a", "alternating_full_refit_b", "cached_repair_c", "unknown"):
        with pytest.raises(ValueError):
            flopsearch.flop_notreks(
                data, 2.0, [(0, 1)], restarts=0, search_version=version)


def test_prune_parents_bic_is_deterministic_and_deletion_only():
    rng = np.random.default_rng(21)
    x0 = rng.normal(size=500)
    data = np.column_stack([
        x0,
        1.8 * x0 + rng.normal(scale=.2, size=500),
        rng.normal(size=500),
    ])
    candidate = np.array([
        [0., 1., 1.],
        [0., 0., 1.],
        [0., 0., 0.],
    ])
    first, coefficients, diagnostics = flopsearch.prune_parents_bic(
        data, candidate, return_coefficients=True, return_diagnostics=True)
    second, coefficients2, diagnostics2 = flopsearch.prune_parents_bic(
        data, candidate, return_coefficients=True, return_diagnostics=True)
    assert np.array_equal(first, second)
    assert np.all((first == 0) | (candidate != 0))
    assert np.allclose(coefficients, coefficients2)
    assert diagnostics["bic"] == diagnostics2["bic"]
    assert diagnostics["edge_count"] == np.count_nonzero(first)
    assert coefficients[0, 1] == pytest.approx(1.8, rel=.05)


def test_prune_parents_bic_rejects_invalid_candidates():
    data = np.random.default_rng(22).normal(size=(100, 3))
    with pytest.raises(ValueError, match="shape"):
        flopsearch.prune_parents_bic(data, np.zeros((2, 2)))
    cyclic = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
    with pytest.raises(ValueError, match="acyclic"):
        flopsearch.prune_parents_bic(data, cyclic)
