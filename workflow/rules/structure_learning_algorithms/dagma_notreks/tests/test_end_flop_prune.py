import numpy as np
import pytest

from workflow.rules.structure_learning_algorithms.dagma.end_flop_prune import (
    end_flop_prune, exact_subset_parents, exact_subset_prune,
    local_gaussian_bic,
)
from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations, is_dag,
)


def data(seed=7, n=500):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = 1.7 * x0 + rng.normal(scale=.25, size=n)
    x2 = -.8 * x1 + rng.normal(scale=.3, size=n)
    x3 = rng.normal(size=n)
    return np.column_stack([x0, x1, x2, x3])


def test_end_flop_is_deterministic_subgraph_and_refits_ols():
    X = data()
    candidate = np.zeros((4, 4))
    candidate[0, 1] = candidate[0, 2] = candidate[1, 2] = 1
    candidate[0, 3] = candidate[2, 3] = 1
    first, coefficients, diagnostics = end_flop_prune(X, candidate)
    second, coefficients2, diagnostics2 = end_flop_prune(X, candidate)
    assert np.array_equal(first, second)
    assert np.all((first == 0) | (candidate != 0))
    assert is_dag(first)
    assert diagnostics["bic"] == diagnostics2["bic"]
    assert diagnostics["local_bics"] == diagnostics2["local_bics"]
    for child in range(4):
        parents = np.flatnonzero(first[:, child])
        if len(parents):
            expected, *_ = np.linalg.lstsq(
                X[:, parents] - X[:, parents].mean(0),
                X[:, child] - X[:, child].mean(), rcond=None)
            assert np.allclose(coefficients[parents, child], expected)
    assert np.allclose(coefficients, coefficients2)


def test_invalid_cycle_empty_and_single_parent():
    X = data()
    with pytest.raises(ValueError, match="DAG"):
        end_flop_prune(X, np.array([[0, 1], [1, 0]]))
    empty, coefficients, _ = end_flop_prune(X, np.zeros((4, 4)))
    assert not empty.any()
    assert not coefficients.any()
    one = np.zeros((4, 4))
    one[0, 1] = 1
    selected, *_ = end_flop_prune(X, one)
    assert np.all((selected == 0) | (one != 0))


def test_local_bic_and_exact_enumeration():
    X = data()
    parents, score = exact_subset_parents(X, 2, [0, 1, 3])
    brute = []
    for mask in range(8):
        subset = tuple(node for bit, node in enumerate([0, 1, 3])
                       if mask & (1 << bit))
        brute.append((local_gaussian_bic(X, 2, subset), len(subset), subset))
    expected = min(brute)
    assert parents == expected[2]
    assert score == pytest.approx(expected[0])

    candidate = np.zeros((4, 4))
    candidate[0, 1:] = 1
    candidate[1, 2:] = 1
    result, local = exact_subset_prune(X, candidate)
    assert np.all((result == 0) | (candidate != 0))
    assert local.sum() == pytest.approx(sum(
        local_gaussian_bic(X, child, np.flatnonzero(result[:, child]))
        for child in range(4)))


def test_deletion_preserves_zero_no_trek_violations():
    candidate = np.zeros((4, 4), dtype=int)
    candidate[0, 2] = candidate[1, 2] = candidate[2, 3] = 1
    pairs = [(0, 1)]
    assert common_ancestor_violations(candidate, pairs) == 0
    selected, *_ = end_flop_prune(data(), candidate)
    assert common_ancestor_violations(selected, pairs) == 0
