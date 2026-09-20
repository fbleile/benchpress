import numpy as np
import pytest

from scripts.causalassembly_protocol import (
    empirical_copula, nested_pairs, nonlinear_screen, oracle_no_treks,
    standardize_discovery,
)
from scripts.notreks_protocol_registry import REGISTRY


def test_causalassembly_is_registered_with_paired_grid():
    spec = REGISTRY["causalassembly"]
    assert spec.dataset == "causalassembly_full"
    assert spec.n_values == (500, 2000, 5000)
    assert spec.q_values == (.10, .25, .50, 1.0)
    assert spec.graph_replicates == 10
    assert spec.attempts_for("flop") == 20
    assert spec.attempts_for("dagma_notreks") == 2


def test_reflexive_ancestor_no_treks_and_nested_rounding():
    # 0 -> 2 <- 1: 0 and 1 have no common ancestor, but each shares itself
    # with its descendants and must not be certified independent.
    graph = np.zeros((3, 3), dtype=np.uint8)
    graph[0, 2] = graph[1, 2] = 1
    assert oracle_no_treks(graph) == [(0, 1)]
    pairs = [(0, 1), (0, 2), (1, 2), (3, 4)]
    first = nested_pairs(pairs, .5, 10)
    assert first == nested_pairs(pairs, .5, 10)
    assert set(first) <= set(pairs)
    assert nested_pairs(pairs, 1.0, 10) == pairs


def test_standardization_is_per_dataset_and_rejects_constants():
    x = np.arange(20, dtype=float).reshape(10, 2)
    z, mean, scale = standardize_discovery(x)
    np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std(0), 1.0)
    assert mean.shape == scale.shape == (2,)
    with pytest.raises(ValueError, match="near-constant"):
        standardize_discovery(np.column_stack([np.ones(10), np.arange(10)]))


def test_empirical_copula_handles_ties_and_is_bounded():
    u = empirical_copula(np.array([[0., 1.], [0., 2.], [1., 3.]]))
    assert np.isfinite(u).all()
    assert np.all((u > 0) & (u < 1))
    assert u[0, 0] == u[1, 0]


def test_nonlinear_screen_is_deterministic_and_detects_strong_nonlinear_signal():
    rng = np.random.default_rng(12)
    x = rng.normal(size=400)
    y = x * x + .02 * rng.normal(size=400)
    z = rng.normal(size=400)
    selected, diagnostics, summary = nonlinear_screen(
        np.column_stack([x, y, z]), candidate_cap=3, blocks=3,
        permutations=9, seed=44)
    selected_again, diagnostics_again, summary_again = nonlinear_screen(
        np.column_stack([x, y, z]), candidate_cap=3, blocks=3,
        permutations=9, seed=44)
    assert selected == selected_again
    assert diagnostics.equals(diagnostics_again)
    assert summary == summary_again
    assert (0, 1) not in selected
