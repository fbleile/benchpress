import numpy as np
import pytest

pyscipopt = pytest.importorskip("pyscipopt")

from workflow.rules.structure_learning_algorithms.flop_notreks.exact_solver import (
    ExactConfig, ancestor_bitsets, exhaustive_optimum, fit_exact_notreks,
    no_trek_violations, trek_witness,
)


@pytest.mark.parametrize("seed", [3, 17, 91])
def test_lazy_branch_and_cut_matches_exhaustive_optimum(seed):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(90, 4))
    data[:, 2] += .7 * data[:, 0]
    data[:, 3] += .6 * data[:, 1] + .3 * data[:, 0]
    pairs = [(2, 3)]
    expected, _ = exhaustive_optimum(data, pairs, 2)
    result = fit_exact_notreks(
        data, pairs, ExactConfig(max_indegree=2, variant="lazy_trek",
                                 time_limit_seconds=30, random_seed=seed))
    assert result.zero_gap_certificate
    assert result.dag_verified and result.notreks_verified
    assert result.objective == pytest.approx(expected, abs=1e-7)


def test_lazy_trek_cut_removes_a_common_ancestor_incumbent():
    rng = np.random.default_rng(9)
    data = rng.normal(size=(1000, 4))
    data[:, 2] = 1.2 * data[:, 0] + rng.normal(scale=.3, size=1000)
    data[:, 3] = 1.1 * data[:, 0] + rng.normal(scale=.3, size=1000)
    source_only = fit_exact_notreks(
        data, [(2, 3)], ExactConfig(variant="source_cut"))
    exact = fit_exact_notreks(
        data, [(2, 3)], ExactConfig(variant="lazy_trek"))
    assert not source_only.notreks_verified
    assert exact.trek_cuts > 0
    assert exact.zero_gap_certificate and exact.notreks_verified


def test_bitset_ancestry_and_witness_paths_are_independently_checkable():
    adjacency = np.zeros((5, 5), dtype=np.uint8)
    adjacency[0, 1] = adjacency[1, 3] = adjacency[0, 2] = adjacency[2, 4] = 1
    ancestors = ancestor_bitsets(adjacency)
    assert ancestors[3] & ancestors[4] == 1
    assert no_trek_violations(adjacency, [(3, 4)]) == [(3, 4)]
    source, edges = trek_witness(adjacency, (3, 4))
    assert source == 0
    assert edges == {(0, 1), (1, 3), (0, 2), (2, 4)}
