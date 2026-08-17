from itertools import product

import numpy as np
import pytest

pytest.importorskip("flopsearch")

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic, is_dag,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.exact_solver import (
    no_trek_violations,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.local_window_ablation import (
    LocalWindowConfig, fit_local_window_ablation, solve_parent_family_block,
)


def case():
    rng = np.random.default_rng(3301)
    X = rng.normal(size=(240, 5))
    X[:, 2] += .8 * X[:, 0]
    X[:, 4] += .7 * X[:, 1]
    return X


def test_block_matches_independent_declared_family_enumeration():
    X = case()
    baseline = np.zeros((5, 5), dtype=np.uint8)
    order = [0, 1, 2, 3, 4]
    block = [2, 4]
    families = {
        2: [(), (0,), (1,)],
        4: [(), (0,), (1,), (2,), (1, 2)],
    }
    result = solve_parent_family_block(
        X, [(2, 4)], baseline, order, block, families,
        time_limit_seconds=10.)
    expected = None
    for parents_2, parents_4 in product(families[2], families[4]):
        graph = baseline.copy()
        graph[list(parents_2), 2] = 1
        graph[list(parents_4), 4] = 1
        if no_trek_violations(graph, [(2, 4)]):
            continue
        score, _ = gaussian_bic(X, graph, lambda_bic=2.)
        key = (score, graph.tobytes(), graph)
        if expected is None or key[:2] < expected[:2]:
            expected = key
    assert result.certified and not result.timed_out
    assert result.absolute_gap == 0
    assert result.score == pytest.approx(expected[0], abs=1e-8)
    np.testing.assert_array_equal(result.adjacency, expected[2])


def test_gflop_low_dimensional_real_smoke_is_hard_feasible():
    result = fit_local_window_ablation(
        case(), [(2, 4)], LocalWindowConfig(
            block_size=4, sweeps=1, initial_flop_runs=2,
            candidate_max_parents=2, max_families_per_node=12,
            block_time_limit_seconds=2., seed=119))
    assert is_dag(result.adjacency)
    assert result.no_trek_violations == 0
    assert result.number_of_block_calls > 0
    assert result.certified_block_solves == result.number_of_block_calls
    assert result.timed_out_block_solves == 0


def test_block_timeout_returns_certified_feasible_baseline_with_gap():
    X = case()
    baseline = np.zeros((5, 5), dtype=np.uint8)
    result = solve_parent_family_block(
        X, [(2, 4)], baseline, [0, 1, 2, 3, 4], [2, 4],
        {2: [(), (0,), (1,)], 4: [(), (0,), (1,), (2,)]},
        time_limit_seconds=0.)
    assert result.timed_out and not result.certified
    assert result.combinations_evaluated == 0
    assert result.absolute_gap >= 0
    np.testing.assert_array_equal(result.adjacency, baseline)
    assert not no_trek_violations(result.adjacency, [(2, 4)])
