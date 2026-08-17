import numpy as np
import pytest

flopsearch = pytest.importorskip("flopsearch")

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    is_dag,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.flop_support import (
    build_flop_union_support,
    excluded_edges,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    ProductionConfig,
    run_production_pipeline,
)


def data():
    rng = np.random.default_rng(281)
    X = rng.normal(size=(180, 6))
    X[:, 2] += .8 * X[:, 0]
    X[:, 4] += .7 * X[:, 1] - .5 * X[:, 3]
    return X


def test_flop_union_is_symmetrized_and_reproducible():
    first = build_flop_union_support(data(), [(2, 4)], runs=2, seed=91)
    second = build_flop_union_support(data(), [(2, 4)], runs=2, seed=91)
    np.testing.assert_array_equal(first.allowed_arcs, first.allowed_arcs.T)
    np.testing.assert_array_equal(first.allowed_arcs, second.allowed_arcs)
    assert np.all(np.diag(first.allowed_arcs) == 0)
    assert first.diagnostics == second.diagnostics
    exclusions = excluded_edges(first.allowed_arcs)
    assert all(not first.allowed_arcs[parent, child]
               for parent, child in exclusions)


def test_real_masked_dagma_notreks_smoke_is_hard_feasible():
    selected, restarts = run_production_pipeline(
        data(), [(2, 4)], ProductionConfig(
            support_mode="flop_union_support", flop_support_runs=2,
            support_initialization="zero_then_best_feasible_flop",
            restarts=2, seed=101, T=1, s=(1.,), warm_iter=30,
            max_iter=30, checkpoint=10, screening_floor=.05))
    assert len(restarts) == 2
    assert selected.support["support_mode"] == "flop_union_support"
    assert selected.support["support_is_symmetrized"]
    assert selected.support["support_density"] < 1
    assert is_dag(selected.adjacency)
    assert common_ancestor_violations(selected.adjacency, [(2, 4)]) == 0
    allowed = build_flop_union_support(
        (data() - data().mean(0)) / data().std(0), [(2, 4)],
        runs=2, seed=101).allowed_arcs
    assert not np.any((selected.adjacency != 0) & (allowed == 0))
