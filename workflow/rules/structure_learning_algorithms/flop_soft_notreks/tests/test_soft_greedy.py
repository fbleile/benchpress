import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import is_dag
from workflow.rules.structure_learning_algorithms.flop_soft_notreks import (
    SoftGreedyConfig,
    fit_soft_notreks,
)


def test_soft_search_returns_dag_and_is_reproducible():
    X = np.random.default_rng(12).normal(size=(80, 5))
    cfg = SoftGreedyConfig(seed=31, restarts=2, max_sweeps=3, lazy_top_k=4)
    first = fit_soft_notreks(X, [(0, 1)], cfg)
    second = fit_soft_notreks(X, [(0, 1)], cfg)
    assert np.array_equal(first.adjacency, second.adjacency)
    assert is_dag(first.adjacency)
    assert np.all(np.diag(first.adjacency) == 0)
    assert first.notreks == 0.0


def test_soft_search_exposes_raw_and_feasible_scores():
    X = np.random.default_rng(13).normal(size=(70, 4))
    result = fit_soft_notreks(
        X, [(0, 1)], SoftGreedyConfig(seed=8, restarts=2, max_sweeps=2))
    assert result.support_evaluations >= 1
    assert result.distinct_supports == result.support_evaluations
    assert np.isfinite(result.bic)
    assert np.isfinite(result.raw_bic)
