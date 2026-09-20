import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import is_dag
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.sortnregress import (
    sortnregress,
)


def test_sortnregress_is_acyclic_and_reproducible():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(160, 8))
    first, first_diag = sortnregress(X, kind="variance")
    second, second_diag = sortnregress(X, kind="variance")
    assert is_dag(first)
    np.testing.assert_array_equal(first, second)
    assert first_diag["sortnregress_order"] == second_diag["sortnregress_order"]


def test_r2_sortnregress_is_invariant_to_coordinate_scaling():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(180, 7))
    scales = np.array([.2, 3.0, 1.7, .5, 4.0, .8, 2.5])
    first, first_diag = sortnregress(X, kind="r2")
    second, second_diag = sortnregress(X * scales, kind="r2")
    assert is_dag(first)
    assert is_dag(second)
    np.testing.assert_array_equal(first, second)
    assert first_diag["sortnregress_order"] == second_diag["sortnregress_order"]

