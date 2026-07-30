import importlib.util

import numpy as np
import pytest

from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag


def test_cpdag_conversion():
    raw = np.array([[0, 2, 1], [2, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(
        convert_flop_cpdag(raw, 3),
        np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]]))
    with pytest.raises(ValueError, match="symmetric"):
        convert_flop_cpdag(np.array([[0, 2], [0, 0]]), 2)


def test_flop_import_or_actionable_skip():
    if importlib.util.find_spec("flopsearch") is None:
        pytest.skip("flopsearch==0.3.0 is not installed; install the released wheel with pip")
    import flopsearch
    assert callable(flopsearch.flop)
