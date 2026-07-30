import numpy as np

from dagma.linear import DagmaLinear
from workflow.rules.structure_learning_algorithms.dagma.shared import SharedDagmaLinear


def test_vanilla_zero_weight_empty_equivalence():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(40, 4))
    settings = dict(T=1, warm_iter=12, max_iter=12, checkpoint=4,
                    lr=.0003, w_threshold=0.0)
    official = DagmaLinear("l2").fit(X.copy(), **settings)
    vanilla = SharedDagmaLinear("l2").fit(X.copy(), **settings)
    zero = SharedDagmaLinear("l2").fit(
        X.copy(), no_trek_pairs=[(0, 2)], trek_weight=0, **settings)
    empty = SharedDagmaLinear("l2").fit(
        X.copy(), no_trek_pairs=[], trek_weight=3, **settings)
    np.testing.assert_array_equal(official, vanilla)
    np.testing.assert_array_equal(vanilla, zero)
    np.testing.assert_array_equal(vanilla, empty)
    np.testing.assert_array_equal(abs(vanilla) >= .01, abs(zero) >= .01)


def test_stage_diagnostics_follow_central_path_schedule():
    X = np.random.default_rng(7).normal(size=(30, 3))
    model = SharedDagmaLinear("l2")
    model.fit(X, T=2, warm_iter=3, max_iter=4, checkpoint=1,
              s=[1.0, 0.9], w_threshold=0.0)
    assert [row["stage"] for row in model.stage_diagnostics] == [1, 2]
    assert [row["s"] for row in model.stage_diagnostics] == [1.0, 0.9]
    assert [row["mu"] for row in model.stage_diagnostics] == [1.0, 0.1]
    assert model.stage_diagnostics[0]["iterations_performed"] <= 3
    assert model.stage_diagnostics[1]["iterations_performed"] <= 4
