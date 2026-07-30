import json

import numpy as np
import pytest

from workflow.rules.structure_learning_algorithms.dagma.knowledge import (
    load_sidecar, named_pairs_to_indices, no_trek_pairs_from_dag, validate_sidecar,
)
from workflow.rules.structure_learning_algorithms.dagma.shared import notreks_value_grad


N = ["X0", "X1", "X2", "X3"]


@pytest.mark.parametrize("A,names,expected", [
    (np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), N[:3], set()),
    (np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]), N[:3], set()),
    (np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]), N[:3], {("X0", "X1")}),
    (np.zeros((3, 3), int), N[:3], {("X0", "X1"), ("X0", "X2"), ("X1", "X2")}),
    (np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]), N, set()),
])
def test_exact_oracle_pairs(A, names, expected):
    assert set(no_trek_pairs_from_dag(A, names)) == expected


@pytest.mark.parametrize("function", ["exp", "log", "inv", "binom"])
def test_path_characterization_and_positive_trek(function):
    # collider 0 -> 2 <- 1: (0,1) has no trek; (0,2) has a trek.
    W = np.array([[0., 0., .4], [0., 0., -.5], [0., 0., 0.]])
    no_trek, _ = notreks_value_grad(W, [(0, 1)], function)
    trek, _ = notreks_value_grad(W, [(0, 2)], function)
    assert no_trek == pytest.approx(0, abs=1e-13)
    assert trek > 0


@pytest.mark.parametrize("function", ["exp", "log", "inv", "binom"])
def test_gradient(function):
    rng = np.random.default_rng(4)
    W = rng.normal(scale=.08, size=(4, 4))
    np.fill_diagonal(W, 0)
    value, gradient = notreks_value_grad(W, [(0, 2), (1, 3)], function)
    numerical = np.zeros_like(W)
    eps = 1e-6
    for i in range(4):
        for j in range(4):
            step = np.zeros_like(W)
            step[i, j] = eps
            numerical[i, j] = (
                notreks_value_grad(W + step, [(0, 2), (1, 3)], function)[0]
                - notreks_value_grad(W - step, [(0, 2), (1, 3)], function)[0]
            ) / (2 * eps)
    assert value >= 0
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-8)


def test_scaling_empty_and_additive():
    W = np.array([[0., .2, .1], [.3, 0., 0.], [0., .4, 0.]])
    empty, grad = notreks_value_grad(W, [], "exp")
    one, _ = notreks_value_grad(W, [(0, 1)], "exp")
    two, _ = notreks_value_grad(W, [(0, 1), (0, 2)], "exp")
    assert empty == 0 and np.array_equal(grad, np.zeros_like(W))
    assert two >= one
    F = __import__("scipy").linalg.expm(W * W)
    raw = (F.T @ F)[0, 1]
    assert one == pytest.approx((2 / (len(W) - 1)) * raw)


def test_sidecar_validation_and_manual_file(tmp_path):
    payload = {"type": "no_trek_pairs", "source": "file",
               "node_names": ["B", "A"], "pairs": [["A", "B"]]}
    path = tmp_path / "knowledge.json"
    path.write_text(json.dumps(payload))
    loaded = load_sidecar(path, ["B", "A"])
    assert loaded["pairs"] == [["B", "A"]]
    assert named_pairs_to_indices(loaded) == [(0, 1)]
    with pytest.raises(ValueError, match="duplicate"):
        validate_sidecar({**payload, "pairs": [["A", "B"], ["B", "A"]]})
    with pytest.raises(ValueError, match="unknown"):
        validate_sidecar({**payload, "pairs": [["A", "C"]]})
    with pytest.raises(ValueError, match="self"):
        validate_sidecar({**payload, "pairs": [["A", "A"]]})
    with pytest.raises(ValueError, match="exactly match"):
        load_sidecar(path, ["A", "B"])
