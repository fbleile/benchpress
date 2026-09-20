import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.shared import (
    SharedDagmaLinear,
)


def _fit(lambda1):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(32, 4))
    model = SharedDagmaLinear("l2")
    model.fit(
        X, initial_W=np.zeros((4, 4)), lambda1=lambda1,
        T=1, mu_schedule=(1.0,), s=(1.1,), warm_iter=3, max_iter=3,
        checkpoint=3,
    )
    return model


def test_scalar_vector_and_matrix_l1_penalties_are_accepted():
    scalar = _fit(.03)
    vector = _fit(np.full(4, .03))
    matrix = _fit(np.full((4, 4), .03))
    assert scalar.lambda1_matrix.shape == (4, 4)
    assert np.allclose(scalar.lambda1_matrix, vector.lambda1_matrix)
    assert np.allclose(scalar.lambda1_matrix, matrix.lambda1_matrix)
    assert np.all(np.diag(matrix.lambda1_matrix) == 0)


def test_target_vector_broadcasts_by_child_and_clears_diagonal():
    values = np.array([.01, .02, .03, .04])
    model = _fit(values)
    expected = np.broadcast_to(values[None, :], (4, 4)).copy()
    np.fill_diagonal(expected, 0.0)
    assert np.array_equal(model.lambda1_matrix, expected)
