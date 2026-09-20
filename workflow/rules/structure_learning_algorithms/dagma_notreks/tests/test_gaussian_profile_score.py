import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.shared import (
    SharedDagmaLinear,
)


def test_gaussian_profile_score_matches_profiled_residual_likelihood():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(80, 4))
    model = SharedDagmaLinear("gaussian_profile")
    model.fit(
        X, initial_W=np.zeros((4, 4)), lambda1=.03,
        T=1, mu_schedule=(1.0,), s=(1.1,), warm_iter=3, max_iter=3,
        checkpoint=3,
    )
    W = rng.normal(scale=.02, size=(4, 4))
    np.fill_diagonal(W, 0.0)
    value, gradient = model._score(W)
    centered = X - X.mean(axis=0, keepdims=True)
    residual = centered - centered @ W
    expected = .5 * np.log(
        np.maximum(np.sum(residual * residual, axis=0) / len(X),
                   model.variance_epsilon)
    ).sum()
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    # The production score uses the repository's float32 tensor path.
    np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-6)
