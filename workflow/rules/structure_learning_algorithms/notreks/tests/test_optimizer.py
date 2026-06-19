from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import scipy.linalg as sla

sys.path.append(str(Path(__file__).resolve().parents[1]))

from notreks_core import NotreksConfig, threshold_adjacency
from optimizer import (
    _dag_value_grad,
    _regularizer_value_grad,
    _score_value_grad,
    _stage_objective_value_grad,
    _trek_value_grad,
    fit_notreks_optimizer,
)


def _cfg(**overrides):
    base = NotreksConfig(
        algorithm_id="notreks-test",
        function_class="linear",
        score="least_squares",
        dag_seq="logdet",
        dag_reg=1.0,
        dag_s=1.0,
        trek_seq="none",
        trek_reg=0.0,
        regularizer="none",
        regularizer_scale=0.0,
        independence_test="none",
        independence_alpha=0.01,
        independence_correction="none",
        seed=1,
        max_iter=20,
        lr=0.0003,
        path_steps=3,
        mu_init=1.0,
        mu_factor=0.1,
        warm_iter=10,
        tol=1e-9,
        threshold=0.3,
        timeout=None,
        init="zero",
        checkpoint=5,
    )
    return replace(base, **overrides)


def _fixed_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 4))
    X[:, 1] += 0.4 * X[:, 0]
    X[:, 2] -= 0.3 * X[:, 1]
    X[:, 3] += 0.2 * X[:, 0] - 0.25 * X[:, 2]
    return X


def _fixed_W():
    W = np.array(
        [
            [0.0, 0.11, -0.17, 0.23],
            [-0.19, 0.0, 0.13, -0.29],
            [0.31, -0.07, 0.0, 0.37],
            [-0.41, 0.43, -0.47, 0.0],
        ],
        dtype=float,
    )
    return W


def _finite_difference_grad(W, objective, eps=1e-6):
    grad = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if i == j:
                continue
            plus = W.copy()
            minus = W.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            grad[i, j] = (objective(plus) - objective(minus)) / (2.0 * eps)
    return grad


def _assert_stage_grad_matches_finite_difference(cfg, pairs, mu=0.37, s=1.0):
    X = _fixed_data()
    W = _fixed_W()

    value, grad, *_ = _stage_objective_value_grad(W, X, cfg, pairs, mu=mu, s=s)
    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))

    def objective(W_candidate):
        obj, *_ = _stage_objective_value_grad(W_candidate, X, cfg, pairs, mu=mu, s=s)
        return obj

    fd_grad = _finite_difference_grad(W, objective)
    mask = ~np.eye(W.shape[0], dtype=bool)
    err = np.max(np.abs(grad[mask] - fd_grad[mask]))
    assert err < 1e-4, f"finite-difference gradient mismatch: {err}"


def _assert_score_grad_matches_finite_difference(score):
    X = _fixed_data()
    W = _fixed_W()
    _, grad = _score_value_grad(X, W, score)

    def objective(W_candidate):
        value, _ = _score_value_grad(X, W_candidate, score)
        return value

    fd_grad = _finite_difference_grad(W, objective)
    mask = ~np.eye(W.shape[0], dtype=bool)
    err = np.max(np.abs(grad[mask] - fd_grad[mask]))
    assert err < 1e-4, f"{score} score gradient mismatch: {err}"


def _assert_dag_grad_matches_finite_difference(seq, s=1.0):
    W = 0.25 * _fixed_W()
    _, grad = _dag_value_grad(W, seq, s)

    def objective(W_candidate):
        value, _ = _dag_value_grad(W_candidate, seq, s)
        return value

    fd_grad = _finite_difference_grad(W, objective)
    mask = ~np.eye(W.shape[0], dtype=bool)
    err = np.max(np.abs(grad[mask] - fd_grad[mask]))
    assert err < 1e-4, f"{seq} DAG gradient mismatch: {err}"


def test_stage_gradient_without_trek():
    cfg = _cfg(trek_seq="none", trek_reg=0.0, regularizer="none")
    _assert_stage_grad_matches_finite_difference(cfg, pairs=[])


def test_stage_gradient_with_trek_inside_mu():
    cfg = _cfg(trek_seq="exp", trek_reg=0.8, regularizer="none")
    _assert_stage_grad_matches_finite_difference(cfg, pairs=[(0, 2), (1, 3)])


def test_stage_trek_scaling_is_inside_mu():
    X = _fixed_data()
    W = _fixed_W()
    cfg_no_trek = _cfg(trek_seq="exp", trek_reg=0.0, regularizer="none")
    cfg_with_trek = _cfg(trek_seq="exp", trek_reg=0.8, regularizer="none")
    pairs = [(0, 2), (1, 3)]

    value_no_trek, *_ = _stage_objective_value_grad(W, X, cfg_no_trek, pairs, mu=0.1, s=1.0)
    value_with_trek, *_ = _stage_objective_value_grad(W, X, cfg_with_trek, pairs, mu=0.1, s=1.0)
    _, _, _, _, trek_value, _ = _stage_objective_value_grad(W, X, cfg_with_trek, pairs, mu=1.0, s=1.0)

    observed = value_with_trek - value_no_trek
    expected = 0.1 * cfg_with_trek.trek_reg * trek_value
    assert abs(observed - expected) < 1e-8


def test_stage_gradient_with_nonsmooth_l1_away_from_zero():
    cfg = _cfg(trek_seq="none", trek_reg=0.0, regularizer="l1", regularizer_scale=0.05)
    _assert_stage_grad_matches_finite_difference(cfg, pairs=[])


def test_score_gradient_least_squares():
    _assert_score_grad_matches_finite_difference("least_squares")


def test_score_gradient_gaussian_likelihood():
    _assert_score_grad_matches_finite_difference("gaussian_likelihood")


def test_dag_gradient_exp():
    _assert_dag_grad_matches_finite_difference("exp")


def test_dag_gradient_logdet():
    _assert_dag_grad_matches_finite_difference("logdet", s=1.0)


def test_regularizer_values_and_gradients():
    W = _fixed_W()

    value, grad = _regularizer_value_grad(W, "none")
    assert value == 0.0
    assert np.array_equal(grad, np.zeros_like(W))

    value, grad = _regularizer_value_grad(W, "l1")
    assert abs(value - np.abs(W).sum()) < 1e-12
    assert np.array_equal(grad, np.sign(W))

    value, grad = _regularizer_value_grad(W, "l2")
    assert abs(value - np.sum(W * W)) < 1e-12
    assert np.allclose(grad, 2.0 * W)


def test_empty_trek_pairs_zero_value_and_gradient():
    W = _fixed_W()
    value, grad = _trek_value_grad(W, "exp", [])
    assert value == 0.0
    assert np.array_equal(grad, np.zeros_like(W))


def test_trek_penalty_sums_over_pairs():
    W = _fixed_W()
    one_value, one_grad = _trek_value_grad(W, "exp", [(0, 2)])
    duplicate_value, duplicate_grad = _trek_value_grad(W, "exp", [(0, 2), (0, 2)])
    assert abs(duplicate_value - 2.0 * one_value) < 1e-12
    assert np.allclose(duplicate_grad, 2.0 * one_grad)


def test_scores_are_invariant_to_duplicate_samples():
    X = _fixed_data()
    X_twice = np.vstack([X, X])
    W = _fixed_W()

    for score in ["least_squares", "gaussian_likelihood"]:
        value, _ = _score_value_grad(X, W, score)
        value_twice, _ = _score_value_grad(X_twice, W, score)
        assert abs(value_twice - value) < 1e-7, score


def test_regularizer_is_raw_not_dimension_averaged():
    W = _fixed_W()

    l1_value, _ = _regularizer_value_grad(W, "l1")
    l2_value, _ = _regularizer_value_grad(W, "l2")

    assert abs(l1_value - np.abs(W).sum()) < 1e-12
    assert abs(l2_value - np.sum(W * W)) < 1e-12


def test_dag_penalties_are_raw_not_dimension_averaged():
    W = 0.25 * _fixed_W()
    d = W.shape[0]

    exp_value, _ = _dag_value_grad(W, "exp", s=1.0)
    expected_exp = np.trace(sla.expm(W * W)) - d
    assert abs(exp_value - expected_exp) < 1e-12

    logdet_value, _ = _dag_value_grad(W, "logdet", s=1.0)
    M = np.eye(d) - W * W
    sign, logdet = np.linalg.slogdet(M)
    assert sign > 0
    expected_logdet = -logdet
    assert abs(logdet_value - expected_logdet) < 1e-12


def test_invalid_trek_pairs_raise_value_error():
    W = _fixed_W()
    for pairs in [[(0, 0)], [(-1, 2)], [(0, W.shape[0])]]:
        try:
            _trek_value_grad(W, "exp", pairs)
        except ValueError as exc:
            assert "independence_pairs" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for pairs={pairs}")


def test_threshold_name_warning_is_present():
    text = (Path(__file__).resolve().parents[1] / "script.py").read_text()
    assert "threshold008" in text
    assert "configured threshold" in text


def test_optimizer_no_trek_sanity():
    X = _fixed_data()
    cfg = _cfg(
        trek_seq="none",
        trek_reg=0.0,
        regularizer="none",
        regularizer_scale=0.0,
        max_iter=30,
        threshold=0.2,
    )
    W, diagnostics = fit_notreks_optimizer(X, cfg, independence_pairs=[])
    assert np.all(np.isfinite(W))
    assert np.allclose(np.diag(W), 0.0)
    assert cfg.init == "zero"
    assert np.any((np.abs(W) > 0.0) & (np.abs(W) < cfg.threshold))
    assert np.array_equal(threshold_adjacency(W, cfg.threshold), (np.abs(W) > cfg.threshold).astype(int))
    assert not np.array_equal(W.astype(int), W)

    successful = [stage for stage in diagnostics.stages if stage.get("success")]
    assert len(successful) >= 1
    objectives = [stage["objective"] for stage in successful]
    assert all(np.isfinite(objectives))
    assert objectives[-1] <= objectives[0]


if __name__ == "__main__":
    test_stage_gradient_without_trek()
    test_stage_gradient_with_trek_inside_mu()
    test_stage_trek_scaling_is_inside_mu()
    test_stage_gradient_with_nonsmooth_l1_away_from_zero()
    test_score_gradient_least_squares()
    test_score_gradient_gaussian_likelihood()
    test_dag_gradient_exp()
    test_dag_gradient_logdet()
    test_regularizer_values_and_gradients()
    test_empty_trek_pairs_zero_value_and_gradient()
    test_trek_penalty_sums_over_pairs()
    test_scores_are_invariant_to_duplicate_samples()
    test_regularizer_is_raw_not_dimension_averaged()
    test_dag_penalties_are_raw_not_dimension_averaged()
    test_invalid_trek_pairs_raise_value_error()
    test_threshold_name_warning_is_present()
    test_optimizer_no_trek_sanity()
    print("optimizer tests passed")
