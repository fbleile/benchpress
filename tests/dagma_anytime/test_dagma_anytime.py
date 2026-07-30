import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagma.linear import DagmaLinear
from workflow.rules.structure_learning_algorithms.dagma_anytime.graph_utils import (
    is_dag,
    postprocess_graph,
    project_to_dag,
)
from workflow.rules.structure_learning_algorithms.dagma_anytime.solver import (
    LinearDagmaKernel,
    fit_linear_dagma_anytime,
    fused_inverse_logdet,
)


def toy_data(seed=7, n=80, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[:, 2] += 0.8 * X[:, 0]
    X[:, 4] += -0.6 * X[:, 1] + 0.3 * X[:, 2]
    return X


def test_baseline_equivalence_small_no_timeout():
    X = toy_data()
    kwargs = dict(lambda1=0.02, w_threshold=0.1, T=2, mu_init=1.0,
                  mu_factor=0.1, s=[1.0, 0.9], warm_iter=5, max_iter=7,
                  lr=0.0003, checkpoint=100, beta_1=0.99, beta_2=0.999)
    vanilla = DagmaLinear("l2", dtype=np.float64).fit(X.copy(), **kwargs)
    res = fit_linear_dagma_anytime(X.copy(), method="dagma_vanilla_anytime", **kwargs)
    assert np.allclose(res.weighted_adjacency_raw * (np.abs(res.weighted_adjacency_raw) >= 0.1), vanilla, atol=1e-10)
    assert np.array_equal(res.adjacency_thresholded_raw, (vanilla != 0).astype(int))
    assert res.iterations_by_stage == [5, 7]


def test_fast64_kernel_values_match_vanilla_components():
    X = toy_data(seed=8)
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / Xc.shape[0]
    W = np.triu(np.ones((5, 5)) * 0.02, 1)
    vanilla = DagmaLinear("l2", dtype=np.float64)
    vanilla.X = Xc.copy(); vanilla.n, vanilla.d = Xc.shape
    vanilla.Id = np.eye(5); vanilla.cov = cov; vanilla.lambda1 = 0.02
    kernel = LinearDagmaKernel(cov, lambda1=0.02, precision_policy="float64")
    score_v, grad_v = vanilla._score(W)
    h_v, gh_v = vanilla._h(W, 1.0)
    score_k, grad_k = kernel.score(W)
    h_k, gh_k, margin = kernel.h(W, 1.0)
    assert np.allclose(score_k, score_v, atol=1e-12)
    assert np.allclose(grad_k, grad_v, atol=1e-12)
    assert np.allclose(h_k, h_v, atol=1e-12)
    assert np.allclose(gh_k, gh_v, atol=1e-12)
    assert margin > 0


def test_fused_inverse_logdet_matches_two_call_reference():
    rng = np.random.default_rng(901)
    W = rng.normal(scale=0.03, size=(5, 5))
    np.fill_diagonal(W, 0)
    matrix = np.eye(5) - W * W
    inverse_transpose, logdet, sign, valid = fused_inverse_logdet(matrix)
    reference_sign, reference_logdet = np.linalg.slogdet(matrix)
    np.testing.assert_allclose(
        inverse_transpose, np.linalg.inv(matrix).T,
        rtol=1e-12, atol=1e-12)
    assert np.isclose(logdet, reference_logdet)
    assert sign == reference_sign
    assert valid


def test_precision_boundary_mixed_more_reliable_than_float32():
    d = 6
    cov = np.eye(d)
    W = np.zeros((d, d))
    for i in range(d - 1):
        W[i, i + 1] = 0.35
    W[0, 2] = 0.2
    k64 = LinearDagmaKernel(cov, 0.01, "float64")
    k32 = LinearDagmaKernel(cov, 0.01, "float32")
    kmix = LinearDagmaKernel(cov, 0.01, "mixed")
    h64, g64, _ = k64.h(W.astype(np.float64), 0.9)
    h32, g32, _ = k32.h(W.astype(np.float32), 0.9)
    hm, gm, _ = kmix.h(W.astype(np.float32), 0.9)
    err32 = abs(h32 - h64) + np.linalg.norm(g32.astype(float) - g64)
    errm = abs(hm - h64) + np.linalg.norm(gm.astype(float) - g64)
    assert errm <= err32 + 1e-8


def test_timeout_returns_result_and_snapshots():
    X = toy_data(n=120, d=8)
    res = fit_linear_dagma_anytime(
        X, method="dagma_fast64", T=3, warm_iter=10000, max_iter=10000,
        checkpoint=20, checkpoint_interval=20, max_runtime_seconds=0.01,
        snapshot_times_seconds=[0.001, 0.005, 0.01], w_threshold=0.1,
    )
    assert res.termination_reason == "max_runtime"
    assert res.timed_out
    assert res.weighted_adjacency_best.shape == (8, 8)
    ts = [c.elapsed_seconds for c in res.checkpoint_history]
    assert ts == sorted(ts)


def test_graph_projection_is_deterministic_dag_and_zero_diagonal():
    A = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
    W = np.array([[9.0, 0.5, 0.0], [0.0, 0.0, 0.2], [0.3, 0.0, 0.0]])
    projected, removed = project_to_dag(A, W)
    assert is_dag(projected)
    assert removed == [(1, 2, 0.2)]
    pp = postprocess_graph(W, 0.1)
    assert np.all(np.diag(pp.thresholded_raw) == 0)
