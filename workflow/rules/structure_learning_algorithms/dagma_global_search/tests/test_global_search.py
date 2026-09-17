import numpy as np

from workflow.rules.structure_learning_algorithms.dagma_global_search.objective import ObjectiveAdapter
from workflow.rules.structure_learning_algorithms.dagma_global_search.methods import (
    _systematic_resample, log_acceptance, run_pt_basin, swap_log_acceptance,
)


def test_adapter_terms_match_existing_kernels():
    rng = np.random.default_rng(4); X = rng.normal(size=(40, 4)); W = rng.normal(scale=.02, size=(4, 4)); np.fill_diagonal(W, 0)
    adapter = ObjectiveAdapter(X, [(0, 1), (2, 3)], trek_weight=1.)
    stage = adapter.stage(1.); got = adapter.components(W, stage)
    data, _ = adapter.model._score(W); dag, _ = adapter.model._h(W, stage.s)
    nt, _ = adapter.model._trek_kernel.value_grad(W, adapter.trek_function, log_terms=adapter.trek_log_terms, inverse_epsilon=adapter.inverse_epsilon)
    assert got["valid_domain"]
    assert np.allclose([got["data"], got["dag"], got["notreks"]], [data, dag, nt])
    assert np.isclose(got["total"], stage.mu * (data + adapter.lambda1*np.abs(W).sum()) + dag + nt)


def test_adapter_gradient_finite_difference():
    rng = np.random.default_rng(5); X = rng.normal(size=(45, 3)); W = rng.normal(scale=.015, size=(3, 3)); np.fill_diagonal(W, 0)
    adapter = ObjectiveAdapter(X, [(0, 1)]); stage = adapter.stage(.8); analytic = adapter.components(W, stage)["gradient"]
    numeric = np.zeros_like(W); eps = 1e-6
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        plus=W.copy(); minus=W.copy(); plus[i,j]+=eps; minus[i,j]-=eps
        numeric[i,j]=(adapter.components(plus,stage)["total"]-adapter.components(minus,stage)["total"])/(2*eps)
    assert np.allclose(analytic[[0,1,2],[1,2,0]], numeric[[0,1,2],[1,2,0]], rtol=2e-3, atol=2e-5)


def test_domain_rejection_is_finite():
    adapter=ObjectiveAdapter(np.ones((20,3)),[(0,1)]); W=np.ones((3,3)); np.fill_diagonal(W,0)
    out=adapter.components(W,adapter.stage(1.)); assert not out["valid_domain"]; assert np.isinf(out["total"]); assert np.isfinite(out["gradient"]).all()


def test_acceptance_signs_and_equal_swap():
    assert log_acceptance(-2., 1.) == 0.; assert log_acceptance(2., 1.) < 0
    assert swap_log_acceptance(1., 1., 1., 2.) == 0.
    assert swap_log_acceptance(1., 2., 1., 2.) < 0


def test_systematic_resampling_seeded():
    idx=_systematic_resample(np.array([.5,.25,.25]),np.random.default_rng(2)); assert np.array_equal(idx,np.array([0,0,2]))


def test_fixed_seed_pt_returns_reproducible_feasible_graph():
    X=np.random.default_rng(8).normal(size=(30,3)); pairs=[(0,1)]
    g1,d1=run_pt_basin(X,pairs,seed=7,replicas=2,stages=2,steps_per_stage=1,warm_iter=2,max_iter=3)
    g2,d2=run_pt_basin(X,pairs,seed=7,replicas=2,stages=2,steps_per_stage=1,warm_iter=2,max_iter=3)
    assert np.array_equal(g1,g2); assert np.all(np.diag(g1)==0); assert d1["archive_size"] == d2["archive_size"]
