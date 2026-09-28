import numpy as np

from udg_pilot.nonparametric import (adjacency_from_pvalues, ensemble,
                                     pairwise_test, stability_selection)


def test_independence_is_not_rejected_reproducibly():
    rng = np.random.default_rng(20)
    x = rng.normal(size=(180, 2))
    a = pairwise_test(x, test="hsic", permutations=39, seed=4)
    b = pairwise_test(x, test="hsic", permutations=39, seed=4)
    assert np.array_equal(a.pvalue, b.pvalue)
    assert a.pvalue[0, 1] > 0.01


def test_zero_pearson_nonlinear_dependence_is_detected():
    rng = np.random.default_rng(21)
    x = rng.uniform(-1, 1, size=240)
    y = x**2 + rng.normal(scale=.04, size=x.size)
    data = np.column_stack([x, y])
    assert abs(np.corrcoef(data.T)[0, 1]) < .15
    hsic = pairwise_test(data, test="hsic", permutations=79, seed=5)
    dcov = pairwise_test(data, test="dcov", permutations=79, seed=5)
    assert hsic.pvalue[0, 1] < .05
    assert dcov.pvalue[0, 1] < .05


def test_dcov_rank_and_ensemble_are_symmetric():
    rng = np.random.default_rng(22)
    x = rng.standard_t(4, size=(160, 3))
    x[:, 1] += x[:, 0]**3
    h = pairwise_test(x, test="hsic", rank=True, permutations=19, seed=6)
    c = pairwise_test(x, test="dcov", rank=True, permutations=19, seed=6)
    assert np.allclose(h.statistic, h.statistic.T)
    assert np.allclose(c.pvalue, c.pvalue.T)
    conservative, liberal, uncertain = ensemble(h, c)
    assert np.array_equal(conservative, conservative.T)
    assert np.array_equal(liberal, liberal.T)
    assert np.all((conservative + uncertain) <= liberal)


def test_stability_path_and_frequency_reproducible():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(100, 4)); x[:, 1] += x[:, 0]
    kwargs = dict(test="dcov", replicates=4, subsample_fraction=.75,
                  permutations=9, seed=7)
    p1, f1 = stability_selection(x, **kwargs)
    p2, f2 = stability_selection(x, **kwargs)
    assert list(p1) == [.5, .6, .7, .8, .9, .95]
    assert np.array_equal(f1, f2)
    assert all(np.array_equal(p1[k], p2[k]) for k in p1)
    assert np.allclose(f1, f1.T)


def test_rank_transform_is_invariant_to_coordinatewise_monotone_maps_and_handles_nan():
    rng = np.random.default_rng(24)
    x = rng.normal(size=(80, 3)); transformed = np.exp(x)
    transformed[3, 1] = np.nan
    a = pairwise_test(x, test="dcov", rank=True, permutations=9, seed=8)
    b = pairwise_test(transformed, test="dcov", rank=True, permutations=9, seed=8)
    # The finite-pair ranks are identical; the NaN only changes the affected
    # pair's sample count, not the symmetry or the supported computation.
    assert np.allclose(a.pvalue[0, 2], b.pvalue[0, 2])
    assert np.allclose(b.pvalue, b.pvalue.T)
    assert b.sample_count[0, 1] == 79
