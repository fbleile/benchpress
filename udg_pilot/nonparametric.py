"""Assumption-light pairwise marginal-dependence evidence.

This module never constructs a DAG/UEC/order or passes pair estimates to a
causal optimizer.  It returns only symmetric pairwise dependence graphs and
diagnostic statistic/p-value matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


@dataclass
class PairwiseEvidence:
    statistic: np.ndarray
    pvalue: np.ndarray
    sample_count: np.ndarray
    permutations: int
    transform: str
    test: str


def _rank_columns(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = rankdata(x[:, j], method="average") / (x.shape[0] + 1.0)
    return out


def _prepare_pair(x, i, j, rank):
    a = np.asarray(x[:, i], dtype=float); b = np.asarray(x[:, j], dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if rank:
        a = rankdata(a, method="average") / (len(a) + 1.0)
        b = rankdata(b, method="average") / (len(b) + 1.0)
    return a, b


def _rbf_kernel(x):
    distances = squareform(pdist(x[:, None], metric="euclidean"))
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    bandwidth = max(bandwidth, np.finfo(float).eps)
    return np.exp(-(distances ** 2) / (2.0 * bandwidth ** 2))


def _hsic_unbiased_from_kernels(k, l):
    n = k.shape[0]
    if n < 4:
        return 0.0
    k = k.copy(); l = l.copy()
    np.fill_diagonal(k, 0.0); np.fill_diagonal(l, 0.0)
    term1 = float(np.sum(k * l))
    term2 = float(np.sum(k) * np.sum(l) / ((n - 1) * (n - 2)))
    term3 = float(2.0 * np.sum(np.sum(k, axis=1) * np.sum(l, axis=1)) / (n - 2))
    return (term1 + term2 - term3) / (n * (n - 3))


def _distance_covariance(a, b):
    aa = squareform(pdist(a[:, None], metric="euclidean"))
    bb = squareform(pdist(b[:, None], metric="euclidean"))
    aa -= aa.mean(axis=0, keepdims=True); aa -= aa.mean(axis=1, keepdims=True); aa += aa.mean()
    bb -= bb.mean(axis=0, keepdims=True); bb -= bb.mean(axis=1, keepdims=True); bb += bb.mean()
    return float(max(0.0, np.mean(aa * bb)))


def _pair_test(a, b, test, permutations, rng):
    n = len(a)
    if n < 8 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0, 1.0
    if test == "hsic":
        ka, kb = _rbf_kernel(a), _rbf_kernel(b)
        observed = max(0.0, _hsic_unbiased_from_kernels(ka, kb))
        null = np.empty(permutations)
        for r in range(permutations):
            p = rng.permutation(n)
            null[r] = max(0.0, _hsic_unbiased_from_kernels(ka, kb[np.ix_(p, p)]))
    else:
        observed = _distance_covariance(a, b)
        null = np.empty(permutations)
        for r in range(permutations):
            p = rng.permutation(n)
            null[r] = _distance_covariance(a, b[p])
    return observed, float((1 + np.sum(null >= observed)) / (permutations + 1))


def pairwise_test(x, *, test="hsic", rank=False, permutations=99, seed=0):
    """Compute symmetric pairwise statistic and permutation p-value matrices."""
    x = np.asarray(x, dtype=float)
    d = x.shape[1]; statistic = np.zeros((d, d)); pvalue = np.ones((d, d)); counts = np.zeros((d, d), dtype=int)
    for i, j in combinations(range(d), 2):
        a, b = _prepare_pair(x, i, j, rank)
        pair_seed = (int(seed) ^ (0x9E3779B97F4A7C15 * (i + 1) + j)) % (2**63 - 1)
        rng = np.random.default_rng(pair_seed)
        stat, p = _pair_test(a, b, test, permutations, rng)
        statistic[i, j] = statistic[j, i] = stat
        pvalue[i, j] = pvalue[j, i] = p
        counts[i, j] = counts[j, i] = len(a)
    return PairwiseEvidence(statistic, pvalue, counts, permutations,
                            "rank" if rank else "raw", test)


def adjacency_from_pvalues(evidence, alpha=0.05):
    # Permutation p-values are discrete; include the boundary so that a
    # prespecified alpha is not made artificially more conservative when
    # (1+B*alpha) is an integer.
    out = (evidence.pvalue <= alpha).astype(np.uint8)
    np.fill_diagonal(out, 0)
    return out


def stability_selection(x, *, test="hsic", rank=False, replicates=20,
                        subsample_fraction=0.8, alpha=0.05, thresholds=(.50, .60, .70, .80, .90, .95),
                        permutations=49, seed=0):
    x = np.asarray(x, dtype=float); n, d = x.shape
    frequencies = np.zeros((d, d), dtype=float); rng = np.random.default_rng(seed)
    for r in range(replicates):
        size = max(8, min(n, int(round(n * subsample_fraction))))
        indices = np.sort(rng.choice(n, size=size, replace=False))
        ev = pairwise_test(x[indices], test=test, rank=rank, permutations=permutations,
                           seed=int(rng.integers(0, 2**32 - 1)))
        frequencies += adjacency_from_pvalues(ev, alpha)
    frequencies /= max(1, replicates)
    path = {float(t): ((frequencies >= t).astype(np.uint8) - np.diag(np.diag(frequencies >= t))).astype(np.uint8)
            for t in thresholds}
    return path, frequencies


def ensemble(hsic, dcov, alpha=0.05):
    h = adjacency_from_pvalues(hsic, alpha).astype(bool)
    c = adjacency_from_pvalues(dcov, alpha).astype(bool)
    conservative = (h & c).astype(np.uint8)
    liberal = (h | c).astype(np.uint8)
    uncertain = np.logical_xor(h, c).astype(np.uint8)
    np.fill_diagonal(conservative, 0); np.fill_diagonal(liberal, 0); np.fill_diagonal(uncertain, 0)
    return conservative, liberal, uncertain
