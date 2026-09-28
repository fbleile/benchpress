"""Direct covariance-graph utilities for the isolated UDG pilot.

The graph constraint is on zeros of Sigma, never on zeros of Sigma^{-1}.
The small Python fitter is a transparent reference implementation; the
benchmark hot path uses the Rust implementation in ``udg_pilot/rust``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from scipy.optimize import minimize


def edge_index(i: int, j: int, d: int) -> int:
    if i > j:
        i, j = j, i
    return sum(d - 1 - k for k in range(i)) + (j - i - 1)


def pairs(d: int):
    return list(combinations(range(d), 2))


def matrix_to_bits(a: np.ndarray) -> int:
    d = a.shape[0]
    bits = 0
    for i, j in pairs(d):
        if a[i, j]:
            bits |= 1 << edge_index(i, j, d)
    return bits


def bits_to_matrix(bits: int, d: int) -> np.ndarray:
    a = np.zeros((d, d), dtype=np.uint8)
    for i, j in pairs(d):
        if bits & (1 << edge_index(i, j, d)):
            a[i, j] = a[j, i] = 1
    return a


def edge_count(bits: int) -> int:
    return int(bits.bit_count())


def smig_valid(bits: int, d: int) -> bool:
    """Check the simple marginal-independence graph characterization.

    A vertex is simplicial when its closed neighbourhood is a clique.  The
    maximal closed neighbourhoods of simplicial vertices are the simplices;
    every edge must occur in one of them.  This is the
    Textor--Idelberger--Lischka characterization for simple MI graphs.
    """
    a = bits_to_matrix(bits, d).astype(bool)
    neigh = []
    for v in range(d):
        closed = {v} | set(np.flatnonzero(a[v]))
        ok = all(a[x, y] for x, y in combinations(sorted(closed), 2))
        if ok:
            neigh.append(frozenset(closed))
    simplices = []
    for s in neigh:
        if not any(s < t for t in neigh):
            if s not in simplices:
                simplices.append(s)
    for i, j in pairs(d):
        if a[i, j] and not any(i in s and j in s for s in simplices):
            return False
    return True


@dataclass
class CovFit:
    sigma: np.ndarray
    logdet: float
    loglik2: float
    bic: float
    iterations: int
    converged: bool
    jitter: float
    failed: bool = False


def sample_covariance(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] <= 1:
        raise ValueError("sample covariance requires a 2D data matrix with at least two rows")
    z = x - x.mean(axis=0, keepdims=True)
    s = z.T @ z / x.shape[0]
    if not np.all(np.isfinite(s)) or not np.allclose(s, s.T, atol=1e-12):
        raise ValueError("sample covariance is non-finite or nonsymmetric")
    return s


def _objective(sigma: np.ndarray, s: np.ndarray, n: int):
    try:
        chol = np.linalg.cholesky(sigma)
        logdet = 2.0 * float(np.log(np.diag(chol)).sum())
        inv = np.linalg.solve(sigma, np.eye(sigma.shape[0]))
    except np.linalg.LinAlgError:
        return math.inf, None, None
    trace = float(np.sum(s * inv.T))
    return float(logdet + trace), logdet, inv


def fit_covariance(bits: int, s: np.ndarray, n: int, *, max_iter=300,
                   tol=1e-9, start: np.ndarray | None = None) -> CovFit:
    """Reference constrained MLE via deterministic projected gradient.

    Only diagonal and UDG-edge covariance entries are free.  This is used for
    correctness tests and tiny exact enumeration; Rust is used for searches.
    """
    d = s.shape[0]
    mask = bits_to_matrix(bits, d).astype(bool)
    free = mask | np.eye(d, dtype=bool)
    sigma = np.array(start if start is not None else np.diag(np.diag(s)), dtype=float)
    sigma[~free] = 0.0
    jitter = 0.0
    for amount in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        trial = sigma.copy(); trial.flat[::d + 1] += amount
        try:
            np.linalg.cholesky(trial)
        except np.linalg.LinAlgError:
            continue
        sigma, jitter = trial, amount; break
    converged = False
    iters = 0
    for it in range(max_iter):
        value, logdet, inv = _objective(sigma, s, n)
        if not np.isfinite(value):
            return CovFit(sigma, math.nan, math.nan, math.nan, it, False, jitter, True)
        grad = inv - inv @ s @ inv
        grad[~free] = 0.0
        norm = float(np.max(np.abs(grad)))
        if norm < tol:
            converged, iters = True, it + 1
            break
        step = 1.0
        accepted = False
        while step > 1e-12:
            candidate = sigma - step * grad
            candidate[~free] = 0.0
            try:
                np.linalg.cholesky(candidate)
            except np.linalg.LinAlgError:
                step *= 0.5
                continue
            new_value, _, _ = _objective(candidate, s, n)
            if new_value <= value - 1e-4 * step * norm * norm:
                sigma, accepted = candidate, True
                break
            step *= 0.5
        iters = it + 1
        if not accepted:
            break
    value, logdet, inv = _objective(sigma, s, n)
    if logdet is None:
        return CovFit(sigma, math.nan, math.nan, math.nan, iters, False, jitter, True)
    loglik2 = -n * (d * math.log(2 * math.pi) + value)
    bic = loglik2 - (d + edge_count(bits)) * math.log(n)
    return CovFit(sigma, float(logdet), float(loglik2), float(bic), iters,
                  converged, jitter, False)


def unrestricted_fit(s: np.ndarray, n: int) -> CovFit:
    bits = (1 << (s.shape[0] * (s.shape[0] - 1) // 2)) - 1
    return fit_covariance(bits, s, n)


def exact_covbic(s: np.ndarray, n: int, *, valid_only: bool = False):
    """Enumerate all UDGs for tiny validation problems (d <= 6)."""
    d = s.shape[0]
    if d > 6:
        raise ValueError("exact enumeration is restricted to d <= 6")
    m = d * (d - 1) // 2
    best = None
    for bits in range(1 << m):
        if valid_only and not smig_valid(bits, d):
            continue
        fit = fit_covariance(bits, s, n, max_iter=800)
        if not fit.failed and (best is None or fit.bic > best[1].bic):
            best = (bits, fit)
    return best
