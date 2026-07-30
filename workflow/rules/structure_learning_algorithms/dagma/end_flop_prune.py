"""Deletion-only node-wise FLOP BIC pruning for a candidate DAG."""

from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np

from .gaussian_bic import is_dag


def local_gaussian_bic(X, child, parents, *, lambda_bic=2.0,
                       variance_epsilon=1e-12):
    """FLOP-compatible local BIC, omitting graph-independent constants."""
    data = np.asarray(X, dtype=np.float64)
    centered = data - data.mean(axis=0, keepdims=True)
    selected = tuple(sorted(int(parent) for parent in parents))
    response = centered[:, child]
    if selected:
        coefficients, *_ = np.linalg.lstsq(
            centered[:, selected], response, rcond=None)
        residual = response - centered[:, selected] @ coefficients
    else:
        residual = response
    variance = max(float(residual @ residual) / len(data), variance_epsilon)
    return (len(data) * np.log(variance)
            + lambda_bic * len(selected) * np.log(len(data)))


def exact_subset_parents(X, child, candidates, *, lambda_bic=2.0,
                         max_candidates=18):
    """Return the exact locally BIC-optimal subset with deterministic ties."""
    data = np.asarray(X, dtype=np.float64)
    centered = data - data.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / len(data)
    return exact_subset_parents_from_covariance(
        covariance, len(data), child, candidates, lambda_bic=lambda_bic,
        max_candidates=max_candidates)


def exact_subset_parents_from_covariance(
        covariance, n, child, candidates, *, lambda_bic=2.0,
        max_candidates=18):
    """Exact subset search using only cached d-by-d covariance operations."""
    candidates = tuple(sorted(set(int(parent) for parent in candidates)))
    if len(candidates) > max_candidates:
        raise ValueError(
            f"node {child} has {len(candidates)} candidates; exact limit is "
            f"{max_candidates}")
    best = None
    for size in range(len(candidates) + 1):
        for parents in combinations(candidates, size):
            if parents:
                pp = covariance[np.ix_(parents, parents)]
                py = covariance[list(parents), child]
                try:
                    beta = np.linalg.solve(pp, py)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(pp, py, rcond=None)[0]
                variance = covariance[child, child] - py @ beta
            else:
                variance = covariance[child, child]
            score = (n * np.log(max(float(variance), 1e-12))
                     + lambda_bic * len(parents) * np.log(n))
            key = (score, len(parents), parents)
            if best is None or key < best[0]:
                best = (key, parents)
    return best[1], best[0][0]


def exact_subset_prune(X, candidate_dag, *, lambda_bic=2.0,
                       max_candidates=18):
    candidate = np.asarray(candidate_dag)
    if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ValueError("candidate_dag must be square")
    if not is_dag(candidate):
        raise ValueError("candidate_dag must be a DAG")
    result = np.zeros(candidate.shape, dtype=int)
    local_scores = []
    for child in range(len(candidate)):
        parents, score = exact_subset_parents(
            X, child, np.flatnonzero(candidate[:, child]),
            lambda_bic=lambda_bic, max_candidates=max_candidates)
        result[list(parents), child] = 1
        local_scores.append(score)
    return result, np.asarray(local_scores)


def end_flop_prune(X, candidate_dag, *, lambda_bic=2.0):
    """Run genuine FLOP grow-shrink restricted to existing candidate edges."""
    import flopsearch

    candidate = (np.asarray(candidate_dag) != 0).astype(float)
    if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ValueError("candidate_dag must be square")
    if not is_dag(candidate):
        raise ValueError("candidate_dag must be a DAG")
    started = perf_counter()
    graph, coefficients, diagnostics = flopsearch.prune_parents_bic(
        np.asarray(X, dtype=np.float64), candidate, lambda_bic,
        return_coefficients=True, return_diagnostics=True)
    graph = np.asarray(graph, dtype=int)
    coefficients = np.asarray(coefficients, dtype=float)
    if np.any((graph != 0) & (candidate == 0)):
        raise RuntimeError("FLOP pruning introduced a non-candidate edge")
    if not is_dag(graph):
        raise RuntimeError("FLOP pruning returned a cyclic graph")
    diagnostics = dict(diagnostics)
    diagnostics["runtime"] = perf_counter() - started
    return graph, coefficients, diagnostics
