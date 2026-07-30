"""Exact graph-level Gaussian BIC and DAGMA candidate archive utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.linalg as sla


def is_dag(adjacency: np.ndarray) -> bool:
    A = np.asarray(adjacency, dtype=bool)
    indegree = A.sum(axis=0).astype(int)
    stack = list(np.flatnonzero(indegree == 0))
    seen = 0
    while stack:
        node = stack.pop()
        seen += 1
        for child in np.flatnonzero(A[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                stack.append(int(child))
    return seen == len(A)


def gaussian_bic(X: np.ndarray, adjacency: np.ndarray,
                 variance_epsilon: float = 1e-8,
                 lambda_bic: float = 1.0) -> tuple[float, np.ndarray]:
    """Return n*sum(log(RSS/n)) + lambda*|E|*log(n)."""
    X = np.asarray(X, dtype=float)
    A = np.asarray(adjacency, dtype=bool)
    if A.shape != (X.shape[1], X.shape[1]) or np.any(np.diag(A)):
        raise ValueError("adjacency shape/diagonal is invalid")
    if not is_dag(A):
        raise ValueError("exact Gaussian DAG BIC is undefined for cyclic support")
    centered = X - X.mean(axis=0, keepdims=True)
    n, d = centered.shape
    coefficients = np.zeros((d, d), dtype=float)
    log_variances = 0.0
    for child in range(d):
        parents = np.flatnonzero(A[:, child])
        if len(parents):
            beta, *_ = sla.lstsq(centered[:, parents], centered[:, child],
                                  lapack_driver="gelsy")
            residual = centered[:, child] - centered[:, parents] @ beta
            coefficients[parents, child] = beta
        else:
            residual = centered[:, child]
        variance = max(float(residual @ residual) / n, variance_epsilon)
        log_variances += np.log(variance)
    return float(
        n * log_variances + lambda_bic * A.sum() * np.log(n)
    ), coefficients


def topological_order(adjacency: np.ndarray) -> list[int]:
    A = np.asarray(adjacency, dtype=bool)
    indegree = A.sum(axis=0).astype(int)
    available = list(np.flatnonzero(indegree == 0))
    order = []
    while available:
        node = min(available)
        available.remove(node)
        order.append(node)
        for child in np.flatnonzero(A[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(int(child))
    if len(order) != len(A):
        raise ValueError("topological order requires a DAG")
    return order


def canonical_signature_masks(adjacency: np.ndarray,
                              pairs: Sequence[tuple[int, int]]) -> list[int]:
    A = np.asarray(adjacency, dtype=bool)
    order = topological_order(A)
    targets = sorted({node for pair in pairs for node in pair})
    bit = {node: i for i, node in enumerate(targets)}
    signatures = [0] * len(A)
    for node in reversed(order):
        signature = (1 << bit[node]) if node in bit else 0
        for child in np.flatnonzero(A[node]):
            signature |= signatures[int(child)]
        signatures[node] = signature
    return signatures


def fixed_order_bic_search(
    X: np.ndarray,
    order: Sequence[int],
    *,
    signatures: Sequence[int] | None = None,
    variance_epsilon: float = 1e-8,
    lambda_bic: float = 1.0,
) -> np.ndarray:
    """Greedy grow-shrink Gaussian-BIC parent fitting under a fixed order."""
    X = np.asarray(X, dtype=float)
    centered = X - X.mean(axis=0, keepdims=True)
    n, d = centered.shape
    position = {node: i for i, node in enumerate(order)}

    def local(child, parents):
        if parents:
            beta, *_ = sla.lstsq(centered[:, parents], centered[:, child],
                                  lapack_driver="gelsy")
            residual = centered[:, child] - centered[:, parents] @ beta
        else:
            residual = centered[:, child]
        variance = max(float(residual @ residual) / n, variance_epsilon)
        return (n * np.log(variance)
                + lambda_bic * len(parents) * np.log(n))

    result = np.zeros((d, d), dtype=int)
    for child in order:
        candidates = [
            parent for parent in order[:position[child]]
            if signatures is None
            or signatures[child] & ~signatures[parent] == 0
        ]
        parents: list[int] = []
        score = local(child, parents)
        while True:
            moves = [(local(child, parents + [candidate]), candidate)
                     for candidate in candidates if candidate not in parents]
            if not moves:
                break
            candidate_score, candidate = min(moves)
            if candidate_score >= score - 1e-10:
                break
            parents.append(candidate)
            score = candidate_score
        while parents:
            moves = [(local(child, [p for p in parents if p != candidate]), candidate)
                     for candidate in parents]
            candidate_score, candidate = min(moves)
            if candidate_score >= score - 1e-10:
                break
            parents.remove(candidate)
            score = candidate_score
        result[parents, child] = 1
    return result


def common_ancestor_violations(adjacency: np.ndarray,
                               pairs: Sequence[tuple[int, int]]) -> int:
    reach = np.asarray(adjacency, dtype=bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(len(reach)):
        reach |= reach[:, [k]] & reach[[k], :]
    return sum(bool(np.any(reach[:, i] & reach[:, j])) for i, j in pairs)


@dataclass(frozen=True)
class Candidate:
    restart: int
    stage: int
    threshold: float
    adjacency: np.ndarray
    exact_bic: float
    violations: int
    edge_count: int


def select_exact_bic(candidates: Iterable[Candidate],
                     require_zero_violations: bool = False) -> Candidate:
    eligible = [candidate for candidate in candidates
                if is_dag(candidate.adjacency)
                and (not require_zero_violations or candidate.violations == 0)]
    if not eligible:
        raise ValueError("no eligible acyclic archived candidate")
    return min(eligible, key=lambda c: (
        c.exact_bic, c.violations, c.edge_count, c.restart, c.stage, c.threshold))
