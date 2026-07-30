from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphPostprocessResult:
    weighted_raw: np.ndarray
    thresholded_raw: np.ndarray
    projected_dag: np.ndarray
    raw_is_dag: bool
    removed_edges: list[tuple[int, int, float]]


def zero_diagonal(A: np.ndarray) -> np.ndarray:
    out = np.array(A, copy=True)
    np.fill_diagonal(out, 0)
    return out


def threshold_adjacency(W: np.ndarray, threshold: float) -> np.ndarray:
    A = (np.abs(W) >= float(threshold)).astype(int)
    np.fill_diagonal(A, 0)
    return A


def is_dag(A: np.ndarray) -> bool:
    A = np.asarray(A).astype(bool)
    d = A.shape[0]
    indegree = A.sum(axis=0).astype(int)
    queue = [i for i in range(d) if indegree[i] == 0]
    seen = 0
    while queue:
        node = queue.pop(0)
        seen += 1
        for child in np.flatnonzero(A[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    return seen == d


def _tarjan_scc(A: np.ndarray) -> list[list[int]]:
    A = np.asarray(A).astype(bool)
    n = A.shape[0]
    index = 0
    stack: list[int] = []
    on_stack = np.zeros(n, dtype=bool)
    indices = -np.ones(n, dtype=int)
    lowlink = np.zeros(n, dtype=int)
    components: list[list[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True
        for w in np.flatnonzero(A[v]):
            w = int(w)
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            components.append(sorted(comp))

    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)
    return components


def cyclic_scc_sizes(A: np.ndarray) -> list[int]:
    return [len(c) for c in _tarjan_scc(A) if len(c) > 1]


def project_to_dag(A: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Remove smallest-magnitude cyclic-SCC edges with deterministic tie breaks."""
    projected = np.array(A, dtype=int, copy=True)
    np.fill_diagonal(projected, 0)
    weights = np.abs(np.asarray(W))
    removed: list[tuple[int, int, float]] = []
    while not is_dag(projected):
        cyclic = [c for c in _tarjan_scc(projected) if len(c) > 1]
        if not cyclic:
            break
        comp = sorted(cyclic, key=lambda c: (-len(c), c))[0]
        candidates = []
        for i in comp:
            for j in comp:
                if projected[i, j]:
                    candidates.append((float(weights[i, j]), int(i), int(j)))
        if not candidates:
            break
        weight, i, j = min(candidates)
        projected[i, j] = 0
        removed.append((i, j, weight))
    return projected, removed


def postprocess_graph(W: np.ndarray, threshold: float) -> GraphPostprocessResult:
    W0 = zero_diagonal(np.asarray(W))
    A = threshold_adjacency(W0, threshold)
    raw_ok = is_dag(A)
    projected, removed = (A.copy(), []) if raw_ok else project_to_dag(A, W0)
    return GraphPostprocessResult(W0, A, projected, raw_ok, removed)
