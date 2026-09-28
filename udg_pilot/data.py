"""Self-contained copy of the repository's standardized linear-Gaussian ER machinery.

The graph convention matches ``local_d20_benchmark.generate``: an ER-k DAG is
sampled by first drawing a random topological order, then each forward pair
with probability k/(d-1). Edge weights are iid signed Uniform[0.5, 1.0].
Innovation log variances are iid Uniform[-log(2), log(2)], and the returned
observations are column-standardized with ddof=0.
"""
from __future__ import annotations

import numpy as np


def graph(d: int, expected_degree: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    dag = np.zeros((d, d), dtype=np.uint8)
    p = float(expected_degree) / max(1, d - 1)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p:
                dag[order[i], order[j]] = 1
    return dag


def _topological_order(dag: np.ndarray) -> list[int]:
    indegree = dag.sum(0).astype(int).tolist()
    remaining = set(range(dag.shape[0]))
    result: list[int] = []
    while remaining:
        ready = sorted(v for v in remaining if indegree[v] == 0)
        if not ready:
            raise ValueError("generated graph is not acyclic")
        for v in ready:
            remaining.remove(v)
            result.append(v)
            for child in np.flatnonzero(dag[v]):
                indegree[int(child)] -= 1
    return result


def data(dag: np.ndarray, n: int, data_seed: int) -> np.ndarray:
    d = dag.shape[0]
    rng = np.random.default_rng(data_seed)
    weights = np.zeros((d, d), dtype=float)
    edges = dag.astype(bool)
    weights[edges] = rng.uniform(0.5, 1.0, size=int(edges.sum()))
    weights[edges] *= rng.choice([-1.0, 1.0], size=int(edges.sum()))
    scales = np.exp(0.5 * rng.uniform(-np.log(2.0), np.log(2.0), size=d))
    x = rng.normal(size=(n, d)) * scales
    for node in _topological_order(dag):
        parents = np.flatnonzero(dag[:, node])
        if len(parents):
            x[:, node] += x[:, parents] @ weights[parents, node]
    x -= x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    return x / np.where(std > 1e-12, std, 1.0)

