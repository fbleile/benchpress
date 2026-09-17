"""FLOP-style DAG search with lazy smooth NOTREKS evaluation.

The order representation keeps DAG validity exact.  Candidate moves are
screened with decomposable Gaussian-BIC deltas; only the best few moves are
refit and evaluated with the existing continuous NOTREKS kernel.  The soft
search may visit violating supports, while the returned graph is repaired and
checked with the canonical exact feasibility predicate.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import scipy.linalg as sla

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    fixed_order_bic_search,
    gaussian_bic,
)
from workflow.rules.structure_learning_algorithms.dagma.shared import (
    notreks_value_grad,
)
from workflow.rules.structure_learning_algorithms.flop_notreks.global_greedy import (
    GlobalGreedyConfig,
    fit_global_greedy_notreks,
)


@dataclass(frozen=True)
class SoftGreedyConfig:
    restarts: int = 8
    max_sweeps: int = 12
    lambda_bic: float = 2.0
    soft_notreks_weight: float = 100.0
    lazy_top_k: int = 12
    seed: int = 1729
    trek_function: str = "inv"
    inverse_epsilon: float = 1e-8


@dataclass
class SoftGreedyResult:
    adjacency: np.ndarray
    score: float
    bic: float
    notreks: float
    coefficients: np.ndarray
    runtime_seconds: float
    support_evaluations: int
    cache_hits: int
    accepted_moves: int
    restarts_completed: int
    distinct_supports: int
    raw_adjacency: np.ndarray
    raw_bic: float
    raw_notreks: float


class _LocalBIC:
    def __init__(self, X: np.ndarray, lambda_bic: float):
        self.X = X - X.mean(axis=0, keepdims=True)
        self.n = len(X)
        self.penalty = float(lambda_bic) * np.log(self.n)
        self.cache: dict[tuple[int, tuple[int, ...]], float] = {}

    def score(self, child: int, parents: Sequence[int]) -> float:
        key = (int(child), tuple(sorted(map(int, parents))))
        if key not in self.cache:
            p = np.asarray(key[1], dtype=int)
            if len(p):
                beta, *_ = sla.lstsq(self.X[:, p], self.X[:, child], lapack_driver="gelsy")
                residual = self.X[:, child] - self.X[:, p] @ beta
            else:
                residual = self.X[:, child]
            variance = max(float(residual @ residual) / self.n, 1e-8)
            self.cache[key] = self.n * np.log(variance) + len(p) * self.penalty
        return self.cache[key]

    def graph_score(self, graph: np.ndarray) -> float:
        return float(sum(self.score(j, np.flatnonzero(graph[:, j])) for j in range(graph.shape[0])))


def _project_to_order(graph: np.ndarray, order: Sequence[int]) -> np.ndarray:
    pos = np.empty(len(order), dtype=int)
    pos[np.asarray(order)] = np.arange(len(order))
    out = np.asarray(graph, dtype=np.uint8).copy()
    out[pos[:, None] >= pos[None, :]] = 0
    np.fill_diagonal(out, 0)
    return out


def _violation_count(graph: np.ndarray, pairs: Sequence[tuple[int, int]]) -> int:
    if not pairs:
        return 0
    reach = np.asarray(graph, dtype=bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(len(reach)):
        reach |= reach[:, [k]] & reach[[k], :]
    return sum(int(np.any(reach[:, i] & reach[:, j])) for i, j in pairs)


def _repair(graph: np.ndarray, coefficients: np.ndarray,
            pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Delete the weakest edge with the largest violation reduction."""
    out = np.asarray(graph, dtype=np.uint8).copy()
    while _violation_count(out, pairs):
        edges = list(zip(*np.nonzero(out)))
        if not edges:
            break
        before = _violation_count(out, pairs)
        ranked = []
        for u, v in edges:
            candidate = out.copy()
            candidate[u, v] = 0
            reduction = before - _violation_count(candidate, pairs)
            ranked.append((-reduction, abs(float(coefficients[u, v])), int(u), int(v)))
        _, _, u, v = min(ranked)
        out[u, v] = 0
    return out


class _SoftScore:
    def __init__(self, X, pairs, config):
        self.X = np.asarray(X, dtype=float)
        self.pairs = tuple(tuple(map(int, p)) for p in pairs)
        self.config = config
        self.cache = {}
        self.hits = 0

    def evaluate(self, graph):
        key = np.asarray(graph, dtype=np.uint8).tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        bic, coefficients = gaussian_bic(self.X, graph, lambda_bic=self.config.lambda_bic)
        try:
            nt, _ = notreks_value_grad(
                coefficients, self.pairs, self.config.trek_function,
                inverse_epsilon=self.config.inverse_epsilon)
            nt = float(nt)
            total = float(bic + self.config.soft_notreks_weight * nt)
            if not np.isfinite(total):
                raise ValueError("non-finite soft NOTREKS score")
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            nt, total = np.inf, np.inf
        value = (float(total), float(bic), float(nt), coefficients)
        self.cache[key] = value
        return value


def fit_soft_notreks(X: np.ndarray,
                     no_trek_pairs: Sequence[tuple[int, int]],
                     config: SoftGreedyConfig = SoftGreedyConfig()) -> SoftGreedyResult:
    """Run multi-restart lazy soft-NOTREKS FLOP search."""
    data = np.asarray(X, dtype=float)
    if data.ndim != 2 or config.restarts < 1 or config.lazy_top_k < 1:
        raise ValueError("invalid data or search configuration")
    d = data.shape[1]
    rng = np.random.default_rng(config.seed)
    local = _LocalBIC(data, config.lambda_bic)
    scorer = _SoftScore(data, no_trek_pairs, config)
    started = perf_counter()
    best = None
    raw_best = None
    completed = 0

    # Feasible incumbent safeguard: soft exploration must not lose the exact
    # hard-feasible FLOP solution when a violating raw support is expensive to
    # repair.  Subsequent lazy moves can still improve on this incumbent.
    hard = fit_global_greedy_notreks(
        data, no_trek_pairs,
        GlobalGreedyConfig(restarts=config.restarts, max_sweeps=4,
                           lambda_bic=config.lambda_bic, seed=config.seed,
                           inner_backend="python"))
    hard_value = scorer.evaluate(hard.adjacency)
    best = (hard_value, hard.adjacency.copy())

    for _ in range(config.restarts):
        order = list(map(int, rng.permutation(d)))
        # FLOP's cheap decomposable inner search supplies a strong support
        # proposal.  The soft NOTREKS term then decides whether to retain,
        # delete, or locally replace edges around that proposal.
        graph = fixed_order_bic_search(
            data, order, lambda_bic=config.lambda_bic).astype(np.uint8)
        current = scorer.evaluate(graph)
        for _sweep in range(config.max_sweeps):
            moves = []
            pos = np.empty(d, dtype=int)
            pos[np.asarray(order)] = np.arange(d)
            for u in range(d):
                for v in range(d):
                    if u == v or pos[u] >= pos[v]:
                        continue
                    proposal = graph.copy()
                    proposal[u, v] ^= 1
                    old = local.score(v, np.flatnonzero(graph[:, v]))
                    new = local.score(v, np.flatnonzero(proposal[:, v]))
                    moves.append((new - old, u, v, proposal))
            moves.sort(key=lambda x: (x[0], x[1], x[2]))
            # Keep deletion moves in the lazy shortlist even when many
            # additions have attractive BIC deltas: NOTREKS improvement can
            # require accepting a small local BIC cost first.
            additions = [m for m in moves if not graph[m[1], m[2]]]
            deletions = [m for m in moves if graph[m[1], m[2]]]
            shortlist = additions[:config.lazy_top_k] + deletions[:config.lazy_top_k]
            candidate_values = []
            for _, _, _, proposal in shortlist:
                candidate_value = scorer.evaluate(proposal)
                candidate_values.append((candidate_value, proposal))
                if _violation_count(proposal, no_trek_pairs) == 0 and (
                        candidate_value[1], proposal.tobytes()) < (
                            best[0][1], best[1].tobytes()):
                    best = (candidate_value, proposal.copy())
            if not candidate_values:
                break
            candidate, proposal = min(candidate_values, key=lambda x: (x[0][0], x[1].tobytes()))
            if candidate[0] >= current[0] - 1e-10:
                break
            graph, current = proposal, candidate
        completed += 1
        if raw_best is None or (current[0], graph.tobytes()) < (raw_best[0][0], raw_best[1].tobytes()):
            raw_best = (current, graph.copy())
        repaired = _repair(graph, current[3], no_trek_pairs)
        feasible = scorer.evaluate(repaired)
        if (feasible[1], repaired.tobytes()) < (best[0][1], best[1].tobytes()):
            best = (feasible, repaired)

    assert best is not None and raw_best is not None
    feasible, adjacency = best
    raw, raw_graph = raw_best
    if common_ancestor_violations(adjacency, no_trek_pairs):
        raise RuntimeError("soft NOTREKS repair returned an infeasible graph")
    return SoftGreedyResult(
        adjacency=adjacency,
        score=feasible[0], bic=feasible[1], notreks=feasible[2], coefficients=feasible[3],
        runtime_seconds=perf_counter() - started,
        support_evaluations=len(scorer.cache), cache_hits=scorer.hits,
        accepted_moves=0, restarts_completed=completed,
        distinct_supports=len(scorer.cache), raw_adjacency=raw_graph,
        raw_bic=raw[1], raw_notreks=raw[2])
