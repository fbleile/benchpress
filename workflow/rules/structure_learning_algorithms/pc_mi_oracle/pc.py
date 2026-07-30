"""Stable-PC with an oracle dispatcher only for empty conditioning sets."""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import norm


@dataclass
class QueryCounts:
    oracle_marginal_queries: int = 0
    oracle_independent_answers: int = 0
    oracle_dependent_answers: int = 0
    statistical_conditional_tests: int = 0


class MIOracleFisherZ:
    def __init__(self, X, no_trek_pairs, alpha=.05):
        self.X = np.asarray(X, dtype=float)
        self.pairs = {tuple(sorted(p)) for p in no_trek_pairs}
        self.alpha = float(alpha)
        self.counts = QueryCounts()

    def independent(self, i, j, conditioning=()):
        conditioning = tuple(conditioning)
        if not conditioning:
            self.counts.oracle_marginal_queries += 1
            answer = tuple(sorted((i, j))) in self.pairs
            if answer:
                self.counts.oracle_independent_answers += 1
            else:
                self.counts.oracle_dependent_answers += 1
            return answer
        self.counts.statistical_conditional_tests += 1
        cols = [i, j, *conditioning]
        corr = np.corrcoef(self.X[:, cols], rowvar=False)
        precision = np.linalg.pinv(corr)
        r = -precision[0, 1] / np.sqrt(max(precision[0, 0] * precision[1, 1], 1e-30))
        r = float(np.clip(r, -.999999, .999999))
        z = np.sqrt(max(len(self.X) - len(conditioning) - 3, 0)) * abs(np.arctanh(r))
        return bool(2 * norm.sf(z) > self.alpha)


def stable_pc(X, test: MIOracleFisherZ, max_cond_set=None):
    """Return a CPDAG matrix (1 directed; symmetric 1 undirected)."""
    d = np.asarray(X).shape[1]
    max_cond_set = d - 2 if max_cond_set is None else int(max_cond_set)
    adj = np.ones((d, d), dtype=bool)
    np.fill_diagonal(adj, False)
    separating = {}
    for level in range(max_cond_set + 1):
        snapshot = adj.copy()
        candidates = [(i, j) for i in range(d) for j in range(i + 1, d) if snapshot[i, j]]
        any_testable = False
        removals = []
        for i, j in candidates:
            neighbors = [k for k in range(d) if snapshot[i, k] and k != j]
            if len(neighbors) < level:
                continue
            any_testable = True
            for S in combinations(neighbors, level):
                if test.independent(i, j, S):
                    removals.append((i, j, frozenset(S)))
                    break
        for i, j, S in removals:
            adj[i, j] = adj[j, i] = False
            separating[i, j] = separating[j, i] = S
        if not any_testable:
            break
    # endpoint matrix: 1 means tail at row -> column; symmetric means undirected.
    cpdag = adj.astype(int)
    for k in range(d):
        neighbors = np.flatnonzero(adj[:, k])
        for i, j in combinations(neighbors, 2):
            if not adj[i, j] and k not in separating.get((i, j), ()):
                cpdag[k, i] = 0  # i -> k
                cpdag[k, j] = 0  # j -> k
    def directed(a, b):
        return cpdag[a, b] == 1 and cpdag[b, a] == 0

    def undirected(a, b):
        return cpdag[a, b] == cpdag[b, a] == 1

    def nonadjacent(a, b):
        return cpdag[a, b] == cpdag[b, a] == 0

    def orient(a, b):
        nonlocal changed
        if undirected(a, b):
            cpdag[b, a] = 0
            changed = True

    # Complete the CPDAG with the standard Meek closure (R1--R3).
    changed = True
    while changed:
        changed = False
        # R1: a -> b - c and a,c nonadjacent implies b -> c.
        for a in range(d):
            for b in range(d):
                for c in range(d):
                    if len({a, b, c}) == 3 and directed(a, b) and undirected(b, c) and nonadjacent(a, c):
                        orient(b, c)
        # R2: a - b and a -> c -> b implies a -> b.
        for a in range(d):
            for b in range(d):
                if undirected(a, b) and any(directed(a, c) and directed(c, b) for c in range(d)):
                    orient(a, b)
        # R3: a-b, a-c, a-d, c->b, d->b, and c,d nonadjacent imply a->b.
        for a in range(d):
            for b in range(d):
                if not undirected(a, b):
                    continue
                candidates = [c for c in range(d) if c not in {a, b}
                              and undirected(a, c) and directed(c, b)]
                if any(nonadjacent(c, e) for c, e in combinations(candidates, 2)):
                    orient(a, b)
    return cpdag, test.counts
