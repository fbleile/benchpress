"""Direct order/source-clique search over UDGs.

The state contains only an order and a cover by source/simplex cliques.  No
orientation, parent set, DAG score, or FLOP output is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from .covariance import edge_count, matrix_to_bits, smig_valid


def canonical_order(order):
    order = tuple(int(x) for x in order)
    if sorted(order) != list(range(len(order))):
        raise ValueError("order must be a permutation of 0..d-1")
    return order


def canonical_cliques(cliques, d):
    result = []
    for clique in cliques:
        c = frozenset(int(x) for x in clique)
        if not c or any(x < 0 or x >= d for x in c):
            raise ValueError("clique contains an invalid node")
        result.append(c)
    return tuple(sorted(set(result), key=lambda c: (min(c), len(c), tuple(sorted(c)))))


def udg_from_cliques(cliques, d):
    udg = np.zeros((d, d), dtype=np.uint8)
    for clique in cliques:
        for i, j in combinations(sorted(clique), 2):
            udg[i, j] = udg[j, i] = 1
    return udg


def clique_cover_valid(cliques, d):
    """Validate a cover and the induced simple-MI graph."""
    try:
        cliques = canonical_cliques(cliques, d)
    except ValueError:
        return False
    udg = udg_from_cliques(cliques, d)
    clique_edges = all(udg[i, j] == 1 for c in cliques for i, j in combinations(sorted(c), 2))
    return clique_edges and smig_valid(matrix_to_bits(udg), d)


@dataclass(frozen=True)
class OrderState:
    order: tuple[int, ...]
    cliques: tuple[frozenset[int], ...]

    def canonical(self):
        return OrderState(canonical_order(self.order), canonical_cliques(self.cliques, len(self.order)))

    def udg(self):
        return udg_from_cliques(self.cliques, len(self.order))

    def key(self):
        return self.canonical()


def _states_from_cliques(order, cliques):
    return OrderState(tuple(order), canonical_cliques((c for c in cliques if c), len(order)))


def initial_states(d, correlation=None, random_restarts=2, seed=0):
    """Deterministic non-oracle initial states."""
    states = [
        _states_from_cliques(range(d), [{i} for i in range(d)]),
        _states_from_cliques(range(d), [set(range(d))]),
    ]
    if correlation is not None:
        strength = np.sum(np.abs(correlation), axis=1)
        order = tuple(np.argsort(-strength, kind="mergesort"))
        # Pair cliques are a conservative proposal-only cover.  The resulting
        # UDG is still checked as a valid final state by the search.
        pairs = sorted(((abs(float(correlation[i, j])), i, j)
                        for i, j in combinations(range(d), 2)), reverse=True)
        states.append(_states_from_cliques(order, [{i, j} for _, i, j in pairs[:max(1, d)]]))
    rng = np.random.default_rng(seed)
    for _ in range(random_restarts):
        states.append(_states_from_cliques(rng.permutation(d), [{i} for i in range(d)]))
    return states


def _order_moves(state):
    order = list(state.order); d = len(order)
    for i in range(d - 1):
        x = order.copy(); x[i], x[i + 1] = x[i + 1], x[i]
        yield _states_from_cliques(x, state.cliques), "adjacent_swap"
    for i in range(d):
        for j in range(d):
            if i == j: continue
            x = order.copy(); node = x.pop(i); x.insert(j, node)
            yield _states_from_cliques(x, state.cliques), "tuck"
    for length in (2, 3):
        for i in range(d - length + 1):
            x = order.copy(); x[i:i + length] = reversed(x[i:i + length])
            yield _states_from_cliques(x, state.cliques), "reverse_segment"


def _clique_moves(state):
    d = len(state.order); cliques = list(state.cliques)
    # Add/remove a node, create/delete cliques, split/merge, and reassign.
    for k, clique in enumerate(cliques):
        for v in range(d):
            if v not in clique:
                x = cliques.copy(); x[k] = frozenset(set(clique) | {v})
                yield _states_from_cliques(state.order, x), "clique_add"
            elif len(clique) > 1:
                x = cliques.copy(); x[k] = frozenset(set(clique) - {v})
                yield _states_from_cliques(state.order, x), "clique_remove"
        if len(clique) > 2:
            a = sorted(clique); mid = len(a) // 2
            x = cliques[:k] + cliques[k + 1:] + [frozenset(a[:mid]), frozenset(a[mid:])]
            yield _states_from_cliques(state.order, x), "clique_split"
        if len(clique) == 1:
            x = cliques[:k] + cliques[k + 1:]
            yield _states_from_cliques(state.order, x), "clique_delete"
    for v in range(d):
        yield _states_from_cliques(state.order, cliques + [frozenset({v})]), "clique_create"
    for a, b in combinations(range(len(cliques)), 2):
        x = [c for k, c in enumerate(cliques) if k not in (a, b)] + [cliques[a] | cliques[b]]
        yield _states_from_cliques(state.order, x), "clique_merge"
    for a, clique in enumerate(cliques):
        for v in sorted(clique):
            for b in range(len(cliques)):
                if a == b: continue
                x = cliques.copy(); x[a] = frozenset(set(x[a]) - {v}); x[b] = frozenset(set(x[b]) | {v})
                yield _states_from_cliques(state.order, x), "clique_reassign"


def search_order_uec(x, *, edge_penalty=None, restarts=16, beam_width=4,
                     max_evaluations=300, random_seed=0, correlation=None):
    """Deterministic beam/greedy search whose objective is direct covBIC."""
    from .covariance import fit_covariance, sample_covariance
    n, d = x.shape; s = sample_covariance(x)
    penalty = math.log(n) if edge_penalty is None else float(edge_penalty)
    cache = {}; diagnostics = {"order_moves": 0, "clique_moves": 0,
                               "candidates": 0, "valid_candidates": 0,
                               "invalid_candidates": 0, "score_evaluations": 0,
                               "cache_hits": 0, "icf_iterations": 0,
                               "icf_failures": 0}

    def score(state):
        state = state.canonical(); udg = state.udg(); bits = matrix_to_bits(udg)
        key = bits
        diagnostics["candidates"] += 1
        if key in cache:
            diagnostics["cache_hits"] += 1
            return cache[key], state
        diagnostics["score_evaluations"] += 1
        fit = fit_covariance(bits, s, n, max_iter=1000)
        diagnostics["icf_iterations"] += fit.iterations
        diagnostics["icf_failures"] += int(fit.failed or not fit.converged)
        cache[key] = fit
        return fit, state

    best = None; best_score = float("-inf")
    starts = initial_states(d, correlation, max(0, restarts - 3), random_seed)
    for start in starts[:restarts]:
        frontier = [start]; local_best = None; local_value = float("-inf")
        while frontier and diagnostics["score_evaluations"] < max_evaluations:
            scored = []
            for state in frontier:
                if not clique_cover_valid(state.cliques, d):
                    diagnostics["invalid_candidates"] += 1; continue
                diagnostics["valid_candidates"] += 1
                fit, state = score(state)
                if fit.failed or not fit.converged: continue
                scored.append((fit.bic if edge_penalty is None else fit.loglik2 - edge_penalty * edge_count(matrix_to_bits(state.udg())), state))
            if not scored: break
            scored.sort(key=lambda z: (-z[0], z[1].order, z[1].cliques))
            value, state = scored[0]
            if value > local_value + 1e-8:
                local_value, local_best = value, state
            else:
                break
            moves = list(_order_moves(state)) + list(_clique_moves(state))
            diagnostics["order_moves"] += sum(kind in {"adjacent_swap", "tuck", "reverse_segment"} for _, kind in moves)
            diagnostics["clique_moves"] += sum(kind not in {"adjacent_swap", "tuck", "reverse_segment"} for _, kind in moves)
            frontier = []
            seen = set()
            for candidate, _ in moves:
                candidate = candidate.canonical()
                if candidate.key() in seen: continue
                seen.add(candidate.key())
                if clique_cover_valid(candidate.cliques, d): frontier.append(candidate)
            # Beam width one is the requested genuine greedy mode.
            ranked = []
            for candidate in frontier:
                if diagnostics["score_evaluations"] >= max_evaluations: break
                fit, candidate = score(candidate)
                if fit.failed or not fit.converged: continue
                val = fit.bic if edge_penalty is None else fit.loglik2 - penalty * edge_count(matrix_to_bits(candidate.udg()))
                ranked.append((val, candidate))
            ranked.sort(key=lambda z: (-z[0], z[1].order, z[1].cliques))
            frontier = [candidate for _, candidate in ranked[:max(1, beam_width)] if ranked]
        if local_best is not None and local_value > best_score:
            best_score, best = local_value, local_best
    if best is None:
        raise RuntimeError("order UEC search found no converged valid candidate")
    fit = cache[matrix_to_bits(best.udg())]
    diagnostics.update({"restarts": len(starts[:restarts]), "beam_width": beam_width,
                        "edge_penalty": penalty, "final_edge_count": int(best.udg().sum() // 2),
                        "cache_size": len(cache), "score_backend": "python_covariance_reference"})
    return best.udg(), diagnostics
