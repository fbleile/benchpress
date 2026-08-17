"""Chromatic source-prefix FLOP.

The no-trek graph H has one vertex per observed variable and one undirected
edge per supplied no-trek relation.  We compute chi(H) exactly with a bounded
DSATUR search.  If the bound is exhausted, a deterministically constructed
clique is returned instead; its size is a certified lower bound on chi(H).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


Pair = tuple[int, int]


@dataclass(frozen=True)
class ChromaticResult:
    value: int
    exact: bool
    clique_lower_bound: int
    coloring_upper_bound: int
    search_nodes: int
    method: str


def _canonical_adjacency(d: int, pairs: Sequence[Pair]) -> list[set[int]]:
    if d < 1:
        raise ValueError("d must be positive")
    adjacency = [set() for _ in range(d)]
    for raw_left, raw_right in pairs:
        left, right = int(raw_left), int(raw_right)
        if left == right:
            raise ValueError("self-pairs are invalid in the no-trek graph")
        if min(left, right) < 0 or max(left, right) >= d:
            raise ValueError(f"no-trek pair {(left, right)} is outside 0..{d}")
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _greedy_clique(adjacency: list[set[int]]) -> tuple[int, ...]:
    """Return a real clique, hence a certificate for a chromatic lower bound."""
    best: tuple[int, ...] = ()
    # Multiple deterministic starts are cheap and materially tighten the
    # fallback without pretending to solve maximum clique.
    starts = sorted(range(len(adjacency)), key=lambda v: (-len(adjacency[v]), v))
    for start in starts:
        clique = [start]
        candidates = set(adjacency[start])
        while candidates:
            vertex = max(candidates, key=lambda v: (len(candidates & adjacency[v]), len(adjacency[v]), -v))
            clique.append(vertex)
            candidates &= adjacency[vertex]
        candidate = tuple(sorted(clique))
        if len(candidate) > len(best) or (len(candidate) == len(best) and candidate < best):
            best = candidate
    return best or (0,)


def _greedy_coloring(adjacency: list[set[int]]) -> list[int]:
    colors = [-1] * len(adjacency)
    uncolored = set(range(len(adjacency)))
    while uncolored:
        vertex = max(
            uncolored,
            key=lambda v: (len({colors[n] for n in adjacency[v] if colors[n] >= 0}), len(adjacency[v]), -v),
        )
        forbidden = {colors[n] for n in adjacency[vertex] if colors[n] >= 0}
        color = 0
        while color in forbidden:
            color += 1
        colors[vertex] = color
        uncolored.remove(vertex)
    return colors


def chromatic_number_or_lower_bound(
    d: int,
    pairs: Sequence[Pair],
    *,
    max_search_nodes: int = 2_000_000,
) -> ChromaticResult:
    """Return exact chi(H), or a certified clique lower bound on interruption."""
    if max_search_nodes < 1:
        raise ValueError("max_search_nodes must be positive")
    adjacency = _canonical_adjacency(d, pairs)
    clique = _greedy_clique(adjacency)
    lower = len(clique)
    greedy = _greedy_coloring(adjacency)
    best = max(greedy) + 1
    if lower == best:
        return ChromaticResult(best, True, lower, best, 0, "clique_equals_greedy")

    colors = [-1] * d
    search_nodes = 0
    exhausted = False

    def visit(colored: int, used: int) -> None:
        nonlocal best, search_nodes, exhausted
        if exhausted or used >= best:
            return
        search_nodes += 1
        if search_nodes > max_search_nodes:
            exhausted = True
            return
        if colored == d:
            best = used
            return
        uncolored = [v for v in range(d) if colors[v] < 0]
        vertex = max(
            uncolored,
            key=lambda v: (len({colors[n] for n in adjacency[v] if colors[n] >= 0}), len(adjacency[v]), -v),
        )
        forbidden = {colors[n] for n in adjacency[vertex] if colors[n] >= 0}
        # Existing colors first, then at most one new color; color labels are
        # symmetric, so considering further new labels is redundant.
        for color in range(min(used + 1, best)):
            if color in forbidden:
                continue
            new_used = max(used, color + 1)
            if new_used >= best:
                continue
            colors[vertex] = color
            visit(colored + 1, new_used)
            colors[vertex] = -1
            if exhausted:
                return

    visit(0, 0)
    if exhausted:
        return ChromaticResult(
            lower, False, lower, best, search_nodes, "greedy_clique_lower_bound")
    return ChromaticResult(best, True, lower, best, search_nodes, "exact_dsatur")


def fit_chromatic_source_flop(
    X: np.ndarray,
    no_trek_pairs: Sequence[Pair],
    *,
    lambda_bic: float = 2.0,
    restarts: int = 1,
    seed: int = 1729,
    max_chromatic_search_nodes: int = 2_000_000,
):
    """Fit source-prefix FLOP and return ``(CPDAG, diagnostics)``."""
    import flopsearch

    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array")
    chromatic = chromatic_number_or_lower_bound(
        data.shape[1], no_trek_pairs, max_search_nodes=max_chromatic_search_nodes)
    cpdag, diagnostics = flopsearch.flop_source_prefix(
        data,
        float(lambda_bic),
        chromatic.value,
        restarts=int(restarts),
        seed=int(seed),
        return_diagnostics=True,
    )
    diagnostics = dict(diagnostics)
    diagnostics.update({f"chromatic_{key}": value for key, value in asdict(chromatic).items()})
    diagnostics["number_of_supplied_constraints"] = len({tuple(sorted(map(int, pair))) for pair in no_trek_pairs})
    diagnostics["search_version"] = "chromatic_source_prefix_flop"
    return np.asarray(cpdag), diagnostics


def fit_chromatic_source_notreks(
    X: np.ndarray,
    no_trek_pairs: Sequence[Pair],
    *,
    lambda_bic: float = 2.0,
    restarts: int = 1,
    seed: int = 1729,
    max_sweeps: int = 20,
    max_chromatic_search_nodes: int = 2_000_000,
):
    """Fit hard global-greedy NOTREKS with the additional chi(H) source prefix."""
    import flopsearch

    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array")
    chromatic = chromatic_number_or_lower_bound(
        data.shape[1], no_trek_pairs, max_search_nodes=max_chromatic_search_nodes)
    cpdag, diagnostics = flopsearch.flop_notreks(
        data,
        float(lambda_bic),
        no_trek_pairs,
        restarts=int(restarts),
        seed=int(seed),
        max_signature_rounds=int(max_sweeps),
        search_version="global_greedy_rust",
        source_prefix=chromatic.value,
        return_diagnostics=True,
    )
    diagnostics = dict(diagnostics)
    diagnostics.update({f"chromatic_{key}": value for key, value in asdict(chromatic).items()})
    diagnostics["search_version"] = "global_greedy_chromatic_sources"
    return np.asarray(cpdag), diagnostics


def fit_chromatic_source_fixed_notreks(
    X: np.ndarray,
    no_trek_pairs: Sequence[Pair],
    *,
    lambda_bic: float = 2.0,
    restarts: int = 1,
    seed: int = 1729,
    max_chromatic_search_nodes: int = 2_000_000,
):
    """Fit the close-to-FLOP fixed-signature kernel with a chi(H) source prefix."""
    import flopsearch

    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array")
    chromatic = chromatic_number_or_lower_bound(
        data.shape[1], no_trek_pairs, max_search_nodes=max_chromatic_search_nodes)
    cpdag, diagnostics = flopsearch.flop_notreks(
        data,
        float(lambda_bic),
        no_trek_pairs,
        restarts=int(restarts),
        seed=int(seed),
        search_version="fixed_signature_a",
        source_prefix=chromatic.value,
        return_diagnostics=True,
    )
    diagnostics = dict(diagnostics)
    diagnostics.update({f"chromatic_{key}": value for key, value in asdict(chromatic).items()})
    diagnostics["search_version"] = "fixed_signature_chromatic_sources"
    return np.asarray(cpdag), diagnostics
