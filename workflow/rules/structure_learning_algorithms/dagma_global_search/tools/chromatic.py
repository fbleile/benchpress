"""Chromatic diagnostics for NOTREKS information graphs."""

from __future__ import annotations

from typing import Sequence


def chromatic_number(
    number_of_vertices: int,
    pairs: Sequence[tuple[int, int]],
) -> int:
    """Compute the exact chromatic number with a DSATUR branch-and-bound."""
    if number_of_vertices < 1:
        return 0
    adjacency = [0] * number_of_vertices
    for left, right in pairs:
        left, right = int(left), int(right)
        if left == right or not (0 <= left < number_of_vertices
                                 and 0 <= right < number_of_vertices):
            raise ValueError("NOTREKS pair is not a valid graph edge")
        adjacency[left] |= 1 << right
        adjacency[right] |= 1 << left

    colors = [-1] * number_of_vertices
    best = number_of_vertices

    def greedy_upper_bound() -> int:
        order = sorted(range(number_of_vertices),
                       key=lambda vertex: adjacency[vertex].bit_count(),
                       reverse=True)
        greedy = [-1] * number_of_vertices
        used = 0
        for vertex in order:
            forbidden = {greedy[neighbor] for neighbor in range(number_of_vertices)
                         if adjacency[vertex] & (1 << neighbor)
                         and greedy[neighbor] >= 0}
            color = 0
            while color in forbidden:
                color += 1
            greedy[vertex] = color
            used = max(used, color + 1)
        return used

    best = greedy_upper_bound()
    if best <= 1:
        return best

    def choose_vertex() -> int:
        selected = -1
        selected_key = (-1, -1)
        for vertex, color in enumerate(colors):
            if color >= 0:
                continue
            neighbor_colors = {
                colors[neighbor]
                for neighbor in range(number_of_vertices)
                if adjacency[vertex] & (1 << neighbor)
                and colors[neighbor] >= 0
            }
            key = (len(neighbor_colors), adjacency[vertex].bit_count())
            if key > selected_key:
                selected = vertex
                selected_key = key
        return selected

    def search(colored: int, used: int) -> None:
        nonlocal best
        if used >= best:
            return
        if colored == number_of_vertices:
            best = used
            return
        vertex = choose_vertex()
        forbidden = {
            colors[neighbor]
            for neighbor in range(number_of_vertices)
            if adjacency[vertex] & (1 << neighbor)
            and colors[neighbor] >= 0
        }
        for color in range(min(used + 1, best - 1)):
            if color in forbidden:
                continue
            colors[vertex] = color
            search(colored + 1, max(used, color + 1))
            colors[vertex] = -1

    search(0, 0)
    return best


def chromatic_upper_bound(
    number_of_vertices: int,
    pairs: Sequence[tuple[int, int]],
) -> int:
    """Return a fast DSATUR greedy coloring upper bound.

    Unlike :func:`chromatic_number`, this never performs branch-and-bound and
    is therefore suitable for the larger-dimensional benchmark runs.
    """
    if number_of_vertices < 1:
        return 0
    adjacency = [0] * number_of_vertices
    for left, right in pairs:
        left, right = int(left), int(right)
        if left == right or not (0 <= left < number_of_vertices
                                 and 0 <= right < number_of_vertices):
            raise ValueError("NOTREKS pair is not a valid graph edge")
        adjacency[left] |= 1 << right
        adjacency[right] |= 1 << left

    colors = [-1] * number_of_vertices
    for _ in range(number_of_vertices):
        candidates = [v for v, color in enumerate(colors) if color < 0]
        if not candidates:
            break
        vertex = max(
            candidates,
            key=lambda v: (
                len({colors[u] for u in range(number_of_vertices)
                     if adjacency[v] & (1 << u) and colors[u] >= 0}),
                adjacency[v].bit_count(),
                -v,
            ),
        )
        forbidden = {
            colors[u] for u in range(number_of_vertices)
            if adjacency[vertex] & (1 << u) and colors[u] >= 0
        }
        color = 0
        while color in forbidden:
            color += 1
        colors[vertex] = color
    return max(colors) + 1
