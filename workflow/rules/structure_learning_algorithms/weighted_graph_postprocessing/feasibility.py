"""Exact discrete feasibility adapters, independent of continuous penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma_anytime.graph_utils import is_dag


class DagConstraint:
    def is_feasible(self, graph):
        return bool(is_dag(np.asarray(graph, dtype=int)))

    def violation_summary(self, graph):
        valid = self.is_feasible(graph)
        return {"DAG_valid": valid, "dag_violations": int(not valid)}


@dataclass(frozen=True)
class NoTreksConstraint:
    pairs: tuple[tuple[int, int], ...]

    def __init__(self, pairs: Sequence[tuple[int, int]]):
        canonical = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("no-trek pairs must have two endpoints")
            i, j = sorted((int(pair[0]), int(pair[1])))
            if i == j:
                raise ValueError("no-trek endpoints must differ")
            canonical.append((i, j))
        object.__setattr__(self, "pairs", tuple(sorted(set(canonical))))

    def violation_count(self, graph):
        adjacency = np.asarray(graph, dtype=bool)
        d = len(adjacency)
        reach = adjacency.copy()
        np.fill_diagonal(reach, True)
        for k in range(d):
            reach |= reach[:, [k]] & reach[[k], :]
        return sum(
            bool(np.any(reach[:, i] & reach[:, j]))
            for i, j in self.pairs)

    def is_feasible(self, graph):
        return self.violation_count(graph) == 0

    def violation_summary(self, graph):
        count = self.violation_count(graph)
        return {
            "notreks_violation_count": count,
            "notreks_pair_count": len(self.pairs),
        }


@dataclass(frozen=True)
class CompositeFeasibility:
    constraints: tuple[object, ...] = (DagConstraint(),)

    def is_feasible(self, graph):
        return all(item.is_feasible(graph) for item in self.constraints)

    def violation_summary(self, graph):
        result = {"feasible": self.is_feasible(graph)}
        for item in self.constraints:
            result.update(item.violation_summary(graph))
        return result
