"""Shared data contracts for weighted-to-graph postprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class WeightedGraphEstimate:
    weights: np.ndarray
    node_names: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError("weights must be a square matrix")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite")
        object.__setattr__(self, "weights", weights.copy())
        if self.node_names is not None and len(self.node_names) != len(weights):
            raise ValueError("node_names must match the weighted matrix")


@dataclass(frozen=True)
class PostprocessingBudget:
    max_seconds: float = 1.0
    max_candidate_evaluations: int = 1000
    max_local_search_iterations: int = 100


@dataclass
class GraphCandidate:
    graph: np.ndarray
    construction: str
    threshold: float | None
    score: float | None = None
    feasible: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostprocessingResult:
    graph: np.ndarray
    score: float
    policy: str
    source_estimate_index: int
    candidates: list[GraphCandidate]
    diagnostics: dict[str, Any]


class GraphScore(Protocol):
    def score_graph(self, graph: np.ndarray, data: Any = None) -> float: ...


class FeasibilityConstraint(Protocol):
    def is_feasible(self, graph: np.ndarray) -> bool: ...
    def violation_summary(self, graph: np.ndarray) -> dict[str, Any]: ...
