"""Experiment-level registry for the NOTREKS synthetic protocol.

The registry is deliberately declarative: sharding and artifact naming can be
validated before expensive solver jobs are submitted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    cells: tuple[tuple[int, str, int], ...]
    n_values: tuple[int, ...]
    q_values: tuple[float, ...]
    q25_rounds: int
    methods: tuple[str, ...]
    graph_replicates: int = 20
    derives_from: str | None = None
    # Practical solver budgets for the protocol.  FLOP is cheap enough to
    # benefit from many order restarts; DAGMA is substantially more costly.
    attempts_by_family: tuple[tuple[str, int], ...] = (("flop", 20), ("dagma", 2))
    dataset: str = "synthetic_linear_gaussian"

    def attempts_for(self, method: str) -> int:
        family = "dagma" if method.startswith("dagma") else "flop"
        return dict(self.attempts_by_family)[family]


MAIN_CELLS = (
    (20, "er", 2), (20, "er", 4), (20, "ws", 2), (20, "ws", 4),
    (50, "er", 2), (50, "er", 4), (50, "er", 8),
    (50, "ws", 2), (50, "ws", 4), (50, "ws", 8),
)

REGISTRY = {
    "main": ExperimentSpec(
        "main", "primary paired recovery benchmark", MAIN_CELLS,
        (100, 500, 2000), (.25, 1.0), 5,
        ("flop", "flop_notreks", "dagma", "dagma_notreks")),
    "integration-ablation": ExperimentSpec(
        "integration-ablation", "integration and post-selection ablation",
        ((50, "er", 2), (50, "er", 4)), (500,), (.25, 1.0), 1,
        ("flop", "flop-nt-edge-mask", "flop-nt-post", "flop_notreks",
         "dagma", "dagma-nt-edge-mask", "dagma-nt-post", "dagma_notreks")),
    "prior-structure": ExperimentSpec(
        "prior-structure", "NOTREKS information-graph structure study",
        ((20, "er", 2), (20, "er", 4), (20, "ws", 2), (20, "ws", 4)),
        (500,), (.25,), 5, ("flop", "flop_notreks", "dagma", "dagma_notreks")),
    "d100-flop": ExperimentSpec(
        "d100-flop", "high-dimensional FLOP scaling study",
        ((100, "er", 2), (100, "er", 4), (100, "er", 8),
         (100, "ws", 2), (100, "ws", 4), (100, "ws", 8)),
        (1000,), (.25, 1.0), 5,
        ("flop", "flop-nt-edge-mask", "flop-nt-post", "flop_notreks")),
    "heterogeneity": ExperimentSpec(
        "heterogeneity", "descriptive stratum heterogeneity summaries", (),
        (), (), 0, (), derives_from="main"),
    "causalassembly": ExperimentSpec(
        "causalassembly",
        "causalAssembly full production-line semisynthetic benchmark",
        (), (500, 2000, 5000), (.10, .25, .50, 1.0), 1,
        ("flop", "flop_notreks", "flop_notreks_edge_mask",
         "flop_notreks_postselection", "dagma", "dagma_notreks",
         "dagma_notreks_postselection"),
        graph_replicates=10, dataset="causalassembly_full"),
}


def select_registry(names: list[str] | None, fraction: float) -> list[ExperimentSpec]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    selected = list(REGISTRY) if names is None or "all" in names else names
    unknown = sorted(set(selected) - set(REGISTRY))
    if unknown:
        raise ValueError(f"unknown protocol experiments: {unknown}")
    return [REGISTRY[name] for name in selected]


def scaled_replicates(spec: ExperimentSpec, fraction: float) -> int:
    return max(1, math.ceil(spec.graph_replicates * fraction))


def planned_solver_rows(spec: ExperimentSpec, fraction: float) -> int:
    if spec.derives_from:
        return 0
    if spec.dataset == "causalassembly_full":
        # The dedicated causalAssembly runner deduplicates vanilla rows, so
        # this is an expanded planning upper bound for registry summaries.
        return (spec.graph_replicates * len(spec.n_values) *
                len(spec.q_values) * len(spec.methods))
    priors = sum(spec.q25_rounds if q == .25 else 1 for q in spec.q_values)
    return (len(spec.cells) * scaled_replicates(spec, fraction) *
            len(spec.n_values) * priors * len(spec.methods))
