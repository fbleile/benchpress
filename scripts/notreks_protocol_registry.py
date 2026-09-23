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
    # Optional per-dimension exclusion, used by the d=100 main slice where
    # DAGMA is intentionally disabled for runtime reasons.
    excluded_methods_by_dimension: tuple[tuple[int, tuple[str, ...]], ...] = ()
    knowledge_strategies: tuple[str, ...] = ("random",)

    def attempts_for(self, method: str) -> int:
        family = "dagma" if method.startswith("dagma") else "flop"
        return dict(self.attempts_by_family)[family]

    def methods_for(self, d: int) -> tuple[str, ...]:
        excluded = dict(self.excluded_methods_by_dimension).get(d, ())
        return tuple(method for method in self.methods if method not in excluded)


MAIN_CELLS = (
    (20, "er", 2), (20, "er", 4), (20, "ws", 2), (20, "ws", 4),
    (50, "er", 2), (50, "er", 4), (50, "er", 8),
    (50, "ws", 2), (50, "ws", 4), (50, "ws", 8),
    (100, "er", 2), (100, "er", 4), (100, "er", 8),
    (100, "ws", 2), (100, "ws", 4), (100, "ws", 8),
)

MAIN_METHODS = (
    "flop", "flop-nt-edge-mask", "flop-nt-post", "flop_notreks",
    "dagma", "dagma-nt-edge-mask", "dagma-nt-post", "dagma_notreks",
    "var_sortnregress", "r2_sortnregress",
)

REGISTRY = {
    "main": ExperimentSpec(
        "main", "primary paired recovery benchmark", MAIN_CELLS,
        (100, 500, 2000), (0.0, .25, 1.0), 5,
        MAIN_METHODS,
        excluded_methods_by_dimension=((100, ("dagma", "dagma-nt-edge-mask",
                                               "dagma-nt-post", "dagma_notreks")),)),
    "integration-ablation": ExperimentSpec(
        "integration-ablation", "integration and post-selection ablation",
        ((50, "er", 2), (50, "er", 4)), (500,), (.25, 1.0), 1,
        ("flop", "flop-nt-edge-mask", "flop-nt-post", "flop_notreks",
         "dagma", "dagma-nt-edge-mask", "dagma-nt-post", "dagma_notreks"),
        derives_from="main"),
    "prior-structure": ExperimentSpec(
        "prior-structure", "NOTREKS information-graph structure study",
        ((20, "er", 2), (20, "er", 4), (20, "ws", 2), (20, "ws", 4)),
        (500,), (.25,), 5, ("flop", "flop_notreks", "dagma", "dagma_notreks"),
        knowledge_strategies=("random", "bipartite-max-capacity", "chromatic-greedy")),
    "d100-flop": ExperimentSpec(
        "d100-flop", "high-dimensional FLOP scaling study",
        ((100, "er", 2), (100, "er", 4), (100, "er", 8),
         (100, "ws", 2), (100, "ws", 4), (100, "ws", 8)),
        (1000,), (.25, 1.0), 5,
        ("flop", "flop-nt-edge-mask", "flop-nt-post", "flop_notreks"),
        derives_from="main"),
    "heterogeneity": ExperimentSpec(
        "heterogeneity", "descriptive stratum heterogeneity summaries", (),
        (), (), 0, (), derives_from="main"),
    "causalassembly": ExperimentSpec(
        "causalassembly", "causalAssembly nonlinear n=500 benchmark", (),
        (500,), (.25, 1.0), 1,
        ("flop", "flop_notreks", "flop-nt-edge-mask", "flop-nt-post",
         "dagma", "dagma_notreks", "dagma-nt-edge-mask", "dagma-nt-post",
         "var_sortnregress", "r2_sortnregress",
         "dagma_nonlinear", "dagma_nonlinear_notreks"),
        graph_replicates=5, dataset="causalassembly_static_n500"),
    "sachs": ExperimentSpec(
        "sachs", "Sachs 50-bootstrap fixed-data benchmark", (),
        (853,), (0.25, 1.0), 1,
        ("flop", "flop_notreks", "flop-nt-edge-mask", "flop-nt-post",
         "dagma", "dagma_notreks", "dagma-nt-edge-mask", "dagma-nt-post",
         "var_sortnregress", "r2_sortnregress",
         "dagma_nonlinear", "dagma_nonlinear_notreks"),
        graph_replicates=50, dataset="sachs_fixed"),
    "pstrek-vs-tcc": ExperimentSpec(
        "pstrek-vs-tcc", "PSTrek versus spectral trek-cycle constraint", (
            (8, "er", 2), (8, "er", 4),
            (10, "er", 2), (10, "er", 4),
            (12, "er", 2), (12, "er", 4)),
        # Match the main protocol's independent graph seeds and partial-
        # knowledge draws: 20 graph replicates, five q=.25 draws, and one
        # complete q=1 draw per graph. There is intentionally no q=0 vanilla
        # row here because this experiment compares two NOTREKS penalties.
        (100,), (0.0, .25, 1.0), 5,
        ("dagma", "var_sortnregress", "r2_sortnregress",
         "dagma_notreks", "dagma_notreks_tcc",
         "dagma-nt-edge-mask", "dagma-nt-post"),
        graph_replicates=20),
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
    return sum(
        scaled_replicates(spec, fraction) * len(spec.n_values)
        * (spec.q25_rounds if q == .25 else 1)
        * len(tuple(
            m for m in spec.methods_for(d)
            if not (spec.name == "main" and (
                (q == 0.0 and m not in {"flop", "dagma", "var_sortnregress", "r2_sortnregress"})
                or (q > 0.0 and m in {"flop", "dagma", "var_sortnregress", "r2_sortnregress"})
            ))
            and not (spec.name != "main" and q != 1.0
                     and m in {"var_sortnregress", "r2_sortnregress"})
        ))
        * len(spec.knowledge_strategies if q != 1.0 else ("random",))
        for d, _family, _density in spec.cells
        for q in spec.q_values
    )
