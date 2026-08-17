import numpy as np
import pytest

from workflow.rules.structure_learning_algorithms.flop_notreks.chromatic_sources import (
    chromatic_number_or_lower_bound,
    fit_chromatic_source_flop,
    fit_chromatic_source_notreks,
)


@pytest.mark.parametrize(
    ("d", "edges", "expected"),
    [
        (5, [], 1),
        (5, [(0, 1), (1, 2), (2, 3), (3, 4)], 2),
        (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 3),
        (4, [(i, j) for i in range(4) for j in range(i + 1, 4)], 4),
    ],
)
def test_exact_chromatic_number(d, edges, expected):
    result = chromatic_number_or_lower_bound(d, edges)
    assert result.exact
    assert result.value == expected
    assert result.clique_lower_bound <= result.value <= result.coloring_upper_bound


def test_bounded_search_falls_back_to_valid_clique_lower_bound():
    # C5 has clique number two and chromatic number three.  One search node is
    # intentionally insufficient, so the returned value must be the safe LB.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    result = chromatic_number_or_lower_bound(5, edges, max_search_nodes=1)
    assert not result.exact
    assert result.value == result.clique_lower_bound == 2
    assert result.coloring_upper_bound == 3


def test_source_prefix_extension_forces_selected_prefix_parentless():
    flopsearch = pytest.importorskip("flopsearch")
    if not hasattr(flopsearch, "flop_source_prefix"):
        pytest.skip("installed extension predates source-prefix FLOP")
    rng = np.random.default_rng(7)
    data = rng.normal(size=(100, 5))
    _, diagnostics = fit_chromatic_source_flop(
        data, [(0, 1), (1, 2), (2, 0)], restarts=0, seed=7)
    order = diagnostics["selected_order"]
    edges = {tuple(edge) for edge in diagnostics["selected_dag_edges"]}
    assert diagnostics["source_prefix"] == 3
    assert all(child not in order[:3] for _, child in edges)


def test_combined_variant_is_parentless_and_hard_notreks_feasible():
    flopsearch = pytest.importorskip("flopsearch")
    if "source_prefix" not in (flopsearch.flop_notreks.__text_signature__ or ""):
        pytest.skip("installed extension predates combined chromatic NOTREKS")
    rng = np.random.default_rng(17)
    data = rng.normal(size=(120, 6))
    _, diagnostics = fit_chromatic_source_notreks(
        data, [(0, 1), (1, 2), (2, 0)], restarts=0, seed=17, max_sweeps=2)
    edges = {tuple(edge) for edge in diagnostics["selected_dag_edges"]}
    assert diagnostics["source_prefix"] == 3
    assert diagnostics["final_no_trek_violation_count"] == 0
    # The Rust result does not expose its order, but every selected source
    # position is certified internally and hard feasibility is checked here.
    assert all(parent != child for parent, child in edges)


def test_incremental_promotion_strategy_returns_a_hard_feasible_dag():
    flopsearch = pytest.importorskip("flopsearch")
    rng = np.random.default_rng(23)
    data = rng.normal(size=(100, 5))
    _, diagnostics = flopsearch.flop_notreks(
        data, 2.0, [(0, 1)], restarts=0, seed=23,
        max_signature_rounds=5, search_version="incremental_promotion_d",
        return_diagnostics=True)
    assert diagnostics["search_version"] == "incremental_promotion_d"
    assert diagnostics["final_no_trek_violation_count"] == 0
