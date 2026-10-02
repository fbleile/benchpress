import numpy as np

from scripts.notreks_protocol import (
    METHODS,
    derive_seed,
    graph_pairs,
    make_data,
    make_graph,
    select_pairs,
)
from scripts.notreks_protocol_registry import REGISTRY
from scripts.lrz_make_jobfarm_cmds import main as make_jobfarm_commands


def test_protocol_uses_the_declared_standard_flop_notreks_method():
    assert "flop_notreks" in METHODS
    assert "flop-nt-local" not in METHODS


def test_seed_derivation_is_stable_and_namespaced():
    assert derive_seed("graph", 20, "er", 2, 0) == derive_seed(
        "graph", 20, "er", 2, 0)
    assert derive_seed("graph", 20, "er", 2, 0) != derive_seed(
        "data", 20, "er", 2, 0)


def test_graph_and_no_trek_artifacts_are_deterministic():
    first = make_graph(20, "er", 2, 1234)
    second = make_graph(20, "er", 2, 1234)
    assert np.array_equal(first, second)
    assert np.all(np.diag(first) == 0)
    assert len(graph_pairs(first)) >= 0


def test_knowledge_rounds_are_reproducible_and_bounded():
    truth = make_graph(20, "er", 2, 1234)
    pairs = graph_pairs(truth)
    selected = select_pairs(pairs, .25, 99)
    assert len(selected) == max(1, round(.25 * len(pairs)))
    assert set(selected) <= set(pairs)
    assert select_pairs(pairs, .25, 99) == selected
    assert select_pairs(pairs, 1.0, 99) == pairs


def test_ws_graph_generation_does_not_depend_on_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = make_graph(5, "ws", 2, 12346)
    assert graph.shape == (5, 5)
    assert np.all(np.diag(graph) == 0)


def test_misspecification_data_arms_are_finite_and_standardized():
    truth = make_graph(10, "er", 2, 1234)
    for model in ("linear_gaussian", "linear_nongaussian", "nonlinear_gaussian"):
        x, weights = make_data(truth, 200, 1234, 5678, model=model)
        assert x.shape == (200, 10)
        assert weights.shape == truth.shape
        assert np.isfinite(x).all()
        assert np.allclose(x.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(x.std(axis=0), 1.0, atol=1e-10)


def test_misspecification_arms_match_main_grid():
    main = REGISTRY["main"]
    for name in ("main-misspec-linear-nongaussian", "main-misspec-nonlinear-gaussian"):
        arm = REGISTRY[name]
        assert arm.cells == main.cells
        assert arm.n_values == main.n_values
        assert arm.q_values == main.q_values
        assert arm.q25_rounds == main.q25_rounds
        assert arm.methods == main.methods
        assert arm.excluded_methods_by_dimension == main.excluded_methods_by_dimension


def test_jobfarm_compiler_applies_dimension_method_exclusions(tmp_path, monkeypatch):
    command_file = tmp_path / "commands.txt"
    output_root = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        [
            "lrz_make_jobfarm_cmds.py",
            "--repo", str(tmp_path),
            "--output-root", str(output_root),
            "--command-file", str(command_file),
            "--synthetic-experiments", "main",
            "--replicate-batch-size", "20",
        ],
    )
    make_jobfarm_commands()
    d100_lines = [
        line for line in command_file.read_text().splitlines()
        if any(f"--cell-start {index} " in line for index in range(10, 16))
    ]
    assert d100_lines
    for line in d100_lines:
        methods = line.split("--methods ", 1)[1].split(" --workers", 1)[0].split()
        assert not set(methods) & {
            "dagma", "dagma-nt-edge-mask", "dagma-nt-post", "dagma_notreks"
        }

    d20_lines = [
        line for line in command_file.read_text().splitlines()
        if "--cell-start 0 " in line
    ]
    assert d20_lines
    methods = d20_lines[0].split("--methods ", 1)[1].split(" --workers", 1)[0].split()
    assert {"flop", "flop-nt-post", "dagma", "dagma-nt-post"} <= set(methods)
