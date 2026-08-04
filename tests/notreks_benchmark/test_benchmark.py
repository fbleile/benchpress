import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.notreks_benchmark import (
    METHOD_IDS, analyse, benchpress_config, collect_results, compile_configs,
    expand_scenarios,
)


def _spec():
    return {
        "models": {"gauss": {"method": "linear", "sem_type": "gauss"}},
        "graphs": {"er": {"method": "er", "expected_degree": 2}},
        "grid": {"models": ["gauss"], "graphs": ["er"],
                 "dimensions": [5, 10], "sample_sizes": [100],
                 "knowledge_fractions": [0.1, 1.0], "seeds": [1, 2]},
        "scenarios": [],
    }


def _defaults():
    return {"knowledge_seed": 1, "algorithm_seed": 2,
            "dagma_trek_weight": 10, "flop_restarts": 4,
            "flop_notreks_restarts": 2, "flop_notreks_max_sweeps": 1,
            "flop_notreks_signature_top_k": 8,
            "flop_notreks_signature_exploration_k": 2,
            "flop_notreks_max_signature_rounds": 20, "n_jobs": 1}


def test_grid_and_single_scenarios_are_easy_to_expand():
    spec = _spec()
    spec["scenarios"] = [{"id": "targeted", "model": "gauss", "graph": "er",
                          "d": 7, "n": 80, "knowledge_fraction": .5,
                          "seeds": [3]}]
    scenarios = expand_scenarios(spec)
    assert len(scenarios) == 5
    assert scenarios[-1]["id"] == "targeted"


def test_compiled_config_contains_exactly_four_methods_and_shared_fraction():
    scenario = expand_scenarios(_spec())[0]
    config = benchpress_config(scenario, _defaults(), smoke=True)
    algorithms = config["resources"]["structure_learning_algorithms"]
    assert tuple(algorithms) == METHOD_IDS
    assert algorithms["dagma_notreks"][0]["knowledge_fraction"] == .1
    assert algorithms["flop_notreks"][0]["knowledge_fraction"] == .1
    assert algorithms["flop_notreks"][0]["search_strategy"] == "global_greedy"
    assert algorithms["dagma"][0]["T"] == 1
    assert algorithms["dagma"][0]["w_threshold"] == .3
    assert algorithms["dagma_notreks"][0]["w_threshold"] == .3
    assert "postselection_policy" not in algorithms["dagma_notreks"][0]


def test_compiler_emits_native_benchpress_cluster_commands(
        tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    spec = _spec()
    spec["tuned_hyperparameters"] = "defaults.json"
    spec_path.write_text(json.dumps(spec))
    (tmp_path / "defaults.json").write_text(json.dumps(_defaults()))
    output_dir = tmp_path / "compiled"

    compile_configs(spec_path, output_dir)

    commands = (output_dir / "commands.txt").read_text().splitlines()
    assert len(commands) == 4
    assert all(command.startswith(
        "snakemake --snakefile workflow/Snakefile --use-apptainer "
    ) for command in commands)
    assert all("--configfile " in command for command in commands)
    assert all("--cores ${NOTREKS_CORES:-1}" in command
               for command in commands)


def test_analysis_only_builds_requested_paired_comparisons(tmp_path: Path):
    rows = []
    for seed in (1, 2):
        for method, shd in zip(METHOD_IDS, (3, 2, 5, 4)):
            rows.append({"seed": seed, "algorithm": method,
                         "SHD_cpdag": shd + 1, "SHD_pattern": shd,
                         "F1_pattern": 1 / (1 + shd),
                         "num_mi_violations": (
                             0 if method.endswith("notreks") else None),
                         "mi_violation_fraction": (
                             0.0 if method.endswith("notreks") else None)})
    source = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    analyse(source, tmp_path / "analysis")
    paired = pd.read_csv(tmp_path / "analysis/paired_per_dataset.csv")
    assert set(paired.comparison) == {
        "flop_vs_flop_notreks", "dagma_vs_dagma_notreks"}
    assert set(paired.delta_SHD_pattern) == {-1}
    assert set(paired.delta_SHD_cpdag) == {-1}
    assert set(paired.notreks_num_mi_violations) == {0}
    assert set(paired.notreks_mi_violation_fraction) == {0.0}
    assert (tmp_path / "analysis/algorithm_summary.csv").exists()
    assert (tmp_path / "analysis/figures/paired_cpdag_shd_change.png").exists()
    report = (tmp_path / "analysis/REPORT.md").read_text()
    assert "Immediate interpretation" in report
    assert "wins/ties/losses" in report
    assert (tmp_path / "analysis/factor_effects.csv").exists()
    assert (tmp_path / "analysis/causal_estimand.dot").exists()


def test_collect_joins_scenario_metadata(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"id": "scenario-a", "model": "gauss", "graph": "er",
                   "d": 5, "n": 100, "knowledge_fraction": .1}]).to_csv(
                       manifest, index=False)
    result_dir = tmp_path / "results/scenario-a"
    result_dir.mkdir(parents=True)
    pd.DataFrame([{"algorithm": "flop", "seed": 1}]).to_csv(
        result_dir / "joint_benchmarks.csv", index=False)
    output = tmp_path / "all.csv"
    collect_results(manifest, tmp_path / "results", output)
    row = pd.read_csv(output).iloc[0]
    assert row.scenario == "scenario-a"
    assert row.knowledge_fraction == .1


def test_analysis_rejects_incomplete_dataset_rows(tmp_path: Path):
    source = tmp_path / "incomplete.csv"
    pd.DataFrame([
        {"scenario": "one", "seed": 1, "algorithm": method,
         "SHD_cpdag": index}
        for index, method in enumerate(METHOD_IDS[:-1])
    ]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="missing methods|exactly once"):
        analyse(source, tmp_path / "analysis")
