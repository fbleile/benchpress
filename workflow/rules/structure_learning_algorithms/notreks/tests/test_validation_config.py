from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from validation import (
    assert_no_fixed_data_duplication_in_dryrun,
    build_selected_benchmark_config,
    expand_grid_config,
    expected_algorithm_run_counts,
    parse_dryrun_job_counts,
    prepare_validation_run,
    select_best_by_method_family,
    write_final_benchmark_config,
)
from cli import (  # noqa: E402
    _tag_benchmark_frame_path,
    _tag_config_path,
    _tag_grid_path,
    _tag_manifest_path,
    _tag_selected_benchmark_config_path,
    _tag_selected_benchmark_manifest_path,
    _tag_selected_dir,
)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "resources/data/mydatasets").mkdir(parents=True)
    (repo / "resources/adjmat/myadjmats").mkdir(parents=True)
    return repo


def _simple_cartesian_grid(repo: Path) -> Path:
    grid_path = repo / "configs/notreks/grids/smoke_grid.json"
    grid_path.parent.mkdir(parents=True)
    grid_path.write_text(
        json.dumps(
            {
                "experiment_id": "notreks_smoke",
                "benchmark_name": "notreks_smoke_validation",
                "filename_prefix": "notreks/smoke/validation/",
                "data": {
                    "type": "fixed",
                    "seeds": [101, 102],
                    "d": 10,
                    "n": 500,
                    "graph": "er",
                    "expected_degree": 2,
                    "standardized": True,
                },
                "methods": {
                    "gcastle_pc": {
                        "enabled": True,
                        "grid": {
                            "variant": ["stable"],
                            "alpha": [0.05],
                            "ci_test": ["fisherz"],
                            "timeout": [None],
                        },
                    },
                    "gcastle_direct_lingam": {
                        "enabled": True,
                        "grid": {"measure": ["pwling"], "thresh": [0.3], "timeout": [None]},
                    },
                    "marginal_trek_graph": {
                        "enabled": True,
                        "grid": {
                            "independence_test": ["spearman"],
                            "independence_alpha": [0.05],
                            "independence_correction": ["benjamini-hochberg"],
                            "independence_cache_dir": [None],
                            "timeout": [None],
                        },
                    },
                    "notreks": {
                        "enabled": True,
                        "grid": {
                            "score": ["least_squares"],
                            "dag_seq": ["logdet"],
                            "trek_seq": ["exp"],
                            "trek_reg": [1.0],
                            "trek_penalty_mu_mode": ["hard_outside_mu"],
                            "regularizer": ["l1"],
                            "regularizer_scale": [0.01],
                            "independence_test": ["spearman", "gcastle_fisherz"],
                            "independence_alpha": [0.05],
                            "independence_correction": ["benjamini-hochberg"],
                            "threshold": [0.1],
                            "max_iter": [3000],
                            "path_steps": [5],
                            "power_iter_steps": [5],
                            "timeout": [None],
                        },
                    },
                },
            }
        )
        + "\n"
    )
    return grid_path


def test_tag_derived_paths_are_standardized() -> None:
    assert _tag_grid_path("hyperparam") == Path("configs/notreks/grids/hyperparam_grid.json")
    assert _tag_config_path("hyperparam") == Path("configs/notreks/expanded/hyperparam_config.json")
    assert _tag_manifest_path("hyperparam") == Path("configs/notreks/expanded/hyperparam_manifest.csv")
    assert _tag_selected_dir() == Path("configs/notreks/selected")
    assert _tag_benchmark_frame_path("full_benchmark") == Path(
        "configs/notreks/benchmarks/full_benchmark_frame.json"
    )
    assert _tag_selected_benchmark_config_path("full_benchmark") == Path(
        "configs/notreks/expanded/selected_full_benchmark_config.json"
    )
    assert _tag_selected_benchmark_manifest_path("full_benchmark") == Path(
        "configs/notreks/expanded/selected_full_benchmark_manifest.csv"
    )


def test_expand_grid_writes_top_level_config_and_manifest(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    grid_path = _simple_cartesian_grid(repo)
    out_config = repo / "configs/notreks/expanded/smoke_config.json"
    out_manifest = repo / "configs/notreks/expanded/smoke_manifest.csv"
    expanded = expand_grid_config(repo, grid_path, out_config, out_manifest)

    assert expanded.config_path == out_config
    assert expanded.manifest_csv == out_manifest
    assert expanded.manifest_json == out_manifest.with_suffix(".json")
    assert expanded.algorithm_counts == {
        "gcastle_pc": 1,
        "gcastle_lingam": 1,
        "marginal_trek_graph": 1,
        "notreks": 2,
    }
    assert "results/" not in str(out_config)
    assert "results/" not in str(out_manifest)

    config = json.loads(out_config.read_text())
    resources = config["resources"]["structure_learning_algorithms"]
    assert len(config["benchmark_setup"][0]["data"]) == 2
    assert len(resources["gcastle_pc"]) == 1
    assert len(resources["gcastle_direct_lingam"]) == 1
    assert len(resources["marginal_trek_graph"]) == 1
    assert len(resources["notreks"]) == 2
    assert resources["notreks"][0] == {
        "id": "notreks__grid000",
        "alg_id": "n000",
        "params_manifest": "configs/notreks/expanded/smoke_manifest.json",
    }
    text = json.dumps(resources["notreks"])
    assert "independence_cache_dir" not in text
    assert "function_class" not in json.dumps(resources["notreks"])

    manifest = pd.read_csv(out_manifest)
    assert set(manifest["method_family"]) == {
        "gcastle_pc",
        "gcastle_lingam",
        "marginal_trek_graph",
        "notreks",
    }
    assert manifest["algorithm_id"].is_unique
    assert set(manifest.loc[manifest["base_method"] == "notreks", "path_id"]) == {"n000", "n001"}
    assert set(manifest["config_path"]) == {"configs/notreks/expanded/smoke_config.json"}


def test_default_benchmark_frames_use_fresh_seeds() -> None:
    repo = Path(__file__).resolve().parents[5]
    for tag in ("smoke", "hyperparam"):
        grid = json.loads((repo / f"configs/notreks/grids/{tag}_grid.json").read_text())
        frame_file = "smoke_benchmark_frame.json" if tag == "smoke" else "full_benchmark_frame.json"
        frame = json.loads((repo / "configs/notreks/benchmarks" / frame_file).read_text())
        assert set(grid["data"]["seeds"]).isdisjoint(set(frame["data"]["seeds"]))


def test_prepare_validation_tiny_writes_one_config_and_manifest(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run_dir = repo / "results/validation-tiny"
    prepared = prepare_validation_run(repo, run_dir, "tiny")

    assert prepared.validation_config_path == run_dir / "configs/validation_hparam_config.json"
    config = json.loads(prepared.validation_config_path.read_text())
    setup = config["benchmark_setup"][0]
    assert len(setup["data"]) == 2
    assert "/" not in setup["data"][0]["data_id"].rsplit("/", 1)[0]
    assert setup["data"][0]["data_id"].endswith("s101.csv")
    assert setup["data"][1]["data_id"].endswith("s102.csv")
    assert all(row["seed_range"] is None for row in setup["data"])
    assert len(setup["evaluation"]["benchmarks"]["ids"]) == 4

    resources = config["resources"]["structure_learning_algorithms"]
    assert set(resources) == {"gcastle_pc", "gcastle_direct_lingam", "notreks"}
    assert len(resources["gcastle_pc"]) == 1
    assert len(resources["gcastle_direct_lingam"]) == 1
    assert len(resources["notreks"]) == 2
    assert resources["notreks"][0] == {
        "id": "notreks__grid000",
        "alg_id": "n000",
        "params_manifest": str(prepared.validation_manifest_json.relative_to(repo)),
    }
    assert resources["notreks"][1]["alg_id"] == "n001"

    manifest = pd.read_csv(prepared.validation_manifest_csv)
    assert set(manifest["method_family"]) == {"gcastle_pc", "gcastle_lingam", "notreks"}
    assert manifest["algorithm_id"].is_unique
    assert set(manifest.loc[manifest["base_method"] == "notreks", "path_id"]) == {"n000", "n001"}
    assert set(manifest["phase"]) == {"validation"}
    assert all(Path(value).name == "validation_hparam_config.json" for value in manifest["config_path"])
    text = prepared.validation_config_path.read_text()
    assert "seed=1/seed=1" not in text
    assert "n=None/seed=1" not in text
    assert "independence_cache_dir" not in text
    assert "function_class" not in json.dumps(resources["notreks"])
    manifest_json = json.loads(prepared.validation_manifest_json.read_text())
    notreks_full = [row for row in manifest_json if row["algorithm_id"] == "notreks__grid001"][0]
    assert notreks_full["hyperparameters"]["independence_test"] == "gcastle_fisherz"
    assert notreks_full["hyperparameters"]["dag_seq"] == "logdet"
    assert notreks_full["hyperparameters"]["independence_cache_dir"].endswith("independence_cache")


def test_prepare_validation_reuses_matching_fixed_data_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run_dir = repo / "results/reuse-fixed-data"
    prepared = prepare_validation_run(repo, run_dir, "tiny")
    config = json.loads(prepared.validation_config_path.read_text())
    data_id = config["benchmark_setup"][0]["data"][0]["data_id"]
    data_path = repo / "resources/data/mydatasets" / data_id
    graph_id = config["benchmark_setup"][0]["data"][0]["graph_id"]
    graph_path = repo / "resources/adjmat/myadjmats" / graph_id
    before = (data_path.stat().st_mtime_ns, graph_path.stat().st_mtime_ns)

    prepare_validation_run(repo, run_dir, "tiny")

    after = (data_path.stat().st_mtime_ns, graph_path.stat().st_mtime_ns)
    assert after == before


def test_validation_tiny_expected_run_counts(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run_dir = repo / "results/counts"
    prepare_validation_run(repo, run_dir, "tiny")
    config = json.loads((run_dir / "configs/validation_hparam_config.json").read_text())
    num_data = len(config["benchmark_setup"][0]["data"])
    resources = config["resources"]["structure_learning_algorithms"]
    assert num_data == 2
    assert len(resources["gcastle_pc"]) * num_data == 2
    assert len(resources["gcastle_direct_lingam"]) * num_data == 2
    assert len(resources["notreks"]) * num_data == 4
    assert expected_algorithm_run_counts(config) == {
        "copy_fixed_data": 2,
        "gcastle_pc": 2,
        "gcastle_direct_lingam": 2,
        "empty_graph": 0,
        "complete_undirected_graph": 0,
        "marginal_trek_graph": 0,
        "notreks": 4,
    }


def test_validation_tiny_dryrun_count_guardrails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run_dir = repo / "results/dryrun-counts"
    prepare_validation_run(repo, run_dir, "tiny")
    config = json.loads((run_dir / "configs/validation_hparam_config.json").read_text())
    expected = expected_algorithm_run_counts(config)
    dryrun = """
Job stats:
job                       count
----------------------  -------
copy_fixed_data              2
gcastle_direct_lingam        2
gcastle_pc                   2
notreks                      4
total                       69
"""
    assert parse_dryrun_job_counts(dryrun)["copy_fixed_data"] == 2
    assert_no_fixed_data_duplication_in_dryrun(dryrun, expected)

    duplicated = dryrun.replace("copy_fixed_data              2", "copy_fixed_data              4")
    try:
        assert_no_fixed_data_duplication_in_dryrun(duplicated, expected)
    except AssertionError as exc:
        assert "copy_fixed_data" in str(exc)
    else:
        raise AssertionError("duplicated fixed-data dry-run count was not rejected")

    bad_path = dryrun + "\nresults/data/foo/n=None/seed=1/seed=1.csv\n"
    try:
        assert_no_fixed_data_duplication_in_dryrun(bad_path, expected)
    except AssertionError as exc:
        assert "duplicated fixed-data path" in str(exc)
    else:
        raise AssertionError("duplicated fixed-data path was not rejected")


def test_validation_tiny_notreks_paths_are_short(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run_dir = repo / "results/path-lengths"
    prepared = prepare_validation_run(repo, run_dir, "tiny")
    config = json.loads(prepared.validation_config_path.read_text())
    resources = config["resources"]["structure_learning_algorithms"]
    for entry in resources["notreks"]:
        path = f"algorithm=/notreks/alg_params=/alg_id={entry['alg_id']}/seed=1/adjmat.csv"
        assert len(path) < 80
        assert "function_class=" not in path
        assert "independence_cache_dir=" not in path
        assert all(len(component) < 200 for component in path.split("/"))


def test_fixed_adjmat_rule_cannot_match_resource_inputs() -> None:
    rule_path = Path(__file__).resolve().parents[3] / "graph/fixed_graph/rule.smk"
    text = rule_path.read_text()
    assert "wildcard_constraints:" in text
    assert 'output_dir="results"' in text


def test_prepare_validation_local10_avoids_logdet_power_iter_duplicates(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run_dir = repo / "results/local10"
    prepare_validation_run(repo, run_dir, "local10")
    config = json.loads((run_dir / "configs/validation_hparam_config.json").read_text())
    manifest = pd.read_csv(run_dir / "configs/validation_hparam_manifest.csv")
    full = [json.loads(value) for value in manifest.loc[manifest["base_method"] == "notreks", "hyperparameters_json"]]
    logdet = [entry for entry in full if entry["dag_seq"] == "logdet"]
    scc = [entry for entry in full if entry["dag_seq"] == "scc_power_iteration"]
    assert len(logdet) == 36
    assert len(scc) == 72
    logdet_unique = {
        (
            entry["independence_test"],
            entry["max_iter"],
            entry["trek_reg"],
            entry["regularizer_scale"],
        )
        for entry in logdet
    }
    assert len(logdet_unique) == len(logdet)


def test_select_by_method_family_and_write_final_config(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run_dir = repo / "results/select-family"
    prepared = prepare_validation_run(repo, run_dir, "tiny")
    joint = prepared.validation_joint_benchmarks_path
    joint.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"id": "gcastle_pc__grid000", "SHD_cpdag": 4, "FPR_skel": 0.1},
            {"id": "gcastle_lingam__grid000", "SHD_cpdag": 5, "FPR_skel": 0.2},
            {"id": "notreks__grid000", "SHD_cpdag": 9, "FPR_skel": 0.3},
            {"id": "notreks__grid001", "SHD_cpdag": 3, "FPR_skel": 0.4},
        ]
    ).to_csv(joint, index=False)

    selected = select_best_by_method_family(run_dir, repo, primary_metric="SHD_cpdag")
    assert selected["notreks"]["selected_algorithm_id"] == "notreks__grid001"
    assert (run_dir / "selection/best_by_method_family.json").is_file()
    assert (run_dir / "selection/validation_summary.csv").is_file()

    final_path = write_final_benchmark_config(repo, run_dir)
    final_config = json.loads(final_path.read_text())
    resources = final_config["resources"]["structure_learning_algorithms"]
    assert len(resources["gcastle_pc"]) == 1
    assert len(resources["gcastle_direct_lingam"]) == 1
    assert resources["notreks"][0]["id"] == "notreks__grid001"
    assert resources["notreks"][0]["alg_id"] == "n001"
    assert "function_class" not in resources["notreks"][0]
    final_manifest = json.loads((run_dir / "configs/final_hparam_manifest.json").read_text())
    assert final_manifest[0]["hyperparameters"]["id"] == "notreks__grid001"
    assert len(final_config["benchmark_setup"][0]["data"]) == 2
    assert final_config["benchmark_setup"][0]["data"][0]["data_id"].endswith("s201.csv")


def test_build_selected_benchmark_uses_fresh_data_and_selected_methods(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    selection_path = repo / "configs/notreks/selected/smoke_best_by_method_family.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(
        json.dumps(
            {
                "gcastle_pc": {
                    "selected_algorithm_id": "gcastle_pc__grid000",
                    "primary_metric_column": "SHD_cpdag",
                    "primary_mean": 2.0,
                    "config": {
                        "id": "gcastle_pc__grid000",
                        "variant": "stable",
                        "alpha": 0.05,
                        "ci_test": "fisherz",
                        "timeout": None,
                    },
                },
                "gcastle_lingam": {
                    "selected_algorithm_id": "gcastle_lingam__grid000",
                    "primary_metric_column": "SHD_cpdag",
                    "primary_mean": 3.0,
                    "config": {
                        "id": "gcastle_lingam__grid000",
                        "measure": "pwling",
                        "thresh": 0.3,
                        "timeout": None,
                    },
                },
                "notreks": {
                    "selected_algorithm_id": "notreks__grid001",
                    "primary_metric_column": "SHD_cpdag",
                    "primary_mean": 1.0,
                    "config": {
                        "id": "notreks__grid001",
                        "function_class": "linear",
                        "score": "least_squares",
                        "dag_seq": "logdet",
                        "dag_reg": 1.0,
                        "dag_s": 1.0,
                        "trek_seq": "exp",
                        "trek_reg": 1.0,
                        "regularizer": "l1",
                        "regularizer_scale": 0.01,
                        "independence_test": "spearman",
                        "independence_alpha": 0.05,
                        "independence_correction": "benjamini-hochberg",
                        "seed": 1,
                        "max_iter": 3000,
                        "lr": 0.0003,
                        "path_steps": 5,
                        "mu_init": 1.0,
                        "mu_factor": 0.1,
                        "warm_iter": 3000,
                        "tol": 1e-6,
                        "threshold": 0.1,
                        "timeout": None,
                        "init": "zero",
                        "checkpoint": 300,
                        "power_iter_steps": 5,
                        "scc_threshold": 1e-8,
                        "independence_cache_dir": "results/notreks_cache/notreks_smoke",
                    },
                },
            }
        )
        + "\n"
    )
    frame_path = repo / "configs/notreks/benchmarks/smoke_benchmark_frame.json"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_text(
        json.dumps(
            {
                "benchmark_id": "notreks_selected_smoke",
                "benchmark_name": "notreks_selected_smoke",
                "filename_prefix": "notreks/benchmark_smoke/",
                "data": {
                    "type": "synthetic_fresh",
                    "seeds": [201, 202],
                    "settings": [
                        {
                            "name": "er_d10_n500_deg2",
                            "graph": "er",
                            "d": 10,
                            "n": 500,
                            "expected_degree": 2,
                        }
                    ],
                },
                "selection": {
                    "include_method_families": ["gcastle_pc", "gcastle_lingam", "notreks"]
                },
            }
        )
        + "\n"
    )
    out_config = repo / "configs/notreks/expanded/selected_smoke_benchmark_config.json"
    out_manifest = repo / "configs/notreks/expanded/selected_smoke_benchmark_manifest.csv"
    built = build_selected_benchmark_config(repo, frame_path, selection_path, out_config, out_manifest)
    assert built.dataset_count == 2
    assert built.algorithm_counts == {"gcastle_pc": 1, "gcastle_lingam": 1, "notreks": 1}

    config = json.loads(out_config.read_text())
    setup = config["benchmark_setup"][0]
    assert len(setup["data"]) == 2
    assert all("s20" in row["data_id"] for row in setup["data"])
    assert not any("s101" in row["data_id"] or "s102" in row["data_id"] for row in setup["data"])
    resources = config["resources"]["structure_learning_algorithms"]
    assert len(resources["gcastle_pc"]) == 1
    assert len(resources["gcastle_direct_lingam"]) == 1
    assert resources["notreks"] == [
        {
            "id": "notreks__selected",
            "alg_id": "nsel",
            "params_manifest": "configs/notreks/expanded/selected_smoke_benchmark_manifest.json",
        }
    ]
    manifest = pd.read_csv(out_manifest)
    assert set(manifest["method_family"]) == {"gcastle_pc", "gcastle_lingam", "notreks"}
    assert set(manifest["source_grid_algorithm_id"]) == {
        "gcastle_pc__grid000",
        "gcastle_lingam__grid000",
        "notreks__grid001",
    }
    assert set(manifest["phase"]) == {"selected_benchmark"}
