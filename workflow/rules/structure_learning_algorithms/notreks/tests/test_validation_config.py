from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from validation import (
    assert_no_fixed_data_duplication_in_dryrun,
    expected_algorithm_run_counts,
    parse_dryrun_job_counts,
    prepare_validation_run,
    select_best_by_method_family,
    write_final_benchmark_config,
)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "resources/data/mydatasets").mkdir(parents=True)
    (repo / "resources/adjmat/myadjmats").mkdir(parents=True)
    return repo


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
