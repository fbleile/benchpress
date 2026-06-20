from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from jobfarm import MANIFEST_FIELDS
from selection import inject_best, select_best


def _write_manifest(run_dir: Path, joint_name: str = "jobs/000_method/joint_benchmarks.csv") -> None:
    manifest_row = {
        "job_id": 0,
        "template_id": "method",
        "relative_config_path": "expanded_configs/method.json",
        "algorithm_ids": "method__grid000;method__grid001",
        "benchmark_title": "test",
        "filename_prefix": "test/",
        "relative_job_dir": "jobs/000_method",
        "relative_status_path": "jobs/000_method/status.json",
        "relative_stdout_path": "jobs/000_method/stdout.log",
        "relative_stderr_path": "jobs/000_method/stderr.log",
        "relative_joint_benchmarks_path": joint_name,
        "relative_roc_data_path": "jobs/000_method/ROC_data.csv",
        "benchpress_joint_benchmarks_path": "results/output/test/benchmarks/test/joint_benchmarks.csv",
        "benchpress_roc_data_path": "results/output/test/benchmarks/test/ROC_data.csv",
    }
    with (run_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(manifest_row)


def _write_index(run_dir: Path, ids: tuple[str, str] = ("method__grid000", "method__grid001")) -> None:
    (run_dir / "grid_index.json").write_text(
        json.dumps(
            [
                {"template_id": "method", "algorithm_id": ids[0], "config": {"id": ids[0], "trek_reg": 0.1}},
                {"template_id": "method", "algorithm_id": ids[1], "config": {"id": ids[1], "trek_reg": 1.0}},
            ]
        )
    )


def _write_selection_run(tmp_path: Path, include_metric: bool = True) -> Path:
    run_dir = tmp_path / "run"
    joint = run_dir / "jobs/000_method/joint_benchmarks.csv"
    joint.parent.mkdir(parents=True)
    data = {
        "id": ["method__grid000", "method__grid000", "method__grid001", "method__grid001"],
        "SHD_pattern": [3, 4, 1, 2],
        "TPR_skel_mean": [0.7, 0.7, 0.3, 0.3],
    }
    if include_metric:
        data["SHD_cpdag"] = [5, 7, 2, 4]
    pd.DataFrame(data).to_csv(joint, index=False)
    _write_manifest(run_dir)
    _write_index(run_dir)
    return run_dir


def test_selection_chooses_lowest_mean_cpdag_shd(tmp_path: Path) -> None:
    run_dir = _write_selection_run(tmp_path)
    output = run_dir / "summaries/selected_best.json"
    selected = select_best(run_dir, "shd_cpdag", None, None, None, output)
    assert selected["method"]["selected_algorithm_id"] == "method__grid001"
    assert selected["method"]["primary_mean"] == 3.0
    assert output.with_suffix(".csv").is_file()


def test_selection_fails_without_cpdag_metric(tmp_path: Path) -> None:
    run_dir = _write_selection_run(tmp_path, include_metric=False)
    try:
        select_best(run_dir, "shd_cpdag", None, None, None, run_dir / "selected.json")
    except ValueError as exc:
        assert "Available columns" in str(exc)
        assert "SHD_pattern" in str(exc)
    else:
        raise AssertionError("Expected missing CPDAG metric to fail")


def test_selection_uses_secondary_tie_breaker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    joint = run_dir / "jobs/000_method/joint_benchmarks.csv"
    joint.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "SHD_cpdag": [1, 1],
            "TPR_skel_mean": [0.2, 0.8],
        }
    ).to_csv(joint, index=False)
    _write_manifest(run_dir)
    _write_index(run_dir, ("a", "b"))
    selected = select_best(
        run_dir,
        "SHD_cpdag",
        "min",
        "TPR_skel_mean",
        "max",
        run_dir / "selected.json",
    )
    assert selected["method"]["selected_algorithm_id"] == "b"


def test_selection_works_after_moving_run_folder(tmp_path: Path) -> None:
    original = _write_selection_run(tmp_path / "original")
    moved = tmp_path / "copied-run"
    import shutil

    shutil.copytree(original, moved)
    selected = select_best(moved, "SHD_cpdag", None, None, None, moved / "selected.json")
    assert selected["method"]["selected_algorithm_id"] == "method__grid001"


def test_inject_best_preserves_non_notreks_algorithms(tmp_path: Path) -> None:
    benchmark = {
        "benchmark_setup": [
            {
                "title": "final",
                "data": [],
                "evaluation": {
                    "benchmarks": {"ids": ["old-notreks", "pc"]},
                    "graph_plots": ["old-notreks", "pc"],
                },
            }
        ],
        "resources": {
            "data": {},
            "graph": {},
            "parameters": {},
            "structure_learning_algorithms": {
                "notreks": [{"id": "old-notreks"}],
                "pc": [{"id": "pc"}],
            },
        },
    }
    selected = {
        "method": {
            "selected_algorithm_id": "new-notreks",
            "primary_metric": "shd_cpdag",
            "primary_mean": 1.0,
            "config": {"id": "new-notreks", "trek_reg": 1.0},
        }
    }
    benchmark_path = tmp_path / "benchmark.json"
    selected_path = tmp_path / "selected.json"
    output_path = tmp_path / "final.json"
    benchmark_path.write_text(json.dumps(benchmark))
    selected_path.write_text(json.dumps(selected))
    injected = inject_best(benchmark_path, selected_path, output_path)
    algorithms = injected["resources"]["structure_learning_algorithms"]
    assert algorithms["pc"] == [{"id": "pc"}]
    assert algorithms["notreks"] == [{"id": "new-notreks", "trek_reg": 1.0}]
    ids = injected["benchmark_setup"][0]["evaluation"]["benchmarks"]["ids"]
    assert ids == ["pc", "new-notreks"]
