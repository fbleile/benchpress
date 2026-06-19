from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from jobfarm import MANIFEST_FIELDS
from selection import inject_best, select_best


def _write_selection_run(tmp_path: Path, include_metric: bool = True) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    joint = run_dir / "joint.csv"
    data = {
        "id": ["method__grid000", "method__grid000", "method__grid001", "method__grid001"],
        "SHD_pattern": [3, 4, 1, 2],
    }
    if include_metric:
        data["SHD_cpdag"] = [5, 7, 2, 4]
    pd.DataFrame(data).to_csv(joint, index=False)
    manifest_row = {
        "job_id": 0,
        "template_id": "method",
        "config_path": str(run_dir / "method.json"),
        "algorithm_ids": "method__grid000;method__grid001",
        "benchmark_title": "test",
        "filename_prefix": "test/",
        "joint_benchmarks_path": str(joint),
        "status_path": str(run_dir / "status.json"),
        "stdout_path": str(run_dir / "stdout"),
        "stderr_path": str(run_dir / "stderr"),
    }
    with (run_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(manifest_row)
    (run_dir / "grid_index.json").write_text(
        json.dumps(
            [
                {
                    "template_id": "method",
                    "algorithm_id": "method__grid000",
                    "config_path": "method.json",
                    "config": {"id": "method__grid000", "trek_reg": 0.1},
                },
                {
                    "template_id": "method",
                    "algorithm_id": "method__grid001",
                    "config_path": "method.json",
                    "config": {"id": "method__grid001", "trek_reg": 1.0},
                },
            ]
        )
    )
    return run_dir


def test_selection_chooses_lowest_mean_cpdag_shd(tmp_path: Path) -> None:
    run_dir = _write_selection_run(tmp_path)
    output = run_dir / "selected_best.json"
    selected = select_best(run_dir, "shd_cpdag", output)
    assert selected["method"]["selected_algorithm_id"] == "method__grid001"
    assert selected["method"]["mean_metric"] == 3.0
    assert output.with_suffix(".csv").is_file()


def test_selection_fails_without_cpdag_metric(tmp_path: Path) -> None:
    run_dir = _write_selection_run(tmp_path, include_metric=False)
    try:
        select_best(run_dir, "shd_cpdag", run_dir / "selected.json")
    except ValueError as exc:
        assert "Available columns" in str(exc)
        assert "SHD_pattern" in str(exc)
    else:
        raise AssertionError("Expected missing CPDAG metric to fail")


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
            "metric": "shd_cpdag",
            "mean_metric": 1.0,
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
