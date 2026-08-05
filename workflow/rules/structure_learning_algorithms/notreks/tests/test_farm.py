from __future__ import annotations

import csv
import os
from pathlib import Path

from workflow.rules.structure_learning_algorithms.notreks.tools.farm import (
    choose_container_flag,
    run_task,
    task_rows,
    validate_cpdag_config,
    validate_required_files,
)


def _config(path: Path) -> None:
    path.write_text(
        '{"benchmark_setup": [{"evaluation": '
        '{"graph_estimation": {"convert_to": ["cpdag"]}}}]}\n'
    )


def test_360_tasks_are_farm_tasks_not_slurm_array(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["job_id", "config_path"])
        writer.writeheader()
        for index in range(360):
            writer.writerow({"job_id": index, "config_path": f"configs/{index}.json"})
    rows = task_rows(tmp_path / "config.json", manifest)
    assert len(rows) == 360
    assert not any("sbatch" in str(row) for row in rows)


def test_container_flag_detects_legacy_snakemake(tmp_path: Path, monkeypatch) -> None:
    snakemake = tmp_path / "snakemake"
    snakemake.write_text("#!/bin/sh\necho 'Usage: --use-singularity'\n")
    singularity = tmp_path / "singularity"
    singularity.write_text("#!/bin/sh\n")
    snakemake.chmod(0o755)
    singularity.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert choose_container_flag(str(snakemake)) == (str(snakemake), "--use-singularity")


def test_cpdag_config_is_required(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "config.json"
    _config(config)
    validate_cpdag_config(config)
    config.write_text('{"benchmark_setup": [{"evaluation": {}}]}\n')
    try:
        validate_cpdag_config(config)
    except ValueError as exc:
        assert "convert_to" in str(exc)
    else:
        raise AssertionError("missing CPDAG conversion was accepted")


def test_missing_required_schema_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    module = repo / "workflow/rules/structure_learning_algorithms/example"
    module.mkdir(parents=True)
    (module / "rule.smk").write_text("")
    config = tmp_path / "config.json"
    config.write_text(
        '{"resources": {"structure_learning_algorithms": {"example": []}}}'
    )
    try:
        validate_required_files(repo, config)
    except FileNotFoundError as exc:
        assert "required Benchpress schema is missing" in str(exc)
    else:
        raise AssertionError("missing schema was accepted")


def test_required_schema_and_rule_are_accepted(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    module = repo / "workflow/rules/structure_learning_algorithms/example"
    module.mkdir(parents=True)
    (module / "rule.smk").write_text("")
    (module / "schema.json").write_text("{}")
    config = tmp_path / "config.json"
    config.write_text(
        '{"resources": {"structure_learning_algorithms": {"example": []}}}'
    )
    validate_required_files(repo, config)


def test_task_timeout_is_recorded_and_resume_skips_success(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake = tmp_path / "snakemake"
    fake.write_text("#!/bin/sh\nsleep 0.05\n")
    fake.chmod(0o755)
    row = {"task_id": "0", "config_path": "config.json", "expected_output": ""}
    config = tmp_path / "config.json"
    _config(config)
    first = run_task(tmp_path, tmp_path / "run", row, str(fake), "--use-singularity", 1, "smoke")
    assert first["status"] == "success"
    second = run_task(tmp_path, tmp_path / "run", row, str(fake), "--use-singularity", 1, "smoke")
    assert second["status"] == "success"
    assert second["start_time"] == first["start_time"]
