from __future__ import annotations

import os
import subprocess
from pathlib import Path

from grid import ExpandedTemplate
from jobfarm import make_manifest, read_manifest, write_command_file
from cli import _write_validation_cmd


MODULE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_DIR.parents[3]


def _records(tmp_path: Path) -> list[ExpandedTemplate]:
    return [
        ExpandedTemplate(
            template_id=f"method-{index}",
            relative_config_path=f"expanded_configs/method-{index}.json",
            algorithm_ids=(f"method-{index}__grid000",),
            benchmark_title=f"benchmark-{index}",
            filename_prefix=f"run/method-{index}/",
            relative_job_dir=f"jobs/{index:03d}_method-{index}",
            relative_joint_benchmarks_path=f"jobs/{index:03d}_method-{index}/joint_benchmarks.csv",
            relative_roc_data_path=f"jobs/{index:03d}_method-{index}/ROC_data.csv",
            benchpress_joint_benchmarks_path=f"results/output/benchmark-{index}/benchmarks/run/method-{index}/joint_benchmarks.csv",
            benchpress_roc_data_path=f"results/output/benchmark-{index}/benchmarks/run/method-{index}/ROC_data.csv",
        )
        for index in range(2)
    ]


def test_manifest_and_command_file_are_unique_and_nonempty(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, _records(tmp_path))
    rows = read_manifest(manifest)
    assert len({row["job_id"] for row in rows}) == len(rows)
    assert len({row["relative_config_path"] for row in rows}) == len(rows)
    assert len({row["relative_status_path"] for row in rows}) == len(rows)
    assert len({row["relative_job_dir"] for row in rows}) == len(rows)
    command_file = write_command_file(REPO_ROOT, tmp_path, manifest)
    assert command_file.stat().st_size > 0
    commands = command_file.read_text().strip().splitlines()
    assert len(commands) == 2
    assert all("tools/cli.py run-config --run-dir" in command for command in commands)


def test_validation_command_file_references_generated_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = run_dir / "configs/validation_hparam_config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n")
    command_file = _write_validation_cmd(run_dir, config)
    text = command_file.read_text()
    commands = text.strip().splitlines()
    assert command_file == run_dir / "cmd.txt"
    assert command_file.stat().st_size > 0
    assert len(commands) == 1
    assert commands[0].startswith('cd "')
    assert "&&" in commands[0]
    assert "snakemake" in text
    assert "--use-apptainer" in text
    assert "--snakefile workflow/Snakefile" in text
    assert "configs/validation_hparam_config.json" in text


def test_slurm_script_rejects_missing_command_file(tmp_path: Path) -> None:
    script = MODULE_DIR / "slurm/notreks_jobfarm.sh"
    run_dir = tmp_path / "run"
    real_smoke_log = REPO_ROOT / "results/notreks_experiments/slurm_smoke/logs/slurm/jobfarm.local.out"
    before = real_smoke_log.read_text() if real_smoke_log.exists() else None
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "REPO_DIR": str(REPO_ROOT),
            "RUN_DIR": str(run_dir),
            "CMD_FILE": str(tmp_path / "missing.txt"),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "CMD_FILE is missing or empty" in completed.stderr
    after = real_smoke_log.read_text() if real_smoke_log.exists() else None
    assert after == before


def test_slurm_script_defaults_to_run_dir_command_file() -> None:
    script = (MODULE_DIR / "slurm/notreks_jobfarm.sh").read_text()
    assert 'CMD_FILE="${CMD_FILE:-${RUN_DIR}/cmd.txt}"' in script
