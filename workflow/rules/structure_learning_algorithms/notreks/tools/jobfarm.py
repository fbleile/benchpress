"""Manifest, command-list, and Benchpress execution helpers."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from grid import ExpandedTemplate


MANIFEST_FIELDS = [
    "job_id",
    "template_id",
    "config_path",
    "algorithm_ids",
    "benchmark_title",
    "filename_prefix",
    "joint_benchmarks_path",
    "status_path",
    "stdout_path",
    "stderr_path",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_manifest(run_dir: Path, records: list[ExpandedTemplate]) -> Path:
    rows = []
    for job_id, record in enumerate(records):
        status_dir = run_dir / "status"
        log_dir = run_dir / "logs"
        filename = _safe_filename(record.template_id)
        rows.append(
            {
                "job_id": job_id,
                "template_id": record.template_id,
                "config_path": str(record.config_path),
                "algorithm_ids": ";".join(record.algorithm_ids),
                "benchmark_title": record.benchmark_title,
                "filename_prefix": record.filename_prefix,
                "joint_benchmarks_path": str(record.joint_benchmarks_path),
                "status_path": str(status_dir / f"{filename}.json"),
                "stdout_path": str(log_dir / f"{filename}.out"),
                "stderr_path": str(log_dir / f"{filename}.err"),
            }
        )
    validate_manifest(rows)
    path = run_dir / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        _write_json(
            Path(row["status_path"]),
            {
                "status": "pending",
                "job_id": row["job_id"],
                "template_id": row["template_id"],
                "start_time": None,
                "end_time": None,
                "runtime": None,
                "exception": None,
            },
        )
    return path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_manifest(rows)
    return rows


def validate_manifest(rows: list[dict]) -> None:
    job_ids = [int(row["job_id"]) for row in rows]
    config_paths = [str(row["config_path"]) for row in rows]
    status_paths = [str(row["status_path"]) for row in rows]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Manifest job_id values must be unique")
    if len(config_paths) != len(set(config_paths)):
        raise ValueError("Manifest config paths must be unique")
    if len(status_paths) != len(set(status_paths)):
        raise ValueError("Manifest status paths must be unique")


def write_command_file(repo_root: Path, run_dir: Path, manifest_path: Path) -> Path:
    cli = repo_root / "workflow/rules/structure_learning_algorithms/notreks/tools/cli.py"
    commands = []
    for row in read_manifest(manifest_path):
        commands.append(
            " ".join(
                [
                    "python",
                    shlex.quote(str(cli.relative_to(repo_root))),
                    "run-config",
                    "--config",
                    shlex.quote(row["config_path"]),
                    "--status",
                    shlex.quote(row["status_path"]),
                    "--stdout",
                    shlex.quote(row["stdout_path"]),
                    "--stderr",
                    shlex.quote(row["stderr_path"]),
                ]
            )
        )
    path = run_dir / "cmd.txt"
    path.write_text("\n".join(commands) + "\n")
    if path.stat().st_size == 0:
        raise ValueError("Generated command file is empty")
    return path


def _snakemake_binary() -> str:
    configured = os.environ.get("SNAKEMAKE_BIN")
    if configured:
        return configured
    found = shutil.which("snakemake")
    if found:
        return found
    fallback = Path("/opt/anaconda3/envs/benchpress-notreks/bin/snakemake")
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError(
        "snakemake was not found. Activate benchpress-notreks or set SNAKEMAKE_BIN."
    )


def run_benchpress_config(
    repo_root: Path,
    config_path: Path,
    status_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    cores: int = 1,
) -> None:
    status = {
        "status": "running",
        "config_path": str(config_path),
        "start_time": _now(),
        "end_time": None,
        "runtime": None,
        "exception": None,
    }
    _write_json(status_path, status)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if platform.system() == "Darwin":
        env.setdefault("BENCHPRESS_SKIP_CONTAINER_CHECK", "1")
    command = [
        _snakemake_binary(),
        "--snakefile",
        "workflow/Snakefile",
        "--cores",
        str(cores),
        "--configfile",
        str(config_path),
    ]
    started = time.perf_counter()
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=True,
                text=True,
            )
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "end_time": _now(),
                "runtime": time.perf_counter() - started,
                "exception": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(status_path, status)
        raise
    status.update(
        {
            "status": "success",
            "end_time": _now(),
            "runtime": time.perf_counter() - started,
        }
    )
    _write_json(status_path, status)
