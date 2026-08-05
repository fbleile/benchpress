"""Single-allocation Benchpress farm for NOTREKS scenario configurations.

The farm owns orchestration only.  Snakemake outputs remain the source of
truth; a task is successful only after its expected Benchpress output exists.
Status files are atomically replaced so interruption cannot create a false
completion marker.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def choose_container_flag(snakemake: str | None = None) -> tuple[str, str]:
    binary = snakemake or shutil.which("snakemake")
    if not binary:
        raise RuntimeError("snakemake was not found; activate benchpress-notreks")
    help_text = subprocess.run(
        [binary, "--help"], capture_output=True, text=True, check=True
    ).stdout
    if "--use-apptainer" in help_text:
        flag = "--use-apptainer"
    elif "--use-singularity" in help_text:
        flag = "--use-singularity"
    else:
        raise RuntimeError("Snakemake supports neither --use-apptainer nor --use-singularity")
    executable = shutil.which("apptainer" if flag == "--use-apptainer" else "singularity")
    if executable is None and shutil.which("apptainer") is None:
        raise RuntimeError(f"{flag} is supported, but no Apptainer/Singularity executable is available")
    return binary, flag


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def task_rows(config: Path, manifest: Path | None) -> list[dict[str, str]]:
    if manifest is None:
        return [{"task_id": "0", "config_path": str(config), "expected_output": ""}]
    rows = _read_csv(manifest)
    if not rows:
        raise ValueError(f"Task manifest is empty: {manifest}")
    result = []
    for index, row in enumerate(rows):
        task_id = str(row.get("job_id", row.get("task_id", row.get("id", index))))
        config_path = (
            row.get("config_path")
            or row.get("relative_config_path")
            or row.get("config")
        )
        if not config_path:
            raise ValueError(f"Manifest row {task_id} has no config path")
        expected = row.get("benchpress_joint_benchmarks_path", "")
        result.append({"task_id": task_id, "config_path": config_path, "expected_output": expected})
    if len({row["task_id"] for row in result}) != len(result):
        raise ValueError("Task IDs must be unique")
    return result


def validate_cpdag_config(config: Path) -> None:
    """Require the Benchpress-native one-time DAG->CPDAG conversion."""
    value = json.loads(config.read_text())
    setups = value.get("benchmark_setup", [])
    if not setups:
        raise ValueError("config has no benchmark_setup entries")
    for setup in setups:
        conversion = setup.get("evaluation", {}).get("graph_estimation", {}).get("convert_to", [])
        if "cpdag" not in conversion:
            raise ValueError(
                "config must request graph_estimation.convert_to=['cpdag']; "
                "regenerate it with the NOTREKS compiler"
            )


def _path(repo: Path, run_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = repo / path
    return candidate if candidate.exists() else run_dir / path


def classify(config_path: Path, explicit: str | None) -> tuple[str, int]:
    classes = {"smoke": 600, "short": 1800, "medium": 7200, "long": 18000}
    if explicit:
        if explicit not in classes:
            raise ValueError(f"Unknown resource class {explicit!r}; choose {sorted(classes)}")
        return explicit, classes[explicit]
    try:
        config = json.loads(config_path.read_text())
        text = json.dumps(config)
        dims = [int(x) for x in re.findall(r'"d"\s*:\s*(\d+)', text)]
        d = max(dims, default=20)
    except (OSError, ValueError, json.JSONDecodeError):
        d = 20
    if d <= 20:
        return "smoke", classes["smoke"]
    if d <= 50:
        return "short", classes["short"]
    if d <= 100:
        return "medium", classes["medium"]
    return "long", classes["long"]


def _status_path(run_dir: Path, task_id: str) -> Path:
    return run_dir / "tasks" / str(task_id) / "status.json"


def _task_status(run_dir: Path, row: dict[str, str]) -> dict | None:
    path = _status_path(run_dir, row["task_id"])
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def run_task(
    repo: Path,
    run_dir: Path,
    row: dict[str, str],
    snakemake: str,
    container_flag: str,
    timeout: int,
    resource_class: str,
    force: bool = False,
) -> dict:
    task_id = row["task_id"]
    task_dir = run_dir / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    status_path = task_dir / "status.json"
    old = _task_status(run_dir, row)
    expected = _path(repo, run_dir, row.get("expected_output", "")) if row.get("expected_output") else None
    if not force and old and old.get("status") == "success" and (expected is None or expected.is_file()):
        return old
    config = _path(repo, run_dir, row["config_path"])
    stdout_path, stderr_path = task_dir / "stdout.log", task_dir / "stderr.log"
    start = time.time()
    running = {"task_id": task_id, "config_path": str(config), "status": "running",
               "resource_class": resource_class, "output_graph_type": "cpdag",
               "start_time": now()}
    atomic_json(status_path, running)
    command = [
        snakemake, "--cores", "1", container_flag, "--nolock", "--rerun-incomplete",
        "--snakefile", "workflow/Snakefile", "--configfile", str(config),
    ]
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    result = dict(running, command=command, timeout_seconds=timeout)
    try:
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            completed = subprocess.run(command, cwd=repo, env=env, stdout=out, stderr=err,
                                       text=True, timeout=timeout)
        if completed.returncode != 0:
            result.update(status="failed", exit_code=completed.returncode)
        elif expected is not None and not expected.is_file():
            result.update(status="failed", exit_code=0,
                          exception=f"expected output missing: {expected}")
        else:
            result.update(status="success", exit_code=0)
    except subprocess.TimeoutExpired:
        result.update(status="TIMEOUT", exit_code=None, exception="task timeout")
    except Exception as exc:  # pragma: no cover - defensive scheduler path
        result.update(status="failed", exit_code=None, exception=f"{type(exc).__name__}: {exc}")
    result.update(end_time=now(), elapsed_seconds=time.time() - start)
    atomic_json(status_path, result)
    return result


def run_farm(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    run_dir = Path(args.run_dir).resolve()
    config = _path(repo, run_dir, args.config)
    if not config.is_file():
        raise FileNotFoundError(f"config does not exist: {config}")
    validate_cpdag_config(config)
    manifest = Path(args.manifest).resolve() if args.manifest else None
    rows = task_rows(config, manifest)
    smk, flag = choose_container_flag(args.snakemake)
    # Snakemake 7 calls the container option ``--use-singularity``.  LRZ
    # installations commonly provide only the Apptainer executable, so use a
    # run-local compatibility shim rather than requiring a cluster-wide alias.
    if flag == "--use-singularity" and shutil.which("singularity") is None:
        shim_dir = run_dir / "bin"
        shim_dir.mkdir(parents=True, exist_ok=True)
        os.symlink(shutil.which("apptainer"), shim_dir / "singularity") if not (shim_dir / "singularity").exists() else None
        os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    workers = max(1, min(int(workers), int(args.max_workers or workers)))
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "farm_config.json", {
        "config": str(config), "manifest": str(manifest) if manifest else None,
        "workers": workers, "container_flag": flag, "snakemake": smk,
        "resource_class": args.resource_class or "auto", "task_count": len(rows),
        "output_graph_type": "cpdag",
    })
    if args.dry_run:
        print(f"dry-run: planned_tasks={len(rows)} workers={workers} per_task_cores=1")
        for row in rows[:5]:
            print(f"  task={row['task_id']} config={row['config_path']}")
        if len(rows) > 5:
            print(f"  ... {len(rows) - 5} more tasks")
        return 0
    pending = []
    for row in rows:
        status = _task_status(run_dir, row)
        expected = _path(repo, run_dir, row.get("expected_output", "")) if row.get("expected_output") else None
        if not args.force and status and status.get("status") == "success" and (expected is None or expected.is_file()):
            continue
        pending.append(row)
    print(f"tasks={len(rows)} pending={len(pending)} workers={workers} container_flag={flag}")
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_task, repo, run_dir, row, smk, flag,
                classify(_path(repo, run_dir, row["config_path"]), args.resource_class)[1],
                classify(_path(repo, run_dir, row["config_path"]), args.resource_class)[0],
                args.force
            ): row
            for row in pending
        }
        for future in as_completed(futures):
            result = future.result()
            print(f"task={result['task_id']} status={result['status']} elapsed={result.get('elapsed_seconds', 0):.1f}s", flush=True)
            if result["status"] != "success":
                failures.append(result)
    statuses = [_task_status(run_dir, row) or {"status": "missing", "task_id": row["task_id"]} for row in rows]
    atomic_json(run_dir / "status.json", {"updated_at": now(), "tasks": statuses})
    if failures or any(item.get("status") != "success" for item in statuses):
        atomic_json(run_dir / "analysis_status.json", {"status": "blocked", "updated_at": now()})
        print(f"{len(failures)} task(s) failed or timed out; analysis not run", file=sys.stderr)
        return 1
    if args.analysis_command:
        analysis_env = os.environ.copy()
        subprocess.run(args.analysis_command, cwd=repo, env=analysis_env, shell=True, check=True)
    atomic_json(run_dir / "analysis_status.json", {"status": "success", "updated_at": now()})
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--resource-class", choices=["smoke", "short", "medium", "long"])
    parser.add_argument("--snakemake")
    parser.add_argument("--analysis-command")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_farm(args))


if __name__ == "__main__":
    main()
