from __future__ import annotations

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
    assert "--use-singularity" in text
    assert "--snakefile workflow/Snakefile" in text
    assert "configs/validation_hparam_config.json" in text


def test_snakemake_driver_rejects_missing_config() -> None:
    script = MODULE_DIR / "slurm/notreks_snakemake_driver_common.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "CONFIG must point to a Benchpress config file" in completed.stderr


def test_slurm_resource_presets_are_lrz_sized() -> None:
    smoke = (MODULE_DIR / "slurm/notreks_driver_serial_smoke.sh").read_text()
    serial_true = (MODULE_DIR / "slurm/notreks_driver_serial_true.sh").read_text()
    cm4_true = (MODULE_DIR / "slurm/notreks_driver_cm4_tiny_true.sh").read_text()
    compat = (MODULE_DIR / "slurm/notreks_jobfarm.sh").read_text()
    all_scripts = "\n".join([smoke, serial_true, cm4_true, compat])

    assert "#SBATCH --partition=serial_std" in smoke
    assert "#SBATCH --clusters=serial" in smoke
    assert "#SBATCH --cpus-per-task=8" in smoke
    assert "#SBATCH --partition=serial_std" in serial_true
    assert "#SBATCH --cpus-per-task=16" in serial_true
    assert "#SBATCH --partition=cm4_tiny" in cm4_true
    assert "#SBATCH --cpus-per-task=32" in cm4_true
    assert "#SBATCH --nodes=2" not in all_scripts
    assert "#SBATCH --ntasks=200" not in all_scripts
    assert "cm4_std" not in all_scripts


def test_snakemake_driver_runs_one_snakemake_process() -> None:
    script = (MODULE_DIR / "slurm/notreks_snakemake_driver_common.sh").read_text()
    assert script.count("\nsnakemake \\") == 1
    assert '"$CONTAINER_FLAG"' in script
    assert 'CONTAINER_FLAG="--use-apptainer"' in script
    assert 'CONTAINER_FLAG="--use-singularity"' in script
    assert 'SMK_MAJOR="${SMK_VER%%.*}"' in script
    assert 'if [[ "$SMK_MAJOR" -ge 8 ]]' in script
    assert "--use-apptainer \\" not in script
    assert "--configfile \"$CONFIG\"" in script
    assert "module load \"$APPTAINER_MODULE\"" in script
    assert 'SQUASHFS_MODULE="${SQUASHFS_MODULE:-squashfs/4.6.1}"' in script
    assert "module load \"$SQUASHFS_MODULE\"" in script
    assert "command -v mksquashfs" in script
    assert "Apptainer cannot convert Docker images to SIF" in script
    assert "Try manually: module load $SQUASHFS_MODULE; which mksquashfs" in script
    assert "SMK_VER=\"$(snakemake --version)\"" in script
    assert "Snakemake 9 is incompatible with Python 3.7 Benchpress gCastle containers" in script
    assert 'SHIM_DIR="${RUN_DIR}/bin"' in script
    assert 'ln -sf "$(command -v apptainer)" "${SHIM_DIR}/singularity"' in script
    assert 'export PATH="${PWD}/${SHIM_DIR}:$PATH"' in script
    assert "singularity --version || true" in script
    assert "micromamba activate \"$CONDA_ENV\"" in script
    assert "SCRIPT_DIR=${SCRIPT_DIR:-}" in script
    assert "REPO_ROOT=${REPO_ROOT:-}" in script


def test_environment_pins_snakemake_seven_for_gcastle_containers() -> None:
    for env_file in (
        MODULE_DIR / "configs/environment/benchpress-notreks.yml",
        MODULE_DIR / "configs/environment/benchpress-notreks-minimal.yml",
    ):
        text = env_file.read_text()
        assert "snakemake=7.32.4" in text
        assert "Snakemake 9" in text


def test_driver_wrappers_find_common_driver_from_spool_dir(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for script_name in (
        "notreks_driver_serial_smoke.sh",
        "notreks_driver_serial_true.sh",
        "notreks_driver_cm4_tiny_true.sh",
    ):
        copied_script = tmp_path / script_name
        copied_script.write_text((MODULE_DIR / "slurm" / script_name).read_text())
        completed = subprocess.run(
            ["bash", str(copied_script)],
            cwd=tmp_path,
            env={
                "REPO_DIR": str(REPO_ROOT),
                "PATH": "/usr/bin:/bin",
                "HOME": str(Path.home()),
            },
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2
        assert "CONFIG must point to a Benchpress config file" in completed.stderr
        assert "common NOTREKS driver not found" not in completed.stderr


def test_driver_wrappers_do_not_use_bare_common_driver_source() -> None:
    for script_name in (
        "notreks_driver_serial_smoke.sh",
        "notreks_driver_serial_true.sh",
        "notreks_driver_cm4_tiny_true.sh",
        "notreks_jobfarm.sh",
    ):
        script = (MODULE_DIR / "slurm" / script_name).read_text()
        assert "source notreks_snakemake_driver_common.sh" not in script
        assert 'source "$COMMON_DRIVER"' in script
        assert 'COMMON_DRIVER="${SCRIPT_DIR}/notreks_snakemake_driver_common.sh"' in script


def test_print_slurm_launch_uses_run_logs_and_driver(tmp_path: Path) -> None:
    run_dir = tmp_path / "slurm_smoke"
    completed = subprocess.run(
        [
            "python",
            str(MODULE_DIR / "tools/cli.py"),
            "print-slurm-launch",
            "--run-dir",
            str(run_dir),
            "--preset",
            "smoke",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = completed.stdout
    assert f"RUN_DIR={run_dir}" in output
    assert f"CONFIG={run_dir}/configs/validation_hparam_config.json" in output
    assert "SNAKEMAKE_CORES=8" in output
    assert "sbatch --clusters=serial" in output
    assert f"-o {run_dir}/logs/slurm/%x-%j.out" in output
    assert f"-e {run_dir}/logs/slurm/%x-%j.err" in output
    assert "notreks_driver_serial_smoke.sh" in output
