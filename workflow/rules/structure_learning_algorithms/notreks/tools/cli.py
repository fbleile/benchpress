#!/usr/bin/env python
"""Phase-based local and SLURM workflow for NOTREKS Benchpress runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TOOLS_DIR.parent
REPO_ROOT = MODULE_DIR.parents[3]
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(MODULE_DIR))

from grid import prepare_hparam_run  # noqa: E402
from independence_tests import pairwise_independence_candidates  # noqa: E402
from jobfarm import (  # noqa: E402
    make_manifest,
    read_manifest,
    run_benchpress_config,
    write_command_file,
)
from selection import inject_best, select_best  # noqa: E402
from validation import (  # noqa: E402
    build_selected_benchmark_config,
    expand_grid_config,
    prepare_validation_run,
    select_best_from_config_manifest,
    select_best_by_method_family,
    write_final_benchmark_config,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_run_path(run_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else run_dir / path


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _tag_grid_path(tag: str) -> Path:
    return Path("configs/notreks/grids") / f"{tag}_grid.json"


def _tag_config_path(tag: str) -> Path:
    return Path("configs/notreks/expanded") / f"{tag}_config.json"


def _tag_manifest_path(tag: str) -> Path:
    return Path("configs/notreks/expanded") / f"{tag}_manifest.csv"


def _tag_selected_dir() -> Path:
    return Path("configs/notreks/selected")


def _tag_benchmark_frame_path(tag: str) -> Path:
    base = Path("configs/notreks/benchmarks")
    preferred = base / f"{tag}_frame.json"
    if preferred.exists():
        return preferred
    return base / f"{tag}_benchmark_frame.json"


def _tag_selected_benchmark_config_path(tag: str) -> Path:
    if tag.endswith("_benchmark"):
        return Path("configs/notreks/expanded") / f"selected_{tag}_config.json"
    return Path("configs/notreks/expanded") / f"selected_{tag}_benchmark_config.json"


def _tag_selected_benchmark_manifest_path(tag: str) -> Path:
    if tag.endswith("_benchmark"):
        return Path("configs/notreks/expanded") / f"selected_{tag}_manifest.csv"
    return Path("configs/notreks/expanded") / f"selected_{tag}_benchmark_manifest.csv"


def _tag_selection_path(tag: str) -> Path:
    return _tag_selected_dir() / f"{tag}_best_by_method_family.json"


def _write_validation_cmd(run_dir: Path, validation_config: Path, *, cores: str = "1") -> Path:
    cmd_path = run_dir / "cmd.txt"
    config_arg = _repo_relative(validation_config)
    command = (
        f'cd "{REPO_ROOT}" && '
        'export PYTHONPATH="$PWD:${PYTHONPATH:-}" && '
        "snakemake "
        f"--cores {cores} "
        "--use-singularity "
        "--snakefile workflow/Snakefile "
        f"--configfile {config_arg}\n"
    )
    cmd_path.write_text(command)
    return cmd_path


def _smoke_meta(preset: str, run_name: str, dag_seq: str) -> dict:
    if preset == "tiny":
        methods = [
            ("exp-no-trek", "none", "none", 0.0),
            ("exp-notreks-spearman", "spearman", "exp", 1.0),
        ]
        max_iter = 500
        path_steps = 2
    elif preset == "small":
        methods = [
            ("exp-no-trek", "none", "none", 0.0),
            ("exp-notreks-spearman", "spearman", "exp", 1.0),
            ("exp-notreks-pearson", "pearson", "exp", 1.0),
            ("exp-notreks-dcor", "dcor", "exp", 1.0),
        ]
        max_iter = 1000
        path_steps = 3
    else:
        raise ValueError(f"Unknown smoke preset: {preset}")

    entries = []
    for method_id, independence_test, trek_seq, trek_reg in methods:
        entries.append(
            {
                "id": method_id,
                "function_class": "linear",
                "score": "least_squares",
                "dag_seq": dag_seq,
                "dag_reg": 1.0,
                "dag_s": 1.0,
                "trek_seq": trek_seq,
                "trek_reg": trek_reg,
                "regularizer": "l1",
                "regularizer_scale": 0.01,
                "independence_test": independence_test,
                "independence_alpha": 0.05,
                "independence_correction": "benjamini-hochberg",
                "seed": 1,
                "max_iter": max_iter,
                "lr": 0.0003,
                "path_steps": path_steps,
                "mu_init": 1.0,
                "mu_factor": 0.1,
                "warm_iter": max_iter,
                "tol": 1e-6,
                "threshold": 0.1,
                "timeout": None,
                "init": "zero",
                "checkpoint": max(max_iter // 2, 1),
            }
        )
    return {
        "notreks_hparam": {
            "run_name": run_name,
            "fixed_data": {
                "d": 10,
                "n_values": [500],
                "seeds": [1],
                "expected_degree": 2.0,
                "standardized": True,
            },
        },
        "structure_learning_algorithms": {"notreks": entries},
    }


def prepare_command(args: argparse.Namespace) -> tuple[Path, Path]:
    run_dir = _resolve(args.out)
    grid_config = _resolve(args.grid_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    records, fixed_reference = prepare_hparam_run(
        REPO_ROOT,
        grid_config,
        run_dir,
        args.grid_mode,
    )
    manifest = make_manifest(run_dir, records)
    command_file = write_command_file(REPO_ROOT, run_dir, manifest)
    print(f"Prepared {len(records)} template jobs")
    print(f"Shared fixed data: {fixed_reference.data_dir}")
    print(f"Manifest: {manifest}")
    print(f"Commands: {command_file}")
    return manifest, command_file


def smoke_command(args: argparse.Namespace) -> None:
    run_dir = _resolve(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "smoke_grid.json"
    meta_path.write_text(
        json.dumps(_smoke_meta(args.preset, f"notreks_smoke_{args.preset}", args.dag_seq), indent=2)
        + "\n"
    )
    prepare_args = argparse.Namespace(
        out=run_dir,
        grid_config=meta_path,
        grid_mode="cartesian",
    )
    manifest, _ = prepare_command(prepare_args)
    frames = []
    failures = []
    for row in read_manifest(manifest):
        try:
            run_benchpress_config(
                REPO_ROOT,
                run_dir,
                _resolve_run_path(run_dir, Path(row["relative_config_path"])),
                int(row["job_id"]),
                _resolve_run_path(run_dir, Path(row["relative_status_path"])),
                _resolve_run_path(run_dir, Path(row["relative_stdout_path"])),
                _resolve_run_path(run_dir, Path(row["relative_stderr_path"])),
                cores=1,
            )
            joint = _resolve_run_path(run_dir, Path(row["relative_joint_benchmarks_path"]))
            if joint.is_file():
                frames.append(pd.read_csv(joint))
        except Exception as exc:
            failures.append(
                {
                    "template_id": row["template_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary.to_csv(run_dir / "smoke_summary.csv", index=False)
    pd.DataFrame(failures, columns=["template_id", "error"]).to_csv(
        run_dir / "failures.csv", index=False
    )
    display = [
        column
        for column in ("id", "SHD_cpdag", "SHD_pattern", "time", "ntests")
        if column in summary.columns
    ]
    print(summary[display].to_string(index=False) if display else "No results.")
    print(f"Failures: {len(failures)}")
    print(f"Outputs: {run_dir}")
    if failures:
        raise SystemExit(1)


def run_config_command(args: argparse.Namespace) -> None:
    run_dir = _resolve(args.run_dir)
    run_benchpress_config(
        REPO_ROOT,
        run_dir,
        _resolve_run_path(run_dir, args.config),
        args.job_id,
        _resolve_run_path(run_dir, args.status)
        if args.status
        else run_dir / "jobs" / f"{args.job_id:03d}" / "status.json",
        _resolve_run_path(run_dir, args.stdout)
        if args.stdout
        else run_dir / "jobs" / f"{args.job_id:03d}" / "stdout.log",
        _resolve_run_path(run_dir, args.stderr)
        if args.stderr
        else run_dir / "jobs" / f"{args.job_id:03d}" / "stderr.log",
        cores=args.cores,
    )


def select_command(args: argparse.Namespace) -> None:
    out_path = (
        _resolve(args.out)
        if args.out is not None
        else _resolve(args.hparam_run) / "summaries/selected_best.json"
    )
    selected = select_best(
        _resolve(args.hparam_run),
        args.primary_metric,
        args.primary_direction,
        args.secondary_metric,
        args.secondary_direction,
        out_path,
    )
    for template_id, value in selected.items():
        print(
            f"{template_id}: {value['selected_algorithm_id']} "
            f"{value['primary_metric_column']}={value['primary_mean']:.6g}"
        )


def inject_command(args: argparse.Namespace) -> None:
    inject_best(
        _resolve(args.benchmark_config),
        _resolve(args.selected),
        _resolve(args.out),
    )
    print(f"Wrote final Benchpress config: {_resolve(args.out)}")


def precompute_independencies_command(args: argparse.Namespace) -> None:
    data_path = _resolve(args.data_csv)
    cache_dir = _resolve(args.cache_dir)
    df = pd.read_csv(data_path)
    result = pairwise_independence_candidates(
        df.to_numpy(dtype=float, copy=True),
        method=args.method,
        alpha=args.alpha,
        correction=args.correction,
        columns=list(df.columns),
        cache_dir=cache_dir,
        dataset_path=data_path,
    )
    print(
        f"independence cache {result.cache_status}: key={result.cache_key}, "
        f"tested={result.number_of_tests}, accepted={len(result.pairs)}, dir={result.cache_dir}"
    )


def prepare_validation_command(args: argparse.Namespace) -> None:
    run_dir = _resolve(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_validation_run(REPO_ROOT, run_dir, args.preset)
    cmd_path = _write_validation_cmd(run_dir, prepared.validation_config_path)
    config = json.loads(prepared.validation_config_path.read_text())
    counts = {
        name: len(entries)
        for name, entries in config["resources"]["structure_learning_algorithms"].items()
    }
    total_variants = sum(counts.values())
    print(f"Validation config: {prepared.validation_config_path}")
    print(f"Validation manifest CSV: {prepared.validation_manifest_csv}")
    print(f"Validation manifest JSON: {prepared.validation_manifest_json}")
    print(f"Command file: {cmd_path}")
    print(f"Expected Benchpress joint benchmark: {prepared.validation_joint_benchmarks_path}")
    print(f"Algorithm variant counts: {counts}")
    print(f"Total algorithm variants: {total_variants}")
    print("Note: Snakemake dry-run job counts include data, plotting, evaluation, and per-dataset jobs; they are not method counts.")
    print("Dry run:")
    print(
        "BENCHPRESS_SKIP_CONTAINER_CHECK=1 snakemake -n --cores all "
        f"--snakefile workflow/Snakefile --configfile {prepared.validation_config_path}"
    )


def prepare_experiment_command(args: argparse.Namespace) -> None:
    grid_path = _resolve(args.grid)
    grid = json.loads(grid_path.read_text())
    preset = grid.get("preset")
    if not preset:
        raise ValueError(f"Grid config must include a 'preset' field: {grid_path}")
    prepare_validation_command(argparse.Namespace(out=args.out, preset=preset))


def expand_grid_command(args: argparse.Namespace) -> None:
    if args.tag:
        args.grid = args.grid or _tag_grid_path(args.tag)
        args.out_config = args.out_config or _tag_config_path(args.tag)
        args.out_manifest = args.out_manifest or _tag_manifest_path(args.tag)
    if args.grid is None or args.out_config is None or args.out_manifest is None:
        raise ValueError("Provide --tag, or provide --grid, --out-config, and --out-manifest")
    expanded = expand_grid_config(
        REPO_ROOT,
        _resolve(args.grid),
        _resolve(args.out_config),
        _resolve(args.out_manifest),
    )
    print(f"Expanded config: {expanded.config_path}")
    print(f"Manifest CSV: {expanded.manifest_csv}")
    print(f"Manifest JSON: {expanded.manifest_json}")
    print(f"Expected Benchpress joint benchmark: {expanded.joint_benchmarks_path}")
    print(f"Algorithm variant counts: {expanded.algorithm_counts}")


def build_selected_benchmark_command(args: argparse.Namespace) -> None:
    if args.tag:
        args.frame = args.frame or _tag_benchmark_frame_path(args.tag)
        if args.selection is None and args.frame is not None:
            frame_path = _resolve(args.frame)
            if frame_path.is_file():
                frame = json.loads(frame_path.read_text())
                source = frame.get("selection", {}).get("source")
                if source:
                    args.selection = Path(source)
        args.selection = args.selection or _tag_selection_path(args.tag)
        args.out_config = args.out_config or _tag_selected_benchmark_config_path(args.tag)
        args.out_manifest = args.out_manifest or _tag_selected_benchmark_manifest_path(args.tag)
    if (
        args.frame is None
        or args.selection is None
        or args.out_config is None
        or args.out_manifest is None
    ):
        raise ValueError(
            "Provide --tag, or provide --frame, --selection, --out-config, and --out-manifest"
        )
    built = build_selected_benchmark_config(
        REPO_ROOT,
        _resolve(args.frame),
        _resolve(args.selection),
        _resolve(args.out_config),
        _resolve(args.out_manifest),
    )
    print(f"Selected benchmark config: {built.config_path}")
    print(f"Manifest CSV: {built.manifest_csv}")
    print(f"Manifest JSON: {built.manifest_json}")
    print(f"Expected Benchpress joint benchmark: {built.joint_benchmarks_path}")
    print(f"Dataset count: {built.dataset_count}")
    print(f"Algorithm variant counts: {built.algorithm_counts}")


def print_slurm_launch_command(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    if args.preset == "smoke":
        script = Path("workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_smoke.sh")
        config = args.config or Path("configs/notreks/expanded/smoke_config.json")
        cores = 8
        cluster = "serial"
    elif args.preset == "heavy":
        script = Path("workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_heavy.sh")
        config = args.config or Path("configs/notreks/expanded/full_benchmark_config.json")
        cores = 16
        cluster = "serial"
    else:
        raise ValueError(f"Unknown SLURM launch preset: {args.preset}")
    log_dir = run_dir / "logs/slurm"
    print(f"mkdir -p {log_dir}")
    print(
        f"RUN_DIR={run_dir} \\\n"
        f"CONFIG={config} \\\n"
        f"SNAKEMAKE_CORES={cores} \\\n"
        f"sbatch --clusters={cluster} "
        f"-o {log_dir}/%x-%j.out "
        f"-e {log_dir}/%x-%j.err "
        f"{script}"
    )


def select_validation_command(args: argparse.Namespace) -> None:
    if args.tag and args.config is None and args.manifest is None and args.out_dir is None:
        args.config = _tag_config_path(args.tag)
        args.manifest = _tag_manifest_path(args.tag)
        args.out_dir = _tag_selected_dir()
    if args.config is not None or args.manifest is not None:
        if args.config is None or args.manifest is None or args.out_dir is None:
            raise ValueError("--config, --manifest, and --out-dir must be provided together")
        selected = select_best_from_config_manifest(
            REPO_ROOT,
            _resolve(args.config),
            _resolve(args.manifest),
            _resolve(args.out_dir),
            args.tag,
            primary_metric=args.primary_metric,
            primary_direction=args.primary_direction,
            secondary_metric=args.secondary_metric,
            secondary_direction=args.secondary_direction,
        )
    else:
        if args.run_dir is None:
            raise ValueError("Either --run-dir or --config/--manifest/--out-dir is required")
        selected = select_best_by_method_family(
            _resolve(args.run_dir),
            REPO_ROOT,
            primary_metric=args.primary_metric,
            primary_direction=args.primary_direction,
            secondary_metric=args.secondary_metric,
            secondary_direction=args.secondary_direction,
        )
    for family, value in selected.items():
        print(
            f"{family}: {value['selected_algorithm_id']} "
            f"{value['primary_metric_column']}={value['primary_mean']:.6g}"
        )


def write_final_config_command(args: argparse.Namespace) -> None:
    path = write_final_benchmark_config(
        REPO_ROOT,
        _resolve(args.run_dir),
        _resolve(args.selected) if args.selected else None,
    )
    print(f"Final benchmark config: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--preset", choices=["tiny", "small"], default="tiny")
    smoke.add_argument(
        "--dag-seq",
        choices=["exp", "logdet", "scc_power_iteration"],
        default="exp",
    )
    smoke.add_argument("--out", type=Path, default=Path("results/notreks_smoke"))
    smoke.set_defaults(func=smoke_command)

    prepare = subparsers.add_parser("prepare-hparam")
    prepare.add_argument("--grid-config", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--grid-mode", choices=["cartesian", "zip"], default="cartesian")
    prepare.set_defaults(func=prepare_command)

    run_config = subparsers.add_parser("run-config")
    run_config.add_argument("--run-dir", type=Path, required=True)
    run_config.add_argument("--config", type=Path, required=True)
    run_config.add_argument("--job-id", type=int, required=True)
    run_config.add_argument("--status", type=Path, default=None)
    run_config.add_argument("--stdout", type=Path, default=None)
    run_config.add_argument("--stderr", type=Path, default=None)
    run_config.add_argument("--cores", type=int, default=1)
    run_config.set_defaults(func=run_config_command)

    select = subparsers.add_parser("select-best")
    select.add_argument("--hparam-run", type=Path, required=True)
    select.add_argument("--primary-metric", default="SHD_cpdag")
    select.add_argument("--primary-direction", choices=["min", "max"], default=None)
    select.add_argument("--secondary-metric", default=None)
    select.add_argument("--secondary-direction", choices=["min", "max"], default=None)
    select.add_argument("--out", type=Path, default=None)
    select.set_defaults(func=select_command)

    inject = subparsers.add_parser("inject-best")
    inject.add_argument("--benchmark-config", type=Path, required=True)
    inject.add_argument("--selected", type=Path, required=True)
    inject.add_argument("--out", type=Path, required=True)
    inject.set_defaults(func=inject_command)

    precompute = subparsers.add_parser("precompute-independencies")
    precompute.add_argument("--data-csv", type=Path, required=True)
    precompute.add_argument("--cache-dir", type=Path, required=True)
    precompute.add_argument(
        "--method",
        choices=[
            "pearson",
            "spearman",
            "hsic",
            "dcor",
            "gcastle_fisherz",
            "gcastle_g2",
            "gcastle_chi2",
        ],
        required=True,
    )
    precompute.add_argument("--alpha", type=float, required=True)
    precompute.add_argument(
        "--correction",
        choices=["none", "bonferroni", "benjamini-hochberg"],
        required=True,
    )
    precompute.set_defaults(func=precompute_independencies_command)

    prepare_validation = subparsers.add_parser("prepare-validation")
    prepare_validation.add_argument(
        "--preset",
        choices=["tiny", "local10", "local10_sensible", "local10_quick"],
        default="tiny",
    )
    prepare_validation.add_argument("--out", type=Path, required=True)
    prepare_validation.set_defaults(func=prepare_validation_command)

    prepare_experiment = subparsers.add_parser("prepare-experiment")
    prepare_experiment.add_argument("--grid", type=Path, required=True)
    prepare_experiment.add_argument("--out", type=Path, required=True)
    prepare_experiment.set_defaults(func=prepare_experiment_command)

    expand_grid = subparsers.add_parser("expand-grid")
    expand_grid.add_argument("--tag", default=None)
    expand_grid.add_argument("--grid", type=Path, default=None)
    expand_grid.add_argument("--out-config", type=Path, default=None)
    expand_grid.add_argument("--out-manifest", type=Path, default=None)
    expand_grid.set_defaults(func=expand_grid_command)

    selected_benchmark = subparsers.add_parser("build-selected-benchmark")
    selected_benchmark.add_argument("--tag", default=None)
    selected_benchmark.add_argument("--frame", type=Path, default=None)
    selected_benchmark.add_argument("--selection", type=Path, default=None)
    selected_benchmark.add_argument("--out-config", type=Path, default=None)
    selected_benchmark.add_argument("--out-manifest", type=Path, default=None)
    selected_benchmark.set_defaults(func=build_selected_benchmark_command)

    slurm_launch = subparsers.add_parser("print-slurm-launch")
    slurm_launch.add_argument("--run-dir", type=Path, required=True)
    slurm_launch.add_argument("--config", type=Path, default=None)
    slurm_launch.add_argument(
        "--preset",
        choices=["smoke", "heavy"],
        required=True,
    )
    slurm_launch.set_defaults(func=print_slurm_launch_command)

    select_validation = subparsers.add_parser("select-validation-best")
    select_validation.add_argument("--run-dir", type=Path, default=None)
    select_validation.add_argument("--config", type=Path, default=None)
    select_validation.add_argument("--manifest", type=Path, default=None)
    select_validation.add_argument("--out-dir", type=Path, default=None)
    select_validation.add_argument("--tag", default="validation")
    select_validation.add_argument("--primary-metric", default="SHD_cpdag")
    select_validation.add_argument("--primary-direction", choices=["min", "max"], default=None)
    select_validation.add_argument("--secondary-metric", default=None)
    select_validation.add_argument("--secondary-direction", choices=["min", "max"], default=None)
    select_validation.set_defaults(func=select_validation_command)

    final_config = subparsers.add_parser("write-final-config")
    final_config.add_argument("--run-dir", type=Path, required=True)
    final_config.add_argument("--selected", type=Path, default=None)
    final_config.set_defaults(func=write_final_config_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
