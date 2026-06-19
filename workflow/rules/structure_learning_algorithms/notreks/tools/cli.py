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

from grid import prepare_hparam_run  # noqa: E402
from jobfarm import (  # noqa: E402
    make_manifest,
    read_manifest,
    run_benchpress_config,
    write_command_file,
)
from selection import inject_best, select_best  # noqa: E402


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _smoke_meta(preset: str, run_name: str) -> dict:
    if preset == "tiny":
        methods = [
            ("exp-no-trek", "none", "none", 0.0),
            ("exp-notreks-spearman", "spearman", "exp", 1.0),
        ]
        max_iter = 60000
        path_steps = 2
    elif preset == "small":
        methods = [
            ("exp-no-trek", "none", "none", 0.0),
            ("exp-notreks-spearman", "spearman", "exp", 1.0),
            ("exp-notreks-pearson", "pearson", "exp", 1.0),
            ("exp-notreks-dcor", "dcor", "exp", 1.0),
        ]
        max_iter = 60000
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
                "dag_seq": "exp",
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
        json.dumps(_smoke_meta(args.preset, f"notreks_smoke_{args.preset}"), indent=2)
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
                Path(row["config_path"]),
                Path(row["status_path"]),
                Path(row["stdout_path"]),
                Path(row["stderr_path"]),
                cores=1,
            )
            joint = Path(row["joint_benchmarks_path"])
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
    run_benchpress_config(
        REPO_ROOT,
        _resolve(args.config),
        _resolve(args.status),
        _resolve(args.stdout),
        _resolve(args.stderr),
        cores=args.cores,
    )


def select_command(args: argparse.Namespace) -> None:
    selected = select_best(
        _resolve(args.hparam_run),
        args.metric,
        _resolve(args.out),
    )
    for template_id, value in selected.items():
        print(
            f"{template_id}: {value['selected_algorithm_id']} "
            f"{value['metric_column']}={value['mean_metric']:.6g}"
        )


def inject_command(args: argparse.Namespace) -> None:
    inject_best(
        _resolve(args.benchmark_config),
        _resolve(args.selected),
        _resolve(args.out),
    )
    print(f"Wrote final Benchpress config: {_resolve(args.out)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--preset", choices=["tiny", "small"], default="tiny")
    smoke.add_argument("--out", type=Path, default=Path("results/notreks_smoke"))
    smoke.set_defaults(func=smoke_command)

    prepare = subparsers.add_parser("prepare-hparam")
    prepare.add_argument("--grid-config", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--grid-mode", choices=["cartesian", "zip"], default="cartesian")
    prepare.set_defaults(func=prepare_command)

    run_config = subparsers.add_parser("run-config")
    run_config.add_argument("--config", type=Path, required=True)
    run_config.add_argument("--status", type=Path, required=True)
    run_config.add_argument("--stdout", type=Path, required=True)
    run_config.add_argument("--stderr", type=Path, required=True)
    run_config.add_argument("--cores", type=int, default=1)
    run_config.set_defaults(func=run_config_command)

    select = subparsers.add_parser("select-best")
    select.add_argument("--hparam-run", type=Path, required=True)
    select.add_argument("--metric", default="shd_cpdag")
    select.add_argument("--out", type=Path, required=True)
    select.set_defaults(func=select_command)

    inject = subparsers.add_parser("inject-best")
    inject.add_argument("--benchmark-config", type=Path, required=True)
    inject.add_argument("--selected", type=Path, required=True)
    inject.add_argument("--out", type=Path, required=True)
    inject.set_defaults(func=inject_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
