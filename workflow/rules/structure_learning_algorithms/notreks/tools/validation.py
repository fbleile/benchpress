"""Generate one-config validation grids and selected final benchmark configs."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fixed_data import FixedDataReference, FixedDataSpec, prepare_fixed_data, safe_name
from grid import load_json
from selection import default_direction, resolve_metric_column


@dataclass(frozen=True)
class ValidationRun:
    run_name: str
    validation_config_path: Path
    validation_manifest_csv: Path
    validation_manifest_json: Path
    validation_joint_benchmarks_path: Path
    final_fixed_data: dict[str, Any]


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _evaluation(algorithm_ids: list[str], prefix: str) -> dict:
    return {
        "benchmarks": {
            "filename_prefix": prefix,
            "show_seed": True,
            "errorbar": True,
            "errorbarh": False,
            "scatter": True,
            "path": True,
            "text": False,
            "ids": algorithm_ids,
        },
        "graph_true_plots": False,
        "graph_true_stats": True,
        "graph_plots": [],
    }


def _fixed_data_metadata(base_dir: Path, reference: FixedDataReference) -> dict[str, Any]:
    metadata_path = base_dir / reference.metadata_path
    return load_json(metadata_path)


def _fixed_data_entries(fixed_data: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for data_file in fixed_data["data_files"]:
        prefix = "resources/data/mydatasets/"
        if not str(data_file).startswith(prefix):
            raise ValueError(f"Expected fixed data file under {prefix}: {data_file}")
        entries.append(
            {
                "graph_id": fixed_data["graph_id"],
                "parameters_id": None,
                "data_id": str(data_file)[len(prefix):],
                "seed_range": None,
            }
        )
    return entries


def _pc_grid(preset: str) -> list[dict[str, Any]]:
    alphas = [0.05] if preset == "tiny" else [0.01, 0.05, 0.1]
    entries = []
    for index, alpha in enumerate(alphas):
        entries.append(
            {
                "id": f"gcastle_pc__grid{index:03d}",
                "variant": "stable",
                "alpha": alpha,
                "ci_test": "fisherz",
                "timeout": None,
            }
        )
    return entries


def _lingam_grid(preset: str) -> list[dict[str, Any]]:
    measures = ["pwling"] if preset == "tiny" else ["pwling", "kernel"]
    return [
        {
            "id": f"gcastle_lingam__grid{index:03d}",
            "measure": measure,
            "thresh": 0.3,
            "timeout": None,
        }
        for index, measure in enumerate(measures)
    ]


def _base_notreks(max_iter: int, threshold: float) -> dict[str, Any]:
    return {
        "function_class": "linear",
        "score": "least_squares",
        "dag_reg": 1.0,
        "dag_s": 1.0,
        "trek_seq": "exp",
        "regularizer": "l1",
        "independence_correction": "benjamini-hochberg",
        "independence_alpha": 0.05,
        "seed": 1,
        "max_iter": max_iter,
        "lr": 0.0003,
        "path_steps": 5,
        "mu_init": 1.0,
        "mu_factor": 0.1,
        "warm_iter": max_iter,
        "tol": 1e-6,
        "threshold": threshold,
        "timeout": None,
        "init": "zero",
        "checkpoint": max(max_iter // 10, 1),
        "scc_threshold": 1e-8,
    }


def _notreks_grid(preset: str, cache_dir: str) -> list[dict[str, Any]]:
    if preset == "tiny":
        specs = [
            {
                "independence_test": "spearman",
                "dag_seq": "logdet",
                "max_iter": 500,
                "trek_reg": 1.0,
                "regularizer_scale": 0.01,
                "power_iter_steps": 5,
            },
            {
                "independence_test": "gcastle_fisherz",
                "dag_seq": "logdet",
                "max_iter": 500,
                "trek_reg": 1.0,
                "regularizer_scale": 0.01,
                "power_iter_steps": 5,
            },
        ]
    elif preset in {"local10_sensible", "local10_quick"}:
        specs = []
        trek_values = [0.5, 1.0] if preset == "local10_sensible" else [1.0]
        max_iter = 30000 if preset == "local10_sensible" else 3000
        for independence_test, dag_seq, trek_reg in itertools.product(
            ["spearman", "dcor", "gcastle_fisherz"],
            ["logdet", "scc_power_iteration"],
            trek_values,
        ):
            specs.append(
                {
                    "independence_test": independence_test,
                    "dag_seq": dag_seq,
                    "max_iter": max_iter,
                    "trek_reg": trek_reg,
                    "regularizer_scale": 0.01,
                    "power_iter_steps": 3 if dag_seq == "scc_power_iteration" else 5,
                }
            )
    else:
        specs = []
        for independence_test, dag_seq, max_iter, trek_reg, regularizer_scale in itertools.product(
            ["spearman", "dcor", "gcastle_fisherz"],
            ["logdet", "scc_power_iteration"],
            [30000, 60000],
            [0.5, 1.0, 2.0],
            [0.001, 0.01],
        ):
            if dag_seq == "scc_power_iteration":
                for power_iter_steps in [3, 5]:
                    specs.append(
                        {
                            "independence_test": independence_test,
                            "dag_seq": dag_seq,
                            "max_iter": max_iter,
                            "trek_reg": trek_reg,
                            "regularizer_scale": regularizer_scale,
                            "power_iter_steps": power_iter_steps,
                        }
                    )
            else:
                specs.append(
                    {
                        "independence_test": independence_test,
                        "dag_seq": dag_seq,
                        "max_iter": max_iter,
                        "trek_reg": trek_reg,
                        "regularizer_scale": regularizer_scale,
                        "power_iter_steps": 5,
                    }
                )

    entries = []
    for index, spec in enumerate(specs):
        entry = _base_notreks(int(spec["max_iter"]), threshold=0.1)
        entry.update(spec)
        entry["id"] = f"notreks__grid{index:03d}"
        entry["independence_cache_dir"] = cache_dir
        entries.append(entry)
    return entries


def _manifest_rows(config_path: Path, phase: str, families: dict[str, tuple[str, list[dict[str, Any]]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for family, (base_method, entries) in families.items():
        for entry in entries:
            rows.append(
                {
                    "algorithm_id": str(entry["id"]),
                    "path_id": _path_id(str(entry["id"])),
                    "method_family": family,
                    "base_method": base_method,
                    "hyperparameters_json": json.dumps(entry, sort_keys=True),
                    "config_path": str(config_path),
                    "phase": phase,
                }
            )
    return rows


def _manifest_json_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    json_rows: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["hyperparameters"] = json.loads(row["hyperparameters_json"])
        json_rows.append(item)
    return json_rows


def _path_id(algorithm_id: str) -> str:
    if algorithm_id.startswith("notreks__grid"):
        return "n" + algorithm_id.rsplit("grid", 1)[1]
    if algorithm_id == "notreks__best":
        return "nbest"
    return safe_name(algorithm_id)


def _write_manifest(rows: list[dict[str, str]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm_id",
                "path_id",
                "method_family",
                "base_method",
                "hyperparameters_json",
                "config_path",
                "phase",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(_manifest_json_rows(rows), indent=2) + "\n")


def _short_notreks_entries(entries: list[dict[str, Any]], manifest_path: Path, repo_root: Path) -> list[dict[str, str]]:
    manifest = _repo_relative(manifest_path, repo_root)
    return [
        {
            "id": str(entry["id"]),
            "alg_id": _path_id(str(entry["id"])),
            "params_manifest": manifest,
        }
        for entry in entries
    ]


def expected_algorithm_run_counts(config: dict[str, Any]) -> dict[str, int]:
    num_data = len(config["benchmark_setup"][0]["data"])
    algorithms = config["resources"]["structure_learning_algorithms"]
    return {
        "copy_fixed_data": num_data,
        "gcastle_pc": len(algorithms.get("gcastle_pc", [])) * num_data,
        "gcastle_direct_lingam": len(algorithms.get("gcastle_direct_lingam", [])) * num_data,
        "notreks": len(algorithms.get("notreks", [])) * num_data,
    }


def parse_dryrun_job_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            counts[parts[0]] = int(parts[1])
    return counts


def assert_no_fixed_data_duplication_in_dryrun(text: str, expected_counts: dict[str, int]) -> None:
    bad_patterns = ("seed=1/seed=1", "n=None/seed=1", "n=500/seed=1/seed=1")
    for pattern in bad_patterns:
        if pattern in text:
            raise AssertionError(f"Dry-run output contains duplicated fixed-data path pattern: {pattern}")
    observed = parse_dryrun_job_counts(text)
    for key in ("copy_fixed_data", "gcastle_pc", "gcastle_direct_lingam", "notreks"):
        if observed.get(key) != expected_counts.get(key):
            raise AssertionError(
                f"Dry-run count mismatch for {key}: observed={observed.get(key)}, "
                f"expected={expected_counts.get(key)}"
            )


def _preset_specs(run_name: str, preset: str) -> tuple[FixedDataSpec, FixedDataSpec]:
    compact = hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:4]
    if preset == "tiny":
        return (
            FixedDataSpec(run_name=f"{compact}v", d=10, n_values=(500,), seeds=(101, 102), expected_degree=2.0),
            FixedDataSpec(run_name=f"{compact}f", d=20, n_values=(500,), seeds=(201, 202), expected_degree=2.0),
        )
    if preset in {"local10", "local10_sensible", "local10_quick"}:
        return (
            FixedDataSpec(run_name=f"{compact}v", d=10, n_values=(500,), seeds=tuple(range(101, 111)), expected_degree=2.0),
            FixedDataSpec(run_name=f"{compact}f", d=20, n_values=(500,), seeds=tuple(range(201, 211)), expected_degree=2.0),
        )
    raise ValueError("preset must be 'tiny', 'local10', 'local10_sensible', or 'local10_quick'")


def prepare_validation_run(repo_root: Path, run_dir: Path, preset: str) -> ValidationRun:
    run_name = safe_name(run_dir.name)
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    validation_spec, final_spec = _preset_specs(run_name, preset)
    validation_base = run_dir / "fixed_data/validation"
    final_base = run_dir / "fixed_data/final"
    validation_fixed = prepare_fixed_data(repo_root, validation_base, validation_spec)
    final_fixed = prepare_fixed_data(repo_root, final_base, final_spec)
    validation_metadata = _fixed_data_metadata(validation_base, validation_fixed)
    final_metadata = _fixed_data_metadata(final_base, final_fixed)

    cache_dir = _repo_relative(run_dir / "independence_cache", repo_root)
    families = {
        "gcastle_pc": ("gcastle_pc", _pc_grid(preset)),
        "gcastle_lingam": ("gcastle_direct_lingam", _lingam_grid(preset)),
        "notreks": ("notreks", _notreks_grid(preset, cache_dir)),
    }
    algorithm_ids = [entry["id"] for _, entries in families.values() for entry in entries]
    title = safe_name(f"{run_name}_validation")
    prefix = f"{run_name}/validation/"
    config = {
        "benchmark_setup": [
            {
                "title": title,
                "data": _fixed_data_entries(validation_metadata),
                "evaluation": _evaluation(algorithm_ids, prefix),
            }
        ],
        "resources": {
            "data": {},
            "graph": {},
            "parameters": {},
            "structure_learning_algorithms": {
                "gcastle_pc": families["gcastle_pc"][1],
                "gcastle_direct_lingam": families["gcastle_lingam"][1],
                "notreks": families["notreks"][1],
            },
        },
    }
    validation_config_path = configs_dir / "validation_hparam_config.json"
    manifest_rows = _manifest_rows(
        validation_config_path.relative_to(run_dir),
        "validation",
        families,
    )
    manifest_csv = configs_dir / "validation_hparam_manifest.csv"
    manifest_json = configs_dir / "validation_hparam_manifest.json"
    _write_manifest(manifest_rows, manifest_csv, manifest_json)
    config["resources"]["structure_learning_algorithms"]["notreks"] = _short_notreks_entries(
        families["notreks"][1],
        manifest_json,
        repo_root,
    )
    validation_config_path.write_text(json.dumps(config, indent=2) + "\n")

    joint_path = repo_root / "results/output" / title / "benchmarks" / prefix / "joint_benchmarks.csv"
    run_info = {
        "preset": preset,
        "run_name": run_name,
        "validation_fixed_data": {**asdict(validation_fixed), "data_files": validation_metadata["data_files"]},
        "final_fixed_data": {**asdict(final_fixed), "data_files": final_metadata["data_files"]},
        "validation_data_entries": _fixed_data_entries(validation_metadata),
        "final_data_entries": _fixed_data_entries(final_metadata),
        "validation_config": str(validation_config_path.relative_to(run_dir)),
        "validation_manifest_csv": str(manifest_csv.relative_to(run_dir)),
        "validation_joint_benchmarks": _repo_relative(joint_path, repo_root),
        "algorithm_counts": {family: len(entries) for family, (_, entries) in families.items()},
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return ValidationRun(
        run_name=run_name,
        validation_config_path=validation_config_path,
        validation_manifest_csv=manifest_csv,
        validation_manifest_json=manifest_json,
        validation_joint_benchmarks_path=joint_path,
        final_fixed_data=asdict(final_fixed),
    )


def _id_column(columns: list[str]) -> str:
    for candidate in ("id", "algorithm_id"):
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find algorithm id column. Available columns: {', '.join(columns)}")


def select_best_by_method_family(
    run_dir: Path,
    repo_root: Path,
    primary_metric: str = "SHD_cpdag",
    primary_direction: str | None = None,
    secondary_metric: str | None = None,
    secondary_direction: str | None = None,
) -> dict[str, Any]:
    configs_dir = run_dir / "configs"
    manifest_path = configs_dir / "validation_hparam_manifest.csv"
    run_info = load_json(run_dir / "run_info.json")
    joint_path = Path(run_info["validation_joint_benchmarks"])
    if not joint_path.is_absolute():
        joint_path = repo_root / joint_path
    if not joint_path.is_file():
        raise FileNotFoundError(f"Missing validation Benchpress result: {joint_path}")

    manifest = pd.read_csv(manifest_path)
    results = pd.read_csv(joint_path)
    id_column = _id_column(list(results.columns))
    metric_column = resolve_metric_column(list(results.columns), primary_metric)
    metric_direction = primary_direction or default_direction(metric_column)
    secondary_column = resolve_metric_column(list(results.columns), secondary_metric) if secondary_metric else None
    secondary_dir = secondary_direction or (default_direction(secondary_column) if secondary_column else None)

    merged = results.merge(manifest, left_on=id_column, right_on="algorithm_id", how="inner")
    if merged.empty:
        raise ValueError("No validation rows matched algorithm ids in validation_hparam_manifest.csv")
    merged[metric_column] = pd.to_numeric(merged[metric_column], errors="coerce")
    summary_rows = []
    selected: dict[str, Any] = {}
    for family, group in merged.groupby("method_family"):
        grouped = group.groupby("algorithm_id", dropna=False)
        ranking = pd.DataFrame({"primary": grouped[metric_column].mean()})
        if secondary_column:
            group = group.copy()
            group[secondary_column] = pd.to_numeric(group[secondary_column], errors="coerce")
            ranking["secondary"] = group.groupby("algorithm_id", dropna=False)[secondary_column].mean()
        ranking["algorithm_id"] = ranking.index.astype(str)
        ranking = ranking.reset_index(drop=True)
        sort_cols = ["primary"]
        ascending = [metric_direction == "min"]
        if secondary_column:
            sort_cols.append("secondary")
            ascending.append(secondary_dir == "min")
        sort_cols.append("algorithm_id")
        ascending.append(True)
        ranking = ranking.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        best_row = ranking.iloc[0]
        best_id = str(best_row["algorithm_id"])
        manifest_row = manifest.loc[manifest["algorithm_id"] == best_id].iloc[0]
        config = json.loads(manifest_row["hyperparameters_json"])
        selected[family] = {
            "selected_algorithm_id": best_id,
            "method_family": family,
            "base_method": manifest_row["base_method"],
            "primary_metric_column": metric_column,
            "primary_direction": metric_direction,
            "primary_mean": float(best_row["primary"]),
            "secondary_metric_column": secondary_column,
            "secondary_direction": secondary_dir,
            "secondary_mean": None if not secondary_column else float(best_row["secondary"]),
            "config": config,
        }
        summary_rows.append(selected[family])

    selection_dir = run_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "best_by_method_family.json").write_text(json.dumps(selected, indent=2) + "\n")
    pd.DataFrame(summary_rows).to_csv(selection_dir / "validation_summary.csv", index=False)
    return selected


def write_final_benchmark_config(repo_root: Path, run_dir: Path, selected_path: Path | None = None) -> Path:
    selected_path = selected_path or run_dir / "selection/best_by_method_family.json"
    selected = load_json(selected_path)
    run_info = load_json(run_dir / "run_info.json")
    final_fixed = run_info["final_fixed_data"]
    selected_entries = {family: value["config"] for family, value in selected.items()}
    algorithm_ids = [entry["id"] for entry in selected_entries.values()]
    title = safe_name(f"{run_info['run_name']}_final")
    prefix = f"{run_info['run_name']}/final/"
    config = {
        "benchmark_setup": [
            {
                "title": title,
                "data": _fixed_data_entries(final_fixed),
                "evaluation": _evaluation(algorithm_ids, prefix),
            }
        ],
        "resources": {
            "data": {},
            "graph": {},
            "parameters": {},
            "structure_learning_algorithms": {},
        },
    }
    resources = config["resources"]["structure_learning_algorithms"]
    out = run_dir / "configs/final_benchmark_config.json"
    if "gcastle_pc" in selected_entries:
        resources["gcastle_pc"] = [selected_entries["gcastle_pc"]]
    if "gcastle_lingam" in selected_entries:
        resources["gcastle_direct_lingam"] = [selected_entries["gcastle_lingam"]]
    if "notreks" in selected_entries:
        selected_manifest_json = run_dir / "configs/final_hparam_manifest.json"
        selected_manifest_csv = run_dir / "configs/final_hparam_manifest.csv"
        rows = _manifest_rows(
            out.relative_to(run_dir),
            "final",
            {"notreks": ("notreks", [selected_entries["notreks"]])},
        )
        _write_manifest(rows, selected_manifest_csv, selected_manifest_json)
        resources["notreks"] = _short_notreks_entries(
            [selected_entries["notreks"]],
            selected_manifest_json,
            repo_root,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2) + "\n")
    return out
