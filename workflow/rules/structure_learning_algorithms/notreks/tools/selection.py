"""Select tuned NOTREKS settings and inject them into final Benchpress configs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from grid import load_json
from jobfarm import read_manifest


def _normalize_metric(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_metric_column(columns: list[str], requested: str) -> str:
    aliases = {
        "shdcpdag": {"shdcpdag", "meanshdcpdag"},
    }
    wanted = _normalize_metric(requested)
    accepted = aliases.get(wanted, {wanted})
    for column in columns:
        if _normalize_metric(column) in accepted:
            return column
    raise ValueError(
        f"CPDAG SHD metric {requested!r} was not found. "
        f"Available columns: {', '.join(columns)}"
    )


def _id_column(columns: list[str]) -> str:
    for candidate in ("id", "algorithm_id"):
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not identify the expanded algorithm id column. "
        f"Available columns: {', '.join(columns)}"
    )


def select_best(
    hparam_run: Path,
    metric: str,
    output_json: Path,
) -> dict:
    manifest = read_manifest(hparam_run / "manifest.csv")
    with (hparam_run / "grid_index.json").open() as handle:
        index = json.load(handle)
    if not isinstance(index, list):
        raise ValueError("grid_index.json must contain a list")
    configs = {row["algorithm_id"]: row["config"] for row in index}
    selected = {}
    summary_rows = []

    for job in manifest:
        template_id = job["template_id"]
        result_path = Path(job["joint_benchmarks_path"])
        if not result_path.is_file():
            raise FileNotFoundError(
                f"Missing Benchpress result for template {template_id}: {result_path}"
            )
        frame = pd.read_csv(result_path)
        metric_column = resolve_metric_column(list(frame.columns), metric)
        id_column = _id_column(list(frame.columns))
        frame[metric_column] = pd.to_numeric(frame[metric_column], errors="coerce")
        means = frame.groupby(id_column, dropna=False)[metric_column].mean().dropna()
        if means.empty:
            raise ValueError(
                f"No numeric {metric_column} values for template {template_id}"
            )
        best_id = str(means.idxmin())
        if best_id not in configs:
            raise ValueError(f"Selected id {best_id!r} is missing from grid_index.json")
        mean_metric = float(means.loc[best_id])
        selected[template_id] = {
            "selected_algorithm_id": best_id,
            "metric": metric,
            "metric_column": metric_column,
            "mean_metric": mean_metric,
            "config": configs[best_id],
        }
        summary_rows.append(
            {
                "template_id": template_id,
                "selected_algorithm_id": best_id,
                "metric": metric,
                "metric_column": metric_column,
                "mean_metric": mean_metric,
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(selected, indent=2) + "\n")
    pd.DataFrame(summary_rows).to_csv(output_json.with_suffix(".csv"), index=False)
    return selected


def _replace_ids(values: list, old_ids: set[str], new_ids: list[str]) -> list:
    retained = [value for value in values if value not in old_ids]
    for value in new_ids:
        if value not in retained:
            retained.append(value)
    return retained


def inject_best(
    benchmark_config_path: Path,
    selected_path: Path,
    output_path: Path,
) -> dict:
    benchmark = load_json(benchmark_config_path)
    selected = load_json(selected_path)
    resources = benchmark.get("resources", {})
    algorithms = resources.get("structure_learning_algorithms", {})
    old_entries = algorithms.get("notreks", [])
    old_ids = {str(entry["id"]) for entry in old_entries}
    selected_entries = [value["config"] for value in selected.values()]
    if not selected_entries:
        raise ValueError("Selected config file does not contain any NOTREKS settings")
    algorithms["notreks"] = selected_entries
    resources["structure_learning_algorithms"] = algorithms
    benchmark["resources"] = resources

    selected_ids = [entry["id"] for entry in selected_entries]
    for setup in benchmark.get("benchmark_setup", []):
        evaluation = setup.get("evaluation", {})
        benchmarks = evaluation.get("benchmarks")
        if isinstance(benchmarks, dict):
            benchmarks["ids"] = _replace_ids(
                list(benchmarks.get("ids", [])), old_ids, selected_ids
            )
        if isinstance(evaluation.get("graph_plots"), list):
            evaluation["graph_plots"] = _replace_ids(
                evaluation["graph_plots"], old_ids, selected_ids
            )
        graph_estimation = evaluation.get("graph_estimation")
        if isinstance(graph_estimation, dict) and isinstance(graph_estimation.get("ids"), list):
            graph_estimation["ids"] = _replace_ids(
                graph_estimation["ids"], old_ids, selected_ids
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark, indent=2) + "\n")
    return benchmark
