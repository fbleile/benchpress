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
        "shdcpdag": {"shdcpdag", "meanshdcpdag", "shdcpdagmean"},
        "shdpattern": {"shdpattern", "shdpatternmean"},
        "f1skel": {"f1skel", "f1skelmean"},
        "time": {"time", "timemean"},
    }
    wanted = _normalize_metric(requested)
    accepted = aliases.get(wanted, {wanted})
    for column in columns:
        if _normalize_metric(column) in accepted:
            return column
    raise ValueError(
        f"Requested metric {requested!r} was not found. "
        f"Available columns: {', '.join(columns)}"
    )


def default_direction(metric: str) -> str:
    normalized = _normalize_metric(metric)
    if any(token in normalized for token in ("tpr", "f1", "precision", "recall")):
        return "max"
    if any(token in normalized for token in ("shd", "fpr", "fnr", "time", "elapsed", "runtime")):
        return "min"
    raise ValueError(f"Metric {metric!r} has ambiguous direction; pass an explicit direction")


def available_metric_columns(frame: pd.DataFrame) -> list[str]:
    ignored = {"id", "algorithm", "adjmat", "parameters", "data", "seed", "curve_param", "curve_value"}
    return [
        column
        for column in frame.columns
        if column not in ignored and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]


def _choose_secondary_metric(columns: list[str]) -> tuple[str | None, str | None]:
    for candidate, direction in [("f1_skel_mean", "max"), ("time_mean", "min"), ("time", "min")]:
        try:
            return resolve_metric_column(columns, candidate), direction
        except ValueError:
            continue
    return None, None


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
    primary_metric: str,
    primary_direction: str | None,
    secondary_metric: str | None,
    secondary_direction: str | None,
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
        result_path = hparam_run / job["relative_joint_benchmarks_path"]
        if not result_path.is_file():
            raise FileNotFoundError(
                f"Missing Benchpress result for template {template_id}: {result_path}"
            )
        frame = pd.read_csv(result_path)
        metric_column = resolve_metric_column(list(frame.columns), primary_metric)
        metric_direction = primary_direction or default_direction(metric_column)
        id_column = _id_column(list(frame.columns))
        frame[metric_column] = pd.to_numeric(frame[metric_column], errors="coerce")
        grouped = frame.groupby(id_column, dropna=False)
        means = grouped[metric_column].mean().dropna()
        if means.empty:
            raise ValueError(
                f"No numeric {metric_column} values for template {template_id}"
            )
        ranking = pd.DataFrame({"primary": means})
        secondary_column = None
        secondary_dir = secondary_direction
        if secondary_metric is not None:
            secondary_column = resolve_metric_column(list(frame.columns), secondary_metric)
            secondary_dir = secondary_dir or default_direction(secondary_column)
        else:
            secondary_column, secondary_dir = _choose_secondary_metric(list(frame.columns))
        if secondary_column is not None:
            frame[secondary_column] = pd.to_numeric(frame[secondary_column], errors="coerce")
            ranking["secondary"] = grouped[secondary_column].mean()
        ranking["algorithm_id"] = ranking.index.astype(str)
        sort_cols = ["primary"]
        ascending = [metric_direction == "min"]
        if "secondary" in ranking.columns:
            sort_cols.append("secondary")
            ascending.append(secondary_dir == "min")
        sort_cols.append("algorithm_id")
        ascending.append(True)
        ranking = ranking.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        best_id = str(ranking.index[0])
        if best_id not in configs:
            raise ValueError(f"Selected id {best_id!r} is missing from grid_index.json")
        mean_metric = float(ranking.loc[best_id, "primary"])
        secondary_mean = None if "secondary" not in ranking.columns else float(ranking.loc[best_id, "secondary"])
        selected[template_id] = {
            "selected_algorithm_id": best_id,
            "primary_metric": primary_metric,
            "primary_metric_column": metric_column,
            "primary_direction": metric_direction,
            "primary_mean": mean_metric,
            "secondary_metric_column": secondary_column,
            "secondary_direction": secondary_dir,
            "secondary_mean": secondary_mean,
            "available_metric_columns": available_metric_columns(frame),
            "config": configs[best_id],
        }
        summary_rows.append(
            {
                "template_id": template_id,
                "selected_algorithm_id": best_id,
                "primary_metric": primary_metric,
                "primary_metric_column": metric_column,
                "primary_direction": metric_direction,
                "primary_mean": mean_metric,
                "secondary_metric_column": secondary_column,
                "secondary_direction": secondary_dir,
                "secondary_mean": secondary_mean,
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
