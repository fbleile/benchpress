"""Expand NOTREKS meta-grids into scalar Benchpress configurations."""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fixed_data import FixedDataReference, FixedDataSpec, prepare_fixed_data, safe_name, spec_from_meta


@dataclass(frozen=True)
class ExpandedTemplate:
    template_id: str
    relative_config_path: str
    algorithm_ids: tuple[str, ...]
    benchmark_title: str
    filename_prefix: str
    relative_job_dir: str
    relative_joint_benchmarks_path: str
    relative_roc_data_path: str
    benchpress_joint_benchmarks_path: str
    benchpress_roc_data_path: str


def load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def extract_notreks_templates(meta_config: dict) -> list[dict[str, Any]]:
    if "resources" in meta_config:
        algorithms = (
            meta_config.get("resources", {})
            .get("structure_learning_algorithms", {})
            .get("notreks", [])
        )
    else:
        algorithms = (
            meta_config.get("structure_learning_algorithms", {})
            .get("notreks", [])
        )
    if not algorithms:
        raise ValueError("Grid config does not contain any NOTREKS method templates")
    ids = [str(item.get("id", "")) for item in algorithms]
    if any(not value for value in ids):
        raise ValueError("Every NOTREKS method template must have a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("NOTREKS method template ids must be unique")
    return [copy.deepcopy(item) for item in algorithms]


def expand_template(template: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    template_id = str(template["id"])
    list_fields = {
        key: value
        for key, value in template.items()
        if key != "id" and isinstance(value, list)
    }
    scalar_fields = {
        key: value
        for key, value in template.items()
        if key != "id" and not isinstance(value, list)
    }

    if mode == "cartesian":
        keys = list(list_fields)
        choices = itertools.product(*(list_fields[key] for key in keys))
        combinations = [dict(zip(keys, values)) for values in choices]
    elif mode == "zip":
        lengths = {len(value) for value in list_fields.values()}
        if len(lengths) > 1:
            details = {key: len(value) for key, value in list_fields.items()}
            raise ValueError(f"Zip grid fields must have equal lengths: {details}")
        count = next(iter(lengths), 1)
        combinations = [
            {key: values[index] for key, values in list_fields.items()}
            for index in range(count)
        ]
    else:
        raise ValueError("grid mode must be 'cartesian' or 'zip'")

    if not combinations:
        raise ValueError(f"Template {template_id!r} contains an empty grid field")

    expanded = []
    for index, combination in enumerate(combinations):
        entry = {
            "id": f"{template_id}__grid{index:03d}",
            **scalar_fields,
            **combination,
        }
        assert_scalar_algorithm(entry)
        expanded.append(entry)
    return expanded


def assert_scalar_algorithm(entry: dict[str, Any]) -> None:
    for key, value in entry.items():
        if isinstance(value, (list, dict)):
            raise ValueError(
                f"Expanded algorithm {entry.get('id')!r} has non-scalar field {key!r}"
            )


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
    }


def make_benchpress_config(
    expanded: list[dict[str, Any]],
    template_id: str,
    run_name: str,
    fixed_data: FixedDataReference,
    independence_cache_dir: str | None = None,
) -> tuple[dict, str, str]:
    slug = safe_name(template_id)
    benchmark_title = safe_name(f"{run_name}__{slug}")
    prefix = f"{safe_name(run_name)}/{slug}/"
    expanded_entries = copy.deepcopy(expanded)
    if independence_cache_dir:
        for entry in expanded_entries:
            entry.setdefault("independence_cache_dir", independence_cache_dir)
    ids = [entry["id"] for entry in expanded_entries]
    config = {
        "benchmark_setup": [
            {
                "title": benchmark_title,
                "data": [
                    {
                        "graph_id": fixed_data.graph_id,
                        "parameters_id": None,
                        "data_id": fixed_data.data_id,
                        "seed_range": None,
                    }
                ],
                "evaluation": _evaluation(ids, prefix),
            }
        ],
        "resources": {
            "data": {},
            "graph": {},
            "parameters": {},
            "structure_learning_algorithms": {"notreks": expanded_entries},
        },
    }
    return config, benchmark_title, prefix


def prepare_hparam_run(
    repo_root: Path,
    grid_config_path: Path,
    run_dir: Path,
    grid_mode: str,
) -> tuple[list[ExpandedTemplate], FixedDataReference]:
    meta_config = load_json(grid_config_path)
    meta = meta_config.get("notreks_hparam", {})
    default_name = safe_name(run_dir.name)
    fixed_spec: FixedDataSpec = spec_from_meta(meta, default_name)
    fixed_reference = prepare_fixed_data(repo_root, run_dir, fixed_spec)
    try:
        cache_dir_text = str((run_dir / "independence_cache").relative_to(repo_root))
    except ValueError:
        cache_dir_text = str(run_dir / "independence_cache")
    templates = extract_notreks_templates(meta_config)

    expanded_dir = run_dir / "expanded_configs"
    expanded_dir.mkdir(parents=True, exist_ok=True)
    records: list[ExpandedTemplate] = []
    index_rows = []
    for template in templates:
        template_id = str(template["id"])
        expanded = expand_template(template, grid_mode)
        config, title, prefix = make_benchpress_config(
            expanded,
            template_id,
            fixed_spec.run_name,
            fixed_reference,
            cache_dir_text,
        )
        config_path = expanded_dir / f"{safe_name(template_id)}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        job_dir = Path("jobs") / f"{len(records):03d}_{safe_name(template_id)}"
        benchpress_dir = Path("results/output") / title / "benchmarks" / prefix
        record = ExpandedTemplate(
            template_id=template_id,
            relative_config_path=str(config_path.relative_to(run_dir)),
            algorithm_ids=tuple(item["id"] for item in expanded),
            benchmark_title=title,
            filename_prefix=prefix,
            relative_job_dir=str(job_dir),
            relative_joint_benchmarks_path=str(job_dir / "joint_benchmarks.csv"),
            relative_roc_data_path=str(job_dir / "ROC_data.csv"),
            benchpress_joint_benchmarks_path=str(benchpress_dir / "joint_benchmarks.csv"),
            benchpress_roc_data_path=str(benchpress_dir / "ROC_data.csv"),
        )
        records.append(record)
        for entry in expanded:
            index_rows.append(
                {
                    "template_id": template_id,
                    "algorithm_id": entry["id"],
                    "relative_config_path": str(config_path.relative_to(run_dir)),
                    "config": entry,
                }
            )

    (run_dir / "grid_index.json").write_text(json.dumps(index_rows, indent=2) + "\n")
    legacy_meta = run_dir / "grid_meta.json"
    if legacy_meta.exists():
        legacy_meta.unlink()
    summaries_dir = run_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "grid_config": _display_path(grid_config_path, repo_root),
                "grid_mode": grid_mode,
                "fixed_data": fixed_reference.__dict__,
                "templates": [record.template_id for record in records],
                "path_convention": "manifest paths prefixed with relative_ are resolved relative to this run directory",
            },
            indent=2,
        )
        + "\n"
    )
    return records, fixed_reference
