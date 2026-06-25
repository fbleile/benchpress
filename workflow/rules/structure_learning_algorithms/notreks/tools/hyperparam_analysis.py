#!/usr/bin/env python3
"""
Hyperparameter + threshold analysis for NOTREKS validation runs.

Expected inputs for tag <tag>:
  configs/notreks/expanded/<tag>_manifest.json
  results/output/notreks_<tag>_validation/benchmarks/notreks/<tag>/validation/ROC_data.csv

Main outputs:
  results/notreks/hyperparam_analysis/<tag>/report.md
  results/notreks/hyperparam_analysis/<tag>/best_threshold_by_algorithm.csv
  results/notreks/hyperparam_analysis/<tag>/threshold_curve_by_algorithm.csv
  results/notreks/hyperparam_analysis/<tag>/algorithm_summary.csv
  results/notreks/hyperparam_analysis/<tag>/hyperparam_main_effects.csv
  results/notreks/hyperparam_analysis/<tag>/numeric_correlations.csv

Example:
  python workflow/rules/structure_learning_algorithms/notreks/tools/hyperparam_analysis.py \
    --tag hyperparam \
    --primary-metric SHD_pattern \
    --metrics SHD_pattern time TPR_pattern FPR_pattern
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_METRICS = ["SHD_pattern", "time"]
DEFAULT_PRIMARY_METRIC = "SHD_pattern"

LOWER_IS_BETTER_HINTS = [
    "shd",
    "fpr",
    "fdr",
    "fnr",
    "time",
    "runtime",
    "loss",
    "error",
    "distance",
]

HIGHER_IS_BETTER_HINTS = [
    "tpr",
    "recall",
    "precision",
    "accuracy",
    "f1",
    "auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze validation hyperparameters and select best thresholds."
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Experiment tag, e.g. smoke, hyperparam, full_benchmark.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest JSON path. Default: configs/notreks/expanded/<tag>_manifest.json.",
    )
    parser.add_argument(
        "--roc-data",
        default=None,
        help=(
            "Optional ROC_data.csv path. Default: "
            "results/output/notreks_<tag>_validation/benchmarks/notreks/<tag>/validation/ROC_data.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: results/notreks/hyperparam_analysis/<tag>.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to analyze. Default: SHD_pattern time.",
    )
    parser.add_argument(
        "--primary-metric",
        default=DEFAULT_PRIMARY_METRIC,
        help="Primary validation metric used for threshold and config ranking.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of best/worst configs/effects shown per family in the report.",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=25,
        help="Maximum number of unique parameter levels for main-effect analysis.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["best", "config", "all"],
        default="best",
        help=(
            "best: choose best threshold per algorithm by primary metric. "
            "config: use threshold/thresh from manifest if available, otherwise best. "
            "all: do not collapse ROC rows for hyperparam summaries."
        ),
    )
    return parser.parse_args()


def lower_is_better(metric: str) -> bool:
    m = metric.lower()

    if any(x in m for x in HIGHER_IS_BETTER_HINTS):
        return False

    if any(x in m for x in LOWER_IS_BETTER_HINTS):
        return True

    # Conservative default for benchmark/error metrics.
    return True


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flatten_dict(value, new_key))
        else:
            out[new_key] = value

    return out


def parse_json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, float) and math.isnan(value):
        return {}

    if isinstance(value, dict):
        return value

    text = str(value).strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except Exception:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def load_manifest(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text())

    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a JSON list: {path}")

    rows: list[dict[str, Any]] = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        algorithm_id = item.get("algorithm_id")
        if algorithm_id is None:
            continue

        hyperparams = item.get("hyperparameters")
        if not isinstance(hyperparams, dict):
            hyperparams = parse_json_dict(item.get("hyperparameters_json"))

        row: dict[str, Any] = {
            "algorithm_id": str(algorithm_id),
            "path_id": item.get("path_id"),
            "method_family": item.get("method_family"),
            "base_method": item.get("base_method"),
            "phase": item.get("phase"),
            "config_path": item.get("config_path"),
        }

        for key, value in flatten_dict(hyperparams).items():
            row[f"param.{key}"] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No manifest rows loaded from {path}")

    if "method_family" not in df.columns or df["method_family"].isna().all():
        df["method_family"] = df["algorithm_id"].str.replace(r"__grid\d+$", "", regex=True)

    if "base_method" not in df.columns:
        df["base_method"] = df["method_family"]

    return df


def load_roc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if "alg_id" not in df.columns:
        raise ValueError(
            f"ROC_data.csv must contain column 'alg_id'. Found columns: {list(df.columns)}"
        )

    df["alg_id"] = df["alg_id"].astype(str)

    return df


def resolve_metric_columns(df: pd.DataFrame, requested: list[str]) -> dict[str, str]:
    """
    Map public metric names such as SHD_pattern to existing columns such as
    SHD_pattern_mean.
    """
    lower_to_real = {c.lower(): c for c in df.columns}
    out: dict[str, str] = {}

    for metric in requested:
        candidates = [
            metric,
            f"{metric}_mean",
        ]

        if metric.endswith("_mean"):
            candidates.append(metric.removesuffix("_mean"))

        found = None

        for candidate in candidates:
            if candidate in df.columns:
                found = candidate
                break

            if candidate.lower() in lower_to_real:
                found = lower_to_real[candidate.lower()]
                break

        if found is None:
            print(f"[warning] metric '{metric}' not found in ROC_data.csv")
        else:
            public_name = metric.removesuffix("_mean")
            out[public_name] = found

    if not out:
        raise ValueError(
            f"None of the requested metrics were found: {requested}\n"
            f"Available columns: {list(df.columns)}"
        )

    return out


def coerce_metric_columns(df: pd.DataFrame, metric_map: dict[str, str]) -> pd.DataFrame:
    """
    Adds canonical public metric columns, e.g. SHD_pattern from SHD_pattern_mean.
    """
    out = df.copy()

    for public_name, source_col in metric_map.items():
        out[public_name] = pd.to_numeric(out[source_col], errors="coerce")

    return out


def metric_list_from_map(metric_map: dict[str, str]) -> list[str]:
    return list(metric_map.keys())


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    values = pd.to_numeric(values, errors="coerce")

    if weights is None:
        return float(values.mean())

    weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)

    if not valid.any():
        return float(values.mean())

    return float(np.average(values[valid], weights=weights[valid]))


def get_config_threshold(row: pd.Series) -> float | None:
    for key in ["param.threshold", "param.thresh"]:
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                pass

    return None


def select_best_thresholds(
    rows: pd.DataFrame,
    metrics: list[str],
    primary_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate ROC rows by algorithm_id × thresh and select the best threshold
    per algorithm_id according to the primary validation metric.
    """
    if "thresh" not in rows.columns:
        raise ValueError("Cannot select thresholds: ROC_data.csv has no 'thresh' column.")

    if primary_metric not in rows.columns:
        raise ValueError(f"Primary metric '{primary_metric}' not found in merged rows.")

    group_cols = ["method_family", "base_method", "algorithm_id", "thresh"]

    temp = rows.copy()
    temp["threshold"] = pd.to_numeric(temp["thresh"], errors="coerce")

    weight_col = "n_seeds" if "n_seeds" in temp.columns else None

    records: list[dict[str, Any]] = []

    for keys, group in temp.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        record = dict(zip(group_cols, keys))
        record["n_rows"] = int(len(group))

        if weight_col is not None:
            weights = group[weight_col]
            record["n_weight"] = float(pd.to_numeric(weights, errors="coerce").fillna(0).sum())
        else:
            weights = None
            record["n_weight"] = float(len(group))

        for metric in metrics:
            if metric in group.columns:
                record[metric] = weighted_mean(group[metric], weights)

        records.append(record)

    threshold_curve = pd.DataFrame(records)

    if threshold_curve.empty:
        raise ValueError("Threshold curve is empty after grouping.")

    primary_lower = lower_is_better(primary_metric)

    best_records = []

    for _, group in threshold_curve.groupby(
        ["method_family", "base_method", "algorithm_id"],
        dropna=False,
    ):
        metric_values = pd.to_numeric(group[primary_metric], errors="coerce")

        if metric_values.notna().sum() == 0:
            continue

        idx = metric_values.idxmin() if primary_lower else metric_values.idxmax()
        chosen = group.loc[idx].copy()

        chosen["selected_primary_metric"] = primary_metric
        chosen["selected_primary_value"] = chosen[primary_metric]
        chosen["lower_is_better"] = primary_lower
        chosen["n_thresholds_considered"] = int(group["thresh"].nunique(dropna=True))

        best_records.append(chosen)

    best_thresholds = pd.DataFrame(best_records)

    if not best_thresholds.empty:
        best_thresholds = best_thresholds.sort_values(
            ["method_family", primary_metric],
            ascending=[True, primary_lower],
        )

    return threshold_curve, best_thresholds


def collapse_rows_for_analysis(
    rows: pd.DataFrame,
    best_thresholds: pd.DataFrame,
    primary_metric: str,
    threshold_mode: str,
) -> pd.DataFrame:
    """
    Returns the row table used for algorithm summaries/effects.

    best:
      Keep only ROC rows whose threshold is selected as best per algorithm.
    config:
      Keep ROC rows closest to manifest threshold/thresh if available,
      otherwise best threshold.
    all:
      Keep all ROC rows.
    """
    if threshold_mode == "all":
        out = rows.copy()
        out["_threshold_selection"] = "all"
        return out

    if "thresh" not in rows.columns:
        out = rows.copy()
        out["_threshold_selection"] = "no_thresh_column"
        return out

    temp = rows.copy()
    temp["thresh_numeric"] = pd.to_numeric(temp["thresh"], errors="coerce")

    chosen_rows = []
    primary_lower = lower_is_better(primary_metric)

    best_lookup = {}
    if not best_thresholds.empty:
        for _, r in best_thresholds.iterrows():
            best_lookup[str(r["algorithm_id"])] = float(r["thresh"])

    for algorithm_id, group in temp.groupby("algorithm_id", dropna=False):
        algorithm_id = str(algorithm_id)
        group = group.copy()

        selected_threshold = None
        selection_reason = None

        if threshold_mode == "config":
            target = get_config_threshold(group.iloc[0])
            if target is not None and group["thresh_numeric"].notna().any():
                selected_threshold = float(
                    group.loc[(group["thresh_numeric"] - target).abs().idxmin(), "thresh_numeric"]
                )
                selection_reason = "closest_to_config_threshold"

        if selected_threshold is None:
            if algorithm_id in best_lookup:
                selected_threshold = best_lookup[algorithm_id]
                selection_reason = "best_validation_threshold"
            else:
                metric_values = pd.to_numeric(group[primary_metric], errors="coerce")
                idx = metric_values.idxmin() if primary_lower else metric_values.idxmax()
                selected_threshold = float(group.loc[idx, "thresh_numeric"])
                selection_reason = "best_validation_threshold_fallback"

        keep = group[
            np.isclose(
                group["thresh_numeric"].astype(float),
                float(selected_threshold),
                equal_nan=False,
            )
        ].copy()

        if keep.empty:
            idx = (group["thresh_numeric"] - selected_threshold).abs().idxmin()
            keep = group.loc[[idx]].copy()

        keep["_selected_threshold"] = selected_threshold
        keep["_threshold_selection"] = selection_reason
        chosen_rows.append(keep)

    out = pd.concat(chosen_rows, ignore_index=True, sort=False)
    out = out.drop(columns=["thresh_numeric"], errors="ignore")

    return out


def parameter_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "param.id",
        "param.independence_cache_dir",
        "param.timeout",
    }

    cols = []

    for col in df.columns:
        if not col.startswith("param."):
            continue

        if col in skip:
            continue

        cols.append(col)

    return cols


def algorithm_summary(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    agg = {}

    for metric in metrics:
        if metric in rows.columns:
            agg[metric] = ["mean", "std", "count"]

    summary = (
        rows.groupby(["method_family", "base_method", "algorithm_id"], dropna=False)
        .agg(agg)
        .reset_index()
    )

    summary.columns = [
        "_".join([x for x in col if x]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    # Add selected threshold if rows were collapsed.
    if "_selected_threshold" in rows.columns:
        th = (
            rows.groupby(["method_family", "base_method", "algorithm_id"], dropna=False)["_selected_threshold"]
            .first()
            .reset_index()
            .rename(columns={"_selected_threshold": "selected_threshold"})
        )
        summary = summary.merge(th, on=["method_family", "base_method", "algorithm_id"], how="left")

    return summary


def compute_main_effects(
    rows: pd.DataFrame,
    metrics: list[str],
    max_levels: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for family, fam in rows.groupby("method_family", dropna=False):
        for param in parameter_columns(fam):
            values = fam[param].fillna("<missing>").astype(str)
            n_levels = values.nunique(dropna=False)

            if n_levels <= 1 or n_levels > max_levels:
                continue

            for metric in metrics:
                if metric not in fam.columns:
                    continue

                temp = fam[[param, metric]].copy()
                temp[param] = temp[param].fillna("<missing>").astype(str)
                temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
                temp = temp.dropna(subset=[metric])

                if temp.empty:
                    continue

                grouped = (
                    temp.groupby(param, dropna=False)[metric]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )

                if len(grouped) <= 1:
                    continue

                metric_lower = lower_is_better(metric)
                best_idx = grouped["mean"].idxmin() if metric_lower else grouped["mean"].idxmax()
                worst_idx = grouped["mean"].idxmax() if metric_lower else grouped["mean"].idxmin()

                best = grouped.loc[best_idx]
                worst = grouped.loc[worst_idx]
                global_mean = float(temp[metric].mean())

                records.append(
                    {
                        "method_family": family,
                        "parameter": param.replace("param.", "", 1),
                        "metric": metric,
                        "lower_is_better": metric_lower,
                        "n_levels": int(n_levels),
                        "best_value": best[param],
                        "best_mean": float(best["mean"]),
                        "best_count": int(best["count"]),
                        "worst_value": worst[param],
                        "worst_mean": float(worst["mean"]),
                        "worst_count": int(worst["count"]),
                        "global_mean": global_mean,
                        "effect_range": float(abs(worst["mean"] - best["mean"])),
                    }
                )

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)

    return out.sort_values(
        ["method_family", "metric", "effect_range"],
        ascending=[True, True, False],
    )


def compute_numeric_correlations(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for family, fam in rows.groupby("method_family", dropna=False):
        for param in parameter_columns(fam):
            x = pd.to_numeric(fam[param], errors="coerce")

            if x.notna().sum() < 3 or x.nunique(dropna=True) < 3:
                continue

            for metric in metrics:
                if metric not in fam.columns:
                    continue

                y = pd.to_numeric(fam[metric], errors="coerce")
                valid = x.notna() & y.notna()

                if valid.sum() < 3:
                    continue

                pearson = x[valid].corr(y[valid], method="pearson")
                spearman = x[valid].corr(y[valid], method="spearman")

                records.append(
                    {
                        "method_family": family,
                        "parameter": param.replace("param.", "", 1),
                        "metric": metric,
                        "n": int(valid.sum()),
                        "pearson": float(pearson) if pd.notna(pearson) else np.nan,
                        "spearman": float(spearman) if pd.notna(spearman) else np.nan,
                        "abs_spearman": abs(float(spearman)) if pd.notna(spearman) else np.nan,
                        "direction_note": "positive means larger parameter values increase the metric",
                    }
                )

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)

    return out.sort_values(
        ["method_family", "metric", "abs_spearman"],
        ascending=[True, True, False],
    )


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No rows._"

    shown = df.head(max_rows).copy()

    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.5g}")
        else:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else str(x))

    columns = list(shown.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    body = []
    for _, row in shown.iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in columns) + " |")

    return "\n".join([header, separator] + body)


def write_report(
    path: Path,
    tag: str,
    manifest_path: Path,
    roc_path: Path,
    all_rows: pd.DataFrame,
    analysis_rows: pd.DataFrame,
    threshold_curve: pd.DataFrame,
    best_thresholds: pd.DataFrame,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    correlations: pd.DataFrame,
    metrics: list[str],
    primary_metric: str,
    threshold_mode: str,
    top_k: int,
) -> None:
    primary_lower = lower_is_better(primary_metric)

    lines: list[str] = []

    lines.append(f"# Hyperparameter analysis for `{tag}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- ROC data: `{roc_path}`")
    lines.append(f"- Raw merged ROC rows: `{len(all_rows)}`")
    lines.append(f"- Rows used for hyperparam analysis: `{len(analysis_rows)}`")
    lines.append(f"- Metrics: `{', '.join(metrics)}`")
    lines.append(f"- Primary metric: `{primary_metric}`")
    lines.append(f"- Threshold mode: `{threshold_mode}`")
    lines.append("")
    lines.append("## Threshold selection")
    lines.append("")
    lines.append(
        f"For each `algorithm_id`, thresholds are selected by the validation metric `{primary_metric}`. "
        "Lower is better for error-like metrics such as SHD/time/FPR; higher is better for TPR/precision/recall."
    )
    lines.append("")

    threshold_cols = [
        c for c in [
            "method_family",
            "base_method",
            "algorithm_id",
            "thresh",
            primary_metric,
            "time",
            "n_rows",
            "n_weight",
            "n_thresholds_considered",
        ]
        if c in best_thresholds.columns
    ]

    lines.append("### Best threshold per algorithm")
    lines.append("")
    if not best_thresholds.empty:
        best_thresholds_sorted = best_thresholds.sort_values(
            ["method_family", primary_metric],
            ascending=[True, primary_lower],
        )
        lines.append(markdown_table(best_thresholds_sorted[threshold_cols], max_rows=100))
    else:
        lines.append("_No threshold rows._")
    lines.append("")

    lines.append("## Overall algorithm summary")
    lines.append("")
    sort_col = f"{primary_metric}_mean"
    if sort_col in summary.columns:
        summary_sorted = summary.sort_values(sort_col, ascending=primary_lower)
    elif primary_metric in summary.columns:
        summary_sorted = summary.sort_values(primary_metric, ascending=primary_lower)
    else:
        summary_sorted = summary

    lines.append(markdown_table(summary_sorted, max_rows=50))
    lines.append("")

    for family in sorted(analysis_rows["method_family"].dropna().astype(str).unique()):
        lines.append(f"## `{family}`")
        lines.append("")

        fam_summary = summary_sorted[summary_sorted["method_family"].astype(str) == family]

        lines.append("### Best configurations")
        lines.append("")
        lines.append(markdown_table(fam_summary, max_rows=top_k))
        lines.append("")

        lines.append("### Worst configurations")
        lines.append("")
        lines.append(markdown_table(fam_summary.tail(top_k), max_rows=top_k))
        lines.append("")

        lines.append("### Largest main effects")
        lines.append("")
        if not effects.empty:
            fam_effects = effects[
                (effects["method_family"].astype(str) == family)
                & (effects["metric"] == primary_metric)
            ].sort_values("effect_range", ascending=False)

            lines.append(markdown_table(fam_effects, max_rows=top_k))
        else:
            lines.append("_No varying hyperparameters with enough observations._")
        lines.append("")

        lines.append("### Numeric correlations")
        lines.append("")
        if not correlations.empty:
            fam_corr = correlations[
                (correlations["method_family"].astype(str) == family)
                & (correlations["metric"] == primary_metric)
            ].sort_values("abs_spearman", ascending=False)

            lines.append(markdown_table(fam_corr, max_rows=top_k))
        else:
            lines.append("_No numeric hyperparameters with enough variation._")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- This is validation analysis. Selecting thresholds by the same validation data is okay for model selection, "
        "but final performance should be reported on separate benchmark/test data."
    )
    lines.append(
        "- Main effects are descriptive. They are useful hints, not causal claims, because hyperparameter interactions can matter."
    )
    lines.append(
        "- Pareto analysis is intentionally not included here; selection is by the primary validation metric."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    tag = args.tag

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else repo_root / "configs" / "notreks" / "expanded" / f"{tag}_manifest.json"
    )

    roc_path = (
        Path(args.roc_data)
        if args.roc_data
        else repo_root
        / "results"
        / "output"
        / f"notreks_{tag}_validation"
        / "benchmarks"
        / "notreks"
        / tag
        / "validation"
        / "ROC_data.csv"
    )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo_root / "results" / "notreks" / "hyperparam_analysis" / tag
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not roc_path.exists():
        raise FileNotFoundError(f"ROC_data.csv not found: {roc_path}")

    manifest = load_manifest(manifest_path)
    roc = load_roc(roc_path)

    merged = roc.merge(
        manifest,
        left_on="id",
        right_on="algorithm_id",
        how="inner",
    )

    if merged.empty:
        roc_ids = set(roc["alg_id"].astype(str))
        manifest_ids = set(manifest["algorithm_id"].astype(str))

        raise ValueError(
            "No rows after joining ROC_data.csv with manifest.json using "
            "ROC_data.alg_id == manifest.algorithm_id.\n"
            f"ROC ids: {len(roc_ids)}\n"
            f"Manifest ids: {len(manifest_ids)}\n"
            f"Matched ids: {len(roc_ids & manifest_ids)}\n"
            f"Example ROC-only ids: {sorted(roc_ids - manifest_ids)[:10]}\n"
            f"Example manifest-only ids: {sorted(manifest_ids - roc_ids)[:10]}"
        )

    metric_map = resolve_metric_columns(merged, args.metrics)
    merged = coerce_metric_columns(merged, metric_map)
    metrics = metric_list_from_map(metric_map)

    primary_metric = args.primary_metric.removesuffix("_mean")
    if primary_metric not in metrics:
        if primary_metric in merged.columns:
            metrics = [primary_metric] + [m for m in metrics if m != primary_metric]
        else:
            print(
                f"[warning] primary metric '{args.primary_metric}' not found. "
                f"Using '{metrics[0]}' instead."
            )
            primary_metric = metrics[0]

    threshold_curve, best_thresholds = select_best_thresholds(
        rows=merged,
        metrics=metrics,
        primary_metric=primary_metric,
    )

    analysis_rows = collapse_rows_for_analysis(
        rows=merged,
        best_thresholds=best_thresholds,
        primary_metric=primary_metric,
        threshold_mode=args.threshold_mode,
    )

    summary = algorithm_summary(analysis_rows, metrics)
    effects = compute_main_effects(
        analysis_rows,
        metrics=metrics,
        max_levels=args.max_levels,
    )
    correlations = compute_numeric_correlations(analysis_rows, metrics)

    merged.to_csv(out_dir / "all_merged_roc_rows.csv", index=False)
    analysis_rows.to_csv(out_dir / "analysis_rows.csv", index=False)
    threshold_curve.to_csv(out_dir / "threshold_curve_by_algorithm.csv", index=False)
    best_thresholds.to_csv(out_dir / "best_threshold_by_algorithm.csv", index=False)
    summary.to_csv(out_dir / "algorithm_summary.csv", index=False)
    effects.to_csv(out_dir / "hyperparam_main_effects.csv", index=False)
    correlations.to_csv(out_dir / "numeric_correlations.csv", index=False)

    report_path = out_dir / "report.md"
    write_report(
        path=report_path,
        tag=tag,
        manifest_path=manifest_path,
        roc_path=roc_path,
        all_rows=merged,
        analysis_rows=analysis_rows,
        threshold_curve=threshold_curve,
        best_thresholds=best_thresholds,
        summary=summary,
        effects=effects,
        correlations=correlations,
        metrics=metrics,
        primary_metric=primary_metric,
        threshold_mode=args.threshold_mode,
        top_k=args.top_k,
    )

    print(f"Read manifest: {manifest_path}")
    print(f"Read ROC data: {roc_path}")
    print(f"Wrote output directory: {out_dir}")
    print("")
    print("Key files:")
    print(f"  {out_dir / 'report.md'}")
    print(f"  {out_dir / 'best_threshold_by_algorithm.csv'}")
    print(f"  {out_dir / 'threshold_curve_by_algorithm.csv'}")
    print(f"  {out_dir / 'algorithm_summary.csv'}")
    print("")
    print("Open report:")
    print(f"  less {report_path}")


if __name__ == "__main__":
    main()