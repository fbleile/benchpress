#!/usr/bin/env python3
"""
Hyperparameter analysis for NOTREKS validation runs.

Expected inputs for tag <tag>:
  configs/notreks/expanded/<tag>_manifest.json
  results/output/notreks_<tag>_validation/benchmarks/notreks/<tag>/validation/joint_benchmarks.csv

Main outputs:
  results/notreks/hyperparam_analysis/<tag>/report.md
  results/notreks/hyperparam_analysis/<tag>/algorithm_summary.csv
  results/notreks/hyperparam_analysis/<tag>/best_configs.csv

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
KEY_HYPERPARAMS = [
    "dag_seq",
    "independence_alpha",
    "independence_correction",
    "threshold",
    "trek_reg",
    "trek_penalty_mu_mode",
    "lr",
    "regularizer_scale",
    "max_iter",
    "path_steps",
    "dag_reg",
    "mu_init",
    "mu_factor",
]
REPORT_METRICS = [
    "SHD_pattern",
    "precision_skel",
    "recall_skel",
    "F1_skel",
    "precision_pattern",
    "recall_pattern",
    "F1_pattern",
    "FPR_skel",
    "FNR_skel",
    "TPR_pattern",
    "FPR_pattern",
    "time",
]
EFFECT_HYPERPARAMS = [
    "independence_alpha",
    "threshold",
    "dag_seq",
    "independence_correction",
    "trek_reg",
    "trek_penalty_mu_mode",
]

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
        description="Analyze validation hyperparameters from joint_benchmarks.csv."
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
        "--joint-benchmarks",
        default=None,
        help="Optional joint_benchmarks.csv path. Default is inferred from the tag.",
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
            "best/config: use configured manifest threshold unless joint_benchmarks has threshold rows. "
            "all: do not collapse threshold rows for hyperparam summaries."
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
            "hyperparameters": hyperparams,
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


def load_joint_benchmarks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if not any(candidate in df.columns for candidate in ("id", "alg_id", "algorithm_id")):
        raise ValueError(
            f"joint_benchmarks.csv must contain an algorithm id column. Found columns: {list(df.columns)}"
        )

    return df


def _first_values(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return df[column].dropna().astype(str).unique().tolist()[:10]


def _normalize_key(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def join_joint_to_manifest(
    joint: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    joint_path: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    candidates = [
        ("id", "algorithm_id"),
        ("alg_id", "algorithm_id"),
        ("id", "path_id"),
        ("alg_id", "path_id"),
        ("algorithm_id", "algorithm_id"),
    ]
    best: tuple[int, int, str, str, pd.DataFrame] | None = None
    for priority, (left_key, right_key) in enumerate(candidates):
        if left_key not in joint.columns or right_key not in manifest.columns:
            continue
        left = joint.copy()
        right = manifest.copy()
        left["_joint_join_key"] = _normalize_key(left[left_key])
        right["_manifest_join_key"] = _normalize_key(right[right_key])
        merged = left.merge(
            right,
            left_on="_joint_join_key",
            right_on="_manifest_join_key",
            how="inner",
            suffixes=("_joint", "_manifest"),
        )
        matches = len(merged)
        if best is None or matches > best[0] or (matches == best[0] and priority < best[1]):
            best = (matches, priority, left_key, right_key, merged)

    if best is None or best[0] == 0:
        raise ValueError(
            "\n".join(
                [
                    "Could not join joint_benchmarks.csv to manifest.",
                    f"joint path: {joint_path}",
                    f"manifest path: {manifest_path}",
                    f"joint columns: {list(joint.columns)}",
                    f"manifest columns: {list(manifest.columns)}",
                    f"joint.id first values: {_first_values(joint, 'id')}",
                    f"joint.alg_id first values: {_first_values(joint, 'alg_id')}",
                    f"manifest.algorithm_id first values: {_first_values(manifest, 'algorithm_id')}",
                    f"manifest.path_id first values: {_first_values(manifest, 'path_id')}",
                ]
            )
        )

    matches, _, left_key, right_key, merged = best
    print(
        "Joined joint_benchmarks with manifest using "
        f"joint_benchmarks.{left_key} == manifest.{right_key} ({matches} rows)."
    )
    result_source = left_key if left_key in merged.columns else f"{left_key}_joint"
    merged["result_id"] = merged.get(result_source, merged["_joint_join_key"]).astype("string")
    merged["manifest_join_key"] = merged["_manifest_join_key"].astype("string")

    if "algorithm_id_manifest" in merged.columns:
        merged["algorithm_id"] = merged["algorithm_id_manifest"].astype(str)
    elif "algorithm_id" in merged.columns:
        merged["algorithm_id"] = merged["algorithm_id"].astype(str)
    elif f"{right_key}_manifest" in merged.columns:
        merged["algorithm_id"] = merged[f"{right_key}_manifest"].astype(str)
    else:
        raise ValueError("Joined table has no recoverable manifest algorithm_id column.")

    if "path_id_manifest" in merged.columns and "path_id" not in merged.columns:
        merged["path_id"] = merged["path_id_manifest"]
    return merged


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
            print(f"[warning] metric '{metric}' not found in joint_benchmarks.csv")
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


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    out = numerator / denominator
    out = out.where(denominator != 0)
    return out


def add_derived_precision_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    metric_specs = [
        ("pattern", "TP_pattern", "FP_pattern", "FN_pattern"),
        ("skel", "TP_skel", "FP_skel", "FN_skel"),
    ]
    for suffix, tp_col, fp_col, fn_col in metric_specs:
        if {tp_col, fp_col, fn_col}.issubset(out.columns):
            tp = pd.to_numeric(out[tp_col], errors="coerce")
            fp = pd.to_numeric(out[fp_col], errors="coerce")
            fn = pd.to_numeric(out[fn_col], errors="coerce")
            precision = _safe_ratio(tp, tp + fp)
            recall = _safe_ratio(tp, tp + fn)
            f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
            out[f"precision_{suffix}"] = precision
            out[f"recall_{suffix}"] = recall
            out[f"F1_{suffix}"] = f1
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


def add_evaluated_threshold(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "thresh" in out.columns:
        out["_threshold_source_row"] = np.where(
            pd.to_numeric(out["thresh"], errors="coerce").notna(),
            "joint_benchmarks_threshold",
            None,
        )
        return out
    out["_threshold_source_row"] = None
    if "curve_param" in out.columns and "curve_value" in out.columns:
        curve_param = out["curve_param"].astype("string").str.lower()
        mask = curve_param.isin(["threshold", "thresh"])
        out["thresh"] = np.nan
        out.loc[mask, "thresh"] = pd.to_numeric(out.loc[mask, "curve_value"], errors="coerce")
        out.loc[mask & out["thresh"].notna(), "_threshold_source_row"] = "joint_benchmarks_threshold"
    return out


def select_best_thresholds(
    rows: pd.DataFrame,
    metrics: list[str],
    primary_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate validation rows and select/report thresholds per algorithm.

    If joint_benchmarks.csv contains threshold rows:
        choose the best threshold per algorithm_id by primary_metric.

    If joint_benchmarks.csv has no 'thresh' column:
        use the configured threshold from the manifest, i.e.
        param.threshold or param.thresh. In this case no threshold optimization
        is possible; the table reports the threshold that was evaluated.
    """

    if primary_metric not in rows.columns:
        raise ValueError(f"Primary metric '{primary_metric}' not found in merged rows.")

    temp = add_evaluated_threshold(rows)
    if "thresh" in temp.columns:
        manifest_thresholds = temp.apply(get_config_threshold, axis=1)
        temp["thresh"] = pd.to_numeric(temp["thresh"], errors="coerce")
        missing_threshold = temp["thresh"].isna()
        temp.loc[missing_threshold, "thresh"] = manifest_thresholds[missing_threshold]
        temp.loc[missing_threshold & temp["thresh"].notna(), "_threshold_source_row"] = "manifest_config_threshold"

    # Case 1: no evaluated threshold column. Use manifest threshold.
    if "thresh" not in temp.columns or temp["thresh"].notna().sum() == 0:
        temp["thresh"] = temp.apply(get_config_threshold, axis=1)

        group_cols = ["method_family", "base_method", "algorithm_id", "thresh"]
        weight_col = "n_seeds" if "n_seeds" in temp.columns else None

        records: list[dict[str, Any]] = []

        for keys, group in temp.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)

            record = dict(zip(group_cols, keys))
            record["n_rows"] = int(len(group))

            if weight_col is not None:
                weights = group[weight_col]
                record["n_weight"] = float(
                    pd.to_numeric(weights, errors="coerce").fillna(0).sum()
                )
            else:
                weights = None
                record["n_weight"] = float(len(group))

            for metric in metrics:
                if metric in group.columns:
                    record[metric] = weighted_mean(group[metric], weights)

            record["selected_primary_metric"] = primary_metric
            record["selected_primary_value"] = record.get(primary_metric, np.nan)
            record["lower_is_better"] = lower_is_better(primary_metric)
            record["n_thresholds_considered"] = 1
            record["threshold_source"] = "manifest_config_threshold"

            records.append(record)

        threshold_curve = pd.DataFrame(records)

        if threshold_curve.empty:
            raise ValueError("Threshold table is empty after grouping.")

        primary_lower = lower_is_better(primary_metric)
        best_thresholds = threshold_curve.sort_values(
            ["method_family", primary_metric],
            ascending=[True, primary_lower],
        ).copy()

        return threshold_curve, best_thresholds

    # Case 2: joint_benchmarks.csv has threshold curve rows.
    temp = temp[temp["thresh"].notna()].copy()
    group_cols = ["method_family", "base_method", "algorithm_id", "thresh"]

    weight_col = "n_seeds" if "n_seeds" in temp.columns else None

    records: list[dict[str, Any]] = []

    for keys, group in temp.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        record = dict(zip(group_cols, keys))
        record["n_rows"] = int(len(group))

        if weight_col is not None:
            weights = group[weight_col]
            record["n_weight"] = float(
                pd.to_numeric(weights, errors="coerce").fillna(0).sum()
            )
        else:
            weights = None
            record["n_weight"] = float(len(group))

        for metric in metrics:
            if metric in group.columns:
                record[metric] = weighted_mean(group[metric], weights)
        sources = set(str(value) for value in group.get("_threshold_source_row", pd.Series()).dropna())
        record["threshold_source"] = (
            "joint_benchmarks_threshold" if "joint_benchmarks_threshold" in sources else "manifest_config_threshold"
        )

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
        chosen["threshold_source"] = (
            "joint_benchmarks_best_threshold"
            if chosen.get("threshold_source") == "joint_benchmarks_threshold"
            else "manifest_config_threshold"
        )

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

    If joint_benchmarks.csv has no thresh column, there is nothing to collapse.
    We attach the manifest threshold and continue.
    """

    if threshold_mode == "all":
        out = add_evaluated_threshold(rows)
        out["_threshold_selection"] = "all"
        if "thresh" not in out.columns:
            out["_selected_threshold"] = out.apply(get_config_threshold, axis=1)
        return out

    rows = add_evaluated_threshold(rows)
    if "thresh" in rows.columns:
        manifest_thresholds = rows.apply(get_config_threshold, axis=1)
        rows["thresh"] = pd.to_numeric(rows["thresh"], errors="coerce")
        missing_threshold = rows["thresh"].isna()
        rows.loc[missing_threshold, "thresh"] = manifest_thresholds[missing_threshold]
        rows.loc[missing_threshold & rows["thresh"].notna(), "_threshold_source_row"] = "manifest_config_threshold"

    # No evaluated threshold column: use manifest threshold.
    if "thresh" not in rows.columns or rows["thresh"].notna().sum() == 0:
        out = rows.copy()
        out["_selected_threshold"] = out.apply(get_config_threshold, axis=1)
        out["_threshold_selection"] = "manifest_param_threshold_no_threshold_curve"
        return out

    temp = rows.copy()
    temp["thresh_numeric"] = pd.to_numeric(temp["thresh"], errors="coerce")
    temp = temp[temp["thresh_numeric"].notna()].copy()

    chosen_rows = []
    primary_lower = lower_is_better(primary_metric)

    best_lookup = {}
    if not best_thresholds.empty:
        for _, r in best_thresholds.iterrows():
            if pd.notna(r["thresh"]):
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
                    group.loc[
                        (group["thresh_numeric"] - target).abs().idxmin(),
                        "thresh_numeric",
                    ]
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


def _metric_mean_col(metric: str) -> str:
    return f"{metric}_mean"


def _metric_std_col(metric: str) -> str:
    return f"{metric}_std"


def available_report_metrics(rows: pd.DataFrame, requested: list[str]) -> list[str]:
    metrics = []
    for metric in list(requested) + REPORT_METRICS:
        if metric in rows.columns and metric not in metrics:
            metrics.append(metric)
    return metrics


def _first_nonnull(series: pd.Series):
    valid = series.dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def build_best_configs(
    rows: pd.DataFrame,
    metrics: list[str],
    primary_metric: str,
    top_k: int,
) -> pd.DataFrame:
    group_cols = ["method_family", "base_method", "algorithm_id"]
    agg: dict[str, list[str] | str] = {}
    for metric in metrics:
        if metric in rows.columns:
            agg[metric] = ["mean", "std", "count"]
    for col in ["path_id", "result_id"]:
        if col in rows.columns:
            agg[col] = "first"
    hp_cols = [col for col in rows.columns if col.startswith("param.")]
    for col in hp_cols:
        agg[col] = "first"

    summary = rows.groupby(group_cols, dropna=False).agg(agg).reset_index()
    flat_cols = []
    for col in summary.columns:
        if isinstance(col, tuple):
            if col[1] in {"mean", "std", "count"}:
                flat_cols.append(f"{col[0]}_{col[1]}")
            else:
                flat_cols.append(col[0])
        else:
            flat_cols.append(col)
    summary.columns = flat_cols
    for col in hp_cols:
        if col in summary.columns:
            summary = summary.rename(columns={col: "hp." + col.removeprefix("param.")})

    primary_col = _metric_mean_col(primary_metric)
    if primary_col not in summary.columns:
        raise ValueError(f"Primary metric mean column not found after summarizing: {primary_col}")
    time_col = _metric_mean_col("time")
    sort_cols = [primary_col]
    ascending = [lower_is_better(primary_metric)]
    if time_col in summary.columns:
        sort_cols.append(time_col)
        ascending.append(True)
    sort_cols.append("algorithm_id")
    ascending.append(True)
    summary = summary.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    count_col = f"{primary_metric}_count"
    if count_col in summary.columns:
        summary["n_datasets"] = summary[count_col]
    return summary


def important_hp_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for key in KEY_HYPERPARAMS:
        col = f"hp.{key}"
        if col in df.columns:
            cols.append(col)
    return cols


def _format_value(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def common_settings(top_configs: pd.DataFrame, *, dominant_fraction: float = 0.7) -> tuple[list[str], list[str], pd.DataFrame]:
    hp_cols = [col for col in top_configs.columns if col.startswith("hp.")]
    n = len(top_configs)
    constants: list[str] = []
    dominant: list[str] = []
    count_rows = []
    if n == 0:
        return constants, dominant, pd.DataFrame()
    for col in hp_cols:
        counts = top_configs[col].fillna("<missing>").astype(str).value_counts(dropna=False)
        if counts.empty:
            continue
        value_counts = ", ".join(f"{value}: {count}" for value, count in counts.items())
        count_rows.append({"hyperparameter": col.removeprefix("hp."), "value counts among top configs": value_counts})
        top_value = counts.index[0]
        top_count = int(counts.iloc[0])
        label = f"{col.removeprefix('hp.')} = {top_value} ({top_count}/{n})"
        if top_count == n:
            constants.append(label)
        elif top_count / max(n, 1) >= dominant_fraction:
            dominant.append(label)
    return constants, dominant, pd.DataFrame(count_rows)


def compact_effect_table(configs: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    wanted_metrics = [metric for metric in ["SHD_pattern", "FPR_skel", "FNR_skel", "time"] if _metric_mean_col(metric) in configs.columns]
    for hp in EFFECT_HYPERPARAMS:
        col = f"hp.{hp}"
        if col not in configs.columns:
            continue
        for value, group in configs.groupby(col, dropna=False):
            row = {
                "hyperparameter": hp,
                "value": _format_value(value),
                "n_configs": int(len(group)),
            }
            for metric in wanted_metrics:
                row[f"mean_{metric}"] = float(pd.to_numeric(group[_metric_mean_col(metric)], errors="coerce").mean())
            rows.append(row)
    return pd.DataFrame(rows)


def interaction_table(configs: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    left_col = f"hp.{left}"
    right_col = f"hp.{right}"
    if left_col not in configs.columns or right_col not in configs.columns:
        return pd.DataFrame()
    metric_cols = [col for col in [_metric_mean_col("SHD_pattern"), _metric_mean_col("time")] if col in configs.columns]
    if not metric_cols:
        return pd.DataFrame()
    records = []
    for keys, group in configs.groupby([left_col, right_col], dropna=False):
        row = {
            left: _format_value(keys[0]),
            right: _format_value(keys[1]),
            "n_configs": int(len(group)),
        }
        for col in metric_cols:
            row[col] = float(pd.to_numeric(group[col], errors="coerce").mean())
        records.append(row)
    return pd.DataFrame(records).sort_values(metric_cols[0], ascending=True) if records else pd.DataFrame()


def baseline_comparison(best_configs: pd.DataFrame, primary_metric: str) -> tuple[pd.DataFrame, list[str]]:
    primary_col = _metric_mean_col(primary_metric)
    rows = []

    def pick(label: str, mask: pd.Series):
        subset = best_configs[mask].copy()
        if subset.empty:
            return
        subset = subset.sort_values([primary_col, _metric_mean_col("time") if _metric_mean_col("time") in subset else primary_col])
        row = subset.iloc[0]
        out = {"method": label, f"mean_{primary_metric}": row.get(primary_col, np.nan)}
        for metric in ["FPR_skel", "FNR_skel", "precision_skel", "recall_skel", "F1_skel", "time"]:
            col = _metric_mean_col(metric)
            if col in row:
                out[f"mean_{metric}"] = row[col]
        rows.append(out)

    pick("empty_graph", best_configs["method_family"].astype(str).eq("empty_graph"))
    pick("complete_undirected_graph", best_configs["method_family"].astype(str).eq("complete_undirected_graph"))
    pick("gcastle_pc", best_configs["method_family"].astype(str).eq("gcastle_pc"))
    pick("marginal_trek_graph", best_configs["method_family"].astype(str).eq("marginal_trek_graph"))
    pick("best_NOTREKS", best_configs["method_family"].astype(str).eq("notreks"))

    table = pd.DataFrame(rows)
    notes = []
    if table.empty or f"mean_{primary_metric}" not in table:
        return table, notes
    values = dict(zip(table["method"], table[f"mean_{primary_metric}"]))
    best = values.get("best_NOTREKS")
    if best is not None and pd.notna(best):
        trivial = [values.get("empty_graph"), values.get("complete_undirected_graph")]
        if all(value is not None and pd.notna(value) and best < value for value in trivial):
            notes.append("NOTREKS beats trivial baselines.")
        elif any(value is not None and pd.notna(value) and best >= value for value in trivial):
            notes.append("NOTREKS not meaningful: it does not beat at least one trivial baseline.")
        pc = values.get("gcastle_pc")
        if pc is not None and pd.notna(pc):
            gap = best - pc
            if best < pc:
                notes.append("NOTREKS beats PC.")
            elif abs(gap) <= 2:
                notes.append("NOTREKS close to PC.")
            else:
                notes.append("NOTREKS does not beat PC.")
    return table, notes


def propose_next_grid(top_notreks: pd.DataFrame) -> dict[str, list[Any]]:
    proposal: dict[str, list[Any]] = {}
    if top_notreks.empty:
        return proposal
    for key in KEY_HYPERPARAMS:
        col = f"hp.{key}"
        if col not in top_notreks.columns:
            continue
        values = top_notreks[col].dropna().tolist()
        if not values:
            continue
        counts = pd.Series(values).astype(str).value_counts()
        chosen = counts.index[:2].tolist()
        parsed = []
        for value in chosen:
            sample = next((item for item in values if str(item) == value), value)
            parsed.append(sample)
        proposal[key] = parsed

    if "independence_alpha" in proposal:
        alphas = sorted({float(value) for value in proposal["independence_alpha"]})
        expanded = set(alphas)
        for alpha in alphas:
            expanded.update([max(0.001, alpha / 2.0), min(0.95, alpha * 1.75)])
        proposal["independence_alpha"] = sorted(round(value, 6) for value in expanded)[:5]
    if "threshold" in proposal:
        thresholds = sorted({float(value) for value in proposal["threshold"]})
        expanded = set(thresholds)
        for threshold in thresholds:
            expanded.update([max(0.0, threshold - 0.1), threshold + 0.1])
        proposal["threshold"] = sorted(round(value, 6) for value in expanded)[:5]
    proposal.setdefault("independence_cache_dir", [None])
    return proposal


def write_report(
    path: Path,
    tag: str,
    manifest_path: Path,
    joint_path: Path,
    merged_rows: pd.DataFrame,
    best_configs: pd.DataFrame,
    top_configs: pd.DataFrame,
    top_notreks: pd.DataFrame,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    interaction_alpha_threshold: pd.DataFrame,
    interaction_alpha_dag: pd.DataFrame,
    baseline_table: pd.DataFrame,
    baseline_notes: list[str],
    next_grid: dict[str, list[Any]],
    metrics: list[str],
    primary_metric: str,
    top_k: int,
) -> None:
    primary_lower = lower_is_better(primary_metric)

    lines: list[str] = []

    lines.append(f"# Hyperparameter analysis for `{tag}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- Joint benchmarks: `{joint_path}`")
    lines.append(f"- Raw merged joint benchmark rows: `{len(merged_rows)}`")
    lines.append(f"- Metrics: `{', '.join(metrics)}`")
    lines.append(f"- Primary metric: `{primary_metric}`")
    lines.append(
        "- Precision is important because FPR can look small in sparse graphs due to many true non-edges. "
        "A useful method should have a good precision/recall tradeoff, not merely low FPR."
    )
    lines.append("")

    lines.append("## Best configs")
    lines.append("")
    for _, row in top_configs.iterrows():
        lines.append(f"Rank {int(row['rank'])}: {row['algorithm_id']}")
        lines.append(f"  method: {row['method_family']} / {row['base_method']}")
        if "path_id" in row and pd.notna(row["path_id"]):
            lines.append(f"  path_id: {row['path_id']}")
        for metric in [primary_metric, "SHD_pattern", "precision_skel", "recall_skel", "F1_skel", "precision_pattern", "recall_pattern", "F1_pattern", "FPR_skel", "FNR_skel", "TPR_pattern", "FPR_pattern", "time"]:
            mean_col = _metric_mean_col(metric)
            std_col = _metric_std_col(metric)
            if mean_col in row and pd.notna(row[mean_col]):
                text = f"  mean {metric}: {_format_value(row[mean_col])}"
                if std_col in row and pd.notna(row[std_col]):
                    text += f" (std {_format_value(row[std_col])})"
                lines.append(text)
        if "n_datasets" in row:
            lines.append(f"  number of datasets: {_format_value(row['n_datasets'])}")
        hp_cols = important_hp_columns(top_configs)
        if hp_cols:
            lines.append("  hyperparameters:")
            for col in hp_cols:
                if col in row and pd.notna(row[col]):
                    lines.append(f"    {col.removeprefix('hp.')}: {_format_value(row[col])}")
        lines.append("")

    lines.append("## Common settings among top configs")
    lines.append("")
    common_base = top_notreks if not top_notreks.empty else top_configs
    constants, dominant, counts = common_settings(common_base)
    lines.append("### Constant across top configs")
    lines.append("")
    lines.extend([f"- {item}" for item in constants] or ["_No constants._"])
    lines.append("")
    lines.append("### Dominant values among top configs")
    lines.append("")
    lines.extend([f"- {item}" for item in dominant] or ["_No dominant values at the 70% threshold._"])
    lines.append("")
    important_counts = counts[counts["hyperparameter"].isin(KEY_HYPERPARAMS)] if not counts.empty else counts
    lines.append(markdown_table(important_counts, max_rows=50))
    lines.append("")

    lines.append("## Trivial baseline comparison")
    lines.append("")
    if not baseline_table.empty:
        lines.append(markdown_table(baseline_table, max_rows=20))
        lines.append("")
        values = dict(zip(baseline_table["method"], baseline_table[f"mean_{primary_metric}"]))
        best = values.get("best_NOTREKS")
        if best is not None and pd.notna(best):
            for method in ["empty_graph", "complete_undirected_graph", "gcastle_pc"]:
                value = values.get(method)
                if value is not None and pd.notna(value):
                    lines.append(f"- best_NOTREKS - {method}: {_format_value(best - value)}")
        lines.extend([f"- {note}" for note in baseline_notes])
    else:
        lines.append("_No baseline rows available._")
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

    lines.append("## Compact hyperparameter effects")
    lines.append("")
    lines.append(markdown_table(effects, max_rows=80))
    lines.append("")
    if not interaction_alpha_threshold.empty:
        lines.append("### independence_alpha x threshold")
        lines.append("")
        lines.append(markdown_table(interaction_alpha_threshold, max_rows=30))
        lines.append("")
    if not interaction_alpha_dag.empty:
        lines.append("### independence_alpha x dag_seq")
        lines.append("")
        lines.append(markdown_table(interaction_alpha_dag, max_rows=30))
        lines.append("")

    lines.append("## Suggested next grid")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(next_grid, indent=2, default=str))
    lines.append("```")
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

    joint_path = (
        Path(args.joint_benchmarks)
        if args.joint_benchmarks
        else repo_root
        / "results"
        / "output"
        / f"notreks_{tag}_validation"
        / "benchmarks"
        / "notreks"
        / tag
        / "validation"
        / "joint_benchmarks.csv"
    )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo_root / "results" / "notreks" / "hyperparam_analysis" / tag
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for obsolete in [
        "all_merged_joint_rows.csv",
        "analysis_rows.csv",
        "threshold_curve_by_algorithm.csv",
        "best_threshold_by_algorithm.csv",
        "hyperparam_main_effects.csv",
        "numeric_correlations.csv",
    ]:
        (out_dir / obsolete).unlink(missing_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not joint_path.exists():
        raise FileNotFoundError(f"joint_benchmarks.csv not found: {joint_path}")

    manifest = load_manifest(manifest_path)
    joint = load_joint_benchmarks(joint_path)

    merged = join_joint_to_manifest(
        joint,
        manifest,
        joint_path=joint_path,
        manifest_path=manifest_path,
    )
    merged = add_derived_precision_metrics(merged)

    requested_metrics = available_report_metrics(merged, args.metrics)
    metric_map = resolve_metric_columns(merged, requested_metrics)
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
    best_configs = build_best_configs(analysis_rows, metrics, primary_metric, args.top_k)
    top_configs = best_configs.head(args.top_k).copy()
    top_notreks = best_configs[best_configs["method_family"].astype(str) == "notreks"].head(args.top_k).copy()
    effects = compact_effect_table(
        best_configs[best_configs["method_family"].astype(str) == "notreks"],
        metrics,
    )
    interaction_alpha_threshold = interaction_table(
        best_configs[best_configs["method_family"].astype(str) == "notreks"],
        "independence_alpha",
        "threshold",
    )
    interaction_alpha_dag = interaction_table(
        best_configs[best_configs["method_family"].astype(str) == "notreks"],
        "independence_alpha",
        "dag_seq",
    )
    baseline_table, baseline_notes = baseline_comparison(best_configs, primary_metric)
    next_grid = propose_next_grid(top_notreks)

    summary.to_csv(out_dir / "algorithm_summary.csv", index=False)
    best_configs.to_csv(out_dir / "best_configs.csv", index=False)
    if next_grid:
        (out_dir / "proposed_next_grid.json").write_text(
            json.dumps(next_grid, indent=2, default=str) + "\n"
        )

    report_path = out_dir / "report.md"
    write_report(
        path=report_path,
        tag=tag,
        manifest_path=manifest_path,
        joint_path=joint_path,
        merged_rows=merged,
        best_configs=best_configs,
        top_configs=top_configs,
        top_notreks=top_notreks,
        summary=summary,
        effects=effects,
        interaction_alpha_threshold=interaction_alpha_threshold,
        interaction_alpha_dag=interaction_alpha_dag,
        baseline_table=baseline_table,
        baseline_notes=baseline_notes,
        next_grid=next_grid,
        metrics=metrics,
        primary_metric=primary_metric,
        top_k=args.top_k,
    )

    print(f"Read manifest: {manifest_path}")
    print(f"Read joint benchmarks: {joint_path}")
    print(f"Wrote output directory: {out_dir}")
    print("")
    print("Key files:")
    print(f"  {out_dir / 'report.md'}")
    print(f"  {out_dir / 'algorithm_summary.csv'}")
    print(f"  {out_dir / 'best_configs.csv'}")
    if next_grid:
        print(f"  {out_dir / 'proposed_next_grid.json'}")
    print("")
    print("Open report:")
    print(f"  less {report_path}")


if __name__ == "__main__":
    main()
