#!/usr/bin/env python3
"""
Descriptive hyperparameter analysis for NOTREKS/Benchpress validation runs.

Example:
  python workflow/rules/structure_learning_algorithms/notreks/tools/hyperparam_analysis.py \
    --tag full_benchmark

The script combines:
  1. configs/notreks/expanded/<tag>_manifest.csv
  2. result/benchmark CSVs found under results/

It writes:
  results/notreks/hyperparam_analysis/<tag>/report.md
  results/notreks/hyperparam_analysis/<tag>/all_rows.csv
  results/notreks/hyperparam_analysis/<tag>/<family>_algorithm_summary.csv
  results/notreks/hyperparam_analysis/<tag>/<family>_hyperparam_effects.csv
  results/notreks/hyperparam_analysis/<tag>/<family>_correlations.csv

Interpretation:
  This is descriptive analysis, not causal proof. For factorial grids,
  main-effect contrasts are useful hints about which hyperparameters worked
  well, but interactions and noise can still matter.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_METRICS = ["SHD_pattern", "time"]
DEFAULT_PRIMARY = "SHD_pattern"

LOWER_IS_BETTER_HINTS = (
    "SHD",
    "FDR",
    "FPR",
    "FNR",
    "time",
    "runtime",
    "elapsed",
    "loss",
    "error",
    "distance",
)

HIGHER_IS_BETTER_HINTS = (
    "TPR",
    "recall",
    "precision",
    "auc",
    "accuracy",
    "f1",
)

ID_COL_CANDIDATES = [
    "algorithm_id",
    "alg_id",
    "method_id",
    "config_id",
    "id",
    "algorithm",
    "method",
]

FAMILY_COL_CANDIDATES = [
    "method_family",
    "family",
    "algorithm_family",
    "method_family_id",
]

PARAM_JSON_COL_CANDIDATES = [
    "parameters_json",
    "params_json",
    "hyperparameters_json",
    "config_json",
]


COMPACT_ID_RE = re.compile(
    r"(notreks|gcastle_pc|gcastle_direct_lingam|gcastle_lingam)__grid\d+"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter performance per method family."
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Experiment tag, e.g. smoke, full_benchmark, selected_full_benchmark.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest CSV path. Default: configs/notreks/expanded/<tag>_manifest.csv.",
    )
    parser.add_argument(
        "--results-glob",
        action="append",
        default=None,
        help=(
            "Glob for result CSVs. Can be repeated. "
            "Default searches likely benchmark/result/summary CSVs under results/."
        ),
    )
    parser.add_argument(
        "--all-csv",
        action="store_true",
        help="Scan all CSVs under results/. Slower, but useful if filenames are nonstandard.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to analyze. Default: SHD_pattern time.",
    )
    parser.add_argument(
        "--primary-metric",
        default=None,
        help="Primary metric for ranking. Default: first metric, usually SHD_pattern.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Default: "
            "results/notreks/hyperparam_analysis/<tag>."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of best/worst configs to show per method family.",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=25,
        help="Maximum number of unique values for a parameter to include in effect analysis.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum number of observations required for a parameter value.",
    )
    return parser.parse_args()


def find_existing_manifest(repo_root: Path, tag: str, explicit: str | None) -> Path:
    if explicit:
        path = repo_root / explicit
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path

    candidates = [
        repo_root / "configs" / "notreks" / "expanded" / f"{tag}_manifest.csv",
        repo_root / "configs" / "notreks" / "expanded" / f"selected_{tag}_manifest.csv",
        repo_root / "configs" / "notreks" / "expanded" / f"{tag}_benchmark_manifest.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find manifest automatically. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nPass --manifest explicitly."
    )


def read_csv_lenient(path: Path, nrows: int | None = None) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return None


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, new_key))
        else:
            out[new_key] = value
    return out


def parse_json_like(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, dict):
        return flatten_dict(value)
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return flatten_dict(parsed)
    return {}


def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_to_original:
            return lower_to_original[c.lower()]
    return None


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)

    id_col = pick_col(manifest, ID_COL_CANDIDATES)
    if id_col is None:
        raise ValueError(
            f"No algorithm id column found in manifest {path}. "
            f"Columns: {list(manifest.columns)}"
        )

    family_col = pick_col(manifest, FAMILY_COL_CANDIDATES)
    if family_col is None:
        # Infer from compact ids such as notreks__grid000.
        manifest["method_family"] = manifest[id_col].astype(str).str.replace(
            r"__grid\d+$", "", regex=True
        )
        family_col = "method_family"

    manifest = manifest.rename(
        columns={id_col: "algorithm_id", family_col: "method_family"}
    )

    param_json_col = pick_col(manifest, PARAM_JSON_COL_CANDIDATES)
    if param_json_col is not None:
        param_rows = [parse_json_like(v) for v in manifest[param_json_col]]
        params = pd.DataFrame(param_rows)
    else:
        reserved = {
            "algorithm_id",
            "method_family",
            "grid_index",
            "index",
            "rank",
            "selected",
        }
        param_cols = [c for c in manifest.columns if c not in reserved]
        params = manifest[param_cols].copy()

    params = params.add_prefix("param.")
    out = pd.concat(
        [manifest[["algorithm_id", "method_family"]].reset_index(drop=True),
         params.reset_index(drop=True)],
        axis=1,
    )
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def default_result_csv_candidates(repo_root: Path, all_csv: bool) -> list[Path]:
    results_root = repo_root / "results"
    if not results_root.exists():
        return []

    paths = list(results_root.rglob("*.csv"))
    if all_csv:
        return sorted(paths)

    keywords = ("benchmark", "summary", "result", "validation", "metric")
    keep = []
    for p in paths:
        s = str(p).lower()
        name = p.name.lower()
        if any(k in s or k in name for k in keywords):
            keep.append(p)
    return sorted(keep)


def find_result_csvs(
    repo_root: Path,
    result_globs: list[str] | None,
    all_csv: bool,
) -> list[Path]:
    if result_globs:
        paths: list[Path] = []
        for g in result_globs:
            paths.extend(repo_root.glob(g))
        return sorted({p for p in paths if p.is_file()})

    return default_result_csv_candidates(repo_root, all_csv)


def normalize_long_metrics(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}
    metric_col = lower.get("metric")
    value_col = lower.get("value") or lower.get("mean") or lower.get("score")

    if metric_col is None or value_col is None:
        return df

    present_metrics = set(df[metric_col].astype(str))
    if not any(m in present_metrics for m in metrics):
        return df

    id_cols = [c for c in df.columns if c not in {metric_col, value_col}]
    wide = (
        df.pivot_table(
            index=id_cols,
            columns=metric_col,
            values=value_col,
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns = [str(c) for c in wide.columns]
    return wide


def infer_algorithm_id_col(
    df: pd.DataFrame,
    manifest_ids: set[str],
) -> str | None:
    direct = pick_col(df, ID_COL_CANDIDATES)
    if direct is not None:
        return direct

    # Prefer a column that overlaps with manifest IDs.
    best_col = None
    best_overlap = 0
    for col in df.columns:
        if not pd.api.types.is_object_dtype(df[col]):
            continue
        values = set(df[col].dropna().astype(str))
        overlap = len(values & manifest_ids)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col

    if best_overlap > 0:
        return best_col

    return None


def extract_compact_id_from_row(row: pd.Series, manifest_ids: set[str]) -> str | None:
    for value in row.astype(str):
        if value in manifest_ids:
            return value
        match = COMPACT_ID_RE.search(value)
        if match:
            compact = match.group(0)
            if compact in manifest_ids:
                return compact
    return None


def load_result_rows(
    csvs: list[Path],
    metrics: list[str],
    manifest_ids: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in csvs:
        header = read_csv_lenient(path, nrows=5)
        if header is None:
            continue

        columns = set(header.columns)
        looks_wide = any(m in columns for m in metrics)
        looks_long = {"metric", "value"}.issubset({c.lower() for c in header.columns})

        if not looks_wide and not looks_long:
            continue

        df = read_csv_lenient(path)
        if df is None or df.empty:
            continue

        df = normalize_long_metrics(df, metrics)

        if not any(m in df.columns for m in metrics):
            continue

        id_col = infer_algorithm_id_col(df, manifest_ids)

        if id_col is not None:
            df = df.rename(columns={id_col: "algorithm_id"})
            df["algorithm_id"] = df["algorithm_id"].astype(str)
        else:
            df["algorithm_id"] = [
                extract_compact_id_from_row(row, manifest_ids)
                for _, row in df.iterrows()
            ]

        df = df[df["algorithm_id"].isin(manifest_ids)].copy()
        if df.empty:
            continue

        df["_source_csv"] = str(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No result CSVs could be matched to manifest algorithm ids.\n"
            "Try passing --results-glob explicitly, e.g.\n"
            "  --results-glob 'results/**/*.csv'\n"
            "or inspect benchmark output CSV column names."
        )

    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def metric_lower_is_better(metric: str) -> bool:
    m = metric.lower()
    if any(h.lower() in m for h in HIGHER_IS_BETTER_HINTS):
        return False
    if any(h.lower() in m for h in LOWER_IS_BETTER_HINTS):
        return True
    # Conservative default for error-like benchmark metrics.
    return True


def numeric_metric_frame(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = df.copy()
    for m in metrics:
        if m in out.columns:
            out[m] = pd.to_numeric(out[m], errors="coerce")
    return out


def algorithm_summary(
    rows: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    agg = {}
    for m in metrics:
        if m in rows.columns:
            agg[m] = ["mean", "std", "count"]

    summary = rows.groupby(
        ["method_family", "algorithm_id"],
        dropna=False,
    ).agg(agg)

    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()
    return summary


def param_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("param.")]


def compute_effects(
    rows: pd.DataFrame,
    metrics: list[str],
    max_levels: int,
    min_count: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    params = param_columns(rows)

    for p in params:
        series = rows[p].fillna("<missing>").astype(str)
        n_levels = series.nunique(dropna=False)
        if n_levels <= 1 or n_levels > max_levels:
            continue

        for metric in metrics:
            if metric not in rows.columns:
                continue

            temp = rows[[p, metric]].copy()
            temp[p] = temp[p].fillna("<missing>").astype(str)
            temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
            temp = temp.dropna(subset=[metric])
            if temp.empty:
                continue

            grouped = (
                temp.groupby(p)[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            grouped = grouped[grouped["count"] >= min_count]
            if grouped.shape[0] <= 1:
                continue

            lower_better = metric_lower_is_better(metric)
            best_idx = grouped["mean"].idxmin() if lower_better else grouped["mean"].idxmax()
            worst_idx = grouped["mean"].idxmax() if lower_better else grouped["mean"].idxmin()

            best = grouped.loc[best_idx]
            worst = grouped.loc[worst_idx]
            global_mean = temp[metric].mean()

            # Positive means "range between worst and best".
            effect_size = abs(float(worst["mean"]) - float(best["mean"]))

            records.append(
                {
                    "parameter": p.replace("param.", "", 1),
                    "metric": metric,
                    "lower_is_better": lower_better,
                    "n_levels": int(n_levels),
                    "best_value": best[p],
                    "best_mean": float(best["mean"]),
                    "best_count": int(best["count"]),
                    "worst_value": worst[p],
                    "worst_mean": float(worst["mean"]),
                    "worst_count": int(worst["count"]),
                    "global_mean": float(global_mean),
                    "best_minus_global": float(best["mean"] - global_mean),
                    "worst_minus_global": float(worst["mean"] - global_mean),
                    "effect_range": effect_size,
                }
            )

    return pd.DataFrame(records).sort_values(
        ["metric", "effect_range"],
        ascending=[True, False],
    )


def compute_correlations(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for p in param_columns(rows):
        x = pd.to_numeric(rows[p], errors="coerce")
        if x.notna().sum() < 3 or x.nunique(dropna=True) < 3:
            continue

        for metric in metrics:
            if metric not in rows.columns:
                continue
            y = pd.to_numeric(rows[metric], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                continue

            pearson = x[valid].corr(y[valid], method="pearson")
            spearman = x[valid].corr(y[valid], method="spearman")

            records.append(
                {
                    "parameter": p.replace("param.", "", 1),
                    "metric": metric,
                    "n": int(valid.sum()),
                    "pearson": float(pearson) if pd.notna(pearson) else np.nan,
                    "spearman": float(spearman) if pd.notna(spearman) else np.nan,
                    "interpretation": (
                        "positive means larger parameter values tend to increase the metric"
                    ),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["parameter", "metric", "n", "pearson", "spearman", "interpretation"]
        )

    out = pd.DataFrame(records)
    out["abs_spearman"] = out["spearman"].abs()
    return out.sort_values(["metric", "abs_spearman"], ascending=[True, False])


def df_to_md(df: pd.DataFrame, max_rows: int = 10, float_digits: int = 4) -> str:
    if df is None or df.empty:
        return "_No rows._"

    shown = df.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.{float_digits}g}"
            )
    return shown.to_markdown(index=False)


def write_report(
    out_dir: Path,
    tag: str,
    manifest_path: Path,
    result_csvs: list[Path],
    merged: pd.DataFrame,
    alg_summary: pd.DataFrame,
    family_outputs: dict[str, dict[str, pd.DataFrame]],
    metrics: list[str],
    primary_metric: str,
    top_k: int,
) -> None:
    lines: list[str] = []
    lines.append(f"# Hyperparameter analysis: `{tag}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- Matched result CSV files: `{len(result_csvs)}`")
    lines.append(f"- Rows after merge: `{len(merged)}`")
    lines.append(f"- Metrics: `{', '.join(metrics)}`")
    lines.append(f"- Primary metric: `{primary_metric}`")
    lines.append("")
    lines.append(
        "This report is descriptive. Main-effect contrasts are useful for "
        "factorial grids, but they are not causal proof; interactions and "
        "dataset noise can change conclusions."
    )
    lines.append("")

    lines.append("## Overall algorithm summary")
    lines.append("")
    lines.append(df_to_md(alg_summary.sort_values(
        f"{primary_metric}_mean",
        ascending=metric_lower_is_better(primary_metric),
    ), max_rows=30))
    lines.append("")

    for family, outputs in family_outputs.items():
        rows = merged[merged["method_family"] == family]
        lines.append(f"## Method family: `{family}`")
        lines.append("")
        lines.append(f"- Rows: `{len(rows)}`")
        lines.append(f"- Configs: `{rows['algorithm_id'].nunique()}`")
        lines.append("")

        fam_alg = outputs["algorithm_summary"]
        sort_col = f"{primary_metric}_mean"
        if sort_col in fam_alg.columns:
            fam_alg_sorted = fam_alg.sort_values(
                sort_col,
                ascending=metric_lower_is_better(primary_metric),
            )
        else:
            fam_alg_sorted = fam_alg

        lines.append("### Best configurations")
        lines.append("")
        lines.append(df_to_md(fam_alg_sorted, max_rows=top_k))
        lines.append("")

        lines.append("### Worst configurations")
        lines.append("")
        lines.append(df_to_md(fam_alg_sorted.tail(top_k), max_rows=top_k))
        lines.append("")

        effects = outputs["effects"]
        if not effects.empty:
            primary_effects = effects[effects["metric"] == primary_metric].sort_values(
                "effect_range",
                ascending=False,
            )
            lines.append("### Largest hyperparameter main effects")
            lines.append("")
            lines.append(df_to_md(primary_effects, max_rows=top_k))
            lines.append("")

            lines.append("### Effects for all requested metrics")
            lines.append("")
            lines.append(df_to_md(effects, max_rows=top_k * len(metrics)))
            lines.append("")
        else:
            lines.append("### Hyperparameter effects")
            lines.append("")
            lines.append("_No varying hyperparameters with enough observations._")
            lines.append("")

        corr = outputs["correlations"]
        if not corr.empty:
            lines.append("### Numeric-parameter correlations")
            lines.append("")
            lines.append(df_to_md(corr, max_rows=top_k * len(metrics)))
            lines.append("")
        else:
            lines.append("### Numeric-parameter correlations")
            lines.append("")
            lines.append("_No numeric parameters with at least three unique values._")
            lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    tag = args.tag
    metrics = args.metrics
    primary_metric = args.primary_metric or metrics[0]

    out_dir = Path(args.out_dir) if args.out_dir else (
        repo_root / "results" / "notreks" / "hyperparam_analysis" / tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = find_existing_manifest(repo_root, tag, args.manifest)
    manifest = load_manifest(manifest_path)
    manifest_ids = set(manifest["algorithm_id"].astype(str))

    csvs = find_result_csvs(repo_root, args.results_glob, args.all_csv)
    result_rows = load_result_rows(csvs, metrics, manifest_ids)

    merged = result_rows.merge(
        manifest,
        on="algorithm_id",
        how="left",
        suffixes=("", "_manifest"),
    )

    if "method_family" not in merged.columns:
        raise ValueError("Merged data has no method_family column.")

    merged = numeric_metric_frame(merged, metrics)
    merged.to_csv(out_dir / "all_rows.csv", index=False)

    alg_summary = algorithm_summary(merged, metrics)
    alg_summary.to_csv(out_dir / "overall_algorithm_summary.csv", index=False)

    family_outputs: dict[str, dict[str, pd.DataFrame]] = {}

    for family, family_rows in merged.groupby("method_family", dropna=False):
        family_name = str(family)
        safe_family = re.sub(r"[^A-Za-z0-9_.-]+", "_", family_name)

        fam_alg_summary = algorithm_summary(family_rows, metrics)
        effects = compute_effects(
            family_rows,
            metrics=metrics,
            max_levels=args.max_levels,
            min_count=args.min_count,
        )
        correlations = compute_correlations(family_rows, metrics)

        fam_alg_summary.to_csv(
            out_dir / f"{safe_family}_algorithm_summary.csv",
            index=False,
        )
        effects.to_csv(
            out_dir / f"{safe_family}_hyperparam_effects.csv",
            index=False,
        )
        correlations.to_csv(
            out_dir / f"{safe_family}_correlations.csv",
            index=False,
        )

        family_outputs[family_name] = {
            "algorithm_summary": fam_alg_summary,
            "effects": effects,
            "correlations": correlations,
        }

    write_report(
        out_dir=out_dir,
        tag=tag,
        manifest_path=manifest_path,
        result_csvs=csvs,
        merged=merged,
        alg_summary=alg_summary,
        family_outputs=family_outputs,
        metrics=metrics,
        primary_metric=primary_metric,
        top_k=args.top_k,
    )

    print(f"Hyperparameter analysis written to: {out_dir}")
    print(f"Report: {out_dir / 'report.md'}")
    print("")
    print("Matched result rows:", len(merged))
    print("Method families:", ", ".join(sorted(map(str, merged["method_family"].dropna().unique()))))
    print("")
    print("Open the report with:")
    print(f"  less {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()