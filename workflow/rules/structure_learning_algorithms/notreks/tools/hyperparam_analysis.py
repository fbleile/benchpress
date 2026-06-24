#!/usr/bin/env python3
"""
Hyperparameter analysis for NOTREKS validation runs.

Inputs for tag <tag>:
  configs/notreks/expanded/<tag>_manifest.json
  results/output/notreks_<tag>_validation/benchmarks/notreks/<tag>/validation/ROC_data.csv

Example:
  python workflow/rules/structure_learning_algorithms/notreks/tools/hyperparam_analysis.py \
    --tag smoke

  python workflow/rules/structure_learning_algorithms/notreks/tools/hyperparam_analysis.py \
    --tag full_benchmark \
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

LOWER_IS_BETTER_HINTS = [
    "shd",
    "fpr",
    "fdr",
    "fnr",
    "time",
    "runtime",
    "loss",
    "error",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--roc-data", default=None)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--primary-metric", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-levels", type=int, default=25)
    parser.add_argument(
        "--threshold-policy",
        choices=["auto", "all", "best"],
        default="auto",
        help=(
            "auto: use config threshold/thresh if available, otherwise best ROC row. "
            "all: keep all ROC rows. "
            "best: choose best row per id by primary metric."
        ),
    )
    return parser.parse_args()


def lower_is_better(metric: str) -> bool:
    m = metric.lower()
    if any(x in m for x in HIGHER_IS_BETTER_HINTS):
        return False
    if any(x in m for x in LOWER_IS_BETTER_HINTS):
        return True
    return True


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def load_manifest(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text())

    rows = []
    for item in raw:
        hp = item.get("hyperparameters")
        if hp is None:
            hp_json = item.get("hyperparameters_json", "{}")
            hp = json.loads(hp_json) if hp_json else {}

        row = {
            "algorithm_id": item["algorithm_id"],
            "path_id": item.get("path_id"),
            "method_family": item.get("method_family"),
            "base_method": item.get("base_method"),
            "phase": item.get("phase"),
        }

        for k, v in flatten_dict(hp).items():
            row[f"param.{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    if "method_family" not in df.columns or df["method_family"].isna().all():
        df["method_family"] = df["algorithm_id"].astype(str).str.replace(
            r"__grid\d+$", "", regex=True
        )

    return df


def load_roc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if "id" not in df.columns:
        raise ValueError(
            f"ROC file must contain column 'id'. Found columns: {list(df.columns)}"
        )

    df = df.rename(columns={"id": "algorithm_id"})
    df["algorithm_id"] = df["algorithm_id"].astype(str)

    return df


def resolve_metric_columns(df: pd.DataFrame, requested: list[str]) -> dict[str, str]:
    lower_to_real = {c.lower(): c for c in df.columns}
    out = {}

    for m in requested:
        candidates = [
            m,
            f"{m}_mean",
            m.removesuffix("_mean"),
        ]

        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
            if c.lower() in lower_to_real:
                found = lower_to_real[c.lower()]
                break

        if found is None:
            print(f"[warning] metric '{m}' not found in ROC_data.csv")
        else:
            out[m] = found

    if not out:
        raise ValueError(
            f"None of the requested metrics were found: {requested}\n"
            f"Available columns: {list(df.columns)}"
        )

    return out


def coerce_metrics(df: pd.DataFrame, metric_map: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for public_name, source_col in metric_map.items():
        out[public_name] = pd.to_numeric(out[source_col], errors="coerce")
    return out


def get_config_threshold(row: pd.Series) -> float | None:
    for key in ["param.threshold", "param.thresh"]:
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                pass
    return None


def collapse_threshold_rows(
    df: pd.DataFrame,
    primary_metric: str,
    policy: str,
) -> pd.DataFrame:
    if policy == "all":
        df = df.copy()
        df["_threshold_policy"] = "all_roc_rows"
        return df

    if "thresh" not in df.columns:
        df = df.copy()
        df["_threshold_policy"] = "no_thresh_column"
        return df

    df = df.copy()
    df["thresh_numeric"] = pd.to_numeric(df["thresh"], errors="coerce")

    rows = []
    primary_lower = lower_is_better(primary_metric)

    for _, group in df.groupby("algorithm_id", dropna=False):
        if len(group) == 1:
            chosen = group.iloc[0].copy()
            chosen["_threshold_policy"] = "single_row"
            rows.append(chosen)
            continue

        if policy == "best":
            metric = pd.to_numeric(group[primary_metric], errors="coerce")
            idx = metric.idxmin() if primary_lower else metric.idxmax()
            chosen = group.loc[idx].copy()
            chosen["_threshold_policy"] = "best_roc_row"
            rows.append(chosen)
            continue

        # auto policy
        target = get_config_threshold(group.iloc[0])
        if target is not None and group["thresh_numeric"].notna().any():
            idx = (group["thresh_numeric"] - target).abs().idxmin()
            chosen = group.loc[idx].copy()
            chosen["_threshold_policy"] = "closest_to_config_threshold"
            chosen["_target_threshold"] = target
            rows.append(chosen)
        else:
            metric = pd.to_numeric(group[primary_metric], errors="coerce")
            idx = metric.idxmin() if primary_lower else metric.idxmax()
            chosen = group.loc[idx].copy()
            chosen["_threshold_policy"] = "best_roc_row_no_config_threshold"
            rows.append(chosen)

    out = pd.DataFrame(rows).drop(columns=["thresh_numeric"], errors="ignore")
    return out


def param_cols(df: pd.DataFrame) -> list[str]:
    skip_patterns = [
        "param.id",
        "param.independence_cache_dir",
        "param.timeout",
    ]
    cols = []
    for c in df.columns:
        if not c.startswith("param."):
            continue
        if c in skip_patterns:
            continue
        cols.append(c)
    return cols


def algorithm_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    agg = {}
    for m in metrics:
        if m in df.columns:
            agg[m] = ["mean", "std", "count"]

    out = (
        df.groupby(["method_family", "algorithm_id"], dropna=False)
        .agg(agg)
        .reset_index()
    )

    out.columns = [
        "_".join([x for x in col if x]) if isinstance(col, tuple) else col
        for col in out.columns
    ]

    return out


def main_effects(
    df: pd.DataFrame,
    metrics: list[str],
    max_levels: int,
) -> pd.DataFrame:
    records = []

    for family, fam in df.groupby("method_family", dropna=False):
        for p in param_cols(fam):
            values = fam[p].fillna("<missing>").astype(str)
            n_levels = values.nunique()

            if n_levels <= 1 or n_levels > max_levels:
                continue

            for metric in metrics:
                temp = fam[[p, metric]].copy()
                temp[p] = temp[p].fillna("<missing>").astype(str)
                temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
                temp = temp.dropna(subset=[metric])

                if temp.empty:
                    continue

                grouped = (
                    temp.groupby(p, dropna=False)[metric]
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
                global_mean = temp[metric].mean()

                records.append(
                    {
                        "method_family": family,
                        "parameter": p.replace("param.", "", 1),
                        "metric": metric,
                        "lower_is_better": metric_lower,
                        "n_levels": int(n_levels),
                        "best_value": best[p],
                        "best_mean": float(best["mean"]),
                        "best_count": int(best["count"]),
                        "worst_value": worst[p],
                        "worst_mean": float(worst["mean"]),
                        "worst_count": int(worst["count"]),
                        "global_mean": float(global_mean),
                        "effect_range": float(abs(worst["mean"] - best["mean"])),
                    }
                )

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    return out.sort_values(["method_family", "metric", "effect_range"], ascending=[True, True, False])


def numeric_correlations(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records = []

    for family, fam in df.groupby("method_family", dropna=False):
        for p in param_cols(fam):
            x = pd.to_numeric(fam[p], errors="coerce")

            if x.notna().sum() < 3 or x.nunique(dropna=True) < 3:
                continue

            for metric in metrics:
                y = pd.to_numeric(fam[metric], errors="coerce")
                valid = x.notna() & y.notna()

                if valid.sum() < 3:
                    continue

                pearson = x[valid].corr(y[valid], method="pearson")
                spearman = x[valid].corr(y[valid], method="spearman")

                records.append(
                    {
                        "method_family": family,
                        "parameter": p.replace("param.", "", 1),
                        "metric": metric,
                        "n": int(valid.sum()),
                        "pearson": pearson,
                        "spearman": spearman,
                        "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
                        "direction_note": "positive means larger parameter values increase the metric",
                    }
                )

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    return out.sort_values(["method_family", "metric", "abs_spearman"], ascending=[True, True, False])


def pareto_front(df: pd.DataFrame, primary_metric: str, time_metric: str = "time") -> pd.DataFrame:
    if primary_metric not in df.columns or time_metric not in df.columns:
        return pd.DataFrame()

    rows = []

    for family, fam in df.groupby("method_family", dropna=False):
        temp = fam.copy()
        temp[primary_metric] = pd.to_numeric(temp[primary_metric], errors="coerce")
        temp[time_metric] = pd.to_numeric(temp[time_metric], errors="coerce")
        temp = temp.dropna(subset=[primary_metric, time_metric])

        if temp.empty:
            continue

        primary_lower = lower_is_better(primary_metric)

        for idx, row in temp.iterrows():
            dominated = False
            for jdx, other in temp.iterrows():
                if idx == jdx:
                    continue

                primary_better_or_equal = (
                    other[primary_metric] <= row[primary_metric]
                    if primary_lower
                    else other[primary_metric] >= row[primary_metric]
                )
                time_better_or_equal = other[time_metric] <= row[time_metric]

                primary_strict = (
                    other[primary_metric] < row[primary_metric]
                    if primary_lower
                    else other[primary_metric] > row[primary_metric]
                )
                time_strict = other[time_metric] < row[time_metric]

                if primary_better_or_equal and time_better_or_equal and (primary_strict or time_strict):
                    dominated = True
                    break

            if not dominated:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["method_family", primary_metric, time_metric])


def md_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "_No rows._"

    shown = df.head(max_rows).copy()

    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else str(x))

    cols = list(shown.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in shown.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    return "\n".join([header, sep] + body)


def write_report(
    path: Path,
    tag: str,
    manifest_path: Path,
    roc_path: Path,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    corrs: pd.DataFrame,
    pareto: pd.DataFrame,
    metrics: list[str],
    primary_metric: str,
    top_k: int,
) -> None:
    lines = []

    lines.append(f"# Hyperparameter analysis for `{tag}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- ROC data: `{roc_path}`")
    lines.append(f"- Rows analyzed: `{len(df)}`")
    lines.append(f"- Method families: `{', '.join(sorted(df['method_family'].dropna().astype(str).unique()))}`")
    lines.append(f"- Metrics: `{', '.join(metrics)}`")
    lines.append(f"- Primary metric: `{primary_metric}`")
    lines.append("")
    lines.append("## Interpretation note")
    lines.append("")
    lines.append(
        "This is a descriptive hyperparameter analysis. "
        "For a factorial validation grid, main-effect contrasts are useful hints, "
        "but they are not causal proof. Interactions between hyperparameters and "
        "dataset-specific effects can still dominate."
    )
    lines.append("")
    lines.append("The script resolves metric names such as `SHD_pattern` to `SHD_pattern_mean` if needed.")
    lines.append("")

    lines.append("## Overall algorithm summary")
    lines.append("")
    primary_col = f"{primary_metric}_mean"
    if primary_col in summary.columns:
        summary_sorted = summary.sort_values(primary_col, ascending=lower_is_better(primary_metric))
    else:
        summary_sorted = summary
    lines.append(md_table(summary_sorted, max_rows=30))
    lines.append("")

    for family in sorted(df["method_family"].dropna().astype(str).unique()):
        fam_summary = summary_sorted[summary_sorted["method_family"].astype(str) == family]
        fam_effects = effects[effects["method_family"].astype(str) == family] if not effects.empty else effects
        fam_corrs = corrs[corrs["method_family"].astype(str) == family] if not corrs.empty else corrs
        fam_pareto = pareto[pareto["method_family"].astype(str) == family] if not pareto.empty else pareto

        lines.append(f"## `{family}`")
        lines.append("")

        lines.append("### Best configurations")
        lines.append("")
        lines.append(md_table(fam_summary, max_rows=top_k))
        lines.append("")

        lines.append("### Worst configurations")
        lines.append("")
        lines.append(md_table(fam_summary.tail(top_k), max_rows=top_k))
        lines.append("")

        lines.append("### Largest main effects")
        lines.append("")
        if not fam_effects.empty:
            fam_primary_effects = fam_effects[fam_effects["metric"] == primary_metric]
            lines.append(md_table(fam_primary_effects, max_rows=top_k))
        else:
            lines.append("_No varying hyperparameters found._")
        lines.append("")

        lines.append("### Numeric correlations")
        lines.append("")
        if not fam_corrs.empty:
            lines.append(md_table(fam_corrs, max_rows=top_k))
        else:
            lines.append("_No numeric hyperparameters with enough variation._")
        lines.append("")

        lines.append("### Accuracy/time Pareto candidates")
        lines.append("")
        if not fam_pareto.empty:
            keep_cols = [
                c for c in [
                    "algorithm_id",
                    primary_metric,
                    "time",
                    "thresh",
                    "_threshold_policy",
                ]
                if c in fam_pareto.columns
            ]
            param_keep = [
                c for c in param_cols(fam_pareto)
                if fam_pareto[c].nunique(dropna=False) > 1
            ][:8]
            lines.append(md_table(fam_pareto[keep_cols + param_keep], max_rows=top_k))
        else:
            lines.append("_No Pareto table available._")
        lines.append("")

    path.write_text("\n".join(lines))


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

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not roc_path.exists():
        raise FileNotFoundError(f"ROC_data.csv not found: {roc_path}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo_root / "results" / "notreks" / "hyperparam_analysis" / tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_metric = args.primary_metric or args.metrics[0]

    manifest = load_manifest(manifest_path)
    roc = load_roc(roc_path)

    merged = roc.merge(manifest, on="algorithm_id", how="inner")

    if merged.empty:
        raise ValueError(
            "No rows after joining ROC_data.csv with manifest.json on id/algorithm_id."
        )

    metric_map = resolve_metric_columns(merged, args.metrics)
    metrics = list(metric_map.keys())
    if primary_metric not in metrics:
        primary_metric = metrics[0]

    merged = coerce_metrics(merged, metric_map)

    before_collapse = len(merged)
    merged = collapse_threshold_rows(
        merged,
        primary_metric=primary_metric,
        policy=args.threshold_policy,
    )
    after_collapse = len(merged)

    summary = algorithm_summary(merged, metrics)
    effects = main_effects(merged, metrics, max_levels=args.max_levels)
    corrs = numeric_correlations(merged, metrics)
    pareto = pareto_front(merged, primary_metric=primary_metric, time_metric="time")

    merged.to_csv(out_dir / "analysis_rows.csv", index=False)
    summary.to_csv(out_dir / "algorithm_summary.csv", index=False)
    effects.to_csv(out_dir / "hyperparam_main_effects.csv", index=False)
    corrs.to_csv(out_dir / "numeric_correlations.csv", index=False)
    pareto.to_csv(out_dir / "pareto_candidates.csv", index=False)

    report_path = out_dir / "report.md"
    write_report(
        path=report_path,
        tag=tag,
        manifest_path=manifest_path,
        roc_path=roc_path,
        df=merged,
        summary=summary,
        effects=effects,
        corrs=corrs,
        pareto=pareto,
        metrics=metrics,
        primary_metric=primary_metric,
        top_k=args.top_k,
    )

    print(f"Read manifest: {manifest_path}")
    print(f"Read ROC data: {roc_path}")
    print(f"Rows before threshold collapse: {before_collapse}")
    print(f"Rows after threshold collapse:  {after_collapse}")
    print(f"Wrote: {out_dir}")
    print("")
    print(f"Open report:")
    print(f"  less {report_path}")


if __name__ == "__main__":
    main()