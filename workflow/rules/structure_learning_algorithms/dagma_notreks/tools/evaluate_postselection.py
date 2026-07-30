#!/usr/bin/env python3
"""Summarize postselection runs without mixing constraint/model regimes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REGIMES = (
    "DAG_only", "DAG_NOTREKS_one_correct",
    "DAG_NOTREKS_several_correct", "DAG_NOTREKS_one_percent")


def _pareto(group):
    metrics = [
        ("candidate_score", False), ("SHD_pattern", False),
        ("skeleton_F1", True), ("postselection_time_seconds", False),
        ("total_runtime", False)]
    available = [(name, maximize) for name, maximize in metrics
                 if name in group and group[name].notna().any()]
    frontier = []
    for index, row in group.iterrows():
        dominated = False
        for other_index, other in group.iterrows():
            if index == other_index:
                continue
            weak, strict = True, False
            for name, maximize in available:
                left, right = row[name], other[name]
                if pd.isna(left) or pd.isna(right):
                    continue
                better = right >= left if maximize else right <= left
                strictly = right > left if maximize else right < left
                weak &= better
                strict |= strictly
            if weak and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(index)
    return group.loc[frontier].copy()


def evaluate(frame):
    data = frame.copy()
    if "run_failure" not in data:
        data["run_failure"] = False
    required = [
        "model_class", "constraint_regime", "postselection_policy",
        "eligible_for_recommendation", "feasible"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError("missing evaluation columns: " + ", ".join(missing))
    eligible = data[
        data.eligible_for_recommendation.astype(bool)
        & data.feasible.astype(bool)
        & ~data.run_failure.astype(bool)].copy()
    summaries, frontiers, warnings = [], [], []
    for (model, regime), group in eligible.groupby(
            ["model_class", "constraint_regime"]):
        numeric = [
            column for column in (
                "candidate_score", "SHD_pattern", "skeleton_F1",
                "pattern_F1", "postselection_time_seconds", "total_runtime")
            if column in group]
        summary = group.groupby(
            "postselection_policy")[numeric].median().reset_index()
        summary.insert(0, "constraint_regime", regime)
        summary.insert(0, "model_class", model)
        summaries.append(summary)
        frontier = _pareto(group)
        frontier.insert(0, "frontier_constraint_regime", regime)
        frontier.insert(0, "frontier_model_class", model)
        frontiers.append(frontier)
        median_shd = (
            float(group.SHD_pattern.median())
            if "SHD_pattern" in group else np.nan)
        median_f1 = (
            float(group.skeleton_F1.median())
            if "skeleton_F1" in group else np.nan)
        poor = (
            not np.isfinite(median_shd)
            or not np.isfinite(median_f1)
            or (np.isfinite(median_shd) and median_shd >= 20)
            or (np.isfinite(median_f1) and median_f1 < .8))
        warnings.append({
            "model_class": model, "constraint_regime": regime,
            "median_SHD_pattern": median_shd,
            "median_skeleton_F1": median_f1,
            "confident_recommendation_allowed": not poor,
            "warning": (
                "Structural metrics are poor; do not freeze a production "
                "default." if poor else ""),
        })
    return (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.concat(frontiers, ignore_index=True) if frontiers else pd.DataFrame(),
        pd.DataFrame(warnings),
    )


def collect_results(root):
    rows = []
    for path in sorted(root.rglob("result.json")):
        payload = json.loads(path.read_text())
        post = payload.get("postselection", {})
        count = int(payload.get("number_of_notreks_pairs", 0))
        regime = (
            "DAG_only" if count == 0
            else "DAG_NOTREKS_one_correct" if count == 1
            else "DAG_NOTREKS_several_correct")
        rows.append({
            "result_path": str(path),
            "model_class": post.get("model_class", "linear"),
            "constraint_regime": payload.get("constraint_regime") or regime,
            "postselection_policy": post.get(
                "postselection_policy",
                payload.get("postselection_policy")),
            "eligible_for_recommendation": post.get(
                "eligible_for_recommendation", False),
            "feasible": post.get("feasible", False),
            "candidate_score": post.get("candidate_score"),
            "postselection_time_seconds": post.get(
                "postselection_time_seconds"),
            "total_runtime": sum(
                item.get("runtime", 0)
                for item in payload.get("restart_diagnostics", [])),
            "SHD_pattern": payload.get("SHD_pattern"),
            "skeleton_F1": payload.get("skeleton_F1"),
            "pattern_F1": payload.get("pattern_F1"),
            "run_failure": False,
        })
    if not rows:
        raise ValueError(f"no result.json files found under {root}")
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--results-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    frame = (
        pd.read_csv(args.input) if args.input is not None
        else collect_results(args.results_root))
    summary, frontier, warnings = evaluate(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "recommendation_summary.csv", index=False)
    frontier.to_csv(args.output_dir / "pareto_frontier.csv", index=False)
    warnings.to_csv(args.output_dir / "recommendation_warnings.csv", index=False)
    report = [
        "# Postselection evaluation", "",
        "Recommendations are separated by model class and active-constraint "
        "regime. The Gaussian-BIC reference is ineligible.", "",
        ("```csv\n" + warnings.to_csv(index=False).strip() + "\n```")
        if len(warnings)
        else "No eligible completed rows were available.", "",
    ]
    text = "\n".join(report)
    (args.output_dir / "REPORT.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
