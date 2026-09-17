"""Aggregate chromatic-prior experiments from multiple graph instances."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.one_graph_prior_chromatic import (
    _safe_correlation,
)


PROPERTIES = [
    "chromatic_number_exact", "maximum_clique_size", "prior_edge_density",
    "prior_degree_mean", "prior_degree_std", "prior_degree_max",
    "prior_isolated_nodes", "prior_endpoint_coverage",
    "prior_connected_components", "prior_largest_component",
    "prior_is_bipartite", "prior_error_alignment",
    "vanilla_notreks_violations", "vanilla_notreks_violation_rate",
]


def correlation_status(left: pd.Series, right: pd.Series) -> str:
    left = left.dropna()
    right = right.loc[left.index].dropna()
    if len(left) < 3:
        return "insufficient_samples"
    if left.nunique() < 2:
        return "constant_predictor"
    if right.nunique() < 2:
        return "constant_gain"
    return "defined"


def correlations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows = []
    groups = [("overall_pooled", "all", "all", frame)]
    groups += [
        ("overall_by_method", method, "all", group)
        for method, group in frame.groupby("method", sort=True)
    ]
    groups += [
        ("by_knowledge_fraction", "all", fraction, group)
        for fraction, group in frame.groupby("knowledge_fraction", sort=True)
    ]
    for scope, method, fraction, group in groups:
        status = correlation_status(
            group["chromatic_number_exact"], group["cpdag_SHD_gain"])
        corr_rows.append({
            "correlation_scope": scope,
            "method": method,
            "knowledge_fraction": fraction,
            "spearman_rho_exact_chromatic": _safe_correlation(
                group["chromatic_number_exact"], group["cpdag_SHD_gain"], "spearman"),
            "pearson_r_exact_chromatic": _safe_correlation(
                group["chromatic_number_exact"], group["cpdag_SHD_gain"], "pearson"),
            "status": status,
            "n_prior_samples": len(group),
        })

    property_rows = []
    for property_name in PROPERTIES:
        for scope, method, fraction, group in groups:
            left = pd.to_numeric(group[property_name], errors="coerce")
            status = correlation_status(left, group["cpdag_SHD_gain"])
            property_rows.append({
                "scope": scope,
                "method": method,
                "knowledge_fraction": fraction,
                "property": property_name,
                "spearman_rho": _safe_correlation(
                    pd.to_numeric(group[property_name], errors="coerce"),
                    group["cpdag_SHD_gain"], "spearman"),
                "pearson_r": _safe_correlation(
                    left,
                    group["cpdag_SHD_gain"], "pearson"),
                "status": status,
                "n": len(group),
            })
    return pd.DataFrame(corr_rows), pd.DataFrame(property_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for directory in args.input_dirs:
        path = directory / "per_prior.csv"
        if not path.exists():
            raise SystemExit(f"missing result file: {path}")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    required = {"method", "knowledge_fraction", "cpdag_SHD_gain", *PROPERTIES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit("result files lack columns: " + ", ".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_prior.csv", index=False)
    corr, properties = correlations(frame)
    corr.to_csv(args.output_dir / "chromatic_gain_correlation.csv", index=False)
    properties.to_csv(args.output_dir / "prior_property_gain_correlation.csv", index=False)
    print(f"aggregated {len(frame)} prior experiments from {len(frames)} graph instances")
    print(corr.to_string(index=False))
    print("\nPrior-property correlation with CPDAG SHD gain:")
    print(properties[properties.scope.isin(["overall_pooled", "overall_by_method"])]
          .to_string(index=False))


if __name__ == "__main__":
    main()
