#!/usr/bin/env python3
"""Graph-seed-aware null tests for prior-structure chromatic experiments.

The omnibus null is that mean structural recovery is equal across realized K.
Graph fixed effects remove between-graph difficulty, while cluster-robust
covariance uses graph_id as the independent sampling unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/lrz_main_synthetic"))
    p.add_argument("--output", type=Path,
                   default=Path("results/lrz_main_synthetic/analysis/prior_structure_flop_null_er4_ws4"))
    args = p.parse_args()

    files = sorted(args.root.glob("job_prior-structure-flop*/raw/results.csv"))
    if not files:
        raise SystemExit(f"no prior-structure-flop results under {args.root}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df[
        (df["method"] == "flop_notreks")
        & df["knowledge_strategy"].astype(str).str.startswith("target-k-")
        & (pd.to_numeric(df["graph_density"], errors="coerce") == 4)
        & df["graph_family"].isin(["er", "ws"])
    ].copy()
    k_field = ("chromatic_number" if "chromatic_number" in df
               else "realized_chromatic_number")
    df["K"] = pd.to_numeric(df[k_field], errors="coerce")
    df["graph_id"] = df["graph_id"].astype(str)
    df = df.dropna(subset=["K", "graph_id"])
    if df.empty:
        raise SystemExit("no FLOP ER4/WS4 target-K rows found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    tests = []
    for family in ["pooled", "er", "ws"]:
        sub = df if family == "pooled" else df[df["graph_family"] == family]
        for metric in ["SHD_cpdag", "Ancestor_AID_cpdag"]:
            x = sub[["graph_id", "K", metric]].copy()
            x[metric] = pd.to_numeric(x[metric], errors="coerce")
            x = x.dropna()
            model = smf.ols(f"{metric} ~ C(graph_id) + C(K)", data=x).fit(
                cov_type="cluster", cov_kwds={"groups": x["graph_id"]})
            table = model.wald_test_terms(skip_single=False).table.loc["C(K)"]
            statistic = float(table["statistic"][0][0])
            df_constraint = int(table["df_constraint"])
            p_value = float(table["pvalue"])
            summaries.append({
                "subset": family,
                "metric": metric,
                "graphs": int(x["graph_id"].nunique()),
                "rows": int(len(x)),
                "K_levels": int(x["K"].nunique()),
                "mean": float(x[metric].mean()),
                "omnibus_wald": statistic,
                "df": df_constraint,
                "p_value": p_value,
                "reject_0.05": bool(p_value < 0.05),
            })
        by_k = sub.groupby(["graph_family", "K"], as_index=False).agg(
            graphs=("graph_id", "nunique"), rows=("K", "size"),
            SHD_cpdag_mean=("SHD_cpdag", "mean"),
            Ancestor_AID_cpdag_mean=("Ancestor_AID_cpdag", "mean"),
        )
        by_k.to_csv(args.output.with_name(args.output.name + f"_{family}_means.csv"), index=False)

    result = pd.DataFrame(summaries)
    result.to_csv(args.output.with_suffix(".csv"), index=False)
    metadata = {
        "test": "graph-fixed-effects OLS with graph_id cluster-robust covariance",
        "null": "equal mean structural recovery across realized chromatic numbers",
        "method": "flop_notreks",
        "families": ["er", "ws"],
        "density": 4,
        "fixed_graph_effect": True,
        "cluster_unit": "graph_id / graph seed",
        "source_files": len(files),
        "rows": len(df),
        "interpretation": "p<0.05 rejects equal recovery across K; this is not a test of independent dots",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
