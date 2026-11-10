#!/usr/bin/env python3
"""Paired significance tests for the synthetic NOTREKS ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


KEYS = ["graph_id", "d", "graph_family", "graph_density"]
METRICS = ["SHD_cpdag", "Ancestor_AID_cpdag"]
METHODS = {
    "flop": {
        "notreks": "flop_notreks",
        "dimension": 100,
        "baselines": {"V": "flop", "E": "flop-nt-edge-mask", "P": "flop-nt-post"},
    },
    "dagma": {
        "notreks": "dagma_notreks",
        "dimension": 50,
        "baselines": {"V": "dagma", "E": "dagma-nt-edge-mask", "P": "dagma-nt-post"},
    },
}


def holm(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy(dtype=float))
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values.iloc[position]))
        adjusted[position] = running
    return pd.Series(adjusted, index=values.index)


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["knowledge_fraction"] = pd.to_numeric(frame["knowledge_fraction"], errors="coerce")
    frame = frame[frame["knowledge_fraction"].isin([0.0, 0.25])]
    records = []
    for method_name, spec in METHODS.items():
        d = spec["dimension"]
        subset = frame[frame["d"].eq(d)]
        nt = subset[(subset["method"] == spec["notreks"]) & subset["knowledge_fraction"].eq(0.25)]
        nt = nt.groupby(KEYS, as_index=False)[METRICS].mean()
        for label, baseline_method in spec["baselines"].items():
            baseline_q = 0.0 if label == "V" else 0.25
            base = subset[(subset["method"] == baseline_method) & subset["knowledge_fraction"].eq(baseline_q)]
            base = base.groupby(KEYS, as_index=False)[METRICS].mean()
            paired = nt.merge(base, on=KEYS, suffixes=("_notreks", "_baseline"))
            for family, density in paired[["graph_family", "graph_density"]].drop_duplicates().itertuples(index=False):
                group = paired[(paired["graph_family"] == family) & paired["graph_density"].eq(density)]
                for metric in METRICS:
                    values = (group[f"{metric}_notreks"] - group[f"{metric}_baseline"]).to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if len(values) == 0 or np.allclose(values, 0.0):
                        statistic, p_value = 0.0, 1.0
                    else:
                        result = wilcoxon(values, alternative="less", zero_method="wilcox", method="auto")
                        statistic, p_value = float(result.statistic), float(result.pvalue)
                    records.append({
                        "method": method_name, "graph_family": family, "graph_density": int(density),
                        "metric": metric, "baseline": label, "N": len(values),
                        "mean_delta": float(values.mean()) if len(values) else np.nan,
                        "wilcoxon_statistic": statistic, "p_value": p_value,
                    })
    result = pd.DataFrame(records)
    result["holm_p_value"] = np.nan
    for _, indices in result.groupby("method").groups.items():
        result.loc[indices, "holm_p_value"] = holm(result.loc[indices, "p_value"])
    result["sign"] = np.where(result["mean_delta"] < 0, "improvement", "worsening")
    result["significant"] = result["holm_p_value"] < 0.05
    result["star"] = np.where(result["significant"] & result["mean_delta"].lt(0), "black", np.where(result["significant"], "red", ""))
    return result


def make_tex(result: pd.DataFrame) -> str:
    lookup = {(r.graph_family, int(r.graph_density), r.metric, r.method, r.baseline): r.star for r in result.itertuples()}
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.82}",
        r"\begin{tabular}{lllrccc|ccc}", r"\toprule",
        r"Arm & $n$ & $k$ & Metric & \multicolumn{3}{c|}{FLOP $d=100$} & \multicolumn{3}{c}{DAGMA $d=50$}\\",
        r" & & & & V & E & P & V & E & P\\", r"\midrule",
    ]
    for family in ["er", "ws"]:
        for density in [2, 4, 8]:
            for metric, label in [("SHD_cpdag", "CPDAG SHD"), ("Ancestor_AID_cpdag", "Ancestor-AID")]:
                cells = []
                for method in ["flop", "dagma"]:
                    for baseline in ["V", "E", "P"]:
                        star = lookup.get((family, density, metric, method, baseline), "")
                        cells.append(r"$^{\ast}$" if star == "black" else (r"$\textcolor{red}{^{\ast}}$" if star == "red" else "--"))
                family_label = "Well-specified ER" if family == "er" else "Well-specified WS"
                lines.append(f"{family_label} & $100,500,2000$ & {density} & {label} & " + " & ".join(cells) + r"\\")
            lines.append(r"\addlinespace")
    lines[-1] = lines[-1].replace(r"\addlinespace", r"\bottomrule")
    lines += [
        r"\end{tabular}",
        r"\caption{Paired ablation tests for the $q=0.25$ NOTREKS integration. A black $^{\ast}$ indicates a significant improvement over the named baseline and a red $^{\ast}$ a significant deterioration; dashes are non-significant. V, E, and P denote vanilla, edge masking, and post-selection. For each graph family and density, graph-level CPDAG-SHD and Ancestor-AID differences are averaged over $n\in\{100,500,2000\}$ before the one-sided paired Wilcoxon test. Holm correction is applied across all baseline, graph-density, and metric tests within each method.}",
        r"\label{tab:paired_ablation_tests}", r"\end{table*}", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/lrz_main_synthetic/figures_png/main/aggregate_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/lrz_main_synthetic/analysis/paired_ablation_tests"))
    parser.add_argument("--tex-output", type=Path, required=True)
    args = parser.parse_args()
    result = compute(pd.read_csv(args.input))
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "test_results.csv", index=False)
    (args.output / "README.md").write_text(
        "Paired ablation tests computed by scripts/test_paired_ablations.py. "
        "Graph-level differences average the five q=.25 prior draws and the three sample sizes; "
        "the one-sided alternative is NOTREKS minus baseline < 0. Holm correction is applied within method.\n"
    )
    args.tex_output.write_text(make_tex(result))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
