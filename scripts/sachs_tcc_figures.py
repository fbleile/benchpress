"""Small PNG summary figures for the Sachs and PSTrek/TCC experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_sachs(path: Path, out: Path) -> None:
    df = pd.read_csv(path)
    df = df[df.status == "ok"].copy()
    # Vanilla rows are duplicated across q; use the complete q=1 slice.
    df = df[df.knowledge_fraction == 1.0]
    summary = df.groupby("method", as_index=False).agg(
        SHD_cpdag=("SHD_cpdag", "mean"),
        runtime=("candidate_runtime", "mean"),
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for _, row in summary.iterrows():
        ax.scatter(row.runtime, row.SHD_cpdag, s=75, label=row.method)
        ax.annotate(row.method, (row.runtime, row.SHD_cpdag),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean runtime (s)")
    ax.set_ylabel("Mean CPDAG SHD")
    ax.set_title("Sachs: 50 bootstrap replicates, 853 observations")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "figure_sachs_paired_pareto.png", dpi=220)
    plt.close(fig)


def write_tcc(path: Path, out: Path) -> None:
    df = pd.read_csv(path)
    df = df[df.solver_status == "ok"].copy()
    df["method"] = df["method"].replace({
        "dagma_notreks": "PSTrek", "dagma_notreks_tcc": "TCC"})
    cells = sorted(df[["d", "graph_type", "density"]].drop_duplicates().itertuples(
        index=False, name=None))
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.8), sharey=True)
    for ax, (d, graph_type, density) in zip(axes.flat, cells):
        sub = df[(df.d == d) & (df.graph_type == graph_type)
                 & (df.density == density)]
        means = sub.groupby("method")["SHD_cpdag"].mean().reindex(
            ["PSTrek", "TCC"])
        ax.bar(means.index, means.values, color=["#0072B2", "#D55E00"])
        ax.set_title(f"d={d}, {graph_type.upper()}{int(density)}")
        ax.grid(axis="y", alpha=.2)
        ax.set_ylabel("Mean CPDAG SHD")
    for ax in axes[-1]:
        ax.set_xlabel("Constraint")
    fig.suptitle("PSTrek versus TCC; n=100, 5 graph replicates")
    fig.tight_layout()
    fig.savefig(out / "figure_pstrek_vs_tcc_six_panels.png", dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sachs-results", type=Path, required=True)
    p.add_argument("--tcc-results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_sachs(args.sachs_results, args.output)
    write_tcc(args.tcc_results, args.output)


if __name__ == "__main__":
    main()
