#!/usr/bin/env python3
"""Analysis and publication-style figures for protocol artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {
    "vanilla_flop": "#0072B2",
    "flop_notreks": "#0072B2",
    "vanilla_dagma": "#D55E00",
    "dagma_notreks": "#D55E00",
}
METHOD_FAMILY = {
    "vanilla_flop": "FLOP", "flop_notreks": "FLOP",
    "vanilla_dagma": "DAGMA", "dagma_notreks": "DAGMA",
}


def _family(method: str) -> str:
    return "DAGMA" if "dagma" in method else "FLOP"


def _color(method: str) -> str:
    return "#D55E00" if _family(method) == "DAGMA" else "#0072B2"


def _save(fig, path: Path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def _read(root: Path) -> pd.DataFrame:
    path = root / "raw" / "results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"method", "SHD_cpdag", "candidate_runtime", "knowledge_fraction",
                "n", "d", "graph_family", "graph_density", "graph_id",
                "data_id", "prior_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"protocol results missing columns: {sorted(missing)}")
    return frame


def analyze(root: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = _read(root)
    out = root / "analysis"
    figdir = out / "figures"
    plotdir = out / "plot_data"
    figdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(plotdir / "raw_results.csv", index=False)

    # Deduplicate vanilla rows for absolute summaries, while retaining their
    # repeated pairing against every knowledge condition in the raw table.
    key = ["data_id", "method"]
    absolute = df.drop_duplicates(key)
    summaries = (absolute.groupby(["method", "knowledge_fraction"], as_index=False)
                 .agg(runs=("method", "size"), SHD_mean=("SHD_cpdag", "mean"),
                      SHD_std=("SHD_cpdag", "std"), runtime_median=("candidate_runtime", "median"),
                      runtime_q25=("candidate_runtime", lambda x: x.quantile(.25)),
                      runtime_q75=("candidate_runtime", lambda x: x.quantile(.75))))
    summaries.to_csv(plotdir / "absolute_summaries.csv", index=False)

    base = df[df.method.isin(["vanilla_flop", "vanilla_dagma"])].drop_duplicates(
        ["data_id", "method"])
    paired_rows = []
    for _, b in base.iterrows():
        nt_method = "flop_notreks" if b.method == "vanilla_flop" else "dagma_notreks"
        current = df[(df.data_id == b.data_id) & (df.method == nt_method)]
        for _, nt in current.iterrows():
            paired_rows.append({
                "data_id": b.data_id, "graph_id": b.graph_id,
                "n": b.n, "d": b.d, "graph_family": b.graph_family,
                "graph_density": b.graph_density,
                "knowledge_fraction": nt.knowledge_fraction,
                "round": nt.knowledge_round,
                "solver": METHOD_FAMILY[b.method],
                "delta_SHD": b.SHD_cpdag - nt.SHD_cpdag,
                "delta_F1": nt.F1_skel - b.F1_skel,
                "base_runtime": b.candidate_runtime,
                "notreks_runtime": nt.candidate_runtime,
                "base_SHD": b.SHD_cpdag, "notreks_SHD": nt.SHD_cpdag,
            })
    effects = pd.DataFrame(paired_rows)
    effects.to_csv(plotdir / "paired_effects.csv", index=False)

    made = []
    # Aggregate Pareto view across graph strata; each data instance remains
    # paired, while marker shape encodes n and fill encodes knowledge.
    rows = []
    for (method, q, n), group in df.groupby(["method", "knowledge_fraction", "n"]):
        if method.startswith("vanilla_") and q != q:  # defensive only
            continue
        z = group.drop_duplicates(["data_id", "method"])
        rows.append({"method": method, "family": _family(method),
                     "q": q, "n": n, "runtime": z.candidate_runtime.median(),
                     "SHD": z.SHD_cpdag.mean(), "SHD_std": z.SHD_cpdag.std(),
                     "runs": len(z)})
    pareto = pd.DataFrame(rows)
    pareto.to_csv(plotdir / "paired_pareto.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    markers = {100: "o", 500: "^", 2000: "s"}
    for row in pareto.itertuples():
        color = _color(row.method)
        face = "none" if row.method.startswith("vanilla_") else (
            "#A8CBE2" if row.family == "FLOP" else "#F0B39A") if row.q < .75 else color
        ax.errorbar(row.runtime, row.SHD, yerr=row.SHD_std if np.isfinite(row.SHD_std) else 0,
                    fmt="none", color=color, alpha=.45, lw=.8, zorder=2)
        ax.scatter(row.runtime, row.SHD, marker=markers.get(int(row.n), "o"),
                   s=46, facecolor=face, edgecolor=color, linewidth=.9, zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("median runtime (s)")
    ax.set_ylabel("CPDAG SHD (lower is better)")
    ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    legend = [Line2D([], [], color="#0072B2", marker="o", ls="None", label="FLOP"),
              Line2D([], [], color="#D55E00", marker="o", ls="None", label="DAGMA"),
              Line2D([], [], color="0.25", marker="o", mfc="none", ls="None", label="vanilla"),
              Line2D([], [], color="0.25", marker="o", mfc="#BBBBBB", ls="None", label="25% NOTREKS"),
              Line2D([], [], color="0.25", marker="o", mfc="0.25", ls="None", label="100% NOTREKS"),
              Line2D([], [], color="0.25", marker="o", ls="None", label="$n=100$"),
              Line2D([], [], color="0.25", marker="^", ls="None", label="$n=500$"),
              Line2D([], [], color="0.25", marker="s", ls="None", label="$n=2000$")]
    ax.legend(handles=legend, ncol=3, fontsize=6.5, frameon=True,
              facecolor="white", framealpha=.92, loc="upper left")
    _save(fig, figdir / "figure1_protocol_paired_pareto")
    plt.close(fig); made.append("figure1_protocol_paired_pareto")

    if not effects.empty:
        effect_summary = (effects.groupby(["solver", "knowledge_fraction", "n"], as_index=False)
                          .agg(mean_delta_SHD=("delta_SHD", "mean"),
                               sd_delta_SHD=("delta_SHD", "std"), runs=("delta_SHD", "size")))
        effect_summary.to_csv(plotdir / "effect_summary.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
        for ax, solver, color in zip(axes, ("FLOP", "DAGMA"), ("#0072B2", "#D55E00")):
            z = effect_summary[effect_summary.solver == solver]
            for q, style in ((.25, "--"), (1.0, "-")):
                qz = z[np.isclose(z.knowledge_fraction, q)]
                if qz.empty: continue
                qz = qz.sort_values("n")
                ax.errorbar(qz.n, qz.mean_delta_SHD,
                            yerr=qz.sd_delta_SHD.fillna(0), marker="o",
                            linestyle=style, color=color, capsize=2,
                            label=f"q={q:g}")
            ax.axhline(0, color="0.4", lw=.8); ax.set_title(solver)
            ax.set_xlabel("sample size $n$"); ax.grid(axis="y", alpha=.18)
        axes[0].set_ylabel("Paired $\\Delta$SHD\n(positive is better)")
        axes[1].legend(frameon=False, fontsize=7)
        _save(fig, figdir / "figure2_protocol_paired_effects")
        plt.close(fig); made.append("figure2_protocol_paired_effects")

    # Audit figure: it is deliberately descriptive and makes missing or
    # failed cells visible before any scientific interpretation.
    audit = (df.groupby(["method", "knowledge_fraction"], as_index=False)
             .agg(runs=("method", "size"), failures=("solver_status", lambda x: (x != "ok").sum()),
                  violations=("violations_after", "max")))
    audit.to_csv(plotdir / "audit_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    labels = [f"{m}\nq={q:g}" for m, q in zip(audit.method, audit.knowledge_fraction)]
    ax.bar(np.arange(len(audit)), audit.runs, color=[_color(m) for m in audit.method], alpha=.75)
    ax.set_xticks(np.arange(len(audit)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("raw result rows"); ax.set_title("Protocol coverage audit")
    ax.grid(axis="y", alpha=.18); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, figdir / "figure3_protocol_coverage_audit")
    plt.close(fig); made.append("figure3_protocol_coverage_audit")
    pd.DataFrame({"figure": made}).to_csv(out / "figures.csv", index=False)
    return made


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    print("generated:", ", ".join(analyze(args.input_root)))


if __name__ == "__main__":
    main()
