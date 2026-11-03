"""CausalAssembly paired Pareto figures for CPDAG recovery metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.notreks_production_figures import DAGMA, FLOP, light_fill, save, with_alpha


METHODS = {
    "flop": ("FLOP", FLOP, "vanilla"),
    "flop-nt-standard": ("FLOP + NOTREKS", FLOP, "nt"),
    "dagma": ("DAGMA", DAGMA, "vanilla"),
    "dagma-pstrek": ("DAGMA + PSTrek", DAGMA, "nt"),
}


def _condition(row: pd.Series) -> str:
    mode = str(row.get("knowledge_mode", ""))
    return "estimated" if mode.startswith("estimated") else "notreks"


def _knowledge_level(row: pd.Series) -> str:
    mode = str(row.get("knowledge_mode", ""))
    if mode == "vanilla":
        return "vanilla"
    if mode.startswith("estimated"):
        return "estimated"
    if "q0.25" in mode:
        return "0.25"
    if "q1" in mode:
        return "1.0"
    return mode


def make_figures(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = df[df.solver_status.eq("ok") & df.method.isin(METHODS)].copy()
    # Vanilla baselines are present in both the known-information and
    # estimated-information command groups.  They are the same seed-level
    # condition, so retain one copy before aggregating.
    identity = ["method", "seed", "n", "d", "knowledge_mode", "knowledge_fraction"]
    identity = [c for c in identity if c in df.columns]
    df = df.drop_duplicates(identity, keep="first")
    df["condition"] = df.apply(_condition, axis=1)
    df["knowledge_level"] = df.apply(_knowledge_level, axis=1)
    rows = []
    for (condition, method, knowledge_level), g in df.groupby(
        ["condition", "method", "knowledge_level"], dropna=False
    ):
        rows.append({
            "condition": condition,
            "method": method,
            "knowledge_level": knowledge_level,
            "knowledge_fraction": pd.to_numeric(g.knowledge_fraction, errors="coerce").mean(),
            "runtime_mean": g.candidate_runtime.mean(),
            "runtime_sd": g.candidate_runtime.std(ddof=1),
            "SHD_cpdag_mean": g.SHD_cpdag.mean(),
            "SHD_cpdag_sd": g.SHD_cpdag.std(ddof=1),
            "Ancestor_AID_cpdag_mean": g.Ancestor_AID_cpdag.mean(),
            "Ancestor_AID_cpdag_sd": g.Ancestor_AID_cpdag.std(ddof=1),
            "runs": len(g),
        })
    plot = pd.DataFrame(rows)
    plot.to_csv(out / "plot_data_causalassembly_paired_pareto.csv", index=False)

    for condition, suffix, condition_label in (
        ("notreks", "notreks", "NOTREKS"),
        ("estimated", "estimated", "Estimated NOTREKS"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.65), sharey=False)
        for ax, (metric, label) in zip(
            axes, (("SHD_cpdag", "CPDAG SHD"),
                   ("Ancestor_AID_cpdag", "CPDAG Ancestor-AID"))):
            sub = plot[plot.condition == condition]
            for r in sub.itertuples():
                colour = METHODS[r.method][1]
                level = r.knowledge_level
                if level == "vanilla":
                    face, marker = "white", "o"
                elif level == "1.0":
                    face, marker = with_alpha(colour, .88), "D"
                else:
                    face, marker = light_fill(colour), "D"
                y = getattr(r, metric + "_mean")
                ysd = getattr(r, metric + "_sd")
                ax.errorbar(r.runtime_mean, y,
                            yerr=None if pd.isna(ysd) else ysd,
                            fmt="none", color=colour, capsize=2, lw=.8, zorder=2)
                if level == "vanilla":
                    ax.scatter(r.runtime_mean, y, marker=marker, s=54,
                               facecolors="white", edgecolors="none", zorder=3.5)
                ax.scatter(r.runtime_mean, y, marker=marker, s=48,
                           color=colour, facecolors=face,
                           edgecolors=with_alpha(colour, .72), linewidths=.7, zorder=5)
            ax.set_xscale("log")
            ax.set_xlabel("mean runtime (s; log scale)")
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
        # Keep the legend in the same three-column layout as the synthetic
        # aggregate Pareto figure.  Each column is padded to three entries so
        # Matplotlib does not stagger the headings/markers vertically.
        blank = dict(color="none", marker="None", linestyle="None")
        method_handles = [
            Line2D([], [], color=FLOP, marker="o", ls="None", label="FLOP"),
            Line2D([], [], color=DAGMA, marker="o", ls="None", label="DAGMA"),
            Line2D([], [], **blank, label=""),
        ]
        knowledge_handles = [
            Line2D([], [], color="#555555", marker="o", mfc="none", ls="None", label="vanilla"),
        ]
        if condition == "notreks":
            knowledge_handles.extend([
                Line2D([], [], color="#555555", marker="D", mfc=light_fill("#555555"), ls="None", label="$q=0.25$"),
                Line2D([], [], color="#555555", marker="D", mfc=with_alpha("#555555", .88), ls="None", label="$q=1.0$"),
            ])
        else:
            knowledge_handles.extend([
                Line2D([], [], color="#555555", marker="D", mfc=light_fill("#555555"), ls="None", label="estimated"),
                Line2D([], [], **blank, label=""),
            ])
        sample_handles = [
            Line2D([], [], color="#555555", marker="o", ls="None", label="$n=500$"),
            Line2D([], [], **blank, label=""),
            Line2D([], [], **blank, label=""),
        ]
        legend_handles = method_handles + knowledge_handles + sample_handles
        axes[0].legend(handles=legend_handles, ncol=3, loc="upper left",
                       bbox_to_anchor=(.01, .985), frameon=True,
                       facecolor="white", edgecolor=".75", framealpha=.9,
                       fontsize=6.4, columnspacing=.75, handletextpad=.25,
                       borderpad=.35, labelspacing=.25)
        fig.text(.99, .012, f"CausalAssembly; {condition_label}; n=500; d=98",
                 ha="right", fontsize=6, color=".3")
        fig.subplots_adjust(wspace=.32, bottom=.18, top=.96)
        save(fig, out / f"figure_causalassembly_paired_pareto_{suffix}")
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    make_figures(pd.read_csv(args.input), args.output)


if __name__ == "__main__":
    main()
