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
    if row.get("run_condition") in {"oracle", "estimated"}:
        return str(row["run_condition"])
    mode = str(row.get("knowledge_mode", ""))
    return "oracle" if mode.startswith("oracle") or mode == "vanilla" else "estimated"


def make_figures(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = df[df.solver_status.eq("ok") & df.method.isin(METHODS)].copy()
    df["condition"] = df.apply(_condition, axis=1)
    rows = []
    for (condition, method, knowledge_mode), g in df.groupby(
        ["condition", "method", "knowledge_mode"], dropna=False
    ):
        rows.append({
            "condition": condition,
            "method": method,
            "knowledge_mode": knowledge_mode,
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

    for metric, label, suffix in (
        ("SHD_cpdag", "CPDAG SHD", "shd_cpdag"),
        ("Ancestor_AID_cpdag", "CPDAG Ancestor-AID", "ancestor_aid_cpdag"),
    ):
        # Keep the same visual grammar and dimensions as the CausalMan
        # paired-Pareto figure; only the assembly metric label and annotation
        # differ.
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.65), sharey=False)
        for ax, condition in zip(axes, ("oracle", "estimated")):
            sub = plot[plot.condition == condition]
            for r in sub.itertuples():
                colour = METHODS[r.method][1]
                kind = METHODS[r.method][2]
                q = r.knowledge_fraction
                if kind == "vanilla":
                    face = "white"
                    marker = "o"
                elif pd.notna(q) and np.isclose(q, 1.0):
                    face = with_alpha(colour, .88)
                    marker = "D"
                else:
                    face = light_fill(colour)
                    marker = "D"
                y = getattr(r, metric + "_mean")
                ysd = getattr(r, metric + "_sd")
                ax.errorbar(r.runtime_mean, y, xerr=None,
                            yerr=None if pd.isna(ysd) else ysd,
                            fmt="none", color=colour,
                            capsize=2, lw=.8, zorder=2)
                if kind == "vanilla":
                    # Mask whiskers underneath the open indicator.
                    ax.scatter(r.runtime_mean, y, marker=marker, s=54,
                               facecolors="white", edgecolors="none", zorder=3.5)
                ax.scatter(r.runtime_mean, y, marker=marker, s=48,
                           color=colour, facecolors=face,
                           edgecolors=with_alpha(colour, .72),
                           linewidths=.7, zorder=5)
            ax.set_title("Oracle NOTREKS" if condition == "oracle" else "Estimated NOTREKS",
                         fontsize=10)
            ax.set_xscale("log")
            ax.set_xlabel("mean runtime (s; log scale)")
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
        blank = dict(color="none", marker="None", linestyle="None")
        handles = [
            Line2D([], [], **blank, label="Method"),
            Line2D([], [], color=FLOP, marker="o", ls="None", label="FLOP"),
            Line2D([], [], color=DAGMA, marker="o", ls="None", label="DAGMA"),
            Line2D([], [], **blank, label=""),
            Line2D([], [], **blank, label="Knowledge"),
            Line2D([], [], color="#555555", marker="o", mfc="none", ls="None", label="vanilla"),
            Line2D([], [], color="#555555", marker="D", mfc=light_fill("#555555"), ls="None", label="$q=0.25$ NOTREKS"),
            Line2D([], [], color="#555555", marker="D", mfc=with_alpha("#555555", .88), ls="None", label="$q=1.0$ NOTREKS"),
            Line2D([], [], **blank, label="Sample size"),
            Line2D([], [], color="#555555", marker="o", ls="None", label="$n=500$"),
            Line2D([], [], **blank, label=""),
        ]
        fig.legend(handles=handles, ncols=3, fontsize=6.5, loc="upper center",
                   bbox_to_anchor=(.5, .995), frameon=True, facecolor="white",
                   framealpha=.9, columnspacing=1.5, handletextpad=.35,
                   borderpad=.45)
        fig.text(.99, .012, "CausalAssembly; n=500; d=98; CPDAG recovery",
                 ha="right", fontsize=6, color=".3")
        fig.subplots_adjust(wspace=.32, bottom=.18, top=.82)
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
