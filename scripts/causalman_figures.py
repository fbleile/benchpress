"""CausalMan paired Pareto figure, reusing the Figure 1 visual grammar."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.notreks_production_figures import DAGMA, FLOP, light_fill, save

LABELS = {
    "flop": ("FLOP", FLOP, "vanilla"),
    "flop-nt-standard": ("FLOP + NOTREKS", FLOP, "informed"),
    "dagma": ("DAGMA", DAGMA, "vanilla"),
    "dagma-pstrek": ("DAGMA + NOTREKS", DAGMA, "informed"),
}


def make_figure(input_csv: Path, output: Path, metric: str = "projected_skeleton_shd",
                n_values=(500,)):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    df = pd.read_csv(input_csv)
    df = df[df.solver_status == "ok"].copy()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in ("causalman_small", "causalman_medium"):
        for method, (label, colour, kind) in LABELS.items():
            for q in (0.0, 0.25, 1.0):
                if (q == 0.0) != (kind == "vanilla"):
                    continue
                for n in sorted(set(n_values)):
                    x = df[(df.variant == variant) & (df.method == method) &
                           (df.n == n) & np.isclose(df.knowledge_fraction, q)]
                    if x.empty or metric not in x:
                        continue
                    rows.append({"variant": variant, "method": method, "solver": label,
                                 "n": int(n), "knowledge": ("vanilla" if q == 0 else str(q)),
                                 "runtime_mean": x.candidate_runtime.mean(),
                                 "runtime_sd": x.candidate_runtime.std(ddof=1),
                                 "metric_mean": x[metric].mean(),
                                 "metric_sd": x[metric].std(ddof=1), "runs": len(x)})
    plot = pd.DataFrame(rows)
    plot.to_csv(output / "plot_data_causalman_paired_pareto.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35), sharey=False)
    markers = {500: "o", 2000: "^", 5000: "s"}
    for ax, variant in zip(axes, ("causalman_small", "causalman_medium")):
        sub = plot[plot.variant == variant]
        for r in sub.itertuples():
            colour = LABELS[r.method][1]
            if r.knowledge == "vanilla":
                face = "white"
            elif r.knowledge == "0.25":
                face = light_fill(colour)
            else:
                face = colour
            ax.errorbar(r.runtime_mean, r.metric_mean,
                        yerr=None if pd.isna(r.metric_sd) else r.metric_sd,
                        fmt="none", color=colour, capsize=2, lw=.8)
            ax.scatter(r.runtime_mean, r.metric_mean, marker=markers.get(r.n, "o"),
                       s=48, color=colour, facecolors=face, edgecolors=colour,
                       linewidths=1.1, zorder=4)
        ax.set_title("CausalMan Small" if variant.endswith("small") else "CausalMan Medium")
        ax.set_xscale("log")
        ax.set_xlabel("mean runtime (s)")
        ax.set_ylabel("Projected directed-skeleton SHD\n(descriptive; not causal truth)")
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    # One shared legend box.  Each column is a semantic group: method,
    # knowledge level, and sample size.
    blank = dict(color="none", marker="None", linestyle="None")
    handles = [
        Line2D([], [], **blank, label="Method"),
        Line2D([], [], color=FLOP, marker="o", ls="None", label="FLOP"),
        Line2D([], [], color=DAGMA, marker="o", ls="None", label="DAGMA"),
        Line2D([], [], **blank, label=""),
        Line2D([], [], **blank, label="Knowledge"),
        Line2D([], [], color="#555555", marker="o", mfc="none", ls="None", label="vanilla"),
        Line2D([], [], color="#555555", marker="o", mfc=light_fill("#555555"), ls="None", label="25% NOTREKS"),
        Line2D([], [], color="#555555", marker="o", mfc="#555555", ls="None", label="100% NOTREKS"),
        Line2D([], [], **blank, label="Sample size"),
        Line2D([], [], color="#555555", marker="o", ls="None", label="$n=500$"),
        Line2D([], [], color="#555555", marker="s", ls="None", label="$n=5000$"),
        Line2D([], [], **blank, label=""),
    ]
    fig.legend(handles=handles, ncols=3, fontsize=6.5, loc="upper center",
               bbox_to_anchor=(.5, .995), frameon=True, facecolor="white",
               framealpha=.9, columnspacing=1.5, handletextpad=.35,
               borderpad=.45)
    fig.subplots_adjust(wspace=.32, bottom=.18, top=.82)
    # Figure 1's helper writes the PNG; CausalMan additionally requires the
    # publication PDF explicitly so the source artifact is unambiguous.
    save(fig, output / "figure_causalman_paired_pareto_small_medium")
    fig.savefig(output / "figure_causalman_paired_pareto_small_medium.pdf",
                bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metric", default="projected_skeleton_shd")
    p.add_argument("--n-values", nargs="+", type=int, default=[500, 5000])
    args = p.parse_args()
    make_figure(args.input, args.output, args.metric, args.n_values)


if __name__ == "__main__":
    main()
