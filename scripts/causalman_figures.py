"""CausalMan paired Pareto figures for ADMG-aware descriptive diagnostics.

The figures separate marginal adjacency recovery from directed-part recovery
and directed ancestor reachability; none is presented as CPDAG truth under
latent confounding.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.notreks_production_figures import DAGMA, FLOP, light_fill, with_alpha, save

LABELS = {
    "flop": ("FLOP", FLOP, "vanilla"),
    "flop-nt-standard": ("FLOP + NOTREKS", FLOP, "informed"),
    "dagma": ("DAGMA", DAGMA, "vanilla"),
    "dagma-pstrek": ("DAGMA + NOTREKS", DAGMA, "informed"),
}


def make_figure(input_csv: Path, output: Path, metric: str = "projected_skeleton_shd",
                n_values=(500,), filename: str | None = None):
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
    data_name = {
        "projected_marginal_adjacency_shd": "plot_data_causalman_projected_marginal_adjacency_shd.csv",
        "projected_skeleton_shd": "plot_data_causalman_projected_skeleton_shd.csv",
        "projected_directed_skeleton_shd": "plot_data_causalman_projected_directed_skeleton_shd.csv",
        "projected_ancestor_disagreement": "plot_data_causalman_projected_ancestor_disagreement.csv",
    }.get(metric, "plot_data_causalman_paired_pareto.csv")
    plot.to_csv(output / data_name, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.65), sharey=False)
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
                face = (*[int(colour[i:i+2], 16) / 255 for i in (1, 3, 5)], .88)
            if r.knowledge == "vanilla":
                ax.scatter(r.runtime_mean, r.metric_mean, marker=markers.get(r.n, "o"),
                           s=54, facecolors="white", edgecolors="none", zorder=3.5)
            ax.errorbar(r.runtime_mean, r.metric_mean,
                        yerr=None if pd.isna(r.metric_sd) else r.metric_sd,
                        fmt="none", color=colour, capsize=2, lw=.8, zorder=2)
            ax.scatter(r.runtime_mean, r.metric_mean, marker=markers.get(r.n, "o"),
                       s=48, color=colour, facecolors=face,
                       edgecolors=with_alpha(colour, .72),
                       linewidths=.7, zorder=5)
        ax.set_title("CausalMan Small" if variant.endswith("small") else "CausalMan Medium",
                     fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel("mean runtime (s; log scale)")
        ylabel = {
            "projected_marginal_adjacency_shd": (
                "Projected marginal-adjacency SHD\n"
                "(directed + bidirected truth; descriptive)"
            ),
            "projected_skeleton_shd": "Projected directed-skeleton SHD\n(descriptive; not causal truth)",
            "projected_directed_skeleton_shd": "Projected directed-skeleton SHD\n(descriptive; not causal truth)",
            "projected_ancestor_disagreement": "Projected ancestor disagreement\n(AID-like; descriptive)",
        }.get(metric, metric)
        ax.set_ylabel(ylabel)
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
        Line2D([], [], color="#555555", marker="o", mfc=light_fill("#555555"), ls="None", label="$q=0.25$ NOTREKS"),
        Line2D([], [], color="#555555", marker="o", mfc=with_alpha("#555555", .88), ls="None", label="$q=1.0$ NOTREKS"),
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
    # Use the shared PNG-only production figure writer.
    save(fig, output / (filename or "figure_causalman_paired_pareto_small_medium"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metric", choices=("projected_marginal_adjacency_shd",
                                         "projected_skeleton_shd",
                                         "projected_directed_skeleton_shd",
                                         "projected_ancestor_disagreement"),
                   default=None)
    p.add_argument("--n-values", nargs="+", type=int, default=[500, 5000])
    args = p.parse_args()
    if args.metric is not None:
        make_figure(args.input, args.output, args.metric, args.n_values)
    else:
        make_figure(
            args.input, args.output, "projected_marginal_adjacency_shd",
            args.n_values, "figure_causalman_projected_marginal_adjacency_small_medium")
        make_figure(
            args.input, args.output, "projected_directed_skeleton_shd",
            args.n_values, "figure_causalman_projected_skeleton_shd_small_medium")
        make_figure(
            args.input, args.output, "projected_ancestor_disagreement",
            args.n_values, "figure_causalman_projected_ancestor_disagreement_small_medium")


if __name__ == "__main__":
    main()
