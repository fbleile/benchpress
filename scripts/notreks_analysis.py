#!/usr/bin/env python3
"""Aggregate atomic NOTREKS jobs and create paired analysis figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PRINCIPAL = {
    "flop": "flop-nt-local",
    "dagma": "dagma-pstrek",
}


def read_raw(root: Path) -> pd.DataFrame:
    files = sorted((root / "raw").rglob("instance_*.csv"))
    frames = [pd.read_csv(path) for path in files]
    if not frames:
        raise RuntimeError(f"no atomic result files found below {root / 'raw'}")
    frame = pd.concat(frames, ignore_index=True)
    if {"run_id", "method"} <= set(frame):
        frame = frame.drop_duplicates(["run_id", "method"], keep="last")
    return frame


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    key = ["experiment_id", "instance_id", "prior_id"]
    rows = []
    for base, nt in PRINCIPAL.items():
        left = frame[frame.method == base].set_index(key)
        right = frame[frame.method == nt].set_index(key)
        common = left.index.intersection(right.index)
        for index in common:
            a, b = left.loc[index], right.loc[index]
            rows.append({
                **dict(zip(key, index if isinstance(index, tuple) else (index,))),
                "comparison": f"{base}_vs_{nt}",
                "d": a["d"], "n": a["n"], "er_degree": a["er_degree"],
                "knowledge_fraction": a["knowledge_fraction"],
                "corruption_fraction": a["corruption_fraction"],
                "G_SHD": (a["SHD_cpdag"] - b["SHD_cpdag"]) / a["d"],
                "G_F1": b["F1_skel"] - a["F1_skel"],
                "runtime_log2": np.log2(
                    max(b["candidate_runtime"], 1e-12) /
                    max(a["candidate_runtime"], 1e-12)),
                "base_shd": a["SHD_cpdag"], "nt_shd": b["SHD_cpdag"],
                "base_runtime": a["candidate_runtime"],
                "nt_runtime": b["candidate_runtime"],
                "nt_violations": b["violations_after"],
            })
    return pd.DataFrame(rows)


def bootstrap_mean(values, groups, seed=20260914, draws=4000):
    values = np.asarray(values, dtype=float)
    # CSV round-tripping can produce mixed string/float identifier types
    # (especially when failed or invalid rows contain missing values).  Group
    # identity is textual here, so normalize it before bootstrap resampling.
    groups = np.asarray(groups).astype(str)
    usable = np.isfinite(values)
    values, groups = values[usable], groups[usable]
    unique = np.unique(groups)
    if not len(unique):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.choice(unique, size=len(unique), replace=True)
        samples.append(np.mean([values[groups == item].mean()
                                for item in selected]))
    return tuple(np.quantile(samples, [.025, .975]))


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False)
    except (ImportError, ValueError) as exc:
        path.with_suffix(".parquet.unavailable.txt").write_text(
            "Parquet export requires pyarrow or fastparquet.\n" + str(exc) + "\n")
        frame.to_csv(path.with_suffix(".csv"), index=False)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def figures(frame, effects, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plot_dir = out / "plot_data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(plot_dir / "raw.csv", index=False)
    effects.to_csv(plot_dir / "paired_effects.csv", index=False)
    made = []

    # Figure 1: paired runtime/recovery trajectories.  Solver family is
    # encoded by colour, graph density by marker shape, and prior strength by
    # fill state.  This deliberately uses the canonical four methods rather
    # than mixing in ablations or imperfect-prior runs.
    main = frame[frame.experiment_id == "main"]
    if not main.empty:
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D

        # One panel per graph family and dimension; sample-size regimes are
        # overlaid within each panel so their runtime/recovery trajectories
        # can be compared directly.
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), squeeze=False)
        colors = {"flop": "#0072B2", "dagma": "#D55E00"}
        nt_methods = {"flop": "flop-nt-local", "dagma": "dagma-pstrek"}
        fill = {"vanilla": None, 0.25: 0.28, 1.0: 1.0}
        n_markers = {100: "o", 500: "^", 2000: "D"}

        for r, degree in enumerate((2, 4)):
            for c, d in enumerate((20, 50)):
                ax = axes[r, c]
                rows = main[(main.d == d) & (main.er_degree == degree)]
                # Draw each solver/sample-size trajectory in the order
                # vanilla -> 25% -> 100%.  Baseline rows are repeated in the
                # raw table for pairing, so they are collapsed before plotting.
                for n, marker in n_markers.items():
                    nrows = rows[rows.n == n]
                    for solver, nt_method in nt_methods.items():
                        points = []
                        base = nrows[nrows.method == solver]
                        if not base.empty:
                            observations = (base.drop_duplicates("seed")
                                            if "seed" in base else base)
                            mean_runtime = observations.candidate_runtime.mean()
                            points.append(("vanilla", mean_runtime,
                                           observations.SHD_cpdag.mean(),
                                           observations.SHD_cpdag.std(ddof=1)
                                           if len(observations) > 1 else 0.0))
                        for q in (0.25, 1.0):
                            current = nrows[(nrows.method == nt_method) &
                                             (nrows.knowledge_fraction == q)]
                            if not current.empty:
                                observations = (current.drop_duplicates("seed")
                                                if "seed" in current else current)
                                mean_runtime = observations.candidate_runtime.mean()
                                points.append((q, mean_runtime,
                                               observations.SHD_cpdag.mean(),
                                               observations.SHD_cpdag.std(ddof=1)
                                               if len(observations) > 1 else 0.0))
                        if not points:
                            continue
                        colour = colors[solver]
                        for state, runtime, shd, shd_std in points:
                            alpha = fill[state]
                            face = "none" if alpha is None else to_rgba(colour, alpha)
                            ax.errorbar(runtime, shd, yerr=shd_std,
                                        fmt=marker, markersize=8,
                                        markerfacecolor=face,
                                        markeredgecolor=colour,
                                        markeredgewidth=1.2,
                                        ecolor=colour, elinewidth=1.0,
                                        capsize=3, zorder=3)
                ax.set_xscale("log")
                ax.set_ylim(bottom=0)
                ax.set_title(f"ER{degree}, $d={d}$")
                ax.set_xlabel("runtime (s)")
                ax.set_ylabel("CPDAG SHD")
                ax.grid(alpha=.2)

        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["flop"],
                   markeredgecolor=colors["flop"], markersize=7, label="FLOP"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["dagma"],
                   markeredgecolor=colors["dagma"], markersize=7, label="DAGMA"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="black", markersize=7, label="vanilla"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="0.75",
                   markeredgecolor="black", markersize=7, label="25% NOTREKS"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
                   markeredgecolor="black", markersize=7, label="100% NOTREKS"),
            Line2D([0], [0], marker="o", color="black", markersize=7,
                   linestyle="None", label="ER2"),
            Line2D([0], [0], marker="s", color="black", markersize=7,
                   linestyle="None", label="ER4"),
            Line2D([0], [0], marker="o", color="0.3", markersize=7,
                   linestyle="None", label="$n=100$"),
            Line2D([0], [0], marker="^", color="0.3", markersize=7,
                   linestyle="None", label="$n=500$"),
            Line2D([0], [0], marker="D", color="0.3", markersize=7,
                   linestyle="None", label="$n=2000$"),
        ]
        # ER is now the panel index, so only solver, knowledge state, and
        # sample-size encodings need to be shown in the legend.
        legend = legend[:5] + legend[7:]
        fig.legend(handles=legend, loc="center left", ncol=1,
                   frameon=False, bbox_to_anchor=(1.01, .5))
        _save(fig, figdir / "figure1_paired_pareto")
        plt.close(fig); made.append("figure1_paired_pareto")

        # Figure 1b: paired recovery by sample size.  This is the primary
        # alternative to the Pareto view: runtime is removed from the
        # recovery comparison, and each solver gets its own row.
        panels = [(2, 20), (2, 50), (4, 20), (4, 50)]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False,
                                 sharex=True)
        offsets = {"vanilla": 0.0, "25%": -0.10, "100%": 0.10}
        for r, (solver, nt_method) in enumerate(nt_methods.items()):
            for c, (degree, d) in enumerate(panels):
                ax = axes[r, c]
                rows = main[(main.er_degree == degree) & (main.d == d)]
                for q, label in ((None, "vanilla"), (0.25, "25%"),
                                  (1.0, "100%")):
                    method = solver if q is None else nt_method
                    current = rows[(rows.method == method) &
                                   (rows.knowledge_fraction == q if q is not None
                                    else rows.method == method)]
                    if current.empty:
                        continue
                    obs = current.groupby("n")["SHD_cpdag"]
                    means, stds = obs.mean(), obs.std().fillna(0.0)
                    x = np.array([100, 500, 2000], dtype=float)
                    valid = np.isin(x, means.index)
                    x = x[valid]
                    y = means.reindex(x).to_numpy()
                    e = stds.reindex(x).fillna(0.0).to_numpy()
                    xpos = np.arange(len(x), dtype=float) + offsets[label]
                    face = "none" if label == "vanilla" else to_rgba(
                        colors[solver], 0.28 if label == "25%" else 1.0)
                    ax.errorbar(xpos, y, yerr=e, fmt="o-", color=colors[solver],
                                markerfacecolor=face, markeredgecolor=colors[solver],
                                capsize=3, lw=1.2, ms=6, label=label)
                ax.axhline(0, color="black", lw=.8)
                ax.set_ylim(bottom=0)
                ax.set_title(f"{solver.upper()}, ER{degree}, $d={d}$")
                ax.set_xticks(range(3), ["100", "500", "2000"])
                ax.set_xlabel("sample size $n$")
                ax.set_ylabel("CPDAG SHD")
                ax.grid(axis="y", alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   frameon=False, bbox_to_anchor=(.5, 1.02))
        _save(fig, figdir / "figure1b_paired_recovery")
        plt.close(fig); made.append("figure1b_paired_recovery")

        # Figure 1c: the corresponding computational cost.  A logarithmic
        # y-axis is appropriate for runtime, while the x-axis remains the
        # scientifically meaningful sample-size regime.
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False,
                                 sharex=True)
        for r, (solver, nt_method) in enumerate(nt_methods.items()):
            for c, (degree, d) in enumerate(panels):
                ax = axes[r, c]
                rows = main[(main.er_degree == degree) & (main.d == d)]
                for q, label in ((None, "vanilla"), (0.25, "25%"),
                                  (1.0, "100%")):
                    method = solver if q is None else nt_method
                    if q is None:
                        current = rows[rows.method == method]
                    else:
                        current = rows[(rows.method == method) &
                                       (rows.knowledge_fraction == q)]
                    if current.empty:
                        continue
                    grouped = current.groupby("n")["candidate_runtime"]
                    means, stds = grouped.mean(), grouped.std().fillna(0.0)
                    x = np.array([100, 500, 2000], dtype=float)
                    valid = np.isin(x, means.index)
                    x = x[valid]
                    y = means.reindex(x).to_numpy()
                    e = stds.reindex(x).fillna(0.0).to_numpy()
                    xpos = np.arange(len(x), dtype=float) + offsets[label]
                    face = "none" if label == "vanilla" else to_rgba(
                        colors[solver], 0.28 if label == "25%" else 1.0)
                    ax.errorbar(xpos, y, yerr=e, fmt="o-", color=colors[solver],
                                markerfacecolor=face, markeredgecolor=colors[solver],
                                capsize=3, lw=1.2, ms=6, label=label)
                ax.set_yscale("log")
                ax.set_title(f"{solver.upper()}, ER{degree}, $d={d}$")
                ax.set_xticks(range(3), ["100", "500", "2000"])
                ax.set_xlabel("sample size $n$")
                ax.set_ylabel("runtime (s)")
                ax.grid(axis="y", alpha=.2, which="both")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   frameon=False, bbox_to_anchor=(.5, 1.02))
        _save(fig, figdir / "figure1c_paired_runtime")
        plt.close(fig); made.append("figure1c_paired_runtime")

    # Figure 2: paired-effect atlas.
    if not effects.empty:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        for r, metric in enumerate(("G_SHD", "G_F1")):
            for c, n in enumerate((100, 500, 2000)):
                ax = axes[r, c]
                rows = effects[effects.n == n]
                for label, group in rows.groupby("comparison"):
                    vals = group[metric].dropna()
                    if not len(vals): continue
                    lo, hi = bootstrap_mean(vals, group.loc[vals.index, "instance_id"])
                    mean = vals.mean()
                    x = 0 if "flop" in label else 1
                    ax.errorbar(x, mean, yerr=[[mean-lo], [hi-mean]],
                                fmt="o", capsize=3, label=label)
                ax.axhline(0, color="black", lw=.8); ax.set_title(f"n={n}")
                ax.set_xticks([0, 1], ["FLOP", "DAGMA"])
                ax.set_ylabel("positive gain" if r == 0 else "")
                ax.grid(axis="y", alpha=.2)
        _save(fig, figdir / "figure2_paired_effect_forest")
        plt.close(fig); made.append("figure2_paired_effect_forest")

    # Figure 3: integration ablation.
    ab = frame[frame.experiment_id == "integration-ablation"]
    if not ab.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
        groups = [("flop", ["flop", "flop-edge-mask", "flop-nt-post",
                             "flop-nt-local", "flop-nt-global"]),
                  ("dagma", ["dagma", "dagma-edge-mask", "dagma-nt-post",
                              "dagma-pstrek"])]
        for ax, (base, names) in zip(axes[0], groups):
            ref = ab[ab.method == base].set_index(["instance_id", "prior_id"])
            for method in names[1:]:
                cur = ab[ab.method == method].set_index(["instance_id", "prior_id"])
                common = ref.index.intersection(cur.index)
                if len(common):
                    x = cur.loc[common, "candidate_runtime"]
                    y = (cur.loc[common, "SHD_cpdag"] -
                         ref.loc[common, "SHD_cpdag"])
                    x = np.asarray(x, dtype=float)
                    y = np.asarray(y, dtype=float)
                    usable = np.isfinite(x) & np.isfinite(y)
                    x, y = x[usable], y[usable]
                    if not len(x):
                        continue
                    colour = None
                    # Keep the method colour assigned by Matplotlib while
                    # showing individual paired instances translucently.
                    # Runtime is summarized by its mean in this ablation;
                    # retain instance-level SHD variation without adding a
                    # horizontal runtime error bar.
                    raw = ax.scatter(np.full_like(x, x.mean()), y,
                                     label=method, alpha=.25)
                    colour = raw.get_facecolor()[0]
                    yerr = np.std(y, ddof=1) if len(y) > 1 else 0.0
                    ax.errorbar(x.mean(), y.mean(), yerr=yerr,
                                fmt="o", color=colour, markersize=7,
                                markeredgecolor="black", markeredgewidth=.7,
                                capsize=3, linewidth=1.2, zorder=4)
            ax.axhline(0, color="black", lw=.8); ax.axvline(0, color="black", lw=.8)
            ax.set_title(base.upper()); ax.set_xlabel("runtime (s)")
            ax.set_ylabel("CPDAG SHD difference\n(variant $-$ vanilla; lower is better)")
            ax.legend(fontsize=8)
        _save(fig, figdir / "figure3_integration_ablation")
        plt.close(fig); made.append("figure3_integration_ablation")

    # Figure 4: imperfect knowledge response.  A heatmap hides whether a
    # change is caused by missing knowledge or by false supplied constraints;
    # lines over corruption make that distinction explicit.
    imp = effects[effects.experiment_id == "imperfect-knowledge"]
    if not imp.empty:
        n_values = sorted(int(value) for value in imp.n.dropna().unique())
        fig, axes = plt.subplots(len(n_values), 2,
                                 figsize=(11, 4.2 * len(n_values)),
                                 squeeze=False, sharey=True)
        comparisons = [("flop_vs_flop-nt-local", "FLOP"),
                       ("dagma_vs_dagma-pstrek", "DAGMA")]
        line_styles = {0.25: "-", 0.5: "--", 1.0: ":"}
        colours = {0.25: "#0072B2", 0.5: "#009E73", 1.0: "#D55E00"}
        for r, n in enumerate(n_values):
            for c, (label, solver) in enumerate(comparisons):
                ax = axes[r, c]
                rows = imp[(imp.comparison == label) & (imp.n == n)].copy()
                for q in (0.25, 0.5, 1.0):
                    qrows = rows[rows.knowledge_fraction == q]
                    if qrows.empty:
                        continue
                    # paired() stores the normalized gain for the general
                    # effect analysis; this figure intentionally uses raw SHD.
                    qrows = qrows.assign(raw_gain=qrows.nt_shd - qrows.base_shd)
                    grouped = qrows.groupby("corruption_fraction", as_index=False).agg(
                        raw_gain=("raw_gain", "mean"),
                        violations=("nt_violations", "mean"))
                    ax.plot(grouped.corruption_fraction, grouped.raw_gain,
                            marker="o", linestyle=line_styles[q],
                            color=colours[q], label=f"q={q:g}")
                    for _, point in grouped.iterrows():
                        if point.violations > 0:
                            ax.annotate(f"v={point.violations:.0f}",
                                        (point.corruption_fraction, point.raw_gain),
                                        xytext=(0, 5), textcoords="offset points",
                                        fontsize=7, color="0.35",
                                        ha="center")
                ax.axhline(0, color="black", lw=.8)
                ax.set_title(f"{solver}, $n={n}$")
                ax.set_xlabel("false-pair fraction $c$")
                ax.set_ylabel("CPDAG SHD difference\n(NTREKS $-$ vanilla)" if c == 0 else "")
                ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   frameon=False, bbox_to_anchor=(.5, 1.01))
        _save(fig, figdir / "figure4_imperfect_knowledge")
        plt.close(fig); made.append("figure4_imperfect_knowledge")

    # Figure 5: synthetic BIC distance from the refitted true graph.
    if {"bic_gap_to_truth", "n", "method"} <= set(frame):
        usable = frame.dropna(subset=["bic_gap_to_truth"])
        if not usable.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for method, group in usable.groupby("method"):
                points = group.groupby("n")["bic_gap_to_truth"].mean()
                ax.plot(points.index, points.values, marker="o", label=method)
            ax.axhline(0, color="black", lw=.8)
            ax.set_xscale("log"); ax.set_xlabel("sample size n")
            ax.set_ylabel("final BIC minus refitted true-graph BIC")
            ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.2)
            _save(fig, figdir / "figure5_bic_gap_to_truth")
            plt.close(fig); made.append("figure5_bic_gap_to_truth")
    return made


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_raw(args.input_root)
    effects = paired(frame)
    write_parquet(frame, args.output_dir / "results.parquet")
    write_parquet(effects, args.output_dir / "paired_effects.parquet")
    effects.to_csv(args.output_dir / "paired_effects.csv", index=False)
    summaries = []
    if not effects.empty:
        for comparison, group in effects.groupby("comparison"):
            for metric, offset in (("G_SHD", 1), ("G_F1", 2),
                                   ("runtime_log2", 3)):
                values = group[metric].to_numpy(float)
                low, high = bootstrap_mean(
                    values, group.instance_id.to_numpy(),
                    seed=20260914 + offset)
                summaries.append({
                    "comparison": comparison, "metric": metric,
                    "mean": np.nanmean(values),
                    "median": np.nanmedian(values),
                    "bootstrap_low": low, "bootstrap_high": high,
                    "n": int(np.isfinite(values).sum()),
                })
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "paired_summary.csv", index=False)
    summary = frame.groupby("method", as_index=False).agg(
        runs=("run_id", "nunique"), SHD_cpdag_mean=("SHD_cpdag", "mean"),
        F1_skel_mean=("F1_skel", "mean"), runtime_mean=("candidate_runtime", "mean"),
        violations_max=("violations_after", "max"),
        final_bic_mean=("final_bic", "mean"),
        bic_gap_to_truth_mean=("bic_gap_to_truth", "mean"),
        failures=("solver_status", lambda x: (x != "ok").sum()))
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame({"figure": figures(frame, effects, args.output_dir)}).to_csv(
        args.output_dir / "figures.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
