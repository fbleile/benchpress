#!/usr/bin/env python3
"""Figures and narrative report for the paired NOTREKS benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = ("flop", "flop_notreks", "dagma", "dagma_notreks")


def _bootstrap_mean_interval(values, seed=20260804, draws=5000):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(1)
    return tuple(np.quantile(means, [.025, .975]))


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    shown = frame.copy()
    for column in shown.select_dtypes(include=["number"]):
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return "\n".join([
        "| " + " | ".join(map(str, shown.columns)) + " |",
        "| " + " | ".join("---" for _ in shown.columns) + " |",
        *("| " + " | ".join(map(str, row)) + " |"
          for row in shown.to_numpy()),
    ])


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def _line_plot(ax, frame, x, y, group, ylabel):
    for name, rows in frame.groupby(group, dropna=False):
        points = rows.groupby(x, dropna=False)[y].agg(["mean", "std"]).reset_index()
        points = points.sort_values(x)
        ax.plot(points[x], points["mean"], marker="o", label=str(name))
        if points["std"].notna().any():
            spread = points["std"].fillna(0)
            ax.fill_between(points[x], points["mean"] - spread,
                            points["mean"] + spread, alpha=.12)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)


def generate_figures(frame: pd.DataFrame, paired: pd.DataFrame,
                     output_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    def finish(fig, name):
        path = figures / f"{name}.png"
        _save(fig, path)
        plt.close(fig)
        made.append(f"figures/{name}.png")

    method_column = "id" if "id" in frame else "algorithm"
    methods = [name for name in METHOD_ORDER if name in set(frame[method_column])]

    for metric, label, name in (
            ("SHD_cpdag", "CPDAG SHD (lower is better)", "recovery_by_sample_size"),
            ("F1_pattern", "Pattern F1 (higher is better)", "f1_by_sample_size")):
        if {"n", metric, method_column}.issubset(frame.columns):
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            selected = frame[frame[method_column].isin(methods)].copy()
            _line_plot(ax, selected, "n", metric, method_column, label)
            ax.set_xscale("log")
            finish(fig, name)

    if {"d", "SHD_cpdag", method_column}.issubset(frame.columns):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        _line_plot(ax, frame[frame[method_column].isin(methods)], "d",
                   "SHD_cpdag", method_column, "CPDAG SHD")
        finish(fig, "recovery_by_dimension")

    for metric, ylabel, name in (
            ("delta_SHD_cpdag", "NOTREKS − baseline CPDAG SHD",
             "paired_cpdag_shd_change"),
            ("delta_F1_pattern", "NOTREKS − baseline pattern F1",
             "paired_pattern_f1_change")):
        if metric in paired:
            groups = [group[metric].dropna().to_numpy()
                      for _, group in paired.groupby("comparison")]
            labels = [str(key) for key, _ in paired.groupby("comparison")]
            if groups and any(len(group) for group in groups):
                fig, ax = plt.subplots(figsize=(7.2, 4.5))
                ax.boxplot(groups, tick_labels=labels, showmeans=True)
                ax.axhline(0, color="black", linewidth=.8)
                ax.set_ylabel(ylabel)
                ax.tick_params(axis="x", rotation=15)
                ax.grid(axis="y", alpha=.2)
                finish(fig, name)

    if {"SHD_cpdag_baseline", "SHD_cpdag_notreks", "comparison"}.issubset(
            paired.columns):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for comparison, rows in paired.groupby("comparison"):
            ax.scatter(rows.SHD_cpdag_baseline, rows.SHD_cpdag_notreks,
                       alpha=.65, label=comparison)
        values = pd.concat([paired.SHD_cpdag_baseline,
                            paired.SHD_cpdag_notreks]).dropna()
        if len(values):
            lower, upper = float(values.min()), float(values.max())
            ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        ax.set_xlabel("Baseline CPDAG SHD")
        ax.set_ylabel("NOTREKS CPDAG SHD")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.2)
        finish(fig, "paired_cpdag_shd_scatter")

    if {"knowledge_fraction", "delta_SHD_cpdag", "comparison"}.issubset(
            paired.columns):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        _line_plot(ax, paired, "knowledge_fraction", "delta_SHD_cpdag",
                   "comparison", "NOTREKS − baseline CPDAG SHD")
        ax.axhline(0, color="black", linewidth=.8)
        finish(fig, "knowledge_fraction_effect")

    if {"knowledge_fraction", "notreks_mi_violation_fraction",
        "comparison"}.issubset(paired.columns):
        usable = paired.dropna(subset=["notreks_mi_violation_fraction"])
        if len(usable):
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            _line_plot(ax, usable, "knowledge_fraction",
                       "notreks_mi_violation_fraction", "comparison",
                       "Residual supplied-pair violation fraction")
            ax.set_ylim(bottom=0)
            finish(fig, "mi_violation_fraction")

    if {"time_baseline", "time_notreks", "d", "comparison"}.issubset(
            paired.columns):
        runtime = paired.copy()
        runtime["runtime_ratio"] = (
            runtime.time_notreks / runtime.time_baseline.replace(0, np.nan))
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        _line_plot(ax, runtime, "d", "runtime_ratio", "comparison",
                   "Runtime ratio (NOTREKS / baseline)")
        ax.axhline(1, color="black", linewidth=.8)
        ax.set_yscale("log")
        finish(fig, "runtime_ratio_by_dimension")

    if {"d", "n", "delta_SHD_cpdag", "comparison"}.issubset(paired.columns):
        for comparison, rows in paired.groupby("comparison"):
            table = rows.pivot_table(index="d", columns="n",
                                     values="delta_SHD_cpdag", aggfunc="mean")
            if table.empty:
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            image = ax.imshow(table.to_numpy(float), aspect="auto", cmap="RdBu_r")
            ax.set_xticks(range(len(table.columns)), labels=table.columns)
            ax.set_yticks(range(len(table.index)), labels=table.index)
            ax.set_xlabel("sample size n")
            ax.set_ylabel("dimension d")
            ax.set_title(comparison.replace("_", " "))
            fig.colorbar(image, ax=ax, label="Mean CPDAG SHD change")
            finish(fig, f"heatmap_{comparison}_cpdag_shd")
    return made


def write_report(frame: pd.DataFrame, paired: pd.DataFrame,
                 output_dir: Path, figures: list[str]) -> None:
    method_column = "id" if "id" in frame else "algorithm"
    requested_metrics = {
        "datasets": ("seed", "nunique"),
        "mean_cpdag_shd": ("SHD_cpdag", "mean"),
        "median_cpdag_shd": ("SHD_cpdag", "median"),
        "mean_pattern_f1": ("F1_pattern", "mean"),
        "mean_runtime_s": ("time", "mean"),
    }
    metrics = {name: operation for name, operation in requested_metrics.items()
               if operation[0] in frame}
    if "num_mi_violations" in frame:
        metrics["mean_mi_violations"] = ("num_mi_violations", "mean")
    overview = frame.groupby(method_column, dropna=False).agg(**metrics).reset_index()

    paired_rows = []
    for comparison, rows in paired.groupby("comparison"):
        delta = rows.get("delta_SHD_cpdag", pd.Series(dtype=float)).dropna()
        f1 = rows.get("delta_F1_pattern", pd.Series(dtype=float)).dropna()
        violations = rows.get(
            "notreks_num_mi_violations", pd.Series(dtype=float)).dropna()
        interval = _bootstrap_mean_interval(delta)
        paired_rows.append({
            "comparison": comparison,
            "mean_delta_cpdag_shd": delta.mean(),
            "median_delta_cpdag_shd": delta.median(),
            "bootstrap_95_low": interval[0],
            "bootstrap_95_high": interval[1],
            "wins_ties_losses": (
                f"{int((delta < 0).sum())}/{int((delta == 0).sum())}/"
                f"{int((delta > 0).sum())}" if len(delta) else ""),
            "mean_delta_pattern_f1": f1.mean(),
            "mean_mi_violations": violations.mean(),
            "maximum_mi_violations": violations.max(),
        })
    paired_overview = pd.DataFrame(paired_rows)
    paired_overview.to_csv(output_dir / "key_findings.csv", index=False)
    figure_purposes = {
        "recovery_by_sample_size": "Absolute CPDAG recovery as sample size changes",
        "f1_by_sample_size": "Absolute oriented-edge F1 as sample size changes",
        "recovery_by_dimension": "Scaling of absolute CPDAG recovery with dimension",
        "paired_cpdag_shd_change": "Distribution of paired NOTREKS effects on CPDAG SHD",
        "paired_pattern_f1_change": "Distribution of paired NOTREKS effects on pattern F1",
        "paired_cpdag_shd_scatter": "Dataset-level baseline versus NOTREKS recovery",
        "knowledge_fraction_effect": "Dose-response across supplied oracle-pair fractions",
        "mi_violation_fraction": "Residual supplied-pair violations after final graph construction",
        "runtime_ratio_by_dimension": "Computational overhead of adding NOTREKS",
    }
    pd.DataFrame([{
        "figure": path,
        "svg": str(Path(path).with_suffix(".svg")),
        "scientific_question": figure_purposes.get(
            Path(path).stem,
            "Dimension-by-sample-size map of paired CPDAG SHD effects"),
    } for path in figures]).to_csv(
        output_dir / "figure_manifest.csv", index=False)

    expected = len(frame[[column for column in (
        "scenario", "seed") if column in frame]].drop_duplicates())
    observed = frame.groupby(method_column).seed.nunique().to_dict()
    completeness = ", ".join(f"{name}: {count}" for name, count in observed.items())
    interpretations = []
    for row in paired_rows:
        change = row["mean_delta_cpdag_shd"]
        if pd.isna(change):
            continue
        direction = "improved" if change < 0 else "worsened" if change > 0 else "tied"
        interpretations.append(
            f"- **{row['comparison']}** {direction} mean CPDAG SHD by "
            f"{abs(change):.3g}; wins/ties/losses were "
            f"{row['wins_ties_losses']}. Mean residual MI violations: "
            f"{row['mean_mi_violations']:.3g}.")
    gallery = "\n".join(f"![{Path(path).stem}]({path})" for path in figures)
    text = f"""# NOTREKS benchmark report

## Run completeness

- Unique scenario/seed combinations observed: **{expected}**
- Per-method seed counts: {completeness}
- Primary contrasts are paired within the same generated dataset.

## Method overview

{_markdown(overview)}

## Paired effects of adding NOTREKS

Negative SHD changes and positive F1 changes favor NOTREKS. MI violations are
computed after each method's final graph construction. DAGMA and
DAGMA+NOTREKS both use the same fixed `|W| >= 0.30` threshold; violations are
measured rather than repaired.

{_markdown(paired_overview)}

## Immediate interpretation

{chr(10).join(interpretations) if interpretations else '_Insufficient paired rows._'}

Do not interpret a lower violation count alone as improved recovery. Read it
together with CPDAG SHD, pattern F1, edge errors, and runtime. Factor-stratified
tables are in `paired_summary.csv`, `factor_effects.csv`, and
`algorithm_summary.csv`. The realized MI-pair count is a mediator of graph
structure and knowledge fraction, as documented in `CAUSAL_ANALYSIS.md`.

## Figures

{gallery}

`figure_manifest.csv` states the scientific question attached to every PNG/SVG
pair. `key_findings.csv` contains the compact paired results behind the
immediate interpretation above.
"""
    (output_dir / "REPORT.md").write_text(text)
