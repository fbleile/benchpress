"""Graph-clustered trend analysis for prior-structure-chrom."""
from pathlib import Path
import argparse
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/prior-structure-chrom"))
    ap.add_argument("--min-coverage", type=float, default=0.90,
                    help="minimum fraction of generated graphs represented at K")
    args = ap.parse_args()
    root = args.root
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    raw = pd.concat([pd.read_csv(p) for p in root.glob("jobs/*/raw/results.csv")], ignore_index=True)
    key = ["method", "graph_id", "prior_id", "knowledge_round", "knowledge_strategy"]
    d = raw.drop_duplicates(key).copy()
    d["target_chromatic_number"] = pd.to_numeric(
        d.knowledge_strategy.astype(str).str.extract(r"target-k-(\d+)", expand=False), errors="coerce"
    )
    d["chromatic_number"] = pd.to_numeric(d.chromatic_number, errors="coerce")
    z = d.target_chromatic_number.notna()
    assert (d.loc[z, "target_chromatic_number"] == d.loc[z, "chromatic_number"]).all()
    for c in ["d", "n", "graph_density", "knowledge_fraction", "graph_replicate", "SHD_cpdag"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["prior_degree_variance"] = pd.to_numeric(
        d["prior_degree_std"], errors="coerce"
    ).pow(2)
    d["family"] = d.graph_family.astype(str).str.lower()
    expected = {50: "dagma_notreks", 100: "flop_notreks"}
    x = d[
        d.method.isin(expected.values()) & d.family.isin(["er", "ws"])
        & d.d.isin(expected) & (d.d.map(expected) == d.method)
        & (d.graph_density == 4) & (d.n == 500) & (d.knowledge_fraction == .25)
        & z & d.knowledge_strategy.astype(str).str.startswith("target-k-")
        & (d.solver_status == "ok")
    ].copy()
    x["graph_cluster"] = x.d.astype(int).astype(str) + ":" + x.family + ":r" + x.graph_replicate.astype(int).astype(str)
    cells = (x.groupby(["method", "d", "family", "graph_cluster", "target_chromatic_number"], as_index=False)
             .agg(mean_SHD=("SHD_cpdag", "mean"), set_count=("SHD_cpdag", "size"))
             .rename(columns={"target_chromatic_number": "K"}))
    cells.to_csv(out / "graph_k_cells.csv", index=False)
    coverage = (cells.groupby(["method", "K"], as_index=False)
                .agg(graphs=("graph_cluster", "nunique")))
    totals = (cells.groupby("method")["graph_cluster"].nunique()
              .rename("total_graphs").reset_index())
    coverage = coverage.merge(totals, on="method")
    coverage["coverage"] = coverage["graphs"] / coverage["total_graphs"]
    coverage.to_csv(out / "support_by_k_all.csv", index=False)
    dispersion = (cells.groupby(["method", "K"])
                  .agg(graphs=("mean_SHD", "size"), mean_SHD=("mean_SHD", "mean"),
                       sd_between_graphs=("mean_SHD", "std"),
                       var_between_graphs=("mean_SHD", "var"),
                       min_SHD=("mean_SHD", "min"), max_SHD=("mean_SHD", "max"),
                       mean_set_count=("set_count", "mean"))
                  .reset_index())
    dispersion.to_csv(out / "dispersion_by_k.csv", index=False)
    degree_variance_cells = (x.groupby(
        ["method", "d", "family", "graph_cluster", "target_chromatic_number",
         "prior_id"], as_index=False
    ).agg(
        prior_degree_variance=("prior_degree_variance", "first"),
        mean_SHD=("SHD_cpdag", "mean"),
        set_count=("SHD_cpdag", "size"),
    ).rename(columns={"target_chromatic_number": "K"}))
    degree_variance_cells.to_csv(
        out / "degree_variance_set_cells.csv", index=False
    )
    degree_variance_summary = (degree_variance_cells.groupby(
        ["method", "d", "family", "K"], as_index=False
    ).agg(
        mean_degree_variance=("prior_degree_variance", "mean"),
        mean_SHD=("mean_SHD", "mean"),
        sd_SHD=("mean_SHD", "std"),
        graphs=("graph_cluster", "nunique"),
        cells=("mean_SHD", "size"),
    ))
    degree_variance_summary.to_csv(
        out / "degree_variance_by_k.csv", index=False
    )
    # Degree-variance figures deliberately do not use K-cells.  They pool all
    # targeted-K prior sets and summarize only along the order-free structural
    # coordinate, separately for ER4 and WS4.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    method_specs = {
        "dagma_notreks": ("DAGMA + NOTREKS", "#228833", 50),
        "flop_notreks": ("FLOP + NOTREKS", "#EE6677", 100),
    }
    marker_specs = {"er": "^", "ws": "o"}
    for method, (title, colour, dimension) in method_specs.items():
        q = degree_variance_cells[
            (degree_variance_cells.method == method)
            & (degree_variance_cells.d == dimension)
        ].copy()
        if q.empty:
            continue
        binned = []
        for family, group in q.groupby("family", sort=True):
            group = group.copy()
            bins = min(12, max(1, group.prior_degree_variance.nunique()))
            group["variance_bin"] = (
                pd.qcut(group.prior_degree_variance, q=bins, duplicates="drop")
                if bins > 1 else "all"
            )
            binned.append(group)
        q = pd.concat(binned, ignore_index=True)
        summary = (q.groupby(["family", "variance_bin"], observed=True)
                   .agg(mean_degree_variance=("prior_degree_variance", "mean"),
                        mean_SHD=("mean_SHD", "mean"),
                        sd_SHD=("mean_SHD", "std"),
                        graphs=("graph_cluster", "nunique"),
                        sets=("mean_SHD", "size"))
                   .reset_index())
        summary.to_csv(out / f"degree_variance_figure_data_{method}.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for family in ("er", "ws"):
            g = summary[summary.family == family].sort_values("mean_degree_variance")
            if g.empty:
                continue
            ax.errorbar(
                g.mean_degree_variance, g.mean_SHD,
                yerr=g.sd_SHD.fillna(0),
                color=colour, ecolor=colour, lw=1.8, elinewidth=1.2,
                capsize=3, capthick=1.0, marker=marker_specs[family],
                ms=6.5, markerfacecolor=colour, markeredgecolor=colour,
                label=family.upper() + "4", zorder=3,
            )
        ax.set_title(title + f" ($d={dimension}$)", fontsize=20)
        ax.set_xlabel("prior degree variance", fontsize=16)
        ax.set_ylabel("Mean CPDAG SHD per graph–prior cell\n(lower is better)", fontsize=15)
        ax.grid(alpha=.18); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Graph family", frameon=False, loc="best", fontsize=11,
                  title_fontsize=11)
        fig.text(.99, .012, f"ER4/WS4; n=500; q=0.25; d={dimension}",
                 ha="right", fontsize=9, color=".3")
        fig.subplots_adjust(left=.14, right=.98, bottom=.16, top=.88)
        fig.savefig(out / f"figure_prior_structure_chrom_degree_variance_{method}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Chromatic-number figures show descriptive graph-level means only.  The
    # points are not joined: K is a discrete target and the feasible graph
    # cohort changes with K.  Whiskers are two-sided 95% t intervals over
    # graph-level means within each graph family.
    for method, (title, colour, dimension) in method_specs.items():
        q = cells[(cells.method == method) & (cells.d == dimension)].copy()
        if q.empty:
            continue
        method_coverage = coverage[coverage.method == method]
        eligible = method_coverage.loc[
            method_coverage.coverage >= args.min_coverage, "K"
        ]
        q = q[q.K.isin(eligible)].copy()
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for family, marker in (("er", "^"),):
            g = q[q.family == family].groupby("K")["mean_SHD"].agg(
                mean="mean", sd="std", graphs="size"
            ).reset_index()
            if g.empty:
                continue
            g["ci"] = g.apply(
                lambda r: stats.t.ppf(.975, int(r.graphs) - 1) * r.sd / (int(r.graphs) ** .5)
                if r.graphs > 1 and pd.notna(r.sd) else 0.0,
                axis=1,
            )
            ax.errorbar(
                g["K"], g["mean"], yerr=g["ci"], fmt=marker, linestyle="none",
                color=colour, ecolor=colour, elinewidth=1.2, capsize=3,
                capthick=1.0, ms=6.5, markerfacecolor=colour,
                markeredgecolor=colour, label=family.upper() + "4", zorder=3,
            )
        ax.set_title(title + f" ($d={dimension}$)", fontsize=20)
        ax.set_xlabel("target chromatic number $K$", fontsize=16)
        ax.set_ylabel("Mean CPDAG SHD per graph--$K$ cell\n(lower is better)", fontsize=15)
        ax.grid(alpha=.18); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Graph family", frameon=False, loc="best", fontsize=11,
                  title_fontsize=11)
        fig.text(.99, .012, f"ER4; n=500; q=0.25; d={dimension}; whiskers: 95% t CI",
                 ha="right", fontsize=9, color=".3")
        fig.subplots_adjust(left=.14, right=.98, bottom=.16, top=.88)
        fig.savefig(out / f"figure_prior_structure_chrom_shd_trend_{method.split('_')[0]}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
    within = (x.groupby(["method", "graph_cluster", "target_chromatic_number"])
              .SHD_cpdag.agg(set_sd="std", set_var="var", sets="size")
              .reset_index().rename(columns={"target_chromatic_number": "K"}))
    (within.groupby(["method", "K"])
           .agg(graphs=("graph_cluster", "size"), mean_within_sd=("set_sd", "mean"),
                mean_within_var=("set_var", "mean"))
           .reset_index().to_csv(out / "within_set_variance_by_k.csv", index=False))
    results = []
    support = []
    adjacent = []
    for method in expected.values():
        q = cells[cells.method == method].copy()
        method_coverage = coverage[coverage.method == method].sort_values("K")
        eligible = method_coverage.loc[
            method_coverage.coverage >= args.min_coverage, "K"
        ]
        if eligible.empty:
            raise RuntimeError(f"No K values meet coverage threshold for {method}")
        max_k = int(eligible.max())
        q = q[q.K <= max_k].copy()
        nk = q.groupby("graph_cluster").K.nunique()
        q = q[q.graph_cluster.isin(nk[nk >= 2].index)]
        G = q.graph_cluster.nunique()
        fit = smf.ols("mean_SHD ~ C(graph_cluster) + K", q).fit(
            cov_type="cluster", cov_kwds={"groups": q.graph_cluster}, use_t=True
        )
        j = fit.model.exog_names.index("K")
        beta, se = float(fit.params.iloc[j]), float(fit.bse.iloc[j])
        df = G - 1
        p = float(2.0 * stats.t.sf(abs(beta / se), df=df))
        crit = float(stats.t.ppf(.975, df=df))
        results.append({"method": method, "beta": beta, "clustered_se": se,
                        "ci_low": beta - crit * se, "ci_high": beta + crit * se,
                        "p_two_sided": p, "df": df, "graphs": G, "cells": len(q)})
        for K in sorted(q.K.unique()):
            row = method_coverage[method_coverage.K == K].iloc[0]
            support.append({"method": method, "K": int(K), "graphs": int(row.graphs),
                            "total_graphs": int(row.total_graphs),
                            "coverage": float(row.coverage),
                            "included_at_min_coverage": bool(K <= max_k),
                            "cells": int((q.K == K).sum())})
            if K + 1 in set(q.K):
                a = q[q.K == K][["graph_cluster", "mean_SHD"]].rename(columns={"mean_SHD": "shd0"})
                b = q[q.K == K + 1][["graph_cluster", "mean_SHD"]].rename(columns={"mean_SHD": "shd1"})
                delta = a.merge(b, on="graph_cluster").eval("shd1-shd0")
                adjacent.append({"method": method, "K_from": int(K), "K_to": int(K + 1),
                                 "graphs": len(delta), "mean_delta": delta.mean()})
    results = pd.DataFrame(results)
    results["min_coverage"] = args.min_coverage
    results["p_holm"] = multipletests(results.p_two_sided, method="holm")[1]
    results.to_csv(out / "cluster_fixed_effects_summary.csv", index=False)
    pd.DataFrame(support).to_csv(out / "support_by_k.csv", index=False)
    pd.DataFrame(adjacent).to_csv(out / "adjacent_paired_changes.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
