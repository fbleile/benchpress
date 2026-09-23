#!/usr/bin/env python3
"""Analysis-only publication figures for completed NOTREKS experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Paul Tol bright palette.
FLOP, DAGMA = "#EE6677", "#228833"
VAR_SORT, R2_SORT, NEUTRAL = "#4477AA", "#AA3377", "#4D4D4D"
METHODS = {"flop": ("FLOP", FLOP, "flop_notreks"),
           "dagma": ("DAGMA", DAGMA, "dagma_notreks")}
SORT_METHODS = {"var_sortnregress": ("Var-SortnRegress", VAR_SORT),
                "r2_sortnregress": ("$R^2$-SortnRegress", R2_SORT)}

def light_fill(hex_colour, fraction=.68):
    """Opaque, lightened fill: distinguishes q=.25 without alpha blending."""
    rgb = np.array([int(hex_colour[i:i+2], 16) for i in (1, 3, 5)])
    return "#" + "".join(f"{int(round(v + (255-v)*fraction)):02X}" for v in rgb)

def knowledge_line(ax, x, y, colour, knowledge, **kwargs):
    """Encode knowledge without uncertainty ribbons.

    Vanilla is visually recessive, 25% knowledge is a translucent band with a
    dashed filled core, and complete knowledge is a solid filled stroke.
    """
    kind = str(knowledge)
    if kind == "vanilla":
        ax.plot(x, y, color=colour, lw=1.8, alpha=.42, linestyle=":", **kwargs)
    elif kind == ".25":
        label = kwargs.pop("label", None)
        ax.plot(x, y, color=colour, lw=4.0, alpha=.20, **kwargs)
        if label is not None:
            kwargs["label"] = label
        ax.plot(x, y, color=colour, lw=1.35, alpha=1.0, linestyle="--", **kwargs)
    else:
        ax.plot(x, y, color=colour, lw=1.7, alpha=1.0, linestyle="-", **kwargs)

def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def ensure_bic_gap(df, root):
    if "bic_gap_to_truth" in df.columns:
        return df
    from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import gaussian_bic
    truth_bic = {}
    for data_id in df.data_id.dropna().unique():
        path = root / "data" / f"{data_id}.npz"
        if path.exists():
            a = np.load(path)
            truth_bic[data_id] = float(gaussian_bic(a["X"], a["truth"], lambda_bic=2.0)[0])
    out = df.copy()
    out["truth_bic"] = out.data_id.map(truth_bic)
    out["bic_gap_to_truth"] = out["final_bic"] - out["truth_bic"]
    return out

def base_rows(df, method):
    x = df[df.method == method].copy()
    keys = [c for c in ("instance_id", "prior_id") if c in x]
    return x.drop_duplicates(keys or ["run_id"])


def _scale_limits(values, lower=None):
    """Readable, data-dependent symmetric limits for signed benchmark deltas."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return (-1.0, 1.0)
    extent = max(float(np.max(np.abs(values))), 1.0)
    extent = 2.0 ** np.ceil(np.log2(extent * 1.12))
    return (-extent if lower is None else lower, extent)


def _metric_label(metric):
    return {"SHD_cpdag": "CPDAG SHD",
            "Parent_AID_cpdag": "CPDAG Parent-AID",
            "Ancestor_AID_cpdag": "CPDAG Ancestor-AID"}.get(metric, metric)

def pareto(df, out, metric="SHD_cpdag", suffix="", all_datasets=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if all_datasets:
        x = df[df.experiment_id == "main"].copy()
    else:
        x = df[(df.experiment_id == "main") & (df.d == 50) &
               (df.graph_family == "er") & (df.graph_density == 8)]
    rows = []
    for base, (solver, colour, nt) in METHODS.items():
        for n in (100, 500, 2000):
            b = base_rows(x[x.n == n], base)
            if not b.empty:
                rows.append(dict(solver=solver, knowledge="vanilla", n=n,
                    runtime_mean=b.candidate_runtime.mean(), runtime_q25=b.candidate_runtime.quantile(.25),
                    runtime_q75=b.candidate_runtime.quantile(.75), SHD_mean=b[metric].mean(),
                    SHD_sd=b[metric].std(ddof=1), runs=len(b)))
            for q, label in ((.25, "25% NOTREKS"), (1., "100% NOTREKS")):
                t=x[(x.n==n)&(x.method==nt)&np.isclose(x.knowledge_fraction.astype(float),q)]
                if t.empty: continue
                rows.append(dict(solver=solver, knowledge=label, n=n,
                    runtime_mean=t.candidate_runtime.mean(), runtime_q25=t.candidate_runtime.quantile(.25),
                    runtime_q75=t.candidate_runtime.quantile(.75), SHD_mean=t[metric].mean(),
                    SHD_sd=t[metric].std(ddof=1), runs=len(t)))
    # Cheap ordering baselines are included when the protocol run contains
    # them.  They have no knowledge-level variants.
    for method, (label, colour) in SORT_METHODS.items():
        for n in (100, 500, 2000):
            t = x[(x.method == method) & (x.n == n)]
            if not t.empty:
                rows.append(dict(solver=label, knowledge="baseline", n=n,
                                 runtime_mean=t.candidate_runtime.mean(),
                                 runtime_q25=t.candidate_runtime.quantile(.25),
                                 runtime_q75=t.candidate_runtime.quantile(.75),
                                 SHD_mean=t[metric].mean(), SHD_sd=t[metric].std(ddof=1),
                                 runs=len(t)))
    src=pd.DataFrame(rows); src.to_csv(out/("plot_data_paired_pareto"+suffix+".csv"),index=False)
    # One shared axis makes the solver comparison direct.  A signed-log scale
    # keeps the low FLOP region visible while still accommodating DAGMA values.
    fig, ax = plt.subplots(figsize=(5.9, 4.35))
    markers={100:"o",500:"^",2000:"s"}
    for r in src.itertuples():
        c = {"FLOP": FLOP, "DAGMA": DAGMA,
             "Var-SortnRegress": VAR_SORT, "$R^2$-SortnRegress": R2_SORT}.get(r.solver, NEUTRAL)
        # White is intentional: the variance whisker must not show through
        # vanilla markers.
        fc = "white" if r.knowledge in ("vanilla", "baseline") else (
            light_fill(c) if r.knowledge == "25% NOTREKS" else c)
        ax.errorbar(r.runtime_mean,r.SHD_mean,
                    yerr=[[r.SHD_sd], [r.SHD_sd]],fmt="none",color=c,
                    capsize=2.2,lw=.8,zorder=2)
        ax.scatter(r.runtime_mean,r.SHD_mean,marker=markers[r.n],s=48,
                   color=c,facecolors=fc,edgecolors=c,linewidths=1.1,zorder=4)
    vals = src.SHD_mean.dropna()
    flo = src.loc[src.solver == "FLOP", "SHD_mean"].dropna()
    linthresh = max(1.0, 2.0 ** np.ceil(np.log2(max(float(flo.max()) if len(flo) else 1.0, 1.0))))
    ymax = max(float(vals.max()) * 1.12 if len(vals) else 1.0, linthresh * 1.5)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=linthresh, linscale=1.2, base=10)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("mean runtime (s)"); ax.set_ylabel(_metric_label(metric)+" (lower is better)")
    ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False)
    # Three explicit legend columns: method, knowledge, sample size.
    h=[Line2D([],[],color=FLOP,marker="o",ls="None",label="FLOP"),
       Line2D([],[],color=DAGMA,marker="o",ls="None",label="DAGMA")]
    if "Var-SortnRegress" in set(src.solver):
        h.append(Line2D([],[],color=VAR_SORT,marker="o",ls="None",label="Var-SortnRegress"))
    if "$R^2$-SortnRegress" in set(src.solver):
        h.append(Line2D([],[],color=R2_SORT,marker="o",ls="None",label="$R^2$-SortnRegress"))
    h += [
       Line2D([],[],linestyle="None",label=""),
       Line2D([],[],color=NEUTRAL,marker="o",mfc="none",ls="None",label="vanilla"),
       Line2D([],[],color=NEUTRAL,marker="o",mfc=light_fill(NEUTRAL),ls="None",label="25% NOTREKS"),
       Line2D([],[],color=NEUTRAL,marker="o",mfc=NEUTRAL,ls="None",label="100% NOTREKS"),
       Line2D([],[],color=NEUTRAL,marker="o",ls="None",label="$n=100$"),
       Line2D([],[],color=NEUTRAL,marker="^",ls="None",label="$n=500$"),
       Line2D([],[],color=NEUTRAL,marker="s",ls="None",label="$n=2000$")]
    # Keep the key inside the plotting area so the exported figure has no
    # oversized header.  The central upper region is intentionally empty in
    # this Pareto layout.
    ax.legend(handles=h,ncol=3,loc="upper left",bbox_to_anchor=(.01,.985),
              frameon=True,facecolor="white",edgecolor=".75",framealpha=.9,
              fontsize=6.4,columnspacing=.55,handletextpad=.22,borderpad=.3)
    footer = ("main: all graph types; d=20/50; n=100/500/2000; q=.25/1"
              if all_datasets else
              "main: d=50, ER8; 2 graph replicates; q=.25 (5 draws), q=1")
    fig.text(.98,.035,footer,
            ha="right",va="bottom",fontsize=5.4,
            color=".28",bbox=dict(facecolor="white",alpha=.82,edgecolor="none",pad=1.5))
    fig.subplots_adjust(top=.91,left=.12,right=.98,bottom=.17)
    save(fig,out/("figure1_paired_pareto_aggregate"+suffix)); plt.close(fig)

def ablation(df,out, dimension=50, metric="SHD_cpdag", suffix=""):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Integration variants now live in the superset main registry.  Keep the
    # old experiment names readable for archived runs, but prefer main.
    experiment = "main" if (df.experiment_id == "main").any() else (
        "integration-ablation" if dimension == 50 else "d100-flop")
    x=df[(df.experiment_id==experiment)&(df.d==dimension)]
    # Deliberately avoid circle/triangle/square: those symbols encode sample
    # size in the Pareto figure.  These symbols encode ablation variant only.
    rows=[]; variants={"edge masking":("{base}-edge-mask","P"),
                      "post-repair":("{base}-nt-post","H"),
                      "NOTREKS integrated":(None,"D")}
    for base,(solver,colour,nt) in METHODS.items():
        b=base_rows(x,base); keys=[c for c in ("instance_id","prior_id") if c in b]
        for name,(template,marker) in variants.items():
            if template is None:
                t=x[x.method==nt]
            else:
                labels = (["flop-nt-edge-mask", "flop-edge-mask"] if base == "flop"
                          else ["dagma-nt-edge-mask", "dagma-edge-mask"])
                chosen = next((label for label in labels if (x.method == label).any()), labels[-1])
                t=x[x.method == chosen]
            for q in sorted(t.knowledge_fraction.dropna().unique()):
                z=t[np.isclose(t.knowledge_fraction.astype(float),q)].merge(b,on=keys,suffixes=("_variant","_base"))
                for _,r in z.iterrows(): rows.append(dict(
                    solver=solver,variant=name,marker=marker,q=q,
                    runtime=r.candidate_runtime_variant,
                    delta_SHD=r["SHD_cpdag_base"]-r["SHD_cpdag_variant"],
                    delta_parent_aid=r["Parent_AID_cpdag_base"]-r["Parent_AID_cpdag_variant"],
                    delta_ancestor_aid=r["Ancestor_AID_cpdag_base"]-r["Ancestor_AID_cpdag_variant"],
                    graph_type=f"{str(r.get('graph_family_variant', r.get('graph_family', ''))).upper()}{int(r.get('graph_density_variant', r.get('graph_density', 0)))}"))
    src=pd.DataFrame(rows); src.to_csv(out/(f"plot_data_ablation_d{dimension}"+suffix+".csv"),index=False)
    if dimension == 100 and not src.empty:
        fig,ax=plt.subplots(figsize=(6.0,4.2))
        graph_types=[g for g in ("ER2","ER4","ER8","WS2","WS4","WS8") if g in set(src.graph_type)]
        variants_order=["edge masking","post-repair","NOTREKS integrated"]
        solver_order=["FLOP"]
        offsets={"edge masking":-.18,"post-repair":0.,"NOTREKS integrated":.18}
        for (solver,variant,q),g in src.groupby(["solver","variant","q"]):
            means=g.groupby("graph_type").delta_SHD.agg(["mean","std"])
            positions=[graph_types.index(gt)+offsets[variant] for gt in means.index]
            c=FLOP if solver=="FLOP" else DAGMA
            ax.errorbar(positions,means["mean"],yerr=means["std"].fillna(0),fmt="none",color=c,capsize=2,lw=.9,zorder=2)
            ax.scatter(positions,means["mean"],marker=g.marker.iloc[0],s=48,color=c,
                       facecolors=light_fill(c) if q<1 else c,edgecolors=c,linewidths=1,zorder=4)
        ax.axhline(0,color=".45",lw=.8); ax.set_xticks(range(len(graph_types)),graph_types)
        ax.set_xlabel("graph type"); ax.set_ylabel("paired $\\Delta$CPDAG SHD\n(positive is better)")
        ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False)
        handles=[Line2D([],[],color=FLOP,marker="o",ls="None",label="FLOP"),
                 Line2D([],[],color=NEUTRAL,marker="P",ls="None",label="edge masking"),
                 Line2D([],[],color=NEUTRAL,marker="H",ls="None",label="post-repair"),
                 Line2D([],[],color=NEUTRAL,marker="D",ls="None",label="NOTREKS integrated"),
                 Line2D([],[],color=NEUTRAL,marker="o",mfc=light_fill(NEUTRAL),ls="None",label="$q=25\\%$"),
                 Line2D([],[],color=NEUTRAL,marker="o",mfc=NEUTRAL,ls="None",label="$q=100\\%$")]
        ax.legend(handles=handles,ncol=3,loc="upper left",frameon=True,facecolor="white",framealpha=.9,fontsize=6.2,columnspacing=.6)
        ax.text(.99,.01,"d=100, n=1000; ER2/4/8 and WS2/4/8; 2 graph replicates; q=.25 (5 draws), q=1",transform=ax.transAxes,ha="right",va="bottom",fontsize=5.4,color=".28",bbox=dict(facecolor="white",alpha=.82,edgecolor="none",pad=1.5))
        fig.subplots_adjust(left=.12,right=.98,bottom=.16,top=.97); save(fig,out/f"figure2_integration_ablation_d{dimension}{suffix}"); plt.close(fig)
        return
    fig,ax=plt.subplots(figsize=(4.45,5.35))
    for (_,variant,q),g in src.groupby(["solver","variant","q"]):
        c=FLOP if g.solver.iloc[0]=="FLOP" else DAGMA; xm=g.runtime.mean(); ym=g.delta_SHD.mean()
        # Draw the variability line first, then an opaque marker on top.  This
        # prevents the line from showing through the q=.25 marker.
        ax.errorbar(xm,ym,yerr=g.delta_SHD.std(ddof=1) if len(g)>1 else 0,
                    fmt="none",color=c,capsize=2.5,lw=1,zorder=2)
        ax.scatter(xm,ym,marker=g.marker.iloc[0],s=58,color=c,
                   facecolors=light_fill(c) if q<1 else c,
                   edgecolors=c,linewidths=1.1,zorder=4)
    # DAGMA gains can be an order of magnitude larger than FLOP gains.  A
    # signed log scale keeps the zero reference while making the small FLOP
    # effects visible instead of flattening them against the axis.
    ax.axhline(0,color=".45",lw=.8); ax.set_xscale("log")
    # Use the same signed symlog transition as the Pareto figure.  The
    # negative side remains visible because ablation effects can be harmful.
    ax.set_yscale("symlog",linthresh=10,linscale=1.4,base=10)
    # Match Figure 1 exactly above zero, with a small negative extension for
    # harmful ablations and the same fine 0--10 tick spacing.
    ax.set_ylim(-10, 300)
    ticks = [-10,-8,-6,-4,-2,0,2,4,6,8,10,20,50,100,200,300]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t).replace("-", "−") for t in ticks])
    ylabel={"SHD_cpdag":"paired $\\Delta$CPDAG SHD","Parent_AID_cpdag":"paired $\\Delta$CPDAG Parent-AID","Ancestor_AID_cpdag":"paired $\\Delta$CPDAG Ancestor-AID"}.get(metric,"paired $\\Delta$"+metric)
    ax.set_xlabel("mean runtime (s)"); ax.set_ylabel(ylabel+" (positive is better)"); ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False)
    h=[Line2D([],[],color=FLOP,marker="o",ls="None",label="FLOP"),Line2D([],[],color=DAGMA,marker="o",ls="None",label="DAGMA"),Line2D([],[],color=NEUTRAL,marker="P",ls="None",label="edge masking"),Line2D([],[],color=NEUTRAL,marker="H",ls="None",label="post-repair"),Line2D([],[],color=NEUTRAL,marker="D",ls="None",label="NOTREKS integrated"),Line2D([],[],color=NEUTRAL,marker="o",mfc=light_fill(NEUTRAL),ls="None",label=r"$q=25\%$"),Line2D([],[],color=NEUTRAL,marker="o",mfc=NEUTRAL,ls="None",label=r"$q=100\%$")]
    blank=Line2D([],[],linestyle="None",label="")
    h=[h[0],h[1],blank,h[2],h[3],h[4],h[5],h[6],blank]
    ax.legend(handles=h,ncol=3,loc="upper left",bbox_to_anchor=(0,.985),
              frameon=True,facecolor="white",edgecolor=".75",framealpha=.9,
              fontsize=6.4,columnspacing=.55,handletextpad=.22,borderpad=.3)
    ax.text(.99,.01,"d=50, n=500; ER2/ER4 pooled; 2 graph replicates; q=.25/1",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=5.4,
            color=".28",bbox=dict(facecolor="white",alpha=.82,edgecolor="none",pad=1.5))
    name = "figure2_integration_ablation" if dimension == 50 and not suffix else f"figure2_integration_ablation_d{dimension}{suffix}"
    fig.subplots_adjust(top=.98,left=.17,right=.98,bottom=.13); save(fig,out/name); plt.close(fig)


def ablation_tradeoff(df, out, dimension=50, suffix="", aid=False):
    """Ablation tradeoff in either SHD/AID or Parent-AID/Ancestor-AID space."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    experiment = "main" if (df.experiment_id == "main").any() else (
        "integration-ablation" if dimension == 50 else "d100-flop")
    x = df[(df.experiment_id == experiment) & (df.d == dimension)].copy()
    variants = {"edge masking": "edge", "post-repair": "post", "NOTREKS integrated": "nt"}
    rows = []
    for base, (solver, colour, nt) in METHODS.items():
        b = base_rows(x, base)
        keys = [c for c in ("instance_id", "prior_id") if c in b]
        for label, kind in variants.items():
            if kind == "nt":
                method_names = [nt]
            elif base == "flop":
                method_names = ["flop-nt-edge-mask", "flop-edge-mask"] if kind == "edge" else ["flop-nt-post"]
            else:
                method_names = ["dagma-nt-edge-mask", "dagma-edge-mask"] if kind == "edge" else ["dagma-nt-post"]
            chosen = next((m for m in method_names if (x.method == m).any()), None)
            if chosen is None:
                continue
            t = x[x.method == chosen]
            for q in sorted(t.knowledge_fraction.dropna().unique()):
                z = t[np.isclose(t.knowledge_fraction.astype(float), q)].merge(
                    b, on=keys, suffixes=("_variant", "_base"))
                for _, r in z.iterrows():
                    gt = f"{str(r.get('graph_family_variant', r.get('graph_family', ''))).upper()}{int(r.get('graph_density_variant', r.get('graph_density', 0)))}"
                    rows.append({"solver": solver, "variant": label, "q": q,
                                 "delta_SHD": r.SHD_cpdag_base - r.SHD_cpdag_variant,
                                 "delta_AID": r.Parent_AID_cpdag_base - r.Parent_AID_cpdag_variant,
                                 "delta_ancestor_AID": r.Ancestor_AID_cpdag_base - r.Ancestor_AID_cpdag_variant,
                                 "graph_type": gt})
    src = pd.DataFrame(rows)
    src.to_csv(out / f"plot_data_ablation_tradeoff_d{dimension}{suffix}.csv", index=False)
    if src.empty:
        return
    solvers = [s for s in ("FLOP", "DAGMA") if s in set(src.solver)]
    fig, axes = plt.subplots(1, len(solvers), figsize=(7.1 if len(solvers) > 1 else 5.35, 4.35),
                             squeeze=False)
    axes = axes[0]
    markers = {"edge masking": "P", "post-repair": "H", "NOTREKS integrated": "D"}
    xcol, ycol = ("delta_ancestor_AID", "delta_SHD") if aid else ("delta_SHD", "delta_ancestor_AID")
    for ax, solver in zip(axes, solvers):
        part = src[src.solver == solver]
        c = FLOP if solver == "FLOP" else DAGMA
        for (solver_name, variant, q), g in part.groupby(["solver", "variant", "q"]):
            xval, yval = g[xcol].mean(), g[ycol].mean()
            ax.errorbar(xval, yval, xerr=g[xcol].std(ddof=1) if len(g) > 1 else 0,
                        yerr=g[ycol].std(ddof=1) if len(g) > 1 else 0,
                        fmt="none", color=c, alpha=.65, capsize=2, lw=.8, zorder=2)
            ax.scatter(xval, yval, marker=markers[variant], s=64,
                       color=c, facecolors="white" if q < 1 else c,
                       edgecolors=c, linewidths=1.1, zorder=4,
                       alpha=.72 if q < 1 else 1.0)
        ax.set_xlim(*_scale_limits(part[xcol]))
        ax.set_ylim(*_scale_limits(part[ycol]))
        # Signed-log axes separate small FLOP effects from the much larger
        # DAGMA AID effects while retaining a readable linear region at zero.
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.15, base=10)
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.15, base=10)
        ax.axhline(0, color=".45", lw=.8); ax.axvline(0, color=".45", lw=.8)
        ax.set_title(solver, color=c, fontsize=9)
        if aid:
            ax.set_xlabel("paired $\\Delta$CPDAG Ancestor-AID\n(positive is better)")
            ylabel = "paired $\\Delta$CPDAG SHD\n(positive is better)"
        else:
            ax.set_xlabel("paired $\\Delta$CPDAG SHD\n(positive is better)")
            ylabel = "paired $\\Delta$CPDAG Ancestor-AID\n(positive is better)"
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.18); ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=FLOP, marker="o", ls="None", label="FLOP"),
               Line2D([], [], color=DAGMA, marker="o", ls="None", label="DAGMA"),
               Line2D([], [], color=".25", marker="P", ls="None", label="edge masking"),
               Line2D([], [], color=".25", marker="H", ls="None", label="post-repair"),
               Line2D([], [], color=".25", marker="D", ls="None", label="NOTREKS integrated"),
               Line2D([], [], color=".25", marker="o", mfc="white", ls="None", label="$q=25\\%$"),
               Line2D([], [], color=".25", marker="o", mfc=".25", ls="None", label="$q=100\\%$")]
    axes[0].legend(handles=handles, ncol=3, fontsize=6.1, loc="upper left",
              frameon=True, facecolor="white", framealpha=.92)
    fig.text(.5, .012, f"d={dimension}; {'ER2/ER4 pooled' if dimension == 50 else 'ER2/4/8 and WS2/4/8'}; paired data",
            ha="center", va="bottom", fontsize=5.2,
            color=".28", bbox=dict(facecolor="white", alpha=.82, edgecolor="none", pad=1.5))
    fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.91, wspace=.28)
    if dimension == 50 and suffix == "":
        name = "figure2_integration_ablation"
    elif dimension == 50 and suffix == "_cpdag_aid":
        name = "figure2_integration_ablation_cpdag_aid"
    else:
        name = f"figure2_integration_ablation_d{dimension}{suffix}"
    save(fig, out / name); plt.close(fig)


def ablation_rank(df, out, dimension=50):
    """Rank available integration variants by graph type and solver."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if dimension == 20:
        experiment = "main"
    elif dimension == 50:
        experiment = "integration-ablation"
    else:
        experiment = "d100-flop"
    x = df[(df.experiment_id == experiment) & (df.d == dimension)].copy()
    method_info = {
        "flop": ("FLOP", "vanilla", FLOP),
        "dagma": ("DAGMA", "vanilla", DAGMA),
        "flop-nt-edge-mask": ("FLOP", "edge masking", FLOP),
        "dagma-nt-edge-mask": ("DAGMA", "edge masking", DAGMA),
        "flop-nt-post": ("FLOP", "post-repair", FLOP),
        "dagma-nt-post": ("DAGMA", "post-repair", DAGMA),
        "flop_notreks": ("FLOP", "NOTREKS integrated", FLOP),
        "dagma_notreks": ("DAGMA", "NOTREKS integrated", DAGMA),
    }
    rows = []
    for (data_id, q, solver), g in x.assign(
            solver=x.method.map({m: info[0] for m, info in method_info.items()})).groupby(
                ["data_id", "knowledge_fraction", "solver"]):
        for method, value in g.groupby("method").SHD_cpdag.mean().items():
            if method not in method_info:
                continue
            solver_name, variant, colour = method_info[method]
            rows.append({"data_id": data_id, "q": q, "solver": solver_name,
                         "variant": variant, "rank_value": value,
                         "graph_type": f"{str(g.graph_family.iloc[0]).upper()}{int(g.graph_density.iloc[0])}"})
    ranks = pd.DataFrame(rows)
    if ranks.empty:
        return
    ranks["rank"] = ranks.groupby(["data_id", "q", "solver"])["rank_value"].rank(
        method="average", ascending=True)
    ranks.to_csv(out / f"plot_data_ablation_rank_d{dimension}.csv", index=False)

    solvers = [s for s in ("FLOP", "DAGMA") if s in set(ranks.solver)]
    fig, axes = plt.subplots(1, len(solvers), figsize=(3.55 * len(solvers), 4.15), squeeze=False)
    axes = axes[0]
    markers = {"vanilla": "o", "edge masking": "P", "post-repair": "H",
               "NOTREKS integrated": "D"}
    for ax, solver in zip(axes, solvers):
        part = ranks[ranks.solver == solver]
        graph_types = sorted(part.graph_type.unique())
        positions = np.arange(len(graph_types))
        for variant in ("vanilla", "edge masking", "post-repair", "NOTREKS integrated"):
            q_values = [0.25] if variant == "vanilla" else sorted(part[part.variant == variant].q.dropna().unique())
            for q in q_values:
                subset = part[part.variant == variant]
                if variant != "vanilla":
                    subset = subset[np.isclose(subset.q, q)]
                z = subset.groupby(
                    "graph_type")["rank"].agg(["mean", "std"]).reindex(graph_types)
                if z["mean"].notna().sum() == 0:
                    continue
                colour = FLOP if solver == "FLOP" else DAGMA
                xoff = -.055 if q < 1 else .055
                ax.errorbar(positions + xoff, z["mean"], yerr=z["std"].fillna(0), fmt="none",
                            color=colour, capsize=2, lw=.8)
                for pos, graph_type in enumerate(graph_types):
                    row = z.loc[graph_type]
                    if pd.notna(row["mean"]):
                        face = "white" if variant == "vanilla" and q != 1 else (
                            light_fill(colour) if q < 1 else colour)
                        ax.scatter(pos + xoff, row["mean"], marker=markers[variant], s=52,
                                   color=colour, facecolors=face, edgecolors=colour,
                                   linewidths=1.0, zorder=4)
        ax.set_xticks(positions, graph_types)
        ax.set_xlabel("graph type")
        ax.set_title(solver, color=FLOP if solver == "FLOP" else DAGMA, fontsize=9)
        ax.set_ylim(.7, max(4.3, float(ranks["rank"].max()) + .3))
        ax.grid(axis="y", alpha=.18); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean within-instance rank (lower is better)")
    handles = []
    for solver, colour in (("FLOP", FLOP), ("DAGMA", DAGMA)):
        if solver in solvers:
            handles.append(Line2D([], [], color=colour, marker="o", ls="None", label=solver))
    handles.append(Line2D([], [], ls="None", label=""))
    available_variants = set(ranks.variant)
    for variant in ("vanilla", "edge masking", "post-repair", "NOTREKS integrated"):
        if variant in available_variants:
            handles.append(Line2D([], [], color=NEUTRAL, marker=markers[variant], ls="None", label=variant))
    handles.append(Line2D([], [], ls="None", label=""))
    handles += [Line2D([], [], color=NEUTRAL, marker="o", mfc=light_fill(NEUTRAL), ls="None", label="$q=25\\%$"),
                Line2D([], [], color=NEUTRAL, marker="o", mfc=NEUTRAL, ls="None", label="$q=100\\%$")]
    fig.legend(handles=handles, ncol=1, loc="upper center", bbox_to_anchor=(.5, 1.02),
               frameon=True, facecolor="white", framealpha=.92, fontsize=6.2)
    fig.text(.5, .012, f"{experiment}: d={dimension}; graph-type ranking; paired data",
             ha="center", va="bottom", fontsize=5.2, color=".28")
    fig.subplots_adjust(top=.72, bottom=.15, left=.09, right=.98, wspace=.28)
    save(fig, out / f"figure2_integration_ablation_rank_d{dimension}"); plt.close(fig)


def ablation_pareto(df, out, dimension=50):
    """Classic runtime/CPDAG-SHD view for the integration ablation."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    experiment = "integration-ablation" if dimension == 50 else "d100-flop"
    x = df[(df.experiment_id == experiment) & (df.d == dimension)].copy()
    info = {"flop": ("FLOP", "vanilla", FLOP),
            "flop-nt-edge-mask": ("FLOP", "edge masking", FLOP),
            "flop-nt-post": ("FLOP", "post-repair", FLOP),
            "flop_notreks": ("FLOP", "NOTREKS", FLOP),
            "dagma": ("DAGMA", "vanilla", DAGMA),
            "dagma-nt-edge-mask": ("DAGMA", "edge masking", DAGMA),
            "dagma-nt-post": ("DAGMA", "post-repair", DAGMA),
            "dagma_notreks": ("DAGMA", "NOTREKS", DAGMA)}
    rows = []
    for method, (solver, variant, colour) in info.items():
        t = x[x.method == method]
        if t.empty:
            continue
        for q in sorted(t.knowledge_fraction.dropna().unique()):
            z = t[np.isclose(t.knowledge_fraction, q)]
            rows.append({"solver": solver, "variant": variant, "q": q,
                         "runtime": z.candidate_runtime.mean(),
                         "runtime_sd": z.candidate_runtime.std(ddof=1),
                         "shd": z.SHD_cpdag.mean(), "shd_sd": z.SHD_cpdag.std(ddof=1)})
    src = pd.DataFrame(rows)
    src.to_csv(out / f"plot_data_integration_pareto_d{dimension}.csv", index=False)
    if src.empty:
        return
    fig, ax = plt.subplots(figsize=(5.9, 4.35))
    markers = {"vanilla": "o", "edge masking": "P", "post-repair": "H", "NOTREKS": "D"}
    for r in src.itertuples():
        colour = FLOP if r.solver == "FLOP" else DAGMA
        face = "white" if r.variant == "vanilla" else (light_fill(colour) if r.q < 1 else colour)
        ax.errorbar(r.runtime, r.shd, xerr=0 if pd.isna(r.runtime_sd) else r.runtime_sd,
                    yerr=0 if pd.isna(r.shd_sd) else r.shd_sd, fmt="none", color=colour,
                    capsize=2, lw=.8, zorder=2)
        ax.scatter(r.runtime, r.shd, marker=markers[r.variant], s=56, color=colour,
                   facecolors=face, edgecolors=colour, linewidths=1.0, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1, linscale=1.15, base=10)
    ax.set_xlabel("mean runtime (s)"); ax.set_ylabel("CPDAG SHD (lower is better)")
    ax.grid(axis="y", alpha=.18); ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=FLOP, marker="o", ls="None", label="FLOP"),
               Line2D([], [], color=DAGMA, marker="o", ls="None", label="DAGMA"),
               Line2D([], [], ls="None", label=""),
               Line2D([], [], color=NEUTRAL, marker="o", mfc="white", ls="None", label="vanilla"),
               Line2D([], [], color=NEUTRAL, marker="P", ls="None", label="edge masking"),
               Line2D([], [], color=NEUTRAL, marker="H", ls="None", label="post-repair"),
               Line2D([], [], color=NEUTRAL, marker="D", ls="None", label="NOTREKS"),
               Line2D([], [], ls="None", label=""),
               Line2D([], [], color=NEUTRAL, marker="o", mfc=light_fill(NEUTRAL), ls="None", label="$q=25\\%$"),
               Line2D([], [], color=NEUTRAL, marker="o", mfc=NEUTRAL, ls="None", label="$q=100\\%$")]
    ax.legend(handles=handles, ncol=1, loc="upper left", frameon=True,
              facecolor="white", framealpha=.92, fontsize=6.2)
    ax.text(.99, .01, f"{experiment}: d={dimension}; paired integration ablation",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color=".28",
            bbox=dict(facecolor="white", alpha=.82, edgecolor="none", pad=1.5))
    fig.subplots_adjust(left=.13, right=.98, bottom=.15, top=.97)
    save(fig, out / ("figure2_integration_ablation" if dimension == 50 else f"figure2_integration_ablation_d{dimension}"))
    plt.close(fig)


def main_rank(df, out, dimension=50):
    """Main-protocol ranking by graph type, sample size, and knowledge."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    x = df[(df.experiment_id == "main") & (df.d == dimension)].copy()
    info = {"flop": ("FLOP", "vanilla", FLOP),
            "flop_notreks": ("FLOP", "NOTREKS", FLOP),
            "dagma": ("DAGMA", "vanilla", DAGMA),
            "dagma_notreks": ("DAGMA", "NOTREKS", DAGMA)}
    rows = []
    for (data_id, n, q, solver), g in x.assign(
            solver=x.method.map({m: v[0] for m, v in info.items()})).groupby(
                ["data_id", "n", "knowledge_fraction", "solver"]):
        for method, value in g.groupby("method").SHD_cpdag.mean().items():
            if method not in info:
                continue
            solver_name, variant, colour = info[method]
            rows.append({"data_id": data_id, "n": n, "q": q,
                         "solver": solver_name, "variant": variant,
                         "rank_value": value,
                         "graph_type": f"{str(g.graph_family.iloc[0]).upper()}{int(g.graph_density.iloc[0])}"})
    ranks = pd.DataFrame(rows)
    if ranks.empty:
        return
    ranks["rank"] = ranks.groupby(["data_id", "n", "q", "solver"])["rank_value"].rank(
        method="average", ascending=True)
    ranks.to_csv(out / f"plot_data_main_rank_d{dimension}.csv", index=False)
    solvers = [s for s in ("FLOP", "DAGMA") if s in set(ranks.solver)]
    fig, axes = plt.subplots(1, len(solvers), figsize=(3.55 * len(solvers), 4.15), squeeze=False)
    axes = axes[0]
    markers = {100: "o", 500: "^", 2000: "s"}
    for ax, solver in zip(axes, solvers):
        part = ranks[ranks.solver == solver]
        graph_types = sorted(part.graph_type.unique())
        positions = np.arange(len(graph_types))
        for n in sorted(part.n.unique()):
            for variant in ("vanilla", "NOTREKS"):
                if variant == "vanilla":
                    q_values = [0.0] if (part.q == 0.0).any() else [0.25]
                else:
                    q_values = sorted(part.q.dropna().unique())
                for q in q_values:
                    subset = part[(part.n == n) & (part.variant == variant)]
                    if variant != "vanilla":
                        subset = subset[np.isclose(subset.q, q)]
                    z = subset.groupby(
                        "graph_type")["rank"].agg(["mean", "std"]).reindex(graph_types)
                    if z["mean"].notna().sum() == 0:
                        continue
                    colour = FLOP if solver == "FLOP" else DAGMA
                    # Nine conditions are separated by sample-size marker and
                    # a small knowledge-level x offset.
                    # Keep all nine sample-size/knowledge conditions visibly
                    # separated even when whiskers are wide.
                    noff = {100: -.18, 500: 0.0, 2000: .18}[int(n)]
                    qoff = -.055 if q < 1 else .055
                    ax.errorbar(positions + noff + qoff, z["mean"], yerr=z["std"].fillna(0),
                                fmt="none", color=colour, capsize=2, lw=.7)
                    for pos, graph_type in enumerate(graph_types):
                        row = z.loc[graph_type]
                        if pd.notna(row["mean"]):
                            face = "white" if variant == "vanilla" else (
                                light_fill(colour) if q < 1 else colour)
                            ax.scatter(pos + noff + qoff, row["mean"], marker=markers[int(n)], s=43,
                                       color=colour, facecolors=face, edgecolors=colour,
                                       linewidths=1.0, zorder=4)
        ax.set_xticks(positions, graph_types); ax.set_xlabel("graph type")
        ax.set_title(solver, color=FLOP if solver == "FLOP" else DAGMA, fontsize=9)
        ax.set_ylim(.7, max(3.3, float(ranks["rank"].max()) + .3))
        ax.grid(axis="y", alpha=.18); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean within-instance CPDAG-SHD rank (lower is better)")
    handles = []
    for solver, colour in (("FLOP", FLOP), ("DAGMA", DAGMA)):
        if solver in solvers:
            handles.append(Line2D([], [], color=colour, marker="o", ls="None", label=solver))
    handles.append(Line2D([], [], ls="None", label=""))
    handles += [Line2D([], [], color=NEUTRAL, marker="o", ls="None", label="vanilla"),
                Line2D([], [], color=NEUTRAL, marker="D", ls="None", label="NOTREKS")]
    handles.append(Line2D([], [], ls="None", label=""))
    handles += [Line2D([], [], color=NEUTRAL, marker="o", ls="None", label="$n=100$"),
                Line2D([], [], color=NEUTRAL, marker="^", ls="None", label="$n=500$"),
                Line2D([], [], color=NEUTRAL, marker="s", ls="None", label="$n=2000$"),
                Line2D([], [], color=NEUTRAL, marker="o", mfc=light_fill(NEUTRAL), ls="None", label="$q=25\\%$"),
                Line2D([], [], color=NEUTRAL, marker="o", mfc=NEUTRAL, ls="None", label="$q=100\\%$")]
    fig.legend(handles=handles, ncol=1, loc="upper center", bbox_to_anchor=(.5, 1.02),
               frameon=True, facecolor="white", framealpha=.92, fontsize=6.2)
    fig.text(.5, .012, f"main: d={dimension}; graph-type ranking; n=100/500/2000; q=.25/1",
             ha="center", va="bottom", fontsize=5.2, color=".28")
    fig.subplots_adjust(top=.72, bottom=.15, left=.09, right=.98, wspace=.28)
    save(fig, out / f"figure1_paired_pareto_rank_d{dimension}"); plt.close(fig)

def _exact_chromatic_number(pairs, d):
    adj = [set() for _ in range(d)]
    for a, b in pairs:
        adj[a].add(b); adj[b].add(a)
    order = sorted(range(d), key=lambda v: len(adj[v]), reverse=True)
    colour = [-1] * d
    best = d
    def search(pos, used):
        nonlocal best
        if used >= best: return
        if pos == d:
            best = used; return
        v = order[pos]
        forbidden = {colour[u] for u in adj[v] if colour[u] >= 0}
        for c in range(min(used + 1, best)):
            if c not in forbidden:
                colour[v] = c
                search(pos + 1, max(used, c + 1))
                colour[v] = -1
    search(0, 0)
    return int(best)


def prior_correlations(df,out,root):
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr,spearmanr
    z=df[df.experiment_id == "prior-structure"].copy()
    z=z[np.isclose(pd.to_numeric(z.knowledge_fraction,errors="coerce"),.25)].copy()
    if z.empty: return
    # Use the integration-ablation performance as the normalization reference
    # rather than the vanilla error of each individual prior instance.  This
    # keeps the scale tied to the benchmark's +NOTREKS operating point.
    ablation = df[df.experiment_id == "integration-ablation"].copy()
    ablation = ablation[np.isclose(
        pd.to_numeric(ablation.knowledge_fraction, errors="coerce"), .25
    )]
    ablation_reference = {}
    for solver, method in (("FLOP", "flop_notreks"), ("DAGMA", "dagma_notreks")):
        values = pd.to_numeric(
            ablation.loc[ablation.method == method, "SHD_cpdag"], errors="coerce"
        ).dropna()
        if not values.empty:
            ablation_reference[solver] = float(values.mean())
    # The protocol runner stores prior properties on every solver row.  Older
    # rows may lack exact chromatic numbers; recover them from the immutable
    # knowledge JSON rather than silently dropping the study.
    for prior_id, idx in z.groupby("prior_id").groups.items():
        path=root / "knowledge" / f"{prior_id}.json"
        if path.exists():
            import json
            pairs=[tuple(p) for p in json.loads(path.read_text())["pairs"]]
            chi=_exact_chromatic_number(pairs, int(z.loc[idx[0], "d"]))
            z.loc[idx, "chromatic_number_exact"] = chi
    # These are properties of the supplied NOTREKS graph H, plus alignment of
    # H with the vanilla solver's forbidden treks—not properties of G_true.
    props=["chromatic_number_exact","prior_endpoint_coverage",
           "prior_connected_components","prior_largest_component",
           "prior_degree_max","trek_error_alignment"]; rows=[]
    paired=[]
    for (data_id, prior_id, method), g in z.groupby(["data_id","prior_id","method"]):
        if method not in ("flop", "flop_notreks", "dagma", "dagma_notreks"): continue
        paired.append(g.iloc[0])
    pz=pd.DataFrame(paired)
    for solver, base, nt in (("FLOP","flop","flop_notreks"),("DAGMA","dagma","dagma_notreks")):
        b=pz[pz.method==base]; t=pz[pz.method==nt]
        base_cols = ["data_id", "prior_id", "SHD_cpdag", "trek_error_alignment"]
        base_cols = [c for c in base_cols if c in b.columns]
        merged=t.merge(b[base_cols],on=["data_id","prior_id"],suffixes=("_nt","_base"))
        merged["cpdag_SHD_gain"] = merged["SHD_cpdag_base"] - merged["SHD_cpdag_nt"]
        # Center by the corresponding mean vanilla+NOTREKS SHD from the
        # integration ablation. Values below that reference are negative and
        # values above it are positive; this is deliberately not fractional.
        reference = ablation_reference.get(solver, np.nan)
        merged["centered_cpdag_SHD_gain"] = merged["cpdag_SHD_gain"] - reference
        merged["ablation_mean_vanilla_NOTREKS_SHD"] = reference
        merged["solver"] = solver
        if not merged.empty:
            merged["method"] = base
            # Alignment must describe the vanilla candidate.  The NOTREKS
            # candidate is feasible by construction, so using its alignment
            # would make this diagnostic identically zero.
            if "trek_error_alignment_base" in merged:
                merged["trek_error_alignment"] = merged["trek_error_alignment_base"]
            elif "trek_error_alignment" not in merged:
                merged["trek_error_alignment"] = np.nan
            # Keep both solver-specific copies for plotting/correlation.
            if "_nt" in "".join(merged.columns): pass
            for prop in props:
                if prop not in merged: merged[prop] = np.nan
            if not merged.empty:
                if "_nt" in "".join(merged.columns):
                    pass
            rows.extend(merged.to_dict("records"))
    z=pd.DataFrame(rows)
    if z.empty: return
    corr_rows=[]
    gain_col = "centered_cpdag_SHD_gain"
    for prop in props:
        if prop not in z: continue
        for scope,g in [("pooled",z),*[(f"method={m}",h) for m,h in z.groupby("solver")]]:
            a=g[[prop,gain_col]].apply(pd.to_numeric,errors="coerce").dropna()
            if len(a)<3: continue
            if a[prop].nunique() < 2 or a[gain_col].nunique() < 2:
                sr = pr = np.nan
            else:
                sr = spearmanr(a[prop],a[gain_col]).statistic
                pr = pearsonr(a[prop],a[gain_col]).statistic
            corr_rows.append(dict(scope=scope,property=prop,knowledge_fraction=.25,
                                  gain_definition="centered_gain=delta_SHD-mean_ablation_vanilla_NOTREKS_SHD",
                                  spearman_rho=sr,pearson_r=pr,n=len(a)))
    corr=pd.DataFrame(corr_rows); corr.to_csv(out/"plot_data_prior_structure_correlations.csv",index=False); available=[p for p in props if p in z and z[p].notna().any()]
    labels={"chromatic_number_exact":"chromatic number",
            "prior_endpoint_coverage":"distinct-node coverage",
            "prior_connected_components":"connected components",
            "prior_largest_component":"largest component",
            "prior_degree_max":"maximum prior degree",
            "trek_error_alignment":"trek-error alignment (ratio)"}
    show=[p for p in ("chromatic_number_exact","prior_endpoint_coverage",
                      "prior_connected_components","prior_largest_component",
                      "prior_degree_max","trek_error_alignment") if p in available][:6]
    fig,axes=plt.subplots(2,3,figsize=(6.65,3.85),squeeze=False)
    for ax,prop in zip(axes.flat,show):
        for solver,g in z.groupby("solver"):
            if prop not in g: continue
            a=g[[prop,gain_col]].copy()
            a[prop]=pd.to_numeric(a[prop],errors="coerce"); a[gain_col]=pd.to_numeric(a[gain_col],errors="coerce"); a=a.dropna()
            ax.scatter(a[prop],a[gain_col],s=13,alpha=.38,edgecolors="none",color=FLOP if solver=="FLOP" else DAGMA,label=solver)
        cc=corr[(corr.property==prop)&(corr.scope=="pooled")]; rr=cc.spearman_rho.iloc[0] if not cc.empty else np.nan
        method_rhos={}
        for method,colour in (("FLOP",FLOP),("DAGMA",DAGMA)):
            m=corr[(corr.property==prop)&(corr.scope==f"method={method}")]
            method_rhos[method]=(m.spearman_rho.iloc[0] if not m.empty else np.nan, colour)
        ax.set_title(labels.get(prop,prop),fontsize=8)
        # Keep the two correlation labels horizontal at the bottom.
        for x_pos, method in ((.04, "FLOP"), (.30, "DAGMA")):
            rho, colour = method_rhos[method]
            m=corr[(corr.property==prop)&(corr.scope==f"method={method}")]
            label=(f"$\\rho_s$={rho:.2f}"
                   if np.isfinite(rho)
                   else "$\\rho_s$=n/a")
            ax.text(x_pos,.04,label,transform=ax.transAxes,va="bottom",fontsize=5.6,
                    color=colour, bbox=dict(boxstyle="round,pad=.22", facecolor=light_fill(colour,.82),
                                            edgecolor=colour, linewidth=.6))
        ax.axhline(0,color=".45",lw=.7,ls="--",alpha=.8)
        # Centered gains can be positive or negative, so a signed-log scale
        # is the meaningful logarithmic analogue here.
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.15, base=10)
        ax.set_xlabel("");
        if ax in axes[:,0]: ax.set_ylabel("Centered paired $\\Delta$SHD")
        else: ax.set_ylabel("")
        ax.grid(alpha=.15); ax.spines[["top","right"]].set_visible(False)
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker="o",ls="None",color=FLOP,label="FLOP"),
                       Line2D([],[],marker="o",ls="None",color=DAGMA,label="DAGMA"),
                       Line2D([],[],marker="",ls="None",color="none",label="Spearman's $\\rho$")],
               loc="upper center",bbox_to_anchor=(.5,1.01),ncol=3,
               frameon=False,fontsize=7)
    fig.text(.5,.015,"prior-structure: d=20, n=500; ER2/4 and WS2/4; q=.25; 2 graph replicates × 5 draws; positive $\\Delta$SHD is better.",ha="center",fontsize=6.5)
    fig.subplots_adjust(top=.88,hspace=.48,wspace=.35,bottom=.14); save(fig,out/"figure3_prior_structure_correlations"); plt.close(fig)

def bic_gap(df,out):
    import matplotlib.pyplot as plt
    x=df[df.experiment_id=="main"]; rows=[]
    for base,(solver,colour,nt) in METHODS.items():
        b=base_rows(x,base)
        for n,g in b.groupby("n"): rows.append(dict(solver=solver,method="vanilla",n=n,mean=g.bic_gap_to_truth.mean(),q25=g.bic_gap_to_truth.quantile(.25),q75=g.bic_gap_to_truth.quantile(.75),runs=len(g)))
        for q,g in x[x.method==nt].groupby("knowledge_fraction"):
            for n,h in g.groupby("n"): rows.append(dict(solver=solver,method=f"NOTREKS q={q:g}",n=n,mean=h.bic_gap_to_truth.mean(),q25=h.bic_gap_to_truth.quantile(.25),q75=h.bic_gap_to_truth.quantile(.75),runs=len(h)))
    src=pd.DataFrame(rows); src.to_csv(out/"plot_data_bic_gap_to_truth.csv",index=False); fig,ax=plt.subplots(figsize=(4.45,4.15))
    for (solver,method),g in src.groupby(["solver","method"]):
        g=g.sort_values("n"); c=FLOP if solver=="FLOP" else DAGMA
        knowledge = "vanilla" if method == "vanilla" else (".25" if "0.25" in method else "1")
        knowledge_line(ax,g.n,g["mean"],c,knowledge,marker="o",ms=4,label=f"{solver} {method}")
    # BIC gaps span a very different range for FLOP and DAGMA.  A signed
    # logarithmic scale preserves the zero reference and exposes small FLOP
    # deviations without flattening them against the DAGMA curves.
    ax.axhline(0,color=".45",lw=.8); ax.set_xscale("log"); ax.set_yscale("symlog",linthresh=10,linscale=1.25,base=10)
    ax.set_xticks([100,500,2000],["100","500","2000"]); ax.set_xlabel("sample size $n$"); ax.set_ylabel("BIC gap to truth (lower is better)"); ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False); ax.legend(ncol=2,fontsize=6.2,frameon=True,facecolor="white",framealpha=.9,loc="lower right",bbox_to_anchor=(.99,.20)); ax.text(.99,.01,"main: d=20/50; ER2/4/8 and WS2/4/8; n=100/500/2000; q=.25/1; 2 graph replicates",transform=ax.transAxes,ha="right",va="bottom",fontsize=5.2,color=".28",bbox=dict(facecolor="white",alpha=.82,edgecolor="none",pad=1.5)); fig.subplots_adjust(left=.16,right=.98,bottom=.15,top=.97); save(fig,out/"figure4_bic_gap_to_truth"); plt.close(fig)

def radar_appendix(df,out):
    """Appendix heterogeneity view: rows are methods, columns dimensions."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    x=df[df.experiment_id=="main"]
    all_graph_types=[("er",2),("er",4),("er",8),("ws",2),("ws",4),("ws",8)]
    n=500
    fig,axes=plt.subplots(2,2,subplot_kw={"projection":"polar"},figsize=(7.1,5.6))
    for r,(base,(solver,c,nt)) in enumerate(METHODS.items()):
        for col,d in enumerate((20,50)):
            ax=axes[r,col]
            graph_types=[g for g in all_graph_types if d == 50 or g[1] in (2,4)]
            ang=np.linspace(0,2*np.pi,len(graph_types),endpoint=False); aa=np.r_[ang,ang[0]]
            for q in (None,.25,1.):
                vals=[]
                for family,density in graph_types:
                    b=base_rows(x[(x.d==d)&(x.graph_family==family)&(x.graph_density==density)&(x.n==n)],base) if q is None else x[(x.d==d)&(x.graph_family==family)&(x.graph_density==density)&(x.n==n)&(x.method==nt)&np.isclose(x.knowledge_fraction.astype(float),q)]
                    vals.append(b.SHD_cpdag.mean() if not b.empty else np.nan)
                if not np.all(np.isnan(vals)):
                    knowledge_line(ax,aa,np.r_[vals,vals[0]],c,
                                   "vanilla" if q is None else (".25" if q == .25 else "1"),
                                   marker="o",ms=2.5)
            ax.set_xticks(ang); ax.set_xticklabels([f"{family.upper()}{density}" for family,density in graph_types],fontsize=5); ax.set_title(f"{solver}, $d={d}$",fontsize=8,pad=10); ax.grid(alpha=.22)
    fig.legend(handles=[Line2D([],[],color=NEUTRAL,ls=":",label="vanilla"),Line2D([],[],color=NEUTRAL,ls="--",label="25% NOTREKS"),Line2D([],[],color=NEUTRAL,ls="-",label="100% NOTREKS")],loc="upper center",bbox_to_anchor=(.5,1.045),ncol=3,frameon=False,fontsize=7)
    fig.text(.5,.02,"main: n=500; d=20 spokes ER2/4 and WS2/4; d=50 spokes ER2/4/8 and WS2/4/8; q=0/.25/1; 2 graph replicates",ha="center",fontsize=6.5); fig.subplots_adjust(top=.88,hspace=.28,wspace=.28); save(fig,out/"supp_heterogeneity_radar"); plt.close(fig)


def _special_sachs_figure(df, out, metric="SHD_cpdag"):
    """Figure-layer Sachs summary, using the same paired-scatter language."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    x = df.copy()
    if "status" in x.columns:
        x = x[x.status == "ok"]
    if x.empty:
        return
    x["knowledge_fraction"] = pd.to_numeric(x["knowledge_fraction"], errors="coerce")
    # Vanilla is independent of q and is shown once. Constrained methods are
    # shown at both requested knowledge levels.
    keep = x[(x.method.isin(["flop", "dagma"]) &
              np.isclose(x.knowledge_fraction, 1.0)) |
             (x.method.isin(["flop-nt-standard", "dagma-pstrek"]) &
              x.knowledge_fraction.isin([.25, 1.0]))].copy()
    if metric not in keep.columns:
        return
    summary = keep.groupby(["method", "knowledge_fraction"], as_index=False).agg(
        value=(metric, "mean"), runtime=("candidate_runtime", "mean"))
    names = {"flop": ("FLOP", FLOP), "flop-nt-standard": ("FLOP+NOTREKS", FLOP),
             "dagma": ("DAGMA", DAGMA), "dagma-pstrek": ("DAGMA+NOTREKS", DAGMA)}
    fig, ax = plt.subplots(figsize=(5.25, 4.7))
    for _, r in summary.iterrows():
        label, colour = names.get(r.method, (r.method, NEUTRAL))
        is_vanilla = r.method in {"flop", "dagma"}
        q = "vanilla" if is_vanilla else ("25%" if np.isclose(r.knowledge_fraction, .25) else "100%")
        face = "none" if q == "vanilla" else (light_fill(colour, .68) if q == "25%" else colour)
        marker = "o" if label.startswith("FLOP") else "s"
        ax.scatter(r.runtime, r.value, s=68, marker=marker, facecolor=face,
                   edgecolor=colour, linewidth=.9, zorder=4)
    handles = [Line2D([], [], marker="o", color=FLOP, markerfacecolor="none", ls="None", label="FLOP"),
               Line2D([], [], marker="s", color=DAGMA, markerfacecolor="none", ls="None", label="DAGMA"),
               Line2D([], [], marker="o", color=NEUTRAL, markerfacecolor="none", ls="None", label="vanilla"),
               Line2D([], [], marker="o", color=NEUTRAL, markerfacecolor=light_fill(FLOP), ls="None", label="25% NOTREKS"),
               Line2D([], [], marker="o", color=NEUTRAL, markerfacecolor=NEUTRAL, ls="None", label="100% NOTREKS")]
    ax.legend(handles=handles, ncol=3, fontsize=7, frameon=True, loc="upper left")
    ylabel = "mean CPDAG SHD" if metric == "SHD_cpdag" else "mean CPDAG Ancestor-AID"
    suffix = "" if metric == "SHD_cpdag" else "_ancestor_aid"
    ax.set_xlabel("mean runtime (s)"); ax.set_ylabel(ylabel)
    ax.set_title("Sachs observational data")
    ax.text(.99, .02, "853 observations; 50 bootstrap replicates; 11 nodes", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color=".3")
    ax.grid(alpha=.18); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save(fig, out / f"figure_sachs_paired_pareto{suffix}"); plt.close(fig)


def _special_tcc_figure(df, out, metric="SHD_cpdag"):
    """Six-panel PSTrek/TCC comparison in the normal protocol figure folder."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    x = df.copy()
    if "status" in x:
        x = x[x.status == "ok"]
    if "solver_status" in x:
        x = x[x.solver_status == "ok"]
    if x.empty:
        return
    # Protocol-all uses graph_family/graph_density; accept the direct names
    # used by diagnostic CSVs as well.
    family = "graph_family" if "graph_family" in x else "graph_type"
    density = "graph_density" if "graph_density" in x else "density"
    cells = sorted(x[["d", family, density]].drop_duplicates().itertuples(index=False, name=None))
    if not cells:
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.25), sharey=True)
    axes = np.asarray(axes).ravel()
    colours = {"dagma_notreks": DAGMA, "dagma_notreks_tcc": "#4477AA",
               "dagma-pstrek": DAGMA, "TCC": "#4477AA"}
    labels = {"dagma_notreks": "PSTrek", "dagma_notreks_tcc": "TCC",
              "dagma-pstrek": "PSTrek", "TCC": "TCC"}
    for ax, cell in zip(axes, cells):
        d, fam, den = cell
        sub = x[(x.d == d) & (x[family] == fam) & (x[density] == den)]
        if metric not in sub.columns:
            continue
        means = sub.groupby("method")[metric].mean()
        methods = [m for m in ("dagma_notreks", "dagma_notreks_tcc", "dagma-pstrek", "TCC") if m in means.index]
        ax.bar([labels[m] for m in methods], [means[m] for m in methods],
               color=[colours[m] for m in methods], width=.62)
        ax.set_title(f"d={int(d)}, {str(fam).upper()}{int(den)}", fontsize=8)
        ax.grid(axis="y", alpha=.18); ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("mean CPDAG SHD" if metric == "SHD_cpdag" else "mean CPDAG Ancestor-AID")
    for ax in axes[len(cells):]:
        ax.set_visible(False)
    axes[-1].set_xlabel("constraint")
    fig.legend(handles=[Patch(facecolor=DAGMA, label="PSTrek"), Patch(facecolor="#4477AA", label="TCC")],
               loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=8)
    fig.text(.5, .01, "n=100; five graph replicates per setting; knowledge fraction q=1", ha="center", fontsize=6)
    fig.subplots_adjust(top=.86, bottom=.11, hspace=.38, wspace=.24)
    suffix = "" if metric == "SHD_cpdag" else "_ancestor_aid"
    save(fig, out / f"figure_pstrek_vs_tcc_six_panels{suffix}"); plt.close(fig)


def write_special_protocol_figures(df, out):
    """Write non-synthetic experiment figures through the shared figure layer."""
    out.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        return
    if "experiment" in df.columns and (df["experiment"] == "sachs_fixed").any():
        sachs = df[df["experiment"] == "sachs_fixed"]
        _special_sachs_figure(sachs, out, "SHD_cpdag")
        _special_sachs_figure(sachs, out, "Ancestor_AID_cpdag")
    if "experiment_id" in df.columns and (df["experiment_id"] == "pstrek-vs-tcc").any():
        tcc = df[df["experiment_id"] == "pstrek-vs-tcc"]
        _special_tcc_figure(tcc, out, "SHD_cpdag")
        _special_tcc_figure(tcc, out, "Ancestor_AID_cpdag")


def write_protocol_figures(df, out, root):
    """Create the agreed publication and appendix figure set."""
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    write_special_protocol_figures(df, out)
    # Non-synthetic experiment tables do not contain the full synthetic
    # schema. Their figures were already written above.
    required_synthetic = {"experiment_id", "data_id", "d", "graph_family",
                          "graph_density", "knowledge_fraction"}
    if not required_synthetic.issubset(df.columns):
        return
    df = ensure_bic_gap(df, root)
    pareto(df, out)
    pareto(df, out, metric="Ancestor_AID_cpdag", suffix="_cpdag_aid")
    pareto(df, out, all_datasets=True, suffix="_all_datasets")
    ablation_pareto(df, out, dimension=50)
    main_rank(df, out, dimension=20)
    main_rank(df, out, dimension=50)
    if (df.experiment_id == "main").any() and (df.d == 100).any():
        main_rank(df, out, dimension=100)
    ablation_rank(df, out, dimension=50)
    if (df.experiment_id == "main").any() and (df.d == 100).any():
        ablation_pareto(df, out, dimension=100)
        ablation_rank(df, out, dimension=100)
    elif (df.experiment_id == "d100-flop").any():
        ablation_pareto(df, out, dimension=100)
        ablation_rank(df, out, dimension=100)
    prior_correlations(df, out, root)
    bic_gap(df, out)
    radar_appendix(df, out)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--prior-root",type=Path,default=None); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); df=pd.read_parquet(a.input) if a.input.suffix == ".parquet" else pd.read_csv(a.input); write_protocol_figures(df,a.output,a.prior_root or a.input.parent.parent.parent); print(f"publication figures written to {a.output}")
if __name__=="__main__": main()
