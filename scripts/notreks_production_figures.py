#!/usr/bin/env python3
"""Analysis-only publication figures for completed NOTREKS experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FLOP, DAGMA, NEUTRAL = "#0072B2", "#D55E00", "#4D4D4D"
METHODS = {"flop": ("FLOP", FLOP, "flop_notreks"),
           "dagma": ("DAGMA", DAGMA, "dagma_notreks")}

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
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


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

def pareto(df, out, metric="SHD_cpdag", suffix=""):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    x = df[(df.experiment_id == "main") & (df.d == 50) &
           (df.graph_family == "er") & (df.graph_density == 4)]
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
    src=pd.DataFrame(rows); src.to_csv(out/("plot_data_paired_pareto"+suffix+".csv"),index=False)
    fig,ax=plt.subplots(figsize=(4.45,5.35)); markers={100:"o",500:"^",2000:"s"}
    for r in src.itertuples():
        c=FLOP if r.solver=="FLOP" else DAGMA
        fc = "none" if r.knowledge == "vanilla" else (light_fill(c) if r.knowledge == "25% NOTREKS" else c)
        # Draw opaque variability bars first and the marker afterwards so the
        # interval line never shows through the point.
        ax.errorbar(r.runtime_mean,r.SHD_mean,
                    yerr=[[r.SHD_sd], [r.SHD_sd]],fmt="none",color=c,
                    capsize=2.2,lw=.8,zorder=2)
        ax.scatter(r.runtime_mean,r.SHD_mean,marker=markers[r.n],s=42,color=c,
                   facecolors=fc,edgecolors=c,linewidths=1.1,zorder=4)
    ax.set_xscale("log"); ax.set_yscale("symlog",linthresh=10,linscale=1.4); ax.set_ylim(0,300)
    ax.set_yticks([0,2,4,6,8,10,20,50,100,200,300]); ax.set_yticklabels(["0","2","4","6","8","10","20","50","100","200","300"])
    ylabel={"SHD_cpdag":"CPDAG SHD","Parent_AID_cpdag":"CPDAG Parent-AID","Ancestor_AID_cpdag":"CPDAG Ancestor-AID"}.get(metric,metric)
    ax.set_xlabel("mean runtime (s)"); ax.set_ylabel(ylabel+" (lower is better)"); ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False)
    # Three explicit legend columns: method, knowledge, sample size.
    h=[Line2D([],[],color=FLOP,marker="o",ls="None",label="FLOP"),
       Line2D([],[],color=DAGMA,marker="o",ls="None",label="DAGMA"),
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
    ax.text(.99,.01,"main: d=50, ER4; 2 graph replicates; q=.25 (5 draws)",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=5.4,
            color=".28",bbox=dict(facecolor="white",alpha=.82,edgecolor="none",pad=1.5))
    fig.subplots_adjust(top=.98,left=.17,right=.98,bottom=.13); save(fig,out/("figure1_paired_pareto_aggregate"+suffix)); plt.close(fig)

def ablation(df,out, dimension=50, metric="SHD_cpdag", suffix=""):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    experiment = "integration-ablation" if dimension == 50 else "d100-flop"
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
                for _,r in z.iterrows(): rows.append(dict(solver=solver,variant=name,marker=marker,q=q,runtime=r.candidate_runtime_variant,delta_SHD=r[f"{metric}_base"]-r[f"{metric}_variant"],graph_type=f"{str(r.get('graph_family_variant', r.get('graph_family', ''))).upper()}{int(r.get('graph_density_variant', r.get('graph_density', 0)))}"))
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
        # Keep the solver labels in a fixed vertical order: FLOP above DAGMA.
        for y_pos, method in ((.15, "FLOP"), (.06, "DAGMA")):
            rho, colour = method_rhos[method]
            m=corr[(corr.property==prop)&(corr.scope==f"method={method}")]
            label=(f"$\\rho_s$={rho:.2f}"
                   if np.isfinite(rho)
                   else "$\\rho_s$=n/a")
            ax.text(.04,y_pos,label,transform=ax.transAxes,va="top",fontsize=5.6,
                    color=colour, bbox=dict(boxstyle="round,pad=.22", facecolor=light_fill(colour,.82),
                                            edgecolor=colour, linewidth=.6))
        ax.axhline(0,color=".45",lw=.7,ls="--",alpha=.8)
        ax.set_xlabel("");
        if ax in axes[:,0]: ax.set_ylabel("Centered paired $\\Delta$SHD")
        else: ax.set_ylabel("")
        ax.grid(alpha=.15); ax.spines[["top","right"]].set_visible(False)
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker="o",ls="None",color=FLOP,label="FLOP"),Line2D([],[],marker="o",ls="None",color=DAGMA,label="DAGMA")],loc="upper center",bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=7,title="Spearman's $\\rho$",title_fontsize=7.2)
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
    ax.set_xticks([100,500,2000],["100","500","2000"]); ax.set_xlabel("sample size $n$"); ax.set_ylabel("BIC gap to truth (lower is better)"); ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False); ax.legend(ncol=2,fontsize=6.2,frameon=True,facecolor="white",framealpha=.9,loc="upper left"); ax.text(.99,.01,"main: d=20/50; ER2/4/8 and WS2/4/8; n=100/500/2000; q=.25/1; 2 graph replicates",transform=ax.transAxes,ha="right",va="bottom",fontsize=5.2,color=".28",bbox=dict(facecolor="white",alpha=.82,edgecolor="none",pad=1.5)); fig.subplots_adjust(left=.16,right=.98,bottom=.15,top=.97); save(fig,out/"figure4_bic_gap_to_truth"); plt.close(fig)

def radar_appendix(df,out):
    """Appendix heterogeneity view: rows are methods, columns dimensions."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    x=df[df.experiment_id=="main"]
    graph_types=[("er",2),("er",4),("er",8),("ws",2),("ws",4),("ws",8)]
    n=500
    ang=np.linspace(0,2*np.pi,len(graph_types),endpoint=False); aa=np.r_[ang,ang[0]]
    fig,axes=plt.subplots(2,2,subplot_kw={"projection":"polar"},figsize=(7.1,5.6))
    for r,(base,(solver,c,nt)) in enumerate(METHODS.items()):
        for col,d in enumerate((20,50)):
            ax=axes[r,col]
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
    fig.legend(handles=[Line2D([],[],color=NEUTRAL,ls=":",label="vanilla"),Line2D([],[],color=NEUTRAL,ls="--",label="25% NOTREKS"),Line2D([],[],color=NEUTRAL,ls="-",label="100% NOTREKS")],loc="upper center",bbox_to_anchor=(.5,1.01),ncol=3,frameon=False,fontsize=7)
    fig.text(.5,.02,"main: n=500; spokes are ER2/4/8 and WS2/4/8; dimensions d=20/50; q=0/.25/1; 2 graph replicates",ha="center",fontsize=6.5); fig.subplots_adjust(top=.88,hspace=.28,wspace=.28); save(fig,out/"supp_heterogeneity_radar"); plt.close(fig)


def write_protocol_figures(df, out, root):
    """Create the agreed publication and appendix figure set."""
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    df = ensure_bic_gap(df, root)
    pareto(df, out)
    ablation(df, out, dimension=50)
    if (df.experiment_id == "d100-flop").any():
        ablation(df, out, dimension=100)
    prior_correlations(df, out, root)
    bic_gap(df, out)
    radar_appendix(df, out)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--prior-root",type=Path,default=None); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); df=pd.read_parquet(a.input) if a.input.suffix == ".parquet" else pd.read_csv(a.input); write_protocol_figures(df,a.output,a.prior_root or a.input.parent.parent.parent); print(f"publication figures written to {a.output}")
if __name__=="__main__": main()
