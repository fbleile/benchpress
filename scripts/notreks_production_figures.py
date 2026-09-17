#!/usr/bin/env python3
"""Analysis-only publication figures for completed NOTREKS experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FLOP, DAGMA, NEUTRAL = "#0072B2", "#D55E00", "#4D4D4D"
METHODS = {"flop": ("FLOP", FLOP, "flop-nt-local"),
           "dagma": ("DAGMA", DAGMA, "dagma-pstrek")}

def light_fill(hex_colour, fraction=.68):
    """Opaque, lightened fill: distinguishes q=.25 without alpha blending."""
    rgb = np.array([int(hex_colour[i:i+2], 16) for i in (1, 3, 5)])
    return "#" + "".join(f"{int(round(v + (255-v)*fraction)):02X}" for v in rgb)

def knowledge_line(ax, x, y, colour, knowledge, **kwargs):
    """Draw knowledge as an opaque tube with a differently filled core."""
    settings = {
        "vanilla": (3.6, .16, 1.0, .42),
        ".25": (4.0, .28, 1.15, .70),
        "1": (4.4, .36, 1.35, 1.0),
    }
    tube_lw, tube_alpha, core_lw, core_alpha = settings[str(knowledge)]
    ax.plot(x, y, color=colour, lw=tube_lw, alpha=tube_alpha, **kwargs)
    ax.plot(x, y, color=colour, lw=core_lw, alpha=core_alpha, **kwargs)

def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")

def base_rows(df, method):
    x = df[df.method == method].copy()
    keys = [c for c in ("instance_id", "prior_id") if c in x]
    return x.drop_duplicates(keys or ["run_id"])

def pareto(df, out, metric="SHD_cpdag", suffix=""):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    x = df[df.experiment_id == "main"]
    rows = []
    for base, (solver, colour, nt) in METHODS.items():
        for n in (100, 500, 2000):
            b = base_rows(x[(x.d == 50) & (x.n == n)], base)
            if not b.empty:
                rows.append(dict(solver=solver, knowledge="vanilla", n=n,
                    runtime_mean=b.candidate_runtime.mean(), runtime_q25=b.candidate_runtime.quantile(.25),
                    runtime_q75=b.candidate_runtime.quantile(.75), SHD_mean=b[metric].mean(),
                    SHD_sd=b[metric].std(ddof=1), runs=len(b)))
            for q, label in ((.25, "25% NOTREKS"), (1., "100% NOTREKS")):
                t=x[(x.d==50)&(x.n==n)&(x.method==nt)&np.isclose(x.knowledge_fraction.astype(float),q)]
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
    h=[Line2D([],[],color=FLOP,marker="o",ls="None",label="FLOP"),Line2D([],[],color=DAGMA,marker="o",ls="None",label="DAGMA"),
       Line2D([],[],color=NEUTRAL,marker="o",mfc="none",ls="None",label="vanilla"),Line2D([],[],color=NEUTRAL,marker="o",mfc=light_fill(NEUTRAL),ls="None",label="25% NOTREKS"),
       Line2D([],[],color=NEUTRAL,marker="o",mfc=NEUTRAL,ls="None",label="100% NOTREKS"),Line2D([],[],color=NEUTRAL,marker="o",ls="None",label="$n=100$"),Line2D([],[],color=NEUTRAL,marker="^",ls="None",label="$n=500$"),Line2D([],[],color=NEUTRAL,marker="s",ls="None",label="$n=2000$")]
    blank=Line2D([],[],linestyle="None",label="")
    h=[h[0],h[1],blank,h[2],h[3],h[4],h[5],h[6],h[7]]
    # Keep the key inside the plotting area so the exported figure has no
    # oversized header.  The central upper region is intentionally empty in
    # this Pareto layout.
    ax.legend(handles=h,ncol=3,loc="upper left",bbox_to_anchor=(.01,.985),
              frameon=True,facecolor="white",edgecolor=".75",framealpha=.9,
              fontsize=6.4,columnspacing=.55,handletextpad=.22,borderpad=.3)
    fig.subplots_adjust(top=.98,left=.17,right=.98,bottom=.13); save(fig,out/("figure1_paired_pareto_aggregate"+suffix)); plt.close(fig)

def ablation(df,out, metric="SHD_cpdag", suffix=""):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    x=df[(df.experiment_id=="integration-ablation")&(df.d==50)&(df.er_degree==4)]
    # Deliberately avoid circle/triangle/square: those symbols encode sample
    # size in the Pareto figure.  These symbols encode ablation variant only.
    rows=[]; variants={"edge masking":("{base}-edge-mask","P"),"post-repair":("{base}-nt-post","H"),"NOTREKS integrated":(None,"D")}
    for base,(solver,colour,nt) in METHODS.items():
        b=base_rows(x,base); keys=[c for c in ("instance_id","prior_id") if c in b]
        for name,(template,marker) in variants.items():
            method=(nt if template is None else template.format(base=base)); t=x[x.method==method]
            for q in sorted(t.knowledge_fraction.dropna().unique()):
                z=t[np.isclose(t.knowledge_fraction.astype(float),q)].merge(b,on=keys,suffixes=("_variant","_base"))
                for _,r in z.iterrows(): rows.append(dict(solver=solver,variant=name,marker=marker,q=q,runtime=r.candidate_runtime_variant,delta_SHD=r[f"{metric}_base"]-r[f"{metric}_variant"]))
    src=pd.DataFrame(rows); src.to_csv(out/("plot_data_ablation"+suffix+".csv"),index=False); fig,ax=plt.subplots(figsize=(4.45,5.35))
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
    fig.subplots_adjust(top=.98,left=.17,right=.98,bottom=.13); save(fig,out/("figure2_integration_ablation"+suffix)); plt.close(fig)

def prior_correlations(df,out,root):
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr,spearmanr
    files=[p for p in root.rglob("per_prior.csv") if "chromatic" in str(p).lower()]; frames=[]
    for p in files:
        try: frames.append(pd.read_csv(p))
        except Exception: pass
    if not frames: return
    z=pd.concat(frames,ignore_index=True).drop_duplicates()
    # Fix the information quantity at q=.25; the main figure covers the
    # already-established effect of changing the quantity itself.
    z=z[np.isclose(pd.to_numeric(z.knowledge_fraction,errors="coerce"),.25)].copy()
    props=["chromatic_number_exact","prior_endpoint_coverage","prior_connected_components",
           "prior_largest_component","trek_error_alignment",
           "trek_forbidden_pairs"]; rows=[]
    for prop in props:
        if prop not in z: continue
        for scope,g in [("pooled",z),*[(f"method={m}",h) for m,h in z.groupby("method")]]:
            a=g[[prop,"cpdag_SHD_gain"]].apply(pd.to_numeric,errors="coerce").dropna()
            if len(a)<3: continue
            if a[prop].nunique() < 2 or a.cpdag_SHD_gain.nunique() < 2:
                sr = pr = np.nan
            else:
                sr = spearmanr(a[prop],a.cpdag_SHD_gain).statistic
                pr = pearsonr(a[prop],a.cpdag_SHD_gain).statistic
            rows.append(dict(scope=scope,property=prop,knowledge_fraction=.25,spearman_rho=sr,pearson_r=pr,n=len(a)))
    corr=pd.DataFrame(rows); corr.to_csv(out/"plot_data_prior_structure_correlations.csv",index=False); available=[p for p in props if p in z and z[p].notna().any()]
    labels={"chromatic_number_exact":"chromatic number",
            "prior_endpoint_coverage":"distinct-node coverage",
            "prior_connected_components":"connected components",
            "prior_largest_component":"largest component",
            "trek_error_alignment":"trek-error alignment (ratio)",
            "trek_forbidden_pairs":"forbidden treks in vanilla (count)"}
    show=[p for p in ("chromatic_number_exact","prior_endpoint_coverage",
                      "prior_connected_components","prior_largest_component",
                      "trek_error_alignment","trek_forbidden_pairs") if p in available][:6]
    fig,axes=plt.subplots(2,3,figsize=(6.65,3.85),squeeze=False)
    for ax,prop in zip(axes.flat,show):
        for solver,g in z.groupby("method"):
            if prop not in g: continue
            a=g[[prop,"cpdag_SHD_gain"]].copy()
            a[prop]=pd.to_numeric(a[prop],errors="coerce"); a.cpdag_SHD_gain=pd.to_numeric(a.cpdag_SHD_gain,errors="coerce"); a=a.dropna()
            ax.scatter(a[prop],a.cpdag_SHD_gain,s=13,alpha=.38,edgecolors="none",color=FLOP if solver=="flop" else DAGMA,label=solver.upper())
        cc=corr[(corr.property==prop)&(corr.scope=="pooled")]; rr=cc.spearman_rho.iloc[0] if not cc.empty else np.nan
        method_rhos={}
        for method,colour in (("flop",FLOP),("dagma",DAGMA)):
            m=corr[(corr.property==prop)&(corr.scope==f"method={method}")]
            method_rhos[method]=(m.spearman_rho.iloc[0] if not m.empty else np.nan, colour)
        ax.set_title(labels.get(prop,prop),fontsize=8)
        for y_pos,(method,(rho,colour)) in zip((.15,.06),method_rhos.items()):
            m=corr[(corr.property==prop)&(corr.scope==f"method={method}")]
            label=(f"$\\rho_s$={rho:.2f}"
                   if np.isfinite(rho)
                   else "$\\rho_s$=n/a")
            ax.text(.04,y_pos,label,transform=ax.transAxes,va="top",fontsize=5.6,
                    color=colour, bbox=dict(boxstyle="round,pad=.22", facecolor=light_fill(colour,.82),
                                            edgecolor=colour, linewidth=.6))
        ax.axhline(0,color=".45",lw=.7); ax.set_xlabel("");
        if ax in axes[:,0]: ax.set_ylabel("Paired $\\Delta$SHD")
        else: ax.set_ylabel("")
        ax.grid(alpha=.15); ax.spines[["top","right"]].set_visible(False)
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker="o",ls="None",color=FLOP,label="FLOP"),Line2D([],[],marker="o",ls="None",color=DAGMA,label="DAGMA")],loc="upper center",bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=7,title="Spearman's $\\rho$",title_fontsize=7.2)
    fig.text(.5,.015,"Fixed $q=0.25$; points are individual paired runs; positive $\\Delta$SHD is better.",ha="center",fontsize=7)
    fig.subplots_adjust(top=.88,hspace=.48,wspace=.35,bottom=.14); save(fig,out/"figure3_prior_structure_correlations"); plt.close(fig)

def bic_gap(df,out):
    import matplotlib.pyplot as plt
    x=df[df.experiment_id=="main"]; rows=[]
    for base,(solver,colour,nt) in METHODS.items():
        b=base_rows(x,base)
        for n,g in b.groupby("n"): rows.append(dict(solver=solver,method="vanilla",n=n,mean=g.bic_gap_to_truth.mean(),q25=g.bic_gap_to_truth.quantile(.25),q75=g.bic_gap_to_truth.quantile(.75),runs=len(g)))
        for q,g in x[x.method==nt].groupby("knowledge_fraction"):
            for n,h in g.groupby("n"): rows.append(dict(solver=solver,method=f"NOTREKS q={q:g}",n=n,mean=h.bic_gap_to_truth.mean(),q25=h.bic_gap_to_truth.quantile(.25),q75=h.bic_gap_to_truth.quantile(.75),runs=len(h)))
    src=pd.DataFrame(rows); src.to_csv(out/"plot_data_bic_gap_to_truth.csv",index=False); fig,ax=plt.subplots(figsize=(6.8,3.8))
    for (solver,method),g in src.groupby(["solver","method"]):
        g=g.sort_values("n"); c=FLOP if solver=="FLOP" else DAGMA
        knowledge = "vanilla" if method == "vanilla" else (".25" if "0.25" in method else "1")
        knowledge_line(ax,g.n,g["mean"],c,knowledge,marker="o",ms=4,label=f"{solver} {method}")
        ax.fill_between(g.n,g.q25,g.q75,color=c,alpha=.08)
    # BIC gaps span a very different range for FLOP and DAGMA.  A signed
    # logarithmic scale preserves the zero reference and exposes small FLOP
    # deviations without flattening them against the DAGMA curves.
    ax.axhline(0,color=".45",lw=.8); ax.set_xscale("log"); ax.set_yscale("symlog",linthresh=10,linscale=1.25,base=10)
    ax.set_xticks([100,500,2000],["100","500","2000"]); ax.set_xlabel("sample size $n$"); ax.set_ylabel("BIC gap to truth (lower is better)"); ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False); ax.legend(ncol=2,fontsize=7,frameon=True,facecolor="white",framealpha=.9); fig.subplots_adjust(left=.12,right=.98,bottom=.16); save(fig,out/"figure4_bic_gap_to_truth"); plt.close(fig)

def radar_appendix(df,out):
    """Appendix-ready heterogeneity view; panels remain unpooled."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    x=df[df.experiment_id=="main"]; spokes=[(2,100),(2,500),(2,2000),(4,100),(4,500),(4,2000)]
    ang=np.linspace(0,2*np.pi,len(spokes),endpoint=False); aa=np.r_[ang,ang[0]]
    fig,axes=plt.subplots(2,2,subplot_kw={"projection":"polar"},figsize=(7.1,5.6))
    for r,(base,(solver,c,nt)) in enumerate(METHODS.items()):
        for col,d in enumerate((20,50)):
            ax=axes[r,col]
            for q in (None,.25,1.):
                vals=[]
                for er,n in spokes:
                    b=base_rows(x[(x.d==d)&(x.er_degree==er)&(x.n==n)],base) if q is None else x[(x.d==d)&(x.er_degree==er)&(x.n==n)&(x.method==nt)&np.isclose(x.knowledge_fraction.astype(float),q)]
                    vals.append(b.SHD_cpdag.mean() if not b.empty else np.nan)
                if not np.all(np.isnan(vals)):
                    knowledge_line(ax,aa,np.r_[vals,vals[0]],c,
                                   "vanilla" if q is None else (".25" if q == .25 else "1"),
                                   marker="o",ms=2.5)
            ax.set_xticks(ang); ax.set_xticklabels([f"ER{er}\n$n={n}$" for er,n in spokes],fontsize=5); ax.set_title(f"{solver}, $d={d}$",fontsize=8,pad=10); ax.grid(alpha=.22)
    fig.legend(handles=[Line2D([],[],color=NEUTRAL,ls=":",label="vanilla"),Line2D([],[],color=NEUTRAL,ls="--",label="25% NOTREKS"),Line2D([],[],color=NEUTRAL,ls="-",label="100% NOTREKS")],loc="upper center",bbox_to_anchor=(.5,1.01),ncol=3,frameon=False,fontsize=7)
    fig.text(.5,.02,"Spokes show ER density and sample size; panels are not pooled.",ha="center",fontsize=7); fig.subplots_adjust(top=.88,hspace=.28,wspace=.28); save(fig,out/"supp_heterogeneity_radar"); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--prior-root",type=Path,default=None); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); df=pd.read_csv(a.input); pareto(df,a.output); ablation(df,a.output)
    for metric,suffix in (("Parent_AID_cpdag","_parent_aid_cpdag"),("Ancestor_AID_cpdag","_ancestor_aid_cpdag")):
        if metric in df.columns:
            pareto(df,a.output,metric,suffix); ablation(df,a.output,metric,suffix)
    prior_correlations(df,a.output,a.prior_root or a.input.parent.parent.parent); bic_gap(df,a.output); radar_appendix(df,a.output); print(f"publication figures written to {a.output}")
if __name__=="__main__": main()
