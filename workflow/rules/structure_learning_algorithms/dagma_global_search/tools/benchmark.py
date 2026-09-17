"""Bounded comparison using the saved oracle d=20 NOTREKS validation data."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import flopsearch

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import gaussian_bic, is_dag
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import ProductionConfig, run_production_pipeline
from workflow.rules.structure_learning_algorithms.flop_soft_notreks import (
    SoftGreedyConfig, fit_soft_notreks,
)


def pairs_from_truth(truth):
    reach = truth.astype(bool).copy(); np.fill_diagonal(reach, True)
    for k in range(len(reach)): reach |= reach[:, [k]] & reach[[k], :]
    return [(i, j) for i in range(len(reach)) for j in range(i+1, len(reach)) if not np.any(reach[:, i] & reach[:, j])]


def metrics(A, truth, X, method, runtime, **extra):
    A = np.asarray(A, dtype=np.uint8); bic, _ = gaussian_bic(X, A, lambda_bic=2.)
    sk=(A|A.T).astype(bool); target=(truth|truth.T).astype(bool); upper=np.triu(np.ones_like(A,dtype=bool),1)
    tp=int(np.sum(sk&target&upper)); fp=int(np.sum(sk&~target&upper)); fn=int(np.sum(~sk&target&upper)); dtp=int(np.sum(A&truth)); dfp=int(np.sum(A&~truth)); dfn=int(np.sum((~A.astype(bool))&truth))
    return {"method":method,"bic":float(bic),"data_fit":float(bic-2*A.sum()*np.log(len(X))),"edges":int(A.sum()),"directed_shd":int(np.sum(A!=truth)),"skeleton_shd":fp+fn,"skeleton_f1":2*tp/max(1,2*tp+fp+fn),"directed_f1":2*dtp/max(1,2*dtp+dfp+dfn),"acyclic":bool(is_dag(A)),"runtime":float(runtime),**extra}


def load(root, seed):
    X=pd.read_csv(root/"resources/data/mydatasets/oracle_validation_d20_inv_t02"/f"s{seed}.csv").to_numpy(float); X=(X-X.mean(0))/X.std(0,ddof=0)
    truth=pd.read_csv(root/"resources/adjmat/myadjmats/oracle_validation_d20_inv_t02"/f"g{seed}.csv").to_numpy(np.uint8)
    return X, truth


def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds",nargs="+",type=int,default=[2001]); p.add_argument("--output-dir",type=Path,default=Path("results/dagma_global_search_pilot")); p.add_argument("--replicas",type=int,default=6); p.add_argument("--particles",type=int,default=32); p.add_argument("--steps-per-stage",type=int,default=2); p.add_argument("--stages",type=int,default=5); p.add_argument("--warm-iter",type=int,default=300); p.add_argument("--max-iter",type=int,default=600); p.add_argument("--full-dagma",action="store_true"); a=p.parse_args(); root=Path(__file__).resolve().parents[5]; a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in a.seeds:
        X,truth=load(root,seed); pairs=pairs_from_truth(truth)
        t=time.perf_counter(); base,restarts=run_production_pipeline(X,[],ProductionConfig() if a.full_dagma else ProductionConfig(restarts=1,warm_iter=a.warm_iter,max_iter=a.max_iter)); rows.append({**metrics(base.adjacency,truth,X,"vanilla_dagma",time.perf_counter()-t),"seed":seed,"objective_evaluations":0,"local_optimizer_calls":len(restarts),"feasible":True})
        t=time.perf_counter(); nt,_=run_production_pipeline(X,pairs,ProductionConfig() if a.full_dagma else ProductionConfig(restarts=1,warm_iter=a.warm_iter,max_iter=a.max_iter)); rows.append({**metrics(nt.adjacency,truth,X,"dagma_notreks",time.perf_counter()-t),"seed":seed,"objective_evaluations":0,"local_optimizer_calls":len(_),"feasible":nt.oracle_violations==0})
        for method,fn in [("vanilla_flop",lambda: flopsearch.flop_notreks(X,2.,[],restarts=1,seed=seed,search_version="global_greedy_rust",return_diagnostics=True)),("flop_notreks_global_greedy",lambda: flopsearch.flop_notreks(X,2.,pairs,restarts=1,seed=seed,search_version="global_greedy_rust",return_diagnostics=True))]:
            t=time.perf_counter(); _,diag=fn(); A=np.zeros_like(truth); [A.__setitem__((int(u),int(v)),1) for u,v in diag["selected_dag_edges"]]; rows.append({**metrics(A,truth,X,method,time.perf_counter()-t),"seed":seed,"objective_evaluations":0,"local_optimizer_calls":0,"feasible":True})
        t=time.perf_counter(); soft=fit_soft_notreks(X,pairs,SoftGreedyConfig(seed=seed,restarts=8,max_sweeps=12,lazy_top_k=12)); rows.append({**metrics(soft.adjacency,truth,X,"flop_soft_notreks",time.perf_counter()-t),"seed":seed,"feasible":True,"soft_score":soft.score,"soft_bic":soft.bic,"soft_notreks":soft.notreks,"raw_bic":soft.raw_bic,"raw_notreks":soft.raw_notreks,"support_evaluations":soft.support_evaluations,"cache_hits":soft.cache_hits,"distinct_supports":soft.distinct_supports,"restarts_completed":soft.restarts_completed})
        print(pd.DataFrame(rows[-6:]).to_string(index=False),flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(a.output_dir/"per_seed.csv",index=False); frame.groupby("method").agg({"bic":["mean","std","median"],"directed_shd":["mean","std"],"runtime":["mean","std"]}).to_csv(a.output_dir/"aggregate.csv"); (a.output_dir/"report.md").write_text("# DAGMA global-search pilot\n\nVanilla DAGMA and DAGMA+NOTREKS use the existing production pipeline. FLOP is structural only; it does not optimize the smooth global NOTREKS objective.\n\n"+frame.to_markdown(index=False)+"\n")

if __name__ == "__main__": main()
