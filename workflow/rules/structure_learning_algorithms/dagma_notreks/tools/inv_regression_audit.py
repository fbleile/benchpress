#!/usr/bin/env python3
"""Factorial audit of the historical and shared-inverse DAGMA-NOTREKS runs."""
from __future__ import annotations

import argparse

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import flopsearch
import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.end_flop_prune import end_flop_prune
from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations, gaussian_bic, is_dag, topological_order,
)
from workflow.rules.structure_learning_algorithms.dagma.knowledge import named_pairs_to_indices
from workflow.rules.structure_learning_algorithms.dagma.shared import (
    SharedDagmaLinear, deterministic_initial_adjacency,
)
from workflow.rules.structure_learning_algorithms.dagma.structural import (
    minimal_feasibility_projection,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.tools.local_smoke import _metrics
from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "results/dagma_notreks_oracle/inv_regression_audit"
HIST = ROOT / "results/dagma_notreks_oracle/flop_notreks_dense_d50_budget"
RECENT = ROOT / "results/dagma_notreks_oracle/inv_dag_notreks_benchmark"
HIST_END = ROOT / "results/dagma_notreks_oracle/end_flop_prune_heldout"
PANELS = {"A": tuple(range(5005, 5011)), "B": tuple(range(8101, 8106))}
LAMBDAS = {"fixed": .03, "scaled": .03 / np.sqrt(2.)}
METHODS = ("dagma_notreks", "inv_notreks")
POST = ("P1_fixed_001", "P2_minimal", "P3_floor_001", "P4_fixed_020")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_paths(panel: str, seed: int):
    if panel == "A":
        data = ROOT / f"resources/data/mydatasets/flop_notreks_dense_d50_budget/s{seed}.csv"
        graph = ROOT / f"resources/adjmat/myadjmats/flop_notreks_dense_d50_budget/g{seed}.csv"
        sidecar = HIST / f"seed_{seed}/no_trek_pairs.json"
    else:
        data = ROOT / f"resources/data/mydatasets/inv_dag_notreks_benchmark/s{seed}_n2000.csv"
        graph = ROOT / f"resources/adjmat/myadjmats/inv_dag_notreks_benchmark/g{seed}.csv"
        sidecar = RECENT / f"seed_{seed}/no_trek_pairs.json"
    return data, graph, sidecar


def load_inputs(panel: str, seed: int):
    data, graph, sidecar = input_paths(panel, seed)
    frame = pd.read_csv(data)
    X = frame.to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    true = pd.read_csv(graph).to_numpy(int)
    pairs = named_pairs_to_indices(json.loads(sidecar.read_text()), list(frame.columns))
    return X, true, pairs, list(frame.columns), graph


def objective(model, W):
    last = model.stage_diagnostics[-1]
    return float(last["mu"] * (model.score_final + model.lambda1 * np.abs(W).sum())
                 + model.h_final + 10. * last["raw_notreks_value"] * (2 / 49))


def continuous_task(task):
    panel, seed, method, lambda_name = task
    case = OUT / "continuous" / panel / str(seed) / method / lambda_name
    done = case / "complete.json"
    if done.exists():
        return json.loads(done.read_text())
    case.mkdir(parents=True, exist_ok=True)
    X, _, pairs, _, _ = load_inputs(panel, seed)
    rows = []
    for restart in range(5):
        path = case / f"restart_{restart}.npy"
        rowpath = case / f"restart_{restart}.json"
        if rowpath.exists() and path.exists():
            rows.append(json.loads(rowpath.read_text()))
            continue
        initial = None if restart == 0 else deterministic_initial_adjacency(
            50, seed * 1009 + restart, .05)
        model = SharedDagmaLinear("l2")
        started = time.perf_counter()
        try:
            W = model.fit(
                X.copy(), initial_W=initial, no_trek_pairs=pairs,
                trek_function="inv", trek_weight=10.,
                lambda1=LAMBDAS[lambda_name], w_threshold=0.,
                mu_schedule=[1., .1, .01, .001, .0001],
                s=[1., .9, .8, .7, .6], warm_iter=30000,
                max_iter=60000, lr=.0003, checkpoint=1000,
                beta_1=.99, beta_2=.999,
                dag_constraint=("inverse_trace" if method == "inv_notreks" else "logdet"),
            )
            np.save(path, W)
            row = {"panel": panel, "seed": seed, "method": method,
                   "lambda_name": lambda_name, "lambda1": LAMBDAS[lambda_name],
                   "restart": restart, "continuous_objective": objective(model, W),
                   "runtime": time.perf_counter() - started,
                   "iterations": int(sum(x["iterations_performed"] for x in model.stage_diagnostics)),
                   "score": float(model.score_final), "h": float(model.h_final),
                   "raw_notreks": float(model.stage_diagnostics[-1]["raw_notreks_value"]),
                   "success": True, "failure": ""}
        except Exception as exc:
            row = {"panel": panel, "seed": seed, "method": method,
                   "lambda_name": lambda_name, "lambda1": LAMBDAS[lambda_name],
                   "restart": restart, "continuous_objective": np.nan,
                   "runtime": time.perf_counter() - started, "iterations": 0,
                   "score": np.nan, "h": np.nan, "raw_notreks": np.nan,
                   "success": False, "failure": repr(exc)}
        rowpath.write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
    done.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def threshold_graph(W, threshold):
    A = (np.abs(W) >= threshold).astype(int)
    np.fill_diagonal(A, 0)
    return A


def graph_metrics(A, true, names, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, columns=names).to_csv(path, index=False)
    true_path = path.with_name("truth.csv")
    pd.DataFrame(true, columns=names).to_csv(true_path, index=False)
    return _metrics(true_path, path, path.with_name(path.stem + "_metrics.csv"))


def postprocess_all(continuous_rows, only_posts=None):
    rows = []
    for c in continuous_rows:
        if not c["success"]:
            continue
        panel, seed, method = c["panel"], int(c["seed"]), c["method"]
        X, true, pairs, names, _ = load_inputs(panel, seed)
        case = OUT / "continuous" / panel / str(seed) / method / c["lambda_name"]
        W = np.load(case / f"restart_{c['restart']}.npy")
        feas = minimal_feasibility_projection(W, pairs)
        definitions = {
            "P1_fixed_001": (.01, False),
            "P2_minimal": (float(feas["tau_feas"]), True),
            "P3_floor_001": (max(float(feas["tau_feas"]), .01),
                              float(feas["tau_feas"]) >= .01),
            "P4_fixed_020": (.20, False),
        }
        for post, (threshold, strict) in definitions.items():
            if only_posts is not None and post not in only_posts:
                continue
            candidate = (feas["graph"].copy() if post == "P2_minimal" else
                         (np.abs(W) > threshold).astype(int) if strict else
                         threshold_graph(W, threshold))
            np.fill_diagonal(candidate, 0)
            violations = common_ancestor_violations(candidate, pairs)
            eligible = is_dag(candidate) and violations == 0
            base = {**c, "postprocessing": post, "candidate_threshold": threshold,
                    "tau_DAG": float(feas["tau_dag"]), "tau_NOTREKS": float(feas["tau_mi"]),
                    "tau_joint": float(feas["tau_feas"]), "eligible": eligible,
                    "candidate_edges": int(candidate.sum()),
                    "candidate_violations": int(violations)}
            truth = true.astype(bool); support = candidate.astype(bool)
            tp = int((truth & support).sum()); fp = int((~truth & support).sum())
            base.update(candidate_true_edge_recall=tp / true.sum(),
                        candidate_precision=tp / max(1, tp + fp),
                        candidate_false_positives=fp,
                        candidate_max_indegree=int(candidate.sum(0).max()))
            if not eligible:
                rows.append({**base, "exact_bic": np.nan, "SHD_pattern": np.nan,
                             "SHD_cpdag": np.nan, "F1_skel": np.nan,
                             "F1_pattern": np.nan, "final_edges": np.nan,
                             "final_violations": np.nan, "pruning_runtime": np.nan,
                             "order_incompatible_true_edges": np.nan,
                             "missing_true_candidate_edges": int((truth & ~support).sum()),
                             "true_edges_rejected": np.nan, "false_edges_retained": np.nan})
                continue
            order = topological_order(candidate); position = np.argsort(order)
            incompatible = int(sum(position[i] > position[j] for i, j in zip(*np.nonzero(true))))
            started = time.perf_counter()
            selected, _, _ = end_flop_prune(X, candidate, lambda_bic=2.)
            prune_runtime = time.perf_counter() - started
            if np.any((selected != 0) & (candidate == 0)):
                raise RuntimeError("End-FLOP added an edge")
            final_violations = common_ancestor_violations(selected, pairs)
            if final_violations:
                raise RuntimeError("End-FLOP broke a zero-violation support")
            metric = graph_metrics(selected, true, names, case / f"r{c['restart']}_{post}.csv")
            final = selected.astype(bool)
            rows.append({**base, "exact_bic": gaussian_bic(X, selected, lambda_bic=2.)[0],
                         "SHD_pattern": metric["SHD_pattern"],
                         "SHD_cpdag": metric["SHD_cpdag"], "F1_skel": metric["F1_skel"],
                         "F1_pattern": metric["F1_pattern"], "final_edges": int(selected.sum()),
                         "final_violations": 0, "pruning_runtime": prune_runtime,
                         "order_incompatible_true_edges": incompatible,
                         "missing_true_candidate_edges": int((truth & ~support).sum()),
                         "true_edges_rejected": int((truth & support & ~final).sum()),
                         "false_edges_retained": int((~truth & final).sum())})
    return pd.DataFrame(rows)


def select_variants(frame):
    out = []
    keys = ["panel", "seed", "method", "lambda_name", "postprocessing"]
    for key, group in frame.groupby(keys):
        for budget in (3, 5):
            available = group[(group.restart < budget) & group.eligible]
            if available.empty:
                continue
            r1_index = available.continuous_objective.idxmin()
            r2_index = available.sort_values(
                ["exact_bic", "final_edges", "candidate_threshold", "restart"],
                ascending=[True, True, False, True]).index[0]
            for policy, index in (("R1_continuous", r1_index), ("R2_bic", r2_index)):
                row = frame.loc[index].to_dict()
                row.update(restarts=budget, selection=policy)
                out.append(row)
    return pd.DataFrame(out)


def historical_metadata():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = json.loads((HIST_END / "input_hashes.json").read_text())
    config = {
        "source_experiment": "flop_notreks_dense_d50_budget",
        "source_report": str(HIST_END / "report.md"), "seeds": list(PANELS["A"]),
        "loss_type": "l2", "lambda1": .03, "trek_function": "inv",
        "trek_weight": 10., "restarts": 5,
        "initializations": "zero; random seeds dataset_seed*1009+restart, scale=.05",
        "mu_schedule": [1., .1, .01, .001, .0001], "s": [1., .9, .8, .7, .6],
        "warm_iter": 30000, "max_iter": 60000, "lr": .0003,
        "checkpoint": 1000, "beta_1": .99, "beta_2": .999,
        "continuous_selection": "lowest final continuous objective, then restart index",
        "historical_threshold_selection": "exact-BIC over [.01,.02,.05,.10,.15,.20,.25,.30]",
        "historically_selected_thresholds": {str(seed): .01 for seed in PANELS["A"]},
        "end_flop": "restricted grow-shrink, lambda_bic=2, deletion only",
        "input_hashes": hashes,
    }
    (OUT / "historical_configuration.json").write_text(json.dumps(config, indent=2) + "\n")


def reproduction(selected):
    historical = pd.read_csv(HIST_END / "per_dataset_selected.csv")
    historical = historical[(historical.method == "l2_dagma_notreks_inv_w10")
                            & (historical.selection == "end_flop_bic_selected")]
    current = selected[(selected.panel == "A") & (selected.method == "dagma_notreks")
                       & (selected.lambda_name == "fixed")
                       & (selected.restarts == 5) & (selected.postprocessing == "P1_fixed_001")
                       & (selected.selection == "R1_continuous")]
    merged = historical.merge(current, on="seed", suffixes=("_historical", "_current"))
    rows = []
    for _, r in merged.iterrows():
        oldW = np.load(HIST / f"seed_{int(r.seed)}/dagma_notreks_multistart_weighted.npy")
        newW = np.load(OUT / f"continuous/A/{int(r.seed)}/dagma_notreks/fixed/restart_{int(r.restart)}.npy")
        old_graph = pd.read_csv(HIST_END / f"seed_{int(r.seed)}/l2_dagma_notreks_inv_w10/end_flop_0.01.csv").to_numpy(int)
        new_graph = pd.read_csv(OUT / f"continuous/A/{int(r.seed)}/dagma_notreks/fixed/r{int(r.restart)}_P1_fixed_001.csv").to_numpy(int)
        old_restart = int(pd.read_csv(HIST / f"seed_{int(r.seed)}/dagma_notreks_multistart_restarts.csv").query("selected").restart_index.iloc[0])
        rows.append({"seed": int(r.seed), "historical_SHD": r.SHD_pattern_historical,
                     "current_SHD": r.SHD_pattern_current,
                     "SHD_difference": r.SHD_pattern_current-r.SHD_pattern_historical,
                     "historical_F1": r.F1_pattern_historical,
                     "current_F1": r.F1_pattern_current,
                     "max_weight_difference": float(np.max(np.abs(oldW-newW))),
                     "relative_frobenius_difference": float(np.linalg.norm(oldW-newW)/np.linalg.norm(oldW)),
                     "support_difference_001": int(np.sum(threshold_graph(oldW,.01)!=threshold_graph(newW,.01))),
                     "historical_restart": old_restart, "current_restart": int(r.restart),
                     "postprocessed_edge_difference": int(np.sum(old_graph != new_graph)),
                     "graph_match": bool(np.array_equal(old_graph, new_graph))})
    return pd.DataFrame(rows)


def summaries(selected):
    group = ["panel", "method", "lambda_name", "restarts", "postprocessing", "selection"]
    summary = selected.groupby(group).agg(
        mean_SHD=("SHD_pattern", "mean"), sd_SHD=("SHD_pattern", "std"),
        median_SHD=("SHD_pattern", "median"), catastrophic=("SHD_pattern", lambda x: int((x>=20).sum())),
        mean_F1=("F1_pattern", "mean"), mean_BIC=("exact_bic", "mean"),
        mean_candidate_recall=("candidate_true_edge_recall", "mean"),
        mean_order_error=("order_incompatible_true_edges", "mean"),
        mean_runtime=("runtime", "mean"), datasets=("seed", "count")).reset_index()
    return summary


def flop_rows():
    rows=[]
    for panel,seeds in PANELS.items():
        for seed in seeds:
            X,true,pairs,names,graph_path=load_inputs(panel,seed)
            if panel == "A":
                path=HIST/f"seed_{seed}/flop_official_r64_adjmat.csv"
                runtime=json.loads((HIST/f"seed_{seed}/flop_official_r64_row.json").read_text())["runtime"]
            else:
                path=RECENT/f"seed_{seed}/n_2000/flop_adjmat.csv"
                d=pd.read_csv(RECENT/f"seed_{seed}/n_2000/per_run.csv")
                runtime=float(d[d.method=='flop'].selected_runtime.iloc[0])
            metric=_metrics(graph_path,path,path.with_name(path.stem+'_audit_metrics.csv'))
            rows.append({"panel":panel,"seed":seed,"method":"flop","SHD_pattern":metric['SHD_pattern'],
                         "SHD_cpdag":metric['SHD_cpdag'],"F1_skel":metric['F1_skel'],
                         "F1_pattern":metric['F1_pattern'],"runtime":runtime})
    return pd.DataFrame(rows)


def main(argv=None):
    argparse.ArgumentParser(
        description=(
            "Reproduce the historical SHD-6 result and audit log-det versus "
            "inverse-trace DAGMA-NOTREKS. This is a long-running diagnostic.")
    ).parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    historical_metadata()
    hashes={}
    for panel,seeds in PANELS.items():
        for seed in seeds:
            for label,path in zip(("data","graph","sidecar"),input_paths(panel,seed)):
                hashes[f"{panel}_{seed}_{label}"]=digest(path)
    (OUT/"input_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    tasks=[(p,s,m,l) for p,seeds in PANELS.items() for s in seeds for m in METHODS for l in LAMBDAS]
    workers=min(8,os.cpu_count() or 1)
    all_rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(continuous_task,t):t for t in tasks}
        for future in as_completed(futures):
            rows=future.result(); all_rows.extend(rows)
            print("completed",futures[future],flush=True)
    continuous=pd.DataFrame(all_rows)
    continuous.to_csv(OUT/"per_restart.csv",index=False)
    if (OUT/"per_variant.csv").exists():
        variants=pd.read_csv(OUT/"per_variant.csv")
    else:
        variants=postprocess_all(all_rows)
        variants.to_csv(OUT/"per_variant.csv",index=False)
    selected=select_variants(variants)
    selected.to_csv(OUT/"paired_comparisons.csv",index=False)
    repro=reproduction(selected); repro.to_csv(OUT/"historical_reproduction.csv",index=False)
    summary=summaries(selected); summary.to_csv(OUT/"dataset_panel_comparison.csv",index=False)
    catastrophic=selected[selected.SHD_pattern>=20].copy()
    catastrophic.to_csv(OUT/"catastrophic_run_diagnostics.csv",index=False)
    selected[[c for c in selected.columns if c in (
        'panel','seed','method','lambda_name','restarts','postprocessing','selection','restart',
        'candidate_threshold','tau_DAG','tau_NOTREKS','tau_joint','candidate_edges',
        'candidate_true_edge_recall','candidate_precision','order_incompatible_true_edges',
        'missing_true_candidate_edges','true_edges_rejected','false_edges_retained','SHD_pattern')]].to_csv(
            OUT/'candidate_order_diagnostics.csv',index=False)
    flop=flop_rows(); flop.to_csv(OUT/'flop_reference.csv',index=False)
    corrected=selected[(selected.postprocessing=='P3_floor_001') & (selected.selection=='R2_bic')
                       & (selected.restarts==5) & (selected.lambda_name=='fixed')]
    corrected_summary=corrected.groupby(['panel','method']).agg(
        mean_SHD=('SHD_pattern','mean'),sd_SHD=('SHD_pattern','std'),
        median_SHD=('SHD_pattern','median'),mean_F1_pattern=('F1_pattern','mean'),
        mean_BIC=('exact_bic','mean'),mean_runtime=('runtime','mean'),
        catastrophic=('SHD_pattern',lambda x:int((x>=20).sum())),
        mean_violations=('final_violations','mean'),datasets=('seed','count')).reset_index()
    corrected_summary.to_csv(OUT/'corrected_method_summary.csv',index=False)
    # Controlled attribution rows, evaluated per panel/method.
    attrs=[]
    def comparison(label,a,b):
        keys=['panel','seed','method']; x=a.merge(b,on=keys,suffixes=('_a','_b'))
        for (panel,method),g in x.groupby(['panel','method']):
            delta=g.SHD_pattern_b-g.SHD_pattern_a
            attrs.append({'factor_changed':label,'panel':panel,'method':method,
                          'mean_SHD_change':delta.mean(),'median_SHD_change':delta.median(),
                          'wins_b':int((delta<0).sum()),'ties':int((delta==0).sum()),'losses_b':int((delta>0).sum()),
                          'catastrophic_change':int((g.SHD_pattern_b>=20).sum()-(g.SHD_pattern_a>=20).sum()),
                          'candidate_recall_change':(g.candidate_true_edge_recall_b-g.candidate_true_edge_recall_a).mean(),
                          'order_error_change':(g.order_incompatible_true_edges_b-g.order_incompatible_true_edges_a).mean(),
                          'runtime_change':(g.runtime_b-g.runtime_a).mean()})
    base=selected[(selected.lambda_name=='fixed')&(selected.restarts==5)&(selected.postprocessing=='P1_fixed_001')&(selected.selection=='R1_continuous')]
    comparison('lambda fixed->scaled',base,selected[(selected.lambda_name=='scaled')&(selected.restarts==5)&(selected.postprocessing=='P1_fixed_001')&(selected.selection=='R1_continuous')])
    comparison('restarts 5->3',base,selected[(selected.lambda_name=='fixed')&(selected.restarts==3)&(selected.postprocessing=='P1_fixed_001')&(selected.selection=='R1_continuous')])
    comparison('threshold P1->P2',base,selected[(selected.lambda_name=='fixed')&(selected.restarts==5)&(selected.postprocessing=='P2_minimal')&(selected.selection=='R1_continuous')])
    comparison('threshold P2->P3',selected[(selected.lambda_name=='fixed')&(selected.restarts==5)&(selected.postprocessing=='P2_minimal')&(selected.selection=='R1_continuous')],selected[(selected.lambda_name=='fixed')&(selected.restarts==5)&(selected.postprocessing=='P3_floor_001')&(selected.selection=='R1_continuous')])
    comparison('selection R1->R2',base,selected[(selected.lambda_name=='fixed')&(selected.restarts==5)&(selected.postprocessing=='P1_fixed_001')&(selected.selection=='R2_bic')])
    pd.DataFrame(attrs).to_csv(OUT/'factor_attribution.csv',index=False)
    (OUT/'runtime_breakdown.csv').write_text(continuous.groupby(['panel','method','lambda_name']).runtime.agg(['mean','sum','count']).reset_index().to_csv(index=False))
    report=("# Inverse regression audit\n\nHistorical reproduction pass: **"
            +str(bool(repro.graph_match.all()))+"**.\n\n## Historical reproduction\n\n```csv\n"
            +repro.to_csv(index=False)+"```\n\n## Factor attribution\n\n```csv\n"
            +pd.DataFrame(attrs).to_csv(index=False)+"```\n\n## Panel summaries\n\n```csv\n"
            +summary.to_csv(index=False)+"```\n")
    (OUT/'report.md').write_text(report)
    (OUT/'command_log.txt').write_text('OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/inv_regression_audit.py\n')


if __name__ == '__main__':
    main()
