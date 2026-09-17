"""Parallel-tempered basin hopping and annealed SMC wrappers."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.shared import deterministic_initial_adjacency
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import postprocess_weighted_adjacency


def log_acceptance(delta, temperature):
    return min(0., -float(delta) / float(temperature))


def swap_log_acceptance(E1, E2, T1, T2):
    return min(0., (1. / T1 - 1. / T2) * (E1 - E2))


def _perturb(W, rng, scale):
    d = len(W); out = np.asarray(W, dtype=float).copy(); kind = int(rng.integers(6))
    if kind == 0:
        out += rng.normal(scale=scale, size=out.shape)
    elif kind == 1:
        for _ in range(max(1, d // 3)):
            i, j = rng.integers(d, size=2)
            if i != j: out[i, j] += rng.normal(scale=scale)
    elif kind == 2:
        node = int(rng.integers(d)); out[node] += rng.normal(scale=scale, size=d); out[:, node] += rng.normal(scale=scale, size=d)
    elif kind == 3:
        node = int(rng.integers(d)); out[node, :] += rng.normal(scale=scale, size=d); out[:, node] += rng.normal(scale=scale, size=d)
    elif kind == 4:
        i, j = rng.choice(d, 2, replace=False); P = np.eye(d); P[[i, j]] = P[[j, i]]; out = P @ out @ P.T
    else:
        i = int(rng.integers(d)); out[i, :] *= rng.uniform(0., .3); out[:, i] *= rng.uniform(0., .3); out += rng.normal(scale=scale*.25, size=out.shape)
    np.fill_diagonal(out, 0.)
    return out


def _archive_result(adapter, archive, X, pairs, method, started, diagnostics):
    feasible = []
    for W, c in archive:
        try:
            graph, coeff, post = postprocess_weighted_adjacency(X, W, pairs, screening_floor=.01, lambda_bic=2.)
            feasible.append((post["postprocessed_bic"], graph, W, c, post))
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    if feasible:
        best = min(feasible, key=lambda x: x[0]); graph, post = best[1], best[4]
        best_feasible = float(best[0]); feasibility = True
    else:
        graph = np.zeros((adapter.d, adapter.d), dtype=np.uint8); best_feasible = np.nan; post = {}; feasibility = False
    best_raw_item = min(archive, key=lambda x: x[1]["total"], default=(None, {"total": np.inf}))
    best_raw = best_raw_item[1]["total"]
    diagnostics.update({"runtime": time.perf_counter()-started, "best_raw_objective": float(best_raw),
                        "best_raw_data": float(best_raw_item[1].get("data", np.nan)),
                        "best_raw_sparsity": float(best_raw_item[1].get("sparsity", np.nan)),
                        "best_raw_dag": float(best_raw_item[1].get("dag", np.nan)),
                        "best_raw_notreks": float(best_raw_item[1].get("notreks", np.nan)),
                        "best_feasible_bic": best_feasible, "feasible": feasibility,
                        "objective_evaluations": len(archive),
                        "final_edges": int(graph.sum()),
                        "selected_threshold": post.get("candidate_threshold", np.nan),
                        "distinct_supports": len({_support(W) for W, _ in archive}),
                        "archive_size": len(archive), "postselection": post})
    return graph, diagnostics


def _support(W): return np.asarray(np.abs(W) >= .3, dtype=np.uint8).tobytes()


def run_pt_basin(X, pairs, *, seed=0, replicas=8, stages=5, steps_per_stage=4,
                 temperatures=None, lambda1=.03, trek_weight=1.,
                 warm_iter=1000, max_iter=2000, lr=.0003, regime="dag_first"):
    rng = np.random.default_rng(seed); started = time.perf_counter(); d = X.shape[1]
    adapter = __import__("workflow.rules.structure_learning_algorithms.dagma_global_search.objective", fromlist=["ObjectiveAdapter"]).ObjectiveAdapter(X, pairs, lambda1=lambda1, trek_weight=trek_weight)
    temps = np.asarray(temperatures if temperatures is not None else np.geomspace(.02, 2., replicas), dtype=float)
    states = [deterministic_initial_adjacency(d, seed+i, .05) for i in range(replicas)]
    archive=[]; swap_accept=0; swap_total=0; accept=0; proposals=0
    for stage_index in range(stages):
        stage=adapter.stage(stage_index/max(1, stages-1), regime); energies=[adapter.components(w,stage)["total"] for w in states]
        for _ in range(steps_per_stage):
            for r in range(replicas):
                scale=max(.01, float(np.median(np.abs(states[r][states[r] != 0])) if np.any(states[r]) else .05))
                proposal=_perturb(states[r],rng,scale); quenched,_=adapter.local_quench(proposal,stage,warm_iter=warm_iter,max_iter=max_iter,lr=lr)
                old=energies[r]; newc=adapter.components(quenched,stage); new=newc["total"]; proposals+=1
                if np.isfinite(new) and np.log(rng.random()) <= log_acceptance(new-old,temps[r]): states[r]=quenched; energies[r]=new; accept+=1
                archive.extend([(states[r].copy(), adapter.components(states[r],stage))])
            for r in range(replicas-1):
                swap_total+=1; la=swap_log_acceptance(energies[r],energies[r+1],temps[r],temps[r+1])
                if np.log(rng.random()) <= la: states[r],states[r+1]=states[r+1],states[r]; energies[r],energies[r+1]=energies[r+1],energies[r]; swap_accept+=1
    return _archive_result(adapter, archive, adapter.X, pairs, "dagma_notreks_pt_basin", started, {"method":"dagma_notreks_pt_basin","replicas":replicas,"swap_rate":swap_accept/max(1,swap_total),"proposal_acceptance":accept/max(1,proposals),"temperatures":temps.tolist(),"regime":regime})


def _systematic_resample(weights, rng):
    n=len(weights); positions=(rng.random()+np.arange(n))/n; c=np.cumsum(weights); return np.searchsorted(c,positions)


def _mala_log_q(source, target, gradient, step, beta=1.0):
    mean_shift = -step * beta * gradient
    residual = target - source - mean_shift
    return -float(np.sum(residual * residual)) / (4. * step)


def run_asmc(X, pairs, *, seed=0, particles=64, stages=8, lambda1=.03,
             trek_weight=1., warm_iter=1000, max_iter=2000, lr=.0003,
             regime="dag_first", mala_step=.001):
    rng=np.random.default_rng(seed); started=time.perf_counter(); d=X.shape[1]
    adapter=__import__("workflow.rules.structure_learning_algorithms.dagma_global_search.objective", fromlist=["ObjectiveAdapter"]).ObjectiveAdapter(X,pairs,lambda1=lambda1,trek_weight=trek_weight)
    particles_=[deterministic_initial_adjacency(d,seed+i,.05) for i in range(particles)]; archive=[]; logw=np.zeros(particles); resamples=0; mala_accept=0; mala_total=0; rw_accept=0; rw_total=0; de_accept=0; de_total=0
    previous=adapter.stage(0.,regime)
    for k in range(1,stages+1):
        stage=adapter.stage(k/stages,regime); oldE=np.array([adapter.components(w,previous)["total"] for w in particles_]); newE=np.array([adapter.components(w,stage)["total"] for w in particles_]);
        delta=np.zeros_like(newE); finite=np.isfinite(oldE)&np.isfinite(newE); delta[finite]=-(newE[finite]-oldE[finite]); delta[~finite]=-np.inf; logw += delta
        shift=np.max(logw); weights=np.exp(logw-shift) if np.isfinite(shift) else np.full(particles,1./particles); total=weights.sum(); weights=weights/total if total > 0 and np.isfinite(total) else np.full(particles,1./particles); ess=1/np.sum(weights*weights)
        if ess < .5*particles:
            idx=_systematic_resample(weights,rng); particles_=[particles_[i].copy() for i in idx]; logw.fill(0); resamples+=1
        for i,w in enumerate(particles_):
            c=adapter.components(w,stage); grad=c["gradient"]; proposal=w-mala_step*grad+rng.normal(scale=np.sqrt(2*mala_step),size=w.shape); np.fill_diagonal(proposal,0); cp=adapter.components(proposal,stage); mala_total+=1
            move = rng.random()
            if move < .2:
                proposal=w+rng.normal(scale=np.sqrt(2*mala_step),size=w.shape); np.fill_diagonal(proposal,0); cp=adapter.components(proposal,stage); rw_total+=1
                loga=-(cp["total"]-c["total"])
                if cp["valid_domain"] and np.log(rng.random()) <= min(0.,loga): particles_[i]=proposal; rw_accept+=1
            elif move < .3 and particles > 2:
                a,b=rng.choice([j for j in range(particles) if j != i],2,replace=False); proposal=w+.8*(particles_[a]-particles_[b])+rng.normal(scale=np.sqrt(2*mala_step)*.1,size=w.shape); np.fill_diagonal(proposal,0); cp=adapter.components(proposal,stage); de_total+=1
                loga=-(cp["total"]-c["total"])
                if cp["valid_domain"] and np.log(rng.random()) <= min(0.,loga): particles_[i]=proposal; de_accept+=1
            else:
                cp=adapter.components(proposal,stage); mala_total+=1
                if cp["valid_domain"]:
                    reverse=_mala_log_q(proposal,w,cp["gradient"],mala_step)
                    forward=_mala_log_q(w,proposal,grad,mala_step)
                    loga=-(cp["total"]-c["total"])+reverse-forward
                    if np.log(rng.random()) <= min(0.,loga): particles_[i]=proposal; mala_accept+=1
            archive.append((particles_[i].copy(),adapter.components(particles_[i],stage))); previous=stage
    # Final deterministic polishing of the best distinct archived states.
    ranked=sorted(archive,key=lambda x:x[1]["total"]); polished=[]
    final=adapter.stage(1.,regime)
    for w,_ in ranked[:min(8,len(ranked))]:
        q,_=adapter.local_quench(w,final,warm_iter=warm_iter,max_iter=max_iter,lr=lr); polished.append((q,adapter.components(q,final)))
    archive.extend(polished)
    return _archive_result(adapter, archive, adapter.X, pairs, "dagma_notreks_asmc", started, {"method":"dagma_notreks_asmc","particles":particles,"resampling_events":resamples,"mala_acceptance":mala_accept/max(1,mala_total),"rw_acceptance":rw_accept/max(1,rw_total),"de_acceptance":de_accept/max(1,de_total),"final_ess":float(ess),"regime":regime})
