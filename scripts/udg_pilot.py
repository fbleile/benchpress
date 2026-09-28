#!/usr/bin/env python3
"""Isolated UDG / marginal-independence scalability pilot.

This script never imports or calls FLOP, DAGMA, or a NOTREKS objective.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udg_pilot.core import (holm_reject, no_trek_pairs, pairwise_evidence,
                            source_mask_udg, source_mask_witness, true_udg)
from udg_pilot.data import data, graph

RUST_MANIFEST = ROOT / "udg_pilot" / "rust" / "Cargo.toml"


def seed_for(*parts: object) -> int:
    raw = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def flat_pairs(d: int):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def metrics(estimated: np.ndarray, truth: np.ndarray):
    est = estimated.astype(bool); tr = truth.astype(bool)
    np.fill_diagonal(est, False); np.fill_diagonal(tr, False)
    iu = np.triu_indices_from(est, 1)
    e = est[iu]; t = tr[iu]
    tp = int(np.sum(e & t)); fp = int(np.sum(e & ~t)); fn = int(np.sum(~e & t))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    no_e, no_t = ~e, ~t
    nt_tp = int(np.sum(no_e & no_t)); nt_fp = int(np.sum(no_e & ~no_t)); nt_fn = int(np.sum(~no_e & no_t))
    nt_p = nt_tp / (nt_tp + nt_fp) if nt_tp + nt_fp else 1.0
    nt_r = nt_tp / (nt_tp + nt_fn) if nt_tp + nt_fn else 1.0
    nt_f1 = 2 * nt_p * nt_r / (nt_p + nt_r) if nt_p + nt_r else 0.0
    return dict(udg_shd=fp + fn, udg_precision=precision, udg_recall=recall,
                udg_f1=f1, no_trek_precision=nt_p, no_trek_recall=nt_r,
                no_trek_f1=nt_f1, no_trek_fdr=fp / max(1, int(np.sum(no_e))))


def fisher_method(x, alpha: float, holm: bool):
    _, z, p = pairwise_evidence(x); pairs = flat_pairs(x.shape[1])
    values = np.array([p[i, j] for i, j in pairs])
    reject = holm_reject(values, alpha) if holm else values < alpha
    adjacency = np.zeros((x.shape[1], x.shape[1]), dtype=np.uint8)
    for (i, j), is_dep in zip(pairs, reject):
        adjacency[i, j] = adjacency[j, i] = int(is_dep)
    return adjacency, dict(alpha=alpha, test="two-sided_fisher_z",
                           p_values=values.tolist(), z_values=[float(z[i, j]) for i, j in pairs],
                           correction="holm" if holm else "none")


def ensure_rust_binary() -> Path:
    binary = ROOT / "udg_pilot" / "rust" / "target" / "release" / "udg_source_mask"
    if not binary.exists():
        subprocess.run(["cargo", "build", "--release", "--manifest-path", str(RUST_MANIFEST)],
                       cwd=ROOT, check=True)
    return binary


def source_mask_method(b, d, seed, restarts, budget, anneal, n):
    binary = ensure_rust_binary()
    temp = ROOT / "udg_pilot" / ".tmp"
    temp.mkdir(parents=True, exist_ok=True)
    evidence_path = temp / f"evidence_{seed}.csv"
    evidence_path.write_text(",".join(str(float(x)) for x in b.ravel()))
    prior = 0.5 * math.log(max(2, n))
    started = time.perf_counter()
    proc = subprocess.run([str(binary), str(d), str(evidence_path), str(seed), str(restarts),
                           str(budget), "anneal" if anneal else "greedy", str(prior)],
                          cwd=ROOT, check=True, capture_output=True, text=True)
    diag = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1); diag[key] = value
    masks = np.array([int(x) for x in diag["masks"].split(",")], dtype=np.uint64)
    return source_mask_udg(masks, d), masks, diag, time.perf_counter() - started


def grues_method(x, seed, iterations):
    try:
        from gues import grues
    except Exception as exc:
        return None, {"status": "unavailable", "error": repr(exc),
                      "package": "gues", "package_version": "not-installed"}
    version = importlib.metadata.version("gues")
    model = grues.InputData(x, np.random.default_rng(seed))
    model.mcmc(init=("gauss", 0.05), max_moves=iterations, prior=None)
    chain = np.asarray(model.uec_markov_chain)
    scores = np.asarray(model.nuc_markov_chain)
    index = int(np.argmax(scores))
    return (chain[index].astype(np.uint8),
            {"status": "ok", "package": "gues", "package_version": version,
             "mcmc_iterations": iterations, "chain_length": int(len(chain)),
             "map_index": index})


def write_figure(rows, out: Path):
    import matplotlib.pyplot as plt
    import pandas as pd
    frame = pd.DataFrame(rows); frame = frame[frame.status == "ok"]
    if frame.empty: return
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), squeeze=False)
    for row, d in enumerate((10, 20)):
        sub = frame[frame.d == d]
        for col, metric in enumerate(("no_trek_precision", "no_trek_recall", "no_trek_f1", "runtime")):
            ax = axes[row, col]
            for method, group in sub.groupby("method"):
                ax.scatter([method] * len(group), group[metric], label=method, alpha=.65)
            ax.set_title(f"d={d}: {metric.replace('_', ' ')}")
            if metric == "runtime": ax.set_yscale("log")
            ax.tick_params(axis="x", rotation=55)
            ax.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, 1)); fig.savefig(out / "udg_metrics_runtime.png", dpi=180); plt.close(fig)


def run(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    methods = args.methods
    rows = []
    for d in args.dimensions:
        for degree in args.degrees:
            for graph_seed in args.graph_seeds:
                dag = graph(d, degree, graph_seed); truth = true_udg(dag)
                true_no = no_trek_pairs(truth)
                for data_seed in args.data_seeds:
                    x = data(dag, args.n, data_seed)
                    corr, _, _ = pairwise_evidence(x)
                    b = np.zeros((d, d), dtype=float)
                    for i, j in flat_pairs(d):
                        r = float(np.clip(corr[i, j], -1 + 1e-12, 1 - 1e-12))
                        b[i, j] = b[j, i] = args.n * (-math.log(1-r*r)) - math.log(args.n)
                    for method in methods:
                        started = time.perf_counter(); status = "ok"; diag = {}
                        try:
                            if method == "pairwise_fisherz_raw": est, diag = fisher_method(x, .05, False)
                            elif method == "pairwise_fisherz_holm": est, diag = fisher_method(x, .05, True)
                            elif method == "grues_exact":
                                est, diag = grues_method(x, seed_for(method,d,degree,graph_seed,data_seed), args.grues_iterations)
                                if est is None:
                                    status = "unavailable"
                                    est = np.zeros_like(truth)
                            else:
                                est, masks, diag, elapsed = source_mask_method(
                                    b, d, seed_for(method,d,degree,graph_seed,data_seed), args.restarts,
                                    args.moves, method == "source_mask_anneal", args.n)
                                witness = source_mask_witness(masks)
                        except Exception as exc:
                            status = "failed"; diag = {"error": repr(exc)}; est = np.zeros_like(truth)
                        elapsed = time.perf_counter() - started
                        row = {"d":d,"er_degree":degree,"n":args.n,"graph_seed":graph_seed,
                               "data_seed":data_seed,"method":method,"status":status,
                               "graph_edges":int(dag.sum()),"true_udg_edges":int(truth.sum()),
                               "true_no_trek_pairs":len(true_no),"runtime":elapsed,
                               "config_hash":seed_for("config",args.moves,args.restarts,args.grues_iterations)}
                        row.update(metrics(est, truth) if status == "ok" else {
                            key: np.nan for key in ("udg_shd", "udg_precision", "udg_recall", "udg_f1",
                                                     "no_trek_precision", "no_trek_recall", "no_trek_f1",
                                                     "no_trek_fdr")})
                        row.update({f"diag_{k}": v for k,v in diag.items()})
                        if method.startswith("source_mask") and status == "ok":
                            row["udg_validity_witness"] = int(np.array_equal(source_mask_udg(masks,d), true_udg(witness)))
                        else: row["udg_validity_witness"] = "NA"
                        rows.append(row)
                        print(f"d={d} ER{degree} graph={graph_seed} data={data_seed} {method} {status} "
                              f"no-trek-F1={row['no_trek_f1']:.3f} runtime={elapsed:.2f}s", flush=True)
    import pandas as pd
    frame = pd.DataFrame(rows); frame.to_csv(out / "results.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "dimensions": args.dimensions, "degrees": args.degrees, "n": args.n,
        "graph_seeds": args.graph_seeds, "data_seeds": args.data_seeds,
        "methods": methods, "restarts": args.restarts, "moves": args.moves,
        "grues_iterations": args.grues_iterations,
        "generator": "udg_pilot.data: standardized linear-Gaussian ER; log innovation variance iid U[-log(2),log(2)]",
        "fisher_alpha": 0.05, "source_prior": "0.5*log(n) per selected singleton source",
        "grues_package": "gues (optional; detected at runtime)",
    }, indent=2) + "\n")
    ok = frame[frame.status == "ok"]
    if not ok.empty:
        ok.groupby("method").agg({"no_trek_precision":["mean","std"],"no_trek_recall":["mean","std"],
                                   "no_trek_f1":["mean","std"],"runtime":["mean","std"]}).to_csv(out / "aggregate.csv")
    if not ok.empty:
        paired = ok.pivot_table(index=["d", "er_degree", "n", "graph_seed", "data_seed"],
                                columns="method", values=["no_trek_precision", "no_trek_recall", "no_trek_f1", "runtime"])
        for method in ("source_mask_greedy", "source_mask_anneal", "pairwise_fisherz_holm"):
            for metric in ("no_trek_precision", "no_trek_recall", "no_trek_f1", "runtime"):
                left = (metric, method); right = (metric, "pairwise_fisherz_raw")
                if left in paired and right in paired:
                    paired[(metric + "_difference_vs_raw", method)] = paired[left] - paired[right]
        paired.to_csv(out / "paired_differences.csv")
    write_figure(rows, out)
    (out / "REPORT.md").write_text(report(frame, args))


def report(frame, args):
    lines = ["# Isolated UDG scalability pilot", "", "This is not a NOTREKS or causal-discovery experiment.", "",
             f"Configuration: dimensions={args.dimensions}, ER degrees={args.degrees}, n={args.n}, "
             f"graph seeds={args.graph_seeds}, data seeds={args.data_seeds}.", "",
             "Data use standardized linear-Gaussian ER DAGs; the no-trek truth is used only for evaluation.", "",
             "GrUES is invoked only when the installed `gues` package is available; otherwise it is reported unavailable.",
             "The existing Benchpress integration uses container `docker://bpimages/grues:0.3.0`; the local pilot environment had no `gues` distribution, so no GrUES run was claimed.", ""]
    if not frame.empty:
        lines += ["## Aggregate", "", "```", frame.groupby(["method","status"])[["no_trek_precision","no_trek_recall","no_trek_f1","runtime"]].mean().to_string(), "```", ""]
    lines += ["## Limitations", "", "Pairwise non-rejection is evidence, not a certificate of independence. "
              "The source-mask methods are structured UDG witnesses, not GrUES posterior samplers."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/udg_pilot"))
    p.add_argument("--dimensions", nargs="+", type=int, default=[10,20])
    p.add_argument("--degrees", nargs="+", type=int, default=[2,4])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--graph-seeds", nargs="+", type=int, default=[7101,7102,7103])
    p.add_argument("--data-seeds", nargs="+", type=int, default=[8101,8102])
    p.add_argument("--methods", nargs="+", default=["pairwise_fisherz_raw","pairwise_fisherz_holm","grues_exact","source_mask_greedy","source_mask_anneal"])
    p.add_argument("--restarts", type=int, default=8); p.add_argument("--moves", type=int, default=3000)
    p.add_argument("--grues-iterations", type=int, default=3000)
    run(p.parse_args())


if __name__ == "__main__": main()
