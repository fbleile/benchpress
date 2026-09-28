#!/usr/bin/env python3
"""Audit direct covariance-graph scoring before changing UDG search.

This script is isolated from NOTREKS/DAGMA.  Fisher and true UDGs are used
only as diagnostic candidates/evaluation labels; no estimated pair set is
passed to a causal optimizer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from udg_pilot.core import pairwise_evidence, true_udg
from udg_pilot.covariance import (bits_to_matrix, edge_count, fit_covariance,
                                  matrix_to_bits, sample_covariance, smig_valid)
from udg_pilot.data import data, graph

MANIFEST = ROOT / "udg_pilot" / "rust" / "Cargo.toml"
BINARY = ROOT / "udg_pilot" / "rust" / "target" / "release" / "udg_covbic"


def build():
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(MANIFEST)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def kv(text):
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1); out[k] = v
    return out


def rust_score(s, n, bits, path, penalty=None):
    path.write_text(",".join(str(float(v)) for v in s.ravel()))
    cmd = [str(BINARY), "score", str(s.shape[0]), str(n), str(path), str(bits), "500"]
    if penalty is not None:
        cmd.append(str(float(penalty)))
    return kv(subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout)


def rust_search(s, n, seed, valid, budget, penalty):
    path = ROOT / "udg_pilot" / ".tmp" / f"audit_search_{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(str(float(v)) for v in s.ravel()))
    cmd = [str(BINARY), "search", str(s.shape[0]), str(n), str(path), str(seed),
           "4", str(budget), "valid" if valid else "unrestricted", "300", "0",
           str(float(penalty))]
    return kv(subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout)


def metric(est, truth):
    e = est[np.triu_indices_from(est, 1)].astype(bool)
    t = truth[np.triu_indices_from(truth, 1)].astype(bool)
    tp = np.sum(e & t); fp = np.sum(e & ~t); fn = np.sum(~e & t)
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    nt_e, nt_t = ~e, ~t
    np_ = np.sum(nt_e & nt_t) / max(1, np.sum(nt_e))
    nr = np.sum(nt_e & nt_t) / max(1, np.sum(nt_t))
    return {"udg_precision": p, "udg_recall": r,
            "udg_f1": 2*p*r/max(1e-15, p+r),
            "no_trek_precision": np_, "no_trek_recall": nr,
            "no_trek_f1": 2*np_*nr/max(1e-15, np_+nr)}


def one_dataset(d, degree, n, graph_seed, data_seed, out, budget):
    dag = graph(d, degree, graph_seed); truth = true_udg(dag)
    x = data(dag, n, data_seed); s = sample_covariance(x)
    _, _, pvals = pairwise_evidence(x)
    raw = np.zeros((d, d), dtype=np.uint8)
    for i in range(d):
        for j in range(i+1, d):
            raw[i, j] = raw[j, i] = int(pvals[i, j] < .05)
    candidates = [("empty", 0), ("fisher_raw", matrix_to_bits(raw)),
                  ("true_udg_evaluation_only", matrix_to_bits(truth))]
    complete = (1 << (d*(d-1)//2)) - 1
    candidates.append(("complete", complete))
    correlations = [(abs(float(s[i, j])), i, j) for i in range(d) for j in range(i+1, d)]
    correlations.sort(reverse=True)
    for label, item in zip(("one_edge_strong", "one_edge_medium", "one_edge_weak"),
                           (correlations[0], correlations[len(correlations)//2], correlations[-1])):
        _, i, j = item; a = np.zeros((d, d), dtype=np.uint8); a[i, j] = a[j, i] = 1
        candidates.append((label, matrix_to_bits(a)))
    try:
        import flopsearch
        flop = np.asarray(flopsearch.flop(x, 2.0, restarts=4))
        candidates.append(("flop_derived", matrix_to_bits(true_udg(flop != 0))))
    except Exception as exc:
        candidates.append(("flop_unavailable", 0))
    rows = []
    for name, bits in candidates:
        py = fit_covariance(bits, s, n, max_iter=800)
        rust = rust_score(s, n, bits, out / f"score_{d}_{graph_seed}_{data_seed}.csv")
        fitted = py.sigma
        free = bits_to_matrix(bits, d).astype(bool) | np.eye(d, dtype=bool)
        nonedge = float(np.max(np.abs(fitted[~free]))) if np.any(~free) else 0.0
        try:
            chol = np.linalg.cholesky(fitted) if not py.failed else np.full((d, d), np.nan)
        except np.linalg.LinAlgError:
            # Preserve the diagnostic row: a non-PD final reference iterate is
            # itself evidence of a fitting failure, not a reason to abort the
            # complete audit.
            chol = np.full((d, d), np.nan)
            py.failed = True
        rows.append({"d": d, "degree": degree, "n": n, "graph_seed": graph_seed,
                     "data_seed": data_seed, "candidate": name, "bits": bits,
                     "edge_count": edge_count(bits), "smig_valid": smig_valid(bits, d),
                     "sample_covariance": json.dumps(s.tolist()),
                     "python_loglik2": py.loglik2, "python_bic": py.bic,
                     "rust_loglik2": float(rust.get("loglik2", "nan")),
                     "rust_bic": float(rust.get("bic", "nan")),
                     "icf_converged": py.converged, "icf_iterations": py.iterations,
                     "jitter": py.jitter, "failed": py.failed,
                     "rust_converged": rust.get("converged"),
                     "rust_failed_fits": rust.get("failed_fits"),
                     "max_abs_nonedge_sigma": nonedge,
                     "min_cholesky_diagonal": float(np.min(np.diag(chol))) if not py.failed else np.nan,
                     "cache_status": "miss", "rust_python_bic_difference":
                         float(rust.get("bic", "nan")) - py.bic})
    # A valid-SMIG search seeded by raw Fisher is recorded separately; it is
    # not used to tune the primary benchmark.
    search = rust_search(s, n, graph_seed + data_seed, True, budget, math.log(n))
    rows.append({"d": d, "degree": degree, "n": n, "graph_seed": graph_seed,
                 "data_seed": data_seed, "candidate": "fisher_valid_smig_search",
                 "bits": int(search["bits"]), "edge_count": int(search["edge_count"]),
                 "smig_valid": True, "rust_bic": float(search["bic"]),
                 "rust_loglik2": np.nan, "python_loglik2": np.nan, "python_bic": np.nan,
                 "icf_converged": search.get("converged"), "icf_iterations": search.get("icf_iterations"),
                 "jitter": search.get("jitter"), "failed": search.get("failed_fits"),
                 "invalid_rejections": search.get("invalid_rejections"),
                 "cache_hits": search.get("cache_hits"), "cache_status": "search",
                 "truth_metrics": json.dumps(metric(bits_to_matrix(int(search["bits"]), d), truth))})
    for penalty in (0.0, 0.25, 0.5, 1.0, 2.0, math.log(n)):
        for valid_graph in (False, True):
            path_search = rust_search(s, n, graph_seed + data_seed + int(1000*penalty) + int(valid_graph),
                                      valid_graph, budget, penalty)
            bits = int(path_search["bits"]); scores = metric(bits_to_matrix(bits, d), truth)
            rows.append({"d": d, "degree": degree, "n": n, "graph_seed": graph_seed,
                         "data_seed": data_seed, "candidate":
                             f"penalty_path_{'valid' if valid_graph else 'unrestricted'}_{penalty:g}",
                         "bits": bits, "edge_count": int(path_search["edge_count"]),
                         "smig_valid": smig_valid(bits, d), "edge_penalty": penalty,
                         "rust_bic": float(path_search["bic"]),
                         "evaluations": path_search.get("evaluations"),
                         "invalid_rejections": path_search.get("invalid_rejections"),
                         "cache_hits": path_search.get("cache_hits"),
                         "icf_converged": path_search.get("converged"),
                         "icf_iterations": path_search.get("icf_iterations"),
                         "failed": path_search.get("failed_fits"),
                         "rust_failed_fits": path_search.get("failed_fits"), **scores})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/udg_pilot/score_audit"))
    p.add_argument("--dimensions", nargs="+", type=int, default=[10, 20])
    p.add_argument("--degrees", nargs="+", type=int, default=[2, 4])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--graph-seeds", nargs="+", type=int, default=[7101, 7102])
    p.add_argument("--data-seeds", nargs="+", type=int, default=[8101])
    p.add_argument("--search-budget", type=int, default=300)
    args = p.parse_args(); build(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in args.dimensions:
        for degree in args.degrees:
            for gs in args.graph_seeds:
                for ds in args.data_seeds:
                    rows.extend(one_dataset(d, degree, args.n, gs, ds, args.output, args.search_budget))
    with (args.output / "score_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); writer.writeheader(); writer.writerows(rows)
    # Keep a typed machine-readable copy as well as the flat CSV.  The CSV is
    # convenient for pandas; the JSON preserves booleans and null diagnostics.
    (args.output / "score_audit.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    print(f"wrote {len(rows)} audit rows to {args.output / 'score_audit.csv'}")


if __name__ == "__main__": main()
