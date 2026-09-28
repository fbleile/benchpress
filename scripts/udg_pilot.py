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
from udg_pilot.covariance import bits_to_matrix, matrix_to_bits, sample_covariance, smig_valid
from udg_pilot.data import data, graph
from udg_pilot.order_search import search_order_uec
from udg_pilot.nonparametric import (adjacency_from_pvalues, ensemble,
                                     pairwise_test, stability_selection)

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
    return dict(predicted_udg_edges=int(np.sum(e)), true_udg_edges=int(np.sum(t)),
                udg_shd=fp + fn, udg_precision=precision, udg_recall=recall,
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
    binary = ROOT / "udg_pilot" / "rust" / "target" / "release" / "udg_covbic"
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(RUST_MANIFEST)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
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


def covariance_method(x, d, n, seed, restarts, budget, valid_graph, max_iter,
                      initial_bits=None, edge_penalty=None):
    """Run the direct Rust covariance-BIC UDG search and parse diagnostics."""
    binary = ensure_rust_binary()
    temp = ROOT / "udg_pilot" / ".tmp"; temp.mkdir(parents=True, exist_ok=True)
    path = temp / f"covariance_{seed}.csv"
    path.write_text(",".join(str(float(v)) for v in sample_covariance(x).ravel()))
    started = time.perf_counter()
    command = [str(binary), "search", str(d), str(n), str(path), str(seed),
                           str(restarts), str(budget), "valid" if valid_graph else "unrestricted",
                           str(max_iter)]
    if initial_bits is not None:
        command.append(str(initial_bits))
    if edge_penalty is not None:
        while len(command) < 11:
            command.append("0")
        command.append(str(edge_penalty))
    proc = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    diag = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1); diag[key] = value
    bits = int(diag["bits"])
    est = bits_to_matrix(bits, d)
    diag["udg_valid"] = str(smig_valid(bits, d)).lower()
    diag["score_method"] = "direct_covariance_bic"
    return est, diag, time.perf_counter() - started


def fisher_pr_curve(x, truth):
    rows = []
    for alpha in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        est, diag = fisher_method(x, alpha, False)
        row = metrics(est, truth); row.update(alpha=alpha)
        rows.append(row)
    return rows


def nonparametric_method(x, method, alpha, permutations, stability_replicates,
                         stability_fraction, stability_threshold, seed):
    """Return a pairwise MIG estimate; no structural optimizer is called."""
    if method.startswith("pairwise_hsic"):
        rank = method.endswith("_rank")
        ev = pairwise_test(x, test="hsic", rank=rank, permutations=permutations, seed=seed)
        return adjacency_from_pvalues(ev, alpha), {
            "test": "hsic", "transform": "rank" if rank else "raw",
            "pairwise_tests": x.shape[1] * (x.shape[1] - 1) // 2,
            "pvalue_matrix": json.dumps(ev.pvalue.tolist()),
            "statistic_matrix": json.dumps(ev.statistic.tolist()),
        }
    if method.startswith("pairwise_dcov"):
        rank = method.endswith("_rank")
        ev = pairwise_test(x, test="dcov", rank=rank, permutations=permutations, seed=seed)
        return adjacency_from_pvalues(ev, alpha), {
            "test": "dcov", "transform": "rank" if rank else "raw",
            "pairwise_tests": x.shape[1] * (x.shape[1] - 1) // 2,
            "pvalue_matrix": json.dumps(ev.pvalue.tolist()),
            "statistic_matrix": json.dumps(ev.statistic.tolist()),
        }
    if method in ("hsic_stability", "dcov_stability"):
        test = "hsic" if method.startswith("hsic") else "dcov"
        path, frequencies = stability_selection(
            x, test=test, rank=False, replicates=stability_replicates,
            subsample_fraction=stability_fraction, alpha=alpha,
            permutations=permutations, seed=seed)
        selected = path[float(stability_threshold)]
        return selected, {
            "test": test, "transform": "raw", "stability_replicates": stability_replicates,
            "stability_fraction": stability_fraction, "stability_threshold": stability_threshold,
            "edge_frequency_matrix": json.dumps(frequencies.tolist()),
            "threshold_path": json.dumps({str(k): int(v.sum() // 2) for k, v in path.items()}),
            "pairwise_tests": stability_replicates * x.shape[1] * (x.shape[1] - 1) // 2,
        }
    if method in ("conservative_ensemble", "liberal_ensemble"):
        hsic = pairwise_test(x, test="hsic", permutations=permutations, seed=seed)
        dcov = pairwise_test(x, test="dcov", permutations=permutations, seed=seed ^ 0xA5A5A5A5)
        conservative, liberal, uncertain = ensemble(hsic, dcov, alpha)
        selected = conservative if method == "conservative_ensemble" else liberal
        return selected, {
            "test": "hsic_and_dcov", "transform": "raw",
            "pairwise_tests": 2 * x.shape[1] * (x.shape[1] - 1) // 2,
            "uncertain_fraction": float(uncertain[np.triu_indices(x.shape[1], 1)].mean()),
            "uncertain_pairs": int(uncertain.sum() // 2),
            "hsic_pvalue_matrix": json.dumps(hsic.pvalue.tolist()),
            "dcov_pvalue_matrix": json.dumps(dcov.pvalue.tolist()),
        }
    raise ValueError(f"unknown nonparametric method {method}")


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


def write_summary(frame, out: Path):
    """Write the compact, paired summary used for pilot comparisons."""
    import pandas as pd
    ok = frame[frame.status == "ok"].copy()
    if ok.empty:
        return
    metrics_to_report = ["udg_precision", "udg_recall", "udg_f1",
                         "no_trek_precision", "no_trek_recall", "no_trek_f1",
                         "predicted_udg_edges", "runtime"]
    summary = ok.groupby("method")[metrics_to_report].agg(["mean", "std", "count"])
    summary.to_csv(out / "summary_by_method.csv")

    ranks = []
    keys = ["d", "er_degree", "n", "graph_seed", "data_seed"]
    for key, group in ok.groupby(keys, dropna=False):
        for metric in ("no_trek_precision", "no_trek_f1", "udg_f1"):
            values = group[metric]
            for index, value in values.items():
                ranks.append({**dict(zip(keys, key)), "method": group.loc[index, "method"],
                              "metric": metric, "value": value,
                              "rank": values.rank(ascending=False, method="min").loc[index]})
    rank_frame = pd.DataFrame(ranks)
    rank_frame.to_csv(out / "paired_ranks.csv", index=False)
    if not rank_frame.empty:
        rank_frame.groupby(["method", "metric"])["rank"].agg(["mean", "std", "count"]).to_csv(
            out / "mean_rank_summary.csv")

    # Compare each method's operating point to the raw Fisher curve at the
    # closest predicted density and closest recall. This avoids rewarding a
    # method merely for returning an unusually sparse graph.
    matched = []
    for _, row in ok.iterrows():
        points = row.get("fisher_pr_points", "")
        if not isinstance(points, str) or not points:
            continue
        curve = pd.DataFrame(json.loads(points))
        if curve.empty:
            continue
        density_match = curve.iloc[(curve.predicted_udg_edges - row.predicted_udg_edges).abs().argmin()]
        recall_match = curve.iloc[(curve.udg_recall - row.udg_recall).abs().argmin()]
        base = {k: row[k] for k in ("d", "er_degree", "n", "graph_seed", "data_seed", "method")}
        matched.append({**base, "target_edges": row.predicted_udg_edges,
                        "target_recall": row.udg_recall,
                        "fisher_density_match_edges": density_match.predicted_udg_edges,
                        "fisher_density_match_no_trek_precision": density_match.no_trek_precision,
                        "method_minus_fisher_density_precision": row.no_trek_precision - density_match.no_trek_precision,
                        "fisher_recall_match_recall": recall_match.udg_recall,
                        "fisher_recall_match_no_trek_precision": recall_match.no_trek_precision,
                        "method_minus_fisher_recall_precision": row.no_trek_precision - recall_match.no_trek_precision})
    if matched:
        matched_frame = pd.DataFrame(matched)
        matched_frame.to_csv(out / "matched_fisher_summary.csv", index=False)
        matched_frame.groupby("method")[["method_minus_fisher_density_precision",
                                           "method_minus_fisher_recall_precision"]].agg(["mean", "std", "count"]).to_csv(
            out / "matched_fisher_aggregate.csv")

    with (out / "SUMMARY.md").open("w") as handle:
        handle.write("# UDG/UEC benchmark summary\n\n")
        handle.write("This summary is paired by graph and data seed. Higher precision, recall, and F1 are better; runtime is lower.\n\n")
        handle.write("## Mean performance\n\n```text\n")
        handle.write(ok.groupby("method")[metrics_to_report].mean().round(3).to_string())
        handle.write("\n```\n\n## Mean rank\n\n")
        if not rank_frame.empty:
            handle.write(rank_frame.groupby(["method", "metric"])["rank"].mean().unstack().round(2).to_string())
        handle.write("\n\n## Interpretation\n\n")
        handle.write("The direct covariance-BIC output is an absolute penalized model-selection point. "
                     "The matched-Fisher files compare methods at comparable output density or recall, "
                     "so sparsity is not mistaken for superior no-trek precision.\n")


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
                    fisher_curve_json = json.dumps(fisher_pr_curve(x, truth), separators=(",", ":"))
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
                            elif method in ("pairwise_hsic_raw", "pairwise_hsic_rank",
                                             "pairwise_dcov_raw", "pairwise_dcov_rank",
                                             "hsic_stability", "dcov_stability",
                                             "conservative_ensemble", "liberal_ensemble"):
                                est, diag = nonparametric_method(
                                    x, method, args.nonparam_alpha, args.nonparam_permutations,
                                    args.stability_replicates, args.stability_fraction,
                                    args.stability_threshold,
                                    seed_for(method, d, degree, graph_seed, data_seed))
                            elif method == "flop_candidates_covbic":
                                # The current FLOP Python API returns one final
                                # adjacency, not the individual restart
                                # candidates.  Keep this method a genuine
                                # candidate reranker: score exactly that one
                                # projected UDG and do not perform covariance
                                # local moves afterwards.  The diagnostics
                                # make the one-candidate limitation explicit.
                                import flopsearch
                                flop_dag = np.asarray(flopsearch.flop(
                                    x, 2.0, restarts=args.flop_restarts))
                                if flop_dag.shape != (d, d):
                                    raise ValueError(f"unexpected FLOP output shape {flop_dag.shape}")
                                est = true_udg(flop_dag != 0)
                                bits = matrix_to_bits(est)
                                digest = hashlib.sha256(str(bits).encode()).hexdigest()[:16]
                                diag = {
                                    "reranking_only": True,
                                    "flop_candidate_count": 1,
                                    "flop_candidate_edge_counts": json.dumps([int(est.sum() // 2)]),
                                    "flop_candidate_hashes": json.dumps([digest]),
                                    "selected_candidate_hash": digest,
                                    "selected_candidate_edge_count": int(est.sum() // 2),
                                    "postselection_local_moves": 0,
                                    "candidate_covbic": "not_available_from_flop_api",
                                    "note": "flopsearch exposes one final result; no hidden restart set was inferred",
                                }
                            elif method == "order_uec_covbic" or method.startswith("order_uec_covbic_lambda_"):
                                edge_penalty = None
                                if method.startswith("order_uec_covbic_lambda_"):
                                    edge_penalty = float(method.rsplit("_", 1)[1])
                                est, diag = search_order_uec(
                                    x, edge_penalty=edge_penalty,
                                    restarts=args.order_restarts,
                                    beam_width=args.order_beam_width,
                                    max_evaluations=args.order_budget,
                                    random_seed=seed_for(method, d, degree, graph_seed, data_seed),
                                    correlation=corr)
                                diag["proposal"] = "order_and_source_clique_moves"
                            elif method in ("uec_covbic_grow_shrink", "fisher_valid_smig", "covbic_unrestricted"):
                                initial = None
                                if method == "fisher_valid_smig":
                                    initial = matrix_to_bits(fisher_method(x, .05, False)[0])
                                est, diag, elapsed_cov = covariance_method(
                                    x, d, args.n, seed_for(method, d, degree, graph_seed, data_seed),
                                    args.covbic_restarts, args.covbic_budget,
                                    method not in ("covbic_unrestricted",), args.icf_iterations, initial)
                                diag["proposal"] = "fisher" if method == "fisher_valid_smig" else "udg_moves"
                            elif method == "flop_to_uec":
                                try:
                                    import flopsearch
                                    flop_estimate = np.asarray(flopsearch.flop(x, 2.0, restarts=args.flop_restarts))
                                    if flop_estimate.shape != (d, d):
                                        raise ValueError(f"unexpected FLOP output shape {flop_estimate.shape}")
                                    est = true_udg(flop_estimate != 0)
                                    diag = {"status": "ok", "projection": "ancestor_overlap",
                                            "flop_restarts": args.flop_restarts}
                                except Exception as exc:
                                    status = "unavailable"; diag = {"error": repr(exc)}; est = np.zeros_like(truth)
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
                            key: np.nan for key in ("predicted_udg_edges", "true_udg_edges", "udg_shd", "udg_precision", "udg_recall", "udg_f1",
                                                     "no_trek_precision", "no_trek_recall", "no_trek_f1",
                            "no_trek_fdr")})
                        # Attach the same Fisher operating curve to every
                        # method on this paired instance, so matched-density
                        # and matched-recall comparisons are meaningful.
                        row["fisher_pr_points"] = fisher_curve_json if status == "ok" else ""
                        row.update({f"diag_{k}": v for k,v in diag.items()})
                        if method.startswith("source_mask") and status == "ok":
                            row["udg_validity_witness"] = int(np.array_equal(source_mask_udg(masks,d), true_udg(witness)))
                        else: row["udg_validity_witness"] = "NA"
                        rows.append(row)
                        print(f"d={d} ER{degree} graph={graph_seed} data={data_seed} {method} {status} "
                              f"no-trek-F1={row['no_trek_f1']:.3f} runtime={elapsed:.2f}s", flush=True)
    import pandas as pd
    frame = pd.DataFrame(rows); frame.to_csv(out / "results.csv", index=False)
    path_rows = []
    for row in rows:
        raw_path = row.get("diag_threshold_path", "")
        if raw_path:
            for threshold, edges in json.loads(raw_path).items():
                path_rows.append({k: row[k] for k in ("d", "er_degree", "n", "graph_seed", "data_seed", "method")} |
                                 {"threshold": float(threshold), "predicted_edges": edges})
    if path_rows:
        pd.DataFrame(path_rows).to_csv(out / "nonparametric_stability_paths.csv", index=False)
    fisher_rows = []
    for row in rows:
        if row.get("fisher_pr_points"):
            for point in json.loads(row["fisher_pr_points"]):
                point.update({k: row[k] for k in ("d", "er_degree", "n", "graph_seed", "data_seed")})
                fisher_rows.append(point)
    if fisher_rows:
        pd.DataFrame(fisher_rows).to_csv(out / "fisher_pr_curve.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "dimensions": args.dimensions, "degrees": args.degrees, "n": args.n,
        "graph_seeds": args.graph_seeds, "data_seeds": args.data_seeds,
        "methods": methods, "restarts": args.restarts, "moves": args.moves,
        "grues_iterations": args.grues_iterations,
        "covbic_restarts": args.covbic_restarts, "covbic_budget": args.covbic_budget,
        "icf_iterations": args.icf_iterations, "flop_restarts": args.flop_restarts,
        "order_restarts": args.order_restarts, "order_beam_width": args.order_beam_width,
        "order_budget": args.order_budget,
        "nonparam_alpha": args.nonparam_alpha,
        "nonparam_permutations": args.nonparam_permutations,
        "stability_replicates": args.stability_replicates,
        "stability_fraction": args.stability_fraction,
        "stability_threshold": args.stability_threshold,
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
    write_summary(frame, out)
    ok = frame[frame.status == "ok"]
    print("\n=== UDG PILOT SUMMARY ===", flush=True)
    if ok.empty:
        print("No successful method rows.", flush=True)
    else:
        shell_metrics = ["no_trek_precision", "no_trek_recall", "no_trek_f1",
                         "udg_precision", "udg_recall", "udg_f1",
                         "predicted_udg_edges", "runtime"]
        print(ok.groupby("method")[shell_metrics].mean().round(3).to_string(), flush=True)
        print("\n=== STATUS COUNTS ===", flush=True)
        print(frame.groupby(["method", "status"]).size().to_string(), flush=True)
        rank_cols = ["no_trek_precision", "no_trek_f1", "udg_f1"]
        rank_rows = []
        for key, group in ok.groupby(["d", "er_degree", "n", "graph_seed", "data_seed"], dropna=False):
            for metric in rank_cols:
                ranks = group[metric].rank(ascending=False, method="min")
                rank_rows.extend({"method": group.loc[index, "method"], "metric": metric,
                                  "rank": value} for index, value in ranks.items())
        if rank_rows:
            print("\n=== MEAN RANK (1=best) ===", flush=True)
            print(pd.DataFrame(rank_rows).groupby(["method", "metric"])["rank"].mean().unstack().round(2).to_string(), flush=True)
    print(f"\nDetailed summary: {out / 'SUMMARY.md'}", flush=True)
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
             "The source-mask methods are structured UDG witnesses, not GrUES posterior samplers.",
             "The direct covariance methods optimize zeros of Sigma. They never pass estimated pairs to a causal optimizer."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/udg_pilot"))
    p.add_argument("--dimensions", nargs="+", type=int, default=[10,20])
    p.add_argument("--degrees", nargs="+", type=int, default=[2,4])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--graph-seeds", nargs="+", type=int, default=[7101,7102,7103])
    p.add_argument("--data-seeds", nargs="+", type=int, default=[8101,8102])
    p.add_argument("--methods", nargs="+", default=["pairwise_fisherz_raw","pairwise_fisherz_holm",
                                                       "flop_to_uec","uec_covbic_grow_shrink",
                                                       "fisher_valid_smig","covbic_unrestricted",
                                                       "order_uec_covbic","pairwise_hsic_raw",
                                                       "pairwise_hsic_rank","pairwise_dcov_raw",
                                                       "pairwise_dcov_rank","hsic_stability",
                                                       "dcov_stability","conservative_ensemble",
                                                       "liberal_ensemble","grues_exact"])
    p.add_argument("--restarts", type=int, default=8); p.add_argument("--moves", type=int, default=3000)
    p.add_argument("--grues-iterations", type=int, default=3000)
    p.add_argument("--covbic-restarts", type=int, default=4)
    p.add_argument("--covbic-budget", type=int, default=300)
    p.add_argument("--icf-iterations", type=int, default=150)
    p.add_argument("--flop-restarts", type=int, default=8)
    p.add_argument("--nonparam-alpha", type=float, default=0.05)
    p.add_argument("--nonparam-permutations", type=int, default=99)
    p.add_argument("--stability-replicates", type=int, default=20)
    p.add_argument("--stability-fraction", type=float, default=0.8)
    p.add_argument("--stability-threshold", type=float, default=0.8)
    p.add_argument("--order-restarts", type=int, default=16)
    p.add_argument("--order-beam-width", type=int, default=4)
    p.add_argument("--order-budget", type=int, default=300)
    run(p.parse_args())


if __name__ == "__main__": main()
