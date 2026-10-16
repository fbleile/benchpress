"""Resume-safe CausalMan benchmark with projection-aware evaluation.

The official CausalMan graph may induce bidirected edges after latent
projection.  In that case causal DAG distances are deliberately left missing;
the projected directed-skeleton numbers below are descriptive only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.causalman_protocol import DISCOVERY_SIZES, VARIANTS, derive_seed
from scripts.notreks_benchmark_pipeline import method_run
from workflow.rules.structure_learning_algorithms.flop_notreks.global_greedy import (
    score_nonlinear_dag,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import (
    order_parent_postselection_from_candidate,
)

METHODS = (
    "flop", "flop-nt-standard", "flop-nt-edge-mask", "flop-nt-post",
    "dagma", "dagma-pstrek", "dagma-nt-edge-mask", "dagma-nt-post",
    "dagma-nonlinear", "dagma-nonlinear-pstrek",
    "flop-nonlinear", "flop-nonlinear-pstrek",
    "flop-nonlinear-admg", "flop-nonlinear-admg-pstrek",
    "var_sortnregress", "r2_sortnregress",
)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _standardize(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return (x - mean) / scale, mean, scale


def _load(cache: Path, variant: str, seed: int):
    root = cache / variant / f"seed_{seed}"
    x = np.load(root.with_suffix(".npy"))
    graph = json.loads(root.with_name(root.name + "_graph.json").read_text())
    manifest = json.loads(root.with_name(root.name + "_manifest.json").read_text())
    return x, graph, manifest


def _pairs_for_q(oracle, q, round_id, seed):
    oracle = [tuple(p) for p in oracle]
    if q <= 0:
        return []
    if q >= 1:
        return oracle
    count = int(round(q * len(oracle)))
    rng = np.random.default_rng(derive_seed("causalman-knowledge", seed, round_id, q))
    chosen = rng.choice(len(oracle), size=count, replace=False) if count else []
    return [oracle[int(i)] for i in np.sort(chosen)]


def _index_pairs(pairs, nodes):
    index = {str(node): i for i, node in enumerate(nodes)}
    return [(index[a], index[b]) for a, b in pairs if a in index and b in index]


def _skeleton(a):
    a = np.asarray(a, dtype=bool)
    return (a | a.T).astype(np.uint8)


def _projection_metrics(candidate, graph, nodes):
    truth = _projected_directed_adjacency(graph, nodes)
    d = len(nodes)
    estimate = np.asarray(candidate, dtype=np.uint8)
    if estimate.shape != truth.shape:
        estimate = estimate[:d, :d]
    ts, es = _skeleton(truth), _skeleton(estimate)
    tp = int(((ts != 0) & (es != 0)).sum() // 2)
    fp = int(((ts == 0) & (es != 0)).sum() // 2)
    fn = int(((ts != 0) & (es == 0)).sum() // 2)
    return {
        "projected_skeleton_shd": fp + fn,
        "projected_skeleton_precision": tp / max(1, tp + fp),
        "projected_skeleton_recall": tp / max(1, tp + fn),
        "projected_skeleton_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "projected_directed_edges": int(truth.sum()),
        "projected_bidirected_edges": len(graph.get("projected", {}).get("bidirected", [])),
    }


def _projected_directed_adjacency(graph, nodes):
    d = len(nodes)
    truth = np.zeros((d, d), dtype=np.uint8)
    index = {str(v): i for i, v in enumerate(nodes)}
    for a, b in graph.get("projected", {}).get("directed", []):
        if a in index and b in index:
            truth[index[a], index[b]] = 1
    return truth


def _violations(candidate, pairs, nodes):
    indexed = _index_pairs(pairs, nodes)
    return int(sum(bool(candidate[i, j] or candidate[j, i]) for i, j in indexed))


def _bidirected_metrics(estimate, graph, nodes):
    estimate = np.asarray(estimate, dtype=bool)
    truth = np.zeros_like(estimate, dtype=bool)
    index = {str(v): i for i, v in enumerate(nodes)}
    for a, b in graph.get("projected", {}).get("bidirected", []):
        if a in index and b in index:
            truth[index[a], index[b]] = truth[index[b], index[a]] = True
    tri = np.triu(np.ones_like(truth, dtype=bool), 1)
    tp = int(np.sum(truth & estimate & tri))
    fp = int(np.sum(~truth & estimate & tri))
    fn = int(np.sum(truth & ~estimate & tri))
    return {"bidirected_edges": int(estimate[tri].sum()),
            "bidirected_precision": tp / max(1, tp + fp),
            "bidirected_recall": tp / max(1, tp + fn),
            "bidirected_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "bidirected_shd": fp + fn}


def _candidate(method, x, pairs, seed, args, cached_vanilla=None):
    if method in {"flop-nt-post", "dagma-nt-post"} and cached_vanilla is not None:
        started = time.perf_counter()
        candidate, diag = order_parent_postselection_from_candidate(x, cached_vanilla[0], pairs)
        diag = {**diag, "candidate_graph": np.asarray(cached_vanilla[0], dtype=np.uint8)}
        return candidate, diag, time.perf_counter() - started
    attempts = (args.flop_attempts if method.startswith("flop")
                else args.dagma_attempts)
    return method_run(method, x, pairs, seed, attempts,
                      args.flop_sweeps, args.dagma_stages,
                      args.dagma_warm_iter, args.dagma_max_iter,
                      args.trek_weight,
                      dagma_nonlinear_hidden_dim=args.dagma_nonlinear_hidden_dim,
                      dagma_nonlinear_score=args.dagma_nonlinear_score,
                      dagma_nonlinear_student_t_nu=args.dagma_nonlinear_student_t_nu,
                      dagma_nonlinear_lambda1=args.dagma_nonlinear_lambda1,
                      flop_nonlinear_hidden_dim=args.flop_nonlinear_hidden_dim,
                      flop_nonlinear_epochs=args.flop_nonlinear_epochs,
                      flop_nonlinear_score=args.flop_nonlinear_score,
                      flop_nonlinear_student_t_nu=args.flop_nonlinear_student_t_nu,
                      flop_nonlinear_lambda_bic=args.flop_nonlinear_lambda_bic,
                      flop_nonlinear_max_block_proposals=(
                          args.flop_nonlinear_max_block_proposals))


def run(args):
    methods = tuple(args.methods or METHODS)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unsupported CausalMan methods: {unknown}")
    output = args.output.resolve()
    raw_path = output / "raw" / "results.csv"
    output.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records")
    done = {(r.get("variant"), int(r.get("seed")), int(r.get("n")),
             r.get("method"), r.get("knowledge_round"), r.get("solver_status")) for r in rows}
    for variant in args.variants:
        for seed in args.seeds:
            data, graph, manifest = _load(args.cache, variant, seed)
            nodes = graph["selected_nodes"]
            oracle = [tuple(p) for p in graph["oracle_no_treks"]]
            if not manifest.get("eligible_for_oracle", False):
                eligible = False
                ineligible_reason = manifest.get("eligibility_reason", "preflight_failed")
            else:
                eligible, ineligible_reason = True, ""
            for n in args.n_values:
                x, mean, scale = _standardize(data[:n])
                cache_vanilla = {}
                score_cache = {}
                projected_reference = _projected_directed_adjacency(graph, nodes)
                reference_score = np.nan
                reference_score_status = "not_computed"
                if any(method.startswith("flop-nonlinear") or
                       method.startswith("dagma-nonlinear") for method in methods):
                    try:
                        reference_score = score_nonlinear_dag(
                            x, projected_reference,
                            seed=derive_seed("causalman-score", variant, seed, n),
                            hidden_dim=args.flop_nonlinear_hidden_dim,
                            epochs=args.flop_nonlinear_epochs,
                            score=args.flop_nonlinear_score,
                            student_t_nu=args.flop_nonlinear_student_t_nu,
                            lambda_bic=args.flop_nonlinear_lambda_bic)
                        reference_score_status = (
                            "projected_directed_only_latent_confounding"
                            if graph.get("projected", {}).get("bidirected")
                            else "causally_sufficient_directed_reference")
                    except Exception as exc:
                        reference_score_status = "failed:" + repr(exc)
                for q in args.q_values:
                    rounds = (0,) if q in (0.0, 1.0) else tuple(range(args.q25_rounds))
                    for round_id in rounds:
                        if q > 0 and not eligible:
                            for method in methods:
                                if (q == 1.0 and method not in {"flop-nt-standard", "dagma-pstrek"}) or (q == .25 and method in {"flop", "dagma"}):
                                    continue
                                rows.append({"variant": variant, "seed": seed, "n": n,
                                             "method": method, "knowledge_fraction": q,
                                             "knowledge_round": round_id,
                                             "solver_status": "ineligible",
                                             "failure_reason": ineligible_reason,
                                             "metric_exclusion_reason": "latent_confounding"})
                            continue
                        eligible_methods = ({"flop", "dagma", "dagma-nonlinear",
                                             "flop-nonlinear", "flop-nonlinear-admg"} if q == 0 else
                                            {"flop-nt-standard", "flop-nt-edge-mask", "flop-nt-post",
                                             "dagma-pstrek", "dagma-nt-edge-mask", "dagma-nt-post",
                                            "dagma-nonlinear-pstrek",
                                            "flop-nonlinear-pstrek", "flop-nonlinear-admg-pstrek"})
                        if q == 0:
                            eligible_methods |= {"var_sortnregress", "r2_sortnregress"}
                        pairs = _pairs_for_q(oracle, q, round_id, seed)
                        for method in methods:
                            if method not in eligible_methods:
                                continue
                            key = (variant, int(seed), int(n), method, round_id, "ok")
                            if key in done:
                                continue
                            started = time.perf_counter()
                            status, error = "ok", ""
                            try:
                                base = cache_vanilla.get((n, "flop" if method.startswith("flop") else "dagma"))
                                candidate, diag, runtime = _candidate(
                                    method, x, _index_pairs(pairs, nodes),
                                    derive_seed("causalman-method", variant, seed, n, method, round_id),
                                    args, base)
                                candidate = np.asarray(diag.get("candidate_graph", candidate), dtype=np.uint8)
                                bm = _bidirected_metrics(
                                    diag.get("bidirected_graph", np.zeros_like(candidate)),
                                    graph, nodes)
                                if method in {"flop", "dagma"}:
                                    cache_vanilla[(n, method)] = (candidate, diag, runtime)
                                pm = _projection_metrics(candidate, graph, nodes)
                                candidate_score = np.nan
                                score_status = "not_computed"
                                score_gap = np.nan
                                if method.startswith(("flop-nonlinear", "dagma-nonlinear")):
                                    score_key = (candidate.tobytes(), args.flop_nonlinear_score,
                                                 args.flop_nonlinear_hidden_dim,
                                                 args.flop_nonlinear_epochs)
                                    if score_key not in score_cache:
                                        score_cache[score_key] = score_nonlinear_dag(
                                            x, candidate,
                                            seed=derive_seed("causalman-score", variant, seed, n),
                                            hidden_dim=args.flop_nonlinear_hidden_dim,
                                            epochs=args.flop_nonlinear_epochs,
                                            score=args.flop_nonlinear_score,
                                            student_t_nu=args.flop_nonlinear_student_t_nu,
                                            lambda_bic=args.flop_nonlinear_lambda_bic)
                                    candidate_score = score_cache[score_key]
                                    score_gap = candidate_score - reference_score
                                    score_status = reference_score_status
                                row = {**pm, "variant": variant, "seed": seed, "n": n,
                                       "method": method, "knowledge_fraction": q,
                                       "knowledge_round": round_id, "requested_knowledge_pairs": len(pairs),
                                       "oracle_knowledge_pairs": len(oracle),
                                       "knowledge_checksum": _hash(pairs),
                                       "knowledge_violations": _violations(candidate, pairs, nodes),
                                       "oracle_no_trek_violations": _violations(candidate, oracle, nodes),
                                       "candidate_edges": int(np.asarray(candidate).sum()),
                                       **bm,
                                       "mixed_graph_score": diag.get("admg_score", np.nan),
                                       "admg_bidirected_edges": diag.get(
                                           "admg_bidirected_edges", np.nan),
                                       "nonlinear_score": candidate_score,
                                       "nonlinear_score_per_observation": (
                                           candidate_score / n if np.isfinite(candidate_score) else np.nan),
                                       "projected_directed_reference_score": reference_score,
                                       "projected_directed_reference_score_per_observation": (
                                           reference_score / n if np.isfinite(reference_score) else np.nan),
                                       "nonlinear_score_gap_to_reference": score_gap,
                                       "nonlinear_score_gap_per_observation": (
                                           score_gap / n if np.isfinite(score_gap) else np.nan),
                                       "nonlinear_score_reference_status": score_status,
                                       "candidate_runtime": float(runtime),
                                       "attempts_requested": (
                                           args.flop_attempts
                                           if method.startswith("flop")
                                           else args.dagma_attempts),
                                       "metric_exclusion_reason": ("latent_confounding" if pm["projected_bidirected_edges"] else ""),
                                       "SHD_cpdag": np.nan, "SID": np.nan, "AID": np.nan,
                                       "sample_mean_checksum": _hash(mean.tolist()),
                                       "sample_scale_checksum": _hash(scale.tolist()),
                                       "solver_status": status, "failure_reason": error}
                            except Exception as exc:
                                row = {"variant": variant, "seed": seed, "n": n,
                                       "method": method, "knowledge_fraction": q,
                                       "knowledge_round": round_id,
                                       "solver_status": "failed", "failure_reason": repr(exc),
                                       "candidate_runtime": time.perf_counter() - started,
                                       "metric_exclusion_reason": "latent_confounding"}
                            rows.append(row)
                            pd.DataFrame(rows).to_csv(raw_path, index=False)
                            print(f"causalman {variant} seed={seed} n={n} q={q} {method} {status}", flush=True)
    frame = pd.DataFrame(rows)
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    # Keep the result schema rectangular even when every row in a small smoke
    # is failed/ineligible or a method does not expose a particular diagnostic.
    # This also makes resumable mixed-method runs safe to aggregate.
    for column in ("projected_skeleton_shd", "projected_skeleton_f1",
                   "oracle_no_trek_violations", "candidate_runtime",
                   "nonlinear_score", "projected_directed_reference_score",
                   "nonlinear_score_gap_to_reference", "bidirected_edges",
                   "nonlinear_score_per_observation",
                   "projected_directed_reference_score_per_observation",
                   "nonlinear_score_gap_per_observation",
                   "bidirected_precision", "bidirected_recall", "bidirected_f1",
                   "bidirected_shd", "mixed_graph_score",
                   "admg_bidirected_edges"):
        if column not in frame:
            frame[column] = np.nan
    frame.to_csv(analysis / "all_rows.csv", index=False)
    if not frame.empty:
        summary = frame.groupby(["variant", "method", "knowledge_fraction", "n"], dropna=False).agg(
            runs=("method", "size"), successes=("solver_status", lambda s: int((s == "ok").sum())),
            failures=("solver_status", lambda s: int((s == "failed").sum())),
            ineligible=("solver_status", lambda s: int((s == "ineligible").sum())),
            projected_skeleton_shd=("projected_skeleton_shd", "mean"),
            projected_skeleton_f1=("projected_skeleton_f1", "mean"),
            oracle_no_trek_violations=("oracle_no_trek_violations", "mean"),
            nonlinear_score=("nonlinear_score", "mean"),
            projected_directed_reference_score=(
                "projected_directed_reference_score", "mean"),
            nonlinear_score_gap_to_reference=(
                "nonlinear_score_gap_to_reference", "mean"),
            nonlinear_score_per_observation=(
                "nonlinear_score_per_observation", "mean"),
            projected_directed_reference_score_per_observation=(
                "projected_directed_reference_score_per_observation", "mean"),
            nonlinear_score_gap_per_observation=(
                "nonlinear_score_gap_per_observation", "mean"),
            bidirected_edges=("bidirected_edges", "mean"),
            bidirected_precision=("bidirected_precision", "mean"),
            bidirected_recall=("bidirected_recall", "mean"),
            bidirected_f1=("bidirected_f1", "mean"),
            bidirected_shd=("bidirected_shd", "mean"),
            mixed_graph_score=("mixed_graph_score", "mean"),
            runtime_mean=("candidate_runtime", "mean"),
        ).reset_index()
        summary.to_csv(analysis / "summary.csv", index=False)
        print("\nCausalMan summary:\n" + summary.to_string(index=False), flush=True)
        report = ["# CausalMan run report", "", f"Rows: {len(frame)}",
                  f"Successful solver rows: {int((frame.solver_status == 'ok').sum())}",
                  f"Failed solver rows: {int((frame.solver_status == 'failed').sum())}",
                  f"Ineligible rows: {int((frame.solver_status == 'ineligible').sum())}",
                  "", "Causal DAG distances are intentionally missing for "
                  "latent-confounded projected cells (`latent_confounding`).",
                  "Projected directed-skeleton metrics are descriptive agreement, "
                  "not causal ground truth.", ""]
        (analysis / "report.md").write_text("\n".join(report))
    (output / "manifest.json").write_text(json.dumps({"variants": args.variants,
        "seeds": args.seeds, "n_values": args.n_values, "methods": methods,
        "q_values": args.q_values, "q25_rounds": args.q25_rounds,
        "metric_policy": "latent_confounding_excludes_causal_DAG_metrics"}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1001, 1011)))
    p.add_argument("--n-values", nargs="+", type=int, default=list(DISCOVERY_SIZES))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--q-values", nargs="+", type=float, default=[0.0, .25, 1.0])
    p.add_argument("--q25-rounds", type=int, default=5)
    p.add_argument("--flop-attempts", type=int, default=20)
    p.add_argument("--dagma-attempts", type=int, default=2)
    p.add_argument("--flop-sweeps", type=int, default=16)
    p.add_argument("--dagma-stages", type=int, default=5)
    p.add_argument("--dagma-warm-iter", type=int, default=30000)
    p.add_argument("--dagma-max-iter", type=int, default=60000)
    p.add_argument("--dagma-nonlinear-hidden-dim", type=int, default=10)
    p.add_argument("--dagma-nonlinear-score",
                   choices=("profiled_log_mse", "student_t"),
                   default="profiled_log_mse")
    p.add_argument("--dagma-nonlinear-student-t-nu", type=float, default=4.0)
    p.add_argument("--dagma-nonlinear-lambda1", type=float, default=0.02)
    p.add_argument("--flop-nonlinear-hidden-dim", type=int, default=16)
    p.add_argument("--flop-nonlinear-epochs", type=int, default=60)
    p.add_argument("--flop-nonlinear-score",
                   choices=("profiled_log_mse", "profiled_log_mse_cv",
                            "student_t", "student_t_cv"),
                   default="profiled_log_mse_cv")
    p.add_argument("--flop-nonlinear-student-t-nu", type=float, default=4.0)
    p.add_argument("--flop-nonlinear-lambda-bic", type=float, default=2.0)
    p.add_argument("--flop-nonlinear-max-block-proposals", type=int, default=0)
    p.add_argument("--trek-weight", type=float, default=1.0)
    args = p.parse_args()
    if any(q not in (0.0, .25, 1.0) for q in args.q_values):
        p.error("--q-values must be drawn from 0, 0.25, 1")
    if args.q25_rounds < 1:
        p.error("--q25-rounds must be positive")
    run(args)


if __name__ == "__main__":
    main()
