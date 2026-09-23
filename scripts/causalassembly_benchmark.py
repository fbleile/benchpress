"""Cached causalAssembly benchmark runner.

Preparation is deliberately separate because official DRF fitting may require
R/rpy2.  This runner never fits DRFs; it consumes the cache made by
``causalassembly_protocol.py prepare`` and is resume-safe at row level.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.causalassembly_protocol import (
    DISCOVERY_SIZES, derive_seed, nested_pairs, nonlinear_screen,
    oracle_no_treks,
)
from scripts.notreks_benchmark_pipeline import method_run
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import metrics


CORE_METHODS = (
    "flop", "flop-nt-standard", "flop-nt-edge-mask", "flop-nt-post",
    "dagma", "dagma-pstrek", "dagma-nt-post",
    "dagma-nonlinear", "dagma-nonlinear-pstrek",
)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def _screen(cache: Path, seed: int, args) -> list[tuple[int, int]]:
    path = cache / f"screen_{seed}.json"
    if path.exists():
        return [tuple(p) for p in json.loads(path.read_text())["pairs"]]
    artifact = np.load(cache / f"seed_{seed}.npz")
    pairs, diagnostics, summary = nonlinear_screen(
        artifact["reference"], candidate_cap=args.screen_candidate_cap,
        blocks=args.screen_blocks, permutations=args.screen_permutations,
        null_quantile=args.screen_null_quantile,
        effect_margin=args.screen_effect_margin,
        seed=derive_seed("screen", seed))
    diagnostics.to_csv(cache / f"screen_{seed}.csv", index=False)
    (cache / f"screen_{seed}.json").write_text(
        json.dumps({**summary, "pairs": [list(p) for p in pairs]}, indent=2) + "\n")
    return pairs


def _ensure_cached_seeds(cache: Path, seeds: tuple[int, ...], manifest: dict) -> None:
    """Materialize missing labels for the fixed upstream static release.

    The official static causalAssembly release is one fixed 500-row dataset;
    protocol seeds are labels, not independent regenerated datasets.  A
    partially prepared cache may therefore safely reuse any existing static
    artifact for missing requested labels.  Dynamic DRF caches are never
    filled this way.
    """
    missing = [seed for seed in seeds
               if not (cache / f"seed_{seed}.npz").exists()]
    if not missing:
        return
    if not manifest.get("fixed_dataset", False):
        raise FileNotFoundError(
            "missing causalAssembly cache artifacts for seeds "
            f"{missing}; prepare those seeds before running the benchmark")
    existing = sorted(cache.glob("seed_*.npz"))
    if not existing:
        raise FileNotFoundError(
            "the fixed causalAssembly cache has no seed artifact; run "
            "prepare-static first")
    source = existing[0]
    with np.load(source) as artifact:
        reference = np.asarray(artifact["reference"])
        discovery = np.asarray(artifact["discovery"])
    for seed in missing:
        np.savez_compressed(cache / f"seed_{seed}.npz",
                            reference=reference, discovery=discovery)
    manifest["completed_seeds"] = sorted(set(manifest.get("completed_seeds", []))
                                          | set(seeds))
    (cache / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n")


def run(args) -> None:
    cache = args.cache.resolve()
    manifest = json.loads((cache / "manifest.json").read_text())
    truth = np.load(cache / "ground_truth.npz")["truth"].astype(np.uint8)
    all_oracle_pairs = oracle_no_treks(truth)
    aliases = {
        "flop_notreks": "flop-nt-standard",
        "flop_notreks_edge_mask": "flop-nt-edge-mask",
        "flop_notreks_postselection": "flop-nt-post",
        "dagma_notreks": "dagma-pstrek",
        "dagma_notreks_postselection": "dagma-nt-post",
        "dagma_nonlinear": "dagma-nonlinear",
        "dagma_nonlinear_notreks": "dagma-nonlinear-pstrek",
    }
    methods = tuple(aliases.get(m, m) for m in (args.methods or CORE_METHODS))
    unknown = set(methods) - set(CORE_METHODS)
    if unknown:
        raise ValueError(f"unknown causalAssembly methods: {sorted(unknown)}")
    sizes = tuple(args.n or DISCOVERY_SIZES)
    seeds = tuple(args.seeds)
    _ensure_cached_seeds(cache, seeds, manifest)
    rows_path = args.output / "raw" / "results.csv"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(rows_path).to_dict("records") if rows_path.exists() else []
    done = {(r.get("seed"), r.get("n"), r.get("knowledge_mode"),
             r.get("knowledge_fraction"), r.get("method"))
            for r in rows if r.get("solver_status") == "ok"}
    for seed in seeds:
        artifact = np.load(cache / f"seed_{seed}.npz")
        if args.mode == "oracle_notreks":
            knowledge = {q: nested_pairs(
                all_oracle_pairs, q,
                derive_seed("oracle-order", seed)) for q in args.q}
        else:
            inferred = _screen(cache, seed, args)
            knowledge = {"estimated": inferred}
        for n in sizes:
            raw = artifact["discovery"][:n]
            # Keep the published causalAssembly variables in their original
            # physical units. Synthetic protocol data are standardized, but
            # this real-world/semisynthetic benchmark is intentionally not.
            X = np.asarray(raw, dtype=float).copy()
            means = X.mean(axis=0)
            scales = X.std(axis=0, ddof=0)
            q_items = list(knowledge.items())
            # Vanilla methods are run once and then paired with every q in
            # analysis; NOTREKS methods run for each supplied set.
            jobs = [("q0", [], 0.0, "vanilla")]
            jobs.extend((str(q), pairs, float(q) if q != "estimated" else np.nan,
                         args.mode) for q, pairs in q_items)
            for q_label, pairs, q_value, knowledge_mode in jobs:
                for method in methods:
                    if method in {"flop", "dagma", "dagma-nonlinear"} and q_label != "q0":
                        continue
                    key = (seed, n, knowledge_mode, q_value, method)
                    if key in done:
                        continue
                    start = time.perf_counter()
                    status, error = "ok", ""
                    try:
                        candidate, diag, runtime = method_run(
                            method, X, pairs, derive_seed("method", seed, n,
                            q_label, method), args.attempts, args.flop_sweeps,
                            args.dagma_stages, args.dagma_warm_iter,
                            args.dagma_max_iter, args.trek_weight)
                        row = metrics(X, truth, pairs, method,
                                      diag["candidate_graph"], candidate,
                                      diag["cpdag"], runtime, args.attempts,
                                      diag.get("optimizer_restarts", args.attempts))
                    except Exception as exc:
                        status, error = "failed", repr(exc)
                        row = {"SHD_cpdag": np.nan, "F1_skel": np.nan,
                               "violations_after": np.nan,
                               "candidate_runtime": time.perf_counter() - start}
                    row.update({
                        "experiment_id": "causalassembly",
                        "dataset": "causalassembly_full", "seed": seed,
                        "n": n, "d": truth.shape[0], "method": method,
                        "knowledge_mode": knowledge_mode,
                        "knowledge_fraction": q_value,
                        "requested_knowledge_pairs": len(pairs),
                        "oracle_knowledge_pairs": len(all_oracle_pairs),
                        "data_model": "causalAssembly_semisynthetic_nonlinear",
                        "discovery_score": "gaussian_bic_existing_solver",
                        "score_misspecification": True,
                        "standardized_for_discovery": False,
                        "sample_mean_checksum": _hash(means.tolist()),
                        "sample_scale_checksum": _hash(scales.tolist()),
                        "sample_checksum": _hash(raw.tolist()),
                        "solver_status": status, "solver_error": error,
                        "package_metadata": json.dumps(manifest.get("package", {})),
                    })
                    supplied = {tuple(map(int, p)) for p in pairs}
                    oracle = {tuple(map(int, p)) for p in all_oracle_pairs}
                    tp = len(supplied & oracle)
                    fp = len(supplied - oracle)
                    fn = len(oracle - supplied)
                    row.update({
                        "knowledge_tp": tp, "knowledge_fp": fp,
                        "knowledge_fn": fn,
                        "knowledge_precision": tp / max(1, tp + fp),
                        "knowledge_recall": tp / max(1, tp + fn),
                        "knowledge_f1": (2 * tp / max(1, 2 * tp + fp + fn)),
                    })
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(rows_path, index=False)
                    print(f"causalassembly seed={seed} n={n} q={q_label} "
                          f"{method} {status} SHD={row.get('SHD_cpdag')}", flush=True)
    out = args.output / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.groupby(["knowledge_mode", "method"], dropna=False).agg(
            runs=("method", "size"), SHD_cpdag_mean=("SHD_cpdag", "mean"),
            SHD_cpdag_std=("SHD_cpdag", "std"), runtime_mean=("candidate_runtime", "mean"),
            failures=("solver_status", lambda s: int((s != "ok").sum()))
        ).reset_index().to_csv(out / "summary.csv", index=False)
    (args.output / "manifest.json").write_text(json.dumps({
        "experiment": "causalassembly", "mode": args.mode,
        "seeds": list(seeds), "n": list(sizes), "q": args.q,
        "methods": list(methods), "attempts": args.attempts,
        "oracle_pairs": len(all_oracle_pairs), "cache_manifest": manifest,
    }, indent=2, default=str) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("oracle_notreks", "estimated_notreks"),
                   default="oracle_notreks")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--n", type=int, nargs="*")
    p.add_argument("--q", type=float, nargs="+", default=[.10, .25, .50, 1.0])
    p.add_argument("--methods", nargs="+")
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--flop-sweeps", type=int, default=16)
    p.add_argument("--dagma-stages", type=int, default=5)
    p.add_argument("--dagma-warm-iter", type=int, default=3000)
    p.add_argument("--dagma-max-iter", type=int, default=6000)
    p.add_argument("--trek-weight", type=float, default=200.0)
    p.add_argument("--screen-candidate-cap", type=int, default=750)
    p.add_argument("--screen-blocks", type=int, default=5)
    p.add_argument("--screen-permutations", type=int, default=99)
    p.add_argument("--screen-null-quantile", type=float, default=.10)
    p.add_argument("--screen-effect-margin", type=float, default=.02)
    args = p.parse_args()
    if args.attempts < 1:
        p.error("--attempts must be positive")
    run(args)


if __name__ == "__main__":
    main()
