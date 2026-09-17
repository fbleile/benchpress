#!/usr/bin/env python3
"""Reproducible paired NOTREKS benchmark driver.

This is deliberately a small host-mode runner around the already validated
local solver implementations.  It writes one atomic CSV per instance/prior
job and invokes the analysis module after all requested jobs finish.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.chromatic import (
    chromatic_upper_bound,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import (
    corrupt_knowledge,
    common_ancestor_violations,
    dagma_candidate,
    flop_notreks_candidate,
    hybrid_flop_notreks_candidate,
    flop_notreks_postselection_candidate,
    flop_notreks_postselection_from_candidate,
    metrics,
    select_knowledge,
    vanilla_flop_candidate,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import (
    generate,
)
from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic,
)


METHODS = (
    "flop", "flop-nt-local", "dagma", "dagma-pstrek",
    "flop-edge-mask", "flop-nt-post", "flop-nt-global",
    "dagma-edge-mask", "dagma-nt-post",
)
EXPERIMENTS = (
    "main", "integration-ablation", "imperfect-knowledge",
    "knowledge-sweep", "high-dimension", "d100-flop", "d100-flop-local",
    "d100-flop-global",
    "exact-reference", "real-data",
)
PRINCIPAL = ("flop", "flop-nt-local", "dagma", "dagma-pstrek")


def parse_range(value: str) -> list[int]:
    try:
        start, end = (int(part) for part in value.split(":", 1))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "seed range must use inclusive START:END syntax") from exc
    if end < start:
        raise argparse.ArgumentTypeError("seed range END must be >= START")
    return list(range(start, end + 1))


def slug(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_metadata(root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True,
                stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    diff = run("diff", "HEAD")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty": bool(run("status", "--porcelain")),
        "git_patch_hash": hashlib.sha256(diff.encode()).hexdigest(),
    }


def nested_pairs(all_pairs, fraction: float, seed: int):
    """Return a prefix of one deterministic permutation of all true pairs."""
    if not 0 <= fraction <= 1:
        raise ValueError("knowledge fraction must lie in [0, 1]")
    pairs = [tuple(sorted((int(a), int(b)))) for a, b in all_pairs]
    rng = np.random.default_rng(seed + 104729)
    order = rng.permutation(len(pairs))
    count = int(round(fraction * len(pairs)))
    return [pairs[int(i)] for i in order[:count]]


def clique_lower_bound(d: int, pairs) -> int:
    """Certified (not necessarily maximum) clique lower bound for H=(V,I)."""
    neighbors = [set() for _ in range(d)]
    for left, right in pairs:
        left, right = int(left), int(right)
        neighbors[left].add(right); neighbors[right].add(left)
    best = 1 if d else 0
    for start in sorted(range(d), key=lambda node: len(neighbors[node]),
                        reverse=True):
        clique = [start]
        candidates = sorted(neighbors[start],
                            key=lambda node: len(neighbors[node]),
                            reverse=True)
        for node in candidates:
            if all(node in neighbors[member] for member in clique):
                clique.append(node)
        best = max(best, len(clique))
    return best


def cells(experiment: str):
    if experiment == "main":
        for d, n, degree, q in itertools.product(
                (20, 50), (100, 500, 2000), (2, 4), (.25, 1.0)):
            yield {"experiment_id": "main", "d": d, "n": n,
                   "degree": degree, "q": q, "c": 0.0,
                   "methods": PRINCIPAL}
    elif experiment == "integration-ablation":
        for degree, q in itertools.product((2, 4), (.25, 1.0)):
            yield {"experiment_id": "integration-ablation", "d": 50,
                   "n": 500, "degree": degree, "q": q, "c": 0.0,
                   "methods": METHODS}
    elif experiment == "imperfect-knowledge":
        # Keep the production corruption study manageable and feasible for
        # unordered, duplicate-free false-pair replacement.  The q=1,c=.25
        # combination is not guaranteed to have enough false pairs on d=50
        # ER2 instances; c=.25 remains available as a separate stress case.
        for q, c in itertools.product((.25, .5, 1.0), (0.0, .05, .1)):
            yield {"experiment_id": "imperfect-knowledge", "d": 50,
                   "n": 500, "degree": 2, "q": q, "c": c,
                   "methods": PRINCIPAL}
    elif experiment == "knowledge-sweep":
        for degree, q in itertools.product((2, 4), (.1, .25, .5, .75, 1.0)):
            yield {"experiment_id": "knowledge-sweep", "d": 50,
                   "n": 500, "degree": degree, "q": q, "c": 0.0,
                   "methods": PRINCIPAL}
    elif experiment == "high-dimension":
        yield {"experiment_id": "high-dimension", "d": 100, "n": 1000,
               "degree": 2, "q": .25, "c": 0.0, "methods": PRINCIPAL}
    elif experiment in {"d100-flop", "d100-flop-local", "d100-flop-global"}:
        # FLOP-only scaling and integration ablation at d=100.  The standard
        # study deliberately excludes the expensive global search; it can be
        # requested separately with d100-flop-global.
        methods = ("flop", "flop-edge-mask", "flop-nt-post",
                   "flop-nt-local")
        if experiment == "d100-flop-global":
            methods = methods + ("flop-nt-global",)
        for n, degree, q in itertools.product(
                (100, 500, 2000), (2, 4), (.25, 1.0)):
            yield {"experiment_id": experiment, "d": 100, "n": n,
                   "degree": degree, "q": q, "c": 0.0,
                   "methods": methods}
    elif experiment in {"exact-reference", "real-data"}:
        yield {"experiment_id": experiment, "d": None, "n": None,
               "degree": None, "q": None, "c": None, "methods": ()}


def method_run(name, X, pairs, seed, attempts, flop_sweeps, dagma_stages,
               dagma_warm_iter, dagma_max_iter, trek_weight):
    started = time.perf_counter()
    if name == "flop":
        candidate, diag = vanilla_flop_candidate(X, seed, attempts)
    elif name == "flop-nt-local":
        candidate, diag = hybrid_flop_notreks_candidate(
            X, pairs, seed, attempts, flop_sweeps,
            search_version="local_greedy_rust", local_greedy_passes=8)
    elif name == "flop-nt-global":
        candidate, diag = hybrid_flop_notreks_candidate(
            X, pairs, seed, attempts, flop_sweeps,
            search_version="global_greedy_rust", local_greedy_passes=8)
    elif name == "flop-edge-mask":
        from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import direct_mask_edges
        candidate, diag = flop_notreks_candidate(
            X, [], seed, attempts, flop_sweeps,
            forbidden_edges=direct_mask_edges(pairs),
            search_version="local_greedy_rust", local_greedy_passes=8)
    elif name == "flop-nt-post":
        candidate, diag = flop_notreks_postselection_candidate(
            X, pairs, seed, attempts)
    elif name == "dagma":
        candidate, diag = dagma_candidate(
            X, [], False, False, seed, attempts, dagma_warm_iter,
            dagma_max_iter, dagma_stages, trek_weight=trek_weight)
    elif name == "dagma-pstrek":
        candidate, diag = dagma_candidate(
            X, pairs, True, False, seed, attempts, dagma_warm_iter,
            dagma_max_iter, dagma_stages, trek_weight=trek_weight)
    elif name == "dagma-edge-mask":
        candidate, diag = dagma_candidate(
            X, pairs, False, True, seed, attempts, dagma_warm_iter,
            dagma_max_iter, dagma_stages, trek_weight=trek_weight)
    elif name == "dagma-nt-post":
        candidate, diag = dagma_candidate(
            X, pairs, False, False, seed, attempts, dagma_warm_iter,
            dagma_max_iter, dagma_stages,
            apply_notreks_postselection=True, trek_weight=trek_weight)
    else:
        raise ValueError(f"unknown method {name}")
    elapsed = time.perf_counter() - started
    return candidate, diag, elapsed


def postselect_cached_flop(X, pairs, cached):
    """Run only NOTREKS post-selection on a cached vanilla FLOP result."""
    base_candidate, base_diag, base_runtime = cached
    started = time.perf_counter()
    candidate, post_diag = flop_notreks_postselection_from_candidate(
        X, base_candidate, pairs)
    elapsed = time.perf_counter() - started
    diag = {
        **post_diag,
        "candidate_graph": base_candidate.copy(),
        "optimizer_restarts": base_diag["optimizer_restarts"],
        "base_solver_runtime": base_runtime,
    }
    return candidate, diag, elapsed


def run_job(root: Path, out: Path, cell: dict, seed: int, attempts: int,
            smoke: bool = False, solver_cache: dict | None = None) -> Path:
    job_started = datetime.now(timezone.utc).isoformat()
    q, c = cell["q"], cell["c"]
    data_id = stable_hash({"d": cell["d"], "n": cell["n"],
                           "degree": cell["degree"], "seed": seed})
    prior_id = stable_hash({"data_id": data_id, "q": q, "c": c})
    instance_id = stable_hash({**cell, "seed": seed})
    raw_dir = out / "raw" / cell["experiment_id"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_path = raw_dir / f"instance_{instance_id}_{prior_id}.csv"
    expected = set(cell["methods"])
    if final_path.exists():
        try:
            if expected <= set(pd.read_csv(final_path)["method"]):
                return final_path
        except Exception:
            pass

    X, truth, all_pairs = generate(
        seed, d=cell["d"], n=cell["n"], graph_type=f"er{cell['degree']}",
        scm="linear", noise="gaussian")
    clean_pairs = nested_pairs(all_pairs, q, seed)
    truth_bic = float(gaussian_bic(X, truth, lambda_bic=2.0)[0])
    try:
        pairs = corrupt_knowledge(clean_pairs, all_pairs, cell["d"], c, seed)
    except ValueError as exc:
        rows = []
        metric_columns = (
            "SHD_cpdag", "SHD_cpdag_normalized", "F1_skel",
            "precision_skel", "recall_skel", "skeleton_SHD",
            "skeleton_TP", "skeleton_FP", "skeleton_FN", "directed_SHD",
            "directed_TP", "directed_FP", "directed_FN", "F1_directed",
            "precision_directed", "recall_directed", "true_edges",
            "candidate_edges", "final_edges", "dag_feasible",
            "violations_before", "violations_after", "full_true_notreks_violations",
            "candidate_runtime", "candidate_bic", "final_bic", "source_count",
        )
        for method in cell["methods"]:
            invalid_row = {column: np.nan for column in metric_columns}
            invalid_row.update({"method": method, "run_id": stable_hash({
                "instance_id": instance_id, "prior_id": prior_id,
                "method": method}), "experiment_id": cell["experiment_id"],
                "instance_id": instance_id, "data_id": data_id,
                "prior_id": prior_id, "seed": seed, "d": cell["d"],
                "n": cell["n"], "er_degree": cell["degree"],
                "scm": "linear", "noise": "gaussian",
                "knowledge_fraction": q, "corruption_fraction": c,
                "supplied_pairs": np.nan,
                "full_true_notreks_pairs": len(all_pairs),
                "true_retained_pairs": np.nan,
                "false_supplied_pairs": np.nan,
                "chromatic_upper": np.nan,
                "full_chromatic_upper": chromatic_upper_bound(
                    cell["d"], all_pairs), "chi_lower": np.nan,
                "truth_bic": truth_bic, "bic_gap_to_truth": np.nan,
                "max_feasible_corruption_fraction": (
                    (cell["d"] * (cell["d"] - 1) / 2 - len(all_pairs)) /
                    max(1, len(clean_pairs))),
                "chi_exact": False, "solver_status": "invalid_prior",
                "exception": repr(exc), "attempts_requested": attempts,
                "attempts_completed": 0})
            rows.append(invalid_row)
        tmp = final_path.with_suffix(".tmp.csv")
        pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(final_path)
        print(f"invalid prior {final_path}: {exc}", file=sys.stderr)
        return final_path
    true_pair_set = {tuple(sorted((int(left), int(right))))
                     for left, right in all_pairs}
    clean_pair_set = {tuple(sorted((int(left), int(right))))
                      for left, right in clean_pairs}
    max_corruption = ((cell["d"] * (cell["d"] - 1) / 2 - len(all_pairs)) /
                      max(1, len(clean_pairs)))
    params = {
        "attempts": 2 if smoke else attempts,
        "flop_sweeps": 4 if smoke else 16,
        "dagma_stages": 2 if smoke else 5,
        "dagma_warm_iter": 100 if smoke else 30000,
        "dagma_max_iter": 200 if smoke else 60000,
        "trek_weight": 1.0,
    }
    rows = []
    for method in cell["methods"]:
        status = "ok"
        error = ""
        diag = {}
        try:
            cache_key = (data_id, method)
            if method == "flop-nt-post" and solver_cache is not None \
                    and (data_id, "flop") in solver_cache:
                candidate, diag, runtime = postselect_cached_flop(
                    X, pairs, solver_cache[(data_id, "flop")])
            elif method in {"flop", "dagma"} and solver_cache is not None \
                    and cache_key in solver_cache:
                candidate, diag, runtime = solver_cache[cache_key]
            else:
                candidate, diag, runtime = method_run(
                    method, X, pairs, seed, **params)
                if method in {"flop", "dagma"} and solver_cache is not None:
                    solver_cache[cache_key] = (candidate, diag, runtime)
            row = metrics(
                X, truth, pairs, method, diag["candidate_graph"], candidate,
                diag["cpdag"], runtime, params["attempts"],
                diag["optimizer_restarts"])
            row["base_solver_runtime"] = diag.get(
                "base_solver_runtime", np.nan)
            row["postselection_runtime"] = (
                runtime if method == "flop-nt-post" else np.nan)
        except Exception as exc:  # keep failed jobs visible and resumable
            status, error = "failed", repr(exc)
            row = {"method": method, "candidate_runtime": np.nan,
                   "attempts_requested": params["attempts"],
                   "attempts_completed": 0,
                   "SHD_cpdag": np.nan, "F1_skel": np.nan,
                   "F1_directed": np.nan, "violations_after": np.nan}
        row.update({
            "run_id": stable_hash({"instance_id": instance_id,
                                    "prior_id": prior_id, "method": method}),
            "experiment_id": cell["experiment_id"],
            "instance_id": instance_id, "data_id": data_id,
            "prior_id": prior_id, "seed": seed,
            "d": cell["d"], "n": cell["n"], "er_degree": cell["degree"],
            "scm": "linear", "noise": "gaussian",
            "knowledge_fraction": q, "corruption_fraction": c,
            "true_edges": int(truth.sum()),
            "supplied_pairs": len(pairs),
            "full_true_notreks_pairs": len(all_pairs),
            "true_retained_pairs": sum(tuple(sorted(p)) in clean_pair_set
                                       for p in pairs),
            "false_supplied_pairs": sum(tuple(sorted(p)) not in true_pair_set
                                         for p in pairs),
            "chromatic_upper": chromatic_upper_bound(cell["d"], pairs),
            "full_chromatic_upper": chromatic_upper_bound(cell["d"], all_pairs),
            "chi_lower": clique_lower_bound(cell["d"], pairs),
            "chi_exact": False,
            "solver_status": status, "exception": error,
            "hybrid_winner": diag.get("hybrid_winner", ""),
            "hybrid_vanilla_feasible": diag.get(
                "hybrid_vanilla_feasible", np.nan),
            "hybrid_vanilla_bic": diag.get("hybrid_vanilla_bic", np.nan),
            "hybrid_constrained_bic": diag.get(
                "hybrid_constrained_bic", np.nan),
            **params,
            "flop_local_passes": 8, "flop_lambda_bic": 2.0,
            "job_started": job_started,
            "job_finished": datetime.now(timezone.utc).isoformat(),
            "max_feasible_corruption_fraction": max_corruption,
        })
        if status == "ok":
            row["SHD_cpdag_normalized"] = row["SHD_cpdag"] / cell["d"]
            row["truth_bic"] = truth_bic
            row["candidate_bic_gap_to_truth"] = (
                row["candidate_bic"] - truth_bic)
            row["bic_gap_to_truth"] = row["final_bic"] - truth_bic
            row["full_true_notreks_violations"] = int(
                common_ancestor_violations(candidate, all_pairs))
            row["source_count"] = int(
                np.sum(np.asarray(candidate).sum(axis=0) == 0))
        rows.append(row)
    frame = pd.DataFrame(rows)
    tmp = final_path.with_suffix(".tmp.csv")
    frame.to_csv(tmp, index=False)
    tmp.replace(final_path)
    return final_path


def run_group(root: Path, out: Path, group: list[tuple[dict, int]],
              attempts: int, smoke: bool) -> list[Path]:
    """Run all priors for one data instance in one worker.

    Grouping preserves the important cache invariant: vanilla FLOP and DAGMA
    are fitted once for a data instance and reused across q/c conditions.
    """
    cache = {}
    return [run_job(root, out, cell, seed, attempts, smoke, cache)
            for cell, seed in group]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", choices=EXPERIMENTS)
    parser.add_argument("--seed-range", type=parse_range)
    parser.add_argument("--output-root", type=Path, default=Path("results/notreks_production"))
    parser.add_argument("--attempts", type=int, default=5)
    default_workers = max(1, min(2, (os.cpu_count() or 2) // 2))
    parser.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("NOTREKS_WORKERS", default_workers)),
        help="parallel data-instance workers; each worker is limited to one thread")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-experiments", action="store_true")
    args = parser.parse_args()
    if args.list_experiments:
        print("\n".join(EXPERIMENTS)); return
    if not args.experiment or not args.seed_range:
        parser.error("provide --experiment and inclusive --seed-range START:END")
    if "real-data" in args.experiment or "exact-reference" in args.experiment:
        raise SystemExit("real-data and exact-reference require separate configured backends")
    root = Path(__file__).resolve().parents[1]
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"experiments": args.experiment, "seeds": args.seed_range,
                "attempts": args.attempts, "smoke": args.smoke,
                "git": git_metadata(root), "python": sys.version,
                "hostname": platform.node(),
                "thread_env": {k: os.environ.get(k) for k in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception:
        freeze = "unavailable"
    manifest["environment_hash"] = hashlib.sha256(freeze.encode()).hexdigest()
    manifest["python_executable"] = sys.executable
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [(cell, seed) for experiment in args.experiment
            for cell in cells(experiment) for seed in args.seed_range]
    planned_rows = []
    for cell, seed in jobs:
        planned_rows.append({
            "experiment_id": cell["experiment_id"], "seed": seed,
            "d": cell["d"], "n": cell["n"], "er_degree": cell["degree"],
            "knowledge_fraction": cell["q"],
            "corruption_fraction": cell["c"],
            "expected_methods": ",".join(cell["methods"]),
            "expected_fit_count": len(cell["methods"]),
        })
    pd.DataFrame(planned_rows).to_csv(out / "planned_jobs.csv", index=False)
    expected = {}
    for row in planned_rows:
        expected[row["experiment_id"]] = expected.get(
            row["experiment_id"], 0) + row["expected_fit_count"]
    (out / "expected_fit_counts.json").write_text(
        json.dumps(expected, indent=2) + "\n")
    print(f"planned atomic jobs: {len(jobs)}")
    if args.dry_run:
        for cell, seed in jobs:
            print(cell["experiment_id"], cell["d"], cell["n"],
                  cell["degree"], cell["q"], cell["c"], seed)
        return
    if args.workers < 1:
        parser.error("--workers must be positive")
    grouped = {}
    for cell, seed in jobs:
        key = (cell["experiment_id"], cell["d"], cell["n"],
               cell["degree"], seed)
        grouped.setdefault(key, []).append((cell, seed))
    groups = list(grouped.values())
    hash_rows = []
    run_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_group, root, out, group, args.attempts,
                                args.smoke) for group in groups]
        for index, future in enumerate(as_completed(futures), 1):
            paths = future.result()
            for path in paths:
                hash_rows.append({"file": str(path), "sha256": file_hash(path)})
            pd.DataFrame(hash_rows).to_csv(out / "result_hashes.csv", index=False)
            elapsed = time.perf_counter() - run_started
            rate = elapsed / index
            eta = rate * (len(groups) - index)
            print(
                f"[{index}/{len(groups)}] data-instance group completed; "
                f"elapsed={elapsed/3600:.2f}h; ETA={eta/3600:.2f}h; "
                f"workers={args.workers}", flush=True)
    subprocess.run([
        sys.executable, str(root / "scripts/notreks_analysis.py"),
        "--input-root", str(out), "--output-dir", str(out / "analysis"),
    ], check=True)


if __name__ == "__main__":
    main()
