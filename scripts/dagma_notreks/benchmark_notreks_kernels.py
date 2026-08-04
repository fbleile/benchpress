#!/usr/bin/env python3
"""Benchmark NOTREKS matrix-function kernels and optimizer integration."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import random
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow.rules.structure_learning_algorithms.dagma.shared import (  # noqa: E402
    SharedDagmaLinear,
    notreks_value_grad,
)
from workflow.rules.structure_learning_algorithms.notreks import (  # noqa: E402
    KERNEL_REGISTRY,
    make_notreks_kernel,
)


KERNEL_FUNCTIONS = {
    "notreks_reference_inv": ("notreks_reference", "inv", "same_function"),
    "dense_inv": ("dense_inv", "inv", "same_function"),
    "selected_inv": ("selected_inv", "inv", "same_function"),
    "dense_exp": ("dense_exp", "exp", "pstrek_family"),
    "poly_selected_walk": ("poly_selected_walk", "walk", "pstrek_family"),
    "poly_selected_exp": ("poly_selected_exp", "poly_exp", "pstrek_family"),
}


def repository_state() -> dict[str, str]:
    def run(cmd: list[str]) -> str:
        proc = subprocess.run(
            cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
        return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    return {
        "pwd": str(ROOT),
        "git_status_short": run(["git", "status", "--short"]),
        "git_branch": run(["git", "branch", "--show-current"]),
        "git_head": run(["git", "rev-parse", "HEAD"]),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def synthetic_w(d: int, seed: int, regime: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if regime == "zero":
        W = np.zeros((d, d))
    elif regime == "sparse_small":
        W = rng.normal(scale=0.04, size=(d, d)) * (rng.random((d, d)) < min(3 / d, 1))
    elif regime == "sparse_moderate":
        W = rng.normal(scale=0.08, size=(d, d)) * (rng.random((d, d)) < min(4 / d, 1))
    elif regime == "dense_small":
        W = rng.normal(scale=0.025, size=(d, d))
    elif regime == "cyclic":
        W = np.zeros((d, d))
        for i in range(d):
            W[i, (i + 1) % d] = 0.12
    elif regime == "near_boundary":
        W = np.zeros((d, d))
        for i in range(d):
            W[i, (i + 1) % d] = 0.98
        rho = max(abs(np.linalg.eigvals(W * W)), default=0.0)
        if rho > 0:
            W *= np.sqrt(0.97 / rho)
    else:
        raise ValueError(f"unknown matrix regime {regime}")
    np.fill_diagonal(W, 0.0)
    return W


def pair_set(d: int, seed: int, regime: str) -> list[tuple[int, int]]:
    all_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    if regime == "one":
        return all_pairs[:1]
    if regime == "five":
        return all_pairs[:5]
    if regime == "ten":
        return all_pairs[:10]
    rng = np.random.default_rng(seed)
    if regime == "one_percent":
        k = max(1, int(round(0.01 * len(all_pairs))))
    elif regime == "five_percent":
        k = max(1, int(round(0.05 * len(all_pairs))))
    elif regime == "all":
        return all_pairs
    else:
        raise ValueError(f"unknown pair regime {regime}")
    idx = rng.choice(len(all_pairs), size=k, replace=False)
    return [all_pairs[int(i)] for i in sorted(idx)]


def finite_difference_direction(
    fn: Callable[[np.ndarray], float], W: np.ndarray, direction: np.ndarray,
    eps: float,
) -> float:
    return float((fn(W + eps * direction) - fn(W - eps * direction)) / (2 * eps))


def run_correctness(args, out: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(2026)
    for d in args.dims:
        for pair_regime in args.pair_regimes:
            pairs = pair_set(d, 700 + d, pair_regime)
            for matrix_regime in args.matrix_regimes:
                W = synthetic_w(d, 800 + d, matrix_regime)
                D = rng.normal(size=W.shape)
                np.fill_diagonal(D, 0.0)
                references = {
                    "inv": notreks_value_grad(W, pairs, "inv", inverse_epsilon=args.inverse_epsilon),
                }
                for label, (kernel_name, function, family_class) in KERNEL_FUNCTIONS.items():
                    try:
                        kernel = make_notreks_kernel(kernel_name, pairs, d)
                        result = kernel.value_and_grad(
                            W, function=function, log_terms=args.poly_degree or d - 1,
                            inverse_epsilon=args.inverse_epsilon)
                        direction_errors = []
                        for eps in (1e-4, 1e-5, 1e-6):
                            fd = finite_difference_direction(
                                lambda Z: kernel.value_grad(
                                    Z, function, log_terms=args.poly_degree or d - 1,
                                    inverse_epsilon=args.inverse_epsilon)[0],
                                W, D, eps)
                            ad = float(np.sum(result.gradient_W * D))
                            direction_errors.append(abs(fd - ad) / max(1.0, abs(fd), abs(ad)))
                        value_error = np.nan
                        grad_error = np.nan
                        if function in references:
                            rv, rg = references[function]
                            value_error = abs(result.penalty_value - rv) / max(1.0, abs(rv))
                            grad_error = float(np.linalg.norm(result.gradient_W - rg) / max(1.0, np.linalg.norm(rg)))
                        rows.append({
                            "kernel_label": label,
                            "kernel_name": kernel_name,
                            "function": function,
                            "family_class": family_class,
                            "d": d,
                            "pair_regime": pair_regime,
                            "num_pairs": len(pairs),
                            "unique_nodes": len(set(x for p in pairs for x in p)),
                            "matrix_regime": matrix_regime,
                            "status": "ok",
                            "value": result.penalty_value,
                            "relative_value_error_same_function": value_error,
                            "relative_gradient_error_same_function": grad_error,
                            "max_directional_derivative_relative_error": max(direction_errors),
                            **asdict(result.diagnostics),
                        })
                    except Exception as exc:  # noqa: BLE001 - benchmark records failures
                        rows.append({
                            "kernel_label": label,
                            "kernel_name": kernel_name,
                            "function": function,
                            "family_class": family_class,
                            "d": d,
                            "pair_regime": pair_regime,
                            "num_pairs": len(pairs),
                            "unique_nodes": len(set(x for p in pairs for x in p)),
                            "matrix_regime": matrix_regime,
                            "status": "failed",
                            "failure_type": type(exc).__name__,
                            "failure_message": str(exc),
                        })
    write_csv(out / "correctness_results.csv", rows)
    write_csv(out / "directional_gradient_results.csv", rows)
    return rows


def timed_call(fn: Callable[[], object]) -> tuple[object, float, int]:
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, int(peak)


def run_microbenchmarks(args, out: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cases = []
    for d in args.dims:
        for pair_regime in args.pair_regimes:
            for matrix_regime in args.matrix_regimes:
                cases.append((d, pair_regime, matrix_regime))
    rng = random.Random(991)
    for repeat in range(args.repeats):
        rng.shuffle(cases)
        for d, pair_regime, matrix_regime in cases:
            pairs = pair_set(d, 900 + d, pair_regime)
            W = synthetic_w(d, 1000 + d + repeat, matrix_regime)
            labels = list(KERNEL_FUNCTIONS)
            rng.shuffle(labels)
            for label in labels:
                kernel_name, function, family_class = KERNEL_FUNCTIONS[label]
                kernel = make_notreks_kernel(kernel_name, pairs, d)
                for _ in range(args.warmups):
                    try:
                        kernel.value_grad(
                            W, function, log_terms=args.poly_degree or d - 1,
                            inverse_epsilon=args.inverse_epsilon)
                    except Exception:
                        break
                try:
                    result, elapsed, peak = timed_call(
                        lambda: kernel.value_and_grad(
                            W, function=function, log_terms=args.poly_degree or d - 1,
                            inverse_epsilon=args.inverse_epsilon))
                    rows.append({
                        "repeat": repeat,
                        "kernel_label": label,
                        "kernel_name": kernel_name,
                        "function": function,
                        "family_class": family_class,
                        "d": d,
                        "pair_regime": pair_regime,
                        "num_pairs": len(pairs),
                        "unique_nodes": len(set(x for p in pairs for x in p)),
                        "matrix_regime": matrix_regime,
                        "status": "ok",
                        "combined_value_grad_seconds": elapsed,
                        "peak_memory_bytes": peak,
                        **asdict(result.diagnostics),
                    })
                except Exception as exc:  # noqa: BLE001
                    rows.append({
                        "repeat": repeat,
                        "kernel_label": label,
                        "kernel_name": kernel_name,
                        "function": function,
                        "family_class": family_class,
                        "d": d,
                        "pair_regime": pair_regime,
                        "num_pairs": len(pairs),
                        "unique_nodes": len(set(x for p in pairs for x in p)),
                        "matrix_regime": matrix_regime,
                        "status": "failed",
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    })
    write_csv(out / "microbenchmark_results.csv", rows)
    write_csv(out / "memory_results.csv", rows)
    return rows


def run_stability(args, out: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scales = np.logspace(-2, 1, 13)
    for d in args.dims:
        base = synthetic_w(d, 3100 + d, "sparse_moderate")
        pairs = pair_set(d, 3200 + d, "ten")
        for label, (kernel_name, function, family_class) in KERNEL_FUNCTIONS.items():
            kernel = make_notreks_kernel(kernel_name, pairs, d)
            for c in scales:
                W = c * base
                try:
                    result = kernel.value_and_grad(
                        W, function=function, log_terms=args.poly_degree or d - 1,
                        inverse_epsilon=args.inverse_epsilon)
                    rows.append({
                        "kernel_label": label,
                        "kernel_name": kernel_name,
                        "function": function,
                        "family_class": family_class,
                        "d": d,
                        "scale_c": c,
                        "status": "ok",
                        "value": result.penalty_value,
                        "gradient_frobenius_norm": float(np.linalg.norm(result.gradient_W)),
                        **asdict(result.diagnostics),
                    })
                except Exception as exc:  # noqa: BLE001
                    rows.append({
                        "kernel_label": label,
                        "kernel_name": kernel_name,
                        "function": function,
                        "family_class": family_class,
                        "d": d,
                        "scale_c": c,
                        "status": "failed",
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    })
    write_csv(out / "stability_results.csv", rows)
    return rows


def simulate_linear_gaussian(d: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    A = np.zeros((d, d), dtype=int)
    p = min(2.0 / max(d - 1, 1), 1.0)
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < p:
                A[order[a], order[b]] = 1
    Wtrue = A * rng.uniform(0.5, 1.0, size=(d, d)) * rng.choice([-1, 1], size=(d, d))
    noise = rng.normal(size=(n, d))
    X = noise @ np.linalg.inv(np.eye(d) - Wtrue)
    std = X.std(axis=0, ddof=0)
    if np.any(std <= 0) or not np.all(np.isfinite(std)):
        raise ValueError("synthetic data cannot be standardized")
    X = (X - X.mean(axis=0)) / std
    assert np.max(np.abs(X.mean(axis=0))) < 1e-12
    assert np.max(np.abs(X.std(axis=0, ddof=0) - 1.0)) < 1e-12
    return X, A


def no_trek_pairs_from_adjacency(A: np.ndarray, max_pairs: int, seed: int) -> list[tuple[int, int]]:
    d = A.shape[0]
    reach = A.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(d):
        reach |= reach[:, [k]] & reach[[k], :]
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            if not np.any(reach[:, i] & reach[:, j]):
                pairs.append((i, j))
    if len(pairs) <= max_pairs:
        return pairs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=max_pairs, replace=False)
    return [pairs[int(i)] for i in sorted(idx)]


def run_end_to_end(args, out: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    kernels = args.end_to_end_kernels
    for d in args.end_to_end_dims:
        for seed in args.end_to_end_seeds:
            X, true_A = simulate_linear_gaussian(d, args.n, seed)
            pair_sets = {
                "one_correct": no_trek_pairs_from_adjacency(true_A, 1, seed),
                "several_correct": no_trek_pairs_from_adjacency(true_A, 5, seed),
                "one_percent": no_trek_pairs_from_adjacency(true_A, max(1, int(0.01 * d * (d - 1) / 2)), seed),
                "five_percent": no_trek_pairs_from_adjacency(true_A, max(1, int(0.05 * d * (d - 1) / 2)), seed),
            }
            for pair_regime, pairs in pair_sets.items():
                for kernel_label in kernels:
                    kernel_name, function, family_class = KERNEL_FUNCTIONS[kernel_label]
                    try:
                        model = SharedDagmaLinear("l2")
                        started = time.perf_counter()
                        W = model.fit(
                            X.copy(), no_trek_pairs=pairs, trek_weight=args.trek_weight,
                            trek_function=function, trek_kernel=kernel_name,
                            trek_log_terms=args.poly_degree or d - 1,
                            trek_inverse_epsilon=args.inverse_epsilon,
                            T=args.T, warm_iter=args.warm_iter,
                            max_iter=args.max_iter, checkpoint=args.checkpoint,
                            lambda1=args.lambda1, w_threshold=args.w_threshold,
                            lr=args.lr)
                        elapsed = time.perf_counter() - started
                        pred = (np.abs(W) >= args.w_threshold).astype(int)
                        np.fill_diagonal(pred, 0)
                        shd = int(np.sum(pred != true_A))
                        true_skel = ((true_A + true_A.T) > 0)
                        pred_skel = ((pred + pred.T) > 0)
                        upper = np.triu(np.ones_like(true_skel, dtype=bool), 1)
                        tp = int(np.sum(true_skel[upper] & pred_skel[upper]))
                        fp = int(np.sum(~true_skel[upper] & pred_skel[upper]))
                        fn = int(np.sum(true_skel[upper] & ~pred_skel[upper]))
                        precision = tp / (tp + fp) if tp + fp else 0.0
                        recall = tp / (tp + fn) if tp + fn else 0.0
                        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                        final_diag = model.stage_diagnostics[-1] if model.stage_diagnostics else {}
                        rows.append({
                            "kernel_label": kernel_label,
                            "kernel_name": kernel_name,
                            "function": function,
                            "family_class": family_class,
                            "d": d,
                            "n": args.n,
                            "data_standardized": True,
                            "seed": seed,
                            "pair_regime": pair_regime,
                            "num_pairs": len(pairs),
                            "status": "ok",
                            "optimizer_time_seconds": elapsed,
                            "notreks_kernel_time_seconds": sum(
                                x.get("notreks_value_gradient_seconds", 0.0)
                                for x in model.stage_diagnostics),
                            "dagma_kernel_time_seconds": sum(
                                x.get("dagma_h_gradient_seconds", 0.0)
                                for x in model.stage_diagnostics),
                            "iterations": sum(
                                x.get("iterations_performed", 0)
                                for x in model.stage_diagnostics),
                            "termination_reason": final_diag.get("termination_reason", "unknown"),
                            "final_dag_penalty": final_diag.get("h", np.nan),
                            "final_notreks_penalty": final_diag.get("raw_notreks_value", np.nan),
                            "directed_adjacency_hamming": shd,
                            "skeleton_F1_python_diagnostic": f1,
                            "predicted_edge_count": int(pred.sum()),
                        })
                    except Exception as exc:  # noqa: BLE001
                        rows.append({
                            "kernel_label": kernel_label,
                            "kernel_name": kernel_name,
                            "function": function,
                            "family_class": family_class,
                            "d": d,
                            "n": args.n,
                            "seed": seed,
                            "pair_regime": pair_regime,
                            "num_pairs": len(pairs),
                            "status": "failed",
                            "failure_type": type(exc).__name__,
                            "failure_message": str(exc),
                        })
    write_csv(out / "end_to_end_results.csv", rows)
    return rows


def summarize(out: Path) -> None:
    lines = ["# DAGMA-NOTREKS Kernel Benchmark Recommendation", ""]
    micro_path = out / "microbenchmark_results.csv"
    if micro_path.exists():
        micro = pd.read_csv(micro_path)
        ok = micro[micro.status == "ok"].copy()
        if len(ok):
            summary = (ok.groupby(["kernel_label", "d", "pair_regime"])
                       ["combined_value_grad_seconds"].median().reset_index())
            write_csv(out / "pareto_frontier.csv", summary.to_dict("records"))
            lines.append("## Microbenchmark Summary")
            lines.append(summary.to_markdown(index=False))
            lines.append("")
    correctness_path = out / "correctness_results.csv"
    if correctness_path.exists():
        corr = pd.read_csv(correctness_path)
        failures = corr[corr.status != "ok"]
        lines.append("## Correctness Failures")
        lines.append("None" if failures.empty else failures.to_markdown(index=False))
        lines.append("")
    e2e_path = out / "end_to_end_results.csv"
    if e2e_path.exists():
        e2e = pd.read_csv(e2e_path)
        ok = e2e[e2e.status == "ok"]
        if len(ok):
            e2e_summary = (ok.groupby(["kernel_label", "d", "pair_regime"])
                           .agg(optimizer_time_seconds=("optimizer_time_seconds", "median"),
                                skeleton_F1_python_diagnostic=("skeleton_F1_python_diagnostic", "median"),
                                directed_adjacency_hamming=("directed_adjacency_hamming", "median"))
                           .reset_index())
            lines.append("## End-to-End Summary")
            lines.append(e2e_summary.to_markdown(index=False))
            lines.append("")
    lines.extend([
        "## Interpretation Guardrails",
        "Dense/selected inverse rows are implementation-equivalent only for trek_function=inv.",
        "Polynomial and exponential rows are different PSTrek families and define different optimization landscapes.",
        "The production default should change only after correctness, gradient, stability and end-to-end results pass together.",
    ])
    (out / "FINAL_RECOMMENDATION.md").write_text("\n".join(lines) + "\n")


def implementation_audit(out: Path) -> None:
    text = """# Implementation Audit

Entry points inspected:

- `workflow/rules/structure_learning_algorithms/dagma_notreks/run.py`
- `workflow/rules/structure_learning_algorithms/dagma/shared.py`
- `workflow/rules/structure_learning_algorithms/dagma/inverse_structural.py`
- `workflow/rules/structure_learning_algorithms/dagma_notreks/schema.json`
- `workflow/rules/structure_learning_algorithms/dagma_notreks/tests/test_analytic_kernels.py`
- `workflow/rules/structure_learning_algorithms/dagma_notreks/tests/test_oracle_notreks.py`

Current reference kernel:

- `workflow.rules.structure_learning_algorithms.dagma.shared.NoTreksKernel`
- `notreks_value_grad(W, pairs, function, log_terms=None, inverse_epsilon=1e-8)`

The existing NOTREKS contract canonicalizes unordered pairs, rejects duplicates,
uses `X = W * W`, applies scale `2 / (d - 1)`, and returns gradients as
`2 * W * grad_X`.  The default Snakemake method uses `trek_function="inv"`.

Existing function families:

- `exp`: SciPy dense matrix exponential with `scipy.linalg.expm_frechet` gradient.
- `inv`: dense solve of `(I - X + epsilon I) F = I` with analytic gradient.
- `log`: finite series `I + sum_{k=1}^K X^k / k` with power-adjoint gradient.
- `binom`: `(I + X)^d` with power-adjoint gradient.

Optimizer interaction:

- `SharedDagmaLinear.minimize` adds `trek_weight * G_nt` to the DAGMA gradient.
- The fixed NOTREKS contribution is outside the DAGMA `mu` factor in the current repository convention.
- Component timing is recorded when `profile_components` is enabled through existing stage diagnostics.

The production default is `fast` (selected-column inverse). The recoverable
`notreks_reference` kernel and all named ablations remain selectable through
`trek_kernel`.
"""
    (out / "implementation_audit.md").write_text(text)


def kernel_registry_table(out: Path) -> None:
    rows = []
    for label, (kernel_name, function, family_class) in KERNEL_FUNCTIONS.items():
        rows.append({
            "kernel_label": label,
            "kernel_name": kernel_name,
            "function": function,
            "family_class": family_class,
            "registered_class": KERNEL_REGISTRY[kernel_name].__name__,
            "implementation_equivalent_to_reference": bool(family_class == "same_function"),
        })
    write_csv(out / "kernel_registry.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/dagma_notreks_kernel_benchmark")
    parser.add_argument("--sections", nargs="+", default=["audit", "correctness", "micro", "stability"],
                        choices=["audit", "correctness", "micro", "stability", "end-to-end", "all"])
    parser.add_argument("--dims", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--pair-regimes", nargs="+", default=["one", "five", "ten", "one_percent", "five_percent", "all"])
    parser.add_argument("--matrix-regimes", nargs="+", default=["zero", "sparse_small", "sparse_moderate", "dense_small", "cyclic", "near_boundary"])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--poly-degree", type=int, default=None)
    parser.add_argument("--inverse-epsilon", type=float, default=1e-8)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--end-to-end-dims", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--end-to-end-seeds", nargs="+", type=int, default=[1001, 1002, 1003])
    parser.add_argument("--end-to-end-kernels", nargs="+", default=["notreks_reference_inv", "selected_inv", "dense_exp", "poly_selected_walk"])
    parser.add_argument("--trek-weight", type=float, default=10.0)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--warm-iter", type=int, default=30000)
    parser.add_argument("--max-iter", type=int, default=60000)
    parser.add_argument("--checkpoint", type=int, default=1000)
    parser.add_argument("--lambda1", type=float, default=0.03)
    parser.add_argument("--w-threshold", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.0003)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sections = set(args.sections)
    if "all" in sections:
        sections = {"audit", "correctness", "micro", "stability", "end-to-end"}
    write_json(out / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "blas_threads": {name: os.environ.get(name) for name in [
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]},
    })
    state = repository_state()
    write_json(out / "repository_state.json", state)
    (out / "repository_state.txt").write_text("\n".join(
        f"{key}: {value}" for key, value in state.items()) + "\n")
    kernel_registry_table(out)
    if "audit" in sections:
        implementation_audit(out)
    if "correctness" in sections:
        run_correctness(args, out)
    if "micro" in sections:
        run_microbenchmarks(args, out)
    if "stability" in sections:
        run_stability(args, out)
    if "end-to-end" in sections:
        run_end_to_end(args, out)
    # Create empty compatibility/dispatch placeholders with documented schema.
    for name in ("dispatch_results.csv", "failures.csv"):
        path = out / name
        if not path.exists():
            write_csv(path, [])
    summarize(out)
    print(json.dumps({
        "output_dir": str(out),
        "sections_completed": sorted(sections),
        "kernel_count": len(KERNEL_FUNCTIONS),
    }, indent=2))


if __name__ == "__main__":
    main()
