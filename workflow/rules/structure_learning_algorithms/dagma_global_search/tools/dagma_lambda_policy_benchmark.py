"""Paired vanilla-DAGMA lambda-policy benchmark.

This runner deliberately has no NOTREKS path.  Each dataset is generated once,
one short pilot is cached for P3--P5, and all policies use the same production
postprocessing and ordinary refitted Gaussian BIC.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.end_flop_prune import (
    end_flop_prune,
)
from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    ProductionConfig,
    run_production_pipeline,
    standardize_training_data,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import (
    generate,
)


POLICIES = (
    "dagma_lambda_fixed_003",
    "dagma_lambda_short_grid_bic",
    "dagma_lambda_wide_short_grid_bic",
    "dagma_lambda_adaptive_grid_bic",
    "dagma_lambda_grid_bic",
    "dagma_lambda_bic_unit",
    "dagma_lambda_pilot_column_bic",
    "dagma_lambda_pilot_score_bic",
    "dagma_lambda_pilot_score_ebic",
    "dagma_lambda_pilot_score_bic_pruned",
)
PILOT_POLICIES = {
    "dagma_lambda_pilot_column_bic",
    "dagma_lambda_pilot_score_bic",
    "dagma_lambda_pilot_score_ebic",
    "dagma_lambda_pilot_score_bic_pruned",
}


def lambda_max(X: np.ndarray) -> float:
    X, _, _ = standardize_training_data(X, ddof=0, std_floor=1e-12)
    covariance = X.T @ X / len(X)
    np.fill_diagonal(covariance, 0.0)
    return float(np.max(np.abs(covariance), initial=0.0))


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                    suffix=".tmp", newline="") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fit(X, penalty, args, *, stages=None, warm_iter=None, max_iter=None):
    config = ProductionConfig(
        restarts=args.restarts,
        seed=args.seed,
        T=args.dagma_stages if stages is None else stages,
        warm_iter=args.dagma_warm_iter if warm_iter is None else warm_iter,
        max_iter=args.dagma_max_iter if max_iter is None else max_iter,
        lambda1=(float(penalty) if np.asarray(penalty).ndim == 0 else .03),
        lambda1_penalty=(None if np.asarray(penalty).ndim == 0
                         else np.asarray(penalty, dtype=float)),
        mu_schedule=tuple(args.mu_schedule),
        s=tuple(args.s_schedule),
        notreks_constraint_active=False,
        dagma_postselection_policy="feasible_parent_shrink",
    )
    started = time.perf_counter()
    selected, restarts = run_production_pipeline(X, [], config)
    elapsed = time.perf_counter() - started
    return selected, restarts, elapsed


def _pilot(X, args):
    budget = max(1000, int(.2 * max(args.dagma_warm_iter,
                                  args.dagma_max_iter)))
    pilot_args = argparse.Namespace(**vars(args))
    pilot_args.restarts = 1
    pilot_args.seed = args.seed
    pilot, _, runtime = _fit(
        X, np.sqrt(np.log(len(X)) / len(X)), pilot_args,
        stages=1, warm_iter=budget, max_iter=budget)
    Xs, _, _ = standardize_training_data(X, ddof=0, std_floor=1e-12)
    _, beta = gaussian_bic(Xs, pilot.adjacency, lambda_bic=2.0)
    residuals = Xs.copy()
    for child in range(Xs.shape[1]):
        parents = np.flatnonzero(pilot.adjacency[:, child])
        if len(parents):
            residuals[:, child] -= Xs[:, parents] @ beta[parents, child]
    n, d = Xs.shape
    scale_floor = np.finfo(float).eps * max(1.0, float(np.std(Xs)))
    sigma = np.sqrt(np.maximum(
        np.sum(residuals * residuals, axis=0)
        / np.maximum(n - pilot.adjacency.sum(axis=0) - 1, 1),
        scale_floor**2))
    score_scale = np.zeros((d, d), dtype=float)
    for parent in range(d):
        for child in range(d):
            if parent != child:
                z = Xs[:, parent] * residuals[:, child]
                score_scale[parent, child] = max(float(np.std(z, ddof=0)),
                                                 scale_floor)
    return {
        "sigma": sigma,
        "score_scale": score_scale,
        "runtime": runtime,
        "pilot_edges": int(pilot.adjacency.sum()),
        "pilot_lambda": float(np.sqrt(np.log(n) / n)),
    }


def _summary(X, selected, runtime, policy, pilot, lam, lam_max_value,
             *, pruning_runtime=0.0, source_policy=None):
    bic, coefficients = gaussian_bic(X, selected.adjacency, lambda_bic=2.0)
    values = np.asarray(lam, dtype=float)
    if values.ndim == 0:
        values = np.full((X.shape[1], X.shape[1]), float(values))
    elif values.ndim == 1:
        values = np.broadcast_to(values[None, :],
                                 (X.shape[1], X.shape[1]))
    values = values[~np.eye(X.shape[1], dtype=bool)]
    return {
        "method": policy,
        "lambda_policy": policy,
        "d": X.shape[1], "n": len(X), "restarts": selected.restart + 1,
        "edges": int(selected.adjacency.sum()),
        "bic": float(bic),
        "bic_likelihood_term": float(bic - selected.adjacency.sum() * np.log(len(X))),
        "bic_complexity_term": float(selected.adjacency.sum() * np.log(len(X))),
        "runtime": float(runtime), "pilot_runtime": float(pilot["runtime"] if pilot else 0.0),
        "pruning_runtime": float(pruning_runtime),
        "lambda_min": float(values.min()),
        "lambda_q25": float(np.quantile(values, .25)),
        "lambda_median": float(np.median(values)),
        "lambda_q75": float(np.quantile(values, .75)),
        "lambda_max": float(values.max()),
        "lambda_max_X": float(lam_max_value),
        "lambda_over_lambda_max_median": float(np.median(values) / lam_max_value)
            if lam_max_value > 0 else np.nan,
        "pilot_sigma_median": float(np.median(pilot["sigma"])) if pilot else np.nan,
        "pilot_score_scale_median": float(np.median(pilot["score_scale"])) if pilot else np.nan,
        "pilot_edges": int(pilot["pilot_edges"]) if pilot else 0,
        "source_policy": source_policy or policy,
        "score_normalization": "standardized X; gaussian_bic refit",
    }


def run_policy(X, args, policy, pilot):
    n, d = X.shape
    lm = lambda_max(X)
    factor = np.sqrt(np.log(n) / n)
    if policy == "dagma_lambda_fixed_003":
        lam = .03
    elif policy == "dagma_lambda_short_grid_bic":
        candidates = []
        short_args = argparse.Namespace(**vars(args))
        short_args.warm_iter = max(1000, args.dagma_warm_iter // 5)
        short_args.max_iter = max(2000, args.dagma_max_iter // 5)
        for ratio in (.3, .03, .003):
            started = time.perf_counter()
            screened, _, _ = _fit(X, lm * ratio, short_args)
            screening_runtime = time.perf_counter() - started
            score = gaussian_bic(
                X, screened.adjacency, lambda_bic=2.0)[0]
            candidates.append((float(score), ratio, screening_runtime))
        _, ratio, screening_runtime = min(
            candidates, key=lambda item: (item[0], item[1]))
        selected, _, final_runtime = _fit(X, lm * ratio, args)
        row = _summary(X, selected, screening_runtime + final_runtime,
                       policy, pilot, lm * ratio, lm,
                       source_policy=f"short_grid:{ratio:g}")
        row.update({
            "short_grid_fit_count": 3,
            "short_grid_screening_runtime": float(screening_runtime),
            "short_grid_final_runtime": float(final_runtime),
            "short_grid_warm_iter": int(short_args.warm_iter),
            "short_grid_max_iter": int(short_args.max_iter),
        })
        return selected, row
    elif policy == "dagma_lambda_wide_short_grid_bic":
        candidates = []
        short_args = argparse.Namespace(**vars(args))
        fraction = float(getattr(args, "wide_screen_fraction", 0.10))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("wide_screen_fraction must lie in (0, 1]")
        short_args.warm_iter = max(100, int(args.dagma_warm_iter * fraction))
        short_args.max_iter = max(200, int(args.dagma_max_iter * fraction))
        ratios = (.3, .1, .03, .01, .003, .001, .0003)
        for ratio in ratios:
            started = time.perf_counter()
            screened, _, _ = _fit(X, lm * ratio, short_args)
            screening_runtime = time.perf_counter() - started
            score = gaussian_bic(
                X, screened.adjacency, lambda_bic=2.0)[0]
            candidates.append((float(score), ratio, screening_runtime))
        _, ratio, _ = min(candidates, key=lambda item: (item[0], item[1]))
        selected, _, final_runtime = _fit(X, lm * ratio, args)
        screening_runtime = float(sum(item[2] for item in candidates))
        row = _summary(X, selected, screening_runtime + final_runtime,
                       policy, pilot, lm * ratio, lm,
                       source_policy=f"wide_short_grid:{ratio:g}")
        row.update({
            "short_grid_fit_count": len(ratios),
            "short_grid_screening_runtime": screening_runtime,
            "short_grid_final_runtime": float(final_runtime),
            "short_grid_warm_iter": int(short_args.warm_iter),
            "short_grid_max_iter": int(short_args.max_iter),
            "short_grid_ratios": ",".join(str(value) for value in ratios),
            "short_grid_fraction": fraction,
        })
        return selected, row
    elif policy == "dagma_lambda_adaptive_grid_bic":
        ratios = (.3, .1, .03, .01, .003, .001, .0003)
        center = len(ratios) // 2
        screen_args = argparse.Namespace(**vars(args))
        fraction = float(getattr(args, "adaptive_screen_fraction", 0.05))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("adaptive_screen_fraction must lie in (0, 1]")
        screen_args.dagma_stages = int(
            getattr(args, "adaptive_screen_stages", 1))
        screen_args.warm_iter = max(100, int(args.dagma_warm_iter * fraction))
        screen_args.max_iter = max(200, int(args.dagma_max_iter * fraction))
        screen_args.mu_schedule = (args.mu_schedule[0],) * screen_args.dagma_stages
        screen_args.s_schedule = (args.s_schedule[0],) * screen_args.dagma_stages

        evaluated = {}

        def evaluate(index):
            if index not in evaluated:
                started = time.perf_counter()
                screened, _, _ = _fit(
                    X, lm * ratios[index], screen_args)
                runtime = time.perf_counter() - started
                evaluated[index] = (
                    float(gaussian_bic(
                        X, screened.adjacency, lambda_bic=2.0)[0]), runtime)
            return evaluated[index]

        best_index = center
        best_score, _ = evaluate(center)
        left = center - 1
        if left >= 0:
            left_score, _ = evaluate(left)
        else:
            left_score = np.inf
        if left_score < best_score:
            best_index, best_score = left, left_score
            while best_index - 1 >= 0:
                candidate = best_index - 1
                score, _ = evaluate(candidate)
                if score < best_score:
                    best_index, best_score = candidate, score
                else:
                    break
        else:
            right = center + 1
            if right < len(ratios):
                right_score, _ = evaluate(right)
            else:
                right_score = np.inf
            if right_score < best_score:
                best_index, best_score = right, right_score
                while best_index + 1 < len(ratios):
                    candidate = best_index + 1
                    score, _ = evaluate(candidate)
                    if score < best_score:
                        best_index, best_score = candidate, score
                    else:
                        break
        ratio = ratios[best_index]
        selected, _, final_runtime = _fit(X, lm * ratio, args)
        screening_runtime = float(sum(item[1] for item in evaluated.values()))
        row = _summary(X, selected, screening_runtime + final_runtime,
                       policy, pilot, lm * ratio, lm,
                       source_policy=f"adaptive_grid:{ratio:g}")
        row.update({
            "short_grid_fit_count": len(evaluated),
            "short_grid_screening_runtime": screening_runtime,
            "short_grid_final_runtime": float(final_runtime),
            "short_grid_warm_iter": int(screen_args.warm_iter),
            "short_grid_max_iter": int(screen_args.max_iter),
            "short_grid_ratios": ",".join(str(ratio) for ratio in ratios),
            "adaptive_screen_fraction": fraction,
            "adaptive_screen_stages": screen_args.dagma_stages,
            "adaptive_screen_path": ";".join(
                f"{ratios[index]}:{evaluated[index][0]:.6g}"
                for index in sorted(evaluated)),
        })
        return selected, row
    elif policy == "dagma_lambda_bic_unit":
        lam = factor
    elif policy == "dagma_lambda_pilot_column_bic":
        lam = pilot["sigma"] * factor
    elif policy == "dagma_lambda_pilot_score_bic":
        lam = pilot["score_scale"] * factor
    elif policy == "dagma_lambda_pilot_score_ebic":
        lam = pilot["score_scale"] * np.sqrt(
            (np.log(n) + 2 * np.log(max(d - 1, 1))) / n)
    elif policy == "dagma_lambda_grid_bic":
        candidates = []
        for ratio in (.3, .1, .03, .01, .003):
            selected, _, runtime = _fit(X, lm * ratio, args)
            score = gaussian_bic(X, selected.adjacency, lambda_bic=2.0)[0]
            candidates.append((float(score), ratio, selected, runtime))
        _, ratio, selected, _ = min(candidates, key=lambda item: (
            item[0], item[1]))
        runtime = float(sum(item[3] for item in candidates))
        return selected, _summary(X, selected, runtime, policy, pilot,
                                  lm * ratio, lm, source_policy=f"grid:{ratio:g}")
    elif policy == "dagma_lambda_pilot_score_bic_pruned":
        selected, p4 = run_policy(X, args, "dagma_lambda_pilot_score_bic", pilot)
        started = time.perf_counter()
        adjacency, _, _ = end_flop_prune(
            X, selected.adjacency, lambda_bic=2.0)
        selected.adjacency = adjacency
        pruning_runtime = time.perf_counter() - started
        bic, _ = gaussian_bic(X, adjacency, lambda_bic=2.0)
        row = dict(p4)
        row.update({"method": policy, "lambda_policy": policy,
                    "source_policy": "dagma_lambda_pilot_score_bic",
                    "edges": int(adjacency.sum()),
                    "bic": float(bic),
                    "bic_likelihood_term": float(
                        bic - adjacency.sum() * np.log(len(X))),
                    "bic_complexity_term": float(
                        adjacency.sum() * np.log(len(X))),
                    "pruning_runtime": pruning_runtime,
                    "runtime": p4["runtime"] + pruning_runtime})
        return selected, row
    else:
        raise ValueError(policy)
    selected, _, runtime = _fit(X, lam, args)
    return selected, _summary(X, selected, runtime, policy, pilot, lam, lm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--graph-types", nargs="+", default=["er2", "er4", "er8"])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[200, 500, 2000])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--methods", nargs="+", choices=["all", *POLICIES], default=["all"])
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--dagma-stages", type=int, default=5)
    parser.add_argument("--dagma-warm-iter", type=int, default=30000)
    parser.add_argument("--dagma-max-iter", type=int, default=60000)
    parser.add_argument("--wide-screen-fraction", type=float, default=0.10,
                        help="fraction of the normal budget per wide-grid screening fit")
    parser.add_argument("--adaptive-screen-fraction", type=float, default=0.05,
                        help="fraction of the normal budget per adaptive screening fit")
    parser.add_argument("--adaptive-screen-stages", type=int, default=1,
                        help="continuation stages used for adaptive screening")
    parser.add_argument("--dagma-mu-schedule", default="1,0.3,0.1,0.01,0.001")
    parser.add_argument("--dagma-s-schedule", default="1.1,1.0,0.9,0.8,0.7")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.mu_schedule = tuple(float(x) for x in args.dagma_mu_schedule.split(","))
    args.s_schedule = tuple(float(x) for x in args.dagma_s_schedule.split(","))
    policies = POLICIES if "all" in args.methods else args.methods
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "per_dataset.csv"
    existing = (pd.read_csv(result_path).to_dict("records")
                if result_path.exists() else [])
    rows = existing[:]
    done = {(r["graph_type"], int(r["n"]), int(r["seed"]), r["method"])
            for r in rows}
    for graph_type in args.graph_types:
        for n in args.sample_sizes:
            for seed in args.seeds:
                X, truth, _ = generate(seed, d=args.d, n=n,
                                        graph_type=graph_type)
                args.seed = seed
                pilot = None
                if any(p in policies for p in PILOT_POLICIES):
                    pilot = _pilot(X, args)
                for policy in policies:
                    key = (graph_type, n, seed, policy)
                    if key in done:
                        continue
                    selected, row = run_policy(X, args, policy, pilot)
                    row.update({"graph_type": graph_type, "n": n,
                                "seed": seed, "truth_edges": int(truth.sum()),
                                "data_model": "linear_gaussian_scm",
                                "notreks_used": False})
                    rows.append(row)
                    _atomic_csv(pd.DataFrame(rows), result_path)
                    print(pd.Series(row).to_string(), flush=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.groupby("method", as_index=False).agg(
            runs=("method", "size"), bic_mean=("bic", "mean"),
            bic_std=("bic", "std"), edges_mean=("edges", "mean"),
            runtime_mean=("runtime", "mean"),
            pilot_runtime_mean=("pilot_runtime", "mean"),
        ).to_csv(args.output_dir / "aggregate.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "policies": policies, "d": args.d, "graph_types": args.graph_types,
        "sample_sizes": args.sample_sizes, "seeds": args.seeds,
        "notreks_used": False, "score_normalization": "standardized X",
        "continuation": {"mu": args.mu_schedule, "s": args.s_schedule},
    }, indent=2))


if __name__ == "__main__":
    main()
