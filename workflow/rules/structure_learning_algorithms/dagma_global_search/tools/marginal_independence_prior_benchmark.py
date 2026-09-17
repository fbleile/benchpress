"""Test whether finite-sample visibility of valid NOTREKS pairs matters.

True no-trek pairs are classified on an independent calibration sample using a
Pearson marginal-independence test.  The solver receives a disjoint sample and
the same-sized prior budget under three deterministic selection policies:
random, pairs that look dependent first (hidden valid constraints), and pairs
that look independent first (visible valid constraints).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import generate
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import (
    cpdag_shd,
    dagma_candidate,
    flop_notreks_candidate,
    metrics,
    vanilla_flop_candidate,
)
from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
)


def _classify(X: np.ndarray, pairs: list[tuple[int, int]], alpha: float):
    rows = []
    for left, right in pairs:
        result = pearsonr(X[:, left], X[:, right])
        pvalue = float(result.pvalue)
        rows.append({
            "left": left,
            "right": right,
            "calibration_r": float(result.statistic),
            "calibration_pvalue": pvalue,
            "test_rejects_independence": int(pvalue < alpha),
        })
    return rows


def _select(pairs, classifications, fraction, seed, policy):
    count = int(round(fraction * len(pairs)))
    if count <= 0:
        return [], []
    rng = np.random.default_rng(seed + 104729)
    indexed = list(enumerate(classifications))
    if policy == "random":
        order = rng.permutation(len(indexed)).tolist()
    elif policy == "hidden-only":
        order = sorted(
            (i for i, row in indexed if row["test_rejects_independence"]),
            key=lambda i: (
                classifications[i]["calibration_pvalue"],
                float(rng.random())))
    else:
        # Rank the complete valid-prior set by the calibration result.  The
        # smallest p-values are valid pairs that look most dependent
        # (hidden finite-sample independences); the largest p-values are the
        # pairs whose independence is most visible in the calibration data.
        tie_break = {i: float(rng.random()) for i, _ in indexed}
        if policy == "hidden-valid-first":
            order = sorted(
                (i for i, _ in indexed),
                key=lambda i: (
                    classifications[i]["calibration_pvalue"], tie_break[i]))
        else:
            order = sorted(
                (i for i, _ in indexed),
                key=lambda i: (
                    -classifications[i]["calibration_pvalue"], tie_break[i]))
    chosen = order[:count]
    return [pairs[i] for i in chosen], [classifications[i] for i in chosen]


def _independent_split(X, fraction):
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration fraction must lie strictly between 0 and 1")
    split = int(round(fraction * len(X)))
    return X[:split], X[split:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--n", type=int, default=200,
                        help="total observations; half are used for calibration by default")
    parser.add_argument("--graph-type", default="er2")
    parser.add_argument("--knowledge-fraction", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--methods", nargs="+", choices=("flop", "dagma"),
                        default=["flop", "dagma"])
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--flop-sweeps", type=int, default=16)
    parser.add_argument("--flop-local-passes", type=int, default=8)
    parser.add_argument("--dagma-warm-iter", type=int, default=30000)
    parser.add_argument("--dagma-max-iter", type=int, default=60000)
    parser.add_argument("--dagma-stages", type=int, default=5)
    parser.add_argument("--adapted-novelty", action="store_true",
                        help="also rerun using only prior pairs violated by vanilla")
    parser.add_argument("--only-adapted-novelty", action="store_true",
                        help="write only the adapted novelty rows")
    parser.add_argument("--append", action="store_true",
                        help="append rows to existing output files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policies = (
        "random",
        "hidden-valid-first",
        "visible-valid-first",
        "hidden-only",
    )
    rows = []
    pair_rows = []
    for seed in args.seeds:
        X, truth, all_pairs = generate(
            seed, d=args.d, n=args.n, graph_type=args.graph_type)
        calibration, solver_data = _independent_split(
            X, args.calibration_fraction)
        classification = _classify(calibration, all_pairs, args.alpha)
        for item in classification:
            pair_rows.append({"seed": seed, **item})

        vanilla_flop = vanilla_flop_candidate(
            solver_data, seed, args.attempts)[0] if "flop" in args.methods else None
        for method in args.methods:
            if method == "flop":
                vanilla = vanilla_flop
            else:
                vanilla, _ = dagma_candidate(
                    solver_data, [], False, False, seed, args.attempts,
                    args.dagma_warm_iter, args.dagma_max_iter,
                    args.dagma_stages)
            vanilla_shd = cpdag_shd(truth, vanilla)

            for policy in policies:
                pairs, selected_classification = _select(
                    all_pairs, classification, args.knowledge_fraction,
                    seed, policy)
                hidden = sum(x["test_rejects_independence"] for x in selected_classification)
                pair_is_novel = [bool(common_ancestor_violations(vanilla, [pair]))
                                 for pair in pairs]
                novel_pairs = [pair for pair, is_novel in zip(pairs, pair_is_novel)
                               if is_novel]
                variants = []
                if not args.only_adapted_novelty:
                    variants.append((pairs, f"{method}-nt-{policy}", False))
                if args.adapted_novelty:
                    variants.append((novel_pairs,
                                     f"{method}-nt-{policy}-novel-only", True))

                for run_pairs, method_name, adapted in variants:
                    run_classification = [row for row, is_novel in
                                          zip(selected_classification, pair_is_novel)
                                          if not adapted or is_novel]
                    run_hidden = sum(
                        row["test_rejects_independence"]
                        for row in run_classification)
                    t0 = time.perf_counter()
                    if method == "flop":
                        estimate, diagnostics = flop_notreks_candidate(
                            solver_data, run_pairs, seed, args.attempts,
                            args.flop_sweeps,
                            local_greedy_passes=args.flop_local_passes)
                    else:
                        estimate, diagnostics = dagma_candidate(
                            solver_data, run_pairs, True, False, seed,
                            args.attempts, args.dagma_warm_iter,
                            args.dagma_max_iter, args.dagma_stages)
                    runtime = time.perf_counter() - t0
                    record = metrics(
                        solver_data, truth, run_pairs, method_name,
                        diagnostics["candidate_graph"], estimate,
                        diagnostics["cpdag"], runtime, args.attempts,
                        diagnostics.get("optimizer_restarts", args.attempts))
                    run_violations = common_ancestor_violations(vanilla, run_pairs)
                    run_hidden_novel = sum(
                        row["test_rejects_independence"]
                        for pair, row in zip(run_pairs, run_classification)
                        if common_ancestor_violations(vanilla, [pair]))
                    record.update({
                        "seed": seed,
                        "knowledge_fraction": args.knowledge_fraction,
                        "selected_pairs_before_adaptation": len(pairs),
                        "supplied_pairs": len(run_pairs),
                        "hidden_valid_pairs": run_hidden,
                        "visible_valid_pairs": len(run_pairs) - run_hidden,
                        "hidden_valid_fraction": run_hidden / max(1, len(run_pairs)),
                        "pvalue_mean": float(np.mean([
                            row["calibration_pvalue"]
                            for row in run_classification
                        ])) if run_classification else float("nan"),
                        "calibration_alpha": args.alpha,
                        "calibration_n": len(calibration),
                        "solver_n": len(solver_data),
                        "vanilla_SHD_cpdag": vanilla_shd,
                        "cpdag_SHD_gain": vanilla_shd - record["SHD_cpdag"],
                        "calibration_hidden_pool": sum(
                            x["test_rejects_independence"] for x in classification),
                        "calibration_valid_pool": len(classification),
                        "vanilla_violating_pairs": int(run_violations),
                        "novel_notrek_count": int(run_violations),
                        "vanilla_satisfied_pairs": int(len(run_pairs) - run_violations),
                        "novel_pair_fraction": run_violations / max(1, len(run_pairs)),
                        "novelty_ratio": run_violations / max(1, len(run_pairs)),
                        "hidden_and_novel_pairs": int(run_hidden_novel),
                        "visible_and_old_pairs": int(
                            len(run_pairs) - run_hidden - run_violations + run_hidden_novel),
                        "adapted_novelty": int(adapted),
                    })
                    rows.append(record)
                    print(
                        f"seed={seed} method={method_name} "
                        f"pairs={len(run_pairs)} hidden={run_hidden} "
                        f"gain={record['cpdag_SHD_gain']:.1f} "
                        f"runtime={runtime:.2f}s", flush=True)

    result = pd.DataFrame(rows)
    result_path = args.output_dir / "per_instance.csv"
    pair_path = args.output_dir / "calibration_pairs.csv"
    if args.append and result_path.exists():
        result = pd.concat([pd.read_csv(result_path), result], ignore_index=True)
        result = result.drop_duplicates(
            subset=["method", "seed", "knowledge_fraction"], keep="last")
    pairs = pd.DataFrame(pair_rows)
    if args.append and pair_path.exists():
        pairs = pd.concat([pd.read_csv(pair_path), pairs], ignore_index=True)
        pairs = pairs.drop_duplicates(subset=["seed", "left", "right"], keep="last")
    result.to_csv(result_path, index=False)
    pairs.to_csv(pair_path, index=False)
    summary = result.groupby(["method", "knowledge_fraction"]).agg(
        runs=("seed", "size"),
        SHD_cpdag_mean=("SHD_cpdag", "mean"),
        SHD_gain_mean=("cpdag_SHD_gain", "mean"),
        SHD_gain_median=("cpdag_SHD_gain", "median"),
        hidden_valid_fraction_mean=("hidden_valid_fraction", "mean"),
        novel_pair_fraction_mean=("novel_pair_fraction", "mean"),
        pvalue_mean=("pvalue_mean", "mean"),
        novelty_ratio_mean=("novelty_ratio", "mean"),
        novel_notrek_count_mean=("novel_notrek_count", "mean"),
        hidden_and_novel_pairs_mean=("hidden_and_novel_pairs", "mean"),
        runtime_mean=("candidate_runtime", "mean"),
    ).reset_index()
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
