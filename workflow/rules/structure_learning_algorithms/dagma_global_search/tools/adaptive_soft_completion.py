"""Partial-information adaptive completion for experimental Trek-Dominance.

The solver-facing entry point deliberately accepts no graph truth.  Truth is
used only by the optional command-line diagnostic layer when scoring synthetic
outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import flopsearch

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import generate


Pair = tuple[int, int]


def canon(pair: Pair) -> Pair:
    return tuple(sorted(pair))  # type: ignore[return-value]


def pairwise_evidence(X: np.ndarray, hard: set[Pair]) -> dict[Pair, tuple[float, float]]:
    """Return (cost declaring no-trek, cost declaring trek) per unknown pair."""
    n, d = X.shape
    out = {}
    for i, j in combinations(range(d), 2):
        if (i, j) in hard:
            continue
        r = float(np.corrcoef(X[:, i], X[:, j])[0, 1])
        r = float(np.clip(r, -1 + 1e-12, 1 - 1e-12))
        improvement = max(0.0, n * (-np.log1p(-r * r)) - np.log(max(n, 2)))
        # Lower cost means better empirical support for the declaration.
        out[(i, j)] = (improvement, 0.0)
    return out


def completion_from_thresholds(X: np.ndarray, hard: set[Pair], fractions: tuple[float, ...]) -> list[set[Pair]]:
    evidence = pairwise_evidence(X, hard)
    ranked = sorted(evidence, key=lambda p: evidence[p][0])
    unknown = len(ranked)
    states = [set(hard)]
    for fraction in fractions:
        count = int(round(fraction * unknown))
        states.append(set(hard) | set(ranked[count:]))
    return states


def candidate_trek_edges(d: int, no_treks: set[Pair]) -> list[Pair]:
    return [(i, j) for i, j in combinations(range(d), 2)
            if (i, j) not in no_treks]


@dataclass
class CompletionResult:
    graph: np.ndarray
    diagnostics: dict


def adaptive_soft_completion(
    X: np.ndarray,
    supplied_pairs: list[Pair] | np.ndarray,
    *,
    total_restarts: int = 5,
    seed: int = 0,
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
    beam_width: int = 4,
) -> CompletionResult:
    """Run adaptive completion using only data and supplied hard pairs."""
    if total_restarts < 1:
        raise ValueError("total_restarts must be positive")
    d = X.shape[1]
    hard = {canon((int(i), int(j))) for i, j in supplied_pairs}
    all_pairs = {canon(p) for p in combinations(range(d), 2)}
    if not hard <= all_pairs:
        raise ValueError("supplied NOTREKS contains an invalid pair")
    states = completion_from_thresholds(X, hard, thresholds)
    unique = []
    seen = set()
    for state in states:
        key = tuple(sorted(state))
        if key not in seen:
            unique.append(state)
            seen.add(key)
    # Every visited completion receives at least one actual FLOP attempt;
    # this prevents a negative ``restarts`` value when a smoke test uses a
    # smaller budget than the default completion beam.
    states = unique[:min(beam_width, total_restarts)]
    # Allocate the fixed budget once; adaptive completion never adds oracle
    # calls beyond this allocation.
    counts = [total_restarts // len(states)] * len(states)
    for i in range(total_restarts % len(states)):
        counts[i] += 1
    best = None
    records = []
    for state, attempts in zip(states, counts):
        T = candidate_trek_edges(d, state)
        raw, info = flopsearch.flop_notreks(
            X, 2.0, list(sorted(hard)), restarts=attempts - 1, seed=seed,
            search_version="trek_dominance", trek_graph=T,
            return_dag=True, return_diagnostics=True)
        graph = np.zeros((d, d), dtype=np.uint8)
        for u, v in info["selected_dag_edges"]:
            graph[int(u), int(v)] = 1
        violations = common_ancestor_violations(graph, list(sorted(hard)))
        if violations:
            raise AssertionError("adaptive completion violated supplied NOTREKS")
        bic = float(info["final_bic"])
        record = {"completion_pairs": len(state), "attempts": attempts,
                  "bic": bic, "violations": violations,
                  "mask_density": info.get("trek_dominance_mask_density", np.nan),
                  "source": "initial_empirical_completion"}
        records.append(record)
        if best is None or bic < best[0]:
            best = (bic, graph, record)
    assert best is not None
    return CompletionResult(best[1], {
        "completion_states_generated": len(states),
        "completion_states_visited": len(records),
        "total_restarts": sum(r["attempts"] for r in records),
        "completion_trajectory": records,
        "winner": best[2],
        "primary_truth_access": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--graph-type", default="er2")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--knowledge-fraction", type=float, default=0.25)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        X, truth, all_pairs = generate(seed, d=args.d, n=args.n,
                                        graph_type=args.graph_type,
                                        scm="linear", noise="gaussian")
        count = int(round(args.knowledge_fraction * len(all_pairs)))
        supplied = list(all_pairs[:count])
        result = adaptive_soft_completion(X, supplied,
                                           total_restarts=args.attempts,
                                           seed=seed)
        rows.append({"seed": seed, "method": "adaptive_soft_completion",
                     "shd_placeholder": np.nan,
                     "violations": common_ancestor_violations(result.graph, supplied),
                     **{k: v for k, v in result.diagnostics.items()
                        if k != "completion_trajectory"}})
        print(rows[-1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "adaptive_results.npy").write_bytes(
        np.array(rows, dtype=object).tobytes())


if __name__ == "__main__":
    main()
