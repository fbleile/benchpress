"""Prior-sampling diagnostic on one fixed graph/data instance.

This is deliberately separate from the production benchmark.  It fixes the
generated graph and standardized data, samples several independent subsets of
the same true NOTREKS set, computes the exact chromatic number of each prior
graph, and compares one cached vanilla FLOP result with local FLOP+NOTREKS.
The primary gain is CPDAG SHD, computed from the returned DAG support. The
skeleton metrics are retained as secondary diagnostics.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    is_dag,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.chromatic import (
    chromatic_number,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import (
    generate,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import (
    cpdag_shd,
    dagma_candidate,
    flop_notreks_candidate,
    vanilla_flop_candidate,
)


def _skeleton_metrics(truth: np.ndarray, estimate: np.ndarray) -> tuple[float, int]:
    truth_skeleton = truth.astype(bool) | truth.astype(bool).T
    estimate_skeleton = estimate.astype(bool) | estimate.astype(bool).T
    tp = int(np.sum(truth_skeleton & estimate_skeleton) // 2)
    fp = int(np.sum(~truth_skeleton & estimate_skeleton) // 2)
    fn = int(np.sum(truth_skeleton & ~estimate_skeleton) // 2)
    f1 = 2.0 * tp / max(1, 2 * tp + fp + fn)
    return f1, fp + fn


def _prior_graph_properties(d: int, pairs) -> dict[str, float | int | bool]:
    """Structural diagnostics for the undirected prior graph H=(V,I)."""
    graph = nx.Graph()
    graph.add_nodes_from(range(d))
    graph.add_edges_from(tuple(sorted(map(int, pair))) for pair in pairs)
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    components = list(nx.connected_components(graph))
    return {
        "prior_edge_density": 2.0 * graph.number_of_edges() / max(1, d * (d - 1)),
        "prior_degree_mean": float(degrees.mean()),
        "prior_degree_std": float(degrees.std()),
        "prior_degree_max": int(degrees.max(initial=0)),
        "prior_isolated_nodes": int(np.sum(degrees == 0)),
        "prior_endpoint_coverage": float(np.mean(degrees > 0)),
        "prior_connected_components": len(components),
        "prior_largest_component": max((len(component) for component in components), default=0),
        "prior_is_bipartite": bool(nx.is_bipartite(graph)),
    }


def _safe_correlation(left: pd.Series, right: pd.Series, method: str):
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return np.nan
    return left.corr(right, method=method)


def _prior_error_alignment(vanilla: np.ndarray, truth: np.ndarray, pairs) -> dict[str, float | int]:
    """Measure how directly supplied pairs target vanilla false positives."""
    vanilla = np.asarray(vanilla, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    false_positive_edges = [
        (int(source), int(target))
        for source, target in zip(*np.nonzero(vanilla & ~truth))
        if source != target
    ]
    prior_pairs = {tuple(sorted(map(int, pair))) for pair in pairs}
    aligned = sum(
        tuple(sorted((source, target))) in prior_pairs
        for source, target in false_positive_edges
    )
    denominator = len(false_positive_edges)
    return {
        "vanilla_false_positive_edges": denominator,
        "prior_forbidden_vanilla_false_positives": int(aligned),
        "prior_error_alignment": aligned / denominator if denominator else np.nan,
    }


def _nested_knowledge(pairs, fraction: float, seed: int):
    """Return a deterministic prefix of one prior-seed permutation."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("knowledge fractions must lie in (0, 1]")
    pairs = list(pairs)
    rng = np.random.default_rng(seed + 104729)
    order = rng.permutation(len(pairs))
    count = int(round(fraction * len(pairs)))
    return [pairs[int(index)] for index in order[:count]]


def _clique_first_knowledge(pairs, fraction: float, seed: int):
    """Use a largest feasible clique, then fill to the fixed pair budget.

    The clique is in the NOTREKS information graph: its edges are supplied
    no-trek pairs.  A clique of size r certifies chromatic number at least r.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("knowledge fractions must lie in (0, 1]")
    pairs = sorted(tuple(sorted(map(int, pair))) for pair in pairs)
    budget = int(round(fraction * len(pairs)))
    graph = nx.Graph()
    graph.add_edges_from(pairs)
    cliques = list(nx.find_cliques(graph))
    feasible = [clique for clique in cliques
                if len(clique) * (len(clique) - 1) // 2 <= budget]
    clique = max(feasible, key=lambda c: (len(c), tuple(sorted(c)))) if feasible else []
    selected = {pair for pair in combinations(sorted(clique), 2)}
    rng = np.random.default_rng(seed + 104729)
    remaining = [pair for pair in pairs if pair not in selected]
    if len(selected) < budget:
        order = rng.permutation(len(remaining))
        selected.update(remaining[int(index)] for index in order[:budget - len(selected)])
    return sorted(selected)


def _bipartite_knowledge(pairs, fraction: float, seed: int):
    """Use a fixed two-colour cut to deliberately keep chi at most two."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("knowledge fractions must lie in (0, 1]")
    pairs = sorted(tuple(sorted(map(int, pair))) for pair in pairs)
    budget = int(round(fraction * len(pairs)))
    vertices = sorted({vertex for pair in pairs for vertex in pair})
    rng = np.random.default_rng(seed + 104729)
    shuffled = list(rng.permutation(vertices))
    left = set(shuffled[:len(shuffled) // 2])
    crossing = [pair for pair in pairs
                if (pair[0] in left) != (pair[1] in left)]
    order = rng.permutation(len(crossing))
    if len(crossing) < budget:
        raise ValueError(
            "bipartite prior cannot satisfy the requested fixed pair budget: "
            f"need {budget}, available crossing pairs {len(crossing)}")
    return sorted(crossing[int(index)] for index in order[:budget])


def _k_partite_knowledge(pairs, fraction: float, seed: int, colors: int):
    """Select a fixed-size k-partite subgraph of the NOTREKS graph."""
    if colors < 2:
        raise ValueError("a partite construction needs at least two colours")
    pairs = sorted(tuple(sorted(map(int, pair))) for pair in pairs)
    budget = int(round(fraction * len(pairs)))
    vertices = sorted({vertex for pair in pairs for vertex in pair})
    rng = np.random.default_rng(seed + 104729)
    shuffled = list(rng.permutation(vertices))
    color_of = {vertex: index % colors
                for index, vertex in enumerate(shuffled)}
    crossing = [pair for pair in pairs
                if color_of[pair[0]] != color_of[pair[1]]]
    if len(crossing) < budget:
        raise ValueError(
            f"{colors}-partite prior needs {budget} crossing pairs, "
            f"but this partition has only {len(crossing)}")
    order = rng.permutation(len(crossing))
    return sorted(crossing[int(index)] for index in order[:budget])


def _bipartite_max_capacity(pairs, fraction: float, seed: int):
    """Use a maximum-capacity two-colour cut of the NOTREKS graph."""
    pairs = sorted(tuple(sorted(map(int, pair))) for pair in pairs)
    budget = int(round(fraction * len(pairs)))
    vertices = sorted({vertex for pair in pairs for vertex in pair})
    if len(vertices) > 24:
        # Exact enumeration is intended for this d=20 diagnostic.  Keep a
        # deterministic large-d fallback rather than silently claiming exactness.
        rng = np.random.default_rng(seed + 104729)
        cuts = [rng.integers(0, 2, size=len(vertices)) for _ in range(2048)]
        best_crossing = max(
            ([(left, right) for left, right in pairs
              if cut[vertices.index(left)] != cut[vertices.index(right)]]
             for cut in cuts),
            key=len,
            default=[],
        )
    else:
        # Fix the first vertex in colour class zero and evaluate all remaining
        # assignments in vectorized chunks.
        vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
        left_indices = np.asarray([vertex_index[left] for left, _ in pairs])
        right_indices = np.asarray([vertex_index[right] for _, right in pairs])
        best_count = -1
        best_cut = None
        for start in range(0, 1 << (len(vertices) - 1), 65536):
            values = np.arange(
                start, min(start + 65536, 1 << (len(vertices) - 1)),
                dtype=np.uint32)
            bits = ((values[:, None] >> np.arange(len(vertices))) & 1)
            counts = np.sum(bits[:, left_indices] != bits[:, right_indices], axis=1)
            index = int(np.argmax(counts))
            if int(counts[index]) > best_count:
                best_count = int(counts[index])
                best_cut = bits[index]
        color = dict(zip(vertices, map(int, best_cut)))
        best_crossing = [pair for pair in pairs
                         if color[pair[0]] != color[pair[1]]]
    capacity = 0 if best_crossing is None else len(best_crossing)
    if capacity < budget:
        # A bipartite prior cannot contain more crossing edges than its
        # maximum cut.  Return the largest feasible prior and let the result
        # table expose the shortfall instead of silently changing the graph or
        # aborting the whole experiment.
        budget = capacity
    rng = np.random.default_rng(seed + 104729)
    order = rng.permutation(len(best_crossing))
    return sorted(best_crossing[int(index)] for index in order[:budget])


def _clique_target_knowledge(pairs, fraction: float, seed: int, target: int):
    """Include a target-size clique, then fill the fixed pair budget."""
    if target < 2:
        raise ValueError("a target clique needs at least two vertices")
    pairs = sorted(tuple(sorted(map(int, pair))) for pair in pairs)
    budget = int(round(fraction * len(pairs)))
    required = target * (target - 1) // 2
    if budget < required:
        raise ValueError(
            f"clique-first-{target} needs {required} pairs, "
            f"but the requested budget is {budget}")
    graph = nx.Graph()
    graph.add_edges_from(pairs)
    candidates = [tuple(sorted(clique)) for clique in nx.find_cliques(graph)
                  if len(clique) >= target]
    if not candidates:
        raise ValueError(f"no clique of size {target} exists in the prior graph")
    clique = min(candidates, key=lambda candidate: (-len(candidate), candidate))
    selected = set(combinations(clique[:target], 2))
    remaining = [pair for pair in pairs if pair not in selected]
    rng = np.random.default_rng(seed + 104729)
    order = rng.permutation(len(remaining))
    selected.update(remaining[int(index)] for index in order[:budget - len(selected)])
    return sorted(selected)


def _chromatic_greedy_knowledge(pairs, fraction: float, seed: int):
    """Fill a fixed pair budget by greedily maximizing exact chromaticity."""
    pairs = sorted(tuple(sorted(map(int, pair))) for pair in pairs)
    budget = int(round(fraction * len(pairs)))
    if budget == 0:
        return []
    graph = nx.Graph()
    graph.add_edges_from(pairs)
    feasible = [clique for clique in nx.find_cliques(graph)
                if len(clique) * (len(clique) - 1) // 2 <= budget]
    clique = max(feasible, key=lambda c: (len(c), tuple(sorted(c)))) if feasible else []
    selected = set(combinations(sorted(clique), 2))
    remaining = [pair for pair in pairs if pair not in selected]
    rng = np.random.default_rng(seed + 104729)
    while len(selected) < budget:
        order = rng.permutation(len(remaining))
        best_pair = None
        best_score = -1
        for index in order:
            pair = remaining[int(index)]
            score = chromatic_number(
                max((max(left, right) for left, right in selected | {pair}), default=-1) + 1,
                selected | {pair})
            if score > best_score:
                best_score = score
                best_pair = pair
        selected.add(best_pair)
        remaining.remove(best_pair)
    return sorted(selected)


def _maximum_clique_size(pairs):
    graph = nx.Graph()
    graph.add_edges_from(pairs)
    return max((len(clique) for clique in nx.find_cliques(graph)), default=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vary nested NOTREKS fractions on one fixed instance.")
    parser.add_argument("--graph-seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--graph-type", default="er2")
    parser.add_argument("--prior-seeds", nargs="+", type=int,
                        default=list(range(1001, 1021)))
    parser.add_argument("--knowledge-fractions", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0],
                        help="nested prior fractions evaluated on this instance")
    parser.add_argument("--knowledge-strategies", nargs="+",
                        choices=(
                            "random", "2-partite", "3-partite", "4-partite",
                            "bipartite", "bipartite-max-capacity",
                            "chromatic-greedy", "clique-first-2", "clique-first-3",
                            "clique-first-4", "clique_first",
                            "maximum-clique-first"),
                        default=["random", "2-partite", "3-partite",
                                 "4-partite", "bipartite-max-capacity",
                                 "chromatic-greedy", "clique-first-2",
                                 "clique-first-3", "clique-first-4",
                                 "maximum-clique-first"],
                        help="prior construction strategies to compare")
    parser.add_argument("--attempts", type=int, default=5,
                        help="total FLOP attempts, including the initial one")
    parser.add_argument("--flop-sweeps", type=int, default=16)
    parser.add_argument("--flop-local-passes", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=("flop", "dagma"),
                        default=["flop"],
                        help="solvers to evaluate for each supplied prior")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(not 0.0 < fraction <= 1.0
           for fraction in args.knowledge_fractions):
        raise SystemExit("knowledge fractions must lie in (0, 1]")
    if len(set(args.knowledge_fractions)) != len(args.knowledge_fractions):
        raise SystemExit("knowledge fractions must be unique")
    if len(set(args.prior_seeds)) != len(args.prior_seeds):
        raise SystemExit("--prior-seeds must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, truth, all_pairs = generate(
        args.graph_seed, d=args.d, n=args.n, graph_type=args.graph_type,
        scm="linear", noise="gaussian")

    started = time.perf_counter()
    vanilla_graph, vanilla_diag = vanilla_flop_candidate(
        X, args.graph_seed, args.attempts)
    vanilla_runtime = time.perf_counter() - started
    vanilla_cpdag = vanilla_diag["cpdag"]
    vanilla_cpdag_shd = cpdag_shd(truth, vanilla_graph)

    metadata = {
        "graph_seed": args.graph_seed,
        "n": args.n,
        "d": args.d,
        "graph_type": args.graph_type,
        "knowledge_fractions": list(args.knowledge_fractions),
        "knowledge_strategies": list(args.knowledge_strategies),
        "prior_seeds": list(args.prior_seeds),
        "attempts": args.attempts,
        "flop_sweeps": args.flop_sweeps,
        "flop_local_passes": args.flop_local_passes,
        "methods": list(args.methods),
        "standardized_data": True,
        "full_notreks_pairs": len(all_pairs),
        "vanilla_runtime": vanilla_runtime,
        "vanilla_dag": bool(is_dag(vanilla_graph)),
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "fixed_instance.npz", X=X, truth=truth,
        true_notreks_pairs=np.asarray(all_pairs, dtype=np.int64),
        vanilla_graph=vanilla_graph, vanilla_cpdag=vanilla_cpdag)

    rows: list[dict[str, object]] = []
    vanilla_f1, vanilla_skeleton_shd = _skeleton_metrics(truth, vanilla_cpdag)
    total_cells = (len(args.methods) * len(args.prior_seeds)
                   * len(args.knowledge_fractions)
                   * len(args.knowledge_strategies))
    experiment_started = time.perf_counter()
    completed = 0
    for prior_seed in args.prior_seeds:
        for knowledge_fraction in args.knowledge_fractions:
            for strategy in args.knowledge_strategies:
                if strategy == "random":
                    pairs = _nested_knowledge(
                        all_pairs, knowledge_fraction, prior_seed)
                elif strategy in {"bipartite", "2-partite"}:
                    pairs = _k_partite_knowledge(
                        all_pairs, knowledge_fraction, prior_seed, 2)
                elif strategy == "bipartite-max-capacity":
                    pairs = _bipartite_max_capacity(
                        all_pairs, knowledge_fraction, prior_seed)
                elif strategy == "3-partite":
                    pairs = _k_partite_knowledge(
                        all_pairs, knowledge_fraction, prior_seed, 3)
                elif strategy == "4-partite":
                    pairs = _k_partite_knowledge(
                        all_pairs, knowledge_fraction, prior_seed, 4)
                elif strategy == "chromatic-greedy":
                    pairs = _chromatic_greedy_knowledge(
                        all_pairs, knowledge_fraction, prior_seed)
                elif strategy.startswith("clique-first-"):
                    target = int(strategy.rsplit("-", 1)[1])
                    pairs = _clique_target_knowledge(
                        all_pairs, knowledge_fraction, prior_seed, target)
                else:
                    pairs = (_clique_first_knowledge(
                        all_pairs, knowledge_fraction, prior_seed)
                        if strategy == "clique_first" else
                        _bipartite_knowledge(
                            all_pairs, knowledge_fraction, prior_seed))
                chi = chromatic_number(args.d, pairs)
                prior_properties = _prior_graph_properties(args.d, pairs)
                error_alignment = _prior_error_alignment(
                    vanilla_graph, truth, pairs)
                vanilla_trek_violations = common_ancestor_violations(
                    vanilla_graph, pairs)
                requested_pairs = int(round(knowledge_fraction * len(all_pairs)))
                if len(pairs) < requested_pairs:
                    print(
                        f"warning: strategy={strategy} q={knowledge_fraction:g} "
                        f"requested={requested_pairs} supplied={len(pairs)}",
                        flush=True,
                    )
                for method in args.methods:
                    started = time.perf_counter()
                    if method == "flop":
                        nt_graph, nt_diag = flop_notreks_candidate(
                            X, pairs, args.graph_seed, args.attempts,
                            args.flop_sweeps,
                            search_version="local_greedy_rust",
                            local_greedy_passes=args.flop_local_passes)
                    else:
                        nt_graph, nt_diag = dagma_candidate(
                            X, pairs, True, False, args.graph_seed,
                            args.attempts, 30000, 60000, 5)
                    nt_runtime = time.perf_counter() - started
                    nt_cpdag = nt_diag["cpdag"]
                    nt_cpdag_shd = cpdag_shd(truth, nt_graph)
                    nt_f1, nt_skeleton_shd = _skeleton_metrics(truth, nt_cpdag)
                    if method == "flop":
                        base_shd, base_skeleton_shd, base_f1 = (
                            vanilla_cpdag_shd, vanilla_skeleton_shd, vanilla_f1)
                    else:
                        started = time.perf_counter()
                        base_graph, base_diag = dagma_candidate(
                            X, [], False, False, args.graph_seed,
                            args.attempts, 30000, 60000, 5)
                        base_runtime = time.perf_counter() - started
                        base_cpdag = base_diag["cpdag"]
                        base_shd = cpdag_shd(truth, base_graph)
                        base_f1, base_skeleton_shd = _skeleton_metrics(
                            truth, base_cpdag)
                    rows.append({
                    "method": method,
                    "graph_seed": args.graph_seed,
                    "prior_seed": prior_seed,
                    "knowledge_strategy": strategy,
                    "n": args.n,
                    "d": args.d,
                    "graph_type": args.graph_type,
                    "knowledge_fraction": knowledge_fraction,
                    "requested_pairs": requested_pairs,
                    "supplied_pairs": len(pairs),
                    "actual_knowledge_fraction": len(pairs) / max(1, len(all_pairs)),
                    **error_alignment,
                    "vanilla_notreks_violations": vanilla_trek_violations,
                    "vanilla_notreks_violation_rate": (
                        vanilla_trek_violations / max(1, len(pairs))),
                    "full_notreks_pairs": len(all_pairs),
                    "maximum_clique_size": _maximum_clique_size(pairs),
                    "chromatic_number_exact": chi,
                    **prior_properties,
                    "vanilla_SHD_cpdag": base_shd,
                    "nt_SHD_cpdag": nt_cpdag_shd,
                    "vanilla_SHD_skeleton": base_skeleton_shd,
                    "nt_SHD_skeleton": nt_skeleton_shd,
                    "vanilla_F1_skeleton": base_f1,
                    "flop_nt_F1_skeleton": nt_f1,
                    "cpdag_SHD_gain": base_shd - nt_cpdag_shd,
                    "skeleton_SHD_gain": base_skeleton_shd - nt_skeleton_shd,
                    "F1_gain": nt_f1 - base_f1,
                    "vanilla_runtime": vanilla_runtime,
                    "nt_runtime": nt_runtime,
                    "violations": common_ancestor_violations(nt_graph, pairs),
                    "is_dag": bool(is_dag(nt_graph)),
                    "attempts_completed": nt_diag["optimizer_restarts"],
                })
                    completed += 1
                    frame = pd.DataFrame(rows)
                    frame.to_csv(args.output_dir / "per_prior.csv", index=False)
                    elapsed = time.perf_counter() - experiment_started
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta = (total_cells - completed) / rate if rate > 0 else float("nan")
                    print(
                        f"[{completed}/{total_cells}] method={method} "
                        f"prior_seed={prior_seed} strategy={strategy} "
                        f"q={knowledge_fraction:g} chi={chi} "
                        f"cpdag_gain={base_shd - nt_cpdag_shd:g} "
                        f"runtime={nt_runtime:.2f}s "
                        f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                        flush=True,
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_prior.csv", index=False)
    correlation_rows = []
    for (method, strategy, fraction), group in frame.groupby(
            ["method", "knowledge_strategy", "knowledge_fraction"], sort=True):
        correlation_rows.append({
            "correlation_scope": "within_strategy",
            "method": method,
            "knowledge_strategy": strategy,
            "knowledge_fraction": fraction,
            "spearman_rho_exact_chromatic": group[
                "chromatic_number_exact"].corr(
                    group["cpdag_SHD_gain"], method="spearman"),
            "pearson_r_exact_chromatic": group[
                "chromatic_number_exact"].corr(
                    group["cpdag_SHD_gain"], method="pearson"),
            "n_prior_samples": len(group),
        })
    for (method, fraction), group in frame.groupby(
            ["method", "knowledge_fraction"], sort=True):
        pooled = group.groupby("knowledge_strategy", as_index=False).agg(
            chromatic_number_exact=("chromatic_number_exact", "mean"),
            cpdag_SHD_gain=("cpdag_SHD_gain", "mean"),
        )
        correlation_rows.append({
            "correlation_scope": "across_strategy_means",
            "method": method,
            "knowledge_strategy": "all",
            "knowledge_fraction": fraction,
            "spearman_rho_exact_chromatic": pooled[
                "chromatic_number_exact"].corr(
                    pooled["cpdag_SHD_gain"], method="spearman"),
            "pearson_r_exact_chromatic": pooled[
                "chromatic_number_exact"].corr(
                    pooled["cpdag_SHD_gain"], method="pearson"),
            "n_prior_samples": len(pooled),
        })
        correlation_rows.append({
            "correlation_scope": "pooled_rows",
            "method": method,
            "knowledge_strategy": "all",
            "knowledge_fraction": fraction,
            "spearman_rho_exact_chromatic": group[
                "chromatic_number_exact"].corr(
                    group["cpdag_SHD_gain"], method="spearman"),
            "pearson_r_exact_chromatic": group[
                "chromatic_number_exact"].corr(
                    group["cpdag_SHD_gain"], method="pearson"),
            "n_prior_samples": len(group),
        })
    # Primary diagnostic: pool all prior experiments.  Strategy-specific
    # correlations are secondary because some strategies have constant chi.
    correlation_rows.append({
        "correlation_scope": "overall_pooled",
        "method": "all",
        "knowledge_strategy": "all",
        "knowledge_fraction": "all",
        "spearman_rho_exact_chromatic": frame[
            "chromatic_number_exact"].corr(
                frame["cpdag_SHD_gain"], method="spearman"),
        "pearson_r_exact_chromatic": frame[
            "chromatic_number_exact"].corr(
                frame["cpdag_SHD_gain"], method="pearson"),
        "n_prior_samples": len(frame),
    })
    for method, group in frame.groupby("method", sort=True):
        correlation_rows.append({
            "correlation_scope": "overall_by_method",
            "method": method,
            "knowledge_strategy": "all",
            "knowledge_fraction": "all",
            "spearman_rho_exact_chromatic": group[
                "chromatic_number_exact"].corr(
                    group["cpdag_SHD_gain"], method="spearman"),
            "pearson_r_exact_chromatic": group[
                "chromatic_number_exact"].corr(
                    group["cpdag_SHD_gain"], method="pearson"),
            "n_prior_samples": len(group),
        })
    property_columns = [
        "chromatic_number_exact",
        "maximum_clique_size",
        "prior_edge_density",
        "prior_degree_mean",
        "prior_degree_std",
        "prior_degree_max",
        "prior_isolated_nodes",
        "prior_endpoint_coverage",
        "prior_connected_components",
        "prior_largest_component",
        "prior_is_bipartite",
        "prior_error_alignment",
        "vanilla_notreks_violations",
        "vanilla_notreks_violation_rate",
    ]
    property_rows = []
    for property_name in property_columns:
        property_rows.append({
            "scope": "overall_pooled",
            "method": "all",
            "knowledge_fraction": "all",
            "property": property_name,
            "spearman_rho": _safe_correlation(
                frame[property_name].astype(float),
                frame["cpdag_SHD_gain"], "spearman"),
            "pearson_r": _safe_correlation(
                frame[property_name].astype(float),
                frame["cpdag_SHD_gain"], "pearson"),
            "n": len(frame),
        })
        for method, group in frame.groupby("method", sort=True):
            property_rows.append({
                "scope": "overall_by_method",
                "method": method,
                "knowledge_fraction": "all",
                "property": property_name,
                "spearman_rho": _safe_correlation(
                    group[property_name].astype(float),
                    group["cpdag_SHD_gain"], "spearman"),
                "pearson_r": _safe_correlation(
                    group[property_name].astype(float),
                    group["cpdag_SHD_gain"], "pearson"),
                "n": len(group),
            })
        for fraction, group in frame.groupby("knowledge_fraction", sort=True):
            property_rows.append({
                "scope": "by_knowledge_fraction",
                "method": "all",
                "knowledge_fraction": fraction,
                "property": property_name,
                "spearman_rho": _safe_correlation(
                    group[property_name].astype(float),
                    group["cpdag_SHD_gain"], "spearman"),
                "pearson_r": _safe_correlation(
                    group[property_name].astype(float),
                    group["cpdag_SHD_gain"], "pearson"),
                "n": len(group),
            })
    property_frame = pd.DataFrame(property_rows)
    property_frame.to_csv(
        args.output_dir / "prior_property_gain_correlation.csv", index=False)
    corr = pd.DataFrame(correlation_rows)
    corr.to_csv(args.output_dir / "chromatic_gain_correlation.csv", index=False)
    frame.groupby(["method", "knowledge_strategy", "knowledge_fraction"], as_index=False).agg(
        chromatic_number_exact_mean=("chromatic_number_exact", "mean"),
        chromatic_number_exact_std=("chromatic_number_exact", "std"),
        cpdag_SHD_gain_mean=("cpdag_SHD_gain", "mean"),
        cpdag_SHD_gain_std=("cpdag_SHD_gain", "std"),
        n=("cpdag_SHD_gain", "size"),
    ).to_csv(args.output_dir / "chromatic_gain_by_strategy.csv", index=False)
    summary = frame.agg({
        "chromatic_number_exact": ["min", "mean", "max"],
        "cpdag_SHD_gain": ["min", "mean", "max"],
        "skeleton_SHD_gain": ["min", "mean", "max"],
        "F1_gain": ["min", "mean", "max"],
        "nt_runtime": ["mean", "std"],
    })
    summary.to_csv(args.output_dir / "summary.csv")
    print(frame[["method", "prior_seed", "knowledge_strategy", "knowledge_fraction",
                 "supplied_pairs",
                 "chromatic_number_exact",
                 "vanilla_SHD_cpdag", "nt_SHD_cpdag",
                 "cpdag_SHD_gain", "nt_runtime"]].to_string(index=False))
    print("\nCorrelation:")
    print(corr.to_string(index=False))
    print("\nPrior-property correlation with CPDAG SHD gain:")
    print(property_frame[
        property_frame["scope"].isin(["overall_pooled", "overall_by_method"])
    ].to_string(index=False))
    print("\nOutputs:", args.output_dir)


if __name__ == "__main__":
    main()
