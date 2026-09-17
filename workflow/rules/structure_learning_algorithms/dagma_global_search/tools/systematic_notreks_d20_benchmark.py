"""Fair d=20 comparison of vanilla methods and NOTREKS extensions."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import flopsearch

from workflow.rules.structure_learning_algorithms.flop.adapter import (
    convert_flop_cpdag,
)
from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    common_ancestor_violations,
    gaussian_bic,
    is_dag,
    topological_order,
)
from workflow.rules.structure_learning_algorithms.dagma.end_flop_prune import (
    end_flop_prune,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    ProductionConfig,
    run_production_pipeline,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.local_d20_benchmark import (
    generate,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.chromatic import (
    chromatic_upper_bound,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.adaptive_soft_completion import (
    adaptive_soft_completion,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.causal_distances import (
    cpdag_aid,
    dag_ancestor_aid,
    dag_parent_aid,
    dag_sid,
)
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.benchmark_config import (
    BenchmarkConfig,
    DEFAULT_METHODS,
    SUPPORTED_METHODS,
)


ROOT = Path(__file__).resolve().parents[5]
R_EVAL = ROOT / "workflow/rules/evaluation/benchmarks/run_summarise.R"


def trek_violation_mass(graph, pairs):
    """Return the exact sum of common-ancestor counts over supplied pairs."""
    graph = np.asarray(graph, dtype=bool)
    d = graph.shape[0]
    reach = graph.copy()
    np.fill_diagonal(reach, True)
    for k in range(d):
        reach |= reach[:, [k]] & reach[[k], :]
    # Rows are descendants, so columns are the ancestor bitsets.
    ancestors = reach.T
    return int(sum(np.count_nonzero(ancestors[i] & ancestors[j])
                   for i, j in pairs))


def trek_graph_edges(graph):
    """Return the undirected trek graph of a DAG as canonical edge pairs."""
    graph = np.asarray(graph, dtype=bool)
    reach = graph.copy()
    np.fill_diagonal(reach, True)
    for k in range(graph.shape[0]):
        reach |= reach[:, [k]] & reach[[k], :]
    ancestors = reach.T
    return [(i, j) for i, j in combinations(range(graph.shape[0]), 2)
            if np.any(ancestors[i] & ancestors[j])]


def source_signature_mask(graph, pairs):
    """Return source signatures and the exact admissible directed-edge mask."""
    graph = np.asarray(graph, dtype=bool)
    d = graph.shape[0]
    order = topological_order(graph.astype(np.uint8))
    sources = [node for node in range(d) if not graph[:, node].any()]
    source_bits = {node: 1 << index for index, node in enumerate(sources)}
    signatures = [0] * d
    for node in order:
        signature = source_bits.get(node, 0)
        for parent in np.flatnonzero(graph[:, node]):
            signature |= signatures[int(parent)]
        signatures[node] = signature
    for left, right in pairs:
        if signatures[int(left)] & signatures[int(right)]:
            raise ValueError(
                "source signatures are not disjoint for supplied NOTREKS")
    allowed = np.zeros((d, d), dtype=bool)
    for parent in range(d):
        for child in range(d):
            if parent != child:
                allowed[parent, child] = (
                    signatures[parent] & ~signatures[child]) == 0
    return sources, signatures, allowed


def _diagnostic_graph(d, diagnostics):
    graph = np.zeros((d, d), dtype=np.uint8)
    for source, target in diagnostics["selected_dag_edges"]:
        graph[int(source), int(target)] = 1
    return graph


def vanilla_flop_candidate(X, seed, attempts, lambda_bic=2.0):
    raw, diagnostics = flopsearch.flop_notreks(
        X, lambda_bic, [], restarts=max(0, attempts - 1),
        seed=seed, search_version="global_greedy_rust",
        return_diagnostics=True)
    graph = _diagnostic_graph(X.shape[1], diagnostics)
    return graph, {
        "candidate_graph": graph.copy(),
        "cpdag": convert_flop_cpdag(raw, X.shape[1]),
        "optimizer_restarts": int(diagnostics["restarts_completed"]),
        "optimizer_final_bic": float(diagnostics["final_bic"]),
        "optimizer_violations": int(
            diagnostics["final_no_trek_violation_count"]),
    }


def flop_notreks_candidate(X, pairs, seed, attempts, sweeps,
                           forbidden_edges=(), random_initial_order=False,
                           lambda_bic=2.0,
                           search_version="local_greedy_rust",
                           local_greedy_passes=8):
    # The Python API's FLOP restart argument counts additional ILS restarts;
    # the initial attempt is implicit. Convert the desired actual attempt
    # count here so FLOP and DAGMA use the same number of attempts.
    raw, diagnostics = flopsearch.flop_notreks(
        X, lambda_bic, pairs,
        restarts=max(0, attempts - 1),
        seed=seed,
        forbidden_edges=forbidden_edges,
        random_initial_order=random_initial_order,
        max_signature_rounds=sweeps,
        search_version=search_version,
        local_greedy_passes=local_greedy_passes,
        return_diagnostics=True,
    )
    graph = _diagnostic_graph(X.shape[1], diagnostics)
    return graph, {
        "candidate_graph": graph.copy(),
        "cpdag": convert_flop_cpdag(raw, X.shape[1]),
        "optimizer_restarts": int(diagnostics["restarts_completed"]),
        "optimizer_final_bic": float(diagnostics["final_bic"]),
        "optimizer_violations": int(
            diagnostics["final_no_trek_violation_count"]),
    }


def flop_notreks_local_adaptive_candidate(
        X, pairs, seed, attempts, sweeps, *, lambda_bic=2.0,
        local_greedy_passes=8):
    return flop_notreks_candidate(
        X, pairs, seed, attempts, sweeps, lambda_bic=lambda_bic,
        search_version="local_greedy_rust_adaptive",
        local_greedy_passes=local_greedy_passes)


def flop_notreks_source_signature_candidate(
        X, pairs, seed, attempts, sweeps, *, lambda_bic=2.0,
        local_greedy_passes=8, polish_outputs=1):
    """Polish up to M feasible local FLOP outputs using source signatures."""
    candidates = []
    for index in range(max(1, polish_outputs)):
        local_seed = int(seed + index * 1_000_003)
        local, local_diag = flop_notreks_candidate(
            X, pairs, local_seed, attempts, sweeps,
            lambda_bic=lambda_bic, search_version="local_greedy_rust",
            local_greedy_passes=local_greedy_passes)
        if common_ancestor_violations(local, pairs) != 0:
            raise RuntimeError("source-signature polishing requires feasible local DAG")
        sources, signatures, allowed = source_signature_mask(local, pairs)
        forbidden = [
            (parent, child)
            for parent in range(X.shape[1])
            for child in range(X.shape[1])
            if parent != child and not allowed[parent, child]
        ]
        polished, polish_diag = flopsearch.flop_source_signature(
            X, np.asarray(local, dtype=np.uint8), lambda_bic, pairs,
            seed=local_seed, return_diagnostics=True)
        polished = np.asarray(polished, dtype=np.uint8)
        if common_ancestor_violations(polished, pairs) != 0:
            raise RuntimeError("source-signature FLOP returned an infeasible DAG")
        candidates.extend([
            (float(gaussian_bic(X, local, lambda_bic=lambda_bic)[0]), local,
             "local", local_diag),
            (float(gaussian_bic(X, polished, lambda_bic=lambda_bic)[0]),
             polished, "polished", polish_diag),
        ])
    winner_bic, winner, winner_source, winner_diag = min(
        candidates, key=lambda item: item[0])
    baseline = min(candidates, key=lambda item: item[0] if item[2] == "local" else np.inf)
    return winner, {
        "candidate_graph": winner.copy(),
        "cpdag": winner.copy(),
        "optimizer_restarts": int(attempts),
        "optimizer_final_bic": winner_bic,
        "optimizer_violations": common_ancestor_violations(winner, pairs),
        "source_signature_source_count": int(len(sources)),
        "source_signature_mask_density": float(
            allowed.sum() / max(1, X.shape[1] * (X.shape[1] - 1))),
        "source_signature_pair_check_passed": True,
        "source_signature_polish_outputs": int(max(1, polish_outputs)),
        "source_signature_polishing_changed": int(
            not np.array_equal(winner, baseline[1])),
        "source_signature_winner": winner_source,
        "source_signature_polish_bic": float(
            min(item[0] for item in candidates if item[2] == "polished")),
        "source_signature_local_bic": float(
            min(item[0] for item in candidates if item[2] == "local")),
        "source_signature_forbidden_edges": forbidden,
        "source_signature_polish_diagnostics": winner_diag,
    }


def flop_notreks_order_guided_candidate(
        X, pairs, seed, attempts, sweeps, *, lambda_bic=2.0,
        local_greedy_passes=8, lex_fraction=0.5,
        repair_candidates=2, coverage_starts=0):
    """Order-guided FLOP with an archive-only NOTREKS repair stage.

    Every optimizer order is fit with ordinary unrestricted FLOP mechanics.
    Only terminal archive candidates are inspected or repaired afterwards;
    no repaired graph is passed back into FLOP or used as a warm start.
    """
    raw, diagnostics = flopsearch.flop_notreks(
        X, lambda_bic, pairs, restarts=max(0, attempts - 1), seed=seed,
        max_signature_rounds=sweeps, search_version="order_guided_local",
        local_greedy_passes=local_greedy_passes,
        order_guided_lex_fraction=lex_fraction,
        order_guided_repair_candidates=repair_candidates,
        order_guided_coverage_starts=coverage_starts,
        return_dag=True, return_diagnostics=True)

    def graph_from_edges(edges):
        graph = np.zeros((X.shape[1], X.shape[1]), dtype=np.uint8)
        for source, target in edges:
            graph[int(source), int(target)] = 1
        return graph

    archive = []
    for edges, bic, violations, exact in zip(
            diagnostics.get("order_guided_archive_edges", []),
            diagnostics.get("order_guided_archive_bics", []),
            diagnostics.get("order_guided_archive_violations", []),
            diagnostics.get("order_guided_archive_exact_violations", [])):
        graph = graph_from_edges(edges)
        archive.append({
            "graph": graph,
            "bic": float(bic),
            "violations": int(violations),
            "exact_violations": int(exact),
        })
    if not archive:
        archive = [{
            "graph": np.asarray(raw, dtype=np.uint8),
            "bic": float(gaussian_bic(X, raw, lambda_bic=lambda_bic)[0]),
            "violations": common_ancestor_violations(raw, pairs),
            "exact_violations": np.nan,
        }]

    raw_feasible = [item for item in archive if item["violations"] == 0]
    infeasible = sorted(
        (item for item in archive if item["violations"] != 0),
        key=lambda item: (item["violations"], item["exact_violations"],
                          item["bic"]))
    repaired = []
    for item in infeasible[:max(0, repair_candidates)]:
        graph = notreks_postselection(X, item["graph"], pairs,
                                      lambda_bic=lambda_bic)
        repaired.append({
            "graph": graph,
            "bic": float(gaussian_bic(X, graph, lambda_bic=lambda_bic)[0]),
            "violations": common_ancestor_violations(graph, pairs),
            "source": "repaired",
        })

    feasible = [{**item, "source": "raw"} for item in raw_feasible]
    feasible.extend(item for item in repaired if item["violations"] == 0)
    if not feasible:
        raise RuntimeError("order-guided search produced no feasible archived candidate")
    winner = min(feasible, key=lambda item: item["bic"])
    raw_winner = min(raw_feasible, key=lambda item: item["bic"] \
                     ) if raw_feasible else None
    archive_violation_counts = [item["violations"] for item in archive]
    archive_violation_masses = [
        trek_violation_mass(item["graph"], pairs) for item in archive]
    return winner["graph"], {
        "candidate_graph": (raw_winner["graph"] if raw_winner is not None
                             else archive[0]["graph"]),
        "cpdag": winner["graph"],
        "optimizer_restarts": int(diagnostics["restarts_completed"]),
        "optimizer_final_bic": float(winner["bic"]),
        "optimizer_violations": int(winner["violations"]),
        "order_guided_raw_feasible_count": len(raw_feasible),
        "order_guided_repaired_count": len(repaired),
        "order_guided_winner": winner["source"],
        "order_guided_archive_max_trek_violations": max(
            archive_violation_counts, default=0),
        "order_guided_archive_max_trek_violation_mass": max(
            archive_violation_masses, default=0),
        "order_guided_archive_bics": diagnostics.get(
            "order_guided_archive_bics", []),
        "order_guided_archive_violations": diagnostics.get(
            "order_guided_archive_violations", []),
        "order_guided_archive_exact_violations": diagnostics.get(
            "order_guided_archive_exact_violations", []),
        "order_guided_violation_trajectory": diagnostics.get(
            "order_guided_violation_trajectory", []),
    }


def flop_notreks_trekcut_candidate(
        X, pairs, seed, attempts, sweeps, *, lambda_bic=2.0,
        local_greedy_passes=8, oracle_budget=32, refinement_passes=4):
    """Run TrekCut-FLOP while retaining local repair as incumbent/fallback."""
    incumbent, incumbent_diag = flop_notreks_candidate(
        X, pairs, seed, attempts, sweeps, lambda_bic=lambda_bic,
        search_version="local_greedy_rust",
        local_greedy_passes=local_greedy_passes)
    incumbent_bic = float(gaussian_bic(X, incumbent, lambda_bic=lambda_bic)[0])
    try:
        raw, trek_diag = flopsearch.flop_notreks(
            X, lambda_bic, pairs, restarts=1, seed=seed,
            search_version="trekcut", trekcut_oracle_budget=oracle_budget,
            trekcut_refinement_passes=refinement_passes,
            return_dag=True, return_diagnostics=True)
        trek_bic = float(gaussian_bic(X, raw, lambda_bic=lambda_bic)[0])
        if trek_bic < incumbent_bic:
            winner = np.asarray(raw, dtype=np.uint8)
            winner_bic = trek_bic
            winner_source = "trekcut"
        else:
            winner = incumbent
            winner_bic = incumbent_bic
            winner_source = "local_incumbent"
        diagnostics = dict(incumbent_diag)
        diagnostics.update({
            "candidate_graph": winner.copy(),
            "cpdag": winner.copy(),
            "optimizer_final_bic": winner_bic,
            "optimizer_violations": common_ancestor_violations(
                winner, pairs),
            "trekcut_winner": winner_source,
            "trekcut_local_incumbent_bic": incumbent_bic,
            "trekcut_search_bic": trek_bic,
            "trekcut_witness_lengths": trek_diag.get(
                "trekcut_witness_lengths", []),
            "trekcut_branch_counts": trek_diag.get(
                "trekcut_branch_counts", []),
            "trekcut_masks": trek_diag.get("trekcut_masks", []),
            "trekcut_archive_bics": trek_diag.get(
                "trekcut_archive_bics", []),
            "trekcut_archive_violations": trek_diag.get(
                "trekcut_archive_violations", []),
            "trekcut_feasible_state_discovery_time": trek_diag.get(
                "trekcut_feasible_state_discovery_time", np.nan),
            "trekcut_oracle_calls": trek_diag.get(
                "trekcut_oracle_calls", 0),
        })
        return winner, diagnostics
    except RuntimeError as error:
        diagnostics = dict(incumbent_diag)
        diagnostics.update({
            "candidate_graph": incumbent.copy(),
            "cpdag": incumbent.copy(),
            "optimizer_final_bic": incumbent_bic,
            "optimizer_violations": common_ancestor_violations(
                incumbent, pairs),
            "trekcut_winner": "local_fallback",
            "trekcut_error": str(error),
            "trekcut_local_incumbent_bic": incumbent_bic,
            "trekcut_search_bic": np.nan,
            "trekcut_oracle_calls": oracle_budget,
        })
        return incumbent, diagnostics


def flop_notreks_prefix_feasible_candidate(
        X, pairs, seed, attempts, sweeps, *, lambda_bic=2.0, beam_width=1):
    """Run the experimental prefix-feasible constructor without repair."""
    raw, diagnostics = flopsearch.flop_notreks(
        X, lambda_bic, pairs, restarts=max(0, attempts - 1), seed=seed,
        max_signature_rounds=sweeps, search_version="prefix_feasible",
        prefix_beam_width=beam_width, return_dag=True,
        return_diagnostics=True)
    graph = _diagnostic_graph(X.shape[1], diagnostics)
    return graph, {
        "candidate_graph": graph.copy(),
        "cpdag": convert_flop_cpdag(raw, X.shape[1]),
        "optimizer_restarts": int(diagnostics["restarts_completed"]),
        "optimizer_final_bic": float(diagnostics["final_bic"]),
        "optimizer_violations": int(diagnostics["final_no_trek_violation_count"]),
        "prefix_admissible_pool_sizes": diagnostics.get(
            "prefix_admissible_pool_sizes", []),
        "prefix_suffix_nodes_rebuilt": diagnostics.get(
            "prefix_suffix_nodes_rebuilt", []),
        "prefix_beam_width": diagnostics.get("prefix_beam_width", beam_width),
        "prefix_beam_alternative_count": diagnostics.get(
            "prefix_beam_alternative_count", 0),
    }


def flop_notreks_trek_dominance_candidate(
        X, pairs, trek_graph, seed, attempts, sweeps, *, lambda_bic=2.0):
    raw, diagnostics = flopsearch.flop_notreks(
        X, lambda_bic, pairs, restarts=max(0, attempts - 1), seed=seed,
        max_signature_rounds=sweeps, search_version="trek_dominance",
        trek_graph=trek_graph,
        return_dag=True, return_diagnostics=True)
    graph = _diagnostic_graph(X.shape[1], diagnostics)
    return graph, {
        "candidate_graph": graph.copy(), "cpdag": convert_flop_cpdag(raw, X.shape[1]),
        "optimizer_restarts": int(diagnostics["restarts_completed"]),
        "optimizer_final_bic": float(diagnostics["final_bic"]),
        "optimizer_violations": int(diagnostics["final_no_trek_violation_count"]),
        "trek_dominance_mask_density": diagnostics.get("trek_dominance_mask_density", np.nan),
    }


def flop_notreks_adaptive_candidate(X, pairs, seed, attempts, *, lambda_bic=2.0):
    result = adaptive_soft_completion(
        X, pairs, total_restarts=attempts, seed=seed)
    graph = result.graph.astype(np.uint8)
    return graph, {
        "candidate_graph": graph.copy(),
        "cpdag": convert_flop_cpdag(graph, X.shape[1]),
        "optimizer_restarts": attempts,
        "optimizer_final_bic": float(result.diagnostics["winner"]["bic"]),
        "optimizer_violations": int(result.diagnostics["winner"]["violations"]),
        "adaptive_completion_states_generated": result.diagnostics[
            "completion_states_generated"],
        "adaptive_completion_states_visited": result.diagnostics[
            "completion_states_visited"],
        "adaptive_completion_winner": result.diagnostics["winner"],
    }


def hybrid_flop_notreks_candidate(X, pairs, seed, attempts, sweeps,
                                   lambda_bic=2.0,
                                   search_version="local_greedy_rust",
                                   local_greedy_passes=8):
    """Pool one vanilla attempt with the remaining constrained attempts.

    The vanilla candidate is retained only when it already satisfies the
    supplied NOTREKS constraints.  The total attempt budget is preserved.
    """
    if attempts < 2:
        return flop_notreks_candidate(
            X, pairs, seed, attempts, sweeps, lambda_bic=lambda_bic,
            search_version=search_version,
            local_greedy_passes=local_greedy_passes)

    vanilla_candidate, vanilla_diag = vanilla_flop_candidate(
        X, seed, 1, lambda_bic=lambda_bic)
    vanilla_bic = float(gaussian_bic(
        X, vanilla_candidate, lambda_bic=lambda_bic)[0])
    vanilla_feasible = common_ancestor_violations(
        vanilla_candidate, pairs) == 0

    constrained_attempts = attempts - 1
    constrained_candidate, constrained_diag = flop_notreks_candidate(
        X, pairs, seed, constrained_attempts, sweeps,
        lambda_bic=lambda_bic, search_version=search_version,
        local_greedy_passes=local_greedy_passes)
    constrained_bic = float(gaussian_bic(
        X, constrained_candidate, lambda_bic=lambda_bic)[0])

    if vanilla_feasible and vanilla_bic <= constrained_bic:
        return vanilla_candidate, {
            **vanilla_diag,
            "candidate_graph": vanilla_candidate.copy(),
            "optimizer_restarts": attempts,
            "hybrid_winner": "vanilla_feasible",
            "hybrid_vanilla_feasible": 1,
            "hybrid_vanilla_bic": vanilla_bic,
            "hybrid_constrained_bic": constrained_bic,
        }
    return constrained_candidate, {
        **constrained_diag,
        "candidate_graph": constrained_candidate.copy(),
        "optimizer_restarts": attempts,
        "hybrid_winner": "local_notreks",
        "hybrid_vanilla_bic": vanilla_bic,
        "hybrid_constrained_bic": constrained_bic,
        "hybrid_vanilla_feasible": int(vanilla_feasible),
    }


def direct_mask_edges(pairs):
    return sorted({(int(left), int(right)) for left, right in pairs} |
                  {(int(right), int(left)) for left, right in pairs})


def ordinary_parent_shrink(X, candidate, lambda_bic=2.0):
    final, _, _ = end_flop_prune(X, np.asarray(candidate, dtype=np.uint8),
                                 lambda_bic=lambda_bic)
    return final


def notreks_postselection(X, candidate, pairs, lambda_bic=2.0):
    """NOTREKS-informed repair followed by ordinary parent shrinking."""
    graph = np.asarray(candidate, dtype=np.uint8).copy()
    while common_ancestor_violations(graph, pairs):
        before = common_ancestor_violations(graph, pairs)
        best = None
        for source, target in zip(*np.nonzero(graph)):
            proposal = graph.copy()
            proposal[source, target] = 0
            reduction = before - common_ancestor_violations(proposal, pairs)
            key = (-int(reduction), int(source), int(target))
            if best is None or key < best[0]:
                best = (key, int(source), int(target))
        if best is None:
            raise RuntimeError("cannot repair vanilla FLOP candidate")
        graph[best[1], best[2]] = 0
    final, _, _ = end_flop_prune(X, graph, lambda_bic=lambda_bic)
    if common_ancestor_violations(final, pairs):
        raise RuntimeError("NOTREKS postselection violated NOTREKS")
    return final


def hybrid_flop_notreks_postselected_candidate(X, pairs, seed, attempts,
                                               sweeps, lambda_bic=2.0):
    """Use one vanilla attempt, then the remaining constrained attempts."""
    vanilla_candidate, vanilla_diag = vanilla_flop_candidate(
        X, seed, 1, lambda_bic=lambda_bic)
    vanilla_final = notreks_postselection(
        X, vanilla_candidate, pairs, lambda_bic=lambda_bic)
    vanilla_bic = float(gaussian_bic(
        X, vanilla_final, lambda_bic=lambda_bic)[0])

    constrained_attempts = attempts - 1
    _, constrained_diag = flopsearch.flop_notreks(
        X, lambda_bic, pairs,
        restarts=max(0, constrained_attempts - 1),
        seed=seed,
        max_signature_rounds=sweeps,
        search_version="global_greedy_rust",
        return_diagnostics=True,
    )
    constrained_final = _diagnostic_graph(X.shape[1], constrained_diag)
    constrained_bic = float(
        gaussian_bic(X, constrained_final, lambda_bic=lambda_bic)[0])

    if vanilla_bic <= constrained_bic:
        return vanilla_final, {
            "candidate_graph": vanilla_candidate.copy(),
            "cpdag": vanilla_final.copy(),
            "optimizer_restarts": attempts,
            "hybrid_winner": "vanilla_postselected",
            "hybrid_vanilla_bic": vanilla_bic,
            "hybrid_constrained_bic": constrained_bic,
        }
    return constrained_final, {
        "candidate_graph": constrained_final.copy(),
        "cpdag": constrained_final.copy(),
        "optimizer_restarts": attempts,
        "hybrid_winner": "constrained_notreks",
        "hybrid_vanilla_bic": vanilla_bic,
        "hybrid_constrained_bic": constrained_bic,
    }


def flop_notreks_postselection_candidate(X, pairs, seed, attempts,
                                         lambda_bic=2.0):
    """Vanilla FLOP followed by NOTREKS-informed post-selection only."""
    candidate, diagnostics = vanilla_flop_candidate(
        X, seed, attempts, lambda_bic=lambda_bic)
    final, post_diag = flop_notreks_postselection_from_candidate(
        X, candidate, pairs, lambda_bic=lambda_bic)
    return final, {
        **post_diag,
        "candidate_graph": candidate.copy(),
        "optimizer_restarts": diagnostics["optimizer_restarts"],
        "postselection_policy": "notreks_repair_parent_shrink",
    }


def flop_notreks_postselection_from_candidate(X, candidate, pairs,
                                              lambda_bic=2.0):
    """Apply FLOP's NOTREKS post-selection to an existing candidate.

    This deliberately contains no FLOP optimization.  It is used when a
    benchmark cell already cached the vanilla FLOP result for this data
    instance, so the measured runtime is the additional post-selection cost.
    """
    final = notreks_postselection(
        X, candidate, pairs, lambda_bic=lambda_bic)
    return final, {
        "cpdag": final.copy(),
        "postselection_policy": "notreks_repair_parent_shrink",
    }


def select_knowledge(pairs, fraction, seed):
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("knowledge_fraction must lie in [0, 1]")
    if fraction >= 1.0:
        return list(pairs)
    if fraction <= 0.0:
        return []
    rng = np.random.default_rng(seed + 104729)
    count = int(round(fraction * len(pairs)))
    selected = rng.choice(len(pairs), size=count, replace=False)
    return [pairs[int(index)] for index in np.sort(selected)]


def corrupt_knowledge(pairs, all_pairs, d, fraction, seed):
    """Replace a fraction of supplied true pairs by genuinely false pairs.

    ``all_pairs`` is the complete true NOTREKS set.  Its complement among
    unordered vertex pairs is exactly the trek graph, so replacements sampled
    from that complement are guaranteed not to be true no-trek constraints.
    The number of supplied pairs is preserved.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("corrupted_knowledge_fraction must lie in [0, 1]")
    pairs = [(int(left), int(right)) for left, right in pairs]
    true_pairs = {tuple(sorted((int(left), int(right))))
                  for left, right in all_pairs}
    known = {tuple(sorted(pair)) for pair in pairs}
    if len(known) != len(pairs):
        raise ValueError("NOTREKS pairs must be unique")
    false_pool = [pair for pair in combinations(range(d), 2)
                  if pair not in true_pairs]
    replace_count = int(round(fraction * len(pairs)))
    if replace_count == 0:
        return pairs
    if replace_count > len(false_pool):
        raise ValueError(
            "not enough genuinely false pairs for requested corruption")
    rng = np.random.default_rng(seed + 15485863)
    removed_indices = set(rng.choice(
        len(pairs), size=replace_count, replace=False).tolist())
    replacement_indices = rng.choice(
        len(false_pool), size=replace_count, replace=False)
    retained = [pair for index, pair in enumerate(pairs)
                if index not in removed_indices]
    retained.extend(false_pool[int(index)] for index in replacement_indices)
    return sorted(retained)


def dagma_mask(d, pairs):
    mask = np.ones((d, d), dtype=float)
    for left, right in direct_mask_edges(pairs):
        mask[left, right] = 0.0
    np.fill_diagonal(mask, 0.0)
    return mask


def dagma_candidate(X, pairs, use_notreks, direct_mask, seed, attempts,
                    warm_iter, max_iter, stages,
                    apply_notreks_postselection=False,
                    trek_weight=1.0,
                    initialization_mode="empty_random",
                    initialization_edge_probability=0.15,
                    record_trajectory=False, proximal_l1=False,
                    postselection_policy="feasible_parent_shrink",
                    mu_schedule=None, s_schedule=None):
    config = ProductionConfig(
        restarts=attempts,
        seed=seed,
        T=stages,
        warm_iter=warm_iter,
        max_iter=max_iter,
        trek_weight=trek_weight,
        initialization_mode=initialization_mode,
        initialization_edge_probability=initialization_edge_probability,
        proximal_l1=proximal_l1,
        dagma_postselection_policy=postselection_policy,
        mu_schedule=(tuple(mu_schedule) if mu_schedule is not None else None),
        s=(tuple(s_schedule) if s_schedule is not None else (1.0, .9, .8, .7, .6)),
        record_trajectory=record_trajectory,
        edge_mask=dagma_mask(X.shape[1], pairs) if direct_mask else None,
    )
    result, all_restarts = run_production_pipeline(
        X, pairs if use_notreks else [], config)
    adjacency = result.adjacency
    if apply_notreks_postselection and not use_notreks:
        adjacency = notreks_postselection(X, adjacency, pairs)
    selected_stages = result.stage_diagnostics
    return adjacency, {
        "candidate_graph": result.candidate_graph.copy(),
        "cpdag": adjacency.copy(),
        "candidate_edges": result.candidate_edges,
        "final_edges": result.final_edges,
        "postselection_policy": "feasible_parent_shrink",
        "optimizer_restarts": len(all_restarts),
        "optimizer_iterations_T": stages,
        "optimizer_warm_iter": warm_iter,
        "optimizer_max_iter": max_iter,
        "proximal_l1": bool(proximal_l1),
        "selected_stage": result.selected_stage,
        "selected_restart": result.restart,
        "checkpoint_selection": [
            {
                "restart": candidate.restart,
                "stage": candidate.selected_stage,
                "bic": candidate.exact_bic,
                "edges": candidate.final_edges,
                "violations": candidate.oracle_violations,
                "adjacency": candidate.adjacency.tolist(),
            }
            for restart in all_restarts
            for candidate in (restart.checkpoint_candidates or [])],
        "dagma_stages_exhausted": int(sum(
            bool(stage.get("exhausted_budget", False))
            for stage in selected_stages)),
        "dagma_stages_stopped_by_tolerance": int(sum(
            bool(stage.get("stopped_by_tolerance", False))
            for stage in selected_stages)),
        "trajectory": (
            [result.stage_adjacencies for result in all_restarts]
            if record_trajectory else None),
        "trajectory_times": (
            [result.trajectory_times for result in all_restarts]
            if record_trajectory else None),
    }


def cpdag_shd(truth, estimate):
    with tempfile.TemporaryDirectory(prefix="notreks_cpdag_eval_") as tmp:
        tmp = Path(tmp)
        true_path = tmp / "true.csv"
        estimate_path = tmp / "estimate.csv"
        output_path = tmp / "metrics.csv"
        pd.DataFrame(truth.astype(int)).to_csv(true_path, index=False)
        pd.DataFrame(estimate.astype(int)).to_csv(estimate_path, index=False)
        subprocess.run(
            ["Rscript", str(R_EVAL), "--adjmat_true", str(true_path),
             "--adjmat_est", str(estimate_path), "--filename",
             str(output_path)], check=True, capture_output=True, text=True)
        return float(pd.read_csv(output_path).iloc[0]["SHD_cpdag"])


def metrics(data, truth, pairs, method, candidate, estimate, estimate_cpdag,
            candidate_runtime, attempts_requested, attempts_completed,
            lambda_bic=2.0):
    truth_bool = np.asarray(truth, dtype=bool)
    estimate_bool = np.asarray(estimate, dtype=bool)
    truth_skeleton = truth_bool | truth_bool.T
    estimate_skeleton = estimate_bool | estimate_bool.T
    directed_tp = int(np.sum(truth_bool & estimate_bool))
    directed_fp = int(np.sum(~truth_bool & estimate_bool))
    directed_fn = int(np.sum(truth_bool & ~estimate_bool))
    directed_f1 = (2 * directed_tp /
                   max(1, 2 * directed_tp + directed_fp + directed_fn))
    directed_precision = directed_tp / max(1, directed_tp + directed_fp)
    directed_recall = directed_tp / max(1, directed_tp + directed_fn)
    tp = int(np.sum(truth_skeleton & estimate_skeleton) // 2)
    fp = int(np.sum(~truth_skeleton & estimate_skeleton) // 2)
    fn = int(np.sum(truth_skeleton & ~estimate_skeleton) // 2)
    f1_skel = 2 * tp / max(1, 2 * tp + fp + fn)
    skeleton_precision = tp / max(1, tp + fp)
    skeleton_recall = tp / max(1, tp + fn)
    candidate_bic = float(gaussian_bic(
        data, candidate, lambda_bic=lambda_bic)[0])
    final_bic = float(gaussian_bic(
        data, estimate, lambda_bic=lambda_bic)[0])
    # These are DAG-level causal distances.  The CPDAG SHD above remains the
    # primary equivalence-class metric; do not feed the CPDAG into a DAG
    # distance implementation.
    parent_aid = dag_parent_aid(truth_bool, estimate_bool)
    ancestor_aid = dag_ancestor_aid(truth_bool, estimate_bool)
    parent_cpdag_norm, parent_cpdag = cpdag_aid(
        truth_bool, estimate_cpdag, "parent")
    ancestor_cpdag_norm, ancestor_cpdag = cpdag_aid(
        truth_bool, estimate_cpdag, "ancestor")
    return {
        "method": method,
        "SHD_cpdag": cpdag_shd(truth, estimate_cpdag),
        "SID_dag": int(dag_sid(truth_bool, estimate_bool)),
        "Parent_AID_dag": int(parent_aid),
        "Ancestor_AID_dag": int(ancestor_aid),
        "Parent_AID_cpdag": int(parent_cpdag),
        "Parent_AID_cpdag_normalized": float(parent_cpdag_norm),
        "Ancestor_AID_cpdag": int(ancestor_cpdag),
        "Ancestor_AID_cpdag_normalized": float(ancestor_cpdag_norm),
        "F1_skel": float(f1_skel),
        "precision_skel": float(skeleton_precision),
        "recall_skel": float(skeleton_recall),
        "skeleton_SHD": int(fp + fn),
        "skeleton_TP": tp,
        "skeleton_FP": fp,
        "skeleton_FN": fn,
        "directed_SHD": int(directed_fp + directed_fn),
        "directed_TP": directed_tp,
        "directed_FP": directed_fp,
        "directed_FN": directed_fn,
        "F1_directed": float(directed_f1),
        "precision_directed": float(directed_precision),
        "recall_directed": float(directed_recall),
        "true_edges": int(truth_bool.sum()),
        "candidate_edges": int(np.asarray(candidate, dtype=bool).sum()),
        "final_edges": int(estimate_bool.sum()),
        "dag_feasible": bool(is_dag(estimate_bool)),
        "violations_before": int(
            common_ancestor_violations(candidate, pairs)),
        "violations_after": int(
            common_ancestor_violations(estimate, pairs)),
        "trek_violations_before": int(
            common_ancestor_violations(candidate, pairs)),
        "trek_violations_after": int(
            common_ancestor_violations(estimate, pairs)),
        "trek_violation_mass_before": trek_violation_mass(candidate, pairs),
        "trek_violation_mass_after": trek_violation_mass(estimate, pairs),
        "candidate_runtime": float(candidate_runtime),
        "attempts_requested": int(attempts_requested),
        "attempts_completed": int(attempts_completed),
        "candidate_bic": candidate_bic,
        "final_bic": final_bic,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--n", type=int, default=None,
                        help="sample size; defaults to 10*d")
    parser.add_argument("--graph-type", default="er2")
    parser.add_argument("--scm", default="linear")
    parser.add_argument("--noise", default="gaussian")
    parser.add_argument("--knowledge-fraction", type=float, default=0.25)
    parser.add_argument("--corrupted-knowledge-fraction", type=float, default=0.0,
                        help="fraction of known pairs replaced by true trek pairs")
    parser.add_argument(
        "--methods", nargs="+",
        choices=SUPPORTED_METHODS,
        default=DEFAULT_METHODS)
    parser.add_argument("--restarts", type=int, default=5,
                        help="total solver attempts per method")
    parser.add_argument("--flop-sweeps", type=int, default=16)
    parser.add_argument("--flop-lambda-bic", type=float, default=2.0)
    parser.add_argument(
        "--flop-search-version",
        choices=("global_greedy_rust", "global_greedy_rust_optimized",
                 "local_greedy_rust"),
        default="local_greedy_rust")
    parser.add_argument("--flop-local-passes", type=int, default=8,
                        help="target-wise passes for flop_notreks_local")
    parser.add_argument("--flop-order-guided-lex-fraction", type=float,
                        default=0.5)
    parser.add_argument("--flop-order-guided-repair-candidates", type=int,
                        default=2)
    parser.add_argument("--flop-order-guided-coverage-starts", type=int,
                        default=1)
    parser.add_argument("--flop-trekcut-oracle-budget", type=int, default=32)
    parser.add_argument("--flop-trekcut-refinement-passes", type=int, default=4)
    parser.add_argument("--flop-prefix-beam-width", type=int, choices=(1, 4), default=1)
    parser.add_argument("--flop-source-signature-polish-outputs", type=int,
                        default=1)
    parser.add_argument("--dagma-stages", type=int, default=5)
    parser.add_argument("--dagma-warm-iter", type=int, default=30000)
    parser.add_argument("--dagma-max-iter", type=int, default=60000)
    parser.add_argument("--dagma-trek-weight", type=float, default=1.0)
    parser.add_argument(
        "--dagma-initialization-mode",
        choices=("empty_random", "empty_feasible_random"),
        default="empty_random")
    parser.add_argument("--dagma-initialization-edge-probability",
                        type=float, default=0.15)
    parser.add_argument("--dagma-log-trajectory", action="store_true",
                        help="save DAGMA initial and continuation-stage matrices")
    parser.add_argument(
        "--dagma-mu-schedule", default=None,
        help="comma-separated continuation values, e.g. 1,0.3,0.1,0.03,0.01")
    parser.add_argument(
        "--dagma-s-schedule", default=None,
        help="comma-separated DAGMA domain values, e.g. 1,0.95,0.9,0.85,0.8")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    parse_schedule = lambda value: (
        tuple(float(item.strip()) for item in value.split(","))
        if value else None)
    config = BenchmarkConfig(
        d=args.d, n=args.n, graph_type=args.graph_type, scm=args.scm,
        noise=args.noise, knowledge_fraction=args.knowledge_fraction,
        corrupted_knowledge_fraction=args.corrupted_knowledge_fraction,
        attempts=args.restarts, flop_sweeps=args.flop_sweeps,
        flop_lambda_bic=args.flop_lambda_bic,
        flop_search_version=args.flop_search_version,
        flop_local_passes=args.flop_local_passes,
        flop_order_guided_lex_fraction=args.flop_order_guided_lex_fraction,
        flop_order_guided_repair_candidates=(
            args.flop_order_guided_repair_candidates),
        flop_order_guided_coverage_starts=(
            args.flop_order_guided_coverage_starts),
        flop_trekcut_oracle_budget=args.flop_trekcut_oracle_budget,
        flop_trekcut_refinement_passes=args.flop_trekcut_refinement_passes,
        flop_prefix_beam_width=args.flop_prefix_beam_width,
        flop_source_signature_polish_outputs=(
            args.flop_source_signature_polish_outputs),
        dagma_stages=args.dagma_stages,
        dagma_warm_iter=args.dagma_warm_iter,
        dagma_max_iter=args.dagma_max_iter,
        dagma_trek_weight=args.dagma_trek_weight,
        dagma_initialization_mode=args.dagma_initialization_mode,
        dagma_initialization_edge_probability=(
            args.dagma_initialization_edge_probability),
        dagma_record_trajectory=args.dagma_log_trajectory,
        dagma_mu_schedule=parse_schedule(args.dagma_mu_schedule),
        dagma_s_schedule=parse_schedule(args.dagma_s_schedule))
    config.validate(args.methods)
    select_knowledge([], config.knowledge_fraction, 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = args.output_dir / "graphs"
    graph_dir.mkdir(exist_ok=True)
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps({
            "d": config.d, "n": config.effective_n,
            "graph_type": config.graph_type, "scm": config.scm,
            "noise": config.noise, "equal_variance": config.equal_variance,
            "knowledge_fraction": config.knowledge_fraction,
            "corrupted_knowledge_fraction": (
                config.corrupted_knowledge_fraction),
            "attempts": config.attempts,
            "flop_sweeps": config.flop_sweeps,
            "flop_lambda_bic": config.flop_lambda_bic,
            "flop_search_version": config.flop_search_version,
            "flop_order_guided_lex_fraction": (
                config.flop_order_guided_lex_fraction),
            "flop_order_guided_repair_candidates": (
                config.flop_order_guided_repair_candidates),
            "flop_order_guided_coverage_starts": (
                config.flop_order_guided_coverage_starts),
            "flop_trekcut_oracle_budget": config.flop_trekcut_oracle_budget,
            "flop_trekcut_refinement_passes": (
                config.flop_trekcut_refinement_passes),
            "flop_source_signature_polish_outputs": (
                config.flop_source_signature_polish_outputs),
            "dagma_stages": config.dagma_stages,
            "dagma_warm_iter": config.dagma_warm_iter,
            "dagma_max_iter": config.dagma_max_iter,
            "dagma_trek_weight": config.dagma_trek_weight,
            "dagma_initialization_mode": config.dagma_initialization_mode,
            "dagma_initialization_edge_probability": (
                config.dagma_initialization_edge_probability),
            "dagma_mu_schedule": config.dagma_mu_schedule,
            "dagma_s_schedule": config.dagma_s_schedule,
            "dagma_log_trajectory": config.dagma_record_trajectory,
            "methods": list(args.methods),
        }, indent=2) + "\n")
    rows = []
    for seed_index, seed in enumerate(args.seeds):
        X, truth, all_pairs = generate(
            seed, d=config.d, n=config.effective_n,
            graph_type=config.graph_type, scm=config.scm,
            noise=config.noise)
        pairs = select_knowledge(
            all_pairs, config.knowledge_fraction, seed)
        clean_pairs = list(pairs)
        pairs = corrupt_knowledge(
            pairs, all_pairs, config.d, config.corrupted_knowledge_fraction,
            seed)
        true_pair_set = {tuple(sorted((int(left), int(right))))
                         for left, right in all_pairs}
        partial_chromatic = chromatic_upper_bound(config.d, pairs)
        full_chromatic = chromatic_upper_bound(config.d, all_pairs)
        candidates = {}
        for method in args.methods:
            started = time.perf_counter()
            if method == "flop":
                candidate, optimizer_diag = vanilla_flop_candidate(
                    X, seed, config.attempts, config.flop_lambda_bic)
            elif method in {"flop_parent_shrink",
                            "flop_edge_mask_parent_shrink"}:
                if method == "flop_parent_shrink":
                    vanilla_candidate, optimizer_diag = vanilla_flop_candidate(
                        X, seed, config.attempts, config.flop_lambda_bic)
                else:
                    vanilla_candidate, optimizer_diag = flop_notreks_candidate(
                        X, [], seed, config.attempts, config.flop_sweeps,
                        forbidden_edges=direct_mask_edges(pairs),
                        lambda_bic=config.flop_lambda_bic)
                candidate = ordinary_parent_shrink(
                    X, vanilla_candidate, config.flop_lambda_bic)
                optimizer_diag = dict(optimizer_diag)
                optimizer_diag["candidate_graph"] = vanilla_candidate.copy()
                optimizer_diag["cpdag"] = candidate.copy()
            elif method == "flop_edge_mask":
                candidate, optimizer_diag = flop_notreks_candidate(
                    X, [], seed, config.attempts, config.flop_sweeps,
                    forbidden_edges=direct_mask_edges(pairs),
                    lambda_bic=config.flop_lambda_bic)
            elif method == "flop_notreks":
                candidate, optimizer_diag = flop_notreks_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic,
                    search_version="flop_like",
                    local_greedy_passes=config.flop_local_passes)
            elif method == "flop_notreks_greedy":
                candidate, optimizer_diag = flop_notreks_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic,
                    search_version="local_greedy_rust",
                    local_greedy_passes=config.flop_local_passes)
            elif method == "flop_notreks_postselection":
                candidate, optimizer_diag = (
                    flop_notreks_postselection_candidate(
                        X, pairs, seed, config.attempts,
                        lambda_bic=config.flop_lambda_bic))
            elif method == "flop_notreks_local":
                candidate, optimizer_diag = flop_notreks_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic,
                    search_version="local_greedy_rust",
                    local_greedy_passes=config.flop_local_passes)
            elif method == "flop_notreks_active_exact":
                candidate, optimizer_diag = flop_notreks_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic,
                    search_version="local_greedy_active_exact",
                    local_greedy_passes=config.flop_local_passes)
            elif method == "flop_notreks_active_reinsert":
                candidate, optimizer_diag = flop_notreks_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic,
                    search_version="local_greedy_active_reinsert",
                    local_greedy_passes=config.flop_local_passes)
            elif method == "flop_notreks_active_exact_edge_mask":
                candidate, optimizer_diag = flop_notreks_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    forbidden_edges=direct_mask_edges(pairs),
                    lambda_bic=config.flop_lambda_bic,
                    search_version="local_greedy_active_exact",
                    local_greedy_passes=config.flop_local_passes)
            elif method == "flop_notreks_local_adaptive":
                candidate, optimizer_diag = flop_notreks_local_adaptive_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic,
                    local_greedy_passes=config.flop_local_passes)
            elif method == "flop_notreks_order_guided":
                candidate, optimizer_diag = flop_notreks_order_guided_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic,
                    local_greedy_passes=config.flop_local_passes,
                    lex_fraction=config.flop_order_guided_lex_fraction,
                    repair_candidates=(
                        config.flop_order_guided_repair_candidates),
                    coverage_starts=(
                        config.flop_order_guided_coverage_starts))
            elif method == "flop_notreks_trekcut":
                candidate, optimizer_diag = flop_notreks_trekcut_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic,
                    local_greedy_passes=config.flop_local_passes,
                    oracle_budget=config.flop_trekcut_oracle_budget,
                    refinement_passes=(
                        config.flop_trekcut_refinement_passes))
            elif method == "flop_notreks_prefix_feasible":
                candidate, optimizer_diag = flop_notreks_prefix_feasible_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic,
                    beam_width=config.flop_prefix_beam_width)
            elif method == "flop_notreks_trek_dominance":
                candidate, optimizer_diag = flop_notreks_trek_dominance_candidate(
                    X, pairs, trek_graph_edges(truth), seed,
                    config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic)
            elif method == "flop_notreks_adaptive_soft_completion":
                candidate, optimizer_diag = flop_notreks_adaptive_candidate(
                    X, pairs, seed, config.attempts,
                    lambda_bic=config.flop_lambda_bic)
            elif method == "flop_notreks_source_signature":
                candidate, optimizer_diag = (
                    flop_notreks_source_signature_candidate(
                        X, pairs, seed, config.attempts, config.flop_sweeps,
                        lambda_bic=config.flop_lambda_bic,
                        local_greedy_passes=config.flop_local_passes,
                        polish_outputs=(
                            config.flop_source_signature_polish_outputs)))
            elif method == "flop_notreks_random_order":
                candidate, optimizer_diag = flop_notreks_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    random_initial_order=True,
                    lambda_bic=config.flop_lambda_bic,
                    search_version=config.flop_search_version)
            elif method == "flop_notreks_warm":
                candidate, optimizer_diag = hybrid_flop_notreks_postselected_candidate(
                    X, pairs, seed, config.attempts, config.flop_sweeps,
                    lambda_bic=config.flop_lambda_bic)
            else:
                # Projection variants are intended as DAGMA+NOTREKS
                # postselection experiments.  They must receive the same
                # supplied pairs during optimization as dagma_notreks.
                use_notreks = (
                    method.startswith("dagma_notreks")
                    or "normalized_projection" in method)
                proximal_l1 = "proximal" in method
                if method.endswith("best_projected_checkpoint"):
                    postselection_policy = "best_projected_checkpoint"
                elif "normalized_projection_shrink" in method:
                    postselection_policy = (
                        "normalized_greedy_projection_refit_shrink")
                elif "normalized_projection" in method:
                    postselection_policy = (
                        "normalized_greedy_projection_refit")
                else:
                    postselection_policy = "feasible_parent_shrink"
                direct_mask = method in {"dagma_edge_mask",
                                         "dagma_edge_mask_postselection",
                                         "dagma_notreks_edge_mask"}
                apply_postselection = method in {
                    "dagma_postselection",
                    "dagma_edge_mask_postselection"}
                candidate, optimizer_diag = dagma_candidate(
                    X, pairs, use_notreks, direct_mask, seed,
                    config.attempts, config.dagma_warm_iter,
                    config.dagma_max_iter, config.dagma_stages,
                    trek_weight=config.dagma_trek_weight,
                    initialization_mode=config.dagma_initialization_mode,
                    initialization_edge_probability=(
                        config.dagma_initialization_edge_probability),
                    record_trajectory=config.dagma_record_trajectory,
                    proximal_l1=proximal_l1,
                    postselection_policy=postselection_policy,
                    mu_schedule=config.dagma_mu_schedule,
                    s_schedule=config.dagma_s_schedule,
                    apply_notreks_postselection=apply_postselection)
            candidates[method] = (
                candidate, optimizer_diag, time.perf_counter() - started)
            if optimizer_diag.get("trajectory") is not None:
                trajectory_dir = args.output_dir / "dagma_trajectories"
                trajectory_dir.mkdir(exist_ok=True)
                payload = {
                    "X": np.asarray(X, dtype=np.float64),
                    "truth": np.asarray(truth, dtype=np.uint8),
                    "pairs": np.asarray(pairs, dtype=np.int64),
                }
                for restart, matrices in enumerate(
                        optimizer_diag["trajectory"]):
                    payload[f"restart_{restart}"] = np.asarray(
                        matrices, dtype=np.float64)
                    payload[f"restart_{restart}_times"] = np.asarray(
                        optimizer_diag["trajectory_times"][restart],
                        dtype=np.float64)
                np.savez_compressed(
                    trajectory_dir / f"seed_{seed}_{method}.npz", **payload)

        for method, (candidate, optimizer_diag,
                     candidate_runtime) in candidates.items():
            row = metrics(
                X, truth, pairs, method, optimizer_diag["candidate_graph"],
                candidate, optimizer_diag["cpdag"], candidate_runtime,
                config.attempts, optimizer_diag["optimizer_restarts"],
                config.flop_lambda_bic)
            row.update({
                "seed": seed,
                "d": config.d,
                "n": config.effective_n,
                "graph_type": config.graph_type,
                "scm": config.scm,
                "noise": config.noise,
                "knowledge_fraction": config.knowledge_fraction,
                "corrupted_knowledge_fraction": (
                    config.corrupted_knowledge_fraction),
                "attempts_requested": config.attempts,
                "flop_lambda_bic": config.flop_lambda_bic,
                "dagma_trek_weight": config.dagma_trek_weight,
                "notreks_pairs": len(pairs),
                "clean_notreks_pairs": len(clean_pairs),
                "corrupted_notreks_pairs": sum(
                    tuple(sorted(pair)) not in true_pair_set for pair in pairs),
                "chromatic_partial_dsat_upper": partial_chromatic,
                "chromatic_full_dsat_upper": full_chromatic,
            })
            for key in (
                    "order_guided_raw_feasible_count",
                    "order_guided_repaired_count",
                    "order_guided_winner",
                    "order_guided_archive_max_trek_violations",
                    "order_guided_archive_max_trek_violation_mass"):
                if key in optimizer_diag:
                    row[key] = optimizer_diag[key]
            for key in (
                    "trekcut_winner", "trekcut_local_incumbent_bic",
                    "trekcut_search_bic", "trekcut_witness_lengths",
                    "trekcut_branch_counts", "trekcut_masks",
                    "trekcut_archive_bics", "trekcut_archive_violations",
                    "trekcut_feasible_state_discovery_time",
                    "trekcut_oracle_calls"):
                if key in optimizer_diag:
                    value = optimizer_diag[key]
                    row[key] = (json.dumps(value) if isinstance(value, list)
                                else value)
            for key in ("selected_stage", "selected_restart",
                        "checkpoint_selection"):
                if key in optimizer_diag:
                    value = optimizer_diag[key]
                    if key == "checkpoint_selection":
                        value = [
                            {
                                **item,
                                "SHD_cpdag": cpdag_shd(
                                    truth,
                                    convert_flop_cpdag(
                                        np.asarray(item["adjacency"],
                                                   dtype=np.uint8),
                                        config.d)),
                            }
                            for item in value]
                        selected = next(
                            (item for item in value
                             if item["stage"] == optimizer_diag.get(
                                 "selected_stage")
                             and item["restart"] == optimizer_diag.get(
                                 "selected_restart")), None)
                        if selected is not None:
                            optimizer_diag["selected_stage_SHD_cpdag"] = (
                                selected["SHD_cpdag"])
                        value = [
                            {key2: value2 for key2, value2 in item.items()
                             if key2 != "adjacency"}
                            for item in value]
                        optimizer_diag["checkpoint_stage_SHD_cpdag"] = value
                    row[key] = (json.dumps(value) if isinstance(value, list)
                                else value)
            if "checkpoint_stage_SHD_cpdag" in optimizer_diag:
                row["checkpoint_stage_SHD_cpdag"] = json.dumps(
                    optimizer_diag["checkpoint_stage_SHD_cpdag"])
                row["selected_stage_SHD_cpdag"] = optimizer_diag.get(
                    "selected_stage_SHD_cpdag")
            for key in (
                    "source_signature_source_count",
                    "source_signature_mask_density",
                    "source_signature_pair_check_passed",
                    "source_signature_polish_outputs",
                    "source_signature_polishing_changed",
                    "source_signature_winner",
                    "source_signature_polish_bic",
                    "source_signature_local_bic",
                    "source_signature_forbidden_edges"):
                if key in optimizer_diag:
                    value = optimizer_diag[key]
                    row[key] = (json.dumps(value) if isinstance(value, list)
                                else value)
            row["order_guided_raw_trek_violations"] = int(
                optimizer_diag.get("order_guided_archive_max_trek_violations", 0))
            row["order_guided_raw_trek_violation_mass"] = int(
                optimizer_diag.get(
                    "order_guided_archive_max_trek_violation_mass", 0))
            if "order_guided_violation_trajectory" in optimizer_diag:
                row["order_guided_violation_trajectory"] = json.dumps(
                    optimizer_diag["order_guided_violation_trajectory"])
            rows.append(row)
        graph_payload = {
            "truth": np.asarray(truth, dtype=np.uint8),
            "true_notreks_pairs": np.asarray(all_pairs, dtype=np.int64),
            "clean_notreks_pairs": np.asarray(clean_pairs, dtype=np.int64),
            "notreks_pairs": np.asarray(pairs, dtype=np.int64),
        }
        for method, (candidate, optimizer_diag, _) in candidates.items():
            graph_payload[f"{method}__candidate"] = np.asarray(
                optimizer_diag["candidate_graph"], dtype=np.uint8)
            graph_payload[f"{method}__final"] = np.asarray(
                candidate, dtype=np.uint8)
            graph_payload[f"{method}__cpdag"] = np.asarray(
                optimizer_diag["cpdag"], dtype=np.uint8)
        np.savez_compressed(graph_dir / f"seed_{seed}.npz", **graph_payload)
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output_dir / "per_seed.csv", index=False)
        # Progress output is intentionally per-seed.  Cumulative means are
        # useful in the final summary, but are easy to confuse with averages
        # over restarts or with the result of the current seed.
        live = frame[frame.seed == seed].copy()
        live_columns = ["seed", "method", "SHD_cpdag", "Parent_AID_cpdag",
                     "Ancestor_AID_cpdag", "SID_dag", "Parent_AID_dag",
                     "Ancestor_AID_dag", "F1_skel",
                     "F1_directed", "candidate_runtime",
                     "attempts_completed"]
        if "selected_stage" in live:
            live_columns.append("selected_stage")
        if "selected_stage_SHD_cpdag" in live:
            live_columns.append("selected_stage_SHD_cpdag")
        if "checkpoint_stage_SHD_cpdag" in live:
            live_columns.append("checkpoint_stage_SHD_cpdag")
        live = live[live_columns].sort_values("method")
        live.to_csv(args.output_dir / "live_current_seed.csv", index=False)
        progress = f"[{seed_index + 1}/{len(args.seeds)}]"
        print(progress, f"seed={seed} completed", flush=True)
        print(live.to_string(index=False), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_seed.csv", index=False)
    summary = frame.groupby(["method"], as_index=False).agg(
            SHD_cpdag_mean=("SHD_cpdag", "mean"),
            SHD_cpdag_std=("SHD_cpdag", "std"),
            SID_dag_mean=("SID_dag", "mean"),
            Parent_AID_dag_mean=("Parent_AID_dag", "mean"),
            Ancestor_AID_dag_mean=("Ancestor_AID_dag", "mean"),
            Parent_AID_cpdag_mean=("Parent_AID_cpdag", "mean"),
            Ancestor_AID_cpdag_mean=("Ancestor_AID_cpdag", "mean"),
            directed_SHD_mean=("directed_SHD", "mean"),
            directed_SHD_std=("directed_SHD", "std"),
            F1_skel_mean=("F1_skel", "mean"),
            F1_skel_std=("F1_skel", "std"),
            F1_directed_mean=("F1_directed", "mean"),
            F1_directed_std=("F1_directed", "std"),
            chromatic_partial_dsat_upper_mean=(
                "chromatic_partial_dsat_upper", "mean"),
            chromatic_full_dsat_upper_mean=(
                "chromatic_full_dsat_upper", "mean"),
            violations_before_mean=("violations_before", "mean"),
            violations_after_max=("violations_after", "max"),
            trek_violations_before_mean=("trek_violations_before", "mean"),
            trek_violations_after_max=("trek_violations_after", "max"),
            trek_violation_mass_before_mean=(
                "trek_violation_mass_before", "mean"),
            trek_violation_mass_after_max=(
                "trek_violation_mass_after", "max"),
            runtime_mean=("candidate_runtime", "mean"),
            attempts_completed_min=("attempts_completed", "min"),
            attempts_completed_max=("attempts_completed", "max"),
        )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
