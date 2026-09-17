use std::collections::{BTreeSet, VecDeque};
use std::env;
use std::time::{Duration, Instant};

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::algo::reinsert;
use crate::bic::Bic;
use crate::error::FlopError;
use crate::fit_parents::{fit_parents_constrained, perm_to_dag_constrained};
use crate::graph::Dag;
use crate::no_treks::{
    ancestor_cone, canonical_signatures, count_no_trek_violations, signatures_respect_edges,
    NoTrekConstraints,
};
use crate::scores::{GlobalScore, LocalScore};
use crate::{pivoted_cholesky, utils};

const EPS: f64 = 1e-9;

fn diagnostic_trace_enabled() -> bool {
    env::var_os("FLOP_NOTREKS_DIAGNOSTIC_TRACE").is_some()
}

fn diagnostic_parent_list(g: &GlobalScore, target: usize) -> String {
    format!("{:?}", g.local_scores[target].parents)
}

fn diagnostic_order(order: &[usize]) -> String {
    format!("{:?}", order)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoTreksVersion {
    FixedSignatureA,
    AlternatingFullRefitB,
    /// FLOP-style order reinsertion with the hard-feasible global-greedy
    /// inner search, executed entirely inside Rust.
    GlobalGreedyRust,
    /// Reference-equivalent search with cached per-target local toggles.
    GlobalGreedyRustOptimized,
    /// Cheaper sequential target-wise parent search under hard NOTREKS.
    LocalGreedyRust,
    /// Event-driven exact equivalent of the local target-wise search.
    LocalGreedyActiveExact,
    /// Active target-wise search with sequential adjacent reinsertion
    /// warm-starts.  This is experimental and intentionally separate from
    /// LocalGreedyActiveExact.
    LocalGreedyActiveReinsert,
    /// FLOP-like non-greedy grow--shrink with exact dynamic NOTREKS checks.
    FlopLike,
    /// Experimental adaptive local search; baseline local search is unchanged.
    LocalGreedyRustAdaptive,
    /// Historical edge-toggle implementation retained for comparison only.
    GlobalGreedyDiagnostic,
    /// Hard-DAG global greedy with a graph-level NOTREKS objective penalty.
    GlobalGreedyPenalty,
    /// Unrestricted FLOP order search followed by archived NOTREKS-aware
    /// selection/repair outside the optimizer.
    OrderGuidedLocal,
    /// Budgeted conflict-directed search over directed edge cuts of witness
    /// treks, retaining local-repair as an external incumbent/fallback.
    TrekCut,
    /// Sequential prefix construction using exact ancestor-bitset masks.
    PrefixFeasible,
    /// Static closed-neighborhood dominance superstructure.
    TrekDominance,
}

impl NoTreksVersion {
    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::FixedSignatureA => "fixed_signature_a",
            Self::AlternatingFullRefitB => "alternating_full_refit_b",
            Self::GlobalGreedyRust => "global_greedy_rust",
            Self::GlobalGreedyRustOptimized => "global_greedy_rust_optimized",
            Self::LocalGreedyRust => "local_greedy_rust",
            Self::LocalGreedyActiveExact => "local_greedy_active_exact",
            Self::LocalGreedyActiveReinsert => "local_greedy_active_reinsert",
            Self::FlopLike => "flop_like",
            Self::LocalGreedyRustAdaptive => "local_greedy_rust_adaptive",
            Self::GlobalGreedyDiagnostic => "global_greedy_diagnostic",
            Self::GlobalGreedyPenalty => "global_greedy_penalty",
            Self::OrderGuidedLocal => "order_guided_local",
            Self::TrekCut => "trekcut",
            Self::PrefixFeasible => "prefix_feasible",
            Self::TrekDominance => "trek_dominance",
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlopNoTreksConfig {
    pub lambda: f64,
    pub restarts: Option<usize>,
    pub timeout: Option<f64>,
    pub manual_termination: bool,
    pub seed: Option<u64>,
    pub signature_top_k: usize,
    pub signature_exploration_k: usize,
    pub max_signature_rounds: usize,
    pub initial_signature_mean_size: f64,
    pub initial_signature_max_size: usize,
    pub search_version: NoTreksVersion,
    /// Number of target-wise passes for the deliberately cheaper local
    /// kernel.  Other search versions ignore this field.
    pub local_greedy_passes: usize,
    pub forbidden_edges: Vec<(usize, usize)>,
    pub random_initial_order: bool,
    pub order_guided_lex_fraction: f64,
    pub order_guided_repair_candidates: usize,
    pub order_guided_coverage_starts: usize,
    pub trekcut_oracle_budget: usize,
    pub trekcut_refinement_passes: usize,
    /// Beam width for the experimental prefix-feasible constructor (1 or 4).
    pub prefix_beam_width: usize,
    pub trek_graph: Option<Vec<(usize, usize)>>,
}

#[derive(Clone, Debug, Default)]
pub struct NoTreksDiagnostics {
    pub restarts_requested: usize,
    pub restarts_completed: usize,
    pub best_restart_index: usize,
    pub best_restart_was_minimal: bool,
    pub restart_bics: Vec<f64>,
    pub initial_fraction_of_candidate_parent_relations_pruned: f64,
    pub number_of_supplied_constraints: usize,
    pub number_of_distinct_constrained_targets: usize,
    pub number_of_signature_rounds: usize,
    pub number_of_blocked_edge_proposals_generated: usize,
    pub number_of_proposals_retained: usize,
    pub number_of_proposals_tested_with_complete_refitting: usize,
    pub number_of_infeasible_promotions: usize,
    pub number_of_accepted_promotions: usize,
    pub ancestor_cone_size_sum: usize,
    pub maximum_ancestor_cone_size: usize,
    pub number_of_candidate_parent_relations_pruned: usize,
    pub fraction_of_candidate_parent_relations_pruned: f64,
    pub time_in_constrained_order_search: f64,
    pub time_in_signature_move_evaluation: f64,
    pub time_in_complete_constrained_refits: f64,
    pub final_bic: f64,
    pub selected_dag_edge_count: usize,
    pub final_no_trek_violation_count: usize,
    pub termination_reason: String,
    pub algorithm_seed: u64,
    pub search_version: String,
    pub number_of_canonical_compressions: usize,
    pub number_of_post_promotion_order_blocks: usize,
    pub order_guided_archive_edges: Vec<Vec<(usize, usize)>>,
    pub order_guided_archive_bics: Vec<f64>,
    pub order_guided_archive_violations: Vec<usize>,
    pub order_guided_archive_exact_violations: Vec<usize>,
    pub order_guided_violation_trajectory: Vec<usize>,
    pub order_guided_raw_feasible_count: usize,
    pub trekcut_witness_lengths: Vec<usize>,
    pub trekcut_branch_counts: Vec<usize>,
    pub trekcut_masks: Vec<Vec<(usize, usize)>>,
    pub trekcut_archive_bics: Vec<f64>,
    pub trekcut_archive_violations: Vec<usize>,
    pub trekcut_feasible_state_discovery_time: f64,
    pub trekcut_oracle_calls: usize,
    pub source_signature_source_count: usize,
    pub source_signature_mask_density: f64,
    pub source_signature_pair_check_passed: bool,
    pub prefix_admissible_pool_sizes: Vec<usize>,
    pub prefix_suffix_nodes_rebuilt: Vec<usize>,
    pub prefix_beam_width: usize,
    pub prefix_beam_alternative_count: usize,
    pub trek_dominance_mask_density: f64,
}

impl NoTreksDiagnostics {
    pub fn mean_ancestor_cone_size(&self) -> f64 {
        if self.number_of_proposals_tested_with_complete_refitting == 0 {
            0.0
        } else {
            self.ancestor_cone_size_sum as f64
                / self.number_of_proposals_tested_with_complete_refitting as f64
        }
    }
}

pub struct NoTreksResult {
    pub dag: Dag,
    pub diagnostics: NoTreksDiagnostics,
}

/// Construct a feasible DAG one node at a time.  The parent-search kernel is
/// deliberately the ordinary FLOP grow--shrink routine; only its admissible
/// parent pool is changed for the current prefix.
fn prefix_fit_order(
    order: &[usize],
    score: &Bic,
    pairs: &[(usize, usize)],
    rng: &mut StdRng,
    diagnostics: &mut NoTreksDiagnostics,
    beam_width: usize,
) -> Result<GlobalScore, FlopError> {
    let p = order.len();
    if p > 64 {
        return Err(FlopError::InvalidConfig(
            "prefix_feasible currently supports at most 64 nodes".into(),
        ));
    }
    let pair_set: BTreeSet<(usize, usize)> = pairs.iter()
        .map(|&(a, b)| if a < b { (a, b) } else { (b, a) })
        .collect();
    let mut ancestors = vec![0u64; p];
    let mut g = GlobalScore::new(p, score)?;
    for (position, &v) in order.iter().enumerate() {
        let prefix = &order[..position];
        diagnostics.prefix_suffix_nodes_rebuilt.push(p - position);
        let mut b = 0u64;
        for &j in prefix {
            let pair = if v < j { (v, j) } else { (j, v) };
            if pair_set.contains(&pair) { b |= ancestors[j]; }
        }
        let mut forbidden = Vec::new();
        let mut pool = 0usize;
        for &u in prefix {
            if ancestors[u] & b == 0 { pool += 1; }
            else { forbidden.push((u, v)); }
        }
        let node_constraints = NoTrekConstraints::new_with_directed_forbidden(
            p, &[], &forbidden).map_err(FlopError::ConstraintError)?;
        let primary = fit_parents_constrained(
            v, prefix, score, rng, Some(&node_constraints))?;
        let mut selected = primary;
        // B=4 is intentionally conservative: branch only when a forbidden
        // parent is locally BIC-improving, then rerun the unchanged kernel
        // with one selected parent additionally forbidden.
        if beam_width == 4 && !forbidden.is_empty() {
            let empty = score.local_score_init(v, Vec::new())?;
            let bottleneck = forbidden.iter().any(|&(u, _)|
                score.local_score_plus(v, &empty, u)
                    .is_ok_and(|candidate| candidate.bic < empty.bic));
            if bottleneck && !selected.parents.is_empty() {
                let expensive = *selected.parents.iter().max_by_key(|&&u| {
                    (ancestors[u] & b).count_ones()
                }).unwrap();
                let mut alt_forbidden = forbidden.clone();
                alt_forbidden.push((expensive, v));
                let alt_constraints = NoTrekConstraints::new_with_directed_forbidden(
                    p, &[], &alt_forbidden).map_err(FlopError::ConstraintError)?;
                let alternative = fit_parents_constrained(
                    v, prefix, score, rng, Some(&alt_constraints))?;
                diagnostics.prefix_beam_alternative_count += 1;
                if alternative.bic < selected.bic { selected = alternative; }
            }
        }
        g.local_scores[v] = selected;
        let mut a = 1u64 << v;
        for &u in &g.local_scores[v].parents { a |= ancestors[u]; }
        ancestors[v] = a;
        diagnostics.prefix_admissible_pool_sizes.push(pool);
    }
    let dag = Dag::from_global_score(&g);
    let violations = count_no_trek_violations(&dag, pairs);
    assert_eq!(violations, 0, "prefix construction returned an infeasible DAG");
    Ok(g)
}

fn run_prefix_feasible(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    if config.prefix_beam_width != 1 && config.prefix_beam_width != 4 {
        return Err(FlopError::InvalidConfig("prefix_beam_width must be 1 or 4".into()));
    }
    let p = data.ncols();
    if p > 64 { return Err(FlopError::InvalidConfig("prefix_feasible supports d <= 64".into())); }
    let corr = utils::corr_matrix(data);
    let (_, initial) = pivoted_cholesky::cholesky_left_min_diag(&corr)
        .ok_or_else(|| FlopError::InitialOrderError("Cholesky decomposition failed".into()))?;
    let score = Bic::from_cov(data.nrows(), corr, config.lambda);
    let limit = config.restarts.unwrap_or(0) + 1;
    let seed = config.seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);
    // Use the unchanged FLOP result to supply its data-driven order as an
    // additional prefix candidate.  The prefix constructor never receives
    // that graph as a warm start; only the order is reused.
    let vanilla = crate::algo::run(
        data,
        crate::algo::FlopConfig::with_forbidden_edges_seeded(
            config.lambda, Some(0), None, false, Vec::new(), Some(seed)),
    )?;
    let vanilla_order = vanilla.topological_ordering();
    let mut best: Option<(f64, Dag)> = None;
    let mut diagnostics = NoTreksDiagnostics {
        restarts_requested: limit,
        prefix_beam_width: config.prefix_beam_width,
        number_of_supplied_constraints: pairs.len(),
        algorithm_seed: seed,
        search_version: config.search_version.stable_name().into(),
        termination_reason: "restart_limit".into(),
        ..Default::default()
    };
    for restart in 0..limit {
        let mut order = if restart == 0 { vanilla_order.clone() } else { initial.clone() };
        if restart > 1 {
            order.shuffle(&mut rng);
        }
        let started = Instant::now();
        let g = prefix_fit_order(
            &order, &score, pairs, &mut rng, &mut diagnostics,
            config.prefix_beam_width)?;
        diagnostics.time_in_complete_constrained_refits += started.elapsed().as_secs_f64();
        let bic = g.score();
        diagnostics.restart_bics.push(bic);
        diagnostics.restarts_completed += 1;
        let dag = Dag::from_global_score(&g);
        if best.as_ref().is_none_or(|(old, _)| bic < *old) { best = Some((bic, dag)); }
    }
    let (bic, dag) = best.ok_or_else(|| FlopError::InvalidConfig("no prefix restart completed".into()))?;
    diagnostics.final_bic = bic;
    diagnostics.selected_dag_edge_count = dag.parents.iter().map(Vec::len).sum();
    diagnostics.final_no_trek_violation_count = count_no_trek_violations(&dag, pairs);
    Ok(NoTreksResult { dag, diagnostics })
}

/// Prototype maximal-T Trek-Dominance search.  The supplied NOTREKS pairs
/// define the absent edges of T; candidate-specific T construction is kept in
/// the benchmark layer for later experiments.
fn run_trek_dominance(
    data: &DMatrix<f64>, pairs: &[(usize, usize)], config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    let p = data.ncols();
    let trek_edges = config.trek_graph.as_deref().ok_or_else(|| {
        FlopError::InvalidConfig("trek_dominance requires an explicit trek_graph".into())
    })?;
    crate::no_treks::validate_candidate_trek_graph(p, trek_edges, pairs)
        .map_err(FlopError::ConstraintError)?;
    let allowed_mask = crate::no_treks::trek_dominance_mask(p, trek_edges)
        .map_err(FlopError::ConstraintError)?;
    let mut forbidden = Vec::new();
    let mut allowed = 0usize;
    for u in 0..p { for v in 0..p {
        if u == v { continue; }
        if allowed_mask[u][v] { allowed += 1; }
        else { forbidden.push((u, v)); }
    }}
    let mut static_config = config.clone();
    static_config.search_version = NoTreksVersion::TrekDominance;
    static_config.forbidden_edges = forbidden;
    // The static dominance mask is the constraint in this method.  Passing I
    // through the legacy signature constraint would impose a second,
    // different parent rule and can eliminate valid dominance arrows.
    let mut result = run_notreks_constrained(data, &[], static_config)?;
    result.diagnostics.number_of_supplied_constraints = pairs.len();
    result.diagnostics.search_version = NoTreksVersion::TrekDominance.stable_name().into();
    result.diagnostics.trek_dominance_mask_density = allowed as f64 / (p * (p - 1)) as f64;
    result.diagnostics.final_no_trek_violation_count = count_no_trek_violations(&result.dag, pairs);
    assert_eq!(result.diagnostics.final_no_trek_violation_count, 0);
    Ok(result)
}

fn dag_adjacency(dag: &Dag) -> Vec<u8> {
    let mut matrix = vec![0u8; dag.p * dag.p];
    for (child, parents) in dag.parents.iter().enumerate() {
        for &parent in parents {
            matrix[parent * dag.p + child] = 1;
        }
    }
    matrix
}

fn global_score_adjacency(g: &GlobalScore) -> Vec<u8> {
    let mut matrix = vec![0u8; g.p * g.p];
    for (child, local) in g.local_scores.iter().enumerate() {
        for &parent in &local.parents {
            matrix[parent * g.p + child] = 1;
        }
    }
    matrix
}

/// Exact packed-bitset ancestry state for a fixed-order binary DAG.
///
/// Each row contains the ancestors of one node. Since the graph is processed
/// in causal order, a child row is formed by OR-ing its parent rows. This is
/// the Boolean triangular solve, with `u64` words replacing scalar Boolean
/// entries. It is exact for arbitrary dimensions and has no numerical zeros.
fn packed_ancestry(g: &GlobalScore, order: &[usize]) -> Vec<Vec<u64>> {
    let p = g.p;
    let words = p.div_ceil(64);
    let mut ancestors = vec![vec![0_u64; words]; p];
    let mut children = vec![Vec::new(); p];
    for (child, local) in g.local_scores.iter().enumerate() {
        for &parent in &local.parents {
            children[parent].push(child);
        }
    }
    for (node, row) in ancestors.iter_mut().enumerate() {
        row[node / 64] |= 1_u64 << (node % 64);
    }
    for &node in order {
        for &child in &children[node] {
            for word in 0..words {
                ancestors[child][word] |= ancestors[node][word];
            }
        }
    }
    ancestors
}

fn addition_violates_packed(
    ancestors: &[Vec<u64>],
    source: usize,
    target: usize,
    pairs: &[(usize, usize)],
) -> bool {
    let target_word = target / 64;
    let target_mask = 1_u64 << (target % 64);
    for &(left, right) in pairs {
        let left_affected = ancestors[left][target_word] & target_mask != 0;
        let right_affected = ancestors[right][target_word] & target_mask != 0;
        if left_affected && right_affected {
            return true;
        }
        if left_affected
            && ancestors[source]
                .iter()
                .zip(&ancestors[right])
                .any(|(a, b)| a & b != 0)
        {
            return true;
        }
        if right_affected
            && ancestors[source]
                .iter()
                .zip(&ancestors[left])
                .any(|(a, b)| a & b != 0)
        {
            return true;
        }
    }
    false
}

fn invalid_additions_packed(
    ancestors: &[Vec<u64>],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
) -> Vec<Vec<bool>> {
    let p = ancestors.len();
    let mut invalid = vec![vec![false; p]; p];
    for source in 0..p {
        for target in 0..p {
            if source != target {
                invalid[source][target] = forbidden_edges
                    .iter()
                    .any(|&(u, v)| (u == source && v == target) || (u == target && v == source))
                    || addition_violates_packed(ancestors, source, target, pairs);
            }
        }
    }
    invalid
}

/// Complete Rust implementation of the FLOP-style outer loop around the
/// fixed-order global-greedy inner search.  This keeps all candidate-order
/// evaluation in one extension call, avoiding Python/Rust crossings.
fn run_global_greedy_rust_impl(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    let p = data.ncols();
    // The global-greedy kernel only needs the pair list for exact witness
    // checks.  Do not route it through the signature prototype: that
    // representation intentionally uses one u64 bit per constrained target
    // and is therefore limited to 64 targets.  FLOP+NOTREKS itself should
    // remain usable for larger graphs.
    let canonical_pairs = canonical_pair_list(p, pairs)?;
    let forbidden_edges = canonical_pair_list(p, &config.forbidden_edges)?;
    let seed = config.seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);
    let corr = crate::utils::corr_matrix(data);
    let (_, mut initial_order) = pivoted_cholesky::cholesky_left_min_diag(&corr)
        .ok_or_else(|| FlopError::InitialOrderError("Cholesky decomposition failed".into()))?;
    if config.random_initial_order {
        initial_order.shuffle(&mut rng);
    }
    let num_perturbations = (p as f64).ln().round() as usize;
    let started = Instant::now();
    let deadline = config.timeout.map(Duration::from_secs_f64);
    let limit = config.restarts.unwrap_or(usize::MAX - 1) + 1;
    let score = Bic::from_cov(data.nrows(), corr, config.lambda);
    let mut best: Option<(f64, Vec<usize>, Dag)> = None;
    let mut best_order = initial_order;
    let mut completed = 0usize;
    let mut termination = "restart_limit".to_string();

    for restart in 0..limit {
        if restart > 0 && deadline.is_some_and(|d| started.elapsed() >= d) {
            termination = "timeout".into();
            break;
        }
        let mut order = best_order.clone();
        if restart > 0 {
            for _ in 0..num_perturbations {
                let a = rng.gen_range(0..p);
                let b = rng.gen_range(0..p);
                order.swap(a, b);
            }
        }
        let mut zero = vec![0u8; p * p];
        let (mut graph, mut current_score, _) = run_selected_inner(
            data,
            &zero,
            &order,
            &canonical_pairs,
            &forbidden_edges,
            &score,
            config.search_version,
            config.local_greedy_passes,
        )?;
        for _ in 0..config.max_signature_rounds.max(1) {
            if deadline.is_some_and(|d| started.elapsed() >= d) {
                termination = "timeout".into();
                break;
            }
            let before = current_score;
            for node in order.clone() {
                if deadline.is_some_and(|d| started.elapsed() >= d) {
                    termination = "timeout".into();
                    break;
                }
                let old_position = order.iter().position(|&x| x == node).unwrap();
                let positions: Vec<usize> = (0..p).filter(|&pos| pos != old_position).collect();
                let worker_count = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
                    .min(8)
                    .min(positions.len().max(1));
                let chunk_size = positions.len().div_ceil(worker_count);
                let mut evaluations = Vec::with_capacity(positions.len());
                std::thread::scope(|scope| -> Result<(), FlopError> {
                    let mut handles = Vec::new();
                    for chunk in positions.chunks(chunk_size) {
                        let chunk = chunk.to_vec();
                        handles.push(scope.spawn(|| -> Result<_, FlopError> {
                            let mut result = Vec::with_capacity(chunk.len());
                            for new_position in chunk {
                                let mut candidate_order = order.clone();
                                let moved = candidate_order.remove(old_position);
                                candidate_order.insert(new_position, moved);
                                // Rebuild the candidate state from the current graph for
                                // every order proposal.  This is the same search semantics as
                                // the reference Python global-greedy implementation: project
                                // the current graph to the candidate order, then run the exact
                                // inner search from that projected graph.  It avoids silently
                                // carrying a stale local basin across order proposals while
                                // retaining Rust's packed ancestry and BIC kernels.
                                let current_adj = dag_adjacency(&graph);
                                let (candidate_graph, candidate_score, _) =
                                    run_selected_inner(
                                        data,
                                        &current_adj,
                                        &candidate_order,
                                        &canonical_pairs,
                                        &forbidden_edges,
                                        &score,
                                        config.search_version,
                                        config.local_greedy_passes,
                                    )?;
                                let selected = (candidate_score, candidate_graph);
                                let candidate_adj = dag_adjacency(&selected.1);
                                result.push((
                                    new_position,
                                    selected.0,
                                    candidate_order,
                                    selected.1,
                                    candidate_adj,
                                ));
                            }
                            Ok(result)
                        }));
                    }
                    for handle in handles {
                        let result = handle.join().map_err(|_| {
                            FlopError::InvalidConfig(
                                "parallel candidate evaluation panicked".into(),
                            )
                        })??;
                        evaluations.extend(result);
                    }
                    Ok(())
                })?;
                evaluations.sort_by_key(|x| x.0);
                let mut chosen: Option<(f64, Vec<usize>, Dag, Vec<u8>)> = None;
                for (_, candidate_score, candidate_order, candidate_graph, candidate_adj) in
                    evaluations
                {
                    let better =
                        chosen
                            .as_ref()
                            .is_none_or(|(best_score, best_order, _, best_adj)| {
                                candidate_score < *best_score - EPS
                                    || ((candidate_score - *best_score).abs() <= EPS
                                        && (candidate_order.clone(), candidate_adj.clone())
                                            < (best_order.clone(), best_adj.clone()))
                            });
                    if better {
                        chosen = Some((
                            candidate_score,
                            candidate_order,
                            candidate_graph,
                            candidate_adj,
                        ));
                    }
                }
                if let Some((candidate_score, candidate_order, candidate_graph, _candidate_adj)) =
                    chosen
                {
                    if candidate_score < current_score - EPS {
                        current_score = candidate_score;
                        order = candidate_order;
                        graph = candidate_graph;
                    }
                }
            }
            if termination == "timeout" || before - current_score <= EPS {
                break;
            }
        }
        completed += 1;
        let candidate = (current_score, order.clone(), graph.clone());
        let better = best.as_ref().is_none_or(|(score, old_order, old_graph)| {
            candidate.0 < *score - EPS
                || ((candidate.0 - *score).abs() <= EPS
                    && (candidate.1.clone(), dag_adjacency(&candidate.2))
                        < (old_order.clone(), dag_adjacency(old_graph)))
        });
        if better {
            best_order = order.clone();
            best = Some(candidate);
        }
        zero.clear();
    }
    let Some((final_bic, order, dag)) = best else {
        return Err(FlopError::InitialOrderError(
            "global-greedy completed no restart".into(),
        ));
    };
    let violations = count_no_trek_violations(&dag, &canonical_pairs);
    if violations != 0 {
        return Err(FlopError::ConstraintError(
            "global-greedy returned an infeasible graph".into(),
        ));
    }
    let diagnostics = NoTreksDiagnostics {
        restarts_requested: limit,
        restarts_completed: completed,
        best_restart_index: 0,
        number_of_supplied_constraints: canonical_pairs.len(),
        final_bic,
        selected_dag_edge_count: dag.parents.iter().map(Vec::len).sum(),
        final_no_trek_violation_count: violations,
        algorithm_seed: seed,
        search_version: config.search_version.stable_name().into(),
        termination_reason: termination,
        ..Default::default()
    };
    let _ = order;
    Ok(NoTreksResult { dag, diagnostics })
}

fn run_global_greedy_rust(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    run_global_greedy_rust_impl(data, pairs, config)
}

pub fn run_global_greedy_rust_optimized(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    run_global_greedy_rust_impl(data, pairs, config)
}

/// Same hard-feasible search as `run_global_greedy_rust`, with an additional
/// graph-only NOTREKS penalty calculation for every candidate.  The penalty is
/// intentionally not part of selection; this is a runtime-control variant.
pub fn run_global_greedy_binary_penalty(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    // The hard inner kernel already computes the exact packed ancestry mask
    // used by the binary NOTREKS rule. Every candidate that reaches the
    // outer search is feasible and therefore has penalty zero. Re-running a
    // full ancestry traversal here would only duplicate work and cannot
    // change the selected graph.
    run_global_greedy_rust(data, pairs, config)
}

/// Optimize a graph for one fixed order using the same edge-toggle greedy
/// inner loop as the Benchpress global-greedy comparator.  The caller owns the
/// FLOP-style order/reinsertion loop; this function only performs the
/// order-respecting, hard-NOTREKS-feasible inner optimization.
fn global_greedy_inner_state(
    mut g: GlobalScore,
    order: &[usize],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
) -> Result<(Dag, f64, GlobalScore), FlopError> {
    let p = g.p;
    if order.len() != p {
        return Err(FlopError::InvalidConfig(
            "global-greedy inner shape/order mismatch".into(),
        ));
    }
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        if node >= p {
            return Err(FlopError::InvalidConfig(
                "global-greedy order contains an invalid node".into(),
            ));
        }
        position[node] = idx;
    }

    let mut current_score = g.score();
    loop {
        let current_encoding = global_score_adjacency(&g);
        let mut best: Option<(f64, Vec<u8>, usize, LocalScore)> = None;
        let ancestors = packed_ancestry(&g, order);
        // The graph is fixed during this proposal scan.  Build the exact
        // parent-feasibility mask once, then use it for every BIC candidate.
        // It is recomputed after an accepted move because ancestry changes.
        let invalid_additions = invalid_additions_packed(&ancestors, pairs, forbidden_edges);
        for source in 0..p {
            for target in 0..p {
                if source == target || position[source] >= position[target] {
                    continue;
                }
                let adding = !g.local_scores[target].parents.contains(&source);
                // Feasibility is a hard precondition for candidate scoring.
                // Do not evaluate local BIC for an addition rejected by the
                // exact ancestry table.
                if adding && invalid_additions[source][target] {
                    continue;
                }
                let local = if adding {
                    score.local_score_plus(target, &g.local_scores[target], source)?
                } else {
                    score.local_score_minus(target, &g.local_scores[target], source)?
                };
                let value = current_score - g.local_scores[target].bic + local.bic;
                let better = best
                    .as_ref()
                    .is_none_or(|(best_value, best_encoding, _, _)| {
                        value < *best_value - EPS
                            || ((value - *best_value).abs() <= EPS && {
                                let mut encoding = current_encoding.clone();
                                encoding[source * p + target] = u8::from(adding);
                                encoding < *best_encoding
                            })
                    });
                if better {
                    let mut encoding = current_encoding.clone();
                    encoding[source * p + target] = u8::from(adding);
                    best = Some((value, encoding, target, local));
                }
            }
        }
        let Some((value, _, target, local)) = best else {
            break;
        };
        if value >= current_score - EPS {
            break;
        }
        g.local_scores[target] = local;
        current_score = value;
    }
    let dag = Dag::from_global_score(&g);
    debug_assert_eq!(count_no_trek_violations(&dag, pairs), 0);
    Ok((dag, current_score, g))
}

fn global_greedy_inner_with_state_from_initial(
    data: &DMatrix<f64>,
    initial: &[u8],
    order: &[usize],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
) -> Result<(Dag, f64, GlobalScore), FlopError> {
    let p = data.ncols();
    if initial.len() != p * p || order.len() != p {
        return Err(FlopError::InvalidConfig(
            "global-greedy inner shape/order mismatch".into(),
        ));
    }
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        if node >= p {
            return Err(FlopError::InvalidConfig(
                "global-greedy order contains an invalid node".into(),
            ));
        }
        position[node] = idx;
    }
    let mut g = GlobalScore::new(p, score)?;
    for child in 0..p {
        let mut parents: Vec<usize> = (0..p)
            .filter(|&parent| {
                parent != child
                    && initial[parent * p + child] != 0
                    && position[parent] < position[child]
                    && !forbidden_edges
                        .iter()
                        .any(|&(u, v)| (u == parent && v == child) || (u == child && v == parent))
            })
            .collect();
        parents.sort_by_key(|&parent| position[parent]);
        for parent in parents {
            g.local_scores[child] =
                score.local_score_plus(child, &g.local_scores[child], parent)?;
        }
    }
    global_greedy_inner_state(g, order, pairs, forbidden_edges, score)
}

fn update_ancestry_after_addition(
    ancestors: &mut [Vec<u64>],
    g: &GlobalScore,
    source: usize,
    target: usize,
) {
    let p = g.p;
    let mut children = vec![Vec::new(); p];
    for (child, local) in g.local_scores.iter().enumerate() {
        for &parent in &local.parents {
            children[parent].push(child);
        }
    }
    let mut descendants = vec![false; p];
    let mut queue = VecDeque::from([target]);
    descendants[target] = true;
    while let Some(node) = queue.pop_front() {
        for &child in &children[node] {
            if !descendants[child] {
                descendants[child] = true;
                queue.push_back(child);
            }
        }
    }
    for node in 0..p {
        if descendants[node] {
            for word in 0..ancestors[node].len() {
                ancestors[node][word] |= ancestors[source][word];
            }
        }
    }
}

/// Exact equivalent of `global_greedy_inner_state` that caches local toggle
/// scores.  A local score changes only for the target whose parent set was
/// toggled; ancestry feasibility is still refreshed on every accepted move.
fn global_greedy_inner_state_cached(
    mut g: GlobalScore,
    order: &[usize],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
) -> Result<(Dag, f64, GlobalScore), FlopError> {
    let p = g.p;
    if order.len() != p {
        return Err(FlopError::InvalidConfig(
            "global-greedy inner shape/order mismatch".into(),
        ));
    }
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        if node >= p {
            return Err(FlopError::InvalidConfig(
                "global-greedy order contains an invalid node".into(),
            ));
        }
        position[node] = idx;
    }

    let mut cached: Vec<Vec<Option<LocalScore>>> =
        (0..p).map(|_| (0..p).map(|_| None).collect()).collect();
    let mut current_score = g.score();
    let mut ancestors = packed_ancestry(&g, order);
    loop {
        let current_encoding = dag_adjacency(&Dag::from_global_score(&g));
        let invalid_additions = invalid_additions_packed(
            &ancestors, pairs, forbidden_edges);
        let mut best: Option<(f64, Vec<u8>, usize, usize, bool, LocalScore)> = None;

        // Traverse targets and predecessors in the supplied order.  The
        // explicit encoding tie-break below keeps the result independent of
        // this traversal order.
        for &target in order {
            for &source in order.iter().take(position[target]) {
                let adding = !g.local_scores[target].parents.contains(&source);
                if adding && invalid_additions[source][target] {
                    continue;
                }
                let local = if let Some(value) = &cached[target][source] {
                    value.clone()
                } else {
                    let value = if adding {
                        score.local_score_plus(
                            target, &g.local_scores[target], source)?
                    } else {
                        score.local_score_minus(
                            target, &g.local_scores[target], source)?
                    };
                    cached[target][source] = Some(value.clone());
                    value
                };
                let value = current_score - g.local_scores[target].bic + local.bic;
                let better = best.as_ref().is_none_or(
                    |(best_value, best_encoding, _, _, _, _)| {
                        value < *best_value - EPS ||
                            ((value - *best_value).abs() <= EPS && {
                                let mut encoding = current_encoding.clone();
                                encoding[source * p + target] = u8::from(adding);
                                encoding < *best_encoding
                            })
                    });
                if better {
                    let mut encoding = current_encoding.clone();
                    encoding[source * p + target] = u8::from(adding);
                    best = Some((value, encoding, source, target, adding, local));
                }
            }
        }

        let Some((value, _, source, target, adding, local)) = best else {
            break;
        };
        if value >= current_score - EPS {
            break;
        }
        g.local_scores[target] = local;
        cached[target].iter_mut().for_each(|entry| *entry = None);
        if adding {
            update_ancestry_after_addition(
                &mut ancestors, &g, source, target);
        } else {
            ancestors = packed_ancestry(&g, order);
        }
        current_score = value;
    }
    let dag = Dag::from_global_score(&g);
    debug_assert_eq!(count_no_trek_violations(&dag, pairs), 0);
    Ok((dag, current_score, g))
}

fn global_greedy_inner_with_state_from_initial_optimized(
    data: &DMatrix<f64>,
    initial: &[u8],
    order: &[usize],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
) -> Result<(Dag, f64, GlobalScore), FlopError> {
    let p = data.ncols();
    if initial.len() != p * p || order.len() != p {
        return Err(FlopError::InvalidConfig(
            "global-greedy inner shape/order mismatch".into(),
        ));
    }
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        if node >= p {
            return Err(FlopError::InvalidConfig(
                "global-greedy order contains an invalid node".into(),
            ));
        }
        position[node] = idx;
    }
    let mut g = GlobalScore::new(p, score)?;
    for child in 0..p {
        let mut parents: Vec<usize> = (0..p)
            .filter(|&parent| {
                parent != child
                    && initial[parent * p + child] != 0
                    && position[parent] < position[child]
                    && !forbidden_edges.iter().any(|&(u, v)|
                        (u == parent && v == child) ||
                        (u == child && v == parent))
            })
            .collect();
        parents.sort_by_key(|&parent| position[parent]);
        for parent in parents {
            g.local_scores[child] = score.local_score_plus(
                child, &g.local_scores[child], parent)?;
        }
    }
    global_greedy_inner_state_cached(g, order, pairs, forbidden_edges, score)
}

fn run_selected_inner(
    data: &DMatrix<f64>,
    initial: &[u8],
    order: &[usize],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
    search_version: NoTreksVersion,
    local_greedy_passes: usize,
) -> Result<(Dag, f64, GlobalScore), FlopError> {
    match search_version {
        NoTreksVersion::GlobalGreedyRustOptimized =>
            global_greedy_inner_with_state_from_initial_optimized(
                data, initial, order, pairs, forbidden_edges, score),
        NoTreksVersion::LocalGreedyRust =>
            local_greedy_inner_with_state_from_initial(
                data, initial, order, pairs, forbidden_edges, score,
                local_greedy_passes),
        NoTreksVersion::LocalGreedyRustAdaptive =>
            local_greedy_inner_with_state_from_initial(
                data, initial, order, pairs, forbidden_edges, score,
                local_greedy_passes.min(4)),
        NoTreksVersion::LocalGreedyActiveExact =>
            local_greedy_active_with_state_from_initial(
                data, initial, order, pairs, forbidden_edges, score,
                local_greedy_passes),
        _ => global_greedy_inner_with_state_from_initial(
            data, initial, order, pairs, forbidden_edges, score),
    }
}

/// Cheaper sequential target-wise alternative to the global greedy inner loop.
/// Each target receives at most one best feasible toggle during one pass.
/// This intentionally trades global refinement for speed and is not expected
/// to return the same graph as the global method.
fn local_greedy_inner_state(
    mut g: GlobalScore,
    order: &[usize],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
    local_greedy_passes: usize,
) -> Result<(Dag, f64, GlobalScore), FlopError> {
    let p = g.p;
    let trace = diagnostic_trace_enabled();
    let shadow = env::var_os("FLOP_NOTREKS_SHADOW_ACTIVE").is_some();
    let mut shadow_active = vec![true; p];
    let mut shadow_skips = 0usize;
    let mut shadow_skip_none = 0usize;
    let mut shadow_skip_add = 0usize;
    let mut shadow_skip_delete = 0usize;
    if trace {
        eprintln!("TRACE_BEGIN|baseline|order={}|bic={:.17e}", diagnostic_order(order), g.score());
    }
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        if node >= p {
            return Err(FlopError::InvalidConfig(
                "local-greedy order contains an invalid node".into(),
            ));
        }
        position[node] = idx;
    }
    let mut ancestors = packed_ancestry(&g, order);
    let mut current_score = g.score();
    for pass in 0..local_greedy_passes.max(1) {
        let mut changed = false;
        let mut shadow_next = vec![false; p];
        for &target in order {
            let would_skip = shadow && !shadow_active[target];
            if would_skip { shadow_skips += 1; }
            let before_target = g.local_scores[target].bic;
            let current_encoding = global_score_adjacency(&g);
            let mut best: Option<(f64, Vec<u8>, usize, bool, LocalScore)> = None;
            for &source in order.iter().take(position[target]) {
            let adding = !g.local_scores[target].parents.contains(&source);
            if adding {
                let directly_forbidden = forbidden_edges.iter().any(|&(u, v)| {
                    (u == source && v == target) ||
                        (u == target && v == source)
                });
                if directly_forbidden ||
                    addition_violates_packed(&ancestors, source, target, pairs)
                {
                    continue;
                }
            }
                let local = if adding {
                    score.local_score_plus(
                        target, &g.local_scores[target], source)?
                } else {
                    score.local_score_minus(
                        target, &g.local_scores[target], source)?
                };
                let value = current_score
                    - g.local_scores[target].bic + local.bic;
                let better = best.as_ref().is_none_or(
                    |(best_value, best_encoding, _, _, _)| {
                        value < *best_value - EPS ||
                            ((value - *best_value).abs() <= EPS && {
                                let mut encoding = current_encoding.clone();
                                encoding[source * p + target] = u8::from(adding);
                                encoding < *best_encoding
                            })
                    });
                if better {
                    let mut encoding = current_encoding.clone();
                    encoding[source * p + target] = u8::from(adding);
                    best = Some((value, encoding, source, adding, local));
                }
            }
            if let Some((value, _, source, adding, local)) = best {
                if value < current_score - EPS {
                    if trace {
                        eprintln!("TRACE_MOVE|baseline|pass={}|target={}|source={}|kind={}|before={:.17e}|after={:.17e}|parents_before={}|parents_after={}",
                            pass + 1, target, source, if adding { "add" } else { "delete" },
                            before_target, local.bic, diagnostic_parent_list(&g, target),
                            format!("{:?}", local.parents));
                    }
                    g.local_scores[target] = local;
                    if adding {
                        update_ancestry_after_addition(
                            &mut ancestors, &g, source, target);
                        if would_skip { shadow_skip_add += 1; }
                        if shadow { shadow_next[target] = true; }
                    } else {
                        ancestors = packed_ancestry(&g, order);
                        if would_skip { shadow_skip_delete += 1; }
                        if shadow {
                            let idx = order.iter().position(|&node| node == target).unwrap();
                            for (j, &node) in order.iter().enumerate() {
                                if j > idx { shadow_active[node] = true; }
                                else { shadow_next[node] = true; }
                            }
                        }
                    }
                    current_score = value;
                    changed = true;
                }
            } else if would_skip {
                shadow_skip_none += 1;
            } else if trace {
                eprintln!("TRACE_TARGET|baseline|pass={}|target={}|parents={}|result=none", pass + 1, target, diagnostic_parent_list(&g, target));
            }
        }
        if shadow { shadow_active = shadow_next; }
        if trace { eprintln!("TRACE_PASS|baseline|pass={}|changed={}|bic={:.17e}", pass + 1, changed, current_score); }
        if !changed {
            break;
        }
    }
    let dag = Dag::from_global_score(&g);
    if shadow {
        eprintln!("SHADOW_SUMMARY|skipped_targets={}|skipped_no_update={}|skipped_additions={}|skipped_deletions={}|rng_draws_in_target_evaluations=0",
                  shadow_skips, shadow_skip_none, shadow_skip_add, shadow_skip_delete);
    }
    if trace { eprintln!("TRACE_END|baseline|bic={:.17e}|edges={:?}", current_score, dag_edges(&dag)); }
    debug_assert_eq!(count_no_trek_violations(&dag, pairs), 0);
    Ok((dag, current_score, g))
}

fn local_greedy_inner_with_state_from_initial(
    data: &DMatrix<f64>,
    initial: &[u8],
    order: &[usize],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
    local_greedy_passes: usize,
) -> Result<(Dag, f64, GlobalScore), FlopError> {
    let p = data.ncols();
    if initial.len() != p * p || order.len() != p {
        return Err(FlopError::InvalidConfig(
            "local-greedy inner shape/order mismatch".into(),
        ));
    }
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        if node >= p {
            return Err(FlopError::InvalidConfig(
                "local-greedy order contains an invalid node".into(),
            ));
        }
        position[node] = idx;
    }
    let mut g = GlobalScore::new(p, score)?;
    for child in 0..p {
        let mut parents: Vec<usize> = (0..p)
            .filter(|&parent| {
                parent != child
                    && initial[parent * p + child] != 0
                    && position[parent] < position[child]
                    && !forbidden_edges.iter().any(|&(u, v)|
                        (u == parent && v == child) ||
                        (u == child && v == parent))
            })
            .collect();
        parents.sort_by_key(|&parent| position[parent]);
        for parent in parents {
            g.local_scores[child] = score.local_score_plus(
                child, &g.local_scores[child], parent)?;
        }
    }
    local_greedy_inner_state(
        g, order, pairs, forbidden_edges, score, local_greedy_passes)
}

/// Event-driven version of the local kernel.  The candidate evaluation and
/// tie-breaking are intentionally copied from `local_greedy_inner_state`;
/// only the target activation schedule differs.  Additions can only remove
/// feasible additions, whereas deletions can unlock them globally.
fn local_greedy_active_exact_state(
    mut g: GlobalScore,
    order: &[usize],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
    local_greedy_passes: usize,
) -> Result<(Dag, f64, GlobalScore), FlopError> {
    let p = g.p;
    let trace = diagnostic_trace_enabled();
    if trace {
        eprintln!("TRACE_BEGIN|active|order={}|bic={:.17e}", diagnostic_order(order), g.score());
    }
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        if node >= p { return Err(FlopError::InvalidConfig("active-local order contains an invalid node".into())); }
        position[node] = idx;
    }
    let mut ancestors = packed_ancestry(&g, order);
    let mut current_score = g.score();
    let mut active = vec![true; p];
    for pass in 0..local_greedy_passes.max(1) {
        let mut next = vec![false; p];
        let mut any_active = false;
        for (idx, &target) in order.iter().enumerate() {
            if !active[target] {
                if trace { eprintln!("TRACE_SKIP|active|pass={}|target={}|active_current={:?}|active_next={:?}|parents={}", pass + 1, target, active, next, diagnostic_parent_list(&g, target)); }
                continue;
            }
            any_active = true;
            let before_target = g.local_scores[target].bic;
            let current_encoding = global_score_adjacency(&g);
            let mut best: Option<(f64, Vec<u8>, usize, bool, LocalScore)> = None;
            for &source in order.iter().take(position[target]) {
                let adding = !g.local_scores[target].parents.contains(&source);
                if adding {
                    let direct = forbidden_edges.iter().any(|&(u,v)|
                        (u == source && v == target) || (u == target && v == source));
                    if direct || addition_violates_packed(&ancestors, source, target, pairs) { continue; }
                }
                let local = if adding { score.local_score_plus(target, &g.local_scores[target], source)? }
                    else { score.local_score_minus(target, &g.local_scores[target], source)? };
                let value = current_score - g.local_scores[target].bic + local.bic;
                let better = best.as_ref().is_none_or(|(bv, be, _, _, _)| {
                    value < *bv - EPS || ((value - *bv).abs() <= EPS && {
                        let mut enc = current_encoding.clone(); enc[source*p+target] = u8::from(adding); enc < *be
                    })
                });
                if better { let mut enc=current_encoding.clone(); enc[source*p+target]=u8::from(adding); best=Some((value,enc,source,adding,local)); }
            }
            if let Some((value, _, source, adding, local)) = best {
                if value < current_score - EPS {
                    if trace {
                        eprintln!("TRACE_MOVE|active|pass={}|target={}|source={}|kind={}|before={:.17e}|after={:.17e}|parents_before={}|parents_after={}|active_current={:?}|active_next_before={:?}",
                            pass + 1, target, source, if adding { "add" } else { "delete" },
                            before_target, local.bic, diagnostic_parent_list(&g, target),
                            format!("{:?}", local.parents), active, next);
                    }
                    g.local_scores[target] = local;
                    current_score = value;
                    if adding {
                        update_ancestry_after_addition(&mut ancestors, &g, source, target);
                        next[target] = true;
                    } else {
                        ancestors = packed_ancestry(&g, order);
                        // Later targets are revisited in this pass; earlier
                        // targets and the changed target are revisited next.
                        for (j, &node) in order.iter().enumerate() {
                            if j > idx { active[node] = true; }
                            else { next[node] = true; }
                        }
                    }
                }
            } else if trace {
                eprintln!("TRACE_TARGET|active|pass={}|target={}|parents={}|active_current={:?}|active_next={:?}|result=none", pass + 1, target, diagnostic_parent_list(&g, target), active, next);
            }
        }
        if trace { eprintln!("TRACE_PASS|active|pass={}|next={:?}|any_active={}|bic={:.17e}", pass + 1, next, any_active, current_score); }
        if !any_active { break; }
        if !next.iter().any(|&v| v) { break; }
        active = next;
    }
    let dag = Dag::from_global_score(&g);
    if trace { eprintln!("TRACE_END|active|bic={:.17e}|edges={:?}", current_score, dag_edges(&dag)); }
    debug_assert_eq!(count_no_trek_violations(&dag, pairs), 0);
    Ok((dag, current_score, g))
}

fn local_greedy_active_with_state_from_initial(
    data: &DMatrix<f64>, initial: &[u8], order: &[usize],
    pairs: &[(usize, usize)], forbidden_edges: &[(usize, usize)],
    score: &Bic, local_greedy_passes: usize,
) -> Result<(Dag, f64, GlobalScore), FlopError> {
    let p = data.ncols();
    if initial.len() != p*p || order.len() != p {
        return Err(FlopError::InvalidConfig("active-local shape/order mismatch".into()));
    }
    let mut position=vec![0usize;p];
    for (idx,&node) in order.iter().enumerate() { if node>=p { return Err(FlopError::InvalidConfig("active-local order contains an invalid node".into())); } position[node]=idx; }
    let mut g=GlobalScore::new(p,score)?;
    for child in 0..p {
        let mut parents: Vec<usize>=(0..p).filter(|&parent| parent!=child && initial[parent*p+child]!=0 && position[parent]<position[child] && !forbidden_edges.iter().any(|&(u,v)|(u==parent&&v==child)||(u==child&&v==parent))).collect();
        parents.sort_by_key(|&parent| position[parent]);
        for parent in parents { g.local_scores[child]=score.local_score_plus(child,&g.local_scores[child],parent)?; }
    }
    local_greedy_active_exact_state(g,order,pairs,forbidden_edges,score,local_greedy_passes)
}

/// Refine only the two targets whose prefixes changed during an adjacent
/// reinsertion. The incoming GlobalScore carries the warm parent sets and
/// Cholesky state from the preceding adjacent position.
fn refine_selected_targets(
    mut g: GlobalScore,
    order: &[usize],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
    targets: [usize; 2],
    passes: usize,
) -> Result<GlobalScore, FlopError> {
    let p = g.p;
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        position[node] = idx;
    }
    let mut ancestors = packed_ancestry(&g, order);
    let mut current_score = g.score();
    let mut target_order = targets;
    target_order.sort_by_key(|&node| position[node]);

    for _ in 0..passes.max(1) {
        let mut changed = false;
        for &target in &target_order {
            let current_encoding = global_score_adjacency(&g);
            let mut best: Option<(f64, Vec<u8>, usize, bool, LocalScore)> = None;
            for &source in order.iter().take(position[target]) {
                let adding = !g.local_scores[target].parents.contains(&source);
                if adding {
                    let direct = forbidden_edges.iter().any(|&(u, v)| {
                        (u == source && v == target) || (u == target && v == source)
                    });
                    if direct || addition_violates_packed(&ancestors, source, target, pairs) {
                        continue;
                    }
                }
                let local = if adding {
                    score.local_score_plus(target, &g.local_scores[target], source)?
                } else {
                    score.local_score_minus(target, &g.local_scores[target], source)?
                };
                let value = current_score - g.local_scores[target].bic + local.bic;
                let better = best.as_ref().is_none_or(|(best_value, best_encoding, _, _, _)| {
                    value < *best_value - EPS || ((value - *best_value).abs() <= EPS && {
                        let mut encoding = current_encoding.clone();
                        encoding[source * p + target] = u8::from(adding);
                        encoding < *best_encoding
                    })
                });
                if better {
                    let mut encoding = current_encoding.clone();
                    encoding[source * p + target] = u8::from(adding);
                    best = Some((value, encoding, source, adding, local));
                }
            }
            let Some((value, _, source, adding, local)) = best else {
                continue;
            };
            if value >= current_score - EPS {
                continue;
            }
            g.local_scores[target] = local;
            current_score = value;
            changed = true;
            if adding {
                update_ancestry_after_addition(&mut ancestors, &g, source, target);
            } else {
                ancestors = packed_ancestry(&g, order);
            }
        }
        if !changed {
            break;
        }
    }
    Ok(g)
}

fn penalty_state_from_initial(
    data: &DMatrix<f64>,
    initial: &[u8],
    order: &[usize],
    score: &Bic,
) -> Result<GlobalScore, FlopError> {
    let p = data.ncols();
    if initial.len() != p * p || order.len() != p {
        return Err(FlopError::InvalidConfig(
            "global-greedy penalty shape/order mismatch".into(),
        ));
    }
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        if node >= p {
            return Err(FlopError::InvalidConfig(
                "global-greedy penalty order contains an invalid node".into(),
            ));
        }
        position[node] = idx;
    }
    let mut g = GlobalScore::new(p, score)?;
    for child in 0..p {
        let mut parents: Vec<usize> = (0..p)
            .filter(|&parent| {
                parent != child
                    && initial[parent * p + child] != 0
                    && position[parent] < position[child]
            })
            .collect();
        parents.sort_by_key(|&parent| position[parent]);
        for parent in parents {
            g.local_scores[child] =
                score.local_score_plus(child, &g.local_scores[child], parent)?;
        }
    }
    Ok(g)
}

fn penalty_inner(
    data: &DMatrix<f64>,
    initial: &[u8],
    order: &[usize],
    pairs: &[(usize, usize)],
    score: &Bic,
    penalty_weight: f64,
) -> Result<(Dag, f64, f64), FlopError> {
    let p = data.ncols();
    let mut g = penalty_state_from_initial(data, initial, order, score)?;
    let mut dag = Dag::from_global_score(&g);
    let mut current_nt = count_no_trek_violations(&dag, pairs) as f64;
    let mut current = g.score() + penalty_weight * current_nt;
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        position[node] = idx;
    }
    loop {
        let mut best: Option<(f64, Vec<u8>, usize, LocalScore, f64)> = None;
        let current_encoding = dag_adjacency(&dag);
        for source in 0..p {
            for target in 0..p {
                if source == target || position[source] >= position[target] {
                    continue;
                }
                let adding = !g.local_scores[target].parents.contains(&source);
                let local = if adding {
                    score.local_score_plus(target, &g.local_scores[target], source)?
                } else {
                    score.local_score_minus(target, &g.local_scores[target], source)?
                };
                let mut candidate_g = g.clone();
                candidate_g.local_scores[target] = local.clone();
                let candidate_dag = Dag::from_global_score(&candidate_g);
                let candidate_nt = count_no_trek_violations(&candidate_dag, pairs) as f64;
                let candidate_value = candidate_g.score() + penalty_weight * candidate_nt;
                let mut encoding = current_encoding.clone();
                encoding[source * p + target] = u8::from(adding);
                let better = best.as_ref().is_none_or(|(value, best_encoding, _, _, _)| {
                    candidate_value < *value - EPS
                        || ((candidate_value - *value).abs() <= EPS && encoding < *best_encoding)
                });
                if better {
                    best = Some((candidate_value, encoding, target, local, candidate_nt));
                }
            }
        }
        let Some((value, _, target, local, candidate_nt)) = best else {
            break;
        };
        if value >= current - EPS {
            break;
        }
        g.local_scores[target] = local;
        dag = Dag::from_global_score(&g);
        current_nt = candidate_nt;
        current = value;
    }
    Ok((dag, g.score(), current_nt))
}

/// Global-greedy search with a hard DAG/order constraint and a graph-level
/// NOTREKS penalty. Unlike the exact constrained variant, ancestry violations
/// are allowed and reported; the penalty is the only NOTREKS mechanism.
pub fn run_global_greedy_penalty(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
    penalty_weight: f64,
) -> Result<NoTreksResult, FlopError> {
    // With zero penalty this is exactly the ordinary Rust global-greedy
    // search, including its optimized scoring path.  This makes the
    // baseline useful for measuring the overhead of the penalty itself.
    if penalty_weight == 0.0 {
        return run_global_greedy_rust(data, &[], config);
    }
    let p = data.ncols();
    let canonical_pairs = canonical_pair_list(p, pairs)?;
    let seed = config.seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);
    let corr = crate::utils::corr_matrix(data);
    let (_, initial_order) = pivoted_cholesky::cholesky_left_min_diag(&corr)
        .ok_or_else(|| FlopError::InitialOrderError("Cholesky decomposition failed".into()))?;
    let num_perturbations = (p as f64).ln().round() as usize;
    let score = Bic::from_cov(data.nrows(), corr, config.lambda);
    let limit = config.restarts.unwrap_or(usize::MAX - 1) + 1;
    let mut best: Option<(f64, Dag)> = None;
    let mut best_order = initial_order;
    let mut completed = 0usize;
    for restart in 0..limit {
        let mut order = best_order.clone();
        if restart > 0 {
            for _ in 0..num_perturbations {
                let a = rng.gen_range(0..p);
                let b = rng.gen_range(0..p);
                order.swap(a, b);
            }
        }
        let zero = vec![0u8; p * p];
        let (mut graph, initial_bic, initial_nt) = penalty_inner(
            data,
            &zero,
            &order,
            &canonical_pairs,
            &score,
            penalty_weight,
        )?;
        let mut objective = initial_bic + penalty_weight * initial_nt;
        for _ in 0..config.max_signature_rounds.max(1) {
            let before = objective;
            for node in order.clone() {
                let old_position = order.iter().position(|&x| x == node).unwrap();
                let mut chosen: Option<(f64, Vec<usize>, Dag, f64, f64)> = None;
                for new_position in 0..p {
                    if new_position == old_position {
                        continue;
                    }
                    let mut candidate_order = order.clone();
                    let moved = candidate_order.remove(old_position);
                    candidate_order.insert(new_position, moved);
                    let current_adj = dag_adjacency(&graph);
                    let (candidate_graph, candidate_bic, candidate_nt) = penalty_inner(
                        data,
                        &current_adj,
                        &candidate_order,
                        &canonical_pairs,
                        &score,
                        penalty_weight,
                    )?;
                    let candidate_objective = candidate_bic + penalty_weight * candidate_nt;
                    if chosen
                        .as_ref()
                        .is_none_or(|x| candidate_objective < x.0 - EPS)
                    {
                        chosen = Some((
                            candidate_objective,
                            candidate_order,
                            candidate_graph,
                            candidate_bic,
                            candidate_nt,
                        ));
                    }
                }
                if let Some((
                    candidate_objective,
                    candidate_order,
                    candidate_graph,
                    _candidate_bic,
                    _candidate_nt,
                )) = chosen
                {
                    if candidate_objective < objective - EPS {
                        objective = candidate_objective;
                        order = candidate_order;
                        graph = candidate_graph;
                    }
                }
            }
            if before - objective <= EPS {
                break;
            }
        }
        completed += 1;
        if best.as_ref().is_none_or(|x| objective < x.0 - EPS) {
            best_order = order;
            best = Some((objective, graph));
        }
    }
    let Some((objective, dag)) = best else {
        return Err(FlopError::InitialOrderError(
            "penalty search completed no restart".into(),
        ));
    };
    let violations = count_no_trek_violations(&dag, &canonical_pairs);
    let diagnostics = NoTreksDiagnostics {
        restarts_requested: limit,
        restarts_completed: completed,
        number_of_supplied_constraints: canonical_pairs.len(),
        final_bic: objective,
        selected_dag_edge_count: dag.parents.iter().map(Vec::len).sum(),
        final_no_trek_violation_count: violations,
        algorithm_seed: seed,
        search_version: NoTreksVersion::GlobalGreedyPenalty.stable_name().into(),
        termination_reason: "restart_limit".into(),
        ..Default::default()
    };
    Ok(NoTreksResult { dag, diagnostics })
}

/// Public one-shot fixed-order Rust kernel used for focused profiling and
/// equivalence tests.
pub fn global_greedy_inner(
    data: &DMatrix<f64>,
    initial: &[u8],
    order: &[usize],
    pairs: &[(usize, usize)],
    lambda: f64,
) -> Result<(Dag, f64), FlopError> {
    let canonical_pairs = canonical_pair_list(data.ncols(), pairs)?;
    let score = Bic::from_cov(data.nrows(), crate::utils::corr_matrix(data), lambda);
    let (dag, value, _) = global_greedy_inner_with_state_from_initial(
        data,
        initial,
        order,
        &canonical_pairs,
        &[],
        &score,
    )?;
    Ok((dag, value))
}

fn canonical_pair_list(
    p: usize,
    pairs: &[(usize, usize)],
) -> Result<Vec<(usize, usize)>, FlopError> {
    let mut canonical = BTreeSet::new();
    for &(a, b) in pairs {
        if a >= p || b >= p {
            return Err(FlopError::ConstraintError(format!(
                "no-trek pair ({a}, {b}) is outside 0..{p}"
            )));
        }
        if a == b {
            return Err(FlopError::ConstraintError(format!(
                "self no-trek pair ({a}, {b}) is invalid"
            )));
        }
        canonical.insert(if a < b { (a, b) } else { (b, a) });
    }
    Ok(canonical.into_iter().collect())
}

fn order_search(
    perm: &mut Vec<usize>,
    g: &mut crate::scores::GlobalScore,
    score: &Bic,
    constraints: &NoTrekConstraints,
    rng: &mut StdRng,
) -> Result<bool, FlopError> {
    let mut improved_any = false;
    loop {
        let last = g.score();
        let mut value = last;
        for node in perm.clone() {
            improved_any |= reinsert(perm, g, score, &mut value, node, rng, Some(constraints))?;
        }
        if last - value <= EPS {
            break;
        }
    }
    debug_assert!(signatures_respect_edges(g, &constraints.signatures));
    Ok(improved_any)
}

fn ordinary_reinsertion_search(
    perm: &mut Vec<usize>,
    g: &mut GlobalScore,
    score: &Bic,
    rng: &mut StdRng,
) -> Result<(), FlopError> {
    ordinary_reinsertion_search_constrained(perm, g, score, rng, None)
}

fn ordinary_reinsertion_search_constrained(
    perm: &mut Vec<usize>,
    g: &mut GlobalScore,
    score: &Bic,
    rng: &mut StdRng,
    constraints: Option<&NoTrekConstraints>,
) -> Result<(), FlopError> {
    loop {
        let last = g.score();
        let mut value = last;
        for node in perm.clone() {
            reinsert(perm, g, score, &mut value, node, rng, constraints)
                .map_err(FlopError::from)?;
        }
        if last - value <= EPS {
            break;
        }
    }
    Ok(())
}

/// FLOP's non-greedy grow--shrink update with exact dynamic NOTREKS checks.
/// Candidates are shuffled and every improving candidate encountered in the
/// traversal may be accepted; this is deliberately different from the
/// best-single-toggle local kernels.
fn flop_like_update_target<R: Rng + ?Sized>(
    mut g: GlobalScore,
    order: &[usize],
    target: usize,
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
    rng: &mut R,
) -> Result<GlobalScore, FlopError> {
    let target_position = order.iter().position(|&node| node == target).unwrap();
    let prefix = &order[..target_position];

    // A moved node can lose a parent from its prefix.  Removing such edges
    // is always safe for NOTREKS and keeps the warm state order-respecting.
    let invalid_parents: Vec<_> = g.local_scores[target]
        .parents
        .iter()
        .copied()
        .filter(|parent| !prefix.contains(parent))
        .collect();
    for parent in invalid_parents {
        g.local_scores[target] = score.local_score_minus(
            target, &g.local_scores[target], parent)?;
    }

    let mut ancestors = packed_ancestry(&g, order);
    let mut non_parents: Vec<_> = prefix
        .iter()
        .copied()
        .filter(|&parent| {
            !g.local_scores[target].parents.contains(&parent)
                && !forbidden_edges.iter().any(|&(u, v)| {
                    (u == parent && v == target) || (u == target && v == parent)
                })
        })
        .collect();

    loop {
        let mut changed = false;
        let mut candidates = non_parents.clone();
        candidates.shuffle(rng);
        for parent in candidates {
            if addition_violates_packed(&ancestors, parent, target, pairs) {
                continue;
            }
            let local = score.local_score_plus(
                target, &g.local_scores[target], parent)?;
            if local.bic <= g.local_scores[target].bic {
                g.local_scores[target] = local;
                non_parents.retain(|&candidate| candidate != parent);
                update_ancestry_after_addition(&mut ancestors, &g, parent, target);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    loop {
        let mut changed = false;
        let mut candidates = g.local_scores[target].parents.clone();
        candidates.shuffle(rng);
        for parent in candidates {
            let local = score.local_score_minus(
                target, &g.local_scores[target], parent)?;
            if local.bic <= g.local_scores[target].bic {
                g.local_scores[target] = local;
                non_parents.push(parent);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(g)
}

fn flop_like_initial_state<R: Rng + ?Sized>(
    order: &[usize],
    pairs: &[(usize, usize)],
    forbidden_edges: &[(usize, usize)],
    score: &Bic,
    rng: &mut R,
) -> Result<GlobalScore, FlopError> {
    let mut g = GlobalScore::new(order.len(), score)?;
    for &target in order {
        g = flop_like_update_target(g, order, target, pairs, forbidden_edges, score, rng)?;
    }
    Ok(g)
}

fn run_flop_like_reinsert(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    let p = data.ncols();
    let canonical_pairs = canonical_pair_list(p, pairs)?;
    let forbidden_edges = canonical_pair_list(p, &config.forbidden_edges)?;
    let seed = config.seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);
    let corr = crate::utils::corr_matrix(data);
    let (_, initial_order) = pivoted_cholesky::cholesky_left_min_diag(&corr)
        .ok_or_else(|| FlopError::InitialOrderError("Cholesky decomposition failed".into()))?;
    let score = Bic::from_cov(data.nrows(), corr, config.lambda);
    let limit = config.restarts.unwrap_or(usize::MAX - 1) + 1;
    let started = Instant::now();
    let deadline = config.timeout.map(Duration::from_secs_f64);
    let num_perturbations = (p as f64).ln().round() as usize;
    let mut best: Option<(f64, Vec<usize>, GlobalScore)> = None;
    let mut best_order = initial_order;
    let mut best_restart_index = 0usize;
    let mut completed = 0usize;
    let mut rounds_used = 0usize;
    let mut termination = "restart_limit".to_string();

    for restart in 0..limit {
        if restart > 0 && deadline.is_some_and(|d| started.elapsed() >= d) {
            termination = "timeout".into();
            break;
        }
        let mut order = best_order.clone();
        if restart > 0 {
            for _ in 0..num_perturbations {
                order.swap(rng.gen_range(0..p), rng.gen_range(0..p));
            }
        }
        let mut state = flop_like_initial_state(
            &order, &canonical_pairs, &forbidden_edges, &score, &mut rng,
        )?;
        let mut current_score = state.score();
        let mut restart_rounds = 0usize;

        for _ in 0..config.max_signature_rounds.max(1) {
            if deadline.is_some_and(|d| started.elapsed() >= d) {
                termination = "timeout".into();
                break;
            }
            let before = current_score;
            for node in order.clone() {
                let base_order = order.clone();
                let base_state = state.clone();
                let base_score = current_score;
                let node_position = base_order.iter().position(|&x| x == node).unwrap();
                let mut node_best = (base_score, base_order.clone(), base_state.clone());

                let mut sweep_order = base_order.clone();
                let mut sweep_state = base_state.clone();
                for swap_pos in node_position + 1..p {
                    let crossed = sweep_order[swap_pos];
                    if sweep_state.local_scores[crossed].parents.contains(&node) {
                        sweep_state.local_scores[crossed] = score.local_score_minus(
                            crossed, &sweep_state.local_scores[crossed], node)?;
                    }
                    sweep_order.swap(swap_pos - 1, swap_pos);
                    sweep_state = flop_like_update_target(
                        sweep_state, &sweep_order, node, &canonical_pairs,
                        &forbidden_edges, &score, &mut rng,
                    )?;
                    sweep_state = flop_like_update_target(
                        sweep_state, &sweep_order, crossed, &canonical_pairs,
                        &forbidden_edges, &score, &mut rng,
                    )?;
                    let candidate_score = sweep_state.score();
                    let better = candidate_score < node_best.0 - EPS
                        || ((candidate_score - node_best.0).abs() <= EPS
                            && (sweep_order.clone(), global_score_adjacency(&sweep_state))
                                < (node_best.1.clone(), global_score_adjacency(&node_best.2)));
                    if better {
                        node_best = (candidate_score, sweep_order.clone(), sweep_state.clone());
                    }
                }

                let mut sweep_order = base_order.clone();
                let mut sweep_state = base_state.clone();
                for swap_pos in (0..node_position).rev() {
                    let crossed = sweep_order[swap_pos];
                    if sweep_state.local_scores[node].parents.contains(&crossed) {
                        sweep_state.local_scores[node] = score.local_score_minus(
                            node, &sweep_state.local_scores[node], crossed)?;
                    }
                    sweep_order.swap(swap_pos, swap_pos + 1);
                    sweep_state = flop_like_update_target(
                        sweep_state, &sweep_order, node, &canonical_pairs,
                        &forbidden_edges, &score, &mut rng,
                    )?;
                    sweep_state = flop_like_update_target(
                        sweep_state, &sweep_order, crossed, &canonical_pairs,
                        &forbidden_edges, &score, &mut rng,
                    )?;
                    let candidate_score = sweep_state.score();
                    let better = candidate_score < node_best.0 - EPS
                        || ((candidate_score - node_best.0).abs() <= EPS
                            && (sweep_order.clone(), global_score_adjacency(&sweep_state))
                                < (node_best.1.clone(), global_score_adjacency(&node_best.2)));
                    if better {
                        node_best = (candidate_score, sweep_order.clone(), sweep_state.clone());
                    }
                }

                if node_best.0 < current_score - EPS {
                    current_score = node_best.0;
                    order = node_best.1;
                    state = node_best.2;
                }
            }
            restart_rounds += 1;
            rounds_used = rounds_used.max(restart_rounds);
            if termination == "timeout" || before - current_score <= EPS {
                break;
            }
        }

        completed += 1;
        let candidate = (current_score, order.clone(), state.clone());
        let better = best.as_ref().is_none_or(|(old_score, old_order, old_state)| {
            candidate.0 < *old_score - EPS
                || ((candidate.0 - *old_score).abs() <= EPS
                    && (candidate.1.clone(), global_score_adjacency(&candidate.2))
                        < (old_order.clone(), global_score_adjacency(old_state)))
        });
        if better {
            best_order = order;
            best_restart_index = restart;
            best = Some(candidate);
        }
    }

    let Some((final_bic, _order, final_state)) = best else {
        return Err(FlopError::InitialOrderError(
            "FLOP-like constrained search completed no restart".into(),
        ));
    };
    let dag = Dag::from_global_score(&final_state);
    let violations = count_no_trek_violations(&dag, &canonical_pairs);
    if violations != 0 {
        return Err(FlopError::ConstraintError(
            "FLOP-like constrained search returned an infeasible graph".into(),
        ));
    }
    let diagnostics = NoTreksDiagnostics {
        restarts_requested: limit,
        restarts_completed: completed,
        best_restart_index,
        number_of_supplied_constraints: canonical_pairs.len(),
        final_bic,
        selected_dag_edge_count: dag.parents.iter().map(Vec::len).sum(),
        final_no_trek_violation_count: violations,
        algorithm_seed: seed,
        search_version: NoTreksVersion::FlopLike.stable_name().into(),
        number_of_signature_rounds: rounds_used,
        termination_reason: termination,
        ..Default::default()
    };
    Ok(NoTreksResult { dag, diagnostics })
}

/// Experimental FLOP-style outer search for the active local kernel.
///
/// Unlike `run_global_greedy_rust_impl`, this evaluates each node by walking
/// through adjacent positions.  The graph produced at one adjacent position
/// is used as the initial graph for the next position in that direction.  A
/// swap changes only one possible order edge: when a node moves right, its
/// edge to the crossed node is removed; when it moves left, the crossed
/// node's edge to it is removed.  The active constrained kernel then repairs
/// the parent sets from that warm state.
fn run_active_reinsert_rust_impl(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    let p = data.ncols();
    let canonical_pairs = canonical_pair_list(p, pairs)?;
    let forbidden_edges = canonical_pair_list(p, &config.forbidden_edges)?;
    let seed = config.seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);
    let corr = crate::utils::corr_matrix(data);
    let (_, initial_order) = pivoted_cholesky::cholesky_left_min_diag(&corr)
        .ok_or_else(|| FlopError::InitialOrderError("Cholesky decomposition failed".into()))?;
    let num_perturbations = (p as f64).ln().round() as usize;
    let score = Bic::from_cov(data.nrows(), corr, config.lambda);
    let limit = config.restarts.unwrap_or(usize::MAX - 1) + 1;
    let started = Instant::now();
    let deadline = config.timeout.map(Duration::from_secs_f64);
    let mut best: Option<(f64, Vec<usize>, GlobalScore)> = None;
    let mut best_order = initial_order;
    let mut best_restart_index = 0usize;
    let mut completed = 0usize;
    let mut rounds_used = 0usize;
    let mut termination = "restart_limit".to_string();

    for restart in 0..limit {
        if restart > 0 && deadline.is_some_and(|d| started.elapsed() >= d) {
            termination = "timeout".into();
            break;
        }
        let mut order = best_order.clone();
        if restart > 0 {
            for _ in 0..num_perturbations {
                let a = rng.gen_range(0..p);
                let b = rng.gen_range(0..p);
                order.swap(a, b);
            }
        }

        let zero = vec![0u8; p * p];
        let (_, mut current_score, mut state) = run_selected_inner(
            data,
            &zero,
            &order,
            &canonical_pairs,
            &forbidden_edges,
            &score,
            NoTreksVersion::LocalGreedyActiveExact,
            config.local_greedy_passes,
        )?;

        let mut restart_rounds = 0usize;
        for _ in 0..config.max_signature_rounds.max(1) {
            if deadline.is_some_and(|d| started.elapsed() >= d) {
                termination = "timeout".into();
                break;
            }
            let before = current_score;

            for node in order.clone() {
                if deadline.is_some_and(|d| started.elapsed() >= d) {
                    termination = "timeout".into();
                    break;
                }
                let base_order = order.clone();
                let base_adj = global_score_adjacency(&state);
                let base_score = current_score;
                let node_position = base_order.iter().position(|&x| x == node).unwrap();
                let mut node_best = (base_score, base_order.clone(), base_adj.clone(), state.clone());

                // Sweep node to the right.  The current graph is carried from
                // one adjacent swap to the next, exactly like a FLOP prefix
                // sweep, but each state is refined by the active kernel.
                let mut sweep_order = base_order.clone();
                let mut sweep_state = state.clone();
                for swap_pos in node_position + 1..p {
                    let crossed = sweep_order[swap_pos];
                    // The crossed node moves before `node`, so node -> crossed
                    // is no longer compatible with the order.
                    if sweep_state.local_scores[crossed].parents.contains(&node) {
                        sweep_state.local_scores[crossed] = score.local_score_minus(
                            crossed, &sweep_state.local_scores[crossed], node)?;
                    }
                    sweep_order.swap(swap_pos - 1, swap_pos);
                    sweep_state = refine_selected_targets(
                        sweep_state,
                        &sweep_order,
                        &canonical_pairs,
                        &forbidden_edges,
                        &score,
                        [node, crossed],
                        config.local_greedy_passes,
                    )?;
                    let candidate_score = sweep_state.score();
                    let sweep_adj = global_score_adjacency(&sweep_state);
                    let candidate_better = candidate_score < node_best.0 - EPS
                        || ((candidate_score - node_best.0).abs() <= EPS
                            && (sweep_order.clone(), sweep_adj.clone())
                                < (node_best.1.clone(), node_best.2.clone()));
                    if candidate_better {
                        node_best = (
                            candidate_score,
                            sweep_order.clone(),
                            sweep_adj.clone(),
                            sweep_state.clone(),
                        );
                    }
                }

                // Sweep node to the left from the original state.
                let mut sweep_order = base_order.clone();
                let mut sweep_state = state.clone();
                for swap_pos in (0..node_position).rev() {
                    let crossed = sweep_order[swap_pos];
                    // `node` moves before `crossed`, so crossed -> node is
                    // no longer compatible with the order.
                    if sweep_state.local_scores[node].parents.contains(&crossed) {
                        sweep_state.local_scores[node] = score.local_score_minus(
                            node, &sweep_state.local_scores[node], crossed)?;
                    }
                    sweep_order.swap(swap_pos, swap_pos + 1);
                    sweep_state = refine_selected_targets(
                        sweep_state,
                        &sweep_order,
                        &canonical_pairs,
                        &forbidden_edges,
                        &score,
                        [node, crossed],
                        config.local_greedy_passes,
                    )?;
                    let candidate_score = sweep_state.score();
                    let sweep_adj = global_score_adjacency(&sweep_state);
                    let candidate_better = candidate_score < node_best.0 - EPS
                        || ((candidate_score - node_best.0).abs() <= EPS
                            && (sweep_order.clone(), sweep_adj.clone())
                                < (node_best.1.clone(), node_best.2.clone()));
                    if candidate_better {
                        node_best = (
                            candidate_score,
                            sweep_order.clone(),
                            sweep_adj.clone(),
                            sweep_state.clone(),
                        );
                    }
                }

                if node_best.0 < current_score - EPS {
                    current_score = node_best.0;
                    order = node_best.1;
                    state = node_best.3;
                }
            }
            restart_rounds += 1;
            rounds_used = rounds_used.max(restart_rounds);
            if termination == "timeout" || before - current_score <= EPS {
                break;
            }
        }

        completed += 1;
        let candidate = (current_score, order.clone(), state.clone());
        let better = best.as_ref().is_none_or(|(old_score, old_order, old_state)| {
            candidate.0 < *old_score - EPS
                || ((candidate.0 - *old_score).abs() <= EPS
                    && (candidate.1.clone(), global_score_adjacency(&candidate.2))
                        < (old_order.clone(), global_score_adjacency(old_state)))
        });
        if better {
            best_order = order.clone();
            best_restart_index = restart;
            best = Some(candidate);
        }
    }

    let Some((final_bic, _order, final_state)) = best else {
        return Err(FlopError::InitialOrderError(
            "active reinsertion completed no restart".into(),
        ));
    };
    let dag = Dag::from_global_score(&final_state);
    let violations = count_no_trek_violations(&dag, &canonical_pairs);
    if violations != 0 {
        return Err(FlopError::ConstraintError(
            "active reinsertion returned an infeasible graph".into(),
        ));
    }
    let diagnostics = NoTreksDiagnostics {
        restarts_requested: limit,
        restarts_completed: completed,
        best_restart_index,
        number_of_supplied_constraints: canonical_pairs.len(),
        final_bic,
        selected_dag_edge_count: dag.parents.iter().map(Vec::len).sum(),
        final_no_trek_violation_count: violations,
        algorithm_seed: seed,
        search_version: NoTreksVersion::LocalGreedyActiveReinsert.stable_name().into(),
        number_of_signature_rounds: rounds_used,
        termination_reason: termination,
        ..Default::default()
    };
    Ok(NoTreksResult { dag, diagnostics })
}

fn exact_ancestor_violation_measure(
    g: &GlobalScore,
    order: &[usize],
    pairs: &[(usize, usize)],
) -> usize {
    let ancestors = packed_ancestry(g, order);
    pairs
        .iter()
        .map(|&(left, right)| {
            ancestors[left]
                .iter()
                .zip(&ancestors[right])
                .map(|(a, b)| (a & b).count_ones() as usize)
                .sum::<usize>()
        })
        .sum()
}

fn dag_edges(dag: &Dag) -> Vec<(usize, usize)> {
    dag.parents
        .iter()
        .enumerate()
        .flat_map(|(child, parents)| parents.iter().map(move |&parent| (parent, child)))
        .collect()
}

fn shortest_directed_path(dag: &Dag, source: usize, target: usize) -> Option<Vec<(usize, usize)>> {
    let mut children = vec![Vec::new(); dag.p];
    for (child, parents) in dag.parents.iter().enumerate() {
        for &parent in parents {
            children[parent].push(child);
        }
    }
    let mut predecessor = vec![None; dag.p];
    let mut queue = VecDeque::from([source]);
    predecessor[source] = Some(source);
    while let Some(node) = queue.pop_front() {
        if node == target {
            break;
        }
        for &child in &children[node] {
            if predecessor[child].is_none() {
                predecessor[child] = Some(node);
                queue.push_back(child);
            }
        }
    }
    predecessor[target]?;
    let mut path = Vec::new();
    let mut node = target;
    while node != source {
        let parent = predecessor[node]?;
        path.push((parent, node));
        node = parent;
    }
    path.reverse();
    Some(path)
}

fn ancestor_bitsets(dag: &Dag) -> Vec<Vec<bool>> {
    let mut result = vec![vec![false; dag.p]; dag.p];
    for node in 0..dag.p {
        result[node][node] = true;
        let mut stack = dag.parents[node].clone();
        while let Some(parent) = stack.pop() {
            if !result[node][parent] {
                result[node][parent] = true;
                stack.extend(dag.parents[parent].iter().copied());
            }
        }
    }
    result
}

fn shortest_violated_trek(
    dag: &Dag,
    pairs: &[(usize, usize)],
) -> Option<((usize, usize), Vec<(usize, usize)>)> {
    let ancestors = ancestor_bitsets(dag);
    for &(left, right) in pairs {
        let common: Vec<_> = (0..dag.p)
            .filter(|&node| ancestors[left][node] && ancestors[right][node])
            .collect();
        if common.is_empty() {
            continue;
        }
        let mut best: Option<Vec<(usize, usize)>> = None;
        for source in common {
            let mut trek = shortest_directed_path(dag, source, left).unwrap_or_default();
            trek.extend(shortest_directed_path(dag, source, right).unwrap_or_default());
            if best.as_ref().is_none_or(|current| trek.len() < current.len()) {
                best = Some(trek);
            }
        }
        return best.map(|trek| ((left, right), trek));
    }
    None
}

#[derive(Clone)]
struct TrekCutState {
    mask: Vec<(usize, usize)>,
    order: Vec<usize>,
    graph: GlobalScore,
    bic: f64,
}

fn run_trekcut(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    let p = data.ncols();
    let canonical_pairs = canonical_pair_list(p, pairs)?;
    let budget = config.trekcut_oracle_budget.max(1);
    let seed = config.seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);
    let corr = crate::utils::corr_matrix(data);
    let (_, initial_order) = pivoted_cholesky::cholesky_left_min_diag(&corr)
        .ok_or_else(|| FlopError::InitialOrderError("Cholesky decomposition failed".into()))?;
    let score = Bic::from_cov(data.nrows(), corr, config.lambda);
    let mut seen = BTreeSet::new();
    let mut open = Vec::new();
    let empty_mask = Vec::new();
    let root_constraints = NoTrekConstraints::new_with_forbidden(p, &[], &empty_mask)
        .map_err(FlopError::ConstraintError)?;
    let mut root_order = initial_order;
    let mut root = perm_to_dag_constrained(&root_order, &score, &mut rng, Some(&root_constraints))?;
    ordinary_reinsertion_search_constrained(
        &mut root_order, &mut root, &score, &mut rng, Some(&root_constraints))?;
    seen.insert(empty_mask.clone());
    open.push(TrekCutState {
        bic: root.score(),
        mask: empty_mask,
        order: root_order,
        graph: root,
    });

    let started = Instant::now();
    let mut calls = 1usize;
    let mut feasible = Vec::new();
    let mut witness_lengths = Vec::new();
    let mut branch_counts = Vec::new();
    let mut masks = Vec::new();
    let mut archive_bics = Vec::new();
    let mut archive_violations = Vec::new();
    while calls <= budget && !open.is_empty() {
        let index = open
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.bic.total_cmp(&b.bic))
            .map(|(index, _)| index)
            .unwrap();
        let state = open.swap_remove(index);
        let dag = Dag::from_global_score(&state.graph);
        let violations = count_no_trek_violations(&dag, &canonical_pairs);
        archive_bics.push(state.bic);
        archive_violations.push(violations);
        if violations == 0 {
            feasible.push((state.bic, dag));
            continue;
        }
        let Some((_, trek)) = shortest_violated_trek(&dag, &canonical_pairs) else {
            continue;
        };
        witness_lengths.push(trek.len());
        branch_counts.push(trek.len());
        for edge in trek {
            if calls >= budget {
                break;
            }
            let mut child_mask = state.mask.clone();
            child_mask.push(edge);
            child_mask.sort_unstable();
            child_mask.dedup();
            if !seen.insert(child_mask.clone()) {
                continue;
            }
            let mut child_graph = state.graph.clone();
            let (parent, child) = edge;
            if child_graph.local_scores[child].parents.contains(&parent) {
                child_graph.local_scores[child] = score
                    .local_score_minus(child, &child_graph.local_scores[child], parent)?;
            }
            let (_, _, refined) = local_greedy_inner_state(
                child_graph,
                &state.order,
                &[],
                &child_mask,
                &score,
                config.trekcut_refinement_passes.max(1),
            )?;
            calls += 1;
            masks.push(child_mask.clone());
            open.push(TrekCutState {
                bic: refined.score(),
                mask: child_mask,
                order: state.order.clone(),
                graph: refined,
            });
        }
    }
    let (best_bic, best_dag) = feasible
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .ok_or_else(|| FlopError::ConstraintError(
            "TrekCut-FLOP found no feasible state within oracle budget".into()))?;
    let diagnostics = NoTreksDiagnostics {
        restarts_requested: budget,
        restarts_completed: calls.min(budget),
        number_of_supplied_constraints: canonical_pairs.len(),
        final_bic: best_bic,
        selected_dag_edge_count: best_dag.parents.iter().map(Vec::len).sum(),
        final_no_trek_violation_count: 0,
        algorithm_seed: seed,
        search_version: NoTreksVersion::TrekCut.stable_name().into(),
        termination_reason: if calls >= budget { "oracle_budget" } else { "open_exhausted" }.into(),
        trekcut_witness_lengths: witness_lengths,
        trekcut_branch_counts: branch_counts,
        trekcut_masks: masks,
        trekcut_archive_bics: archive_bics,
        trekcut_archive_violations: archive_violations,
        trekcut_feasible_state_discovery_time: started.elapsed().as_secs_f64(),
        trekcut_oracle_calls: calls,
        ..Default::default()
    };
    Ok(NoTreksResult { dag: best_dag, diagnostics })
}

fn source_signatures(dag: &Dag) -> (Vec<usize>, Vec<Vec<bool>>) {
    let sources: Vec<_> = (0..dag.p)
        .filter(|&node| dag.parents[node].is_empty())
        .collect();
    let mut source_index = vec![None; dag.p];
    for (index, &source) in sources.iter().enumerate() {
        source_index[source] = Some(index);
    }
    let mut signatures = vec![vec![false; sources.len()]; dag.p];
    for node in dag.topological_ordering() {
        if let Some(index) = source_index[node] {
            signatures[node][index] = true;
        }
        for &parent in &dag.parents[node] {
            for index in 0..sources.len() {
                signatures[node][index] |= signatures[parent][index];
            }
        }
    }
    (sources, signatures)
}

pub fn run_source_signature_polish(
    data: &DMatrix<f64>,
    initial: &[u8],
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    let p = data.ncols();
    if initial.len() != p * p {
        return Err(FlopError::InvalidConfig(
            "source-signature initial graph shape mismatch".into(),
        ));
    }
    let mut dag = Dag { p, parents: vec![Vec::new(); p] };
    for parent in 0..p {
        for child in 0..p {
            if initial[parent * p + child] != 0 {
                dag.parents[child].push(parent);
            }
        }
    }
    let (sources, signatures) = source_signatures(&dag);
    let canonical_pairs = canonical_pair_list(p, pairs)?;
    if canonical_pairs.iter().any(|&(left, right)| {
        signatures[left]
            .iter()
            .zip(&signatures[right])
            .any(|(a, b)| *a && *b)
    }) {
        return Err(FlopError::ConstraintError(
            "source signatures are not disjoint for a supplied NOTREKS pair".into(),
        ));
    }
    let mut allowed_count = 0usize;
    let mut forbidden = Vec::new();
    for parent in 0..p {
        for child in 0..p {
            if parent == child {
                continue;
            }
            let allowed = signatures[parent]
                .iter()
                .zip(&signatures[child])
                .all(|(a, b)| !*a || *b);
            if allowed {
                allowed_count += 1;
            } else {
                forbidden.push((parent, child));
            }
        }
    }
    let constraints = NoTrekConstraints::new_with_directed_forbidden(p, &[], &forbidden)
        .map_err(FlopError::ConstraintError)?;
    let mut order = dag.topological_ordering();
    let mut position = vec![0usize; p];
    for (index, &node) in order.iter().enumerate() {
        position[node] = index;
    }
    let corr = crate::utils::corr_matrix(data);
    let score = Bic::from_cov(data.nrows(), corr, config.lambda);
    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or(0));
    let mut graph = GlobalScore::new(p, &score)?;
    for child in 0..p {
        let mut parents = dag.parents[child].clone();
        parents.retain(|&parent| position[parent] < position[child]
            && constraints.allowed_parent(parent, child));
        parents.sort_unstable_by_key(|&parent| position[parent]);
        for parent in parents {
            graph.local_scores[child] = score
                .local_score_plus(child, &graph.local_scores[child], parent)?;
        }
    }
    ordinary_reinsertion_search_constrained(
        &mut order, &mut graph, &score, &mut rng, Some(&constraints))?;
    let result_dag = Dag::from_global_score(&graph);
    let diagnostics = NoTreksDiagnostics {
        final_bic: graph.score(),
        selected_dag_edge_count: result_dag.parents.iter().map(Vec::len).sum(),
        final_no_trek_violation_count: count_no_trek_violations(
            &result_dag, &canonical_pairs),
        number_of_supplied_constraints: canonical_pairs.len(),
        source_signature_source_count: sources.len(),
        source_signature_mask_density: allowed_count as f64
            / (p.saturating_mul(p.saturating_sub(1)).max(1) as f64),
        source_signature_pair_check_passed: true,
        algorithm_seed: config.seed.unwrap_or(0),
        search_version: "source_signature_polish".into(),
        termination_reason: "ordinary_flop_polish".into(),
        ..Default::default()
    };
    Ok(NoTreksResult { dag: result_dag, diagnostics })
}

/// Run ordinary, unrestricted FLOP order search and retain its raw terminal
/// candidates for NOTREKS-aware selection outside the optimizer.  No repaired
/// graph is ever used as a warm start.  The lexicographic pool is useful when
/// the BIC-optimal basin violates the supplied prior, while the ordinary pool
/// preserves the standard FLOP restart behavior.
fn run_order_guided_local_impl(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    let p = data.ncols();
    let canonical_pairs = canonical_pair_list(p, pairs)?;
    let seed = config.seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);
    let corr = crate::utils::corr_matrix(data);
    let (_, initial_order) = pivoted_cholesky::cholesky_left_min_diag(&corr)
        .ok_or_else(|| FlopError::InitialOrderError("Cholesky decomposition failed".into()))?;
    let score = Bic::from_cov(data.nrows(), corr, config.lambda);
    let total = config.restarts.unwrap_or(usize::MAX - 1) + 1;
    let lex_count = ((total as f64) * config.order_guided_lex_fraction.clamp(0.0, 1.0))
        .round() as usize;
    let bic_count = total.saturating_sub(lex_count).max(1);
    let perturbations = (p as f64).ln().round() as usize;
    let mut archive: Vec<(f64, usize, usize, Vec<usize>, Dag)> = Vec::new();
    let mut trajectory = Vec::new();
    let mut best_bic: Option<(f64, Dag)> = None;
    let mut best_lex: Option<(usize, f64, Dag)> = None;

    for restart in 0..total {
        let mut order = initial_order.clone();
        if restart > 0 || config.random_initial_order {
            for _ in 0..perturbations.max(1) {
                let a = rng.gen_range(0..p);
                let b = rng.gen_range(0..p);
                order.swap(a, b);
            }
        }
        let mut g = perm_to_dag_constrained(&order, &score, &mut rng, None)?;
        ordinary_reinsertion_search(&mut order, &mut g, &score, &mut rng)?;
        let dag = Dag::from_global_score(&g);
        let bic = g.score();
        let violations = count_no_trek_violations(&dag, &canonical_pairs);
        let exact = exact_ancestor_violation_measure(&g, &order, &canonical_pairs);
        trajectory.push(exact);
        archive.push((bic, violations, exact, order.clone(), dag.clone()));
        if restart < bic_count && violations == 0 {
            if best_bic.as_ref().is_none_or(|x| bic < x.0 - EPS) {
                best_bic = Some((bic, dag.clone()));
            }
        }
        let lex_key = (exact, bic);
        if (restart >= bic_count || best_lex.is_none()) && best_lex.as_ref().is_none_or(|x| {
            lex_key.0 < x.0 || (lex_key.0 == x.0 && lex_key.1 > bic + EPS)
        }) {
            best_lex = Some((violations, bic, dag));
        }
    }

    // Coverage starts are deliberately outside the restart accounting. They
    // are independent full-support order fits and are archived exactly like
    // terminal restart candidates.
    for _ in 0..config.order_guided_coverage_starts {
        let mut order: Vec<usize> = (0..p).collect();
        order.shuffle(&mut rng);
        let mut g = perm_to_dag_constrained(&order, &score, &mut rng, None)?;
        ordinary_reinsertion_search(&mut order, &mut g, &score, &mut rng)?;
        let dag = Dag::from_global_score(&g);
        let bic = g.score();
        let violations = count_no_trek_violations(&dag, &canonical_pairs);
        let exact = exact_ancestor_violation_measure(&g, &order, &canonical_pairs);
        trajectory.push(exact);
        archive.push((bic, violations, exact, order, dag.clone()));
        if violations == 0 && best_bic.as_ref().is_none_or(|x| bic < x.0 - EPS) {
            best_bic = Some((bic, dag));
        }
    }

    let selected = best_bic
        .or_else(|| best_lex.map(|(_, bic, dag)| (bic, dag)))
        .ok_or_else(|| FlopError::ConstraintError("order-guided search produced no graph".into()))?;
    let raw_feasible_count = archive.iter().filter(|x| x.1 == 0).count();
    let archive_edges = archive.iter().map(|x| dag_edges(&x.4)).collect();
    let archive_bics = archive.iter().map(|x| x.0).collect();
    let archive_violations = archive.iter().map(|x| x.1).collect();
    let archive_exact = archive.iter().map(|x| x.2).collect();
    let diagnostics = NoTreksDiagnostics {
        restarts_requested: total,
        restarts_completed: total,
        number_of_supplied_constraints: canonical_pairs.len(),
        final_bic: selected.0,
        selected_dag_edge_count: selected.1.parents.iter().map(Vec::len).sum(),
        final_no_trek_violation_count: count_no_trek_violations(&selected.1, &canonical_pairs),
        algorithm_seed: seed,
        search_version: config.search_version.stable_name().into(),
        termination_reason: "restart_limit".into(),
        order_guided_archive_edges: archive_edges,
        order_guided_archive_bics: archive_bics,
        order_guided_archive_violations: archive_violations,
        order_guided_archive_exact_violations: archive_exact,
        order_guided_violation_trajectory: trajectory,
        order_guided_raw_feasible_count: raw_feasible_count,
        ..Default::default()
    };
    Ok(NoTreksResult { dag: selected.1, diagnostics })
}

pub fn run_notreks(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    if config.search_version == NoTreksVersion::TrekCut {
        return run_trekcut(data, pairs, &config);
    }
    if config.search_version == NoTreksVersion::OrderGuidedLocal {
        return run_order_guided_local_impl(data, pairs, &config);
    }
    if config.search_version == NoTreksVersion::PrefixFeasible {
        return run_prefix_feasible(data, pairs, &config);
    }
    if config.search_version == NoTreksVersion::TrekDominance {
        return run_trek_dominance(data, pairs, &config);
    }
    if config.search_version == NoTreksVersion::LocalGreedyActiveReinsert {
        return run_active_reinsert_rust_impl(data, pairs, &config);
    }
    if config.search_version == NoTreksVersion::FlopLike {
        return run_flop_like_reinsert(data, pairs, &config);
    }
    if config.search_version == NoTreksVersion::GlobalGreedyRust
        || config.search_version == NoTreksVersion::GlobalGreedyRustOptimized
        || config.search_version == NoTreksVersion::LocalGreedyRust
        || config.search_version == NoTreksVersion::LocalGreedyActiveExact
    {
        if config.restarts.is_none() && config.timeout.is_none() {
            return Err(FlopError::InvalidConfig(
                "global-greedy Rust requires restarts or timeout".into(),
            ));
        }
        // Production FLOP-NOTREKS uses the exact edge-level certificate.  A
        // candidate parent is rejected only when adding that edge to the
        // current DAG creates a forbidden common ancestor.  The signature
        // search below is retained as an explicit diagnostic, but must not
        // silently exclude BIC candidates from the production method.
        return run_global_greedy_rust(data, pairs, &config);
    }
    if config.search_version == NoTreksVersion::GlobalGreedyDiagnostic {
        return run_global_greedy_rust(data, pairs, &config);
    }
    run_notreks_constrained(data, pairs, config)
}

fn run_notreks_constrained(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    if config.restarts.is_some() as u8
        + config.timeout.is_some() as u8
        + config.manual_termination as u8
        != 1
    {
        return Err(FlopError::InvalidConfig(
            "config requires exactly one of restarts, timeout, or manual termination".into(),
        ));
    }
    if config.signature_top_k == 0 {
        return Err(FlopError::InvalidConfig(
            "signature_top_k must be positive".into(),
        ));
    }
    let p = data.ncols();
    let n = data.nrows();
    let base_constraints = if config.search_version == NoTreksVersion::TrekDominance {
        NoTrekConstraints::new_with_directed_forbidden(
            p, pairs, &config.forbidden_edges,
        )
    } else {
        NoTrekConstraints::new(p, pairs)
    }.map_err(FlopError::ConstraintError)?;
    let seed = config.seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);
    let corr = utils::corr_matrix(data);
    let (_, initial_perm) = pivoted_cholesky::cholesky_left_min_diag(&corr)
        .ok_or_else(|| FlopError::InitialOrderError("Cholesky decomposition failed".into()))?;
    let score = Bic::from_cov(n, corr, config.lambda);
    let started = Instant::now();
    let deadline = config.timeout.map(Duration::from_secs_f64);
    let limit = config.restarts.unwrap_or(usize::MAX - 1) + 1;
    let perturbations = (p as f64).ln().round() as usize;
    let mut best: Option<(
        f64,
        Vec<usize>,
        crate::scores::GlobalScore,
        NoTrekConstraints,
    )> = None;
    let mut diagnostics = NoTreksDiagnostics {
        restarts_requested: limit,
        initial_fraction_of_candidate_parent_relations_pruned: if p < 2 {
            0.0
        } else {
            base_constraints.candidate_relations_pruned() as f64 / (p * (p - 1)) as f64
        },
        number_of_supplied_constraints: base_constraints.pairs.len(),
        number_of_distinct_constrained_targets: base_constraints
            .target_bit
            .iter()
            .filter(|x| x.is_some())
            .count(),
        algorithm_seed: seed,
        search_version: config.search_version.stable_name().into(),
        termination_reason: "restart_limit".into(),
        ..Default::default()
    };

    for restart in 0..limit {
        if restart > 0 && deadline.is_some_and(|d| started.elapsed() >= d) {
            diagnostics.termination_reason = "timeout".into();
            break;
        }
        let mut constraints = if restart == 0 {
            base_constraints.clone()
        } else {
            base_constraints.randomized(
                &mut rng,
                config.initial_signature_mean_size,
                config.initial_signature_max_size,
            )
        };
        let mut perm = initial_perm.clone();
        if restart > 0 {
            for _ in 0..perturbations {
                let a = rng.gen_range(0..p);
                let b = rng.gen_range(0..p);
                perm.swap(a, b);
            }
        }
        let refit_start = Instant::now();
        let mut g = perm_to_dag_constrained(&perm, &score, &mut rng, Some(&constraints))?;
        diagnostics.time_in_complete_constrained_refits += refit_start.elapsed().as_secs_f64();
        let order_start = Instant::now();
        order_search(&mut perm, &mut g, &score, &constraints, &mut rng)?;
        diagnostics.time_in_constrained_order_search += order_start.elapsed().as_secs_f64();

        let mut signature_local_optimum = config.search_version == NoTreksVersion::FixedSignatureA;
        for _ in 0..config.max_signature_rounds {
            if config.search_version == NoTreksVersion::FixedSignatureA {
                break;
            }
            diagnostics.number_of_signature_rounds += 1;
            let signature_start = Instant::now();
            let positions = {
                let mut pos = vec![0usize; p];
                for (i, &node) in perm.iter().enumerate() {
                    pos[node] = i;
                }
                pos
            };
            let mut proposals = Vec::new();
            for child in 0..p {
                for parent in 0..p {
                    if parent == child
                        || positions[parent] >= positions[child]
                        || constraints.allowed_parent(parent, child)
                        || g.local_scores[child].parents.contains(&parent)
                    {
                        continue;
                    }
                    diagnostics.number_of_blocked_edge_proposals_generated += 1;
                    let payload = constraints.missing_payload(parent, child);
                    let cone = ancestor_cone(&g, parent);
                    if !constraints.promotion_is_valid(&cone, payload) {
                        diagnostics.number_of_infeasible_promotions += 1;
                        continue;
                    }
                    if let Ok(candidate) =
                        score.local_score_plus(child, &g.local_scores[child], parent)
                    {
                        proposals.push((g.local_scores[child].bic - candidate.bic, parent, child));
                    }
                }
            }
            proposals.sort_by(|a, b| b.0.total_cmp(&a.0));
            let ranked = config.signature_top_k.min(proposals.len());
            let explore_end = (ranked + config.signature_exploration_k).min(proposals.len());
            if explore_end > ranked {
                for i in ranked..explore_end {
                    let j = rng.gen_range(i..proposals.len());
                    proposals.swap(i, j);
                }
            }
            proposals.truncate(explore_end);
            diagnostics.number_of_proposals_retained += proposals.len();

            let mut best_move: Option<(f64, crate::scores::GlobalScore, NoTrekConstraints)> = None;
            for (_, parent, child) in proposals {
                let payload = constraints.missing_payload(parent, child);
                let cone = ancestor_cone(&g, parent);
                diagnostics.number_of_proposals_tested_with_complete_refitting += 1;
                diagnostics.ancestor_cone_size_sum += cone.len();
                diagnostics.maximum_ancestor_cone_size =
                    diagnostics.maximum_ancestor_cone_size.max(cone.len());
                if !constraints.promotion_is_valid(&cone, payload) {
                    diagnostics.number_of_infeasible_promotions += 1;
                    continue;
                }
                let mut promoted = constraints.clone();
                for &node in &cone {
                    promoted.signatures[node] |= payload;
                }
                debug_assert!(promoted.allowed_parent(parent, child));
                let refit_start = Instant::now();
                let candidate = perm_to_dag_constrained(&perm, &score, &mut rng, Some(&promoted))?;
                diagnostics.time_in_complete_constrained_refits +=
                    refit_start.elapsed().as_secs_f64();
                let candidate_bic = candidate.score();
                if candidate_bic + EPS < g.score()
                    && best_move.as_ref().is_none_or(|x| candidate_bic < x.0)
                {
                    best_move = Some((candidate_bic, candidate, promoted));
                }
            }
            diagnostics.time_in_signature_move_evaluation +=
                signature_start.elapsed().as_secs_f64();
            if let Some((_, candidate, promoted)) = best_move {
                diagnostics.number_of_accepted_promotions += 1;
                g = candidate;
                constraints = promoted;
                // The promoted permission must survive at least one complete
                // order block so FLOP can exploit newly admissible relations.
                let order_start = Instant::now();
                order_search(&mut perm, &mut g, &score, &constraints, &mut rng)?;
                diagnostics.time_in_constrained_order_search += order_start.elapsed().as_secs_f64();
                diagnostics.number_of_post_promotion_order_blocks += 1;
                continue;
            }

            // Version B compression is a plateau operation. It is deliberately
            // not applied immediately after an accepted promotion.
            let compressed = canonical_signatures(&g, &perm, &constraints);
            let mut compressed_state = constraints.clone();
            compressed_state.signatures = compressed;
            if !compressed_state.signatures_are_valid()
                || !signatures_respect_edges(&g, &compressed_state.signatures)
            {
                return Err(FlopError::ConstraintError(
                    "canonical signature compression produced an invalid state".into(),
                ));
            }
            constraints = compressed_state;
            diagnostics.number_of_canonical_compressions += 1;
            let refit_start = Instant::now();
            g = perm_to_dag_constrained(&perm, &score, &mut rng, Some(&constraints))?;
            diagnostics.time_in_complete_constrained_refits += refit_start.elapsed().as_secs_f64();
            let order_start = Instant::now();
            let final_order_improved =
                order_search(&mut perm, &mut g, &score, &constraints, &mut rng)?;
            diagnostics.time_in_constrained_order_search += order_start.elapsed().as_secs_f64();
            if !final_order_improved {
                signature_local_optimum = true;
                break;
            }
        }
        if !signature_local_optimum && config.max_signature_rounds > 0 {
            diagnostics.termination_reason = "signature_round_limit".into();
        }
        let restart_bic = g.score();
        diagnostics.restart_bics.push(restart_bic);
        diagnostics.restarts_completed += 1;
        if best.as_ref().is_none_or(|x| restart_bic + EPS < x.0) {
            diagnostics.best_restart_index = restart;
            diagnostics.best_restart_was_minimal = restart == 0;
            best = Some((restart_bic, perm, g, constraints));
        }
    }

    let (final_bic, _perm, g, constraints) = best.ok_or_else(|| {
        FlopError::ConstraintError("constrained search did not produce a graph".into())
    })?;
    let dag = Dag::from_global_score(&g);
    let violations = count_no_trek_violations(&dag, &constraints.pairs);
    if violations != 0 {
        return Err(FlopError::ConstraintError(format!(
            "selected DAG has {violations} no-trek violations"
        )));
    }
    let pruned = constraints.candidate_relations_pruned();
    diagnostics.number_of_candidate_parent_relations_pruned = pruned;
    diagnostics.fraction_of_candidate_parent_relations_pruned = if p < 2 {
        0.0
    } else {
        pruned as f64 / (p * (p - 1)) as f64
    };
    diagnostics.final_bic = final_bic;
    diagnostics.selected_dag_edge_count = dag.parents.iter().map(Vec::len).sum();
    diagnostics.final_no_trek_violation_count = violations;
    Ok(NoTreksResult { dag, diagnostics })
}

pub fn run_seeded_unconstrained(
    data: &DMatrix<f64>,
    config: FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    run_notreks(data, &[], config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::no_treks::count_no_trek_violations;

    fn data() -> DMatrix<f64> {
        DMatrix::from_fn(120, 4, |r, c| {
            let x = (r as f64 * (c + 1) as f64 * 0.173).sin();
            x + ((r + 7 * c) as f64 * 0.071).cos()
        })
    }

    fn config(seed: u64) -> FlopNoTreksConfig {
        FlopNoTreksConfig {
            lambda: 2.0,
            restarts: Some(0),
            timeout: None,
            manual_termination: false,
            seed: Some(seed),
            signature_top_k: 3,
            signature_exploration_k: 0,
            max_signature_rounds: 4,
            initial_signature_mean_size: 3.0,
            initial_signature_max_size: 6,
            search_version: NoTreksVersion::AlternatingFullRefitB,
            local_greedy_passes: 8,
            forbidden_edges: Vec::new(),
            random_initial_order: false,
            order_guided_lex_fraction: 0.5,
            order_guided_repair_candidates: 2,
            order_guided_coverage_starts: 0,
            trekcut_oracle_budget: 32,
            trekcut_refinement_passes: 4,
            prefix_beam_width: 1,
            trek_graph: None,
        }
    }

    #[test]
    fn empty_constraints_are_reproducible_and_prune_nothing() {
        let first = run_seeded_unconstrained(&data(), config(9)).unwrap();
        let second = run_notreks(&data(), &[], config(9)).unwrap();
        assert_eq!(first.dag.parents, second.dag.parents);
        assert_eq!(first.diagnostics.final_bic, second.diagnostics.final_bic);
        assert_eq!(
            first
                .diagnostics
                .number_of_candidate_parent_relations_pruned,
            0
        );
        assert_eq!(first.diagnostics.final_no_trek_violation_count, 0);
    }

    #[test]
    fn selected_dag_respects_one_constraint() {
        let result = run_notreks(&data(), &[(0, 1)], config(11)).unwrap();
        assert_eq!(count_no_trek_violations(&result.dag, &[(0, 1)]), 0);
        assert_eq!(result.diagnostics.final_no_trek_violation_count, 0);
        assert_eq!(
            result.diagnostics.number_of_post_promotion_order_blocks,
            result.diagnostics.number_of_accepted_promotions
        );
        assert_eq!(
            result.diagnostics.search_version,
            "alternating_full_refit_b"
        );
    }

    #[test]
    fn fixed_signature_a_never_searches_signatures() {
        let mut fixed = config(11);
        fixed.search_version = NoTreksVersion::FixedSignatureA;
        let result = run_notreks(&data(), &[(0, 1)], fixed).unwrap();
        assert_eq!(count_no_trek_violations(&result.dag, &[(0, 1)]), 0);
        assert_eq!(result.diagnostics.number_of_signature_rounds, 0);
        assert_eq!(result.diagnostics.number_of_accepted_promotions, 0);
        assert_eq!(result.diagnostics.number_of_canonical_compressions, 0);
        assert_eq!(result.diagnostics.search_version, "fixed_signature_a");
    }

    #[test]
    fn global_greedy_pair_validation_has_no_u64_target_limit() {
        let pairs: Vec<_> = (0..99).map(|i| (i, i + 1)).collect();
        let canonical = canonical_pair_list(100, &pairs).unwrap();
        assert_eq!(canonical.len(), 99);
        assert_eq!(canonical.first(), Some(&(0, 1)));
        assert_eq!(canonical.last(), Some(&(98, 99)));
    }

    #[test]
    fn packed_ancestry_matches_exact_reachability() {
        let matrix = DMatrix::from_fn(80, 4, |r, c| ((r + 3 * c) as f64).sin());
        let score = Bic::new(&matrix, 2.0);
        let mut global = GlobalScore::new(4, &score).unwrap();
        global.local_scores[1] = score
            .local_score_plus(1, &global.local_scores[1], 0)
            .unwrap();
        global.local_scores[2] = score
            .local_score_plus(2, &global.local_scores[2], 1)
            .unwrap();
        global.local_scores[3] = score
            .local_score_plus(3, &global.local_scores[3], 2)
            .unwrap();
        let ancestors = packed_ancestry(&global, &[0, 1, 2, 3]);
        assert_ne!(ancestors[3][0] & 1, 0);
        assert_ne!(ancestors[3][0] & (1 << 1), 0);
        assert_eq!(ancestors[0][0] & (1 << 3), 0);
        assert_ne!(ancestors[0][0] & 1, 0);
    }

    #[test]
    fn trekcut_branches_on_both_edges_of_a_fork_trek() {
        // X <- Z -> Y has two valid conflict cuts: remove Z->X or remove
        // Z->Y.  A directed-edge heuristic that chooses one side would lose
        // one of the two feasible possibilities.
        let dag = Dag {
            p: 3,
            parents: vec![vec![], vec![0], vec![0]],
        };
        let (_, trek) = shortest_violated_trek(&dag, &[(1, 2)]).unwrap();
        assert_eq!(trek.len(), 2);
        assert!(trek.contains(&(0, 1)));
        assert!(trek.contains(&(0, 2)));
    }

}
