use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::algo::reinsert;
use crate::bic::Bic;
use crate::error::FlopError;
use crate::fit_parents::{perm_to_dag, perm_to_dag_constrained};
use crate::graph::Dag;
use crate::no_treks::{
    ancestor_cone, canonical_signatures, count_no_trek_violations, signatures_respect_edges,
    NoTrekConstraints,
};
use crate::scores::{GlobalScore, LocalScore};
use crate::{pivoted_cholesky, utils};

const EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoTreksVersion {
    FixedSignatureA,
    AlternatingFullRefitB,
    /// FLOP-style order reinsertion with the hard-feasible global-greedy
    /// inner search, executed entirely inside Rust.
    GlobalGreedyRust,
    /// Run ordinary FLOP first, then delete a small number of edges using
    /// the path-sum no-trek value and cached local BIC scores.
    CachedRepairC,
}

impl NoTreksVersion {
    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::FixedSignatureA => "fixed_signature_a",
            Self::AlternatingFullRefitB => "alternating_full_refit_b",
            Self::GlobalGreedyRust => "global_greedy_rust",
            Self::CachedRepairC => "cached_repair_c",
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
    pub repair_edges_removed: usize,
    pub repair_continuous_evaluations: usize,
    pub repair_initial_violation_count: usize,
    pub repair_initial_continuous_value: f64,
    pub repair_final_continuous_value: f64,
    pub repair_fallback_used: bool,
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

fn dag_adjacency(dag: &Dag) -> Vec<u8> {
    let mut matrix = vec![0u8; dag.p * dag.p];
    for (child, parents) in dag.parents.iter().enumerate() {
        for &parent in parents {
            matrix[parent * dag.p + child] = 1;
        }
    }
    matrix
}

fn order_key(order: &[usize]) -> Vec<usize> {
    order.to_vec()
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
    for (node, row) in ancestors.iter_mut().enumerate() {
        row[node / 64] |= 1_u64 << (node % 64);
    }
    for &node in order {
        for child in 0..p {
            if g.local_scores[child].parents.contains(&node) {
                let source = ancestors[node].clone();
                for (word, value) in source.into_iter().enumerate() {
                    ancestors[child][word] |= value;
                }
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
    for &(left, right) in pairs {
        let target_word = target / 64;
        let target_mask = 1_u64 << (target % 64);
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

/// Complete Rust implementation of the FLOP-style outer loop around the
/// fixed-order global-greedy inner search.  This keeps all candidate-order
/// evaluation in one extension call, avoiding Python/Rust crossings.
fn run_global_greedy_rust(
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
    let seed = config.seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);
    let corr = crate::utils::corr_matrix(data);
    pivoted_cholesky::cholesky_left_min_diag(&corr)
        .ok_or_else(|| FlopError::InitialOrderError("Cholesky decomposition failed".into()))?;
    let started = Instant::now();
    let deadline = config.timeout.map(Duration::from_secs_f64);
    let limit = config.restarts.unwrap_or(usize::MAX - 1) + 1;
    let score = Bic::from_cov(data.nrows(), corr, config.lambda);
    let mut best: Option<(f64, Vec<usize>, Dag)> = None;
    let mut completed = 0usize;
    let mut termination = "restart_limit".to_string();

    for restart in 0..limit {
        if restart > 0 && deadline.is_some_and(|d| started.elapsed() >= d) {
            termination = "timeout".into();
            break;
        }
        let mut order: Vec<usize> = (0..p).collect();
        order.shuffle(&mut rng);
        let mut zero = vec![0u8; p * p];
        let (mut graph, mut current_score) = global_greedy_inner_with_score(
            data, &zero, &order, &canonical_pairs, &score,
        )?;
        let mut graph_adj = dag_adjacency(&graph);
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
                let mut chosen: Option<(f64, Vec<usize>, Dag, Vec<u8>)> = None;
                for new_position in 0..p {
                    if new_position == old_position {
                        continue;
                    }
                    let mut candidate_order = order.clone();
                    let moved = candidate_order.remove(old_position);
                    candidate_order.insert(new_position, moved);
                    let (candidate_graph, candidate_score) = global_greedy_inner_with_score(
                        data, &graph_adj, &candidate_order, &canonical_pairs, &score,
                    )?;
                    let candidate_adj = dag_adjacency(&candidate_graph);
                    let better = chosen.as_ref().is_none_or(|(score, ord, _, enc)| {
                        candidate_score < *score - EPS
                            || ((candidate_score - *score).abs() <= EPS
                                && (order_key(&candidate_order), candidate_adj.clone())
                                    < (order_key(ord), enc.clone()))
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
                if let Some((score, candidate_order, candidate_graph, candidate_adj)) = chosen {
                    if score < current_score - EPS {
                        current_score = score;
                        order = candidate_order;
                        graph = candidate_graph;
                        graph_adj = candidate_adj;
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
        search_version: NoTreksVersion::GlobalGreedyRust.stable_name().into(),
        termination_reason: termination,
        ..Default::default()
    };
    let _ = order;
    Ok(NoTreksResult { dag, diagnostics })
}

/// Optimize a graph for one fixed order using the same edge-toggle greedy
/// inner loop as the Benchpress global-greedy comparator.  The caller owns the
/// FLOP-style order/reinsertion loop; this function only performs the
/// order-respecting, hard-NOTREKS-feasible inner optimization.
fn global_greedy_inner_with_score(
    data: &DMatrix<f64>,
    initial: &[u8],
    order: &[usize],
    pairs: &[(usize, usize)],
    score: &Bic,
) -> Result<(Dag, f64), FlopError> {
    let p = data.ncols();
    if initial.len() != p * p || order.len() != p {
        return Err(FlopError::InvalidConfig(
            "global-greedy inner shape/order mismatch".into(),
        ));
    }
    let mut position = vec![0usize; p];
    for (idx, &node) in order.iter().enumerate() {
        if node >= p {
            return Err(FlopError::InvalidConfig("global-greedy order contains an invalid node".into()));
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
            g.local_scores[child] = score.local_score_plus(
                child,
                &g.local_scores[child],
                parent,
            )?;
        }
    }

    let mut current_score = g.score();
    loop {
        let current_encoding = dag_adjacency(&Dag::from_global_score(&g));
        let mut best: Option<(f64, Vec<u8>, usize, LocalScore)> = None;
        let ancestors = packed_ancestry(&g, order);
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
                if adding
                    && addition_violates_packed(
                        &ancestors, source, target, pairs)
                {
                    continue;
                }
                let value = current_score - g.local_scores[target].bic + local.bic;
                let better = best.as_ref().is_none_or(|(best_value, best_encoding, _, _)| {
                    value < *best_value - EPS
                        || ((value - *best_value).abs() <= EPS
                            && {
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
    if count_no_trek_violations(&dag, pairs) != 0 {
        return Err(FlopError::ConstraintError(
            "global-greedy inner returned an infeasible graph".into(),
        ));
    }
    Ok((dag, current_score))
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
    let score = Bic::from_cov(
        data.nrows(),
        crate::utils::corr_matrix(data),
        lambda,
    );
    global_greedy_inner_with_score(data, initial, order, &canonical_pairs, &score)
}

fn canonical_pair_list(p: usize, pairs: &[(usize, usize)]) -> Result<Vec<(usize, usize)>, FlopError> {
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

fn unconstrained_order_search(
    perm: &mut Vec<usize>,
    g: &mut crate::scores::GlobalScore,
    score: &Bic,
    rng: &mut StdRng,
) -> Result<(), FlopError> {
    loop {
        let last = g.score();
        let mut value = last;
        for node in perm.clone() {
            reinsert(perm, g, score, &mut value, node, rng, None)?;
        }
        if last - value <= EPS {
            break;
        }
    }
    Ok(())
}

/// Fit the standardized linear parent coefficients represented by a DAG.
/// The resulting matrix uses the FLOP convention `W[parent, child]`.
fn regression_weights(cov: &DMatrix<f64>, dag: &Dag) -> DMatrix<f64> {
    let mut weights = DMatrix::<f64>::zeros(dag.p, dag.p);
    for child in 0..dag.p {
        refit_weight_column(cov, dag, child, &mut weights);
    }
    weights
}

fn refit_weight_column(cov: &DMatrix<f64>, dag: &Dag, child: usize, weights: &mut DMatrix<f64>) {
    for parent in 0..dag.p {
        weights[(parent, child)] = 0.0;
    }
    let parents = &dag.parents[child];
    if parents.is_empty() {
        return;
    }
    let parent_cov = utils::submatrix(cov, parents);
    let rhs = DMatrix::from_iterator(
        parents.len(),
        1,
        parents.iter().map(|&parent| cov[(parent, child)]),
    );
    if let Some(beta) = parent_cov.lu().solve(&rhs) {
        for (row, &parent) in parents.iter().enumerate() {
            weights[(parent, child)] = beta[(row, 0)];
        }
    }
}

/// Continuous inverse-resolvent NOTREKS value for a weighted DAG.
///
/// The weights are fitted parent-regression coefficients and the nonnegative
/// map is `abs(W)`, rather than `W * W`. Since the graph is acyclic, the
/// resolvent is well-defined and its off-diagonal Gram entries measure shared
/// weighted ancestry. This value guides repair only; exact ancestry remains
/// the final certificate.
fn continuous_notreks_value(weights: &DMatrix<f64>, pairs: &[(usize, usize)]) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let nonnegative = weights.map(|x| x.abs());
    let p = nonnegative.nrows();
    let identity = DMatrix::<f64>::identity(p, p);
    let Some(resolvent) = (identity - nonnegative).try_inverse() else {
        return f64::INFINITY;
    };
    let gram = resolvent.transpose() * resolvent;
    pairs
        .iter()
        .map(|&(i, j)| gram[(i, j)] + gram[(j, i)])
        .sum()
}

fn cached_repair(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: &FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    let p = data.ncols();
    let n = data.nrows();
    let canonical_pairs = NoTrekConstraints::new(p, pairs)
        .map_err(FlopError::ConstraintError)?
        .pairs;
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
    let mut best: Option<(f64, crate::scores::GlobalScore, usize)> = None;
    let mut diagnostics = NoTreksDiagnostics {
        restarts_requested: limit,
        number_of_supplied_constraints: canonical_pairs.len(),
        number_of_distinct_constrained_targets: canonical_pairs
            .iter()
            .flat_map(|&(a, b)| [a, b])
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
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
        let mut perm = initial_perm.clone();
        if restart > 0 {
            for _ in 0..perturbations {
                perm.swap(rng.gen_range(0..p), rng.gen_range(0..p));
            }
        }
        let mut g = perm_to_dag(&perm, &score, &mut rng)?;
        unconstrained_order_search(&mut perm, &mut g, &score, &mut rng)?;
        let mut dag = Dag::from_global_score(&g);
        let mut weights = regression_weights(score.covariance(), &dag);
        let initial_violations = count_no_trek_violations(&dag, &canonical_pairs);
        let initial_path = continuous_notreks_value(&weights, &canonical_pairs);
        diagnostics.repair_initial_violation_count += initial_violations;
        diagnostics.repair_initial_continuous_value += initial_path;
        let mut path_value = initial_path;
        let max_steps = g
            .local_scores
            .iter()
            .map(|x| x.parents.len())
            .sum::<usize>();
        let mut removed = 0usize;
        for _ in 0..max_steps {
            let current_violations = count_no_trek_violations(&dag, &canonical_pairs);
            if current_violations == 0 {
                break;
            }
            let mut best_move: Option<(usize, f64, f64, crate::scores::LocalScore, usize, usize)> =
                None;
            for child in 0..p {
                for &parent in &g.local_scores[child].parents {
                    let candidate_local =
                        score.local_score_minus(child, &g.local_scores[child], parent)?;
                    let mut candidate_g = g.clone();
                    candidate_g.local_scores[child] = candidate_local.clone();
                    let candidate_dag = Dag::from_global_score(&candidate_g);
                    let mut candidate_weights = weights.clone();
                    refit_weight_column(
                        score.covariance(),
                        &candidate_dag,
                        child,
                        &mut candidate_weights,
                    );
                    let candidate_path =
                        continuous_notreks_value(&candidate_weights, &canonical_pairs);
                    diagnostics.repair_continuous_evaluations += 1;
                    let candidate_violations =
                        count_no_trek_violations(&candidate_dag, &canonical_pairs);
                    // Deletion is allowed only when it reduces the exact hard
                    // violation count. The continuous value ranks those
                    // necessary repairs; it is never an independent sparsity
                    // objective.
                    if candidate_violations >= current_violations {
                        continue;
                    }
                    let bic_cost = candidate_local.bic - g.local_scores[child].bic;
                    let better = best_move.as_ref().is_none_or(|x| {
                        candidate_violations < x.0
                            || (candidate_violations == x.0
                                && (candidate_path < x.1 - EPS
                                    || ((candidate_path - x.1).abs() <= EPS && bic_cost < x.2)))
                    });
                    if better {
                        best_move = Some((
                            candidate_violations,
                            candidate_path,
                            bic_cost,
                            candidate_local,
                            parent,
                            child,
                        ));
                    }
                }
            }
            let Some((_, candidate_path, _, candidate_local, _parent, child)) = best_move else {
                break;
            };
            g.local_scores[child] = candidate_local;
            dag = Dag::from_global_score(&g);
            refit_weight_column(score.covariance(), &dag, child, &mut weights);
            path_value = candidate_path;
            removed += 1;
        }
        let final_violations = count_no_trek_violations(&dag, &canonical_pairs);
        if final_violations == 0 {
            diagnostics.repair_edges_removed += removed;
            diagnostics.repair_final_continuous_value += path_value;
            diagnostics.restarts_completed += 1;
            diagnostics.restart_bics.push(g.score());
            if best.as_ref().is_none_or(|x| g.score() + EPS < x.0) {
                diagnostics.best_restart_index = restart;
                best = Some((g.score(), g, restart));
            }
        }
    }
    if let Some((final_bic, g, restart)) = best {
        let dag = Dag::from_global_score(&g);
        diagnostics.best_restart_index = restart;
        diagnostics.final_bic = final_bic;
        diagnostics.selected_dag_edge_count = dag.parents.iter().map(Vec::len).sum();
        diagnostics.final_no_trek_violation_count =
            count_no_trek_violations(&dag, &canonical_pairs);
        return Ok(NoTreksResult { dag, diagnostics });
    }
    // The repair is deliberately conservative: if it cannot certify a result,
    // use the established constrained search rather than returning an invalid graph.
    diagnostics.repair_fallback_used = true;
    let mut fallback = config.clone();
    fallback.search_version = NoTreksVersion::AlternatingFullRefitB;
    let mut result = run_notreks_constrained(data, pairs, fallback)?;
    result.diagnostics.repair_fallback_used = true;
    result.diagnostics.search_version = NoTreksVersion::CachedRepairC.stable_name().into();
    result.diagnostics.termination_reason = "repair_fallback_constrained".into();
    Ok(result)
}

pub fn run_notreks(
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    config: FlopNoTreksConfig,
) -> Result<NoTreksResult, FlopError> {
    if config.search_version == NoTreksVersion::GlobalGreedyRust {
        if config.restarts.is_none() && config.timeout.is_none() {
            return Err(FlopError::InvalidConfig(
                "global-greedy Rust requires restarts or timeout".into(),
            ));
        }
        return run_global_greedy_rust(data, pairs, &config);
    }
    if config.search_version == NoTreksVersion::CachedRepairC {
        if config.restarts.is_none() && config.timeout.is_none() && !config.manual_termination {
            return Err(FlopError::InvalidConfig(
                "repair requires restarts or timeout".into(),
            ));
        }
        return cached_repair(data, pairs, &config);
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
    let base_constraints = NoTrekConstraints::new(p, pairs).map_err(FlopError::ConstraintError)?;
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

    let (final_bic, _, g, constraints) = best.ok_or_else(|| {
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
    fn cached_repair_certifies_the_returned_graph() {
        let mut cfg = config(1);
        cfg.search_version = NoTreksVersion::CachedRepairC;
        let result = run_notreks(&data(), &[(0, 1), (2, 3)], cfg).unwrap();
        assert_eq!(result.diagnostics.search_version, "cached_repair_c");
        assert_eq!(result.diagnostics.final_no_trek_violation_count, 0);
        assert_eq!(count_no_trek_violations(&result.dag, &[(0, 1), (2, 3)]), 0);
    }
}
