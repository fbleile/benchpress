use std::time::{Duration, Instant};

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::algo::reinsert;
use crate::bic::Bic;
use crate::error::FlopError;
use crate::fit_parents::perm_to_dag_constrained;
use crate::graph::Dag;
use crate::no_treks::{
    ancestor_cone, canonical_signatures, count_no_trek_violations, signatures_respect_edges,
    NoTrekConstraints,
};
use crate::{pivoted_cholesky, utils};

const EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoTreksVersion {
    FixedSignatureA,
    AlternatingFullRefitB,
}

impl NoTreksVersion {
    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::FixedSignatureA => "fixed_signature_a",
            Self::AlternatingFullRefitB => "alternating_full_refit_b",
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

pub fn run_notreks(
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
}
