use std::thread;
use std::time::Duration;

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use std::sync::atomic::Ordering;

use crate::bic::Bic;
use crate::error::{FlopError, ScoreError};
use crate::global_abort::GLOBAL_ABORT;
use crate::graph::Dag;
use crate::no_treks::NoTrekConstraints;
use crate::scores::{GlobalScore, LocalScore};
use crate::token_buffer::TokenBuffer;
use crate::{fit_parents, pivoted_cholesky, utils};

static EPS: f64 = 1e-9;

pub struct FlopConfig {
    lambda: f64,
    restarts: Option<usize>,
    timeout: Option<f64>,
    manual_termination: bool,
}

impl FlopConfig {
    pub fn new(
        lambda: f64,
        restarts: Option<usize>,
        timeout: Option<f64>,
        manual_termination: bool,
    ) -> Self {
        Self {
            lambda,
            restarts,
            timeout,
            manual_termination,
        }
    }
}

pub fn run(data: &DMatrix<f64>, config: FlopConfig) -> Result<Dag, FlopError> {
    // exactly one termination criterion can be configured
    if config.restarts.is_some() as u8
        + config.timeout.is_some() as u8
        + config.manual_termination as u8
        != 1
    {
        return Err(FlopError::InvalidConfig(
            "config is missing number of restarts xor timeout xor manual termination flag"
                .to_owned(),
        ));
    }

    if config.manual_termination {
        ctrlc::set_handler(|| {
            GLOBAL_ABORT.store(true, Ordering::SeqCst);
        })
        .expect("Error setting Ctrl-C handler");
    }

    if let Some(timeout) = config.timeout {
        thread::spawn(move || {
            thread::sleep(Duration::from_secs_f64(timeout));
            GLOBAL_ABORT.store(true, Ordering::SeqCst);
        });
    }

    let p = data.ncols();
    let n = data.nrows();

    let mut rng = thread_rng();
    let num_perturbations = (p as f64).ln().round() as usize;

    let corr = utils::corr_matrix(data);

    let mut best_perm = match pivoted_cholesky::cholesky_left_min_diag(&corr) {
        None => Err(FlopError::InitialOrderError(
            "Cholesky decomposition failed".to_owned(),
        ))?,
        Some((_, order)) => order,
    };

    let score = Bic::from_cov(n, corr, config.lambda);

    let mut best_bic = f64::MAX;
    let mut best_g = None;

    // number of ILS + 1 and if not configured essentially infinite
    let limit = config.restarts.unwrap_or(usize::MAX - 1) + 1;

    for iter in 0..limit {
        let mut perm = best_perm.clone();
        if iter > 0 {
            for _ in 0..num_perturbations {
                let u = rng.gen_range(0..perm.len());
                let v = rng.gen_range(0..perm.len());
                perm.swap(u, v);
            }
        }

        let mut g = fit_parents::perm_to_dag(&perm, &score, &mut rng)?;
        let mut bic = g.score();

        'outer: loop {
            let last_bic = bic;

            for x in perm.clone() {
                reinsert(&mut perm, &mut g, &score, &mut bic, x, &mut rng, None)?;
                if iter > 0 && GLOBAL_ABORT.load(Ordering::SeqCst) {
                    break 'outer;
                }
            }

            // break if no improvement during full iteration
            if last_bic - bic <= EPS {
                break;
            }
        }

        // need to be at least EPS better than previous optimum
        if best_bic - bic > EPS {
            best_bic = bic;
            best_perm = perm;
            best_g = Some(g);
        }
    }

    Ok(Dag::from_global_score(&best_g.unwrap()))
}

#[derive(Clone, Debug)]
pub struct SourcePrefixDiagnostics {
    pub source_prefix: usize,
    pub restarts_completed: usize,
    pub full_order_refits: usize,
    pub accepted_reinsertions: usize,
    pub total_sweeps: usize,
    pub maximum_sweeps_per_restart: usize,
    pub final_bic: f64,
    pub selected_order: Vec<usize>,
}

/// FLOP order search with only one extra restriction: the first k variables
/// in every candidate causal order are fitted as source nodes.
///
/// A move across the k-boundary changes which variable is forced to be a
/// source.  The incremental reinsertion kernel handles only that boundary
/// transition with a from-scratch local parent fit; all other positions reuse
/// ordinary FLOP's two-local-score updates.
pub fn run_source_prefix(
    data: &DMatrix<f64>,
    lambda: f64,
    source_prefix: usize,
    restarts: usize,
    seed: u64,
) -> Result<(Dag, SourcePrefixDiagnostics), FlopError> {
    let p = data.ncols();
    if p == 0 || source_prefix == 0 || source_prefix > p {
        return Err(FlopError::InvalidConfig(format!(
            "source_prefix must be in 1..={p}"
        )));
    }
    let corr = utils::corr_matrix(data);
    let mut best_perm = pivoted_cholesky::cholesky_left_min_diag(&corr)
        .ok_or_else(|| FlopError::InitialOrderError("Cholesky decomposition failed".to_owned()))?
        .1;
    let score = Bic::from_cov(data.nrows(), corr, lambda);
    let mut rng = StdRng::seed_from_u64(seed);
    let perturbations = (p as f64).ln().round() as usize;
    let mut winner: Option<(f64, Vec<usize>, GlobalScore)> = None;
    let mut full_order_refits = 0usize;
    let mut accepted_reinsertions = 0usize;
    let mut total_sweeps = 0usize;
    let mut maximum_sweeps_per_restart = 0usize;

    // Match FLOP's convention: `restarts` means the initial run plus this
    // many perturbed starts.
    for iter in 0..=restarts {
        let mut perm = best_perm.clone();
        if iter > 0 {
            for _ in 0..perturbations {
                let u = rng.gen_range(0..p);
                let v = rng.gen_range(0..p);
                perm.swap(u, v);
            }
        }
        let mut graph =
            fit_parents::perm_to_dag_source_prefix(&perm, source_prefix, &score, &mut rng)?;
        full_order_refits += 1;
        let mut current = graph.score();

        let mut restart_sweeps = 0usize;
        loop {
            restart_sweeps += 1;
            total_sweeps += 1;
            let sweep_start = current;
            for node in perm.clone() {
                if reinsert_source_prefix(
                    &mut perm,
                    &mut graph,
                    &score,
                    &mut current,
                    node,
                    source_prefix,
                    &mut rng,
                    None,
                )? {
                    accepted_reinsertions += 1;
                }
            }
            if sweep_start - current <= EPS {
                break;
            }
        }
        maximum_sweeps_per_restart = maximum_sweeps_per_restart.max(restart_sweeps);
        let better = winner.as_ref().is_none_or(|(old_score, old_order, _)| {
            current < *old_score - EPS || ((current - *old_score).abs() <= EPS && perm < *old_order)
        });
        if better {
            best_perm = perm.clone();
            winner = Some((current, perm, graph));
        }
    }
    let (final_bic, selected_order, graph) = winner.ok_or_else(|| {
        FlopError::InitialOrderError("source-prefix FLOP completed no run".to_owned())
    })?;
    let dag = Dag::from_global_score(&graph);
    debug_assert!(selected_order[..source_prefix]
        .iter()
        .all(|&node| dag.parents[node].is_empty()));
    Ok((
        dag,
        SourcePrefixDiagnostics {
            source_prefix,
            restarts_completed: restarts + 1,
            full_order_refits,
            accepted_reinsertions,
            total_sweeps,
            maximum_sweeps_per_restart,
            final_bic,
            selected_order,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn reinsert_source_prefix<R: Rng + ?Sized>(
    perm: &mut Vec<usize>,
    g: &mut GlobalScore,
    score: &Bic,
    score_value: &mut f64,
    v: usize,
    source_prefix: usize,
    rng: &mut R,
    constraints: Option<&NoTrekConstraints>,
) -> Result<bool, ScoreError> {
    let v_index = perm.iter().position(|&x| x == v).unwrap();
    let mut v_curr_local = g.local_scores[v].clone();
    let mut best_diff = EPS;
    let mut best_ins_pos = v_index;
    let mut curr_diff = 0.0;
    let mut v_best_local: Vec<Option<LocalScore>> = vec![None; perm.len()];
    let mut z_best_local: Vec<Option<LocalScore>> = vec![None; perm.len()];
    let mut tokens = TokenBuffer::new(g.p);

    for pos in (0..v_index).rev() {
        let z = perm[pos];
        let mut prefix = perm[0..pos].to_vec();
        let v_new_local = if pos < source_prefix {
            score.local_score_init(v, Vec::new())?
        } else {
            fit_parents::fit_parents_minus_constrained(
                v,
                &v_curr_local,
                &prefix,
                z,
                score,
                &mut tokens,
                rng,
                constraints,
            )?
        };
        let v_score_diff = v_new_local.bic - v_curr_local.bic;
        v_curr_local = v_new_local.clone();

        prefix.push(v);
        let z_curr_local = &g.local_scores[z];
        let z_new_position = pos + 1;
        let z_new_local = if z_new_position < source_prefix {
            score.local_score_init(z, Vec::new())?
        } else if pos < source_prefix {
            // z is the single node leaving the source block.
            fit_parents::fit_parents_constrained(z, &prefix, score, rng, constraints)?
        } else {
            fit_parents::fit_parents_plus_constrained(
                z,
                z_curr_local,
                &prefix,
                v,
                score,
                &mut tokens,
                rng,
                constraints,
            )?
        };
        curr_diff += v_score_diff + z_new_local.bic - z_curr_local.bic;
        if curr_diff < best_diff {
            best_diff = curr_diff;
            best_ins_pos = pos;
            v_best_local[pos] = Some(v_new_local);
        }
        z_best_local[pos] = Some(z_new_local);
    }

    curr_diff = 0.0;
    v_curr_local = g.local_scores[v].clone();
    for pos in v_index + 1..perm.len() {
        let z = perm[pos];
        let mut prefix = perm[0..pos + 1].to_vec();
        utils::rem_first(&mut prefix, v);
        let v_new_local = if pos < source_prefix {
            score.local_score_init(v, Vec::new())?
        } else if pos == source_prefix && v_index < source_prefix {
            // v is the single node leaving the source block.
            fit_parents::fit_parents_constrained(v, &prefix, score, rng, constraints)?
        } else {
            fit_parents::fit_parents_plus_constrained(
                v,
                &v_curr_local,
                &prefix,
                z,
                score,
                &mut tokens,
                rng,
                constraints,
            )?
        };
        let v_score_diff = v_new_local.bic - v_curr_local.bic;
        v_curr_local = v_new_local.clone();

        utils::rem_first(&mut prefix, z);
        let z_curr_local = &g.local_scores[z];
        let z_new_local = if pos - 1 < source_prefix {
            score.local_score_init(z, Vec::new())?
        } else {
            fit_parents::fit_parents_minus_constrained(
                z,
                z_curr_local,
                &prefix,
                v,
                score,
                &mut tokens,
                rng,
                constraints,
            )?
        };
        curr_diff += v_score_diff + z_new_local.bic - z_curr_local.bic;
        if curr_diff < best_diff {
            best_diff = curr_diff;
            best_ins_pos = pos;
            v_best_local[pos] = Some(v_new_local);
        }
        z_best_local[pos] = Some(z_new_local);
    }

    if best_ins_pos == v_index {
        return Ok(false);
    }
    *score_value += best_diff;
    g.local_scores[v] = v_best_local[best_ins_pos].clone().unwrap();
    if best_ins_pos < v_index {
        for (i, &z) in perm[best_ins_pos..v_index].iter().enumerate() {
            g.local_scores[z] = z_best_local[best_ins_pos + i].clone().unwrap();
        }
    } else {
        for (i, &z) in perm[v_index + 1..best_ins_pos + 1].iter().enumerate() {
            g.local_scores[z] = z_best_local[v_index + i + 1].clone().unwrap();
        }
    }
    perm.remove(v_index);
    perm.insert(best_ins_pos, v);
    Ok(true)
}

pub(crate) fn reinsert<R: Rng + ?Sized>(
    perm: &mut Vec<usize>,
    g: &mut GlobalScore,
    score: &Bic,
    score_value: &mut f64,
    v: usize,
    rng: &mut R,
    constraints: Option<&NoTrekConstraints>,
) -> Result<bool, ScoreError> {
    let v_index = perm.iter().position(|&x| x == v).unwrap();
    let mut v_curr_local = g.local_scores[v].clone();

    let mut best_diff = EPS; // allow small worsening in single moves
    let mut best_ins_pos = v_index;
    let mut curr_diff = 0.0;

    let mut v_best_local: Vec<Option<LocalScore>> = vec![None; perm.len()];
    let mut z_best_local: Vec<Option<LocalScore>> = vec![None; perm.len()];
    let mut tokens = TokenBuffer::new(g.p);

    // look at positions preceding v
    for pos in (0..v_index).rev() {
        // try to reinsert BEFORE element at pos, which we term z
        let z = perm[pos];
        let mut prefix = perm[0..pos].to_vec();

        let v_new_local = fit_parents::fit_parents_minus_constrained(
            v,
            &v_curr_local,
            &prefix,
            z,
            score,
            &mut tokens,
            rng,
            constraints,
        )?;
        let v_score_diff = v_new_local.bic - v_curr_local.bic;
        v_curr_local = v_new_local.clone();

        // parents of z are updated based on addition of v
        prefix.push(v);
        let z_curr_local = &g.local_scores[z];
        let z_new_local = fit_parents::fit_parents_plus_constrained(
            z,
            z_curr_local,
            &prefix,
            v,
            score,
            &mut tokens,
            rng,
            constraints,
        )?;
        let z_score_diff = z_new_local.bic - z_curr_local.bic;

        curr_diff += v_score_diff + z_score_diff;
        if curr_diff < best_diff {
            best_diff = curr_diff;
            best_ins_pos = pos;
            // this will only be needed if v is put at pos
            v_best_local[pos] = Some(v_new_local);
        }
        z_best_local[pos] = Some(z_new_local);
    }
    // look at positions succeeding v
    // start with some resets
    curr_diff = 0.0;
    v_curr_local = g.local_scores[v].clone();

    for pos in v_index + 1..perm.len() {
        // try to reinsert AFTER element at pos, which we again term z
        let z = perm[pos];
        let mut prefix = perm[0..pos + 1].to_vec();
        // remove v from prefix
        utils::rem_first(&mut prefix, v);
        // parents of v are updated based on addition of z
        let v_new_local = fit_parents::fit_parents_plus_constrained(
            v,
            &v_curr_local,
            &prefix,
            z,
            score,
            &mut tokens,
            rng,
            constraints,
        )?;
        let v_score_diff = v_new_local.bic - v_curr_local.bic;
        v_curr_local = v_new_local.clone();

        // remove z from prefix
        utils::rem_first(&mut prefix, z);
        let z_curr_local = &g.local_scores[z];
        // parents of z are updated based on removal of v
        let z_new_local = fit_parents::fit_parents_minus_constrained(
            z,
            z_curr_local,
            &prefix,
            v,
            score,
            &mut tokens,
            rng,
            constraints,
        )?;
        let z_score_diff = z_new_local.bic - z_curr_local.bic;

        curr_diff += v_score_diff + z_score_diff;
        if curr_diff < best_diff {
            best_diff = curr_diff;
            best_ins_pos = pos;
            // this will only be needed if v is put at pos
            v_best_local[pos] = Some(v_new_local);
        }
        z_best_local[pos] = Some(z_new_local);
    }

    if best_ins_pos == v_index {
        return Ok(false);
    }

    *score_value += best_diff;

    g.local_scores[v] = v_best_local[best_ins_pos].clone().unwrap();
    if best_ins_pos < v_index {
        for (i, &z) in perm[best_ins_pos..v_index].iter().enumerate() {
            g.local_scores[z] = z_best_local[best_ins_pos + i].clone().unwrap();
        }
    } else {
        for (i, &z) in perm[v_index + 1..best_ins_pos + 1].iter().enumerate() {
            g.local_scores[z] = z_best_local[v_index + i + 1].clone().unwrap();
        }
    }
    perm.remove(v_index);
    perm.insert(best_ins_pos, v);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_prefix_nodes_are_parentless_and_result_is_reproducible() {
        let data = DMatrix::from_row_slice(
            8,
            4,
            &[
                0., 1., 1., 2., 1., 0., 2., 1., 2., 1., 3., 3., 3., 2., 5., 4., 4., 3., 7., 6., 5.,
                5., 8., 7., 6., 8., 11., 10., 7., 13., 13., 12.,
            ],
        );
        let first = run_source_prefix(&data, 2.0, 2, 0, 91).unwrap();
        let second = run_source_prefix(&data, 2.0, 2, 0, 91).unwrap();
        assert_eq!(first.1.selected_order, second.1.selected_order);
        assert_eq!(first.1.final_bic, second.1.final_bic);
        for &node in &first.1.selected_order[..2] {
            assert!(first.0.parents[node].is_empty());
        }
        let mut position = vec![0usize; 4];
        for (index, &node) in first.1.selected_order.iter().enumerate() {
            position[node] = index;
        }
        for (child, parents) in first.0.parents.iter().enumerate() {
            assert!(parents
                .iter()
                .all(|&parent| position[parent] < position[child]));
        }
    }

    #[test]
    fn source_prefix_rejects_zero_and_overdimensioned_values() {
        let data = DMatrix::identity(3, 3);
        assert!(run_source_prefix(&data, 2.0, 0, 0, 1).is_err());
        assert!(run_source_prefix(&data, 2.0, 4, 0, 1).is_err());
    }
}
