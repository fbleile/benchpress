use rand::seq::SliceRandom;
use rand::Rng;

use crate::bic::Bic;
use crate::error::ScoreError;
use crate::no_treks::NoTrekConstraints;
use crate::scores::{GlobalScore, LocalScore};
use crate::token_buffer::TokenBuffer;
use crate::utils;

fn grow<R: Rng + ?Sized>(
    v: usize,
    v_local: &mut LocalScore,
    non_parents: &mut Vec<usize>,
    score: &Bic,
    rng: &mut R,
) -> Result<(), ScoreError> {
    loop {
        let mut done = true;
        let mut shuffled_non_parents = non_parents.clone();
        shuffled_non_parents.shuffle(rng);
        for &w in shuffled_non_parents.iter() {
            let v_local_new = score.local_score_plus(v, v_local, w)?;
            if v_local_new.bic <= v_local.bic {
                *v_local = v_local_new;
                utils::rem_first(non_parents, w);
                done = false;
            }
        }
        if done {
            break;
        }
    }
    Ok(())
}

fn shrink<R: Rng + ?Sized>(
    v: usize,
    v_local: &mut LocalScore,
    non_parents: &mut Vec<usize>,
    score: &Bic,
    rng: &mut R,
) -> Result<(), ScoreError> {
    loop {
        let mut done = true;
        let mut shuffled_parents = v_local.parents.clone();
        shuffled_parents.shuffle(rng);
        for &w in shuffled_parents.iter() {
            let v_local_new = score.local_score_minus(v, v_local, w)?;
            if v_local_new.bic <= v_local.bic {
                *v_local = v_local_new;
                non_parents.push(w);
                done = false;
            }
        }
        if done {
            break;
        }
    }
    Ok(())
}

// fit parents from scratch
#[allow(dead_code)]
pub fn fit_parents<R: Rng + ?Sized>(
    v: usize,
    prefix: &[usize],
    score: &Bic,
    rng: &mut R,
) -> Result<LocalScore, ScoreError> {
    fit_parents_constrained(v, prefix, score, rng, None)
}

pub fn fit_parents_constrained<R: Rng + ?Sized>(
    v: usize,
    prefix: &[usize],
    score: &Bic,
    rng: &mut R,
    constraints: Option<&NoTrekConstraints>,
) -> Result<LocalScore, ScoreError> {
    let parents = Vec::new();
    let mut non_parents: Vec<_> = prefix
        .iter()
        .copied()
        .filter(|&u| constraints.is_none_or(|c| c.allowed_parent(u, v)))
        .collect();
    let mut v_local = score.local_score_init(v, parents)?;
    grow(v, &mut v_local, &mut non_parents, score, rng)?;
    shrink(v, &mut v_local, &mut non_parents, score, rng)?;
    Ok(v_local)
}

fn set_diff(tokens: &mut TokenBuffer, s1: &[usize], s2: &[usize]) -> Vec<usize> {
    tokens.clear();
    for &x in s2.iter() {
        tokens.set(x);
    }
    let res = s1.iter().copied().filter(|&x| !tokens.check(x)).collect();
    res
}

#[allow(dead_code)]
pub fn fit_parents_minus<R: Rng + ?Sized>(
    v: usize,
    v_local: &LocalScore,
    prefix: &[usize],
    r: usize,
    score: &Bic,
    tokens: &mut TokenBuffer,
    rng: &mut R,
) -> Result<LocalScore, ScoreError> {
    fit_parents_minus_constrained(v, v_local, prefix, r, score, tokens, rng, None)
}

#[allow(clippy::too_many_arguments)]
pub fn fit_parents_minus_constrained<R: Rng + ?Sized>(
    v: usize,
    v_local: &LocalScore,
    prefix: &[usize],
    r: usize,
    score: &Bic,
    tokens: &mut TokenBuffer,
    rng: &mut R,
    constraints: Option<&NoTrekConstraints>,
) -> Result<LocalScore, ScoreError> {
    if !v_local.parents.contains(&r) {
        return Ok(v_local.clone());
    }

    let mut v_local_new = score.local_score_minus(v, v_local, r)?;
    let admissible: Vec<_> = prefix
        .iter()
        .copied()
        .filter(|&u| constraints.is_none_or(|c| c.allowed_parent(u, v)))
        .collect();
    let mut non_parents = set_diff(tokens, &admissible, &v_local_new.parents);

    grow(v, &mut v_local_new, &mut non_parents, score, rng)?;
    shrink(v, &mut v_local_new, &mut non_parents, score, rng)?;

    Ok(v_local_new)
}

#[allow(dead_code)]
pub fn fit_parents_plus<R: Rng + ?Sized>(
    v: usize,
    v_local: &LocalScore,
    prefix: &[usize],
    r: usize,
    score: &Bic,
    tokens: &mut TokenBuffer,
    rng: &mut R,
) -> Result<LocalScore, ScoreError> {
    fit_parents_plus_constrained(v, v_local, prefix, r, score, tokens, rng, None)
}

#[allow(clippy::too_many_arguments)]
pub fn fit_parents_plus_constrained<R: Rng + ?Sized>(
    v: usize,
    v_local: &LocalScore,
    prefix: &[usize],
    r: usize,
    score: &Bic,
    tokens: &mut TokenBuffer,
    rng: &mut R,
    constraints: Option<&NoTrekConstraints>,
) -> Result<LocalScore, ScoreError> {
    if constraints.is_some_and(|c| !c.allowed_parent(r, v)) {
        return Ok(v_local.clone());
    }
    // check if adding r is an improvement
    let mut v_local_new = score.local_score_plus(v, v_local, r)?;

    if v_local_new.bic > v_local.bic {
        return Ok(v_local.clone());
    }
    let admissible: Vec<_> = prefix
        .iter()
        .copied()
        .filter(|&u| constraints.is_none_or(|c| c.allowed_parent(u, v)))
        .collect();
    let mut non_parents = set_diff(tokens, &admissible, &v_local_new.parents);

    grow(v, &mut v_local_new, &mut non_parents, score, rng)?;
    shrink(v, &mut v_local_new, &mut non_parents, score, rng)?;

    Ok(v_local_new)
}

// fit permutation from scratch
pub fn perm_to_dag<R: Rng + ?Sized>(
    perm: &[usize],
    score: &Bic,
    rng: &mut R,
) -> Result<GlobalScore, ScoreError> {
    perm_to_dag_constrained(perm, score, rng, None)
}

pub fn perm_to_dag_constrained<R: Rng + ?Sized>(
    perm: &[usize],
    score: &Bic,
    rng: &mut R,
    constraints: Option<&NoTrekConstraints>,
) -> Result<GlobalScore, ScoreError> {
    let mut g = GlobalScore::new(perm.len(), score)?;
    for (i, &v) in perm.iter().enumerate() {
        g.local_scores[v] = fit_parents_constrained(v, &perm[0..i], score, rng, constraints)?;
    }
    Ok(g)
}

/// Fit the ordinary FLOP local models for an order while forcing its first
/// `source_prefix` vertices to have empty parent sets.
///
/// This is deliberately not a NOTREKS-constrained parent search: after the
/// prefix, parent fitting is byte-for-byte the ordinary FLOP grow/shrink
/// routine.  The only intervention is the theorem-motivated source prefix.
pub fn perm_to_dag_source_prefix<R: Rng + ?Sized>(
    perm: &[usize],
    source_prefix: usize,
    score: &Bic,
    rng: &mut R,
) -> Result<GlobalScore, ScoreError> {
    let mut g = GlobalScore::new(perm.len(), score)?;
    for (i, &v) in perm.iter().enumerate() {
        g.local_scores[v] = if i < source_prefix {
            score.local_score_init(v, Vec::new())?
        } else {
            fit_parents(v, &perm[0..i], score, rng)?
        };
    }
    Ok(g)
}

pub fn perm_to_dag_constrained_source_prefix<R: Rng + ?Sized>(
    perm: &[usize],
    source_prefix: usize,
    score: &Bic,
    rng: &mut R,
    constraints: Option<&NoTrekConstraints>,
) -> Result<GlobalScore, ScoreError> {
    let mut g = GlobalScore::new(perm.len(), score)?;
    for (i, &v) in perm.iter().enumerate() {
        g.local_scores[v] = if i < source_prefix {
            score.local_score_init(v, Vec::new())?
        } else {
            fit_parents_constrained(v, &perm[0..i], score, rng, constraints)?
        };
    }
    Ok(g)
}
