//! Deterministic node-wise FLOP BIC pruning on a fixed candidate DAG.
//!
//! This deliberately performs no order search.  Each node's grow-shrink search
//! is restricted to the incoming neighbours supplied by the candidate DAG.

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::bic::Bic;
use crate::error::ScoreError;
use crate::fit_parents::fit_parents;

#[derive(Clone, Debug)]
pub struct PruneResult {
    pub parents: Vec<Vec<usize>>,
    pub local_bics: Vec<f64>,
}

impl PruneResult {
    pub fn bic(&self) -> f64 {
        self.local_bics.iter().sum()
    }

    pub fn edge_count(&self) -> usize {
        self.parents.iter().map(Vec::len).sum()
    }
}

fn candidate_is_dag(candidate: &DMatrix<f64>) -> bool {
    let p = candidate.nrows();
    let mut indegree = vec![0usize; p];
    let mut children = vec![Vec::new(); p];
    for parent in 0..p {
        for child in 0..p {
            if candidate[(parent, child)] != 0.0 {
                if parent == child {
                    return false;
                }
                indegree[child] += 1;
                children[parent].push(child);
            }
        }
    }
    let mut stack: Vec<_> = (0..p).filter(|&node| indegree[node] == 0).collect();
    let mut seen = 0;
    while let Some(node) = stack.pop() {
        seen += 1;
        for &child in &children[node] {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                stack.push(child);
            }
        }
    }
    seen == p
}

pub fn prune_parents_bic(
    data: &DMatrix<f64>,
    candidate: &DMatrix<f64>,
    lambda_bic: f64,
) -> Result<PruneResult, ScoreError> {
    let p = data.ncols();
    assert_eq!(candidate.nrows(), p);
    assert_eq!(candidate.ncols(), p);
    assert!(candidate_is_dag(candidate), "candidate graph must be a DAG");

    let score = Bic::new(data, lambda_bic);
    let mut parents = Vec::with_capacity(p);
    let mut local_bics = Vec::with_capacity(p);
    for child in 0..p {
        let candidates: Vec<_> = (0..p)
            .filter(|&parent| candidate[(parent, child)] != 0.0)
            .collect();
        // A node-specific seed makes the result independent of traversal or
        // thread scheduling while retaining FLOP's genuine grow-shrink code.
        let mut rng = StdRng::seed_from_u64(0x454e_4446_4c4f_5000 ^ child as u64);
        let local = fit_parents(child, &candidates, &score, &mut rng)?;
        debug_assert!(local
            .parents
            .iter()
            .all(|&parent| { candidate[(parent, child)] != 0.0 }));
        parents.push(local.parents);
        local_bics.push(local.bic);
    }
    Ok(PruneResult {
        parents,
        local_bics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pruning_is_deterministic_and_never_adds_edges() {
        let data = DMatrix::from_fn(100, 3, |r, c| {
            let x = (r as f64 - 50.0) / 25.0;
            match c {
                0 => x,
                1 => 2.0 * x + (r % 7) as f64 * 0.01,
                _ => (r % 11) as f64,
            }
        });
        let candidate = DMatrix::from_row_slice(3, 3, &[0., 1., 1., 0., 0., 1., 0., 0., 0.]);
        let first = prune_parents_bic(&data, &candidate, 2.0).unwrap();
        let second = prune_parents_bic(&data, &candidate, 2.0).unwrap();
        assert_eq!(first.parents, second.parents);
        for (child, parents) in first.parents.iter().enumerate() {
            assert!(parents
                .iter()
                .all(|&parent| candidate[(parent, child)] != 0.0));
        }
    }

    #[test]
    #[should_panic(expected = "candidate graph must be a DAG")]
    fn cyclic_candidate_is_rejected() {
        let data = DMatrix::zeros(10, 2);
        let candidate = DMatrix::from_row_slice(2, 2, &[0., 1., 1., 0.]);
        let _ = prune_parents_bic(&data, &candidate, 2.0);
    }
}
