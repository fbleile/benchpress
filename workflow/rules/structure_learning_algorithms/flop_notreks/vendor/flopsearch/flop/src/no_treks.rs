use std::collections::{BTreeSet, VecDeque};

use rand::Rng;

use crate::graph::Dag;
use crate::scores::GlobalScore;

#[derive(Clone, Debug)]
pub struct NoTrekConstraints {
    pub p: usize,
    pub pairs: Vec<(usize, usize)>,
    pub target_bit: Vec<Option<u8>>,
    pub incompatibility: Vec<u64>,
    pub signatures: Vec<u64>,
    pub forbidden_edges: Vec<(usize, usize)>,
    pub directed_forbidden_edges: Vec<(usize, usize)>,
}

impl NoTrekConstraints {
    pub fn new(p: usize, pairs: &[(usize, usize)]) -> Result<Self, String> {
        Self::new_with_forbidden(p, pairs, &[])
    }

    pub fn new_with_forbidden(
        p: usize,
        pairs: &[(usize, usize)],
        forbidden_edges: &[(usize, usize)],
    ) -> Result<Self, String> {
        let mut canonical = BTreeSet::new();
        for &(a, b) in pairs {
            if a >= p || b >= p {
                return Err(format!("no-trek pair ({a}, {b}) is outside 0..{p}"));
            }
            if a == b {
                return Err(format!("self no-trek pair ({a}, {b}) is invalid"));
            }
            canonical.insert(if a < b { (a, b) } else { (b, a) });
        }
        let pairs: Vec<_> = canonical.into_iter().collect();
        let targets: Vec<_> = pairs
            .iter()
            .flat_map(|&(a, b)| [a, b])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if targets.len() > 64 {
            return Err(format!(
                "{} constrained targets exceed the u64 prototype limit of 64",
                targets.len()
            ));
        }
        let mut target_bit = vec![None; p];
        for (bit, &node) in targets.iter().enumerate() {
            target_bit[node] = Some(bit as u8);
        }
        let mut incompatibility = vec![0u64; targets.len()];
        for &(a, b) in &pairs {
            let ba = target_bit[a].unwrap() as usize;
            let bb = target_bit[b].unwrap() as usize;
            incompatibility[ba] |= 1u64 << bb;
            incompatibility[bb] |= 1u64 << ba;
        }
        let mut signatures = vec![0u64; p];
        for node in 0..p {
            if let Some(bit) = target_bit[node] {
                signatures[node] = 1u64 << bit;
            }
        }
        let forbidden_edges = forbidden_edges
            .iter()
            .map(|&(u, v)| {
                if u >= p || v >= p || u == v {
                    Err(format!("forbidden edge ({u}, {v}) is invalid for 0..{p}"))
                } else {
                    Ok(if u < v { (u, v) } else { (v, u) })
                }
            })
            .collect::<Result<BTreeSet<_>, _>>()?
            .into_iter()
            .collect();
        Ok(Self {
            p,
            pairs,
            target_bit,
            incompatibility,
            signatures,
            forbidden_edges,
            directed_forbidden_edges: Vec::new(),
        })
    }

    pub fn new_with_directed_forbidden(
        p: usize,
        pairs: &[(usize, usize)],
        forbidden_edges: &[(usize, usize)],
    ) -> Result<Self, String> {
        let mut result = Self::new(p, pairs)?;
        let mut directed = BTreeSet::new();
        for &(u, v) in forbidden_edges {
            if u >= p || v >= p || u == v {
                return Err(format!("forbidden directed edge ({u}, {v}) is invalid for 0..{p}"));
            }
            directed.insert((u, v));
        }
        result.directed_forbidden_edges = directed.into_iter().collect();
        Ok(result)
    }

    #[inline(always)]
    pub fn allowed_parent(&self, parent: usize, child: usize) -> bool {
        self.signatures[child] & !self.signatures[parent] == 0
            && !self
                .forbidden_edges
                .iter()
                .any(|&(u, v)| (u == parent && v == child) || (u == child && v == parent))
            && !self
                .directed_forbidden_edges
                .iter()
                .any(|&(u, v)| u == parent && v == child)
    }

    pub fn signature_is_valid(&self, signature: u64) -> bool {
        let mut bits = signature;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            if self.incompatibility[bit] & signature != 0 {
                return false;
            }
        }
        true
    }

    pub fn signatures_are_valid(&self) -> bool {
        self.signatures
            .iter()
            .enumerate()
            .all(|(node, &signature)| {
                self.signature_is_valid(signature)
                    && self.target_bit[node].is_none_or(|bit| signature & (1u64 << bit) != 0)
            })
    }

    pub fn promotion_is_valid(&self, nodes: &[usize], payload: u64) -> bool {
        nodes
            .iter()
            .all(|&node| node < self.p && self.signature_is_valid(self.signatures[node] | payload))
    }

    #[inline(always)]
    pub fn missing_payload(&self, parent: usize, child: usize) -> u64 {
        self.signatures[child] & !self.signatures[parent]
    }

    pub fn randomized<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        mean_size: f64,
        max_size: usize,
    ) -> Self {
        let mut result = self.clone();
        let probability = if self.incompatibility.is_empty() {
            0.0
        } else {
            (mean_size / self.incompatibility.len() as f64).clamp(0.0, 1.0)
        };
        for node in 0..self.p {
            let required = self.target_bit[node].map_or(0, |bit| 1u64 << bit);
            let mut signature = required;
            let mut bits: Vec<_> = (0..self.incompatibility.len()).collect();
            for i in (1..bits.len()).rev() {
                bits.swap(i, rng.gen_range(0..=i));
            }
            for bit in bits {
                if signature.count_ones() as usize >= max_size {
                    break;
                }
                if rng.gen_bool(probability) && self.signature_is_valid(signature | (1u64 << bit)) {
                    signature |= 1u64 << bit;
                }
            }
            result.signatures[node] = signature;
        }
        result
    }

    pub fn candidate_relations_pruned(&self) -> usize {
        (0..self.p)
            .flat_map(|u| (0..self.p).map(move |v| (u, v)))
            .filter(|&(u, v)| u != v && !self.allowed_parent(u, v))
            .count()
    }
}

pub fn ancestor_cone(g: &GlobalScore, u: usize) -> Vec<usize> {
    let mut seen = vec![false; g.p];
    let mut queue = VecDeque::from([u]);
    seen[u] = true;
    while let Some(v) = queue.pop_front() {
        for &parent in &g.local_scores[v].parents {
            if !seen[parent] {
                seen[parent] = true;
                queue.push_back(parent);
            }
        }
    }
    seen.iter()
        .enumerate()
        .filter_map(|(v, &yes)| yes.then_some(v))
        .collect()
}

pub fn signatures_respect_edges(g: &GlobalScore, signatures: &[u64]) -> bool {
    g.local_scores.iter().enumerate().all(|(child, local)| {
        local
            .parents
            .iter()
            .all(|&parent| signatures[child] & !signatures[parent] == 0)
    })
}

pub fn canonical_signatures(
    g: &GlobalScore,
    perm: &[usize],
    constraints: &NoTrekConstraints,
) -> Vec<u64> {
    let mut signatures = vec![0u64; g.p];
    let mut children = vec![Vec::new(); g.p];
    for (child, local) in g.local_scores.iter().enumerate() {
        for &parent in &local.parents {
            children[parent].push(child);
        }
    }
    for &node in perm.iter().rev() {
        let own = constraints.target_bit[node].map_or(0, |bit| 1u64 << bit);
        signatures[node] = children[node]
            .iter()
            .fold(own, |acc, &child| acc | signatures[child]);
    }
    signatures
}

pub fn has_common_ancestor(g: &Dag, i: usize, j: usize) -> bool {
    fn ancestors(g: &Dag, node: usize) -> Vec<bool> {
        let mut seen = vec![false; g.p];
        let mut queue = VecDeque::from([node]);
        seen[node] = true;
        while let Some(v) = queue.pop_front() {
            for &parent in &g.parents[v] {
                if !seen[parent] {
                    seen[parent] = true;
                    queue.push_back(parent);
                }
            }
        }
        seen
    }
    let ai = ancestors(g, i);
    let aj = ancestors(g, j);
    ai.iter().zip(aj).any(|(&a, b)| a && b)
}

pub fn count_no_trek_violations(g: &Dag, pairs: &[(usize, usize)]) -> usize {
    pairs
        .iter()
        .filter(|&&(i, j)| has_common_ancestor(g, i, j))
        .count()
}

/// Validate an explicit undirected candidate trek graph.  Pairs are accepted
/// in either orientation but duplicate/self edges are rejected explicitly.
pub fn validate_candidate_trek_graph(
    p: usize,
    trek_edges: &[(usize, usize)],
    hard_notreks: &[(usize, usize)],
) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for &(a, b) in trek_edges {
        if a >= p || b >= p || a == b {
            return Err(format!("candidate trek edge ({a}, {b}) is invalid"));
        }
        let edge = if a < b { (a, b) } else { (b, a) };
        if !seen.insert(edge) {
            return Err(format!("candidate trek graph contains duplicate edge {edge:?}"));
        }
    }
    for &(a, b) in hard_notreks {
        let edge = if a < b { (a, b) } else { (b, a) };
        if seen.contains(&edge) {
            return Err(format!("candidate trek graph contains hard NOTREKS pair {edge:?}"));
        }
    }
    Ok(())
}

/// Static Trek-Dominance mask from an explicit undirected trek graph.
/// Rows are parents and columns are children.
pub fn trek_dominance_mask(
    p: usize,
    trek_edges: &[(usize, usize)],
) -> Result<Vec<Vec<bool>>, String> {
    if p > 64 {
        return Err("trek-dominance bitsets support at most 64 nodes".into());
    }
    let mut closed = vec![0u64; p];
    for v in 0..p { closed[v] = 1u64 << v; }
    for &(a, b) in trek_edges {
        if a >= p || b >= p || a == b {
            return Err(format!("invalid trek edge ({a}, {b})"));
        }
        closed[a] |= 1u64 << b;
        closed[b] |= 1u64 << a;
    }
    let mask = (0..p).map(|u| {
        (0..p).map(|v| u != v && (closed[u] & !closed[v]) == 0).collect()
    }).collect();
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bic::Bic;
    use crate::scores::GlobalScore;
    use nalgebra::DMatrix;

    #[test]
    fn trek_dominance_hand_checked_masks_and_prefix_candidates() {
        let t = vec![(0, 2), (1, 2)];
        let mask = trek_dominance_mask(3, &t).unwrap();
        assert_eq!(mask, vec![vec![false, false, true],
                              vec![false, false, true],
                              vec![false, false, false]]);
        let mut closed = vec![1u64, 2u64, 4u64];
        for &(a, b) in &t { closed[a] |= 1 << b; closed[b] |= 1 << a; }
        println!("T adjacency: {:?}", t);
        println!("closed-neighborhood bitsets: {:?}", closed);
        println!("allowed[parent,child]: {:?}", mask);
        assert_eq!(closed, vec![0b101, 0b110, 0b111]);
        assert_eq!(vec![Vec::<usize>::new(), Vec::new(), vec![0, 1]],
                   vec![Vec::<usize>::new(), Vec::new(),
                        (0..3).filter(|&u| mask[u][2]).collect()]);
    }

    #[test]
    fn trek_dominance_isolated_and_fork_masks() {
        let isolated = trek_dominance_mask(3, &[(0, 1)]).unwrap();
        assert!(isolated[0][1] && isolated[1][0]);
        assert!(!(isolated[0][2] || isolated[2][0] || isolated[1][2] || isolated[2][1]));
        let fork = trek_dominance_mask(3, &[(0, 1), (0, 2), (1, 2)]).unwrap();
        assert!((0..3).all(|u| (0..3).all(|v| u == v || fork[u][v])));
    }

    #[test]
    fn trek_dominance_extended_collider_retains_true_edges() {
        let mask = trek_dominance_mask(4, &[(0, 2), (1, 2), (2, 3), (0, 3), (1, 3)]).unwrap();
        assert!(mask[0][2] && mask[1][2] && mask[2][3]);
    }

    #[test]
    fn normalization_and_signature_rules() {
        let c = NoTrekConstraints::new(3, &[(1, 0), (0, 1), (1, 2)]).unwrap();
        assert_eq!(c.pairs, vec![(0, 1), (1, 2)]);
        let b0 = 1 << c.target_bit[0].unwrap();
        let b1 = 1 << c.target_bit[1].unwrap();
        let b2 = 1 << c.target_bit[2].unwrap();
        assert!(c.signature_is_valid(b0 | b2));
        assert!(!c.signature_is_valid(b0 | b1));
        assert!(!c.signature_is_valid(b1 | b2));
    }

    #[test]
    fn paths_forks_and_colliders() {
        let chain = Dag::from_edge_list(3, vec![(0, 2), (2, 1)]);
        assert!(has_common_ancestor(&chain, 0, 1));
        let fork = Dag::from_edge_list(3, vec![(2, 0), (2, 1)]);
        assert!(has_common_ancestor(&fork, 0, 1));
        let collider = Dag::from_edge_list(3, vec![(0, 2), (1, 2)]);
        assert!(!has_common_ancestor(&collider, 0, 1));
    }

    #[test]
    fn invalid_inputs() {
        assert!(NoTrekConstraints::new(2, &[(0, 0)]).is_err());
        assert!(NoTrekConstraints::new(2, &[(0, 2)]).is_err());
        let pairs: Vec<_> = (1..65).map(|x| (0, x)).collect();
        assert!(NoTrekConstraints::new(65, &pairs).is_err());
    }

    fn score_graph(p: usize, edges: &[(usize, usize)]) -> GlobalScore {
        let data = DMatrix::from_fn(20, p, |r, c| ((r + 3 * c) as f64).sin());
        let score = Bic::new(&data, 2.0);
        let mut graph = GlobalScore::new(p, &score).unwrap();
        for &(parent, child) in edges {
            graph.local_scores[child].parents.push(parent);
        }
        graph
    }

    #[test]
    fn ancestry_cone_promotes_complete_chain() {
        let graph = score_graph(5, &[(0, 1), (1, 2)]);
        assert_eq!(ancestor_cone(&graph, 2), vec![0, 1, 2]);
        let constraints = NoTrekConstraints::new(5, &[(3, 4)]).unwrap();
        let payload = constraints.missing_payload(2, 3);
        let cone = ancestor_cone(&graph, 2);
        assert!(constraints.promotion_is_valid(&cone, payload));
        let mut promoted = constraints.clone();
        for node in cone {
            promoted.signatures[node] |= payload;
        }
        assert!(promoted.allowed_parent(2, 3));
        assert_eq!(promoted.signatures[0], promoted.signatures[1]);
        assert_eq!(promoted.signatures[1], promoted.signatures[2]);
    }

    #[test]
    fn infeasible_promotion_is_rejected_before_refit() {
        let constraints = NoTrekConstraints::new(3, &[(0, 1)]).unwrap();
        let bit_one = 1u64 << constraints.target_bit[1].unwrap();
        assert!(!constraints.promotion_is_valid(&[0], bit_one));
    }

    #[test]
    fn canonical_compression_preserves_edges_and_removes_excess_bits() {
        let graph = score_graph(4, &[(0, 2), (1, 2)]);
        let perm = vec![0, 1, 3, 2];
        let mut constraints = NoTrekConstraints::new(4, &[(0, 1)]).unwrap();
        let bit_zero = 1u64 << constraints.target_bit[0].unwrap();
        constraints.signatures[3] = bit_zero; // unnecessary certificate bit
        let canonical = canonical_signatures(&graph, &perm, &constraints);
        assert_eq!(canonical[3], 0);
        assert!(signatures_respect_edges(&graph, &canonical));
        assert!(canonical
            .iter()
            .zip(&constraints.signatures)
            .all(|(&new, &old)| new & !old == 0));
    }
}
