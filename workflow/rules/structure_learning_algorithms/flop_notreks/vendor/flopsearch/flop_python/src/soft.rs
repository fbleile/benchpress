use nalgebra::{DMatrix, DVector};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

#[derive(Clone)]
struct Eval {
    total: f64,
    bic: f64,
    coef: DMatrix<f64>,
}

fn local_bic(
    data: &DMatrix<f64>,
    child: usize,
    parents: &[usize],
    lambda: f64,
) -> (f64, DVector<f64>) {
    let n = data.nrows();
    let p = parents.len();
    let means: Vec<f64> = (0..data.ncols())
        .map(|j| data.column(j).sum() / n as f64)
        .collect();
    let y = DVector::from_iterator(n, (0..n).map(|i| data[(i, child)] - means[child]));
    if p == 0 {
        let var = (y.dot(&y) / n as f64).max(1e-8);
        return (n as f64 * var.ln(), DVector::zeros(0));
    }
    let design = DMatrix::from_fn(n, p, |i, j| data[(i, parents[j])] - means[parents[j]]);
    let beta = design
        .clone()
        .svd(true, true)
        .solve(&y, 1e-12)
        .unwrap_or_else(|_| DVector::zeros(p));
    let residual = &y - design * &beta;
    let var = (residual.dot(&residual) / n as f64).max(1e-8);
    (
        n as f64 * var.ln() + lambda * p as f64 * (n as f64).ln(),
        beta,
    )
}

fn evaluate(
    data: &DMatrix<f64>,
    graph: &[u8],
    pairs: &[(usize, usize)],
    lambda: f64,
    weight: f64,
) -> Eval {
    let d = data.ncols();
    let mut coef = DMatrix::zeros(d, d);
    let mut bic = 0.0;
    for child in 0..d {
        let parents: Vec<usize> = (0..d).filter(|&u| graph[u * d + child] != 0).collect();
        let (score, beta) = local_bic(data, child, &parents, lambda);
        bic += score;
        for (k, &parent) in parents.iter().enumerate() {
            coef[(parent, child)] = beta[k];
        }
    }
    if pairs.is_empty() {
        return Eval {
            total: bic,
            bic,
            coef,
        };
    }
    let a = coef.map(|x| x * x);
    let mut m = DMatrix::<f64>::identity(d, d);
    m -= &a;
    m += DMatrix::<f64>::identity(d, d) * 1e-8;
    let f = match m.try_inverse() {
        Some(x) => x,
        None => {
            return Eval {
                total: f64::INFINITY,
                bic,
                coef,
            }
        }
    };
    let ftf = f.transpose() * &f;
    let scale = 2.0 / (d.saturating_sub(1).max(1) as f64);
    let mut nt = 0.0;
    for &(i, j) in pairs {
        nt += ftf[(i, j)] + ftf[(j, i)];
    }
    nt *= 0.5 * scale;
    Eval {
        total: bic + weight * nt,
        bic,
        coef,
    }
}

fn reach(graph: &[u8], d: usize) -> Vec<bool> {
    let mut r = vec![false; d * d];
    for i in 0..d {
        r[i * d + i] = true;
    }
    for i in 0..d {
        for j in 0..d {
            r[i * d + j] |= graph[i * d + j] != 0;
        }
    }
    for k in 0..d {
        for i in 0..d {
            if r[i * d + k] {
                for j in 0..d {
                    r[i * d + j] |= r[k * d + j];
                }
            }
        }
    }
    r
}

fn violations(graph: &[u8], d: usize, pairs: &[(usize, usize)]) -> usize {
    let r = reach(graph, d);
    pairs
        .iter()
        .filter(|&&(i, j)| (0..d).any(|a| r[a * d + i] && r[a * d + j]))
        .count()
}

fn fixed_order_start(data: &DMatrix<f64>, order: &[usize], lambda: f64) -> Vec<u8> {
    let d = data.ncols();
    let mut graph = vec![0u8; d * d];
    loop {
        let mut best: Option<(f64, usize, usize)> = None;
        for pi in 0..d {
            for pj in (pi + 1)..d {
                let u = order[pi];
                let v = order[pj];
                if graph[u * d + v] != 0 {
                    continue;
                }
                let old: Vec<usize> = (0..d).filter(|&x| graph[x * d + v] != 0).collect();
                let old_score = local_bic(data, v, &old, lambda).0;
                let mut new = old.clone();
                new.push(u);
                let delta = local_bic(data, v, &new, lambda).0 - old_score;
                if delta < -1e-10 && best.map_or(true, |b| (delta, u, v) < b) {
                    best = Some((delta, u, v));
                }
            }
        }
        match best {
            Some((_, u, v)) => graph[u * d + v] = 1,
            None => break,
        }
    }
    graph
}

fn project_to_order(graph: &[u8], order: &[usize]) -> Vec<u8> {
    let d = order.len();
    let mut position = vec![0usize; d];
    for (index, &node) in order.iter().enumerate() {
        position[node] = index;
    }
    let mut result = graph.to_vec();
    for u in 0..d {
        for v in 0..d {
            if u == v || position[u] >= position[v] {
                result[u * d + v] = 0;
            }
        }
    }
    result
}

fn reinsert(order: &[usize], node: usize, position: usize) -> Vec<usize> {
    let mut result: Vec<usize> = order.iter().copied().filter(|&x| x != node).collect();
    result.insert(position.min(result.len()), node);
    result
}

fn evaluate_normalized(
    data: &DMatrix<f64>,
    graph: &[u8],
    pairs: &[(usize, usize)],
    lambda: f64,
    weight: f64,
) -> Eval {
    // Gaussian BIC is O(n), so put the continuous NOTREKS value on the same
    // sample-size scale before applying the dimensionless continuation weight.
    evaluate(data, graph, pairs, lambda, weight * data.nrows() as f64)
}

pub fn fit_soft_redesigned<'py>(
    py: pyo3::Python<'py>,
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    restarts: usize,
    sweeps: usize,
    lambda: f64,
    top_k: usize,
    seed: u64,
) -> pyo3::PyResult<(pyo3::Bound<'py, numpy::PyArray2<f64>>, usize)> {
    let d = data.ncols();
    let mut rng = StdRng::seed_from_u64(seed);
    let schedule = [0.01_f64, 0.1, 1.0, 10.0];
    let mut best_graph: Option<Vec<u8>> = None;
    let mut best_bic = f64::INFINITY;
    let mut evaluations = 0usize;
    for _ in 0..restarts {
        let mut order: Vec<usize> = (0..d).collect();
        order.shuffle(&mut rng);
        let mut graph = fixed_order_start(data, &order, lambda);
        for &weight in &schedule {
            let mut current = evaluate_normalized(data, &graph, pairs, lambda, weight);
            evaluations += 1;
            for _ in 0..sweeps {
                let mut candidates: Vec<(f64, Vec<u8>)> = Vec::new();
                let mut position = vec![0usize; d];
                for (index, &node) in order.iter().enumerate() {
                    position[node] = index;
                }
                let mut moves: Vec<(f64, usize, usize)> = Vec::new();
                for u in 0..d {
                    for v in 0..d {
                        if u != v && position[u] < position[v] {
                            let parents: Vec<usize> =
                                (0..d).filter(|&x| graph[x * d + v] != 0).collect();
                            let old = local_bic(data, v, &parents, lambda).0;
                            let mut next = parents.clone();
                            if graph[u * d + v] != 0 {
                                next.retain(|&x| x != u);
                            } else {
                                next.push(u);
                            }
                            let delta = local_bic(data, v, &next, lambda).0 - old;
                            moves.push((delta, u, v));
                        }
                    }
                }
                moves.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for &(_, u, v) in moves.iter().take(top_k) {
                    let mut proposal = graph.clone();
                    proposal[u * d + v] ^= 1;
                    let value = evaluate_normalized(data, &proposal, pairs, lambda, weight);
                    evaluations += 1;
                    candidates.push((value.total, proposal));
                }
                // FLOP-style outer order reinsertion: evaluate every single
                // node move and project the incumbent to the proposed order.
                for node in 0..d {
                    for position in 0..d {
                        let proposed_order = reinsert(&order, node, position);
                        let proposal = project_to_order(&graph, &proposed_order);
                        let value = evaluate_normalized(data, &proposal, pairs, lambda, weight);
                        evaluations += 1;
                        candidates.push((value.total, proposal));
                    }
                }
                let Some((value, proposal)) = candidates.into_iter().min_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(&b.1))
                }) else {
                    break;
                };
                if value >= current.total - 1e-10 {
                    break;
                }
                graph = proposal;
                current = evaluate_normalized(data, &graph, pairs, lambda, weight);
                evaluations += 1;
            }
        }
        let mut repaired = graph.clone();
        while violations(&repaired, d, pairs) > 0 {
            let mut best: Option<(usize, f64, usize, usize)> = None;
            let current_eval = evaluate(data, &repaired, pairs, lambda, 0.0);
            for u in 0..d {
                for v in 0..d {
                    if repaired[u * d + v] != 0 {
                        let mut candidate = repaired.clone();
                        candidate[u * d + v] = 0;
                        let reduction = violations(&repaired, d, pairs)
                            .saturating_sub(violations(&candidate, d, pairs));
                        let strength = current_eval.coef[(u, v)].abs();
                        let key = (reduction, strength, u, v);
                        if best.map_or(true, |b| key.0 > b.0 || (key.0 == b.0 && key.1 < b.1)) {
                            best = Some(key);
                        }
                    }
                }
            }
            if let Some((_, _, u, v)) = best {
                repaired[u * d + v] = 0;
            } else {
                break;
            }
        }
        let final_eval = evaluate(data, &repaired, pairs, lambda, 0.0);
        evaluations += 1;
        if final_eval.bic < best_bic {
            best_bic = final_eval.bic;
            best_graph = Some(repaired);
        }
    }
    let graph = best_graph.unwrap_or_else(|| vec![0u8; d * d]);
    Ok((graph_matrix(py, &graph, d)?, evaluations))
}

fn graph_matrix<'py>(
    py: pyo3::Python<'py>,
    graph: &[u8],
    d: usize,
) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray2<f64>>> {
    let rows: Vec<Vec<f64>> = (0..d)
        .map(|i| (0..d).map(|j| graph[i * d + j] as f64).collect())
        .collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

pub fn fit_soft<'py>(
    py: pyo3::Python<'py>,
    data: &DMatrix<f64>,
    pairs: &[(usize, usize)],
    restarts: usize,
    sweeps: usize,
    lambda: f64,
    weight: f64,
    top_k: usize,
    seed: u64,
) -> pyo3::PyResult<(pyo3::Bound<'py, numpy::PyArray2<f64>>, usize)> {
    let d = data.ncols();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best_graph = vec![0u8; d * d];
    let mut best_eval = evaluate(data, &best_graph, pairs, lambda, weight);
    let mut evaluations = 1usize;
    for _ in 0..restarts {
        let mut order: Vec<usize> = (0..d).collect();
        order.shuffle(&mut rng);
        let mut graph = fixed_order_start(data, &order, lambda);
        let mut current = evaluate(data, &graph, pairs, lambda, weight);
        evaluations += 1;
        for _ in 0..sweeps {
            let mut moves: Vec<(f64, usize, usize)> = Vec::new();
            let mut pos = vec![0usize; d];
            for (i, &u) in order.iter().enumerate() {
                pos[u] = i;
            }
            for u in 0..d {
                for v in 0..d {
                    if u != v && pos[u] < pos[v] {
                        let parents: Vec<usize> =
                            (0..d).filter(|&x| graph[x * d + v] != 0).collect();
                        let old = local_bic(data, v, &parents, lambda).0;
                        let mut next = parents.clone();
                        if graph[u * d + v] != 0 {
                            next.retain(|&x| x != u);
                        } else {
                            next.push(u);
                        }
                        let delta = local_bic(data, v, &next, lambda).0 - old;
                        moves.push((delta, u, v));
                    }
                }
            }
            moves.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut candidates = Vec::new();
            let add_limit = top_k.min(moves.len());
            let mut additions = 0;
            let mut deletions = 0;
            for &(_, u, v) in &moves {
                if graph[u * d + v] == 0 && additions < top_k {
                    candidates.push((u, v));
                    additions += 1;
                }
            }
            for &(_, u, v) in &moves {
                if graph[u * d + v] != 0 && deletions < top_k {
                    candidates.push((u, v));
                    deletions += 1;
                }
            }
            let _ = add_limit;
            let mut best_move: Option<(Eval, Vec<u8>)> = None;
            for (u, v) in candidates {
                let mut proposal = graph.clone();
                proposal[u * d + v] ^= 1;
                let value = evaluate(data, &proposal, pairs, lambda, weight);
                evaluations += 1;
                if best_move.as_ref().map_or(true, |x| {
                    value.total < x.0.total || (value.total == x.0.total && proposal < x.1)
                }) {
                    best_move = Some((value, proposal));
                }
            }
            match best_move {
                Some((value, proposal)) if value.total < current.total - 1e-10 => {
                    graph = proposal;
                    current = value;
                }
                _ => break,
            }
        }
        let mut repaired = graph.clone();
        while violations(&repaired, d, pairs) > 0 {
            let mut best: Option<(usize, f64, usize, usize)> = None;
            for u in 0..d {
                for v in 0..d {
                    if repaired[u * d + v] != 0 {
                        let mut candidate = repaired.clone();
                        candidate[u * d + v] = 0;
                        let reduction = violations(&repaired, d, pairs)
                            .saturating_sub(violations(&candidate, d, pairs));
                        let strength =
                            evaluate(data, &repaired, pairs, lambda, weight).coef[(u, v)].abs();
                        let key = (reduction, strength, u, v);
                        if best.map_or(true, |b| key.0 > b.0 || (key.0 == b.0 && key.1 < b.1)) {
                            best = Some(key);
                        }
                    }
                }
            }
            if let Some((_, _, u, v)) = best {
                repaired[u * d + v] = 0;
            } else {
                break;
            }
        }
        let feasible = evaluate(data, &repaired, pairs, lambda, weight);
        evaluations += 1;
        if feasible.bic < best_eval.bic || (feasible.bic == best_eval.bic && repaired < best_graph)
        {
            best_eval = feasible;
            best_graph = repaired;
        }
    }
    Ok((graph_matrix(py, &best_graph, d)?, evaluations))
}
