use ::flop::algo::FlopConfig;
use ::flop::constrained_algo::{
    run_notreks, run_notreks_source_prefix, FlopNoTreksConfig, NoTreksDiagnostics, NoTreksVersion,
};
use nalgebra::DMatrix;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict, PyTuple},
};

/// Run the FLOP causal discovery algorithm.
///
/// Parameters:
///     data: A data matrix with rows corresponding to observations and columns to variables/nodes.
///     lambda_bic: The penalty parameter of the BIC, a typical value for structure learning is 2.0.
///     restarts: Optional parameter specifying the number of ILS restarts. Either restarts or timeout (below) need to be specified.
///     timeout: Optional parameter specifying a timeout after which the search returns. At least one local search is run up to a local optimum. Either restarts or timeout need to be specified.
///
/// Returns:
///     A matrix encoding a CPDAG. The entry in row i and column j is 1 in case of a directed edge from i to j and 2 in case of an undirected edge between those nodes (the entry in row j and column i will also be a 2, that is each undirected edge induces two 2's in the matrix).
#[pyfunction]
#[pyo3(signature = (data, lambda_bic, *, restarts=None, timeout=None))]
fn flop<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    lambda_bic: f64,
    restarts: Option<usize>,
    timeout: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if restarts.is_none() && timeout.is_none() {
        return Err(PyValueError::new_err(
            "Config error: neither number of restarts nor timeout was specified, e.g., pass restarts=50 as optional argument",
        ));
    }
    let flop_config = FlopConfig::new(lambda_bic, restarts, timeout, false);
    let data_matrix = DMatrix::from(data.as_matrix());
    let g = match ::flop::algo::run(&data_matrix, flop_config) {
        Ok(res) => res,
        Err(err) => Err(PyRuntimeError::new_err(format!("FLOP error: {}", err)))?,
    };

    let mut res = vec![vec![0.0; g.p]; g.p];
    let g = g.to_cpdag();
    for (u, row) in res.iter_mut().enumerate() {
        for &v in g.undir_neighbors[u].iter() {
            row[v] = 2.0;
        }
        for &v in g.out_neighbors[u].iter() {
            row[v] = 1.0;
        }
    }

    PyArray2::from_vec2(py, &res).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Run FLOP with the first k positions of every causal order forced to be
/// parentless.  `source_prefix` is normally chi(H), computed by the Python
/// adapter from the supplied no-trek graph H.
#[pyfunction]
#[pyo3(signature = (data, lambda_bic, source_prefix, *, restarts=1, seed=1729, return_diagnostics=false))]
fn flop_source_prefix<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    lambda_bic: f64,
    source_prefix: usize,
    restarts: usize,
    seed: u64,
    return_diagnostics: bool,
) -> PyResult<Py<PyAny>> {
    let data_matrix = DMatrix::from(data.as_matrix());
    let (dag, diagnostics) =
        ::flop::algo::run_source_prefix(&data_matrix, lambda_bic, source_prefix, restarts, seed)
            .map_err(|err| PyRuntimeError::new_err(format!("source-prefix FLOP error: {err}")))?;
    let matrix = graph_matrix(py, &dag, false)?;
    if !return_diagnostics {
        return Ok(matrix.into_any().unbind());
    }
    let details = PyDict::new(py);
    details.set_item("source_prefix", diagnostics.source_prefix)?;
    details.set_item("restarts_completed", diagnostics.restarts_completed)?;
    details.set_item("full_order_refits", diagnostics.full_order_refits)?;
    details.set_item("accepted_reinsertions", diagnostics.accepted_reinsertions)?;
    details.set_item("total_sweeps", diagnostics.total_sweeps)?;
    details.set_item(
        "maximum_sweeps_per_restart",
        diagnostics.maximum_sweeps_per_restart,
    )?;
    details.set_item("final_bic", diagnostics.final_bic)?;
    details.set_item("selected_order", diagnostics.selected_order)?;
    let dag_edges: Vec<(usize, usize)> = dag
        .parents
        .iter()
        .enumerate()
        .flat_map(|(child, parents)| parents.iter().map(move |&parent| (parent, child)))
        .collect();
    details.set_item("selected_dag_edges", dag_edges)?;
    Ok(PyTuple::new(py, [matrix.into_any(), details.into_any()])?
        .into_any()
        .unbind())
}

/// Optimize one fixed order with the Rust implementation of the global-
/// greedy inner edge-toggle search.  The FLOP-style outer reinsertion loop is
/// intentionally left to the caller.
#[pyfunction]
#[pyo3(signature = (data, initial, order, no_trek_pairs, *, lambda_bic=2.0, return_dag=true))]
fn global_greedy_inner<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    initial: PyReadonlyArray2<u8>,
    order: Vec<usize>,
    no_trek_pairs: &Bound<'py, PyAny>,
    lambda_bic: f64,
    return_dag: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = data.shape()[1];
    if initial.shape() != [p, p] {
        return Err(PyValueError::new_err("initial adjacency must be d x d"));
    }
    let pairs = parse_pairs(no_trek_pairs, p)?;
    let initial_vec: Vec<u8> = initial.as_array().iter().copied().collect();
    let data_matrix = DMatrix::from(data.as_matrix());
    let (dag, _) = ::flop::constrained_algo::global_greedy_inner(
        &data_matrix,
        &initial_vec,
        &order,
        &pairs,
        lambda_bic,
    )
    .map_err(|err| PyRuntimeError::new_err(format!("global-greedy inner error: {err}")))?;
    graph_matrix(py, &dag, return_dag)
}

fn parse_pairs(obj: &Bound<'_, PyAny>, p: usize) -> PyResult<Vec<(usize, usize)>> {
    let raw: Vec<(i64, i64)> = if let Ok(pairs) = obj.extract() {
        pairs
    } else if let Ok(array) = obj.extract::<PyReadonlyArray2<i64>>() {
        let view = array.as_array();
        if view.ncols() != 2 {
            return Err(PyValueError::new_err(
                "no_trek_pairs array must have shape (m, 2)",
            ));
        }
        view.rows()
            .into_iter()
            .map(|row| (row[0], row[1]))
            .collect()
    } else {
        return Err(PyValueError::new_err(
            "no_trek_pairs must be a sequence of integer pairs or an integer array of shape (m, 2)",
        ));
    };
    let mut result = Vec::with_capacity(raw.len());
    for (a, b) in raw {
        if a < 0 || b < 0 || a as usize >= p || b as usize >= p {
            return Err(PyValueError::new_err(format!(
                "no-trek pair ({a}, {b}) is outside 0..{p}"
            )));
        }
        if a == b {
            return Err(PyValueError::new_err("self no-trek pairs are invalid"));
        }
        result.push((a as usize, b as usize));
    }
    Ok(result)
}

fn graph_matrix<'py>(
    py: Python<'py>,
    dag: &::flop::graph::Dag,
    return_dag: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut result = vec![vec![0.0; dag.p]; dag.p];
    if return_dag {
        for (child, parents) in dag.parents.iter().enumerate() {
            for &parent in parents {
                result[parent][child] = 1.0;
            }
        }
    } else {
        let cpdag = dag.to_cpdag();
        for (node, row) in result.iter_mut().enumerate() {
            for &other in &cpdag.undir_neighbors[node] {
                row[other] = 2.0;
            }
            for &child in &cpdag.out_neighbors[node] {
                row[child] = 1.0;
            }
        }
    }
    PyArray2::from_vec2(py, &result).map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
    let mut ready: Vec<_> = (0..p).filter(|&node| indegree[node] == 0).collect();
    let mut seen = 0;
    while let Some(node) = ready.pop() {
        seen += 1;
        for &child in &children[node] {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                ready.push(child);
            }
        }
    }
    seen == p
}

fn ols_coefficients(data: &DMatrix<f64>, parents: &[Vec<usize>]) -> Result<Vec<Vec<f64>>, String> {
    let n = data.nrows();
    let p = data.ncols();
    let means: Vec<_> = (0..p)
        .map(|column| data.column(column).sum() / n as f64)
        .collect();
    let mut coefficients = vec![vec![0.0; p]; p];
    for (child, selected) in parents.iter().enumerate() {
        if selected.is_empty() {
            continue;
        }
        let design = DMatrix::from_fn(n, selected.len(), |row, column| {
            data[(row, selected[column])] - means[selected[column]]
        });
        let response = nalgebra::DVector::from_fn(n, |row, _| data[(row, child)] - means[child]);
        let fitted = design
            .svd(true, true)
            .solve(&response, 1e-12)
            .map_err(|err| format!("OLS refit failed for node {child}: {err}"))?;
        for (index, &parent) in selected.iter().enumerate() {
            coefficients[parent][child] = fitted[index];
        }
    }
    Ok(coefficients)
}

/// Prune a fixed candidate DAG by FLOP's node-wise Gaussian-BIC grow-shrink search.
#[pyfunction]
#[pyo3(signature = (
    data, candidate_dag, lambda_bic=2.0, *,
    return_coefficients=false, return_diagnostics=false
))]
fn prune_parents_bic<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    candidate_dag: PyReadonlyArray2<f64>,
    lambda_bic: f64,
    return_coefficients: bool,
    return_diagnostics: bool,
) -> PyResult<Py<PyAny>> {
    let shape = candidate_dag.shape();
    let p = data.shape()[1];
    if shape != [p, p] {
        return Err(PyValueError::new_err(format!(
            "candidate_dag must have shape ({p}, {p})"
        )));
    }
    let candidate = DMatrix::from(candidate_dag.as_matrix());
    if !candidate.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "candidate_dag must contain only finite values",
        ));
    }
    if !candidate_is_dag(&candidate) {
        return Err(PyValueError::new_err(
            "candidate_dag must be a directed acyclic graph",
        ));
    }
    let data_matrix = DMatrix::from(data.as_matrix());
    let result = ::flop::prune::prune_parents_bic(&data_matrix, &candidate, lambda_bic)
        .map_err(|err| PyRuntimeError::new_err(format!("FLOP pruning error: {err}")))?;

    let mut adjacency = vec![vec![0.0; p]; p];
    for (child, parents) in result.parents.iter().enumerate() {
        for &parent in parents {
            adjacency[parent][child] = 1.0;
        }
    }
    let graph = PyArray2::from_vec2(py, &adjacency)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let coefficients = if return_coefficients {
        Some(
            PyArray2::from_vec2(
                py,
                &ols_coefficients(&data_matrix, &result.parents)
                    .map_err(PyRuntimeError::new_err)?,
            )
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?,
        )
    } else {
        None
    };
    let diagnostics = if return_diagnostics {
        let values = PyDict::new(py);
        values.set_item("bic", result.bic())?;
        values.set_item("edge_count", result.edge_count())?;
        values.set_item("local_bics", &result.local_bics)?;
        values.set_item(
            "candidate_edge_count",
            candidate.iter().filter(|&&x| x != 0.0).count(),
        )?;
        Some(values)
    } else {
        None
    };
    match (coefficients, diagnostics) {
        (Some(coef), Some(diag)) => Ok(PyTuple::new(
            py,
            [graph.as_any(), coef.as_any(), diag.as_any()],
        )?
        .into_any()
        .unbind()),
        (Some(coef), None) => Ok(PyTuple::new(py, [graph.as_any(), coef.as_any()])?
            .into_any()
            .unbind()),
        (None, Some(diag)) => Ok(PyTuple::new(py, [graph.as_any(), diag.as_any()])?
            .into_any()
            .unbind()),
        (None, None) => Ok(graph.into_any().unbind()),
    }
}

fn diagnostics_dict<'py>(
    py: Python<'py>,
    d: &NoTreksDiagnostics,
    dag: &::flop::graph::Dag,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("restarts_requested", d.restarts_requested)?;
    result.set_item("restarts_completed", d.restarts_completed)?;
    result.set_item("best_restart_index", d.best_restart_index)?;
    result.set_item("best_restart_was_minimal", d.best_restart_was_minimal)?;
    result.set_item("restart_bics", &d.restart_bics)?;
    result.set_item(
        "initial_fraction_of_candidate_parent_relations_pruned",
        d.initial_fraction_of_candidate_parent_relations_pruned,
    )?;
    result.set_item(
        "number_of_supplied_constraints",
        d.number_of_supplied_constraints,
    )?;
    result.set_item(
        "number_of_distinct_constrained_targets",
        d.number_of_distinct_constrained_targets,
    )?;
    result.set_item("number_of_signature_rounds", d.number_of_signature_rounds)?;
    result.set_item(
        "number_of_blocked_edge_proposals_generated",
        d.number_of_blocked_edge_proposals_generated,
    )?;
    result.set_item(
        "number_of_proposals_retained",
        d.number_of_proposals_retained,
    )?;
    result.set_item(
        "number_of_proposals_tested_with_complete_refitting",
        d.number_of_proposals_tested_with_complete_refitting,
    )?;
    result.set_item(
        "number_of_infeasible_promotions",
        d.number_of_infeasible_promotions,
    )?;
    result.set_item(
        "number_of_accepted_promotions",
        d.number_of_accepted_promotions,
    )?;
    result.set_item("mean_ancestor_cone_size", d.mean_ancestor_cone_size())?;
    result.set_item("maximum_ancestor_cone_size", d.maximum_ancestor_cone_size)?;
    result.set_item(
        "number_of_candidate_parent_relations_pruned",
        d.number_of_candidate_parent_relations_pruned,
    )?;
    result.set_item(
        "fraction_of_candidate_parent_relations_pruned",
        d.fraction_of_candidate_parent_relations_pruned,
    )?;
    result.set_item(
        "time_in_constrained_order_search",
        d.time_in_constrained_order_search,
    )?;
    result.set_item(
        "time_in_signature_move_evaluation",
        d.time_in_signature_move_evaluation,
    )?;
    result.set_item(
        "time_in_complete_constrained_refits",
        d.time_in_complete_constrained_refits,
    )?;
    result.set_item("final_bic", d.final_bic)?;
    result.set_item("selected_dag_edge_count", d.selected_dag_edge_count)?;
    result.set_item(
        "final_no_trek_violation_count",
        d.final_no_trek_violation_count,
    )?;
    result.set_item("termination_reason", &d.termination_reason)?;
    result.set_item("algorithm_seed", d.algorithm_seed)?;
    result.set_item("search_version", &d.search_version)?;
    result.set_item(
        "number_of_canonical_compressions",
        d.number_of_canonical_compressions,
    )?;
    result.set_item(
        "number_of_post_promotion_order_blocks",
        d.number_of_post_promotion_order_blocks,
    )?;
    result.set_item("repair_edges_removed", d.repair_edges_removed)?;
    result.set_item(
        "repair_continuous_evaluations",
        d.repair_continuous_evaluations,
    )?;
    result.set_item(
        "repair_initial_violation_count",
        d.repair_initial_violation_count,
    )?;
    result.set_item(
        "repair_initial_continuous_value",
        d.repair_initial_continuous_value,
    )?;
    result.set_item(
        "repair_final_continuous_value",
        d.repair_final_continuous_value,
    )?;
    result.set_item("repair_fallback_used", d.repair_fallback_used)?;
    let edges: Vec<_> = dag
        .parents
        .iter()
        .enumerate()
        .flat_map(|(child, parents)| parents.iter().map(move |&parent| (parent, child)))
        .collect();
    result.set_item("selected_dag_edges", edges)?;
    Ok(result)
}

/// Run FLOP with hard structural no-trek constraints.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    data, lambda_bic, no_trek_pairs, *,
    restarts=None, timeout=None, seed=None, signature_top_k=5,
    signature_exploration_k=0, max_signature_rounds=20,
    initial_signature_mean_size=3.0, initial_signature_max_size=6,
    search_version="global_greedy_rust", source_prefix=0,
    return_dag=false, return_diagnostics=false
))]
fn flop_notreks<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    lambda_bic: f64,
    no_trek_pairs: &Bound<'py, PyAny>,
    restarts: Option<usize>,
    timeout: Option<f64>,
    seed: Option<u64>,
    signature_top_k: usize,
    signature_exploration_k: usize,
    max_signature_rounds: usize,
    initial_signature_mean_size: f64,
    initial_signature_max_size: usize,
    search_version: &str,
    source_prefix: usize,
    return_dag: bool,
    return_diagnostics: bool,
) -> PyResult<Py<PyAny>> {
    if restarts.is_none() && timeout.is_none() {
        return Err(PyValueError::new_err(
            "Config error: specify restarts or timeout",
        ));
    }
    if restarts.is_some() && timeout.is_some() {
        return Err(PyValueError::new_err(
            "Config error: specify only one of restarts or timeout",
        ));
    }
    let p = data.shape()[1];
    let pairs = parse_pairs(no_trek_pairs, p)?;
    // The schema keeps global_greedy_rust as the sole production path.  The
    // fixed-signature path remains callable here as a research comparator for
    // experiments against the historical, closest-to-FLOP formulation.
    let search_version = match search_version {
        "global_greedy_rust" => NoTreksVersion::GlobalGreedyRust,
        "global_greedy_cached" => NoTreksVersion::GlobalGreedyCached,
        "global_greedy_parallel" => NoTreksVersion::GlobalGreedyParallel,
        "global_greedy_hybrid" => NoTreksVersion::GlobalGreedyHybrid,
        "fixed_signature_a" => NoTreksVersion::FixedSignatureA,
        "incremental_promotion_d" => NoTreksVersion::IncrementalPromotionD,
        other => return Err(PyValueError::new_err(format!(
            "unsupported search_version {other:?}; expected 'global_greedy_rust', 'global_greedy_cached', 'global_greedy_parallel', 'global_greedy_hybrid', 'fixed_signature_a', or 'incremental_promotion_d'"
        ))),
    };
    let config = FlopNoTreksConfig {
        lambda: lambda_bic,
        restarts,
        timeout,
        manual_termination: false,
        seed,
        signature_top_k,
        signature_exploration_k,
        max_signature_rounds,
        initial_signature_mean_size,
        initial_signature_max_size,
        search_version,
    };
    let data_matrix = DMatrix::from(data.as_matrix());
    let result = if source_prefix > 0 {
        run_notreks_source_prefix(&data_matrix, &pairs, config, source_prefix)
    } else {
        run_notreks(&data_matrix, &pairs, config)
    }
    .map_err(|err| PyRuntimeError::new_err(format!("FLOP-NOTREKS error: {err}")))?;
    let graph = graph_matrix(py, &result.dag, return_dag)?;
    if return_diagnostics {
        let diagnostics = diagnostics_dict(py, &result.diagnostics, &result.dag)?;
        diagnostics.set_item("source_prefix", source_prefix)?;
        Ok(PyTuple::new(py, [graph.as_any(), diagnostics.as_any()])?
            .into_any()
            .unbind())
    } else {
        Ok(graph.into_any().unbind())
    }
}

#[pymodule]
fn flopsearch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::flop, m)?)?;
    m.add_function(wrap_pyfunction!(crate::global_greedy_inner, m)?)?;
    m.add_function(wrap_pyfunction!(crate::flop_source_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(crate::flop_notreks, m)?)?;
    m.add_function(wrap_pyfunction!(crate::prune_parents_bic, m)?)?;
    Ok(())
}
