"""Exact finite-support structural feasibility diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .gaussian_bic import common_ancestor_violations, is_dag


def literal_support(weighted):
    support = (np.asarray(weighted) != 0).astype(int)
    np.fill_diagonal(support, 0)
    return support


def support_at_threshold(weighted, threshold):
    support = (np.abs(np.asarray(weighted)) > float(threshold)).astype(int)
    np.fill_diagonal(support, 0)
    return support


def support_diagnostics(weighted, pairs=()):
    support = literal_support(weighted)
    count, labels = connected_components(
        csr_matrix(support), directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=count)
    nontrivial = sizes[sizes > 1]
    return {
        "raw_exact_nonzero_count": int(support.sum()),
        "raw_literal_support_is_dag": bool(is_dag(support)),
        "raw_support_scc_count": int(count),
        "raw_support_largest_scc": int(sizes.max(initial=0)),
        "raw_support_nontrivial_scc_count": int(len(nontrivial)),
        "raw_oracle_violations": int(
            common_ancestor_violations(support, pairs)),
    }


def _thresholds(weighted):
    values = np.abs(np.asarray(weighted, dtype=float)).copy()
    np.fill_diagonal(values, 0.)
    return np.unique(np.concatenate(([0.], values[values != 0])))


def _minimum_threshold(weighted, predicate):
    thresholds = _thresholds(weighted)
    # Feasibility is monotone under edge deletion, so binary search is exact
    # over the finite set of distinct magnitudes.
    low, high = 0, len(thresholds) - 1
    if predicate(support_at_threshold(weighted, thresholds[low])):
        index = low
    else:
        while low < high:
            middle = (low + high) // 2
            if predicate(support_at_threshold(weighted, thresholds[middle])):
                high = middle
            else:
                low = middle + 1
        index = low
    graph = support_at_threshold(weighted, thresholds[index])
    if not predicate(graph):
        raise RuntimeError("even the empty support is not structurally feasible")
    previous = None
    previous_feasible = None
    if index > 0:
        previous = float(thresholds[index - 1])
        previous_feasible = bool(predicate(
            support_at_threshold(weighted, thresholds[index - 1])))
        if previous_feasible:
            raise RuntimeError("feasibility threshold is not minimal")
    return float(thresholds[index]), graph, previous, previous_feasible


def feasibility_thresholds(weighted, pairs=()):
    pairs = tuple(pairs)
    dag_tau, _, _, _ = _minimum_threshold(weighted, is_dag)
    if pairs:
        mi_predicate = lambda graph: common_ancestor_violations(
            graph, pairs) == 0
        mi_tau, _, _, _ = _minimum_threshold(weighted, mi_predicate)
        joint_predicate = lambda graph: (
            is_dag(graph)
            and common_ancestor_violations(graph, pairs) == 0)
        joint_tau, graph, previous, previous_feasible = _minimum_threshold(
            weighted, joint_predicate)
    else:
        mi_tau = 0.0
        joint_tau, graph, previous, previous_feasible = _minimum_threshold(
            weighted, is_dag)
    return {
        "tau_dag": dag_tau,
        "tau_mi": mi_tau,
        "tau_feas": joint_tau,
        "graph": graph,
        "previous_threshold": previous,
        "previous_support_feasible": previous_feasible,
    }


def minimal_feasibility_projection(weighted, pairs=()):
    result = feasibility_thresholds(weighted, pairs)
    graph = result["graph"]
    if not is_dag(graph):
        raise RuntimeError("minimal feasibility projection is cyclic")
    if pairs and common_ancestor_violations(graph, pairs):
        raise RuntimeError("minimal feasibility projection violates knowledge")
    return result
