"""Variance- and R2-SortnRegress diagnostic baselines."""
from __future__ import annotations

from typing import Literal

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic,
    is_dag,
)
from workflow.rules.structure_learning_algorithms.flop.adapter import (
    convert_flop_cpdag,
)


def _sort_order(X: np.ndarray, kind: Literal["variance", "r2"]):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    centered = X - X.mean(axis=0, keepdims=True)
    if kind == "variance":
        values = np.var(X, axis=0, ddof=0)
    elif kind == "r2":
        values = np.zeros(d, dtype=float)
        for target in range(d):
            others = [j for j in range(d) if j != target]
            beta, *_ = np.linalg.lstsq(
                centered[:, others], centered[:, target], rcond=None)
            residual = centered[:, target] - centered[:, others] @ beta
            total = float(centered[:, target] @ centered[:, target])
            values[target] = (0.0 if total <= 1e-12 else
                              1.0 - float(residual @ residual) / total)
    else:
        raise ValueError(f"unknown SortnRegress order: {kind}")
    order = sorted(range(d), key=lambda node: (float(values[node]), node))
    return order, values


def _lasso_parent_fit(X, target, parents, *, lambda_bic, alpha_count):
    n = X.shape[0]
    if not parents:
        centered = X[:, target] - X[:, target].mean()
        bic = n * np.log(max(float(centered @ centered) / n, 1e-12))
        return np.zeros(0), float(bic), 0.0
    y = X[:, target]
    design = X[:, parents]
    yc = y - y.mean()
    dc = design - design.mean(axis=0, keepdims=True)
    alpha_max = float(np.max(np.abs(dc.T @ yc)) / n)
    if not np.isfinite(alpha_max) or alpha_max <= 1e-14:
        return np.zeros(len(parents)), float(
            n * np.log(max(float(yc @ yc) / n, 1e-12))), 0.0
    alphas = np.geomspace(alpha_max, max(alpha_max * 1e-4, 1e-14), alpha_count)
    best = None
    for alpha in alphas:
        coefficients = np.zeros(len(parents), dtype=float)
        residual = yc.copy()
        column_norms = np.sum(dc * dc, axis=0) / n
        for _ in range(10000):
            maximum_change = 0.0
            for column in range(len(parents)):
                if column_norms[column] <= 1e-14:
                    continue
                partial = residual + dc[:, column] * coefficients[column]
                rho = float(dc[:, column] @ partial) / n
                updated = np.sign(rho) * max(abs(rho) - alpha, 0.0)
                updated /= column_norms[column]
                change = abs(updated - coefficients[column])
                if change:
                    residual -= dc[:, column] * (updated - coefficients[column])
                    coefficients[column] = updated
                    maximum_change = max(maximum_change, change)
            if maximum_change <= 1e-8:
                break
        support = np.abs(coefficients) > 1e-10
        bic = (n * np.log(max(float(residual @ residual) / n, 1e-12))
               + lambda_bic * int(support.sum()) * np.log(n))
        key = (float(bic), int(support.sum()), -float(alpha))
        if best is None or key < best[0]:
            best = (key, coefficients.copy(), float(alpha))
    assert best is not None
    return best[1], float(best[0][0]), best[2]


def sortnregress(X: np.ndarray, *, kind: Literal["variance", "r2"],
                 lambda_bic: float = 2.0, alpha_count: int = 32):
    """Return an acyclic weighted DAG and diagnostics."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite two-dimensional array")
    order, order_values = _sort_order(X, kind)
    if kind == "r2":
        # R2-SortnRegress is defined on standardized coordinates.  Doing this
        # locally also makes the diagnostic invariant to input rescaling.
        mean = X.mean(axis=0, keepdims=True)
        scale = X.std(axis=0, ddof=0, keepdims=True)
        scale = np.where(scale > 1e-12, scale, 1.0)
        X = (X - mean) / scale
    d = X.shape[1]
    weighted = np.zeros((d, d), dtype=float)
    local_bics, selected_alphas = [], []
    for position, target in enumerate(order):
        parents = order[:position]
        coefficients, bic, alpha = _lasso_parent_fit(
            X, target, parents, lambda_bic=lambda_bic,
            alpha_count=alpha_count)
        if parents:
            weighted[np.asarray(parents), target] = coefficients
        local_bics.append(bic)
        selected_alphas.append(alpha)
    adjacency = (np.abs(weighted) > 1e-10).astype(np.uint8)
    np.fill_diagonal(adjacency, 0)
    if not is_dag(adjacency):
        raise AssertionError("SortnRegress produced a cyclic graph")
    return adjacency, {
        "candidate_graph": adjacency.copy(),
        "weighted_adjacency": weighted,
        "cpdag": convert_flop_cpdag(adjacency, d),
        "optimizer_restarts": 1,
        "optimizer_final_bic": float(gaussian_bic(
            X, adjacency, lambda_bic=lambda_bic)[0]),
        "optimizer_violations": 0,
        "sortnregress_order": order,
        "sortnregress_order_values": order_values.tolist(),
        "sortnregress_selected_alphas": selected_alphas,
        "sortnregress_local_bics": local_bics,
        "sortnregress_standardized_input": kind == "r2",
    }
