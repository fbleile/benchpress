"""Pure UDG definitions and pairwise evidence for the isolated pilot."""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.special import ndtr


def true_udg(dag: np.ndarray) -> np.ndarray:
    """Return the ancestor-overlap UDG, including each node as its own ancestor."""
    d = dag.shape[0]
    reach = dag.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(d):
        reach |= reach[:, [k]] & reach[[k], :]
    ancestors = reach.T
    out = np.zeros((d, d), dtype=np.uint8)
    for i, j in combinations(range(d), 2):
        if np.any(ancestors[i] & ancestors[j]):
            out[i, j] = out[j, i] = 1
    return out


def pairwise_evidence(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return correlations, Fisher-z statistics, and two-sided p-values."""
    n, d = x.shape
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    clipped = np.clip(corr, -1.0 + 1e-12, 1.0 - 1e-12)
    z = np.arctanh(clipped) * math.sqrt(max(1, n - 3))
    p = 2.0 * ndtr(-np.abs(z))
    np.fill_diagonal(z, 0.0)
    np.fill_diagonal(p, 0.0)
    return corr, z, p


def holm_reject(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Holm step-down rejection mask for a vector of p-values."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="mergesort")
    reject = np.zeros(p_values.size, dtype=bool)
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (p_values.size - rank):
            reject[index] = True
        else:
            break
    return reject


def no_trek_pairs(udg: np.ndarray) -> list[tuple[int, int]]:
    return [(i, j) for i, j in combinations(range(udg.shape[0]), 2)
            if not udg[i, j]]


def source_masks_udg(masks: np.ndarray, d: int) -> np.ndarray:
    out = np.zeros((d, d), dtype=np.uint8)
    for i, j in combinations(range(d), 2):
        if int(masks[i]) & int(masks[j]):
            out[i, j] = out[j, i] = 1
    return out


source_mask_udg = source_masks_udg


def source_mask_witness(masks: np.ndarray) -> np.ndarray:
    """Construct the source-to-node witness DAG for a valid mask state."""
    d = len(masks)
    out = np.zeros((d, d), dtype=np.uint8)
    for source in range(d):
        bit = 1 << source
        if int(masks[source]) != bit:
            continue
        for node in range(d):
            if node != source and int(masks[node]) & bit:
                out[source, node] = 1
    return out


def composite_score(masks: np.ndarray, evidence: np.ndarray, prior: float) -> float:
    d = len(masks)
    total = -prior * sum(int(masks[s]) == (1 << s) for s in range(d))
    for i, j in combinations(range(d), 2):
        if int(masks[i]) & int(masks[j]):
            total += float(evidence[i, j])
    return float(total)


def toggle_score_delta(masks: np.ndarray, evidence: np.ndarray, node: int,
                       source: int, add: bool, prior: float) -> float:
    candidate = masks.copy()
    bit = np.uint64(1 << source)
    if add:
        candidate[node] |= bit
    else:
        candidate[node] &= ~bit
    return composite_score(candidate, evidence, prior) - composite_score(masks, evidence, prior)
