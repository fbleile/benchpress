from typing import Sequence, Tuple

import numpy as np
from scipy.linalg import expm


Pair = Tuple[int, int]


def dag_penalty_value(W: np.ndarray, seq: str, dag_s: float) -> float:
    A = np.asarray(W, dtype=float) ** 2
    d = A.shape[0]
    if seq == "none":
        return 0.0
    if seq == "exp":
        return float(np.trace(expm(A)) - d)
    if seq == "log":
        sign, logdet = np.linalg.slogdet(np.eye(d) - A / max(d, 1))
        if sign <= 0:
            return float("inf")
        return float(-logdet)
    if seq == "inv":
        try:
            return float(np.trace(np.linalg.inv(np.eye(d) - A / max(d, 1))) - d)
        except np.linalg.LinAlgError:
            return float("inf")
    if seq == "logdet":
        s = float(dag_s)
        sign, logdet = np.linalg.slogdet(s * np.eye(d) - A)
        if sign <= 0:
            return float("inf")
        return float(-logdet + d * np.log(s))
    raise ValueError(f"Unsupported dag_seq: {seq}")


def trek_penalty_value(W: np.ndarray, seq: str, pairs: Sequence[Pair]) -> float:
    if seq == "none" or len(pairs) == 0:
        return 0.0

    A = np.abs(np.asarray(W, dtype=float))
    d = A.shape[0]
    if seq == "exp":
        M = expm(A)
    elif seq == "log":
        M = -np.log(np.maximum(1.0 - A / max(d, 1), 1e-12))
    elif seq == "inv":
        try:
            M = np.linalg.inv(np.eye(d) - A / max(d, 1))
        except np.linalg.LinAlgError:
            return float("inf")
    else:
        raise ValueError(f"Unsupported trek_seq: {seq}")

    return float(sum(M[i, j] + M[j, i] for i, j in pairs))


def regularizer_value(W: np.ndarray, regularizer: str) -> float:
    if regularizer == "none":
        return 0.0
    if regularizer == "l1":
        return float(np.sum(np.abs(W)))
    if regularizer == "l2":
        return float(np.sum(W * W))
    raise ValueError(f"Unsupported regularizer: {regularizer}")


def penalty_diagnostics(W: np.ndarray, cfg, independence_pairs: Sequence[Pair]) -> dict:
    return {
        "dag_penalty": cfg.dag_reg * dag_penalty_value(W, cfg.dag_seq, cfg.dag_s),
        "trek_penalty": cfg.trek_reg * trek_penalty_value(W, cfg.trek_seq, independence_pairs),
        "regularizer": cfg.regularizer_scale * regularizer_value(W, cfg.regularizer),
        "n_independence_candidates": len(independence_pairs),
    }
