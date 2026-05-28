from typing import Sequence, Tuple

import numpy as np
from scipy.linalg import expm, solve


Pair = Tuple[int, int]
TREK_INV_EPS = 1e-8


def _series_I_minus_log_I_minus_W(A: np.ndarray, K: int) -> np.ndarray:
    d = A.shape[0]
    F = np.eye(d, dtype=float)
    power = A.copy()
    for k in range(1, int(K) + 1):
        F = F + power / float(k)
        power = power @ A
    return F


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

    A = np.asarray(W, dtype=float) ** 2
    d = A.shape[0]
    if seq == "exp":
        F = expm(A)
    elif seq == "log":
        F = _series_I_minus_log_I_minus_W(A, K=2 * d)
    elif seq == "inv":
        try:
            I = np.eye(d, dtype=float)
            F = solve(I - A + TREK_INV_EPS * I, I)
        except np.linalg.LinAlgError:
            return float("inf")
    else:
        raise ValueError(f"Unsupported trek_seq: {seq}")

    H = F.T @ F
    return float(sum(0.5 * (H[i, j] + H[j, i]) for i, j in pairs))


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
