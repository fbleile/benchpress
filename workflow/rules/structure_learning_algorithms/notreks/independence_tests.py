from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


Pair = Tuple[int, int]


@dataclass(frozen=True)
class IndependenceResult:
    pairs: List[Pair]
    number_of_tests: int
    pvalues: Dict[str, float]
    alpha_used: float
    method: str
    correction: str


def _pair_key(i: int, j: int, columns: Optional[Sequence[str]]) -> str:
    if columns is None:
        return f"{i},{j}"
    return f"{columns[i]},{columns[j]}"


def _test_pair(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return 1.0
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 1.0
    if method == "pearson":
        return float(stats.pearsonr(x, y).pvalue)
    if method == "spearman":
        return float(stats.spearmanr(x, y).pvalue)
    if method == "hsic":
        try:
            from hyppo.independence import Hsic
        except ImportError as exc:
            raise ImportError(
                "independence_test='hsic' requires the optional package hyppo. "
                "Install it with: pip install hyppo"
            ) from exc
        _, pvalue = Hsic().test(x.reshape(-1, 1), y.reshape(-1, 1))
        return float(pvalue)
    if method == "dcor":
        try:
            from hyppo.independence import Dcorr
        except ImportError:
            try:
                import dcor
            except ImportError as exc:
                raise ImportError(
                    "independence_test='dcor' requires the optional package hyppo "
                    "(preferred, install with: pip install hyppo) or dcor "
                    "(install with: pip install dcor)."
                ) from exc
            result = dcor.independence.distance_covariance_test(
                x.reshape(-1, 1),
                y.reshape(-1, 1),
                num_resamples=100,
            )
            return float(result.pvalue)
        _, pvalue = Dcorr().test(x.reshape(-1, 1), y.reshape(-1, 1))
        return float(pvalue)
    raise ValueError(f"Unsupported independence test: {method}")


def _benjamini_hochberg_accept_independence(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    m = pvalues.size
    if m == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    rejected = sorted_p <= thresholds

    dependent = np.zeros(m, dtype=bool)
    if np.any(rejected):
        k = np.max(np.nonzero(rejected)[0])
        dependent[order[: k + 1]] = True
    return ~dependent


def pairwise_independence_candidates(
    X: np.ndarray,
    *,
    method: str,
    alpha: float,
    correction: str,
    columns: Optional[Sequence[str]] = None,
) -> IndependenceResult:
    method = str(method)
    correction = str(correction)
    X = np.asarray(X, dtype=float)
    d = X.shape[1]

    if method == "none":
        return IndependenceResult([], 0, {}, float(alpha), method, correction)

    raw_pairs: List[Pair] = []
    pvals: List[float] = []
    pvalue_map: Dict[str, float] = {}

    for i in range(d - 1):
        for j in range(i + 1, d):
            pvalue = _test_pair(X[:, i], X[:, j], method)
            raw_pairs.append((i, j))
            pvals.append(pvalue)
            pvalue_map[_pair_key(i, j, columns)] = pvalue

    pvalues = np.asarray(pvals, dtype=float)
    m = pvalues.size

    if correction == "none":
        alpha_used = float(alpha)
        accepted = pvalues > alpha_used
    elif correction == "bonferroni":
        alpha_used = float(alpha) / max(m, 1)
        accepted = pvalues > alpha_used
    elif correction == "benjamini-hochberg":
        alpha_used = float(alpha)
        accepted = _benjamini_hochberg_accept_independence(pvalues, float(alpha))
    else:
        raise ValueError(f"Unsupported independence correction: {correction}")

    pairs = [pair for pair, keep in zip(raw_pairs, accepted) if keep]
    return IndependenceResult(pairs, m, pvalue_map, alpha_used, method, correction)
