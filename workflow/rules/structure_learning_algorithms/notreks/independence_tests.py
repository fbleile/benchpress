import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


Pair = Tuple[int, int]
INDEPENDENCE_CACHE_VERSION = "notreks-independence-cache-v1"


@dataclass(frozen=True)
class IndependenceResult:
    pairs: List[Pair]
    number_of_tests: int
    pvalues: Dict[str, float]
    alpha_used: float
    method: str
    correction: str
    cache_status: str = "disabled"
    cache_key: Optional[str] = None
    cache_dir: Optional[str] = None
    results: Optional[List[Dict[str, object]]] = None


def _pair_key(i: int, j: int, columns: Optional[Sequence[str]]) -> str:
    if columns is None:
        return f"{i},{j}"
    return f"{columns[i]},{columns[j]}"


def _test_pair(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan"), 1.0
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0
    if method == "pearson":
        result = stats.pearsonr(x, y)
        return float(result.statistic), float(result.pvalue)
    if method == "spearman":
        result = stats.spearmanr(x, y)
        return float(result.statistic), float(result.pvalue)
    if method == "hsic":
        try:
            from hyppo.independence import Hsic
        except ImportError as exc:
            raise ImportError(
                "independence_test='hsic' requires the optional package hyppo. "
                "Install it with: pip install hyppo"
            ) from exc
        stat, pvalue = Hsic().test(x.reshape(-1, 1), y.reshape(-1, 1))
        return float(stat), float(pvalue)
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
            return float(getattr(result, "statistic", np.nan)), float(result.pvalue)
        stat, pvalue = Dcorr().test(x.reshape(-1, 1), y.reshape(-1, 1))
        return float(stat), float(pvalue)
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


def _file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(X: np.ndarray) -> str:
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(X.shape).encode("utf-8"))
    digest.update(X.tobytes())
    return digest.hexdigest()


def _cache_payload(
    X: np.ndarray,
    *,
    method: str,
    alpha: float,
    correction: str,
    columns: Optional[Sequence[str]],
    dataset_path: Optional[Path],
    extra_params: Optional[Dict[str, object]],
) -> Dict[str, object]:
    dataset_path_text = str(dataset_path.resolve()) if dataset_path is not None else None
    return {
        "cache_version": INDEPENDENCE_CACHE_VERSION,
        "dataset_hash": dataset_sha256(X),
        "dataset_path": dataset_path_text,
        "dataset_file_hash": _file_sha256(dataset_path) if dataset_path is not None else None,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "columns": list(columns) if columns is not None else None,
        "independence_test": method,
        "independence_alpha": float(alpha),
        "independence_correction": correction,
        "extra_params": extra_params or {},
    }


def independence_cache_key(payload: Dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _apply_correction(raw_pairs: List[Pair], pvalues: np.ndarray, alpha: float, correction: str) -> Tuple[np.ndarray, float]:
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
    if len(raw_pairs) != accepted.size:
        raise ValueError("Internal error: pair and p-value counts differ")
    return accepted, alpha_used


def _rows_to_result(
    rows: List[Dict[str, object]],
    *,
    method: str,
    correction: str,
    alpha_used: float,
    pvalue_map: Dict[str, float],
    cache_status: str,
    cache_key: Optional[str],
    cache_dir: Optional[Path],
) -> IndependenceResult:
    pairs = [
        (int(row["i"]), int(row["j"]))
        for row in rows
        if bool(row["accepted_independence"])
    ]
    return IndependenceResult(
        pairs=pairs,
        number_of_tests=len(rows),
        pvalues=pvalue_map,
        alpha_used=alpha_used,
        method=method,
        correction=correction,
        cache_status=cache_status,
        cache_key=cache_key,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        results=rows,
    )


def _load_cache_entry(entry_dir: Path, payload: Dict[str, object]) -> Optional[IndependenceResult]:
    metadata_path = entry_dir / "metadata.json"
    results_path = entry_dir / "all_test_results.csv"
    if not metadata_path.is_file() or not results_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("payload") != payload:
        return None

    rows: List[Dict[str, object]] = []
    pvalue_map: Dict[str, float] = {}
    with results_path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {
                "i": int(raw["i"]),
                "j": int(raw["j"]),
                "pair_key": raw["pair_key"],
                "statistic": float(raw["statistic"]) if raw["statistic"] else float("nan"),
                "p_value": float(raw["p_value"]) if raw["p_value"] else float("nan"),
                "adjusted_p_value": float(raw["adjusted_p_value"]) if raw.get("adjusted_p_value") else float("nan"),
                "alpha_used": float(raw["alpha_used"]),
                "accepted_independence": raw["accepted_independence"].lower() == "true",
            }
            rows.append(row)
            pvalue_map[str(row["pair_key"])] = float(row["p_value"])
    return _rows_to_result(
        rows,
        method=str(metadata["test_name"]),
        correction=str(metadata["correction"]),
        alpha_used=float(metadata["alpha_used"]),
        pvalue_map=pvalue_map,
        cache_status="hit",
        cache_key=str(metadata["cache_key"]),
        cache_dir=entry_dir,
    )


def _write_cache_entry(
    entry_dir: Path,
    payload: Dict[str, object],
    result: IndependenceResult,
) -> None:
    entry_dir.mkdir(parents=True, exist_ok=True)
    rows = result.results or []
    fields = [
        "i",
        "j",
        "pair_key",
        "statistic",
        "p_value",
        "adjusted_p_value",
        "alpha_used",
        "accepted_independence",
    ]
    with (entry_dir / "all_test_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    with (entry_dir / "accepted_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["i", "j"])
        writer.writeheader()
        for i, j in result.pairs:
            writer.writerow({"i": i, "j": j})

    metadata = {
        "cache_key": result.cache_key,
        "payload": payload,
        "dataset_hash": payload["dataset_hash"],
        "dataset_path": payload["dataset_path"],
        "test_name": result.method,
        "correction": result.correction,
        "alpha_used": result.alpha_used,
        "number_of_tests": result.number_of_tests,
        "number_of_accepted_independence_pairs": len(result.pairs),
        "accepted_fraction": len(result.pairs) / max(result.number_of_tests, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "implementation_version": INDEPENDENCE_CACHE_VERSION,
    }
    (entry_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _compute_result_from_pvalues(
    raw_pairs: List[Pair],
    pvals: List[float],
    stats_values: List[float],
    *,
    method: str,
    alpha: float,
    correction: str,
    columns: Optional[Sequence[str]],
    cache_status: str,
    cache_key: Optional[str],
    cache_dir: Optional[Path],
) -> IndependenceResult:
    pvalues = np.asarray(pvals, dtype=float)
    accepted, alpha_used = _apply_correction(raw_pairs, pvalues, float(alpha), correction)
    rows: List[Dict[str, object]] = []
    pvalue_map: Dict[str, float] = {}
    for (i, j), statistic, pvalue, keep in zip(raw_pairs, stats_values, pvals, accepted):
        pair_key = _pair_key(i, j, columns)
        pvalue_map[pair_key] = float(pvalue)
        if correction == "none":
            adjusted_pvalue = float(pvalue)
        elif correction == "bonferroni":
            adjusted_pvalue = min(1.0, float(pvalue) * max(len(raw_pairs), 1))
        else:
            adjusted_pvalue = float("nan")
        rows.append(
            {
                "i": int(i),
                "j": int(j),
                "pair_key": pair_key,
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "adjusted_p_value": adjusted_pvalue,
                "alpha_used": float(alpha_used),
                "accepted_independence": bool(keep),
            }
        )
    return _rows_to_result(
        rows,
        method=method,
        correction=correction,
        alpha_used=alpha_used,
        pvalue_map=pvalue_map,
        cache_status=cache_status,
        cache_key=cache_key,
        cache_dir=cache_dir,
    )


def pairwise_independence_candidates(
    X: np.ndarray,
    *,
    method: str,
    alpha: float,
    correction: str,
    columns: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    extra_params: Optional[Dict[str, object]] = None,
) -> IndependenceResult:
    method = str(method)
    correction = str(correction)
    X = np.asarray(X, dtype=float)
    d = X.shape[1]

    if method == "none":
        return IndependenceResult([], 0, {}, float(alpha), method, correction)

    cache_entry_dir = None
    cache_key = None
    payload = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        dataset_path = Path(dataset_path) if dataset_path is not None else None
        payload = _cache_payload(
            X,
            method=method,
            alpha=float(alpha),
            correction=correction,
            columns=columns,
            dataset_path=dataset_path,
            extra_params=extra_params,
        )
        cache_key = independence_cache_key(payload)
        cache_entry_dir = cache_dir / cache_key
        cached = _load_cache_entry(cache_entry_dir, payload)
        if cached is not None:
            return cached

    raw_pairs: List[Pair] = []
    pvals: List[float] = []
    stats_values: List[float] = []

    for i in range(d - 1):
        for j in range(i + 1, d):
            statistic, pvalue = _test_pair(X[:, i], X[:, j], method)
            raw_pairs.append((i, j))
            stats_values.append(statistic)
            pvals.append(pvalue)

    result = _compute_result_from_pvalues(
        raw_pairs,
        pvals,
        stats_values,
        method=method,
        alpha=float(alpha),
        correction=correction,
        columns=columns,
        cache_status="miss" if cache_entry_dir is not None else "disabled",
        cache_key=cache_key,
        cache_dir=cache_entry_dir,
    )
    if cache_entry_dir is not None and payload is not None:
        _write_cache_entry(cache_entry_dir, payload, result)
    return result


def graph_implied_no_trek_pairs(adjmat: np.ndarray) -> set[Pair]:
    graph = np.asarray(adjmat, dtype=bool)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjmat must be a square matrix")
    d = graph.shape[0]
    reach = graph.copy()
    np.fill_diagonal(reach, True)
    for k in range(d):
        reach = reach | (reach[:, [k]] & reach[[k], :])

    no_trek: set[Pair] = set()
    for i in range(d - 1):
        for j in range(i + 1, d):
            has_trek = bool(np.any(reach[:, i] & reach[:, j]))
            if not has_trek:
                no_trek.add((i, j))
    return no_trek


def evaluate_no_trek_independence(
    accepted_pairs: Sequence[Pair],
    adjmat: np.ndarray,
) -> Dict[str, float]:
    true_pairs = graph_implied_no_trek_pairs(adjmat)
    accepted = {tuple(sorted((int(i), int(j)))) for i, j in accepted_pairs}
    tested_count = adjmat.shape[0] * (adjmat.shape[0] - 1) // 2
    tp = len(accepted & true_pairs)
    fp = len(accepted - true_pairs)
    fn = len(true_pairs - accepted)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "num_true_no_trek_pairs": int(len(true_pairs)),
        "num_tested_pairs": int(tested_count),
        "num_accepted_pairs": int(len(accepted)),
        "true_positive_no_trek_pairs": int(tp),
        "false_positive_pairs": int(fp),
        "false_negative_no_trek_pairs": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def write_no_trek_diagnostics(
    cache_entry_dir: Path,
    accepted_pairs: Sequence[Pair],
    adjmat: np.ndarray,
) -> Dict[str, float]:
    cache_entry_dir = Path(cache_entry_dir)
    summary = evaluate_no_trek_independence(accepted_pairs, adjmat)
    (cache_entry_dir / "ground_truth_no_trek_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    true_pairs = graph_implied_no_trek_pairs(adjmat)
    accepted = {tuple(sorted((int(i), int(j)))) for i, j in accepted_pairs}
    rows = []
    d = adjmat.shape[0]
    for i in range(d - 1):
        for j in range(i + 1, d):
            pair = (i, j)
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "graph_implied_no_trek": pair in true_pairs,
                    "accepted_independence": pair in accepted,
                }
            )
    with (cache_entry_dir / "ground_truth_no_trek_eval.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["i", "j", "graph_implied_no_trek", "accepted_independence"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary
