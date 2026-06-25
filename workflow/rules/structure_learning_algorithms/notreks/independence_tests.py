import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


Pair = Tuple[int, int]
INDEPENDENCE_CACHE_VERSION = "notreks-independence-cache-v2"


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


def _gcastle_ci_test(data: np.ndarray, i: int, j: int, method: str) -> Tuple[float, float, float, str]:
    try:
        from castle.common.independence_tests import CITest
    except ImportError as exc:
        raise ImportError(
            f"independence_test={method!r} requires gCastle. Install it with: pip install gcastle"
        ) from exc

    mapping = {
        "gcastle_fisherz": ("fisherz_test", "fisherz"),
        "gcastle_g2": ("g2_test", "g2"),
        "gcastle_chi2": ("chi2_test", "chi2"),
    }
    function_name, raw_name = mapping[method]
    result = getattr(CITest, function_name)(data, int(i), int(j), [])
    if isinstance(result, tuple):
        if len(result) < 3:
            raise ValueError(f"gCastle {function_name} returned an unexpected tuple: {result!r}")
        statistic, dof, pvalue = result[0], result[1], result[-1]
    else:
        statistic, dof, pvalue = np.nan, np.nan, result
    statistic_value = float("nan") if statistic is None else float(statistic)
    dof_value = float("nan") if dof is None else float(dof)
    return statistic_value, float(pvalue), dof_value, raw_name


def _test_pair(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float, float, str]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan"), 1.0, float("nan"), method
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0, float("nan"), method
    if method == "pearson":
        result = stats.pearsonr(x, y)
        return float(result.statistic), float(result.pvalue), float("nan"), method
    if method == "spearman":
        result = stats.spearmanr(x, y)
        return float(result.statistic), float(result.pvalue), float("nan"), method
    if method == "hsic":
        try:
            from hyppo.independence import Hsic
        except ImportError as exc:
            raise ImportError(
                "independence_test='hsic' requires the optional package hyppo. "
                "Install it with: pip install hyppo"
            ) from exc
        stat, pvalue = Hsic().test(x.reshape(-1, 1), y.reshape(-1, 1))
        return float(stat), float(pvalue), float("nan"), method
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
            return float(getattr(result, "statistic", np.nan)), float(result.pvalue), float("nan"), method
        stat, pvalue = Dcorr().test(x.reshape(-1, 1), y.reshape(-1, 1))
        return float(stat), float(pvalue), float("nan"), method
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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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


def _load_cache_entry(entry_dir: Path, payload: Dict[str, object]) -> Optional[List[Dict[str, object]]]:
    metadata_path = entry_dir / "metadata.json"
    results_path = entry_dir / "all_test_results.csv"
    if not metadata_path.is_file() or not results_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("payload") != payload:
            return None

        rows: List[Dict[str, object]] = []
        with results_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"i", "j", "pair_key", "statistic", "p_value", "dof", "raw_test_name"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                return None
            for raw in reader:
                p_value = float(raw["p_value"])
                if not np.isfinite(p_value):
                    return None
                row = {
                    "i": int(raw["i"]),
                    "j": int(raw["j"]),
                    "pair_key": raw["pair_key"],
                    "statistic": float(raw["statistic"]) if raw["statistic"] else float("nan"),
                    "p_value": p_value,
                    "dof": float(raw["dof"]) if raw.get("dof") else float("nan"),
                    "raw_test_name": raw.get("raw_test_name") or str(payload["independence_test"]),
                }
                rows.append(row)
        expected = int(payload["d"]) * (int(payload["d"]) - 1) // 2
        if len(rows) != expected:
            return None
        return rows
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, csv.Error):
        return None


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
        "dof",
        "raw_test_name",
        "adjusted_p_value",
        "alpha_used",
        "accepted_independence",
    ]
    _atomic_write_csv(
        entry_dir / "all_test_results.csv",
        fields,
        [{field: row.get(field) for field in fields} for row in rows],
    )
    _atomic_write_csv(
        entry_dir / "accepted_pairs.csv",
        ["i", "j"],
        [{"i": i, "j": j} for i, j in result.pairs],
    )

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
    _atomic_write_text(entry_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n")


def _compute_result_from_pvalues(
    raw_pairs: List[Pair],
    pvals: List[float],
    stats_values: List[float],
    *,
    dof_values: Optional[List[float]] = None,
    raw_test_names: Optional[List[str]] = None,
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
    dof_values = dof_values or [float("nan")] * len(raw_pairs)
    raw_test_names = raw_test_names or [method] * len(raw_pairs)
    for (i, j), statistic, pvalue, dof, raw_test_name, keep in zip(
        raw_pairs, stats_values, pvals, dof_values, raw_test_names, accepted
    ):
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
                "dof": float(dof),
                "raw_test_name": str(raw_test_name),
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


def _compute_result_from_raw_rows(
    rows: List[Dict[str, object]],
    *,
    method: str,
    alpha: float,
    correction: str,
    columns: Optional[Sequence[str]],
    cache_status: str,
    cache_key: Optional[str],
    cache_dir: Optional[Path],
) -> IndependenceResult:
    raw_pairs = [(int(row["i"]), int(row["j"])) for row in rows]
    pvals = [float(row["p_value"]) for row in rows]
    stats_values = [float(row["statistic"]) for row in rows]
    dof_values = [float(row.get("dof", float("nan"))) for row in rows]
    raw_test_names = [str(row.get("raw_test_name", method)) for row in rows]
    return _compute_result_from_pvalues(
        raw_pairs,
        pvals,
        stats_values,
        dof_values=dof_values,
        raw_test_names=raw_test_names,
        method=method,
        alpha=float(alpha),
        correction=correction,
        columns=columns,
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
        cached_rows = _load_cache_entry(cache_entry_dir, payload)
        if cached_rows is not None:
            result = _compute_result_from_raw_rows(
                cached_rows,
                method=method,
                alpha=float(alpha),
                correction=correction,
                columns=columns,
                cache_status="hit",
                cache_key=cache_key,
                cache_dir=cache_entry_dir,
            )
            return result

    raw_pairs: List[Pair] = []
    pvals: List[float] = []
    stats_values: List[float] = []
    dof_values: List[float] = []
    raw_test_names: List[str] = []

    for i in range(d - 1):
        for j in range(i + 1, d):
            if method in {"gcastle_fisherz", "gcastle_g2", "gcastle_chi2"}:
                statistic, pvalue, dof, raw_test_name = _gcastle_ci_test(X, i, j, method)
            else:
                statistic, pvalue, dof, raw_test_name = _test_pair(X[:, i], X[:, j], method)
            raw_pairs.append((i, j))
            stats_values.append(statistic)
            pvals.append(pvalue)
            dof_values.append(dof)
            raw_test_names.append(raw_test_name)

    result = _compute_result_from_pvalues(
        raw_pairs,
        pvals,
        stats_values,
        dof_values=dof_values,
        raw_test_names=raw_test_names,
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
