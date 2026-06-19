"""Create deterministic data consumed through Benchpress's fixed-data modules."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FixedDataSpec:
    run_name: str
    d: int = 10
    n_values: tuple[int, ...] = (500,)
    seeds: tuple[int, ...] = (1, 2, 3)
    expected_degree: float = 2.0
    standardized: bool = True
    graph_seed: int = 1729
    weight_seed: int = 2718


@dataclass(frozen=True)
class FixedDataReference:
    data_id: str
    graph_id: str
    data_dir: str
    graph_path: str
    metadata_path: str


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    if not cleaned:
        raise ValueError("run_name must contain at least one filename-safe character")
    return cleaned


def spec_from_meta(meta: dict, default_run_name: str) -> FixedDataSpec:
    raw = meta.get("fixed_data", {})
    n_raw = raw.get("n_values", raw.get("n", [500]))
    seed_raw = raw.get("seeds", [1, 2, 3])
    if isinstance(n_raw, int):
        n_raw = [n_raw]
    if isinstance(seed_raw, int):
        seed_raw = [seed_raw]
    spec = FixedDataSpec(
        run_name=safe_name(str(meta.get("run_name", default_run_name))),
        d=int(raw.get("d", 10)),
        n_values=tuple(int(value) for value in n_raw),
        seeds=tuple(int(value) for value in seed_raw),
        expected_degree=float(raw.get("expected_degree", 2.0)),
        standardized=bool(raw.get("standardized", True)),
        graph_seed=int(raw.get("graph_seed", 1729)),
        weight_seed=int(raw.get("weight_seed", 2718)),
    )
    if spec.d < 2:
        raise ValueError("fixed_data.d must be at least 2")
    if not spec.n_values or any(value < 2 for value in spec.n_values):
        raise ValueError("fixed_data.n_values must contain positive sample sizes")
    if not spec.seeds:
        raise ValueError("fixed_data.seeds must not be empty")
    return spec


def _generate_dag(d: int, expected_degree: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    edge_probability = min(float(expected_degree) / max(d - 1, 1), 1.0)
    adjacency = np.zeros((d, d), dtype=int)
    for left in range(d):
        for right in range(left + 1, d):
            if rng.random() < edge_probability:
                adjacency[order[left], order[right]] = 1
    return adjacency


def _sample_weights(adjacency: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.zeros_like(adjacency, dtype=float)
    rows, columns = np.where(adjacency != 0)
    for row, column in zip(rows, columns):
        sign = -1.0 if rng.random() < 0.5 else 1.0
        weights[row, column] = sign * rng.uniform(0.5, 1.0)
    return weights


def _sample_linear_gaussian(weights: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, weights.shape[0]))
    return noise @ np.linalg.inv(np.eye(weights.shape[0]) - weights)


def _standardize(data: np.ndarray) -> np.ndarray:
    centered = data - np.mean(data, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    return centered / scale


def prepare_fixed_data(
    repo_root: Path,
    run_dir: Path,
    spec: FixedDataSpec,
) -> FixedDataReference:
    """Write one fixed DAG and shared datasets in Benchpress resource folders."""

    run_name = safe_name(spec.run_name)
    data_id = f"notreks_hparam/{run_name}"
    graph_id = f"notreks_hparam_{run_name}.csv"
    data_dir = repo_root / "resources/data/mydatasets" / data_id
    graph_path = repo_root / "resources/adjmat/myadjmats" / graph_id
    metadata_path = run_dir / "fixed_data/metadata.json"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=False)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    adjacency = _generate_dag(spec.d, spec.expected_degree, spec.graph_seed)
    weights = _sample_weights(adjacency, spec.weight_seed)
    columns = [f"X{index}" for index in range(spec.d)]
    pd.DataFrame(adjacency, columns=columns).to_csv(graph_path, index=False)

    files = []
    for n in spec.n_values:
        for seed in spec.seeds:
            data = _sample_linear_gaussian(weights, n, seed)
            if spec.standardized:
                data = _standardize(data)
            filename = f"linear_gaussian_d{spec.d}_n{n}_seed{seed}.csv"
            path = data_dir / filename
            pd.DataFrame(data, columns=columns).to_csv(path, index=False)
            files.append(str(path.relative_to(repo_root)))

    payload = {
        "spec": asdict(spec),
        "data_id": data_id,
        "graph_id": graph_id,
        "data_files": files,
        "graph_path": str(graph_path.relative_to(repo_root)),
        "format": "Benchpress fixed data and fixed adjacency CSV",
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
    return FixedDataReference(
        data_id=data_id,
        graph_id=graph_id,
        data_dir=str(data_dir.relative_to(repo_root)),
        graph_path=str(graph_path.relative_to(repo_root)),
        metadata_path=str(metadata_path),
    )
