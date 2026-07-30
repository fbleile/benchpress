"""Validated JSON sidecars for structural oracle no-trek knowledge."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def no_trek_pairs_from_dag(adjacency: np.ndarray, node_names: list[str]) -> list[tuple[str, str]]:
    A = np.asarray(adjacency)
    d = len(node_names)
    if A.shape != (d, d) or len(set(node_names)) != d:
        raise ValueError("true DAG shape and unique node_names must agree")
    reach = (A != 0).copy()
    np.fill_diagonal(reach, True)
    for k in range(d):
        reach |= reach[:, [k]] & reach[[k], :]
    # Benchpress convention A[parent, child]; ancestors are rows in each column.
    return [(node_names[i], node_names[j]) for i, j in combinations(range(d), 2)
            if not np.any(reach[:, i] & reach[:, j])]


def validate_sidecar(payload: dict, expected_node_names=None) -> dict:
    if not isinstance(payload, dict) or payload.get("type") != "no_trek_pairs":
        raise ValueError("sidecar type must be 'no_trek_pairs'")
    names = payload.get("node_names")
    if not isinstance(names, list) or not all(isinstance(x, str) and x for x in names):
        raise ValueError("node_names must be a list of nonempty strings")
    if len(names) != len(set(names)):
        raise ValueError("duplicate node names are not allowed")
    if expected_node_names is not None and list(expected_node_names) != names:
        raise ValueError("sidecar node_names do not exactly match data columns")
    known, canonical = set(names), []
    seen = set()
    for pair in payload.get("pairs", []):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("each pair must be a two-element JSON list")
        a, b = pair
        if a not in known or b not in known:
            raise ValueError(f"unknown node in pair {pair}")
        if a == b:
            raise ValueError("self-pairs are not allowed")
        item = tuple(sorted((a, b), key=names.index))
        if item in seen:
            raise ValueError(f"duplicate unordered pair {pair}")
        seen.add(item)
        canonical.append(list(item))
    canonical.sort(key=lambda p: (names.index(p[0]), names.index(p[1])))
    return {
        "type": "no_trek_pairs", "source": payload.get("source", "file"),
        "node_names": names, "pairs": canonical,
        "metadata": {"number_of_nodes": len(names), "number_of_pairs": len(canonical)},
    }


def load_sidecar(path, expected_node_names=None) -> dict:
    with open(path, encoding="utf-8") as handle:
        return validate_sidecar(json.load(handle), expected_node_names)


def write_oracle_sidecar(adjacency, node_names, path):
    payload = validate_sidecar({
        "type": "no_trek_pairs", "source": "true_dag", "node_names": list(node_names),
        "pairs": [list(x) for x in no_trek_pairs_from_dag(adjacency, list(node_names))],
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def named_pairs_to_indices(payload: dict, expected_node_names=None) -> list[tuple[int, int]]:
    valid = validate_sidecar(payload, expected_node_names)
    index = {name: i for i, name in enumerate(valid["node_names"])}
    return [(index[a], index[b]) for a, b in valid["pairs"]]
