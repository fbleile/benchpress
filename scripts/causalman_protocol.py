"""Official CausalMan preparation and preflight for the NOTREKS protocol.

This module is intentionally separate from the causalAssembly adapter.  It
does not contain a fallback generator: the official CausalMan package is an
optional, large upstream dependency and preparation fails explicitly when it
is unavailable.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SOURCE_URL = "https://github.com/boschresearch/CausalMan"
PAPER_URL = "https://arxiv.org/abs/2502.12707"
VARIANTS = ("causalman_small", "causalman_medium")
DISCOVERY_SIZES = (500, 2000, 5000)
MASTER_SEED = 20260917


@contextlib.contextmanager
def _manifest_lock(cache: Path):
    """Serialize only the small shared manifest update, not simulation work."""
    lock_path = cache / ".manifest.lock"
    lock_path.touch(exist_ok=True)
    try:
        import fcntl
        with lock_path.open("r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover - Windows fallback
        yield


def derive_seed(namespace: str, *parts: object, master: int = MASTER_SEED) -> int:
    payload = json.dumps([namespace, master, *parts], separators=(",", ":"),
                         sort_keys=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2**32


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_upstream():
    try:
        from causalman import CausalMan
    except ImportError as exc:
        raise RuntimeError(
            "CausalMan preparation is blocked: the official upstream package "
            "is not installed. Install the pinned CausalMan release and its "
            "precomputed assets in the preparation environment; no micro or "
            "hand-built substitute is used.") from exc
    return CausalMan


def package_metadata() -> dict[str, str]:
    result = {"source_url": SOURCE_URL, "paper_url": PAPER_URL,
              "license": "AGPL-3.0", "upstream_revision":
              os.environ.get("CAUSALMAN_UPSTREAM_REVISION", "unknown")}
    for name in ("causalman", "numpy", "pandas", "networkx"):
        try:
            result[f"{name}_version"] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[f"{name}_version"] = "unavailable"
    return result


def _as_frame(value, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise ValueError(f"CausalMan returned an invalid {name}") from exc


def _observational_mask(int_table: pd.DataFrame, n: int) -> tuple[np.ndarray, dict]:
    if int_table is None or int_table.empty:
        return np.ones(n, dtype=bool), {"intervention_columns": [],
                                        "intervention_rows": 0}
    if len(int_table) != n:
        raise ValueError(f"int_table has {len(int_table)} rows but obs_df has {n}")
    flags = np.zeros(n, dtype=bool)
    columns = []
    for column in int_table.columns:
        values = int_table[column]
        if pd.api.types.is_bool_dtype(values):
            active = values.fillna(False).to_numpy(bool)
        elif pd.api.types.is_numeric_dtype(values):
            active = values.fillna(0).to_numpy(float) != 0
        else:
            text = values.fillna("").astype(str).str.strip()
            active = (~text.isin(("", "0", "none", "None", "nan"))).to_numpy(bool)
        if active.any():
            columns.append(str(column))
            flags |= active
    return ~flags, {"intervention_columns": columns,
                    "intervention_rows": int(flags.sum())}


def _continuous_columns(frame: pd.DataFrame, mask: np.ndarray) -> tuple[list[str], dict[str, str]]:
    keep, dropped = [], {}
    for column in frame.columns:
        # ``batch_ID_num`` is a numeric encoding of a batch label, not a
        # measured process variable.  Keep it in the original cached data and
        # manifest, but never expose it as a continuous discovery feature.
        if str(column).strip().lower() == "batch_id_num":
            dropped[str(column)] = "categorical_identifier"
            continue
        values = frame.loc[mask, column]
        if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(values):
            dropped[str(column)] = "non_numeric_or_boolean"
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            dropped[str(column)] = "nonfinite"
            continue
        if numeric.nunique(dropna=True) <= 1:
            dropped[str(column)] = "constant"
            continue
        # CausalMan contains switches and numeric-coded categorical controls.
        # A finite integer column is not silently treated as a sensor.
        if pd.api.types.is_integer_dtype(values):
            dropped[str(column)] = "integer_or_discrete"
            continue
        if numeric.nunique(dropna=True) <= min(10, max(2, len(values) // 20)):
            dropped[str(column)] = "low_cardinality_discrete"
            continue
        keep.append(str(column))
    return keep, dropped


def _graph_nodes(graph) -> list:
    if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiDiGraph)):
        graph = nx.DiGraph(graph)
    return list(graph.nodes)


def _adjacency(graph, nodes: list) -> np.ndarray:
    return nx.to_numpy_array(graph, nodelist=nodes, dtype=np.uint8)


def full_oracle_no_treks(graph: nx.DiGraph, selected: list) -> list[tuple[str, str]]:
    """No-treks from the complete DAG, before observable projection."""
    nodes = list(graph.nodes)
    reach = nx.transitive_closure_dag(graph) if nx.is_directed_acyclic_graph(graph) else nx.transitive_closure(graph)
    ancestors = {node: set(reach.predecessors(node)) | {node} for node in nodes}
    return [(str(a), str(b)) for i, a in enumerate(selected) for b in selected[i + 1:]
            if ancestors[a].isdisjoint(ancestors[b])]


def latent_projection(graph: nx.DiGraph, observed: list) -> dict:
    """Return directed/bidirected edge lists using the official projection API."""
    try:
        from causalman.graph_projections import get_latent_projection_single
        graph = graph.copy()
        observed_set = set(observed)
        for node in graph.nodes:
            graph.nodes[node]["Observable"] = node in observed_set
        projected = get_latent_projection_single(graph)
    except Exception as exc:
        raise RuntimeError(
            "CausalMan latent projection failed; the upstream graph projection "
            "API is required and no directed-only fallback is valid") from exc
    directed, bidirected = [], []
    if isinstance(projected, nx.MultiDiGraph):
        for u, v, data in projected.edges(data=True):
            kind = data.get("edge_type", data.get("type", "directed"))
            if kind == "bidirected":
                if str(u) < str(v):
                    bidirected.append((str(u), str(v)))
            else:
                directed.append((str(u), str(v)))
    else:
        directed = [(str(u), str(v)) for u, v in projected.edges()]
    return {"directed": sorted(set(directed)),
            "bidirected": sorted(set(bidirected))}


def _sample(CausalMan, variant: str, seed: int, target_rows: int,
            run_path: Path):
    run_path.mkdir(parents=True, exist_ok=True)
    common = {"parallelize": False, "save_path": str(run_path),
              "debug_mode": False}
    simulator = CausalMan(name=variant, seed=int(seed),
                           batch_multiplier=1, **common)
    # CausalMan 0.1.0 added exact n_samples support and returns six values;
    # older upstream revisions documented a four-value return.  Prefer the
    # exact API to avoid generating and then discarding a large batch.
    try:
        return simulator.sample(n_samples=target_rows), 1
    except TypeError as exc:
        if "n_samples" not in str(exc):
            raise
        result = simulator.sample()
        return result, 1


def _unpack_sample(result):
    """Normalize documented four-value and CausalMan 0.1 six-value APIs."""
    if not isinstance(result, (tuple, list)):
        raise ValueError("CausalMan sample() did not return a tuple/list")
    if len(result) == 4:
        obs, int_table, path_df, causal_dag = result
        full = obs
        api = "four_value"
    elif len(result) == 6:
        # CausalMan 0.1.0 returns (observable, MAG, ADMG, full, DAG,
        # intervention_table). It no longer returns path_df directly.
        obs, _mag, _admg, full, causal_dag, int_table = result
        path_df = pd.DataFrame()
        api = "causalman_0.1_six_value"
    else:
        raise ValueError(
            f"unsupported CausalMan sample() return arity {len(result)}; "
            "expected the documented four-value or pinned six-value API")
    return (_as_frame(obs, "obs_df"), _as_frame(int_table, "int_table"),
            _as_frame(path_df, "path_df"), _as_frame(full, "full_observable_df"),
            causal_dag, api)


def prepare_one(cache: Path, variant: str, seed: int, target_rows: int = 5000) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    CausalMan = _require_upstream()
    result, multiplier = _sample(
        CausalMan, variant, seed, target_rows,
        cache / "upstream_runs" / variant / f"seed_{seed}")
    obs, int_table, path_df, full_observable, causal_dag, api = _unpack_sample(result)
    obs.columns = [str(column) for column in obs.columns]
    graph = causal_dag if isinstance(causal_dag, nx.DiGraph) else nx.DiGraph(causal_dag)
    # Portable artifacts use string node IDs; relabel only after sampling so
    # the simulator object itself remains untouched.
    graph = nx.relabel_nodes(graph, {node: str(node) for node in graph.nodes}, copy=True)
    nodes = _graph_nodes(graph)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("CausalMan causal_dag is not acyclic")
    mask, intervention = _observational_mask(int_table, len(obs))
    obs = obs.loc[mask].reset_index(drop=True)
    path_df = path_df.loc[mask].reset_index(drop=True) if len(path_df) == len(mask) else path_df
    observable = [node for node in nodes if bool(graph.nodes[node].get("Observable", graph.nodes[node].get("observable", False)))]
    selected = [str(c) for c in obs.columns if str(c) in set(observable)]
    if not selected:
        selected, dropped = _continuous_columns(obs, np.ones(len(obs), dtype=bool))
    else:
        selected, dropped = _continuous_columns(obs.loc[:, selected], np.ones(len(obs), dtype=bool))
    missing = [node for node in selected if node not in obs.columns]
    if missing:
        raise ValueError(f"selected CausalMan nodes missing from obs_df: {missing}")
    values = obs.loc[:, selected].to_numpy(float)
    if len(values) < target_rows:
        raise ValueError(f"only {len(values)} observational rows remain; {target_rows} required")
    values = values[:target_rows]
    projection = latent_projection(graph, selected)
    pairs = full_oracle_no_treks(graph, selected)
    corr = np.corrcoef(values, rowvar=False) if len(selected) > 1 else np.zeros((len(selected), len(selected)))
    selected_index = {name: i for i, name in enumerate(selected)}
    oracle_corr = [abs(float(corr[selected_index[a], selected_index[b]]))
                   for a, b in pairs if a in selected_index and b in selected_index]
    components = list(nx.connected_components(nx.Graph(graph)))
    source_nodes = {node for node in nodes if graph.in_degree(node) == 0}
    pair_count = len(selected) * (len(selected) - 1) // 2
    source_source = sum(a in source_nodes and b in source_nodes for a, b in pairs)
    source_incident = sum(a in source_nodes or b in source_nodes for a, b in pairs)
    non_source = sum(a not in source_nodes and b not in source_nodes for a, b in pairs)
    selected_component = {}
    for component_id, component in enumerate(components):
        for node in component:
            selected_component[str(node)] = component_id
    disconnected = sum(selected_component.get(a) != selected_component.get(b)
                       for a, b in pairs)
    variant_dir = cache / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    prefix = variant_dir / f"seed_{seed}"
    np.save(prefix.with_suffix(".npy"), values)
    np.save(prefix.with_name(prefix.name + "_full_adjacency.npy"), _adjacency(graph, nodes))
    pd.DataFrame(path_df).to_csv(prefix.with_name(prefix.name + "_path.csv"), index=False)
    (prefix.with_name(prefix.name + "_graph.json")).write_text(json.dumps({
        "nodes": [str(v) for v in nodes], "selected_nodes": selected,
        "full_edges": [(str(u), str(v)) for u, v in graph.edges()],
        "projected": projection, "oracle_no_treks": pairs,
    }, indent=2) + "\n")
    manifest = {
        "dataset": variant, "seed": int(seed), "target_rows": target_rows,
        "observational_rows": int(len(obs)), "stored_rows": int(len(values)),
        "nodes_full": len(nodes), "edges_full": int(graph.number_of_edges()),
        "nodes_observed_continuous": len(selected), "selected_nodes": selected,
        "dropped_variables": dropped, "intervention": intervention,
        "batch_multiplier": multiplier, "projection": projection,
        "sample_api": api, "path_metadata_available": not path_df.empty,
        "oracle_no_trek_pairs": len(pairs), "total_pairs": pair_count,
        "source_source_no_trek_pairs": int(source_source),
        "source_incident_no_trek_pairs": int(source_incident),
        "non_source_no_trek_pairs": int(non_source),
        "disconnected_section_pairs": int(disconnected),
        "remaining_nontrivial_pairs": int(non_source),
        "oracle_pair_abs_corr_mean": float(np.mean(oracle_corr)) if oracle_corr else 0.0,
        "oracle_pair_abs_corr_max": float(np.max(oracle_corr)) if oracle_corr else 0.0,
        "data_checksum": sha256_bytes(values.tobytes()),
        "full_graph_checksum": sha256_bytes(_adjacency(graph, nodes).tobytes()),
        "package": package_metadata(), "master_seed": MASTER_SEED,
        "eligible_for_oracle": bool(len(pairs) > 0 and not intervention["intervention_rows"]),
        "eligibility_reason": ("ok" if pairs and not intervention["intervention_rows"]
                               else "interventions_or_no_valid_pairs"),
    }
    (prefix.with_name(prefix.name + "_manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_cache(cache: Path, variants: list[str], seeds: list[int], target_rows: int = 5000):
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for variant in variants:
        for seed in seeds:
            path = cache / variant / f"seed_{seed}_manifest.json"
            if path.exists():
                cached = json.loads(path.read_text())
                # Rebuild caches made before batch_ID_num was classified as a
                # categorical identifier.  The original upstream data remain
                # untouched; only the derived analysis matrix is refreshed.
                if "batch_ID_num" not in cached.get("selected_nodes", []):
                    rows.append(cached)
                    continue
            try:
                rows.append(prepare_one(cache, variant, seed, target_rows))
            except Exception as exc:
                errors.append({"variant": variant, "seed": seed,
                               "error": repr(exc)})
    with _manifest_lock(cache):
        old = {}
        manifest_path = cache / "manifest.json"
        if manifest_path.exists():
            try:
                old = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                old = {}
        old_rows = old.get("preflight_rows", [])
        merged = {(r.get("dataset"), r.get("seed")): r
                  for r in old_rows if isinstance(r, dict)}
        merged.update({(r.get("dataset"), r.get("seed")): r for r in rows})
        merged_errors = {(e.get("variant"), e.get("seed")): e
                         for e in old.get("errors", []) if isinstance(e, dict)}
        merged_errors.update({(e.get("variant"), e.get("seed")): e for e in errors})
        # A later successful retry supersedes a historical failure for the
        # same variant/seed.  Keep the root manifest as current status, while
        # the per-seed manifests remain the audit trail.
        for key in merged:
            merged_errors.pop(key, None)
        payload = {"variants": sorted(set(old.get("variants", [])) | set(variants)),
                   "seeds": sorted(set(old.get("seeds", [])) | set(seeds)),
                   "target_rows": target_rows, "completed": len(merged),
                   "errors": list(merged_errors.values()),
                   "preflight_rows": list(merged.values()),
                   "package": package_metadata(), "source_url": SOURCE_URL,
                   "paper_url": PAPER_URL}
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
        if merged:
            pd.DataFrame(merged.values()).to_csv(cache / "preflight.csv", index=False)
    return payload


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--cache", type=Path, required=True)
    q.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    q.add_argument("--seeds", nargs="+", type=int, required=True)
    q.add_argument("--target-rows", type=int, default=5000)
    q.add_argument("--json", action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        try:
            result = prepare_cache(args.cache, args.variants, args.seeds, args.target_rows)
        except RuntimeError as exc:
            result = {"completed": 0, "errors": [{"error": str(exc)}],
                      "source_url": SOURCE_URL}
            print(json.dumps(result, indent=2))
            raise SystemExit(2)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
