"""causalAssembly preparation, graphical knowledge, and practical screen.

The upstream package is optional.  Ordinary protocol imports do not import it;
preparation fails with an actionable message when causalAssembly/R/rpy2 are
not installed.  Structure-learning runs consume only the portable cache made
by :func:`prepare_cache`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


SOURCE_URL = "https://github.com/boschresearch/causalAssembly"
STATIC_DATA_URL = ("https://raw.githubusercontent.com/boschresearch/causalAssembly/"
                   "main/data/data_sets/n_500_synthdata/assembly_line_500.csv")
STATIC_TRUTH_URL = ("https://raw.githubusercontent.com/boschresearch/causalAssembly/"
                    "main/data/ground_truth/ground_truth.json")
PAPER_URL = "https://proceedings.mlr.press/v236/gobler24a.html"
N_NODES = 98
DISCOVERY_SIZES = (500, 2000, 5000)
REFERENCE_SIZE = 5000


class _NormalizingChoiceRNG:
    """Adapter for upstream DRF weights with harmless roundoff drift."""
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def choice(self, a, size=None, replace=True, p=None):
        if p is not None:
            p = np.asarray(p, dtype=float)
            total = float(p.sum())
            if not np.isfinite(total) or total <= 0:
                raise ValueError("DRF returned invalid sampling probabilities")
            p = p / total
        return self.rng.choice(a, size=size, replace=replace, p=p)


def derive_seed(namespace: str, *parts: object, master: int = 20260917) -> int:
    payload = json.dumps([namespace, master, *parts], sort_keys=False,
                         separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _require_upstream():
    try:
        # Homebrew R can be newer than the rpy2 wheel's API-mode build.  ABI
        # mode uses the active R installation and is the supported fallback.
        os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
        from causalAssembly.drf_fitting import DRF
        from causalAssembly.models_dag import ProductionLineGraph
    except ImportError as exc:
        raise RuntimeError(
            "causalAssembly preparation requires the official causalAssembly "
            "package plus its DRF dependencies (R and rpy2). Install the "
            "upstream package in the preparation environment; no substitute "
            "generator is used."
        ) from exc
    return ProductionLineGraph, DRF


def package_metadata() -> dict[str, str]:
    result = {"source_url": SOURCE_URL, "paper_url": PAPER_URL}
    for name in ("causalAssembly", "rpy2", "numpy", "pandas"):
        try:
            result[f"{name}_version"] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[f"{name}_version"] = "unavailable"
    return result


def _graph_and_data():
    ProductionLineGraph, _ = _require_upstream()
    data = ProductionLineGraph.get_data()
    truth = ProductionLineGraph.get_ground_truth()
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    graph = getattr(truth, "graph", truth)
    if not isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)
    nodes = list(graph.nodes)
    if len(nodes) != N_NODES or len(set(nodes)) != N_NODES:
        raise ValueError(f"causalAssembly ground truth has {len(nodes)} nodes, expected 98")
    missing = [node for node in nodes if node not in data.columns]
    if missing:
        raise ValueError(f"causalAssembly data is missing graph columns: {missing[:5]}")
    data = data.loc[:, nodes].copy()
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.uint8)
    np.fill_diagonal(adjacency, 0)
    return data, adjacency, nodes


def _static_graph_and_data(data_csv: Path, truth_json: Path):
    """Load the official upstream fixed n=500 causalAssembly release."""
    data = pd.read_csv(data_csv)
    truth_payload = json.loads(truth_json.read_text())
    nodes = [entry["id"] for entry in truth_payload["nodes"]]
    if data.shape != (500, len(nodes)):
        raise ValueError(
            f"static causalAssembly data has shape {data.shape}, expected "
            f"(500, {len(nodes)})")
    if list(data.columns) != nodes:
        missing = sorted(set(nodes) - set(data.columns))
        extra = sorted(set(data.columns) - set(nodes))
        raise ValueError(f"static columns do not match ground truth; "
                         f"missing={missing[:3]}, extra={extra[:3]}")
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.uint8)
    index = {node: i for i, node in enumerate(nodes)}
    for source, children in enumerate(truth_payload["adjacency"]):
        for child in children:
            adjacency[source, index[child["id"]]] = 1
    np.fill_diagonal(adjacency, 0)
    if not nx.is_directed_acyclic_graph(nx.DiGraph(adjacency)):
        raise ValueError("static causalAssembly ground truth is cyclic")
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("static causalAssembly data contains nonfinite values")
    return data, adjacency, nodes


def prepare_static_cache(cache_dir: Path, seeds: list[int], data_csv: Path,
                         truth_json: Path) -> dict:
    """Create seed-compatible cache artifacts without DRF regeneration.

    The upstream release is one fixed 500-row dataset.  Requested seeds are
    retained as protocol labels, but intentionally point to identical data;
    the manifest records this so they cannot be mistaken for independent
    generated replicates.
    """
    data, truth, nodes = _static_graph_and_data(data_csv, truth_json)
    values = data.to_numpy(dtype=float)
    pairs = oracle_no_treks(truth)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / "ground_truth.npz", truth=truth)
    manifest = {
        "dataset": "causalassembly_upstream_static_n500",
        "node_count": len(nodes), "nodes": nodes, "truth_edges": int(truth.sum()),
        "oracle_no_trek_pairs": len(pairs), "seeds": list(seeds),
        "source_url": SOURCE_URL, "data_url": STATIC_DATA_URL,
        "truth_url": STATIC_TRUTH_URL, "data_csv": str(data_csv),
        "truth_json": str(truth_json), "fixed_dataset": True,
        "independent_seed_replicates": False,
        "data_checksum": hashlib.sha256(values.tobytes()).hexdigest(),
        "truth_checksum": hashlib.sha256(truth.tobytes()).hexdigest(),
        "completed_seeds": [],
    }
    for seed in seeds:
        np.savez_compressed(cache_dir / f"seed_{seed}.npz",
                            reference=values, discovery=values)
        manifest["completed_seeds"].append(seed)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def oracle_no_treks(truth: np.ndarray) -> list[tuple[int, int]]:
    """Complete unordered graphical no-trek set, with reflexive ancestors."""
    truth = np.asarray(truth, dtype=bool)
    reach = truth.copy()
    np.fill_diagonal(reach, True)
    for k in range(len(truth)):
        reach |= reach[:, [k]] & reach[[k], :]
    ancestors = reach.T
    return [(i, j) for i in range(len(truth)) for j in range(i + 1, len(truth))
            if not np.any(ancestors[i] & ancestors[j])]


def nested_pairs(pairs, fraction: float, seed: int) -> list[tuple[int, int]]:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0, 1]")
    pairs = [tuple(map(int, p)) for p in pairs]
    if fraction == 1:
        return pairs
    count = int(round(fraction * len(pairs)))
    order = np.random.default_rng(seed).permutation(len(pairs))
    return [pairs[int(i)] for i in order[:count]]


def standardize_discovery(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("causalAssembly discovery data contains nonfinite values")
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    if np.any(scale <= 1e-12):
        bad = np.flatnonzero(scale <= 1e-12).tolist()
        raise ValueError(f"near-constant causalAssembly columns: {bad}")
    return (x - mean) / scale, mean, scale


def empirical_copula(x: np.ndarray) -> np.ndarray:
    """Average-rank copula scores; ties receive their average rank."""
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("reference data contains nonfinite values")
    if np.any(np.ptp(x, axis=0) <= 1e-12):
        raise ValueError("constant reference columns cannot be screened")
    n = len(x)
    return np.column_stack([(rankdata(x[:, j], method="average") - .5) / n
                            for j in range(x.shape[1])])


def _features(u: np.ndarray, harmonics: int = 4) -> np.ndarray:
    # Deterministic bounded random-free characteristic features.  This is a
    # controlled O(n p^2 k^2) practical nonlinear screen, not Pearson-only.
    angles = 2 * np.pi * np.arange(1, harmonics + 1, dtype=float)
    return np.concatenate([np.sin(u[..., None] * angles),
                           np.cos(u[..., None] * angles)], axis=-1)


def nonlinear_screen(reference: np.ndarray, *, candidate_cap: int = 750,
                     blocks: int = 5, permutations: int = 99,
                     null_quantile: float = .10, effect_margin: float = .02,
                     seed: int = 0) -> tuple[list[tuple[int, int]], pd.DataFrame, dict]:
    """Frozen practical-independence screen with deterministic diagnostics.

    The primary score is a characteristic-feature cross-covariance norm.  It
    is intentionally labelled a practical screen: non-rejection is not proof
    of graphical no-trek status.  Permutation calibration is performed only
    for the capped low-score candidate set.
    """
    u = empirical_copula(reference)
    n, d = u.shape
    feat = _features(u)
    centered = feat - feat.mean(axis=0, keepdims=True)
    scores = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i + 1, d):
            cross = centered[:, i, :].T @ centered[:, j, :] / max(n - 1, 1)
            scores[i, j] = scores[j, i] = float(np.linalg.norm(cross, "fro"))
    off = scores[np.triu_indices(d, 1)]
    envelope = float(np.quantile(off, null_quantile)) if len(off) else 0.0
    candidates = [(i, j) for i in range(d) for j in range(i + 1, d)
                  if scores[i, j] <= envelope]
    candidates.sort(key=lambda p: (scores[p], p))
    candidates = candidates[:candidate_cap]
    rng = np.random.default_rng(seed)
    rows = []
    selected = []
    block_ids = np.array_split(np.arange(n), max(1, blocks))
    for i, j in candidates:
        block_scores = []
        for idx in block_ids:
            a, b = centered[idx, i], centered[idx, j]
            cross = a.T @ b / max(len(idx) - 1, 1)
            block_scores.append(float(np.linalg.norm(cross, "fro")))
        null = []
        for _ in range(permutations):
            perm = rng.permutation(n)
            cross = centered[:, i].T @ centered[perm, j] / max(n - 1, 1)
            null.append(float(np.linalg.norm(cross, "fro")))
        null_q = float(np.quantile(null, .95)) if null else 0.0
        observed = float(max(block_scores))
        # A pair is retained only when every block is small and below the
        # calibrated null envelope plus the explicit effect margin.
        stable = bool(all(v <= null_q + effect_margin for v in block_scores))
        if stable:
            selected.append((i, j))
        rows.append({"i": i, "j": j, "score": scores[i, j],
                     "block_max": observed, "null_q95": null_q,
                     "stable_near_null": stable, "blocks": len(block_ids),
                     "permutations": permutations})
    diagnostics = {"n_reference": n, "d": d, "candidate_cap": candidate_cap,
                   "candidate_count": len(candidates), "selected_count": len(selected),
                   "null_quantile": null_quantile, "effect_margin": effect_margin,
                   "screen": "copula_characteristic_features_permutation"}
    return selected, pd.DataFrame(rows), diagnostics


def prepare_cache(cache_dir: Path, seeds: list[int], *, fit_seed: int = 20260917,
                  sizes: tuple[int, ...] = DISCOVERY_SIZES,
                  reference_size: int = REFERENCE_SIZE,
                  num_trees: int = 2000,
                  num_threads: int | None = None) -> dict:
    """Fit official DRFs once and write portable paired sample artifacts."""
    data, truth, nodes = _graph_and_data()
    _, DRF = _require_upstream()
    graph_obj = __import__("causalAssembly.models_dag", fromlist=["ProductionLineGraph"]
                           ).ProductionLineGraph.get_ground_truth()
    # Fit and sample one node at a time.  The upstream convenience function
    # retains all fitted forests in graph_obj.drf; for the 98-node line that
    # can exceed the workstation memory even though the official forest
    # configuration itself is unchanged.
    cache_dir.mkdir(parents=True, exist_ok=True)
    truth_pairs = oracle_no_treks(truth)
    manifest = {"dataset": "causalassembly_full", "node_count": len(nodes),
                "nodes": nodes, "truth_edges": int(truth.sum()),
                "oracle_no_trek_pairs": len(truth_pairs),
                "package": package_metadata(), "fit_seed": fit_seed,
                "reference_size": reference_size, "discovery_sizes": list(sizes),
                "drf_num_trees": num_trees,
                "drf_num_threads": num_threads,
                "seeds": list(seeds), "source_url": SOURCE_URL,
                "sampling_seeds": {
                    str(seed): {
                        "reference": derive_seed("reference", seed),
                        "discovery": derive_seed("discovery", seed),
                    } for seed in seeds},
                "truth_checksum": hashlib.sha256(truth.tobytes()).hexdigest()}
    np.savez_compressed(cache_dir / "ground_truth.npz", truth=truth)
    manifest_path = cache_dir / "manifest.json"
    manifest["completed_seeds"] = []
    existing = set()
    max_size = max(sizes)
    for seed in seeds:
        seed_path = cache_dir / f"seed_{seed}.npz"
        if not seed_path.exists():
            continue
        try:
            cached = np.load(seed_path)
            valid = (cached["reference"].shape == (reference_size, len(nodes))
                     and cached["discovery"].shape == (max_size, len(nodes))
                     and np.isfinite(cached["reference"]).all()
                     and np.isfinite(cached["discovery"]).all())
            cached.close()
        except Exception:
            valid = False
        if valid:
            existing.add(seed)
            manifest["completed_seeds"].append(seed)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    pending = [seed for seed in seeds if seed not in existing]
    if not pending:
        return manifest
    from scipy.stats import gaussian_kde
    generated = {}
    for seed in pending:
        generated[seed] = {
            "reference": np.empty((reference_size, len(nodes)), dtype=float),
            "discovery": np.empty((max_size, len(nodes)), dtype=float),
        }
    reference_rngs = {seed: np.random.default_rng(derive_seed("reference", seed))
                      for seed in pending}
    discovery_rngs = {seed: np.random.default_rng(derive_seed("discovery", seed))
                      for seed in pending}
    for node in graph_obj.causal_order:
        node_index = nodes.index(node)
        parents = graph_obj.parents(of_node=node)
        if not parents:
            kde = gaussian_kde(data[node].to_numpy())
            for seed in pending:
                generated[seed]["reference"][:, node_index] = kde.resample(
                    reference_size, seed=reference_rngs[seed])[0]
                generated[seed]["discovery"][:, node_index] = kde.resample(
                    max_size, seed=discovery_rngs[seed])[0]
            del kde
            continue
        fit_params = {"min_node_size": 15, "num_trees": num_trees,
                      "splitting_rule": "FourierMMD",
                      "seed": derive_seed("drf-fit", fit_seed, node_index)}
        if num_threads is not None:
            fit_params["num_threads"] = num_threads
        forest = DRF(**fit_params)
        forest.fit(data[parents], data[node])
        for seed in pending:
            ref_parents = pd.DataFrame(generated[seed]["reference"][:,
                                      [nodes.index(p) for p in parents]],
                                       columns=parents)
            disc_parents = pd.DataFrame(generated[seed]["discovery"][:,
                                       [nodes.index(p) for p in parents]],
                                        columns=parents)
            generated[seed]["reference"][:, node_index] = forest.produce_sample(
                ref_parents, _NormalizingChoiceRNG(reference_rngs[seed]))
            generated[seed]["discovery"][:, node_index] = forest.produce_sample(
                disc_parents, _NormalizingChoiceRNG(discovery_rngs[seed]))
        del forest
    for seed in pending:
        reference = generated[seed]["reference"]
        discovery = generated[seed]["discovery"]
        np.savez_compressed(cache_dir / f"seed_{seed}.npz",
                            reference=reference, discovery=discovery)
        manifest["completed_seeds"].append(seed)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--seeds", type=int, nargs="+", required=True)
    prep.add_argument("--fit-seed", type=int, default=20260917)
    prep.add_argument("--drf-num-trees", type=int, default=2000)
    prep.add_argument("--drf-num-threads", type=int, default=None,
                      help="threads used inside each official R DRF fit")
    static = sub.add_parser("prepare-static")
    static.add_argument("--cache", type=Path, required=True)
    static.add_argument("--seeds", type=int, nargs="+", required=True)
    static.add_argument("--data-csv", type=Path, required=True)
    static.add_argument("--truth-json", type=Path, required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--cache", type=Path, required=True)
    screen.add_argument("--seed", type=int, required=True)
    screen.add_argument("--candidate-cap", type=int, default=750)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--cache", type=Path, required=True)
    dry.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_cache(args.cache, args.seeds,
                                        fit_seed=args.fit_seed,
                                        num_trees=args.drf_num_trees,
                                        num_threads=args.drf_num_threads), indent=2))
    elif args.command == "prepare-static":
        print(json.dumps(prepare_static_cache(
            args.cache, args.seeds, args.data_csv, args.truth_json), indent=2))
    elif args.command == "screen":
        artifact = np.load(args.cache / f"seed_{args.seed}.npz")
        selected, diagnostics, summary = nonlinear_screen(
            artifact["reference"], candidate_cap=args.candidate_cap,
            seed=derive_seed("screen", args.seed))
        diagnostics.to_csv(args.cache / f"screen_{args.seed}.csv", index=False)
        (args.cache / f"screen_{args.seed}.json").write_text(
            json.dumps({**summary, "pairs": [list(p) for p in selected]}, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    else:
        manifest_path = args.cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        print(json.dumps({
            "dataset": "causalassembly_full", "cache": str(args.cache),
            "cache_exists": manifest is not None,
            "seeds": args.seeds, "n_values": list(DISCOVERY_SIZES),
            "reference_size": REFERENCE_SIZE, "q_values": [.10, .25, .50, 1.0],
            "methods": ["flop", "flop-nt-standard", "flop-nt-edge-mask",
                        "flop-nt-post", "dagma", "dagma-pstrek", "dagma-nt-post"],
            "drf_preparation": "requires official causalAssembly, R, and rpy2",
            "expected_reference_screens": len(args.seeds),
        }, indent=2))


if __name__ == "__main__":
    main()
