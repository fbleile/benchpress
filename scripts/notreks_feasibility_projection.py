#!/usr/bin/env python3
"""Post-hoc hard-NOTREKS feasibility projection for FLOP and DAGMA outputs.

This is a diagnostic projection, not a structure-learning method. It never
changes the farm outputs. For each completed FLOP or DAGMA DAG, it repeatedly
removes the edge carrying the largest number of currently witnessed forbidden
treks. Exact ancestry is recomputed after every deletion. Ties are resolved by
the number of witnesses removed and then by edge indices.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.knowledge import (
    load_sidecar, named_pairs_to_indices,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.tools.local_smoke import (
    _python_metrics,
)


def reachability(graph: np.ndarray) -> np.ndarray:
    reach = np.asarray(graph, dtype=bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(len(reach)):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def witnesses(graph: np.ndarray, pairs: list[tuple[int, int]]) -> list[tuple[int, int, list[int]]]:
    reach = reachability(graph)
    return [(i, j, [a for a in range(len(graph)) if reach[a, i] and reach[a, j]])
            for i, j in pairs if any(reach[a, i] and reach[a, j] for a in range(len(graph)))]


def project(graph: np.ndarray, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, dict]:
    graph = (np.asarray(graph) != 0).astype(np.uint8)
    np.fill_diagonal(graph, 0)
    initial = len(witnesses(graph, pairs))
    removed: list[tuple[int, int]] = []
    while True:
        reach = reachability(graph)
        current = [(i, j, [a for a in range(len(graph))
                            if reach[a, i] and reach[a, j]])
                   for i, j in pairs
                   if any(reach[a, i] and reach[a, j] for a in range(len(graph)))]
        if not current:
            break
        edges = np.argwhere(graph != 0)
        if not len(edges):
            break
        scored = []
        for source, target in edges:
            # Weight an edge by the number of witness treks whose path can
            # pass through it. This is the intentionally simple projection
            # rule; it does not evaluate a BIC objective or search alternatives.
            attached = 0
            for left, right, ancestors in current:
                for ancestor in ancestors:
                    attached += int(reach[ancestor, source] and
                                    (reach[target, left] or reach[target, right]))
            scored.append((attached, -int(source), -int(target), int(source), int(target)))
        _, _, _, source, target = max(scored)
        graph[source, target] = 0
        removed.append((source, target))
    return graph, {
        "initial_violations": initial,
        "final_violations": len(witnesses(graph, pairs)),
        "edges_before": int(np.asarray(graph).sum()) + len(removed),
        "edges_after": int(graph.sum()),
        "edges_removed": len(removed),
        "removed_edges": removed,
    }


def _find_one(paths, pattern):
    return next((p for p in paths if pattern in str(p)), None)


def run(run_dir: Path, output_dir: Path, limit: int | None = None) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_files = list(run_dir.rglob("workspace/results/adjmat_estimate/**/adjmat.csv"))
    rows = []
    indexed: dict[Path, tuple[list[Path], list[Path]]] = {}
    processed = 0
    for raw in raw_files:
        text = str(raw)
        method_match = re.search(r"/algorithm=/([^/]+)/", text)
        seed_match = re.search(r"/seed=(\d+)/adjmat\.csv$", text)
        if not method_match or not seed_match or method_match.group(1) not in {"flop", "dagma"}:
            continue
        method, seed = method_match.group(1), int(seed_match.group(1))
        task = raw.parts[raw.parts.index("tasks") + 1]
        parts = list(raw.parts)
        workspace_index = parts.index("workspace")
        results = Path(*parts[:workspace_index + 2])
        if results not in indexed:
            indexed[results] = (list(results.rglob("*.csv")), list(results.rglob("*.json")))
        csv_files, json_files = indexed[results]
        data = next((p for p in csv_files if "/data/" in str(p) and f"seed={seed}.csv" in str(p)), None)
        sidecar = next((p for p in json_files if "oracle_knowledge" in str(p) and f"seed={seed}.json" in str(p)), None)
        truth = next((p for p in csv_files if "/adjmat/" in str(p) and "/adjmat_estimate/" not in str(p) and f"seed={seed}.csv" in str(p)), None)
        if not data or not sidecar or not truth:
            continue
        matrix = pd.read_csv(raw).to_numpy(dtype=np.uint8)
        names = list(pd.read_csv(data, nrows=0).columns)
        pairs = named_pairs_to_indices(load_sidecar(sidecar, names), names)
        projected, diagnostics = project(matrix, pairs)
        projected_path = output_dir / f"{task}__{method}__seed{seed}.csv"
        pd.DataFrame(projected, columns=names).to_csv(projected_path, index=False)
        metrics = _python_metrics(truth, projected_path)
        row = {"task_id": task, "method": method, "seed": seed,
               **diagnostics, **metrics, "output_graph_type": "projected_dag"}
        rows.append(row)
        processed += 1
        if limit is not None and processed >= limit:
            break
        (output_dir / f"{task}__{method}__seed{seed}.json").write_text(json.dumps(row, indent=2, default=int) + "\n")
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_run.csv", index=False)
    if len(frame):
        frame.groupby("method", dropna=False).agg(
            runs=("task_id", "count"), mean_shd=("SHD_pattern", "mean"),
            mean_f1=("F1_pattern", "mean"), mean_removed=("edges_removed", "mean"),
            mean_initial_violations=("initial_violations", "mean"),
            maximum_final_violations=("final_violations", "max"),
        ).reset_index().to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "REPORT.md").write_text(
        "# Post-hoc NOTREKS feasibility projection\n\n"
        "This diagnostic starts from completed FLOP and DAGMA graphs and deletes "
        "edges carrying the largest number of exact forbidden-trek witnesses. "
        "It is not an optimizer and does not alter primary benchmark outputs.\n\n"
        + (frame.groupby("method")["final_violations"].max().to_string() if len(frame) else "No completed source graphs found.")
        + "\n")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most this many completed source graphs")
    args = parser.parse_args()
    frame = run(args.run_dir, args.output_dir, args.limit)
    print(f"projected {len(frame)} completed FLOP/DAGMA graphs")
    print((args.output_dir / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
