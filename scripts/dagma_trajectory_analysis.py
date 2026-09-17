#!/usr/bin/env python3
"""Plot diagnostic DAGMA trajectories saved by the systematic benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    production_candidate_graph,
)


def _mds3(values: np.ndarray) -> np.ndarray:
    """Classical metric MDS from Frobenius distances between matrices."""
    differences = values[:, None, :] - values[None, :, :]
    distances_sq = np.einsum("ijk,ijk->ij", differences, differences)
    n = len(values)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ distances_sq @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    coordinates = []
    for index in order:
        if eigenvalues[index] > 0 and len(coordinates) < 3:
            coordinates.append(
                eigenvectors[:, index] * np.sqrt(eigenvalues[index]))
    while len(coordinates) < 3:
        coordinates.append(np.zeros(n))
    return np.column_stack(coordinates)


def _read(path: Path):
    payload = np.load(path, allow_pickle=False)
    X = np.asarray(payload["X"], dtype=float)
    truth = np.asarray(payload["truth"], dtype=np.uint8)
    pairs = [tuple(map(int, row)) for row in np.asarray(payload["pairs"])]
    trajectories = []
    times = []
    for key in sorted(
            (key for key in payload.files
             if key.startswith("restart_") and not key.endswith("_times")),
            key=lambda key: int(key.split("_")[1])):
        trajectories.append(np.asarray(payload[key], dtype=float))
        time_key = f"{key}_times"
        times.append(np.asarray(payload[time_key], dtype=float)
                     if time_key in payload.files
                     else np.arange(len(trajectories[-1]), dtype=float))
    return X, truth, pairs, trajectories, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-fraction", type=float, default=1.0,
                        help="retain this fraction of best-BIC checkpoints per restart (default: all)")
    args = parser.parse_args()
    if not 0.0 < args.keep_fraction <= 1.0:
        raise SystemExit("--keep-fraction must lie in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted((args.input_dir / "dagma_trajectories").glob("*.npz"))
    if not files:
        raise SystemExit("no DAGMA trajectory files found")

    records = []
    vectors = []
    for path in files:
        X, truth, pairs, trajectories, trajectory_times = _read(path)
        truth_bic = float(gaussian_bic(X, truth, lambda_bic=2.0)[0])
        method = path.stem.split("_", 2)[-1]
        for restart, matrices in enumerate(trajectories):
            for stage, matrix in enumerate(matrices):
                vectors.append(matrix.ravel())
                try:
                    candidate, diagnostic = production_candidate_graph(
                        matrix, pairs if method == "dagma_notreks" else [],
                        screening_floor=0.01,
                        notreks_active=(method == "dagma_notreks"))
                    bic = float(gaussian_bic(
                        X, candidate, lambda_bic=2.0)[0])
                    gap = bic - truth_bic
                    edges = int(candidate.sum())
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    gap, edges = np.nan, np.nan
                    diagnostic = {}
                records.append({
                    "file": path.name, "method": method, "restart": restart,
                    "stage": stage, "bic_gap_to_truth": gap,
                    "elapsed_seconds": float(trajectory_times[restart][stage]),
                    "candidate_edges": edges,
                    "candidate_threshold": diagnostic.get(
                        "candidate_threshold", np.nan),
                })

    coordinates = _mds3(np.asarray(vectors))
    frame = pd.DataFrame(records)
    frame[["pca1", "pca2", "pca3"]] = coordinates
    frame["frobenius_distance_from_previous"] = np.nan
    frame["distance_file"] = ""
    distance_rows = []
    for (filename, restart), group in frame.groupby(
            ["file", "restart"], sort=False):
        indices = group.sort_values("stage").index.to_numpy()
        matrices = _read(args.input_dir / "dagma_trajectories" / filename)[3][restart]
        for left in range(len(matrices)):
            for right in range(left + 1, len(matrices)):
                distance_rows.append({
                    "file": filename, "restart": restart,
                    "stage_left": left, "stage_right": right,
                    "frobenius_distance": float(
                        np.linalg.norm(matrices[right] - matrices[left])),
                })
        for offset in range(1, len(indices)):
            frame.loc[indices[offset], "frobenius_distance_from_previous"] = (
                np.linalg.norm(matrices[offset] - matrices[offset - 1]))
    pd.DataFrame(distance_rows).to_csv(
        args.output_dir / "dagma_trajectory_distances.csv", index=False)

    frame["retained"] = False
    keep_fraction = float(args.keep_fraction)
    for (filename, restart), group in frame.groupby(
            ["file", "restart"], sort=False):
        valid = group[np.isfinite(group.bic_gap_to_truth)]
        keep = max(2, int(np.ceil(keep_fraction * max(1, len(valid)))))
        selected = valid.nsmallest(keep, "bic_gap_to_truth").index
        frame.loc[selected, "retained"] = True
    filtered = frame[frame.retained].copy()
    frame.to_csv(args.output_dir / "plot_data_dagma_trajectories.csv", index=False)
    finite_gap = filtered.bic_gap_to_truth[
        np.isfinite(filtered.bic_gap_to_truth)]
    finite_time = filtered.elapsed_seconds[
        np.isfinite(filtered.elapsed_seconds)]
    if finite_gap.empty or finite_time.empty:
        raise SystemExit("trajectory contains no valid thresholded DAG points")
    norm = Normalize(vmin=float(finite_time.min()),
                     vmax=float(finite_time.max()))
    cmap = plt.colormaps["viridis"]
    fig = plt.figure(figsize=(7.0, 4.5))
    methods = list(dict.fromkeys(frame.method))
    for panel, method in enumerate(methods[:2], start=1):
        ax = fig.add_subplot(1, min(2, len(methods)), panel, projection="3d")
        subset = filtered[filtered.method == method]
        for restart, group in subset.groupby("restart"):
            group = group.sort_values("stage")
            xyz = group[["pca1", "pca2", "pca3"]].to_numpy()
            times = group.elapsed_seconds.to_numpy()
            for i in range(len(xyz) - 1):
                if not np.all(np.isfinite(xyz[i:i + 2])):
                    continue
                t = np.linspace(0.0, 1.0, 16)
                segment = xyz[i] + t[:, None] * (xyz[i + 1] - xyz[i])
                c0 = cmap(norm(times[i])) if np.isfinite(times[i]) else (0.5,)*3 + (1,)
                c1 = cmap(norm(times[i + 1])) if np.isfinite(times[i + 1]) else c0
                colors = [tuple(np.asarray(c0) * (1 - q) + np.asarray(c1) * q)
                          for q in t[:-1]]
                ax.add_collection3d(Line3DCollection(
                    [segment[j:j + 2] for j in range(len(segment) - 1)],
                    colors=colors, linewidths=1.0, alpha=.75))
            valid = np.isfinite(times)
            ax.scatter(xyz[valid, 0], xyz[valid, 1], xyz[valid, 2],
                       c=times[valid], cmap=cmap, norm=norm, s=16,
                       depthshade=False)
        ax.set_title(method.replace("_", " + "))
        ax.set_xlabel("MDS 1", labelpad=-5)
        ax.set_ylabel("MDS 2", labelpad=-5)
        ax.set_zlabel("MDS 3", labelpad=0)
        ax.tick_params(labelsize=7, pad=-2)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(finite_time.to_numpy())
    fig.subplots_adjust(left=.02, right=.98, top=.90, bottom=.18,
                        wspace=.02)
    fig.colorbar(sm, ax=fig.axes, orientation="horizontal", fraction=.045,
                 pad=.10, aspect=45,
                 label="elapsed optimizer time (s); retained points have best BIC gaps")
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(args.output_dir / f"dagma_trajectory_3d.{suffix}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
