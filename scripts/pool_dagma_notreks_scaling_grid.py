#!/usr/bin/env python3
"""Pool completed DAGMA-NOTREKS lambda-grid cells and audit missing cells."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CELL_RE = re.compile(
    r"^d(?P<d>\d+)_n(?P<n>\d+)_(?P<graph>[a-z0-9]+)_lambdaNT(?P<lambda>[-+0-9.eE]+)$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    cells: list[dict[str, object]] = []
    for cell in sorted(args.input.iterdir()):
        if not cell.is_dir():
            continue
        match = CELL_RE.match(cell.name)
        if not match:
            continue
        meta = {
            "cell": cell.name,
            "d": int(match["d"]),
            "n": int(match["n"]),
            "graph_type": match["graph"],
            "lambda_nt": float(match["lambda"]),
        }
        per_seed = cell / "per_seed.csv"
        summary = cell / "summary.csv"
        completed = per_seed.exists() and summary.exists()
        if completed:
            frame = pd.read_csv(per_seed)
            for key, value in meta.items():
                frame[key] = value
            rows.append(frame)
            cells.append({**meta, "status": "completed", "reason": ""})
        else:
            reason = "unsupported_graph_type" if str(meta["graph_type"]).startswith("sf") else "incomplete_or_failed"
            cells.append({**meta, "status": "missing", "reason": reason})

    cell_frame = pd.DataFrame(cells)
    per_seed_frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    cell_frame.to_csv(args.output / "cell_status.csv", index=False)
    if not per_seed_frame.empty:
        per_seed_frame.to_csv(args.output / "pooled_per_seed.csv", index=False)

        # Rank lambda values within each paired experiment.  Lower CPDAG SHD
        # is better; average ranks handle exact ties deterministically.
        nt = per_seed_frame[per_seed_frame["method"] == "dagma_notreks"].copy()
        rank_keys = ["seed", "d", "n", "graph_type"]
        nt["lambda_rank"] = nt.groupby(rank_keys)["SHD_cpdag"].rank(
            method="average", ascending=True
        )
        nt.to_csv(args.output / "dagma_notreks_lambda_ranks_per_seed.csv", index=False)

        mean_rank = (
            nt.groupby("lambda_nt", dropna=False)["lambda_rank"]
            .agg(runs="size", mean_rank="mean", std_rank="std", median_rank="median", min_rank="min", max_rank="max")
            .reset_index()
            .sort_values(["mean_rank", "lambda_nt"])
        )
        mean_rank.to_csv(args.output / "dagma_notreks_mean_rank_by_lambda.csv", index=False)

        wins = (
            nt.loc[nt["lambda_rank"] == 1]
            .groupby("lambda_nt", dropna=False)
            .size()
            .rename("rank1_count")
            .reset_index()
        )
        wins["rank1_fraction"] = wins["rank1_count"] / nt[rank_keys].drop_duplicates().shape[0]
        wins.to_csv(args.output / "dagma_notreks_rank1_by_lambda.csv", index=False)

        metric = "SHD_cpdag"
        group_cols = ["method", "d", "n", "graph_type", "lambda_nt"]
        pooled = (
            per_seed_frame.groupby(group_cols, dropna=False)[metric]
            .agg(runs="size", mean="mean", std="std", median="median", min="min", max="max")
            .reset_index()
        )
        pooled.to_csv(args.output / "pooled_by_cell.csv", index=False)

        overall = (
            per_seed_frame.groupby(["method", "lambda_nt"], dropna=False)[metric]
            .agg(runs="size", mean="mean", std="std", median="median", min="min", max="max")
            .reset_index()
        )
        overall.to_csv(args.output / "pooled_by_lambda.csv", index=False)

        by_design = (
            per_seed_frame.groupby(["method", "d", "n", "graph_type"], dropna=False)[metric]
            .agg(runs="size", mean="mean", std="std", median="median", min="min", max="max")
            .reset_index()
        )
        by_design.to_csv(args.output / "pooled_by_design.csv", index=False)

    status = (
        cell_frame.groupby(["d", "n", "graph_type"], dropna=False)
        .agg(planned=("cell", "size"), completed=("status", lambda x: (x == "completed").sum()),
             missing=("status", lambda x: (x == "missing").sum()))
        .reset_index()
    )
    status["completion_rate"] = status["completed"] / status["planned"]
    status.to_csv(args.output / "completion_by_design.csv", index=False)

    reason = cell_frame.groupby("reason", dropna=False).size().rename("cells").reset_index()
    reason["fraction_of_cells"] = reason["cells"] / len(cell_frame)
    reason.to_csv(args.output / "failure_statistics.csv", index=False)

    lines = [
        f"planned cells: {len(cell_frame)}",
        f"completed cells: {(cell_frame.status == 'completed').sum()}",
        f"missing cells: {(cell_frame.status == 'missing').sum()}",
        f"completed fraction: {(cell_frame.status == 'completed').mean():.3f}",
        "",
        "Failure/status counts:",
        reason.to_string(index=False),
        "",
        "Completion by design:",
        status.to_string(index=False),
    ]
    (args.output / "pool_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
