#!/usr/bin/env python3
"""Generate the shared production figure set separately for each main arm.

JobFarm collection produces one aggregate table containing several experiment
arms.  The production figure grammar is intentionally arm-local: experiment
identity is not added as another visual encoding or legend entry.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.notreks_production_figures import write_protocol_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="collected JobFarm CSV")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="directory receiving one subdirectory per main arm")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {"experiment_id", "method", "solver_status"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"input is missing required columns: {sorted(missing)}")

    arms = sorted(
        arm for arm in frame["experiment_id"].dropna().unique()
        if str(arm).startswith("main")
    )
    if not arms:
        raise SystemExit("no main* experiment arms found")

    for arm in arms:
        subset = frame[(frame["experiment_id"] == arm) &
                       (frame["solver_status"] == "ok")].copy()
        if subset.empty:
            raise SystemExit(f"no successful rows for {arm}")

        arm_root = args.output_root / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        subset.to_csv(arm_root / "aggregate_results.csv", index=False)

        # The existing production layer uses the main protocol name to select
        # the main-grid plots.  The arm is already isolated in this loop, so
        # normalize only the plotting copy; the saved aggregate keeps its real
        # experiment_id.
        plotting = subset.copy()
        plotting["experiment_id"] = "main"
        write_protocol_figures(
            plotting,
            arm_root / "analysis" / "figures",
            arm_root,
        )
        print(f"generated {arm} figures from {len(subset)} rows")


if __name__ == "__main__":
    main()
