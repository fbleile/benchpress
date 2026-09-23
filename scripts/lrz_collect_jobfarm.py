#!/usr/bin/env python3
"""Merge isolated JobFarm outputs without rerunning any solver."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frames = []
    for path in sorted(args.root.glob("job_*/raw/results.csv")):
        frame = pd.read_csv(path)
        frame["job_directory"] = str(path.parent.parent)
        frames.append(frame)
    for path in sorted(args.root.glob("job_*/results.csv")):
        frame = pd.read_csv(path)
        frame["job_directory"] = str(path.parent)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no completed job tables found below {args.root}")
    result = pd.concat(frames, ignore_index=True, sort=False)
    if "run_id" in result:
        result = result.drop_duplicates("run_id", keep="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"merged {len(result)} rows from {len(frames)} job tables into {args.output}")


if __name__ == "__main__":
    main()
