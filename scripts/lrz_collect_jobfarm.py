#!/usr/bin/env python3
"""Merge isolated JobFarm outputs without rerunning any solver."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--command-file", type=Path,
                   help="optional JobFarm command list for completeness validation")
    p.add_argument("--require-complete", action="store_true",
                   help="fail unless every command has a successful result table")
    args = p.parse_args()
    expected_outputs = []
    if args.command_file is not None:
        for line in args.command_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("#!"):
                continue
            tokens = shlex.split(line)
            if "--output-root" in tokens:
                expected_outputs.append(Path(tokens[tokens.index("--output-root") + 1]))
    frames = []
    # Synthetic runs keep isolated processes in results/<experiment>/jobs/;
    # rglob also remains compatible with the older flat layout.
    for path in sorted(args.root.rglob("job_*/raw/results.csv")):
        frame = pd.read_csv(path)
        frame["job_directory"] = str(path.parent.parent)
        frames.append(frame)
    for path in sorted(args.root.rglob("job_*/results.csv")):
        frame = pd.read_csv(path)
        frame["job_directory"] = str(path.parent)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no completed job tables found below {args.root}")
    if args.require_complete:
        missing = [path for path in expected_outputs
                   if not (path / "raw" / "results.csv").is_file()]
        failed = []
        for path in expected_outputs:
            result_path = path / "raw" / "results.csv"
            if result_path.is_file():
                frame = pd.read_csv(result_path)
                if "solver_status" not in frame or frame.empty or \
                        (frame["solver_status"] != "ok").any():
                    failed.append(str(path))
        if missing or failed:
            raise SystemExit(
                f"incomplete JobFarm results: missing={len(missing)} "
                f"failed_or_incomplete={len(failed)}")
    result = pd.concat(frames, ignore_index=True, sort=False)
    if "run_id" in result:
        result = result.drop_duplicates("run_id", keep="last")
    else:
        key = [column for column in (
            "experiment_id", "data_id", "knowledge_fraction",
            "knowledge_round", "knowledge_strategy", "method")
               if column in result.columns]
        if key:
            result = result.drop_duplicates(key, keep="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"merged {len(result)} rows from {len(frames)} job tables into {args.output}")


if __name__ == "__main__":
    main()
