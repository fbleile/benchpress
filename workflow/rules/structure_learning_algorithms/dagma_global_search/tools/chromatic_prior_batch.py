"""Run one chromatic-prior experiment over several graph seeds and pool it."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--graph-type", default="er2")
    parser.add_argument("--prior-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--knowledge-fractions", nargs="+", type=float, required=True)
    parser.add_argument("--knowledge-strategies", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=("flop", "dagma"), required=True)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--flop-sweeps", type=int, default=16)
    parser.add_argument("--flop-local-passes", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    script = Path(__file__).with_name("one_graph_prior_chromatic.py")
    instance_dirs = []
    for graph_seed in args.graph_seeds:
        instance_dir = args.output_dir / f"graph_{graph_seed}"
        instance_dirs.append(instance_dir)
        command = [
            sys.executable, str(script),
            "--d", str(args.d), "--n", str(args.n),
            "--graph-type", args.graph_type,
            "--graph-seed", str(graph_seed),
            "--prior-seeds", *map(str, args.prior_seeds),
            "--knowledge-fractions", *map(str, args.knowledge_fractions),
            "--knowledge-strategies", *args.knowledge_strategies,
            "--methods", *args.methods,
            "--attempts", str(args.attempts),
            "--flop-sweeps", str(args.flop_sweeps),
            "--flop-local-passes", str(args.flop_local_passes),
            "--output-dir", str(instance_dir),
        ]
        print(f"\n=== graph_seed={graph_seed} ===", flush=True)
        subprocess.run(command, check=True)

    aggregate = Path(__file__).with_name("aggregate_chromatic_prior_analysis.py")
    subprocess.run([
        sys.executable, str(aggregate),
        "--input-dirs", *(str(path) for path in instance_dirs),
        "--output-dir", str(args.output_dir / "aggregate"),
    ], check=True)


if __name__ == "__main__":
    main()
