"""Small CLI for canonical DAGMA-fast smoke and weighted-output runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .lambda_policy import resolve_lambda1
from .objective import LinearL2Objective
from .optimizer import DagmaFastConfig, fit_weighted_adjacency


def smoke(args):
    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.n, args.dimension))
    X[:, 1] += 0.8 * X[:, 0]
    resolved = resolve_lambda1(
        args.lambda_policy, n=args.n, d=args.dimension,
        fixed=args.lambda1)
    result = fit_weighted_adjacency(
        LinearL2Objective(X),
        DagmaFastConfig(
            lambda1=resolved.value, T=2, warm_iter=5, max_iter=8,
            checkpoint=4, max_runtime_seconds=5.0))
    payload = {
        "method": "dagma_fast",
        "lambda_policy": resolved.policy,
        "lambda1_resolved": resolved.value,
        "n": args.n,
        "d": args.dimension,
        "iterations": result.iterations,
        "termination_reason": result.termination_reason,
        "factorization_count": result.factorization_count,
        "zero_diagonal": bool(np.all(np.diag(result.weighted_adjacency) == 0)),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    command = sub.add_parser("smoke")
    command.add_argument("--dimension", type=int, default=5)
    command.add_argument("--n", type=int, default=100)
    command.add_argument("--seed", type=int, default=1701)
    command.add_argument(
        "--lambda-policy",
        choices=("fixed", "sqrt_log_d_over_n"),
        default="fixed")
    command.add_argument("--lambda1", type=float, default=0.03)
    command.add_argument("--output")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "smoke":
        smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
