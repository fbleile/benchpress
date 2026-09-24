#!/usr/bin/env python3
"""Create one disjoint LRZ JobFarm command per experiment and method.

Each command writes to its own result directory.  The collector can merge the
tables afterwards, so no worker ever writes the same CSV or checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--python", default=".venv-lrz/bin/python")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--command-file", type=Path, required=True)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="synthetic protocol fraction (default: 1.0)")
    p.add_argument("--causal-seeds", default="1001 1002 1003 1004 1005")
    p.add_argument("--sachs-seed", default="1")
    p.add_argument("--sachs-bootstrap-replicates", type=int, default=50)
    args = p.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        p.error("--fraction must lie in (0, 1]")
    root = args.repo.resolve()
    py = str((root / args.python).resolve()) if not str(args.python).startswith("/") else args.python
    out = args.output_root.resolve()
    lines: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    env = f"env PYTHONPATH={root} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"

    synthetic = {
        "main": ("flop flop-nt-edge-mask flop-nt-post flop_notreks "
                 "dagma dagma-nt-edge-mask dagma-nt-post dagma_notreks "
                 "var_sortnregress r2_sortnregress"),
        "prior-structure": "flop flop_notreks dagma dagma_notreks",
        "pstrek-vs-tcc": ("dagma var_sortnregress r2_sortnregress "
                           "dagma_notreks dagma_notreks_tcc dagma-nt-edge-mask dagma-nt-post"),
    }
    for experiment, methods in synthetic.items():
        for method in methods.split():
            job_out = out / f"job_{experiment}_{method.replace('-', '_')}"
            lines.append(
                f"{env} {py} {root}/scripts/notreks_protocol_all.py "
                f"--experiments {experiment} --fraction {args.fraction:g} --methods {method} "
                f"--workers 1 --flop-sweeps 16 --dagma-stages 5 "
                f"--dagma-warm-iter 30000 --dagma-max-iter 60000 "
                f"--max-wall-hours 24 --output-root {job_out}")

    sachs_methods = (
        "flop flop-nt-standard flop-nt-edge-mask flop-nt-post "
        "dagma dagma-pstrek dagma-nt-edge-mask dagma-nt-post "
        "var_sortnregress r2_sortnregress dagma-nonlinear dagma-nonlinear-pstrek"
    )
    for method in sachs_methods.split():
        job_out = out / f"job_sachs_{method.replace('-', '_')}"
        lines.append(
            f"{env} {py} {root}/scripts/sachs_benchmark.py "
            f"--data {root}/resources/data/mydatasets/2005_sachs/1_cd3cd28_n854.csv "
            f"--truth {root}/resources/adjmat/myadjmats/sachs.csv "
            f"--seeds {args.sachs_seed} --bootstrap-replicates {args.sachs_bootstrap_replicates} "
            f"--knowledge-fraction 0.25 1.0 --methods {method} "
            f"--flop-attempts 20 --dagma-attempts 2 --flop-sweeps 16 "
            f"--dagma-stages 5 --dagma-warm-iter 30000 --dagma-max-iter 60000 "
            f"--output {job_out}")

    causal_methods = (
        "flop flop-nt-standard flop-nt-edge-mask flop-nt-post "
        "dagma dagma-pstrek dagma-nt-edge-mask dagma-nt-post "
        "var_sortnregress r2_sortnregress dagma-nonlinear dagma-nonlinear-pstrek"
    )
    seeds = args.causal_seeds
    for mode in ("oracle_notreks", "estimated_notreks"):
        for method in causal_methods.split():
            job_out = out / f"job_causalassembly_{mode}_{method.replace('-', '_')}"
            lines.append(
                f"{env} {py} {root}/scripts/causalassembly_benchmark.py "
                f"--cache {out}/causalassembly_cache --output {job_out} "
                f"--mode {mode} --seeds {seeds} --n 500 --q 0.25 1.0 "
                f"--methods {method} --flop-attempts 20 --dagma-attempts 2 "
                f"--flop-sweeps 16 --dagma-stages 5 --dagma-warm-iter 30000 "
                f"--dagma-max-iter 60000")

    args.command_file.parent.mkdir(parents=True, exist_ok=True)
    args.command_file.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)-3} disjoint commands to {args.command_file}")


if __name__ == "__main__":
    main()
