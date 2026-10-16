"""Compile disjoint, resumable CausalMan commands for local/JobFarm runs."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--python", default=".venv-lrz/bin/python")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--command-file", type=Path, required=True)
    p.add_argument("--variants", nargs="+", default=("causalman_small", "causalman_medium"))
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1001, 1011)))
    p.add_argument("--n-values", nargs="+", type=int, default=(500, 2000, 5000))
    p.add_argument("--flop-attempts", type=int, default=20)
    p.add_argument("--dagma-attempts", type=int, default=2)
    args = p.parse_args()
    root, py = args.repo.resolve(), args.python
    env = f"env PYTHONPATH={root} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
    families = {
        "flop": "flop flop-nt-standard flop-nt-edge-mask flop-nt-post",
        "dagma": ("dagma dagma-pstrek dagma-nt-edge-mask dagma-nt-post "
                  "dagma-nonlinear dagma-nonlinear-pstrek"),
        "baseline": "var_sortnregress r2_sortnregress",
    }
    lines = []
    index = 0
    for variant in args.variants:
        for seed in args.seeds:
            for n in args.n_values:
                for family, methods in families.items():
                    out = args.output_root / f"job_causalman_{variant}_{seed}_{n}_{family}"
                    lines.append(
                        f"{env} {root / py} {root}/scripts/causalman_benchmark.py "
                        f"--cache {args.cache} --output {out} --variants {variant} "
                        f"--seeds {seed} --n-values {n} --methods {methods} "
                        f"--flop-attempts {args.flop_attempts} "
                        f"--dagma-attempts {args.dagma_attempts} "
                        f"--flop-sweeps 16 --dagma-stages 5 "
                        f"--dagma-warm-iter 30000 --dagma-max-iter 60000")
                    index += 1
    args.command_file.parent.mkdir(parents=True, exist_ok=True)
    args.command_file.write_text("\n".join(lines) + "\n")
    print(f"wrote {index} disjoint CausalMan commands to {args.command_file}")


if __name__ == "__main__":
    main()
