#!/usr/bin/env python3
"""Create one disjoint LRZ JobFarm command per experiment and method.

Each command writes to its own result directory.  The collector can merge the
tables afterwards, so no worker ever writes the same CSV or checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.notreks_protocol_registry import REGISTRY, scaled_replicates


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
    p.add_argument("--synthetic-experiments", nargs="+",
                   default=("main", "main-misspec-linear-nongaussian",
                            "main-misspec-nonlinear-gaussian",
                            "pstrek-vs-tcc"),
                   help="registered synthetic arms to shard")
    p.add_argument("--batch-size", type=int, default=16,
                   help="methods per JobFarm command (default: all eligible methods)")
    p.add_argument("--cell-batch-size", type=int, default=1,
                   help="registered graph cells per JobFarm command (default: one cell)")
    p.add_argument("--cell-indices", nargs="+", type=int, default=None,
                   help="explicit registered cell indices for a smoke")
    p.add_argument("--replicate-batch-size", type=int, default=2,
                   help="graph replicates per task; all replicates remain in the full run")
    p.add_argument("--split-priors", action="store_true",
                   help="make one task per n/q/strategy/knowledge-round shard")
    p.add_argument("--include-real-world", action="store_true",
                   help="also add Sachs and causalAssembly commands")
    p.add_argument("--attempts", type=int, default=None,
                   help="override registry attempts; omit for FLOP=20/DAGMA=2")
    p.add_argument("--flop-sweeps", type=int, default=16)
    p.add_argument("--dagma-stages", type=int, default=5)
    p.add_argument("--dagma-warm-iter", type=int, default=30000)
    p.add_argument("--dagma-max-iter", type=int, default=60000)
    p.add_argument("--max-wall-hours", type=float, default=23.0,
                   help="per-task solver guard; leave one hour before cm4_std timeout")
    p.add_argument("--master-seed", type=int, default=20260917,
                   help="explicit protocol master seed")
    p.add_argument("--n-values", nargs="+", type=int, default=None)
    p.add_argument("--knowledge-rounds", type=int, default=None)
    args = p.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        p.error("--fraction must lie in (0, 1]")
    if args.batch_size < 1:
        p.error("--batch-size must be at least 1")
    if args.cell_batch_size < 1:
        p.error("--cell-batch-size must be at least 1")
    if args.replicate_batch_size < 1:
        p.error("--replicate-batch-size must be at least 1")
    if args.cell_indices is not None and any(index < 0 for index in args.cell_indices):
        p.error("--cell-indices must be nonnegative")
    if args.attempts is not None and args.attempts < 1:
        p.error("--attempts must be at least 1")
    root = args.repo.resolve()
    # Preserve the environment wrapper path.  Resolving a venv symlink can
    # turn .venv-local-smoke/bin/python into the base interpreter path.
    py = str(root / args.python) if not str(args.python).startswith("/") else args.python
    out = args.output_root.resolve()
    # JobFarm input files must contain commands only.  Shell headers would be
    # interpreted as additional worker tasks by JobFarm.
    lines: list[str] = []
    env = f"env PYTHONPATH={root} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"

    synthetic = {
        "main": ("flop flop-nt-edge-mask flop-nt-post flop_notreks "
                 "dagma dagma-nt-edge-mask dagma-nt-post dagma_notreks "
                 "var_sortnregress r2_sortnregress"),
        "main-misspec-linear-nongaussian": (
            "flop flop-nt-edge-mask flop-nt-post flop_notreks "
            "dagma dagma-nt-edge-mask dagma-nt-post dagma_notreks "
            "var_sortnregress r2_sortnregress"),
        "main-misspec-nonlinear-gaussian": (
            "flop flop-nt-edge-mask flop-nt-post flop_notreks "
            "dagma dagma-nt-edge-mask dagma-nt-post dagma_notreks "
            "var_sortnregress r2_sortnregress"),
        "pstrek-vs-tcc": ("dagma var_sortnregress r2_sortnregress "
                           "dagma_notreks dagma_notreks_tcc dagma-nt-edge-mask dagma-nt-post"),
    }
    unknown = sorted(set(args.synthetic_experiments) - set(synthetic))
    if unknown:
        p.error(f"unknown synthetic experiments: {unknown}")

    def method_batches(methods: list[str]) -> list[list[str]]:
        """Keep vanilla/post-selection methods in one process, by family.

        The protocol runner's candidate cache is process-local.  FLOP and
        DAGMA therefore need separate family batches so their post-selection
        variants see the exact vanilla candidate, while a single task does
        not combine both expensive solver families and exceed the wall limit.
        """
        families: list[list[str]] = []
        current_family = None
        current: list[str] = []
        for method in methods:
            family = "dagma" if method.startswith("dagma") else (
                "baseline" if method in {"var_sortnregress", "r2_sortnregress"}
                else "flop")
            if current and family != current_family:
                families.append(current)
                current = []
            current_family = family
            current.append(method)
        if current:
            families.append(current)
        batches: list[list[str]] = []
        for family in families:
            for index in range(0, len(family), args.batch_size):
                batches.append(family[index:index + args.batch_size])
        return batches

    for experiment in args.synthetic_experiments:
        methods = synthetic[experiment].split()
        spec = REGISTRY[experiment]
        cell_count = len(spec.cells)
        task_index = 0
        if args.cell_indices is not None:
            cell_starts = [(index, 1) for index in args.cell_indices
                           if index < cell_count]
        else:
            # Never create a command spanning cells with different method
            # eligibility.  In particular, the protocol excludes DAGMA for
            # d=100, and JobFarm must not receive a command that names DAGMA
            # for such a cell even though the runtime would filter it later.
            cell_starts = []
            cell_start = 0
            while cell_start < cell_count:
                dimension = spec.cells[cell_start][0]
                cell_limit = 0
                while (cell_start + cell_limit < cell_count
                       and cell_limit < args.cell_batch_size
                       and spec.cells[cell_start + cell_limit][0] == dimension):
                    cell_limit += 1
                cell_starts.append((cell_start, cell_limit))
                cell_start += cell_limit
        for cell_start, cell_limit in cell_starts:
            cell_dimensions = {
                spec.cells[index][0]
                for index in range(cell_start, cell_start + cell_limit)
            }
            if len(cell_dimensions) != 1:
                raise ValueError(
                    f"cell batch {cell_start}:{cell_limit} crosses dimensions "
                    f"for {experiment}; split it before compiling commands")
            dimension = next(iter(cell_dimensions))
            eligible_methods = set(spec.methods_for(dimension))
            cell_methods = [method for method in methods
                            if method in eligible_methods]
            if not cell_methods:
                continue
            replicate_count = scaled_replicates(REGISTRY[experiment], args.fraction)
            for replicate_start in range(0, replicate_count,
                                         args.replicate_batch_size):
                replicate_limit = min(args.replicate_batch_size,
                                      replicate_count - replicate_start)
                for batch in method_batches(cell_methods):
                    method_args = " ".join(batch)
                    batch_label = "_".join(method.replace('-', '_') for method in batch)
                    n_shards = args.n_values or [None]
                    prior_shards = [(None, None, None)]
                    if args.split_priors:
                        prior_shards = []
                        for q in spec.q_values:
                            strategies = spec.knowledge_strategies if q == .25 else ("random",)
                            rounds = spec.q25_rounds if q == .25 else 1
                            for strategy in strategies:
                                for round_id in range(rounds):
                                    prior_shards.append((q, strategy, round_id))
                    for n_value in n_shards:
                        for q_value, strategy, round_id in prior_shards:
                            job_label = f"job_{experiment}_{task_index:05d}_c{cell_start:02d}_r{replicate_start:02d}"
                            if n_value is not None:
                                job_label += f"_n{n_value}"
                            if q_value is not None:
                                job_label += f"_q{q_value:g}_{strategy}_k{round_id:02d}"
                            job_out = out / f"{job_label}_{batch_label}"
                            optional = (f" --replicate-start {replicate_start}"
                                        f" --replicate-limit {replicate_limit}")
                            if n_value is not None:
                                optional += f" --n-values {n_value}"
                            if q_value is not None:
                                optional += f" --q-values {q_value:g}"
                                optional += f" --knowledge-strategies {strategy}"
                                optional += f" --knowledge-round-start {round_id} --knowledge-round-limit 1"
                            if args.knowledge_rounds is not None:
                                optional += f" --knowledge-rounds {args.knowledge_rounds}"
                            attempts = f" --attempts {args.attempts}" if args.attempts is not None else ""
                            lines.append(
                                f"{env} {py} {root}/scripts/notreks_protocol_all.py "
                                f"--experiments {experiment} --fraction {args.fraction:g} "
                                f"--master-seed {args.master_seed} "
                                f"--cell-start {cell_start} --cell-limit {cell_limit} "
                                f"--methods {method_args} --workers 1{attempts} "
                                f"--flop-sweeps {args.flop_sweeps} --dagma-stages {args.dagma_stages} "
                                f"--dagma-warm-iter {args.dagma_warm_iter} "
                                f"--dagma-max-iter {args.dagma_max_iter} "
                                f"--max-wall-hours {args.max_wall_hours:g}{optional} "
                                f"--skip-figures --output-root {job_out}")
                            task_index += 1

    sachs_methods = (
        "flop flop-nt-standard flop-nt-edge-mask flop-nt-post "
        "dagma dagma-pstrek dagma-nt-edge-mask dagma-nt-post "
        "var_sortnregress r2_sortnregress dagma-nonlinear dagma-nonlinear-pstrek"
    )
    if args.include_real_world:
        for batch_index in range(0, len(sachs_methods.split()), args.batch_size):
            batch = sachs_methods.split()[batch_index:batch_index + args.batch_size]
            method_args = " ".join(batch)
            batch_label = "_".join(method.replace('-', '_') for method in batch)
            job_out = out / f"job_sachs_{batch_index // args.batch_size:02d}_{batch_label}"
            lines.append(
                f"{env} {py} {root}/scripts/sachs_benchmark.py "
                f"--data {root}/resources/data/mydatasets/2005_sachs/1_cd3cd28_n854.csv "
                f"--truth {root}/resources/adjmat/myadjmats/sachs.csv "
                f"--seeds {args.sachs_seed} --bootstrap-replicates {args.sachs_bootstrap_replicates} "
                f"--knowledge-fraction 0.25 1.0 --methods {method_args} "
                f"--flop-attempts 20 --dagma-attempts 2 --flop-sweeps 16 "
                f"--dagma-stages 5 --dagma-warm-iter 30000 --dagma-max-iter 60000 "
                f"--output {job_out}")

    causal_methods = (
        "flop flop-nt-standard flop-nt-edge-mask flop-nt-post "
        "dagma dagma-pstrek dagma-nt-edge-mask dagma-nt-post "
        "var_sortnregress r2_sortnregress dagma-nonlinear dagma-nonlinear-pstrek"
    )
    seeds = args.causal_seeds
    if args.include_real_world:
        for mode in ("oracle_notreks", "estimated_notreks"):
            methods = causal_methods.split()
            for batch_index in range(0, len(methods), args.batch_size):
                batch = methods[batch_index:batch_index + args.batch_size]
                method_args = " ".join(batch)
                batch_label = "_".join(method.replace('-', '_') for method in batch)
                job_out = out / f"job_causalassembly_{mode}_{batch_index // args.batch_size:02d}_{batch_label}"
                lines.append(
                    f"{env} {py} {root}/scripts/causalassembly_benchmark.py "
                    f"--cache {out}/causalassembly_cache --output {job_out} "
                    f"--mode {mode} --seeds {seeds} --n 500 --q 0.25 1.0 "
                    f"--methods {method_args} --flop-attempts 20 --dagma-attempts 2 "
                    f"--flop-sweeps 16 --dagma-stages 5 --dagma-warm-iter 30000 "
                    f"--dagma-max-iter 60000")

    args.command_file.parent.mkdir(parents=True, exist_ok=True)
    args.command_file.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} disjoint commands to {args.command_file}")


if __name__ == "__main__":
    main()
