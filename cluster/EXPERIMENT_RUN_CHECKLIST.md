# Synthetic experiment runbook

Repository:

```bash
/Users/fbleile/Projects/benchpress-dagma-notreks-oracle
```

The four stages are: local smoke, LRZ login-node smoke, LRZ JobFarm smoke, and the full synthetic JobFarm run.

All stages use the deterministic master seed `20260917`. The protocol derives graph, data, NOTREKS, and solver seeds from this master seed. If a different seed is desired, add `--master-seed <integer>` to the protocol command or generator invocation.

## 1. Local smoke

Run from the repository root. This tests one representative cell from every synthetic arm, with low solver budgets.

```bash
cd /Users/fbleile/Projects/benchpress-dagma-notreks-oracle

PYTHONPATH=. .venv-local-smoke/bin/python \
  scripts/notreks_protocol_all.py \
    --experiments main main-misspec-linear-nongaussian \
      main-misspec-nonlinear-gaussian pstrek-vs-tcc \
    --fraction 0.01 \
    --cell-indices 0 \
    --n-values 100 \
    --knowledge-rounds 1 \
    --methods \
      flop flop-nt-edge-mask flop-nt-post flop_notreks \
      dagma dagma-nt-edge-mask dagma-nt-post dagma_notreks \
      dagma_notreks_tcc var_sortnregress r2_sortnregress \
    --attempts 1 \
    --flop-sweeps 2 \
    --dagma-stages 2 \
    --dagma-warm-iter 1000 \
    --dagma-max-iter 2000 \
    --max-wall-hours 1 \
    --skip-figures \
    --output-root results/local_smoke
```

This is a direct local run, not JobFarm.

## 2. LRZ login-node smoke

After every new login:

```bash
cd /dss/dsshome1/0C/ge86xim2/benchpress
source cluster/lrz_session_setup.sh
```

Verify the environment:

```bash
"$PY" -c 'import numpy, scipy, pandas, sklearn, torch, tqdm, flopsearch; from dagma.linear import DagmaLinear; print("Python environment OK")'
"$PY" scripts/notreks_protocol_all.py --help >/tmp/notreks_help.txt
```

Run the same small smoke on the login node, using the LRZ Python environment:

```bash
"$PY" scripts/notreks_protocol_all.py \
  --experiments main main-misspec-linear-nongaussian \
    main-misspec-nonlinear-gaussian pstrek-vs-tcc \
  --fraction 0.01 \
  --cell-indices 0 \
  --n-values 100 \
  --knowledge-rounds 1 \
  --methods flop dagma flop_notreks dagma_notreks dagma_notreks_tcc \
    var_sortnregress r2_sortnregress \
  --attempts 1 \
  --flop-sweeps 2 \
  --dagma-stages 2 \
  --dagma-warm-iter 1000 \
  --dagma-max-iter 2000 \
  --max-wall-hours 1 \
  --skip-figures \
  --output-root results/login_smoke
```

Do not run a substantial benchmark on the login node.

## 3. LRZ JobFarm smoke

Generate a fresh, small command list. The generator creates one method × one graph cell task. At a 1% fraction, the replicate count is also scaled down.

```bash
"$PY" scripts/lrz_make_jobfarm_cmds.py \
  --repo "$PROJECT_DIR" \
  --python .venv-lrz/bin/python \
  --output-root results/lrz_smoke \
  --command-file cluster/notreks_smoke_cmd.txt \
  --fraction 0.01 \
  --synthetic-experiments \
    main main-misspec-linear-nongaussian \
    main-misspec-nonlinear-gaussian pstrek-vs-tcc \
  --cell-indices 0 \
  --n-values 100 \
  --knowledge-rounds 1 \
  --replicate-batch-size 1 \
  --attempts 1 \
  --flop-sweeps 2 \
  --dagma-stages 2 \
  --dagma-warm-iter 1000 \
  --dagma-max-iter 2000 \
  --max-wall-hours 1

wc -l cluster/notreks_smoke_cmd.txt
```

Submit through Slurm; do not execute the JobFarm script with `bash`:

```bash
export CMD_FILE="$PROJECT_DIR/cluster/notreks_smoke_cmd.txt"
export RESET_JOBFARM=1
sbatch "$PROJECT_DIR/scripts/lrz_jobfarm.sh"
```

Check the allocation:

```bash
squeue --clusters=cm4 --partition=cm4_std -u "$USER"
find "${CMD_FILE}_res" -maxdepth 1 -type f -name '[0-9]*' | wc -l
tail -f cluster/slurm_logs/jobfarm.*.out
```

## 4. Full synthetic JobFarm run

Generate the full synthetic protocol only. Do not pass `--attempts`: registry defaults are FLOP=20 and DAGMA=2.

```bash
"$PY" scripts/lrz_make_jobfarm_cmds.py \
  --repo "$PROJECT_DIR" \
  --python .venv-lrz/bin/python \
  --output-root results/lrz_full \
  --command-file cluster/notreks_full_cmd.txt \
  --fraction 1.0 \
  --synthetic-experiments \
    main main-misspec-linear-nongaussian \
    main-misspec-nonlinear-gaussian pstrek-vs-tcc \
  --batch-size 1 \
  --cell-batch-size 1 \
  --replicate-batch-size 2 \
  --flop-sweeps 16 \
  --dagma-stages 5 \
  --dagma-warm-iter 30000 \
  --dagma-max-iter 60000 \
  --max-wall-hours 23

wc -l cluster/notreks_full_cmd.txt
sha256sum cluster/notreks_full_cmd.txt
```

Submit:

```bash
export CMD_FILE="$PROJECT_DIR/cluster/notreks_full_cmd.txt"
export RESET_JOBFARM=1
sbatch "$PROJECT_DIR/scripts/lrz_jobfarm.sh"
```

The task layout is deterministic. Every task has one method, one graph cell, and two graph replicates. All graph replicates are included across the command list.

## Restart after a Slurm timeout

Do not create a manual retry list and do not reuse JobFarm state.

1. Keep the same command-file arguments and output root.
2. Regenerate the command file if needed.
3. Submit a fresh JobFarm allocation with `RESET_JOBFARM=1`.

The protocol reads each task's existing `raw/results.csv`. Valid rows (`solver_status=ok` with finite SHD and runtime) are skipped. Failed, malformed, or missing rows are recomputed. Existing experiment output directories are not deleted by `RESET_JOBFARM=1`.

## Collection

Only run this after all tasks finish:

```bash
"$PY" scripts/lrz_collect_jobfarm.py \
  --root results/lrz_full \
  --command-file cluster/notreks_full_cmd.txt \
  --require-complete \
  --output results/lrz_full/combined_results.csv
```

If collection reports missing or failed tasks, repeat the same fresh JobFarm submission before generating figures.

## Figures and analysis

Figures are generated separately, after collection:

```bash
"$PY" scripts/notreks_protocol_analysis.py \
  --input-root results/lrz_full
```

The generated figures are under:

```text
results/lrz_full/analysis/figures/
```

Never interpret a partial collection as a completed benchmark.
