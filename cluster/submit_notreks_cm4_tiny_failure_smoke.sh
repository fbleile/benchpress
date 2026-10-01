#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/dss/dsshome1/0C/ge86xim2/benchpress
MAMBA_ENV="$PROJECT_DIR/.venv-lrz"
OUT="$PROJECT_DIR/results/lrz_failure_smoke"

mkdir -p "$PROJECT_DIR/cluster/slurm_logs"

sbatch \
  --clusters=cm4 \
  --partition=cm4_tiny \
  --qos=cm4_tiny \
  --chdir="$PROJECT_DIR" \
  --mail-user=f.bleile@tum.de \
  --mail-type=BEGIN,FAIL,END,TIME_LIMIT \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --time=00:30:00 \
  --output="$PROJECT_DIR/cluster/slurm_logs/notreks.failure-smoke.%j.out" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",MAMBA_ENV="$MAMBA_ENV",OUT="$OUT" \
  --wrap='set -euo pipefail
module load slurm_setup
cd "$PROJECT_DIR"
test -f resources/binarydatagen/generate_DAG.R
env PYTHONPATH="$PROJECT_DIR" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$MAMBA_ENV/bin/python" -c '\''from scripts.notreks_protocol import make_graph; import numpy as np; a=make_graph(20,"ws",2,20260917); assert a.shape==(20,20); assert np.all(np.diag(a)==0); print("WS generation OK", int(a.sum()))'\''
env PYTHONPATH="$PROJECT_DIR" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$MAMBA_ENV/bin/python" scripts/notreks_protocol_all.py \
    --experiments main --fraction 0.02 --graph-replicates 1 \
    --master-seed 20260917 \
    --cell-start 2 --cell-limit 1 --replicate-start 0 --replicate-limit 1 \
    --n-values 100 --methods flop flop-nt-edge-mask flop-nt-post flop_notreks dagma dagma-nt-edge-mask dagma-nt-post dagma_notreks \
    --workers 1 --attempts 1 --flop-sweeps 2 --dagma-stages 2 \
    --dagma-warm-iter 1000 --dagma-max-iter 2000 --knowledge-rounds 1 \
    --skip-figures --output-root "$OUT"'
