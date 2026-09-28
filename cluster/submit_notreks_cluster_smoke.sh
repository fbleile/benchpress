#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/dss/dsshome1/0C/ge86xim2/benchpress
MAMBA_ENV="$PROJECT_DIR/.venv-lrz"
CMD_FILE="$PROJECT_DIR/cluster/notreks_cluster_smoke_cmd.txt"

cd "$PROJECT_DIR"
unset TASKDB RESET_JOBFARM

sbatch \
  --clusters=cm4 \
  --partition=cm4_std \
  --qos=cm4_std \
  --chdir="$PROJECT_DIR" \
  --mail-user="f.bleile@tum.de" \
  --mail-type=BEGIN,FAIL,END,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90,REQUEUE \
  --nodes=2 \
  --ntasks=224 \
  --cpus-per-task=1 \
  --time=00:30:00 \
  --output="$PROJECT_DIR/cluster/slurm_logs/jobfarm.smoke.%N.%j.out" \
  --export=ALL,CMD_FILE="$CMD_FILE",MAMBA_ENV="$MAMBA_ENV" \
  "$PROJECT_DIR/scripts/lrz_jobfarm.sh"
