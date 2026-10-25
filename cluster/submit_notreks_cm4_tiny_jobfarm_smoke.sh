#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/dss/dsshome1/0C/ge86xim2/benchpress
MAMBA_ENV="$PROJECT_DIR/.venv-lrz"
CMD_FILE="$PROJECT_DIR/cluster/notreks_cm4_tiny_jobfarm_smoke.cmd"

cd "$PROJECT_DIR"
unset TASKDB RESET_JOBFARM
mkdir -p "$PROJECT_DIR/cluster/slurm_logs"

exec sbatch \
  --clusters=cm4 \
  --partition=cm4_tiny \
  --qos=cm4_tiny \
  --chdir="$PROJECT_DIR" \
  --mail-user=f.bleile@tum.de \
  --mail-type=BEGIN,FAIL,END,TIME_LIMIT \
  --nodes=1 \
  --ntasks=17 \
  --cpus-per-task=1 \
  --time=00:15:00 \
  --output="$PROJECT_DIR/cluster/slurm_logs/jobfarm.tiny.%j.out" \
  --export=ALL,CMD_FILE="$CMD_FILE",MAMBA_ENV="$MAMBA_ENV" \
  "$PROJECT_DIR/scripts/lrz_jobfarm.sh"
