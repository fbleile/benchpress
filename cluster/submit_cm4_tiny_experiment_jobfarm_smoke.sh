#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/dss/dsshome1/0C/ge86xim2/benchpress
MAMBA_ENV="$PROJECT_DIR/.venv-lrz"

case "${1:-}" in
  causalassembly)
    TAG=causalassembly
    CMD_FILE="$PROJECT_DIR/cluster/causalassembly_cm4_tiny_jobfarm_smoke.cmd"
    ;;
  prior-structure)
    TAG=prior-structure
    CMD_FILE="$PROJECT_DIR/cluster/prior_structure_cm4_tiny_jobfarm_smoke.cmd"
    ;;
  *)
    echo "usage: $0 {causalassembly|prior-structure}" >&2
    exit 2
    ;;
esac

cd "$PROJECT_DIR"
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
  --time=00:30:00 \
  --output="$PROJECT_DIR/cluster/slurm_logs/jobfarm.${TAG}.tiny.%j.out" \
  --export=ALL,CMD_FILE="$CMD_FILE",MAMBA_ENV="$MAMBA_ENV",RESET_JOBFARM=1 \
  "$PROJECT_DIR/scripts/lrz_jobfarm.sh"
