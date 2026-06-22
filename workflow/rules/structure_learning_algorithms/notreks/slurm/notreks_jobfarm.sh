#!/usr/bin/env bash
#SBATCH -o slurm_logs/jobfarm.%N.%j.out
#SBATCH -J NotreksJobFarm
#SBATCH --mail-user=f.bleile@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-/dss/dsshome1/0C/ge86xim2/benchpress}"
CONDA_ENV="${CONDA_ENV:-benchpress-notreks}"
RUN_DIR="${RUN_DIR:-results/notreks_experiments/slurm_smoke}"
CONFIG="${CONFIG:-}"
CMD_FILE="${CMD_FILE:-${RUN_DIR}/cmd.txt}"
TASKDB="${TASKDB:-$RUN_DIR/logs/slurm/cmd}"
FRESH="${FRESH:-0}"
DRY_RUN="${DRY_RUN:-0}"
SNAKEMAKE_CORES="${SNAKEMAKE_CORES:-all}"
SNAKEMAKE_CONTAINER_ARG="${SNAKEMAKE_CONTAINER_ARG:---use-apptainer}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: REPO_DIR does not exist: $REPO_DIR" >&2
  exit 2
fi
cd "$REPO_DIR"

if [[ ! -s "$CMD_FILE" ]]; then
  echo "ERROR: CMD_FILE is missing or empty: $CMD_FILE" >&2
  exit 2
fi
if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda initialization script not found: $CONDA_SH" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/logs/slurm"
RUN_LOG="$RUN_DIR/logs/slurm/jobfarm.${SLURM_JOB_ID:-local}.out"
exec > >(tee -a "$RUN_LOG") 2>&1

module load slurm_setup
module load jobfarm

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export TMPDIR="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$RUN_DIR/logs/slurm" slurm_logs

if [[ "$FRESH" == "1" ]]; then
  rm -f "${TASKDB}.db"
  rm -rf "${TASKDB}.txt_res"
fi

echo "Starting NOTREKS JobFarm"
echo "  host=$(hostname)"
echo "  date=$(date -Is)"
echo "  git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  REPO_DIR=$REPO_DIR"
echo "  CONDA_ENV=$CONDA_ENV"
echo "  RUN_DIR=$RUN_DIR"
echo "  CONFIG=${CONFIG:-<from CMD_FILE>}"
echo "  CMD_FILE=$CMD_FILE"
echo "  TASKDB=$TASKDB"
echo "  FRESH=$FRESH"
echo "  DRY_RUN=$DRY_RUN"
echo "  first command:"
head -n 1 "$CMD_FILE"

jobfarm start "$CMD_FILE"
