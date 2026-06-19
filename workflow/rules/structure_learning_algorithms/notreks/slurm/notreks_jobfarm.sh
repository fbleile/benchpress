#!/usr/bin/env bash
#SBATCH -o slurm_logs/jobfarm.%N.%j.out
#SBATCH -J NotreksJobFarm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
#SBATCH --nodes=2
#SBATCH --ntasks=200
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-/dss/dsshome1/0C/ge86xim2/benchpress}"
CONDA_ENV="${CONDA_ENV:-benchpress-notreks}"
CMD_FILE="${CMD_FILE:-cmd.txt}"
TASKDB="${TASKDB:-cmd}"
FRESH="${FRESH:-0}"
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

module load slurm_setup
module load jobfarm

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export TMPDIR="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" slurm_logs

if [[ "$FRESH" == "1" ]]; then
  rm -f "${TASKDB}.db"
  rm -rf "${TASKDB}.txt_res"
fi

echo "Starting NOTREKS JobFarm"
echo "  REPO_DIR=$REPO_DIR"
echo "  CONDA_ENV=$CONDA_ENV"
echo "  CMD_FILE=$CMD_FILE"
echo "  TASKDB=$TASKDB"
echo "  FRESH=$FRESH"

jobfarm start "$CMD_FILE"
