#!/usr/bin/env bash
# One Slurm allocation; scenario concurrency is managed inside farm.py.
#SBATCH -J notreks-farm
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=24:00:00
#SBATCH --export=ALL

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
CONFIG="${CONFIG:?CONFIG must be set}"
cd "$REPO_DIR"
mkdir -p "$RUN_DIR/logs/slurm"
printf '%s\n' "${SLURM_JOB_ID:-local}" > "$RUN_DIR/driver_job_id"
exec > >(tee -a "$RUN_DIR/logs/slurm/driver-${SLURM_JOB_ID:-local}.log") 2>&1
if [[ -n "${MICROMAMBA_BIN:-}" && -n "${CONDA_ENV:-}" ]]; then
  eval "$(${MICROMAMBA_BIN} shell hook -s bash)"
  micromamba activate "$CONDA_ENV"
fi
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKERS="${SNAKEMAKE_CORES:-${SLURM_CPUS_PER_TASK:-1}}"
args=(--repo "$REPO_DIR" --run-dir "$RUN_DIR" --config "$CONFIG" --workers "$WORKERS")
if [[ -n "${MANIFEST:-}" ]]; then args+=(--manifest "$MANIFEST"); fi
if [[ -n "${RESOURCE_CLASS:-}" ]]; then args+=(--resource-class "$RESOURCE_CLASS"); fi
if [[ -n "${ANALYSIS_COMMAND:-}" ]]; then args+=(--analysis-command "$ANALYSIS_COMMAND"); fi
python workflow/rules/structure_learning_algorithms/notreks/tools/farm.py "${args[@]}"
