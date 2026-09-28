#!/usr/bin/env bash
# Source this after every LRZ login:
#   source /dss/dsshome1/0C/ge86xim2/benchpress/cluster/lrz_session_setup.sh
# It only prepares the shell; it does not submit or execute a job.

export PROJECT_DIR="/dss/dsshome1/0C/ge86xim2/benchpress"
export MAMBA_ENV="$PROJECT_DIR/.venv-lrz"
export LRZ_EMAIL="f.bleile@tum.de"
export PY="$MAMBA_ENV/bin/python"

module load slurm_setup >/dev/null 2>&1 || true
module load jobfarm >/dev/null 2>&1 || true

if ! command -v micromamba >/dev/null 2>&1; then
    echo "micromamba is not available in PATH" >&2
    return 1 2>/dev/null || exit 1
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$MAMBA_ENV"

export PATH="$MAMBA_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$PROJECT_DIR/cluster/slurm_logs"
cd "$PROJECT_DIR" || return 1 2>/dev/null || exit 1

echo "LRZ Benchpress environment ready"
echo "  project: $PROJECT_DIR"
echo "  python:  $(command -v python)"
echo "  email:   $LRZ_EMAIL"
echo "  host:    $(hostname)"
