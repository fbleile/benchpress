#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS="${NOTREKS_THREADS_PER_WORKER:-1}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$REPO_DIR/.venv-local-smoke/bin/python" \
  "$REPO_DIR/scripts/notreks_benchmark_pipeline.py" "$@"
