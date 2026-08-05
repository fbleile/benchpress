#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${1:?usage: $0 CONFIG RUN_DIR MANIFEST WORKERS}"
RUN_DIR="${2:?usage: $0 CONFIG RUN_DIR MANIFEST WORKERS}"
MANIFEST="${3:?usage: $0 CONFIG RUN_DIR MANIFEST WORKERS}"
WORKERS="${4:?usage: $0 CONFIG RUN_DIR MANIFEST WORKERS}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NOTREKS_CONTAINER_MODE=host
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

bash scripts/check_notreks_environment.sh "$CONFIG"

PY="$(command -v python)"
RUN_ABS="$(cd "$(dirname "$RUN_DIR")" && mkdir -p "$(basename "$RUN_DIR")" && cd "$(basename "$RUN_DIR")" && pwd)"
ANALYSIS_COMMAND="\"$PY\" \"$REPO_DIR/scripts/notreks_benchmark.py\" collect --manifest \"$MANIFEST\" --results-root \"$RUN_ABS\" --output \"$RUN_ABS/results.csv\" && \"$PY\" \"$REPO_DIR/scripts/notreks_benchmark.py\" analyse --results-csv \"$RUN_ABS/results.csv\" --output-dir \"$RUN_ABS/analysis\""

# One srun step owns the allocation.  The farm then runs one-core Snakemake
# processes concurrently, but every process has its own workspace/results
# tree, metadata database, cache, and temporary directory.
exec srun --exclusive --ntasks=1 --cpus-per-task="$WORKERS" \
  env ANALYSIS_COMMAND="$ANALYSIS_COMMAND" \
  python workflow/rules/structure_learning_algorithms/notreks/tools/farm.py \
    --repo "$REPO_DIR" --run-dir "$RUN_ABS" --config "$CONFIG" \
    --manifest "$MANIFEST" --workers "$WORKERS" --isolate-tasks \
    --no-task-timeout \
    --analysis-command "$ANALYSIS_COMMAND"
