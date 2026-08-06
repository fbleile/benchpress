#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPEC="${NOTREKS_SPEC:-$ROOT_DIR/configs/notreks_benchmark/linear_gaussian_d20_d50_v1.json}"
GENERATED_DIR="${NOTREKS_GENERATED_DIR:-$ROOT_DIR/configs/notreks_benchmark/generated_linear_gaussian_d20_d50_v1}"
MANIFEST="$GENERATED_DIR/scenario_manifest.csv"
RUN_DIR="${1:-results/notreks/linear_gaussian_d20_d50_v1}"
WORKERS="${NOTREKS_WORKERS:-3}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NOTREKS_CONTAINER_MODE=host
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

PYTHON_BIN="$PYTHON_BIN" bash "$ROOT_DIR/scripts/install_flopsearch_host.sh"

"$PYTHON_BIN" -c 'import dagma, flopsearch' >/dev/null || {
  echo "ERROR: dagma and flopsearch must be importable in the active environment." >&2
  echo "Run: PYTHON_BIN=$PYTHON_BIN bash scripts/install_flopsearch_host.sh" >&2
  exit 2
}

rm -rf -- "$GENERATED_DIR"
"$PYTHON_BIN" scripts/notreks_benchmark.py compile \
  --spec "$SPEC" --output-dir "$GENERATED_DIR"

RUN_ABS="$ROOT_DIR/$RUN_DIR"
ANALYSIS_COMMAND="$PYTHON_BIN scripts/notreks_benchmark.py collect --manifest '$MANIFEST' --results-root '$RUN_ABS' --output '$RUN_ABS/results.csv' && $PYTHON_BIN scripts/notreks_benchmark.py analyse --results-csv '$RUN_ABS/results.csv' --output-dir '$RUN_ABS/analysis' && PYTHONPATH='$ROOT_DIR' $PYTHON_BIN scripts/notreks_feasibility_projection.py --run-dir '$RUN_ABS' --output-dir '$RUN_ABS/analysis/feasibility_projection'"

exec "$PYTHON_BIN" workflow/rules/structure_learning_algorithms/notreks/tools/farm.py \
  --repo "$ROOT_DIR" --run-dir "$RUN_DIR" \
  --config "$GENERATED_DIR/linear-gaussian__er2__d20__n100__k0p25.json" \
  --manifest "$MANIFEST" --workers "$WORKERS" --isolate-tasks \
  --no-task-timeout --analysis-command "$ANALYSIS_COMMAND"
