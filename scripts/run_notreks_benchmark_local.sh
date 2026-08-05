#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: $0 CONFIG RUN_DIR [MANIFEST]}"
RUN_DIR="${2:?usage: $0 CONFIG RUN_DIR [MANIFEST]}"
MANIFEST="${3:-${MANIFEST:-}}"
export REPO_DIR CONFIG RUN_DIR MANIFEST
bash "$REPO_DIR/scripts/check_notreks_environment.sh" "$CONFIG"
mkdir -p "$REPO_DIR/$RUN_DIR/logs"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${SNAKEMAKE_CORES:-1}" != "1" ]]; then
  echo "local workflow: forcing SNAKEMAKE_CORES=1 because scenarios share Benchpress results paths" >&2
fi
export SNAKEMAKE_CORES=1
if [[ -z "${ANALYSIS_COMMAND:-}" ]]; then
  if [[ -z "$MANIFEST" ]]; then
    echo "ANALYSIS_COMMAND must be provided when no scenario manifest is supplied" >&2
    exit 2
  fi
  ANALYSIS_COMMAND="python scripts/notreks_benchmark.py collect --manifest '$MANIFEST' --results-root '$REPO_DIR/results/output' --output '$REPO_DIR/$RUN_DIR/results.csv' && python scripts/notreks_benchmark.py analyse --results-csv '$REPO_DIR/$RUN_DIR/results.csv' --output-dir '$REPO_DIR/$RUN_DIR/analysis'"
fi
export ANALYSIS_COMMAND
args=(--repo "$REPO_DIR" --run-dir "$RUN_DIR" --config "$CONFIG" --workers "$SNAKEMAKE_CORES")
if [[ -n "$MANIFEST" ]]; then args+=(--manifest "$MANIFEST"); fi
if [[ "${NOTREKS_DRY_RUN:-0}" == "1" ]]; then args+=(--dry-run); fi
if [[ -n "${ANALYSIS_COMMAND:-}" ]]; then args+=(--analysis-command "$ANALYSIS_COMMAND"); fi
exec python "$REPO_DIR/workflow/rules/structure_learning_algorithms/notreks/tools/farm.py" "${args[@]}"
