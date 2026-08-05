#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${CONFIG:-${1:-}}"
echo "repository=$REPO_DIR"
echo "branch=$(git -C "$REPO_DIR" branch --show-current 2>/dev/null || echo unknown)"
echo "commit=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
missing=0
for command in python snakemake sbatch srun; do
  if command -v "$command" >/dev/null 2>&1; then
    echo "$command=$(command -v "$command")"
    "$command" --version 2>&1 | head -1 || true
  else
    echo "MISSING: $command" >&2
    missing=1
  fi
done
if command -v snakemake >/dev/null 2>&1; then
  HELP="$(snakemake --help 2>&1)"
  if grep -q -- '--use-apptainer' <<<"$HELP"; then
    echo "container_flag=--use-apptainer"
    command -v apptainer >/dev/null 2>&1 || { echo "MISSING: apptainer" >&2; missing=1; }
  elif grep -q -- '--use-singularity' <<<"$HELP"; then
    echo "container_flag=--use-singularity"
    if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
      echo "MISSING: singularity or apptainer" >&2; missing=1
    fi
  else
    echo "MISSING: Snakemake container flag" >&2
    missing=1
  fi
fi
test -f "$REPO_DIR/workflow/Snakefile" || { echo "MISSING: workflow/Snakefile" >&2; missing=1; }
if [[ -n "$CONFIG" ]]; then
  test -f "$REPO_DIR/$CONFIG" || test -f "$CONFIG" || { echo "MISSING: config=$CONFIG" >&2; missing=1; }
  config_path="$CONFIG"
  [[ -f "$REPO_DIR/$CONFIG" ]] && config_path="$REPO_DIR/$CONFIG"
  if [[ -f "$config_path" ]]; then
    if ! python - "$config_path" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
ok=all('cpdag' in s.get('evaluation', {}).get('graph_estimation', {}).get('convert_to', [])
       for s in x.get('benchmark_setup', []))
raise SystemExit(0 if ok and x.get('benchmark_setup') else 1)
PY
    then
      echo "CONFIG ERROR: every benchmark_setup must request graph_estimation.convert_to=['cpdag']" >&2
      missing=1
    fi
  fi
fi
if [[ "${CLUSTERS:-serial}:${PARTITION:-serial_std}" == "serial:cm4_std" ]]; then
  echo "invalid resource pair: serial + cm4_std" >&2
  missing=1
fi
if [[ "$missing" -ne 0 ]]; then
  echo "preflight: FAILED" >&2
  exit 1
fi
echo "preflight: OK"
