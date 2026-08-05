#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${1:-${CONFIG:-}}"
# Interactive command substitution/copy-paste can accidentally introduce a
# newline into an exported config path.  Normalize it before file checks so a
# harmless wrapped variable does not look like a missing configuration.
CONFIG="${CONFIG//$'\n'/}"
CONFIG="${CONFIG//$'\r'/}"
echo "repository=$REPO_DIR"
echo "branch=$(git -C "$REPO_DIR" branch --show-current 2>/dev/null || echo unknown)"
echo "commit=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
missing=0
required_commands=(python snakemake)
for command in "${required_commands[@]}"; do
  if command -v "$command" >/dev/null 2>&1; then
    echo "$command=$(command -v "$command")"
    "$command" --version 2>&1 | head -1 || true
  else
    echo "MISSING: $command" >&2
    missing=1
  fi
done
if command -v snakemake >/dev/null 2>&1; then
  if [[ "${NOTREKS_CONTAINER_MODE:-auto}" == "host" ]]; then
    echo "container_mode=host"
    if python -c 'import dagma' >/dev/null 2>&1; then
      echo "dagma_python=available"
    else
      echo "MISSING: Python package dagma (required for host-mode DAGMA rules)" >&2
      echo "Install dagma==1.1.1 in the selected environment or use a reachable DAGMA container." >&2
      missing=1
    fi
    # Host mode bypasses the FLOP container as well.  Check the extension
    # before submission so a farm task cannot fail later with an opaque
    # ModuleNotFoundError from a Snakemake script.
    if [[ -n "${CONFIG:-}" ]]; then
      host_config="$CONFIG"
      [[ -f "$REPO_DIR/$CONFIG" ]] && host_config="$REPO_DIR/$CONFIG"
      if python - "$host_config" 2>/dev/null <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
names = payload.get("resources", {}).get("structure_learning_algorithms", {})
raise SystemExit(0 if any(str(n) in {"flop", "flop_notreks"} for n in names) else 1)
PY
      then
        if python -c 'import flopsearch' >/dev/null 2>&1; then
          echo "flopsearch_python=available"
        else
          echo "MISSING: Python package flopsearch (required for host-mode FLOP rules)" >&2
          echo "Install it with: bash scripts/install_flopsearch_host.sh" >&2
          missing=1
        fi
      fi
    fi
  else
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
fi
test -f "$REPO_DIR/workflow/Snakefile" || { echo "MISSING: workflow/Snakefile" >&2; missing=1; }
if [[ -n "$CONFIG" ]]; then
  if [[ "$CONFIG" = /* ]]; then
    config_path="$CONFIG"
  else
    config_path="$REPO_DIR/$CONFIG"
  fi
  test -f "$config_path" || { echo "MISSING: config=$config_path" >&2; missing=1; }
  if [[ -f "$config_path" ]]; then
    if ! PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python - "$REPO_DIR" "$config_path" <<'PY'
import sys
from pathlib import Path
from workflow.rules.structure_learning_algorithms.notreks.tools.farm import (
    validate_cpdag_config,
    validate_required_files,
)
repo, config = Path(sys.argv[1]), Path(sys.argv[2])
validate_cpdag_config(config)
validate_required_files(repo, config)
PY
    then
      echo "CONFIG ERROR: required Benchpress files or CPDAG conversion are invalid" >&2
      missing=1
    fi
  fi
fi
if [[ "$missing" -ne 0 ]]; then
  echo "preflight: FAILED" >&2
  exit 1
fi
if [[ -n "$CONFIG" && -f "$config_path" ]]; then
  parse_log="$(mktemp)"
  # Snakemake creates a source cache during parsing.  On managed macOS and
  # shared filesystems TMPDIR can point at a stale or non-writable location;
  # keep this short-lived validation state inside the repository instead.
  preflight_tmp="$(dirname "$config_path")"
  parse_bin=""
  if [[ "${NOTREKS_CONTAINER_MODE:-auto}" == "host" ]] && \
     ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
    parse_bin="$(mktemp -d)"
    printf '%s\n' '#!/bin/sh' "echo 'singularity version 3.8.0'" > "$parse_bin/singularity"
    chmod +x "$parse_bin/singularity"
  fi
  trap 'rm -f "$parse_log"; test -z "$parse_bin" || rm -rf "$parse_bin"' EXIT
  if ! (cd "$REPO_DIR" && TMPDIR="$preflight_tmp" PATH="${parse_bin:+$parse_bin:}$PATH" snakemake --dry-run --quiet --nolock \
      --snakefile workflow/Snakefile --configfile "$config_path" --cores 1) \
      >"$parse_log" 2>&1; then
    echo "ERROR: Snakemake could not parse the selected configuration:" >&2
    sed -n '1,40p' "$parse_log" >&2
    exit 1
  fi
  rm -f "$parse_log"
  test -z "$parse_bin" || rm -rf "$parse_bin"
  trap - EXIT
fi
echo "preflight: OK"
