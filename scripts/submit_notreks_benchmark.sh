#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: $0 CONFIG RUN_DIR [MANIFEST]}"
RUN_DIR="${2:?usage: $0 CONFIG RUN_DIR [MANIFEST]}"
MANIFEST="${3:-${MANIFEST:-}}"
export REPO_DIR CONFIG RUN_DIR MANIFEST
bash "$REPO_DIR/scripts/check_notreks_cluster_environment.sh" "$CONFIG"
mkdir -p "$REPO_DIR/$RUN_DIR/logs/slurm"
cluster="${NOTREKS_CLUSTER:-serial}"
partition="${NOTREKS_PARTITION:-serial_std}"
if [[ "$cluster:$partition" == "serial:cm4_std" ]]; then
  echo "invalid resource pair: cm4_std belongs to cluster cm4" >&2
  exit 2
fi
qos_args=()
if [[ "$cluster:$partition" == "serial:serial_long" ]]; then qos_args+=(--qos=cm4_serial_long); fi
export SNAKEMAKE_CORES="${SNAKEMAKE_CORES:-16}"
if [[ -z "${ANALYSIS_COMMAND:-}" ]]; then
  config_name="$(basename "$CONFIG" .json)"
  config_name="${config_name#selected_}"
  config_name="${config_name%_config}"
  ANALYSIS_COMMAND="python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py analyze-benchmark --tag '$config_name' --output-dir '$REPO_DIR/$RUN_DIR/analysis'"
fi
export ANALYSIS_COMMAND
if [[ "${NOTREKS_DRY_RUN:-0}" == "1" ]]; then
  export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
  dry_args=(--repo "$REPO_DIR" --run-dir "$RUN_DIR" --config "$CONFIG" --workers "$SNAKEMAKE_CORES" --dry-run)
  if [[ -n "$MANIFEST" ]]; then dry_args+=(--manifest "$MANIFEST"); fi
  python "$REPO_DIR/workflow/rules/structure_learning_algorithms/notreks/tools/farm.py" "${dry_args[@]}"
  exit 0
fi
echo "Submitting one driver allocation: cluster=$cluster partition=$partition workers=$SNAKEMAKE_CORES"
job_id="$(sbatch --parsable --clusters="$cluster" --partition="$partition" "${qos_args[@]}" \
  --cpus-per-task="$SNAKEMAKE_CORES" --mem="${NOTREKS_MEMORY:-64G}" \
  --time="${NOTREKS_TIME_LIMIT:-24:00:00}" --export=ALL \
  -o "$REPO_DIR/$RUN_DIR/logs/slurm/%x-%j.out" -e "$REPO_DIR/$RUN_DIR/logs/slurm/%x-%j.err" \
  "$REPO_DIR/workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_farm.sh")"
job_id="${job_id%%;*}"
printf '%s\n' "$job_id" > "$REPO_DIR/$RUN_DIR/driver_job_id"
echo "driver_job_id=$job_id"
echo "status: bash scripts/notreks_farm_status.sh $RUN_DIR"
echo "cancel: bash scripts/notreks_farm_cancel.sh $RUN_DIR"
