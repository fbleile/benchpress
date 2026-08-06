#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: $0 CONFIG RUN_DIR MANIFEST}"
RUN_DIR="${2:?usage: $0 CONFIG RUN_DIR MANIFEST}"
MANIFEST="${3:?usage: $0 CONFIG RUN_DIR MANIFEST}"
WORKERS="${NOTREKS_WORKERS:-16}"
TIME_LIMIT="${NOTREKS_TIME_LIMIT:-168:00:00}"
MEMORY="${NOTREKS_MEMORY:-64G}"
CLUSTER="${NOTREKS_CLUSTER:-serial}"
PARTITION="${NOTREKS_PARTITION:-serial_long}"
QOS="${NOTREKS_QOS:-cm4_serial_long}"
# The cluster driver runs the already-installed Python implementations.  Host
# mode is the safe default: it avoids pulling the unavailable public DAGMA
# image and is also passed explicitly to the submitted allocation below.
export NOTREKS_CONTAINER_MODE="${NOTREKS_CONTAINER_MODE:-host}"

if [[ "$CLUSTER/$PARTITION" == "serial/serial_long" ]]; then
  (( WORKERS >= 1 && WORKERS <= 16 )) || {
    echo "ERROR: serial_long supports 1-16 workers; got $WORKERS" >&2; exit 2;
  }
elif [[ "$CLUSTER/$PARTITION" == "serial/serial_std" ]]; then
  (( WORKERS >= 1 && WORKERS <= 16 )) || {
    echo "ERROR: serial_std supports 1-16 workers; got $WORKERS" >&2; exit 2;
  }
  QOS="${NOTREKS_QOS:-}"
elif [[ "$CLUSTER/$PARTITION" == "cm4/cm4_tiny" ]]; then
  (( WORKERS >= 17 && WORKERS <= 112 )) || {
    echo "ERROR: cm4_tiny supports 17-112 workers; got $WORKERS" >&2; exit 2;
  }
  QOS="${NOTREKS_QOS:-}"
else
  echo "ERROR: unsupported cluster/partition: $CLUSTER/$PARTITION" >&2
  echo "       supported: serial/serial_std, serial/serial_long, cm4/cm4_tiny" >&2
  exit 2
fi
cd "$REPO_DIR"
bash scripts/check_notreks_environment.sh "$CONFIG"
mkdir -p "$RUN_DIR/logs/slurm"

SBATCH_ARGS=(--parsable --clusters="$CLUSTER" --partition="$PARTITION"
  --nodes=1 --ntasks=1 --cpus-per-task="$WORKERS" --mem="$MEMORY" --time="$TIME_LIMIT"
  --chdir="$REPO_DIR"
  --export=ALL,REPO_DIR="$REPO_DIR",NOTREKS_CONTAINER_MODE=host
  -o "$RUN_DIR/logs/slurm/notreks-%j.out"
  -e "$RUN_DIR/logs/slurm/notreks-%j.err")
if [[ -n "$QOS" ]]; then SBATCH_ARGS+=(--qos="$QOS"); fi
JOB_RAW="$(sbatch "${SBATCH_ARGS[@]}" \
  "$REPO_DIR/scripts/run_notreks_cluster_driver.sh" "$CONFIG" "$RUN_DIR" "$MANIFEST" "$WORKERS")"
# Slurm may append ``;cluster`` in parsable output when clusters are enabled.
# Keep only the numeric job id for status/cancellation commands.
JOB_ID="${JOB_RAW%%;*}"
printf '%s\n' "$JOB_ID" > "$RUN_DIR/driver_job_id"
printf 'submitted job=%s cluster=%s partition=%s workers=%s memory=%s time=%s\n' "$JOB_ID" "$CLUSTER" "$PARTITION" "$WORKERS" "$MEMORY" "$TIME_LIMIT"
printf 'status: bash scripts/notreks_farm_status.sh %q\n' "$RUN_DIR"
printf 'cancel: scancel %s\n' "$JOB_ID"
