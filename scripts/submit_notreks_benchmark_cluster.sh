#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: $0 CONFIG RUN_DIR MANIFEST}"
RUN_DIR="${2:?usage: $0 CONFIG RUN_DIR MANIFEST}"
MANIFEST="${3:?usage: $0 CONFIG RUN_DIR MANIFEST}"
WORKERS="${NOTREKS_WORKERS:-16}"
TIME_LIMIT="${NOTREKS_TIME_LIMIT:-168:00:00}"
MEMORY="${NOTREKS_MEMORY:-64G}"
# The cluster driver runs the already-installed Python implementations.  Host
# mode is the safe default: it avoids pulling the unavailable public DAGMA
# image and is also passed explicitly to the submitted allocation below.
export NOTREKS_CONTAINER_MODE="${NOTREKS_CONTAINER_MODE:-host}"

if (( WORKERS < 1 || WORKERS > 16 )); then
  echo "ERROR: serial partitions support 1-16 workers; got $WORKERS" >&2
  exit 2
fi
cd "$REPO_DIR"
bash scripts/check_notreks_environment.sh "$CONFIG"
mkdir -p "$RUN_DIR/logs/slurm"

JOB_RAW="$(sbatch --parsable \
  --clusters=serial --partition=serial_long --qos=cm4_serial_long \
  --nodes=1 --ntasks=1 --cpus-per-task="$WORKERS" --mem="$MEMORY" --time="$TIME_LIMIT" \
  --chdir="$REPO_DIR" \
  --export=ALL,REPO_DIR="$REPO_DIR",NOTREKS_CONTAINER_MODE=host \
  -o "$RUN_DIR/logs/slurm/notreks-%j.out" \
  -e "$RUN_DIR/logs/slurm/notreks-%j.err" \
  "$REPO_DIR/scripts/run_notreks_cluster_driver.sh" "$CONFIG" "$RUN_DIR" "$MANIFEST" "$WORKERS")"
# Slurm may append ``;cluster`` in parsable output when clusters are enabled.
# Keep only the numeric job id for status/cancellation commands.
JOB_ID="${JOB_RAW%%;*}"
printf '%s\n' "$JOB_ID" > "$RUN_DIR/driver_job_id"
printf 'submitted job=%s workers=%s memory=%s partition=serial_long time=%s\n' "$JOB_ID" "$WORKERS" "$MEMORY" "$TIME_LIMIT"
printf 'status: bash scripts/notreks_farm_status.sh %q\n' "$RUN_DIR"
printf 'cancel: scancel %s\n' "$JOB_ID"
