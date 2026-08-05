#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:?usage: $0 RUN_DIR}"
JOB_ID="$(sed -n '1p' "$RUN_DIR/driver_job_id")"
test -n "$JOB_ID"
scancel "$JOB_ID"
echo "cancelled driver $JOB_ID; rerun the submission command to resume completed tasks"
