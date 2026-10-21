#!/usr/bin/env bash
# Submit the disjoint command list through LRZ JobFarm on CoolMUC-4.
# CoolMUC-4 cm4_std requires 2--4 exclusive nodes and at least 112 physical
# cores per node. Four nodes provide 448 one-core JobFarm tasks.
# Keep #SBATCH directives immediately after the shebang. Slurm ignores
# directives that occur after executable shell statements.
#SBATCH -J notreks-jobfarm
#SBATCH --get-user-env
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
# Request exactly four nodes for the production farm.  The previous full run
# was historically submitted with two nodes and a four-hour override.
#SBATCH --nodes=4-4
#SBATCH --ntasks=448
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
# Notify only when the allocation starts or ends.  Avoid progress and
# requeue notifications for the long-running production farms.
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=f.bleile@tum.de

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the repository path on LRZ}"
CMD_FILE="${CMD_FILE:-$PROJECT_DIR/cluster/notreks_cmd.txt}"
TASKDB="${TASKDB:-$PROJECT_DIR/cluster/notreks_cmd}"
MAMBA_ENV="${MAMBA_ENV:-$PROJECT_DIR/.venv-lrz}"
MASTER_SEED="${MASTER_SEED:-20260917}"

# Use the same environment setup as interactive LRZ sessions.  This also
# places the bundled Rscript next to the venv Python before JobFarm starts.
source "$PROJECT_DIR/cluster/lrz_session_setup.sh"
export PATH="$MAMBA_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PROJECT_DIR/cluster/slurm_logs"
cd "$PROJECT_DIR"

# Capture the exact runtime used by every JobFarm allocation and fail before
# launching workers if the batch environment cannot import the protocol.
env_manifest="$PROJECT_DIR/cluster/slurm_logs/jobfarm.env.${SLURM_JOB_ID:-manual}.txt"
{
  echo "timestamp=$(date -Is)"
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "project=$PROJECT_DIR"
  echo "mamba_env=$MAMBA_ENV"
  echo "master_seed=$MASTER_SEED"
  echo "python=$(command -v python)"
  python --version
  echo "rscript=$(command -v Rscript || true)"
  "$MAMBA_ENV/bin/Rscript" --version 2>&1 || true
  git rev-parse HEAD 2>/dev/null || true
  git diff --quiet 2>/dev/null && echo "git_dirty=false" || echo "git_dirty=true"
  sha256sum "$CMD_FILE" 2>/dev/null || true
} > "$env_manifest"
if ! "$MAMBA_ENV/bin/python" "$PROJECT_DIR/scripts/notreks_protocol_all.py" --help \
    > "$env_manifest.protocol_help" 2>&1; then
  echo "Protocol import/CLI preflight failed; see $env_manifest.protocol_help" >&2
  exit 1
fi
if [[ "${RESET_JOBFARM:-0}" == "1" ]]; then
  # JobFarm keys its database/results to the input filename, not TASKDB.
  # Archive the old state instead of deleting/recreating the same directory.
  # On the shared filesystem, compute-side workers can retain a directory
  # handle while Slurm starts the next allocation; deleting that directory
  # can then produce "Stale file handle" and cascading worker Bus errors.
  old_result_dir="${CMD_FILE}_res"
  if [[ -e "$old_result_dir" ]]; then
    archive_dir="${old_result_dir}.archive.${SLURM_JOB_ID:-$$}"
    mv -- "$old_result_dir" "$archive_dir"
    echo "Archived previous JobFarm state: $archive_dir"
  fi
fi
set +e
jobfarm start "$CMD_FILE"
jobfarm_rc=$?
set -e

# JobFarm can return success even when individual tasks failed.  Do not use
# `jobfarm status` here: LRZ's status helper depends on `bc`, which is not
# guaranteed on compute nodes.  The per-task result markers are sufficient.
result_dir="${CMD_FILE}_res"
total_count="$(grep -c '^env ' "$CMD_FILE" || true)"
success_count=0
failed_count=0
processed_count=0
shopt -s nullglob
for result_file in "$result_dir"/[0-9]*; do
  if grep -q "STOP SUCCESS" "$result_file"; then
    success_count=$((success_count + 1))
    processed_count=$((processed_count + 1))
  elif grep -q "STOP FAILED" "$result_file"; then
    failed_count=$((failed_count + 1))
    processed_count=$((processed_count + 1))
  fi
done
printf 'JobFarm task markers: success=%d failed=%d processed=%d total=%d\n' \
  "$success_count" "$failed_count" "$processed_count" "$total_count"
if (( failed_count > 0 || processed_count < total_count )); then
  echo "JobFarm reported failed or incomplete tasks; marking Slurm job failed." >&2
  exit 1
fi
exit "$jobfarm_rc"
