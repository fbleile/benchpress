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
#SBATCH --nodes=4
#SBATCH --ntasks=448
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,FAIL,END,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90,REQUEUE
#SBATCH --mail-user=f.bleile@tum.de

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the repository path on LRZ}"
CMD_FILE="${CMD_FILE:-$PROJECT_DIR/cluster/notreks_cmd.txt}"
TASKDB="${TASKDB:-$PROJECT_DIR/cluster/notreks_cmd}"
MAMBA_ENV="${MAMBA_ENV:-$PROJECT_DIR/.venv-lrz}"

module load slurm_setup
module load jobfarm

# Login-shell activation is not guaranteed to be inherited by Slurm/JobFarm
# workers.  Recreate it explicitly in the batch shell and export the runtime
# paths inherited by every command in the command file.
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate "$MAMBA_ENV"
else
  echo "micromamba is not available in the batch environment" >&2
  exit 127
fi
export PATH="$MAMBA_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$PROJECT_DIR/cluster/slurm_logs"
cd "$PROJECT_DIR"
if [[ "${RESET_JOBFARM:-0}" == "1" ]]; then
  rm -f "${TASKDB}.db"
  rm -rf "${TASKDB}.txt_res"
fi
set +e
jobfarm start "$CMD_FILE"
jobfarm_rc=$?
set -e

# JobFarm can return success even when individual tasks failed.  Do not use
# `jobfarm status` here: LRZ's status helper depends on `bc`, which is not
# guaranteed on compute nodes.  The per-task result markers are sufficient.
result_dir="${CMD_FILE}_res"
total_count="$(awk 'NF && $0 !~ /^#/ && $0 != "set -euo pipefail" {n++} END {print n+0}' "$CMD_FILE")"
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
