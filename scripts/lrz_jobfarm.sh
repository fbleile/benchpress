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

# JobFarm can return success even when individual tasks failed.  Convert that
# condition into a failed Slurm job so --mail-type=FAIL notifies the user.
status_text="$(jobfarm status "$CMD_FILE" 2>&1 || true)"
status_counts="$(sed -n 's/.*= \[ \([0-9][0-9]*\) \([0-9][0-9]*\) \([0-9][0-9]*\) \].*/\1 \2 \3/p' <<<"$status_text")"
if [[ -z "$status_counts" ]]; then
  echo "$status_text" >&2
  echo "Could not parse JobFarm status; marking Slurm job failed." >&2
  exit 1
fi
read -r success_count failed_count total_count <<<"$status_counts"
if (( failed_count > 0 || success_count + failed_count < total_count )); then
  echo "$status_text" >&2
  echo "JobFarm reported failed tasks; marking Slurm job failed." >&2
  exit 1
fi
exit "$jobfarm_rc"
