#!/usr/bin/env bash
# Submit the disjoint command list through LRZ JobFarm on CoolMUC-4.
# CoolMUC-4 cm4_std requires 2--4 exclusive nodes and at least 112 physical
# cores per node. Four nodes provide 448 one-core JobFarm tasks.
# Keep #SBATCH directives immediately after the shebang. Slurm ignores
# directives that occur after executable shell statements.
#SBATCH -J notreks-jobfarm
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
#SBATCH --nodes=4
#SBATCH --ntasks=448
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the repository path on LRZ}"
CMD_FILE="${CMD_FILE:-$PROJECT_DIR/cluster/notreks_cmd.txt}"
TASKDB="${TASKDB:-$PROJECT_DIR/cluster/notreks_cmd}"

module load slurm_setup
module load jobfarm
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$PROJECT_DIR/cluster/slurm_logs"
cd "$PROJECT_DIR"
if [[ "${RESET_JOBFARM:-0}" == "1" ]]; then
  rm -f "${TASKDB}.db"
  rm -rf "${TASKDB}.txt_res"
fi
jobfarm start "$CMD_FILE"
