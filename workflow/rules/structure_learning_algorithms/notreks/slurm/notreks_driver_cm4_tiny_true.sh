#!/usr/bin/env bash
#SBATCH -o slurm_logs/notreks-true-cm4tiny.%N.%j.out
#SBATCH -J NotreksTrueTiny
#SBATCH --mail-user=f.bleile@tum.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000M
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SNAKEMAKE_CORES="${SNAKEMAKE_CORES:-32}"
COMMON_DRIVER="${SCRIPT_DIR}/notreks_snakemake_driver_common.sh"
if [[ ! -f "$COMMON_DRIVER" && -n "${REPO_DIR:-}" ]]; then
  COMMON_DRIVER="${REPO_DIR}/workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_snakemake_driver_common.sh"
fi
if [[ ! -f "$COMMON_DRIVER" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  COMMON_DRIVER="${SLURM_SUBMIT_DIR}/workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_snakemake_driver_common.sh"
fi
if [[ ! -f "$COMMON_DRIVER" ]]; then
  echo "ERROR: common NOTREKS driver not found: $COMMON_DRIVER" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$COMMON_DRIVER"
