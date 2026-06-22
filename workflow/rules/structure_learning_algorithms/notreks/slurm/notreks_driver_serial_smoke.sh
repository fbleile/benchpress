#!/usr/bin/env bash
#SBATCH -o slurm_logs/notreks-smoke.%N.%j.out
#SBATCH -J NotreksSmoke
#SBATCH --mail-user=f.bleile@tum.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --time=02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SNAKEMAKE_CORES="${SNAKEMAKE_CORES:-8}"
exec "$SCRIPT_DIR/notreks_snakemake_driver_common.sh"
