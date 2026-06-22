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
exec "$SCRIPT_DIR/notreks_snakemake_driver_common.sh"
