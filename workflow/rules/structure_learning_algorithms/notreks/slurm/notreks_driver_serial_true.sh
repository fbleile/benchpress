#!/usr/bin/env bash
#SBATCH -o slurm_logs/notreks-true-serial.%N.%j.out
#SBATCH -J NotreksTrueSerial
#SBATCH --mail-user=f.bleile@tum.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SNAKEMAKE_CORES="${SNAKEMAKE_CORES:-16}"
exec "$SCRIPT_DIR/notreks_snakemake_driver_common.sh"
