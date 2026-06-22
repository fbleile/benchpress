#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/dss/dsshome1/0C/ge86xim2/benchpress}"
RUN_DIR="${RUN_DIR:-results/notreks_experiments/slurm_smoke}"
CONFIG="${CONFIG:-}"
SNAKEMAKE_CORES="${SNAKEMAKE_CORES:-${SLURM_CPUS_PER_TASK:-1}}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/bin/micromamba}"
CONDA_ENV="${CONDA_ENV:-benchpress-notreks}"
APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer/1.3.4}"
SQUASHFS_MODULE="${SQUASHFS_MODULE:-squashfs/4.6.1}"

if [[ -z "$CONFIG" ]]; then
  echo "ERROR: CONFIG must point to a Benchpress config file" >&2
  exit 2
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: REPO_DIR does not exist: $REPO_DIR" >&2
  exit 2
fi

cd "$REPO_DIR"
REPO_ROOT="$PWD"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: CONFIG does not exist: $CONFIG" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/logs/slurm"
RUN_LOG="$RUN_DIR/logs/slurm/driver-${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "NOTREKS Snakemake driver configuration"
echo "  SCRIPT_DIR=${SCRIPT_DIR:-}"
echo "  REPO_ROOT=${REPO_ROOT:-}"
echo "  PWD=$(pwd)"
echo "  RUN_DIR=${RUN_DIR:-}"
echo "  CONFIG=${CONFIG:-}"
echo "  SNAKEMAKE_CORES=${SNAKEMAKE_CORES:-}"

module load "$APPTAINER_MODULE"
module load "$SQUASHFS_MODULE"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found after module load $APPTAINER_MODULE" >&2
  exit 3
fi

if ! command -v mksquashfs >/dev/null 2>&1; then
  echo "ERROR: mksquashfs not found after module load $SQUASHFS_MODULE. Apptainer cannot convert Docker images to SIF." >&2
  echo "PATH=$PATH" >&2
  echo "Try manually: module load $SQUASHFS_MODULE; which mksquashfs" >&2
  exit 3
fi

echo "apptainer=$(command -v apptainer)"
apptainer --version
echo "mksquashfs=$(command -v mksquashfs)"
mksquashfs -version || true

if [[ ! -x "$MICROMAMBA_BIN" ]]; then
  echo "ERROR: MICROMAMBA_BIN is not executable: $MICROMAMBA_BIN" >&2
  exit 2
fi
eval "$("$MICROMAMBA_BIN" shell hook -s bash)"
micromamba activate "$CONDA_ENV"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

echo "NOTREKS Snakemake driver"
echo "  hostname=$(hostname)"
echo "  date=$(date -Is)"
echo "  pwd=$PWD"
echo "  git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
echo "  python=$(command -v python)"
python --version
echo "  snakemake=$(command -v snakemake)"
SMK_VER="$(snakemake --version)"
echo "$SMK_VER"
if [[ "$SMK_VER" == 9.* ]] && grep -Eq '"gcastle_(pc|direct_lingam)"' "$CONFIG"; then
  echo "ERROR: Snakemake 9 is incompatible with Python 3.7 Benchpress gCastle containers. Use the pinned environment." >&2
  exit 3
fi
SMK_MAJOR="${SMK_VER%%.*}"
if [[ "$SMK_MAJOR" -ge 8 ]]; then
  CONTAINER_FLAG="--use-apptainer"
else
  CONTAINER_FLAG="--use-singularity"
  if ! command -v singularity >/dev/null 2>&1; then
    SHIM_DIR="${RUN_DIR}/bin"
    mkdir -p "$SHIM_DIR"
    ln -sf "$(command -v apptainer)" "${SHIM_DIR}/singularity"
    export PATH="${PWD}/${SHIM_DIR}:$PATH"
  fi
fi
echo "  apptainer=$(command -v apptainer)"
apptainer --version
echo "  container_flag=$CONTAINER_FLAG"
echo "  singularity=$(command -v singularity || true)"
singularity --version || true
echo "  RUN_DIR=$RUN_DIR"
echo "  CONFIG=$CONFIG"
echo "  SNAKEMAKE_CORES=$SNAKEMAKE_CORES"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "  log=$RUN_LOG"

snakemake \
  --cores "$SNAKEMAKE_CORES" \
  "$CONTAINER_FLAG" \
  --snakefile workflow/Snakefile \
  --configfile "$CONFIG"
