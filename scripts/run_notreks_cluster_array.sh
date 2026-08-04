#!/usr/bin/env bash
set -euo pipefail

repo_dir="${NOTREKS_REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
command_file="${1:-$repo_dir/configs/notreks_benchmark/generated_v1/commands.txt}"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"
line_number=$((task_id + 1))
command="$(sed -n "${line_number}p" "$command_file")"

if [[ -z "$command" ]]; then
  echo "No benchmark command at array index $task_id" >&2
  exit 2
fi

cd "$repo_dir"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NOTREKS_CORES="${SLURM_CPUS_PER_TASK:-1}"

echo "[$(date -Iseconds)] task=$task_id command=$command"
bash -lc "$command"
