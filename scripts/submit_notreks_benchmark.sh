#!/usr/bin/env bash
set -euo pipefail

repo_dir="${NOTREKS_REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
spec="${1:-$repo_dir/configs/notreks_benchmark/large_benchmark_v1.json}"
generated="${2:-$repo_dir/configs/notreks_benchmark/generated_v1}"
output_dir="${3:-$repo_dir/results/dagma_notreks_oracle/main_benchmark}"
python_bin="${NOTREKS_PYTHON:-python}"

cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
command -v sbatch >/dev/null || {
  echo "sbatch is required to submit the cluster benchmark" >&2
  exit 2
}
"$python_bin" -c 'import numpy, pandas, matplotlib' || {
  echo "NOTREKS_PYTHON must provide NumPy, pandas, and matplotlib" >&2
  exit 2
}
"$python_bin" scripts/notreks_benchmark.py compile \
  --spec "$spec" --output-dir "$generated"

command_file="$generated/commands.txt"
manifest="$generated/scenario_manifest.csv"
mkdir -p "$output_dir/logs"
row_count="$(wc -l < "$command_file" | tr -d ' ')"
if [[ "$row_count" -lt 1 ]]; then
  echo "No benchmark scenarios were generated" >&2
  exit 2
fi

array_max=$((row_count - 1))
echo "Submitting $row_count native Benchpress scenarios (array 0-$array_max)."
echo "Each task uses workflow/Snakefile with Apptainer; completed Snakemake outputs are resumable."
array_job="$(sbatch --parsable \
  --job-name=notreks-benchmark \
  --output="$output_dir/logs/benchpress_%A_%a.out" \
  --array="0-$array_max" \
  --cpus-per-task="${NOTREKS_CPUS_PER_TASK:-4}" \
  --mem="${NOTREKS_MEMORY:-16G}" \
  --time="${NOTREKS_TIME_LIMIT:-2-00:00:00}" \
  scripts/run_notreks_cluster_array.sh "$command_file")"
array_job="${array_job%%;*}"

analysis_job="$(sbatch --parsable \
  --job-name=notreks-analysis \
  --output="$output_dir/logs/analysis_%j.out" \
  --dependency="afterok:$array_job" \
  --cpus-per-task=1 \
  --mem="${NOTREKS_ANALYSIS_MEMORY:-16G}" \
  --time="${NOTREKS_ANALYSIS_TIME_LIMIT:-02:00:00}" \
  scripts/run_notreks_cluster_finalize.sh \
  "$manifest" "$repo_dir/results/output" "$output_dir")"
analysis_job="${analysis_job%%;*}"

printf 'Benchmark array job: %s\nAnalysis job: %s (runs automatically after success)\n' \
  "$array_job" "$analysis_job"
printf 'Final report: %s/analysis/REPORT.md\n' "$output_dir"
