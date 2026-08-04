#!/usr/bin/env bash
set -euo pipefail

repo_dir="${NOTREKS_REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
manifest="${1:-$repo_dir/configs/notreks_benchmark/generated_v1/scenario_manifest.csv}"
results_root="${2:-$repo_dir/results/output}"
output_dir="${3:-$repo_dir/results/dagma_notreks_oracle/main_benchmark}"
python_bin="${NOTREKS_PYTHON:-python}"

cd "$repo_dir"
mkdir -p "$output_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" scripts/notreks_benchmark.py collect \
  --manifest "$manifest" \
  --results-root "$results_root" \
  --output "$output_dir/all_runs.csv"
"$python_bin" scripts/notreks_benchmark.py analyse \
  --results-csv "$output_dir/all_runs.csv" \
  --output-dir "$output_dir/analysis"

echo "Analysis complete: $output_dir/analysis/REPORT.md"
cat "$output_dir/analysis/REPORT.md"
