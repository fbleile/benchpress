# NOTREKS command sheet

This sheet assumes the LRZ cluster setup with Snakemake 7, Apptainer loaded as
Singularity-compatible backend, and `squashfs/4.6.1` available.

Use `--use-singularity` with Snakemake 7. The driver internally handles
Apptainer/Singularity compatibility.

Validation grids tune/select methods. Selected benchmarks use fresh benchmark
frames and selected methods only.

Pipeline:

```text
validation grid JSON -> expand-grid -> validation config + manifest
  -> local dry-run or SLURM run -> select-validation-best -> selection JSON
  -> build-selected-benchmark + fresh benchmark frame
  -> selected benchmark config + manifest -> local dry-run or SLURM run
```

Local dry-runs use `snakemake -n` to validate the DAG and do not run jobs.
SLURM runs actually execute the benchmark on LRZ.

## 0. Cluster setup

```bash
cd ~/benchpress
git pull --ff-only origin add-notreks-module

eval "$(~/bin/micromamba shell hook -s bash)"
micromamba activate benchpress-notreks

module load apptainer/1.3.4
module load squashfs/4.6.1

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## 1. Manual local example

```bash
python workflow/rules/structure_learning_algorithms/notreks/tests/manual_notreks_example.py
```

## 2. Smoke validation grid

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py expand-grid --tag smoke
```

```bash
snakemake -n \
  --cores 1 \
  --use-singularity \
  --snakefile workflow/Snakefile \
  --configfile configs/notreks/expanded/smoke_config.json
```

```bash
mkdir -p results/notreks/smoke/logs/slurm

RUN_DIR=results/notreks/smoke \
CONFIG=configs/notreks/expanded/smoke_config.json \
SNAKEMAKE_CORES=8 \
sbatch --clusters=serial \
  --export=ALL,RUN_DIR=results/notreks/smoke,CONFIG=configs/notreks/expanded/smoke_config.json,SNAKEMAKE_CORES=8 \
  -o results/notreks/smoke/logs/slurm/%x-%j.out \
  -e results/notreks/smoke/logs/slurm/%x-%j.err \
  workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_smoke.sh
```

## 3. Select best methods from smoke validation

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  select-validation-best --tag smoke --primary-metric SHD_pattern
```

## 4. Smoke selected benchmark

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  build-selected-benchmark --tag smoke
```

```bash
snakemake -n \
  --cores 1 \
  --use-singularity \
  --snakefile workflow/Snakefile \
  --configfile configs/notreks/expanded/selected_smoke_benchmark_config.json
```

```bash
mkdir -p results/notreks/benchmark_smoke/logs/slurm

RUN_DIR=results/notreks/benchmark_smoke \
CONFIG=configs/notreks/expanded/selected_smoke_benchmark_config.json \
SNAKEMAKE_CORES=8 \
sbatch --clusters=serial \
  --export=ALL,RUN_DIR=results/notreks/benchmark_smoke,CONFIG=configs/notreks/expanded/selected_smoke_benchmark_config.json,SNAKEMAKE_CORES=8 \
  -o results/notreks/benchmark_smoke/logs/slurm/%x-%j.out \
  -e results/notreks/benchmark_smoke/logs/slurm/%x-%j.err \
  workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_smoke.sh
```

## 5. Hyperparameter validation grid

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py expand-grid --tag hyperparam
```

```bash
snakemake -n \
  --cores 1 \
  --use-singularity \
  --snakefile workflow/Snakefile \
  --configfile configs/notreks/expanded/hyperparam_config.json
```

```bash
mkdir -p results/notreks/hyperparam/logs/slurm

RUN_DIR=results/notreks/hyperparam \
CONFIG=configs/notreks/expanded/hyperparam_config.json \
SNAKEMAKE_CORES=16 \
sbatch --clusters=serial \
  --export=ALL,RUN_DIR=results/notreks/hyperparam,CONFIG=configs/notreks/expanded/hyperparam_config.json,SNAKEMAKE_CORES=16 \
  -o results/notreks/hyperparam/logs/slurm/%x-%j.out \
  -e results/notreks/hyperparam/logs/slurm/%x-%j.err \
  workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_heavy.sh
```

## 6. Select best methods from hyperparameter validation

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  select-validation-best --tag hyperparam --primary-metric SHD_cpdag
```

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/hyperparam_analysis.py \
  --tag hyperparam
```

## 7. Full selected benchmark

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  build-selected-benchmark --tag full_benchmark
```

```bash
snakemake -n \
  --cores 1 \
  --use-singularity \
  --snakefile workflow/Snakefile \
  --configfile configs/notreks/expanded/selected_full_benchmark_config.json
```

```bash
mkdir -p results/notreks/benchmark_full/logs/slurm

RUN_DIR=results/notreks/benchmark_full \
CONFIG=configs/notreks/expanded/selected_full_benchmark_config.json \
SNAKEMAKE_CORES=16 \
sbatch --clusters=serial \
  --export=ALL,RUN_DIR=results/notreks/benchmark_full,CONFIG=configs/notreks/expanded/selected_full_benchmark_config.json,SNAKEMAKE_CORES=16 \
  -o results/notreks/benchmark_full/logs/slurm/%x-%j.out \
  -e results/notreks/benchmark_full/logs/slurm/%x-%j.err \
  workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_heavy.sh
```

## 8. Monitoring and logs

```bash
squeue -u $USER --clusters=serial,cm4
```

```bash
ls -t results/notreks/smoke/logs/slurm/*.out | head -1
tail -f "$(ls -t results/notreks/smoke/logs/slurm/*.out | head -1)"
```

```bash
ls -t results/notreks/benchmark_full/logs/slurm/*.out | head -1
tail -f "$(ls -t results/notreks/benchmark_full/logs/slurm/*.out | head -1)"
```

```bash
sacct -M serial \
  --starttime="$(date -d '1 day ago' '+%Y-%m-%dT%H:%M:%S')" \
  --endtime=now \
  --user=$USER \
  --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,MaxRSS,ReqCPUS,ReqMem
```

```bash
scancel --clusters=serial <JOBID>
scancel --clusters=cm4 <JOBID>
```

## 9. Cleanup

Warning: this deletes benchmark outputs and Snakemake state. Do not delete
`configs/notreks/...` unless intentionally resetting configs.

```bash
rm -rf results
rm -rf .snakemake
mkdir -p results
```
