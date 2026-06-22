# NOTREKS SLURM Scripts

Start with
[../docs/HOWTO_CLUSTER_BOOTSTRAP.md](../docs/HOWTO_CLUSTER_BOOTSTRAP.md) on a
new cluster checkout.

The active launch path is one SLURM driver job running one Snakemake process.
Smoke and true runs use the same mechanism; only grid size and requested
resources differ.

Driver scripts:

```text
workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_serial_smoke.sh
workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_serial_true.sh
workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_cm4_tiny_true.sh
```

Smoke submission example:

```bash
mkdir -p results/notreks_experiments/slurm_smoke/logs/slurm
RUN_DIR=results/notreks_experiments/slurm_smoke \
CONFIG=results/notreks_experiments/slurm_smoke/configs/validation_hparam_config.json \
SNAKEMAKE_CORES=8 \
sbatch --clusters=serial \
  -o results/notreks_experiments/slurm_smoke/logs/slurm/%x-%j.out \
  -e results/notreks_experiments/slurm_smoke/logs/slurm/%x-%j.err \
  workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_serial_smoke.sh
```

True validation on `serial_std`:

```bash
RUN_DIR=results/notreks_experiments/slurm_true \
CONFIG=results/notreks_experiments/slurm_true/configs/validation_hparam_config.json \
SNAKEMAKE_CORES=16 \
sbatch --clusters=serial \
  -o results/notreks_experiments/slurm_true/logs/slurm/%x-%j.out \
  -e results/notreks_experiments/slurm_true/logs/slurm/%x-%j.err \
  workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_serial_true.sh
```

True validation on `cm4_tiny`:

```bash
RUN_DIR=results/notreks_experiments/slurm_true \
CONFIG=results/notreks_experiments/slurm_true/configs/validation_hparam_config.json \
SNAKEMAKE_CORES=32 \
sbatch --clusters=cm4 \
  -o results/notreks_experiments/slurm_true/logs/slurm/%x-%j.out \
  -e results/notreks_experiments/slurm_true/logs/slurm/%x-%j.err \
  workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_cm4_tiny_true.sh
```

The default SLURM notification email in the script headers is
`f.bleile@tum.de`. Change the `#SBATCH --mail-user=...` line if needed.

Set these variables as needed:

```bash
REPO_DIR=/path/to/benchpress
CONDA_ENV=benchpress-notreks
MICROMAMBA_BIN=$HOME/bin/micromamba
RUN_DIR=results/notreks_experiments/slurm_smoke
CONFIG=results/notreks_experiments/slurm_smoke/configs/validation_hparam_config.json
SNAKEMAKE_CORES=8
```

`notreks_jobfarm.sh` is kept only as a deprecated compatibility wrapper and now
delegates to the same single-Snakemake driver. Do not use JobFarm command lists
for NOTREKS validation runs; multiple concurrent Snakemake processes in one
checkout can conflict over locks and metadata.
