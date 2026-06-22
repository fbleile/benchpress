# NOTREKS SLURM Workflow

Start with [HOWTO_CLUSTER_BOOTSTRAP.md](HOWTO_CLUSTER_BOOTSTRAP.md) on a new
cluster checkout. That guide covers GitHub SSH, cloning, branch checkout,
module inspection, environment creation, tests, and the first smoke run.

## Smoke validation

Prepare a small smoke run:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  prepare-experiment \
  --grid workflow/rules/structure_learning_algorithms/notreks/configs/grids/slurm_smoke_grid.json \
  --out results/notreks_experiments/slurm_smoke
```

This creates `results/notreks_experiments/slurm_smoke/cmd.txt`, the command
file consumed by JobFarm.

Dry-run on the login node if cluster policy allows it:

```bash
snakemake -n \
  --cores 1 \
  --use-apptainer \
  --snakefile workflow/Snakefile \
  --configfile results/notreks_experiments/slurm_smoke/configs/validation_hparam_config.json
```

Submit:

```bash
RUN_DIR=results/notreks_experiments/slurm_smoke \
CONFIG=results/notreks_experiments/slurm_smoke/configs/validation_hparam_config.json \
CMD_FILE=results/notreks_experiments/slurm_smoke/cmd.txt \
FRESH=1 \
sbatch workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_jobfarm.sh
```

The SLURM header sends notifications to `f.bleile@tum.de` by default. Change
the `#SBATCH --mail-user=...` line in
`workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_jobfarm.sh`
if a different notification address is needed.

The wrapper also supports an explicit command list:

```bash
CMD_FILE=results/notreks_experiments/slurm_smoke/cmd.txt \
FRESH=1 \
sbatch workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_jobfarm.sh
```

Monitor:

```bash
squeue -u "$USER"
tail -f results/notreks_experiments/slurm_smoke/logs/slurm/*.out
```

## Selection after validation

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  select-validation-best \
  --run-dir results/notreks_experiments/slurm_smoke \
  --primary-metric SHD_cpdag
```

Write the final benchmark config:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  write-final-config \
  --run-dir results/notreks_experiments/slurm_smoke
```

## True validation run

Until the grid-file interface is enabled, use the validation preset that matches
the intended scale:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  prepare-experiment \
  --grid workflow/rules/structure_learning_algorithms/notreks/configs/grids/slurm_true_grid.json \
  --out results/notreks_experiments/slurm_true
```

Then dry-run and submit with the same pattern as the smoke run, replacing
`slurm_smoke` with `slurm_true`.

## Resuming and rerunning

Use `FRESH=0` to preserve JobFarm state where possible. Use `FRESH=1` only when
you deliberately want to reset the JobFarm database/log state for the run.

Avoid changing manifests behind existing short algorithm IDs. If the grid
changes, use a fresh run directory.
