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

This creates the validation config, manifests, fixed-data references, and a
legacy `cmd.txt` command file for local inspection. The SLURM path does not use
JobFarm command lists. Smoke and true runs both use one SLURM driver job that
starts exactly one Snakemake process; Snakemake then parallelizes Benchpress
jobs with the requested `--cores`.

Dry-run on the login node if cluster policy allows it:

```bash
snakemake -n \
  --cores 1 \
  --use-apptainer \
  --snakefile workflow/Snakefile \
  --configfile results/notreks_experiments/slurm_smoke/configs/validation_hparam_config.json
```

On LRZ, Apptainer image pulls require `mksquashfs`. The NOTREKS SLURM driver
loads both `apptainer/1.3.4` and `squashfs/4.6.1` before Snakemake starts. To
diagnose the module setup manually:

```bash
module load apptainer/1.3.4
module load squashfs/4.6.1
which apptainer
apptainer --version
which mksquashfs
mksquashfs -version
```

Snakemake/container Python compatibility: the Benchpress gCastle image
`docker://bpimages/gcastle:1.0.3` uses Python 3.7. Snakemake injects its Python
package into containerized `script:` jobs, so Snakemake 9 is incompatible with
that container. Use the pinned `snakemake=7.32.4` environment from
`benchpress-notreks.yml`. If this appears:

```text
TypeError: unsupported operand type(s) for |: 'NoneType' and 'type'
```

recreate or update the cluster environment from the repository environment
file.

Snakemake 7 uses the older container flag `--use-singularity`, while Snakemake
8/9 can use `--use-apptainer`. LRZ provides Apptainer. The NOTREKS SLURM driver
therefore creates a run-local `singularity -> apptainer` shim under
`results/notreks_experiments/<run>/bin/` when Snakemake 7 is active and no
`singularity` binary is already on `PATH`.

Print the exact submit command:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  print-slurm-launch \
  --run-dir results/notreks_experiments/slurm_smoke \
  --preset smoke
```

Submit the smoke run on LRZ `serial_std` with 8 cores:

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

The SLURM header sends notifications to `f.bleile@tum.de` by default. Change
the `#SBATCH --mail-user=...` line in the driver script if a different
notification address is needed.

We intentionally do not launch many concurrent `snakemake` processes from
JobFarm. Independent Snakemake processes in one checkout can fight over
`.snakemake` locks and metadata. The driver starts one Snakemake process and
lets Snakemake manage the DAG.


Monitor:

```bash
squeue -u "$USER" --clusters=serial,cm4
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

Then dry-run and submit with the same mechanism as the smoke run, replacing
`slurm_smoke` with `slurm_true`.

Default true run on `serial_std` with 16 cores:

```bash
mkdir -p results/notreks_experiments/slurm_true/logs/slurm
RUN_DIR=results/notreks_experiments/slurm_true \
CONFIG=results/notreks_experiments/slurm_true/configs/validation_hparam_config.json \
SNAKEMAKE_CORES=16 \
sbatch --clusters=serial \
  -o results/notreks_experiments/slurm_true/logs/slurm/%x-%j.out \
  -e results/notreks_experiments/slurm_true/logs/slurm/%x-%j.err \
  workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_serial_true.sh
```

If 16 cores are too slow and the workload fits one shared CM4 node, use
`cm4_tiny` with 32 cores:

```bash
RUN_DIR=results/notreks_experiments/slurm_true \
CONFIG=results/notreks_experiments/slurm_true/configs/validation_hparam_config.json \
SNAKEMAKE_CORES=32 \
sbatch --clusters=cm4 \
  -o results/notreks_experiments/slurm_true/logs/slurm/%x-%j.out \
  -e results/notreks_experiments/slurm_true/logs/slurm/%x-%j.err \
  workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_cm4_tiny_true.sh
```

LRZ resource guidance: use `serial_std` for 1-16 cores on one shared node,
`cm4_tiny` for 17-112 cores on one shared node, and reserve `cm4_std` for
intentional multi-node workloads. The NOTREKS smoke is not a multi-node job and
must not request 2 nodes or hundreds of CPUs.

## Resuming and rerunning

Rerun the same `sbatch` command after fixing the cause of a failure. Snakemake
will reuse completed outputs where valid. Do not change manifests behind
existing short algorithm IDs; if the grid changes, use a fresh run directory.

Useful SLURM commands:

```bash
squeue -u "$USER" --clusters=serial,cm4
scontrol -M serial show job <jobid>
scontrol -M cm4 show job <jobid>
scancel --clusters=serial <jobid>
scancel --clusters=cm4 <jobid>
```
