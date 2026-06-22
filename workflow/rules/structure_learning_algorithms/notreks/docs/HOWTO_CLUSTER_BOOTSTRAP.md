# NOTREKS Cluster Bootstrap

This guide gets a clean SLURM checkout ready to run the Benchpress NOTREKS
pipeline. Run commands from the cluster unless explicitly noted.

## 1. Log in and prepare a workspace

```bash
ssh <user>@<cluster-host>
mkdir -p ~/projects
cd ~/projects
```

Replace `<user>` and `<cluster-host>` with the values for the target cluster.

## 2. Verify GitHub SSH access

```bash
ssh -T git@github.com
```

If GitHub does not recognize your key, create one on the cluster:

```bash
ssh-keygen -t ed25519 -C "fabian.bleile@posteo.de"
cat ~/.ssh/id_ed25519.pub
```

Add the printed public key to GitHub under **Settings > SSH and GPG keys**.
Then retry:

```bash
ssh -T git@github.com
```

## 3. Clone Benchpress and checkout the NOTREKS branch

```bash
git clone git@github.com:fbleile/benchpress.git
cd benchpress
git checkout add-notreks-module
git pull
```

## 4. Inspect cluster modules

Cluster module names vary. Inspect what is available before assuming module
names:

```bash
module avail
module avail anaconda
module avail miniconda
module avail apptainer
module avail squashfs
module avail singularity
module avail R
```

Load the matching modules for the cluster. Examples:

```bash
module load anaconda
module load apptainer
module load squashfs
```

or:

```bash
module load miniconda
module load singularity
```

Use the actual module names reported by `module avail`.

On LRZ, Apptainer image pulls require `mksquashfs`. Load `squashfs/4.6.1`
together with `apptainer/1.3.4`. The NOTREKS SLURM driver does this
automatically, but this manual diagnostic is useful before submitting:

```bash
module load apptainer/1.3.4
module load squashfs/4.6.1
which apptainer
apptainer --version
which mksquashfs
mksquashfs -version
```

## 5. Create the conda/mamba environment

Do not copy a macOS conda environment directory to Linux. Create the environment
on the cluster from the repository file:

```bash
conda env create -f workflow/rules/structure_learning_algorithms/notreks/configs/environment/benchpress-notreks.yml
conda activate benchpress-notreks
```

If mamba is available, it is usually faster:

```bash
mamba env create -f workflow/rules/structure_learning_algorithms/notreks/configs/environment/benchpress-notreks.yml
conda activate benchpress-notreks
```

Anaconda Cloud backup can be useful as a reference, but the repository
environment file is the reproducible source of truth. If package solving fails,
save the exact error and update the environment file.

## 6. Verify the NOTREKS environment

```bash
python workflow/rules/structure_learning_algorithms/notreks/tests/run_tests.py
```

Expected result:

```text
All NOTREKS tests passed
```

Also verify the container runtime for final cluster runs:

```bash
module load apptainer/1.3.4
module load squashfs/4.6.1
apptainer --version
mksquashfs -version
```

If Apptainer or `mksquashfs` is unavailable, stop and fix the cluster
environment before running the final benchmark. Without `mksquashfs`, Apptainer
cannot convert Docker images to SIF on LRZ.

## 7. Prepare the SLURM smoke experiment

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  prepare-experiment \
  --grid workflow/rules/structure_learning_algorithms/notreks/configs/grids/slurm_smoke_grid.json \
  --out results/notreks_experiments/slurm_smoke
```

This writes:

```text
results/notreks_experiments/slurm_smoke/configs/validation_hparam_config.json
results/notreks_experiments/slurm_smoke/configs/validation_hparam_manifest.csv
results/notreks_experiments/slurm_smoke/configs/validation_hparam_manifest.json
results/notreks_experiments/slurm_smoke/cmd.txt
```

`cmd.txt` is kept as a readable local command record. The SLURM submission path
uses the Snakemake driver scripts below, not a JobFarm command list.

## 8. Dry-run the smoke config

On the cluster, use the normal Benchpress container path:

```bash
snakemake -n \
  --cores 1 \
  --use-apptainer \
  --snakefile workflow/Snakefile \
  --configfile results/notreks_experiments/slurm_smoke/configs/validation_hparam_config.json
```

If the cluster uses Singularity instead of Apptainer, use the Benchpress
container option supported by that checkout.

## 9. Submit the SLURM smoke

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

The job script currently sets SLURM mail notifications to
`f.bleile@tum.de`. Edit the `#SBATCH --mail-user=...` header in
`workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_serial_smoke.sh`
if the cluster run should notify a different address.

Smoke and true use the same mechanism: one SLURM job starts one Snakemake
process, and Snakemake parallelizes Benchpress jobs. The smoke grid only uses
fewer datasets, fewer hyperparameter variants, and smaller SLURM resources.
This avoids `.snakemake` lock and metadata conflicts from running many
independent Snakemake processes at once.

The script is configurable through environment variables. The most important
ones are:

```bash
REPO_DIR=/path/to/benchpress
CONDA_ENV=benchpress-notreks
MICROMAMBA_BIN=$HOME/bin/micromamba
RUN_DIR=results/notreks_experiments/slurm_smoke
CONFIG=results/notreks_experiments/slurm_smoke/configs/validation_hparam_config.json
SNAKEMAKE_CORES=8
```

LRZ resource guidance: smoke uses `serial_std` with 8 cores. The true grid can
use `serial_std` with 16 cores, or `cm4_tiny` with 32 cores if more local
parallelism is justified. Do not use `cm4_std` or multi-node allocations for
the smoke run.

## 10. Monitor the smoke run

```bash
squeue -u "$USER" --clusters=serial,cm4
tail -f results/notreks_experiments/slurm_smoke/logs/slurm/*.out
```

Cancel/check commands:

```bash
scancel --clusters=serial <jobid>
scancel --clusters=cm4 <jobid>
scontrol -M serial show job <jobid>
scontrol -M cm4 show job <jobid>
```

If the run fails, keep the run directory and logs. Do not delete the fixed data,
manifest, or config before diagnosing the issue.

## 11. Select best settings after smoke validation

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  select-validation-best \
  --run-dir results/notreks_experiments/slurm_smoke \
  --primary-metric SHD_cpdag
```

Then write the final benchmark config:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  write-final-config \
  --run-dir results/notreks_experiments/slurm_smoke
```

## 12. Environment troubleshooting

- Prefer the repository `benchpress-notreks.yml` over Anaconda Cloud backup for
  reproducibility. Anaconda Cloud can help identify package versions, but it is
  not the primary setup path.
- If `conda env create` fails, try `mamba env create` with the same file.
- If a Python package is unavailable, record the exact solver or pip error and
  update `configs/environment/benchpress-notreks.yml`.
- If R packages are missing, check whether the cluster provides R modules or
  whether the conda R packages need adjustment. Benchpress evaluation uses R.
- If Apptainer/Singularity is unavailable, stop. Final cluster benchmark runs
  should use real containers.
- If JAX tries to use an unavailable accelerator, force CPU mode:

  ```bash
  export JAX_PLATFORM_NAME=cpu
  ```

- If gCastle import fails, verify `gcastle` was installed in the active
  `benchpress-notreks` environment:

  ```bash
  python -c "import castle; print('castle ok')"
  ```

- If Snakemake behavior differs from local runs, check the installed Snakemake
  version and pin it in the environment file if needed.
