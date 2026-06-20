# NOTREKS Conda Environment

`benchpress-notreks.yml` is a Linux-oriented conda environment for running the
NOTREKS Benchpress workflow on a SLURM cluster. It is intentionally hand-written
rather than exported directly from the local macOS environment, so it avoids
macOS build strings and absolute `prefix:` entries.

Create it with conda:

```bash
conda env create -f workflow/rules/structure_learning_algorithms/notreks/configs/environment/benchpress-notreks.yml
conda activate benchpress-notreks
```

or with mamba:

```bash
mamba env create -f workflow/rules/structure_learning_algorithms/notreks/configs/environment/benchpress-notreks.yml
conda activate benchpress-notreks
```

The file includes:

- Python packages needed by the NOTREKS wrapper and tooling;
- Snakemake for running Benchpress;
- R and common R packages used by Benchpress evaluation;
- pip packages for gCastle, dCor/HSIC via `hyppo`, LiNGAM, pgmpy, CDT, and JAX CPU.

This environment file should be tested on the target cluster. If dependency
solving fails, record the exact package error and adjust this file rather than
copying a local macOS conda environment to Linux.

`benchpress-notreks-minimal.yml` currently mirrors the main file. Keep it as the
place for a smaller environment later if the full R/Python stack needs to be
split by cluster module policy.
