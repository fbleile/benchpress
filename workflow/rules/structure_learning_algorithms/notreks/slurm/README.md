# NOTREKS SLURM Scripts

Start with
[../docs/HOWTO_CLUSTER_BOOTSTRAP.md](../docs/HOWTO_CLUSTER_BOOTSTRAP.md) on a
new cluster checkout.

The main wrapper is:

```text
workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_jobfarm.sh
```

Smoke submission example:

```bash
RUN_DIR=results/notreks_experiments/slurm_smoke \
CONFIG=results/notreks_experiments/slurm_smoke/configs/validation_hparam_config.json \
CMD_FILE=results/notreks_experiments/slurm_smoke/cmd.txt \
FRESH=1 \
sbatch workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_jobfarm.sh
```

The default SLURM notification email in the script header is
`f.bleile@tum.de`. Change the `#SBATCH --mail-user=...` line if needed.

Set these variables as needed:

```bash
REPO_DIR=/path/to/benchpress
CONDA_ENV=benchpress-notreks
CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh
RUN_DIR=results/notreks_experiments/slurm_smoke
CONFIG=results/notreks_experiments/slurm_smoke/configs/validation_hparam_config.json
CMD_FILE=results/notreks_experiments/slurm_smoke/cmd.txt
FRESH=0
```

Use `FRESH=1` only when you intentionally want to reset JobFarm state for that
run.
