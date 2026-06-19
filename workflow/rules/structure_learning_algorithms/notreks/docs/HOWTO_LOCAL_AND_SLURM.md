# NOTREKS Local and SLURM Workflow

Run all commands from the Benchpress repository root.

## 1. Environment setup

Local:

```bash
conda activate benchpress-notreks
```

The environment needs Benchpress/Snakemake plus the NOTREKS Python
dependencies. `hyppo` is needed for HSIC and dCor independence tests.

Create a compatible environment independently on the Linux cluster. Do not
copy a macOS conda environment directory to Linux. If a project environment
file is available, create the cluster environment from that file; otherwise
install the same packages into a cluster-created `benchpress-notreks`
environment.

Benchpress itself must be available on the cluster checkout because graph/data
handling and evaluation use the normal Benchpress workflow.

## 2. Local smoke test

Tiny, one dataset, no-trek plus Spearman:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  smoke --preset tiny
```

Small, also including Pearson and dCor:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  smoke --preset small
```

The default output is `results/notreks_smoke/`. On macOS the CLI enables the
existing local container-check bypass. On Linux it leaves the normal
Apptainer/Singularity check enabled.

## 3. Prepare a hyperparameter run

A usable example is included at
`config/notreks_hparam_grid.json`:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  prepare-hparam \
  --grid-config config/notreks_hparam_grid.json \
  --out results/notreks_hparam_run \
  --grid-mode cartesian
```

Use `--grid-mode zip` to zip list-valued fields. Zip mode requires every list
to have the same length.

The input is a NOTREKS meta-config, not a direct Snakemake config. It can use a
short form:

```json
{
  "notreks_hparam": {
    "run_name": "notreks_basic_validation",
    "fixed_data": {
      "d": 10,
      "n_values": [500],
      "seeds": [101, 102, 103],
      "expected_degree": 2,
      "standardized": true
    }
  },
  "structure_learning_algorithms": {
    "notreks": [
      {
        "id": "notreks-exp-spearman",
        "function_class": "linear",
        "score": "least_squares",
        "dag_seq": "exp",
        "dag_reg": 1.0,
        "dag_s": 1.0,
        "trek_seq": "exp",
        "trek_reg": [0.1, 1.0, 10.0],
        "regularizer": "l1",
        "regularizer_scale": [0.001, 0.01],
        "independence_test": "spearman",
        "independence_alpha": [0.01, 0.05],
        "independence_correction": "benjamini-hochberg",
        "seed": 1,
        "max_iter": [30000, 60000],
        "lr": 0.0003,
        "path_steps": 5,
        "mu_init": 1.0,
        "mu_factor": 0.1,
        "warm_iter": 30000,
        "tol": 0.000001,
        "threshold": 0.1,
        "timeout": null
      }
    ]
  }
}
```

Preparation creates:

```text
results/notreks_hparam_run/
  expanded_configs/
  fixed_data/metadata.json
  grid_index.json
  manifest.csv
  cmd.txt
  status/
  logs/
```

Each original template gets one scalar Benchpress config containing all its
expanded settings. IDs are traceable as `template-id__grid000`, etc.

Shared data are stored using Benchpress conventions:

```text
resources/data/mydatasets/notreks_hparam/<run_name>/
resources/adjmat/myadjmats/notreks_hparam_<run_name>.csv
```

Every expanded config references those same files. Re-preparing the same run
name replaces only that run's generated fixed-data folder.

## 4. Run hyperparameter jobs locally

The generated command file is ordinary shell:

```bash
bash results/notreks_hparam_run/cmd.txt
```

Each command runs one template config through normal Benchpress/Snakemake and
writes status plus stdout/stderr logs.

## 5. Run hyperparameter jobs with SLURM JobFarm

```bash
mkdir -p slurm_logs

FRESH=1 \
REPO_DIR=/dss/dsshome1/0C/ge86xim2/benchpress \
CONDA_ENV=benchpress-notreks \
CMD_FILE=results/notreks_hparam_run/cmd.txt \
sbatch workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_jobfarm.sh
```

The SLURM script contains no benchmark logic. It executes the same `cmd.txt`
used locally. Set `CONDA_SH` if conda is installed somewhere other than
`$HOME/miniconda3`.

Use `FRESH=0` (the default) to preserve JobFarm's resumable database and result
state. Use `FRESH=1` only to intentionally start fresh.

## 6. Select best hyperparameters

After all jobs have produced Benchpress `joint_benchmarks.csv` files:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  select-best \
  --hparam-run results/notreks_hparam_run \
  --metric shd_cpdag \
  --out results/notreks_hparam_run/selected_best.json
```

Selection minimizes mean `SHD_cpdag` independently for every original method
template. It also writes `selected_best.csv`. If no CPDAG-SHD-like column is
present, the command fails and lists all available columns. It does not fall
back to directed adjacency Hamming distance.

## 7. Inject selected settings into a final benchmark

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  inject-best \
  --benchmark-config config/your_final_benchmark_template.json \
  --selected results/notreks_hparam_run/selected_best.json \
  --out config/final_benchmark_with_best_notreks.json
```

Only `resources.structure_learning_algorithms.notreks` and matching NOTREKS IDs
in evaluation lists are replaced. Non-NOTREKS algorithms are preserved.
The final benchmark template is intentionally not generated automatically:
its broader data regimes, seeds, and external baselines are an experimental
design choice rather than a NOTREKS tooling default.

## 8. Run the final benchmark

For a small final benchmark, use normal Benchpress:

```bash
BENCHPRESS_SKIP_CONTAINER_CHECK=1 snakemake \
  --snakefile workflow/Snakefile \
  --cores 1 \
  --configfile config/final_benchmark_with_best_notreks.json
```

The bypass is only for local macOS development. Linux/SLURM benchmark runs
should use the normal Apptainer/Singularity setup.

For a large final benchmark, generate an appropriate command list calling
normal Benchpress configs and submit it through the same JobFarm wrapper.

## 9. Tune/test separation

Use a small, basic validation setting in the hyperparameter meta-config.
Choose one setting per method template there, then inject those settings into a
broader final benchmark with different seeds and, where appropriate, different
dimensions, sample sizes, graph families, or data-generating regimes.

Do not select hyperparameters on the final benchmark results.

## 10. Troubleshooting

`cmd.txt` missing or empty:
: Re-run `prepare-hparam`. The SLURM script exits before loading JobFarm when
  the command file is missing or empty.

Conda environment not found:
: Set `CONDA_ENV` and, if needed, `CONDA_SH` when submitting.

Fixed data missing:
: Re-run `prepare-hparam` from the Benchpress repository root. Check
  `fixed_data/metadata.json` for the exact resource paths.

CPDAG SHD metric not found:
: Confirm the Benchpress evaluation completed and inspect the columns printed
  by `select-best`.

Failed jobs:
: Inspect `status/*.json` and `logs/*.err`. Run the corresponding line from
  `cmd.txt` locally for debugging.

Resume JobFarm:
: Submit with `FRESH=0` so existing JobFarm state is retained.
