# NOTREKS Local and SLURM Workflow

Run commands from the Benchpress repository root unless noted otherwise.

For a fresh cluster setup, first read
[HOWTO_CLUSTER_BOOTSTRAP.md](HOWTO_CLUSTER_BOOTSTRAP.md). For the concise SLURM
submission workflow, see [HOWTO_SLURM.md](HOWTO_SLURM.md). For generated file
locations and cleanup boundaries, see
[EXPERIMENT_LAYOUT.md](EXPERIMENT_LAYOUT.md).

## 1. Environment setup

Local:

```bash
conda activate benchpress-notreks
```

The environment needs Benchpress, Snakemake, NumPy, pandas, SciPy, and JAX for
the current matrix-function trek gradients. `hyppo` is optional and only needed
for HSIC and dCor independence tests.

Create a compatible environment independently on the Linux cluster. Do not copy
a macOS conda environment directory to Linux. If an environment file is
available, create the cluster environment from that file; otherwise install the
same packages into a cluster-created `benchpress-notreks` environment.

Benchpress must be available on the cluster checkout because data generation
and evaluation use the normal Benchpress workflow.

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

Experimental SCC power-iteration DAG penalty smoke:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  smoke --preset tiny --dag-seq scc_power_iteration
```

The default output is `results/notreks_smoke/`. On macOS the CLI enables the
local container-check bypass for `container: None` development. On Linux it
leaves the normal Apptainer/Singularity check enabled.

## 3. Prepare a hyperparameter run

For the current validation/final workflow, prefer a single validation config
containing all validation datasets and all method/hyperparameter variants:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  prepare-validation \
  --preset tiny \
  --out results/notreks_validation_tiny
```

The `tiny` preset creates two validation fixed datasets with `d=10, n=500` and
two final fixed datasets with `d=20, n=500`. It includes a very small method
grid for gCastle PC, gCastle DirectLiNGAM, and NOTREKS, and is intended only to
verify config generation, Benchpress dry runs, selection, and final-config
writing.

The `local10` preset creates ten validation fixed datasets with `d=10, n=500`
and ten final fixed datasets with `d=20, n=500`, using different seeds. It
generates one validation config:

```text
results/<run_name>/configs/validation_hparam_config.json
```

and method-family manifests:

```text
results/<run_name>/configs/validation_hparam_manifest.csv
results/<run_name>/configs/validation_hparam_manifest.json
```

The validation config contains:

- all validation fixed-data references;
- `gcastle_pc__grid...` entries for gCastle PC alpha variants;
- `gcastle_lingam__grid...` entries for gCastle DirectLiNGAM variants;
- `notreks__grid...` entries for NOTREKS variants.

NOTREKS `power_iter_steps` is only expanded for
`dag_seq="scc_power_iteration"`, so logdet configs are not duplicated by an
irrelevant power-iteration setting.

Dry-run the one-config validation benchmark locally:

```bash
BENCHPRESS_SKIP_CONTAINER_CHECK=1 snakemake -n \
  --cores all \
  --snakefile workflow/Snakefile \
  --configfile results/notreks_validation_tiny/configs/validation_hparam_config.json
```

On Linux with containers available, use the normal Benchpress container path:

```bash
snakemake -n \
  --cores all \
  --use-apptainer \
  --snakefile workflow/Snakefile \
  --configfile results/notreks_validation_tiny/configs/validation_hparam_config.json
```

The dry run should show separate planned jobs for method/data/hyperparameter
combinations; Snakemake can parallelize those jobs from this one config file.

After the validation benchmark has produced `joint_benchmarks.csv`, select one
best setting per method family:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  select-validation-best \
  --run-dir results/notreks_validation_tiny \
  --primary-metric SHD_cpdag
```

This writes:

```text
results/<run_name>/selection/best_by_method_family.json
results/<run_name>/selection/validation_summary.csv
```

Then generate the final benchmark config with exactly one selected setting for
each method family:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  write-final-config \
  --run-dir results/notreks_validation_tiny
```

The final config is written to:

```text
results/<run_name>/configs/final_benchmark_config.json
```

It contains the final fixed datasets and exactly one selected gCastle PC,
gCastle DirectLiNGAM, and NOTREKS configuration.

### Legacy per-template hparam command

A small example meta-config is included at `config/notreks_hparam_grid.json`:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  prepare-hparam \
  --grid-config config/notreks_hparam_grid.json \
  --out results/notreks_hparam_run \
  --grid-mode cartesian
```

Use `--grid-mode zip` to zip list-valued fields. Zip mode requires every
list-valued field in a method template to have the same length.

The input is a NOTREKS meta-config, not a direct Snakemake config. List-valued
fields are expanded into scalar Benchpress algorithm entries, and each original
method template gets one expanded Benchpress config. Expanded IDs are traceable
as `template-id__grid000`, `template-id__grid001`, and so on.

The run directory is intentionally self-describing:

```text
results/notreks_hparam_run/
  run_info.json
  fixed_data/
    metadata.json
  expanded_configs/
    <template_id>.json
  manifest.csv
  cmd.txt
  grid_index.json
  independence_cache/
    <cache_key>/
      metadata.json
      all_test_results.csv
      accepted_pairs.csv
  jobs/
    000_<template_id>/
      status.json
      stdout.log
      stderr.log
      joint_benchmarks.csv
      ROC_data.csv
  summaries/
```

`manifest.csv` stores run-relative paths for configs, status, logs, and copied
Benchpress result files. It also stores the native Benchpress output locations
as repository-relative paths so a job can copy results back into the movable
run folder after Snakemake finishes.

## 4. Fixed data

The preparation step creates shared fixed data using Benchpress conventions:

```text
resources/data/mydatasets/notreks_hparam/<run_name>/
resources/adjmat/myadjmats/notreks_hparam_<run_name>.csv
```

Every expanded config in the same run references those exact fixed-data
resources. The data are generated once for the run, not once per hyperparameter
setting. Re-preparing the same run name replaces only that run's generated
NOTREKS fixed-data folder.

## 5. Independence-test caching

Pairwise independence tests can dominate hyperparameter searches because many
NOTREKS settings reuse the same dataset and the same raw independence test.
Generated hyperparameter configs therefore include an `independence_cache_dir`
pointing at:

```text
results/<run_name>/independence_cache/
```

Each cache key includes:

- a SHA-256 hash of the numeric dataset values;
- dataset path and file hash when available;
- `n`, `d`, and column names;
- raw independence test name;
- extra test parameters when supplied;
- the NOTREKS cache implementation version.

Alpha and multiple-testing correction are not part of the raw-test cache key.
The cache stores raw test statistics and p-values. NOTREKS then recomputes
accepted marginal independence pairs for each alpha/correction setting.

Each cache entry stores:

```text
metadata.json
all_test_results.csv
accepted_pairs.csv
```

`all_test_results.csv` contains one row per tested pair with `i`, `j`,
statistic, p-value, optional degrees of freedom, raw test name, alpha used after
correction, and the final `accepted_independence` decision. The first run for a
dataset/raw-test setting is a cache miss and writes the entry. Later runs with
the same dataset/raw-test setting are cache hits, even when alpha or correction
changes. A different test type, data seed, dimension, sample size, or data hash
creates a different entry.

### gCastle-backed marginal independence tests

NOTREKS can reuse gCastle's low-level `castle.common.independence_tests.CITest`
functions:

- `gcastle_fisherz` -> `CITest.fisherz_test`
- `gcastle_g2` -> `CITest.g2_test`
- `gcastle_chi2` -> `CITest.chi2_test`

NOTREKS calls these functions with an empty conditioning set (`z=[]`) to test
marginal independence. This is not a full PC run and does not use the PC output
graph. gCastle returns raw p-values/statistics; NOTREKS applies Bonferroni,
Benjamini-Hochberg, or no correction itself. `gcastle_fisherz` is appropriate
for continuous approximately Gaussian data. `gcastle_g2` and `gcastle_chi2` are
more appropriate for discrete data.

You can precompute one cache entry directly:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  precompute-independencies \
  --data-csv path/to/data.csv \
  --cache-dir results/notreks_hparam_run/independence_cache \
  --method spearman \
  --alpha 0.05 \
  --correction benjamini-hochberg
```

Delete a specific `independence_cache/<cache_key>/` directory to invalidate one
entry, or delete the whole `independence_cache/` directory to force all tests to
recompute.

## 6. Ground-truth no-trek diagnostic

When a ground-truth graph is available to the helper functions, accepted
independence pairs can be compared with graph-implied no-trek marginal
independencies. A pair `(i, j)` is graph-implied no-trek independent when the
ground-truth graph contains no trek connecting the variables. The diagnostic
reports:

```text
num_true_no_trek_pairs
num_tested_pairs
num_accepted_pairs
true_positive_no_trek_pairs
false_positive_pairs
false_negative_no_trek_pairs
precision
recall
f1
```

Interpret this as graph-implied no-trek marginal independence, not all true
statistical marginal independencies. In linear Gaussian SEMs with independent
errors, no-trek is the relevant structural criterion for generic covariance
independence. In nonlinear or non-Gaussian data, and under special parameter
cancellations, it is a structural diagnostic rather than a complete statistical
truth label.

## 7. Run hyperparameter jobs locally

The generated command file is ordinary shell:

```bash
bash results/notreks_hparam_run/cmd.txt
```

Each command runs one expanded template config through normal Benchpress and
writes per-job status and logs under `jobs/<job_id>_<template_id>/`.

## 8. Run validation jobs on SLURM

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  print-slurm-launch \
  --run-dir results/notreks_experiments/slurm_smoke \
  --preset smoke
```

Smoke and true validation runs use the same SLURM mechanism: one driver job
starts one Snakemake process, and Snakemake parallelizes Benchpress jobs. The
smoke run only uses fewer datasets, fewer method variants, and smaller LRZ
resources.

The smoke preset uses LRZ `serial_std` with 8 cores. The true run defaults to
`serial_std` with 16 cores, with an optional `cm4_tiny` 32-core script when more
single-node parallelism is justified. Do not start many concurrent Snakemake
processes from JobFarm in one checkout; that can conflict over `.snakemake`
locks and metadata.

## 9. Select best hyperparameters

Default selection minimizes CPDAG SHD when available and uses a secondary
skeleton-F1-like metric if available:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  select-best \
  --hparam-run results/notreks_hparam_run
```

The default output is
`results/notreks_hparam_run/summaries/selected_best.json`, with a matching
`selected_best.csv`.

Custom primary and secondary metrics:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  select-best \
  --hparam-run results/notreks_hparam_run \
  --primary-metric SHD_pattern_mean \
  --primary-direction min \
  --secondary-metric TPR_skel_mean \
  --secondary-direction max \
  --out results/notreks_hparam_run/summaries/selected_best.json
```

Selection reads the copied per-job `joint_benchmarks.csv` files inside the run
folder. If a requested metric is unavailable, the command fails and prints the
available metric columns. It does not silently fall back to local directed
adjacency Hamming distance.

Benchpress `ROC_data.csv` summary columns are produced by
`workflow/rules/evaluation/benchmarks/combine_ROC_data.R`. In that file, `q1`
and `q3` are the 5% and 95% empirical quantiles, not the 25% and 75%
quartiles.

## 10. Inject selected settings into a final benchmark

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  inject-best \
  --benchmark-config config/your_final_benchmark_template.json \
  --selected results/notreks_hparam_run/summaries/selected_best.json \
  --out config/final_benchmark_with_best_notreks.json
```

Only `resources.structure_learning_algorithms.notreks` and matching NOTREKS IDs
in evaluation lists are replaced. Non-NOTREKS algorithms are preserved. The
final benchmark template is intentionally not generated automatically: its
broader data regimes, seeds, and external baselines are an experimental design
choice rather than a NOTREKS tooling default.

## 11. Run the final benchmark

For a small final benchmark, use normal Benchpress:

```bash
BENCHPRESS_SKIP_CONTAINER_CHECK=1 snakemake \
  --snakefile workflow/Snakefile \
  --cores 1 \
  --configfile config/final_benchmark_with_best_notreks.json
```

The bypass is only for local macOS development. Linux/SLURM benchmark runs
should use the normal Apptainer/Singularity setup.

For large final benchmarks, use the same single-Snakemake SLURM driver pattern
as validation, with a final benchmark config passed through `CONFIG=...`.

## 12. Tune/test separation

Use a small, basic validation setting in the hyperparameter meta-config. Choose
one setting per method template there, then inject those settings into a
broader final benchmark with different seeds and, where appropriate, different
dimensions, sample sizes, graph families, or data-generating regimes.

Do not select hyperparameters on final benchmark results.

## 13. DAG constraints and references

NOTREKS currently exposes these DAG constraints:

### `dag_seq="exp"`

This is the NOTEARS exponential-trace acyclicity constraint:

```text
h_exp(W) = trace(expm(W * W)) - d
```

Reference:

```bibtex
@article{zheng2018dags,
  title={Dags with no tears: Continuous optimization for structure learning},
  author={Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep K and Xing, Eric P},
  journal={Advances in neural information processing systems},
  volume={31},
  year={2018}
}
```

Code reference: <https://github.com/xunzheng/notears>

### `dag_seq="logdet"`

This is the DAGMA M-matrix/log-det acyclicity barrier:

```text
h_logdet(W; s) = -logdet(s * I - W * W) + d * log(s)
```

Reference:

```bibtex
@article{bello2022dagma,
  title={Dagma: Learning dags via m-matrices and a log-determinant acyclicity characterization},
  author={Bello, Kevin and Aragam, Bryon and Ravikumar, Pradeep},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={8226--8239},
  year={2022}
}
```

Code reference: <https://github.com/kevinsbello/dagma>

### `dag_seq="scc_power_iteration"`

`dag_seq = "scc_power_iteration"` adds an optional experimental SDCD-style
acyclicity surrogate. SDCD already uses a nonnegative adjacency proxy; for
signed linear NOTREKS weights the analogue is:

```text
A = W * W
A = A * offdiag_mask
```

The implementation detects nontrivial strongly connected components from
`A > scc_threshold`, then approximates left/right Perron vectors within each
SCC block with fixed-step power iteration:

```text
v <- normalize(A_scc @ v + eps)
u <- normalize(A_scc.T @ u + eps)
G_scc = outer(u, v) / (dot(u, v) + eps)
h(W) = sum(stop_gradient(G) * A)
```

SCC detection is graph-structural and is not differentiated through. The
default `power_iter_steps` is 5, chosen to keep the value/gradient call
reasonably close to logdet in small timing checks. The old experimental names
`power_iteration` and `spectral_radius` are not accepted; use only
`scc_power_iteration`. This leaves the existing `dag_seq = "exp"` and
`dag_seq = "logdet"` behavior unchanged. The branch is experimental until
larger benchmarks validate it.

Reference:

```bibtex
@article{nazaret2023stable,
  title={Stable differentiable causal discovery},
  author={Nazaret, Achille and Hong, Justin and Azizi, Elham and Blei, David},
  journal={arXiv preprint arXiv:2311.10263},
  year={2023}
}
```

Code reference: <https://github.com/azizilab/sdcd/tree/master>

Example algorithm field:

```json
{
  "dag_seq": "scc_power_iteration",
  "dag_reg": 1.0,
  "power_iter_steps": 5,
  "scc_threshold": 1e-8
}
```

## 14. SLURM readiness status

The current NOTREKS-local workflow keeps tooling under the module directory,
uses Benchpress fixed-data resources for shared datasets, writes run-relative
manifest paths where possible, and uses one Snakemake driver process on SLURM.
Independence caching is enabled for generated hyperparameter configs.

Before a first SLURM smoke, verify:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tests/run_tests.py
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py smoke --preset tiny
```

## 15. Troubleshooting

Wrong working directory:
: Run commands from the Benchpress repository root so relative config and
  resource paths resolve as expected.

SLURM config missing:
: Re-run `prepare-experiment` or `prepare-validation`. The Snakemake driver
  fails before loading the environment when `CONFIG` is missing.

Conda environment not found:
: Set `CONDA_ENV` and, if needed, `MICROMAMBA_BIN` when submitting.

Fixed data missing:
: Re-run `prepare-hparam` from the Benchpress repository root. Check
  `fixed_data/metadata.json` for the exact resource paths.

Metric not found:
: Confirm Benchpress evaluation completed and inspect the columns printed by
  `select-best`. Use explicit `--primary-direction` for ambiguous metrics.

Failed jobs:
: Inspect `jobs/<job>/status.json`, `stdout.log`, and `stderr.log`. Run the
  corresponding line from `cmd.txt` locally for debugging.

Resume Snakemake:
: Re-submit the same driver command after fixing the failure. Snakemake reuses
  valid completed outputs.
