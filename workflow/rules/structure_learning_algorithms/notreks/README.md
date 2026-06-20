# NOTREKS Benchpress Module

This directory contains the self-contained Benchpress implementation of
NOTREKS and its local/SLURM experiment tooling.

The optimizer is used unchanged by all execution modes. The tooling under
`tools/` prepares ordinary Benchpress configs, shared fixed datasets, JobFarm
manifests, and selected final configs.

Quick local check:

```bash
conda activate benchpress-notreks
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  smoke --preset tiny
```

See [docs/HOWTO_LOCAL_AND_SLURM.md](docs/HOWTO_LOCAL_AND_SLURM.md) for the
complete tuning, selection, injection, local, and JobFarm workflow. For a fresh
cluster checkout, start with
[docs/HOWTO_CLUSTER_BOOTSTRAP.md](docs/HOWTO_CLUSTER_BOOTSTRAP.md), then use
[docs/HOWTO_SLURM.md](docs/HOWTO_SLURM.md). The generated-file map is in
[docs/EXPERIMENT_LAYOUT.md](docs/EXPERIMENT_LAYOUT.md).

Current validation/final workflow:

```bash
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  prepare-validation --preset tiny --out results/notreks_validation_tiny
```

This writes one Benchpress validation config containing all validation fixed
datasets and all gCastle PC, gCastle DirectLiNGAM, and NOTREKS hyperparameter
variants. After Benchpress evaluation, use `select-validation-best` to choose
one setting per method family and `write-final-config` to write the final
benchmark config.

Current continuation policy: every central-path stage receives `max_iter`.
`warm_iter` remains accepted only for compatibility with older configs.

## DAG constraints

Accepted `dag_seq` names are:

- `dag_seq="exp"`: NOTEARS exponential-trace acyclicity constraint from Zheng,
  Aragam, Ravikumar, and Xing (2018), code reference
  <https://github.com/xunzheng/notears>. Rough formula:
  `trace(expm(W * W)) - d`.
- `dag_seq="logdet"`: DAGMA M-matrix/log-det acyclicity barrier from Bello,
  Aragam, and Ravikumar (2022), code reference
  <https://github.com/kevinsbello/dagma>. Rough formula:
  `-logdet(s * I - W * W) + d * log(s)`.
- `dag_seq="scc_power_iteration"`: experimental SCC-blockwise power-iteration
  surrogate inspired by Stable Differentiable Causal Discovery (SDCD) by
  Nazaret, Hong, Azizi, and Blei (2023), code reference
  <https://github.com/azizilab/sdcd/tree/master>. SDCD works with a
  nonnegative learned adjacency proxy. NOTREKS linear weights are signed, so
  this branch uses the smooth nonnegative NOTEARS-style proxy `A = W * W` with
  the diagonal masked out, computes SCCs from the support of `A`, and applies
  the SCC-blockwise detached Perron-gradient surrogate.

The obsolete experimental names `power_iteration` and `spectral_radius` are no
longer accepted. Use `dag_seq="scc_power_iteration"`.

## Independence-test cache

Generated hyperparameter configs include an `independence_cache_dir` pointing
to the run-local `independence_cache/` directory. Cache keys include the data
hash, data path and file hash when available, `n`, `d`, raw test name, columns,
extra test parameters, and cache implementation version. Alpha and
multiple-testing correction are intentionally not part of the raw-test cache
key; NOTREKS recomputes accepted pairs from cached raw p-values for each
alpha/correction setting. Each entry stores `metadata.json`,
`all_test_results.csv`, and `accepted_pairs.csv`.

`gcastle_fisherz`, `gcastle_g2`, and `gcastle_chi2` call gCastle's low-level
`CITest` functions with an empty conditioning set. These are marginal tests
used to build NOTREKS no-trek constraints, not full PC runs. Correction remains
inside NOTREKS. `gcastle_fisherz` is the continuous/Gaussian option; `gcastle_g2`
and `gcastle_chi2` are mainly for discrete data.

Ground-truth diagnostics, when available, are reported as graph-implied no-trek
marginal independence rather than all statistical marginal independencies.
