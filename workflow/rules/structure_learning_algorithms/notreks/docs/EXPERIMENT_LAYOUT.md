# NOTREKS Experiment Layout

For first-time cluster setup, see
[HOWTO_CLUSTER_BOOTSTRAP.md](HOWTO_CLUSTER_BOOTSTRAP.md). For SLURM command
examples, see [HOWTO_SLURM.md](HOWTO_SLURM.md).

All NOTREKS-specific source tooling lives under:

```text
workflow/rules/structure_learning_algorithms/notreks/
```

Generated experiment artifacts should live under:

```text
results/notreks_experiments/<run_name>/
```

Recommended run names:

```text
slurm_smoke
slurm_true
```

## Source files

```text
workflow/rules/structure_learning_algorithms/notreks/
  configs/environment/       conda environment files
  docs/                      local, SLURM, bootstrap, and layout docs
  slurm/                     SLURM Snakemake driver scripts
  tools/                     config generation, selection, cache helpers
  tests/                     NOTREKS tests
```

Never delete source files as part of experiment cleanup.

## Generated run files

Typical generated validation run:

```text
results/notreks_experiments/<run_name>/
  configs/
    validation_hparam_config.json
    validation_hparam_manifest.csv
    validation_hparam_manifest.json
    final_benchmark_config.json
  cmd.txt                         legacy/local command record
  independence_cache/
  selection/
    best_by_method_family.json
    validation_summary.csv
  logs/
    slurm/
  reports/
```

Benchpress also writes normal outputs under repository-level `results/` paths,
including:

```text
results/adjmat_estimate/
results/time/
results/ntests/
results/result/
results/output/<run_name>_validation/
results/output/<run_name>_final/
```

The validation manifest maps short algorithm IDs, such as `notreks__grid003`,
back to full hyperparameters. Keep the manifest with the generated config.

SLURM logs should be written under the run directory, for example:

```text
results/notreks_experiments/slurm_smoke/logs/slurm/
results/notreks_experiments/slurm_true/logs/slurm/
```

The SLURM workflow uses one driver job running one Snakemake process. The
generated `cmd.txt` is useful for inspection and local debugging, but it is not
the active JobFarm input for validation runs.

## Fixed data

The current generator uses Benchpress fixed-data conventions under:

```text
resources/data/mydatasets/
resources/adjmat/myadjmats/
```

These are reproducible generated resources for the run. Do not edit them by
hand. If a grid changes, prefer a fresh run directory and fresh fixed-data IDs.

## Independence cache

Run-local NOTREKS independence-test caches live under:

```text
results/notreks_experiments/<run_name>/independence_cache/
```

Cache entries store raw pairwise test results and derived accepted pairs. The
cache is safe to delete if you are willing to recompute independence tests.

## Safe-to-delete generated folders

After confirming you no longer need a run, these generated folders can be
deleted:

```text
results/notreks_experiments/<run_name>/
results/output/<run_name>_validation/
results/output/<run_name>_final/
```

Only delete `resources/data/mydatasets/...` and `resources/adjmat/myadjmats/...`
entries when you are sure they belong to the run and are reproducible.
