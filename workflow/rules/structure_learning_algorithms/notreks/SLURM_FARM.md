# NOTREKS single-allocation farm

`notreks_driver_farm.sh` submits one Slurm allocation. It does not submit a
Slurm array: `farm.py` runs independent Benchpress/Snakemake configurations in
a standard-library worker pool, with one Snakemake core per task and at most
`SLURM_CPUS_PER_TASK` concurrent tasks.

Each task has `tasks/<id>/status.json`, `stdout.log`, and `stderr.log`. Status
files are atomically replaced. A task is successful only when Snakemake exits
zero and its manifest-declared Benchpress output exists. Interrupted, failed,
and timed-out tasks are retried by rerunning the same submission command;
completed Snakemake outputs are reused by `--rerun-incomplete`.

The default resource class is selected from the dimension when it is available:

| class | task timeout | intended use |
|---|---:|---|
| smoke | 10 min | d <= 20 |
| short | 30 min | d <= 50 |
| medium | 2 h | d <= 100 |
| long | 5 h | larger or explicitly difficult tasks |

Use `RESOURCE_CLASS=smoke|short|medium|long` for a benchmark-specific override.
The driver itself defaults to `serial_std`, 16 physical cores, 64 GB, and 24 h;
this is a single allocation rather than 360 submitted jobs. `serial_long`
requires `--clusters=serial --partition=serial_long --qos=cm4_serial_long` and
is selected explicitly with `NOTREKS_PARTITION=serial_long`.
`cm4_std` is rejected for serial jobs; it belongs to cluster `cm4` and requires
an exclusive multi-node allocation. `cm4_tiny` is not selected by default
because this farm needs at most 16 concurrent one-core tasks.

Snakemake capability detection runs `snakemake --help` and selects
`--use-apptainer` or, for legacy installations, `--use-singularity`. If only
Apptainer is installed with a legacy Snakemake, the farm creates a run-local
`singularity` compatibility symlink. No cluster software or shell setup is
modified.

Generated benchmark configurations request the canonical Benchpress
DAG-to-CPDAG conversion exactly once with
`benchmark_setup[].evaluation.graph_estimation.convert_to=["cpdag"]`. Raw
algorithm DAGs remain in the `original` graph output; CPDAG outputs are the
ones consumed by metrics and downstream analysis. The farm records
`output_graph_type=cpdag` in its machine-readable metadata.

Canonical commands from the repository root:

```bash
bash scripts/check_notreks_cluster_environment.sh CONFIG.json
bash scripts/submit_notreks_benchmark.sh CONFIG.json results/notreks/full_benchmark [MANIFEST.csv]
bash scripts/notreks_farm_status.sh results/notreks/full_benchmark
bash scripts/notreks_farm_first_error.sh results/notreks/full_benchmark
bash scripts/notreks_farm_cancel.sh results/notreks/full_benchmark
```

The same submission command is the resume command. Analysis runs automatically
only after all required task statuses and outputs succeed.
