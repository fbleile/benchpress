# NOTREKS benchmark configurations

`large_benchmark_v1.json` is the frozen large design. `benchmark_grid.json`
is the editable design template. Both define reusable named model and graph families, a
Cartesian grid, and an optional `scenarios` list for targeted additions.  The
compiler emits one normal Benchpress JSON configuration per scenario.  Every
configuration contains exactly four IDs: `flop`, `flop_notreks`, `dagma`, and
`dagma_notreks`. The FLOP+NOTREKS arm uses the hard-feasible
`global_greedy` strategy; the older signature-alternating implementation is a
diagnostic only.

Prior-knowledge fractions deterministically subsample the full oracle no-trek
sidecar with the same dataset-derived seed in both constrained methods. A positive fraction
retains at least one available pair.  Truth is never passed to an optimizer.

The one-time smoke calibration freezes the DAGMA NOTREKS weight and the
FLOP-NOTREKS proposal budget in `tuned_hyperparameters.json`; these values are
then reused across every benchmark scenario.

The primary DAGMA pair intentionally uses the ordinary DAGMA fixed threshold
of 0.30 in both arms. DAGMA+NOTREKS performs no feasibility repair or discrete
search after thresholding. This isolates the continuous NOTREKS penalty from
postselection strength; residual supplied-pair violation counts and fractions
are reported as outcomes rather than forced to zero.

```bash
PYTHONPATH=. .venv-local-smoke/bin/python scripts/notreks_benchmark.py compile --spec configs/notreks_benchmark/benchmark_grid.json --output-dir configs/notreks_benchmark/generated
```

Add a single scenario by appending an object with `id`, `model`, `graph`, `d`,
`n`, `knowledge_fraction`, and a contiguous `seeds` list to `scenarios`.

Preview the complete report format locally on the frozen one-seed smoke:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venv-local-smoke/bin/python scripts/run_notreks_pipeline_smoke.py \
  --output-dir results/dagma_notreks_oracle/pipeline_smoke_v2
```

Open `pipeline_smoke_v2/analysis/REPORT.md`; its `figures/` directory contains
both PNG and SVG versions. This smoke is a pipeline preview, not evidence for a
scientific conclusion.

## Cluster workflow

The generated `commands.txt` has one resumable Snakemake command per scenario
and is suitable as the command source for a scheduler array. The repository
and input resources should be transferred together so the fixed scenario IDs
and seeds remain unchanged.

Every command explicitly runs `workflow/Snakefile` with `--use-apptainer`.
Thus the cluster execution is native Benchpress: Benchpress generates the
graph and SEM data, constructs the oracle knowledge sidecar, executes the four
registered structure-learning rules in their containers, and runs the normal
Benchpress evaluation rules. `run_notreks_pipeline_smoke.py` is only a local
container-free integration check and is not used by the cluster array.

The canonical submission command compiles the frozen design, submits the
Benchpress array, and submits a dependent analysis job. No second cluster
command is required:

```bash
NOTREKS_PYTHON=python scripts/submit_notreks_benchmark.sh
```

Override scheduler resources with `NOTREKS_CPUS_PER_TASK`, `NOTREKS_MEMORY`,
`NOTREKS_TIME_LIMIT`, `NOTREKS_ANALYSIS_MEMORY`, and
`NOTREKS_ANALYSIS_TIME_LIMIT`. The selected Python environment must provide
NumPy, pandas, and matplotlib. The submission script prints both Slurm job IDs;
the analysis job has an `afterok` dependency on the complete array.

The `d=100` greedy jobs can be substantially slower than the calibration.
For the frozen ordering, the `d=20` scenarios are array indices
`0-19,60-79,120-139,180-199,240-259,300-319`; run those first as a staged
cluster validation before releasing the complete array. Do not infer greedy
wall time from DAGMA or vanilla FLOP.

The dependent job automatically collects and analyses all rows. Its outputs
include `all_runs.csv`, paired and factor summaries, a causal-analysis note,
PNG and SVG figures, and a self-contained Markdown report at:

```text
results/dagma_notreks_oracle/main_benchmark/analysis/REPORT.md
```

The analysis deliberately produces only the two scientifically relevant
paired contrasts: FLOP versus FLOP+NOTREKS and DAGMA versus DAGMA+NOTREKS.
