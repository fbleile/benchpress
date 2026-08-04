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
of 0.30 in both arms. DAGMA+NOTREKS may make only the deterministic feasibility
repair required by its active hard constraints. This prevents postselection
strength from being confounded with the continuous NOTREKS penalty.

```bash
PYTHONPATH=. .venv-local-smoke/bin/python scripts/notreks_benchmark.py compile --spec configs/notreks_benchmark/benchmark_grid.json --output-dir configs/notreks_benchmark/generated
```

Add a single scenario by appending an object with `id`, `model`, `graph`, `d`,
`n`, `knowledge_fraction`, and a contiguous `seeds` list to `scenarios`.

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

Compile the frozen design and submit it as a Slurm array (adjust account,
partition, memory, and time for the target cluster):

```bash
PYTHONPATH=. python scripts/notreks_benchmark.py compile --spec configs/notreks_benchmark/large_benchmark_v1.json --output-dir configs/notreks_benchmark/generated_v1
wc -l configs/notreks_benchmark/generated_v1/commands.txt
sbatch --array=0-359 --cpus-per-task=4 --mem=16G --time=2-00:00:00 scripts/run_notreks_cluster_array.sh configs/notreks_benchmark/generated_v1/commands.txt
```

The `d=100` greedy jobs can be substantially slower than the calibration.
For the frozen ordering, the `d=20` scenarios are array indices
`0-19,60-79,120-139,180-199,240-259,300-319`; run those first as a staged
cluster validation before releasing the complete array. Do not infer greedy
wall time from DAGMA or vanilla FLOP.

After all jobs finish, collect and analyse them with:

```bash
PYTHONPATH=. python scripts/notreks_benchmark.py collect --manifest configs/notreks_benchmark/generated/scenario_manifest.csv --results-root results/output --output results/dagma_notreks_oracle/main_benchmark/all_runs.csv
PYTHONPATH=. python scripts/notreks_benchmark.py analyse --results-csv results/dagma_notreks_oracle/main_benchmark/all_runs.csv --output-dir results/dagma_notreks_oracle/main_benchmark/analysis
```

The analysis deliberately produces only the two scientifically relevant
paired contrasts: FLOP versus FLOP+NOTREKS and DAGMA versus DAGMA+NOTREKS.
