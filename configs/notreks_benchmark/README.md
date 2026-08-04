# NOTREKS benchmark configurations

`benchmark_grid.json` defines reusable named model and graph families, a
Cartesian grid, and an optional `scenarios` list for targeted additions.  The
compiler emits one normal Benchpress JSON configuration per scenario.  Every
configuration contains exactly four IDs: `flop`, `flop_notreks`, `dagma`, and
`dagma_notreks`.

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

After all jobs finish, collect and analyse them with:

```bash
PYTHONPATH=. python scripts/notreks_benchmark.py collect --manifest configs/notreks_benchmark/generated/scenario_manifest.csv --results-root results/output --output results/dagma_notreks_oracle/main_benchmark/all_runs.csv
PYTHONPATH=. python scripts/notreks_benchmark.py analyse --results-csv results/dagma_notreks_oracle/main_benchmark/all_runs.csv --output-dir results/dagma_notreks_oracle/main_benchmark/analysis
```

The analysis deliberately produces only the two scientifically relevant
paired contrasts: FLOP versus FLOP+NOTREKS and DAGMA versus DAGMA+NOTREKS.
