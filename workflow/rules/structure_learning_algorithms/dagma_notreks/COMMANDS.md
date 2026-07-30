# DAGMA-NOTREKS commands

The maintained commands, purpose, runtime class, inputs, and outputs are
documented in [README.md](README.md).

```bash
# Fast deterministic smoke test
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/production.py smoke

# Production pipeline
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/production.py run --data DATA.csv --knowledge no_trek_pairs.json --output-dir results/dagma_notreks_oracle/production_run

# Main log-det versus inverse-trace comparison (long)
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/inv_dag_benchmark.py --n-jobs 5

# Historical regression audit (long)
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/inv_regression_audit.py

# Kernel timing benchmark (diagnostic)
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/profile_runtime.py
```
