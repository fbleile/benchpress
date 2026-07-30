# DAGMA-NOTREKS

DAGMA-NOTREKS adds supplied marginal-independence knowledge to linear
log-determinant DAGMA. Knowledge is a validated JSON sidecar containing
unordered node-name pairs. The sidecar is converted to column indices before
optimisation; the algorithm never receives the true graph.

## Production pipeline

The canonical implementation is [`pipeline.py`](pipeline.py), with a safe CLI
in [`tools/production.py`](tools/production.py). Its fixed production settings
are:

- log-det DAGMA with the ordinary five-stage positive-\(\mu\) path;
- L2 loss, `lambda1=0.03`, and five deterministic restarts;
- analytic inverse NOTREKS with `trek_weight=10`;
- candidate threshold
  \(\tau=\max(\tau_{\mathrm{feas}},0.01)\);
- deletion-only, fixed-order FLOP parent shrinking;
- restart selection by the exact postprocessed Gaussian BIC.

The continuous objective follows the existing shared solver convention:

\[
\mu\{\operatorname{score}(W)+\lambda_1\lVert W\rVert_1\}
+h_{\mathrm{logdet}}(W)+\lambda_{\mathrm{NT}}R_{\mathcal I}^{\mathrm{inv}}(W).
\]

The pipeline is deliberately compositional:

```text
data + no-trek sidecar
        |
        v
log-det DAGMA + analytic inverse NOTREKS (five restarts)
        |
        v
minimum joint feasibility threshold
        |
        v
max(feasibility_threshold, screening_floor=0.01)
        |
        v
candidate_graph
        |
        v
fixed_order_parent_shrink (FLOP local Gaussian BIC)
        |
        v
postprocessed_bic restart selection
```

Fixed-order parent shrinking never changes the order, adds or reverses an
edge, and therefore preserves DAG and no-trek feasibility.

## Commands

Quick deterministic smoke test:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/production.py smoke
```

Production run on a CSV with a matching sidecar:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/production.py run --data DATA.csv --knowledge no_trek_pairs.json --output-dir results/dagma_notreks_oracle/production_run
```

The main retained comparison (long-running; do not use as a smoke test):

```bash
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/inv_dag_benchmark.py --n-jobs 5
```

Historical SHD-6 regression audit (long-running):

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/inv_regression_audit.py
```

Kernel timing diagnostic:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/profile_runtime.py
```

All commands write below `results/dagma_notreks_oracle/`.

## Supported and diagnostic variants

`dagma_notreks` is the production method. `inv_notreks` replaces the log-det
DAG penalty with inverse trace and remains a diagnostic ablation. Explicit
fixed thresholds and minimum-feasibility-only screening are historical
diagnostics. Gaussian-profile loss and terminal-zero-stage experiments are
historical only and are not production entry points.

The shared solver retains lightweight function boundaries for the data loss,
DAG penalty, NOTREKS penalty, feasibility projection, BIC scoring, and
postprocessing. Mathematical formulas and scaling live in the shared
`dagma/` modules; experiment runners must not duplicate them.

## Sidecar

The sidecar schema is:

```json
{
  "type": "no_trek_pairs",
  "source": "true_dag",
  "node_names": ["X1", "X2"],
  "pairs": [["X1", "X2"]]
}
```

Names must exactly match the data-column order. Incorrect hard knowledge can
exclude the true graph.

## Generic weighted-graph postprocessing

New development uses the optimizer-independent
`weighted_graph_postprocessing` package. DAGMA-fast produces a weighted
matrix; the shared second stage performs candidate generation, exact
feasibility checks, score-based local search, and truth-free selection.
Historical policy names remain compatibility aliases.

## Standardized feasible postselection

The public production CLI exposes `PS1_joint_feasible_greedy_score`,
`PS2_joint_feasible_local_search`, `PS3_joint_feasible_budgeted_search`,
`PS4_joint_violation_repair`, and
`PS5_fixed_threshold_joint_feasible`.
`REF_threshold_grid_scc_bic_infeasible` is a linear-Gaussian comparison only
and is never recommendation-eligible. No policy is hard-coded as the
scientific recommendation.

Observed columns are standardized from training statistics with `ddof=0` and
a `1e-12` standard-deviation floor. Production candidates are certified using
exact graph acyclicity and common-ancestor reachability for active NOTREKS
pairs. L1 candidates use coordinate-descent refits; L2 candidates use ridge
solves. Candidate pools, feasibility, search, model refitting, scoring, and
selection remain separate.

The target contains no nonlinear DAGMA solver. The graph machinery accepts a
nonlinear DAGMA adjacency proxy and a model-specific masked-refit callback,
but nonlinear production postselection remains disabled until an actual
nonlinear model provides them. Gaussian BIC is never a nonlinear fallback.

Run one policy:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/production.py run --data DATA.csv --knowledge no_trek_pairs.json --output-dir results/postselection/ps1 --lambda-policy fixed_0.03 --regularizer-type L1 --postselection-policy PS1_joint_feasible_greedy_score --candidate-edge-pool threshold_grid
```

Evaluate a directory of completed policy runs:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/evaluate_postselection.py --results-root results/postselection --output-dir results/postselection/evaluation
```

Validate deterministic transferred semantics:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python workflow/rules/structure_learning_algorithms/dagma_notreks/tools/validate_postselection_transfer.py --source-root /Users/fbleile/Projects/benchpress --output-dir results/dagma_notreks_oracle/postselection_transfer_validation
```
