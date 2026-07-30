# DAGMA-fast

`dagma_fast` is the canonical continuous optimizer. It uses the exact fused
float64 LU path for the DAGMA log-determinant value and gradient. Historical
names `dagma_fast64` and `dagma_fused64_exact` are compatibility aliases.

```text
data/function class
    -> objective
    -> dagma_fast
    -> weighted W
    -> generic weighted-graph postprocessing
    -> feasibility checker
    -> graph score/local search
    -> final graph
```

The optimizer ends at weighted `W`. The rounding policy is not part of the
continuous optimizer. The same generic postprocessor can consume weighted
outputs from DAGMA, NOTEARS, GOLEM, or nonlinear methods.

The optimizer consumes a small objective protocol, a DAG-penalty object, and
zero or more structural components. `LinearL2Objective` is currently
implemented. Callable mock/nonlinear objectives and alternative scalar DAG
penalties work without optimizer changes. A NOTEARS scalar penalty is
structurally pluggable, but reproducing NOTEARS requires its augmented-
Lagrangian outer controller.

Lambda policies are resolved in `lambda_policy.py`; graph conversion is in
`weighted_graph_postprocessing/`.

Smoke:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python -m workflow.rules.structure_learning_algorithms.dagma_fast.runner smoke
```
