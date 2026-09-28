# Direct order-based UEC search

`order_uec_covbic` searches states `(node_order, source_clique_cover)`.
The induced undirected graph is the union of the clique edge sets and is the
only object scored and returned. No edge orientation, parent set, DAG score,
FLOP result, or estimated no-trek pair is used by the proposed method.

The search uses deterministic adjacent swaps, tuck moves, short reversals,
clique add/remove/create/delete/split/merge/reassignment moves, validity
checking, score caching, multiple starts, and optional beam width. The
default is the audited covariance-graph BIC. Names of the form
`order_uec_covbic_lambda_0`, `..._0.25`, `..._0.5`, `..._1`, `..._2`, and
`..._6.214608` are diagnostic penalty-path runs; they are not used to tune the
primary BIC result.

Reproduction:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-local-smoke/bin/python scripts/udg_pilot.py \
  --output results/udg_pilot/order_uec \
  --dimensions 10 20 --degrees 2 4 --n 200  # use one n per invocation
```

The output contains `results.csv`, `SUMMARY.md`, paired ranks, matched-Fisher
comparisons, and the Fisher operating curve. Per-row diagnostics include
order/clique moves, valid/invalid candidates, score evaluations, cache hits,
ICF iterations/failures, edge penalty, final UDG density, and runtime.
