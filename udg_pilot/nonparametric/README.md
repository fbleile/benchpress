# Nonparametric marginal-independence pilot

This namespace estimates only a pairwise marginal-dependence graph. It does
not assume acyclicity, Gaussianity, an order, a DAG, a UEC, or a covariance
graph, and it never passes its pair estimates to a causal optimizer.

Implemented evidence methods:

- `pairwise_hsic_raw`, `pairwise_hsic_rank`: unbiased RBF HSIC with median
  bandwidth and deterministic permutation calibration;
- `pairwise_dcov_raw`, `pairwise_dcov_rank`: distance covariance with
  deterministic permutation calibration;
- `hsic_stability`, `dcov_stability`: bootstrap/subsample selection-frequency
  paths at thresholds 0.50, 0.60, 0.70, 0.80, 0.90, and 0.95;
- `conservative_ensemble`: dependence only when both tests reject;
- `liberal_ensemble`: dependence when either test rejects. Pairs rejected by
  exactly one test remain explicitly uncertain in diagnostics.

Run the original linear-Gaussian pilot with the unchanged Fisher and
FLOP-to-UEC baselines plus the frozen order negative control:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-local-smoke/bin/python scripts/udg_pilot.py \
  --output results/udg_pilot/nonparametric \
  --dimensions 10 20 --degrees 2 4 --n 500 \
  --graph-seeds 7101 7102 --data-seeds 8101 \
  --methods pairwise_fisherz_raw pairwise_fisherz_holm flop_to_uec \
    order_uec_covbic pairwise_hsic_raw pairwise_hsic_rank \
    pairwise_dcov_raw pairwise_dcov_rank hsic_stability dcov_stability \
    conservative_ensemble liberal_ensemble
```

Run model-mismatch arms without assigning them a DAG/UEC interpretation:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python -m udg_pilot.nonparametric_benchmark \
  --output results/udg_pilot/nonparametric_mismatch \
  --dimensions 10 20 --n 500 --permutations 99
```

The mismatch benchmark includes nonlinear zero-correlation dependence and
non-Gaussian dependence. `results.csv` and `summary.csv` report pairwise MIG
precision/recall/F1, edge count, runtime, and ensemble uncertainty.
