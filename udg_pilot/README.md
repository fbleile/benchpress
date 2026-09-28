# Isolated UDG scalability pilot

This namespace estimates the unconditional-dependence graph (UDG) / missing
no-trek pair set only. Estimated pairs are never passed to FLOP, DAGMA,
NOTREKS, or another causal optimizer. All benchmark artifacts stay below
`results/udg_pilot/`.

## Methods

* `pairwise_fisherz_raw`: two-sided Fisher-z tests at `alpha=0.05`.
* `pairwise_fisherz_holm`: the same tests with Holm correction across all
  unordered pairs.
* `grues_exact`: the repository's existing `gues.InputData(...).mcmc(...)`
  interface, only when `gues` is installed. The current local environment
  reports `gues` unavailable; this is never replaced by a surrogate.
* `source_mask_greedy` and `source_mask_anneal`: the isolated Rust prototype
  in `udg_pilot/rust`. They optimize a pairwise composite BIC evidence score,
  with fixed source prior `0.5*log(n)` per singleton source.
* `uec_covbic_grow_shrink`: direct score-based UDG search. Its state is a
  canonical edge bitset and its score is
  `2 loglik(Sigma_hat_U) - (d + |E(U)|) log(n)`, with zeros constrained in
  `Sigma`, not in the precision matrix. The Rust fitter uses Cholesky,
  stable solves, deterministic edge add/remove/swap proposals, empty,
  complete, and seeded starts, and a score cache.
* `fisher_valid_smig`: direct covariance-BIC search seeded by raw Fisher-z,
  constrained to valid simple marginal-independence graphs.
* `covbic_unrestricted`: the same direct score search without the validity
  restriction.
* `flop_to_uec`: the existing FLOP output projected only for evaluation by
  ancestor overlap; it is not optimized after projection.
* `flop_candidates_covbic`: diagnostic direct reranking seeded by the FLOP
  candidate. It is secondary and does not change the primary FLOP baseline.

The SMIG checker identifies simplicial vertices, forms maximal closed
neighbourhood simplices, and requires every edge to lie in a simplex. The
`exact_covbic` Python helper enumerates all graphs only for d <= 6 validation.
The Python covariance fitter is a transparent reference; Rust is the search
hot path. Failed/non-converged fits remain explicit diagnostics and are never
replaced by Fisher or graphical-lasso output.

The Rust state uses one `u64` mask per node. A singleton mask identifies a
source label and the induced UDG is the bitwise-overlap graph. Toggle proposals
use incremental pair updates; create/delete/merge/split proposals use exact
full recomputation and are counted separately.

## Reproduction

Smoke:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-local-smoke/bin/python scripts/udg_pilot.py \
  --output results/udg_pilot/smoke \
  --dimensions 10 20 --degrees 2 4 --n 500 \
  --graph-seeds 7101 7102 7103 --data-seeds 8101 8102 \
  --restarts 8 --moves 3000 --grues-iterations 3000
```

Expanded pilot:

```bash
for n in 200 500 1000; do
  PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv-local-smoke/bin/python scripts/udg_pilot.py \
    --output "results/udg_pilot/expanded_n${n}" \
    --dimensions 10 20 --degrees 2 4 --n "$n" \
    --graph-seeds 7201 7202 7203 7204 7205 \
    --data-seeds 8201 8202 8203 --restarts 8 --moves 5000 \
    --grues-iterations 5000
done
```

Direct covariance smoke:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-local-smoke/bin/python scripts/udg_pilot.py \
  --output results/udg_pilot/direct_smoke \
  --dimensions 10 20 --degrees 2 4 --n 500 \
  --graph-seeds 7101 7102 7103 --data-seeds 8101 8102 \
  --methods pairwise_fisherz_raw pairwise_fisherz_holm flop_to_uec \
    uec_covbic_grow_shrink fisher_valid_smig covbic_unrestricted grues_exact \
  --covbic-restarts 4 --covbic-budget 300 --icf-iterations 150
```

GrUES is not replaced if unavailable. The exact existing integration is the
`gues` package/container `docker://bpimages/grues:0.3.0`; the pilot records
`status=unavailable` and the import/container reason otherwise. The primary
fair run uses no oracle source-count prior.

The predeclared gate for d=50 is: the direct method must improve no-trek
precision over FLOP->UEC and raw Fisher-z at matched predicted edge count or
matched recall, without catastrophic recall/runtime loss; the d<=6 exact
optimizer gap must be small and covariance fits stable. If not, the result is
reported as a negative pilot and d=50 is not attempted.

Each run also writes `SUMMARY.md`, `summary_by_method.csv`,
`mean_rank_summary.csv`, `paired_ranks.csv`, and matched-Fisher files. These
make the comparison paired by graph/data seed and expose the direct method's
operating density instead of hiding sparsity behind no-trek precision.

## Score audit

Before interpreting direct-search performance, run:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-local-smoke/bin/python scripts/udg_score_audit.py \
  --output results/udg_pilot/score_audit_d10_d20 \
  --dimensions 10 20 --degrees 2 4 --n 500 \
  --graph-seeds 7101 7102 --data-seeds 8101 --search-budget 300
```

The audit writes `score_audit.csv` and `score_audit.json`. It compares empty,
one-edge, Fisher, true-UDG (evaluation only), FLOP-derived, and complete
candidates, and records Rust/Python score differences, Cholesky diagnostics,
non-edge covariance residuals, and convergence. It also reports the
non-oracle penalty path `lambda in {0,.25,.5,1,2,log(n)}` for unrestricted and
valid-SMIG search. The Rust direct search uses three fixed `u64` words for
the canonical bitset: d=20 has 190 undirected edges and cannot fit in one
`u128`.
