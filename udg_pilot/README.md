# Isolated UDG scalability pilot

This namespace estimates the unconditional-dependence graph (UDG) / missing
no-trek pair set only. It does not invoke FLOP, DAGMA, NOTREKS, or any causal
discovery optimizer.

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

