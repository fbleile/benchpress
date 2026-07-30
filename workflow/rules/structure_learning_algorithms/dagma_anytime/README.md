# DAGMA Anytime Variants

This module adds separate anytime-capable linear DAGMA variants while leaving the existing `dagma` Benchpress module untouched.

Implemented identifiers via `solver_variant`:

- `dagma_vanilla_anytime`: instrumented reference following upstream linear DAGMA updates, L1 subgradient, continuation, and thresholding.
- `dagma_fast64`: float64 implementation backend for kernel/timing experiments.
- `dagma_float32`: full single-precision ablation with explicit precision policy.
- `dagma_mixed`: float32 optimizer state with float64 log-det/domain calculation.
- `dagma_prox_adaptive`: algorithmic variant using zero-diagonal masking, proximal L1 handling, and adaptive stopping.
- `dagma_hybrid`: experimental placeholder on the proximal/adaptive path; robust quasi-Newton polishing is not yet production-ready.

The normal Benchpress outputs are `adjmat.csv`, `time.txt`, and `ntests.txt`. The module also writes `diagnostics.json` containing timing, checkpoints, best-so-far diagnostics, precision policy, factorization counts, domain backtracks, and projection status.

Smoke tests:

```bash
.venv-local-smoke/bin/python -m pytest tests/dagma_anytime/test_dagma_anytime.py -q
.venv-local-smoke/bin/python scripts/dagma_anytime/benchmark_dagma_anytime.py --smoke --dims 20 50 --seeds 1001 --methods dagma_vanilla dagma_vanilla_anytime dagma_fast64 dagma_float32 dagma_mixed dagma_prox_adaptive --warm-iter 5 --max-iter 8 --T 2 --checkpoint 4 --budget 0.5 --output-dir results/dagma_anytime/smoke_local
```

Prepared full benchmark command:

```bash
.venv-local-smoke/bin/python scripts/dagma_anytime/benchmark_dagma_anytime.py --full --dims 20 50 --seeds 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 --methods dagma_vanilla dagma_vanilla_anytime dagma_fast64 dagma_float32 dagma_mixed dagma_prox_adaptive dagma_hybrid --warm-iter 300 --max-iter 600 --T 3 --checkpoint 50 --budget 60 --output-dir results/dagma_anytime/full_d20_d50
```

Caveats: Python benchmark metrics mirror Benchpress metric names for smoke testing. CPDAG SHD should be produced by the native R evaluator in full Benchpress workflows when available.
