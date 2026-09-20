# causalAssembly protocol

`causalassembly_full` is the 98-node semisynthetic production-line benchmark
from the official [causalAssembly repository](https://github.com/boschresearch/causalAssembly)
and paper ([Göbler et al., 2024](https://proceedings.mlr.press/v236/gobler24a.html)).
The package is optional: DRF preparation may require R and `rpy2`, while cached
benchmark runs require only the repository's Python solver environment.

## Prepare the official cache

```bash
RPY2_CFFI_MODE=ABI PYTHONPATH=. .venv-local-smoke/bin/python \
  scripts/causalassembly_protocol.py prepare \
  --cache results/causalassembly_cache \
  --seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010
```

Preparation validates the 98 graph nodes and column names, fits the official
DRFs once, and stores paired 5,000-row reference/discovery samples plus a
provenance manifest. It does not silently replace causalAssembly if the
upstream package or R backend is unavailable.

## Screen the reference samples

```bash
PYTHONPATH=. .venv-local-smoke/bin/python \
  scripts/causalassembly_protocol.py screen \
  --cache results/causalassembly_cache --seed 1001
```

The practical screen uses empirical-copula data and deterministic
characteristic features with capped permutation confirmation. Non-rejection is
not treated as proof of a graphical no-trek pair; the cache stores all screen
diagnostics.

## Dry-run / smoke benchmark

```bash
PYTHONPATH=. .venv-local-smoke/bin/python \
  scripts/causalassembly_benchmark.py \
  --cache results/causalassembly_cache \
  --output results/causalassembly_smoke \
  --seeds 1001 --n 500 \
  --methods flop flop-nt-standard dagma dagma-pstrek \
  --attempts 1 --flop-sweeps 4 --dagma-stages 2 \
  --dagma-warm-iter 100 --dagma-max-iter 200
```

The production grid is `n={500,2000,5000}`, ten paired seeds, and oracle
fractions `{.10,.25,.50,1.0}`. `scripts/notreks_protocol_registry.py` records
the experiment and its default FLOP/DAGMA restart policy; the dedicated runner
keeps vanilla rows deduplicated and resumes at row granularity.

Estimated mode uses the same cached reference screen:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python \
  scripts/causalassembly_benchmark.py \
  --cache results/causalassembly_cache \
  --output results/causalassembly_estimated \
  --mode estimated_notreks --seeds 1001 1002 1003 1004 1005
```

Discovery data are standardized separately for every prefix. The existing
FLOP/DAGMA scores are retained and explicitly marked as potentially
misspecified for this nonlinear, non-Gaussian benchmark.
