# Section 20 — CausalMan Small and Medium

CausalMan is an official simulator benchmark from Bosch Research, not a
production-data set.  The adapter uses only the upstream `causalman_small` and
`causalman_medium` variants; it never substitutes the `micro` variant or a
hand-built SEM.  The upstream source and paper are recorded in every cache
manifest: <https://github.com/boschresearch/CausalMan> and
<https://arxiv.org/abs/2502.12707>.

## Architecture

- preparation, observational filtering, complete-DAG oracle pairs and latent
  projection: `scripts/causalman_protocol.py`;
- resumable method execution and projection-aware metrics:
  `scripts/causalman_benchmark.py`;
- registry IDs: `causalman_small`, `causalman_medium` in
  `scripts/notreks_protocol_registry.py`;
- paired Pareto aggregation and PDF: `scripts/causalman_figures.py`;
- portable cache: `results/causalman_cache/{causalman_small,causalman_medium}`.

The selected feature set is fixed before solver execution from finite,
nonconstant, genuinely continuous observable sensors.  Each prefix is
standardized independently with `ddof=0`.  The full DAG is retained for
ancestor/no-trek checks; the observed graph is a latent projection with
directed and bidirected edges.  If bidirected edges are present, CPDAG SHD,
SID and AID are missing with reason `latent_confounding`; projected directed
skeleton agreement is descriptive only.

## Commands

Preparation is idempotent and records measured dimensions, checksums, package
versions, intervention filtering, graph projection and eligibility:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python scripts/causalman_protocol.py prepare \
  --cache results/causalman_cache \
  --variants causalman_small causalman_medium \
  --seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010
```

The one-replicate smoke (both variants, all production method mechanisms) is:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python scripts/causalman_benchmark.py \
  --cache results/causalman_cache --output results/causalman_smoke \
  --seeds 1001 --n-values 500 --attempts 1 --flop-sweeps 2 \
  --dagma-stages 2 --dagma-warm-iter 1000 --dagma-max-iter 2000
```

The complete feasible run is:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python scripts/causalman_benchmark.py \
  --cache results/causalman_cache --output results/causalman_full \
  --seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 \
  --attempts 2 --flop-sweeps 16 --dagma-stages 5 \
  --dagma-warm-iter 30000 --dagma-max-iter 60000
```

Build the two-panel figure only after execution/collection:

```bash
PYTHONPATH=. .venv-local-smoke/bin/python scripts/causalman_figures.py \
  --input results/causalman_full/raw/results.csv \
  --output results/causalman_full/analysis/figures
```

This writes `figure_causalman_paired_pareto_small_medium.pdf` (and PNG), the
plot-data CSV and the descriptive metric label.  It does not modify the
original Figure 1 artifacts.

If preparation reports that `causalman` is unavailable, stop: install the
official pinned upstream package/assets and rerun preparation.  No replacement
data may be used.
