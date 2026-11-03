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
directed and bidirected edges.

Because the estimators produce DAG-like adjacency matrices rather than ADMGs,
we do not interpret a directed estimate between two confounded observed
variables as proof of a directed causal relation.  Such an edge can instead
be the estimator's best available representation of their marginal
dependence.

We therefore report three descriptive diagnostics.  Projected
marginal-adjacency SHD compares the estimated DAG skeleton with the union of
directed and bidirected adjacencies in the latent projection; it asks whether
the observed dependence neighbourhood is recovered, while ignoring edge type
and direction.  Projected directed-skeleton SHD compares the estimated
skeleton only with the directed part of the projection; it is a stricter
diagnostic of the directed observed component.  Projected ancestor disagreement
compares directed reachability in the estimate with directed reachability in
that same directed component.  Bidirected edges and paths involving them are
excluded from the latter two diagnostics because they encode latent
confounding, not directed observed-variable ancestry.

None of these quantities is CPDAG SHD, SID, or CPDAG Ancestor-AID.  The
marginal-adjacency diagnostic is the fairest structural comparison when an
estimator has no bidirected-edge output, whereas the directed-skeleton and
ancestor diagnostics retain causal-direction information at the cost of
ignoring latent-confounding structure.  Smaller values indicate fewer
adjacency or reachability disagreements, but all three metrics should be
interpreted as descriptive agreement with the ADMG projection rather than as
complete causal recovery.

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

This writes PNG figures for projected marginal-adjacency SHD, projected
directed-skeleton SHD, and projected ancestor disagreement, together with
their plot-data CSVs.  It does not modify the original Figure 1 artifacts.

A suitable caption is:

~~~latex
\caption{CausalMan Small and Medium benchmarks.  The plotted diagnostics are
descriptive comparisons with the latent projection of the observed graph.
Projected marginal-adjacency SHD evaluates the estimated skeleton against the
union of directed and bidirected projection adjacencies, thereby measuring
recovery of observed marginal dependence without assigning causal meaning to
an estimated edge.  Projected directed-skeleton SHD evaluates only the
directed component, and projected ancestor disagreement compares directed
reachability in that component.  Bidirected edges are excluded from the latter
two diagnostics because they represent latent confounding rather than
directed observed-variable ancestry.  Lower values are better; none of these
metrics is CPDAG SHD or CPDAG Ancestor-AID.}
~~~

If preparation reports that `causalman` is unavailable, stop: install the
official pinned upstream package/assets and rerun preparation.  No replacement
data may be used.
