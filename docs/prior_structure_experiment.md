# Prior-structure experiment

This experiment asks which properties of the supplied NOTREKS graph are
associated with recovery improvement, conditional on information quantity.
The primary analysis fixes the quantity at `q=0.25`; the main benchmark is
used for the separate quantity effect.

## Primary design

- graph families: ER2 and ER4;
- dimensions: `d=20` and `d=50`;
- sample sizes: `n=100`, `500`, and `2000`;
- graph/data seeds: 20 shared seeds;
- prior seeds: 3 or more shared seeds per instance;
- prior strategies: `random`, `bipartite-max-capacity`, and
  `chromatic-greedy`;
- methods: vanilla FLOP and vanilla DAGMA, each paired with its NOTREKS
  variant;
- knowledge fraction: exactly `q=0.25`;
- attempts: identical for all methods.

Every method must use the same data instance and every solver must receive the
same unordered prior-pair set for a given graph/prior seed. Prior properties
and paired CPDAG gains are recorded per run; no aggregation is performed
before the paired rows are joined.

## Recorded prior properties

The structural table should contain:

- exact chromatic number (and the certified bounds when exact coloring is not
  available);
- distinct-node coverage / endpoint coverage;
- connected-component count;
- largest-component size;
- maximum clique size and bipartiteness;
- prior degree mean, standard deviation, and maximum;
- number of isolated prior nodes;
- trek-error alignment count and ratio, where the count is the number of
  supplied no-trek pairs for which the vanilla DAG contains a trek, and the
  ratio divides this count by the number of supplied pairs. This is a
  trek-level diagnostic, not merely a direct-edge false-positive count.

The response is paired CPDAG improvement
`delta_SHD = SHD_vanilla - SHD_NOTREKS`; positive values are better.
The primary descriptive analysis reports pooled and solver-specific Pearson
and Spearman correlations at fixed `q=0.25`. Stratified estimates by
dimension, density, and sample size are retained in the source table so that
the pooled association is not mistaken for a causal effect.

The final Figure 3 uses compact scatter panels for chromatic number,
distinct-node coverage, connected components, largest component, and the
trek-error alignment diagnostics. FLOP and DAGMA are encoded by the shared
blue/orange palette. The CSV source includes the paired rows and all
correlation estimates.
