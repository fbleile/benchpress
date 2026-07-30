# Weighted graph postprocessing

This module is not part of DAGMA.

It converts dense weighted estimates from arbitrary continuous
causal-discovery methods into feasible discrete graphs. The weights guide
candidate generation and local-search proposals. The configured graph score
and structural constraints determine the accepted final graph.

```text
weighted estimate(s)
    -> threshold/weight-ranked candidate starts
    -> exact feasibility checks
    -> budgeted score-based local search
    -> truth-free deterministic selection
    -> final graph
```

The implementation never branches on the source optimizer name. The same API
accepts DAGMA, NOTEARS, GOLEM, nonlinear, or mock weighted estimates. Gaussian
BIC is one score adapter; arbitrary global or nonlinear scores can be supplied
through `CallableGraphScore`.

Canonical policies are `threshold_grid_score_search`,
`weighted_feasible_local_search`, and `fixed_threshold`. Historical DAGMA
policy names remain deprecated aliases where inexpensive.
