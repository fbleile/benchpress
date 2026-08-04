# Benchmark estimand graph

```text
model ---------------------> data distribution -----> recovery
graph family --> true graph ------------------------> recovery
dimension -----> true graph
true graph ----> available oracle MI pairs
knowledge fraction --------> supplied MI pairs
available oracle MI pairs -> supplied MI pairs
sample size ---------------> data information ------> recovery
method/NOTREKS ------------> optimization/constraints -> recovery
supplied MI pairs ----------> optimization/constraints
```

The primary outcomes are paired constrained-minus-unconstrained differences
in CPDAG SHD and pattern F1. The full factorial varies model, graph family,
dimension, sample size, and knowledge fraction. These support controlled
design contrasts.

The realized number of oracle MI pairs is downstream of the sampled graph,
graph family, and dimension. The supplied count is additionally downstream of
the knowledge fraction. Consequently, pair counts are mediators. Analyses
conditioning on them are useful mechanistic descriptions but are not estimates
of the total effect of graph family or dimension.
