FLOP-NOTREKS
=============

FLOP-NOTREKS has one production implementation: the Rust global-greedy
search. It keeps FLOP's order/reinsertion outer loop and performs a complete
BIC edge-toggle search for each fixed order. Every proposed parent addition is
checked with the exact discrete ancestry witness test, so the returned DAG has
zero violations before it is converted once to the Benchpress CPDAG format.

The Rust implementation receives the canonical list of supplied NOTREKS
pairs directly. It does not use the old 64-bit signature prototype, so the
production path is not limited to 64 constrained targets. Runtime still grows
with dimension because each order evaluates many local parent regressions;
larger dimensions therefore need an appropriate restart count and wall-time
budget.

The continuous DAGMA inverse kernel is not used by FLOP-NOTREKS. FLOP needs an
exact hard certificate, and the Boolean ancestry closure is both exact and
cheaper than recomputing a dense inverse for every candidate edge.

Configuration
-------------

``search_strategy`` and ``search_version`` are retained in serialized paths
for Benchpress compatibility, but both are fixed to ``global_greedy_rust`` and
are not tuning knobs. The only production controls are the data, supplied
pair fraction, BIC penalty, deterministic seed, restart count, and the outer
reinsertion sweep count.

The method never receives graph truth. Incorrect prior pairs can exclude the
true graph, so knowledge quality remains a scientific assumption rather than
an optimizer fallback.
