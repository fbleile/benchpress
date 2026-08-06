FLOP-NOTREKS
=============

FLOP-NOTREKS has one production implementation: a Rust global-greedy
search. It keeps FLOP's order-reinsertion outer loop and performs a complete
BIC edge-toggle search for each fixed order. Every proposed parent addition is
checked with the exact discrete ancestry witness test, so the returned DAG has
zero violations before it is converted once to the Benchpress CPDAG format.

The implementation accepts arbitrary graph dimensions and receives the
canonical supplied NOTREKS pair list directly. Runtime grows with dimension
because each order evaluates many local parent regressions; larger jobs need
an appropriate restart count and wall-time budget.

The continuous DAGMA kernel is not part of FLOP-NOTREKS. FLOP uses its exact
hard certificate and Boolean ancestry closure during the discrete search.

Configuration
-------------

The serialized implementation fields are fixed to ``global_greedy_rust`` by
the schema and compiler. They are provenance fields, not tuning controls.
The production controls are the data, supplied-pair fraction, BIC penalty,
deterministic seed, restart count, and outer reinsertion sweep count.

The method never receives graph truth. Supplied pairs are the only structural
prior, so their quality remains an explicit scientific assumption.
