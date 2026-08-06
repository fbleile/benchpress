FLOP-NOTREKS
=============

Purpose
-------

FLOP-NOTREKS is an exact, discrete feasibility extension of FLOP. It is
important to distinguish the two algorithms:

* **FLOP** searches over causal orders. For a proposed order, it allows only
  forward-pointing edges, so every candidate is a DAG. It chooses parent sets
  by the configured Gaussian BIC score and improves the order with node
  reinsertion moves. FLOP has no NOTREKS constraint and does not receive
  NOTREKS pairs.
* **FLOP-NOTREKS** keeps that same order search, BIC score, and reinsertion
  structure. The new part is a hard feasibility check for supplied NOTREKS
  pairs. Candidate edge additions that would create a forbidden common
  ancestor are rejected before BIC selection.

Thus FLOP-NOTREKS is not a different continuous objective and does not call
DAGMA. It is FLOP's discrete order/BIC search with an additional exact
constraint on the admissible parent sets.

Ordinary FLOP
-------------

For a fixed node order ``pi``, ordinary FLOP searches graphs satisfying

``position(parent) < position(child)``.

This forward-edge rule is the source of FLOP's DAG guarantee. Its objective is
the Gaussian BIC score, and its outer loop proposes node reinsertions in the
order. The final selected DAG is converted once to the Benchpress CPDAG
representation for evaluation.

No-trek feasibility added by FLOP-NOTREKS
-----------------------------------------

For each current candidate DAG, FLOP-NOTREKS maintains an exact ancestry
lookup table ``R``:

``R[a, b] = 1`` exactly when ``a`` is an ancestor of ``b``.

The diagonal is one, because a node is its own ancestor. For every supplied
pair ``(i, j)``, the hard condition is

``An(i) intersection An(j) = empty``.

When an edge ``u -> v`` is proposed, the lookup table is used to determine
whether the proposal would make any supplied pair share an ancestor. Such a
proposal is infeasible and is never scored as a BIC candidate. Feasible
proposals are compared using the same BIC calculation as ordinary FLOP.

After the search, the complete ancestry condition is checked once more. The
production result is accepted only when the exact certificate reports zero
violations. The forward-edge order rule supplies the DAG certificate; the
ancestry lookup supplies the independent NOTREKS certificate.

What is and is not shared
-------------------------

Shared with FLOP:

* causal-order representation;
* node-reinsertion outer search;
* BIC objective and local parent-set score updates;
* deterministic tie-breaking and restart handling;
* final DAG-to-CPDAG conversion.

Added by FLOP-NOTREKS:

* the supplied unordered NOTREKS pair list;
* incremental ancestry lookup for candidate feasibility;
* rejection of infeasible parent additions;
* an exact final no-trek certificate.

This is different from differentiable DAGMA-NOTREKS. DAGMA uses a smooth
NOTREKS penalty and its gradient while optimizing a dense weighted matrix.
FLOP-NOTREKS uses no NOTREKS gradient and does not minimize a continuous trek
value: it uses the discrete condition as a hard admissibility rule during BIC
search.

Prior-knowledge interpretation
------------------------------

The method never infers the supplied pairs from the data. If the benchmark
uses pairs generated from the true graph, that run is an oracle-prior-
knowledge diagnostic. It measures the value of exact prior structural
knowledge and must not be presented as an ordinary data-only FLOP method.

Configuration
-------------

The serialized implementation fields are fixed to ``global_greedy_rust`` by
the schema and compiler. They identify the production implementation and are
not tuning controls. The production controls are the data, supplied-pair
fraction, BIC penalty, deterministic seed, restart count, and outer
reinsertion sweep count.
