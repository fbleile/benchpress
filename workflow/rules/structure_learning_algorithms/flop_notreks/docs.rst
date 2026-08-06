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

How the lookup is used efficiently
-----------------------------------

The lookup state stores only ancestry support, not regression coefficients or
floating-point path values. In the Rust implementation, each node owns a
packed bitset of its ancestors. Starting with one diagonal bit, a child bitset
is formed by word-level OR operations with the bitsets of its parents. This is
the Boolean triangular solve in causal order. It is exact for arbitrary
dimensions and has no numerical zeros or thresholds.

For a proposed ``u -> v``, only ancestry relations that can change need to be
considered: ancestors of ``u`` can become ancestors of ``v`` and of the
descendants of ``v``. The feasibility test checks those affected relations
against the supplied forbidden pairs. It rejects an addition as soon as one
pair would have a common ancestor. Edge deletions cannot create a new common
ancestor, so they do not need a NOTREKS rejection test. The BIC candidate is
then compared using FLOP's local parent-score update; no dense NOTREKS inverse
or gradient is evaluated.

The current ``global_greedy_rust`` implementation rebuilds these packed
bitsets after each accepted inner-loop move. This is deliberate and simple:
moves may add or delete edges, and rebuilding guarantees that the state is
never stale. The dominant cost at larger dimensions is usually the many BIC
parent-set proposals, rather than these word-level support operations.

Relation to the power-series trek criterion
--------------------------------------------

The packed lookup is an exact Boolean implementation of the support criterion
in :ref:`sec:Power Series Trek constraint`. For a binary adjacency matrix
``B`` in a DAG, let

``A = I + B + ... + B**(d-1)``.

With ordinary arithmetic, ``A[c, j]`` is the number of directed paths from
``c`` to ``j`` (including the length-zero path). Therefore

``(A.T @ A)[i, j] > 0``

if and only if ``i`` and ``j`` have a common ancestor, which is precisely the
existence of an ``i``-``j`` trek. The packed implementation computes the same
zero/nonzero information over the Boolean semiring. It stores the support of
each column of ``A`` as an ancestor bitset ``a_j`` and evaluates

``(a_i & a_j) != 0``.

This implementation answers the feasibility question only: it records each
ancestor once and therefore does not reproduce the multiplicity in the
numeric value ``(A.T @ A)[i, j]``. That distinction is intentional. FLOP-
NOTREKS needs to know whether a forbidden trek exists, not how many paths
realize it. Since FLOP's order construction is a DAG, the truncated series is
finite and the Boolean closure is exact; no inverse convergence condition,
matrix exponential, eigenvalue test, or floating-point threshold is needed.

The correspondence is therefore:

``power series / A.T A``
    path-support calculation used to characterize trek existence;
``packed ancestry / bitset intersection``
    exact word-level implementation of the same support test for the discrete
    FLOP search.

The continuous ``h_inv`` and ``h_exp`` penalties in DAGMA-NOTREKS are a
separate optimization device. They operate on dense real-valued matrices and
provide gradients; the FLOP-NOTREKS lookup is a hard certificate on the
current discrete graph.

Incrementally updating the closure after an edge addition is also possible by
propagating the affected ancestor/descendant region. Deletions are harder:
removing an edge can invalidate reachability that has another supporting path,
so a safe implementation must either track path support counts or rebuild the
affected closure. The production implementation uses the simpler exact
rebuild rather than an unvalidated dynamic-closure cache.

The bit-packed signature representation in other constrained-search code is a
different optimization for a different search path. It must not be confused
with the arbitrary-dimension Boolean closure used by the production global
greedy implementation.

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
