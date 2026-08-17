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

Chromatic source-prefix comparator
----------------------------------

``chromatic_source_prefix_flop`` is a data-plus-prior comparator that does not
enforce the individual NOTREKS constraints during parent selection.  It forms
the undirected no-trek graph ``H`` (one edge per supplied pair), computes
``chi(H)`` by bounded exact DSATUR, and forces the first ``chi(H)`` positions
of every candidate FLOP causal order to have no parents.  All later nodes use
the ordinary FLOP Gaussian-BIC grow/shrink parent fitter.

A reinsertion can cross the source-prefix boundary and thereby change which
node is parentless. The incremental kernel handles that single boundary swap
with a from-scratch local parent fit for the node leaving the source block.
All other crossed nodes use FLOP's ordinary plus/minus local-score updates, so
a sweep has the same asymptotic work as a FLOP sweep rather than refitting the
complete graph for every candidate order.

If exact coloring exceeds its configured search-node budget, the method uses
the size of an explicitly constructed clique.  That is a certified lower
bound on ``chi(H)``.  Diagnostics record ``chromatic_exact``, the clique lower
bound, the current coloring upper bound, and the number of DSATUR search
nodes, so an exact and fallback run cannot be confused.

Unlike hard FLOP-NOTREKS, this comparator does not guarantee zero no-trek
violations.  The pairs influence only the scalar source-prefix length.  It is
therefore appropriately described as FLOP with theorem-motivated source prior
knowledge, rather than as another exact NOTREKS constraint solver.

The separate ``global_greedy_chromatic_sources`` comparator combines both
restrictions: every edge toggle must preserve all hard no-trek constraints,
and targets in the first ``chi(H)`` order positions reject all incoming
edges. It retains the exact zero-violation certificate, but searches a strict
subset of the ordinary global-greedy feasible graphs.

``fixed_signature_chromatic_sources`` applies the same boundary-aware source
prefix to the historical fixed-signature kernel. This is the combined variant
closest to ordinary FLOP: it retains FLOP's grow/shrink parent fitter and
incremental node reinsertion, while the fixed ancestry signatures screen
inadmissible parents.

``incremental_promotion_d`` addresses the fixed-signature dead end without
switching to global edge-toggle search. At a constrained FLOP plateau it ranks
currently blocked forward relations by their local BIC improvement. For the
best valid proposal it adds the missing signature payload to the proposed
parent and its ancestor cone, updates only the affected local parent score,
and resumes ordinary incremental node reinsertion. Promotions are monotone
and every returned DAG still receives the exact no-trek certificate.

Global-greedy execution modes
-----------------------------

``global_greedy_rust`` is the preserved sequential reference kernel.
``global_greedy_cached`` adds an exact cross-order local-score cache but keeps
candidate evaluation sequential. ``global_greedy_parallel`` evaluates the
independent reinsertion positions concurrently and applies the original
deterministic score/order/adjacency tie-break only after all results return.
It does not parallelize dependent accepted moves or restarts. The sequential
reference remains selectable for regression checks and immediate rollback.
``global_greedy_hybrid`` is an opt-in strategy experiment built on the
parallel kernel. It retains a seeded vanilla-FLOP DAG as an incumbent when
that DAG is already feasible, uses constraint degree only to bias the first
order (without imposing sources), and stops after a conservative restart
plateau. Subsequent restarts remain full seeded random permutations.

Generic gFLOP
-------------

``gflop`` is the generic full-global reference implementation. Its outer
engine knows only causal orders, dense edge strengths, complete scalar merits,
optimizer states, and hard-candidate callbacks. It does not invoke nodewise
scores, Gaussian BIC caches, or DAGMA internals.

For every order the backend receives the complete forward-triangular mask.
Every dense fixed-order state is therefore acyclic and the DAGMA backend sets
``dag_penalty_weight=0``. Edges absent from an initializer remain available.
Every candidate reinsertion transports the dense state and globally
reoptimizes all permitted parameters. The current order and screened
candidates receive equal screening and refinement budgets before comparison.

The NOTREKS continuation permits infeasible intermediate states. Its complete
merit is the backend base score plus regularizer plus the configured weight
times the repository PSTrek penalty. Smallest-feasible-threshold attempts run
during search; final selection uses the existing budgeted,
nondecomposability-safe postselection. Feasible initializations and later hard
candidates are protected by unpenalized target score. Only an independently
certified DAG is returned. ``use_notreks=false`` runs the identical engine and
backend without the penalty or hard NOTREKS constraint.

``DagmaOrderBackend`` reuses ``SharedDagmaLinear``, PSTrek, lambda policies,
the optimizer, standardization, and postselection. The backend is linear now;
the protocol permits later nonlinear parameter states and edge gates or
black-box optimizers.

The former parent-family block prototype is retained only as
``local_window_ablation``. It is not gFLOP and is not the default selector.

The focused tests and single-seed smoke are run with::

  PYTHONPATH=. python -m pytest \
    workflow/rules/structure_learning_algorithms/flop_notreks/tests/test_gflop_generic.py \
    workflow/rules/structure_learning_algorithms/flop_notreks/tests/test_local_window_ablation.py -q
  PYTHONPATH=. python \
    workflow/rules/structure_learning_algorithms/flop_notreks/tools/gflop_generic_smoke.py

A later paired multi-seed d=20 benchmark can be generated without changing
method settings by running::

  for seed in 8401 8402 8403 8404 8405; do
    PYTHONPATH=. python \
      workflow/rules/structure_learning_algorithms/flop_notreks/tools/gflop_generic_smoke.py \
      --dimension 20 --sample-size 500 --data-seed "$seed" \
      --algorithm-seed 7401 --knowledge-seed 9401 --append
  done

Configuration
-------------

Exact branch-and-cut experiment
-------------------------------

``exact_solver.py`` provides an optional PySCIPOpt family-variable model for
low-dimensional certification experiments. It enumerates every parent set up
to the explicitly reported indegree cap and performs no score screening.
A persistent SCIP constraint handler separates standard cluster inequalities
and lazy forbidden-trek path-union cuts. Pair endpoints are forbidden as
direct parents, and the source inequality uses either exact ``chi(H)`` or the
certified clique lower bound returned by ``chromatic_sources.py``. Compact
topological-order constraints independently enforce acyclicity if fractional
cluster separation finds no violated cut. Source-family variables receive
early branching priority, and a feasible heuristic DAG can be supplied as a
MIP start after score-optimal deletion of excess parents.

``tools/exact_notreks_benchmark.py`` is resumable and reports the incumbent,
dual bound, MIP gap, parent-family count, generated cuts, source count, and
independent DAG/NOTREKS verification. A time-limited run is never reported as
exact; only SCIP status ``optimal`` with zero gap is a certificate. Any
certificate is conditional on the declared indegree cap (and would also be
conditional on a superstructure if one were added; the current experiment
uses none).

The serialized ``search_strategy`` selects ``global_greedy_rust`` (hard
pairwise feasibility), ``chromatic_source_prefix_flop`` (source-only prior),
``global_greedy_chromatic_sources`` (both restrictions), ``gflop`` (generic
full-global order search), or ``local_window_ablation`` (the old block method).
The research-only ``fixed_signature_chromatic_sources`` strategy combines the
prefix with the close-to-FLOP fixed-signature kernel.
The latter also exposes ``max_chromatic_search_nodes``; exhausting it switches
to the reported clique lower bound. Shared controls are the data,
supplied-pair fraction, BIC penalty, deterministic seed, and restart count.
