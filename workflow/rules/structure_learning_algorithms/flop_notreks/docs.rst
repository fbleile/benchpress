FLOP-NOTREKS
=============

Feasibility note: FLOP-NOTREKS uses an exact discrete ancestry certificate,
not DAGMA's continuous inverse NOTREKS kernel. For each supplied pair
``(i, j)``, it requires ``An(i) intersection An(j) == empty`` (with each node
counting as its own ancestor). The global-greedy implementation computes a
Boolean transitive closure, precomputes invalid additions, and performs one
final full DAG+NOTREKS check. This is the hard certificate used during search.

For diagnostics only, an inverse-resolvent check with ``inverse_epsilon=0`` is
mathematically equivalent on a valid DAG because
``(I - A)^(-1) = I + A + A^2 + ...``. Its pair entries are zero up to
floating-point roundoff when no common ancestor exists. It is not an acyclicity
certificate: a cyclic matrix can still have an inverse. The continuous DAGMA
path therefore keeps its epsilon-protected inverse objective, while FLOP keeps
the exact ancestry check. Recomputing an inverse for every edge proposal is
also slower than the cached closure/table method.

FLOP-NOTREKS consumes the same validated ``no_trek_pairs`` JSON sidecar as
DAGMA-NOTREKS and PC-MIOracle. It never receives the true graph. For every
edge ``u -> v``, the ancestry certificates require ``S_v`` to be a subset of
``S_u``. Each signature is an independent set in the incompatibility graph,
which guarantees that every supplied pair has no common ancestor.

Two exact hard-constraint versions are available:

``fixed_signature_a``
  Version A fixes the initial signatures and runs ordinary constrained FLOP.
  It is a correctness/runtime baseline and is expected to be restrictive with
  minimal signatures.

``alternating_full_refit_b``
  Version B is the scientific default. It alternates constrained FLOP order
  blocks with complete ancestry-cone promotions. A direct edge gain ranks a
  proposal, but acceptance always uses complete constrained parent refitting
  and total BIC.

Version B returns to an order block immediately after every accepted
promotion. Canonical reverse-topological compression is applied only at a
plateau, followed by a full refit and final order block. Thus a newly promoted
permission cannot be erased before FLOP has an opportunity to exploit it.

``incremental_c`` and ``hybrid_bc`` are reserved future optimizations and are
not implemented. Version C begins only when Version B's full proposal refit is
replaced by exact affected-node incremental updates; larger proposal pools,
restarts, randomized initialization, and delayed compression remain Version B.

The prototype supports at most 64 distinct constrained targets. Its selected
DAG is checked for zero violations before conversion to FLOP's CPDAG output.
Incorrect hard knowledge can exclude the true DAG. Current limitations include
full refits for signature proposals, no dynamic ancestry cache, and sequential
proposal evaluation.

All constrained versions preserve order precedence, the signature-subset edge
rule, independent-set signatures, target self-membership, and zero selected-DAG
violations.

Global greedy comparator
------------------------

``global_greedy.fit_global_greedy_notreks`` is a separate paper comparator.
It is exposed through ``search_strategy=global_greedy`` and is the strategy
used by the paper-facing NOTREKS benchmark. It returns a selected DAG directly.
The historical Rust implementation remains available through
``search_strategy=signature_alternating`` and returns the usual CPDAG output
plus a selected-DAG diagnostic.
It retains FLOP-style node reinsertion outside, but applies a complete-graph
Gaussian BIC edge-toggle search inside and checks exact DAG/NOTREKS feasibility
for every move. It does not use graph truth or silently fall back to ordinary
FLOP. It is intentionally kept distinct from Versions A and B because it does
not use their ancestry-signature representation.
