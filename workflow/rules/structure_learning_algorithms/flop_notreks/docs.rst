FLOP-NOTREKS
=============

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
