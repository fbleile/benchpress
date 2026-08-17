DAGMA-NOTREKS
==============

DAGMA-NOTREKS augments linear log-det DAGMA with supplied structural
no-trek pairs. The production implementation uses analytic inverse NOTREKS,
joint structural feasibility screening, deletion-only fixed-order FLOP parent
shrinking, and exact-BIC restart selection.

See ``README.md`` in this directory for the objective, architecture, sidecar
format, supported diagnostics, and canonical commands.

FLOP-union optimization support
-------------------------------

``support_mode=flop_union_support`` runs one, two, four, or eight independently
seeded vanilla FLOP searches, unions their skeletons, and symmetrizes the
result before DAGMA optimization. Every directed arc outside that
superstructure is passed through DAGMA's existing ``exclude_edges`` mask.
The symmetrization is mandatory: a directed FLOP representative must not
exclude another orientation in the same equivalence class.

The production pipeline supports zero initialization, a zero restart followed
by the best hard-feasible masked FLOP coefficient initialization, and explicit
masked-random restarts. Optimization and postselection both verify that no
outside edge was introduced. Result diagnostics report support size, density,
FLOP seeds, unique DAGs/skeletons, and the number of feasible FLOP initializers.
This is a heuristic superstructure restriction; performance or optimality is
conditional on the union containing the relevant adjacencies.
