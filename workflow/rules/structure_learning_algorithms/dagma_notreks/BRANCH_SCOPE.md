# Development branch split

## `dagma-notreks-oracle` — paper-facing NOTREKS integration

This branch integrates NOTREKS into established causal-discovery methods. Its
scope is deliberately convergent:

- vanilla `dagma_fast`, with optional method-independent NOTREKS components;
- `fast` selected-inverse NOTREKS by default and named kernel ablations;
- ordinary FLOP;
- signature-based FLOP-NOTREKS Versions A and B;
- the hard-feasible FLOP-outer/global-greedy comparator;
- stable benchmarks, regression tests, and paper-ready reporting.

It is not the branch for inventing a new general nonlinear causal-discovery
method.

## `global-flop-exploration` — generalized order-search research

This branch starts from commit `a61939e8` and owns exploratory work that lifts
FLOP-style reinsertion to generic nonlinear or globally coupled objectives,
including greedy and affected-block continuous inner optimizers. Results there
must remain diagnostic until matched function classes, complexity penalties,
and held-out evaluation establish a stable method.

## Legacy `add-notreks-module`

The older sibling branch is retained because it is checked out in another
worktree and contains earlier standalone module development. It is not the
paper-facing integration branch and should be harvested selectively rather
than merged wholesale.
