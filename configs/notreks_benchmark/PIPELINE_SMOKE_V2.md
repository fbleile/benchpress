# One-seed matched-threshold pipeline smoke v2

The complete container-free pipeline was executed on a new standardized
linear-Gaussian ER instance (`seed=9201`, `d=20`, `n=200`) with 25% of the
available oracle no-trek pairs. It compiled the corresponding Benchpress
configuration, ran all four production algorithm paths, evaluated graphs with
Benchpress's R evaluator, and generated paired/factor/causal-analysis outputs.
Both DAGMA arms used only the same fixed `|W| >= 0.30` threshold. No hard
feasibility repair or discrete postselection was applied to DAGMA+NOTREKS.

| Method | CPDAG SHD | Pattern SHD | Pattern F1 | Runtime (s) | supplied-pair violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| FLOP | 2 | 2 | 0.953 | 0.015 | not constrained |
| greedy FLOP + NOTREKS | 0 | 0 | 1.000 | 95.704 | 0 |
| DAGMA | 18 | 17 | 0.614 | 0.592 | not constrained |
| DAGMA + NOTREKS | 11 | 11 | 0.703 | 2.586 | 0 |

## Exact implementation acceleration

The hard-feasible FLOP+NOTREKS inner loop was subsequently accelerated without
changing its search, Gaussian-BIC objective, deterministic tie-breaking, or
hard constraints. Exact node-local BIC scores are cached, and invalid NOTREKS
edge additions are precomputed from the current transitive closure. Repeating
this same seed and pipeline selected the same perfect graph with zero supplied
pair violations in **2.180 seconds**, versus **95.704 seconds** above (a
**43.9x** end-to-end method speedup). The original table is retained as the
historical pre-optimization result.

The paired CPDAG-SHD changes were `-2` for FLOP+NOTREKS and `-7` for
DAGMA+NOTREKS. This is a pipeline check on one graph, not a scientific effect
estimate. Raw local artifacts live under
`results/dagma_notreks_oracle/pipeline_smoke_v2/` and are intentionally
git-ignored. The generated analysis includes the same tables, narrative report,
and 11-figure PNG/SVG gallery that the dependent cluster-analysis job will
produce. The zero DAGMA+NOTREKS violations here are observed, not enforced.
