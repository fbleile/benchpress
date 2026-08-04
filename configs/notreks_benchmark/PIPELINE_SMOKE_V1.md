# One-seed pipeline smoke v1

The complete container-free pipeline was executed on a new standardized
linear-Gaussian ER instance (`seed=9201`, `d=20`, `n=200`) with 25% of the
available oracle no-trek pairs. It compiled the corresponding Benchpress
configuration, ran all four production algorithm paths, evaluated graphs with
Benchpress's R evaluator, and generated paired/factor/causal-analysis outputs.

| Method | CPDAG SHD | Pattern SHD | Pattern F1 | Runtime (s) | supplied-pair violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| FLOP | 2 | 2 | 0.953 | 0.014 | not constrained |
| greedy FLOP + NOTREKS | 0 | 0 | 1.000 | 100.043 | 0 |
| DAGMA | 18 | 17 | 0.614 | 0.609 | not constrained |
| DAGMA + NOTREKS | 11 | 11 | 0.703 | 2.638 | 0 |

The paired CPDAG-SHD changes were `-2` for FLOP+NOTREKS and `-7` for
DAGMA+NOTREKS. This is a pipeline check on one graph, not a scientific effect
estimate. Raw local artifacts live under
`results/dagma_notreks_oracle/pipeline_smoke_v1/` and are intentionally
git-ignored.
