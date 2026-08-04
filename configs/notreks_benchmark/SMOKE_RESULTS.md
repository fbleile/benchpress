# One-time NOTREKS calibration smoke

Calibration used standardized linear-Gaussian ER data at `d=8`, `n=200`,
seeds 9101 and 9102, and 25% of the available oracle no-trek pairs. Selection
used only zero-violation eligibility and postprocessed Gaussian BIC; truth
metrics below are diagnostic.

The tested DAGMA-NOTREKS weights 0.3, 1, 3, 10, and 30 produced the same
postselected graphs and the same mean BIC (`-528.784`). The calibration was
therefore uninformative about strength, and the deterministic weaker-penalty
tie break froze `trek_weight=0.3`. This should be described as a conservative
calibration choice, not evidence that 0.3 is universally optimal.

For FLOP-NOTREKS, `(signature_top_k=32, max_signature_rounds=100)` had the best
mean selected-DAG BIC (`-523.277`) among the three tested budgets and was
frozen. Its exploration budget is 8.

| Method | mean SHD pattern | mean F1 pattern | mean edges | mean runtime (s) | max supplied-pair violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| FLOP | 6.0 | 0.667 | 9.0 | 0.001 | unavailable |
| FLOP + NOTREKS | 12.5 | 0.232 | 8.5 | 0.020 | 0 |
| DAGMA | 6.0 | 0.535 | 7.0 | 0.399 | 0 |
| DAGMA + NOTREKS | 6.0 | 0.535 | 7.0 | 1.186 | 0 |

This tiny smoke validates execution and feasibility, not superiority. In
particular, FLOP-NOTREKS was worse than FLOP on these two seeds, so the cluster
benchmark must report individual scenarios and paired effects rather than
assuming prior knowledge helps.
