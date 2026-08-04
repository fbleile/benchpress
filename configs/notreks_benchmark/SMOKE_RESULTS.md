# One-time NOTREKS calibration smoke (`d=20`)

Calibration used standardized linear-Gaussian ER data at `d=20`, `n=200`,
seeds 9101–9105, and 25% of the available oracle no-trek pairs. Hyperparameter
selection used only zero-violation eligibility and graph-level Gaussian BIC
after the same fixed `|W| >= 0.30` threshold used by both DAGMA benchmark arms;
the truth metrics below are diagnostic.

## What FLOP+NOTREKS means here

The benchmark uses `search_strategy=global_greedy`, not the historical Rust
signature search. It is a standalone FLOP-style method:

1. propose node reinsertions in a causal order;
2. for each order, greedily toggle edges using complete-graph Gaussian BIC;
3. reject every toggle that creates a directed cycle or violates a supplied
   no-trek pair;
4. select the best restart by BIC.

It does not call vanilla FLOP internally. The older
`signature_alternating` implementation remains available as a diagnostic but
is not selected by the paper benchmark configurations.

## Frozen calibration

For DAGMA-NOTREKS, weights 0.3, 3, and 10 were tested. The matched-threshold
recheck produced mean BIC values `-1248.27`, `-1311.01`, and `-1383.45`,
respectively. Weight 10 retained the best mean BIC and is frozen. All 15
thresholded calibration graphs had zero supplied-pair violations; this was
measured rather than enforced by postselection.

For greedy FLOP-NOTREKS, `(restarts, sweeps)` values `(1,1)`, `(2,1)`, and
`(2,2)` were tested. `(2,2)` obtained the best mean BIC and is frozen. It is
substantially slower and should receive explicit cluster resources.

| Method | mean SHD | median SHD | mean F1 pattern | mean edges | mean runtime (s) | maximum supplied-pair violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLOP | 7.6 | 8 | 0.829 | 25.6 | 0.004 | unavailable |
| greedy FLOP + NOTREKS | 4.0 | 4 | 0.890 | 19.2 | 103.844 | 0 |
| DAGMA | 29.4 | 23 | 0.340 | 23.8 | 0.556 | not constrained |
| DAGMA + NOTREKS | 12.6 | 16 | 0.662 | 17.8 | 2.487 | 0 |

Greedy FLOP+NOTREKS improved over FLOP on four seeds and tied it on one. Its
per-seed SHDs were `10, 0, 4, 4, 2`, versus `10, 1, 5, 14, 8` for FLOP.
DAGMA+NOTREKS improved over DAGMA on every seed. This is encouraging, but the
panel is a small oracle calibration set and is not evidence for the main
benchmark by itself.
