# Upstream inspection record

Inspected 2026-07-25:

- DAGMA repository HEAD `088616885d71b56c0573cd4902c1fcbac02e649f`;
  released package `dagma==1.1.1`, Apache-2.0.
- FLOP repository HEAD `84a4a6ab7ef3afb44835a41010062487f7706187`;
  released package `flopsearch==0.3.0`, MPL-2.0.

Both packages were installed from PyPI into a temporary Python 3.13 virtual
environment. DAGMA depends on NumPy, SciPy, Torch, tqdm, and igraph. FLOP 0.3.0
provided a platform wheel and required no local Rust compiler. The installed
sources and runtime signatures were inspected. DAGMA returns a float64 weighted
square matrix and applies `w_threshold` by zeroing weights below the threshold;
it did not mutate the input in the direct test. FLOP accepts
`flop(data, lambda_bic, *, restarts=None, timeout=None)` and documents a square
CPDAG with directed value 1 and symmetric undirected value 2. Its direct tiny
call exceeded one minute in this local environment and was terminated.

Benchpress pins these versions in the module Dockerfiles; the shared optimizer
is an attributed integration adaptation of the official DAGMA linear loop.
