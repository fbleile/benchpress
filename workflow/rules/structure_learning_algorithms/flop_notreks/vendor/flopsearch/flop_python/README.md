# flopsearch

This Benchpress research fork also exposes:

```python
graph = flopsearch.flop_notreks(
    X, 2.0, no_trek_pairs,
    restarts=1, seed=1001,
    signature_top_k=5, max_signature_rounds=20,
    search_version="alternating_full_refit_b",
)
```

Pairs are zero-based integer node pairs. The default graph uses FLOP's CPDAG
encoding (directed ``1``; symmetric undirected ``2``). ``return_dag=True``
returns the selected DAG. ``return_diagnostics=True`` returns
``(graph, diagnostics)``; diagnostics include the selected DAG edge list,
total BIC, signature-search counts, timing, pruning, and the independently
verified final violation count.

Hard constraints are represented by at most 64 constrained target bits.
An edge ``u -> v`` is admissible only when ``S_v`` is a subset of ``S_u``.
Incorrect supplied knowledge can therefore exclude the true DAG.

``search_version="fixed_signature_a"`` selects the fixed-signature baseline.
The default ``alternating_full_refit_b`` performs exact full-refit
ancestry-cone promotion search and compresses signatures only at plateaus.
``incremental_c`` and ``hybrid_bc`` are reserved names and currently raise a
clear error.

The research-only deletion postprocessor is:

```python
dag, coefficients, diagnostics = flopsearch.prune_parents_bic(
    X, candidate_dag, 2.0,
    return_coefficients=True,
    return_diagnostics=True,
)
```

It runs FLOP's local grow-shrink BIC fitting independently for each node, with
the incoming edges of `candidate_dag` as the complete candidate-parent pool.
It performs no order search, is deterministic, and guarantees that the result
is a subgraph of the supplied acyclic candidate. Coefficients are refitted by
OLS after support selection.

Python package providing an implementation of the [FLOP causal discovery algorithm](https://arxiv.org/abs/2510.04970) for linear additive noise models.

## Installation
flopsearch can be installed via pip:

```bash
pip install flopsearch
```

## Citing FLOP
If you use FLOP in your scientific work, please cite this paper:
```bibtex
@article{embracing2026,
  author  = {Marcel Wien{\"o}bst and Leonard Henckel and Sebastian Weichwald},
  title   = {{Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning}},
  journal = {International Conference on Learning Representations (ICLR)},
  year    = {2026}
}
```

## Example
A simple example run of the FLOP algorithm provided by flopsearch.

``` py
import flopsearch
import numpy as np
from scipy import linalg

p = 10
W = np.diag(np.ones(p - 1), 1)
X = np.random.randn(10000, p).dot(linalg.inv(np.eye(p) - W))
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
flopsearch.flop(X_std, 2.0, restarts=50)
```

## Input and Output
As input, FLOP takes the data matrix, the BIC penalty parameter (we recommend ```2.0``` as a default choice) and either a ```timeout``` (in seconds) or the number of ILS restarts to control how long the search runs.

The output of FLOP is a CPDAG encoded with an adjacency matrix whose entry in row i and column j is 1 in case of a directed edge from the i-th to the j-th variable and 2 in case of an undirected edge between those variables (in case of an undirected edge, the entry in row j and column i is also 2, that is each undirected edge induces two 2's in the matrix).
