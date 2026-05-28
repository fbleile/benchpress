NOTREKS
=======

This is a local, minimal Benchpress structure-learning module for NOTREKS.  The
implementation is intentionally small: it reads Benchpress CSV input, parses the
NOTREKS configuration, optionally computes pairwise independence-candidate
pairs, builds a weighted adjacency estimate from a linear baseline plus a first
local optimizer path, thresholds that matrix, and writes the standard Benchpress
adjacency matrix, runtime, and number-of-tests outputs.

Only ``function_class = "linear"`` is implemented.  Nonlinear function classes
may be added later.

Score
-----

``score = "least_squares"`` is intended for linear SEM-style additive-noise
models and NOTEARS-like squared-loss comparisons.

``score = "gaussian_likelihood"`` is intended for Gaussian linear-model or
covariance-based scoring, but is not optimized yet and currently raises
``NotImplementedError`` in the optimizer path.

Possible future score types include ``logistic``, ``poisson``,
``generalized_linear``, and ``nonlinear_mlp``.

DAG penalty
-----------

``dag_seq`` selects the acyclicity penalty family:

* ``none`` disables the DAG penalty.
* ``exp`` denotes a NOTEARS-style exponential trace / power-series penalty.
* ``log`` denotes a logarithmic sequence variant.
* ``inv`` denotes an inverse/resolvent-style sequence variant.
* ``logdet`` denotes a DAGMA-style log-det barrier with parameter ``dag_s``.

``dag_reg`` is the non-negative scaling factor for this penalty.  ``dag_s`` is
kept in every config for simplicity and is relevant to ``dag_seq = "logdet"``.

The current optimizer supports ``dag_seq = "none"`` and ``dag_seq = "logdet"``.
The log-det case uses the DAGMA-style barrier
``-logdet(dag_s * I - W * W) + d * log(dag_s)``.  The ``exp``, ``log``, and
``inv`` variants remain documented placeholders and raise ``NotImplementedError``
in the optimizer path.

Trek penalty
------------

``trek_seq`` selects the trek penalty sequence:

* ``none`` disables the trek penalty.
* ``exp``, ``log``, and ``inv`` select simple sequence variants.

``trek_reg`` is the non-negative scaling factor.  The candidate marginal
independencies used by the trek penalty are supplied by the optional pairwise
tests.  The current optimizer wires ``trek_seq = "exp"`` as a simple smooth
penalty that suppresses direct coefficients in both directions for accepted
candidate-independence pairs.  The exact path/trek penalty is still future work.
``trek_seq`` values ``log`` and ``inv`` raise ``NotImplementedError`` in the
optimizer path.

Regularizer
-----------

``regularizer`` is one of ``none``, ``l1``, or ``l2``.  The baseline initializer
uses ridge-style fitting for ``l2`` and simple coefficient soft-thresholding for
``l1``.  The optimizer uses a smooth approximation for ``l1`` and a squared
penalty for ``l2``.

Independence tests
------------------

``independence_test = "none"`` skips testing and uses an empty independence set.
``pearson`` and ``spearman`` test all ``d(d-1)/2`` variable pairs using
``scipy.stats``.  Large p-values are interpreted as compatibility with
independence, so those pairs are returned as candidate marginal independencies.

Pearson tests zero linear correlation.  Spearman tests zero rank correlation.
Both are proxies for marginal independence, not general independence tests;
Pearson has the usual Gaussian interpretation under Gaussian assumptions.

``hsic`` and ``dcor`` are nonlinear dependence tests when the optional
``hyppo`` package is installed.  The module imports ``hyppo`` lazily only for
those tests, so Pearson and Spearman runs do not require it.  Install with
``pip install hyppo`` to use ``independence_test = "hsic"`` or
``independence_test = "dcor"``.  For ``dcor``, the standalone ``dcor`` package
is used as a secondary fallback when available.

``independence_correction`` can be ``none``, ``bonferroni``, or
``benjamini-hochberg``.

Tolerance
---------

``tol`` is a positive numerical stopping tolerance reserved for the future
optimizer.  In the current local optimizer it controls relative objective
improvement.  Later it may also refer to gradient norm, constraint residual, or
a related stopping criterion.

Current optimizer support
-------------------------

Implemented:

* ``function_class = "linear"``
* ``score = "least_squares"``
* ``dag_seq`` in ``{"none", "logdet"}``
* ``trek_seq`` in ``{"none", "exp"}``
* ``regularizer`` in ``{"none", "l1", "l2"}``

Not yet implemented in the optimizer:

* ``score = "gaussian_likelihood"``
* ``dag_seq`` in ``{"exp", "log", "inv"}``
* ``trek_seq`` in ``{"log", "inv"}``

Unsupported optimizer settings raise ``NotImplementedError`` rather than being
silently mapped to another objective.
