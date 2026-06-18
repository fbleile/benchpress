NOTREKS
=======

This is a local, minimal Benchpress structure-learning module for NOTREKS.  The
implementation is intentionally small: it reads Benchpress CSV input, parses the
NOTREKS configuration, optionally computes pairwise independence-candidate
pairs, builds a weighted adjacency estimate from a linear baseline plus a
DAGMA-style central-path optimizer, thresholds that matrix, and writes the
standard Benchpress adjacency matrix, runtime, and number-of-tests outputs.

Only ``function_class = "linear"`` is implemented.  Nonlinear function classes
may be added later.

Score
-----

``score = "least_squares"`` is intended for linear SEM-style additive-noise
models and NOTEARS-like squared-loss comparisons.

``score = "gaussian_likelihood"`` uses a minimal diagonal-noise Gaussian linear
SEM score based on residual variances from ``X @ (I - W)``.

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

The current optimizer supports ``dag_seq`` values ``none``, ``exp``, and
``logdet``.
The log-det case uses the DAGMA-style barrier
``-logdet(s * I - W * W) + d * log(s)`` at each central-path stage.  The
``exp`` case uses the NOTEARS-style ``trace(expm(W * W)) - d`` penalty.  The
``log`` and ``inv`` variants remain documented placeholders and raise
``NotImplementedError`` in the optimizer path.

Trek penalty
------------

``trek_seq`` selects the trek penalty sequence:

* ``none`` disables the trek penalty.
* ``exp``, ``log``, and ``inv`` select simple sequence variants.

``trek_reg`` is the non-negative scaling factor.  The candidate marginal
independencies used by the trek penalty are supplied by the optional pairwise
tests.

The implemented trek penalty is matrix-function based.  Let ``A = W * W``.
The sequence determines a matrix ``F(A)``:

* ``exp`` uses ``F = expm(A)``.
* ``inv`` uses ``F = (I - A + eps I)^(-1)`` with a small ridge.
* ``log`` uses the truncated series
  ``F = I + A + A^2 / 2 + ... + A^K / K`` with ``K = 2d``.

Then ``H = F.T @ F``.  For each accepted marginal-independence pair ``(i, j)``,
the optimizer penalizes ``H[i, j]``.  Since ``H`` is symmetric and pairs are
stored once as undirected pairs, this is equivalent to using the symmetric
entry.  The current trek penalty is the sum over accepted pairs.  The penalty
discourages shared trek/connectivity mass between variables that the pairwise
tests accepted as candidate marginal independencies.  ``binom`` is a possible
future sequence but is not exposed in the schema.

Central-path optimizer
----------------------

For each central-path stage, the implemented objective is:

``mu * [score(W; X) + regularizer_scale * R(W) + trek_reg * T(W; I)] + dag_reg * h(W; s)``

The score, ordinary coefficient regularizer, and current NOTREKS trek
regularizer are multiplied by ``mu``.  The DAGMA log-det barrier is outside
``mu``.

``dag_reg`` is not ignored: for ``dag_seq = "logdet"`` it multiplies the
log-det barrier.  ``dag_reg = 1`` gives the faithful DAGMA scaling.

Optimizer controls:

* ``mu_init`` is the first central-path multiplier.
* ``mu_factor`` multiplies ``mu`` after each successful stage.
* ``path_steps`` is the number of central-path stages.
* ``max_iter`` is the inner iteration limit for every stage under the current
  ``max_every_stage`` policy.
* ``warm_iter`` remains accepted for compatibility with older configs, but the
  current optimizer does not consume it.
* ``lr`` is the Adam learning rate, with backtracking if a step leaves the
  log-det domain or increases the objective.
* ``tol`` is the relative objective-improvement stopping tolerance inside a
  stage.

The optimizer records the stage index, ``mu``, iteration budget, actual
iterations, objective terms, and convergence status. Using fewer iterations in
early continuation stages can be faster, but may pass an under-solved point to
later stages. The current implementation avoids that tradeoff by applying
``max_iter`` to every stage.

For ``dag_seq = "logdet"``, the optimizer checks the DAGMA M-matrix domain:
``s * I - W * W`` must be invertible and its inverse must not have substantially
negative entries.  Invalid trial steps are rejected and retried with a smaller
learning rate.

Regularizer
-----------

``regularizer`` is one of ``none``, ``l1``, or ``l2``.  The baseline initializer
uses ridge-style fitting for ``l2`` and simple coefficient soft-thresholding for
``l1``.  The optimizer uses raw ``sum(abs(W))`` for ``l1`` with the
corresponding ``sign(W)`` subgradient and raw ``sum(W * W)`` for ``l2`` with
gradient ``2W``.  To run without ordinary coefficient regularization, use
``regularizer = "none"`` and ``regularizer_scale = 0``.

Scaling conventions
-------------------

The objective follows the earlier local DAGMA-like scaling used for the
NOTREKS experiments:

* ``least_squares`` is centered and uses
  ``0.5 * ||XW - X||_F^2 / n``.
* ``gaussian_likelihood`` is ``0.5 * sum_j log(sigma_j^2)``.
* ``l1`` is ``sum(abs(W))``.
* ``l2`` is ``sum(W * W)``.
* ``dag_seq = "exp"`` is ``trace(expm(W * W)) - d``.
* ``dag_seq = "logdet"`` is
  ``-logdet(sI - W * W) + d log(s)``.
* ``trek_seq`` penalties use
  ``T(W; I) = sum_{(i,j) in I} H[i,j]``.

The DAG terms are raw constraint violations.  The current regularizer and trek
terms are intentionally unnormalized, matching the earlier local runs where the
module performed best under the tested hyperparameters.  If no independence
pairs are accepted, the trek value and gradient are exactly zero.

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
* ``score`` in ``{"least_squares", "gaussian_likelihood"}``
* ``dag_seq`` in ``{"none", "exp", "logdet"}``
* ``trek_seq`` in ``{"none", "exp", "log", "inv"}``
* ``regularizer`` in ``{"none", "l1", "l2"}``

Not yet implemented in the optimizer:

* ``dag_seq`` in ``{"log", "inv"}``

Unsupported optimizer settings raise ``NotImplementedError`` rather than being
silently mapped to another objective.
