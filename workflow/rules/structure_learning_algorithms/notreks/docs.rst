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
* ``scc_power_iteration`` denotes an experimental SCC-blockwise SDCD-style
  power-iteration surrogate on ``W * W``.

``dag_reg`` is the non-negative scaling factor for this penalty.  ``dag_s`` is
kept in every config for simplicity and is relevant to ``dag_seq = "logdet"``.

The current optimizer supports ``dag_seq`` values ``none``, ``exp``,
``logdet``, and ``scc_power_iteration``.
The log-det case uses the DAGMA-style barrier
``-logdet(s * I - W * W) + d * log(s)`` at each central-path stage.  The
``exp`` case uses the NOTEARS-style ``trace(expm(W * W)) - d`` penalty.  The
``scc_power_iteration`` case follows an SCC-blockwise SDCD detached-gradient
surrogate.  For signed NOTREKS weights it constructs the NOTEARS-style
nonnegative proxy ``A = W * W`` with a zero diagonal, detects nontrivial
strongly connected components from ``A > scc_threshold``, approximates
left/right Perron vectors inside each SCC by fixed-step power iteration, forms
``G_scc = outer(u, v) / (dot(u, v) + eps)``, and optimizes
``sum(stop_gradient(G) * A)`` with JAX.  SCC detection is structural and is not
differentiated through.  The default ``power_iter_steps`` is 5 to keep the
value/gradient call closer to logdet cost in small-matrix timing checks.  This
branch is experimental and should be benchmarked before large use.  The old
names ``power_iteration`` and ``spectral_radius`` are rejected; use
``scc_power_iteration``.  The
``log`` and ``inv`` variants remain documented placeholders and raise
``NotImplementedError`` in the optimizer path.

References for the implemented DAG constraints:

* ``dag_seq = "exp"`` follows NOTEARS: Zheng, Aragam, Ravikumar, and Xing,
  "DAGs with NO TEARS: Continuous Optimization for Structure Learning", NeurIPS
  2018. Code reference: https://github.com/xunzheng/notears
* ``dag_seq = "logdet"`` follows DAGMA: Bello, Aragam, and Ravikumar, "DAGMA:
  Learning DAGs via M-matrices and a Log-Determinant Acyclicity
  Characterization", NeurIPS 2022. Code reference:
  https://github.com/kevinsbello/dagma
* ``dag_seq = "scc_power_iteration"`` is inspired by SDCD: Nazaret, Hong,
  Azizi, and Blei, "Stable differentiable causal discovery", arXiv 2023. Code
  reference: https://github.com/azizilab/sdcd/tree/master

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
* ``dag_seq = "scc_power_iteration"`` uses an SCC-blockwise SDCD-style
  surrogate on ``W * W`` with a zero diagonal.
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

Independence-test caching
-------------------------

Hyperparameter searches often reuse the same dataset and independence-test
parameters.  Generated NOTREKS hyperparameter configs therefore include an
optional ``independence_cache_dir``.  When present, the module checks that
directory before computing tests.  Cache keys include a hash of the numeric
dataset, dataset path and file hash when available, ``n``, ``d``, column names,
test name, alpha, multiple-testing correction, extra test parameters, and the
cache implementation version.

Each cache entry stores ``metadata.json``, ``all_test_results.csv``, and
``accepted_pairs.csv``.  The current cache is parameter-specific: the same
dataset plus the same test settings produces a hit, while changing alpha,
correction, test type, dimensions, data seed, or data contents produces a
different entry.  Raw test statistics and p-values are written, but decisions
are not currently recomputed across different alpha/correction settings.

When a ground-truth graph is available to the helper functions, accepted
independence pairs can be compared with graph-implied no-trek marginal
independencies.  This diagnostic reports true-positive no-trek pairs,
false-positive accepted pairs, false-negative no-trek pairs, precision, recall,
and F1.  It should be read as a graph-implied no-trek diagnostic, not as a
complete list of all true statistical marginal independencies in nonlinear or
non-Gaussian settings.

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
* ``dag_seq`` in
  ``{"none", "exp", "logdet", "scc_power_iteration"}``
* ``trek_seq`` in ``{"none", "exp", "log", "inv"}``
* ``regularizer`` in ``{"none", "l1", "l2"}``

Not yet implemented in the optimizer:

* ``dag_seq`` in ``{"log", "inv"}``

Unsupported optimizer settings raise ``NotImplementedError`` rather than being
silently mapped to another objective.
