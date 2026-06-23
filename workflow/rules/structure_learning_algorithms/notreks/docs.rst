NOTREKS
=======

NOTREKS is a Benchpress structure-learning module with a linear optimizer,
optional no-trek regularization, cached marginal-independence tests, and a
small hyperparameter-selection workflow.

Experiment layout
-----------------

Human-edited grids live under::

  configs/notreks/grids/

Selected-method benchmark frames live under::

  configs/notreks/benchmarks/

Expanded Benchpress configs and manifests are written under::

  configs/notreks/expanded/

Selection outputs that define the next run are written under::

  configs/notreks/selected/

Benchmark outputs remain in Benchpress's normal ``results/`` tree.  Generated
NOTREKS fixed-data resources are reproducible and are ignored by git.

Validation phase
----------------

Smoke grid::

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
    expand-grid \
    --grid configs/notreks/grids/smoke_grid.json \
    --out-config configs/notreks/expanded/smoke_config.json \
    --out-manifest configs/notreks/expanded/smoke_manifest.csv

Full validation grid::

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
    expand-grid \
    --grid configs/notreks/grids/full_benchmark_grid.json \
    --out-config configs/notreks/expanded/full_benchmark_config.json \
    --out-manifest configs/notreks/expanded/full_benchmark_manifest.csv

The grid format is intentionally simple: each enabled method has a ``grid``
dictionary, and list-valued entries are expanded by Cartesian product.  The
manifest maps compact ids such as ``notreks__grid000`` back to full
hyperparameters, keeping Snakemake output paths short.

After validation finishes, select one setting per method family::

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
    select-validation-best \
    --config configs/notreks/expanded/smoke_config.json \
    --manifest configs/notreks/expanded/smoke_manifest.csv \
    --out-dir configs/notreks/selected \
    --tag smoke \
    --primary-metric SHD_cpdag

Benchmark phase
---------------

Selected-method benchmark configs combine fresh benchmark data with a selection
JSON.  They do not rerun the whole hyperparameter grid.

Smoke selected benchmark::

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
    build-selected-benchmark \
    --frame configs/notreks/benchmarks/smoke_benchmark_frame.json \
    --selection configs/notreks/selected/smoke_best_by_method_family.json \
    --out-config configs/notreks/expanded/selected_smoke_benchmark_config.json \
    --out-manifest configs/notreks/expanded/selected_smoke_benchmark_manifest.csv

Full selected benchmark::

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
    build-selected-benchmark \
    --frame configs/notreks/benchmarks/full_benchmark_frame.json \
    --selection configs/notreks/selected/full_benchmark_best_by_method_family.json \
    --out-config configs/notreks/expanded/selected_full_benchmark_config.json \
    --out-manifest configs/notreks/expanded/selected_full_benchmark_manifest.csv

Local dry-run
-------------

Benchpress currently uses Snakemake 7 for Python-3.7 gCastle container
compatibility, so use ``--use-singularity`` locally and on LRZ::

  snakemake -n \
    --cores 1 \
    --use-singularity \
    --snakefile workflow/Snakefile \
    --configfile configs/notreks/expanded/smoke_config.json

Manual NOTREKS example
----------------------

A tiny direct Python example can be run without Snakemake::

  python workflow/rules/structure_learning_algorithms/notreks/tests/manual_notreks_example.py

SLURM modes
-----------

There are only two user-facing SLURM scripts:

* ``notreks_driver_smoke.sh``: ``serial_std``, 8 cores, short walltime.
* ``notreks_driver_heavy.sh``: ``serial_std``, 16 cores, 24 hour walltime.

Both use the same common Snakemake driver.  Smoke is smaller only in datasets,
variants, and resources.  On LRZ the driver loads ``apptainer/1.3.4`` and
``squashfs/4.6.1`` because Apptainer image pulls need ``mksquashfs``.  With
Snakemake 7 the driver chooses ``--use-singularity`` and creates a local
``singularity -> apptainer`` shim when needed.

Smoke submit::

  mkdir -p results/notreks/smoke/logs/slurm
  RUN_DIR=results/notreks/smoke \
  CONFIG=configs/notreks/expanded/smoke_config.json \
  SNAKEMAKE_CORES=8 \
  sbatch --clusters=serial \
    --export=ALL,RUN_DIR=results/notreks/smoke,CONFIG=configs/notreks/expanded/smoke_config.json,SNAKEMAKE_CORES=8 \
    -o results/notreks/smoke/logs/slurm/%x-%j.out \
    -e results/notreks/smoke/logs/slurm/%x-%j.err \
    workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_smoke.sh

Heavy validation or selected-benchmark submit on ``serial_std``::

  mkdir -p results/notreks/full_benchmark/logs/slurm
  RUN_DIR=results/notreks/full_benchmark \
  CONFIG=configs/notreks/expanded/full_benchmark_config.json \
  SNAKEMAKE_CORES=16 \
  sbatch --clusters=serial \
    --export=ALL,RUN_DIR=results/notreks/full_benchmark,CONFIG=configs/notreks/expanded/full_benchmark_config.json,SNAKEMAKE_CORES=16 \
    -o results/notreks/full_benchmark/logs/slurm/%x-%j.out \
    -e results/notreks/full_benchmark/logs/slurm/%x-%j.err \
    workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_heavy.sh

DAG constraints
---------------

Implemented names:

* ``dag_seq="exp"``: NOTEARS exponential-trace acyclicity constraint from
  Zheng et al. (2018), code reference https://github.com/xunzheng/notears.
* ``dag_seq="logdet"``: DAGMA log-det acyclicity barrier from Bello et al.
  (2022), code reference https://github.com/kevinsbello/dagma.
* ``dag_seq="scc_power_iteration"``: experimental SCC-blockwise SDCD-style
  detached Perron-gradient surrogate inspired by Nazaret et al. (2023), code
  reference https://github.com/azizilab/sdcd/tree/master.

For ``scc_power_iteration``, NOTREKS uses the smooth NOTEARS-style nonnegative
proxy ``A = W * W`` with a zero diagonal, computes SCCs from the support of
``A``, and applies blockwise power iteration inside nontrivial SCCs.  The old
aliases ``power_iteration`` and ``spectral_radius`` are intentionally rejected.

Independence tests and cache
----------------------------

NOTREKS supports ``none``, ``pearson``, ``spearman``, ``hsic``, ``dcor``,
``gcastle_fisherz``, ``gcastle_g2``, and ``gcastle_chi2``.  The gCastle-backed
tests call the low-level CI test with an empty conditioning set; this is not a
full PC run.  Multiple-testing correction remains in NOTREKS.

When ``independence_cache_dir`` is present in the manifest-resolved
hyperparameters, raw test statistics and p-values are cached.  Changing alpha
or correction reuses the raw cache and recomputes accepted pairs.  Diagnostics
can compare accepted pairs with graph-implied no-trek marginal independence;
that phrase is structural and should not be read as all statistical marginal
independencies in every nonlinear or non-Gaussian regime.
