NOTREKS
=======

NOTREKS is a Benchpress structure-learning module with a linear optimizer,
optional no-trek regularization, cached marginal-independence tests, and a
small hyperparameter-selection workflow.

Experiment layout
-----------------

Human-edited grids live under::

  configs/notreks/grids/

Generated Benchpress configs and manifests are written under::

  configs/notreks/expanded/

Selection outputs that define the next run are written under::

  configs/notreks/selected/

Benchmark outputs remain in Benchpress's normal ``results/`` tree.  Generated
NOTREKS fixed-data resources are reproducible and are ignored by git.

Grid expansion
--------------

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

Local dry-run
-------------

Benchpress currently uses Snakemake 7 for Python-3.7 gCastle container
compatibility, so use ``--use-singularity`` locally and on LRZ::

  snakemake -n \
    --cores 1 \
    --use-singularity \
    --snakefile workflow/Snakefile \
    --configfile configs/notreks/expanded/smoke_config.json

SLURM
-----

Smoke and full validation use the same Snakemake-driver mechanism; smoke only
uses fewer datasets, fewer variants, and smaller resources.  On LRZ the driver
loads ``apptainer/1.3.4`` and ``squashfs/4.6.1`` because Apptainer image pulls
need ``mksquashfs``.  With Snakemake 7 the driver chooses
``--use-singularity`` and creates a local ``singularity -> apptainer`` shim when
needed.

Smoke submit::

  mkdir -p results/notreks/smoke/logs/slurm
  RUN_DIR=results/notreks/smoke \
  CONFIG=configs/notreks/expanded/smoke_config.json \
  SNAKEMAKE_CORES=8 \
  sbatch --clusters=serial \
    --export=ALL,RUN_DIR=results/notreks/smoke,CONFIG=configs/notreks/expanded/smoke_config.json,SNAKEMAKE_CORES=8 \
    -o results/notreks/smoke/logs/slurm/%x-%j.out \
    -e results/notreks/smoke/logs/slurm/%x-%j.err \
    workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_serial_smoke.sh

Full validation submit on ``serial_std``::

  mkdir -p results/notreks/full_benchmark/logs/slurm
  RUN_DIR=results/notreks/full_benchmark \
  CONFIG=configs/notreks/expanded/full_benchmark_config.json \
  SNAKEMAKE_CORES=16 \
  sbatch --clusters=serial \
    --export=ALL,RUN_DIR=results/notreks/full_benchmark,CONFIG=configs/notreks/expanded/full_benchmark_config.json,SNAKEMAKE_CORES=16 \
    -o results/notreks/full_benchmark/logs/slurm/%x-%j.out \
    -e results/notreks/full_benchmark/logs/slurm/%x-%j.err \
    workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_serial_true.sh

If the full grid needs more one-node parallelism, use the existing
``notreks_driver_cm4_tiny_true.sh`` script with ``SNAKEMAKE_CORES=32``.
Do not use multi-node ``cm4_std`` resources unless the workload is explicitly
changed to a distributed workflow.

Selection
---------

After validation finishes, select one setting per method family::

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
    select-validation-best \
    --config configs/notreks/expanded/smoke_config.json \
    --manifest configs/notreks/expanded/smoke_manifest.csv \
    --out-dir configs/notreks/selected \
    --tag smoke \
    --primary-metric SHD_cpdag

The selector joins Benchpress metrics to the manifest by ``algorithm_id`` and
writes ``<tag>_best_by_method_family.json`` plus
``<tag>_validation_summary.csv``.

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
