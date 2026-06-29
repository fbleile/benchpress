NOTREKS
=======

NOTREKS is a Benchpress structure-learning module with a linear optimizer,
optional no-trek regularization, cached marginal-independence tests, and a
small validation/selection workflow.  The concise copy-paste command sheet is
``workflow/rules/structure_learning_algorithms/notreks/COMMANDS.md``.

NOTREKS experiment workflow
---------------------------

There are two phases.

1. Validation / selection phase
   Expand a Cartesian hyperparameter grid, run it, then select one best
   configuration per method family.

2. Selected benchmark phase
   Combine fresh benchmark data with the selected methods and run the final
   benchmark.  This phase does not rerun the full hyperparameter grid.

::

  Validation grid JSON
        |
        v
  expand-grid
        |
        v
  expanded validation Benchpress config + manifest
        |
        v
  local dry-run or SLURM run
        |
        v
  select-validation-best
        |
        v
  selection JSON
        |
        v
  build-selected-benchmark + benchmark frame
        |
        v
  selected benchmark Benchpress config + manifest
        |
        v
  local dry-run or SLURM run

Tag-based paths
---------------

The normal CLI interface is driven by ``--tag``.  For example,
``--tag hyperparam`` derives:

* grid: ``configs/notreks/grids/hyperparam_grid.json``
* expanded config: ``configs/notreks/expanded/hyperparam_config.json``
* manifest CSV/JSON: ``configs/notreks/expanded/hyperparam_manifest.csv/json``
* selection outputs: ``configs/notreks/selected/hyperparam_*``
* run directory: ``results/notreks/hyperparam``

Explicit path arguments remain available for debugging and unusual runs.

Local dry-run vs SLURM run
--------------------------

A local dry-run only checks that the generated Benchpress/Snakemake config
produces a valid DAG.  It does not run the benchmark.

Use local dry-runs before submitting to SLURM.

A SLURM run actually executes the jobs on LRZ.  Snakemake 7 is used for
compatibility with the Python-3.7 gCastle containers, so the flag is
``--use-singularity``.  The LRZ driver loads Apptainer and creates a
``singularity -> apptainer`` shim if needed.

Dry-run template::

  snakemake -n \
    --cores 1 \
    --use-singularity \
    --snakefile workflow/Snakefile \
    --configfile <CONFIG>

Cluster setup
-------------

::

  cd ~/benchpress
  git pull --ff-only origin add-notreks-module

  eval "$(~/bin/micromamba shell hook -s bash)"
  micromamba activate benchpress-notreks

  module load apptainer/1.3.4
  module load squashfs/4.6.1

  export PYTHONPATH="$PWD:${PYTHONPATH:-}"

Four canonical workflows
------------------------

1. Smoke validation + selection::

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py expand-grid --tag smoke

  snakemake -n \
    --cores 1 \
    --use-singularity \
    --snakefile workflow/Snakefile \
    --configfile configs/notreks/expanded/smoke_config.json

  mkdir -p results/notreks/smoke/logs/slurm
  RUN_DIR=results/notreks/smoke \
  CONFIG=configs/notreks/expanded/smoke_config.json \
  SNAKEMAKE_CORES=8 \
  sbatch --clusters=serial \
    --export=ALL,RUN_DIR=results/notreks/smoke,CONFIG=configs/notreks/expanded/smoke_config.json,SNAKEMAKE_CORES=8 \
    -o results/notreks/smoke/logs/slurm/%x-%j.out \
    -e results/notreks/smoke/logs/slurm/%x-%j.err \
    workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_smoke.sh

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
    select-validation-best --tag smoke --primary-metric SHD_cpdag

2. Smoke selected benchmark::

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
    build-selected-benchmark --tag smoke

  snakemake -n \
    --cores 1 \
    --use-singularity \
    --snakefile workflow/Snakefile \
    --configfile configs/notreks/expanded/selected_smoke_benchmark_config.json

  mkdir -p results/notreks/benchmark_smoke/logs/slurm
  RUN_DIR=results/notreks/benchmark_smoke \
  CONFIG=configs/notreks/expanded/selected_smoke_benchmark_config.json \
  SNAKEMAKE_CORES=8 \
  sbatch --clusters=serial \
    --export=ALL,RUN_DIR=results/notreks/benchmark_smoke,CONFIG=configs/notreks/expanded/selected_smoke_benchmark_config.json,SNAKEMAKE_CORES=8 \
    -o results/notreks/benchmark_smoke/logs/slurm/%x-%j.out \
    -e results/notreks/benchmark_smoke/logs/slurm/%x-%j.err \
    workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_smoke.sh

3. Hyperparameter validation + selection::

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py expand-grid --tag hyperparam

  snakemake -n \
    --cores 1 \
    --use-singularity \
    --snakefile workflow/Snakefile \
    --configfile configs/notreks/expanded/hyperparam_config.json

  mkdir -p results/notreks/hyperparam/logs/slurm
  RUN_DIR=results/notreks/hyperparam \
  CONFIG=configs/notreks/expanded/hyperparam_config.json \
  SNAKEMAKE_CORES=16 \
  sbatch --clusters=serial \
    --export=ALL,RUN_DIR=results/notreks/hyperparam,CONFIG=configs/notreks/expanded/hyperparam_config.json,SNAKEMAKE_CORES=16 \
    -o results/notreks/hyperparam/logs/slurm/%x-%j.out \
    -e results/notreks/hyperparam/logs/slurm/%x-%j.err \
    workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_heavy.sh

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
    select-validation-best --tag hyperparam --primary-metric SHD_cpdag

  python workflow/rules/structure_learning_algorithms/notreks/tools/hyperparam_analysis.py \
    --tag hyperparam

4. Full selected benchmark::

  python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
    build-selected-benchmark --tag full_benchmark

  snakemake -n \
    --cores 1 \
    --use-singularity \
    --snakefile workflow/Snakefile \
    --configfile configs/notreks/expanded/selected_full_benchmark_config.json

  mkdir -p results/notreks/benchmark_full/logs/slurm
  RUN_DIR=results/notreks/benchmark_full \
  CONFIG=configs/notreks/expanded/selected_full_benchmark_config.json \
  SNAKEMAKE_CORES=16 \
  sbatch --clusters=serial \
    --export=ALL,RUN_DIR=results/notreks/benchmark_full,CONFIG=configs/notreks/expanded/selected_full_benchmark_config.json,SNAKEMAKE_CORES=16 \
    -o results/notreks/benchmark_full/logs/slurm/%x-%j.out \
    -e results/notreks/benchmark_full/logs/slurm/%x-%j.err \
    workflow/rules/structure_learning_algorithms/notreks/slurm/notreks_driver_heavy.sh

Marginal trek graph baseline
----------------------------

``marginal_trek_graph`` is a standalone Benchpress structure-learning method.
It tests only marginal pairs ``X_i`` and ``X_j`` with no conditioning set,
starts from a complete undirected graph, and removes an edge when marginal
independence is accepted.  The output ``adjmat.csv`` is symmetric with a zero
diagonal.  Interpret it mainly through pattern/skeleton metrics such as
``SHD_pattern`` and skeleton FPR/FNR.

Manual NOTREKS example
----------------------

This is independent of Snakemake and intended as a small direct NOTREKS
example::

  python workflow/rules/structure_learning_algorithms/notreks/tests/manual_notreks_example.py

Monitoring and cleanup
----------------------

::

  squeue -u $USER --clusters=serial,cm4
  tail -f "$(ls -t results/notreks/smoke/logs/slurm/*.out | head -1)"
  tail -f "$(ls -t results/notreks/benchmark_full/logs/slurm/*.out | head -1)"

  sacct -M serial \
    --starttime="$(date -d '1 day ago' '+%Y-%m-%dT%H:%M:%S')" \
    --endtime=now \
    --user=$USER \
    --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,MaxRSS,ReqCPUS,ReqMem

Cleanup warning: this deletes benchmark outputs and Snakemake state.  Do not
delete ``configs/notreks`` unless intentionally resetting experiment configs.::

  rm -rf results
  rm -rf .snakemake
  mkdir -p results

Thresholds
----------

NOTREKS uses the canonical config parameter ``threshold`` to turn fitted
weights into ``adjmat.csv``.  Benchpress ROC output, when present, uses
``thresh`` for evaluation rows.  gCastle DirectLiNGAM also has its own wrapper
parameter named ``thresh``.  Internally, NOTREKS does not infer thresholds from
algorithm ids.

DAG constraints
---------------

Implemented names:

* ``dag_seq="None"`` or ``dag_seq="none"``: no DAG penalty.
* ``dag_seq="exp"``: NOTEARS exponential-trace acyclicity constraint.
  Reference: Zheng et al. (2018).
* ``dag_seq="logdet"``: DAGMA log-det acyclicity barrier.
  Reference: Bello et al. (2022).
* ``dag_seq="scc_power_iteration"``: experimental SCC-blockwise SDCD-style
  detached Perron-gradient surrogate using ``A = W * W`` with zero diagonal.
  Reference: Nazaret et al. (2023).

The old aliases ``power_iteration`` and ``spectral_radius`` are intentionally
rejected.

Trek penalty placement
----------------------

``trek_penalty_mu_mode`` controls the no-trek penalty scaling.

* ``hard_outside_mu`` (default):
  ``mu * (score + regularizer_scale * R) + dag_reg * h + trek_reg * T``.
* ``soft_inside_mu``:
  ``mu * (score + regularizer_scale * R + trek_reg * T) + dag_reg * h``.

Independence cache
------------------

When ``independence_cache_dir`` is present in the manifest-resolved
hyperparameters, NOTREKS caches marginal-independence test statistics and
p-values in a parameter-specific entry.  Cache writes use atomic temporary-file
replacement, and malformed cache entries are ignored and recomputed instead of
crashing a run.  Diagnostics can compare accepted pairs with graph-implied no-trek
marginal independence; that phrase is structural and should not be read
as all statistical marginal independencies in every nonlinear or non-Gaussian
regime.

Hyperparameter analysis
-----------------------

``tools/hyperparam_analysis.py`` reads Benchpress ``joint_benchmarks.csv`` and
joins it to ``configs/notreks/expanded/<tag>_manifest.json`` by resolving
Benchpress result ids against manifest ``algorithm_id`` and short ``path_id``
fields.  It no longer reads ``ROC_data.csv``.  If no evaluated threshold column
is present, the report uses the configured ``threshold`` from the manifest.
