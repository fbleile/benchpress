# LRZ JobFarm run

The cluster preparation creates one independent command per method. A failed
method therefore does not abort the other methods, and JobFarm can resume a
partially completed task database.

The current cluster target is CoolMUC-4 `cm4_std`. LRZ documents this queue as
an exclusive 2--4 node queue with a maximum wall time of 24 hours; the
provided submission wrapper uses four nodes and one CPU per task. See the
[LRZ job-processing documentation](https://doku.lrz.de/job-processing-on-the-linux-cluster-10745970.html)
for current queue limits.

## First login and environment smoke test

On an LRZ login node, clone or update this branch and note the absolute path:

```bash
git clone git@github.com:fbleile/benchpress.git
cd benchpress
git fetch origin
git checkout notreks-oracle
git pull --ff-only origin notreks-oracle
export PROJECT_DIR="$PWD"
```

Use the cluster's available Miniconda/Conda module to create an environment.
The exact module name is site-specific, so inspect it first:

```bash
module avail 2>&1 | egrep -i 'conda|miniconda|python|jobfarm|slurm'
module load slurm_setup
module load jobfarm
```

Then create or activate the project environment and install the repository's
Python dependencies. Build/install the bundled FLOP extension in that same
environment. Before submitting the farm, verify:

```bash
env PYTHONPATH="$PROJECT_DIR" .venv-lrz/bin/python -c \
  'import numpy, scipy, pandas, sklearn, torch, flopsearch; print("environment ok")'
```

If the cluster environment uses a different Python path, pass it to the
command generator with `--python /absolute/path/to/python`.

## Official static causalAssembly cache

The benchmark should use the official static upstream data, not the expensive
R/DRF generator, when that data is available. Download the upstream data and
truth JSON into the project (or copy them from a shared project filesystem),
then prepare a resumable cache. The repository records the official upstream
URLs; on the cluster the files can be fetched directly with:

```bash
mkdir -p "$PROJECT_DIR/cluster/causalassembly_input"
curl -L --fail --retry 3 \
  https://raw.githubusercontent.com/boschresearch/causalAssembly/main/data/data_sets/n_500_synthdata/assembly_line_500.csv \
  -o "$PROJECT_DIR/cluster/causalassembly_input/assembly_line_500.csv"
curl -L --fail --retry 3 \
  https://raw.githubusercontent.com/boschresearch/causalAssembly/main/data/ground_truth/ground_truth.json \
  -o "$PROJECT_DIR/cluster/causalassembly_input/ground_truth.json"
```

Then prepare a resumable cache:

```bash
env PYTHONPATH="$PROJECT_DIR" .venv-lrz/bin/python \
  scripts/causalassembly_protocol.py prepare-static \
  --cache "$PROJECT_DIR/cluster/causalassembly_cache" \
  --data-csv "$PROJECT_DIR/cluster/causalassembly_input/assembly_line_500.csv" \
  --truth-json "$PROJECT_DIR/cluster/causalassembly_input/ground_truth.json" \
  --seeds 1001 1002 1003 1004 1005
```

`prepare-static` writes one compressed `.npz` per seed and a manifest. It can
be rerun safely; completed seeds remain cached. The current production
protocol uses `n=500`. A separate paper-style causalAssembly reproduction
(98 nodes, 485 edges, 50 resamples of 5000 observations) should be launched
as a separately named job set because its cost and method exclusions differ.

## Generate the disjoint command file

```bash
env PYTHONPATH="$PROJECT_DIR" .venv-lrz/bin/python \
  scripts/lrz_make_jobfarm_cmds.py \
  --repo "$PROJECT_DIR" \
  --python .venv-lrz/bin/python \
  --output-root "$PROJECT_DIR/results/lrz_full" \
  --command-file "$PROJECT_DIR/cluster/notreks_cmd.txt" \
  --causal-seeds "1001 1002 1003 1004 1005"
wc -l "$PROJECT_DIR/cluster/notreks_cmd.txt"
```

Every line invokes one method and writes to a separate `job_*` directory.
The synthetic protocol itself preserves the registered dimension/method
rules, including the current explicit DAGMA exclusion at `d=100`.

## Smoke JobFarm submission

Before the full farm, make a one-line command file for one small synthetic
job and submit the wrapper:

```bash
printf '%s\n' "env PYTHONPATH=$PROJECT_DIR $PROJECT_DIR/.venv-lrz/bin/python $PROJECT_DIR/scripts/notreks_protocol_all.py --experiments main --fraction 0.1 --methods flop --graph-replicates 1 --workers 1 --max-wall-hours 1 --output-root $PROJECT_DIR/results/lrz_smoke" > "$PROJECT_DIR/cluster/smoke_cmd.txt"
CMD_FILE="$PROJECT_DIR/cluster/smoke_cmd.txt" TASKDB="$PROJECT_DIR/cluster/lrz_smoke" \
  sbatch --chdir="$PROJECT_DIR" scripts/lrz_jobfarm.sh
```

Inspect the Slurm output and the smoke result before launching the full list.

## Full submission and resume

```bash
CMD_FILE="$PROJECT_DIR/cluster/notreks_cmd.txt" \
TASKDB="$PROJECT_DIR/cluster/notreks_cmd" \
  sbatch --chdir="$PROJECT_DIR" scripts/lrz_jobfarm.sh
```

Do not set `RESET_JOBFARM=1` for a resume. To intentionally start a new farm,
use a new `TASKDB` and output root, or explicitly set `RESET_JOBFARM=1` after
checking the paths.

After completion, merge isolated result tables:

```bash
env PYTHONPATH="$PROJECT_DIR" .venv-lrz/bin/python \
  scripts/lrz_collect_jobfarm.py \
  --root "$PROJECT_DIR/results/lrz_full" \
  --output "$PROJECT_DIR/results/lrz_full/merged_results.csv"
```

Figure generation remains a separate, single post-processing step on the
merged tables so that plotting never occupies solver workers.
