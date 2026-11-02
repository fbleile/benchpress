# `prior-structure-chrom`

This arm tests the effect of the chromatic number of fixed-size NOTREKS
sets.  It is separate from `prior-structure-flop`.

The registered grid is ER4 and WS4 with `n=500`, 100 graph seeds per cell,
and target K values from 2 through the graph-specific feasible maximum,
capped at 20.  DAGMA is run at `d=50`; FLOP is run at `d=100`.  Both the
vanilla baseline and the corresponding NOTREKS method use one solver run per
condition (`attempts=1`).

The fixed pair budgets are the rounded quarter of the ER4 mean full
NOTREKS-pair count: 138 pairs for d=50 and 695 pairs for d=100.  Each graph
seed/K cell requests 50 sets.  Sets are canonically hashed and deduplicated
across all K values for that graph seed.  Candidate construction is bounded
by `--prior-chrom-set-attempts`; unavailable sets are recorded as such rather
than retried indefinitely.

Each successful target-K construction is certified by construction: a planted
K-clique provides `chi >= K`, while the K-partition provides `chi <= K`.
Solver rows therefore store one `chromatic_number` field without running an
exact coloring solver. Detailed target, hash, attempt, and availability
metadata remain in the knowledge JSON and `set_audit.csv`.

## Tiny smoke on `cm4_tiny`

The smoke command file exercises ER4 and WS4 at both dimensions with one
graph seed, K up to 4, and three requested sets per K:

```bash
cd /dss/dsshome1/0C/ge86xim2/benchpress
CMD_FILE="$PWD/cluster/prior_structure_chrom_cm4_tiny_smoke.cmd"

JOB=$(PROJECT_DIR="$PWD" CMD_FILE="$CMD_FILE" RESET_JOBFARM=1 \
  sbatch --parsable --clusters=cm4 --partition=cm4_tiny --qos=cm4_tiny \
  --nodes=1 --ntasks=17 --cpus-per-task=1 --time=04:00:00 \
  --chdir="$PWD" \
  --export=ALL,PROJECT_DIR="$PWD",CMD_FILE="$CMD_FILE",RESET_JOBFARM=1 \
  scripts/lrz_jobfarm.sh)
JOB=${JOB%%;*}
echo "$JOB"
```

Verify that the four task markers complete and inspect `set_audit.csv` in the
four isolated output directories before launching the full arm.

## Full JobFarm command file

The full command file contains 400 disjoint tasks: four graph cells times 100
graph seeds.  Each task owns its own output directory, so no two workers
write the same CSV.

```bash
CMD_FILE="$PWD/cluster/prior_structure_chrom_cluster.cmd"
JOB=$(PROJECT_DIR="$PWD" CMD_FILE="$CMD_FILE" RESET_JOBFARM=1 \
  sbatch --parsable --clusters=cm4 --partition=cm4_std --qos=cm4_std \
  --nodes=4 --ntasks=137 --cpus-per-task=1 --time=1-00:00:00 \
  --chdir="$PWD" \
  --export=ALL,PROJECT_DIR="$PWD",CMD_FILE="$CMD_FILE",RESET_JOBFARM=1 \
  scripts/lrz_jobfarm.sh)
JOB=${JOB%%;*}
echo "$JOB"
```

The full run should only be launched after the tiny smoke confirms the
cluster environment, WS graph generation, one-restart solver paths, and
deduplication audit.
