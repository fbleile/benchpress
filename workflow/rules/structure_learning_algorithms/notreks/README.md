# NOTREKS Benchpress Module

This directory contains the self-contained Benchpress implementation of
NOTREKS and its local/SLURM experiment tooling.

The optimizer is used unchanged by all execution modes. The tooling under
`tools/` prepares ordinary Benchpress configs, shared fixed datasets, JobFarm
manifests, and selected final configs.

Quick local check:

```bash
conda activate benchpress-notreks
python workflow/rules/structure_learning_algorithms/notreks/tools/cli.py \
  smoke --preset tiny
```

See [docs/HOWTO_LOCAL_AND_SLURM.md](docs/HOWTO_LOCAL_AND_SLURM.md) for the
complete tuning, selection, injection, local, and JobFarm workflow.

Current continuation policy: every central-path stage receives `max_iter`.
`warm_iter` remains accepted only for compatibility with older configs.
