# Linear FLOP/DAGMA benchmark

This is the terminal recipe for the current linear-Gaussian comparison. It
uses standardized observations, while the generating SCM has independent
noise scales drawn in `[0.8, 1.2]`. Knowledge is sampled independently at
fractions 0, 0.25, and 0.75. The four methods are vanilla FLOP,
`global_greedy_hybrid` (the FLOP+NOTREKS benchmark), vanilla DAGMA, and
DAGMA+NOTREKS.

## Environment

```bash
cd /Users/fbleile/Projects/benchpress/.worktrees/flop-chi-sources
conda activate <your-benchpress-environment>
# Install maturin into this exact environment; this avoids a PATH-dependent
# `command not found: maturin` failure.
python -m pip install --upgrade maturin
# If needed, install the ordinary Python dependencies in that environment:
# python -m pip install numpy scipy pandas scikit-learn tqdm torch dagma

cd workflow/rules/structure_learning_algorithms/flop_notreks/vendor/flopsearch/flop_python
PYO3_PYTHON="$(which python)" python -m maturin develop --release
cd /Users/fbleile/Projects/benchpress/.worktrees/flop-chi-sources
export PYTHONPATH=.
python -c 'import dagma, flopsearch; print("DAGMA/FLOP imports OK")'
```

The script uses the separate fused `dagma_fast` optimizer, with ordinary
checkpoint convergence (not forced full-budget execution), five restarts by
default, with canonical L1 value `0.03`. It forces the standard
descending DAGMA threshold search beginning at 0.30 (`0.30, 0.20, 0.10,
0.05, 0.03, 0.01`), selecting only feasible candidates by refit/BIC, followed by
NOTREKS feasibility repair when needed. The current exploratory coefficients
are `--dagma-weight 0.5` and `--notreks-weight 0.5` to reduce over-deletion;
the DAG and NOTREKS constraints remain independently verified after
postselection. DAGMA's existing postselection and
independent DAG/NOTREKS verification are retained.

For positive knowledge fractions, DAGMA+NOTREKS now follows a soft
continuation: unconstrained DAGMA, then 25% of the requested NOTREKS weight,
then the requested weight. All stages are retained for fixed-threshold
postselection, so NOTREKS can steer the basin without automatically forcing
the final graph to be the most aggressively pruned stage.

The default `hadamard` mapping uses (W\circ W). The alternative
`phi_log` mapping uses

\[
\phi_d(x)=\frac{2}{d}\log(1+|x|),
\]

for both the DAGMA acyclicity and NOTREKS path constraints, including the
chain-rule derivative. Select it with `--adjacency-mapping phi_log`.

## First real d=20 smoke

Run one graph and one algorithm seed across all knowledge levels:

```bash
python workflow/rules/structure_learning_algorithms/flop_notreks/tools/linear_flop_dagma_benchmark.py \
  --dimension 20 --sample-size 200 \
  --graph-seeds 2001 --algorithm-seeds 7001 \
  --knowledge-fractions 0 .25 .75 --knowledge-seed 9101 \
  --graph-family er --noise-scale-spread .2 \
  --flop-restarts 2 --flop-sweeps 100 --dagma-restarts 2 \
  --dagma-lambda1 0.03 --dagma-weight 0.5 --notreks-weight 0.5 \
  --output-dir results/dagma_notreks_oracle/linear_d20_smoke
```

The run is intentionally not iteration-capped below production defaults and
can take a while. The CSV is written when the invocation finishes. For a
restartable multi-case run, invoke one graph/seed at a time and add
`--append`; completed invocations are de-duplicated by graph, algorithm,
knowledge, graph family, and method.

```bash
for g in 2001 2002 2003 2004 2005; do
  python workflow/rules/structure_learning_algorithms/flop_notreks/tools/linear_flop_dagma_benchmark.py \
    --dimension 20 --sample-size 200 --graph-seeds "$g" --algorithm-seeds 7001 8017 \
    --knowledge-fractions 0 .25 .75 --knowledge-seed 9101 \
    --graph-family er --flop-restarts 2 --flop-sweeps 100 --dagma-restarts 2 \
    --dagma-lambda1 0.03 --dagma-weight 0.5 --notreks-weight 0.5 --append \
    --output-dir results/dagma_notreks_oracle/linear_d20_multiseed
done
```

The script prints per-run rows and a grouped mean table. For interpretation,
prioritize zero independently verified NOTREKS violations, skeleton SHD/F1,
directed SHD, and Gaussian BIC (all final supports are refit under the same
standardized data). Compare runtimes separately; DAGMA optimizer time is also
recorded. At each knowledge fraction, pair methods on the same graph,
algorithm seed, and knowledge seed before counting wins/ties/losses.

## Scaling after d=20

Only proceed if the smoke does not show a clear hard-feasibility or recovery
failure for the NOTREKS methods. Start with two graph seeds and one algorithm
seed; add seeds only after inspecting the paired table.

```bash
# d=50, ER graphs
python workflow/rules/structure_learning_algorithms/flop_notreks/tools/linear_flop_dagma_benchmark.py \
  --dimension 50 --sample-size 1000 --graph-seeds 5001 5002 \
  --algorithm-seeds 7001 --knowledge-fractions 0 .25 .75 \
  --graph-family er --flop-restarts 2 --flop-sweeps 100 --dagma-restarts 2 \
  --notreks-weight 1.0 --output-dir results/dagma_notreks_oracle/linear_d50_er

# d=100, ER graphs (run only after d=50 is acceptable)
python workflow/rules/structure_learning_algorithms/flop_notreks/tools/linear_flop_dagma_benchmark.py \
  --dimension 100 --sample-size 1500 --graph-seeds 10001 10002 \
  --algorithm-seeds 7001 --knowledge-fractions 0 .25 .75 \
  --graph-family er --flop-restarts 2 --flop-sweeps 100 --dagma-restarts 2 \
  --notreks-weight 1.0 --output-dir results/dagma_notreks_oracle/linear_d100_er
```

For structural diversity, repeat the same command with `--graph-family hub`,
`chain_fork`, and `preferential`. These are still linear SCMs; they change
only the DAG topology. Keep `--noise-scale-spread .2` fixed for the primary
comparison, then optionally repeat with `.4` as a robustness check.

## Future benchmark tracks

The generic gFLOP order engine is nonlinear-backend-ready, but this branch
does not yet contain a validated nonlinear DAGMA backend. Do not label a
neural/generic run as available until that backend is implemented and given
the same independent support verification. The next safe track is therefore
the graph-family and d=50/d=100 sweep above. A future nonlinear command should
reuse this seed/knowledge layout and replace only the backend, for example:

```bash
# Planned interface; currently unavailable until the nonlinear backend lands.
python workflow/rules/structure_learning_algorithms/flop_notreks/tools/nonlinear_flop_dagma_benchmark.py \
  --score neural_anm --dimension 20 --knowledge-fractions 0 .25 .75 \
  --graph-seeds 2001 2002 --algorithm-seeds 7001 8017
```
