"""Run the paired FLOP/DAGMA Sachs experiment on the bundled fixed data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.notreks_benchmark_pipeline import method_run
from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import metrics


DATA = Path("resources/data/mydatasets/2005_sachs_2_cd3cd28icam2_log_std.csv")
TRUTH = Path("resources/adjmat/myadjmats/sachs.csv")
METHODS = (
    "flop", "flop-nt-standard", "dagma", "dagma-pstrek",
    "dagma-nonlinear", "dagma-nonlinear-pstrek",
)


def no_trek_pairs(truth: np.ndarray) -> list[tuple[int, int]]:
    """Return unordered pairs with disjoint reflexive ancestor sets."""
    reach = np.asarray(truth, dtype=bool).copy()
    for k in range(reach.shape[0]):
        reach |= reach[:, [k]] & reach[[k], :]
    ancestor = reach | np.eye(reach.shape[0], dtype=bool)
    return [(i, j) for i in range(len(ancestor)) for j in range(i + 1, len(ancestor))
            if not np.any(ancestor[:, i] & ancestor[:, j])]


def select_pairs(pairs, fraction: float, seed: int):
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("knowledge fraction must lie in [0, 1]")
    count = int(round(fraction * len(pairs)))
    if count == len(pairs):
        return list(pairs)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(pairs), size=count, replace=False)
    return [pairs[int(i)] for i in np.sort(selected)]


def run(args):
    data_path = Path(args.data)
    truth_path = Path(args.truth)
    X = pd.read_csv(data_path).to_numpy(dtype=float)
    truth = pd.read_csv(truth_path, index_col=False).to_numpy(dtype=np.uint8)
    if X.shape[1] != truth.shape[0] or truth.shape != (X.shape[1], X.shape[1]):
        raise ValueError(f"data/graph shape mismatch: {X.shape}, {truth.shape}")
    pairs = no_trek_pairs(truth)
    methods = tuple(args.methods) if args.methods else METHODS
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown Sachs methods: {sorted(unknown)}")
    names = list(pd.read_csv(data_path, nrows=0).columns)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(int(r["seed"]), int(r.get("bootstrap", 0)),
             float(r["knowledge_fraction"]), r["method"])
            for r in rows if r.get("status") == "ok"}
    pair_path = args.output / "notreks_pairs.csv"
    for seed in args.seeds:
        for bootstrap in range(args.bootstrap_replicates):
            sample_seed = seed * 1000003 + bootstrap
            rng = np.random.default_rng(sample_seed)
            X_boot = X[rng.integers(0, len(X), size=len(X))]
            for fraction in args.knowledge_fraction:
                constrained_pairs = select_pairs(
                    pairs, fraction, sample_seed + int(round(10000 * fraction)))
                named_pairs = [(names[i], names[j]) for i, j in constrained_pairs]
                print(f"sachs seed={seed} bootstrap={bootstrap} q={fraction:g} "
                      f"NOTREKS pairs={named_pairs}", flush=True)
                pair_row = pd.DataFrame([{
                    "seed": seed, "bootstrap": bootstrap,
                    "knowledge_fraction": fraction,
                    "pair_count": len(named_pairs),
                    "pairs": json.dumps(named_pairs),
                }])
                if pair_path.exists():
                    pair_row = pd.concat([pd.read_csv(pair_path), pair_row],
                                         ignore_index=True)
                pair_row.drop_duplicates(
                    subset=["seed", "bootstrap", "knowledge_fraction"],
                    keep="last").to_csv(pair_path, index=False)
                for method in methods:
                    effective_pairs = (
                        [] if method in {"flop", "dagma"}
                        else constrained_pairs)
                    key = (seed, bootstrap, float(fraction), method)
                    if key in done and not args.force:
                        continue
                    started = time.perf_counter()
                    status, error = "ok", ""
                    try:
                        candidate, diag, runtime = method_run(
                            method, X_boot, effective_pairs, sample_seed,
                            args.attempts, args.flop_sweeps,
                            args.dagma_stages, args.dagma_warm_iter,
                            args.dagma_max_iter, args.trek_weight)
                        row = metrics(
                            X_boot, truth, effective_pairs, method,
                            diag["candidate_graph"], candidate,
                            diag["cpdag"], runtime, args.attempts,
                            diag.get("optimizer_restarts", args.attempts))
                    except Exception as exc:
                        status, error = "failed", repr(exc)
                        row = {"SHD_cpdag": np.nan,
                               "candidate_runtime": time.perf_counter() - started}
                    row.update({
                        "experiment": "sachs_fixed",
                        "data_file": str(data_path),
                        "truth_file": str(truth_path),
                        "seed": seed, "bootstrap": bootstrap,
                        "n": X_boot.shape[0], "d": X_boot.shape[1],
                        "method": method, "knowledge_fraction": fraction,
                        "notreks_pairs": len(effective_pairs),
                        "notreks_pairs_named": json.dumps(
                            [(names[i], names[j]) for i, j in effective_pairs]),
                        "all_no_trek_pairs": len(pairs),
                        "standardized_input": data_path.name.endswith("_std.csv"),
                        "status": status, "error": error,
                    })
                    if args.force:
                        rows = [r for r in rows if not (
                            int(r.get("seed", -1)) == seed
                            and int(r.get("bootstrap", -1)) == bootstrap
                            and float(r.get("knowledge_fraction", -1)) == float(fraction)
                            and r.get("method") == method)]
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(path, index=False)
                    print(f"sachs seed={seed} bootstrap={bootstrap} q={fraction:g} "
                          f"{method} {status} SHD={row.get('SHD_cpdag')}", flush=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.groupby(["knowledge_fraction", "method"], dropna=False).agg(
            runs=("method", "size"), SHD_cpdag_mean=("SHD_cpdag", "mean"),
            SHD_cpdag_std=("SHD_cpdag", "std"),
            runtime_mean=("candidate_runtime", "mean"),
            failures=("status", lambda s: int((s != "ok").sum()))
        ).reset_index().to_csv(args.output / "summary.csv", index=False)
    (args.output / "manifest.json").write_text(json.dumps({
        "experiment": "sachs_fixed", "data": str(data_path),
        "truth": str(truth_path), "n": int(X.shape[0]),
        "d": int(X.shape[1]), "seeds": args.seeds,
        "knowledge_fraction": args.knowledge_fraction,
        "bootstrap_replicates": args.bootstrap_replicates,
        "methods": methods, "all_no_trek_pairs": len(pairs),
        "input_is_prestandardized": data_path.name.endswith("_std.csv"),
    }, indent=2) + "\n")
    # Use the same analysis/figures layer as the synthetic protocol. This is
    # deliberately after checkpointing, so resumed runs regenerate complete
    # figures from the accumulated table.
    from scripts.notreks_production_figures import write_protocol_figures
    write_protocol_figures(frame, args.output / "analysis" / "figures", args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--truth", type=Path, default=TRUTH)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="rerun and replace matching existing rows")
    parser.add_argument("--bootstrap-replicates", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--knowledge-fraction", type=float, nargs="+",
                        default=[.25, 1.0])
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--flop-sweeps", type=int, default=16)
    parser.add_argument("--dagma-stages", type=int, default=5)
    parser.add_argument("--dagma-warm-iter", type=int, default=30000)
    parser.add_argument("--dagma-max-iter", type=int, default=60000)
    parser.add_argument("--trek-weight", type=float, default=200.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
