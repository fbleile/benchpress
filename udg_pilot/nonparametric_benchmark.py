"""Model-mismatch benchmark for the isolated nonparametric MIG pilot."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .nonparametric import adjacency_from_pvalues, ensemble, pairwise_test


def mismatch_data(arm, d, n, seed):
    rng = np.random.default_rng(seed)
    truth = np.zeros((d, d), dtype=np.uint8)
    if arm == "nonlinear_zero_corr":
        x = rng.normal(size=(n, d))
        x[:, 1] = x[:, 0] ** 2 - 1.0 + rng.normal(scale=.15, size=n)
        truth[0, 1] = truth[1, 0] = 1
        if d >= 4:
            x[:, 3] = np.sin(x[:, 2]) + rng.normal(scale=.15, size=n)
            truth[2, 3] = truth[3, 2] = 1
    elif arm == "non_gaussian":
        x = rng.laplace(size=(n, d))
        x[:, 1] = .8 * x[:, 0] + rng.laplace(scale=.5, size=n)
        truth[0, 1] = truth[1, 0] = 1
        if d >= 4:
            skew = rng.exponential(size=n) - 1.0
            x[:, 3] = x[:, 2] + .5 * skew
            truth[2, 3] = truth[3, 2] = 1
    else:
        raise ValueError(f"unknown mismatch arm: {arm}")
    x -= x.mean(axis=0)
    return x, truth


def metrics(est, truth):
    iu = np.triu_indices_from(truth, 1); e = est[iu].astype(bool); t = truth[iu].astype(bool)
    tp = np.sum(e & t); fp = np.sum(e & ~t); fn = np.sum(~e & t)
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"predicted_edges": int(e.sum()), "precision": p, "recall": r,
            "f1": 2 * p * r / max(1e-15, p + r)}


def fisher_adjacency(x, alpha):
    d = x.shape[1]; out = np.zeros((d, d), dtype=np.uint8)
    for i in range(d):
        for j in range(i + 1, d):
            p = pearsonr(x[:, i], x[:, j]).pvalue
            out[i, j] = out[j, i] = int(p <= alpha)
    return out


def run(args):
    rows = []
    methods = args.methods
    for arm in args.arms:
        for d in args.dimensions:
            for seed in args.seeds:
                x, truth = mismatch_data(arm, d, args.n, seed)
                cache = {}
                for method in methods:
                    started = time.perf_counter()
                    if method == "pairwise_fisherz_raw":
                        est = fisher_adjacency(x, args.alpha); diag = {"test": "pearson_fisher_baseline"}
                    elif method.startswith("pairwise_hsic"):
                        ev = pairwise_test(x, test="hsic", rank=method.endswith("_rank"),
                                           permutations=args.permutations, seed=seed)
                        est = adjacency_from_pvalues(ev, args.alpha)
                        diag = {"test": "hsic", "rank": method.endswith("_rank")}
                    elif method.startswith("pairwise_dcov"):
                        ev = pairwise_test(x, test="dcov", rank=method.endswith("_rank"),
                                           permutations=args.permutations, seed=seed)
                        est = adjacency_from_pvalues(ev, args.alpha)
                        diag = {"test": "dcov", "rank": method.endswith("_rank")}
                    elif method == "conservative_ensemble" or method == "liberal_ensemble":
                        h = cache.setdefault("h", pairwise_test(x, test="hsic", permutations=args.permutations, seed=seed))
                        c = cache.setdefault("c", pairwise_test(x, test="dcov", permutations=args.permutations, seed=seed ^ 17))
                        conservative, liberal, uncertain = ensemble(h, c, args.alpha)
                        est = conservative if method == "conservative_ensemble" else liberal
                        diag = {"uncertain_fraction": float(uncertain[np.triu_indices(d, 1)].mean())}
                    else:
                        raise ValueError(method)
                    rows.append({"arm": arm, "d": d, "n": args.n, "seed": seed,
                                 "method": method, "runtime": time.perf_counter() - started,
                                 **metrics(est, truth), **{f"diag_{k}": v for k, v in diag.items()}})
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows); frame.to_csv(out / "results.csv", index=False)
    frame.groupby(["arm", "method"])[["precision", "recall", "f1", "predicted_edges", "runtime"]].mean().to_csv(out / "summary.csv")
    (out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    print(frame.groupby(["arm", "method"])[["precision", "recall", "f1", "predicted_edges", "runtime"]].mean().round(3).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/udg_pilot/nonparametric_mismatch"))
    p.add_argument("--arms", nargs="+", default=["nonlinear_zero_corr", "non_gaussian"])
    p.add_argument("--dimensions", nargs="+", type=int, default=[10, 20])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seeds", nargs="+", type=int, default=[9101, 9102])
    p.add_argument("--permutations", type=int, default=99)
    p.add_argument("--alpha", type=float, default=.05)
    p.add_argument("--methods", nargs="+", default=["pairwise_fisherz_raw", "pairwise_hsic_raw", "pairwise_hsic_rank", "pairwise_dcov_raw", "pairwise_dcov_rank", "conservative_ensemble", "liberal_ensemble"])
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
