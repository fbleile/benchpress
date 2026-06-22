#!/usr/bin/env python
"""
Small manually runnable NOTREKS example.

Run from repo root:
python workflow/rules/structure_learning_algorithms/notreks/tests/manual_notreks_example.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from independence_tests import pairwise_independence_candidates  # noqa: E402
from notreks_core import NotreksConfig, threshold_adjacency  # noqa: E402
from optimizer import fit_notreks_optimizer  # noqa: E402


def _standardize(data: np.ndarray) -> np.ndarray:
    centered = data - np.mean(data, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    return centered / scale


def _synthetic_dataset(seed: int = 123, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    weights = np.array(
        [
            [0.0, 0.7, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.6, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, weights.shape[0]))
    data = noise @ np.linalg.inv(np.eye(weights.shape[0]) - weights)
    return _standardize(data), (weights != 0.0).astype(int)


def _config() -> NotreksConfig:
    return NotreksConfig(
        algorithm_id="manual-notreks-example",
        function_class="linear",
        score="least_squares",
        dag_seq="exp",
        dag_reg=1.0,
        dag_s=1.0,
        trek_seq="exp",
        trek_reg=1.0,
        regularizer="l1",
        regularizer_scale=0.01,
        independence_test="spearman",
        independence_alpha=0.05,
        independence_correction="benjamini-hochberg",
        seed=123,
        max_iter=300,
        lr=0.0003,
        path_steps=2,
        mu_init=1.0,
        mu_factor=0.1,
        tol=1e-6,
        threshold=0.1,
        timeout=None,
        init="zero",
        checkpoint=100,
        power_iter_steps=5,
        scc_threshold=1e-8,
        independence_cache_dir=None,
        warm_iter=None,
    )


def main() -> None:
    seed = 123
    n = 300
    X, true_adj = _synthetic_dataset(seed=seed, n=n)
    cfg = _config()
    columns = [f"X{i}" for i in range(X.shape[1])]

    indep = pairwise_independence_candidates(
        X,
        method=cfg.independence_test,
        alpha=cfg.independence_alpha,
        correction=cfg.independence_correction,
        columns=columns,
    )
    start = time.time()
    W, diagnostics = fit_notreks_optimizer(X, cfg, indep.pairs)
    runtime = time.time() - start
    estimated_adj = threshold_adjacency(W, cfg.threshold)

    out_dir = Path(tempfile.mkdtemp(prefix="notreks_manual_example_"))
    pd.DataFrame(X, columns=columns).to_csv(out_dir / "data.csv", index=False)
    pd.DataFrame(true_adj, columns=columns).to_csv(out_dir / "true_adjacency.csv", index=False)
    pd.DataFrame(estimated_adj, columns=columns).to_csv(out_dir / "estimated_adjacency.csv", index=False)
    (out_dir / "diagnostics.json").write_text(json.dumps(asdict(diagnostics), indent=2) + "\n")

    print("Generated dataset:")
    print(f"  n={n}, d={X.shape[1]}, seed={seed}")
    print()
    print("True adjacency:")
    print(true_adj)
    print()
    print("NOTREKS config:")
    print(f"  score={cfg.score}")
    print(f"  dag_seq={cfg.dag_seq}")
    print(f"  trek_reg={cfg.trek_reg}")
    print(f"  independence_test={cfg.independence_test}")
    print()
    print("Accepted marginal independence pairs:")
    print(f"  {indep.pairs}")
    print()
    print("Estimated adjacency:")
    print(estimated_adj)
    print()
    print("Diagnostics:")
    print(f"  runtime={runtime:.3f}s")
    print(f"  n_independence_tests={indep.number_of_tests}")
    print(f"  accepted_independence_pairs={len(indep.pairs)}")
    print(f"  threshold={cfg.threshold}")
    print(f"  optimizer_converged={diagnostics.converged}")
    print(f"  path_steps_completed={diagnostics.path_steps_completed}")
    print(f"  objective={diagnostics.objective:.6g}")
    print(f"  score={diagnostics.score:.6g}")
    print(f"  dag_penalty={diagnostics.dag_penalty:.6g}")
    print(f"  trek_penalty={diagnostics.trek_penalty:.6g}")
    print()
    print(f"Saved outputs: {out_dir}")


if __name__ == "__main__":
    main()
