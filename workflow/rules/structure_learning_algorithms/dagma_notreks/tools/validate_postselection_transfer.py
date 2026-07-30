#!/usr/bin/env python3
"""Compare deterministic transferable policies against the source module."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.rules.structure_learning_algorithms.dagma_notreks.postselection import (
    LinearCandidateScorer,
    PostselectionConfig,
    select_postselection_candidate,
    standardize_training_data,
)


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SOURCE = Path("/Users/fbleile/Projects/benchpress")
EQUIVALENT_POLICIES = (
    "PS1_joint_feasible_greedy_score",
    "PS4_joint_violation_repair",
    "PS5_fixed_threshold_joint_feasible",
)


def load_source(source_root):
    path = source_root / (
        "workflow/rules/structure_learning_algorithms/"
        "dagma_anytime/postselection.py")
    name = (
        "workflow.rules.structure_learning_algorithms.dagma_anytime."
        "_transfer_source_postselection")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(source_root):
    source = load_source(source_root)
    rows = []
    for d in (20, 50):
        for seed in (1001, 1002, 1003):
            rng = np.random.default_rng(seed + d)
            X, _, _ = standardize_training_data(
                rng.normal(size=(200, d)))
            covariance = X.T @ X / len(X)
            W = (
                rng.normal(scale=.15, size=(d, d))
                * (rng.random((d, d)) < .01))
            np.fill_diagonal(W, 0)
            pair_sets = {
                "DAG_only": (),
                "DAG_NOTREKS_one_correct": ((0, 1),),
                "DAG_NOTREKS_several_correct": tuple(
                    (index, index + 1)
                    for index in range(0, min(d - 1, 8), 2)),
            }
            for regime, pairs in pair_sets.items():
                for policy in EQUIVALENT_POLICIES:
                    target = select_postselection_candidate(
                        W, scorer=LinearCandidateScorer(
                            X, regularizer_weight=.03),
                        config=PostselectionConfig(
                            policy=policy,
                            notreks_constraint_active=bool(pairs)),
                        notreks_pairs=pairs)
                    old = source.select_postselection_candidate(
                        W, covariance, len(X), policy=policy,
                        notreks_pairs=pairs,
                        notreks_constraint_active=bool(pairs),
                        regularizer_type="L1", regularizer_weight=.03)
                    same = np.array_equal(target.adjacency, old.adjacency)
                    rows.append({
                        "d": d, "seed": seed, "constraint_regime": regime,
                        "policy": policy, "support_equal": same,
                        "target_edges": target.predicted_edges,
                        "source_edges": old.predicted_edges,
                        "target_DAG_valid": target.DAG_valid,
                        "source_DAG_valid": old.DAG_valid,
                        "target_notreks_violations":
                            target.notreks_violation_count,
                        "source_notreks_violations":
                            old.notreks_violation_count,
                        "target_score": target.candidate_score,
                        "source_score": old.candidate_score,
                        "score_absolute_difference": abs(
                            target.candidate_score - old.candidate_score),
                    })
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "results/dagma_notreks_oracle/postselection_transfer_validation"))
    args = parser.parse_args(argv)
    frame = run(args.source_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "source_equivalence.csv", index=False)
    failures = frame[~frame.support_equal]
    print(frame.groupby(
        ["d", "constraint_regime", "policy"]).support_equal.mean())
    if len(failures):
        raise SystemExit(
            f"{len(failures)} deterministic source-equivalence rows differ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
