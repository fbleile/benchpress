import json
from pathlib import Path

import numpy as np

from workflow.rules.structure_learning_algorithms.pc_mi_oracle.pc import (
    MIOracleFisherZ,
    stable_pc,
)


ROOT = Path(__file__).resolve().parents[5]


def test_dispatch_and_counts_use_only_supplied_pairs():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 3))
    test = MIOracleFisherZ(X, [(0, 1)], alpha=.05)
    assert test.independent(0, 1)  # collider parents supplied as no-trek
    assert not test.independent(0, 2)  # chain/fork-style trek: dependent structurally
    assert not test.independent(1, 2)
    test.independent(0, 1, [2])
    assert test.counts.oracle_marginal_queries == 3
    assert test.counts.oracle_independent_answers == 1
    assert test.counts.oracle_dependent_answers == 2
    assert test.counts.statistical_conditional_tests == 1


def test_declared_cpdag_metadata_and_endpoint_representation():
    info = json.loads(
        (ROOT / "workflow/rules/structure_learning_algorithms/pc_mi_oracle/info.json"
         ).read_text())
    assert info["graph_types"] == ["CPDAG"]
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 4))
    graph, _ = stable_pc(X, MIOracleFisherZ(X, []), max_cond_set=1)
    assert graph.shape == (4, 4)
    assert set(np.unique(graph)).issubset({0, 1})
    assert np.all(np.diag(graph) == 0)
