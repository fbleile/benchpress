import json
from pathlib import Path

import numpy as np
import pytest

from workflow.rules.structure_learning_algorithms.dagma.knowledge import (
    named_pairs_to_indices, validate_sidecar,
)
from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag
from workflow.rules.structure_learning_algorithms.flop_notreks.adapter import (
    count_no_trek_violations, selected_dag_from_diagnostics,
)


ROOT = Path(__file__).resolve().parents[5]


def test_shared_sidecar_alignment_and_selected_dag_violation_check():
    payload = validate_sidecar({
        "type": "no_trek_pairs",
        "node_names": ["A", "B", "C"],
        "pairs": [["A", "B"]],
    }, ["A", "B", "C"])
    assert named_pairs_to_indices(payload, ["A", "B", "C"]) == [(0, 1)]
    with pytest.raises(ValueError):
        named_pairs_to_indices(payload, ["B", "A", "C"])
    collider = selected_dag_from_diagnostics(
        {"selected_dag_edges": [[0, 2], [1, 2]]}, 3)
    assert count_no_trek_violations(collider, [(0, 1)]) == 0
    fork = selected_dag_from_diagnostics(
        {"selected_dag_edges": [[2, 0], [2, 1]]}, 3)
    assert count_no_trek_violations(fork, [(0, 1)]) == 1


def test_cpdag_encoding_conversion():
    raw = np.array([[0, 2, 1], [2, 0, 0], [0, 0, 0]])
    converted = convert_flop_cpdag(raw, 3)
    assert np.array_equal(converted, np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]]))


def test_cleaned_main_comparison_has_distinct_stable_method_names():
    config = json.loads(
        (ROOT / "configs/dagma_notreks_oracle/grids/"
         "main_comparison.json").read_text())
    assert config["methods"] == ["flop", "dagma_notreks", "inv_notreks"]
    assert len(config["methods"]) == len(set(config["methods"]))
