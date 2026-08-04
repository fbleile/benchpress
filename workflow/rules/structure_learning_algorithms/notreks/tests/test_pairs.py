import pytest

from workflow.rules.structure_learning_algorithms.notreks import subsample_no_trek_pairs


def test_pair_subsampling_is_canonical_deterministic_and_shared():
    pairs = [(4, 1), (0, 2), (1, 4), (2, 3), (0, 4)]
    first = subsample_no_trek_pairs(pairs, 0.5, 19)
    second = subsample_no_trek_pairs(list(reversed(pairs)), 0.5, 19)
    assert first == second
    assert len(first) == 2
    assert all(i < j for i, j in first)


def test_pair_subsampling_boundaries_and_validation():
    pairs = [(0, 1), (0, 2)]
    assert subsample_no_trek_pairs(pairs, 0.0, 1) == []
    assert subsample_no_trek_pairs(pairs, 0.01, 1)
    assert subsample_no_trek_pairs(pairs, 1.0, 1) == pairs
    with pytest.raises(ValueError):
        subsample_no_trek_pairs(pairs, 1.01, 1)
