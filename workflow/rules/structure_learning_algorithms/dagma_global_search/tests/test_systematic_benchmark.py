from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.systematic_notreks_d20_benchmark import (
    corrupt_knowledge,
    select_knowledge,
)


def test_corruption_replaces_exact_fraction_with_true_treks():
    true_no_treks = [(0, 1), (0, 2), (2, 3), (3, 4)]
    known = list(true_no_treks)
    corrupted = corrupt_knowledge(
        known, true_no_treks, d=5, fraction=.5, seed=11)
    true_set = set(true_no_treks)
    assert len(corrupted) == len(known)
    assert sum(pair not in true_set for pair in corrupted) == 2
    assert len(set(corrupted)) == len(corrupted)


def test_corruption_is_seeded_and_uses_only_false_pairs():
    all_pairs = [(0, 1), (1, 2)]
    known = select_knowledge(all_pairs, 1.0, seed=3)
    first = corrupt_knowledge(known, all_pairs, d=4, fraction=1.0, seed=9)
    second = corrupt_knowledge(known, all_pairs, d=4, fraction=1.0, seed=9)
    assert first == second
    assert set(first).isdisjoint(set(all_pairs))


def test_zero_corruption_preserves_selected_pairs():
    pairs = [(0, 1), (2, 3)]
    assert corrupt_knowledge(pairs, pairs, d=4, fraction=0., seed=1) == pairs
