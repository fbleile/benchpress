import numpy as np

from udg_pilot.core import (composite_score, source_mask_udg, source_mask_witness,
                            toggle_score_delta, true_udg)


def independent_udg(dag):
    d = dag.shape[0]
    ancestors = [set([i]) for i in range(d)]
    changed = True
    while changed:
        changed = False
        for parent in range(d):
            for child in np.flatnonzero(dag[parent]):
                before = len(ancestors[child])
                ancestors[child] |= ancestors[parent]
                changed |= len(ancestors[child]) != before
    out = np.zeros((d, d), dtype=np.uint8)
    for i in range(d):
        for j in range(i + 1, d):
            out[i, j] = out[j, i] = bool(ancestors[i] & ancestors[j])
    return out


def test_hand_graphs_and_independent_ancestor_implementation():
    chain = np.zeros((3, 3), dtype=np.uint8); chain[0, 1] = chain[1, 2] = 1
    fork = np.zeros((3, 3), dtype=np.uint8); fork[0, 1] = fork[0, 2] = 1
    collider = np.zeros((3, 3), dtype=np.uint8); collider[0, 2] = collider[1, 2] = 1
    disconnected = np.zeros((3, 3), dtype=np.uint8)
    assert np.array_equal(true_udg(chain), independent_udg(chain))
    assert np.array_equal(true_udg(fork), independent_udg(fork))
    assert np.array_equal(true_udg(collider), independent_udg(collider))
    assert np.array_equal(true_udg(disconnected), independent_udg(disconnected))
    assert true_udg(collider)[0, 1] == 0


def test_source_masks_have_a_valid_witness():
    masks = np.array([1, 3, 4, 5], dtype=np.uint64)
    witness = source_mask_witness(masks)
    assert np.array_equal(source_mask_udg(masks, 4), true_udg(witness))
    assert np.all(np.diag(source_mask_udg(masks, 4)) == 0)


def test_incremental_toggle_delta_matches_full_score_for_random_proposals():
    rng = np.random.default_rng(23)
    d = 10
    masks = np.array([1 << i for i in range(d)], dtype=np.uint64)
    evidence = rng.normal(size=(d, d)); evidence = (evidence + evidence.T) / 2
    prior = 1.7
    for _ in range(100):
        node, source = rng.integers(0, d, size=2)
        bit = np.uint64(1 << source)
        add = not bool(masks[node] & bit)
        if not add and int(masks[node]).bit_count() == 1:
            continue
        before = composite_score(masks, evidence, prior)
        delta = toggle_score_delta(masks, evidence, int(node), int(source), add, prior)
        candidate = masks.copy()
        if add: candidate[node] |= bit
        else: candidate[node] &= ~bit
        assert np.isclose(before + delta, composite_score(candidate, evidence, prior))
        masks = candidate
