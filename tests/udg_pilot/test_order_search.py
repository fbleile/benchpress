import numpy as np
import pytest

from udg_pilot.order_search import (OrderState, canonical_cliques, canonical_order,
                                    clique_cover_valid, initial_states,
                                    search_order_uec, udg_from_cliques)


def test_order_and_clique_canonicalization():
    assert canonical_order([2, 0, 1]) == (2, 0, 1)
    assert canonical_cliques([{2, 1}, {1, 2}, {0}], 3) == (frozenset({0}), frozenset({1, 2}))
    with pytest.raises(ValueError):
        canonical_order([0, 0, 1])


def test_udg_from_clique_cover_is_symmetric_and_valid():
    udg = udg_from_cliques([{0, 1, 2}, {3}], 4)
    assert np.array_equal(udg, udg.T)
    assert not np.any(np.diag(udg))
    assert clique_cover_valid([{0, 1, 2}, {3}], 4)


def test_initial_states_include_empty_complete_and_deterministic_random():
    states = initial_states(5, None, random_restarts=2, seed=7)
    assert len(states) == 4
    assert states[0].udg().sum() == 0
    assert states[1].udg().sum() == 20
    assert [s.order for s in states] == [s.order for s in initial_states(5, None, 2, 7)]


def test_order_search_is_deterministic_and_reports_cache():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(120, 4))
    kwargs = dict(restarts=3, beam_width=1, max_evaluations=30, random_seed=9)
    a, da = search_order_uec(x, **kwargs)
    b, db = search_order_uec(x, **kwargs)
    assert np.array_equal(a, b)
    assert da["score_evaluations"] == db["score_evaluations"]
    assert da["cache_hits"] == db["cache_hits"]
    assert np.array_equal(a, a.T)
