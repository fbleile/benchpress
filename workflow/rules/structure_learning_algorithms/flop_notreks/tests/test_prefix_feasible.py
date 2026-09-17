import numpy as np
import flopsearch


def _run(pairs, beam=1):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(160, 4))
    return flopsearch.flop_notreks(
        x, 2.0, pairs, restarts=1, seed=17,
        search_version="prefix_feasible", prefix_beam_width=beam,
        return_dag=True, return_diagnostics=True)


def test_prefix_reversed_collider_is_exactly_feasible():
    _, diagnostics = _run([(0, 1)])
    assert diagnostics["final_no_trek_violation_count"] == 0
    assert diagnostics["prefix_beam_width"] == 1
    assert diagnostics["prefix_admissible_pool_sizes"]


def test_prefix_beam_flag_is_available_and_feasible():
    _, diagnostics = _run([(0, 1)], beam=4)
    assert diagnostics["final_no_trek_violation_count"] == 0
    assert diagnostics["prefix_beam_width"] == 4
