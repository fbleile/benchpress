import numpy as np

from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.thermalgagge import (
    OBSERVED_NAMES,
    ROOT_NAMES,
    ThermalConfig,
    _parameters,
    clamped_sensation,
    equilibrate,
    generate,
)


def test_thermal_generation_is_reproducible_and_in_domain():
    first = generate(1729, 4)
    second = generate(1729, 4)
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[0].shape == (4, len(OBSERVED_NAMES))
    assert np.all((first[0][:, 0] >= 18) & (first[0][:, 0] <= 32))
    assert np.all((first[0][:, 2] >= .05) & (first[0][:, 2] <= .8))
    assert np.all((first[0][:, 3] >= 20) & (first[0][:, 3] <= 80))


def test_equilibrium_residuals_and_time_constant_trace():
    _, _, _, metadata = generate(1730, 2, diagnostics=True)
    assert metadata["max_equilibrium_residual"] < 1e-4
    for trace in metadata["integration_traces"]:
        assert len(trace) >= 20
        assert all(item["E"] == trace[0]["E"] for item in trace)
        assert all(item["P"] == trace[0]["P"] for item in trace)
        assert all(item["U_T"] == trace[0]["U_T"] for item in trace)


def test_oracle_has_only_root_to_outcome_and_cross_block_pairs():
    _, truth, pairs, metadata = generate(1731, 2)
    skin, sensation = len(ROOT_NAMES), len(ROOT_NAMES) + 1
    assert not np.any(np.diag(truth))
    assert np.all(truth[:len(ROOT_NAMES), skin] >= 0)
    assert np.all(truth[:len(ROOT_NAMES), sensation] >= 0)
    assert truth[:7, skin].sum() >= 5
    assert truth[:7, sensation].sum() >= 1
    assert truth[skin, sensation] == 1
    assert len(pairs) == 5 * 4
    assert metadata["thermal_package"] == "pythermalcomfort"


def test_configuration_has_strict_equilibrium_defaults():
    config = ThermalConfig()
    assert config.tol_core == 1e-4
    assert config.tol_skin == 1e-4
    assert config.consecutive_steps == 20
    assert config.max_minutes == 2000


def test_equilibrium_agrees_across_initial_states_and_reduced_readout():
    config = ThermalConfig()
    params = _parameters(np.zeros(len(ROOT_NAMES)))
    states = [
        equilibrate(params, 0.0, config, initial=initial)
        for initial in ((36.8, 33.7), (40.0, 28.0), (30.0, 38.0))]
    for state in states[1:]:
        assert abs(state["core"] - states[0]["core"]) < 1e-3
        assert abs(state["skin"] - states[0]["skin"]) < 1e-3
    reduced = clamped_sensation(params, states[0]["skin"], 0.0, config)
    assert abs(reduced - states[0]["t_sens"]) < 1e-4
