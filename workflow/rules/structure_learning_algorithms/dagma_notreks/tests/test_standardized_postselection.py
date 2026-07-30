import numpy as np
import pytest

from workflow.rules.structure_learning_algorithms.dagma_notreks.postselection import (
    ALL_POLICIES,
    CallbackCandidateScorer,
    EdgeStrengthExtractor,
    FeasibilityChecker,
    LinearCandidateScorer,
    PostselectionConfig,
    REFERENCE_POLICY,
    apply_standardization,
    canonical_pairs,
    lambda_policy,
    select_postselection_candidate,
    standardize_training_data,
)
from workflow.rules.structure_learning_algorithms.dagma_notreks.tools.evaluate_postselection import (
    evaluate,
)


def data(seed=1, n=80, d=5):
    return np.random.default_rng(seed).normal(size=(n, d))


def weighted():
    W = np.zeros((5, 5))
    W[0, 1], W[1, 2], W[2, 0] = .7, .6, .5
    W[3, 1], W[3, 4], W[4, 2] = .4, .3, .2
    return W


def scorer(kind="L1"):
    return LinearCandidateScorer(
        data(), regularizer_type=kind, regularizer_weight=.03)


def test_training_standardization_ddof0_floor_validation_and_no_mutation():
    X = np.array([[1., 2., 3.], [3., 2., 7.], [5., 2., 11.]])
    copy = X.copy()
    Z, means, stds = standardize_training_data(X)
    np.testing.assert_array_equal(X, copy)
    assert stds[1] == 1e-12
    np.testing.assert_allclose(Z[:, [0, 2]].mean(0), 0, atol=1e-15)
    np.testing.assert_allclose(Z[:, [0, 2]].std(0, ddof=0), 1)
    test = apply_standardization([[7., 2., 15.]], means, stds)
    np.testing.assert_allclose(test[0, [0, 2]], [2.44948974, 2.44948974])


def test_lambda_registry_exact_values():
    assert lambda_policy("fixed_0.02", 20, 100)[-1] == .02
    assert lambda_policy("sqrt_c0.5", 20, 100)[-1] == pytest.approx(
        .5 * np.sqrt(np.log(20) / 100))


def test_exact_dag_and_notreks_feasibility_and_inactive_constraints():
    A = np.zeros((4, 4), int)
    A[0, 2] = A[1, 2] = 1  # collider, allowed for (0,1)
    checker = FeasibilityChecker(
        d=4, notreks_constraint_active=True, notreks_pairs=[(0, 1)])
    assert checker.check(A).feasible
    A[2, 0] = 1
    assert not checker.check(A).DAG_valid
    assert not checker.check(A).feasible
    inactive = FeasibilityChecker(
        d=4, dag_constraint_active=False,
        notreks_constraint_active=False, notreks_pairs=[(0, 1)])
    assert inactive.check(A).feasible


def test_pairs_are_canonical_and_reversed_duplicates_removed():
    assert canonical_pairs([(1, 0), (0, 1), (3, 2)], 4) == (
        (0, 1), (2, 3))
    with pytest.raises(ValueError):
        canonical_pairs([(1, 1)], 4)


@pytest.mark.parametrize("policy", ALL_POLICIES[:-1])
def test_every_production_policy_returns_exactly_feasible_graph(policy):
    result = select_postselection_candidate(
        weighted(), scorer=scorer(),
        config=PostselectionConfig(
            policy=policy, fixed_threshold=.1,
            notreks_constraint_active=True,
            max_search_seconds=.01, max_expanded_nodes=50,
            max_ambiguous_edges=6),
        notreks_pairs=[(0, 3)])
    assert result.feasible and result.DAG_valid
    assert result.notreks_violation_count == 0
    assert result.eligible_for_recommendation


def test_empty_graph_is_feasible_fallback():
    result = select_postselection_candidate(
        np.zeros((5, 5)), scorer=scorer(),
        config=PostselectionConfig(
            policy="PS5_fixed_threshold_joint_feasible",
            fixed_threshold=1.))
    assert result.feasible and result.predicted_edges == 0


def test_ps1_and_ps5_are_deterministic():
    for policy in (
            "PS1_joint_feasible_greedy_score",
            "PS5_fixed_threshold_joint_feasible"):
        first = select_postselection_candidate(
            weighted(), scorer=scorer(),
            config=PostselectionConfig(policy=policy))
        second = select_postselection_candidate(
            weighted(), scorer=scorer(),
            config=PostselectionConfig(policy=policy))
        np.testing.assert_array_equal(first.adjacency, second.adjacency)


def test_ps2_never_leaves_feasible_region_and_ps3_cutoff_has_incumbent():
    ps2 = select_postselection_candidate(
        weighted(), scorer=scorer(),
        config=PostselectionConfig(
            policy="PS2_joint_feasible_local_search",
            max_expanded_nodes=20, max_search_seconds=.001))
    ps3 = select_postselection_candidate(
        weighted(), scorer=scorer(),
        config=PostselectionConfig(
            policy="PS3_joint_feasible_budgeted_search",
            max_expanded_nodes=1, max_search_seconds=.00001))
    assert ps2.feasible and ps3.feasible
    assert ps3.search.cutoff_reason in {
        "max_search_seconds", "max_expanded_nodes", "queue_exhausted"}


def test_ps4_repairs_cycles_and_forbidden_treks():
    result = select_postselection_candidate(
        weighted(), scorer=scorer(),
        config=PostselectionConfig(
            policy="PS4_joint_violation_repair",
            notreks_constraint_active=True),
        notreks_pairs=[(0, 3)])
    assert result.feasible and result.notreks_violation_count == 0


def test_l1_and_l2_refits_are_distinct_and_cached():
    A = np.zeros((5, 5), int)
    A[0, 1] = A[2, 1] = 1
    l1, l2 = scorer("L1"), scorer("L2")
    score1, score2 = l1.score(A), l2.score(A)
    assert np.isfinite(score1) and np.isfinite(score2)
    assert score1 != score2
    l1.score(A.copy())
    assert l1.refit_calls == 1


def test_nonlinear_callback_respects_mask_and_never_uses_bic():
    seen = []
    callback = CallbackCandidateScorer(
        lambda mask: seen.append(mask.copy()) or float(mask.sum()),
        regularizer_type="L2", regularizer_weight=.1,
        loss_name="mock_nonlinear_loss")
    result = select_postselection_candidate(
        weighted(), scorer=callback,
        config=PostselectionConfig(
            policy="PS1_joint_feasible_greedy_score"),
        model_class="nonlinear_dagma_proxy")
    assert seen and result.score_name == "configured_model_refit_score"
    with pytest.raises(ValueError, match="linear-only"):
        select_postselection_candidate(
            weighted(), scorer=callback,
            config=PostselectionConfig(policy=REFERENCE_POLICY),
            model_class="nonlinear_dagma_proxy")


def test_reference_is_never_recommendation_eligible():
    result = select_postselection_candidate(
        weighted(), scorer=scorer(),
        config=PostselectionConfig(policy=REFERENCE_POLICY))
    assert result.reference_only
    assert not result.eligible_for_recommendation
    assert result.hard_feasibility_not_guaranteed


def test_edge_strength_uses_linear_weights_or_provided_nonlinear_proxy():
    linear = EdgeStrengthExtractor().extract(
        weighted(), model_class="linear_dagma")
    nonlinear = EdgeStrengthExtractor().extract(
        weighted(), model_class="nonlinear_dagma_proxy")
    assert linear.edge_strength_shape == (5, 5)
    assert nonlinear.edge_strength_definition.endswith("_proxy_abs")


def test_recommendations_are_separate_and_reference_is_excluded():
    import pandas as pd
    rows = pd.DataFrame([
        {"model_class": "linear", "constraint_regime": "DAG_only",
         "postselection_policy": "PS1_joint_feasible_greedy_score",
         "eligible_for_recommendation": True, "feasible": True,
         "SHD_pattern": 5, "skeleton_F1": .9,
         "candidate_score": 1., "postselection_time_seconds": .1},
        {"model_class": "linear",
         "constraint_regime": "DAG_NOTREKS_one_correct",
         "postselection_policy": "PS5_fixed_threshold_joint_feasible",
         "eligible_for_recommendation": True, "feasible": True,
         "SHD_pattern": 48, "skeleton_F1": .575,
         "candidate_score": 2., "postselection_time_seconds": .01},
        {"model_class": "linear", "constraint_regime": "DAG_only",
         "postselection_policy": REFERENCE_POLICY,
         "eligible_for_recommendation": False, "feasible": True,
         "SHD_pattern": 1, "skeleton_F1": 1.,
         "candidate_score": 0., "postselection_time_seconds": .001},
    ])
    summary, frontier, warnings = evaluate(rows)
    assert set(summary.constraint_regime) == {
        "DAG_only", "DAG_NOTREKS_one_correct"}
    assert REFERENCE_POLICY not in set(frontier.postselection_policy)
    warning = warnings[
        warnings.constraint_regime == "DAG_NOTREKS_one_correct"].iloc[0]
    assert not bool(warning.confident_recommendation_allowed)
