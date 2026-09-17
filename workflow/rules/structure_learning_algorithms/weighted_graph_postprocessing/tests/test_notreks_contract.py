import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import is_dag
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.core import (
    FeasibilityChecker,
    ancestor_matrix,
    notreks_violation_count,
)


def test_ancestor_table_matches_graph_theoretic_reachability_and_diagonal():
    graph = np.zeros((4, 4), dtype=int)
    graph[0, 1] = graph[1, 2] = graph[2, 3] = 1
    reach = ancestor_matrix(graph)
    assert is_dag(graph)
    assert reach[0, 3]
    assert not reach[3, 0]
    assert np.all(np.diag(reach))


def test_every_supplied_pair_is_checked_exactly():
    graph = np.zeros((4, 4), dtype=int)
    graph[0, 2] = graph[2, 3] = 1
    assert notreks_violation_count(graph, [(0, 1), (1, 3)]) == 0
    assert notreks_violation_count(graph, [(0, 3)]) == 1
    checker = FeasibilityChecker(
        d=4, dag_constraint_active=True, notreks_constraint_active=True,
        notreks_pairs=[(0, 3)])
    assert not checker.check(graph).feasible


def test_empty_prior_is_a_noop_for_feasibility():
    graph = np.zeros((3, 3), dtype=int)
    graph[0, 1] = graph[1, 2] = 1
    checker = FeasibilityChecker(
        d=3, dag_constraint_active=True, notreks_constraint_active=True,
        notreks_pairs=[])
    assert checker.check(graph).feasible
    assert checker.check(graph).notreks_violation_count == 0
