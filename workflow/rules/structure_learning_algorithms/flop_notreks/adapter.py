import numpy as np


def selected_dag_from_diagnostics(diagnostics, d):
    dag = np.zeros((d, d), dtype=int)
    for edge in diagnostics.get("selected_dag_edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("malformed selected_dag_edges diagnostic")
        parent, child = map(int, edge)
        if parent == child or min(parent, child) < 0 or max(parent, child) >= d:
            raise ValueError("invalid selected DAG edge")
        dag[parent, child] = 1
    return dag


def count_no_trek_violations(dag, pairs):
    dag = np.asarray(dag) != 0
    reach = dag.copy()
    np.fill_diagonal(reach, True)
    for k in range(len(dag)):
        reach |= reach[:, [k]] & reach[[k], :]
    return sum(bool(np.any(reach[:, i] & reach[:, j])) for i, j in pairs)
