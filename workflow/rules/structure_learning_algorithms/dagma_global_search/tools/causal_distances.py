"""Small DAG causal-distance diagnostics.

These implement the DAG distances used in the adjustment-identification
framework of Henckel, Würtzen and Weichwald (2024).  In particular,
Parent-AID is SID for DAG inputs.  The benchmark deliberately keeps these
functions DAG-only: applying them directly to a CPDAG would be a different
estimand and would require the paper's CPDAG extension.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path


def _dag(a: np.ndarray) -> nx.DiGraph:
    a = np.asarray(a, dtype=bool)
    g = nx.DiGraph()
    g.add_nodes_from(range(a.shape[0]))
    g.add_edges_from((int(i), int(j)) for i, j in zip(*np.nonzero(a)))
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("causal distances require DAG inputs")
    return g


def _desc(g: nx.DiGraph, node: int) -> set[int]:
    return set(nx.descendants(g, node))


def _ancestors(g: nx.DiGraph, node: int) -> set[int]:
    return set(nx.ancestors(g, node))


def _adjustment_valid(true: nx.DiGraph, treatment: int, outcome: int,
                      adjustment: set[int]) -> bool:
    """Back-door adjustment criterion for a DAG and singleton treatment."""
    if treatment in adjustment or outcome in adjustment:
        return False
    if adjustment & _desc(true, treatment):
        return False
    backdoor = true.copy()
    backdoor.remove_edges_from(list(true.out_edges(treatment)))
    return nx.is_d_separator(backdoor, {treatment}, {outcome}, adjustment)


def _aid(true: np.ndarray, guess: np.ndarray, strategy: str) -> int:
    truth = _dag(true)
    estimated = _dag(guess)
    distance = 0
    for treatment in estimated.nodes:
        estimated_desc = _desc(estimated, treatment)
        true_desc = _desc(truth, treatment)
        if strategy == "parent":
            adjustment = set(estimated.predecessors(treatment))
        elif strategy == "ancestor":
            adjustment = _ancestors(estimated, treatment)
        else:
            raise ValueError(f"unknown DAG AID strategy: {strategy}")
        for outcome in estimated.nodes:
            if outcome == treatment:
                continue
            guess_claims_effect = outcome in estimated_desc
            truth_has_effect = outcome in true_desc
            if guess_claims_effect:
                correct = truth_has_effect and _adjustment_valid(
                    truth, treatment, outcome, adjustment)
            else:
                correct = not truth_has_effect
            distance += int(not correct)
    return distance


def dag_parent_aid(true: np.ndarray, guess: np.ndarray) -> int:
    """Parent-AID; for DAG inputs this is exactly SID."""
    return _aid(true, guess, "parent")


def dag_sid(true: np.ndarray, guess: np.ndarray) -> int:
    """Structural intervention distance for two DAGs."""
    return dag_parent_aid(true, guess)


def dag_ancestor_aid(true: np.ndarray, guess: np.ndarray) -> int:
    """Ancestor-AID for two DAGs."""
    return _aid(true, guess, "ancestor")


def _encode_cpdag(a: np.ndarray) -> np.ndarray:
    """Convert a binary CPDAG adjacency to gadjid's 1/2 encoding."""
    a = np.asarray(a, dtype=np.int8)
    out = (a != 0).astype(np.int8)
    undirected = (out != 0) & (out.T != 0)
    out[undirected] = 2
    np.fill_diagonal(out, 0)
    return out


@lru_cache(maxsize=256)
def _dag_cpdag_cached(payload: bytes, d: int) -> np.ndarray:
    dag = np.frombuffer(payload, dtype=np.uint8).reshape((d, d))
    # pcalg provides the reference DAG -> CPDAG conversion used elsewhere in
    # this repository for CPDAG SHD.  Cache by truth graph: all methods on a
    # paired instance share the same conversion.
    with tempfile.TemporaryDirectory(prefix="notreks_dag_cpdag_") as tmp:
        tmp = Path(tmp)
        dag_path, cpdag_path = tmp / "dag.csv", tmp / "cpdag.csv"
        np.savetxt(dag_path, dag, fmt="%d", delimiter=",")
        code = (
            "args <- commandArgs(TRUE); "
            "a <- as.matrix(read.csv(args[1], header=FALSE)); "
            "library(pcalg); "
            "write.table(dag2cpdag(a)*1, args[2], sep=',', "
            "row.names=FALSE, col.names=FALSE)"
        )
        subprocess.run(["Rscript", "-e", code, str(dag_path),
                         str(cpdag_path)], check=True,
                        capture_output=True, text=True)
        return np.loadtxt(cpdag_path, delimiter=",", dtype=np.int8)


def dag_to_cpdag(dag: np.ndarray) -> np.ndarray:
    dag = np.asarray(dag, dtype=np.uint8)
    return _dag_cpdag_cached(dag.tobytes(), dag.shape[0]).copy()


def cpdag_aid(true_dag: np.ndarray, guess_cpdag: np.ndarray,
              strategy: str) -> tuple[float, int]:
    """Exact scalar CPDAG AID via the authors' gadjid implementation."""
    import gadjid

    true_cpdag = _encode_cpdag(dag_to_cpdag(true_dag))
    guess = _encode_cpdag(guess_cpdag)
    fn = {"parent": gadjid.parent_aid, "ancestor": gadjid.ancestor_aid}[strategy]
    normalized, count = fn(true_cpdag, guess, "from row to column")
    return float(normalized), int(count)
