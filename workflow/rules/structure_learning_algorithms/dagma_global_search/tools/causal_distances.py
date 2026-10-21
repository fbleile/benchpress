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
import os
import shutil
import subprocess
import sys
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


def _dag_to_cpdag_python(dag: np.ndarray) -> np.ndarray:
    """Convert a DAG to its CPDAG using v-structures and Meek R1--R3.

    This is the pure-Python counterpart of ``pcalg::dag2cpdag``.  The
    benchmark's DAG inputs have no background knowledge, so the standard
    v-structure initialization followed by the three ordinary Meek rules is
    sufficient; the fourth rule is the background-knowledge extension.
    Entries are returned in the same binary adjacency convention used by the
    R path: a directed edge has one nonzero entry and an undirected edge has
    both entries nonzero.
    """
    dag = np.asarray(dag, dtype=np.uint8)
    d = dag.shape[0]
    if dag.shape != (d, d):
        raise ValueError("DAG adjacency must be square")
    if np.any(np.diag(dag)) or not nx.is_directed_acyclic_graph(
            nx.from_numpy_array(dag, create_using=nx.DiGraph)):
        raise ValueError("CPDAG conversion requires a DAG")

    skeleton = (dag != 0) | (dag.T != 0)
    directed = np.zeros((d, d), dtype=bool)
    undirected = skeleton.copy()

    def adjacent(i: int, j: int) -> bool:
        return bool(skeleton[i, j])

    def orient(i: int, j: int) -> bool:
        if directed[i, j]:
            return False
        if directed[j, i]:
            raise ValueError("inconsistent CPDAG orientation")
        if not undirected[i, j]:
            raise ValueError("attempted to orient a non-adjacent pair")
        undirected[i, j] = undirected[j, i] = False
        directed[i, j] = True
        return True

    # Preserve exactly the unshielded colliders of the input DAG.
    for child in range(d):
        parents = np.flatnonzero(dag[:, child])
        for offset, left in enumerate(parents):
            for right in parents[offset + 1:]:
                if not adjacent(int(left), int(right)):
                    orient(int(left), child)
                    orient(int(right), child)

    changed = True
    while changed:
        changed = False

        # R1: a -> b - c and a not adjacent c implies b -> c.
        for a in range(d):
            for b in np.flatnonzero(directed[a]):
                for c in np.flatnonzero(undirected[b]):
                    c = int(c)
                    if not adjacent(a, c):
                        changed |= orient(int(b), c)

        # R2: a - b and a -> c -> b implies a -> b.
        for a in range(d):
            for b in np.flatnonzero(undirected[a]):
                b = int(b)
                if any(directed[a, c] and directed[c, b]
                       for c in range(d) if c not in (a, b)):
                    changed |= orient(a, b)

        # R3: a - b, a - c, a - d, c -> b, d -> b, and c not adjacent d
        # implies a -> b.
        for a in range(d):
            for b in np.flatnonzero(undirected[a]):
                b = int(b)
                candidates = [
                    int(c) for c in np.flatnonzero(undirected[a])
                    if c != b and directed[int(c), b]
                ]
                if any(not adjacent(c, e)
                       for offset, c in enumerate(candidates)
                       for e in candidates[offset + 1:]):
                    changed |= orient(a, b)

    result = directed.astype(np.uint8)
    result[undirected] = 1
    return result


@lru_cache(maxsize=256)
def _dag_cpdag_cached(payload: bytes, d: int) -> np.ndarray:
    dag = np.frombuffer(payload, dtype=np.uint8).reshape((d, d))
    backend = os.environ.get("NOTREKS_CPDAG_BACKEND", "r").lower()
    if backend == "python":
        return _dag_to_cpdag_python(dag)
    if backend not in {"r", "auto"}:
        raise ValueError(
            "NOTREKS_CPDAG_BACKEND must be 'r', 'auto', or 'python'"
        )
    # JobFarm workers invoke the absolute venv Python path.  Resolve the
    # sibling Rscript first so this remains reliable even if the worker does
    # not preserve the interactive shell's PATH.
    rscript = Path(sys.executable).with_name("Rscript")
    if not rscript.is_file():
        found = shutil.which("Rscript")
        rscript = Path(found) if found else None
    if rscript is None:
        return _dag_to_cpdag_python(dag)
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
        run_kwargs = {
            "check": True,
            "capture_output": True,
            "text": True,
        }
        timeout = os.environ.get("NOTREKS_CPDAG_R_TIMEOUT")
        if timeout:
            run_kwargs["timeout"] = float(timeout)
        try:
            subprocess.run([str(rscript), "-e", code, str(dag_path),
                            str(cpdag_path)], **run_kwargs)
        except subprocess.TimeoutExpired:
            # Exact pcalg conversion can be pathological for some larger
            # graphs.  Preserve progress and the benchmark's binary output
            # convention by falling back to the in-process converter.
            print(
                f"CPDAG R conversion timed out after {timeout}s; "
                "using Python fallback",
                flush=True,
            )
            return _dag_to_cpdag_python(dag)
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
