import numpy as np
from typing import Any, Optional, Sequence

def flop(
    data: np.ndarray,
    lambda_bic: float,
    *,
    restarts: Optional[int] = None,
    timeout: Optional[float] = None,
) -> np.ndarray:
    """
    Run the FLOP causal discovery algorithm.

    Parameters
    ----------
    data: A data matrix with rows corresponding to observations and columns to variables/nodes.
    lambda_bic: The penalty parameter of the BIC, a typical value for structure learning is 2.0.
    restarts: Optional parameter specifying the number of ILS restarts. Either restarts or timeout (below) need to be specified.
    timeout: Optional parameter specifying a timeout after which the search returns. At least one local search is run up to a local optimum. Either restarts or timeout need to be specified.

    Returns
    -------
    A matrix encoding a CPDAG or DAG. The entry in row i and column j is 1 in case of a directed edge from i to j and 2 in case of an undirected edge between those nodes (the entry in row j and column i will also be a 2, that is each undirected edge induces two 2's in the matrix).
    """
    ...

def flop_notreks(
    data: np.ndarray,
    lambda_bic: float,
    no_trek_pairs: Sequence[tuple[int, int]] | np.ndarray,
    *,
    restarts: Optional[int] = None,
    timeout: Optional[float] = None,
    seed: Optional[int] = None,
    signature_top_k: int = 5,
    signature_exploration_k: int = 0,
    max_signature_rounds: int = 20,
    initial_signature_mean_size: float = 3.0,
    initial_signature_max_size: int = 6,
    forbidden_edges: Sequence[tuple[int, int]] | np.ndarray | None = None,
    search_version: str = "global_greedy_rust",
    random_initial_order: bool = False,
    local_greedy_passes: int = 8,
    order_guided_lex_fraction: float = 0.5,
    order_guided_repair_candidates: int = 2,
    order_guided_coverage_starts: int = 0,
    trekcut_oracle_budget: int = 32,
    trekcut_refinement_passes: int = 4,
    prefix_beam_width: int = 1,
    trek_graph: Sequence[tuple[int, int]] | np.ndarray | None = None,
    return_dag: bool = False,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Run FLOP with hard structural no-trek constraints."""
    ...

def flop_source_signature(
    data: np.ndarray,
    initial_graph: np.ndarray,
    lambda_bic: float,
    no_trek_pairs: Sequence[tuple[int, int]] | np.ndarray,
    *,
    seed: Optional[int] = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    ...

def prune_parents_bic(
    data: np.ndarray,
    candidate_dag: np.ndarray,
    lambda_bic: float = 2.0,
    *,
    return_coefficients: bool = False,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Deterministically prune only existing candidate parents using FLOP BIC."""
    ...

def soft_global_greedy(
    data: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]] | np.ndarray,
    *,
    restarts: int = 8,
    max_sweeps: int = 12,
    lambda_bic: float = 2.0,
    soft_notreks_weight: float = 100.0,
    lazy_top_k: int = 12,
    seed: int = 1729,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the Rust lazy soft continuous-NOTREKS global-greedy search."""
    ...

def soft_global_greedy_redesigned(
    data: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]] | np.ndarray,
    *,
    restarts: int = 8,
    max_sweeps: int = 8,
    lambda_bic: float = 2.0,
    lazy_top_k: int = 16,
    seed: int = 1729,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the experimental normalized-continuation soft Rust search."""
    ...

def global_greedy_penalty(
    data: np.ndarray,
    lambda_bic: float,
    no_trek_pairs: Sequence[tuple[int, int]] | np.ndarray,
    *,
    restarts: int = 1,
    max_sweeps: int = 8,
    penalty_weight: float = 1.0,
    seed: int = 1729,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Hard-DAG global greedy with a continuous NOTREKS objective penalty."""
    ...

def global_greedy_binary_penalty(
    data: np.ndarray,
    lambda_bic: float,
    no_trek_pairs: Sequence[tuple[int, int]] | np.ndarray,
    *,
    restarts: int = 1,
    max_sweeps: int = 8,
    seed: int = 1729,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Hard-feasible search with an extra graph-only NOTREKS calculation."""
    ...
