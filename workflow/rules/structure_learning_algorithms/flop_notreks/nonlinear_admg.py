"""ADMG-aware extension around the genuine nonlinear FLOP skeleton.

The directed component is obtained from the normal nonlinear FLOP search.  A
second, explicitly named residual-covariance stage then searches bidirected
edges using a Gaussian covariance likelihood.  This is intentionally not
DAGMA and does not interpret residual covariance as directed causation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


def _fit_residuals(x: np.ndarray, graph: np.ndarray, *, hidden_dim: int,
                   epochs: int, seed: int) -> np.ndarray:
    import torch
    from torch import nn

    x = np.asarray(x, dtype=float)
    d = x.shape[1]
    residuals = np.empty_like(x)
    for child in range(d):
        parents = tuple(np.flatnonzero(graph[:, child]).tolist())
        y = torch.tensor(x[:, child:child + 1], dtype=torch.float32)
        if not parents:
            residuals[:, child] = x[:, child] - x[:, child].mean()
            continue
        features = torch.tensor(x[:, parents], dtype=torch.float32)
        torch.manual_seed(int(seed + 7919 * (child + 1)))
        model = nn.Sequential(nn.Linear(len(parents), hidden_dim), nn.Tanh(),
                              nn.Linear(hidden_dim, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(features) - y) ** 2)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            residuals[:, child] = (y - model(features)).numpy().ravel()
    return residuals


def covariance_graph_score(residuals: np.ndarray,
                           bidirected: Sequence[tuple[int, int]]):
    """Constrained Gaussian covariance likelihood for residual dependence.

    Omega has free diagonal entries and free covariance entries only on the
    proposed bidirected support.  This is a covariance, not precision, model.
    """
    residuals = np.asarray(residuals, dtype=float)
    # The residual covariance model is a correlation-structure diagnostic.
    # Normalize each residual channel before fitting so an over-flexible MLP
    # cannot make the covariance likelihood degenerate merely by shrinking a
    # residual variance toward zero.
    residuals = residuals - residuals.mean(axis=0, keepdims=True)
    residual_scale = residuals.std(axis=0, ddof=0)
    residuals = residuals / np.where(residual_scale > 1e-8, residual_scale, 1.0)
    n, d = residuals.shape
    sample = np.cov(residuals, rowvar=False, bias=True)
    sample = 0.999 * sample + 0.001 * np.eye(d)
    pairs = tuple(sorted(tuple(sorted(map(int, pair))) for pair in bidirected))
    diagonal = np.maximum(np.diag(sample), 1e-6)
    initial = np.concatenate((np.log(diagonal),
                              [sample[i, j] for i, j in pairs]))

    def unpack(theta):
        # Keep the diagonal parameterization numerically bounded.  Without
        # bounds L-BFGS can probe exp(theta) at overflow while it is trying a
        # non-PD covariance proposal.
        log_diagonal = np.clip(np.asarray(theta[:d]), -20.0, 20.0)
        omega = np.diag(np.exp(log_diagonal))
        for value, (i, j) in zip(theta[d:], pairs):
            omega[i, j] = omega[j, i] = value
        return omega

    def objective(theta):
        omega = unpack(theta)
        sign, logdet = np.linalg.slogdet(omega)
        if sign <= 0:
            return 1e12 + 1e8 * max(0.0, -np.linalg.eigvalsh(omega).min())
        try:
            trace = np.trace(np.linalg.solve(omega, sample))
        except np.linalg.LinAlgError:
            return 1e12
        return float(n * (logdet + trace) + len(pairs) * np.log(n))

    bounds = [(-20.0, 20.0)] * d + [(None, None)] * len(pairs)
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-10})
    return float(result.fun), unpack(result.x), bool(result.success)


def add_bidirected_edges(residuals: np.ndarray,
                         no_trek_pairs: Sequence[tuple[int, int]],
                         *, candidate_pairs=None, max_passes: int = 4):
    """Search residual-covariance edges in a declared candidate scope.

    The integrated pilot uses the current directed skeleton as a conservative
    scope: a pair can become bidirected only when it is already a directed
    parent/child adjacency.  This keeps the residual stage local and avoids
    pretending that an unsearched pair is evidence against independence.
    Passing ``candidate_pairs=None`` retains the complete all-pairs diagnostic
    mode.
    """
    forbidden = {tuple(sorted(map(int, pair))) for pair in no_trek_pairs}
    if candidate_pairs is None:
        candidates = set(combinations(range(residuals.shape[1]), 2))
        scope = "all_pairs"
    else:
        candidates = {
            tuple(sorted((int(i), int(j))))
            for i, j in candidate_pairs if int(i) != int(j)
        }
        scope = "directed_adjacencies"
    candidates -= forbidden
    current: set[tuple[int, int]] = set()
    current_score, omega, converged = covariance_graph_score(residuals, current)
    evaluations = 1
    for _ in range(max_passes):
        best = (current_score, tuple(sorted(current)), omega, converged)
        for pair in sorted(candidates):
            if pair in current:
                continue
            proposal = current | {pair}
            score, candidate_omega, ok = covariance_graph_score(residuals, proposal)
            evaluations += 1
            if score < best[0] - 1e-8 or (abs(score - best[0]) <= 1e-8 and
                                          tuple(sorted(proposal)) < best[1]):
                best = (score, tuple(sorted(proposal)), candidate_omega, ok)
        for pair in sorted(current):
            proposal = current - {pair}
            score, candidate_omega, ok = covariance_graph_score(residuals, proposal)
            evaluations += 1
            if score < best[0] - 1e-8 or (abs(score - best[0]) <= 1e-8 and
                                          tuple(sorted(proposal)) < best[1]):
                best = (score, tuple(sorted(proposal)), candidate_omega, ok)
        if best[0] >= current_score - 1e-8:
            break
        current_score, selected, omega, converged = best
        current = set(selected)
    graph = np.zeros((residuals.shape[1], residuals.shape[1]), dtype=np.uint8)
    for i, j in current:
        graph[i, j] = graph[j, i] = 1
    return graph, {"admg_score": current_score, "admg_covariance": omega,
                   "admg_score_evaluations": evaluations,
                   "admg_covariance_converged": converged,
                   "admg_bidirected_edges": int(graph.sum() // 2),
                   "admg_notreks_forbidden_edges": len(forbidden),
                   "admg_bidirected_candidate_scope": scope,
                   "admg_bidirected_candidate_edges": len(candidates)}


def _is_dag(graph: np.ndarray) -> bool:
    graph = np.asarray(graph, dtype=bool)
    d = graph.shape[0]
    indegree = graph.sum(axis=0).astype(int)
    queue = [int(i) for i in np.flatnonzero(indegree == 0)]
    seen = 0
    for node in queue:
        seen += 1
        for child in np.flatnonzero(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    return seen == d


def _violates_notreks(graph: np.ndarray,
                      no_trek_pairs: Sequence[tuple[int, int]]) -> bool:
    graph = np.asarray(graph, dtype=bool)
    reach = graph.copy()
    np.fill_diagonal(reach, True)
    for k in range(graph.shape[0]):
        reach |= reach[:, [k]] & reach[[k], :]
    ancestors = reach.T
    return any(np.any(ancestors[i] & ancestors[j])
               for i, j in no_trek_pairs)


def joint_admg_refine(x: np.ndarray, directed: np.ndarray,
                      no_trek_pairs: Sequence[tuple[int, int]], *,
                      directed_score, hidden_dim: int, epochs: int,
                      seed: int, max_passes: int = 1,
                      max_proposals: int = 32, admg_weight: float = 1.0):
    """Bounded joint directed/bidirected refinement.

    The nonlinear FLOP graph is the starting point.  Each accepted proposal
    is evaluated by the directed nonlinear score plus the residual covariance
    score.  Bidirected candidates remain restricted to directed adjacencies;
    only directed DAG/NOTREKS-feasible proposals are considered.
    """
    current = np.asarray(directed, dtype=np.uint8).copy()
    if not _is_dag(current) or _violates_notreks(current, no_trek_pairs):
        raise ValueError("joint ADMG refinement requires a feasible DAG")

    evaluations = 0

    def evaluate(graph):
        nonlocal evaluations
        residuals = _fit_residuals(
            x, graph, hidden_dim=hidden_dim, epochs=epochs, seed=seed + 31337)
        candidate_pairs = list(zip(*np.nonzero(np.asarray(graph, dtype=bool))))
        bidirected, residual_diag = add_bidirected_edges(
            residuals, no_trek_pairs, candidate_pairs=candidate_pairs)
        total = float(directed_score(graph)) + float(admg_weight) * float(
            residual_diag["admg_score"])
        evaluations += 1
        return total, bidirected, residuals, residual_diag

    current_score, current_bidirected, current_residuals, current_diag = evaluate(current)
    accepted = 0
    d = current.shape[0]
    for _ in range(max(0, int(max_passes))):
        proposals = []
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                proposal = current.copy()
                proposal[i, j] ^= 1
                if _is_dag(proposal) and not _violates_notreks(
                        proposal, no_trek_pairs):
                    proposals.append((i, j, proposal))
        proposals = proposals[:max(0, int(max_proposals))]
        best = None
        for i, j, proposal in proposals:
            score, bidirected, residuals, residual_diag = evaluate(proposal)
            key = (score, proposal.tobytes(), i, j)
            if best is None or key < best[0]:
                best = (key, proposal, bidirected, residuals, residual_diag)
        if best is None or best[0][0] >= current_score - 1e-8:
            break
        current_score = float(best[0][0])
        current, current_bidirected, current_residuals, current_diag = (
            best[1], best[2], best[3], best[4])
        accepted += 1

    return current, current_bidirected, current_residuals, {
        **current_diag,
        "admg_joint_refinement": True,
        "admg_joint_score": current_score,
        "admg_joint_score_evaluations": evaluations,
        "admg_joint_accepted_moves": accepted,
    }
