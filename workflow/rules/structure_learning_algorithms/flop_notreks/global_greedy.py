"""FLOP-style order reinsertion with a hard-NOTREKS global greedy inner loop.

This diagnostic/paper comparator reuses FLOP's outer idea and scores complete
graphs under exact discrete DAG and NOTREKS feasibility. Gaussian BIC is
decomposable, so the implementation caches exact node-local scores while
retaining the same graph-level acceptance objective and deterministic ties.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from typing import Sequence

import numpy as np
import scipy.linalg as sla

from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.feasibility import (
    CompositeFeasibility,
    DagConstraint,
    NoTreksConstraint,
)
from workflow.rules.structure_learning_algorithms.weighted_graph_postprocessing.scores import (
    GaussianBICGraphScore,
)


@dataclass(frozen=True)
class GlobalGreedyConfig:
    restarts: int = 8
    max_sweeps: int = 4
    lambda_bic: float = 2.0
    seed: int = 1729
    time_limit_seconds: float | None = None
    # Keep the reference implementation as the production default.  The Rust
    # kernel is available explicitly for one-shot fixed-order profiling; using
    # it for every Python outer-loop proposal would repeatedly allocate Rust
    # score state and is not yet a safe d=50 path.
    inner_backend: str = "python"
    score_backend: str = "gaussian"
    nonlinear_hidden_dim: int = 16
    nonlinear_epochs: int = 60
    nonlinear_learning_rate: float = 0.01
    nonlinear_score: str = "profiled_log_mse_cv"
    nonlinear_student_t_nu: float = 4.0
    nonlinear_validation_fraction: float = 0.2
    # A neural parent-set score must charge its actual parameter count.  The
    # old 0.05 default materially under-penalized parent additions and caused
    # dense nonlinear FLOP outputs on CausalMan.
    nonlinear_complexity_scale: float = 1.0
    # Disabled by default: the current in-sample profiled MLP score can make
    # joint proposals look better while harming recovery.  Keep the feature
    # available for controlled diagnostics only.
    max_block_proposals_per_child: int = 0


@dataclass
class GlobalGreedyResult:
    adjacency: np.ndarray
    score: float
    order: tuple[int, ...]
    restart: int
    runtime_seconds: float
    graph_score_evaluations: int
    local_score_cache_misses: int
    full_feasibility_evaluations: int
    feasibility_rejections: int
    accepted_edge_moves: int
    accepted_reinsertions: int
    termination_reason: str
    notreks_violation_count: int
    inner_backend: str


def _project_to_order(graph: np.ndarray, order: Sequence[int]) -> np.ndarray:
    position = np.empty(len(order), dtype=int)
    position[np.asarray(order, dtype=int)] = np.arange(len(order))
    result = np.asarray(graph, dtype=np.uint8).copy()
    result[position[:, None] >= position[None, :]] = 0
    np.fill_diagonal(result, 0)
    return result


def _reinsert(order: Sequence[int], node: int, position: int) -> list[int]:
    candidate = list(order)
    candidate.remove(node)
    candidate.insert(position, node)
    return candidate


class _LocalGaussianBIC:
    """Exact cached local scores matching the graph-level BIC convention."""

    def __init__(self, data: np.ndarray, lambda_bic: float):
        self.data = data - data.mean(axis=0, keepdims=True)
        self.n = len(data)
        self.penalty = float(lambda_bic) * np.log(self.n)
        self.cache: dict[tuple[int, tuple[int, ...]], float] = {}
        self.cache_misses = 0

    def score(self, child: int, parents: Sequence[int]) -> float:
        key = (int(child), tuple(sorted(map(int, parents))))
        if key not in self.cache:
            parent_array = np.asarray(key[1], dtype=int)
            if len(parent_array):
                beta, *_ = sla.lstsq(
                    self.data[:, parent_array], self.data[:, child],
                    lapack_driver="gelsy")
                residual = (self.data[:, child]
                            - self.data[:, parent_array] @ beta)
            else:
                residual = self.data[:, child]
            variance = max(
                float(residual @ residual) / self.n, 1e-8)
            self.cache[key] = (
                self.n * np.log(variance) + len(parent_array) * self.penalty)
            self.cache_misses += 1
        return self.cache[key]

    def graph_score(self, graph: np.ndarray) -> float:
        return float(sum(
            self.score(child, np.flatnonzero(graph[:, child]))
            for child in range(graph.shape[0])))


class _LocalNonlinearScore:
    """Standalone nodewise MLP score for the FLOP order search.

    This is not DAGMA: there is no global weight matrix, acyclicity penalty,
    continuation schedule, or DAGMA optimizer in this scorer.
    """

    def __init__(self, data: np.ndarray, config: GlobalGreedyConfig):
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("nonlinear FLOP requires PyTorch") from exc
        self.torch, self.nn = torch, nn
        self.data = np.asarray(data, dtype=np.float64)
        self.data = self.data - self.data.mean(axis=0, keepdims=True)
        scale = self.data.std(axis=0, keepdims=True)
        self.data /= np.where(scale > 1e-12, scale, 1.0)
        self.n = len(self.data)
        self.penalty = float(config.lambda_bic) * np.log(self.n)
        self.hidden_dim = int(config.nonlinear_hidden_dim)
        self.epochs = int(config.nonlinear_epochs)
        self.learning_rate = float(config.nonlinear_learning_rate)
        self.score_kind = str(config.nonlinear_score)
        self.student_t_nu = float(config.nonlinear_student_t_nu)
        self.validation_fraction = float(config.nonlinear_validation_fraction)
        self.complexity_scale = float(config.nonlinear_complexity_scale)
        self.seed = int(config.seed)
        rng = np.random.default_rng(self.seed + 424242)
        order = rng.permutation(self.n)
        split = int(round(self.n * (1.0 - self.validation_fraction)))
        self.train_idx = np.sort(order[:split])
        self.valid_idx = np.sort(order[split:])
        self.cache: dict[tuple[int, tuple[int, ...]], float] = {}
        self.state_cache = {}
        self.cache_misses = 0

    def _fit(self, child: int, parents: tuple[int, ...]) -> float:
        torch, nn = self.torch, self.nn
        use_cv = self.score_kind.endswith("_cv")
        base_kind = self.score_kind[:-3] if use_cv else self.score_kind
        train = self.train_idx if use_cv else np.arange(self.n)
        valid = self.valid_idx if use_cv else train
        y_train = torch.tensor(self.data[train, child:child + 1], dtype=torch.float32)
        y_valid = torch.tensor(self.data[valid, child:child + 1], dtype=torch.float32)
        if not parents:
            residual = y_valid - y_train.mean()
        else:
            x_train = torch.tensor(self.data[np.ix_(train, parents)], dtype=torch.float32)
            x_valid = torch.tensor(self.data[np.ix_(valid, parents)], dtype=torch.float32)
            torch.manual_seed(self.seed + 1009 * (child + 1) + sum(9176 * (p + 1) for p in parents))
            model = nn.Sequential(nn.Linear(len(parents), self.hidden_dim),
                                  nn.Tanh(), nn.Linear(self.hidden_dim, 1)).float()
            # Warm-start from the largest previously fitted parent subset.
            subsets = [key for key in self.state_cache
                       if key[0] == child and set(key[1]).issubset(parents)]
            if subsets:
                source = max(subsets, key=lambda key: len(key[1]))
                old = self.state_cache[source]
                with torch.no_grad():
                    model[0].weight.zero_()
                    for old_pos, parent in enumerate(source[1]):
                        model[0].weight[:, parents.index(parent)] = old[
                            "0.weight"][:, old_pos]
                    model[0].bias.copy_(old["0.bias"])
                    model[2].weight.copy_(old["2.weight"])
                    model[2].bias.copy_(old["2.bias"])
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for _ in range(self.epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((model(x_train) - y_train) ** 2)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                residual = y_valid - model(x_valid)
                self.state_cache[(child, parents)] = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()}
        variance = torch.clamp(torch.mean(residual ** 2), min=1e-8)
        if base_kind == "profiled_log_mse":
            # The likelihood is evaluated on `valid` for CV and on `train`
            # otherwise.  Using self.n here inflated the CV fit term and made
            # the effective complexity penalty inconsistent with the data
            # actually scored.
            fit = len(valid) * float(torch.log(variance))
        elif base_kind == "student_t":
            nu = self.student_t_nu
            fit = len(valid) * float(torch.log(variance)) + len(valid) * float(
                torch.mean((nu + 1.0) * torch.log1p(
                    residual ** 2 / (nu * variance))))
        else:
            raise ValueError("unknown nonlinear FLOP score: " + self.score_kind)
        parameter_count = self.hidden_dim * (len(parents) + 1) + self.hidden_dim + 1
        return fit + self.complexity_scale * parameter_count * self.penalty

    def score(self, child: int, parents: Sequence[int]) -> float:
        key = (int(child), tuple(sorted(map(int, parents))))
        if key not in self.cache:
            self.cache[key] = self._fit(*key)
            self.cache_misses += 1
        return self.cache[key]

    def graph_score(self, graph: np.ndarray) -> float:
        return float(sum(self.score(child, np.flatnonzero(graph[:, child]))
                         for child in range(graph.shape[0])))


def _transitive_reach(graph: np.ndarray) -> np.ndarray:
    reach = np.asarray(graph, dtype=bool).copy()
    np.fill_diagonal(reach, True)
    for node in range(len(reach)):
        reach |= reach[:, [node]] & reach[[node], :]
    return reach


def _addition_violates_notreks(
    reach: np.ndarray,
    source: int,
    target: int,
    pairs: np.ndarray,
) -> bool:
    """Check an order-valid edge addition from the current feasible graph."""
    if not len(pairs):
        return False
    new_ancestors = reach[:, source]
    for left, right in pairs:
        left_affected = bool(reach[target, left])
        right_affected = bool(reach[target, right])
        if left_affected and right_affected:
            return True
        if left_affected and np.any(new_ancestors & reach[:, right]):
            return True
        if right_affected and np.any(new_ancestors & reach[:, left]):
            return True
    return False


def _invalid_notreks_additions(
    reach: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """Return exact invalid ``source -> target`` additions for a feasible DAG."""
    d = reach.shape[0]
    invalid = np.zeros((d, d), dtype=bool)
    if not len(pairs):
        return invalid
    # common[i, j] is true exactly when i and j currently share an ancestor.
    common = (reach.T.astype(np.uint16) @ reach.astype(np.uint16)) > 0
    for left, right in pairs:
        left_affected = reach[:, left]
        right_affected = reach[:, right]
        invalid |= np.outer(common[:, right], left_affected)
        invalid |= np.outer(common[:, left], right_affected)
        invalid[:, left_affected & right_affected] = True
    return invalid


def fit_global_greedy_notreks(
    X: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]],
    config: GlobalGreedyConfig = GlobalGreedyConfig(),
) -> GlobalGreedyResult:
    """Fit a hard-feasible graph without consulting graph truth."""
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array")
    if config.restarts < 1 or config.max_sweeps < 1:
        raise ValueError("restarts and max_sweeps must be positive")
    if config.time_limit_seconds is not None and config.time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    d = data.shape[1]
    constraints = CompositeFeasibility((
        DagConstraint(), NoTreksConstraint(no_trek_pairs)))
    scorer = GaussianBICGraphScore(data, lambda_bic=config.lambda_bic)
    if config.score_backend == "gaussian":
        local_scorer = _LocalGaussianBIC(data, config.lambda_bic)
    elif config.score_backend == "nonlinear":
        local_scorer = _LocalNonlinearScore(data, config)
    else:
        raise ValueError("unknown FLOP score backend: " + config.score_backend)
    rust_inner = None
    if config.score_backend == "gaussian" and config.inner_backend in {"auto", "rust"}:
        try:
            import flopsearch  # type: ignore
            rust_inner = getattr(flopsearch, "global_greedy_inner", None)
        except ImportError:
            rust_inner = None
        if config.inner_backend == "rust" and rust_inner is None:
            raise RuntimeError("requested Rust global-greedy inner kernel is unavailable")
    canonical_pairs = np.asarray(sorted({
        tuple(sorted(map(int, pair))) for pair in no_trek_pairs
    }), dtype=int).reshape(-1, 2)
    score_evaluations = feasibility_rejections = accepted_moves = 0
    full_feasibility_evaluations = 0

    def improve(graph: np.ndarray, order: Sequence[int]):
        nonlocal score_evaluations, feasibility_rejections, accepted_moves
        if rust_inner is not None:
            result = np.asarray(rust_inner(
                data,
                np.asarray(graph, dtype=np.uint8),
                list(map(int, order)),
                [tuple(map(int, pair)) for pair in canonical_pairs.tolist()],
                lambda_bic=config.lambda_bic,
                return_dag=True,
            ), dtype=np.uint8)
            if not constraints.is_feasible(result):
                raise RuntimeError(
                    "Rust global-greedy inner returned an infeasible graph")
            return result, local_scorer.graph_score(result)
        current = _project_to_order(graph, order)
        if not constraints.is_feasible(current):
            raise RuntimeError(
                "global-greedy inner received an infeasible initial graph")
        current_score = local_scorer.graph_score(current)
        position = np.empty(d, dtype=int)
        position[np.asarray(order, dtype=int)] = np.arange(d)
        edges = [
            (i, j) for i in range(d) for j in range(d)
            if position[i] < position[j]]
        while True:
            reach = _transitive_reach(current)
            invalid_additions = _invalid_notreks_additions(
                reach, canonical_pairs)
            current_bytes = current.tobytes()
            best = (current_score, current_bytes, None)
            # The same parent set is queried for every candidate edge into a
            # child.  Materialising these once avoids O(d^2) repeated scans.
            parents_by_child = [
                np.flatnonzero(current[:, child]) for child in range(d)
            ]
            for i, j in edges:
                adding = not bool(current[i, j])
                if adding and invalid_additions[i, j]:
                    feasibility_rejections += 1
                    continue
                old_parents = parents_by_child[j]
                if adding:
                    new_parents = np.append(old_parents, i)
                else:
                    new_parents = old_parents[old_parents != i]
                value = (current_score
                         - local_scorer.score(j, old_parents)
                         + local_scorer.score(j, new_parents))
                score_evaluations += 1
                # Most proposals are worse than the incumbent.  Defer the
                # full matrix copy and bytewise tie key until a proposal can
                # actually compete with the current best.
                if value < best[0] - 1e-12:
                    proposal = current.copy()
                    proposal[i, j] ^= 1
                    best = (value, proposal.tobytes(), proposal)
                elif abs(value - best[0]) <= 1e-12:
                    proposal = current.copy()
                    proposal[i, j] ^= 1
                    candidate = (value, proposal.tobytes(), proposal)
                    if candidate[1] < best[1]:
                        best = candidate
            # Nonlinear effects can require two parents jointly.  Try a
            # bounded, deterministic set of two-edge additions after the
            # ordinary FLOP grow/shrink proposals.  The final graph, rather
            # than either intermediate edge, is checked for NOTREKS validity.
            if config.score_backend == "nonlinear" and config.max_block_proposals_per_child > 0:
                for child in range(d):
                    absent = [i for i, j in edges
                              if j == child and not bool(current[i, child])]
                    for left, right in list(combinations(absent, 2))[
                            :config.max_block_proposals_per_child]:
                        proposal = current.copy()
                        proposal[left, child] = 1
                        proposal[right, child] = 1
                        if not constraints.is_feasible(proposal):
                            feasibility_rejections += 1
                            continue
                        old_parents = parents_by_child[child]
                        new_parents = np.concatenate((old_parents, [left, right]))
                        value = (current_score
                                 - local_scorer.score(child, old_parents)
                                 + local_scorer.score(child, new_parents))
                        score_evaluations += 1
                        candidate = (value, proposal.tobytes(), proposal)
                        if value < best[0] - 1e-12 or (
                                abs(value - best[0]) <= 1e-12 and
                                candidate[1] < best[1]):
                            best = candidate
            if best[2] is None or best[0] >= current_score - 1e-10:
                return current, current_score
            current, current_score = best[2], best[0]
            current_score = local_scorer.graph_score(current)
            accepted_moves += 1

    rng = np.random.default_rng(config.seed)
    started = perf_counter()
    deadline = (started + config.time_limit_seconds
                if config.time_limit_seconds is not None else None)
    winner = None
    accepted_reinsertions = 0
    for restart in range(config.restarts):
        if restart > 0 and deadline is not None and perf_counter() >= deadline:
            break
        order = list(map(int, rng.permutation(d)))
        graph, current = improve(np.zeros((d, d), dtype=np.uint8), order)
        termination = "maximum_sweeps"
        for _ in range(config.max_sweeps):
            before = current
            for node in order.copy():
                if deadline is not None and perf_counter() >= deadline:
                    termination = "time_limit"
                    break
                old_position = order.index(node)
                best = (current, tuple(order), graph, order)
                for new_position in range(d):
                    if new_position == old_position:
                        continue
                    candidate_order = _reinsert(order, node, new_position)
                    candidate_graph, candidate_score = improve(
                        graph, candidate_order)
                    key = (candidate_score, tuple(candidate_order))
                    if key < best[:2]:
                        best = (
                            candidate_score, tuple(candidate_order),
                            candidate_graph, candidate_order)
                if best[0] < current - 1e-10:
                    current, graph, order = best[0], best[2], best[3]
                    accepted_reinsertions += 1
            if termination == "time_limit":
                break
            if before - current <= 1e-10:
                termination = "joint_plateau"
                break
        candidate = (current, graph.tobytes(), restart, graph, tuple(order), termination)
        if winner is None or candidate[:3] < winner[:3]:
            winner = candidate
    assert winner is not None
    full_feasibility_evaluations += 1
    summary = constraints.violation_summary(winner[3])
    if not summary["feasible"]:
        raise RuntimeError("global greedy returned an infeasible graph")
    exact_score = local_scorer.graph_score(winner[3])
    if not np.isclose(exact_score, winner[0], rtol=1e-10, atol=1e-8):
        raise RuntimeError("incremental and graph-level Gaussian BIC disagree")
    return GlobalGreedyResult(
        adjacency=winner[3], score=float(exact_score), order=winner[4],
        restart=int(winner[2]), runtime_seconds=perf_counter() - started,
        graph_score_evaluations=score_evaluations,
        local_score_cache_misses=local_scorer.cache_misses,
        full_feasibility_evaluations=full_feasibility_evaluations,
        feasibility_rejections=feasibility_rejections,
        accepted_edge_moves=accepted_moves,
        accepted_reinsertions=accepted_reinsertions,
        termination_reason=winner[5],
        notreks_violation_count=int(summary["notreks_violation_count"]),
        inner_backend=("rust" if rust_inner is not None else
                       ("python-nonlinear" if config.score_backend == "nonlinear"
                        else "python")),
    )


def fit_global_greedy_nonlinear_notreks(
    X: np.ndarray,
    no_trek_pairs: Sequence[tuple[int, int]],
    config: GlobalGreedyConfig = GlobalGreedyConfig(),
) -> GlobalGreedyResult:
    """Genuine nonlinear FLOP-style search with unchanged outer mechanics."""
    if config.score_backend != "gaussian":
        return fit_global_greedy_notreks(X, no_trek_pairs, config)
    nonlinear_config = GlobalGreedyConfig(
        **{**config.__dict__, "score_backend": "nonlinear",
           "inner_backend": "python"})
    return fit_global_greedy_notreks(X, no_trek_pairs, nonlinear_config)


def score_nonlinear_dag(
    X: np.ndarray, graph: np.ndarray, *, seed: int = 1729,
    hidden_dim: int = 16, epochs: int = 60,
    score: str = "profiled_log_mse_cv", student_t_nu: float = 4.0,
    lambda_bic: float = 2.0) -> float:
    """Score a fixed DAG with exactly the nonlinear FLOP local score.

    This is intended for score suitability audits: compare the score of the
    generating DAG with the scores of linear/nonlinear FLOP outputs without
    using truth in optimization or stopping decisions.
    """
    config = GlobalGreedyConfig(
        score_backend="nonlinear", seed=seed, nonlinear_hidden_dim=hidden_dim,
        nonlinear_epochs=epochs, nonlinear_score=score,
        nonlinear_student_t_nu=student_t_nu, lambda_bic=lambda_bic)
    scorer = _LocalNonlinearScore(np.asarray(X, dtype=float), config)
    return scorer.graph_score(np.asarray(graph, dtype=np.uint8))
