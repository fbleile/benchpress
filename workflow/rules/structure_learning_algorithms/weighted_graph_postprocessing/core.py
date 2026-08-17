"""Standardized, model-score-consistent DAG/NOTREKS postselection.

Adjacency orientation is ``A[i, j] == 1`` for ``i -> j``.  Continuous
penalties propose edge strengths; this module certifies hard constraints
combinatorially and never reruns the continuous optimizer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import time
from typing import Callable, Protocol, Sequence

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import (
    gaussian_bic,
)
from workflow.rules.structure_learning_algorithms.dagma_anytime.graph_utils import (
    is_dag,
    postprocess_graph,
)


DEFAULT_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.20, 0.30)
LAMBDA_POLICIES = (
    "fixed_0.01", "fixed_0.02", "fixed_0.03", "fixed_0.05",
    "sqrt_c0.5", "sqrt_c1.0", "sqrt_c2.0")
PRODUCTION_POLICIES = (
    "PS1_joint_feasible_greedy_score",
    "PS2_joint_feasible_local_search",
    "PS3_joint_feasible_budgeted_search",
    "PS4_joint_violation_repair",
    "PS5_fixed_threshold_joint_feasible",
)
REFERENCE_POLICY = "REF_threshold_grid_scc_bic_infeasible"
ALL_POLICIES = PRODUCTION_POLICIES + (REFERENCE_POLICY,)


def standardize_training_data(X, *, ddof=0, std_floor=1e-12):
    original = np.asarray(X)
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be two-dimensional")
    means = data.mean(axis=0)
    stds = data.std(axis=0, ddof=ddof)
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)):
        raise ValueError("standardization statistics are not finite")
    safe = np.maximum(stds, float(std_floor))
    standardized = (data - means) / safe
    if not np.all(np.isfinite(standardized)):
        raise ValueError("standardized data contain non-finite values")
    # Never expose a view capable of mutating the caller.
    return np.array(standardized, copy=True, order="C"), means.copy(), safe.copy()


def apply_standardization(X, means, stds):
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(means):
        raise ValueError("validation data do not match training columns")
    return np.array((data - means) / stds, copy=True, order="C")


def lambda_policy(policy, d, n):
    fixed = {
        "fixed_0.01": .01, "fixed_0.02": .02,
        "fixed_0.03": .03, "fixed_0.05": .05}
    if policy in fixed:
        return policy, fixed[policy], 1.0, fixed[policy]
    multipliers = {"sqrt_c0.5": .5, "sqrt_c1.0": 1., "sqrt_c2.0": 2.}
    if policy not in multipliers:
        raise ValueError(f"unknown lambda policy: {policy}")
    multiplier = multipliers[policy]
    return (
        "sqrt_log_d_over_n", None, multiplier,
        float(multiplier * np.sqrt(np.log(d) / n)))


def canonical_pairs(pairs, d):
    canonical = sorted({tuple(sorted((int(i), int(j)))) for i, j in pairs})
    if any(i < 0 or j >= d or i == j for i, j in canonical):
        raise ValueError("NOTREKS pairs must be valid distinct node pairs")
    return tuple(canonical)


def ancestor_matrix(A):
    reach = np.asarray(A, dtype=bool).copy()
    if reach.ndim != 2 or reach.shape[0] != reach.shape[1]:
        raise ValueError("adjacency must be square")
    np.fill_diagonal(reach, True)
    for node in range(len(reach)):
        reach |= reach[:, [node]] & reach[[node], :]
    return reach


def notreks_violation_count(A, pairs):
    if not pairs:
        return 0
    reach = ancestor_matrix(A)
    return int(sum(
        bool(np.any(reach[:, i] & reach[:, j])) for i, j in pairs))


@dataclass(frozen=True)
class StandardizationMetadata:
    data_standardised: bool
    standardisation_ddof: int
    standardisation_std_floor: float
    training_means: tuple[float, ...]
    training_standard_deviations: tuple[float, ...]
    n: int
    d: int


@dataclass(frozen=True)
class EdgeStrengthResult:
    strengths: np.ndarray
    model_class: str
    edge_strength_definition: str
    edge_strength_shape: tuple[int, int]


class EdgeStrengthExtractor:
    def extract(self, weighted_adjacency, *, model_class="linear_dagma"):
        strengths = np.abs(np.asarray(weighted_adjacency, dtype=np.float64))
        if strengths.ndim != 2 or strengths.shape[0] != strengths.shape[1]:
            raise ValueError("DAGMA edge-strength proxy must be square")
        np.fill_diagonal(strengths, 0)
        definitions = {
            "linear_dagma": "absolute_linear_weight",
            "nonlinear_dagma_proxy": "provided_dagma_acyclicity_proxy_abs",
        }
        if model_class not in definitions:
            raise ValueError(
                "nonlinear DAGMA must provide its dxd acyclicity proxy")
        return EdgeStrengthResult(
            strengths, model_class, definitions[model_class],
            strengths.shape)


@dataclass(frozen=True)
class FeasibilityResult:
    DAG_valid: bool
    notreks_violation_count: int
    feasible: bool


class FeasibilityChecker:
    def __init__(self, *, d, dag_constraint_active=True,
                 notreks_constraint_active=False, notreks_pairs=()):
        self.d = int(d)
        self.dag_active = bool(dag_constraint_active)
        self.notreks_active = bool(notreks_constraint_active)
        self.pairs = canonical_pairs(notreks_pairs, d)

    def check(self, A):
        graph = np.asarray(A, dtype=int)
        dag = bool(is_dag(graph))
        violations = (
            notreks_violation_count(graph, self.pairs)
            if self.notreks_active else 0)
        return FeasibilityResult(
            dag, violations,
            (dag or not self.dag_active)
            and (violations == 0 or not self.notreks_active))


@dataclass(frozen=True)
class CandidateSupport:
    adjacency: np.ndarray
    threshold: float
    provenance: tuple[float, ...]


def support_hash(A):
    return hashlib.blake2b(
        np.asarray(A, dtype=np.uint8).tobytes(), digest_size=16).hexdigest()


class CandidatePoolBuilder:
    def __init__(self, thresholds=DEFAULT_THRESHOLDS, fixed_threshold=.30):
        self.thresholds = tuple(float(value) for value in thresholds)
        self.fixed_threshold = float(fixed_threshold)

    def _support(self, strengths, threshold):
        graph = (np.asarray(strengths) >= threshold).astype(int)
        np.fill_diagonal(graph, 0)
        return graph

    def threshold_grid(self, strengths):
        deduplicated = {}
        for threshold in self.thresholds:
            graph = self._support(strengths, threshold)
            key = support_hash(graph)
            if key in deduplicated:
                old = deduplicated[key]
                deduplicated[key] = CandidateSupport(
                    old.adjacency, old.threshold,
                    old.provenance + (threshold,))
            else:
                deduplicated[key] = CandidateSupport(
                    graph, threshold, (threshold,))
        return list(deduplicated.values())

    def fixed(self, strengths):
        graph = self._support(strengths, self.fixed_threshold)
        return [CandidateSupport(
            graph, self.fixed_threshold, (self.fixed_threshold,))]

    def low_threshold(self, strengths, threshold):
        graph = self._support(strengths, float(threshold))
        return [CandidateSupport(graph, float(threshold), (float(threshold),))]

    def union(self, strengths):
        supports = self.threshold_grid(strengths)
        union = np.maximum.reduce([item.adjacency for item in supports])
        return [CandidateSupport(union, min(self.thresholds), self.thresholds)]


class CandidateScorer(Protocol):
    score_name: str
    loss_name: str
    regularizer_type: str
    regularizer_weight: float
    refit_method: str
    def score(self, A: np.ndarray) -> float: ...


@dataclass
class LinearCandidateScorer:
    X: np.ndarray
    regularizer_type: str = "L1"
    regularizer_weight: float = .03
    max_coordinate_iterations: int = 500
    tolerance: float = 1e-10
    cache: dict[str, float] = field(default_factory=dict)
    refit_calls: int = 0
    score_name: str = "configured_model_refit_score"
    loss_name: str = "linear_squared_loss"
    refit_method: str = "coordinate_descent_or_ridge"

    def __post_init__(self):
        data = np.asarray(self.X, dtype=np.float64)
        self.X = data
        self.cov = data.T @ data / len(data)
        self.regularizer_type = self.regularizer_type.upper()
        if self.regularizer_type not in {"L1", "L2"}:
            raise ValueError("regularizer_type must be L1 or L2")

    def _fit(self, node, parents):
        parents = np.asarray(parents, dtype=int)
        if not len(parents):
            return np.zeros(0), .5 * float(self.cov[node, node])
        C = self.cov[np.ix_(parents, parents)]
        target = self.cov[parents, node]
        if self.regularizer_type == "L2":
            beta = np.linalg.solve(
                C + (self.regularizer_weight + 1e-10) * np.eye(len(parents)),
                target)
            penalty = .5 * self.regularizer_weight * float(beta @ beta)
        else:
            beta = np.zeros(len(parents))
            diagonal = np.maximum(np.diag(C), 1e-12)
            for _ in range(self.max_coordinate_iterations):
                old = beta.copy()
                for index in range(len(beta)):
                    rho = (
                        target[index] - C[index] @ beta
                        + C[index, index] * beta[index])
                    beta[index] = (
                        np.sign(rho)
                        * max(abs(rho) - self.regularizer_weight, 0)
                        / diagonal[index])
                if np.max(np.abs(beta - old)) <= self.tolerance:
                    break
            penalty = self.regularizer_weight * float(np.abs(beta).sum())
        residual = float(
            self.cov[node, node] - 2 * target @ beta + beta @ C @ beta)
        return beta, .5 * max(residual, 0) + penalty

    def _local(self, node, parents):
        return self._fit(node, parents)[1]

    def score(self, A):
        graph = np.asarray(A, dtype=int)
        key = support_hash(graph)
        if key in self.cache:
            return self.cache[key]
        self.refit_calls += 1
        value = float(sum(
            self._local(node, np.flatnonzero(graph[:, node]))
            for node in range(graph.shape[0])))
        self.cache[key] = value
        return value

    def coefficients(self, A):
        graph = np.asarray(A, dtype=int)
        coefficients = np.zeros_like(graph, dtype=float)
        for node in range(len(graph)):
            parents = np.flatnonzero(graph[:, node])
            beta, _ = self._fit(node, parents)
            coefficients[parents, node] = beta
        return coefficients


@dataclass
class CallbackCandidateScorer:
    """Nonlinear/model-specific scorer supplied by the target model."""
    callback: Callable[[np.ndarray], float]
    regularizer_type: str
    regularizer_weight: float
    loss_name: str
    refit_method: str = "model_specific_masked_refit"
    score_name: str = "configured_model_refit_score"
    cache: dict[str, float] = field(default_factory=dict)
    refit_calls: int = 0

    def score(self, A):
        graph = np.asarray(A, dtype=int)
        key = support_hash(graph)
        if key not in self.cache:
            self.refit_calls += 1
            self.cache[key] = float(self.callback(graph.copy()))
        return self.cache[key]


@dataclass(frozen=True)
class PostselectionConfig:
    policy: str = "PS1_joint_feasible_greedy_score"
    candidate_edge_pool: str = "threshold_grid"
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLDS
    fixed_threshold: float = .30
    low_threshold: float = .01
    max_search_seconds: float = 1.
    max_expanded_nodes: int = 1000
    max_queue_size: int = 1000
    max_ambiguous_edges: int = 20
    max_indegree: int | None = None
    dag_constraint_active: bool = True
    notreks_constraint_active: bool = False


@dataclass
class SearchDiagnostics:
    search_nodes_expanded: int = 0
    search_nodes_pruned_dag: int = 0
    search_nodes_pruned_notreks: int = 0
    search_nodes_pruned_score: int = 0
    incumbent_improvements: int = 0
    cutoff_reason: str = "completed"
    optimality_certified: bool = False


@dataclass
class PostselectionResult:
    postselection_policy: str
    adjacency: np.ndarray
    selected_threshold: float
    candidate_score: float
    predicted_edges: int
    DAG_valid: bool
    notreks_violation_count: int
    feasible: bool
    reference_only: bool
    eligible_for_recommendation: bool
    hard_feasibility_not_guaranteed: bool
    model_class: str
    edge_strength_definition: str
    edge_strength_shape: tuple[int, int]
    score_name: str
    loss_name: str
    regularizer_type: str
    regularizer_weight: float
    refit_method: str
    postselection_time_seconds: float
    thresholds_requested: int
    unique_supports: int
    search: SearchDiagnostics
    effective_configuration: dict

    def to_row(self):
        row = asdict(self)
        row.pop("adjacency")
        row.update(asdict(self.search))
        row.pop("search")
        return row


def _edge_order(pool, strengths):
    edges = [
        (float(strengths[i, j]), int(i), int(j))
        for i, j in zip(*np.nonzero(pool))]
    return sorted(edges, key=lambda item: (-item[0], item[1], item[2]))


def _indegree_ok(A, maximum):
    return maximum is None or bool(np.all(np.sum(A, axis=0) <= maximum))


def _greedy(pool, strengths, checker, maximum, diagnostics):
    graph = np.zeros_like(pool, dtype=int)
    for _, source, target in _edge_order(pool, strengths):
        proposal = graph.copy()
        proposal[source, target] = 1
        diagnostics.search_nodes_expanded += 1
        feasible = checker.check(proposal)
        if feasible.feasible and _indegree_ok(proposal, maximum):
            graph = proposal
        else:
            diagnostics.search_nodes_pruned_dag += int(
                checker.dag_active and not feasible.DAG_valid)
            diagnostics.search_nodes_pruned_notreks += int(
                checker.notreks_active
                and feasible.notreks_violation_count > 0)
    return graph


def _local_search(initial, pool, scorer, checker, config, diagnostics):
    graph = initial.copy()
    best = scorer.score(graph)
    started = time.perf_counter()
    while diagnostics.search_nodes_expanded < config.max_expanded_nodes:
        if time.perf_counter() - started >= config.max_search_seconds:
            diagnostics.cutoff_reason = "max_search_seconds"
            break
        best_move = None
        for source, target in zip(*np.nonzero(pool)):
            moves = []
            if graph[source, target]:
                deletion = graph.copy()
                deletion[source, target] = 0
                moves.append(deletion)
                reversal = deletion.copy()
                if pool[target, source]:
                    reversal[target, source] = 1
                    moves.append(reversal)
            else:
                addition = graph.copy()
                addition[source, target] = 1
                moves.append(addition)
            for proposal in moves:
                diagnostics.search_nodes_expanded += 1
                feasibility = checker.check(proposal)
                if not feasibility.feasible or not _indegree_ok(
                        proposal, config.max_indegree):
                    diagnostics.search_nodes_pruned_dag += int(
                        checker.dag_active and not feasibility.DAG_valid)
                    diagnostics.search_nodes_pruned_notreks += int(
                        checker.notreks_active
                        and feasibility.notreks_violation_count > 0)
                    continue
                score = scorer.score(proposal)
                key = (score, int(proposal.sum()), support_hash(proposal))
                if score < best - 1e-12 and (
                        best_move is None or key < best_move[0]):
                    best_move = (key, proposal)
                else:
                    diagnostics.search_nodes_pruned_score += 1
                if diagnostics.search_nodes_expanded >= config.max_expanded_nodes:
                    break
            if diagnostics.search_nodes_expanded >= config.max_expanded_nodes:
                break
        if best_move is None:
            diagnostics.cutoff_reason = "local_optimum"
            break
        graph = best_move[1]
        best = best_move[0][0]
        diagnostics.incumbent_improvements += 1
    return graph


def _budgeted(initial, pool, strengths, scorer, checker, config, diagnostics):
    edges = []
    for source, target in zip(*np.nonzero(pool)):
        strength = float(strengths[source, target])
        boundary_distance = min(
            abs(strength - threshold) for threshold in config.threshold_grid)
        opposite_competitor = bool(pool[target, source])
        edges.append((
            boundary_distance, not opposite_competitor,
            strength, int(source), int(target)))
    edges.sort()
    edges = [
        (strength, source, target)
        for _, _, strength, source, target
        in edges[:config.max_ambiguous_edges]]
    best, best_score = initial.copy(), scorer.score(initial)
    queue = [(best_score, 0, 0, np.zeros_like(pool, dtype=int))]
    serial = 1
    started = time.perf_counter()
    while queue:
        if time.perf_counter() - started >= config.max_search_seconds:
            diagnostics.cutoff_reason = "max_search_seconds"
            break
        if diagnostics.search_nodes_expanded >= config.max_expanded_nodes:
            diagnostics.cutoff_reason = "max_expanded_nodes"
            break
        _, index, _, graph = heapq.heappop(queue)
        if index == len(edges):
            score = scorer.score(graph)
            if score < best_score:
                best, best_score = graph.copy(), score
                diagnostics.incumbent_improvements += 1
            continue
        _, source, target = edges[index]
        for include in (False, True):
            proposal = graph.copy()
            if include:
                proposal[source, target] = 1
            diagnostics.search_nodes_expanded += 1
            feasibility = checker.check(proposal)
            if not feasibility.feasible or not _indegree_ok(
                    proposal, config.max_indegree):
                diagnostics.search_nodes_pruned_dag += int(
                    checker.dag_active and not feasibility.DAG_valid)
                diagnostics.search_nodes_pruned_notreks += int(
                    checker.notreks_active
                    and feasibility.notreks_violation_count > 0)
                continue
            score = scorer.score(proposal)
            heapq.heappush(queue, (score, index + 1, serial, proposal))
            serial += 1
        if len(queue) > config.max_queue_size:
            queue = heapq.nsmallest(config.max_queue_size, queue)
            heapq.heapify(queue)
    else:
        diagnostics.cutoff_reason = "queue_exhausted"
        diagnostics.optimality_certified = (
            len(edges) == int(pool.sum())
            and diagnostics.search_nodes_expanded < config.max_expanded_nodes)
    return best


def _repair(initial, strengths, scorer, checker, config, diagnostics):
    graph = initial.copy()
    best = None
    best_score = float("inf")
    # Follow one deterministic weakest-edge deletion path.  Unlike the old
    # feasibility-only repair, retain every feasible incumbent and select the
    # best refit/BIC score along the complete path.
    while graph.any():
        feasibility = checker.check(graph)
        if feasibility.feasible:
            score = float(scorer.score(graph))
            if (score < best_score - 1e-12 or
                    (abs(score - best_score) <= 1e-12 and
                     (best is None or graph.sum() < best.sum()))):
                best, best_score = graph.copy(), score
        candidates = []
        for source, target in zip(*np.nonzero(graph)):
            candidates.append((
                float(strengths[source, target]),
                int(source), int(target)))
        _, source, target = min(candidates)
        graph[source, target] = 0
        diagnostics.search_nodes_expanded += 1
    diagnostics.cutoff_reason = "repaired"
    if best is None:
        return np.zeros_like(graph, dtype=int)
    diagnostics.incumbent_improvements += 1
    return best


def select_postselection_candidate(
    weighted_adjacency,
    *,
    scorer: CandidateScorer,
    config=PostselectionConfig(),
    model_class="linear_dagma",
    notreks_pairs=(),
):
    started = time.perf_counter()
    edge_result = EdgeStrengthExtractor().extract(
        weighted_adjacency, model_class=model_class)
    strengths = edge_result.strengths
    d = len(strengths)
    checker = FeasibilityChecker(
        d=d, dag_constraint_active=config.dag_constraint_active,
        notreks_constraint_active=config.notreks_constraint_active,
        notreks_pairs=notreks_pairs)
    builder = CandidatePoolBuilder(
        config.threshold_grid, config.fixed_threshold)
    if config.policy == "PS5_fixed_threshold_joint_feasible":
        pools = builder.fixed(strengths)
    elif config.candidate_edge_pool == "threshold_grid":
        pools = builder.threshold_grid(strengths)
    elif config.candidate_edge_pool == "fixed_threshold":
        pools = builder.fixed(strengths)
    elif config.candidate_edge_pool == "low_threshold_supergraph":
        pools = builder.low_threshold(strengths, config.low_threshold)
    elif config.candidate_edge_pool == "union_threshold_supports":
        pools = builder.union(strengths)
    else:
        raise ValueError("unknown candidate_edge_pool")

    if config.policy == REFERENCE_POLICY:
        if model_class != "linear_dagma":
            raise ValueError("Gaussian BIC reference is linear-only")
        X = getattr(scorer, "X", None)
        if X is None:
            raise ValueError("Gaussian BIC reference requires linear data")
        candidates = []
        for pool in pools:
            projected = postprocess_graph(
                weighted_adjacency, pool.threshold).projected_dag
            bic, _ = gaussian_bic(X, projected, lambda_bic=1.)
            candidates.append((bic, pool.threshold, int(projected.sum()), projected))
        score, threshold, _, graph = min(candidates)
        feasibility = checker.check(graph)
        diagnostics = SearchDiagnostics(cutoff_reason="reference_only")
        return PostselectionResult(
            config.policy, graph, threshold, score, int(graph.sum()),
            feasibility.DAG_valid, feasibility.notreks_violation_count,
            feasibility.feasible, True, False, True, model_class,
            edge_result.edge_strength_definition, edge_result.edge_strength_shape,
            "gaussian_bic_reference", "linear_gaussian", "none", 0.,
            "OLS", time.perf_counter() - started, len(pools), len(pools),
            diagnostics, asdict(config))

    if config.policy not in PRODUCTION_POLICIES:
        raise ValueError("unknown postselection policy")
    candidate_rows = []
    for pool in pools:
        diagnostics = SearchDiagnostics()
        if config.policy == "PS4_joint_violation_repair":
            graph = _repair(
                pool.adjacency, strengths, scorer, checker, config, diagnostics)
        else:
            graph = _greedy(
                pool.adjacency, strengths, checker,
                config.max_indegree, diagnostics)
            if config.policy == "PS2_joint_feasible_local_search":
                graph = _local_search(
                    graph, pool.adjacency, scorer, checker, config, diagnostics)
            elif config.policy == "PS3_joint_feasible_budgeted_search":
                graph = _budgeted(
                    graph, pool.adjacency, strengths, scorer, checker,
                    config, diagnostics)
        feasibility = checker.check(graph)
        if not feasibility.feasible:
            graph = np.zeros((d, d), dtype=int)
            feasibility = checker.check(graph)
            diagnostics.cutoff_reason += "_empty_fallback"
        candidate_rows.append((
            scorer.score(graph), int(graph.sum()), pool.threshold,
            graph, feasibility, diagnostics))
    score, edges, threshold, graph, feasibility, diagnostics = min(
        candidate_rows, key=lambda item: (
            item[0], item[1], item[2], support_hash(item[3])))
    return PostselectionResult(
        config.policy, graph, threshold, score, edges,
        feasibility.DAG_valid, feasibility.notreks_violation_count,
        feasibility.feasible, False, True, False, model_class,
        edge_result.edge_strength_definition, edge_result.edge_strength_shape,
        scorer.score_name, scorer.loss_name, scorer.regularizer_type,
        scorer.regularizer_weight, scorer.refit_method,
        time.perf_counter() - started, len(pools), len(pools),
        diagnostics, asdict(config))
