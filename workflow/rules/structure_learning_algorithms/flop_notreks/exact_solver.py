"""Persistent SCIP family-variable branch-and-cut for exact DAG/NOTREKS search.

All parent sets up to ``max_indegree`` are included unless a logically valid
forbidden-arc rule removes them.  There is deliberately no score screening.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product
from math import inf
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from .chromatic_sources import chromatic_number_or_lower_bound
from .global_greedy import _LocalGaussianBIC

try:
    from pyscipopt import Conshdlr, Model, SCIP_RESULT, quicksum
except ImportError:  # pragma: no cover - exercised only without optional SCIP
    Conshdlr = object
    Model = None
    SCIP_RESULT = None
    quicksum = None


Pair = tuple[int, int]


@dataclass(frozen=True)
class ExactConfig:
    max_indegree: int = 2
    lambda_bic: float = 2.0
    time_limit_seconds: float = 300.0
    variant: str = "lazy_trek"
    chromatic_search_nodes: int = 2_000_000
    random_seed: int = 0
    verbose: bool = False


@dataclass
class ExactResult:
    adjacency: np.ndarray
    objective: float | None
    dual_bound: float | None
    mip_gap: float | None
    status: str
    zero_gap_certificate: bool
    runtime_seconds: float
    parent_set_count: int
    cluster_cuts: int
    trek_cuts: int
    source_bound: int
    source_bound_exact: bool
    source_count: int
    no_trek_violations: int
    dag_verified: bool
    notreks_verified: bool
    max_indegree: int
    optimality_scope: str
    solver_nodes: int
    warm_start_accepted: bool


def canonical_pairs(p: int, pairs: Sequence[Pair]) -> tuple[Pair, ...]:
    result = set()
    for raw_left, raw_right in pairs:
        left, right = sorted((int(raw_left), int(raw_right)))
        if left == right or left < 0 or right >= p:
            raise ValueError(f"invalid no-trek pair {(raw_left, raw_right)}")
        result.add((left, right))
    return tuple(sorted(result))


def enumerate_families(p: int, max_indegree: int, pairs: Sequence[Pair],
                       forbid_pair_arcs: bool):
    forbidden = {frozenset(pair) for pair in pairs}
    families = []
    by_child = [[] for _ in range(p)]
    for child in range(p):
        others = [node for node in range(p) if node != child]
        for size in range(min(max_indegree, p - 1) + 1):
            for parents in combinations(others, size):
                if forbid_pair_arcs and any(
                        frozenset((parent, child)) in forbidden
                        for parent in parents):
                    continue
                index = len(families)
                families.append((child, parents))
                by_child[child].append(index)
    return families, by_child


def adjacency_from_families(p: int, families, selected: Iterable[int]):
    adjacency = np.zeros((p, p), dtype=np.uint8)
    for index in selected:
        child, parents = families[index]
        adjacency[list(parents), child] = 1
    return adjacency


def ancestor_bitsets(adjacency: np.ndarray) -> list[int]:
    """Exact source-ancestor sets encoded as arbitrary-width Python bitsets."""
    p = len(adjacency)
    ancestors = [1 << node for node in range(p)]
    indegree = adjacency.sum(axis=0).astype(int)
    ready = [node for node in range(p) if indegree[node] == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for child in np.flatnonzero(adjacency[node]):
            ancestors[child] |= ancestors[node]
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    if seen != p:
        raise ValueError("ancestry requested for a cyclic graph")
    return ancestors


def dag_verified(adjacency: np.ndarray) -> bool:
    indegree = adjacency.sum(axis=0).astype(int)
    ready = [node for node in range(len(adjacency)) if indegree[node] == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    return seen == len(adjacency)


def topological_positions(adjacency: np.ndarray):
    indegree = adjacency.sum(axis=0).astype(int)
    ready = sorted(node for node in range(len(adjacency)) if indegree[node] == 0)
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
                ready.sort()
    if len(order) != len(adjacency):
        raise ValueError("topological positions requested for cyclic graph")
    positions = np.empty(len(order), dtype=int)
    positions[order] = np.arange(len(order))
    return positions


def no_trek_violations(adjacency: np.ndarray, pairs: Sequence[Pair]):
    if not dag_verified(adjacency):
        return list(pairs)
    ancestors = ancestor_bitsets(adjacency)
    return [pair for pair in pairs if ancestors[pair[0]] & ancestors[pair[1]]]


def project_warm_start(data: np.ndarray, adjacency: np.ndarray,
                       max_indegree: int, lambda_bic: float = 2.0):
    """Delete excess parents using exact local BIC; never add an edge."""
    projected = np.asarray(adjacency, dtype=np.uint8).copy()
    scorer = _LocalGaussianBIC(np.asarray(data, dtype=float), lambda_bic)
    for child in range(len(projected)):
        parents = tuple(map(int, np.flatnonzero(projected[:, child])))
        if len(parents) <= max_indegree:
            continue
        retained = min(combinations(parents, max_indegree),
                       key=lambda subset: (scorer.score(child, subset), subset))
        projected[:, child] = 0
        projected[list(retained), child] = 1
    return projected


def _path(adjacency: np.ndarray, source: int, target: int) -> list[tuple[int, int]]:
    if source == target:
        return []
    predecessor = {source: None}
    queue = [source]
    for node in queue:
        for child in np.flatnonzero(adjacency[node]):
            child = int(child)
            if child not in predecessor:
                predecessor[child] = node
                queue.append(child)
    if target not in predecessor:
        raise AssertionError("common ancestor has no directed witness path")
    edges = []
    node = target
    while predecessor[node] is not None:
        edges.append((predecessor[node], node))
        node = predecessor[node]
    return edges


def trek_witness(adjacency: np.ndarray, pair: Pair):
    ancestors = ancestor_bitsets(adjacency)
    common = ancestors[pair[0]] & ancestors[pair[1]]
    if not common:
        return None
    source = (common & -common).bit_length() - 1
    return source, frozenset(_path(adjacency, source, pair[0])
                             + _path(adjacency, source, pair[1]))


def _strong_components(adjacency: np.ndarray):
    reach = adjacency.astype(bool).copy()
    for node in range(len(reach)):
        reach |= reach[:, [node]] & reach[[node], :]
    unseen = set(range(len(reach)))
    result = []
    while unseen:
        node = min(unseen)
        component = {other for other in unseen
                     if (other == node or (reach[node, other] and reach[other, node]))}
        unseen -= component
        if len(component) > 1 or reach[node, node]:
            result.append(frozenset(component))
    return result


class _DagNoTreksHandler(Conshdlr):
    def __init__(self, p, pairs, families, variables, by_child,
                 enforce_treks, exhaustive_cluster_limit=16):
        self.p = p
        self.pairs = pairs
        self.families = families
        self.variables = variables
        self.by_child = by_child
        self.enforce_treks = enforce_treks
        self.exhaustive_cluster_limit = exhaustive_cluster_limit
        self.cluster_keys = set()
        self.trek_keys = set()
        self.cluster_cuts = 0
        self.trek_cuts = 0

    def _values(self, solution=None):
        return np.asarray([self.model.getSolVal(solution, variable)
                           for variable in self.variables])

    def _selected_adjacency(self, values):
        selected = []
        for child in range(self.p):
            index = max(self.by_child[child], key=lambda i: values[i])
            selected.append(index)
        return adjacency_from_families(self.p, self.families, selected)

    def _cluster_lhs(self, cluster):
        return quicksum(self.variables[index]
                        for child in cluster for index in self.by_child[child]
                        if not cluster.intersection(self.families[index][1]))

    def _add_cluster(self, cluster):
        key = frozenset(cluster)
        if len(key) < 2 or key in self.cluster_keys:
            return False
        self.model.addCons(self._cluster_lhs(key) >= 1,
                           name=f"cluster_{self.cluster_cuts}",
                           initial=False, removable=True)
        self.cluster_keys.add(key)
        self.cluster_cuts += 1
        return True

    def _add_trek(self, edges):
        key = frozenset(edges)
        if not key or key in self.trek_keys:
            return False
        arc_sum = quicksum(
            self.variables[index]
            for parent, child in key for index in self.by_child[child]
            if parent in self.families[index][1])
        self.model.addCons(arc_sum <= len(key) - 1,
                           name=f"trek_{self.trek_cuts}",
                           initial=False, removable=False)
        self.trek_keys.add(key)
        self.trek_cuts += 1
        return True

    def _separate(self, values, require_integral):
        added = False
        violated = False
        arc_values = np.zeros((self.p, self.p))
        for index, value in enumerate(values):
            child, parents = self.families[index]
            for parent in parents:
                arc_values[parent, child] += value
        if require_integral:
            adjacency = self._selected_adjacency(values)
            for component in _strong_components(adjacency):
                violated = True
                added |= self._add_cluster(component)
            if dag_verified(adjacency) and self.enforce_treks:
                for pair in no_trek_violations(adjacency, self.pairs):
                    violated = True
                    _, edges = trek_witness(adjacency, pair)
                    added |= self._add_trek(edges)
            return added, violated

        candidates = set()
        if self.p <= self.exhaustive_cluster_limit:
            for size in range(2, self.p + 1):
                candidates.update(frozenset(c) for c in combinations(range(self.p), size))
        else:
            for threshold in (.15, .3, .5):
                candidates.update(_strong_components(arc_values > threshold))
        for cluster in candidates:
            lhs = sum(values[index]
                      for child in cluster for index in self.by_child[child]
                      if not cluster.intersection(self.families[index][1]))
            if lhs < 1 - 1e-7:
                added |= self._add_cluster(cluster)
        return added, added

    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        values = self._values()
        integral = bool(np.all(np.minimum(abs(values), abs(values - 1)) < 1e-7))
        added, violated = self._separate(values, integral)
        result = (SCIP_RESULT.CONSADDED if added else
                  SCIP_RESULT.INFEASIBLE if violated else SCIP_RESULT.FEASIBLE)
        return {"result": result}

    def conscheck(self, constraints, solution, checkintegrality, checklprows,
                  printreason, completely):
        adjacency = self._selected_adjacency(self._values(solution))
        feasible = dag_verified(adjacency)
        if feasible and self.enforce_treks:
            feasible = not no_trek_violations(adjacency, self.pairs)
        return {"result": SCIP_RESULT.FEASIBLE if feasible else SCIP_RESULT.INFEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        pass


def fit_exact_notreks(data: np.ndarray, pairs: Sequence[Pair],
                      config: ExactConfig = ExactConfig(),
                      warm_start: np.ndarray | None = None) -> ExactResult:
    started = perf_counter()
    if Model is None:
        raise RuntimeError("PySCIPOpt is required for exact NOTREKS")
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("data must be a nonempty two-dimensional matrix")
    p = data.shape[1]
    pairs = canonical_pairs(p, pairs)
    variants = {"ordinary", "forbidden_arcs", "source_cut", "lazy_trek"}
    if config.variant not in variants:
        raise ValueError(f"variant must be one of {sorted(variants)}")
    forbid_arcs = config.variant != "ordinary"
    use_source = config.variant in {"source_cut", "lazy_trek"}
    use_treks = config.variant == "lazy_trek"
    families, by_child = enumerate_families(
        p, config.max_indegree, pairs, forbid_arcs)
    scorer = _LocalGaussianBIC(data, config.lambda_bic)
    scores = [scorer.score(child, parents) for child, parents in families]
    model = Model(f"exact_notreks_{config.variant}")
    model.hideOutput(not config.verbose)
    model.setRealParam("limits/time", config.time_limit_seconds)
    model.setIntParam("randomization/randomseedshift", config.random_seed)
    # Python constraint handlers retain references to the original family
    # variables.  Prevent presolve aggregation from invalidating callback
    # separation; SCIP still performs propagation and node processing.
    model.setIntParam("presolving/maxrounds", 0)
    variables = [model.addVar(vtype="B", obj=score,
                              name=f"f_{child}_{'_'.join(map(str, parents)) or 'empty'}")
                 for score, (child, parents) in zip(scores, families)]
    for child in range(p):
        model.addCons(quicksum(variables[i] for i in by_child[child]) == 1,
                      name=f"family_{child}")
    # Exact extended topological-order formulation.  Cluster cuts remain the
    # standard family-variable strengthening separated below; these compact
    # constraints ensure integer acyclicity even if no fractional cluster is
    # found by the d=50 heuristic separator.
    order_variables = [model.addVar(vtype="I", lb=0, ub=p - 1,
                                    name=f"order_{node}") for node in range(p)]
    for parent in range(p):
        for child in range(p):
            if parent == child:
                continue
            arc = quicksum(variables[index] for index in by_child[child]
                           if parent in families[index][1])
            model.addCons(order_variables[parent] + 1
                          <= order_variables[child] + p * (1 - arc),
                          name=f"order_arc_{parent}_{child}")
    source_bound = 0
    source_exact = True
    if use_source:
        chromatic = chromatic_number_or_lower_bound(
            p, pairs, max_search_nodes=config.chromatic_search_nodes)
        source_bound, source_exact = chromatic.value, chromatic.exact
        empty_variables = [variables[next(i for i in by_child[node]
                                          if not families[i][1])]
                           for node in range(p)]
        model.addCons(quicksum(empty_variables) >= source_bound,
                      name="certified_source_bound")
        for variable in empty_variables:
            model.chgVarBranchPriority(variable, 100000)
    handler = _DagNoTreksHandler(
        p, pairs, families, variables, by_child, use_treks)
    model.includeConshdlr(
        handler, "dag_notreks", "cluster and lazy forbidden-trek cuts",
        sepapriority=-10, enfopriority=-10, chckpriority=-10,
        sepafreq=1, needscons=True)
    model.addPyCons(model.createCons(handler, "dag_notreks_constraint"))
    warm_start_accepted = False
    if warm_start is not None:
        warm_start = project_warm_start(
            data, warm_start, config.max_indegree, config.lambda_bic)
        selected = []
        valid = warm_start.shape == (p, p) and dag_verified(warm_start)
        valid &= not use_treks or not no_trek_violations(warm_start, pairs)
        for child in range(p):
            parents = tuple(map(int, np.flatnonzero(warm_start[:, child])))
            match = next((i for i in by_child[child] if families[i][1] == parents), None)
            valid &= match is not None
            selected.append(match)
        if valid:
            solution = model.createSol()
            selected_set = set(selected)
            for index, variable in enumerate(variables):
                model.setSolVal(solution, variable, float(index in selected_set))
            for node, position in enumerate(topological_positions(warm_start)):
                model.setSolVal(solution, order_variables[node], int(position))
            warm_start_accepted = bool(model.addSol(solution))
    model.optimize()
    runtime = perf_counter() - started
    status = str(model.getStatus())
    solution = model.getBestSol()
    if solution is None:
        adjacency = np.zeros((p, p), dtype=np.uint8)
        objective = None
    else:
        selected = [index for index, variable in enumerate(variables)
                    if model.getSolVal(solution, variable) > .5]
        adjacency = adjacency_from_families(p, families, selected)
        objective = float(model.getSolObjVal(solution))
    dual = float(model.getDualbound())
    gap = float(model.getGap()) if solution is not None else inf
    dag_ok = dag_verified(adjacency) if solution is not None else False
    violations = no_trek_violations(adjacency, pairs) if dag_ok else list(pairs)
    sources = int(np.sum(adjacency.sum(axis=0) == 0)) if solution is not None else 0
    zero_gap = status == "optimal" and gap <= 1e-9
    return ExactResult(
        adjacency=adjacency, objective=objective, dual_bound=dual,
        mip_gap=gap, status=status, zero_gap_certificate=zero_gap,
        runtime_seconds=runtime, parent_set_count=len(families),
        cluster_cuts=handler.cluster_cuts, trek_cuts=handler.trek_cuts,
        source_bound=source_bound, source_bound_exact=source_exact,
        source_count=sources, no_trek_violations=len(violations),
        dag_verified=dag_ok, notreks_verified=dag_ok and not violations,
        max_indegree=config.max_indegree,
        optimality_scope=f"all parent sets with indegree <= {config.max_indegree}",
        solver_nodes=int(model.getNNodes()),
        warm_start_accepted=warm_start_accepted)


def result_dict(result: ExactResult):
    output = asdict(result)
    output.pop("adjacency")
    return output


def exhaustive_optimum(data: np.ndarray, pairs: Sequence[Pair], max_indegree: int,
                       lambda_bic: float = 2.0):
    """Independent tiny-instance oracle used only for exactness tests."""
    p = data.shape[1]
    pairs = canonical_pairs(p, pairs)
    families, by_child = enumerate_families(p, max_indegree, pairs, True)
    scorer = _LocalGaussianBIC(data, lambda_bic)
    scores = [scorer.score(child, parents) for child, parents in families]
    best = (inf, None)
    for selected in product(*by_child):
        adjacency = adjacency_from_families(p, families, selected)
        if dag_verified(adjacency) and not no_trek_violations(adjacency, pairs):
            objective = sum(scores[index] for index in selected)
            if objective < best[0] - 1e-9:
                best = objective, adjacency
    return best
