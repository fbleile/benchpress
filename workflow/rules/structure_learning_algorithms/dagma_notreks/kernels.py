"""Benchmarkable PST/NOTREKS value-gradient kernels.

All kernels in this module use the repository's existing NOTREKS contract:

    X = W * W
    F = f(X)
    R_I(W) = c_I sum_{(i,j) in I} [F.T F]_{ij}

with the same unordered-pair scaling c_I = 2 / (d - 1) used by the original
``NoTreksKernel`` in ``dagma.shared``.  The public optimizer hook deliberately
keeps these kernels behind explicit names so the existing implementation remains
recoverable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Literal, Sequence

import numpy as np
import scipy.linalg as sla


Pair = tuple[int, int]
PairSemantics = Literal["unordered", "ordered"]


def validate_pairs(
    pairs: Sequence[Pair],
    d: int,
    *,
    semantics: PairSemantics = "unordered",
) -> np.ndarray:
    """Validate pair indices and canonicalise unordered pairs."""
    out = np.asarray(pairs, dtype=int)
    if out.size == 0:
        return np.empty((0, 2), dtype=int)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError("pairs must have shape (m, 2)")
    if np.any(out < 0) or np.any(out >= d) or np.any(out[:, 0] == out[:, 1]):
        raise ValueError("pair indices must be distinct and in range")
    out = out.copy()
    if semantics == "unordered":
        out.sort(axis=1)
    elif semantics != "ordered":
        raise ValueError("semantics must be 'unordered' or 'ordered'")
    if len({tuple(x) for x in out}) != len(out):
        raise ValueError("duplicate no-trek pairs")
    out.setflags(write=False)
    return out


def pair_mask(
    pairs: np.ndarray,
    d: int,
    *,
    semantics: PairSemantics = "unordered",
    pair_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Return the full pair mask used for dense adjoint formulas."""
    mask = np.zeros((d, d), dtype=float)
    if len(pairs):
        weights = (np.ones(len(pairs), dtype=float) if pair_weights is None
                   else np.asarray(pair_weights, dtype=float))
        if weights.shape != (len(pairs),):
            raise ValueError("pair_weights must have one value per pair")
        if not np.all(np.isfinite(weights)):
            raise ValueError("pair_weights must be finite")
        for (i, j), weight in zip(pairs, weights):
            mask[i, j] += weight
            if semantics == "unordered":
                mask[j, i] += weight
    mask.setflags(write=False)
    return mask


def selected_nodes(pairs: np.ndarray) -> np.ndarray:
    """Sorted unique nodes appearing in the constrained pair set."""
    if len(pairs) == 0:
        return np.empty(0, dtype=int)
    nodes = np.unique(pairs.reshape(-1))
    nodes.setflags(write=False)
    return nodes


@dataclass
class KernelDiagnostics:
    name: str
    function: str
    implementation_class: str
    forward_time_seconds: float = 0.0
    backward_time_seconds: float = 0.0
    total_kernel_time_seconds: float = 0.0
    peak_memory_bytes: int = 0
    domain_valid: bool = True
    numerical_status: str = "ok"
    matrix_factorizations: int = 0
    matrix_function_evaluations: int = 0
    linear_solves: int = 0
    right_hand_sides: int = 0
    dense_matrix_multiplications: int = 0
    selected_column_count: int = 0
    full_matrix_materialized: bool = True
    condition_number: float = math.nan
    spectral_radius: float = math.nan
    domain_margin: float = math.nan


@dataclass
class ValueGradResult:
    penalty_value: float
    gradient_W: np.ndarray
    diagnostics: KernelDiagnostics


@dataclass
class BaseNoTreksKernel:
    """Base class implementing validation, pair masks and timing storage."""

    d: int
    pairs: np.ndarray
    semantics: PairSemantics = "unordered"
    pair_weights: np.ndarray | None = None
    scale: float = field(init=False)
    mask: np.ndarray = field(init=False)
    nodes: np.ndarray = field(init=False)
    maximum_condition_number: float = 0.0
    solve_failures: int = 0
    _last: KernelDiagnostics | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.scale = 2.0 / (self.d - 1) if self.d > 1 else 0.0
        self.mask = pair_mask(
            self.pairs, self.d, semantics=self.semantics,
            pair_weights=self.pair_weights)
        self.nodes = selected_nodes(self.pairs)

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[Pair],
        d: int,
        *,
        semantics: PairSemantics = "unordered",
        pair_weights: Sequence[float] | None = None,
    ):
        canonical = validate_pairs(pairs, d, semantics=semantics)
        weights = None if pair_weights is None else np.asarray(pair_weights, dtype=float)
        return cls(d, canonical, semantics=semantics, pair_weights=weights)

    @property
    def empty(self) -> bool:
        return self.d < 2 or len(self.pairs) == 0 or self.scale == 0.0

    @property
    def pair_mask(self) -> np.ndarray:
        """Compatibility alias for the existing SharedDagmaLinear code."""
        return self.mask

    def value_and_grad(
        self,
        W: np.ndarray,
        pairs: Sequence[Pair] | None = None,
        pair_weights: Sequence[float] | None = None,
        state=None,
        *,
        function: str = "exp",
        log_terms: int | None = None,
        inverse_epsilon: float = 1e-8,
    ) -> ValueGradResult:
        if pairs is not None or pair_weights is not None:
            kernel = type(self).from_pairs(
                self.pairs if pairs is None else pairs,
                self.d,
                semantics=self.semantics,
                pair_weights=self.pair_weights if pair_weights is None else pair_weights,
            )
            return kernel.value_and_grad(
                W, function=function, log_terms=log_terms,
                inverse_epsilon=inverse_epsilon)
        value, grad, diag = self._value_grad_impl(
            W, function=function, log_terms=log_terms,
            inverse_epsilon=inverse_epsilon)
        self._last = diag
        if np.isfinite(diag.condition_number):
            self.maximum_condition_number = max(self.maximum_condition_number, float(diag.condition_number))
        if not diag.domain_valid or diag.numerical_status != "ok":
            self.solve_failures += 1
        return ValueGradResult(value, grad, diag)

    def value_grad(
        self,
        W: np.ndarray,
        function: str = "exp",
        *,
        log_terms: int | None = None,
        inverse_epsilon: float = 1e-8,
    ) -> tuple[float, np.ndarray]:
        result = self.value_and_grad(
            W, function=function, log_terms=log_terms,
            inverse_epsilon=inverse_epsilon)
        return result.penalty_value, result.gradient_W

    def diagnostics(self) -> dict[str, object]:
        return {} if self._last is None else dict(self._last.__dict__)

    def _empty_result(self, W: np.ndarray, name: str, function: str) -> tuple[float, np.ndarray, KernelDiagnostics]:
        return 0.0, np.zeros_like(W, dtype=float), KernelDiagnostics(
            name=name, function=function, implementation_class=type(self).__name__,
            full_matrix_materialized=False, selected_column_count=0)

    def _value_grad_impl(self, W: np.ndarray, *, function: str, log_terms: int | None, inverse_epsilon: float):
        raise NotImplementedError


def _path_matrix(A: np.ndarray, function: str, log_terms: int, inverse_epsilon: float) -> np.ndarray:
    d = A.shape[0]
    eye = np.eye(d, dtype=A.dtype)
    if function == "exp":
        return sla.expm(A)
    if function == "inv":
        return sla.solve(eye - A + inverse_epsilon * eye, eye, check_finite=False)
    if function == "log":
        F = eye.copy()
        power = eye.copy()
        for k in range(1, log_terms + 1):
            power = power @ A
            F += power / k
        return F
    if function == "binom":
        return np.linalg.matrix_power(eye + A, d)
    raise ValueError("trek_function must be one of: exp, log, inv, binom")


def _power_adjoint(B: np.ndarray, G: np.ndarray, degree: int, coefficient: float = 1.0) -> np.ndarray:
    result = np.zeros_like(B)
    powers = [np.eye(B.shape[0], dtype=B.dtype)]
    for _ in range(degree):
        powers.append(powers[-1] @ B)
    for r in range(degree):
        result += powers[r].T @ G @ powers[degree - 1 - r].T
    return coefficient * result


class DenseCurrentNoTreksKernel(BaseNoTreksKernel):
    """Dense implementation equivalent to the current repository kernel."""

    name = "dense_current"

    def _value_grad_impl(self, W: np.ndarray, *, function: str, log_terms: int | None, inverse_epsilon: float):
        started = time.perf_counter()
        W = np.asarray(W, dtype=float)
        if W.shape != (self.d, self.d):
            raise ValueError(f"W must have shape {(self.d, self.d)}")
        if self.empty:
            return self._empty_result(W, self.name, function)
        K = 2 * self.d if log_terms is None else int(log_terms)
        if K < 1 or inverse_epsilon < 0:
            raise ValueError("log_terms must be positive and inverse_epsilon non-negative")
        A = W * W
        matrix_function_evaluations = 1
        factorizations = 0
        solves = 0
        condition = math.nan
        if function == "inv":
            system = np.eye(self.d) - A + inverse_epsilon * np.eye(self.d)
            condition = float(np.linalg.cond(system))
            if not np.isfinite(condition) or condition > 1e12:
                diag = KernelDiagnostics(
                    self.name, function, type(self).__name__,
                    domain_valid=False, numerical_status=f"ill-conditioned cond={condition:.3e}")
                self._last = diag
                raise np.linalg.LinAlgError(diag.numerical_status)
            factorizations = 1
            solves = 1
            F = sla.solve(system, np.eye(self.d), check_finite=False)
        else:
            F = _path_matrix(A, function, K, inverse_epsilon)
        forward_done = time.perf_counter()
        Gf = self.scale * (F @ self.mask)
        value = 0.5 * float(np.sum(self.mask * (F.T @ F))) * self.scale
        if function == "exp":
            Ga = sla.expm_frechet(A.T, Gf, compute_expm=False)
        elif function == "inv":
            Ga = F.T @ Gf @ F.T
        elif function == "log":
            Ga = np.zeros_like(A)
            for k in range(1, K + 1):
                Ga += _power_adjoint(A, Gf, k, 1.0 / k)
        else:
            Ga = _power_adjoint(np.eye(self.d) + A, Gf, self.d)
        grad = 2.0 * W * Ga
        finished = time.perf_counter()
        diag = KernelDiagnostics(
            name=self.name,
            function=function,
            implementation_class=type(self).__name__,
            forward_time_seconds=forward_done - started,
            backward_time_seconds=finished - forward_done,
            total_kernel_time_seconds=finished - started,
            matrix_factorizations=factorizations,
            matrix_function_evaluations=matrix_function_evaluations,
            linear_solves=solves,
            right_hand_sides=self.d if solves else 0,
            dense_matrix_multiplications=4,
            selected_column_count=self.d,
            full_matrix_materialized=True,
            condition_number=condition,
        )
        return value, grad, diag


class SelectedInverseNoTreksKernel(BaseNoTreksKernel):
    """Selected-column inverse PST kernel equivalent for function='inv'."""

    name = "selected_inv"

    def _value_grad_impl(self, W: np.ndarray, *, function: str, log_terms: int | None, inverse_epsilon: float):
        if function != "inv":
            raise ValueError("selected_inv only implements trek_function='inv'")
        started = time.perf_counter()
        W = np.asarray(W, dtype=float)
        if W.shape != (self.d, self.d):
            raise ValueError(f"W must have shape {(self.d, self.d)}")
        if self.empty:
            return self._empty_result(W, self.name, function)
        A = W * W
        alpha = 1.0 + inverse_epsilon
        system = alpha * np.eye(self.d) - A
        rho = float(max(abs(np.linalg.eigvals(A)), default=0.0))
        margin = alpha - rho
        if not np.isfinite(rho) or margin <= 0:
            diag = KernelDiagnostics(
                self.name, function, type(self).__name__, domain_valid=False,
                numerical_status=f"outside domain rho={rho:.6g} alpha={alpha:.6g}",
                spectral_radius=rho, domain_margin=margin)
            self._last = diag
            raise np.linalg.LinAlgError(diag.numerical_status)
        condition = float(np.linalg.cond(system))
        if not np.isfinite(condition) or condition > 1e12:
            diag = KernelDiagnostics(
                self.name, function, type(self).__name__, domain_valid=False,
                numerical_status=f"ill-conditioned cond={condition:.3e}",
                condition_number=condition, spectral_radius=rho, domain_margin=margin)
            self._last = diag
            raise np.linalg.LinAlgError(diag.numerical_status)
        lu, piv = sla.lu_factor(system, check_finite=False)
        E = np.eye(self.d)[:, self.nodes]
        Y = sla.lu_solve((lu, piv), E, check_finite=False)
        forward_done = time.perf_counter()
        node_pos = {int(node): idx for idx, node in enumerate(self.nodes)}
        B = np.zeros((len(self.nodes), len(self.nodes)), dtype=float)
        weights = (np.ones(len(self.pairs), dtype=float) if self.pair_weights is None
                   else np.asarray(self.pair_weights, dtype=float))
        for (i, j), weight in zip(self.pairs, weights):
            a, b = node_pos[int(i)], node_pos[int(j)]
            B[a, b] += weight
            if self.semantics == "unordered":
                B[b, a] += weight
        gram = Y.T @ Y
        value = 0.5 * self.scale * float(np.sum(B * gram))
        GY = self.scale * (Y @ B)
        Z = sla.lu_solve((lu, piv), GY, trans=1, check_finite=False)
        GX = Z @ Y.T
        grad = 2.0 * W * GX
        finished = time.perf_counter()
        diag = KernelDiagnostics(
            name=self.name,
            function=function,
            implementation_class=type(self).__name__,
            forward_time_seconds=forward_done - started,
            backward_time_seconds=finished - forward_done,
            total_kernel_time_seconds=finished - started,
            matrix_factorizations=1,
            matrix_function_evaluations=0,
            linear_solves=2,
            right_hand_sides=2 * len(self.nodes),
            dense_matrix_multiplications=3,
            selected_column_count=len(self.nodes),
            full_matrix_materialized=False,
            condition_number=condition,
            spectral_radius=rho,
            domain_margin=margin,
        )
        return value, grad, diag


class PolynomialSelectedNoTreksKernel(BaseNoTreksKernel):
    """Selected-column finite polynomial PST kernel with reverse recurrence."""

    name = "poly_selected_walk"

    coefficients: tuple[float, ...] | None = None

    def _coefficients(self, function: str, log_terms: int | None) -> np.ndarray:
        degree = self.d - 1 if log_terms is None else int(log_terms)
        if degree < 0:
            raise ValueError("polynomial degree must be non-negative")
        if function in {"poly_walk", "walk"}:
            return np.ones(degree + 1, dtype=float)
        if function in {"poly_exp", "truncated_exp"}:
            coeffs = np.empty(degree + 1, dtype=float)
            coeffs[0] = 1.0
            factorial = 1.0
            for k in range(1, degree + 1):
                factorial *= k
                coeffs[k] = 1.0 / factorial
            return coeffs
        raise ValueError("polynomial selected kernel supports poly_walk/walk/poly_exp/truncated_exp")

    def _value_grad_impl(self, W: np.ndarray, *, function: str, log_terms: int | None, inverse_epsilon: float):
        started = time.perf_counter()
        W = np.asarray(W, dtype=float)
        if W.shape != (self.d, self.d):
            raise ValueError(f"W must have shape {(self.d, self.d)}")
        if self.empty:
            return self._empty_result(W, self.name, function)
        coeffs = self._coefficients(function, log_terms)
        X = W * W
        E = np.eye(self.d)[:, self.nodes]
        powers = [E]
        Y = coeffs[0] * E
        for k in range(1, len(coeffs)):
            powers.append(X @ powers[-1])
            Y = Y + coeffs[k] * powers[-1]
        forward_done = time.perf_counter()
        node_pos = {int(node): idx for idx, node in enumerate(self.nodes)}
        B = np.zeros((len(self.nodes), len(self.nodes)), dtype=float)
        weights = (np.ones(len(self.pairs), dtype=float) if self.pair_weights is None
                   else np.asarray(self.pair_weights, dtype=float))
        for (i, j), weight in zip(self.pairs, weights):
            a, b = node_pos[int(i)], node_pos[int(j)]
            B[a, b] += weight
            if self.semantics == "unordered":
                B[b, a] += weight
        value = 0.5 * self.scale * float(np.sum(B * (Y.T @ Y)))
        GY = self.scale * (Y @ B)
        GX = np.zeros_like(X)
        adj = np.zeros_like(E)
        for k in range(len(coeffs) - 1, 0, -1):
            total = adj + coeffs[k] * GY
            GX += total @ powers[k - 1].T
            adj = X.T @ total
        grad = 2.0 * W * GX
        finished = time.perf_counter()
        diag = KernelDiagnostics(
            name=self.name,
            function=function,
            implementation_class=type(self).__name__,
            forward_time_seconds=forward_done - started,
            backward_time_seconds=finished - forward_done,
            total_kernel_time_seconds=finished - started,
            dense_matrix_multiplications=2 * max(0, len(coeffs) - 1) + 2,
            selected_column_count=len(self.nodes),
            full_matrix_materialized=False,
        )
        return value, grad, diag


KERNEL_REGISTRY = {
    "notreks_reference": DenseCurrentNoTreksKernel,
    "dense_current": DenseCurrentNoTreksKernel,
    "dense_exp": DenseCurrentNoTreksKernel,
    "dense_inv": DenseCurrentNoTreksKernel,
    "selected_inv": SelectedInverseNoTreksKernel,
    "poly_selected_walk": PolynomialSelectedNoTreksKernel,
    "poly_selected_exp": PolynomialSelectedNoTreksKernel,
}


def make_notreks_kernel(
    name: str,
    pairs: Sequence[Pair],
    d: int,
    *,
    semantics: PairSemantics = "unordered",
    pair_weights: Sequence[float] | None = None,
) -> BaseNoTreksKernel:
    """Instantiate a named benchmarkable NOTREKS kernel."""
    if name not in KERNEL_REGISTRY:
        raise ValueError(
            f"unknown NOTREKS kernel {name!r}; expected one of {sorted(KERNEL_REGISTRY)}")
    return KERNEL_REGISTRY[name].from_pairs(
        pairs, d, semantics=semantics, pair_weights=pair_weights)


def notreks_value_grad_kernel(
    W: np.ndarray,
    pairs: Sequence[Pair],
    function: str = "exp",
    *,
    kernel_name: str = "notreks_reference",
    log_terms: int | None = None,
    inverse_epsilon: float = 1e-8,
) -> tuple[float, np.ndarray, KernelDiagnostics]:
    """Convenience wrapper returning value, gradient and diagnostics."""
    kernel = make_notreks_kernel(kernel_name, pairs, np.asarray(W).shape[0])
    result = kernel.value_and_grad(
        W, function=function, log_terms=log_terms,
        inverse_epsilon=inverse_epsilon)
    return result.penalty_value, result.gradient_W, result.diagnostics
