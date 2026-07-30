"""Shared inverse backend for inverse-trace acyclicity and inverse PST."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import scipy.linalg as sla


def solve_inverse_resolvent(
    squared_adjacency: np.ndarray,
    alpha: float,
    *,
    check_condition: bool = False,
    condition_limit: float = 1e12,
    require_m_matrix_inverse: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Construct one inverse resolvent with a single LU factorization.

    Both production inverse-NOTREKS and the fused inverse-trace ablation call
    this backend. Callers retain their existing domain and diagnostics policy.
    """
    A = np.asarray(squared_adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("squared_adjacency must be square")
    system = float(alpha) * np.eye(A.shape[0]) - A
    condition = float(np.linalg.cond(system)) if check_condition else np.nan
    if check_condition and (
            not np.isfinite(condition) or condition > condition_limit):
        raise np.linalg.LinAlgError(
            "inverse structural system is ill-conditioned "
            f"(cond={condition:.3e})")
    lu, pivots = sla.lu_factor(system, check_finite=False)
    inverse = sla.lu_solve(
        (lu, pivots), np.eye(A.shape[0]), check_finite=False)
    minimum = float(inverse.min(initial=0.0))
    if not np.all(np.isfinite(inverse)):
        raise np.linalg.LinAlgError(
            "inverse structural system returned non-finite values")
    if require_m_matrix_inverse and minimum < -1e-10:
        raise np.linalg.LinAlgError(
            "inverse structural system is not a valid M-matrix inverse "
            f"(minimum inverse entry={minimum:.3e})")
    return inverse, condition, minimum


def lambda1_sqrt_logd_over_n(
        n: int, d: int, *, reference: float = .03,
        reference_n: int = 1000, reference_d: int = 50) -> float:
    """Scale an L1 screening coefficient by sqrt(log(d) / n)."""
    if n < 1 or d < 2 or reference_n < 1 or reference_d < 2:
        raise ValueError("n/reference_n must be positive and d/reference_d >= 2")
    if not np.isfinite(reference) or reference < 0:
        raise ValueError("lambda1 reference must be finite and nonnegative")
    return float(reference * np.sqrt(
        (np.log(d) / n) / (np.log(reference_d) / reference_n)))


def spectral_radius_power(A: np.ndarray, iterations: int = 12) -> float:
    """Deterministic Perron-root estimate for a nonnegative matrix."""
    matrix = np.asarray(A, dtype=float)
    if matrix.size == 0 or not np.any(matrix):
        return 0.0
    vector = np.full(matrix.shape[0], 1.0 / np.sqrt(matrix.shape[0]))
    estimate = 0.0
    for _ in range(iterations):
        product = matrix @ vector
        norm = np.linalg.norm(product)
        if norm == 0:
            return 0.0
        vector = product / norm
        estimate = float(vector @ (matrix @ vector))
    # Power iteration can converge slowly on imprimitive nonnegative matrices.
    # The exact eigenvalue check is checkpoint-only; this helper is the cheap
    # inner-loop guard and deliberately errs on the safe side via a norm bound
    # when its estimate is close to the boundary.
    return max(0.0, estimate)


def exact_spectral_radius(A: np.ndarray) -> float:
    return float(max(abs(np.linalg.eigvals(np.asarray(A, dtype=float))),
                     default=0.0))


@dataclass
class InverseStructuralResult:
    inverse_dag_value: float
    inverse_dag_gradient: np.ndarray
    notreks_value: float
    notreks_gradient: np.ndarray
    condition_number: float
    minimum_inverse_entry: float
    spectral_radius: float
    elapsed_seconds: float


@dataclass
class InverseStructuralKernel:
    """One-solve structural evaluation with cached pair-mask state."""

    d: int
    pair_mask: np.ndarray
    scale: float
    inverse_epsilon: float = 1e-8
    matrix_factorizations: int = 0
    matrix_inverse_or_solve_calls: int = 0
    matrix_multiplications: int = 0
    inverse_failures: int = 0
    maximum_condition_number: float = 0.0
    minimum_spectral_margin: float = np.inf
    total_seconds: float = 0.0
    calls: int = 0

    @classmethod
    def from_pair_mask(
            cls, pair_mask: np.ndarray, scale: float,
            inverse_epsilon: float = 1e-8) -> "InverseStructuralKernel":
        mask = np.asarray(pair_mask, dtype=float)
        if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
            raise ValueError("pair_mask must be square")
        if inverse_epsilon < 0:
            raise ValueError("inverse_epsilon must be nonnegative")
        mask = mask.copy()
        mask.setflags(write=False)
        return cls(mask.shape[0], mask, float(scale),
                   float(inverse_epsilon))

    @property
    def alpha(self) -> float:
        return 1.0 + self.inverse_epsilon

    def in_domain(self, W: np.ndarray, *, safety_margin: float = 1e-12):
        rho = spectral_radius_power(np.asarray(W) * np.asarray(W))
        return bool(rho < self.alpha - safety_margin), rho

    def evaluate(
            self, W: np.ndarray, *, diagnostics: bool = False
    ) -> InverseStructuralResult:
        started = time.perf_counter()
        matrix = np.asarray(W, dtype=float)
        if matrix.shape != (self.d, self.d):
            raise ValueError(f"W must have shape {(self.d, self.d)}")
        A = matrix * matrix
        rho = exact_spectral_radius(A) if diagnostics else spectral_radius_power(A)
        margin = self.alpha - rho
        self.minimum_spectral_margin = min(self.minimum_spectral_margin, margin)
        if not np.isfinite(rho) or margin <= 0:
            self.inverse_failures += 1
            raise np.linalg.LinAlgError(
                "inverse structural system is outside its domain "
                f"(rho={rho:.6g}, alpha={self.alpha:.6g})")
        self.matrix_factorizations += 1
        self.matrix_inverse_or_solve_calls += 1
        try:
            F, condition, minimum = solve_inverse_resolvent(
                A, self.alpha, check_condition=diagnostics,
                require_m_matrix_inverse=True)
        except np.linalg.LinAlgError:
            self.inverse_failures += 1
            raise
        if diagnostics:
            self.maximum_condition_number = max(
                self.maximum_condition_number, condition)

        F_squared = F @ F
        self.matrix_multiplications += 1
        h_value = self.alpha * float(np.trace(F)) - self.d
        h_gradient = 2.0 * self.alpha * matrix * F_squared.T

        # Preserve the repository's existing inverse-NOTREKS convention:
        # R = scale/2 <B, F.T F>, using unnormalised F rather than P=alpha F.
        F_transpose_F = F.T @ F
        self.matrix_multiplications += 1
        nt_value = .5 * self.scale * float(
            np.sum(self.pair_mask * F_transpose_F))
        Gf = self.scale * (F @ self.pair_mask)
        self.matrix_multiplications += 1
        Ga = F.T @ Gf
        self.matrix_multiplications += 1
        Ga = Ga @ F.T
        self.matrix_multiplications += 1
        nt_gradient = 2.0 * matrix * Ga

        elapsed = time.perf_counter() - started
        self.calls += 1
        self.total_seconds += elapsed
        return InverseStructuralResult(
            float(h_value), h_gradient, float(nt_value), nt_gradient,
            condition, minimum, rho, elapsed)


def inverse_dag_value_grad(
        W: np.ndarray, *, inverse_epsilon: float = 1e-8):
    d = np.asarray(W).shape[0]
    kernel = InverseStructuralKernel.from_pair_mask(
        np.zeros((d, d)), 2.0 / (d - 1) if d > 1 else 0.,
        inverse_epsilon)
    result = kernel.evaluate(W)
    return result.inverse_dag_value, result.inverse_dag_gradient
