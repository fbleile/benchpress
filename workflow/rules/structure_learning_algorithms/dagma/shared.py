"""Shared linear DAGMA optimizer with an optional PST/NOTREKS penalty.

The central-path loop is a small adaptation of ``dagma==1.1.1`` (Apache-2.0),
https://github.com/kevinsbello/dagma, inspected at commit
088616885d71b56c0573cd4902c1fcbac02e649f.
"""
from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np
import scipy.linalg as sla
from dagma.linear import DagmaLinear

from .inverse_structural import (
    InverseStructuralKernel, exact_spectral_radius, solve_inverse_resolvent,
)
from .structural import feasibility_thresholds, support_diagnostics


Pair = tuple[int, int]


def _validate_pairs(pairs: Sequence[Pair], d: int) -> np.ndarray:
    out = np.asarray(pairs, dtype=int)
    if out.size == 0:
        return np.empty((0, 2), dtype=int)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError("no-trek pairs must have shape (m, 2)")
    if np.any(out < 0) or np.any(out >= d) or np.any(out[:, 0] == out[:, 1]):
        raise ValueError("no-trek pair indices must be distinct and in range")
    out.sort(axis=1)
    if len({tuple(x) for x in out}) != len(out):
        raise ValueError("duplicate no-trek pairs")
    return out


def _path_matrix(A: np.ndarray, function: str, log_terms: int, inverse_epsilon: float) -> np.ndarray:
    d = A.shape[0]
    eye = np.eye(d, dtype=A.dtype)
    if function == "exp":
        return sla.expm(A)
    if function == "inv":
        return sla.solve(eye - A + inverse_epsilon * eye, eye)
    if function == "log":
        F, power = eye.copy(), eye.copy()
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


@dataclass
class NoTreksKernel:
    """Cached pair data and exact NumPy/SciPy PST value-gradient kernel."""

    d: int
    pairs: np.ndarray
    pair_mask: np.ndarray
    scale: float
    empty: bool
    maximum_condition_number: float = 0.0
    solve_failures: int = 0

    @classmethod
    def from_pairs(cls, pairs: Sequence[Pair], d: int) -> "NoTreksKernel":
        canonical = _validate_pairs(pairs, d)
        mask = np.zeros((d, d), dtype=float)
        if len(canonical):
            mask[canonical[:, 0], canonical[:, 1]] = 1.0
            mask[canonical[:, 1], canonical[:, 0]] = 1.0
        mask.setflags(write=False)
        canonical.setflags(write=False)
        return cls(d, canonical, mask, 2.0 / (d - 1) if d > 1 else 0.0,
                   d < 2 or not len(canonical))

    def value_grad(
        self,
        W: np.ndarray,
        function: str = "exp",
        *,
        log_terms: int | None = None,
        inverse_epsilon: float = 1e-8,
    ) -> tuple[float, np.ndarray]:
        W = np.asarray(W, dtype=float)
        if W.shape != (self.d, self.d):
            raise ValueError(f"W must have shape {(self.d, self.d)}")
        if self.empty:
            return 0.0, np.zeros_like(W)
        K = 2 * self.d if log_terms is None else int(log_terms)
        if K < 1 or inverse_epsilon < 0:
            raise ValueError("log_terms must be positive and inverse_epsilon non-negative")
        A = W * W
        if function == "inv":
            try:
                F, condition, _ = solve_inverse_resolvent(
                    A, 1.0 + inverse_epsilon, check_condition=True)
            except np.linalg.LinAlgError:
                self.solve_failures += 1
                raise
            self.maximum_condition_number = max(
                self.maximum_condition_number, float(condition))
        else:
            F = _path_matrix(A, function, K, inverse_epsilon)
        Gf = self.scale * (F @ self.pair_mask)
        # scale/2 * <B, F.T F>; B is symmetric and has two entries per pair.
        value = 0.5 * float(np.sum(self.pair_mask * (F.T @ F))) * self.scale
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
        return value, 2.0 * W * Ga


def notreks_value_grad(
    W: np.ndarray,
    pairs: Sequence[Pair],
    function: str = "exp",
    *,
    log_terms: int | None = None,
    inverse_epsilon: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Return R_I(W) and its gradient, including the exact 2/(d-1) scale."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    d = W.shape[0]
    return NoTreksKernel.from_pairs(pairs, d).value_grad(
        W, function, log_terms=log_terms, inverse_epsilon=inverse_epsilon)


@dataclass
class FitDiagnostics:
    raw_notreks_penalty: float
    scaled_notreks_penalty: float
    score: float
    h: float


class SharedDagmaLinear(DagmaLinear):
    """Official linear DAGMA loop with one additive, optional NOTREKS gradient."""

    def __init__(self, loss_type: str, verbose: bool = False,
                 dtype: type = np.float64):
        if loss_type == "gaussian_profile":
            super().__init__("l2", verbose=verbose, dtype=dtype)
            self.loss_type = loss_type
        else:
            super().__init__(loss_type, verbose=verbose, dtype=dtype)

    def _score(self, W):
        if self.loss_type != "gaussian_profile":
            return super()._score(W)
        # q_j = Sigma_jj - 2 Sigma_:j^T W_:j + W_:j^T Sigma W_:j.
        SW = self.cov @ W
        q = (np.diag(self.cov) - 2.0 * np.sum(self.cov * W, axis=0)
             + np.sum(W * SW, axis=0))
        denominator = q + self.variance_epsilon
        if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
            raise FloatingPointError("non-positive Gaussian profile residual variance")
        value = 0.5 * float(np.log(denominator).sum())
        gradient = (SW - self.cov) / denominator[np.newaxis, :]
        return value, gradient

    def _h(self, W, s=1.0):
        if getattr(self, "dag_constraint", "logdet") == "inverse_trace":
            result = self._inverse_structural_kernel.evaluate(W)
            return result.inverse_dag_value, result.inverse_dag_gradient
        if getattr(self, "dag_penalty_weight", 1.0) != 0.0:
            return super()._h(W, s)
        # Report the raw log-determinant value, but do not evaluate its
        # gradient or impose its domain when the term is disabled.
        M = s * self.Id - W * W
        h = -np.linalg.slogdet(M)[1] + self.d * np.log(s)
        return float(h), np.zeros_like(W)

    def minimize(self, W, mu, max_iter, s, lr, tol=1e-6, beta_1=.99, beta_2=.999, pbar=None):
        stage_started = time.perf_counter()
        failures_before = self._trek_kernel.solve_failures
        profile = getattr(self, "profile_components", False)
        component_times = {
            "dagma_score_gradient_seconds": 0.0,
            "dagma_h_gradient_seconds": 0.0,
            "notreks_value_gradient_seconds": 0.0,
            "shared_inverse_structural_seconds": 0.0,
            "optimizer_update_seconds": 0.0,
            "diagnostics_logging_seconds": 0.0,
        }
        inverse_counters_before = {
            "calls": self._inverse_structural_kernel.calls,
            "factorizations": self._inverse_structural_kernel.matrix_factorizations,
            "solves":
                self._inverse_structural_kernel.matrix_inverse_or_solve_calls,
            "multiplications":
                self._inverse_structural_kernel.matrix_multiplications,
            "failures": self._inverse_structural_kernel.inverse_failures,
        }
        domain_rejections_before = self.domain_rejections
        backtracking_before = self.backtracking_steps
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        mask_exc = np.ones((self.d, self.d), dtype=self.dtype)
        if (not self.dag_penalty_weight
                or self.dag_constraint == "inverse_trace"):
            # A weighted adjacency never admits self-loops. The log-det term
            # ordinarily drives these entries to zero; the no-h ablation must
            # retain the structural diagonal mask explicitly.
            np.fill_diagonal(mask_exc, 0.0)
        if self.exc_c is not None:
            mask_exc[self.exc_r, self.exc_c] = 0.
        mask_inc = np.zeros((self.d, self.d))
        if self.inc_c is not None:
            mask_inc[self.inc_r, self.inc_c] = -2 * mu * self.lambda1
        grad = np.zeros_like(W)
        stopped_by_tolerance = False
        zero_stage = bool(mu == 0.0 and getattr(
            self, "terminal_zero_stage", False))
        termination_reason = "maximum iterations"
        iteration = 0
        for iteration in range(1, int(max_iter) + 1):
            inverse_result = None
            if self.dag_constraint == "inverse_trace":
                started = time.perf_counter()
                try:
                    inverse_result = self._inverse_structural_kernel.evaluate(W)
                except np.linalg.LinAlgError:
                    return W, False
                if profile:
                    component_times[
                        "shared_inverse_structural_seconds"
                    ] += time.perf_counter() - started
                M = np.zeros_like(W)
            elif self.dag_penalty_weight:
                started = time.perf_counter()
                M = sla.inv(s * self.Id - W * W) + 1e-16
                if profile:
                    component_times["dagma_h_gradient_seconds"] += time.perf_counter() - started
                while np.any(M < 0):
                    if iteration == 1 or s <= .9:
                        return W, False
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, True
                    W -= lr * grad
                    M = sla.inv(s * self.Id - W * W) + 1e-16
            else:
                # h_s and its M-matrix domain are absent from this ablation.
                M = np.zeros_like(W)
            started = time.perf_counter()
            _, score_gradient = self._score(W)
            G_score = mu * score_gradient
            if profile:
                component_times["dagma_score_gradient_seconds"] += time.perf_counter() - started
            if self.dag_constraint == "inverse_trace":
                G_h = self.gamma_inv * inverse_result.inverse_dag_gradient
                G_nt = (
                    inverse_result.notreks_gradient
                    if self.trek_weight != 0.0 and len(self.no_trek_pairs)
                    else np.zeros_like(W))
            elif self.trek_weight == 0.0 or not len(self.no_trek_pairs):
                G_nt = np.zeros_like(W)
            else:
                started = time.perf_counter()
                _, G_nt = self._trek_kernel.value_grad(
                    W, self.trek_function,
                    log_terms=self.trek_log_terms,
                    inverse_epsilon=self.trek_inverse_epsilon,
                )
                if profile:
                    component_times["notreks_value_gradient_seconds"] += time.perf_counter() - started
            G_l1 = mu * self.lambda1 * np.sign(W)
            if self.dag_constraint != "inverse_trace":
                G_h = self.dag_penalty_weight * 2 * W * M.T
            G_structural = G_h + self.trek_weight * G_nt
            Gobj = (G_score + G_l1 + G_h
                    + mask_inc * np.sign(W) + self.trek_weight * G_nt)
            started = time.perf_counter()
            grad = self._adam_update(Gobj, iteration, beta_1, beta_2)
            if self.dag_constraint == "inverse_trace":
                step_lr = lr
                while True:
                    proposal = (W - step_lr * grad) * mask_exc
                    inside, _ = self._inverse_structural_kernel.in_domain(
                        proposal)
                    if inside:
                        W = proposal
                        lr = step_lr
                        break
                    self.domain_rejections += 1
                    self.backtracking_steps += 1
                    step_lr *= .5
                    if step_lr <= 1e-16:
                        return W, False
            else:
                W -= lr * grad
                W *= mask_exc
            if profile:
                component_times["optimizer_update_seconds"] += time.perf_counter() - started
            diagnostic_interval = (
                self.zero_block_iterations if zero_stage else self.checkpoint)
            if iteration % diagnostic_interval == 0 or iteration == int(max_iter):
                started = time.perf_counter()
                score, _ = self._score(W)
                if self.dag_constraint == "inverse_trace":
                    checkpoint_inverse = (
                        self._inverse_structural_kernel.evaluate(
                            W, diagnostics=True))
                    h = checkpoint_inverse.inverse_dag_value
                    nt = checkpoint_inverse.notreks_value
                else:
                    h = self._h(W, s)[0] if self.dag_penalty_weight else 0.0
                    nt, _ = self._trek_kernel.value_grad(
                        W, self.trek_function,
                        log_terms=self.trek_log_terms,
                        inverse_epsilon=self.trek_inverse_epsilon,
                    )
                structural_weight = (
                    self.gamma_inv if self.dag_constraint == "inverse_trace"
                    else self.dag_penalty_weight)
                obj_new = (mu * (score + self.lambda1 * np.abs(W).sum())
                           + structural_weight * h
                           + self.trek_weight * nt)
                if profile:
                    component_times["diagnostics_logging_seconds"] += time.perf_counter() - started
                if zero_stage:
                    feasibility = feasibility_thresholds(
                        W, self.no_trek_pairs)
                    structural = support_diagnostics(W, self.no_trek_pairs)
                    absolute = np.abs(W)
                    off_diagonal = absolute[
                        ~np.eye(self.d, dtype=bool)]
                    nonzero = off_diagonal[off_diagonal != 0]
                    quantiles = (
                        np.quantile(nonzero, [
                            0., .01, .05, .25, .5, .75, .95, .99, 1.])
                        if len(nonzero) else np.zeros(9))
                    structural_gradient_norm = float(
                        np.linalg.norm(G_structural))
                    trajectory = {
                        "stage": len(self.stage_diagnostics) + 1,
                        "zero_stage_iteration": int(iteration),
                        "mu": 0.0, "s": float(s), "h": float(h),
                        "notreks_value": float(nt),
                        "h_gradient_norm": float(np.linalg.norm(G_h)),
                        "notreks_gradient_norm": float(np.linalg.norm(
                            self.trek_weight * G_nt)),
                        "structural_gradient_norm": structural_gradient_norm,
                        "data_score": float(score),
                        "l1_norm": float(np.abs(W).sum()),
                        "maximum_absolute_weight": float(
                            absolute.max(initial=0.)),
                        "minimum_nonzero_absolute_weight": float(
                            nonzero.min()) if len(nonzero) else 0.,
                        "exact_nonzero_count": int(len(nonzero)),
                        "tau_dag": feasibility["tau_dag"],
                        "tau_mi": feasibility["tau_mi"],
                        "tau_feas": feasibility["tau_feas"],
                        **structural,
                    }
                    for q, value in zip(
                            ("q00", "q01", "q05", "q25", "q50", "q75",
                             "q95", "q99", "q100"), quantiles):
                        trajectory[f"absolute_weight_{q}"] = float(value)
                    self.zero_stage_trajectory.append(trajectory)
                    nt_ok = (not len(self.no_trek_pairs)
                             or nt <= self.notreks_tolerance)
                    converged = (
                        h <= self.h_tolerance and nt_ok
                        and feasibility["tau_feas"]
                        <= self.feasibility_threshold_tolerance
                        and structural_gradient_norm
                        <= self.gradient_tolerance)
                    if converged:
                        stopped_by_tolerance = True
                        termination_reason = (
                            "literal support feasible"
                            if structural["raw_literal_support_is_dag"]
                            and structural["raw_oracle_violations"] == 0
                            else "numerically feasible")
                        break
                elif abs((obj_prev - obj_new) / obj_prev) <= tol:
                    stopped_by_tolerance = True
                    termination_reason = "relative objective tolerance"
                    if pbar is not None:
                        pbar.update(int(max_iter) - iteration + 1)
                    break
                obj_prev = obj_new
            if pbar is not None:
                pbar.update(1)
        score, _ = self._score(W)
        if self.dag_constraint == "inverse_trace":
            final_inverse = self._inverse_structural_kernel.evaluate(
                W, diagnostics=True)
            h = final_inverse.inverse_dag_value
            h_gradient = self.gamma_inv * final_inverse.inverse_dag_gradient
            nt = final_inverse.notreks_value
            nt_gradient = final_inverse.notreks_gradient
            condition_number = final_inverse.condition_number
            spectral_radius = exact_spectral_radius(W * W)
            minimum_inverse_entry = final_inverse.minimum_inverse_entry
        else:
            h, h_gradient = self._h(W, s)
            nt, nt_gradient = self._trek_kernel.value_grad(
                W, self.trek_function,
                log_terms=self.trek_log_terms,
                inverse_epsilon=self.trek_inverse_epsilon,
            )
            condition_number = self._trek_kernel.maximum_condition_number
            spectral_radius = exact_spectral_radius(W * W)
            minimum_inverse_entry = np.nan
        absolute = np.abs(W)
        nonzero = absolute[absolute > 0]
        self.stage_diagnostics.append({
            "stage": len(self.stage_diagnostics) + 1,
            "mu": float(mu), "s": float(s),
            "iterations_performed": int(iteration),
            "stopped_by_tolerance": bool(stopped_by_tolerance),
            "exhausted_budget": bool(not stopped_by_tolerance and iteration == int(max_iter)),
            "stage_runtime_seconds": float(time.perf_counter() - stage_started),
            "mean_seconds_per_iteration":
                float((time.perf_counter() - stage_started) / max(iteration, 1)),
            "maximum_inverse_condition_number": float(
                condition_number),
            "inverse_solve_failures": int(
                (self._inverse_structural_kernel.inverse_failures
                 - inverse_counters_before["failures"])
                if self.dag_constraint == "inverse_trace"
                else self._trek_kernel.solve_failures - failures_before),
            "inverse_stage_retries": 0,
            "spectral_radius_A": float(spectral_radius),
            "alpha_minus_spectral_radius": float(
                self._inverse_structural_kernel.alpha - spectral_radius)
                if self.dag_constraint == "inverse_trace" else np.nan,
            "minimum_inverse_entry": float(minimum_inverse_entry),
            "domain_rejections": int(
                self.domain_rejections - domain_rejections_before),
            "backtracking_steps": int(
                self.backtracking_steps - backtracking_before),
            "matrix_factorizations": int(
                self._inverse_structural_kernel.matrix_factorizations
                - inverse_counters_before["factorizations"]),
            "matrix_inverse_or_solve_calls": int(
                self._inverse_structural_kernel.matrix_inverse_or_solve_calls
                - inverse_counters_before["solves"]),
            "matrix_multiplications": int(
                self._inverse_structural_kernel.matrix_multiplications
                - inverse_counters_before["multiplications"]),
            "inverse_structural_calls": int(
                self._inverse_structural_kernel.calls
                - inverse_counters_before["calls"]),
            "termination_reason": termination_reason,
            "score_gradient_contribution_norm": float(
                np.linalg.norm(mu * self._score(W)[1])),
            "l1_gradient_contribution_norm": float(
                np.linalg.norm(mu * self.lambda1 * np.sign(W))),
            "h_gradient_norm": float(np.linalg.norm(
                h_gradient)) if (
                    self.gamma_inv if self.dag_constraint == "inverse_trace"
                    else self.dag_penalty_weight) else 0.0,
            "notreks_gradient_norm": float(np.linalg.norm(
                self.trek_weight * nt_gradient)),
            **component_times,
            "score": float(score), "h": float(h),
            "raw_notreks_value": float(nt / (2.0 / (W.shape[0] - 1)))
                if W.shape[0] > 1 else 0.0,
            "scaled_notreks_contribution": float(self.trek_weight * nt),
            "maximum_absolute_weight": float(absolute.max(initial=0.0)),
            "median_nonzero_absolute_weight": float(np.median(nonzero)) if nonzero.size else 0.0,
            **{f"number_weights_above_{threshold}": int(np.sum(absolute >= threshold))
               for threshold in (0.05, 0.1, 0.2, 0.3)},
        })
        self.stage_adjacencies.append(W.copy())
        return W, True

    def fit(self, X, *, no_trek_pairs=(), trek_weight=0.0, trek_function="inv",
            dag_penalty_weight=1.0, trek_log_terms=None,
            trek_inverse_epsilon=1e-8, variance_epsilon=1e-8,
            dag_constraint="logdet", gamma_inv=1.0,
            trek_kernel="fast",
            initial_W=None, mu_schedule=None, terminal_zero_stage=False,
            zero_block_iterations=10000, maximum_zero_iterations=300000,
            h_tolerance=1e-12, notreks_tolerance=1e-12,
            feasibility_threshold_tolerance=1e-6,
            gradient_tolerance=1e-8, **kwargs):
        self.no_trek_pairs = _validate_pairs(no_trek_pairs, np.asarray(X).shape[1])
        self.trek_kernel_name = str(trek_kernel)
        if self.trek_kernel_name == "notreks_reference":
            self._trek_kernel = NoTreksKernel.from_pairs(
                self.no_trek_pairs, np.asarray(X).shape[1])
        else:
            from workflow.rules.structure_learning_algorithms.notreks import (
                make_notreks_kernel,
            )

            self._trek_kernel = make_notreks_kernel(
                self.trek_kernel_name, self.no_trek_pairs, np.asarray(X).shape[1])
        if dag_constraint not in {"logdet", "inverse_trace"}:
            raise ValueError("dag_constraint must be logdet or inverse_trace")
        self.dag_constraint = dag_constraint
        self.gamma_inv = float(gamma_inv)
        if self.gamma_inv < 0:
            raise ValueError("gamma_inv must be non-negative")
        self.trek_weight = float(trek_weight)
        self.dag_penalty_weight = float(dag_penalty_weight)
        if self.dag_penalty_weight < 0:
            raise ValueError("dag_penalty_weight must be non-negative")
        self.trek_function = trek_function
        self.trek_log_terms = 2 * np.asarray(X).shape[1] if trek_log_terms is None else int(trek_log_terms)
        self.trek_inverse_epsilon = float(trek_inverse_epsilon)
        self._inverse_structural_kernel = InverseStructuralKernel.from_pair_mask(
            self._trek_kernel.pair_mask, self._trek_kernel.scale,
            self.trek_inverse_epsilon)
        self.domain_rejections = 0
        self.backtracking_steps = 0
        self.variance_epsilon = float(variance_epsilon)
        if self.variance_epsilon <= 0:
            raise ValueError("variance_epsilon must be positive")
        self.stage_diagnostics = []
        self.stage_adjacencies = []
        self.zero_stage_trajectory = []
        validated_mu_schedule = None
        if mu_schedule is not None:
            schedule = np.asarray(mu_schedule, dtype=float)
            if (schedule.ndim != 1 or not len(schedule)
                    or not np.all(np.isfinite(schedule))
                    or np.any(schedule < 0)):
                raise ValueError(
                    "mu_schedule must be a nonempty finite nonnegative list")
            if np.any(schedule[:-1] == 0):
                raise ValueError("zero is allowed only as the final mu value")
            validated_mu_schedule = schedule.tolist()
        self.terminal_zero_stage = bool(
            terminal_zero_stage or (
                validated_mu_schedule is not None
                and validated_mu_schedule[-1] == 0.0))
        self.zero_block_iterations = int(zero_block_iterations)
        self.maximum_zero_iterations = int(maximum_zero_iterations)
        self.h_tolerance = float(h_tolerance)
        self.notreks_tolerance = float(notreks_tolerance)
        self.feasibility_threshold_tolerance = float(
            feasibility_threshold_tolerance)
        self.gradient_tolerance = float(gradient_tolerance)
        if self.zero_block_iterations < 1 or self.maximum_zero_iterations < 1:
            raise ValueError("zero-stage iteration limits must be positive")
        mu_schedule = validated_mu_schedule
        if self.loss_type == "gaussian_profile" and initial_W is None:
            initial_W = np.zeros((np.asarray(X).shape[1],) * 2, dtype=self.dtype)
        if initial_W is None and mu_schedule is None:
            return super().fit(X, **kwargs)
        if initial_W is None:
            initial_W = np.zeros((np.asarray(X).shape[1],) * 2, dtype=self.dtype)
        return self._fit_initialized(
            np.asarray(X), np.asarray(initial_W),
            mu_schedule=mu_schedule, **kwargs)

    def _fit_initialized(self, X, initial_W, lambda1=.03, w_threshold=.3, T=5,
                         mu_init=1., mu_factor=.1, s=(1., .9, .8, .7, .6),
                         warm_iter=30000, max_iter=60000, lr=.0003,
                         checkpoint=1000, beta_1=.99, beta_2=.999,
                         exclude_edges=None, include_edges=None,
                         mu_schedule=None):
        """Run the upstream central path from a caller-supplied valid W0."""
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        if self.loss_type in {"l2", "gaussian_profile"}:
            self.X -= X.mean(axis=0, keepdims=True)
        self.exc_r, self.exc_c = None, None
        self.inc_r, self.inc_c = None, None
        if exclude_edges:
            self.exc_r, self.exc_c = zip(*exclude_edges)
        if include_edges:
            self.inc_r, self.inc_c = zip(*include_edges)
        self.cov = self.X.T @ self.X / float(self.n)
        if initial_W.shape != (self.d, self.d):
            raise ValueError(f"initial_W must have shape {(self.d, self.d)}")
        self.W_est = initial_W.astype(self.dtype, copy=True)
        np.fill_diagonal(self.W_est, 0.)
        mus = ([float(mu_init) * float(mu_factor) ** stage
                for stage in range(int(T))]
               if mu_schedule is None else list(mu_schedule))
        T = len(mus)
        schedule = list(s) if not np.isscalar(s) else [float(s)] * T
        if len(schedule) < T:
            schedule += [schedule[-1]] * (T - len(schedule))
        for stage, mu in enumerate(mus):
            success, lr_adam = False, lr
            if mu == 0.0 and self.terminal_zero_stage:
                budget = self.maximum_zero_iterations
            else:
                last_positive = max(i for i, value in enumerate(mus)
                                    if value > 0)
                budget = int(max_iter) if stage == last_positive else int(warm_iter)
            while not success:
                candidate, success = self.minimize(
                    self.W_est.copy(), mu, budget, schedule[stage], lr_adam,
                    beta_1=beta_1, beta_2=beta_2)
                if not success:
                    lr_adam *= .5
                    if mu == 0.0 and self.terminal_zero_stage:
                        # The explicit structural stage must retain its
                        # configured log-det domain. Retry with a smaller step
                        # instead of silently changing s or accepting a stage
                        # that produced no diagnostics.
                        if lr_adam <= 1e-16:
                            raise FloatingPointError(
                                "terminal zero stage could not remain in the "
                                f"M-matrix domain at s={schedule[stage]}")
                    else:
                        schedule[stage] += .1
            self.W_est = candidate
        self.h_final, _ = self._h(self.W_est)
        self.score_final, _ = self._score(self.W_est)
        self.W_est[np.abs(self.W_est) < w_threshold] = 0.
        return self.W_est


def deterministic_initial_adjacency(d: int, seed: int, scale: float = .05,
                                    s0: float = 1.0) -> np.ndarray:
    """Small zero-diagonal W0 scaled conservatively inside the DAGMA domain."""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=scale, size=(d, d))
    np.fill_diagonal(W, 0.)
    rho = max(abs(np.linalg.eigvals(W * W)), default=0.)
    limit = .25 * s0
    if rho > limit:
        W *= np.sqrt(limit / float(rho))
    return W
