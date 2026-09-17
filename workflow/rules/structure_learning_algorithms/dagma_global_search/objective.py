"""Exact adapter for the existing SharedDagmaLinear objective.

The adapter delegates score, log-det DAG, and NOTREKS values/gradients to the
same kernels used by SharedDagmaLinear.  It only supplies continuation
weights and never substitutes a smoothed sparsity term for the configured L1.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.shared import (
    SharedDagmaLinear,
)
from workflow.rules.structure_learning_algorithms.notreks import make_notreks_kernel
from workflow.rules.structure_learning_algorithms.dagma.inverse_structural import (
    InverseStructuralKernel,
)


@dataclass(frozen=True)
class ContinuationStage:
    alpha: float
    mu: float
    s: float
    dag_weight: float
    notreks_weight: float


class ObjectiveAdapter:
    """Term-by-term view of the exact objective used by SharedDagmaLinear."""

    def __init__(self, X, pairs, *, lambda1=.03, trek_weight=1.,
                 trek_function="inv", trek_kernel="fast", dag_weight=1.,
                 mu_init=1., mu_factor=.1, s=(1., .9, .8, .7, .6),
                 dag_constraint="logdet", trek_log_terms=None,
                 inverse_epsilon=1e-8):
        self.X = np.asarray(X, dtype=np.float64).copy()
        self.X -= self.X.mean(axis=0, keepdims=True)
        self.d = self.X.shape[1]
        self.n = self.X.shape[0]
        self.pairs = tuple(tuple(map(int, p)) for p in pairs)
        self.lambda1 = float(lambda1)
        self.final_trek_weight = float(trek_weight)
        self.trek_function = str(trek_function)
        self.trek_kernel_name = str(trek_kernel)
        self.final_dag_weight = float(dag_weight)
        self.mu_init = float(mu_init)
        self.mu_factor = float(mu_factor)
        self.s_schedule = tuple(float(x) for x in s)
        self.trek_log_terms = 2 * self.d if trek_log_terms is None else int(trek_log_terms)
        self.inverse_epsilon = float(inverse_epsilon)
        self.model = SharedDagmaLinear("l2")
        self.model.X = self.X.copy(); self.model.n = self.n; self.model.d = self.d
        self.model.Id = np.eye(self.d, dtype=np.float64)
        self.model.cov = self.X.T @ self.X / float(self.n)
        self.model.lambda1 = self.lambda1
        self.model.dag_penalty_weight = self.final_dag_weight
        self.model.dag_constraint = str(dag_constraint)
        self.model.trek_weight = self.final_trek_weight
        self.model.trek_function = self.trek_function
        self.model.trek_log_terms = self.trek_log_terms
        self.model.trek_inverse_epsilon = self.inverse_epsilon
        self.model.no_trek_pairs = np.asarray(self.pairs, dtype=int).reshape((-1, 2)) if self.pairs else np.empty((0, 2), dtype=int)
        self.model._trek_kernel = make_notreks_kernel(
            trek_kernel, self.model.no_trek_pairs, self.d)
        self.model._inverse_structural_kernel = InverseStructuralKernel.from_pair_mask(
            self.model._trek_kernel.pair_mask, self.model._trek_kernel.scale,
            self.inverse_epsilon)

    def stage(self, alpha, regime="dag_first"):
        alpha = float(np.clip(alpha, 0., 1.))
        stage_index = min(int(round(alpha * (len(self.s_schedule)-1))), len(self.s_schedule)-1)
        s = self.s_schedule[stage_index]
        if regime == "simultaneous":
            dag_alpha = nt_alpha = alpha
        elif regime == "dag_first":
            dag_alpha = min(1., 2. * alpha)
            nt_alpha = max(0., 2. * alpha - 1.)
        else:
            raise ValueError("regime must be dag_first or simultaneous")
        return ContinuationStage(alpha, self.mu_init * self.mu_factor ** stage_index,
                                 s, self.final_dag_weight * dag_alpha,
                                 self.final_trek_weight * nt_alpha)

    def components(self, W, stage: ContinuationStage):
        W = np.asarray(W, dtype=np.float64).copy(); np.fill_diagonal(W, 0.)
        if W.shape != (self.d, self.d) or not np.all(np.isfinite(W)):
            return {"total": np.inf, "data": np.inf, "sparsity": np.inf,
                    "dag": np.inf, "notreks": np.inf, "gradient": np.zeros_like(W),
                    "valid_domain": False}
        M = stage.s * self.model.Id - W * W
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0 or not np.isfinite(logdet) or np.any(~np.isfinite(M)):
            return {"total": np.inf, "data": np.inf, "sparsity": np.inf,
                    "dag": np.inf, "notreks": np.inf, "gradient": np.zeros_like(W),
                    "valid_domain": False}
        try:
            data, g_data = self.model._score(W)
            dag, g_dag = self.model._h(W, stage.s)
            nt, g_nt = self.model._trek_kernel.value_grad(
                W, self.trek_function, log_terms=self.trek_log_terms,
                inverse_epsilon=self.inverse_epsilon)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return {"total": np.inf, "data": np.inf, "sparsity": np.inf,
                    "dag": np.inf, "notreks": np.inf, "gradient": np.zeros_like(W),
                    "valid_domain": False}
        sparsity = self.lambda1 * float(np.abs(W).sum())
        total = stage.mu * (data + sparsity) + stage.dag_weight * dag + stage.notreks_weight * nt
        gradient = (stage.mu * (g_data + self.lambda1 * np.sign(W))
                    + stage.dag_weight * g_dag + stage.notreks_weight * g_nt)
        np.fill_diagonal(gradient, 0.)
        return {"total": float(total), "data": float(data), "sparsity": float(sparsity),
                "dag": float(dag), "notreks": float(nt), "gradient": gradient,
                "valid_domain": True}

    def local_quench(self, W, stage, *, warm_iter=1000, max_iter=2000,
                     lr=.0003):
        model = SharedDagmaLinear("l2")
        out = model.fit(self.X.copy(), no_trek_pairs=self.pairs,
                        trek_weight=stage.notreks_weight,
                        trek_function=self.trek_function,
                        trek_kernel=self.trek_kernel_name,
                        initial_W=np.asarray(W, dtype=np.float64),
                        lambda1=self.lambda1, w_threshold=0., T=1,
                        mu_schedule=[stage.mu], s=[stage.s],
                        dag_penalty_weight=stage.dag_weight,
                        warm_iter=int(warm_iter), max_iter=int(max_iter),
                        lr=float(lr), checkpoint=max(1, int(max_iter)))
        return np.asarray(out, dtype=np.float64), model
