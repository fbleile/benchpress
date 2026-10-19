"""Small nonlinear DAGMA adapter with an adjacency-level NOTREKS term.

The upstream nonlinear DAGMA model exposes its induced adjacency through
``fc1_to_adj``.  We use the same adjacency for the DAGMA log-det and for a
single differentiable NOTREKS resolvent penalty during every continuation
stage.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
from workflow.rules.structure_learning_algorithms.dagma_notreks.pipeline import (
    production_candidate_graph,
)


class NonlinearDagmaNoTreks(DagmaNonlinear):
    def __init__(self, model, pair_mask: np.ndarray | None = None,
                 trek_weight: float = 0.0,
                 edge_mask: np.ndarray | None = None,
                 group_l1: bool = True,
                 score: str = "profiled_log_mse",
                 student_t_nu: float = 4.0,
                 verbose: bool = False):
        super().__init__(model, verbose=verbose, dtype=torch.float64)
        d = model.d
        mask = np.zeros((d, d), dtype=float) if pair_mask is None else pair_mask
        self.nt_mask = torch.as_tensor(mask, dtype=torch.float64)
        allowed = (np.ones((d, d), dtype=float) if edge_mask is None
                   else np.asarray(edge_mask, dtype=float))
        if allowed.shape != (d, d):
            raise ValueError("edge_mask must have shape (d, d)")
        np.fill_diagonal(allowed, 0.0)
        self.edge_mask = torch.as_tensor(allowed, dtype=torch.float64)
        self.group_l1 = bool(group_l1)
        if score not in {"profiled_log_mse", "student_t"}:
            raise ValueError("score must be profiled_log_mse or student_t")
        if student_t_nu <= 0:
            raise ValueError("student_t_nu must be positive")
        self.score_name = score
        self.student_t_nu = float(student_t_nu)
        self.trek_weight = float(trek_weight)
        self.last_notreks_penalty = 0.0

    def log_mse_loss(self, output: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
        """Profiled unequal-variance nonlinear Gaussian score.

        The upstream nonlinear DAGMA loss pools squared residuals across
        coordinates.  That makes raw, unstandardized data dominated by the
        largest-scale variables.  Profiling one residual variance per target
        gives the nonlinear analogue of the linear unequal-variance score.
        Coordinatewise rescaling changes this only by a graph-independent
        additive constant.
        """
        residual = output - target
        variance = torch.mean(residual.square(), dim=0)
        variance = torch.clamp(variance, min=torch.finfo(variance.dtype).tiny)
        return 0.5 * torch.sum(torch.log(variance))

    def score_loss(self, output: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        """Return the configured scale-aware nonlinear nodewise score.

        ``student_t`` profiles one scale per target and uses a robust
        Student-t residual term.  This protects the graph search from a few
        large manufacturing/switching residuals while retaining invariance
        to coordinate rescaling.  The same method is used for optimization
        and fixed-support refitting.
        """
        if self.score_name == "profiled_log_mse":
            return self.log_mse_loss(output, target)
        residual = output - target
        variance = torch.mean(residual.square(), dim=0)
        variance = torch.clamp(variance, min=torch.finfo(variance.dtype).tiny)
        nu = self.student_t_nu
        robust_term = torch.mean(
            torch.log1p(residual.square() / (nu * variance)), dim=0)
        return (0.5 * torch.sum(torch.log(variance))
                + 0.5 * (nu + 1.0) * torch.sum(robust_term))

    def _adjacency(self):
        weights = self.model.fc1.weight.view(self.model.d, -1, self.model.d)
        adjacency = torch.sum(weights ** 2, dim=1).t()
        return adjacency * self.edge_mask

    def _group_l1_reg(self):
        """Causal-edge group penalty over all hidden units.

        ``fc1`` is grouped as [target, hidden unit, parent].  A parent is
        therefore one causal edge, not one individual neural-network weight.
        The sqrt(hidden) factor keeps the coefficient roughly comparable to
        the elementwise L1 coefficient used by the upstream model.
        """
        weights = self.model.fc1.weight.view(self.model.d, -1, self.model.d)
        groups = torch.sqrt(torch.sum(weights.square(), dim=1) + 1e-24)
        groups = groups * self.edge_mask.t()
        return groups.sum() * torch.sqrt(torch.tensor(
            float(weights.shape[1]), dtype=weights.dtype,
            device=weights.device))

    def _regularizer(self):
        return self._group_l1_reg() if self.group_l1 else self.model.fc1_l1_reg()

    def _prox_group_l1(self, step_size: float):
        """Apply the proximal group shrink for the edge blocks."""
        if not self.group_l1:
            return
        weights = self.model.fc1.weight.view(self.model.d, -1, self.model.d)
        groups = torch.sqrt(torch.sum(weights.square(), dim=1) + 1e-24)
        coefficient = (float(step_size) * self.trek_mu
                       * float(self.group_lambda1)
                       * float(np.sqrt(weights.shape[1])))
        factor = torch.clamp(1.0 - coefficient / groups, min=0.0)
        factor = factor * self.edge_mask.t()
        with torch.no_grad():
            # The structural objective uses the same edge block convention.
            weights.mul_(factor[:, None, :])

    def _zero_forbidden_parameters(self):
        """Keep hard directed exclusions absent from the MLP support."""
        # fc1 is grouped as [output-node, hidden-unit, input-node]; the
        # adjacency returned by fc1_to_adj is [input-node, output-node].
        hidden = self.model.fc1.weight.shape[0] // self.model.d
        allowed = self.edge_mask.t().repeat_interleave(hidden, dim=0)
        with torch.no_grad():
            self.model.fc1.weight.mul_(allowed)

    def _notreks(self, s):
        if self.trek_weight == 0.0 or not bool(torch.any(self.nt_mask)):
            return self._adjacency().sum() * 0.0
        adjacency = self._adjacency()
        resolvent = torch.linalg.inv(s * self.model.I - adjacency)
        gram = resolvent.transpose(0, 1) @ resolvent
        # The supplied mask is symmetric and each unordered pair is counted
        # once, matching the production kernel's pair semantics.
        return (self.trek_weight * torch.sum(gram * self.nt_mask)
                / max(1, self.model.d - 1))

    def minimize(self, max_iter, lr, lambda1, lambda2, mu, s,
                 lr_decay=False, tol=1e-6, pbar=None):
        self.trek_mu = float(mu)
        self.group_lambda1 = float(lambda1)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(.99, .999),
            weight_decay=mu * lambda2)
        scheduler = (torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.8)
                     if lr_decay else None)
        previous = 1e16
        for i in range(int(max_iter)):
            optimizer.zero_grad()
            h_value = self.model.h_func(s)
            if h_value.item() < 0 or not torch.isfinite(h_value):
                return False
            prediction = self.model(self.X)
            score = self.score_loss(prediction, self.X)
            l1 = lambda1 * self._regularizer()
            nt = self._notreks(s)
            objective = mu * (score + l1) + h_value + nt
            if not torch.isfinite(objective):
                return False
            objective.backward()
            optimizer.step()
            self._prox_group_l1(lr)
            self._zero_forbidden_parameters()
            if scheduler is not None and (i + 1) % 1000 == 0:
                scheduler.step()
            if i % self.checkpoint == 0 or i == int(max_iter) - 1:
                value = float(objective.detach())
                if abs((previous - value) / previous) <= tol:
                    if pbar is not None:
                        pbar.update(max(0, int(max_iter) - i))
                    self.last_notreks_penalty = float(nt.detach())
                    return True
                previous = value
            if pbar is not None:
                pbar.update(1)
        self.last_notreks_penalty = float(self._notreks(s).detach())
        return True


def nonlinear_dagma_candidate(X: np.ndarray, pairs, seed: int, *,
                              trek_weight: float = 200.0,
                              hidden_dim: int = 10, stages: int = 4,
                              warm_iter: int = 3000, max_iter: int = 6000,
                              lambda1: float = .02, lambda2: float = .005,
                              threshold: float = .3,
                              score: str = "profiled_log_mse",
                              student_t_nu: float = 4.0,
                              postselection_policy: str = "feasible_parent_shrink",
                              screening_floor: float = .01,
                              lambda_bic: float = 2.0,
                              forbidden_edges=(),
                              fixed_support: np.ndarray | None = None,
                              refit_warm_iter: int = 1000,
                              refit_max_iter: int = 2000):
    """Fit upstream nonlinear DAGMA with optional adjacency-level NOTREKS."""
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed) % (2**31 - 1))
    pair_mask = np.zeros((d, d), dtype=float)
    for left, right in pairs:
        pair_mask[int(left), int(right)] = 1.0
        pair_mask[int(right), int(left)] = 1.0
    edge_mask = (np.ones((d, d), dtype=float) if fixed_support is None
                 else np.asarray(fixed_support, dtype=float).copy())
    if edge_mask.shape != (d, d):
        raise ValueError("fixed_support must have shape (d, d)")
    edge_mask = (edge_mask != 0).astype(float)
    for parent, child in forbidden_edges:
        edge_mask[int(parent), int(child)] = 0.0
    np.fill_diagonal(edge_mask, 0.0)
    model = DagmaMLP(dims=[d, hidden_dim, 1], bias=True, dtype=torch.float64)
    # Use the same scale-aware score for vanilla, NOTREKS, and hard-mask
    # variants.  The adjacency-level NOTREKS term remains optional.
    solver = NonlinearDagmaNoTreks(
        model, pair_mask=pair_mask if pairs else None,
        trek_weight=trek_weight if pairs else 0.0,
        edge_mask=edge_mask, score=score, student_t_nu=student_t_nu)
    started = time.perf_counter()
    solver.fit(X, lambda1=lambda1, lambda2=lambda2, T=stages,
               mu_init=.1, mu_factor=.1, s=1.0, warm_iter=warm_iter,
               max_iter=max_iter, lr=.0002, w_threshold=threshold,
               checkpoint=max(100, min(1000, warm_iter // 3)))
    weighted = model.fc1_to_adj()
    weighted = np.asarray(weighted, dtype=float) * edge_mask
    # The nonlinear optimizer and postselection now use the same MLP score.
    # We retain the existing deletion-only feasibility screening to construct
    # one support, then refit that fixed support with the same MLP family.
    if fixed_support is None:
        adjacency, support_diag = production_candidate_graph(
            weighted, pairs, screening_floor=screening_floor,
            notreks_active=bool(pairs))
    else:
        # This is the honest FLOP-order hybrid: FLOP determines the support;
        # nonlinear DAGMA only refits that fixed support.  Do not silently
        # turn this into a second nonlinear structure search.
        adjacency = edge_mask.astype(np.uint8)
        support_diag = {
            "support_source": "flop_fixed_support",
            "support_edges": int(adjacency.sum()),
            "support_screening_floor": float(screening_floor),
            "support_notreks_active": bool(pairs),
        }
    refit_model = DagmaMLP(
        dims=[d, hidden_dim, 1], bias=True, dtype=torch.float64)
    refit_solver = NonlinearDagmaNoTreks(
        refit_model,
        pair_mask=pair_mask if pairs else None,
        trek_weight=trek_weight if pairs else 0.0,
        edge_mask=adjacency, score=score, student_t_nu=student_t_nu)
    refit_solver.fit(
        X, lambda1=lambda1, lambda2=lambda2, T=1,
        mu_init=.1, mu_factor=.1, s=1.0,
        warm_iter=refit_warm_iter, max_iter=refit_max_iter,
        lr=.0002, w_threshold=0.0,
        checkpoint=max(100, min(1000, refit_warm_iter // 3)))
    with torch.no_grad():
        refit_prediction = refit_model(torch.as_tensor(X, dtype=torch.float64))
        nonlinear_score = float(
            refit_solver.score_loss(
                refit_prediction, torch.as_tensor(X, dtype=torch.float64)))
        nonlinear_l1 = float(refit_solver._regularizer())
    coefficients = np.asarray(refit_model.fc1_to_adj(), dtype=float) * adjacency
    post_diag = {
        **support_diag,
        "postselection_policy": "nonlinear_mlp_fixed_support_refit",
        "postprocessed_score": nonlinear_score + lambda1 * nonlinear_l1,
        "nonlinear_log_mse": nonlinear_score,
        "nonlinear_score": score,
        "support_mode": "fixed_flop_support" if fixed_support is not None
        else "nonlinear_search",
        "student_t_nu": float(student_t_nu),
        "nonlinear_l1": nonlinear_l1,
        "linear_bic_used": False,
        "refit_warm_iter": int(refit_warm_iter),
        "refit_max_iter": int(refit_max_iter),
        "raw_weight_max": float(np.max(np.abs(weighted), initial=0.0)),
        "raw_edges_above_1e-8": int(np.count_nonzero(np.abs(weighted) > 1e-8)),
        "raw_edges_above_screening_floor": int(
            np.count_nonzero(np.abs(weighted) >= screening_floor)),
    }
    from workflow.rules.structure_learning_algorithms.dagma_global_search.tools.causal_distances import (
        dag_to_cpdag,
    )
    runtime = time.perf_counter() - started
    return adjacency, {
        "candidate_graph": adjacency.copy(),
        "coefficients": coefficients,
        "weighted_adjacency": weighted,
        "cpdag": dag_to_cpdag(adjacency),
        "optimizer_restarts": 1,
        "optimizer_violations": 0,
        "nonlinear_dagma_notreks_penalty": (
            solver.last_notreks_penalty if pairs else 0.0),
        "postselection": post_diag,
        "raw_weight_max": post_diag["raw_weight_max"],
        "raw_edges_above_1e-8": post_diag["raw_edges_above_1e-8"],
        "raw_edges_above_screening_floor": post_diag[
            "raw_edges_above_screening_floor"],
        "postselection_policy": post_diag["postselection_policy"],
        "postprocessed_score": post_diag["postprocessed_score"],
        "nonlinear_dagma_runtime": runtime,
        "nonlinear_score": score,
        "support_mode": post_diag["support_mode"],
    }
