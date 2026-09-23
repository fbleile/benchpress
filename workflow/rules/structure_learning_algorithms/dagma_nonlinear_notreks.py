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
    postprocess_weighted_adjacency,
)


class NonlinearDagmaNoTreks(DagmaNonlinear):
    def __init__(self, model, pair_mask: np.ndarray | None = None,
                 trek_weight: float = 0.0, verbose: bool = False):
        super().__init__(model, verbose=verbose, dtype=torch.float64)
        d = model.d
        mask = np.zeros((d, d), dtype=float) if pair_mask is None else pair_mask
        self.nt_mask = torch.as_tensor(mask, dtype=torch.float64)
        self.trek_weight = float(trek_weight)
        self.last_notreks_penalty = 0.0

    def _adjacency(self):
        weights = self.model.fc1.weight.view(self.model.d, -1, self.model.d)
        return torch.sum(weights ** 2, dim=1).t()

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
            score = self.log_mse_loss(prediction, self.X)
            l1 = lambda1 * self.model.fc1_l1_reg()
            nt = self._notreks(s)
            objective = mu * (score + l1) + h_value + nt
            if not torch.isfinite(objective):
                return False
            objective.backward()
            optimizer.step()
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
                              postselection_policy: str = "feasible_parent_shrink",
                              screening_floor: float = .01,
                              lambda_bic: float = 2.0):
    """Fit upstream nonlinear DAGMA with optional adjacency-level NOTREKS."""
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed) % (2**31 - 1))
    pair_mask = np.zeros((d, d), dtype=float)
    for left, right in pairs:
        pair_mask[int(left), int(right)] = 1.0
        pair_mask[int(right), int(left)] = 1.0
    model = DagmaMLP(dims=[d, hidden_dim, 1], bias=True, dtype=torch.float64)
    # Keep vanilla nonlinear DAGMA on the upstream implementation exactly.
    # The subclass is needed only when a NOTREKS term is actually requested.
    if pairs:
        solver = NonlinearDagmaNoTreks(
            model, pair_mask=pair_mask, trek_weight=trek_weight)
    else:
        solver = DagmaNonlinear(model, verbose=False, dtype=torch.float64)
    started = time.perf_counter()
    solver.fit(X, lambda1=lambda1, lambda2=lambda2, T=stages,
               mu_init=.1, mu_factor=.1, s=1.0, warm_iter=warm_iter,
               max_iter=max_iter, lr=.0002, w_threshold=threshold,
               checkpoint=max(100, min(1000, warm_iter // 3)))
    weighted = model.fc1_to_adj()
    adjacency, coefficients, post_diag = postprocess_weighted_adjacency(
        X, weighted, pairs, screening_floor=screening_floor,
        lambda_bic=lambda_bic, notreks_active=bool(pairs),
        postselection_policy=postselection_policy)
    from workflow.rules.structure_learning_algorithms.flop.adapter import convert_flop_cpdag
    runtime = time.perf_counter() - started
    return adjacency, {
        "candidate_graph": adjacency.copy(),
        "coefficients": coefficients,
        "weighted_adjacency": weighted,
        "cpdag": convert_flop_cpdag(adjacency, d),
        "optimizer_restarts": 1,
        "optimizer_violations": 0,
        "nonlinear_dagma_notreks_penalty": (
            solver.last_notreks_penalty if pairs else 0.0),
        "postselection": post_diag,
        "nonlinear_dagma_runtime": runtime,
    }
