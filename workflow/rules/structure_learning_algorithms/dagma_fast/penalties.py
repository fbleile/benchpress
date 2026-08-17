"""DAG-penalty boundary for the canonical optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma_anytime.solver import (
    fused_inverse_logdet,
)


@dataclass
class LogDetDagPenalty:
    dimension: int
    adjacency_mapping: str = "hadamard"
    factorization_count: int = 0

    def _mapped(self, W):
        W = np.asarray(W, dtype=float)
        if self.adjacency_mapping == "hadamard":
            return W * W, 2.0 * W
        if self.adjacency_mapping == "phi_log":
            scale = 2.0 / max(1, self.dimension)
            mapped = scale * np.log1p(np.abs(W))
            jacobian = scale * np.sign(W) / (1.0 + np.abs(W))
            return mapped, jacobian
        raise ValueError("unknown adjacency mapping")

    def is_valid(self, W, s=1.0):
        mapped, _ = self._mapped(W)
        matrix = s * np.eye(self.dimension) - mapped
        _, logdet, sign, valid = fused_inverse_logdet(matrix)
        return bool(sign > 0 and np.isfinite(logdet) and valid)

    def value_and_grad(self, W, s=1.0):
        mapped, jacobian = self._mapped(W)
        matrix = s * np.eye(self.dimension) - mapped
        inverse_transpose, logdet, sign, valid = fused_inverse_logdet(matrix)
        self.factorization_count += 1
        if sign <= 0 or not np.isfinite(logdet) or not valid:
            raise FloatingPointError("invalid log-det M-matrix domain")
        value = -float(logdet) + self.dimension * np.log(float(s))
        return value, jacobian * inverse_transpose

    def value(self, W, s=1.0):
        return self.value_and_grad(W, s)[0]


@dataclass
class CallableDagPenalty:
    value_grad_callback: Callable[[np.ndarray, float], tuple[float, np.ndarray]]
    validity_callback: Callable[[np.ndarray, float], bool] = (
        lambda W, s: True)

    def is_valid(self, W, s=1.0):
        return bool(self.validity_callback(W, s))

    def value_and_grad(self, W, s=1.0):
        value, gradient = self.value_grad_callback(W, s)
        return float(value), np.asarray(gradient, dtype=np.float64)

    def value(self, W, s=1.0):
        return self.value_and_grad(W, s)[0]
