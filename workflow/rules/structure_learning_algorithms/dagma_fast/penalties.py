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
    factorization_count: int = 0

    def is_valid(self, W, s=1.0):
        matrix = s * np.eye(self.dimension) - W * W
        _, logdet, sign, valid = fused_inverse_logdet(matrix)
        return bool(sign > 0 and np.isfinite(logdet) and valid)

    def value_and_grad(self, W, s=1.0):
        matrix = s * np.eye(self.dimension) - W * W
        inverse_transpose, logdet, sign, valid = fused_inverse_logdet(matrix)
        self.factorization_count += 1
        if sign <= 0 or not np.isfinite(logdet) or not valid:
            raise FloatingPointError("invalid log-det M-matrix domain")
        value = -float(logdet) + self.dimension * np.log(float(s))
        return value, 2.0 * W * inverse_transpose

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
