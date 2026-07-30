"""Graph-score adapters; generic search contains no OLS assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from workflow.rules.structure_learning_algorithms.dagma.gaussian_bic import gaussian_bic


@dataclass
class GaussianBICGraphScore:
    data: np.ndarray
    lambda_bic: float = 1.0

    def score_graph(self, graph, data=None):
        value, _ = gaussian_bic(
            self.data if data is None else data,
            np.asarray(graph, dtype=int),
            lambda_bic=self.lambda_bic)
        return float(value)


@dataclass
class CallableGraphScore:
    callback: Callable[[np.ndarray, object], float]
    context: object = None

    def score_graph(self, graph, data=None):
        return float(self.callback(np.asarray(graph, dtype=int), self.context))
