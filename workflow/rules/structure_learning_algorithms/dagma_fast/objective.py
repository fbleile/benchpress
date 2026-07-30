"""Data/function-class objective contracts for DAGMA-fast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class DagmaObjective(Protocol):
    dimension: int
    def value(self, W: np.ndarray) -> float: ...
    def value_and_grad(self, W: np.ndarray) -> tuple[float, np.ndarray]: ...
    def data_loss_value(self, W: np.ndarray) -> float: ...
    def data_loss_value_and_grad(
        self, W: np.ndarray,
    ) -> tuple[float, np.ndarray]: ...


class ObjectiveComponent(Protocol):
    def value(self, W: np.ndarray) -> float: ...
    def value_and_grad(self, W: np.ndarray) -> tuple[float, np.ndarray]: ...


@dataclass
class LinearL2Objective:
    covariance: np.ndarray

    def __init__(self, data=None, *, covariance=None):
        if covariance is None:
            X = np.asarray(data, dtype=np.float64)
            if X.ndim != 2:
                raise ValueError("linear data must be two-dimensional")
            X = X - X.mean(axis=0, keepdims=True)
            covariance = X.T @ X / len(X)
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.dimension = len(self.covariance)
        self.identity = np.eye(self.dimension)

    def data_loss_value_and_grad(self, W):
        difference = self.identity - np.asarray(W, dtype=np.float64)
        rhs = self.covariance @ difference
        return float(0.5 * np.trace(difference.T @ rhs)), -rhs

    value_and_grad = data_loss_value_and_grad

    def data_loss_value(self, W):
        return self.data_loss_value_and_grad(W)[0]

    value = data_loss_value


@dataclass
class CallableObjective:
    dimension: int
    value_grad_callback: Callable[[np.ndarray], tuple[float, np.ndarray]]

    def data_loss_value_and_grad(self, W):
        value, gradient = self.value_grad_callback(
            np.asarray(W, dtype=np.float64))
        return float(value), np.asarray(gradient, dtype=np.float64)

    value_and_grad = data_loss_value_and_grad

    def data_loss_value(self, W):
        return self.data_loss_value_and_grad(W)[0]

    value = data_loss_value
