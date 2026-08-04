"""Optimizer-neutral NOTREKS objective component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .kernels import FAST_KERNEL, BaseNoTreksKernel, make_notreks_kernel


@dataclass
class NoTreksPenalty:
    """Weighted NOTREKS component for any weighted-adjacency optimizer.

    The component deliberately knows nothing about DAGMA, continuation
    schedules, or data losses. Its only contract is ``value_and_grad(W)``.
    """

    pairs: Sequence[tuple[int, int]]
    dimension: int
    weight: float = 1.0
    function: str = "inv"
    kernel: str = FAST_KERNEL
    inverse_epsilon: float = 1e-8
    log_terms: int | None = None
    _implementation: BaseNoTreksKernel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("NOTREKS weight must be finite and non-negative")
        self._implementation = make_notreks_kernel(
            self.kernel, self.pairs, self.dimension)

    def value_and_grad(self, W: np.ndarray) -> tuple[float, np.ndarray]:
        result = self._implementation.value_and_grad(
            W,
            function=self.function,
            log_terms=self.log_terms,
            inverse_epsilon=self.inverse_epsilon,
        )
        return (
            self.weight * result.penalty_value,
            self.weight * result.gradient_W,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "kernel": self.kernel,
            "function": self.function,
            "weight": self.weight,
            **self._implementation.diagnostics(),
        }
