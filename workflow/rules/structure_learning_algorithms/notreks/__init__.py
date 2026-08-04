"""Reusable NOTREKS penalties and pair utilities.

This package is independent of DAGMA. Continuous estimators opt into a
NOTREKS penalty by constructing a component and passing it through their
structural-penalty interface.
"""

from .component import NoTreksPenalty
from .kernels import (
    FAST_KERNEL,
    KERNEL_REGISTRY,
    make_notreks_kernel,
    notreks_value_grad_kernel,
    validate_pairs,
)
from .pairs import subsample_no_trek_pairs

__all__ = [
    "FAST_KERNEL",
    "KERNEL_REGISTRY",
    "NoTreksPenalty",
    "make_notreks_kernel",
    "notreks_value_grad_kernel",
    "subsample_no_trek_pairs",
    "validate_pairs",
]
