"""Lazy FLOP-style search with a soft continuous NOTREKS score."""

from .soft_greedy import SoftGreedyConfig, SoftGreedyResult, fit_soft_notreks

__all__ = ["SoftGreedyConfig", "SoftGreedyResult", "fit_soft_notreks"]
