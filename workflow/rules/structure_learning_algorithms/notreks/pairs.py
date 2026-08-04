"""Deterministic handling of supplied no-trek prior knowledge."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def subsample_no_trek_pairs(
    pairs: Sequence[tuple[int, int]],
    fraction: float = 1.0,
    seed: int = 0,
) -> list[tuple[int, int]]:
    """Return a reproducible fraction of canonical unordered pairs."""
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("knowledge_fraction must be between zero and one")
    canonical = sorted({tuple(sorted(map(int, pair))) for pair in pairs})
    if fraction == 0.0 or not canonical:
        return []
    count = min(len(canonical), max(1, int(np.ceil(fraction * len(canonical)))))
    if count == len(canonical):
        return canonical
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(len(canonical), size=count, replace=False)
    return [canonical[index] for index in sorted(map(int, selected))]
