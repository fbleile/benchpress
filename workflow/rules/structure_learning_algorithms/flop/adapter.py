import numpy as np


def convert_flop_cpdag(raw, d):
    raw = np.asarray(raw)
    if raw.shape != (d, d) or not set(np.unique(raw)).issubset({0, 1, 2}):
        raise ValueError("unexpected FLOP CPDAG representation")
    if np.any((raw == 2) != (raw.T == 2)):
        raise ValueError("FLOP undirected edges must be symmetric value-2 pairs")
    result = raw.copy()
    result[result == 2] = 1
    return result.astype(int)
