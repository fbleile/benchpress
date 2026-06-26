import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("workflow/scripts/utils")
sys.path.append(str(Path(__file__).resolve().parents[1] / "notreks"))

from add_timeout import timeoutf
from independence_tests import pairwise_independence_candidates


def _is_none(value) -> bool:
    return value is None or str(value) in {"", "None", "none", "null"}


def fit_marginal_trek_graph(
    df: pd.DataFrame,
    *,
    independence_test: str,
    independence_alpha: float,
    independence_correction: str,
    independence_cache_dir=None,
    dataset_path=None,
) -> tuple[np.ndarray, int]:
    X = df.to_numpy(dtype=float, copy=True)
    independence = pairwise_independence_candidates(
        X,
        method=str(independence_test),
        alpha=float(independence_alpha),
        correction=str(independence_correction),
        columns=list(df.columns),
        cache_dir=Path(independence_cache_dir) if independence_cache_dir else None,
        dataset_path=Path(dataset_path) if dataset_path else None,
    )
    d = X.shape[1]
    adjmat = np.ones((d, d), dtype=int)
    np.fill_diagonal(adjmat, 0)
    for i, j in independence.pairs:
        adjmat[int(i), int(j)] = 0
        adjmat[int(j), int(i)] = 0
    return adjmat, int(independence.number_of_tests)


def run_marginal_trek_graph(
    data_csv: Path,
    adjmat_out: Path,
    time_out: Path,
    ntests_out: Path,
    *,
    independence_test: str,
    independence_alpha: float,
    independence_correction: str,
    independence_cache_dir=None,
) -> None:
    start = time.perf_counter()
    df = pd.read_csv(data_csv)
    adjmat, ntests = fit_marginal_trek_graph(
        df,
        independence_test=independence_test,
        independence_alpha=independence_alpha,
        independence_correction=independence_correction,
        independence_cache_dir=None if _is_none(independence_cache_dir) else independence_cache_dir,
        dataset_path=data_csv,
    )
    pd.DataFrame(adjmat, columns=df.columns).to_csv(adjmat_out, index=False)
    time_out.write_text(str(time.perf_counter() - start))
    ntests_out.write_text(str(ntests))


def _wildcard(name, default=None):
    if hasattr(snakemake.wildcards, name):
        return getattr(snakemake.wildcards, name)
    if default is not None:
        return default
    raise KeyError(name)


def wrapper():
    run_marginal_trek_graph(
        Path(snakemake.input["data"]),
        Path(snakemake.output["adjmat"]),
        Path(snakemake.output["time"]),
        Path(snakemake.output["ntests"]),
        independence_test=str(_wildcard("independence_test")),
        independence_alpha=float(_wildcard("independence_alpha")),
        independence_correction=str(_wildcard("independence_correction")),
        independence_cache_dir=_wildcard("independence_cache_dir", None),
    )


if "snakemake" in globals():
    timeout_value = _wildcard("timeout", None)
    if _is_none(timeout_value):
        wrapper()
    else:
        with timeoutf(
            int(float(timeout_value)),
            snakemake.output["adjmat"],
            snakemake.output["time"],
            snakemake.output["ntests"],
            time.perf_counter(),
        ):
            wrapper()
