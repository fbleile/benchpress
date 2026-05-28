import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("workflow/scripts/utils")
sys.path.append(str(Path(__file__).resolve().parent))

from add_timeout import timeoutf
from independence_tests import pairwise_independence_candidates
from notreks_core import NotreksConfig, fit_linear_baseline, threshold_adjacency
from optimizer import fit_notreks_optimizer
from penalties import penalty_diagnostics


def _wildcard(name, default=None):
    if hasattr(snakemake.wildcards, name):
        return getattr(snakemake.wildcards, name)
    if default is not None:
        return default
    raise KeyError(name)


def _same_config_value(config_value, wildcard_value):
    if config_value is None:
        return str(wildcard_value) == "None"
    return str(config_value) == str(wildcard_value)


def _algorithm_config():
    if not hasattr(snakemake, "config"):
        return {}

    algs = (
        snakemake.config.get("resources", {})
        .get("structure_learning_algorithms", {})
        .get("notreks", [])
    )
    matches = []
    for alg in algs:
        ok = True
        for key, value in alg.items():
            if key in {"id", "threshold"}:
                continue
            if hasattr(snakemake.wildcards, key) and not _same_config_value(value, getattr(snakemake.wildcards, key)):
                ok = False
                break
        if ok:
            matches.append(alg)
    if len(matches) == 1:
        return matches[0]
    return {}


def _config_value(name, default=None):
    alg = _algorithm_config()
    if name in alg:
        return alg[name]
    return _wildcard(name, default)


def _timeout_is_none(value):
    return str(value) in {"None", "none", "null", ""}


def _read_config():
    return NotreksConfig(
        algorithm_id=str(_config_value("id", "notreks")),
        function_class=str(_config_value("function_class")),
        score=str(_config_value("score")),
        dag_seq=str(_config_value("dag_seq")),
        dag_reg=float(_config_value("dag_reg")),
        dag_s=float(_config_value("dag_s")),
        trek_seq=str(_config_value("trek_seq")),
        trek_reg=float(_config_value("trek_reg")),
        regularizer=str(_config_value("regularizer")),
        regularizer_scale=float(_config_value("regularizer_scale")),
        independence_test=str(_config_value("independence_test")),
        independence_alpha=float(_config_value("independence_alpha")),
        independence_correction=str(_config_value("independence_correction")),
        seed=int(_config_value("seed")),
        max_iter=int(_config_value("max_iter")),
        lr=float(_config_value("lr")),
        path_steps=int(_config_value("path_steps")),
        mu_init=float(_config_value("mu_init")),
        mu_factor=float(_config_value("mu_factor")),
        warm_iter=int(_config_value("warm_iter")),
        tol=float(_config_value("tol")),
        threshold=float(_config_value("threshold")),
        timeout=None if _timeout_is_none(_config_value("timeout")) else float(_config_value("timeout")),
    )


def wrapper():
    cfg = _read_config()
    if cfg.function_class != "linear":
        raise NotImplementedError("Only function_class='linear' is implemented in the local notreks module")
    rng = np.random.default_rng(cfg.seed)

    df = pd.read_csv(snakemake.input["data"])
    X = df.to_numpy(dtype=float, copy=True)

    independence = pairwise_independence_candidates(
        X,
        method=cfg.independence_test,
        alpha=cfg.independence_alpha,
        correction=cfg.independence_correction,
        columns=list(df.columns),
    )

    W_init = fit_linear_baseline(
        X,
        score=cfg.score,
        regularizer=cfg.regularizer,
        regularizer_scale=cfg.regularizer_scale,
        independence_pairs=independence.pairs,
        rng=rng,
    )
    W_est, _ = fit_notreks_optimizer(X, cfg, independence.pairs, W_init=W_init)
    _ = penalty_diagnostics(W_est, cfg, independence.pairs)

    adjmat = threshold_adjacency(W_est, cfg.threshold)
    adjmat_df = pd.DataFrame(adjmat, columns=df.columns)
    adjmat_df.to_csv(snakemake.output["adjmat"], index=False)

    tottime = time.perf_counter() - start
    with open(snakemake.output["time"], "w") as text_file:
        text_file.write(str(tottime))

    with open(snakemake.output["ntests"], "w") as text_file:
        text_file.write(str(independence.number_of_tests))


start = time.perf_counter()

timeout_value = _wildcard("timeout")
if _timeout_is_none(timeout_value):
    wrapper()
else:
    with timeoutf(
        int(float(timeout_value)),
        snakemake.output["adjmat"],
        snakemake.output["time"],
        snakemake.output["ntests"],
        start,
    ):
        wrapper()
