import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("workflow/scripts/utils")
sys.path.append(str(Path(__file__).resolve().parent))

from add_timeout import timeoutf
from independence_tests import pairwise_independence_candidates
from notreks_core import NotreksConfig, fit_linear_baseline, threshold_adjacency
from optimizer import fit_notreks_optimizer


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
    alg_id = getattr(snakemake.wildcards, "alg_id", None)
    if alg_id is not None:
        for alg in algs:
            if str(alg.get("alg_id", alg.get("id"))) != str(alg_id):
                continue
            manifest_path = alg.get("params_manifest")
            if not manifest_path:
                return alg
            path = Path(manifest_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            with path.open() as handle:
                rows = json.load(handle)
            for row in rows:
                if str(row.get("path_id", row.get("algorithm_id"))) == str(alg_id):
                    if "hyperparameters" in row:
                        return row["hyperparameters"]
                    if "hyperparameters_json" in row:
                        return json.loads(row["hyperparameters_json"])
            raise KeyError(f"Could not resolve NOTREKS alg_id={alg_id!r} in params manifest {path}")

    matches = []
    for alg in algs:
        ok = True
        for key, value in alg.items():
            if key in {"id", "threshold", "seed"}:
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


def _optional_config_value(name, default=None):
    try:
        return _config_value(name, default)
    except KeyError:
        return default


def _read_config():
    cfg = NotreksConfig(
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
        tol=float(_config_value("tol")),
        threshold=float(_config_value("threshold")),
        timeout=None if _timeout_is_none(_config_value("timeout")) else float(_config_value("timeout")),
        init=str(_config_value("init", "zero")),
        checkpoint=int(_config_value("checkpoint", 1000)),
        power_iter_steps=int(_config_value("power_iter_steps", 5)),
        scc_threshold=float(_config_value("scc_threshold", 1e-8)),
        independence_cache_dir=None
        if _optional_config_value("independence_cache_dir", None) in {None, "", "None", "none", "null"}
        else str(_optional_config_value("independence_cache_dir")),
        warm_iter=int(_config_value("warm_iter", _config_value("max_iter"))),
    )
    if cfg.init not in {"zero", "linear_baseline"}:
        raise ValueError("init must be one of {'zero', 'linear_baseline'}")
    return cfg


def _print_config_sanity(cfg: NotreksConfig) -> None:
    print(
        "NOTREKS run config: "
        f"id={cfg.algorithm_id}, threshold={cfg.threshold}, init={cfg.init}, "
        f"score={cfg.score}, dag_reg={cfg.dag_reg}, trek_seq={cfg.trek_seq}, trek_reg={cfg.trek_reg}, "
        f"stage_iter_policy=max_every_stage, max_iter={cfg.max_iter}, path_steps={cfg.path_steps}, "
        f"power_iter_steps={cfg.power_iter_steps}, scc_threshold={cfg.scc_threshold}, "
        f"independence_cache_dir={cfg.independence_cache_dir}"
    )


def _write_diagnostics(path: Path, cfg: NotreksConfig, diagnostics) -> None:
    records = []
    for stage in diagnostics.stages:
        base = {
            "algorithm_id": cfg.algorithm_id,
            "init": cfg.init,
            "path_steps_completed": diagnostics.path_steps_completed,
            "final_mu": diagnostics.final_mu,
            "optimizer_converged": diagnostics.converged,
        }
        base.update(stage)
        records.append(base)
        for checkpoint in stage.get("checkpoints", []):
            row = base.copy()
            row.update(checkpoint)
            row["row_type"] = "checkpoint"
            records.append(row)
    if not records:
        records.append({
            "algorithm_id": cfg.algorithm_id,
            "init": cfg.init,
            "path_steps_completed": diagnostics.path_steps_completed,
            "final_mu": diagnostics.final_mu,
            "optimizer_converged": diagnostics.converged,
            "objective": diagnostics.objective,
        })
    pd.DataFrame.from_records(records).to_csv(path, index=False)


def wrapper():
    cfg = _read_config()
    _print_config_sanity(cfg)
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
        cache_dir=Path(cfg.independence_cache_dir) if cfg.independence_cache_dir else None,
        dataset_path=Path(snakemake.input["data"]),
    )
    print(
        "NOTREKS independence tests: "
        f"method={independence.method}, cache={independence.cache_status}, "
        f"cache_key={independence.cache_key}, accepted={len(independence.pairs)}, "
        f"tested={independence.number_of_tests}"
    )

    if cfg.init == "linear_baseline":
        W_init = fit_linear_baseline(
            X,
            score=cfg.score,
            regularizer=cfg.regularizer,
            regularizer_scale=cfg.regularizer_scale,
            independence_pairs=independence.pairs,
            rng=rng,
        )
    else:
        W_init = None
    W_est, diagnostics = fit_notreks_optimizer(X, cfg, independence.pairs, W_init=W_init)

    adjmat = threshold_adjacency(W_est, cfg.threshold)
    adjmat_df = pd.DataFrame(adjmat, columns=df.columns)
    adjmat_df.to_csv(snakemake.output["adjmat"], index=False)

    tottime = time.perf_counter() - start
    with open(snakemake.output["time"], "w") as text_file:
        text_file.write(str(tottime))

    with open(snakemake.output["ntests"], "w") as text_file:
        text_file.write(str(independence.number_of_tests))

    diagnostics_path = Path(snakemake.output["adjmat"]).with_name("diagnostics.csv")
    _write_diagnostics(diagnostics_path, cfg, diagnostics)


start = time.perf_counter()

timeout_value = _optional_config_value("timeout", None)
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
