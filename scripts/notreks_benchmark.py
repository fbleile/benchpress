#!/usr/bin/env python3
"""Compile and analyse the paper-facing four-method NOTREKS benchmark.

The compiler emits one ordinary Benchpress configuration per scenario.  A
scenario can come from a Cartesian grid or be listed explicitly, which keeps
cluster submission and targeted follow-ups equally simple.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_IDS = ("flop", "flop_notreks", "dagma", "dagma_notreks")
DAGMA_DEFAULTS = {
    "loss_type": "l2", "lambda1": 0.03, "w_threshold": 0.3,
    "T": 5, "mu_init": 1.0, "mu_factor": 0.1,
    "s": "1.0,0.9,0.8,0.7,0.6", "warm_iter": 30000,
    "max_iter": 60000, "lr": 0.0003, "checkpoint": 1000,
    "beta_1": 0.99, "beta_2": 0.999, "timeout": None,
}


def _markdown(frame: pd.DataFrame) -> str:
    table = frame.reset_index()
    table.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple) else str(column)
        for column in table.columns]
    rows = [["" if pd.isna(value) else str(value) for value in row]
            for row in table.to_numpy()]
    return "\n".join([
        "| " + " | ".join(map(str, table.columns)) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def _slug(value: object) -> str:
    return str(value).lower().replace(".", "p").replace("_", "-")


def expand_scenarios(spec: dict) -> list[dict]:
    """Expand a Cartesian grid and append explicit scenario overrides."""
    grid = spec.get("grid", {})
    keys = ("models", "graphs", "dimensions", "sample_sizes",
            "knowledge_fractions")
    missing = [key for key in keys if not grid.get(key)]
    if missing:
        raise ValueError(f"benchmark grid is missing nonempty fields: {missing}")
    scenarios = []
    for model, graph, d, n, fraction in itertools.product(
            *(grid[key] for key in keys)):
        model_spec = spec["models"][model]
        graph_spec = spec["graphs"][graph]
        scenario = {
            "model": model, "graph": graph, "d": int(d), "n": int(n),
            "knowledge_fraction": float(fraction),
            "seeds": list(grid.get("seeds", spec.get("seeds", [1]))),
            **{f"model_{key}": value for key, value in model_spec.items()},
            **{f"graph_{key}": value for key, value in graph_spec.items()},
        }
        scenarios.append(scenario)
    scenarios.extend(spec.get("scenarios", []))
    seen: set[str] = set()
    result = []
    for scenario in scenarios:
        identifier = scenario.get("id") or "__".join(map(_slug, (
            scenario["model"], scenario["graph"], f"d{scenario['d']}",
            f"n{scenario['n']}", f"k{scenario['knowledge_fraction']}")))
        if identifier in seen:
            raise ValueError(f"duplicate scenario id: {identifier}")
        seen.add(identifier)
        result.append({**scenario, "id": identifier})
    return result


def _algorithms(scenario: dict, defaults: dict, smoke: bool) -> dict:
    dagma = {**DAGMA_DEFAULTS, "id": "dagma"}
    if smoke:
        dagma.update(T=1, warm_iter=100, max_iter=250, checkpoint=50)
    knowledge = {
        "knowledge_source": "oracle_true_graph", "knowledge_file": None,
        "knowledge_fraction": scenario["knowledge_fraction"],
        "knowledge_seed": int(defaults["knowledge_seed"]),
    }
    dagma_notreks = {
        **dagma, **knowledge, "id": "dagma_notreks",
        "trek_regularizer": "pst", "trek_function": "inv",
        "trek_kernel": "fast", "trek_weight": defaults["dagma_trek_weight"],
        "trek_log_terms": 2 * int(scenario["d"]),
        "trek_inverse_epsilon": 1e-8,
        "postselection_policy": "PS5_fixed_threshold_joint_feasible",
        "candidate_edge_pool": "fixed_threshold",
        "fixed_threshold": 0.30,
        "dag_constraint_active": True,
        "notreks_constraint_active": True,
    }
    flop_restarts = 2 if smoke else int(defaults["flop_restarts"])
    flop_notreks_restarts = (1 if smoke else int(
        defaults["flop_notreks_restarts"]))
    flop = {
        "id": "flop", "lambda_bic": 2.0, "restarts": flop_restarts,
        "search_timeout": None, "timeout": None,
    }
    flop_notreks = {
        **knowledge, "id": "flop_notreks", "lambda_bic": 2.0,
        "restarts": flop_notreks_restarts, "search_timeout": None,
        "timeout": None,
        "algorithm_seed": int(defaults["algorithm_seed"]),
        "search_strategy": "global_greedy",
        "max_sweeps": int(defaults["flop_notreks_max_sweeps"]),
        "signature_top_k": int(defaults["flop_notreks_signature_top_k"]),
        "signature_exploration_k": int(
            defaults["flop_notreks_signature_exploration_k"]),
        "max_signature_rounds": int(
            defaults["flop_notreks_max_signature_rounds"]),
        "initial_signature_mean_size": 3.0,
        "initial_signature_max_size": 6,
        "n_jobs": int(defaults.get("n_jobs", 1)),
        "search_version": "alternating_full_refit_b",
    }
    return {"flop": [flop], "flop_notreks": [flop_notreks],
            "dagma": [dagma], "dagma_notreks": [dagma_notreks]}


def benchpress_config(scenario: dict, defaults: dict, smoke: bool = False) -> dict:
    graph_id = f"graph_{scenario['id']}"
    data_id = f"data_{scenario['id']}"
    parameter_id = f"sem_{scenario['id']}"
    graph_method = scenario.get("graph_method", "er")
    graph_degree = float(scenario.get("graph_expected_degree", 2.0))
    model_method = scenario.get("model_method", "linear")
    sem_type = scenario.get("model_sem_type", "gauss")
    seeds = list(map(int, scenario["seeds"]))
    if not seeds or seeds != list(range(min(seeds), max(seeds) + 1)):
        raise ValueError("Benchpress scenario seeds must form a contiguous range")
    algorithms = _algorithms(scenario, defaults, smoke)
    return {
        "benchmark_setup": [{
            "title": scenario["id"],
            "data": [{"graph_id": graph_id, "parameters_id": parameter_id,
                      "data_id": data_id,
                      "seed_range": [min(seeds), max(seeds)]}],
            "evaluation": {"benchmarks": {
                "filename_prefix": f"notreks_benchmark/{scenario['id']}/",
                "show_seed": True, "errorbar": True, "errorbarh": False,
                "scatter": True, "path": True, "text": False,
                "ids": list(METHOD_IDS)}, "graph_true_plots": False,
                "graph_true_stats": True, "graph_plots": []},
        }],
        "resources": {
            "graph": {"gcastle_dag": [{
                "id": graph_id, "n_nodes": int(scenario["d"]),
                "n_edges": max(1, int(round(
                    int(scenario["d"]) * graph_degree / 2.0))),
                "method": graph_method}]},
            "parameters": {"sem_params": [{
                "id": parameter_id,
                "min": float(scenario.get("model_coefficient_min", 0.5)),
                "max": float(scenario.get("model_coefficient_max", 1.0))}]},
            "data": {"gcastle_iidsim": [{
                "id": data_id, "n": [int(scenario["n"])],
                "method": model_method, "sem_type": sem_type,
                "noise_scale": float(scenario.get("model_noise_scale", 1.0)),
                "standardized": True}]},
            "structure_learning_algorithms": algorithms,
        },
    }


def compile_configs(spec_path: Path, output_dir: Path, smoke: bool = False) -> None:
    spec = json.loads(spec_path.read_text())
    defaults_path = spec_path.parent / spec.get(
        "tuned_hyperparameters", "tuned_hyperparameters.json")
    defaults = json.loads(defaults_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in expand_scenarios(spec):
        config_path = output_dir / f"{scenario['id']}.json"
        config_path.write_text(json.dumps(
            benchpress_config(scenario, defaults, smoke), indent=2) + "\n")
        rows.append({**scenario, "config": str(config_path)})
    pd.DataFrame(rows).to_csv(output_dir / "scenario_manifest.csv", index=False)
    commands = [
        "snakemake --configfile " + row["config"]
        + " --cores ${NOTREKS_CORES:-1} --rerun-incomplete"
        for row in rows]
    (output_dir / "commands.txt").write_text("\n".join(commands) + "\n")
    print(f"wrote {len(rows)} Benchpress configs to {output_dir}")


def collect_results(manifest_path: Path, results_root: Path, output_path: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    files = list(results_root.rglob("joint_benchmarks.csv"))
    diagnostic_paths = list(results_root.rglob("*.diagnostics.json"))
    collected = []
    for row in manifest.to_dict("records"):
        scenario = str(row["id"])
        matches = [path for path in files if scenario in str(path)]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one joint_benchmarks.csv for {scenario}; "
                f"found {len(matches)}")
        frame = pd.read_csv(matches[0])
        for key in ("id", "model", "graph", "d", "n", "knowledge_fraction"):
            if key in row:
                frame["scenario" if key == "id" else key] = row[key]
        diagnostics = []
        for path in diagnostic_paths:
            if scenario not in str(path):
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if {"method_id", "seed"}.issubset(payload):
                diagnostics.append({
                    "id": payload["method_id"], "seed": payload["seed"],
                    "num_oracle_mi_pairs": payload.get(
                        "number_of_oracle_pairs"),
                    "num_supplied_mi_pairs": payload.get(
                        "number_of_supplied_constraints",
                        payload.get("number_of_no_trek_pairs")),
                })
        if diagnostics and {"id", "seed"}.issubset(frame.columns):
            frame = frame.merge(pd.DataFrame(diagnostics).drop_duplicates(
                ["id", "seed"]), on=["id", "seed"], how="left")
            for column in ("num_oracle_mi_pairs",):
                frame[column] = frame.groupby("seed")[column].transform("max")
            frame["num_supplied_mi_pairs"] = frame[
                "num_supplied_mi_pairs"].fillna(0)
        collected.append(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(collected, ignore_index=True).to_csv(output_path, index=False)
    print(f"collected {len(collected)} scenarios into {output_path}")


def analyse(results_csv: Path, output_dir: Path) -> None:
    frame = pd.read_csv(results_csv)
    for column in ("SHD_cpdag", "SHD_pattern", "SHD_skel", "time",
                   "TP_pattern", "FP_pattern", "FN_pattern",
                   "TP_skel", "FP_skel", "FN_skel"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for suffix in ("pattern", "skel"):
        tp, fp, fn = (f"TP_{suffix}", f"FP_{suffix}", f"FN_{suffix}")
        if {tp, fp, fn}.issubset(frame.columns):
            precision = frame[tp] / (frame[tp] + frame[fp]).replace(0, np.nan)
            recall = frame[tp] / (frame[tp] + frame[fn]).replace(0, np.nan)
            frame[f"F1_{suffix}"] = (2 * precision * recall /
                                      (precision + recall)).fillna(0.0)
    method_column = next((column for column in ("id", "algorithm")
                          if column in frame and set(METHOD_IDS).issubset(
                              set(frame[column]))), None)
    if method_column is None:
        raise ValueError("cannot identify a method-ID column containing all four methods")
    seed_columns = [name for name in ("scenario", "model", "graph", "d", "n",
                                      "knowledge_fraction", "seed") if name in frame]
    required = set(METHOD_IDS)
    if not required.issubset(set(frame[method_column])):
        raise ValueError(f"results are missing methods {sorted(required - set(frame[method_column]))}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for baseline, constrained in (("flop", "flop_notreks"),
                                  ("dagma", "dagma_notreks")):
        left = frame[frame[method_column] == baseline]
        right = frame[frame[method_column] == constrained]
        merged = left.merge(right, on=seed_columns, suffixes=("_baseline", "_notreks"))
        for metric in ("SHD_cpdag", "SHD_pattern", "F1_pattern", "SHD_skel",
                       "F1_skel", "time"):
            lhs, rhs = f"{metric}_baseline", f"{metric}_notreks"
            if lhs in merged and rhs in merged:
                merged[f"delta_{metric}"] = merged[rhs] - merged[lhs]
        merged["comparison"] = f"{baseline}_vs_{constrained}"
        pairs.append(merged)
    paired = pd.concat(pairs, ignore_index=True)
    paired.to_csv(output_dir / "paired_per_dataset.csv", index=False)
    delta_columns = [column for column in paired if column.startswith("delta_")]
    group_columns = ["comparison"] + [name for name in (
        "model", "graph", "d", "n", "knowledge_fraction") if name in paired]
    summary = paired.groupby(group_columns, dropna=False)[delta_columns].agg(
        ["mean", "std", "median", "count"])
    summary.to_csv(output_dir / "paired_summary.csv")
    _factor_and_causal_analysis(paired, output_dir)
    (output_dir / "REPORT.md").write_text(
        "# Paired NOTREKS benchmark\n\n"
        "Only matched FLOP/FLOP+NOTREKS and DAGMA/DAGMA+NOTREKS contrasts "
        "are summarized. Negative SHD deltas and positive F1 deltas favour "
        "the constrained method.\n\n" + _markdown(summary) + "\n")
    print((output_dir / "REPORT.md").read_text())


def _fit_effect_model(frame: pd.DataFrame, outcome: str, include_pairs: bool):
    usable = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome]).copy()
    columns = {"intercept": np.ones(len(usable))}
    for name in ("d", "n"):
        if name in usable:
            columns[f"log_{name}"] = np.log(usable[name].astype(float))
    if "knowledge_fraction" in usable:
        columns["knowledge_fraction"] = usable.knowledge_fraction.astype(float)
    if include_pairs and "num_supplied_mi_pairs_notreks" in usable:
        columns["log1p_supplied_pairs"] = np.log1p(
            usable.num_supplied_mi_pairs_notreks.astype(float))
    for name in ("model", "graph"):
        if name in usable:
            levels = sorted(map(str, usable[name].dropna().unique()))
            for level in levels[1:]:
                columns[f"{name}={level}"] = (usable[name].astype(str) == level).astype(float)
    design = pd.DataFrame(columns, index=usable.index)
    if len(usable) <= design.shape[1]:
        return pd.DataFrame()
    X = design.to_numpy(float)
    y = usable[outcome].to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    sigma2 = float(residual @ residual / dof)
    covariance = sigma2 * np.linalg.pinv(X.T @ X)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0))
    return pd.DataFrame({"term": design.columns, "estimate": beta,
                         "standard_error": standard_error,
                         "outcome": outcome,
                         "model": "mechanistic" if include_pairs else "total_design"})


def _factor_and_causal_analysis(paired: pd.DataFrame, output_dir: Path) -> None:
    outcomes = [name for name in ("delta_SHD_cpdag", "delta_F1_pattern")
                if name in paired]
    factor_rows = []
    for factor in ("model", "graph", "d", "n", "knowledge_fraction"):
        if factor not in paired:
            continue
        for keys, group in paired.groupby(["comparison", factor], dropna=False):
            for outcome in outcomes:
                factor_rows.append({"comparison": keys[0], "factor": factor,
                                    "level": keys[1], "outcome": outcome,
                                    "mean": group[outcome].mean(),
                                    "median": group[outcome].median(),
                                    "sample_size": group[outcome].notna().sum()})
    pd.DataFrame(factor_rows).to_csv(output_dir / "factor_effects.csv", index=False)
    models = []
    for comparison, group in paired.groupby("comparison"):
        for outcome in outcomes:
            for include_pairs in (False, True):
                fitted = _fit_effect_model(group, outcome, include_pairs)
                if not fitted.empty:
                    fitted.insert(0, "comparison", comparison)
                    models.append(fitted)
    pd.concat(models, ignore_index=True).to_csv(
        output_dir / "effect_models.csv", index=False) if models else None
    dag = """digraph benchmark_estimand {
  model -> data_distribution;
  graph_family -> true_graph;
  dimension -> true_graph;
  true_graph -> oracle_mi_pairs;
  knowledge_fraction -> supplied_mi_pairs;
  oracle_mi_pairs -> supplied_mi_pairs;
  sample_size -> data_information;
  data_distribution -> recovery;
  true_graph -> recovery;
  data_information -> recovery;
  method_notreks -> optimization_and_constraints;
  supplied_mi_pairs -> optimization_and_constraints;
  optimization_and_constraints -> recovery;
}
"""
    (output_dir / "causal_estimand.dot").write_text(dag)
    (output_dir / "CAUSAL_ANALYSIS.md").write_text(
        "# Factorial and causal-effect analysis\n\n"
        "The primary estimands are paired NOTREKS-minus-baseline changes. The "
        "full-factorial configured factors (model, graph family, dimension, "
        "sample size, and knowledge fraction) support controlled marginal "
        "contrasts. `effect_models.csv` contains a total-design model that "
        "does not adjust for MI-pair counts, and a mechanistic model that "
        "does. Realized MI-pair count is caused by graph family, dimension, "
        "and the sampled graph; it is therefore a mediator, not a harmless "
        "baseline covariate. Its coefficient must not be called the total "
        "causal effect of graph type. The DOT file records the assumed graph.\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile")
    compile_parser.add_argument("--spec", type=Path, required=True)
    compile_parser.add_argument("--output-dir", type=Path, required=True)
    compile_parser.add_argument("--smoke", action="store_true")
    analyse_parser = sub.add_parser("analyse")
    analyse_parser.add_argument("--results-csv", type=Path, required=True)
    analyse_parser.add_argument("--output-dir", type=Path, required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--manifest", type=Path, required=True)
    collect_parser.add_argument("--results-root", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "compile":
        compile_configs(args.spec, args.output_dir, args.smoke)
    elif args.command == "analyse":
        analyse(args.results_csv, args.output_dir)
    else:
        collect_results(args.manifest, args.results_root, args.output)


if __name__ == "__main__":
    main()
