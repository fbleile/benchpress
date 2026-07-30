#!/usr/bin/env python3
"""Generate, select, and analyze tag-derived oracle experiments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
CONFIG_ROOT = ROOT / "configs/dagma_notreks_oracle"
DEFAULTS = dict(loss_type="l2", lambda1=.03, w_threshold=.3, T=5, mu_init=1.,
                mu_factor=.1, s="1.0,0.9,0.8,0.7,0.6", warm_iter=30000,
                max_iter=60000, lr=.0003, checkpoint=1000, beta_1=.99,
                beta_2=.999, timeout=None)


def paths(tag):
    return {
        "config": CONFIG_ROOT / "expanded" / f"{tag}_config.json",
        "manifest": CONFIG_ROOT / "expanded" / f"{tag}_manifest.csv",
        "joint": ROOT / "results/output" / tag / "benchmarks/dagma_notreks_oracle" / tag / "joint_benchmarks.csv",
        "report": ROOT / "results/dagma_notreks_oracle" / tag / "report",
    }


def _fixed_data(meta):
    tag, spec = meta["tag"], meta["data"]
    source_tag = meta.get("reuse_inputs_from")
    if source_tag:
        source_grid = json.loads(
            (CONFIG_ROOT / "grids" / f"{source_tag}.json").read_text())
        if source_grid["data"] != spec:
            raise ValueError("reused experiment data specification must match exactly")
        return [{"graph_id": f"g{source_tag}.csv", "parameters_id": None,
                 "data_id": f"{source_tag}/s{seed}.csv", "seed_range": None}
                for seed in spec["seeds"]]
    d, n = int(spec["d"]), int(spec["n"])
    if spec.get("independent_instances", False):
        names = [f"X{i + 1}" for i in range(d)]
        data_dir = ROOT / "resources/data/mydatasets" / tag
        graph_dir = ROOT / "resources/adjmat/myadjmats" / tag
        data_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)
        refs, seed_rows, hashes = [], [], []
        for dataset_seed in spec["seeds"]:
            seed_sequence = np.random.SeedSequence(int(dataset_seed))
            graph_ss, coefficient_ss, noise_ss = seed_sequence.spawn(3)
            graph_seed = int(graph_ss.generate_state(1, dtype=np.uint32)[0])
            coefficient_seed = int(
                coefficient_ss.generate_state(1, dtype=np.uint32)[0])
            noise_seed = int(noise_ss.generate_state(1, dtype=np.uint32)[0])
            graph_rng = np.random.default_rng(graph_seed)
            order = graph_rng.permutation(d)
            A = np.zeros((d, d), int)
            p = min(float(spec["expected_degree"]) / max(d - 1, 1), 1.)
            for a in range(d):
                for b in range(a + 1, d):
                    if graph_rng.random() < p:
                        A[order[a], order[b]] = 1
            coefficient_rng = np.random.default_rng(coefficient_seed)
            W = A * coefficient_rng.uniform(.5, 1., size=A.shape) * \
                coefficient_rng.choice([-1, 1], size=A.shape)
            noise = np.random.default_rng(noise_seed).normal(size=(n, d))
            X = noise @ np.linalg.inv(np.eye(d) - W)
            if spec["standardized"]:
                X = (X - X.mean(0)) / X.std(0)
            graph_name, data_name = f"g{dataset_seed}.csv", f"s{dataset_seed}.csv"
            graph_path, data_path = graph_dir / graph_name, data_dir / data_name
            pd.DataFrame(A, columns=names).to_csv(graph_path, index=False)
            pd.DataFrame(X, columns=names).to_csv(data_path, index=False)
            graph_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
            hashes.append(graph_hash)
            refs.append({"graph_id": f"{tag}/{graph_name}", "parameters_id": None,
                         "data_id": f"{tag}/{data_name}", "seed_range": None})
            seed_rows.append({
                "dataset_seed": int(dataset_seed), "experiment_seed": int(dataset_seed),
                "graph_seed": graph_seed, "coefficient_seed": coefficient_seed,
                "noise_seed": noise_seed, "graph_hash": graph_hash,
            })
        if len(hashes) > 1 and len(set(hashes)) == 1:
            raise ValueError("independent graph generation produced identical graph hashes")
        meta["_generated_seed_manifest"] = seed_rows
        return refs
    graph_rng = np.random.default_rng(1729)
    order = graph_rng.permutation(d)
    A = np.zeros((d, d), int)
    p = min(float(spec["expected_degree"]) / max(d - 1, 1), 1.)
    for a in range(d):
        for b in range(a + 1, d):
            if graph_rng.random() < p:
                A[order[a], order[b]] = 1
    weight_rng = np.random.default_rng(2718)
    W = A * weight_rng.uniform(.5, 1., size=A.shape) * weight_rng.choice([-1, 1], size=A.shape)
    names = [f"X{i + 1}" for i in range(d)]
    data_dir = ROOT / "resources/data/mydatasets" / tag
    graph_path = ROOT / "resources/adjmat/myadjmats" / f"g{tag}.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, columns=names).to_csv(graph_path, index=False)
    refs = []
    for seed in spec["seeds"]:
        noise = np.random.default_rng(seed).normal(size=(n, d))
        X = noise @ np.linalg.inv(np.eye(d) - W)
        if spec["standardized"]:
            X = (X - X.mean(0)) / X.std(0)
        filename = f"s{seed}.csv"
        pd.DataFrame(X, columns=names).to_csv(data_dir / filename, index=False)
        refs.append({"graph_id": f"g{tag}.csv", "parameters_id": None,
                     "data_id": f"{tag}/{filename}", "seed_range": None})
    return refs


def expand(grid_path):
    meta = json.loads(Path(grid_path).read_text())
    tag = meta["tag"]
    output = paths(tag)
    output["config"].parent.mkdir(parents=True, exist_ok=True)
    data_refs = _fixed_data(meta)
    smoke = {**meta.get("smoke_overrides", {}), **meta.get("local_overrides", {})}
    ordinary = {**DEFAULTS}
    if "w_threshold" in meta:
        ordinary["w_threshold"] = float(meta["w_threshold"])
    for key in ("T", "warm_iter", "max_iter", "checkpoint"):
        if key in smoke:
            ordinary[key] = smoke[key]
    dagma = [{**ordinary, "id": "dagma"}]
    notreks, rows = [], [{"family": "dagma", "id": "dagma", "trek_weight": ""}]
    for index, weight in enumerate(meta["trek_weight"]):
        item = {**ordinary, "id": f"dagma_notreks_w{index:02d}",
                "trek_regularizer": "pst",
                "trek_function": meta.get("trek_function", "inv"),
                "trek_weight": weight, "trek_log_terms": 2 * int(meta["data"]["d"]),
                "trek_inverse_epsilon": 1e-8, "knowledge_source": "oracle_true_graph",
                "knowledge_file": None}
        notreks.append(item)
        rows.append({"family": "dagma_notreks", "id": item["id"], "trek_weight": weight})
    algorithms = {
        "dagma": dagma, "dagma_notreks": notreks,
        "pc_mi_oracle": [{"id": "pc_mi_oracle", "alpha": .05, "variant": "stable",
                          "ci_test": "fisherz", "max_cond_set": int(meta["data"]["d"]) - 2,
                          "knowledge_source": "oracle_true_graph", "knowledge_file": None,
                          "timeout": None}],
        "flop": [{"id": "flop", "lambda_bic": 2., "restarts": smoke.get("flop_restarts", 50),
                  "search_timeout": None, "timeout": None}],
        "flop_notreks": [{
            "id": "flop_notreks", "lambda_bic": 2.,
            "restarts": smoke.get("flop_restarts", 1), "search_timeout": None,
            "timeout": smoke.get("flop_timeout", 120), "algorithm_seed": 0,
            "signature_top_k": 5, "max_signature_rounds": 20,
            "search_version": "alternating_full_refit_b",
            "knowledge_source": "oracle_true_graph", "knowledge_file": None,
        }],
    }
    if meta.get("dense_budget"):
        algorithms["dagma"][0].update(
            id="dagma_multistart", restarts=5, n_jobs=5,
            algorithm_seed=7319, initialization_scale=.05)
        algorithms["dagma_notreks"][0].update(
            id="dagma_notreks_multistart", restarts=5, n_jobs=5,
            algorithm_seed=7319, initialization_scale=.05)
        algorithms["flop"] = [{
            "id": "flop_official_r64", "lambda_bic": 2., "restarts": 64,
            "search_timeout": 1800, "timeout": None}]
        common = {
            "lambda_bic": 2., "search_timeout": 1800, "timeout": None,
            "algorithm_seed": 99173, "knowledge_source": "oracle_true_graph",
            "knowledge_file": None, "n_jobs": 16,
        }
        algorithms["flop_notreks"] = [
            {**common, "id": "flop_seeded_empty", "restarts": 64,
             "signature_top_k": 1, "signature_exploration_k": 0,
             "max_signature_rounds": 0, "initial_signature_mean_size": 0.,
             "initial_signature_max_size": 0,
             "search_version": "alternating_full_refit_b"},
            {**common, "id": "flop_notreks_a_r64", "restarts": 64,
             "signature_top_k": 1, "signature_exploration_k": 0,
             "max_signature_rounds": 0, "initial_signature_mean_size": 3.,
             "initial_signature_max_size": 6, "search_version": "fixed_signature_a"},
            {**common, "id": "flop_notreks_b_medium", "restarts": 64,
             "signature_top_k": 64, "signature_exploration_k": 16,
             "max_signature_rounds": 500, "initial_signature_mean_size": 3.,
             "initial_signature_max_size": 6,
             "search_version": "alternating_full_refit_b"},
            {**common, "id": "flop_notreks_b_large", "restarts": 128,
             "search_timeout": 3600, "signature_top_k": 96,
             "signature_exploration_k": 32, "max_signature_rounds": 1000,
             "initial_signature_mean_size": 3.5, "initial_signature_max_size": 7,
             "search_version": "alternating_full_refit_b"},
        ]
        rows = [
            {"family": "dagma", "id": "dagma_multistart", "trek_weight": ""},
            {"family": "dagma_notreks", "id": "dagma_notreks_multistart",
             "trek_weight": 10.0},
            {"family": "pc_mi_oracle", "id": "pc_mi_oracle", "trek_weight": ""},
            {"family": "flop", "id": "flop_official_r64", "trek_weight": ""},
        ] + [{"family": "flop_notreks", "id": x["id"], "trek_weight": ""}
             for x in algorithms["flop_notreks"]]
    rows += [{"family": "pc_mi_oracle", "id": "pc_mi_oracle", "trek_weight": ""},
             {"family": "flop", "id": "flop", "trek_weight": ""},
             {"family": "flop_notreks", "id": "flop_notreks", "trek_weight": ""}]
    if meta.get("dense_budget"):
        available = {item["id"] for family in algorithms.values() for item in family}
        rows = [row for row in rows if row["id"] in available]
    ids = [row["id"] for row in rows]
    config = {
        "benchmark_setup": [{"title": tag, "data": data_refs, "evaluation": {
            "benchmarks": {"filename_prefix": f"dagma_notreks_oracle/{tag}/",
                           "show_seed": True, "errorbar": True, "errorbarh": False,
                           "scatter": True, "path": True, "text": False, "ids": ids},
            "graph_true_plots": False,
            "graph_true_stats": not meta.get("dense_budget", False),
            "graph_plots": []}}],
        "resources": {"data": {}, "graph": {}, "parameters": {},
                      "structure_learning_algorithms": algorithms},
    }
    output["config"].write_text(json.dumps(config, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(output["manifest"], index=False)
    (output["manifest"].with_suffix(".json")).write_text(json.dumps(
        {"tag": tag, "data": meta["data"], "rows": rows,
         "seed_manifest": meta.get("_generated_seed_manifest", []),
         "config": str(output["config"].relative_to(ROOT))}, indent=2) + "\n")
    print(f"tag={tag}\nconfig={output['config']}\nmanifest={output['manifest']}\n"
          f"joint_benchmarks={output['joint']}")


def _metric(df, *names):
    for name in names:
        if name in df:
            return name
    return None


def _markdown(frame, include_index=True):
    table = frame.reset_index() if include_index else frame.copy()
    table.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple)
                     else str(col) for col in table.columns]
    values = [["" if pd.isna(x) else str(x) for x in row] for row in table.to_numpy()]
    return "\n".join([
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
        *["| " + " | ".join(row) + " |" for row in values],
    ])


def _derive_metrics(df):
    df = df.copy()
    for suffix in ("skel", "pattern"):
        tp, fp, fn = (f"TP_{suffix}", f"FP_{suffix}", f"FN_{suffix}")
        if {tp, fp, fn}.issubset(df.columns):
            df[f"precision_{suffix}"] = (
                df[tp] / (df[tp] + df[fp]).replace(0, np.nan)).fillna(0.0)
            df[f"recall_{suffix}"] = (
                df[tp] / (df[tp] + df[fn]).replace(0, np.nan)).fillna(0.0)
            p, r = df[f"precision_{suffix}"], df[f"recall_{suffix}"]
            df[f"F1_{suffix}"] = (
                2 * p * r / (p + r).replace(0, np.nan)).fillna(0.0)
            if suffix == "skel":
                df["SHD_skel"] = df[fp] + df[fn]
    return df


def _bootstrap_mean_interval(values, seed=20260725, repetitions=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def analyze(args):
    tag_paths = paths(args.tag) if args.tag else {}
    joint = Path(args.joint_benchmarks or tag_paths["joint"])
    manifest_path = Path(args.manifest or tag_paths["manifest"])
    output = Path(args.output_dir or tag_paths["report"])
    output.mkdir(parents=True, exist_ok=True)
    df, manifest = pd.read_csv(joint), pd.read_csv(manifest_path)
    df = _derive_metrics(df)
    family_col = _metric(df, "algorithm", "family")
    id_col = _metric(df, "id", "algorithm_id")
    if family_col is None:
        raise ValueError("joint benchmarks lacks algorithm/family column")
    shd = _metric(df, "SHD_pattern", "SHD_cpdag", "shd_cpdag", "SHD")
    metrics = [x for x in ["SHD_cpdag", "SHD_pattern", "SHD_skel", "precision_skel",
               "recall_skel", "F1_skel", "precision_pattern", "recall_pattern",
               "F1_pattern", "FPR_skel", "FNR_skel", "time", "runtime"] if x in df]
    grouping = [family_col, id_col] if id_col else [family_col]
    summary = df.groupby(grouping)[metrics].agg(["mean", "std", "count"])
    summary.to_csv(output / "algorithm_summary.csv")
    df.to_csv(output / "per_dataset_comparison.csv", index=False)
    local_results = output.parent
    raw_path = local_results / "raw_per_run.csv"
    if raw_path.exists():
        pd.read_csv(raw_path).to_csv(output / "raw_per_run.csv", index=False)
    for diagnostic_name in ("stage_diagnostics.csv", "threshold_diagnostics.csv",
                            "dataset_manifest.csv"):
        diagnostic_path = local_results / diagnostic_name
        if diagnostic_path.exists():
            pd.read_csv(diagnostic_path).to_csv(output / diagnostic_name, index=False)
    best, hyper = pd.DataFrame(), pd.DataFrame()
    if shd and id_col:
        nt = df[df[family_col] == "dagma_notreks"]
        candidates = nt.groupby(id_col)[shd].agg(["mean", "std", "count"])
        if len(candidates):
            hyper = candidates.reset_index().merge(
                manifest[["id", "trek_weight"]], left_on=id_col, right_on="id", how="left")
            runtime = _metric(nt, "runtime", "time")
            if runtime:
                hyper = hyper.merge(nt.groupby(id_col)[runtime].mean().rename("runtime_mean"),
                                    left_on=id_col, right_index=True)
            if {"trek_weight", "fraction_of_oracle_pairs_violated_after_threshold"}.issubset(nt):
                violations = nt.groupby("trek_weight")[
                    "fraction_of_oracle_pairs_violated_after_threshold"].mean().rename(
                        "mean_final_notreks_violation")
                hyper = hyper.merge(violations, on="trek_weight", how="left")
            selection = hyper.copy()
            if "F1_pattern" in nt:
                selection = selection.merge(
                    nt.groupby(id_col)["F1_pattern"].mean().rename("mean_F1_pattern"),
                    left_on=id_col, right_index=True)
            if runtime:
                selection["selection_runtime"] = selection["runtime_mean"]
            if "mean_F1_pattern" not in selection:
                selection["mean_F1_pattern"] = np.nan
            if "selection_runtime" not in selection:
                selection["selection_runtime"] = np.nan
            selection = selection.sort_values(
                ["mean", "mean_F1_pattern", "selection_runtime", "trek_weight"],
                ascending=[True, False, True, True], na_position="last")
            best = selection.head(1)
    best.to_csv(output / "best_configs.csv", index=False)
    weight_summary = pd.DataFrame()
    nt = df[df[family_col] == "dagma_notreks"].copy()
    if "trek_weight" in nt:
        summary_metrics = [x for x in (
            "SHD_pattern", "SHD_cpdag", "F1_skel", "F1_pattern",
            "estimated_edges", "number_of_oracle_pairs_violated_after_threshold",
            "runtime") if x in nt]
        records = []
        for weight, group in nt.groupby("trek_weight", sort=True):
            row = {"trek_weight": weight, "successful_runs": len(group)}
            for metric in summary_metrics:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                row.update({
                    f"{metric}_mean": values.mean(),
                    f"{metric}_std": values.std(ddof=1),
                    f"{metric}_median": values.median(),
                    f"{metric}_iqr": values.quantile(.75) - values.quantile(.25),
                    f"{metric}_min": values.min(),
                    f"{metric}_max": values.max(),
                    f"{metric}_count": values.count(),
                })
            records.append(row)
        weight_summary = pd.DataFrame(records)
    weight_summary.to_csv(output / "weight_summary.csv", index=False)
    direct_rows = []
    if shd and id_col:
        vanilla = df[df[family_col] == "dagma"]
        key = _metric(df, "data", "dataset", "seed")
        runtime_col = _metric(df, "runtime", "time")
        for algorithm_id, group in df[df[family_col] == "dagma_notreks"].groupby(id_col):
            weight = group["trek_weight"].iloc[0] if "trek_weight" in group else np.nan
            if not key or weight == 0:
                continue
            columns = [key, "SHD_cpdag", "SHD_pattern"]
            if runtime_col:
                columns.append(runtime_col)
            if "number_of_oracle_pairs_violated_after_threshold" in df:
                columns.append("number_of_oracle_pairs_violated_after_threshold")
            paired = group[columns].merge(
                vanilla[columns], on=key, suffixes=("_nt", "_vanilla"))
            cpdag_delta = paired["SHD_cpdag_nt"] - paired["SHD_cpdag_vanilla"]
            pattern_delta = paired["SHD_pattern_nt"] - paired["SHD_pattern_vanilla"]
            ci_low, ci_high = _bootstrap_mean_interval(pattern_delta)
            row = {
                "algorithm_id": algorithm_id,
                "trek_weight": weight,
                "mean_delta_SHD_cpdag": cpdag_delta.mean(),
                "mean_delta_SHD_pattern": pattern_delta.mean(),
                "median_delta_SHD_pattern": pattern_delta.median(),
                "paired_standard_error_SHD_pattern": pattern_delta.std(ddof=1) / np.sqrt(len(pattern_delta)),
                "bootstrap_95ci_low_SHD_pattern": ci_low,
                "bootstrap_95ci_high_SHD_pattern": ci_high,
                "wins": int((pattern_delta < 0).sum()),
                "ties": int((pattern_delta == 0).sum()),
                "losses": int((pattern_delta > 0).sum()),
            }
            if "number_of_oracle_pairs_violated_after_threshold_nt" in paired:
                row["mean_change_oracle_violations"] = (
                    paired["number_of_oracle_pairs_violated_after_threshold_nt"]
                    - paired["number_of_oracle_pairs_violated_after_threshold_vanilla"]
                ).mean()
            if runtime_col:
                row["runtime_ratio"] = (
                    paired[f"{runtime_col}_nt"] / paired[f"{runtime_col}_vanilla"]
                ).mean()
            direct_rows.append(row)
    pd.DataFrame(direct_rows).to_csv(output / "direct_dagma_comparison.csv", index=False)
    flop_notreks_rows = []
    key = _metric(df, "data", "dataset", "seed")
    runtime_col = _metric(df, "runtime", "time")
    if key and "SHD_pattern" in df:
        constrained = df[df[family_col] == "flop_notreks"]
        for other in ("flop", "pc_mi_oracle", "dagma_notreks"):
            baseline = df[df[family_col] == other]
            if other == "dagma_notreks" and "trek_weight" in baseline:
                baseline = baseline[baseline["trek_weight"] == 1.0]
            columns = [key, "SHD_pattern"]
            for optional in (runtime_col, "final_bic",
                             "number_of_oracle_pairs_violated_after_threshold"):
                if optional and optional in df and optional not in columns:
                    columns.append(optional)
            paired = constrained[columns].merge(
                baseline[columns], on=key, suffixes=("_flop_notreks", "_other"))
            if not len(paired):
                continue
            delta = paired["SHD_pattern_flop_notreks"] - paired["SHD_pattern_other"]
            row = {
                "comparison": f"flop_notreks_vs_{other}",
                "mean_delta_SHD_pattern": delta.mean(),
                "wins": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta > 0).sum()),
            }
            if runtime_col:
                row["runtime_ratio"] = (
                    paired[f"{runtime_col}_flop_notreks"]
                    / paired[f"{runtime_col}_other"]).mean()
            if "final_bic_flop_notreks" in paired:
                row["mean_delta_BIC"] = (
                    paired["final_bic_flop_notreks"] - paired["final_bic_other"]).mean()
            violation = "number_of_oracle_pairs_violated_after_threshold"
            if f"{violation}_flop_notreks" in paired:
                row["mean_change_no_trek_violations"] = (
                    paired[f"{violation}_flop_notreks"]
                    - paired[f"{violation}_other"]).mean()
            flop_notreks_rows.append(row)
    pd.DataFrame(flop_notreks_rows).to_csv(
        output / "direct_flop_notreks_comparison.csv", index=False)
    exp_inv = pd.DataFrame()
    if args.tag and args.tag.endswith("_inv"):
        exp_tag = args.tag.removesuffix("_inv")
        exp_path = ROOT / "results/dagma_notreks_oracle" / exp_tag / "raw_per_run.csv"
        inv_path = output.parent / "raw_per_run.csv"
        if exp_path.exists() and inv_path.exists():
            exp = pd.read_csv(exp_path)
            inv = pd.read_csv(inv_path)
            exp = exp[exp["algorithm"] == "dagma_notreks"]
            inv = inv[inv["algorithm"] == "dagma_notreks"]
            columns = ["seed", "trek_weight", "SHD_cpdag", "SHD_pattern",
                       "number_of_oracle_pairs_violated_after_threshold",
                       "runtime", "optimizer_iterations"]
            # Older exp results predate optimizer_iterations; recover them from
            # their saved stage diagnostics without rerunning the experiment.
            if "optimizer_iterations" not in exp:
                stage_path = exp_path.parent / "stage_diagnostics.csv"
                stage = pd.read_csv(stage_path)
                totals = stage[stage.algorithm == "dagma_notreks"].groupby(
                    ["seed", "trek_weight"]).iterations_performed.sum()
                exp = exp.merge(totals.rename("optimizer_iterations"),
                                on=["seed", "trek_weight"], how="left")
            exp_inv = exp[columns].merge(
                inv[columns], on=["seed", "trek_weight"],
                suffixes=("_exp", "_inv"))
            exp_inv["runtime_speedup_exp_over_inv"] = (
                exp_inv["runtime_exp"] / exp_inv["runtime_inv"])
    exp_inv.to_csv(output / "exp_vs_inv_comparison.csv", index=False)
    expected = len(json.loads(manifest_path.with_suffix(".json").read_text()).get("data", {}).get("seeds", [])) if manifest_path.with_suffix(".json").exists() else None
    warnings = []
    if shd is None:
        warnings.append("CPDAG metrics are unavailable.")
    if expected and any(df.groupby(family_col).size() < expected):
        warnings.append("Some methods have fewer datasets than expected.")
    metadata = json.loads(manifest_path.with_suffix(".json").read_text()) if manifest_path.with_suffix(".json").exists() else {}
    if metadata.get("data", {}).get("standardized") is False:
        warnings.append("This run contains non-standardized data.")
    if {"number_of_no_trek_pairs", "sidecar_pair_count"}.issubset(df.columns):
        if not (df["number_of_no_trek_pairs"] == df["sidecar_pair_count"]).all():
            warnings.append("The sidecar pair count does not match diagnostics.")
    report = ["# Oracle no-trek experiment report", "",
              f"- Tag: {args.tag or 'explicit paths'}", f"- Input: `{joint}`",
              f"- Manifest: `{manifest_path}`", f"- Methods: {', '.join(map(str, sorted(df[family_col].unique()))) }",
              "", "## Aggregate method table", "", _markdown(summary), "",
              "## Direct comparisons", ""]
    if shd:
        key = _metric(df, "data", "dataset", "seed")
        for other in ("dagma", "pc_mi_oracle", "flop", "flop_notreks"):
            left = df[df[family_col] == "dagma_notreks"]
            if id_col and len(best):
                left = left[left[id_col] == best.iloc[0][id_col]]
            right = df[df[family_col] == other]
            if key and len(left) and len(right):
                paired = left[[key, shd]].merge(right[[key, shd]], on=key, suffixes=("_nt", "_other"))
                delta = paired[f"{shd}_nt"] - paired[f"{shd}_other"]
                runtime = _metric(df, "runtime", "time")
                ratio_text = ""
                if runtime:
                    runtimes = left[[key, runtime]].merge(
                        right[[key, runtime]], on=key, suffixes=("_nt", "_other"))
                    ratio_text = f" Runtime ratio: {(runtimes[f'{runtime}_nt'] / runtimes[f'{runtime}_other']).mean():.3g}."
                report += [f"### DAGMA-NOTREKS vs {other}", "",
                           f"Mean/median SHD difference: {delta.mean():.3g} / {delta.median():.3g}; "
                           f"wins/ties/losses: {(delta < 0).sum()}/{(delta == 0).sum()}/{(delta > 0).sum()}."
                           f"{ratio_text}", ""]
    report += ["## Hyperparameter table", "", _markdown(hyper, include_index=False) if len(hyper) else "Unavailable.", "",
               "## Sanity warnings", "", *(f"- {x}" for x in warnings or ["None detected by the compact analyzer."]),
               "", "## Final verdict", "",
               f"Numerical comparisons above are diagnostic results from {expected or df['data'].nunique()} "
               "dataset(s); they do not support broad claims.", ""]
    (output / "report.md").write_text("\n".join(report))
    print(f"report={output / 'report.md'}")


def select(tag):
    p = paths(tag)
    df, manifest = pd.read_csv(p["joint"]), pd.read_csv(p["manifest"])
    family = "algorithm" if "algorithm" in df else "family"
    ident = "id" if "id" in df else "algorithm_id"
    shd = _metric(df, "SHD_cpdag", "shd_cpdag", "SHD")
    means = df[df[family] == "dagma_notreks"].groupby(ident)[shd].mean()
    selected = means.idxmin()
    out = CONFIG_ROOT / "expanded" / f"{tag}_selection.json"
    out.write_text(json.dumps({"tag": tag, "selected_dagma_notreks_id": selected,
                               "mean_validation_shd": means[selected]}, indent=2) + "\n")
    print(f"selection={out}\nselected={selected}")


def build_selected(tag, selected_tag):
    source, target = paths(tag), paths(selected_tag)
    config = json.loads(source["config"].read_text())
    selection = json.loads((CONFIG_ROOT / "expanded" / f"{tag}_selection.json").read_text())
    algs = config["resources"]["structure_learning_algorithms"]
    algs["dagma_notreks"] = [x for x in algs["dagma_notreks"]
                              if x["id"] == selection["selected_dagma_notreks_id"]]
    families = ("dagma", "dagma_notreks", "pc_mi_oracle", "flop", "flop_notreks")
    ids = [algs[name][0]["id"] for name in families]
    config["benchmark_setup"][0]["title"] = selected_tag
    config["benchmark_setup"][0]["evaluation"]["benchmarks"]["filename_prefix"] = f"dagma_notreks_oracle/{selected_tag}/"
    config["benchmark_setup"][0]["evaluation"]["benchmarks"]["ids"] = ids
    target["config"].parent.mkdir(parents=True, exist_ok=True)
    target["config"].write_text(json.dumps(config, indent=2) + "\n")
    pd.DataFrame([{"family": name, "id": algs[name][0]["id"],
                   "trek_weight": algs[name][0].get("trek_weight", "")}
                  for name in families]).to_csv(target["manifest"], index=False)
    print(f"config={target['config']}\nmanifest={target['manifest']}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("expand"); exp.add_argument("--grid", required=True)
    ana = sub.add_parser("analyze"); ana.add_argument("--tag"); ana.add_argument("--joint-benchmarks")
    ana.add_argument("--manifest"); ana.add_argument("--output-dir")
    sel = sub.add_parser("select"); sel.add_argument("--tag", required=True)
    build = sub.add_parser("build-selected"); build.add_argument("--tag", required=True)
    build.add_argument("--selected-tag", default="oracle_selected_benchmark")
    args = parser.parse_args()
    if args.command == "expand": expand(args.grid)
    elif args.command == "analyze": analyze(args)
    elif args.command == "select": select(args.tag)
    else: build_selected(args.tag, args.selected_tag)


if __name__ == "__main__":
    main()
