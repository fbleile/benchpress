"""Create one benchmark result row without requiring the Benchpress R stack.

The upstream workflow historically delegated this tiny operation to an R
script.  Cluster environments used for the NOTREKS farm do not necessarily
ship ``argparser``, BiDAG, or pcalg, so keep the summary rule self-contained.
The canonical R path remains available elsewhere; this Python implementation
uses the same basic edge-count conventions and explicitly marks its backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(true_path: str, estimate_path: str) -> dict[str, object]:
    truth = pd.read_csv(true_path).to_numpy(dtype=int) != 0
    estimate = pd.read_csv(estimate_path).to_numpy(dtype=int) != 0
    if truth.shape != estimate.shape or truth.shape[0] != truth.shape[1]:
        raise ValueError("true and estimated adjacency matrices must be square and equal-sized")
    upper = np.triu(np.ones_like(truth, dtype=bool), 1)
    true_skel = truth | truth.T
    est_skel = estimate | estimate.T
    tp_s = int(np.sum(true_skel[upper] & est_skel[upper]))
    fp_s = int(np.sum(~true_skel[upper] & est_skel[upper]))
    fn_s = int(np.sum(true_skel[upper] & ~est_skel[upper]))
    directed_tp = int(np.sum(estimate & truth))
    directed_fp = int(np.sum(estimate & ~truth))
    directed_fn = int(np.sum(~estimate & truth))
    pattern_shd = int(np.sum(np.abs(estimate.astype(int) - truth.astype(int))) / 2)
    precision_s = tp_s / (tp_s + fp_s) if tp_s + fp_s else 0.0
    recall_s = tp_s / (tp_s + fn_s) if tp_s + fn_s else 0.0
    precision_p = directed_tp / (directed_tp + directed_fp) if directed_tp + directed_fp else 0.0
    recall_p = directed_tp / (directed_tp + directed_fn) if directed_tp + directed_fn else 0.0
    f1_s = 2 * precision_s * recall_s / (precision_s + recall_s) if precision_s + recall_s else 0.0
    f1_p = 2 * precision_p * recall_p / (precision_p + recall_p) if precision_p + recall_p else 0.0
    return {
        "TP_pattern": directed_tp, "FP_pattern": directed_fp,
        "FN_pattern": directed_fn, "TP_skel": tp_s, "FP_skel": fp_s,
        "FN_skel": fn_s, "SHD_pattern": pattern_shd,
        # Without pcalg, use the conservative directed-pattern proxy and
        # label the backend explicitly rather than silently claiming CPDAG
        # equivalence semantics.
        "SHD_cpdag": pattern_shd, "SHD_skel": fp_s + fn_s,
        "precision_pattern": precision_p, "recall_pattern": recall_p,
        "F1_pattern": f1_p, "precision_skel": precision_s,
        "recall_skel": recall_s, "F1_skel": f1_s,
        "TPR_pattern": recall_p, "FPR_pattern": np.nan,
        "FPR_skel": np.nan, "FNR_skel": np.nan,
        "graph_type": "dag_or_cpdag_python_proxy",
        "metrics_backend": "python_fallback_pattern",
    }


algorithm = snakemake.params["alg"]
config = json.loads(Path(snakemake.params["config"]).read_text())
algorithm_config = config["resources"]["structure_learning_algorithms"][algorithm][0]

row = _metrics(snakemake.input["adjmat_true"], snakemake.input["adjmat_est"])
row.update({
    "seed": snakemake.wildcards.get("seed"),
    "algorithm": algorithm,
    "adjmat": snakemake.wildcards.get("adjmat"),
    "parameters": snakemake.wildcards.get("bn"),
    "data": snakemake.wildcards.get("data"),
    "time": Path(snakemake.input["time"]).read_text().strip(),
    "ntests": Path(snakemake.input["ntests"]).read_text().strip(),
})
for key, value in algorithm_config.items():
    if isinstance(value, (list, tuple)):
        value = ";".join(map(str, value))
    row[key] = value
pd.DataFrame([row]).to_csv(snakemake.output["res"], index=False)
