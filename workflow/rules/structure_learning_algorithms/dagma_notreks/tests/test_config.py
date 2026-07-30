import json
from pathlib import Path

import pandas as pd
from workflow.rules.structure_learning_algorithms.dagma_notreks.tools.cli import _derive_metrics


ROOT = Path(__file__).resolve().parents[5]


def test_production_configuration_is_unambiguous():
    config = json.loads((ROOT / "configs/dagma_notreks_oracle/grids/"
                         "production_pipeline.json").read_text())
    assert config["method"] == "dagma_notreks"
    assert config["dag_constraint"] == "logdet"
    assert config["lambda1"] == 0.03
    assert config["trek_function"] == "inv"
    assert config["trek_weight"] == 10
    assert config["restarts"] == 5
    assert config["screening_floor"] == 0.01
    assert config["postprocessing"] == "fixed_order_parent_shrink"
    assert config["restart_selection"] == "postprocessed_bic"


def test_report_metrics_for_empty_estimate_are_zero_not_missing():
    raw = pd.DataFrame([{
        "TP_skel": 0, "FP_skel": 0, "FN_skel": 5,
        "TP_pattern": 0, "FP_pattern": 0, "FN_pattern": 5,
    }])
    result = _derive_metrics(raw).iloc[0]
    assert result["precision_skel"] == result["F1_skel"] == 0
    assert result["precision_pattern"] == result["F1_pattern"] == 0
    assert result["SHD_skel"] == 5


def test_inverse_is_user_facing_default_and_exp_remains_supported():
    schema = json.loads(
        (ROOT / "workflow/rules/structure_learning_algorithms/dagma_notreks/schema.json"
         ).read_text())
    prop = schema["items"]["properties"]["trek_function"]
    assert prop["default"] == "inv"
    assert set(prop["enum"]) == {"inv", "exp", "log", "binom"}
    config = json.loads((ROOT / "configs/dagma_notreks_oracle/grids/"
                         "production_pipeline.json").read_text())
    assert config["trek_function"] == "inv"


def test_only_distinct_retained_configuration_categories_exist():
    names = {path.name for path in (
        ROOT / "configs/dagma_notreks_oracle/grids").glob("*.json")}
    assert names == {
        "production_pipeline.json",
        "main_comparison.json",
        "inv_dag_notreks_benchmark.json",
        "historical_regression.json",
        "oracle_tiny_smoke.json",
    }
