from pathlib import Path
import sys

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from hyperparam_analysis import join_joint_to_manifest, select_best_thresholds  # noqa: E402


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "algorithm_id": "gcastle_pc__grid000",
                "path_id": "gcastle_pc__grid000",
                "method_family": "gcastle_pc",
                "base_method": "gcastle_pc",
                "param.threshold": 0.1,
            },
            {
                "algorithm_id": "notreks__grid044",
                "path_id": "n044",
                "method_family": "notreks",
                "base_method": "notreks",
                "param.threshold": 0.2,
            },
        ]
    )


def test_joint_manifest_join_uses_full_algorithm_id() -> None:
    joint = pd.DataFrame(
        [
            {
                "id": "gcastle_pc__grid000",
                "alg_id": pd.NA,
                "SHD_pattern": 4,
                "time": 1.0,
            }
        ]
    )
    merged = join_joint_to_manifest(
        joint,
        _manifest(),
        joint_path=Path("joint_benchmarks.csv"),
        manifest_path=Path("manifest.json"),
    )
    assert list(merged["algorithm_id"]) == ["gcastle_pc__grid000"]
    assert list(merged["result_id"].astype(str)) == ["gcastle_pc__grid000"]


def test_joint_manifest_join_uses_short_path_id() -> None:
    joint = pd.DataFrame(
        [
            {
                "id": "n044",
                "alg_id": pd.NA,
                "SHD_pattern": 3,
                "time": 2.0,
            }
        ]
    )
    merged = join_joint_to_manifest(
        joint,
        _manifest(),
        joint_path=Path("joint_benchmarks.csv"),
        manifest_path=Path("manifest.json"),
    )
    assert list(merged["algorithm_id"]) == ["notreks__grid044"]
    assert list(merged["result_id"].astype(str)) == ["n044"]


def test_thresholds_use_curve_param_or_manifest_threshold() -> None:
    rows = pd.DataFrame(
        [
            {
                "algorithm_id": "notreks__grid044",
                "method_family": "notreks",
                "base_method": "notreks",
                "curve_param": "threshold",
                "curve_value": 0.3,
                "param.threshold": 0.2,
                "SHD_pattern": 5,
                "time": 1.0,
            },
            {
                "algorithm_id": "gcastle_pc__grid000",
                "method_family": "gcastle_pc",
                "base_method": "gcastle_pc",
                "curve_param": "not_threshold",
                "curve_value": 999,
                "param.threshold": 0.1,
                "SHD_pattern": 4,
                "time": 1.0,
            },
        ]
    )
    curve, best = select_best_thresholds(rows, ["SHD_pattern", "time"], "SHD_pattern")
    assert set(curve["algorithm_id"]) == {"gcastle_pc__grid000", "notreks__grid044"}
    assert set(curve["thresh"]) == {0.1, 0.3}

    manifest_only = rows.drop(columns=["curve_param", "curve_value"])
    curve, best = select_best_thresholds(manifest_only, ["SHD_pattern", "time"], "SHD_pattern")
    assert set(curve["threshold_source"]) == {"manifest_config_threshold"}
    assert set(curve["thresh"]) == {0.1, 0.2}
