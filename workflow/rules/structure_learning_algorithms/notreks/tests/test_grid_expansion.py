from __future__ import annotations

import json
from pathlib import Path

from grid import (
    assert_scalar_algorithm,
    expand_template,
    prepare_hparam_run,
)


def _algorithm(template_id: str) -> dict:
    return {
        "id": template_id,
        "function_class": "linear",
        "score": "least_squares",
        "dag_seq": "exp",
        "dag_reg": 1.0,
        "dag_s": 1.0,
        "trek_seq": "exp",
        "trek_reg": [0.1, 1.0],
        "regularizer": "l1",
        "regularizer_scale": [0.01, 0.1],
        "independence_test": "spearman",
        "independence_alpha": 0.05,
        "independence_correction": "benjamini-hochberg",
        "seed": 1,
        "max_iter": 10,
        "lr": 0.0003,
        "path_steps": 1,
        "mu_init": 1.0,
        "mu_factor": 0.1,
        "warm_iter": 10,
        "tol": 1e-6,
        "threshold": 0.1,
        "timeout": None,
    }


def test_cartesian_grid_expansion() -> None:
    expanded = expand_template(_algorithm("cartesian"), "cartesian")
    assert len(expanded) == 4
    assert len({entry["id"] for entry in expanded}) == 4
    for entry in expanded:
        assert_scalar_algorithm(entry)


def test_zip_grid_expansion() -> None:
    expanded = expand_template(_algorithm("zipped"), "zip")
    assert len(expanded) == 2
    assert [(entry["trek_reg"], entry["regularizer_scale"]) for entry in expanded] == [
        (0.1, 0.01),
        (1.0, 0.1),
    ]


def test_zip_grid_rejects_inconsistent_lengths() -> None:
    template = _algorithm("bad-zip")
    template["max_iter"] = [10, 20, 30]
    try:
        expand_template(template, "zip")
    except ValueError as exc:
        assert "equal lengths" in str(exc)
    else:
        raise AssertionError("Expected inconsistent zip lengths to fail")


def test_prepare_writes_one_config_per_template_and_shared_fixed_data(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_dir = repo / "results/run"
    (repo / "resources/data/mydatasets").mkdir(parents=True)
    (repo / "resources/adjmat/myadjmats").mkdir(parents=True)
    config = {
        "notreks_hparam": {
            "run_name": "grid-test",
            "fixed_data": {"d": 5, "n_values": [20], "seeds": [1]},
        },
        "structure_learning_algorithms": {
            "notreks": [_algorithm("method-a"), _algorithm("method-b")]
        },
    }
    grid_path = repo / "grid.json"
    grid_path.write_text(json.dumps(config))

    optimizer = (
        Path(__file__).resolve().parents[1] / "optimizer.py"
    )
    before = optimizer.read_bytes()
    records, fixed = prepare_hparam_run(repo, grid_path, run_dir, "cartesian")
    assert optimizer.read_bytes() == before

    assert len(records) == 2
    assert len({record.relative_config_path for record in records}) == 2
    assert fixed.data_id == "grid-test"
    assert (run_dir / "run_info.json").is_file()
    assert not (run_dir / "grid_meta.json").exists()
    data_ids = set()
    graph_ids = set()
    for record in records:
        config_path = run_dir / record.relative_config_path
        expanded_config = json.loads(config_path.read_text())
        setup = expanded_config["benchmark_setup"][0]["data"][0]
        data_ids.add(setup["data_id"])
        graph_ids.add(setup["graph_id"])
        entries = expanded_config["resources"]["structure_learning_algorithms"]["notreks"]
        assert len(entries) == 4
        for entry in entries:
            assert_scalar_algorithm(entry)
    assert data_ids == {fixed.data_id}
    assert graph_ids == {fixed.graph_id}
