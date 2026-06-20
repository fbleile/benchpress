from __future__ import annotations

import csv
import json
from pathlib import Path

from grid import prepare_hparam_run
from jobfarm import make_manifest
from test_grid_expansion import _algorithm


def test_run_folder_uses_relative_paths_and_clean_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_dir = repo / "results/run"
    (repo / "resources/data/mydatasets").mkdir(parents=True)
    (repo / "resources/adjmat/myadjmats").mkdir(parents=True)
    config = {
        "notreks_hparam": {
            "run_name": "path-test",
            "fixed_data": {"d": 5, "n_values": [20], "seeds": [1]},
        },
        "structure_learning_algorithms": {
            "notreks": [_algorithm("method-a"), _algorithm("method-b")]
        },
    }
    grid_path = repo / "grid.json"
    grid_path.write_text(json.dumps(config))

    records, _ = prepare_hparam_run(repo, grid_path, run_dir, "cartesian")
    manifest_path = make_manifest(run_dir, records)

    assert (run_dir / "run_info.json").is_file()
    assert not (run_dir / "grid_meta.json").exists()
    assert (run_dir / "summaries").is_dir()

    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for key, value in row.items():
            if key.startswith("relative_"):
                assert value
                assert not Path(value).is_absolute()
                assert (run_dir / value).parent.exists()

    run_info = json.loads((run_dir / "run_info.json").read_text())
    assert "relative_" in run_info["path_convention"]
    assert not Path(run_info["fixed_data"]["metadata_path"]).is_absolute()

    cache_dirs = set()
    for record in records:
        expanded = json.loads((run_dir / record.relative_config_path).read_text())
        entries = expanded["resources"]["structure_learning_algorithms"]["notreks"]
        cache_dirs.update(entry["independence_cache_dir"] for entry in entries)
    assert len(cache_dirs) == 1
    assert cache_dirs.pop().endswith("results/run/independence_cache")


def test_docs_reference_dag_constraints_and_literature() -> None:
    module_dir = Path(__file__).resolve().parents[1]
    text = "\n".join(
        [
            (module_dir / "README.md").read_text(),
            (module_dir / "docs/HOWTO_LOCAL_AND_SLURM.md").read_text(),
        ]
    )
    for token in [
        "dag_seq=\"exp\"",
        "NOTEARS",
        "Zheng",
        "dag_seq=\"logdet\"",
        "DAGMA",
        "Bello",
        "dag_seq=\"scc_power_iteration\"",
        "SDCD",
        "Nazaret",
        "W * W",
        "independence_cache",
        "graph-implied no-trek",
    ]:
        assert token in text
