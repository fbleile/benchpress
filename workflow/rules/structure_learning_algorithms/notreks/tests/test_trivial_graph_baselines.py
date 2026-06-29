import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_ROOT = Path(__file__).resolve().parents[2]


def _load_script(method: str):
    path = METHOD_ROOT / method / "script.py"
    spec = importlib.util.spec_from_file_location(f"{method}_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    return module, spec.loader, path


def _run_with_fake_snakemake(method: str, tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_path = tmp_path / "data.csv"
    pd.DataFrame(np.arange(12).reshape(4, 3), columns=["a", "b", "c"]).to_csv(data_path, index=False)
    module, loader, _ = _load_script(method)

    class FakeSnakemake:
        input = {"data": str(data_path)}
        output = {
            "adjmat": str(tmp_path / "adjmat.csv"),
            "time": str(tmp_path / "time.txt"),
            "ntests": str(tmp_path / "ntests.txt"),
        }

    module.snakemake = FakeSnakemake()
    loader.exec_module(module)
    return (
        pd.read_csv(tmp_path / "adjmat.csv").to_numpy(dtype=int),
        int((tmp_path / "ntests.txt").read_text()),
        float((tmp_path / "time.txt").read_text()),
    )


def test_empty_graph_outputs_zero_adjacency(tmp_path: Path) -> None:
    adjmat, ntests, runtime = _run_with_fake_snakemake("empty_graph", tmp_path / "empty")
    assert adjmat.shape == (3, 3)
    assert np.array_equal(adjmat, np.zeros((3, 3), dtype=int))
    assert ntests == 0
    assert runtime >= 0


def test_complete_undirected_graph_outputs_all_offdiag(tmp_path: Path) -> None:
    adjmat, ntests, runtime = _run_with_fake_snakemake(
        "complete_undirected_graph",
        tmp_path / "complete",
    )
    assert adjmat.shape == (3, 3)
    assert np.array_equal(np.diag(adjmat), np.zeros(3, dtype=int))
    assert np.array_equal(adjmat, adjmat.T)
    assert int(adjmat.sum()) == 6
    assert ntests == 0
    assert runtime >= 0
