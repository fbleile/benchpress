import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[1]
METHOD_SCRIPT = MODULE_DIR.parent / "marginal_trek_graph" / "script.py"

sys.path.append(str(MODULE_DIR))


def _load_baseline_module():
    spec = importlib.util.spec_from_file_location("marginal_trek_graph_script", METHOD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(77)
    x0 = rng.normal(size=80)
    x1 = x0 + 0.05 * rng.normal(size=80)
    x2 = rng.normal(size=80)
    x3 = x2 + 0.05 * rng.normal(size=80)
    return pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3})


def test_marginal_trek_graph_outputs_undirected_contract(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    module = _load_baseline_module()
    data_path = tmp_path / "data.csv"
    adjmat_path = tmp_path / "adjmat.csv"
    time_path = tmp_path / "time.txt"
    ntests_path = tmp_path / "ntests.txt"
    _data().to_csv(data_path, index=False)

    module.run_marginal_trek_graph(
        data_path,
        adjmat_path,
        time_path,
        ntests_path,
        independence_test="spearman",
        independence_alpha=0.05,
        independence_correction="none",
        independence_cache_dir=tmp_path / "cache",
    )

    adjmat = pd.read_csv(adjmat_path).to_numpy(dtype=int)
    assert adjmat.shape == (4, 4)
    assert np.array_equal(adjmat, adjmat.T)
    assert np.array_equal(np.diag(adjmat), np.zeros(4, dtype=int))
    assert int(ntests_path.read_text()) == 4 * 3 // 2
    assert float(time_path.read_text()) >= 0.0
    assert adjmat_path.is_file()
