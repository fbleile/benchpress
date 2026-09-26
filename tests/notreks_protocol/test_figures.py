import pandas as pd

from scripts.notreks_production_figures import pareto


def test_empty_pareto_subset_is_a_valid_noop(tmp_path):
    frame = pd.DataFrame({
        "experiment_id": ["main"],
        "d": [20],
        "graph_family": ["er"],
        "graph_density": [2],
        "n": [100],
        "method": ["flop"],
        "instance_id": ["x"],
        "run_id": ["x"],
    })
    pareto(frame, tmp_path)
    assert (tmp_path / "plot_data_paired_pareto.csv").exists()
