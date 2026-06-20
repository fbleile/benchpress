from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from independence_tests import (  # noqa: E402
    _cache_payload,
    evaluate_no_trek_independence,
    graph_implied_no_trek_pairs,
    independence_cache_key,
    pairwise_independence_candidates,
)


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    x0 = rng.normal(size=40)
    x1 = x0 + 0.05 * rng.normal(size=40)
    x2 = rng.normal(size=40)
    return pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})


def test_independence_cache_key_changes_with_parameters(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    df = _data()
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    X = df.to_numpy(dtype=float)
    base = _cache_payload(
        X,
        method="spearman",
        alpha=0.05,
        correction="none",
        columns=list(df.columns),
        dataset_path=data_path,
        extra_params=None,
    )
    changed_alpha = _cache_payload(
        X,
        method="spearman",
        alpha=0.1,
        correction="bonferroni",
        columns=list(df.columns),
        dataset_path=data_path,
        extra_params=None,
    )
    changed_method = dict(base)
    changed_method["independence_test"] = "pearson"
    changed_data = dict(base)
    changed_data["dataset_hash"] = "different"

    assert independence_cache_key(base) == independence_cache_key(changed_alpha)
    assert independence_cache_key(base) != independence_cache_key(changed_method)
    assert independence_cache_key(base) != independence_cache_key(changed_data)


def test_independence_cache_hit_miss_and_metadata(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    df = _data()
    data_path = tmp_path / "data.csv"
    cache_dir = tmp_path / "cache"
    df.to_csv(data_path, index=False)
    X = df.to_numpy(dtype=float)

    first = pairwise_independence_candidates(
        X,
        method="spearman",
        alpha=0.05,
        correction="bonferroni",
        columns=list(df.columns),
        cache_dir=cache_dir,
        dataset_path=data_path,
    )
    second = pairwise_independence_candidates(
        X,
        method="spearman",
        alpha=0.05,
        correction="bonferroni",
        columns=list(df.columns),
        cache_dir=cache_dir,
        dataset_path=data_path,
    )

    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert first.cache_key == second.cache_key
    assert first.pairs == second.pairs
    assert first.pvalues == second.pvalues
    assert (cache_dir / str(first.cache_key) / "metadata.json").is_file()
    assert (cache_dir / str(first.cache_key) / "accepted_pairs.csv").is_file()
    assert (cache_dir / str(first.cache_key) / "all_test_results.csv").is_file()


def test_cached_accepted_pairs_match_fresh_result(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    df = _data()
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    X = df.to_numpy(dtype=float)
    cached = pairwise_independence_candidates(
        X,
        method="pearson",
        alpha=0.05,
        correction="none",
        columns=list(df.columns),
        cache_dir=tmp_path / "cache",
        dataset_path=data_path,
    )
    fresh = pairwise_independence_candidates(
        X,
        method="pearson",
        alpha=0.05,
        correction="none",
        columns=list(df.columns),
    )
    assert cached.pairs == fresh.pairs
    assert cached.number_of_tests == fresh.number_of_tests


def test_gcastle_fisherz_runs_empty_conditioning_set() -> None:
    df = _data()
    X = df.to_numpy(dtype=float)
    result = pairwise_independence_candidates(
        X,
        method="gcastle_fisherz",
        alpha=0.05,
        correction="none",
        columns=list(df.columns),
    )
    assert result.number_of_tests == 3
    assert all(np.isfinite(row["p_value"]) for row in result.results or [])
    rows = {(row["i"], row["j"]): row for row in result.results or []}
    assert rows[(0, 1)]["p_value"] < 0.05
    assert (0, 1) not in result.pairs
    assert rows[(0, 2)]["p_value"] > 0.05
    assert (0, 2) in result.pairs
    assert rows[(0, 2)]["raw_test_name"] == "fisherz"


def test_gcastle_raw_cache_reused_for_alpha_and_correction(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    df = _data()
    data_path = tmp_path / "data.csv"
    cache_dir = tmp_path / "cache"
    df.to_csv(data_path, index=False)
    X = df.to_numpy(dtype=float)

    first = pairwise_independence_candidates(
        X,
        method="gcastle_fisherz",
        alpha=0.01,
        correction="none",
        columns=list(df.columns),
        cache_dir=cache_dir,
        dataset_path=data_path,
    )
    second = pairwise_independence_candidates(
        X,
        method="gcastle_fisherz",
        alpha=0.5,
        correction="bonferroni",
        columns=list(df.columns),
        cache_dir=cache_dir,
        dataset_path=data_path,
    )
    assert first.cache_key == second.cache_key
    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert len(second.pairs) <= len(first.pairs)
    metadata = (cache_dir / str(first.cache_key) / "metadata.json").read_text()
    assert "gcastle_fisherz" in metadata
    rows = (cache_dir / str(first.cache_key) / "all_test_results.csv").read_text()
    assert "raw_test_name" in rows


def test_no_trek_ground_truth_diagnostic_tiny_graph() -> None:
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = 1
    adj[2, 3] = 1
    true_no_trek = graph_implied_no_trek_pairs(adj)
    assert (1, 3) in true_no_trek
    assert (0, 1) not in true_no_trek

    summary = evaluate_no_trek_independence([(1, 3), (0, 1)], adj)
    assert summary["true_positive_no_trek_pairs"] == 1
    assert summary["false_positive_pairs"] == 1
    assert summary["precision"] == 0.5


def test_repeated_independence_settings_reuse_cache(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    df = _data()
    data_path = tmp_path / "data.csv"
    cache_dir = tmp_path / "cache"
    df.to_csv(data_path, index=False)
    X = df.to_numpy(dtype=float)

    statuses = []
    for _ in range(3):
        result = pairwise_independence_candidates(
            X,
            method="spearman",
            alpha=0.1,
            correction="benjamini-hochberg",
            columns=list(df.columns),
            cache_dir=cache_dir,
            dataset_path=data_path,
        )
        statuses.append(result.cache_status)
    assert statuses == ["miss", "hit", "hit"]
