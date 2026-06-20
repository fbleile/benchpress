#!/usr/bin/env python
"""Run NOTREKS tests without requiring pytest."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TESTS_DIR.parent
TOOLS_DIR = MODULE_DIR / "tools"
for path in (TESTS_DIR, MODULE_DIR, TOOLS_DIR):
    sys.path.insert(0, str(path))

import test_grid_expansion as grid_tests  # noqa: E402
import test_independence_cache as independence_tests  # noqa: E402
import test_jobfarm_manifest as jobfarm_tests  # noqa: E402
import test_optimizer as optimizer_tests  # noqa: E402
import test_paths as path_tests  # noqa: E402
import test_selection as selection_tests  # noqa: E402
import test_validation_config as validation_tests  # noqa: E402


def main() -> None:
    optimizer_functions = [
        value
        for name, value in vars(optimizer_tests).items()
        if name.startswith("test_") and callable(value)
    ]
    for test in optimizer_functions:
        test()

    grid_tests.test_cartesian_grid_expansion()
    grid_tests.test_zip_grid_expansion()
    grid_tests.test_zip_grid_rejects_inconsistent_lengths()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        independence_tests.test_independence_cache_key_changes_with_parameters(path / "key")
        independence_tests.test_independence_cache_hit_miss_and_metadata(path / "hit")
        independence_tests.test_cached_accepted_pairs_match_fresh_result(path / "fresh")
        independence_tests.test_gcastle_fisherz_runs_empty_conditioning_set()
        independence_tests.test_gcastle_raw_cache_reused_for_alpha_and_correction(path / "gcastle-cache")
        independence_tests.test_no_trek_ground_truth_diagnostic_tiny_graph()
        independence_tests.test_repeated_independence_settings_reuse_cache(path / "reuse")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        grid_tests.test_prepare_writes_one_config_per_template_and_shared_fixed_data(
            path / "grid"
        )
        jobfarm_tests.test_manifest_and_command_file_are_unique_and_nonempty(
            path / "jobfarm"
        )
        jobfarm_tests.test_slurm_script_rejects_missing_command_file(path / "slurm")
        path_tests.test_run_folder_uses_relative_paths_and_clean_metadata(path / "paths")
        path_tests.test_docs_reference_dag_constraints_and_literature()
        for name in ("selection", "selection-missing", "selection-tie", "selection-move", "inject"):
            (path / name).mkdir()
        selection_tests.test_selection_chooses_lowest_mean_cpdag_shd(path / "selection")
        selection_tests.test_selection_fails_without_cpdag_metric(path / "selection-missing")
        selection_tests.test_selection_uses_secondary_tie_breaker(path / "selection-tie")
        selection_tests.test_selection_works_after_moving_run_folder(path / "selection-move")
        selection_tests.test_inject_best_preserves_non_notreks_algorithms(path / "inject")
        validation_tests.test_prepare_validation_tiny_writes_one_config_and_manifest(path / "validation-tiny")
        validation_tests.test_validation_tiny_expected_run_counts(path / "validation-counts")
        validation_tests.test_prepare_validation_local10_avoids_logdet_power_iter_duplicates(path / "validation-local10")
        validation_tests.test_validation_tiny_notreks_paths_are_short(path / "validation-paths")
        validation_tests.test_select_by_method_family_and_write_final_config(path / "validation-select")

    print("All NOTREKS tests passed")


if __name__ == "__main__":
    main()
