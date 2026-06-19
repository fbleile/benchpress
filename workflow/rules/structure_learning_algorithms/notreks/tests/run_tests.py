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
import test_jobfarm_manifest as jobfarm_tests  # noqa: E402
import test_optimizer as optimizer_tests  # noqa: E402
import test_selection as selection_tests  # noqa: E402


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
        grid_tests.test_prepare_writes_one_config_per_template_and_shared_fixed_data(
            path / "grid"
        )
        jobfarm_tests.test_manifest_and_command_file_are_unique_and_nonempty(
            path / "jobfarm"
        )
        jobfarm_tests.test_slurm_script_rejects_missing_command_file(path / "slurm")
        for name in ("selection", "selection-missing", "inject"):
            (path / name).mkdir()
        selection_tests.test_selection_chooses_lowest_mean_cpdag_shd(path / "selection")
        selection_tests.test_selection_fails_without_cpdag_metric(path / "selection-missing")
        selection_tests.test_inject_best_preserves_non_notreks_algorithms(path / "inject")

    print("All NOTREKS tests passed")


if __name__ == "__main__":
    main()
