#!/usr/bin/env python
"""Run the focused NOTREKS tests without requiring pytest."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TESTS_DIR.parent
TOOLS_DIR = MODULE_DIR / "tools"
for path in (TESTS_DIR, MODULE_DIR, TOOLS_DIR):
    sys.path.insert(0, str(path))

import test_farm as farm_tests  # noqa: E402
import test_component as component_tests  # noqa: E402
import test_pairs as pair_tests  # noqa: E402


def main() -> None:
    for module in (component_tests, pair_tests):
        for name, test in vars(module).items():
            if name.startswith("test_") and callable(test):
                test()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        farm_tests.test_360_tasks_are_farm_tasks_not_slurm_array(path / "farm-360")
        farm_tests.test_cpdag_config_is_required(path / "farm-cpdag")
        farm_tests.test_task_timeout_is_recorded_and_resume_skips_success(path / "farm-resume")

    print("All NOTREKS tests passed")


if __name__ == "__main__":
    main()
