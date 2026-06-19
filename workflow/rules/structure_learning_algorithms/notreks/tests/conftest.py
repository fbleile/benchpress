"""Test path setup for NOTREKS-local modules."""

from __future__ import annotations

import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = MODULE_DIR / "tools"
for path in (MODULE_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
