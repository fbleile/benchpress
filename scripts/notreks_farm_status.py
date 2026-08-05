#!/usr/bin/env python
from __future__ import annotations
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
counts = {key: 0 for key in ("total", "completed", "running", "pending", "failed", "timeout")}
expected_total = 0
farm_config = run / "farm_config.json"
if farm_config.exists():
    try: expected_total = int(json.loads(farm_config.read_text()).get("task_count", 0))
    except (ValueError, json.JSONDecodeError): pass
for path in sorted((run / "tasks").glob("*/status.json")):
    try:
        row = json.loads(path.read_text())
    except json.JSONDecodeError:
        row = {"status": "failed"}
    counts["total"] += 1
    state = str(row.get("status", "pending")).lower()
    if state == "success": counts["completed"] += 1
    elif state == "running": counts["running"] += 1
    elif state in {"timeout", "timed_out"}: counts["timeout"] += 1
    elif state == "failed": counts["failed"] += 1
    else: counts["pending"] += 1
if expected_total > counts["total"]:
    counts["pending"] += expected_total - counts["total"]
    counts["total"] = expected_total
print(" ".join(f"{key}={value}" for key, value in counts.items()))
analysis = run / "analysis_status.json"
print("analysis=" + (json.loads(analysis.read_text()).get("status", "pending") if analysis.exists() else "pending"))
