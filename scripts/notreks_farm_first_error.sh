#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:?usage: $0 RUN_DIR}"
python - "$RUN_DIR" <<'PY'
import json, pathlib, sys
run=pathlib.Path(sys.argv[1])
for path in sorted((run/'tasks').glob('*/status.json')):
    try: row=json.loads(path.read_text())
    except json.JSONDecodeError: continue
    if row.get('status') in {'failed','TIMEOUT'}:
        print(json.dumps({k:row.get(k) for k in ('task_id','config_path','command','exit_code','status','exception')}, indent=2))
        err=path.parent/'stderr.log'
        if err.exists(): print('\n--- stderr ---\n'+err.read_text()[:6000])
        break
else:
    print('No failed or timed-out task found.')
PY
