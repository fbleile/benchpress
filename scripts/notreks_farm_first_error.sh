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
        if err.exists():
            text=err.read_text(errors='replace')
            # Snakemake prints a very large DAG before the useful exception.
            # Show both the beginning (environment/config context) and the
            # end (the actual rule/error) so this remains useful on clusters.
            if len(text) > 7500:
                text = text[:1200] + '\n... [stderr middle elided] ...\n' + text[-6200:]
            print('\n--- stderr ---\n'+text)
        break
else:
    print('No failed or timed-out task found.')
PY
