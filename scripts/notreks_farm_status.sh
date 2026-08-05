#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:?usage: $0 RUN_DIR}"
python "$(dirname "${BASH_SOURCE[0]}")/notreks_farm_status.py" "$RUN_DIR"
