#!/usr/bin/env bash
set -euo pipefail

# Install the Linux wheel used by the ordinary FLOP rule.  If a wheel is not
# available for the active Python, build the checked-in source distribution.
# This is deliberately explicit: host mode must never silently substitute a
# different FLOP implementation.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENDOR_DIR="$ROOT_DIR/workflow/rules/structure_learning_algorithms/flop_notreks/vendor/flopsearch/flop_python"
MATURIN_BIN="${MATURIN_BIN:-}"
if [[ -z "$MATURIN_BIN" && -x "$ROOT_DIR/.venv-local-smoke/bin/maturin" ]]; then
  MATURIN_BIN="$ROOT_DIR/.venv-local-smoke/bin/maturin"
fi
CARGO_BIN="${CARGO_BIN:-$(command -v cargo || true)}"
if [[ -z "$CARGO_BIN" && -x "$HOME/.cargo/bin/cargo" ]]; then
  CARGO_BIN="$HOME/.cargo/bin/cargo"
fi
if [[ -n "$CARGO_BIN" ]]; then
  export PATH="$(dirname "$CARGO_BIN"):$PATH"
fi

backend_available() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import flopsearch
import numpy as np

# Importing an old wheel is not enough: the production path must expose the
# Rust global-greedy backend introduced in this checkout.
data = np.random.default_rng(1).normal(size=(32, 3))
flopsearch.flop_notreks(
    data, 2.0, [], restarts=0, seed=1,
    search_version="global_greedy_rust", return_diagnostics=True)
PY
}

if backend_available
then
  echo "flopsearch global_greedy_rust backend already available"
  exit 0
fi

echo "Installing flopsearch==0.3.0 into $($PYTHON_BIN -c 'import sys; print(sys.prefix)')"
if "$PYTHON_BIN" -m pip install --no-cache-dir "flopsearch==0.3.0"; then
  if backend_available; then
    echo "flopsearch global_greedy_rust backend installed"
    exit 0
  fi
  echo "Installed wheel lacks global_greedy_rust; building vendored source." >&2
fi

[[ -n "$MATURIN_BIN" && -x "$MATURIN_BIN" ]] || {
  echo "ERROR: no compatible flopsearch wheel and maturin is unavailable." >&2
  echo "Set MATURIN_BIN to a maturin executable, then rerun this script." >&2
  exit 1
}
[[ -n "$CARGO_BIN" && -x "$CARGO_BIN" ]] || {
  echo "ERROR: no compatible flopsearch wheel and cargo is unavailable." >&2
  exit 1
}
test -f "$VENDOR_DIR/pyproject.toml" || {
  echo "ERROR: vendored flopsearch source is missing: $VENDOR_DIR" >&2
  exit 1
}

wheel_dir="$(mktemp -d)"
trap 'rm -rf "$wheel_dir"' EXIT
echo "Building vendored flopsearch from $VENDOR_DIR"
"$MATURIN_BIN" build --interpreter "$PYTHON_BIN" \
  --manifest-path "$VENDOR_DIR/Cargo.toml" --release --out "$wheel_dir"
"$PYTHON_BIN" -m pip install --no-cache-dir "$wheel_dir"/*.whl
backend_available || {
  echo "ERROR: vendored flopsearch build does not expose global_greedy_rust." >&2
  exit 1
}
echo "flopsearch installation succeeded from vendored source"
