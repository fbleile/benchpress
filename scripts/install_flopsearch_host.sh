#!/usr/bin/env bash
set -euo pipefail

# Install the Linux wheel used by the ordinary FLOP rule.  If a wheel is not
# available for the active Python, build the checked-in source distribution.
# This is deliberately explicit: host mode must never silently substitute a
# different FLOP implementation.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENDOR_DIR="$ROOT_DIR/workflow/rules/structure_learning_algorithms/flop_notreks/vendor/flopsearch/flop_python"

if "$PYTHON_BIN" -c 'import flopsearch' >/dev/null 2>&1; then
  echo "flopsearch already available: $($PYTHON_BIN -c 'import flopsearch; print(getattr(flopsearch, "__version__", "installed"))')"
  exit 0
fi

echo "Installing flopsearch==0.3.0 into $($PYTHON_BIN -c 'import sys; print(sys.prefix)')"
if "$PYTHON_BIN" -m pip install --no-cache-dir "flopsearch==0.3.0"; then
  echo "flopsearch installation succeeded"
  exit 0
fi

command -v maturin >/dev/null 2>&1 || {
  echo "ERROR: no compatible flopsearch wheel and maturin is unavailable." >&2
  echo "Install maturin and Rust/cargo, then rerun this script." >&2
  exit 1
}
command -v cargo >/dev/null 2>&1 || {
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
maturin build --manifest-path "$VENDOR_DIR/Cargo.toml" --release --out "$wheel_dir"
"$PYTHON_BIN" -m pip install --no-cache-dir "$wheel_dir"/*.whl
echo "flopsearch installation succeeded from vendored source"
