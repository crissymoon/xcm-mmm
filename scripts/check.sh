#!/usr/bin/env bash
# Release check: the same gates a hosted CI job would run, run locally.
# Usage: scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== format"
cargo fmt --all -- --check

echo "== lint"
cargo clippy --all-targets -- -D warnings

echo "== test"
cargo test --release

echo "== docs"
python3 scripts/check_docs.py
cargo doc --no-deps

echo "all checks passed"
