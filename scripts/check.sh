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

echo "== scripts"
# The generator is the only thing that builds the kernel, so it is compiled and
# run here: a broken cell module would otherwise be found on Kaggle, not here.
PYTHONPYCACHEPREFIX=tmp/pycache python3 -m compileall -q scripts
python3 scripts/kaggle-notebook.py

echo "== frequency table"
python3 scripts/lex-freq.py --check

echo "all checks passed"
