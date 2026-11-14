#!/usr/bin/env bash
# Rebuild the detector folder: export the three inputs, fit, check.
#
# The inputs are exported from the marketlab corpus rather than edited here, so
# the cue patterns and the bank cannot drift from the readings they were
# published with. Everything is written into OUT, which is the folder the
# repository ships; nothing is committed by this script.
#
#   scripts/detect-data.sh
#   XCB=/path/to/XcaliburBiz OUT=/tmp/detectors scripts/detect-data.sh
#   FOLDS=3 TOP_TERMS=60 scripts/detect-data.sh
#
# Exits non-zero if the fit refuses a bank, if a declared example fails, or if
# the rebuilt folder disagrees with its own manifest.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XCB="${XCB:-$HOME/Documents/sql-mgr/XcaliburBiz}"
OUT="${OUT:-$ROOT/data/detectors}"
BIN="${BIN:-$ROOT/target/release/xcm-mmm}"
FOLDS="${FOLDS:-5}"
TOP_TERMS="${TOP_TERMS:-120}"

if [ ! -x "$BIN" ]; then
    echo "no binary at $BIN; run cargo build --release first" >&2
    exit 1
fi

python3 "$ROOT/scripts/export-detect-data.py" --xcb "$XCB" --out "$OUT"

"$BIN" detect train \
    --bank "$OUT/bank.jsonl" \
    --ads "$OUT/ads.jsonl" \
    --cues "$OUT/cues.json" \
    --out "$OUT" \
    --folds "$FOLDS" \
    --top-terms "$TOP_TERMS"

"$BIN" detect check --dir "$OUT"
