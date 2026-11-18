#!/usr/bin/env bash
# Rebuild the detector folder: export the three inputs, fit, check.
#
# The inputs are exported from the marketlab corpus rather than edited here, so
# the cue patterns and the bank cannot drift from the readings they were
# published with. The legal language bank is folded into the same bank, which is
# the only source of statute register and of plain sentences about the law; set
# EXTRA=none to fit the ad corpus alone. Everything is written into OUT, which is
# the folder the repository ships; nothing is committed by this script.
#
#   scripts/detect-data.sh
#   XCB=/path/to/XcaliburBiz OUT=/tmp/detectors scripts/detect-data.sh
#   FOLDS=3 TOP_TERMS=60 scripts/detect-data.sh
#   EXTRA=none scripts/detect-data.sh
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
# The legal language bank joins the same two axes: statute sentences as legal
# register, and plain sentences about the law as the negative side, which the
# ad corpus cannot supply. EXTRA=none fits the ad corpus alone.
EXTRA="${EXTRA:-$XCB/data/marketlab-legal-language/bank.jsonl}"

if [ ! -x "$BIN" ]; then
    echo "no binary at $BIN; run cargo build --release first" >&2
    exit 1
fi

EXPORT_ARGS=(--xcb "$XCB" --out "$OUT")
if [ "$EXTRA" != "none" ]; then
    if [ ! -f "$EXTRA" ]; then
        echo "no legal language bank at $EXTRA; set EXTRA=none to fit the ad corpus alone" >&2
        exit 1
    fi
    EXPORT_ARGS+=(--extend "$EXTRA")
fi

python3 "$ROOT/scripts/export-detect-data.py" "${EXPORT_ARGS[@]}"

"$BIN" detect train \
    --bank "$OUT/bank.jsonl" \
    --ads "$OUT/ads.jsonl" \
    --cues "$OUT/cues.json" \
    --out "$OUT" \
    --folds "$FOLDS" \
    --top-terms "$TOP_TERMS"

"$BIN" detect check --dir "$OUT"
