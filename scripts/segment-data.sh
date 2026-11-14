#!/usr/bin/env bash
# Rebuild the segment folder: extract the panel, fit, check.
#
# The panel is exported by the archive reader rather than edited here, so the
# entities cannot drift from the ads they describe. Everything is written into
# OUT, which is the folder the repository ships; nothing is committed by this
# script.
#
#   scripts/segment-data.sh
#   XCB=/path/to/XcaliburBiz OUT=/tmp/segments scripts/segment-data.sh
#   K_MIN=3 K_MAX=5 MIN_SHARE=0.08 scripts/segment-data.sh
#
# Exits non-zero if the fit refuses its own gates or if the rebuilt folder
# disagrees with its manifest.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XCB="${XCB:-$HOME/Documents/sql-mgr/XcaliburBiz}"
PANEL="${PANEL:-$XCB/data/marketlab-ads-language/ads_clean_all.jsonl}"
OUT="${OUT:-$ROOT/data/segments}"
BIN="${BIN:-$ROOT/target/release/xcm-mmm}"
K_MIN="${K_MIN:-3}"
K_MAX="${K_MAX:-6}"
MIN_SHARE="${MIN_SHARE:-0.05}"
SEED="${SEED:-20261114}"

if [ ! -x "$BIN" ]; then
    echo "no binary at $BIN; run cargo build --release first" >&2
    exit 1
fi
if [ ! -f "$PANEL" ]; then
    echo "no panel at $PANEL; set PANEL or XCB" >&2
    exit 1
fi

"$BIN" segment extract --ads "$PANEL" --out "$OUT/features.jsonl"

"$BIN" segment fit \
    --features "$OUT/features.jsonl" \
    --out "$OUT" \
    --k-min "$K_MIN" \
    --k-max "$K_MAX" \
    --min-share "$MIN_SHARE" \
    --seed "$SEED"

"$BIN" segment check --dir "$OUT"
