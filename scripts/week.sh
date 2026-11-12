#!/usr/bin/env bash
# One weekly step in one command: mine each day's labelled rows into pairs, fold
# the best of them into the retention bank, audit the bank, and optionally train
# on it.
#
# Usage: scripts/week.sh --bank data/week.jsonl [options] day1.jsonl day2.jsonl
#
#   --bank PATH       corpus the bank keeps, required
#   --budget N        rows the bank may hold (default 60000)
#   --min-margin N    outcome gap floor while mining (default 0.01)
#   --min-score N     retention floor (default 0.45)
#   --novelty N       simhash bits a new anchor must differ by (default 4)
#   --per-day N       rows per label day (default 4000)
#   --per-anchor N    rows per anchor document (default 3)
#   --train           train the transformer on the bank after the audit
#   --model PATH      model file for --train (default beside the bank)
#   --train-limit N   rows the training run may read, best first (default all)
#   --resume          continue the model named by --model instead of starting over
#
# Rows may be gzip compressed. Nothing is overwritten except the bank and the
# files that belong to it: <bank>.idx.json, <bank>.ledger.jsonl and a work
# directory for the day files, which is removed on exit.
set -euo pipefail
cd "$(dirname "$0")/.."

BANK=""
BUDGET=60000
MIN_MARGIN=0.01
MIN_SCORE=0.45
NOVELTY=4
PER_DAY=4000
PER_ANCHOR=3
TRAIN=0
RESUME=0
MODEL=""
TRAIN_LIMIT=0
DAYS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --bank) BANK="${2:?--bank needs a path}"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --min-margin) MIN_MARGIN="$2"; shift 2 ;;
        --min-score) MIN_SCORE="$2"; shift 2 ;;
        --novelty) NOVELTY="$2"; shift 2 ;;
        --per-day) PER_DAY="$2"; shift 2 ;;
        --per-anchor) PER_ANCHOR="$2"; shift 2 ;;
        --train) TRAIN=1; shift ;;
        --resume) RESUME=1; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --train-limit) TRAIN_LIMIT="$2"; shift 2 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) DAYS+=("$1"); shift ;;
    esac
done

if [ -z "$BANK" ]; then
    echo "error: --bank is required" >&2
    exit 2
fi
if [ ${#DAYS[@]} -eq 0 ]; then
    echo "error: give at least one day file" >&2
    exit 2
fi
for day in "${DAYS[@]}"; do
    if [ ! -f "$day" ]; then
        echo "error: no such file: $day" >&2
        exit 2
    fi
done

echo "== build"
cargo build --release -q
BIN=./target/release/xcm-mmm

mkdir -p "$(dirname "$BANK")"
WORK="$(mktemp -d "${BANK}.work.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

MINED=()
for day in "${DAYS[@]}"; do
    stem="$(basename "$day")"
    stem="${stem%.gz}"
    stem="${stem%.jsonl}"
    out="$WORK/$stem.pairs.jsonl"
    echo "== mine $day"
    "$BIN" pairs mine "$day" "$out" --min-margin "$MIN_MARGIN" >/dev/null
    rows=$(wc -l < "$out" | tr -d ' ')
    echo "   $rows pairs"
    if [ "$rows" -gt 0 ]; then
        MINED+=("$out")
    fi
done

if [ ${#MINED[@]} -eq 0 ]; then
    echo "error: no day produced a pair, so the bank is untouched" >&2
    exit 1
fi

join() {
    local out=""
    local part
    for part in "$@"; do
        if [ -z "$out" ]; then out="$part"; else out="$out,$part"; fi
    done
    printf '%s' "$out"
}

echo "== retain"
"$BIN" pairs retain \
    --input "$(join "${MINED[@]}")" \
    --out "$BANK" \
    --budget "$BUDGET" \
    --per-day "$PER_DAY" \
    --per-anchor "$PER_ANCHOR" \
    --min-score "$MIN_SCORE" \
    --novelty-bits "$NOVELTY"

echo "== audit"
"$BIN" pairs audit "$BANK" --min-margin "$MIN_MARGIN"

echo "== bank"
"$BIN" pairs bank --out "$BANK"

if [ "$TRAIN" -eq 1 ]; then
    if [ -z "$MODEL" ]; then
        MODEL="${BANK%.jsonl}.model"
    fi
    echo "== train $MODEL"
    extra=()
    if [ "$RESUME" -eq 1 ] && [ -f "$MODEL" ]; then
        extra=(--resume "$MODEL")
    fi
    "$BIN" transformer train \
        --pairs "$BANK" \
        --out "$MODEL" \
        --limit "$TRAIN_LIMIT" \
        --checkpoint "$MODEL" \
        ${extra[@]+"${extra[@]}"}
fi

echo "done: $BANK"
