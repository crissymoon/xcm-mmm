#!/usr/bin/env bash
# Label rows in, the training set in this repository out.
#
# The three stages are separate commands and this script is only the wiring:
# mine turns one file of labelled rows into rank, preference and contrast pairs,
# retain folds every file into one bank that only improves, and dataset samples
# that bank into the files under data/, with a manifest and a card. The bank
# itself stays out of the repository, because a bank grows every week and a
# clone should not grow with it.
#
# Usage:
#   scripts/train-data.sh --rows DIR              # mine, retain, ship
#   scripts/train-data.sh --bank data/bank.jsonl  # ship an existing bank
#   scripts/train-data.sh --rows DIR --budget 30000
#
#   --rows DIR      folder of labelled JSON Lines, one month per file
#   --bank PATH     bank to sample; mined rows are retained into it
#   --out DIR       shipped folder (default data/training)
#   --budget N      rows the shipped sample may hold (default 20000)
#   --seed N        seed of the sample (default 20261114)
#   --name NAME     name the card gives the set
#   --dry           print the commands and stop
set -euo pipefail
cd "$(dirname "$0")/.."

ROWS=""
BANK=""
OUT="data/training"
BUDGET=20000
SEED=20261114
NAME="marketlab-news-pairs"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rows) ROWS="$2"; shift 2 ;;
    --bank) BANK="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --dry) DRY=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$ROWS" && -z "$BANK" ]]; then
  echo "one of --rows or --bank is required" >&2
  exit 2
fi

BIN="${XCM_MMM:-./target/release/xcm-mmm}"
if [[ ! -x "$BIN" ]]; then
  echo "no binary at $BIN; run cargo build --release or set XCM_MMM" >&2
  exit 2
fi

run() {
  if [[ "$DRY" == "1" ]]; then
    echo "$@"
  else
    "$@"
  fi
}

if [[ -n "$ROWS" ]]; then
  [[ -d "$ROWS" ]] || { echo "no rows folder at $ROWS" >&2; exit 2; }
  BANK="${BANK:-tmp/train-bank.jsonl}"
  mkdir -p tmp/pairs
  mined=()
  for rows in "$ROWS"/*.jsonl; do
    [[ -e "$rows" ]] || { echo "no *.jsonl under $ROWS" >&2; exit 2; }
    name="$(basename "$rows" .jsonl)"
    run "$BIN" pairs mine "$rows" "tmp/pairs/$name.jsonl"
    mined+=("tmp/pairs/$name.jsonl")
  done
  # The bank only improves, so a rerun adds what is new and keeps what already
  # cleared the floor. Every month file is offered in one call so a better pair
  # can push out a worse one across months rather than within one. Retain takes
  # the list as one comma separated value rather than one argument per file.
  list="$(IFS=,; echo "${mined[*]}")"
  run "$BIN" pairs retain --input "$list" --out "$BANK"
fi

run "$BIN" dataset build \
  --input "$BANK" \
  --out "$OUT" \
  --budget "$BUDGET" \
  --seed "$SEED" \
  --name "$NAME"

run "$BIN" dataset check --dir "$OUT"
