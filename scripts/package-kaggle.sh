#!/usr/bin/env bash
# Stage the Linux binary as a Kaggle dataset.
#
#   scripts/package-kaggle.sh                 # build and stage dist/kaggle
#   scripts/package-kaggle.sh --push          # stage, then upload with the kaggle CLI
#   scripts/package-kaggle.sh --push --new    # first upload: create instead of version
#
# Why a dataset and not a private source build: Kaggle kernels run with no
# network unless the setting is enabled, and even with network there is no Rust
# toolchain in the image. A dataset is the only artefact a kernel can mount.
#
# Two facts the notebook exists to prove, because both are easy to get wrong:
#   1. Kaggle does not preserve the executable bit on dataset files that were
#      not shipped inside a zip, so /kaggle/input/<ds>/xcm-mmm cannot be run in
#      place. The notebook copies it to /kaggle/working and chmods it.
#   2. /kaggle/input is mounted read only, so nothing may write there.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SLUG="${SLUG:-xcm-mmm-static-linux-x86-64}"
OUT="${OUT:-dist/kaggle}"
PUSH=0
NEW=0
for arg in "$@"; do
  case "$arg" in
    --push) PUSH=1 ;;
    --new) NEW=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
mkdir -p "$OUT"
OUT="$OUT" scripts/build-linux.sh >/dev/null
SHA="$(shasum -a 256 "$OUT/xcm-mmm" | cut -d' ' -f1)"
BYTES="$(wc -c < "$OUT/xcm-mmm" | tr -d ' ')"

cp LICENSE "$OUT/LICENSE"

# One source of truth for the notice: the instrument's own settings.json, read
# rather than retyped, so the card, the dataset description and `legal explain`
# say the same thing.
DISCLAIMER="$(python3 -c 'import json; print(json.load(open("data/legal/settings.json"))["disclaimer"])')"

# The card is the only place a reader learns that the binary is static, why
# that matters, and how the notebook is expected to invoke it.
cat > "$OUT/README.md" <<CARD
# xcm-mmm ${VERSION} for Linux x86-64

The command line front end of [xcm-mmm](https://github.com/crissymoon/xcm-mmm),
built as a static-pie ELF so it runs unchanged inside a Kaggle notebook without
a Rust toolchain and without enabling the network.

| field | value |
| --- | --- |
| version | ${VERSION} |
| bytes | ${BYTES} |
| sha256 | ${SHA} |
| target | x86_64-unknown-linux-musl, static-pie |
| licence | MIT |

`provenance.json` beside the binary carries the same hash with the full commit,
the target triple, the `rustc` that produced it and the flags the build used. The
notebook reads both and fails if they disagree, which is the only reason to
trust either.

## Using it in a notebook

    import hashlib, os, shutil, stat, subprocess

    SRC = "/kaggle/input/${SLUG}/xcm-mmm"
    DST = "/kaggle/working/xcm-mmm"
    shutil.copyfile(SRC, DST)
    os.chmod(DST, os.stat(DST).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(subprocess.run([DST, "--help"], capture_output=True, text=True).stdout)

Copy, then chmod. A dataset file arrives with no execute bit (0644 by the
documentation, 0666 on the image measured in November 2026) and /kaggle/input is
read only, so running it in place fails with EACCES rather than a useful message.

## What it was measured on

| field | value |
| --- | --- |
| image | Ubuntu 24.04, Python 3.13.18, glibc 2.39 |
| cpu | 4 cores, rayon uses all of them |
| memory | 32 GB |
| working disk | 20 GB at /kaggle/working |
| session | 12 hours, no internet needed |

A 4000 row 24 feature forecast with a 3000 evaluation budget took 3.8 seconds on
that image; four shards of 1500 took 2.1 seconds wall against 8.0 seconds of
shard time.

## What is inside

Runs of the tool need no arguments beyond a file, and each reads only what it is
given: \`detect score\` reads a text or JSONL file against the shipped lexical
model, \`read\` scores one text on every model, \`pairs mine\` and \`pairs retain\`
build and prune training pairs, \`pairs bank\` summarises one. The models and the
corpora are not in this dataset, because this upload is the executable only:
attach \`crissymoon/xcm-mmm-models\` as well, copy its \`data/\` tree beside the
binary and change directory, and the CLI's defaults resolve with no path flags.

## About this upload

${DISCLAIMER}

The legal reading commands exist to study how legal English can be rewritten
and explained, and they are part of a research project. Read a brief as a
starting point for reading the document itself.

Built from $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo "an uncommitted tree").
CARD

cat > "$OUT/dataset-metadata.json" <<META
{
  "title": "xcm-mmm Linux x86-64 binary",
  "id": "crissymoon/${SLUG}",
  "licenses": [{"name": "MIT"}],
  "subtitle": "Static ELF build of the xcm-mmm command line tool, for Kaggle notebooks",
  "description": "The xcm-mmm command line front end compiled for Linux x86-64 as a static-pie ELF, so it runs in a Kaggle notebook with no Rust toolchain and no network. Version ${VERSION}, ${BYTES} bytes, sha256 ${SHA}. Copy it out of /kaggle/input and chmod +x before running: Kaggle does not keep the executable bit on dataset files and /kaggle/input is read only. Commands used from notebooks are detect score, read, segment assign, legal scan, legal explain, churn check, route organize and forecast; the artefacts those text commands read live in the companion dataset crissymoon/xcm-mmm-models, which stages them as data/<model>/ so that the CLI defaults resolve once the tree sits beside the binary. ${DISCLAIMER}",
  "keywords": ["mathematics", "software"],
  "isPrivate": false
}
META

echo "staged $OUT"
echo "sha256 $SHA"
ls -l "$OUT"

if [ "$PUSH" = 1 ]; then
  KAGGLE="${KAGGLE:-$(command -v kaggle || echo "$HOME/Documents/mcp-v5.4/venv/bin/kaggle")}"
  # --public on create, because `datasets create` ignores "isPrivate" in the
  # metadata and defaults to private; "-r zip" because the default dir-mode is
  # "skip". Neither omission fails loudly.
  if [ "$NEW" = 1 ]; then
    "$KAGGLE" datasets create -p "$OUT" -r zip --public
  else
    "$KAGGLE" datasets version -p "$OUT" -m "xcm-mmm ${VERSION} static build" -r zip
  fi
fi
