#!/usr/bin/env bash
# Stage the fitted artefacts as a Kaggle dataset, laid out exactly as the
# binary expects them.
#
#   scripts/package-models.sh                 # stage dist/kaggle-models
#   scripts/package-models.sh --push          # stage, then upload a new version
#   scripts/package-models.sh --push --new    # first upload: create the dataset
#
# Why a flat `data/` tree and not a folder per model: every command in the CLI
# defaults to a path under `data/` (`data/detectors/model.json`,
# `data/legal`, `data/segments/features.jsonl`). A notebook that copies this
# tree next to the binary and cds into that directory runs `detect score`,
# `read`, `segment assign`, `legal scan`, `legal explain` and `churn check` with
# no flags at all,
# which is the same invocation the repository's own README shows.
#
# data/training is deliberately not staged: it is 20 MB of raw source that no
# shipped model reads, and the dataset is meant to mount fast.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SLUG="${SLUG:-xcm-mmm-models}"
OUT="${OUT:-dist/kaggle-models}"
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
rm -rf "$OUT"
mkdir -p "$OUT/data"

for d in detectors segments legal emotion churn route; do
  cp -R "data/$d" "$OUT/data/$d"
done

BYTES="$(find "$OUT/data" -type f -exec cat {} + | wc -c | tr -d ' ')"
FILES="$(find "$OUT/data" -type f | wc -l | tr -d ' ')"
DIGEST="$(cd "$OUT" && find data -type f | sort | xargs shasum -a 256 | shasum -a 256 | cut -d' ' -f1)"

cp LICENSE "$OUT/LICENSE"

# Read from the instrument, never retyped, so the card and the dataset
# description carry the same notice as `legal explain` prints.
DISCLAIMER="$(python3 -c 'import json; print(json.load(open("data/legal/settings.json"))["disclaimer"])')"

card="README.md"
cat > "$OUT/$card" <<CARD
# xcm-mmm fitted artefacts (${VERSION})

The folders the xcm-mmm command line tool reads, staged in the same layout the
repository uses so that the CLI's defaults resolve with no flags.

| field | value |
| --- | --- |
| version | ${VERSION} |
| files | ${FILES} |
| bytes | ${BYTES} |
| tree digest | ${DIGEST} |
| licence | MIT |

## What is inside

| folder | what it holds | read by |
| --- | --- | --- |
| \`data/detectors\` | seven-axis lexical model, term bank, cue table, 518 ads | \`detect score\`, \`detect info\`, \`read\` |
| \`data/emotion\` | emotion model, listed and analogy tiers, word bank | \`read\`, \`emotion coverage\` |
| \`data/segments\` | behavioural segment model, 251 entities, feature rows | \`segment info\`, \`segment assign\` |
| \`data/legal\` | glossary, rules, pairs, rewrites, settings | \`legal scan\`, \`legal translate\`, \`legal explain\` |
| \`data/churn\` | churn model, panel, reasons and the source it rebuilds from | \`churn check\`, \`churn score\` |
| \`data/route\` | the routing policy and seven sample arrivals | \`route decide\`, \`route organize\`, \`route check\` |

The binary itself is a separate dataset (\`crissymoon/xcm-mmm-static-linux-x86-64\`).
This one has no executable in it.

## Using it in a notebook

Attach both datasets, then put the two pieces side by side, which is the layout
the defaults were written for:

    import os, pathlib, shutil, stat, subprocess

    IN   = pathlib.Path("/kaggle/input")
    RUN  = pathlib.Path("/kaggle/working/xcm-mmm-run")
    RUN.mkdir(parents=True, exist_ok=True)

    BIN  = RUN / "xcm-mmm"
    shutil.copyfile(IN / "${SLUG%-models}-static-linux-x86-64/xcm-mmm", BIN)
    BIN.chmod(BIN.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    shutil.copytree(IN / "${SLUG}/data", RUN / "data", dirs_exist_ok=True)

    os.chdir(RUN)
    print(subprocess.run([str(BIN), "detect", "score", "--text",
                          "We are delighted to confirm the renewal signed today."],
                         capture_output=True, text=True).stdout)

Both dataset mounts are read only, so the copy is not optional: writing a
scored row next to the model fails with \`EROFS\`. Copying the tree costs about
5 MB and a fraction of a second.

## Measured

On the Kaggle image (Ubuntu 24.04, 4 cores, 32 GB, no network):
\`detect score\` over the 518 ads in \`data/detectors/ads.jsonl\` took 0.11 s,
\`churn check\`, which rebuilds the folder from its source, reported the folder
clean, and \`segment assign\` placed all 251 entities.

## About this upload

${DISCLAIMER}

The \`data/legal\` folder is part of a research project on legal English. A
legal brief from the CLI is a reading of the words in a document, useful for
study and for finding what a clause asks, and not a substitute for a lawyer.

Built from $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo "an uncommitted tree").
CARD

cat > "$OUT/dataset-metadata.json" <<META
{
  "title": "xcm-mmm fitted artefacts",
  "id": "crissymoon/${SLUG}",
  "licenses": [{"name": "MIT"}],
  "subtitle": "Detector, emotion, segment, legal, churn and routing artefacts",
  "description": "The data folders read by the xcm-mmm command line tool: data/detectors (seven lexical axes plus 518 ads), data/emotion (listed and analogy tiers), data/segments (4 behavioural groups over 251 entities), data/legal (glossary, rules, pairs, rewrites), data/churn (model, panel, reasons, source) and data/route (the routing policy and seven sample arrivals). ${FILES} files, ${BYTES} bytes, version ${VERSION}, tree digest ${DIGEST}, MIT. Staged as data/<model>/ so the CLI defaults resolve: copy the tree beside the xcm-mmm binary, cd into that directory and detect score, read, segment assign, legal scan, legal explain, churn check and route organize all run with no path flags. The executable is a separate dataset, crissymoon/xcm-mmm-static-linux-x86-64. ${DISCLAIMER}",
  "keywords": ["mathematics", "software"],
  "isPrivate": false
}
META

echo "staged $OUT"
echo "files $FILES bytes $BYTES"
echo "digest $DIGEST"
du -sh "$OUT/data"

if [ "$PUSH" = 1 ]; then
  KAGGLE="${KAGGLE:-$(command -v kaggle || echo "$HOME/Documents/mcp-v5.4/venv/bin/kaggle")}"
  # --dir-mode is not optional here. The CLI default is "skip", which uploads
  # the loose files, prints "Skipping folder: data" in passing, and publishes a
  # dataset with no models in it. zip is what the mount expects: Kaggle unpacks
  # it back to data/<model>/ and the CLI defaults resolve as written.
  #
  # --public is not optional either. `datasets create` ignores "isPrivate" in
  # dataset-metadata.json and defaults to private, so a dataset that reads as
  # public in its own metadata is invisible to anyone who is not signed in.
  if [ "$NEW" = 1 ]; then
    "$KAGGLE" datasets create -p "$OUT" -r zip --public
  else
    "$KAGGLE" datasets version -p "$OUT" -m "xcm-mmm ${VERSION} artefacts" -r zip
  fi
fi
