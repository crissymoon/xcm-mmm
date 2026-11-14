#!/usr/bin/env python3
"""Write the detector training folder this repository ships.

Three files come out of the marketlab ad corpus, and nothing here fits a model:

* ``bank.jsonl`` is the sentence bank the lexicon axes are learned from. Each
  row names its axis, its label, the source it came from and its text. Legal
  sentences are Federal Register abstracts (label 1), jargon sentences are
  grants.gov synopses (label 1), and ad copy is the plain side (label 0).
* ``ads.jsonl`` is ad copy at document grain, with the source split kept, so the
  index axes fit their reference quantiles on the training ads and nothing else.
* ``cues.json`` is the declared cue table for the five index axes, including
  the declared examples each construct must fire on, and the feature list the
  two learned axes are fitted on. It is read out of the Python modules rather
  than retyped, so the patterns in this repository cannot drift from the ones
  the corpus was scored with.

Run from the repository root:

    scripts/export-detect-data.py --xcb /path/to/XcaliburBiz

The default ``--xcb`` is the author's checkout. Everything is sorted before it
is written, so an unchanged corpus exports the same bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path("data/detectors")
CORPUS = Path("data/marketlab-ads-language")


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item, sort_keys=True, ensure_ascii=True) + "\n" for item in items),
        encoding="utf-8",
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def export_bank(corpus: Path, out: Path) -> int:
    bank = rows(corpus / "bank_sentences.jsonl")
    kept = [
        {"axis": r["axis"], "label": int(r["label"]), "source": r["source"], "text": r["text"]}
        for r in bank
    ]
    kept.sort(key=lambda r: (r["axis"], r["label"], r["source"], r["text"]))
    write_jsonl(out / "bank.jsonl", kept)
    return len(kept)


def export_ads(corpus: Path, out: Path) -> int:
    seen = set()
    kept = []
    for row in rows(corpus / "ads_clean.jsonl"):
        ident = str(row["id"])
        if ident in seen:
            continue
        seen.add(ident)
        kept.append({"id": ident, "split": row["split"], "text": row["text"]})
    kept.sort(key=lambda r: r["id"])
    write_jsonl(out / "ads.jsonl", kept)
    return len(kept)


def export_cues(xcb: Path, out: Path) -> int:
    sys.path.insert(0, str(xcb))
    from xcb.marketing.marketlab import adscampaign, adsdetect  # noqa: E402

    axes = {}
    for construct in adscampaign.CONSTRUCTS:
        axes[construct.key] = {
            "method": "index",
            "tag": construct.tag,
            "question": construct.question,
            "components": dict(construct.components),
            "cues": dict(construct.cues),
            "gate": list(construct.gate),
            "min_groups": construct.min_groups,
            "halves": [list(construct.halves[0]), list(construct.halves[1])],
            "checks": [
                {"text": text, "expected": bool(expected)}
                for text, expected in adscampaign.CHECKS.get(construct.key, ())
            ],
        }
    # Fear is an index too, but its components are declared in adsdetect beside
    # the seed patterns and it carries no halves. The Python axis adds a learned
    # threat lexicon to the threat component; this folder fits the cues only,
    # because no labelled fear bank travels with the corpus.
    axes["fear"] = {
        "method": "index",
        "tag": "#IntentionalFear",
        "question": "Does the ad name a harm and tell the reader to act before a clock?",
        "components": dict(adsdetect.FEAR_WEIGHTS),
        "cues": dict(adsdetect.FEAR_SEED),
        "gate": ["threat"],
        "min_groups": 1,
        "halves": [["threat"], ["victim", "urgency"]],
        "checks": [],
    }
    # The learned axes read a feature list, and every surface feature is a
    # pattern the corpus already declares, so the table travels with the cues
    # rather than being retyped in Rust. Fear is an index here and is left out.
    surface = {name: pattern.pattern for name, pattern in (
        ("citation", adsdetect.CITE), ("authority", adsdetect.AUTHORITY),
        ("nominal", adsdetect.NOMINAL), ("agentless", adsdetect.AGENTLESS),
        ("directive", adsdetect.DIRECTIVE), ("victim", adsdetect.VICTIM),
        ("urgency", adsdetect.URGENCY),
    )}
    features = {
        axis: {
            "names": list(names),
            "patterns": {name: surface[name] for name in names if name in surface},
        }
        for axis, names in adsdetect.FEATURES.items()
        if axis != "fear"
    }
    write_json(out / "cues.json", {"version": 1, "axes": axes, "features": features})
    return len(axes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xcb", default=str(Path.home() / "Documents/sql-mgr/XcaliburBiz"))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.xcb) / CORPUS
    if not corpus.is_dir():
        parser.error(f"no ad corpus under {corpus}")

    bank = export_bank(corpus, out)
    ads = export_ads(corpus, out)
    axes = export_cues(Path(args.xcb), out)
    print(f"bank      {bank} sentences -> {out / 'bank.jsonl'}")
    print(f"ads       {ads} documents -> {out / 'ads.jsonl'}")
    print(f"cues      {axes} index axes -> {out / 'cues.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
