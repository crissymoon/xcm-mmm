#!/usr/bin/env python3
"""Write the general-English frequency table the rarity reading is measured on.

The glossary knows which words a lawyer writes. It does not know which of those
words a reader has ever met, and `hereunder` and `under` sit in the same list
with the same weight. This script supplies the missing axis: how often each word
occurs in general English, so a term can be graded by the rarest word it
contains rather than counted as one hit among equals.

The reference is the Brown corpus, 1,161,192 tokens of American English from
1961, reached through NLTK. It is old and small, and both matter: a word absent
from it is not proof the word is unknown, only that a million words of edited
prose never used it. The threshold in settings.json says how strict the reading
is, and the file records the corpus it came from so the numbers are never
mistaken for a modern count.

Only words the glossary actually contains are written. A full table would be
fifty thousand lines of words no row uses, and the point of the file is that a
person can read it beside the glossary.

    python3 scripts/lex-freq.py            # rewrite data/legal/freq.json
    python3 scripts/lex-freq.py --check    # refuse if the table is stale
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TERMS = ROOT / "data" / "legal" / "terms.jsonl"
OUT = ROOT / "data" / "legal" / "freq.json"

SOURCE = (
    "Brown corpus (Kučera & Francis 1967), 1,161,192 tokens of American "
    "English, 1961, via NLTK"
)
NOTE = (
    "Counts are per million tokens of general English, not of legal English. A "
    "word the glossary carries and this table does not was never used in a "
    "million words of 1961 edited prose; the reading treats that as a word a "
    "reader may not have met, and settings.json holds the cut."
)

WORD = re.compile(r"[a-z]+")


def words(name: str) -> set[str]:
    """Every alphabetic word in a glossary spelling, lower-cased."""
    return set(WORD.findall(name.lower()))


def needed() -> set[str]:
    """The words any row of the glossary can put in front of a reader."""
    out: set[str] = set()
    for line in TERMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out |= words(row["term"])
        out |= words(row.get("plain", ""))
        for variant in row.get("variants", []):
            out |= words(variant)
    return out


def brown() -> tuple[int, collections.Counter]:
    try:
        from nltk.corpus import brown
    except ImportError:
        sys.exit("nltk is not installed: pip install nltk, then rerun")
    try:
        tokens = brown.words()
    except LookupError:
        import nltk

        nltk.download("brown", quiet=True)
        from nltk.corpus import brown as corpus

        tokens = corpus.words()
    counted = collections.Counter(
        token.lower() for token in tokens if token.isalpha()
    )
    return len(tokens), counted


def build() -> dict:
    tokens, counted = brown()
    wanted = needed()
    per_million = {
        word: round(1e6 * counted[word] / tokens, 2)
        for word in sorted(wanted)
        if counted[word] > 0
    }
    return {
        "source": SOURCE,
        "note": NOTE,
        "tokens": tokens,
        "per_million": per_million,
        "absent": sorted(wanted - set(per_million)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when the shipped table no longer matches the glossary",
    )
    args = parser.parse_args()

    table = build()
    text = json.dumps(table, indent=1, sort_keys=False, ensure_ascii=False) + "\n"
    if args.check:
        if not OUT.exists():
            sys.exit(f"{OUT.relative_to(ROOT)} is missing: run scripts/lex-freq.py")
        if OUT.read_text(encoding="utf-8") != text:
            sys.exit(
                f"{OUT.relative_to(ROOT)} is stale: the glossary changed, rerun "
                "scripts/lex-freq.py"
            )
        print(f"freq ok: {len(table['per_million'])} words, {table['tokens']} tokens")
        return 0

    OUT.write_text(text, encoding="utf-8")
    print(
        f"wrote {OUT.relative_to(ROOT)}: {len(table['per_million'])} words counted, "
        f"{len(table['absent'])} absent from {table['tokens']} tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
