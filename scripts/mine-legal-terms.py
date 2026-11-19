#!/usr/bin/env python3
"""Find drafting vocabulary the glossary does not hold, from the bank itself.

The coverage report says the glossary reaches one court-rule row in five while
it reaches more than half the statute rows. That is a fact about the word list,
not about the scanner, so the fix has to come from the text: phrases that court
rules use habitually and the glossary has no row for.

The score is a document frequency log-odds ratio between the target register and
the contrast rows, with a Dirichlet prior, which is the same estimator the ad
detector uses (Monroe, Colaresi and Quinn 2008). A phrase that appears equally
often on both sides is ordinary English and scores near zero. A phrase that
appears only in the target register scores high, and high is a claim that the
phrase is *characteristic* of the register, not that it is legalese. The second
question is a person's: the report is a reading queue, never an import.

Usage:
    python3 scripts/mine-legal-terms.py [--register court-rule] [--min-df 5]
                                        [--top 60] [--json PATH]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "legal"

# Same markers the Rust register table uses. Kept in one place here because a
# mismatch would silently mine the wrong register.
MARKERS = {
    "court-rule": ("frcp-", "frap-", "fre-", "ca-crc-", "crc-"),
    "statute": (
        "usc-", "ca-ccp-", "ca-civ-", "ca-gov-", "ca-lamc-",
        "ca-lab-", "ca-pen-", "ca-bpc-", "ca-edu-", "ca-hs-",
    ),
    "contract": ("cuad-",),
    "regulation": ("cfr-", "ca-code-regs", "ca-reg-"),
    "agency-notice": ("fedreg-",),
    "opinion": ("courtlistener-", "cap-"),
}

WORD = re.compile(r"[a-z][a-z'\u2019-]*")
# A phrase may not begin or end on a function word: those positions are where
# the window crossed into the next clause rather than found a fixed phrase.
EDGE = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with",
    "and", "or", "that", "this", "these", "those", "is", "are", "be",
}
# Drafting glue: a phrase holding one of these is a connective, not a domain
# name. "in the event that" is rewritable; "class action" is a concept.
GLUE = {
    "in", "of", "to", "under", "upon", "for", "with", "by", "on", "at",
    "without", "within", "from", "as", "that", "which", "not", "no", "any",
    "each", "every", "all", "such", "shall", "may", "must", "herein",
    "thereof", "where", "when", "if", "unless", "until", "before", "after",
}
# Tokens that only ever appear in citation furniture (Fed. R. Civ. P. 23(b)),
# which the scanner already has rules for and which a reader skips anyway.
CITATION = {"fed", "r", "p", "civ", "subdivision", "section", "sec", "no",
            "a", "b", "c", "d", "e", "f", "g", "h", "i"}


def rows() -> list[dict]:
    """The shipped bank, plus the collected extension when one exists."""
    out = []
    for name in ("bank.jsonl", "bank-ext.jsonl"):
        path = DATA / name
        if path.exists():
            out.extend(json.loads(line) for line in open(path, encoding="utf-8"))
    return out


def in_register(source: str, markers: tuple[str, ...]) -> bool:
    return any(source.startswith(m) for m in markers)


def tokenise(text: str) -> list[str]:
    # "rules--amendment" and "rules- amendment" are one word to a reader and
    # three to the pattern; stripping the dangling hyphen stops the hyphen from
    # becoming the most common token in the register.
    return [w.rstrip("-\u2019'") for w in WORD.findall(text.lower())]


def phrases(text: str, low: int = 2, high: int = 4) -> set[tuple[str, ...]]:
    words = tokenise(text)
    out: set[tuple[str, ...]] = set()
    for n in range(low, high + 1):
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i + n])
            if any(not word for word in gram):
                continue
            if gram[0] in EDGE or gram[-1] in EDGE:
                continue
            out.add(gram)
    return out


def is_glue(gram: tuple[str, ...]) -> bool:
    return any(word in GLUE for word in gram)


def is_citation(gram: tuple[str, ...]) -> bool:
    return all(word in CITATION for word in gram)


def log_odds(docs: list[set[tuple[str, ...]]]) -> dict[tuple[str, ...], float]:
    """Smoothed log-odds of a phrase between the two document sets."""
    target, contrast = docs
    nt, nc = len(target), len(contrast)
    df_t: collections.Counter = collections.Counter()
    df_c: collections.Counter = collections.Counter()
    for doc in target:
        df_t.update(doc)
    for doc in contrast:
        df_c.update(doc)
    alpha = 0.5
    scored: dict[tuple[str, ...], float] = {}
    for gram, count in df_t.items():
        other = df_c.get(gram, 0)
        p_t = (count + alpha) / (nt + 2 * alpha)
        p_c = (other + alpha) / (nc + 2 * alpha)
        scored[gram] = math.log(p_t / (1 - p_t)) - math.log(p_c / (1 - p_c))
    return scored


def known(terms: list[dict]) -> set[str]:
    """Every word a glossary row already accounts for, lowercase."""
    bag: set[str] = set()
    for row in terms:
        for phrase in [row["term"], *row.get("variants", [])]:
            bag.update(WORD.findall(phrase.lower()))
    return bag


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", default="court-rule", choices=sorted(MARKERS))
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--glue", action="store_true",
                    help="only phrases holding a drafting connective")
    ap.add_argument("--keep-citation", action="store_true",
                    help="keep citation furniture such as 'fed r civ p'")
    args = ap.parse_args(argv)

    bank = rows()
    terms = [json.loads(l) for l in open(DATA / "terms.jsonl", encoding="utf-8")]
    markers = MARKERS[args.register]

    target = [r for r in bank if r["label"] == 1 and in_register(r["source"], markers)]
    contrast = [r for r in bank if r["label"] == 0]
    if not target:
        print(f"no {args.register} rows", file=sys.stderr)
        return 1

    df_t: collections.Counter = collections.Counter()
    sets = [phrases(r["text"]) for r in target]
    for doc in sets:
        df_t.update(doc)

    scores = log_odds([sets, [phrases(r["text"]) for r in contrast]])
    covered = known(terms)

    # A phrase every word of which a glossary row already accounts for cannot
    # add vocabulary; it would only add a rule.
    queue = [
        (gram, df_t[gram], scores[gram])
        for gram in scores
        if df_t[gram] >= args.min_df
        and not all(w in covered for w in gram)
        and (not args.glue or is_glue(gram))
        and (args.keep_citation or not is_citation(gram))
    ]
    queue.sort(key=lambda row: (-row[2], -row[1], row[0]))

    print(f"{args.register}: {len(target)} target rows, {len(contrast)} contrast rows, "
          f"{len(queue)} phrases at df >= {args.min_df} outside the glossary\n")
    print(f"{'phrase':44s} {'df':>4s} {'lo':>7s}")
    for gram, count, score in queue[:args.top]:
        print(f"{' '.join(gram):44s} {count:4d} {score:7.2f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "register": args.register,
            "target_rows": len(target),
            "contrast_rows": len(contrast),
            "min_df": args.min_df,
            "queue": [
                {"phrase": " ".join(g), "df": c, "log_odds": round(s, 4)}
                for g, c, s in queue
            ],
        }, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
