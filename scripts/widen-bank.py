#!/usr/bin/env python3
"""Collect text for the registers the shipped bank does not hold.

The coverage report says four registers have no rows at all: contracts,
judicial opinions, regulations and agency notices. A widening effort that
starts with words would be guessing, because there is nothing to read. This
collects the text first, from three keyless public sources, and leaves the
words to a later pass that can be measured against it.

Sources and why these three:

* regulation - eCFR (ecfr.gov/api/versioner/v1). The codified federal rules,
  current, XML, and the register with the densest cross references. The
  endpoint refuses a request that does not accept compression, which the
  client below always sends.
* agency-notice - Federal Register API. Notices, bulletins and orders: text
  that tells a person what to do next, which is the register the plain
  language movement was invented for.
* contract - CUAD (theatticusproject/cuad on the Hugging Face datasets
  server). 510 commercial contracts from SEC filings with clause labels,
  the corpus Hendrycks et al. built precisely because contracts are a
  register of their own.

Every sentence is filtered for shape (length, word count, letter density) and
deduplicated against itself and against the shipped bank. Rows are written to
`bank-ext.jsonl`, never into `bank.jsonl`: the shipped bank has a manifest and
a recorded hash, and an extension that can be deleted in one command is
easier to defend than a bank whose contents drifted.

Usage:
    python3 scripts/widen-bank.py                        # all three
    python3 scripts/widen-bank.py --register regulation  # one
    python3 scripts/widen-bank.py --limit 120 --refresh  # refetch
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "legal"
CACHE = ROOT / "tmp" / "widen"

UA = "xcm-mmm-widen/1.0 (+plain language research)"
ECFR_DATE = "2026-08-10"  # latest issue date reported by titles.json

# Parts chosen for drafting density rather than subject matter: each of these
# is written in the cross referenced, shall heavy style the instrument exists
# to read. (title, part, label)
ECFR_PARTS = [
    (29, "1604", "EEOC sex discrimination guidelines"),
    (24, "982", "housing choice vouchers"),
    (45, "160", "HIPAA privacy rule"),
    (7, "273", "SNAP certification"),
]
# Federal Register notices. The type filter is what makes this the notice
# register rather than rulemaking prose.
FR_NOTICES = 12

MIN_CHARS, MAX_CHARS = 45, 320
MIN_WORDS = 8
MIN_ALPHA = 0.55
MAX_DIGITS = 0.20

SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n{2,}")
SPACE = re.compile(r"\s+")


def cache_path(register: str, name: str, suffix: str) -> Path:
    folder = CACHE / register
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-z0-9._-]+", "-", name.lower())
    return folder / f"{safe}{suffix}"


# The datasets server throttles a burst of paginated rows and answers 429 or
# 502; the Federal Register occasionally 502s under load. Three retries with a
# doubling wait is the difference between a usable corpus and a lost run.
RETRIES = 3
RETRY_STATUS = {429, 500, 502, 503, 504}


def fetch(url: str, path: Path, refresh: bool, accept: str | None = None) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes()
    headers = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    if accept:
        headers["Accept"] = accept
    pause = 1.0 if "huggingface.co" in url else 0.4
    delay = 2.0
    for attempt in range(RETRIES + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                body = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
            path.write_bytes(body)
            time.sleep(pause)  # public and unauthenticated: stay polite
            return body
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUS or attempt == RETRIES:
                raise
            wait = float(error.headers.get("Retry-After") or delay)
            time.sleep(min(wait, 30.0))
            delay *= 2
    raise RuntimeError(f"unreachable: {url}")


def sentences(text: str) -> list[str]:
    out = []
    for chunk in SENTENCE.split(text):
        line = SPACE.sub(" ", chunk).strip()
        if line:
            out.append(line)
    return out


def keep(sentence: str) -> bool:
    if not (MIN_CHARS <= len(sentence) <= MAX_CHARS):
        return False
    words = sentence.split()
    if len(words) < MIN_WORDS:
        return False
    letters = sum(c.isalpha() for c in sentence)
    digits = sum(c.isdigit() for c in sentence)
    if letters / len(sentence) < MIN_ALPHA:
        return False
    if digits / len(sentence) > MAX_DIGITS:
        return False
    if sentence.isupper():
        return False
    if not sentence[0].isalpha() and sentence[0] not in "(\u201c\"":
        return False
    if sentence.endswith((":", ";")):
        # section headings and lead-ins split off mid rule
        return False
    return True


def rows_of(register: str, docs: list[tuple[str, str, str]]) -> list[dict]:
    """Turn (source, url, text) documents into sentence rows."""
    out = []
    for source, url, text in docs:
        for sentence in sentences(text):
            if keep(sentence):
                out.append({"axis": "legal", "label": 1, "source": source,
                            "text": sentence, "url": url})
    return out


def regulation(limit: int, refresh: bool) -> list[dict]:
    docs = []
    for title, part, label in ECFR_PARTS:
        url = (f"https://www.ecfr.gov/api/versioner/v1/full/{ECFR_DATE}"
               f"/title-{title}.xml?part={part}")
        path = cache_path("regulation", f"title-{title}-part-{part}", ".xml")
        root = ET.fromstring(fetch(url, path, refresh, "application/xml"))
        text = " ".join(node.text or "" for node in root.iter() if node.tag in
                        {"P", "FP", "PSPACE", "HEAD"})
        docs.append((f"cfr-{title}-{part}", url, f"{label}. {text}"))
    return rows_of("regulation", docs)[:limit]


def agency_notice(limit: int, refresh: bool) -> list[dict]:
    index_url = ("https://www.federalregister.gov/api/v1/documents.json"
                 f"?per_page={FR_NOTICES}&order=newest"
                 "&conditions%5Btype%5D%5B%5D=NOTICE"
                 "&fields%5B%5D=document_number&fields%5B%5D=raw_text_url")
    path = cache_path("agency-notice", "index", ".json")
    index = json.loads(fetch(index_url, path, refresh))
    docs = []
    for doc in index.get("results", []):
        number = doc["document_number"]
        raw = doc.get("raw_text_url")
        if not raw:
            continue
        body = fetch(raw, cache_path("agency-notice", number, ".txt"), refresh)
        docs.append((f"fedreg-{number}", raw, body.decode("utf-8", "replace")))
    return rows_of("agency-notice", docs)[:limit]


def contract(limit: int, refresh: bool) -> list[dict]:
    """CUAD, sampled evenly across the split so one filing cannot dominate.

    The dataset is one row per paragraph (about 84,000 of them), so a row is a
    clause fragment and consecutive rows belong to one contract. The whole
    text column is pulled once from the parquet export and read a block of a
    hundred paragraphs at a time, which keeps the sentence splitter inside a
    real clause instead of starting every sentence at a heading. The rows API
    was the first attempt and is not used here: it answers 429 under
    pagination and a hundred small requests to avoid one file is a worse
    trade than the file.
    """
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        raise SystemExit("contract needs pyarrow: python3 -m pip install pyarrow")

    url = ("https://huggingface.co/api/datasets/theatticusproject/cuad"
           "/parquet/default/train/0.parquet")
    path = cache_path("contract", "cuad-train", ".parquet")
    fetch(url, path, refresh)
    texts = parquet.read_table(path, columns=["text"]).column("text").to_pylist()

    block, wanted = 100, 24
    step = max(block, len(texts) // wanted) if texts else block
    docs = []
    for index in range(wanted):
        start = index * step
        paragraphs = [t for t in texts[start:start + block]
                      if t and len(t) > 40 and not set(t) <= {"=", "-", " "}]
        if paragraphs:
            docs.append((f"cuad-{start:05d}",
                         "https://huggingface.co/datasets/theatticusproject/cuad",
                         "\n".join(paragraphs)))
    return rows_of("contract", docs)[:limit]


def opinion(limit: int, refresh: bool) -> list[dict]:
    """Judicial opinions from the Caselaw Access Project static files.

    CAP publishes every volume as a metadata list plus one JSON per case, all
    on a public host with no key. Opinions are the register that argues in
    prose instead of clauses, so they are the one where a glossary read off
    statutes is most likely to be silent for the wrong reason.
    """
    volumes = [("f3d", "200"), ("f3d", "600"), ("f3d", "900")]
    per_volume = 10
    docs = []
    for reporter, volume in volumes:
        base = f"https://static.case.law/{reporter}/{volume}"
        meta_path = cache_path("opinion", f"{reporter}-{volume}-index", ".json")
        cases = json.loads(fetch(f"{base}/CasesMetadata.json", meta_path, refresh))
        # A filing is long; the first case of each volume is as good as any,
        # and taking them in order keeps the sample reproducible.
        for case in cases[:per_volume]:
            name = case.get("file_name")
            if not name:
                continue
            name = name if name.endswith(".json") else f"{name}.json"
            path = cache_path("opinion", f"{reporter}-{volume}-{name}", ".json")
            body = json.loads(fetch(f"{base}/cases/{name}", path, refresh))
            opinions = body.get("casebody", {}).get("opinions", [])
            text = "\n".join(o.get("text") or "" for o in opinions)
            if len(text) < 400:
                continue
            cite = (body.get("citations") or [{}])[0].get("cite") or name
            docs.append((f"cap-{reporter}-{volume}-{case.get('id')}",
                         f"{base}/cases/{name}", text))
    return rows_of("opinion", docs)[:limit]


FETCH = {"regulation": regulation, "agency-notice": agency_notice,
         "contract": contract, "opinion": opinion}


def dedupe(rows: list[dict], seen: set[str]) -> list[dict]:
    out = []
    for row in rows:
        key = hashlib.sha1(SPACE.sub(" ", row["text"].lower()).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", choices=sorted(FETCH), action="append")
    ap.add_argument("--limit", type=int, default=250,
                    help="rows per register after filtering")
    ap.add_argument("--refresh", action="store_true",
                    help="refetch instead of using tmp/widen")
    ap.add_argument("--out", type=Path, default=DATA / "bank-ext.jsonl")
    args = ap.parse_args(argv)

    wanted = args.register or sorted(FETCH)
    seen: set[str] = set()
    for line in open(DATA / "bank.jsonl", encoding="utf-8"):
        seen.add(hashlib.sha1(SPACE.sub(" ", json.loads(line)["text"].lower())
                              .encode()).hexdigest())

    report, all_rows = {}, []
    for name in wanted:
        before = len(seen)
        rows = dedupe(FETCH[name](args.limit, args.refresh), seen)
        report[name] = {"rows": len(rows), "duplicates_dropped": len(seen) - before}
        all_rows.extend(rows)
        print(f"{name}: {len(rows)} sentences", file=sys.stderr)

    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                for row in all_rows), encoding="utf-8")
    meta = {"rows": len(all_rows), "registers": report,
            "sources": {"regulation": f"eCFR {ECFR_DATE}",
                        "agency-notice": "Federal Register API",
                        "contract": "theatticusproject/cuad"}}
    (args.out.with_suffix(".json")).write_text(json.dumps(meta, indent=1),
                                               encoding="utf-8")
    print(f"wrote {args.out} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
