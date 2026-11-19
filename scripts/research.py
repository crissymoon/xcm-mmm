#!/usr/bin/env python3
"""Pull the differential evolution and MOEA/D literature through the house scraper.

The point is provenance: every claim the documentation makes about a technique
has to trace back to a paper whose abstract this script actually downloaded, not
to a memory of one. Pages are cached under research/raw so a rerun reads the same
bytes, and the merged corpus lands in research/papers.json.

    python3 scripts/research.py pull          fetch and merge, then digest
    python3 scripts/research.py digest        reprint the digest from cache
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research")
RAW = os.path.join(OUT, "raw")
CORPUS = os.path.join(OUT, "papers.json")
DIGEST = os.path.join(OUT, "digest.txt")

ARXIV = ("https://export.arxiv.org/api/query?search_query={query}"
         "&start=0&max_results={rows}")
CROSSREF = ("https://api.crossref.org/works?query.bibliographic={query}"
            "&rows={rows}&select=DOI,title,issued,container-title,"
            "is-referenced-by-count,abstract,author")
#: OpenAlex is the only one of the three that returns abstracts for a large
#: share of the IEEE and Elsevier back catalogue, which is where the mechanism
#: papers live. Its abstracts arrive as an inverted index and are rebuilt here.
OPENALEX = ("https://api.openalex.org/works?search={query}&per_page={rows}"
            "&sort=cited_by_count:desc"
            "&select=title,publication_year,cited_by_count,doi,"
            "abstract_inverted_index,primary_location")

#: One query per mechanism this repository implements or is considering. The
#: label is what the digest groups by, so keep it short and mechanism shaped.
ARXIV_QUERIES: tuple[tuple[str, str], ...] = (
    ("shade", 'abs:"differential evolution" AND abs:"success-history"'),
    ("lshade", 'abs:"L-SHADE" OR abs:"success-history based adaptive"'),
    ("jso", 'abs:"differential evolution" AND abs:"population size reduction"'),
    ("restart", 'abs:"differential evolution" AND abs:"restart strategy"'),
    ("opposition", 'abs:"differential evolution" AND abs:"opposition-based learning"'),
    ("moead", 'abs:"MOEA/D" AND abs:"decomposition"'),
    ("moead-de", 'abs:"MOEA/D" AND abs:"differential evolution"'),
    ("moead-resource", 'abs:"MOEA/D" AND abs:"resource allocation"'),
    ("moead-pbi", 'abs:"MOEA/D" AND abs:"penalty-based boundary intersection"'),
    ("many-objective", 'abs:"many-objective" AND abs:"decomposition"'),
    #: The legal front end is scored by the same rule as the optimizer: a claim
    #: about legal English cites a benchmark whose abstract this script read.
    ("legal-bench", 'ti:"LegalBench"'),
    ("legal-lexglue", 'ti:"LexGLUE"'),
    ("legal-cuad", 'abs:"contract review" AND abs:"dataset" AND abs:"legal"'),
    ("legal-benchmark", 'abs:"legal" AND abs:"benchmark" AND abs:"language model"'),
    #: Advanced improvements to the legal front end. Each label names the
    #: mechanism a commit would change, so an empty group means the question
    #: is unread rather than answered.
    ("lex-simplify", 'abs:"lexical simplification" AND abs:"evaluation"'),
    ("complex-word", 'abs:"complex word identification"'),
    ("simplify-metric", 'abs:"text simplification" AND abs:"SARI"'),
    ("readability", 'abs:"readability" AND abs:"legal"'),
    ("deontic", 'abs:"deontic" AND abs:"extraction"'),
    ("clause-type", 'abs:"contract" AND abs:"clause" AND abs:"classification"'),
    ("citation-parse", 'abs:"citation" AND abs:"extraction" AND abs:"legal"'),
    ("term-extract", 'abs:"term extraction" AND abs:"domain" AND abs:"corpus"'),
    ("legalese", 'abs:"legalese"'),
    ("plain-language", 'abs:"plain language" AND abs:"legal"'),
)

CROSSREF_QUERIES: tuple[tuple[str, str], ...] = (
    ("shade", "SHADE differential evolution algorithm with strategy adaptation for global numerical optimization"),
    ("lshade", "improving the search performance of SHADE using linear population size reduction"),
    ("jso", "jSO a new differential evolution algorithm with a self-adaptation and population size reduction"),
    ("moead", "MOEA/D a multiobjective evolutionary algorithm based on decomposition"),
    ("moead-de", "multiobjective optimization problems with complicated Pareto sets MOEA/D and NSGA-II"),
    ("moead-resource", "MOEA/D with dynamic resource allocation for many-objective optimization"),
    ("moead-pbi", "adaptive penalty based boundary intersection decomposition multi-objective"),
    ("survey", "differential evolution a survey of the state-of-the-art"),
    ("legal-bench", "LegalBench a collaboratively built benchmark for measuring legal reasoning in large language models"),
    ("legal-lexglue", "LexGLUE a benchmark for legal language understanding in English"),
    ("legal-cuad", "CUAD an expert-annotated NLP dataset for legal contract review"),
    ("bench-calibration", "on calibration of modern neural networks"),
    ("bench-bootstrap", "bootstrap methods for standard errors confidence intervals and other measures of statistical accuracy"),
    ("bench-mcnemar", "approximate statistical tests for comparing supervised classification learning algorithms"),
)

#: OpenAlex relevance drift is real: an unguarded search for "jSO" returns
#: dentistry. Each query therefore carries the words its titles must contain,
#: and rows that fail the test are dropped before ranking by citations.
OPENALEX_QUERIES: tuple[tuple[str, str, str], ...] = (
    ("shade", "SHADE success-history differential evolution adaptation",
     r"(?i)shade|success.?histor"),
    ("lshade", "L-SHADE linear population size reduction differential evolution",
     r"(?i)l-?shade|linear population size reduction"),
    ("jso", "jSO self-adaptive differential evolution population size reduction",
     r"(?i)jso|j21|\bdifferential evolution\b"),
    ("restart", "differential evolution restart strategy stagnation",
     r"(?i)differential evolution"),
    ("moead", "MOEA/D multiobjective evolutionary algorithm based on decomposition",
     r"(?i)moea/d"),
    ("moead-de", "MOEA/D-DE differential evolution complicated Pareto sets",
     r"(?i)moea/d"),
    ("moead-resource", "MOEA/D dynamic resource allocation many-objective",
     r"(?i)moea/d"),
    ("moead-pbi", "MOEA/D penalty-based boundary intersection adaptive weight",
     r"(?i)moea/d|pbi|boundary intersection"),
    ("moead-weight", "MOEA/D adaptive weight vector adjustment nonuniform Pareto front",
     r"(?i)moea/d|weight vector"),
    ("many-objective", "many-objective evolutionary algorithm decomposition",
     r"(?i)many-objective|decomposition"),
    ("legal-bench", "LegalBench legal reasoning benchmark tasks",
     r"(?i)legalbench|legal reasoning"),
    ("legal-lexglue", "LexGLUE legal language understanding benchmark",
     r"(?i)lexglue"),
    ("legal-cuad", "CUAD contract review dataset",
     r"(?i)cuad|contract review"),
    ("bench-calibration", "calibration of neural networks expected calibration error",
     r"(?i)calibrat"),
    ("bench-bootstrap", "bootstrap confidence intervals evaluation metrics",
     r"(?i)bootstrap"),
    ("bench-mcnemar", "McNemar test comparing classifiers significance",
     r"(?i)mcnemar|significan"),
    ("lex-simplify", "lexical simplification benchmark complex word substitution",
     r"(?i)lexical simplification|complex word"),
    ("complex-word", "complex word identification annotation dataset",
     r"(?i)complex word"),
    ("simplify-metric", "text simplification evaluation SARI metric",
     r"(?i)simplification|sari"),
    ("readability", "readability legal text assessment formula",
     r"(?i)readab"),
    ("deontic", "deontic modality legal rule extraction obligation permission",
     r"(?i)deontic|obligation"),
    ("clause-type", "contract clause classification legal text segmentation",
     r"(?i)clause"),
    ("citation-parse", "legal citation extraction parsing reference",
     r"(?i)citation"),
    ("term-extract", "domain term extraction corpus statistical terminology",
     r"(?i)term extraction|terminolog"),
    ("legalese", "legalese plain english legal writing simplification",
     r"(?i)legalese|plain (english|language)"),
    ("plain-language", "plain language legal documents comprehension",
     r"(?i)plain (language|english)"),
)

#: Both arXiv and Crossref take a free-text query and answer loosely. `abs:"text simplification" AND abs:"SARI"`
#: still returns papers on image synthesis, because the parser matches the
#: token anywhere and the AND is advisory. Every arXiv label therefore carries
#: the words its title and abstract must contain, as OpenAlex does, and a row
#: that fails the test is dropped before it can be counted under the label.
MUST: dict[str, str] = {
    "shade": r"(?i)shade|success.?histor",
    "lshade": r"(?i)l-?shade|linear population size reduction",
    "jso": r"(?i)\bjso\b|population size reduction",
    "restart": r"(?i)differential evolution",
    "opposition": r"(?i)differential evolution",
    "moead": r"(?i)moea/d",
    "moead-de": r"(?i)moea/d",
    "moead-resource": r"(?i)moea/d",
    "moead-pbi": r"(?i)moea/d|boundary intersection",
    "many-objective": r"(?i)many-objective",
    "legal-bench": r"(?i)legalbench",
    "legal-lexglue": r"(?i)lexglue",
    "legal-cuad": r"(?i)\bcuad\b|contract review",
    "legal-benchmark": r"(?i)legal",
    "lex-simplify": r"(?i)lexical simplification",
    "complex-word": r"(?i)complex word",
    "simplify-metric": r"(?i)simplif",
    "readability": r"(?i)readab",
    "deontic": r"(?i)deontic|obligation|permission|prohibition",
    "clause-type": r"(?i)clause",
    "citation-parse": r"(?i)citation",
    "term-extract": r"(?i)term extraction|terminolog",
    "legalese": r"(?i)legalese|plain (english|language)",
    "plain-language": r"(?i)plain (language|english)",
}

#: Two groups are named after a topic a paper can mention in passing, and the
#: substring test above is too loose for them. ``machine-readable`` contains
#: ``readab``, so a paper on machine-readable legislation scored as a
#: readability paper, and a red-teaming paper that recommended a finding be
#: written in plain language scored as a plain-language paper. For these two
#: labels the topic must be in the title, which is where a paper about a topic
#: puts it, and ``machine-readable`` is not a reading of a text.
TITLE: dict[str, str] = {
    "readability": r"(?i)(?<!machine-)(?<!machine )\breadab",
    "plain-language": r"(?i)plain (language|english)",
}


def keeps(label: str, row: dict, extra: str = "") -> bool:
    """Whether one returned row belongs under the label that found it.

    ``extra`` is an OpenAlex query's own title test, which several labels carry
    instead of an entry in MUST. A source with no test at all keeps the row.
    """
    text = row["title"] + " " + row["abstract"][:400]
    for pattern in (MUST.get(label, ""), extra):
        if pattern and not re.search(pattern, text):
            return False
    titled = TITLE.get(label)
    return not titled or bool(re.search(titled, row["title"]))

ATOM = "{http://www.w3.org/2005/Atom}"


def scrape(url: str, timeout: int = 60) -> str:
    """Read one URL with the house scrape tool, which paces its own hosts."""
    binary = shutil.which("scrape") or os.path.expanduser("~/.local/bin/scrape")
    if not os.path.exists(binary):
        return ""
    done = subprocess.run([binary, "--timeout", str(timeout), url],
                          capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else ""


def get(url: str, name: str, timeout: int = 40) -> str:
    """Fetch once and keep the bytes. A failed fetch is never cached.

    arXiv answers a bare urllib client with 429 when several queries arrive in
    one minute, so the scrape tool is tried first and urllib is the fallback.
    """
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, name)
    if os.path.exists(path) and os.path.getsize(path) > 200:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    body = scrape(url)
    if not body.strip().startswith(("{", "<")):
        body = fetch(url, timeout)
    if not body.strip().startswith(("{", "<")):
        raise RuntimeError(f"no usable answer from {url}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    time.sleep(0.6)
    return body


def fetch(url: str, timeout: int = 40, tries: int = 4) -> str:
    """One GET with bounded backoff.

    OpenAlex answers a burst of query strings with 429, and the whole corpus
    used to be lost to one of them. A 429 is a wait, not an answer, so it is
    retried with the server's own Retry-After when one is given, and the wait
    doubles so a busy host is left alone rather than hammered.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "xcm-mmm/0.1"})
    wait = 2.0
    last: Exception | None = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            last = error
            if error.code not in (429, 500, 502, 503, 504):
                raise
            told = error.headers.get("Retry-After") if error.headers else None
            pause = float(told) if told and told.strip().isdigit() else wait
            time.sleep(min(pause, 30.0))
            wait = min(wait * 2, 30.0)
        except urllib.error.URLError as error:
            last = error
            time.sleep(min(wait, 30.0))
            wait = min(wait * 2, 30.0)
    raise RuntimeError(f"no usable answer from {url}: {last}")


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_arxiv(body: str) -> list[dict]:
    rows = []
    for entry in ET.fromstring(body).findall(ATOM + "entry"):
        pick = lambda tag: clean((entry.find(ATOM + tag).text or ""))  # noqa: E731
        rows.append({
            "source": "arxiv",
            "id": pick("id"),
            "title": pick("title"),
            "date": pick("published")[:10],
            "venue": "arXiv",
            "citations": 0,
            "authors": [clean(a.text) for a in entry.findall(ATOM + "author/" + ATOM + "name")][:4],
            "abstract": pick("summary"),
        })
    return rows


def parse_crossref(body: str) -> list[dict]:
    rows = []
    for item in json.loads(body).get("message", {}).get("items", []):
        issued = item.get("issued", {}).get("date-parts", [[None]])[0][0]
        rows.append({
            "source": "crossref",
            "id": "doi:" + item.get("DOI", ""),
            "title": clean((item.get("title") or [""])[0]),
            "date": str(issued or ""),
            "venue": clean((item.get("container-title") or [""])[0]),
            "citations": item.get("is-referenced-by-count", 0),
            "authors": [clean(f"{a.get('given', '')} {a.get('family', '')}")
                        for a in item.get("author", [])][:4],
            "abstract": clean(re.sub(r"<[^>]+>", " ", item.get("abstract") or ""))[:1500],
        })
    return rows


def rebuild_abstract(index: dict | None) -> str:
    """Turn OpenAlex's inverted index back into running text."""
    if not index:
        return ""
    placed: list[tuple[int, str]] = []
    for word, spots in index.items():
        placed.extend((spot, word) for spot in spots)
    placed.sort()
    return clean(" ".join(word for _, word in placed))


def parse_openalex(body: str) -> list[dict]:
    rows = []
    for item in json.loads(body).get("results", []):
        location = item.get("primary_location") or {}
        source = (location.get("source") or {}).get("display_name") or ""
        rows.append({
            "source": "openalex",
            "id": item.get("doi") or "",
            "title": clean(item.get("title")),
            "date": str(item.get("publication_year") or ""),
            "venue": clean(source),
            "citations": item.get("cited_by_count", 0),
            "authors": [],
            "abstract": rebuild_abstract(item.get("abstract_inverted_index"))[:1500],
        })
    return rows


def absorb(found: dict[str, dict], row: dict) -> None:
    """Keep one record per title, taking the longest abstract and hit count."""
    if len(row["title"]) < 13:
        return
    current = found.get(key(row))
    if current is None:
        found[key(row)] = row
        return
    if len(row["abstract"]) > len(current["abstract"]):
        current["abstract"] = row["abstract"]
    current["citations"] = max(current["citations"], row["citations"])
    current["venue"] = current["venue"] or row["venue"]
    current["date"] = current["date"] or row["date"]
    current["authors"] = current["authors"] or row["authors"]


def key(row: dict) -> str:
    return re.sub(r"[^a-z0-9]", "", row["title"].lower())[:80]


def sources() -> set[str]:
    """Which back ends this run may ask. A parked host is one environment
    variable away from being left alone, and `RESEARCH_SOURCES=arxiv` keeps a
    pull cheap when only the arXiv rows are wanted."""
    told = os.environ.get("RESEARCH_SOURCES", "openalex,arxiv,crossref")
    return {name.strip().lower() for name in told.split(",") if name.strip()}


def attempt(misses: list[str], source: str, label: str, work) -> None:
    """Run one query. A host that refuses is recorded, not fatal.

    Losing the whole corpus to one 429 is how a digest ends up with no legal
    benchmarks in it and nobody noticing, so a miss is named in the readings
    beside the counts.
    """
    try:
        work()
    except Exception as error:  # a parked host must not lose the other queries
        misses.append(f"{source}:{label} ({str(error)[:60]})")


def pull(rows: int = 10, wide: int = 40) -> dict:
    found: dict[str, dict] = {}
    misses: list[str] = []
    wanted = sources()

    if "openalex" in wanted:
        for label, query, must in OPENALEX_QUERIES:
            def work(label=label, query=query, must=must):
                body = get(OPENALEX.format(query=urllib.parse.quote(query), rows=wide),
                           f"openalex-{label}.json")
                for row in parse_openalex(body):
                    if keeps(label, row, must):
                        row["query"] = label
                        absorb(found, row)
            attempt(misses, "openalex", label, work)

    if "arxiv" in wanted:
        for label, query in ARXIV_QUERIES:
            def work(label=label, query=query):
                body = get(ARXIV.format(query=urllib.parse.quote(query), rows=rows),
                           f"arxiv-{label}.xml")
                for row in parse_arxiv(body):
                    if not keeps(label, row):
                        continue
                    row["query"] = label
                    absorb(found, row)
            attempt(misses, "arxiv", label, work)

    if "crossref" in wanted:
        for label, query in CROSSREF_QUERIES:
            def work(label=label, query=query):
                body = get(CROSSREF.format(query=urllib.parse.quote(query), rows=rows),
                           f"crossref-{label}.json")
                for row in parse_crossref(body):
                    if not keeps(label, row):
                        continue
                    row["query"] = label
                    absorb(found, row)
            attempt(misses, "crossref", label, work)

    # The must-regexes are applied once more over what the three sources
    # returned. OpenAlex honours its filter per query, but a cached group is
    # merged by title, so a row kept under one label can arrive under another
    # and has to be judged against the label it lands in.
    papers = [row for row in found.values() if keeps(row["query"], row)]
    papers.sort(key=lambda row: (row["query"], -row["citations"], row["date"]))
    readings = {
        "queries": len(OPENALEX_QUERIES) + len(ARXIV_QUERIES) + len(CROSSREF_QUERIES),
        "papers": len(papers),
        "by_source": {s: sum(1 for r in papers if r["source"] == s)
                      for s in ("openalex", "arxiv", "crossref")},
        "with_abstract": sum(1 for r in papers if len(r["abstract"]) > 200),
        "misses": len(misses),
        "missed": misses[:8],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(CORPUS, "w", encoding="utf-8") as handle:
        json.dump({"readings": readings, "papers": papers}, handle, indent=1)
    return readings


def digest(width: int = 420, per_label: int = 5) -> str:
    """Print the most cited rows per mechanism, so a group stays readable."""
    with open(CORPUS, encoding="utf-8") as handle:
        corpus = json.load(handle)
    lines = []
    for reading, value in corpus["readings"].items():
        lines.append(f"{reading}: {value}")
    grouped: dict[str, list[dict]] = {}
    for row in corpus["papers"]:
        grouped.setdefault(row["query"], []).append(row)
    for label, rows in grouped.items():
        lines.append("")
        lines.append(f"## {label}")
        rows = sorted(rows, key=lambda row: -row["citations"])[:per_label]
        for row in rows:
            lines.append(
                f"- {row['title']} ({row['date']}, {row['venue']}, cited {row['citations']})")
            if len(row["abstract"]) > 200:
                lines.append(f"  {row['abstract'][:width]}")
    text = "\n".join(lines) + "\n"
    with open(DIGEST, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "pull"
    if command == "digest":
        print(digest())
        return 0
    readings = pull()
    print(json.dumps(readings, indent=1))
    print(digest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
