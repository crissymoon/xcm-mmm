#!/usr/bin/env python3
"""Check that the documentation is internally consistent.

Three rules, all cheap and all things a reader notices:

* every relative link resolves to a file that exists,
* every page linked from the README exists and is listed once,
* no em dash or en dash sneaks into the prose.

Run from the repository root. Exit status is 1 on any failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DOCS = ROOT / "README" / "DOCS"

LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
BAD_DASHES = ("\u2014", "\u2013")


def markdown_files() -> list[Path]:
    files = [README]
    files.extend(sorted(DOCS.glob("*.md")))
    return [f for f in files if f.exists()]


def check_links(path: Path) -> list[str]:
    problems: list[str] = []
    text = path.read_text(encoding="utf-8")
    for label, target in LINK.findall(text):
        target = target.strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative = target.split("#", 1)[0]
        if not relative:
            continue
        resolved = (path.parent / relative).resolve()
        if not resolved.exists():
            problems.append(f"{path.relative_to(ROOT)}: link {label!r} points at missing {relative}")
    return problems


def check_dashes(path: Path) -> list[str]:
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for dash in BAD_DASHES:
            if dash in line:
                problems.append(
                    f"{path.relative_to(ROOT)}:{lineno}: em or en dash in prose"
                )
                break
    return problems


def check_docs_index() -> list[str]:
    problems: list[str] = []
    if not README.exists():
        return ["README.md is missing"]
    text = README.read_text(encoding="utf-8")
    linked = {
        target.split("#", 1)[0]
        for _, target in LINK.findall(text)
        if target.startswith("README/DOCS/")
    }
    on_disk = {f"README/DOCS/{p.name}" for p in sorted(DOCS.glob("*.md"))}
    for missing in sorted(on_disk - linked):
        problems.append(f"README.md: {missing} exists but is not linked")
    for extra in sorted(linked - on_disk):
        problems.append(f"README.md: links {extra} which does not exist")
    return problems


def main() -> int:
    files = markdown_files()
    problems: list[str] = []
    for path in files:
        problems.extend(check_links(path))
        problems.extend(check_dashes(path))
    problems.extend(check_docs_index())

    if problems:
        for problem in problems:
            print(f"docs: {problem}", file=sys.stderr)
        print(f"docs: {len(problems)} problem(s) in {len(files)} file(s)", file=sys.stderr)
        return 1
    links = sum(len(LINK.findall(p.read_text(encoding="utf-8"))) for p in files)
    print(f"docs: {len(files)} files, {links} links, no problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
