#!/usr/bin/env python3
"""Check that the documentation is internally consistent.

Four rules, all cheap and all things a reader notices:

* every relative link resolves to a file that exists,
* every page linked from the README exists and is listed once,
* no em dash or en dash sneaks into the prose,
* no generated sentence joins two words that a line continuation split apart.

The last rule exists because Rust drops the newline and the leading whitespace
of a continued string literal, so ``"one two and\\`` followed by ``"three"``
renders as ``andthree``. The churn card shipped that bug twice, in prose nobody
reads twice, and it is invisible to every compiler check.
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


def check_continuations() -> list[str]:
    """Flag a Rust string continuation that glues two words together.

    A line ending in a lone backslash continues a string literal, and Rust
    strips the newline plus the indent of the next line. If the text on either
    side of that seam ends and starts with a word character, the rendered text
    loses a space that the source appears to have.
    """
    problems: list[str] = []
    sources = sorted((ROOT / "src").rglob("*.rs"))
    for path in sources:
        lines = path.read_text(encoding="utf-8").splitlines()
        for index in range(len(lines) - 1):
            line = lines[index].rstrip()
            if not line.endswith("\\") or line.endswith("\\\\"):
                continue
            left = line[:-1]
            right = lines[index + 1].lstrip()
            if not left or not right or left.endswith(" "):
                continue
            if left.endswith("\\n") or left.endswith("\\r"):
                continue
            joined = left[-1].isalnum() or left[-1] in "`)]"
            starts = right[0].isalnum() or right[0] == "`"
            if joined and starts:
                problems.append(
                    f"{path.relative_to(ROOT)}:{index + 1}: continuation joins "
                    f"{left[-24:]!r} to {right[:24]!r} without a space"
                )
    return problems


def main() -> int:
    files = markdown_files()
    problems: list[str] = []
    for path in files:
        problems.extend(check_links(path))
        problems.extend(check_dashes(path))
    problems.extend(check_docs_index())
    problems.extend(check_continuations())

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
