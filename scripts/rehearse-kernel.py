#!/usr/bin/env python3
"""Run part of the generated kernel here, against the repository, before pushing.

The notebook is written for ``/kaggle/input`` and ``/kaggle/working``, and a bad
cell is otherwise only found by pushing it and reading a failed kernel run. The
cells already accept ``XCM_INPUT`` and ``XCM_WORK``, so staging the binary and
the data tree under ``tmp/rehearse`` is enough to run them on this machine.

    python3 scripts/rehearse-kernel.py                 # the setup and the panel cells
    python3 scripts/rehearse-kernel.py --match route   # any phrase a cell contains

Only the cells needed to reach the phrase are executed: the path setup, the
plotting kit, the copy and chmod, then every cell whose source matches. A full
run takes the forecast shards and is left to Kaggle.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp" / "rehearse"


def stage() -> tuple[pathlib.Path, pathlib.Path]:
    """Lay the two datasets out where the notebook's defaults expect them."""
    if TMP.exists():
        shutil.rmtree(TMP)
    into, work = TMP / "input", TMP / "work"
    for slug, source, name in (
        ("xcm-mmm-static-linux-x86-64", ROOT / "target/release/xcm-mmm", "xcm-mmm"),
        ("xcm-mmm-models", ROOT / "data", "data"),
    ):
        target = into / slug
        target.mkdir(parents=True)
        if source.is_dir():
            shutil.copytree(source, target / name)
        else:
            shutil.copyfile(source, target / name)
    work.mkdir(parents=True)
    return into, work


def wanted(sources: list[str], phrase: str) -> list[str]:
    """The cells that must run: the layout, the kit, then the phrase."""
    keys = ("xcm-mmm-run", "INK = {", "RUN.mkdir", phrase)
    return [src for src in sources if any(key in src for key in keys)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="topics", help="phrase a cell has to contain")
    args = ap.parse_args(argv)

    if not (ROOT / "target/release/xcm-mmm").exists():
        raise SystemExit("build first: cargo build --release")

    into, work = stage()
    os.environ["XCM_INPUT"] = str(into)
    os.environ["XCM_WORK"] = str(work)

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/kaggle-notebook.py")], cwd=ROOT, check=True
    )
    nb = json.loads((ROOT / "dist/kaggle-kernel/notebook.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    cells = wanted(sources, args.match)
    if not any(args.match in src for src in cells):
        raise SystemExit(f"no cell contains {args.match!r}")

    print(f"rehearsing {len(cells)} of {len(sources)} code cells, match {args.match!r}")
    names: dict = {"__name__": "__cell__"}
    for i, src in enumerate(cells):
        try:
            exec(compile(src, f"<cell {i}>", "exec"), names)
        except NameError as exc:
            # A cell that reads a variable an earlier cell computed cannot be
            # rehearsed on its own. Say which name is missing instead of
            # letting a traceback look like a broken notebook.
            print(f"cell {i}: {exc.name} is not set here; this cell depends on "
                  "state built above and has to run in the full kernel",
                  file=sys.stderr)
            return 2

    charts = sorted((work / "charts").glob("*.png"))
    print("charts:", [p.name for p in charts] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
