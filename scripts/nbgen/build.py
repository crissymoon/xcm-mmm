"""Assemble the kernel: cells in order, then the metadata that pins them."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess

from .consts import KERNEL, MODELS, OUT, SLUG
from .setup import CELLS_SETUP
from .text import CELLS_TEXT

CELLS = CELLS_SETUP + CELLS_TEXT

if not CELLS or any(kind not in {"markdown", "code"} or not src.strip() for kind, src in CELLS):
    raise SystemExit("nbgen: a cell is empty or names no known kind")


def cell(kind: str, source: str, index: int) -> dict:
    body = source.strip("\n").splitlines(keepends=True)
    # nbformat 4.5 wants an id per cell; papermill warns loudly without one.
    base = {"id": f"c{index:02d}", "metadata": {}, "source": body}
    if kind == "markdown":
        return {"cell_type": "markdown", **base}
    return {"cell_type": "code", "execution_count": None, "outputs": [], **base}


def notebook() -> dict:
    return {
        "cells": [cell(kind, src, i) for i, (kind, src) in enumerate(CELLS)],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def metadata() -> dict:
    # A notebook that names a dataset by slug alone follows every future
    # version of it, so a re-run can quietly score a different binary. The two
    # version environment variables append the number as a third path segment,
    # which is the documented pin, and leave the slug alone when unset.
    def pinned(slug: str, version: str) -> str:
        version = version.strip()
        return f"crissymoon/{slug}" + (f"/{version}" if version else "")

    binary = pinned(SLUG, os.environ.get("XCM_BINARY_VERSION", ""))
    models = pinned(MODELS, os.environ.get("XCM_MODELS_VERSION", ""))
    return {
        "id": KERNEL,
        "title": "xcm-mmm binary in a notebook",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        # Public: the figures and the assertions are the point, and a reader who
        # cannot open the kernel cannot check either one.
        "is_private": False,
        "enable_gpu": False,
        "enable_tpu": False,
        # No network, on purpose: the notebook has to prove a static binary and
        # a local data tree are all it needs.
        "enable_internet": False,
        "dataset_sources": [binary, models],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="push the kernel after staging")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "notebook.ipynb").write_text(json.dumps(notebook(), indent=1) + "\n")
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    print(f"staged {OUT}/notebook.ipynb ({len(CELLS)} cells)")

    if args.push:
        cli = shutil.which("kaggle") or str(
            pathlib.Path.home() / "Documents/mcp-v5.4/venv/bin/kaggle"
        )
        r = subprocess.run([cli, "kernels", "push", "-p", str(OUT)])
        if r.returncode != 0:
            return r.returncode
    return 0
