"""Paths and build facts the notebook generator and its cells share.

`version` and `disclaimer` read the repository rather than restating it, so a
released notebook cannot claim a version or a notice the tool does not carry.
Both resolve against this file rather than the working directory, so a lint that
imports the generator from somewhere else still reads the same tree.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SLUG = "xcm-mmm-static-linux-x86-64"
MODELS = "xcm-mmm-models"
KERNEL = "crissymoon/xcm-mmm-binary-in-a-notebook"
OUT = ROOT / "dist" / "kaggle-kernel"


def version() -> str:
    for line in (ROOT / "Cargo.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    return "0.1.0"


def disclaimer() -> str:
    """The notice the instrument carries, read from the file it ships in."""
    path = ROOT / "data" / "legal" / "settings.json"
    if not path.exists():
        return ""
    return json.loads(path.read_text()).get("disclaimer", "").strip()
