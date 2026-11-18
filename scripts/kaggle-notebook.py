#!/usr/bin/env python3
"""Write the Kaggle notebook that runs the xcm-mmm binary, plus its metadata.

The notebook is generated rather than hand edited so the commands it shows stay
in step with the CLI and the dataset slugs. It is written to answer four
questions a reader would otherwise have to guess at:

  * can a dataset binary run at all, given that Kaggle drops the executable bit
    and mounts /kaggle/input read only;
  * does the static build depend on the image's glibc or on the network;
  * how do the fitted artefacts reach the CLI, given that both mounts are read
    only and every command defaults to a path under data/;
  * what does a real workload look like when the tool is the worker inside a
    map over shards, which is the pattern that makes a notebook advanced.

    python3 scripts/kaggle-notebook.py            # stage dist/kaggle-kernel
    python3 scripts/kaggle-notebook.py --push     # stage, then push the kernel

The two dataset paths are read from XCM_INPUT and XCM_WORK, so the same cells
can be executed off Kaggle against local copies before they are pushed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess

SLUG = "xcm-mmm-static-linux-x86-64"
MODELS = "xcm-mmm-models"
KERNEL = "crissymoon/xcm-mmm-binary-in-a-notebook"
OUT = pathlib.Path("dist/kaggle-kernel")


def cell(kind: str, source: str, index: int) -> dict:
    body = source.strip("\n").splitlines(keepends=True)
    # nbformat 4.5 wants an id per cell; papermill warns loudly without one.
    base = {"id": f"c{index:02d}", "metadata": {}, "source": body}
    if kind == "markdown":
        return {"cell_type": "markdown", **base}
    return {"cell_type": "code", "execution_count": None, "outputs": [], **base}


def version() -> str:
    for line in pathlib.Path("Cargo.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    return "0.1.0"


def disclaimer() -> str:
    """The notice the instrument carries, read from the file it ships in."""
    path = pathlib.Path("data/legal/settings.json")
    if not path.exists():
        return ""
    return json.loads(path.read_text()).get("disclaimer", "").strip()


CELLS = [
    ("markdown", f"""
# Running the xcm-mmm binary inside a Kaggle notebook

> **{disclaimer()}**

The notebook does four things and asserts each one, because all four are things
people get wrong when they ship a compiled tool to Kaggle.

1. **A dataset file arrives without the executable bit** and `/kaggle/input` is
   mounted read only. The first cell prints the mode it actually gets; the next
   copies the binary to `/kaggle/working` and chmods it. Running it in place
   fails with `EACCES`, not with a message that explains itself.
2. **The build is a static-pie ELF**, so it does not track the image's glibc.
   The environment cell prints `ldd --version` next to the ELF header so the
   point is visible rather than claimed.
3. **The fitted artefacts are a second dataset**, and they are staged as a
   plain `data/` tree because every command in the CLI defaults to a path under
   it. Copy that tree beside the binary, change directory, and `detect score`,
   `read`, `segment assign`, `legal scan` and `churn check` all run with no path
   flags, exactly as they do in the repository.
4. **No network is needed.** The kernel metadata sets `enable_internet: false`
   and every command below is a local run.

The numeric half of the notebook treats the binary as a worker: a synthetic
panel is written, split into shards, scored by parallel `forecast` runs, and the
shard reports reduce into one table. The text half runs the shipped models over
the 518 ads they were fitted on. A Rust binary in a notebook is useful for the
work it does per core, not for one interactive call.

xcm-mmm {version()} is MIT licensed. Source: https://github.com/crissymoon/xcm-mmm
"""),
    ("code", """
import hashlib
import json
import os
import pathlib
import platform
import shutil
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

IN = pathlib.Path(os.environ.get("XCM_INPUT", "/kaggle/input"))
WORK = pathlib.Path(os.environ.get("XCM_WORK", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)

SRC = IN / \"""" + SLUG + """\" / "xcm-mmm"
ARTEFACTS = IN / \"""" + MODELS + """\" / "data"
RUN = WORK / "xcm-mmm-run"
BIN = RUN / "xcm-mmm"


def sh(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=RUN if RUN.exists() else WORK)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd} -> {r.returncode}\\n{r.stdout}\\n{r.stderr}")
    return r


def run(*args):
    \"\"\"Run a subcommand from the directory that holds the artefacts.\"\"\"
    return sh([str(BIN), *map(str, args)])


print("kernel run type:", os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "?"))
print("python        :", sys.version.split()[0], platform.machine())
print("uname         :", " ".join(platform.uname()))
print("cpu count     :", os.cpu_count())
# ldd is a glibc program; the guard keeps this cell usable off the Kaggle image
# instead of dying on a host that has no such binary.
try:
    glibc = subprocess.run(["ldd", "--version"], capture_output=True, text=True,
                           check=False).stdout.splitlines()
except OSError:
    glibc = []
print("ldd           :", glibc[0] if glibc else "none (not a glibc image)")
# /proc is Linux only; the fallback keeps the cell runnable off Kaggle.
try:
    mem = pathlib.Path("/proc/meminfo").read_text().splitlines()[0]
except OSError:
    mem = f"{platform.machine()} host, /proc not mounted"
print("memory        :", mem)
"""),
    ("markdown", """
## The claim, printed before it is acted on

`stat` on the file inside `/kaggle/input` shows mode `0666` on this image and no
execute bit for anyone; the mount is read only as well, so the chmod cannot
happen in place. Both facts are printed here rather than trusted, and the last
line of the cell is the failure a reader would otherwise have to debug:
`PermissionError` from `subprocess`, raised before the binary runs at all.
"""),
    ("code", """
print("exists   :", SRC.exists())
mode = SRC.stat().st_mode
print("mode     :", oct(stat.S_IMODE(mode)), "exec bit:", bool(mode & 0o111))

try:
    SRC.parent.joinpath(".probe").write_text("x")
    print("input dir is writable")
except OSError as exc:
    print("input dir is read only:", exc.__class__.__name__)

try:
    direct = subprocess.run([str(SRC), "--version"], capture_output=True, text=True)
    print("running in place ->", direct.returncode, (direct.stderr or direct.stdout).strip()[-120:])
except OSError as exc:
    # The interesting failure, not a crash: EACCES from the exec, before the
    # tool ever gets to print anything.
    print("running in place -> refused:", exc.__class__.__name__, exc.strerror)
"""),
    ("markdown", """
## Assemble the layout the defaults were written for

The binary and the artefacts are separate datasets, and both mounts are read
only, so both have to move. The tree is copied to the same directory as the
executable, which is what makes `data/detectors/model.json` resolve without a
`--model` flag. The ELF header is checked before anything is executed from it.
"""),
    ("code", """
RUN.mkdir(parents=True, exist_ok=True)
shutil.copyfile(SRC, BIN)
BIN.chmod(BIN.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
shutil.copytree(ARTEFACTS, RUN / "data", dirs_exist_ok=True)
os.chdir(RUN)

raw = BIN.read_bytes()
print("bytes    :", len(raw))
print("sha256   :", hashlib.sha256(raw).hexdigest())
print("is ELF   :", raw[:4] == b"\\x7fELF")
print("machine  :", int.from_bytes(raw[18:20], "little"), "(62 is x86-64)")
print("type     :", int.from_bytes(raw[16:18], "little"), "(3 is static-pie)")

files = [p for p in (RUN / "data").rglob("*") if p.is_file()]
print("artefacts:", len(files), "files",
      f"{sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
for rel in ("detectors/model.json", "emotion/model.json", "segments/model.json",
            "segments/features.jsonl", "legal/settings.json", "churn/source"):
    print(f"  data/{rel:26s}", (RUN / 'data' / rel).exists())

print(run("--version").stdout.strip())
"""),
    ("markdown", """
## A workload, and the timing that matters

`synth` writes a panel with a known signal, `forecast` runs the walk-forward
search over it behind an embargo, and `--nested` makes the inner loop choose
while the outer loop scores. The number to watch is the honest one: the nested
skill, not the score from the selection that picked itself.
"""),
    ("code", """
panel = WORK / "panel.csv"
print(run("synth", "--rows", 4000, "--features", 24, "--informative", 6,
          "--seed", 7, "--out", panel).stdout.strip())

t0 = time.perf_counter()
flat = run("forecast", str(panel), "--folds", 4, "--horizon", 3,
           "--budget", 3000, "--seed", 1).stdout
flat_s = time.perf_counter() - t0
print(flat)
print(f"elapsed {flat_s:.2f}s on {os.cpu_count()} cores")
"""),
    ("markdown", """
## The pattern: the binary as a worker in a shard map

One `forecast` call already uses every core through rayon, so launching it four
times does not make it four times faster; the thread pool is the parallelism.
What the shard map buys is independence: four panels, four seeds, four runs that
do not contend for a fold. That is how a notebook scales a compiled worker past
one interactive call, and it is how a long sweep survives the 12 hour limit,
because each shard is a finished unit that can be resumed.
"""),
    ("code", """
def shard(i: int) -> dict:
    path = WORK / f"shard{i}.csv"
    run("synth", "--rows", 2000, "--features", 20, "--informative", 5,
        "--seed", 100 + i, "--out", path)
    t0 = time.perf_counter()
    out = run("forecast", str(path), "--folds", 3, "--horizon", 2,
              "--budget", 1500, "--seed", 10 + i).stdout
    elapsed = time.perf_counter() - t0
    best = [ln for ln in out.splitlines() if "best" in ln.lower()]
    return {"shard": i, "seconds": round(elapsed, 2), "best": best[0].strip() if best else out.strip()[:80]}


t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
    reports = list(pool.map(shard, range(4)))
wall = time.perf_counter() - t0

for r in reports:
    print(f"shard {r['shard']}: {r['seconds']:>6.2f}s  {r['best']}")
print(f"wall {wall:.2f}s, sum of shard time {sum(r['seconds'] for r in reports):.2f}s")
"""),
    ("markdown", """
## The text side, now that the artefacts are present

`detect info` proves which model answered and what it was fitted on. `detect
score` reads every axis for one text and prints the evidence term beside the
score. `read` runs one text through the detectors, the emotion model, the
language measures, the segment model and the brand map in a single pass; the
`--ads` flag hands it a copy set so the brand map has a panel to be fitted on.

None of those calls pass a path. They resolve because the working directory is
the one that holds `data/`.
"""),
    ("code", """
print(run("detect", "info").stdout)

print(run("detect", "score",
          "--text", "We are delighted to confirm the renewal signed today.",
          "--text", "The deadline is tonight and they are counting on you.",
          "--text", "National security demands we act now.",
          "--hits", 3).stdout)
"""),
    ("code", """
reading = run("read", "--ads", "data/detectors/ads.jsonl",
              "Our card charges 19.9% APR and the fee applies from day one.").stdout
print("\\n".join(reading.splitlines()[:16]))
print("...")
print("\\n".join(reading.splitlines()[-6:]))
"""),
    ("markdown", """
## Batch: the model as a function over a file

`--input` reads JSON Lines and `--out` writes one scored row per input row, so
the tool stays a filter in a pipeline and the table work happens in pandas. The
518 ads in `data/detectors/ads.jsonl` carry the labels the model was fitted
against and a `train`/`test` split, which makes the output a check on the model
rather than a demo of it.
"""),
    ("code", """
import pandas as pd

SCORED = WORK / "scored.jsonl"
t0 = time.perf_counter()
print(run("detect", "score", "--input", "data/detectors/ads.jsonl",
          "--out", SCORED).stdout.strip())
print(f"{time.perf_counter() - t0:.2f}s")

rows = [json.loads(line) for line in SCORED.read_text().splitlines() if line.strip()]
axes = sorted(rows[0]["axes"])
summary = pd.DataFrame([{
    "axis": a,
    "mean": round(sum(r["axes"][a]["score"] for r in rows) / len(rows), 3),
    "flagged": sum(r["axes"][a]["level"] != "none" for r in rows),
    "clear+": sum(r["axes"][a]["level"] in ("clear", "strong") for r in rows),
} for a in axes])
print(summary.to_string(index=False))
print("rows", len(rows), "| split", {s: sum(r["split"] == s for r in rows) for s in ("train", "test")})

peaks = sorted(rows, key=lambda r: max(v["score"] for v in r["axes"].values()), reverse=True)[:5]
for r in peaks:
    axis, v = max(r["axes"].items(), key=lambda kv: kv[1]["score"])
    flat_text = " ".join(r["text"].split())[:64]
    print(f"  {v['score']:.2f}  {axis:12s} {r['split']:5s} {flat_text}")
"""),
    ("markdown", """
## The other four folders

`segment assign` places entities using `data/segments/features.jsonl`;
`legal scan` reads `data/legal`; `churn check` rebuilds a folder from its own
source and reports disagreements, which is the cheapest way to prove an
artefact has not drifted from what produced it. `emotion coverage` counts how
many words reach an answer above the neutral prior, which is the number that
says whether the emotion model is guessing.
"""),
    ("code", '''
print("\\n".join(run("segment", "info").stdout.splitlines()[:6]))
print()
print("\\n".join(run("segment", "assign", "--limit", 3).stdout.splitlines()[:8]))
print()
print(run("legal", "scan",
          "The tenant waives all claims and indemnifies the landlord against any liability.").stdout)
notice = """The tenant shall pay the rent within five (5) days of the due date.
If the tenant fails to pay, the landlord may terminate this lease."""

brief = run("legal", "explain", notice).stdout
print(brief)
# The notice is part of the instrument, so it must survive the trip.
assert "not legal advice" in brief.lower(), "the brief must carry the research notice"

# The coverage check answers with an exit status, so a pipeline can gate on it:
# every defined term the text needs either has a glossary row or it does not.
clean = run("legal", "explain", notice, "--check")
print("check, known terms :", clean.stderr.strip())
gap = sh([str(BIN), "legal", "explain", "--check",
          "The Fee (the \\"Service Fee\\") means the amount in Schedule 1."], check=False)
print("check, unknown term:", gap.returncode, gap.stderr.strip().splitlines()[-1])
assert clean.returncode == 0, "a brief over known terms must pass the check"
assert gap.returncode != 0, "an unglossed defined term must fail the check"
print(run("churn", "check").stdout)
print("\\n".join(run("emotion", "coverage",
                     "--words", "data/emotion/listed.tsv",
                     "--corpus", "data/emotion/bank.jsonl").stdout.splitlines()[-5:]))
'''),
    ("markdown", """
## What to carry forward

- Copy then chmod. Never assume the executable bit survives a dataset upload.
- Prefer static-pie for anything that ships to a notebook image you do not pin.
- Ship models as a `data/` tree and `chdir` into the directory that holds it:
  the CLI's defaults then do the argument passing, and the notebook reads like
  the documentation instead of like a list of paths.
- Keep `enable_internet: false` and prove the notebook still runs; a kernel that
  quietly needs the network breaks for every reader who leaves the default on.
- Wrap the binary in one `run()` helper that raises with the tool's own stderr,
  and make every step a shard that can be re-run alone.
"""),
]


def notebook() -> dict:
    return {
        "cells": [cell(kind, src, i) for i, (kind, src) in enumerate(CELLS)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def metadata() -> dict:
    return {
        "id": KERNEL,
        "title": "xcm-mmm binary in a notebook",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [f"crissymoon/{SLUG}", f"crissymoon/{MODELS}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="push the kernel after staging")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "notebook.ipynb").write_text(json.dumps(notebook(), indent=1) + "\n")
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    print(f"staged {OUT}/notebook.ipynb ({len(CELLS)} cells)")

    if args.push:
        cli = shutil.which("kaggle") or str(pathlib.Path.home() / "Documents/mcp-v5.4/venv/bin/kaggle")
        r = subprocess.run([cli, "kernels", "push", "-p", str(OUT)])
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
