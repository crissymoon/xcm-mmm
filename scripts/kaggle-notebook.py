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
    map over shards, which is the pattern that makes a notebook advanced;
  * which binary and which artefacts produced the numbers, which is answered by
    a provenance record that travels with the file rather than by a hash in a
    card nobody recomputes.

The evaluation and packaging decisions each cite the source that forced them,
listed in full in README/DOCS/RESEARCH.md and repeated at the end of the
notebook.

The kernel is public: a reader can open it, run it and read the figures without
owning the datasets. Every figure is drawn from numbers the binary printed in
the same session, saved as a PNG under charts/ and displayed inline, so the
pictures cannot drift from the run that produced them.

    python3 scripts/kaggle-notebook.py            # stage dist/kaggle-kernel
    python3 scripts/kaggle-notebook.py --push     # stage, then push the kernel

The two dataset paths are read from XCM_INPUT and XCM_WORK, so the same cells
can be executed off Kaggle against local copies before they are pushed.
"""
from __future__ import annotations

import argparse
import json
import os
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

Four figures follow, and each one reads a number the tool printed in this
session rather than a constant copied into the notebook:

| figure | what it answers |
| --- | --- |
| `01-nested-folds.png` | which fold the nested selection scored highest, and which columns survived all four folds |
| `02-shard-timings.png` | whether four shards in threads beat one shard in wall clock, given rayon already fills the cores |
| `03-detector-axes.png` | how far each of the seven axes moves over 518 ads, and the shape of the top score by split |
| `04-emotion-coverage.png` | how much of the emotion list and of running text reaches an answer above the neutral prior |

The figures are written to `charts/` and kept as part of the kernel output.

xcm-mmm {version()} is MIT licensed. Source: https://github.com/crissymoon/xcm-mmm
"""),
    ("markdown", """
## Research behind the four steps

Two literatures decide what the numbers below are worth: how to evaluate a model
that was chosen from many, and how to ship a compiled artefact somewhere it
cannot be rebuilt.

The numeric half is a nested selection over a synthetic panel, and it is nested
rather than a single cross-validated maximum for a reason that is settled rather
than stylistic. Cawley and Talbot (2010) show the optimism of a selected score
grows with the number of candidates compared against the noise of the criterion,
and Varma and Simon (2006) prescribe the outer loop that `--nested` implements.
Tashman (2000) fixes the rolling-origin vocabulary, which is why the folds are
successive origins rather than shuffled rows, and Bergmeir, Hyndman and Koo
(2018) are why the fold geometry should follow the residual autocorrelation
instead of assuming that blocked folds are always required.

| the question | what answers it | what it forced in this notebook |
| --- | --- | --- |
| is the selected score an estimate? | Cawley and Talbot (2010); Varma and Simon (2006) | the outer loop is run and scored per fold, never as a mean of the inner scores |
| how should folds be laid out? | Tashman (2000); Bergmeir, Hyndman and Koo (2018) | successive origins with an embargo, and four of them described as a small sample |
| which search runs inside | Tanabe and Fukunaga (2013, 2014); Zhang and Li (2007) | the 24 candidates are SHADE and MOEA/D configurations, not arbitrary settings |
| why repeat rather than widen | Cerqueira, Torgo and Mozetič (2020) | the shard map, which repeats a cheap scheme four times |
| can the artefact be trusted? | Reproducible Builds; the Cargo and rustc references | a byte identical rebuild, a build id, and a hash the notebook recomputes |

The full list, with links, is `README/DOCS/RESEARCH.md` in the repository and is
repeated at the end of this notebook. The provenance of the binary this run used
is not a citation: it is printed two cells from here and asserted against the
copy the dataset shipped.
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
## The plotting kit

Every figure below is drawn from output the binary printed in this same run, so
a picture cannot outlive the number behind it. matplotlib is pinned to `Agg`
because a notebook image supplies no display, the PNG is written under `charts/`
first and then displayed from that file, and the palette is named once so four
figures read as one set.
"""),
    ("code", '''
import matplotlib

# Agg is the backend a headless image has; the saved PNG is what gets displayed,
# not a live canvas, so the file on disk and the inline picture are the same
# bytes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import pandas as pd
import re

try:
    from IPython.display import Image, display
except ImportError:  # the cells still run when extracted to a plain script
    def display(*_args, **_kwargs):
        return None

    class Image:  # noqa: D101
        def __init__(self, filename):
            self.filename = filename

CHARTS = WORK / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

INK = {"text": "#1f2933", "grid": "#d7dde3", "bar": "#3b6ea5",
       "accent": "#c25e2a", "muted": "#8a9baa"}


def decorate(ax, axis="y"):
    """House axes: two spines, one grid direction, no box."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["grid"])
    ax.tick_params(colors=INK["text"], labelsize=9)
    ax.grid(axis=axis, color=INK["grid"], linewidth=0.6, alpha=0.7)
    ax.grid(axis="x" if axis == "y" else "y", visible=False)
    ax.set_axisbelow(True)
    return ax


def panels(count=1, width=9.0, height=3.6, **kwargs):
    fig, axes = plt.subplots(1, count, figsize=(width, height), dpi=130, **kwargs)
    fig.patch.set_facecolor("white")
    axes = [axes] if count == 1 else list(axes)
    for ax in axes:
        decorate(ax)
    return fig, axes


def show(fig, name, note):
    """Save, then display the file, so the notebook and charts/ cannot diverge."""
    path = CHARTS / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"{path.name}  {path.stat().st_size / 1e3:5.0f} KB  {note}")
    display(Image(filename=str(path)))
    return path


print("matplotlib", matplotlib.__version__, "| pandas", pd.__version__, "| charts ->", CHARTS)
'''),
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
## What built this, asked of three sources

A checksum in a card proves nothing on its own: it says the file did not change
after it was hashed, not that it came from the commit beside it. Three
independent descriptions of the same artefact are printed here instead.

| source | what it can speak for |
| --- | --- |
| `xcm-mmm provenance --json` | the crate version, the commit, the target and the `rustc` that compiled it, baked in by `build.rs` |
| `provenance.json` beside the binary in the dataset | the sha256 and byte count the packaging step measured, written by the same fetch that produced `dist/kaggle` |
| the interpreter | Python, pandas, matplotlib and the core count the four figures were drawn on |

The two that describe the binary must agree on the hash and on the commit. The
record is written to `provenance.json` in the working directory as well, so a
reader who forks the kernel has the same facts for their own run rather than
mine.
"""),
    ("code", '''
# Nothing in this cell is a constant: every value is read from the binary, the
# dataset or the interpreter, so the record cannot survive a change that the
# code did not make.
stated = json.loads(run("provenance", "--json").stdout)
for key in ("version", "commit", "dirty", "target", "rustc", "built"):
    print(f"{key:<8} {stated[key]}")

digest = hashlib.sha256(BIN.read_bytes()).hexdigest()
card = SRC.parent / "provenance.json"
if card.exists():
    shipped = json.loads(card.read_text())
    agrees = shipped["sha256"] == digest
    print(f"card     sha256 {'agrees' if agrees else 'DIFFERS'} with the mounted file")
    assert agrees, "the mounted binary is not the one the dataset description names"
    assert shipped["commit"].startswith(stated["commit"]), (
        f"card commit {shipped['commit'][:7]} against binary commit {stated['commit']}")
else:
    print(f"card     absent at {card}; only the binary reports for itself")

record = {
    "binary": {**stated, "sha256": digest, "bytes": BIN.stat().st_size},
    "runtime": {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    },
    "datasets": ["crissymoon/xcm-mmm-static-linux-x86-64", "crissymoon/xcm-mmm-models"],
    "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(RUN / "provenance.json").write_text(json.dumps(record, indent=2) + "\\n")
print("wrote    provenance.json,", len(json.dumps(record)), "bytes")
'''),
    ("markdown", """
## A workload, and the timing that matters

`synth` writes a panel with a known signal, `forecast` runs the walk-forward
search over it behind an embargo, and `--nested` makes the inner loop choose
while the outer loop scores. The number to watch is the honest one: the nested
skill, not the score from a selection that picked itself on the rows it is then
scored on.

`--json` and `--out` are what make the result a figure instead of a paragraph:
the JSON carries one record per outer fold, including the columns the inner
search chose, and the CSV holds the same rows for a reader who works in R or in
awk. Neither flag changes the run.
"""),
    ("code", """
panel = WORK / "panel.csv"
FOLDS = WORK / "folds.csv"
print(run("synth", "--rows", 4000, "--features", 24, "--informative", 6,
          "--seed", 7, "--out", panel).stdout.strip())

t0 = time.perf_counter()
report = json.loads(run("forecast", str(panel), "--folds", 4, "--horizon", 3,
                        "--nested", "--inner-folds", 3, "--nested-budget", 3000,
                        "--seed", 1, "--json", "--out", FOLDS).stdout)
nested_s = time.perf_counter() - t0

table = pd.DataFrame([{
    "fold": f["fold"],
    "test": f"{f['test_start']}-{f['test_end']}",
    "mse": f["mse"],
    "r2": f["r2"],
    "ic": f["ic"],
    "skill": f["skill"],
    "columns": len(f["names"]),
} for f in report["outer"]])
print(f"nested   {report['outer_folds']} outer x {report['inner_folds']} inner folds, "
      f"{report['rows']} rows, {report['features']} candidates, {nested_s:.2f}s on {os.cpu_count()} cores")
print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"skill    {report['mean_skill'] * 100:.1f}% +/- {report['spread'] * 100:.1f}   "
      f"columns {report['mean_features']:.1f} of {report['features']}")
print(f"folds.csv {FOLDS.stat().st_size} bytes")

# A nested run that keeps every candidate has selected nothing, and a skill at
# or below zero is worse than predicting the training mean. Both are the
# failures that look like a table of plausible numbers.
assert len(report["outer"]) == 4, "four folds were asked for"
assert report["mean_features"] < report["features"], "selection kept every column"
assert report["mean_skill"] > 0.0, "nested skill must beat the training mean"
"""),
    ("code", '''
fig, (left, right) = panels(2, width=11.0, height=3.9,
                            gridspec_kw={"width_ratios": [1.0, 1.25]})

xs = [f["fold"] for f in report["outer"]]
skill = [f["skill"] for f in report["outer"]]
# Skill and r2 land within a point of each other on this panel, so drawing both
# would hide one behind the other. Skill is the line; r2 keeps its place in the
# tick label, where the table in the cell above prints it too.
low = max(0.0, min(skill) - 0.06)
high = min(1.0, max(skill) + 0.09)
left.plot(xs, skill, "o-", color=INK["accent"], markersize=7, linewidth=1.4,
          label="skill over the mean")
for x, v in zip(xs, skill):
    left.annotate(f"{v * 100:.1f}%", (x, v), textcoords="offset points",
                  xytext=(0, 10), ha="center", fontsize=8, color=INK["text"])
left.axhline(report["mean_skill"], color=INK["muted"], linestyle="--", linewidth=1.0)
left.text(xs[0], low + 0.01, f"mean skill {report['mean_skill'] * 100:.1f}%",
          color=INK["text"], fontsize=8, va="bottom")
left.set_xticks(xs)
left.set_xticklabels([f"fold {x} (r2 {f['r2']:.3f})" for x, f in zip(xs, report["outer"])],
                     fontsize=8)
left.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
left.set_ylim(low, high)
left.set_ylabel("skill, axis over the fold range only", color=INK["text"], fontsize=9)
left.set_title("Nested walk-forward: the inner loop chooses, the outer loop scores",
               fontsize=10, color=INK["text"], loc="left")
left.legend(frameon=False, fontsize=8, loc="upper left")

chosen = report["selection"][:12][::-1]
right.barh([c["name"] for c in chosen], [c["folds"] for c in chosen],
           color=INK["bar"], height=0.62)
for i, c in enumerate(chosen):
    right.text(c["folds"] + 0.06, i, str(c["folds"]), va="center",
               fontsize=8, color=INK["text"])
right.set_xlim(0, report["outer_folds"] + 0.6)
right.set_xlabel("outer folds that chose the column", color=INK["text"], fontsize=9)
right.set_title(f"Which of {report['features']} candidates survived all "
                f"{report['outer_folds']} folds", fontsize=10, color=INK["text"], loc="left")
right.tick_params(axis="y", labelsize=8)
decorate(right, axis="x")

show(fig, "01-nested-folds", "fold skill beside selection stability")
'''),
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
ORIGIN = time.perf_counter()


def shard(i: int) -> dict:
    path = WORK / f"shard{i}.csv"
    run("synth", "--rows", 2000, "--features", 20, "--informative", 5,
        "--seed", 100 + i, "--out", path)
    start = time.perf_counter() - ORIGIN
    out = run("forecast", str(path), "--folds", 3, "--horizon", 2,
              "--budget", 1500, "--seed", 10 + i).stdout
    end = time.perf_counter() - ORIGIN
    best = [ln for ln in out.splitlines() if "best" in ln.lower()]
    return {"shard": i, "start": round(start, 3), "end": round(end, 3),
            "seconds": round(end - start, 3),
            "best": best[0].strip() if best else out.strip()[:80]}


t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
    reports = list(pool.map(shard, range(4)))
wall = time.perf_counter() - ORIGIN

for r in reports:
    print(f"shard {r['shard']}: {r['seconds']:>6.2f}s  started {r['start']:>5.2f}s  {r['best']}")
print(f"wall {wall:.2f}s, sum of shard time {sum(r['seconds'] for r in reports):.2f}s")
"""),
    ("code", '''
fig, axes = panels(width=9.0, height=3.0)
ax = axes[0]
for r in reports:
    ax.barh(f"shard {r['shard']}", r["end"] - r["start"], left=r["start"],
            height=0.55, color=INK["bar"])
    ax.text(r["start"] + wall * 0.012, r["shard"], f"{r['end'] - r['start']:.2f}s",
            va="center", fontsize=8, color="white")
ax.axvline(wall, color=INK["accent"], linestyle="--", linewidth=1.2)
ax.text(wall * 0.985, -0.8, f"wall clock {wall:.2f}s for all four", color=INK["accent"],
        fontsize=8, ha="right")
ax.set_xlim(0, wall * 1.08)
ax.set_ylim(len(reports) - 0.38, -1.1)
ax.set_xlabel("seconds since the pool started", color=INK["text"], fontsize=9)
ax.set_title("Four forecasts at once: the bars overlap, so the wall clock is not their sum",
             fontsize=10, color=INK["text"], loc="left")
decorate(ax, axis="x")

show(fig, "02-shard-timings",
     f"{len(reports)} shards, {sum(r['seconds'] for r in reports):.2f}s of work in "
     f"{wall:.2f}s of wall clock")
'''),
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
    ("code", '''
fig, (left, right) = panels(2, width=11.0, height=3.9,
                            gridspec_kw={"width_ratios": [1.1, 1.0]})

ordered = summary.sort_values("mean")
ys = list(range(len(ordered)))
left.barh(ys, ordered["mean"], 0.6, color=INK["bar"])
for y, mean, flagged in zip(ys, ordered["mean"], ordered["flagged"]):
    left.text(mean + ordered["mean"].max() * 0.02, y, f"{flagged} ads",
              va="center", fontsize=7.5, color=INK["muted"])
left.set_yticks(ys)
left.set_yticklabels(ordered["axis"])
left.set_xlim(0, ordered["mean"].max() * 1.32)
left.set_xlabel("mean score across the ad set, with the ads it flags", color=INK["text"], fontsize=9)
left.set_title(f"Seven axes over {len(rows)} ads", fontsize=10, color=INK["text"], loc="left")
decorate(left, axis="x")

# The top score per ad, split the way the model was split. Train is drawn as
# bars and test as an outline, because 127 rows behind 391 taller bars would
# otherwise be invisible, and the comparison is the point of the panel. The bins
# start at the lowest observed score, not at zero: no ad is near zero and an
# empty half axis would hide the shape.
tops = {split: [max(v["score"] for v in r["axes"].values())
                for r in rows if r["split"] == split]
        for split in ("train", "test")}
start = max(0.0, math.floor(min(map(min, tops.values())) / 0.05) * 0.05)
stop = math.ceil(max(map(max, tops.values())) / 0.05) * 0.05 + 0.02
step = (stop - start) / 20
bins = [start + i * step for i in range(21)]
# Counts per bin would put 391 train rows against 127 test rows and say nothing
# about shape; density is what makes the two curves comparable.
right.hist(tops["train"], bins=bins, density=True, color=INK["bar"], alpha=0.45,
           edgecolor="white", linewidth=0.4, label=f"train ({len(tops['train'])})")
right.hist(tops["test"], bins=bins, density=True, histtype="step", color=INK["accent"],
           linewidth=1.4, label=f"test ({len(tops['test'])})")
right.set_xlim(start, stop)
right.set_xlabel("highest axis score for one ad", color=INK["text"], fontsize=9)
right.set_ylabel("share of ads per bin", color=INK["text"], fontsize=9)
right.set_title("Top score by split", fontsize=10, color=INK["text"], loc="left")
right.legend(frameon=False, fontsize=8)

show(fig, "03-detector-axes",
     f"{len(ordered)} axes, {int(summary['flagged'].sum())} flagged cells of "
     f"{len(rows) * len(ordered)}")
'''),
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
coverage = run("emotion", "coverage",
               "--words", "data/emotion/listed.tsv",
               "--corpus", "data/emotion/bank.jsonl").stdout
for line in coverage.splitlines()[-5:]:
    print(line)
'''),
    ("markdown", """
## Coverage, as a picture

The last count in that cell is the one worth a chart: of the 1 108 tokens in the
corpus, 70.9% reach no tier at all and fall back to the neutral prior, which is
the honest way to say how much of running text the emotion model actually reads.
`prior` is drawn in the accent colour because it is the fallback, not an answer.
"""),
    ("code", '''
# The coverage table is indented two spaces; the summary lines are not, which is
# what keeps "corpus" and "evidence" out of the tier counts.
TIER = re.compile(r"^\\s{2,}(\\w+)\\s+([\\d,]+)\\s+([\\d.]+)%$")
tiers = {m.group(1): (int(m.group(2).replace(",", "")), float(m.group(3)))
         for m in map(TIER.match, coverage.splitlines()) if m}
assert "prior" in tiers, "the coverage table must name the fallback tier"

order = sorted(tiers, key=lambda name: (-tiers[name][0], name))
fig, axes = panels(width=9.0, height=3.6)
ax = axes[0]
bars = ax.bar(order, [tiers[name][0] for name in order], width=0.6,
              color=[INK["accent"] if name == "prior" else INK["bar"] for name in order])
top = max(count for count, _ in tiers.values())
for bar, name in zip(bars, order):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.02,
            f"{tiers[name][1]:.1f}%", ha="center", fontsize=8, color=INK["text"])
ax.set_ylabel("tokens in the corpus", color=INK["text"], fontsize=9)
ax.set_ylim(0, top * 1.15)
ax.set_title("Which tier answers for a token, out of "
             f"{sum(count for count, _ in tiers.values())} tokens",
             fontsize=10, color=INK["text"], loc="left")

show(fig, "04-emotion-coverage",
     f"prior is {tiers['prior'][1]:.1f}% of running text")
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
- Draw every figure from output the same run produced, save the PNG, then
  display the file. A screenshot outlives the run; a chart built from the JSON
  the tool just printed cannot.
- Make the artefact describe itself. A hash in a dataset card is a receipt; the
  same hash recomputed here and compared against the record shipped beside the
  binary is what turns it into evidence.
- Cite what constrains the method. A nested loop is either the procedure the
  literature prescribes or a shape someone liked, and the reader cannot tell
  without the reference.

The kernel is public, so the run above and the four figures under `charts/` are
visible without the datasets: fork it, swap the two `dataset_sources` for your
own binary and artefacts, and the rest of the notebook is the part you keep.
"""),
    ("markdown", """
## References

The evaluation, the search and the packaging, in the order the notebook uses
them. DOIs and the page that explains each decision are in
`README/DOCS/RESEARCH.md` of the repository.

1. Cawley, G. C. and Talbot, N. L. C. (2010). On over-fitting in model selection
   and subsequent selection bias in performance evaluation. JMLR 11, 2079-2107.
   [jmlr.org/papers/v11/cawley10a.html](https://www.jmlr.org/papers/v11/cawley10a.html)
2. Varma, S. and Simon, R. (2006). Bias in error estimation when using
   cross-validation for model selection. BMC Bioinformatics 7, 91.
   [doi:10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91)
3. Bergmeir, C., Hyndman, R. J. and Koo, B. (2018). A note on the validity of
   cross-validation for evaluating autoregressive time series prediction.
   CSDA 120, 70-83. [doi:10.1016/j.csda.2017.11.003](https://doi.org/10.1016/j.csda.2017.11.003)
4. Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy.
   International Journal of Forecasting 16(4), 437-450.
   [doi:10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0)
5. Bergmeir, C. and Benítez, J. M. (2012). On the use of cross-validation for
   time series predictor evaluation. Information Sciences 191, 192-213.
   [doi:10.1016/j.ins.2011.12.028](https://doi.org/10.1016/j.ins.2011.12.028)
6. Cerqueira, V., Torgo, L. and Mozetič, I. (2020). Evaluating time series
   forecasting models. Machine Learning 109(11), 1997-2028.
   [doi:10.1007/s10994-020-05910-7](https://doi.org/10.1007/s10994-020-05910-7)
7. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data
   sets. JMLR 7, 1-30.
   [jmlr.org/papers/v7/demsar06a.html](https://www.jmlr.org/papers/v7/demsar06a.html)
8. Tanabe, R. and Fukunaga, A. (2013). Success-history based parameter
   adaptation for differential evolution. IEEE CEC. DOI 10.1109/CEC.2013.6557555
9. Tanabe, R. and Fukunaga, A. (2014). Improving the search performance of
   SHADE using linear population size reduction. IEEE CEC.
   DOI 10.1109/CEC.2014.6900380
10. Zhang, Q. and Li, H. (2007). MOEA/D: a multiobjective evolutionary algorithm
    based on decomposition. IEEE TEVC 11(6), 712-731.
    DOI 10.1109/TEVC.2007.892759
11. Lavinas, Y., Aranha, C., Ladeira, M. and Campelo, F. (2020). MOEA/D with
    random partial update strategy. IEEE CEC. DOI 10.1109/CEC48606.2020.9185527
12. Reproducible Builds. SOURCE_DATE_EPOCH.
    [reproducible-builds.org/docs/source-date-epoch](https://reproducible-builds.org/docs/source-date-epoch/)
13. The Cargo Book. Profiles, the `strip` option.
    [doc.rust-lang.org/cargo/reference/profiles.html](https://doc.rust-lang.org/cargo/reference/profiles.html)
14. The rustc book. `--remap-path-prefix`.
    [doc.rust-lang.org/rustc/command-line-arguments.html](https://doc.rust-lang.org/rustc/command-line-arguments.html)
15. Kaggle. Notebooks, technical specifications.
    [kaggle.com/docs/notebooks](https://www.kaggle.com/docs/notebooks#technical-specifications)

`provenance.json` in this version's output tab carries the binary's own report,
the runtime versions and the two dataset slugs, so a fork records its own run
rather than inheriting this one.
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
