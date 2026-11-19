"""The first half of the kernel: packaging, provenance, the numeric run.

These cells prove the binary runs and use it as a worker; the text cells follow.
"""
from __future__ import annotations

from .consts import MODELS, SLUG, disclaimer, version

CELLS_SETUP = [
    ("markdown", f"""
# Running the xcm-mmm binary inside a Kaggle notebook

> **{disclaimer()}**

The notebook does five things and asserts each one, because all five are things
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
5. **The binary can name its own origin.** `xcm-mmm provenance --json` reports
   the commit, target and toolchain compiled in by `build.rs`, and the notebook
   hashes the file and compares it with the record shipped beside it. A hash in
   a card is a receipt; the same hash recomputed by the consumer is evidence.

The numeric half of the notebook treats the binary as a worker: a synthetic
panel is written, split into shards, scored by parallel `forecast` runs, and the
shard reports reduce into one table. The text half runs the shipped models over
the 518 ads they were fitted on. A Rust binary in a notebook is useful for the
work it does per core, not for one interactive call.

The routing section is different in kind: `route` turns a reading into an
action, so the notebook checks the decision a message receives, not only the
score it carries. The section after it reads the panel the ads were drawn from,
where two floors refuse to rank a group too narrow to describe.

Six figures follow, and each one reads a number the tool printed in this
session rather than a constant copied into the notebook:

| figure | what it answers |
| --- | --- |
| `01-nested-folds.png` | which fold the nested selection scored highest, and how often each column was chosen |
| `02-shard-timings.png` | whether four shards in threads beat one shard in wall clock, given rayon already fills the cores |
| `03-detector-axes.png` | how far each of the seven axes moves over 518 ads, and the shape of the top score by split |
| `04-emotion-coverage.png` | how much of the emotion list and of running text reaches an answer above the neutral prior |
| `05-routing-cost.png` | why a demand letter is held while a newsletter is ignored, from the cost of each action |
| `06-hot-topics.png` | which topics carried the panel's reach, and how many ads each one needed |

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
repeated at the end of this notebook. The provenance of the binary this run uses
is not a citation: it is printed in the provenance section below and asserted
against the record the dataset shipped.
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
import matplotlib.patches

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
# The label sits on the dashed line it names, at the right edge. Putting it at
# the bottom of the axis read as a caption for the axis rather than for the mean.
left.text(xs[-1] + 0.35, report["mean_skill"],
          f"mean {report['mean_skill'] * 100:.1f}%", color=INK["muted"],
          fontsize=8, va="bottom", ha="right")
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
right.set_title(f"How often each column was chosen, top {len(chosen)} of "
                f"{report['features']} over {report['outer_folds']} folds",
                fontsize=10, color=INK["text"], loc="left")
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
]
