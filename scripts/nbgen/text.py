"""The second half of the kernel: the fitted models, then the panel.

One cell per artefact tree, then the two floors that refuse a narrow group.
"""
from __future__ import annotations

from .consts import MODELS, SLUG, disclaimer, version

CELLS_TEXT = [
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
## Deciding what to do, not only what it says

Every command so far describes a text. `route` answers a different question: one
of five actions. A message from a person who cannot log in is answered; an
invoice is forwarded to the queue that pays it; a demand letter is held for a
person; a newsletter is ignored; an application is filed. The evidence is the
header block first, because `Auto-Submitted`, `List-Id` and a null reverse-path
are exact strings, and the class cues second.

The rule is the cheapest expected cost under an asymmetric loss matrix (Elkan,
2001). Answering a newsletter costs ten units and ignoring it costs one; holding
a demand letter costs one and answering it unreviewed costs nine. Those numbers
live in `data/route/policy.json`, so the routing is a rulebook a reader can
argue with rather than a model nobody can inspect.

Two parts of the design are not prices. The vetoes come from the protocol:
RFC 3834 forbids an answer to a message carrying `Auto-Submitted`, to a null
reverse-path, or to a machine mailbox, and no cost matrix may buy past that. The
reject option is Chow's (1970): when the top share is below the threshold, or
the two cheapest actions sit within the margin, the message goes to a person
instead. The shares are shares of evidence, not probabilities, because an
uncalibrated score read as a probability is the error Guo and others (2017)
document.

Nothing here sends mail. A reply or a forward is written as a draft with
`Auto-Submitted: auto-replied` and its `In-Reply-To` set, and a person or a queue
releases it. Every filing is a copy beside a ledger that carries the digest of
the policy that produced it and the second the decision was made, so the run can
be audited after the rulebook changes and the next run can hold the repeat
window of RFC 3834, section 7: a sender answered inside `reply_window_days` is
held for a person instead of answered twice.
"""),
    ("code", """
ROUTE = WORK / "route"
SAMPLES = RUN / "data/route/samples"

print(run("route", "check").stdout.strip())

dry = json.loads(run("route", "organize", "--inbox", SAMPLES, "--out", ROUTE,
                     "--json").stdout)
# A dry run is the default for a reason: it must not create the output tree.
assert not ROUTE.exists(), "the dry run wrote something"
print(f"dry run  {dry['files']} arrivals, no bytes written")

applied = json.loads(run("route", "organize", "--inbox", SAMPLES, "--out", ROUTE,
                          "--apply", "--json").stdout)
rows = pd.DataFrame([{
    "file": row["file"],
    "class": row["class"],
    "action": row["action"],
    "filed as": row["dest"] or "-",
} for row in applied["rows"]])
print(rows.to_string(index=False))

ledger = [json.loads(line) for line in (ROUTE / "ledger.jsonl").read_text().splitlines()]
drafts = sorted(p.name for p in (ROUTE / "drafts").glob("*.eml"))
print(f"ledger   {len(ledger)} rows, policy {applied['policy']['fnv64']}")
print("drafts   ", ", ".join(drafts))

# The counts are the claim: a demand letter is held, a newsletter is ignored,
# and the two invoices reach the same queue whether or not the file had headers.
assert applied["by_action"] == {"file": 1, "forward": 2, "hold": 2, "ignore": 1, "reply": 1}
assert all(row["decision"]["policy"] == applied["policy"]["fnv64"] for row in ledger)

reply = next((ROUTE / "drafts").glob("*-reply.eml")).read_text()
assert "Auto-Submitted: auto-replied" in reply, "RFC 3834 header is missing"
assert "In-Reply-To: <support-1@example.com>" in reply
print()
print("\\n".join(reply.splitlines()[:6]))
"""),
    ("code", """
# The ledger is also the memory behind the repeat window of RFC 3834, section 7.
# Two runs over the same arrivals with a fixed clock, a day apart, show it: the
# second run reads the first run's ledger and refuses the answer it already gave.
WINDOW = WORK / "route-window"
shutil.rmtree(WINDOW, ignore_errors=True)
DAY = 86_400
T0 = 1_760_000_000

first = json.loads(run("route", "organize", "--inbox", SAMPLES, "--out", WINDOW,
                       "--apply", "--at", str(T0), "--json").stdout)
second = json.loads(run("route", "organize", "--inbox", SAMPLES, "--out", WINDOW,
                        "--apply", "--at", str(T0 + DAY), "--json").stdout)

print(f"day 0  {first['by_action']}")
print(f"day 1  {second['by_action']}")
print(f"window {second['window_days']} days, "
      f"ledger held {second['history']['dated_replies']} dated replies, "
      f"{second['window_blocked']} answer refused")

held = {row["file"]: row for row in second["rows"]}
assert first["window_blocked"] == 0, "an empty ledger blocks nothing"
assert second["window_blocked"] == 1, second["window_blocked"]
assert held["support-login.eml"]["action"] == "hold"
assert held["support-login.eml"]["drafts"] == [], "a refused answer wrote a draft"
assert held["support-login.eml"]["dest"] == "review/support-login.eml"
assert second["history"]["dated_replies"] == 1
assert all(row["at"] == T0 + DAY for row in second["rows"]), "the ledger carries no clock"
"""),
    ("code", """
def reading(name):
    out = json.loads(run("route", "decide", "--file", SAMPLES / name, "--json").stdout)
    return out["decision"], {action: cost for action, cost in out["decision"]["expected"]}

legal, legal_cost = reading("legal-demand.eml")
news, news_cost = reading("newsletter.eml")
ORDER = ["reply", "forward", "file", "hold", "ignore"]

width = 0.38
xs = list(range(len(ORDER)))
fig, axes = panels(width=9.6, height=3.9)
ax = axes[0]
for offset, label, doc, cost, colour in (
    (-width / 2, "demand letter", legal, legal_cost, INK["bar"]),
    (width / 2, "list message", news, news_cost, INK["muted"]),
):
    bars = ax.bar([x + offset for x in xs], [cost[a] for a in ORDER], width=width,
                  label=label, color=colour)
    # The chosen action keeps its series colour and gains a hatch, because
    # recolouring it orange would erase which message the bar belongs to.
    chosen = ORDER.index(doc["action"])
    bars[chosen].set_hatch("//")
    bars[chosen].set_edgecolor(INK["accent"])
    bars[chosen].set_linewidth(1.6)

ax.set_xticks(xs)
ax.set_xticklabels(ORDER)
ax.set_ylabel("expected cost, cost units", color=INK["text"], fontsize=9)
ax.set_title("Expected cost per action for two messages: hatched bars are the actions taken",
             fontsize=10, color=INK["text"], loc="left")
# The hatch needs its own key, or it reads as a third series.
taken = matplotlib.patches.Patch(facecolor="white", edgecolor=INK["accent"], hatch="//",
                                 label="action taken (hatched)")
ax.legend(handles=[*ax.containers, taken], frameon=False, fontsize=8, loc="upper right")
top = max(max(legal_cost.values()), max(news_cost.values()))
ax.set_ylim(0, top * 1.15)
for bar in ax.patches:
    ax.annotate(f"{bar.get_height():.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center",
                fontsize=7, color=INK["text"])

show(fig, "05-routing-cost",
     f"hold {legal_cost['hold']:.2f} vs reply {legal_cost['reply']:.2f} for the letter")

print(f"demand letter -> {legal['action']} ({legal['class']}), {len(legal['vetoes'])} vetoes")
print(f"list message  -> {news['action']} ({news['class']}), {len(news['vetoes'])} vetoes")
for line in news["vetoes"]:
    print(" ", line)
"""),
    ("markdown", """
## The panel the ads came from

`topics` reads the creative bank as a panel: 209 creatives, the reach each one
measured, and twelve topics recovered by clustering the terms that share ads.
Reach is the DSA figure for a European ad and the midpoint of a published
impression range for an American one, so the two are never summed into a single
column; the panel total is the union of people, not a sum of platforms.

Two floors keep the ranking honest, and both refuse rather than smooth. A group
under eight ads keeps its count and loses its term rank, because a lift measured
inside six ads describes the six ads. A term needs two ads inside its group
before it counts as that group's language. The cell asks the narrowest area for
its terms and prints the refusal, so the floor is a behaviour the notebook
exercises rather than a sentence in a README.
"""),
    ("code", '''
print(run("topics", "info").stdout)

PANEL = re.compile(r"panel\\s+([\\d,]+) creatives\\s+([\\d,]+) with reach\\s+([\\d,]+) people")
SHAPE = re.compile(r"shape\\s+(\\d+) topics\\s+([\\d,]+) terms")
AREAS = re.compile(r"^areas\\s+((?:\\w+ \\d+\\s*)+)$", re.M)
head = run("topics", "info").stdout
panel, shape = PANEL.search(head), SHAPE.search(head)
assert panel and shape, "topics info must print a panel line and a shape line"
creatives, with_reach, people = (int(x.replace(",", "")) for x in panel.groups())
topics, terms = int(shape.group(1)), int(shape.group(2).replace(",", ""))
assert 0 < with_reach <= creatives, "a creative with reach cannot outnumber the creatives"
print(f"{topics} topics from {terms} terms over {with_reach} creatives")

# The two floors: a group under eight ads refuses a term rank by name.
areas = {name: int(n) for name, n in re.findall(r"(\\w+) (\\d+)", AREAS.search(head).group(1))}
narrowest = min(areas, key=lambda name: areas[name])
if areas[narrowest] < 8:
    refusal = run("topics", "keywords", "--axis", "area", "--name", narrowest).stdout
    print(refusal)
    assert "under the floor" in refusal, "a group under eight ads must refuse its terms"

# The list is a fixed-width table: reach, ads, a padded name, then the tickers.
ROW = re.compile(r"^\\s*([\\d,]+)\\s+(\\d+)\\s{2,}(\\S.*?)\\s*$")
rows = []
for line in run("topics", "list", "--top", 8).stdout.splitlines():
    m = ROW.match(line)
    if m:
        rows.append((m.group(3), int(m.group(1).replace(",", "")), int(m.group(2))))
assert len(rows) >= 8, "the topic list must return the rows it was asked for"
# One creative can share terms with two clusters, and the panel total is a union
# of people rather than a sum of platforms, so only a single topic is bounded.
assert rows[0][1] <= people, "a topic cannot reach more people than the panel"
''' ),
    ("markdown", """
## Reach by topic, as a picture

The bars are the reach the register measured, not a score, and the count beside
each bar is how many ads carried it. The bars do not add up to the panel: a
creative can share terms with two clusters, and a creative that shares no term
with any cluster is left out rather than assigned to the nearest one.
"""),
    ("code", '''
names = [name.split("  ")[0].strip() for name, _, _ in rows]
reach = [value / 1e6 for _, value, _ in rows]
ads = [n for _, _, n in rows]

fig, axes = panels(width=9.0, height=4.2)
ax = axes[0]
bars = ax.barh(names[::-1], reach[::-1], height=0.62, color=INK["bar"])
top = max(reach)
for bar, n in zip(bars, ads[::-1]):
    ax.text(bar.get_width() + top * 0.015, bar.get_y() + bar.get_height() / 2,
            f"{n} ad{'s' if n != 1 else ''}", va="center", fontsize=8, color=INK["text"])
ax.set_xlabel("reach, millions of people", color=INK["text"], fontsize=9)
ax.set_xlim(0, top * 1.16)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", labelsize=8)
ax.set_title(f"Reach per topic, top {len(rows)} of {topics} topics over "
             f"{people / 1e6:.0f} million people in the panel",
             fontsize=10, color=INK["text"], loc="left")
# The y grid is off, so the x grid has to be asked for again on the flipped axes.
decorate(ax, axis="x")

show(fig, "06-hot-topics",
     f"{names[0]}: {reach[0]:.1f} M over {ads[0]} ads")
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
- A rank over a narrow group measures the group, not the language inside it:
  keep the count, drop the rank, and print the reason rather than the score.
- Draw every figure from output the same run produced, save the PNG, then
  display the file. A screenshot outlives the run; a chart built from the JSON
  the tool just printed cannot.
- Make the artefact describe itself. A hash in a dataset card is a receipt; the
  same hash recomputed here and compared against the record shipped beside the
  binary is what turns it into evidence.
- Put the costs in data. A router whose tradeoffs live in code cannot be argued
  with by the person who owns the consequences.
- Keep protocol rules out of the cost matrix. RFC 3834 vetoes an answer whatever
  a number says, and a matrix that could outvote it would be wrong at any price.
- Cite what constrains the method. A nested loop is either the procedure the
  literature prescribes or a shape someone liked, and the reader cannot tell
  without the reference.

The kernel is public, so the run above and the four figures under `charts/` are
visible without the datasets: fork it, swap the two `dataset_sources` for your
own binary and artefacts, and the rest of the notebook is the part you keep.
"""),
    ("markdown", """
## References

The evaluation, the search, the packaging and the routing, in the order the
notebook uses them. DOIs and the page that explains each decision are in
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
16. Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI 2001,
    973-978. [cseweb.ucsd.edu/~elkan/rescale.pdf](https://cseweb.ucsd.edu/~elkan/rescale.pdf)
17. Chow, C. K. (1970). On optimum recognition error and reject tradeoff.
    IEEE Transactions on Information Theory 16(1), 41-46.
    [doi:10.1109/TIT.1970.1054406](https://doi.org/10.1109/TIT.1970.1054406)
18. Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of
    modern neural networks. ICML 2017, PMLR 70, 1321-1330.
    [proceedings.mlr.press/v70/guo17a.html](https://proceedings.mlr.press/v70/guo17a.html)
19. Parasuraman, R. and Riley, V. (1997). Humans and automation: use, misuse,
    disuse, abuse. Human Factors 39(2), 230-253.
    [doi:10.1518/001872097778543886](https://doi.org/10.1518/001872097778543886)
20. Moore, K. (2004). Recommendations for automatic responses to electronic
    mail. RFC 3834.
    [rfc-editor.org/rfc/rfc3834.html](https://www.rfc-editor.org/rfc/rfc3834.html)
21. Guenther, P. and Showalter, T. (2008). Sieve: an email filtering language.
    RFC 5228.
    [rfc-editor.org/rfc/rfc5228.html](https://www.rfc-editor.org/rfc/rfc5228.html)

`provenance.json` in this version's output tab carries the binary's own report,
the runtime versions and the two dataset slugs, so a fork records its own run
rather than inheriting this one.
"""),
]
