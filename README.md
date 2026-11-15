# xcm-mmm

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/crissymoon/xcm-mmm/blob/main/LICENSE)
[![Rust 1.75+](https://img.shields.io/badge/rust-1.75%2B-blue.svg)](https://github.com/crissymoon/xcm-mmm)
[![Website](https://img.shields.io/badge/website-xcaliburmoon.net-black.svg)](https://xcaliburmoon.net/)

Differential evolution and MOEA/D in Rust, with walk-forward forecasting.

xcm-mmm solves single and multi-objective optimisation problems with two research grade engines. SHADE carries a success history of scale factors and crossover rates, an external archive and the L-SHADE population schedule; jDE and the classic Storn and Price algorithm are included for comparison. MOEA/D-DE decomposes a front into weighted subproblems, mates inside a neighbourhood and updates a bounded number of neighbours per child. Both engines evaluate on a rayon pool and both are bit for bit reproducible at any thread count.

The forecasting module turns feature selection into a search problem over blocked walk-forward folds. A ridge model is fitted on a training block, scored on the block that follows, and the search sees only the cross-validated error. Run it with two objectives and it returns the whole accuracy against model size curve, so a smaller model can be chosen on evidence rather than hope.

**Version:** 0.1.0

## Why this exists

Most of the prediction accuracy in a panel is lost before the model is fitted: features are chosen on the same rows they are scored on, the split ignores time, and a handful of noisy columns are kept because they helped in sample. This crate makes the honest version the easy one.

- Ridge is closed form, deterministic and fast enough to be evaluated tens of thousands of times inside an evolutionary loop.
- Every fold trains on the past and tests on the future, with an embargo between them, so a label can never look at its own future.
- The single objective search minimises error plus a sparsity penalty; the two objective search returns the trade-off between error and model size.
- The report is out of sample: mean squared error, R2, mean absolute error, rank information coefficient, and skill against predicting the training mean.

On the synthetic panel the search lowers its own walk-forward error: all 12 features score R2 0.661, and the 4 true features score 0.695. That comparison flatters the search, because it scores every fold it also uses to choose the subset. Holding the last quarter of the panel back with `--holdout 0.25` removes the advantage: 0.7026 for the selected four against 0.7058 for all twelve. On a wide noisy panel selection pays; on a panel whose columns are already few and informative it does not, and the tool is built to show you which case you are in.

## What is inside

- **SHADE and L-SHADE.** Current-to-pbest mutation, success history memory with a weighted Lehmer mean for F, Cauchy sampling for F and normal sampling for CR, an external archive, and the linear population reduction schedule.
- **jDE and classic DE.** Per individual self adaptation, and the Storn and Price rand/1/bin baseline, with binomial or exponential crossover.
- **Constraints.** Deb's feasibility rules everywhere: a feasible point always beats an infeasible one, and two infeasible points compare by total violation.
- **MOEA/D-DE.** Das and Dennis simplex weights, Euclidean neighbourhoods, Tchebycheff, PBI and weighted sum scalarisation, neighbourhood mating with delta, two neighbour replacements per child, polynomial mutation, and partial updates that vary a fifth of the subproblems per generation, weighted by relative improvement with a stagnation reset.
- **Quality indicators.** Exact hypervolume in two and three dimensions, seeded Monte Carlo above that with a reported error, IGD, generational distance, spacing, and the standard ZDT and DTLZ reference fronts.
- **Parallel and deterministic.** Evaluation runs on rayon. Mutation, selection and all random draws stay on one thread, so the result does not depend on the thread count.
- **Forecasting.** CSV and JSON Lines reading, blocked walk-forward folds with an embargo, ridge regression by Cholesky, and a report with per fold detail.
- **Corpus cleaning and pairs.** Character folding, markup and boilerplate screening, quality rules with a called-out reason per rejection, MinHash near duplicate clustering, two selection rules (thresholded marginal relevance and target-aware importance resampling), and a pair miner that writes rank, preference and contrast banks in one schema with an audit that gates leakage, triviality and contamination.
- **Seven text detectors.** Two learned axes (legal register, abstraction) fitted as smoothed log odds over a labelled bank plus a logistic regression over surface readings, five declared cue indices behind a gate (fear, publicity, manipulation, followers, narrative), and an artifact that carries its terms, cuts, metrics and audit beside the data it was fitted on.
- **Emotion readings.** Eight Plutchik axes over the whole English vocabulary: a 673 word seed, a VADER valence list, a labelled bank of twenty contact-centre transcripts, suffix and prefix derivation, and a character-analogy fallback that needs a shared opening three letters unless the spelling is very close. Seven tiers, every word in a list resolved to one of them, and `emotion coverage` counts which tier answered for a 235,976 word dictionary and for running text.
- **Churn outcome.** A 7,043 customer telecom panel into a churn label, a retention hazard by tenure month with right censoring, and the 1,869 churn reasons read on the emotion axes. A fourteen-feature logistic with folds assigned by customer id reports held-out AUC beside the AUC of tenure and monthly charge alone.
- **Behavioural segments.** Advertisers grouped by what they did in an ad panel: volume, cadence and its spread, banded spend, copy length, terms bought against, creative reuse and copy concentration. k-means++ with a silhouette sweep, a shuffled control and a resampling audit, and a folder that rebuilds from its own entities.
- **One transformer for every task.** A byte level byte pair tokenizer with no unknown token, a pre norm encoder whose backward pass is checked against finite differences, six task heads on one shared trunk, AdamW with a warmup and cosine schedule, early stopping on the validation split the pairs carry, and one model file holding the tokenizer, the config and the weights.
- **One report per text.** `read` puts one piece of copy through the detectors, the emotion scanner, the segment centres and the learned copy map and prints every score beside its evidence: the words that carried the emotional mass, the cues that fired, the surface readings behind a register, Flesch ease and grade, and the segment dimensions the text cannot show.
- **Two front ends.** A library with a small trait surface, and a CLI that can run a benchmark suite, solve a problem or score a panel.

## Training data

`data/training/` is the set this tool ships: 20,000 rank, preference and contrast pairs mined from the public marketlab news corpus, in `train.jsonl`, `valid.jsonl` and `test.jsonl` beside a generated `card.md` and a `manifest.json` that names the digest and row count of every file.

The set is a stratified sample of a retention bank that keeps growing, so it is fixed rather than current. The same source and the same seed rebuild the same bytes:

```bash
./xcm-mmm dataset build --input bank.jsonl --out data/training --budget 20000 --seed 20261114
./xcm-mmm dataset check --dir data/training
```

`dataset check` reads the folder back, parses every row and compares each file with its manifest, so a copy that lost a line or a byte is caught rather than trained on. The full path from labelled rows is one script:

```bash
scripts/train-data.sh --rows rows/ --out data/training
```

It mines each month, folds the pairs into a bank that only improves, samples the bank, and checks the result. The bank lives under `tmp/` and stays out of the repository because it grows every week; what enters version control is the sample and the card that describes it.

## Text detectors

`data/detectors/` is the second shipped set: seven text detectors over advertising copy, fitted in Rust from the three files that travel beside the model.

| axis | method | readings | held out |
|---|---|---|---:|
| legal | learned lexicon and logistic | 120 terms, 3 features | balanced 0.840, accuracy 0.860 |
| jargon | learned lexicon and logistic | 120 terms, 4 features | balanced 0.766, accuracy 0.827 |
| fear | declared cue index | 4 cue groups | fires on 10.6 percent |
| publicity | declared cue index | 4 cue groups | fires on 0.0 percent |
| manipulation | declared cue index | 6 cue groups | fires on 0.0 percent |
| followers | declared cue index | 4 cue groups | fires on 0.6 percent |
| narrative | declared cue index | 3 cue groups | fires on 0.8 percent |

`bank.jsonl` holds 2,847 labelled sentences: legal register from Federal Register abstracts and abstraction from grants.gov synopses, each set against advertisers' own copy as the negative side. `ads.jsonl` holds 518 ads with their source split, so the index references are fitted on training ads and nothing else. `cues.json` holds the declared cue table and the feature list, exported from the Python module that produced the published readings rather than retyped, so a pattern cannot drift between the two.

The learned axes are fitted in two steps. Smoothed log odds (Monroe, Colaresi and Quinn) give every term a z from the two banks behind three one-sided filters: occurrences only in the positive bank, a frequency ceiling on both sides, and a document frequency ceiling in the negative bank that switches off when the negative bank is too small to have a granularity. A logistic regression then reads that density beside the surface readings: citations and authority verbs for legal, nominalisations, agentless passives and syllables for jargon. Both axes are scored out of fold, with folds assigned by the digest of the text so a repeated sentence cannot straddle a fold, and the same loop runs again on shuffled labels. Legal scores 0.840 balanced against a 0.734 majority and 0.487 on its control; jargon 0.766 against 0.738 and 0.496.

The other five have no labelled corpus, so nothing here is an accuracy. They ship as declared cue sets behind a gate, where a construct that names no gated component scores zero, and the audit reports whether two disjoint halves of the cues pick out the same ads. Fear's halves agree on 47 percent of the corpus and correlate at -0.08, which is the published finding rather than a defect: harm and urgency travel apart in this corpus. Publicity and manipulation never fire on these 518 ads, which is a fact about Los Angeles political and issue advertising, not a claim that the constructs are absent from the form.

```bash
./xcm-mmm detect train --bank data/detectors/bank.jsonl --ads data/detectors/ads.jsonl \
    --cues data/detectors/cues.json --out data/detectors
./xcm-mmm detect score --text "guaranteed returns, before it is too late"
./xcm-mmm detect info
./xcm-mmm detect check --dir data/detectors
```

`detect check` rebuilds the folder from its own inputs, compares the result with the shipped fit and runs every declared example in the cue table, so a model edited after its manifest was stamped is a fault rather than a note. `scripts/detect-data.sh` runs the whole path: export the three inputs, fit, check.

## Behavioural segments

`data/segments/` is the third shipped set: advertisers grouped by what they did in an ad panel, fitted in Rust. Nothing descriptive enters a feature. The eight readings are ad volume, the median gap between the days it started an ad, the spread of those gaps, the published spend band midpoint, copy length, terms bought against, creative reuse and copy concentration.

`features.jsonl` holds 251 advertisers and the 1,355 ads behind them, folded from the archive panel by `segment extract`. The fit standardises each column, clips at three standard deviations, and searches k from 3 to 6, because a two way split of an advertising panel is a high and low score rather than a segmentation. The winner is the highest silhouette among the partitions that keep every group above five percent of the panel.

| segment | advertisers | share | behaviour |
|---|---:|---:|---|
| creative reuse high | 57 | 22.7% | 4.2 ads, 0.95 of them repeating one fingerprint, 0.90 concentration |
| start gap spread high, ad volume high | 42 | 16.7% | 20.5 ads, gap spread 1.04, 0.52 concentration |
| creative reuse low, ad volume low | 115 | 45.8% | one ad each, bought once |
| copy concentration low, creative reuse low | 37 | 14.7% | 3.9 ads, 0.40 concentration, 0.08 reuse |

Four groups were chosen at a silhouette of 0.286. There are no labels, so that is not an accuracy: it is the separation the partition achieves. Two audits decide whether the fit ships. The same protocol on columns shuffled independently scores 0.168, so the groups are not an artefact of the method, and refitting on fifty resamples of the advertisers agrees with the full panel at 0.759 mean adjusted Rand index, tenth percentile 0.552. A fit below either floor is refused with a reason rather than written.

```bash
./xcm-mmm segment extract --ads panel.jsonl --out data/segments/features.jsonl
./xcm-mmm segment fit --features data/segments/features.jsonl --out data/segments
./xcm-mmm segment info
./xcm-mmm segment assign --entities data/segments/features.jsonl
./xcm-mmm segment check --dir data/segments
```

`segment check` rebuilds the folder from its own entities, compares the result with the shipped fit and verifies each digest, so a centroid edited after the manifest was stamped is a fault rather than a note. `scripts/segment-data.sh` runs the whole path: extract the panel, fit, check.

### Learned copy map

`textmap.json` asks the same question with a fitted model instead of a distance: a multinomial logistic map from what copy reads as, plus 96 hashed vocabulary buckets, onto the segment the advertiser was assigned. Two units are fitted from the panel and only one of them clears its gate.

| unit | input | rows | accuracy | balanced | chance | shuffled | used |
|---|---|---|---|---|---|---|---|
| ad | one ad's copy | 1,355 | 44.5% | 30.8% | 25.0% | 25.3% | no |
| brand | one advertiser's copy set | 251 | 55.4% | 45.5% | 25.0% | 25.5% | yes |

Both readings come from five folds that hold out whole page ids, so no ad of a scored advertiser was seen in training. The ad unit is withheld because 63.5% of the panel sits in one segment: at 44.5% overall it is worse than naming that segment every time, and its 30.8% balanced is six points over chance, real but too thin to act on. The brand unit clears the gate at 45.5% balanced and 55.4% overall, which is what a copy set adds: cross-ad reuse, vocabulary breadth and the size of the set, none of which one ad shows. Its lift is 27 percent of the way from chance to perfect.

The gate is explicit: at least two points over chance, at least ten percent lift, better than the largest class, and two points over the shuffled-label control. A unit that fails is still written and still measured, and is never used to place a text.

```bash
./xcm-mmm segment text-fit --ads panel.jsonl --out data/segments
./xcm-mmm segment text-check --dir data/segments
./xcm-mmm read --ads brand.jsonl
```

`text_features.jsonl` ships the derived rows, readings and vocabulary buckets with no ad copy, so `segment text-check` rebuilds the map offline and the digest in `textmap.json` covers the rows it was fitted from. `read` prints the withheld measurement for one text rather than a placement, and uses the brand unit only when `--ads` names a set of more than one ad, taken from JSON Lines with a `text` field or one ad per line.

Two limits are worth stating. Recency is deliberately not a feature: the panel is a snapshot weighted towards recent ads, so the distance from a last start would measure the archive's coverage as much as the advertiser's activity. And copy concentration is bounded by the ad count, so an advertiser with one ad scores one by construction; that group is a real behaviour, the advertiser who bought once, but its concentration reading adds nothing to its volume.

## Emotion and churn

`data/emotion/` ships a seed, a VADER valence list (MIT), twenty labelled transcripts, the fitted artifact, a manifest and a card. `data/churn/` ships the source table, the panel, the retention series and the reason readings.

```bash
./xcm-mmm emotion fit
./xcm-mmm emotion score --text "This is the third time the unit has failed."
./xcm-mmm emotion coverage --examples 4
./xcm-mmm churn extract
./xcm-mmm churn fit
./xcm-mmm churn info
```

On the bank the reading is 0.850 held out against a 0.350 majority baseline. Coverage over `/usr/share/dict/words` is 100% resolved: 39.0% of the list lands above the neutral prior, 9.8% of it with a reading strong enough to matter, and the rest reads neutral because a dictionary holds far more nouns and technical terms than feelings. On the transcript corpus 51.9% of content tokens read above the neutral prior. The churn baseline is AUC 0.867 held out, against 0.801 for tenure and monthly charge alone. `emotion check` and `churn check` rebuild each folder and report every disagreement.

## Reading one text

`read` puts one piece of copy through every model the tool ships and prints each score beside the evidence for it. Nothing is blended: every number is the score of the model that owns it, and a dimension no model can see from one text is named as unobserved instead of guessed.

```bash
./xcm-mmm read "Guaranteed returns or the crash takes your savings. Only 3 seats left, and the insiders already know. Act now."
./xcm-mmm read --json --triggers 10 --file ad.txt
cat ad.txt | ./xcm-mmm read
```

```text
text      110 chars  18 words  3 sentences  6.0 words/sentence
          "Guaranteed returns or the crash takes your savings. Only 3 seats left, a..."
emotion   neutral      valence -0.01  arousal 0.20  intensity 0.31
  trust         0.31  ##########
  sadness       0.23  #######
  anticipation  0.10  ###
language  direct address 5.6%  punch 6.0 words/sentence  specificity 5.6 numbers/100w  emotional 44.4%
          distinct 17 of 18 words  type/token 0.94  reuse 0.00  focus 0.00
triggers  word             axis        tier      weight  valence
          guaranteed       trust       exact       1.00   +0.70
          crash            sadness     listed      0.42   -0.97
          left             sadness     exact       0.40   -1.00
detectors axis          score  level   evidence
          fear          0.000  none    gate closed: no cue from threat  fired: act now, now, your
          manipulation  0.320  weak    per 100 words: clock 5.56, guarantee 5.56  fired: act now, guaranteed returns
          narrative     0.000  none    gate closed: no cue from framing, withheld, verdict
semantics flesch 69.15  grade 5.11  syllables/word 1.56  long words 27.8%  second person 5.6%
          abstraction register      15.789
segment   copy fit against every behavioural centre (text reach 22%)
          *  2 creative reuse low, ad volume low  fit  36.1%  distance 0.15  panel share  45.8%
             3 copy concentration low, creative reuse low fit  35.8%  distance 0.23  panel share  14.7%
             1 start gap spread high, ad volume high fit  17.6%  distance 1.55  panel share  16.7%
             0 creative reuse high                fit  10.4%  distance 2.04  panel share  22.7%
          copy length          text 18 words text                   segment 67.4    -1.46 z  weight 0.09
          creative reuse       text 0 share phrase reuse           segment 0       +0.00 z  weight 0.91
          observed copy length, creative reuse
          unobserved ad volume, time between starts, start gap spread, banded spend, targeting terms, copy concentration
          the copy supplies 2 of 8 dimensions and those carry 22% of the separation between segments: the fit ranks creative style, it does not assign a segment, because volume, cadence and spend decide that
          nearest and second differ by 0.08 distance, against distances of 0.15 and 0.23
learned   an ad map over 1,355 rows is withheld: grouped folds recover 31% balanced and 45% overall, against 25% chance, 63% for the largest class and 25% on shuffled labels, so the largest class alone scores higher; it is not used to place this text
```

With `--ads brand.jsonl` the learned block scores the whole set instead and names the segment the map believes, which the section on the learned copy map explains.

The blocks answer separate questions. *Emotion* is the eight Plutchik axes with the words trimmed away, and the modifiers the scanner saw. *Language* reads the copy as marketing: direct address, punch per sentence, numeric specificity, the share of words that carry emotional mass, and the two shingle readings `reuse` and `focus`. *Triggers* are the words that carried the mass, the axis each leans on and the tier that answered for it, so a word read by analogy is visible as one. *Detectors* is the seven axes with their level and evidence: a fired cue list, or for a cue index the component densities per hundred words and whether the gate opened at all. *Semantics* is measured rather than modelled: sentence and syllable rates, Flesch ease and grade, second person share, questions and exclamations, plus the surface readings the learned axes used (abstraction register, nominalisations, agentless passives, citations, authority words).

*Segment* is a fit score, not a label. The text supplies the columns it can actually fill: copy length, and creative reuse read as the share of the copy sitting inside a four word phrase it uses more than once, which is the same habit the panel sees as an advertiser reusing a creative across ads. `--terms N` adds the buy. Each column is weighted by how far apart the segment centres sit on it, so reuse at a full standard deviation of separation decides the fit while copy length, which moves a tenth of one across the four segments, barely moves it. `fit` is a softmax over the weighted z distance and sums to 100 across the segments, `matches` and `clashes` name only the columns that carried a real share of the distance, and `reach` states how much of the model's separation the copy can see. At 22% the note says plainly that the ranking is creative style and that volume, cadence and spend decide the segment. That is the honest ceiling of one text: cross-ad concentration and media behaviour are not in it.

### Data that would sharpen the segments

The eight columns were chosen because a public ad panel publishes them. The gaps are known and closable with public data; these were checked against live sources and none is vendored in this repository:

- **Wesleyan Media Project, Meta 2022 election ads** (figshare 26124325, text at 26124295, CC BY 4.0): ad id, page id, delivery start and stop, spend and impression bands, platform split, plus full creative body, title, link text and video transcript. All eight columns, at hundreds of thousands of ads. The only source found that supplies per-ad text *and* time *and* money, which is what would let the learned map see the behaviour columns a 251 advertiser panel cannot separate.
- **Meta Ad Library API**: spend, impressions, delivery dates, targeting and creative body per ad, banded in the EU. Free, but the US spend field is often a single wide band and there is no click data.
- **TikTok Commercial Content Library**: similar per-ad fields with a free API, stronger on commercial advertisers and weaker on political ones.
- **Google Ads Transparency Center** (`bigquery-public-data.google_ads_transparency_center.creative_stats`, EEA and Turkey): advertiser id, creative id, format, first and last shown, impression bounds per region and per surface, plus demographic, geo, contextual and topic criteria. No creative text, so copy length and fingerprints need the detail page; strong on volume, cadence and targeting breadth for commercial advertisers.
- **Criteo Sponsored Search Conversion Log** (CC BY-NC-SA): 90 days of clicks and conversions per product and partner, with audience ids and category, which supplies volume, cadence, burstiness, targeting breadth and a conversion label, but no creative text and no media spend.
- **Outbrain Click Prediction** (Kaggle, ~36 GB): real advertiser_id, campaign_id, ad document ids, timestamps and clicks, plus a document topic taxonomy. The best public shape for native-advertiser behaviour; no copy text.
- **KDD Cup 2012 Track 2** (Tencent): advertiser_id, ad_id, keyword_id, query, position and click, at 150M impressions. Targeting and cadence are strong; there is no spend.

What none of them has is the join this crate would need to fit text to segments: one advertiser's creative text and its full behaviour history under the same stable id. The Wesleyan set is the closest, and it is the recommended next source if `copy_words` and `repeat` are to be predicted rather than proxied.

## Quick start

```bash
cargo build --release
./target/release/xcm-mmm list
./target/release/xcm-mmm de --problem sphere --dim 30
./target/release/xcm-mmm moead --problem zdt1
```

A forecast needs a CSV with a header row and a numeric target column. To try the whole path without data of your own:

```bash
./target/release/xcm-mmm synth --rows 800 --features 12 --informative 4 --out tmp/panel.csv
./target/release/xcm-mmm forecast tmp/panel.csv --method both --folds 4
```

That prints one line per model:

```text
rows        800   features 12   folds 4
model        feat          mse       r2      ic   skill
all            12     0.371781   0.6616   0.796   67.8%
de              4     0.335217   0.6949   0.820   71.0%
  keeps: x1, x2, x3, x4
moead-knee      2     0.524956   0.5222   0.718   54.6%
  keeps: x1, x3
moead-best      4     0.335217   0.6949   0.820   71.0%
  keeps: x1, x2, x3, x4
  front: 4 points, knee 2, best 0
```

From labelled rows to one trained model:

```bash
./target/release/xcm-mmm pairs mine tmp/rows.jsonl tmp/pairs.jsonl
./target/release/xcm-mmm transformer train --pairs tmp/pairs.jsonl --out tmp/model.json
./target/release/xcm-mmm transformer eval tmp/model.json --pairs tmp/pairs.jsonl
./target/release/xcm-mmm transformer score tmp/model.json --rows tmp/rows.jsonl --out tmp/scores.jsonl
```

Training fills a batch in parallel and can be paused: `--checkpoint tmp/run.cp.json` writes the optimizer state after each epoch group, and `transformer train --pairs tmp/pairs.jsonl --out tmp/model.json --resume tmp/run.cp.json` continues the same plan and lands on the model the uninterrupted run produces.

## Benchmarks

Both tables come from `xcm-mmm bench --runs 5` on one development machine, release build. Times are wall clock per run and only show scale; compare the objective values, not the clocks.

Differential evolution, SHADE with the L-SHADE reduction, 300 000 evaluations at 30 variables:

| problem | best | median | mean | seconds |
|---|---:|---:|---:|---:|
| sphere | 1.510e-13 | 1.629e-13 | 1.639e-13 | 0.21 |
| rosenbrock | 1.371e-13 | 1.502e-13 | 1.509e-13 | 0.20 |
| rastrigin | 5.684e-14 | 1.137e-13 | 1.023e-13 | 0.21 |
| ackley | 4.405e-13 | 5.649e-13 | 5.585e-13 | 0.22 |

MOEA/D with Tchebycheff scalarisation, 100 subproblems, 50 000 evaluations, five runs, partial updates at a 0.2 share weighted by relative improvement:

| problem | variables | IGD | hypervolume | points | seconds |
|---|---:|---:|---:|---:|---:|
| zdt1 | 30 | 0.00550 | 0.86646 | 100 | 0.119 |
| zdt2 | 30 | 0.00561 | 0.53220 | 98 | 0.119 |
| zdt3 | 30 | 0.01230 | 1.30270 | 84 | 0.122 |
| zdt4 | 10 | 0.00710 | 0.86345 | 100 | 0.109 |
| zdt6 | 10 | 0.00190 | 0.50436 | 100 | 0.105 |
| dtlz1 | 7 | 0.03186 | 0.18272 | 63 | 0.110 |
| dtlz2 | 12 | 0.07700 | 0.70066 | 60 | 0.112 |

Hypervolume uses the reference point 1.1 in every objective for the ZDT set, 0.6 for DTLZ1 and 1.1 for DTLZ2. IGD is measured against a 500 point sample of the true front, which the DTLZ rows serve with a 496 point lattice; the three objective rows are resolution limited, and the scaling law behind that is in [Benchmarks](README/DOCS/BENCHMARKS.md). Reproduce with `xcm-mmm bench --runs 5`.

## Documentation

- [Algorithms](README/DOCS/ALGORITHMS.md) - the operators, the success history formulas, decomposition, constraints and every default.
- [Usage](README/DOCS/USAGE.md) - the full command line, JSON reports and library examples.
- [Forecasting](README/DOCS/FORECASTING.md) - data formats, folds, metrics, how to read the front and where accuracy comes from.
- [Benchmarks](README/DOCS/BENCHMARKS.md) - methodology, the tables above and how they compare with published results.
- [Install](README/DOCS/INSTALL.md) - requirements, build, install and using the crate as a dependency.
- [Development](README/DOCS/DEVELOPMENT.md) - layout, design rules, tests, lints and how to add a problem.

## Testing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

The suite has 395 tests. It covers convergence on sphere, Rosenbrock, Rastrigin, Ackley, ZDT1 to ZDT4, ZDT6, DTLZ1 and DTLZ2, the exact hypervolume identities, the weight lattice counts, constraint handling, reproducibility at one and four threads, the forecasting path on a synthetic panel, the detector fit from bank to artifact including its refusals, the segmentation path from panel to artifact including its refusals, the reader over the shipped models including its stdin and file inputs, the learned copy map from fit to gate to offline rebuild, and every CLI subcommand.

## License

MIT License, copyright 2026 Crissy Moon. See [LICENSE](LICENSE) for the full text.

Crissy Moon - [https://xcaliburmoon.net/](https://xcaliburmoon.net/)

Repository: [https://github.com/crissymoon/xcm-mmm](https://github.com/crissymoon/xcm-mmm)
