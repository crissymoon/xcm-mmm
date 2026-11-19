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
- **Legal words and plain words.** A 171 row glossary of legal English and the rule table built from it, including six authored clause rewrites that reorder a construction rather than replace one word, a reader that names every term in a document with its offsets and bands its density, a translator that rewrites the avoidable terms, rewrites the constructions, and glosses the terms of art it must keep, with the page each meaning was checked against carried in the row, and a brief that sorts a document by what it requires, forbids and allows and defines the terms it reads.
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

`bank.jsonl` holds 4,672 labelled sentences: legal register from Federal Register abstracts and abstraction from grants.gov synopses, each set against advertisers' own copy as the negative side, plus the legal language bank's 1,825 statute sentences and plain sentences about the law. That last part arrives through an extension bank: the exporter folds it in under the same labels and prefixes its source with the bank's folder name, and the card names what joined. `ads.jsonl` holds 518 ads with their source split, so the index references are fitted on training ads and nothing else. `cues.json` holds the declared cue table and the feature list, exported from the Python module that produced the published readings rather than retyped, so a pattern cannot drift between the two.

The learned axes are fitted in two steps. Smoothed log odds (Monroe, Colaresi and Quinn) give every term a z from the two banks behind three one-sided filters: occurrences only in the positive bank, a frequency ceiling on both sides, and a document frequency ceiling in the negative bank that switches off when the negative bank is too small to have a granularity. A logistic regression then reads that density beside the surface readings: citations and authority verbs for legal, nominalisations, agentless passives and syllables for jargon. Both axes are scored out of fold, with folds assigned by the digest of the text so a repeated sentence cannot straddle a fold, and the same loop runs again on shuffled labels. Legal scores 0.836 balanced against a 0.509 majority and 0.493 on its control; jargon 0.779 against 0.780 and 0.502. The legal bank is the harder bank, and it is the one that reads a statute: the same axis fitted on the ad corpus alone scored 0.670 balanced on 1,825 statute sentences it had never seen, with 0.556 recall.

The other five have no labelled corpus, so nothing here is an accuracy. They ship as declared cue sets behind a gate, where a construct that names no gated component scores zero, and the audit reports whether two disjoint halves of the cues pick out the same ads. Fear's halves agree on 47 percent of the corpus and correlate at -0.08, which is the published finding rather than a defect: harm and urgency travel apart in this corpus. Publicity and manipulation never fire on these 518 ads, which is a fact about Los Angeles political and issue advertising, not a claim that the constructs are absent from the form.

```bash
./xcm-mmm detect train --bank data/detectors/bank.jsonl --ads data/detectors/ads.jsonl \
    --cues data/detectors/cues.json --out data/detectors
./xcm-mmm detect score --text "guaranteed returns, before it is too late"
./xcm-mmm detect info
./xcm-mmm detect check --dir data/detectors
```

`detect check` rebuilds the folder from its own inputs, compares the result with the shipped fit and runs every declared example in the cue table, so a model edited after its manifest was stamped is a fault rather than a note. `scripts/detect-data.sh` runs the whole path: export the three inputs, fit, check.

## Legal words and plain words

`data/legal/` holds an instrument for two jobs: naming the words that make a sentence legal English, and writing it again in plain words. Both run from data rather than from a fitted model.

- `terms.jsonl` is the glossary, 171 rows. Each row carries the plain reading, the kind (modal, deictic, doublet, latin, doctrine, procedure), whether the term is avoidable, the authority behind it where one exists, and the page its meaning was checked against.
- `rewrites.jsonl` holds 110 rules: 104 derived from the glossary, so the two cannot disagree, and 6 clause rules written in the corpus builder. 108 edit and 2 only advise, because `said` means "the" in `said vehicle` and "spoke" in `he said`. A replacement may name its own capture groups, which is what lets a clause rule reorder what it matched: `shall not be entitled to payment or reimbursement for` becomes `shall not be paid for`, where a word rule had produced `must not be entitled to`.
- `bank.jsonl` is the labelled side: 1,402 sentences from statutes and rules fetched from their publisher, and 922 plain sentences about legal subjects from court self-help pages and advertisers' copy. The self-help sentences are the point of the file, because a detector fitted only against ads learns that courts are legal English rather than what legal English is.
- `pairs.jsonl` is 71 legal to plain pairs, 43 published (a page prints both sides) and 28 written here from authority sentences and tagged editorial. Twelve of the editorial pairs quote a contract clause or a code of ethics, each with the document's URL on the pair and in `sources.json`, because the published pairs are thin on contract and securities wording.
- `settings.json` holds the citation patterns a rewrite must not touch, the density bands, the notice every brief carries, and the clause cues `legal explain` sorts a sentence by: duty, prohibition, permission, condition, deadline, negation, statement and abbreviation. On this bank every one of the 41 self-help documents lands in `plain`, at 0.48 avoidable terms per hundred words at the top, while the 46 statutory documents spread across light, heavy and dense.

The distinction the reading is built on is avoidable against necessary. `pursuant to` becomes `under`; `res judicata` keeps its name and gets a gloss on first use, because replacing it would change what the document claims. A term inside a citation or a quotation is reported and never edited.

`legal explain` is the third reading. It sorts every sentence by the verb that carries it, so a duty, a prohibition, a permission and a condition are reported apart from each other and from statements that do neither. Each clause carries the actor, the condition that limits it and the deadline that dates it; each term of art the text relies on gets its plain reading, the note that says why it is kept, and its authority. A sentence that holds two rules, `..., and the Licensee shall be liable`, is cut in two rather than filed under one verb. The cue words come from `settings.json` and the definitions from `terms.jsonl`, so a brief cannot disagree with a scan of the same text. Every brief ends with the notice `settings.json` carries, so the text output, the JSON and the dataset cards on Kaggle state the same thing: this is a research project, a brief is a reading of the words in the document, and it is not legal advice.

```bash
./xcm-mmm legal terms --kind latin
./xcm-mmm legal scan --file notice.txt
./xcm-mmm legal translate --explain --file notice.txt
./xcm-mmm legal explain --file notice.txt
./xcm-mmm legal explain --file notice.txt --check
./xcm-mmm legal check --verbose
./xcm-mmm legal check --fix
```

`legal explain --check` is the coverage question a pipeline asks. A glossary answer has to start somewhere, so a name the document defines and the glossary has no row for is reported under `UNGLOSSED TERMS`, counted in a warning and, with `--check`, turned into a non-zero exit. The definitional cues it reads are the document's own: `(the "Service Fee")` and `Service Fee means`. A name a contract defines for its own cast, `the Buyer`, `the Agreement`, `the Term`, is not a term of art, so `clauses.defined_ignore` in `settings.json` lists those and the check stays quiet about them; that list is data, so a corpus with a different cast can change it without a build.

`legal check` recomputes the folder's digests and its coverage, then runs the translator over the whole bank. On the legal side 1.41 avoidable terms per hundred words fall to 0.27, of which 95 are the two advisory words and 4 sit inside citations; 310 of the 1402 legal rows change at all, and 10 of the 922 plain rows. The corpus is built by `tools/build-marketlab-legallanguage.py` in the marketlab repository, which writes both this folder and `data/marketlab-legal-language/` from the same bytes.

## Behavioural segments

`data/segments/` is the third shipped set: advertisers grouped by what they did in an ad panel, fitted in Rust. Nothing descriptive enters a feature. The eight readings are ad volume, the median gap between the days it started an ad, the spread of those gaps, the published spend band midpoint, copy length, terms bought against, creative reuse and copy concentration.

`features.jsonl` holds 251 advertisers and the 1,355 ads behind them, folded from the archive panel by `segment extract`. The fit standardises each column, clips at three standard deviations, and searches k from 3 to 6 in three covariance families, because a two way split of an advertising panel is a high and low score rather than a segmentation, and because the shape of a group is a choice the folder used to make without saying so.

A k-means fit is the maximum likelihood fit of a mixture whose groups are spheres of one shared width: equally spread on every column, no two columns co-varying. This panel is not that. The mean absolute correlation between its columns is 0.21, and ad volume and gap spread run together at 0.74, so a sphere cannot use most of the signal. Three beliefs are therefore fitted and compared: one width per group (the k-means shape), one width per group per column, and a covariance per group. Each is scored on rows it never saw, and the comparison is paired by fold: a larger k or a freer family had to read 0.020 nats per entity better than its own standard error before it was preferred, so fold noise cannot buy parameters. A partition also has to keep its placement: a fold model has to put its held-out rows where the shipped model puts them, at 0.60 agreement or better.

| segment | advertisers | share | behaviour |
|---|---:|---:|---|
| copy concentration low, creative reuse low | 18 | 7.2% | 2.3 ads, none repeating a fingerprint, 0.44 concentration, 85 word copy |
| start gap spread high, time between starts high | 65 | 25.9% | 8.4 ads, 9.9 days between starts, gap spread 0.75 |
| creative reuse high, start gap spread low | 53 | 21.1% | 3.6 ads, 0.96 of them repeating one fingerprint, 0.88 concentration |
| creative reuse low, ad volume low | 115 | 45.8% | one ad each, bought once |

Four groups, read through a full covariance. On rows it never saw the fit reads 0.609 nats per entity against -9.777 for the k-means partition the folder used to ship, a gain of 10.4 with a fold spread of 2.4; that spread is wide and it is printed rather than smoothed. A fold model places its held-out rows at 0.813 agreement with the shipped partition. The Euclidean silhouette is 0.222 against 0.168 for the same k-means protocol on columns shuffled independently and 0.286 for k-means itself, so it clears the control but not the incumbent: a partition read through a covariance is not a sphere, and silhouette charges it for the shape that lets it predict better. Refitting on fifty resamples of the advertisers agrees with the full panel at 0.712 mean adjusted Rand index, tenth percentile 0.451. A fit that fails any of those floors is refused with the reason it failed, not written.

Every candidate keeps its verdict in the card, so the table accounts for all twelve rows and not only for the winner, and BIC is printed beside the held-out reading. BIC is not the rule here: at 251 entities the freest candidate holds 224 parameters, and its in-sample BIC beats the whole grid while its held-out reading is worse than the fit that ships.

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
| ad | one ad's copy | 1,355 | 56.2% | 33.7% | 25.0% | 24.9% | no |
| brand | one advertiser's copy set | 251 | 62.2% | 46.0% | 25.0% | 25.4% | yes |

Both readings come from five folds that hold out whole page ids, so no ad of a scored advertiser was seen in training. The ad unit is withheld because ads are weighted by how many each advertiser bought: 70.8% of the 1,355 ads belong to the 26% of advertisers in one segment, so naming that segment every time scores 70.8% and the map's 56.2% is below it. Its 33.7% balanced is nine points over chance, real but too thin to act on. The brand unit clears the gate at 46.0% balanced and 62.2% overall, which is what a copy set adds: cross-ad reuse, vocabulary breadth and the size of the set, none of which one ad shows. Its lift is 28 percent of the way from chance to perfect.

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
./xcm-mmm churn publish
./xcm-mmm churn versions
```

On the bank the reading is 0.850 held out against a 0.350 majority baseline. Coverage over `/usr/share/dict/words` is 100% resolved: 39.0% of the list lands above the neutral prior, 9.8% of it with a reading strong enough to matter, and the rest reads neutral because a dictionary holds far more nouns and technical terms than feelings. On the transcript corpus 51.9% of content tokens read above the neutral prior. The churn baseline is AUC 0.867 held out, against 0.801 for tenure and monthly charge alone, and the card beside that number carries what the interval is worth: at 251 rows with 50 positives the bootstrap covers 0.938, BCa 0.936 and DeLong 0.947 at nominal 0.95, each with its own Wilson interval so a reader can see how much of the reading is simulation noise. An earlier run of that simulation used 200 trials and reported DeLong at 0.925, which read as a failure of the analytic interval; the gap was the trial count, and the card says so. It also carries the window contract, time zero at the join, 1,869 events and 356 rows censored at month 72, and the class conditional conformal split with per-class coverage intervals that refuse an `alpha` its calibration rows cannot support. The support reading is a six point curve drawn on a split of its own, salted by customer id: accuracy 0.793 at 528 training rows and 0.790 at 5,282, against an estimated label ceiling of 0.879, so the remaining gap is the features or the operator rather than the row count, and the card prints the inverse power fit only to say it does not rise through these points. `churn publish` writes a version carrying the SHA-256 of the artifact, the SHA-256 of the panel, the window range and the readings, and `churn versions` prints every version with a verification column, so an edited copy reads as a digest mismatch rather than as current. `emotion check` and `churn check` rebuild each folder and report every disagreement.

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
./target/release/xcm-mmm forecast tmp/panel.csv --method both --folds 4 --cpcv
```

The first run prints one line per model. The second adds the audit:

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

With `--cpcv` the run adds the split audit underneath:

```text
cpcv        6 groups, 2 tested per split, purge 1, 15 splits, 5 paths
cand  feat    train mse     test mse  chosen
0        5     0.315765     0.330183    0/15
1        7     0.313527     0.331671   15/15
4        4     0.317641     0.329029    0/15
  path mse 0.331671 +/- 0.002731 across 5 paths
  pbo 0.67   mean rank logit -0.17
  verdict: the training winner lands in the better half on 33% of splits, so selection is no better than a coin at this candidate count
```

The audit is the point. Candidate 1 wins every training split and is beaten on the test rows by a smaller candidate, the path spread is 0.0027 in MSE, and the probability of backtest overfitting is 0.67. A run whose verdict reads like that one should not be quoted as a discovery, and the deflated gate that prices the search says the same thing from the trial count.

From labelled rows to one trained model:

```bash
./target/release/xcm-mmm pairs mine tmp/rows.jsonl tmp/pairs.jsonl
./target/release/xcm-mmm transformer train --pairs tmp/pairs.jsonl --out tmp/model.json
./target/release/xcm-mmm transformer eval tmp/model.json --pairs tmp/pairs.jsonl
./target/release/xcm-mmm transformer score tmp/model.json --rows tmp/rows.jsonl --out tmp/scores.jsonl
```

Training fills a batch in parallel and can be paused: `--checkpoint tmp/run.cp.json` writes the optimizer state after each epoch group, and `transformer train --pairs tmp/pairs.jsonl --out tmp/model.json --resume tmp/run.cp.json` continues the same plan and lands on the model the uninterrupted run produces.

### In a Kaggle notebook

Kaggle images hold no Rust toolchain and, by default, no network, so the tool arrives as two datasets: the static binary (`crissymoon/xcm-mmm-static-linux-x86-64`) and the fitted artefacts (`crissymoon/xcm-mmm-models`). Both mounts are read only and neither keeps the executable bit, so a notebook copies them out. Because every command defaults to a path under `data/`, the copy puts the artefacts beside the binary and changes directory, after which `detect score`, `read`, `segment assign`, `legal scan`, `legal explain` and `churn check` run with no path flags.

```bash
scripts/package-kaggle.sh --push    # binary dataset
scripts/package-models.sh --push    # artefacts dataset, staged as data/<model>/
python3 scripts/kaggle-notebook.py --push   # the notebook that uses both
```

The kernel ([`crissymoon/xcm-mmm-binary-in-a-notebook`](https://www.kaggle.com/code/crissymoon/xcm-mmm-binary-in-a-notebook)) is public. It runs the binary with `enable_internet: false`, asserts each of the four claims above, and draws four figures from numbers the same run printed, saved under `charts/` and displayed inline: the nested fold skill beside selection stability, four shard forecasts as overlapping bars, the seven detector axes with the top-score distribution by split, and the emotion tiers with the neutral prior as its own bar. A research section states which source forces each methodological choice, a provenance section compares the binary's own report with the record shipped beside it and writes `provenance.json` into the output, and the references at the end carry the links.

`scripts/build-linux.sh` is the part that is not obvious: cross compiling from macOS fails until `rust-lld` from the musl target is named as the linker, because cargo otherwise hands the link to Apple's `cc`, which rejects every GNU flag. The result is a static-pie ELF that does not track the image's glibc.

The build is reproducible, and the binary states its own origin: `xcm-mmm provenance` prints the commit, target, `rustc` and build date that `build.rs` compiled in, `scripts/build-linux.sh` writes the same facts with the sha256 to `provenance.json`, and the notebook recomputes the hash and fails if the two disagree. The reasoning behind the evaluation, the settings that make a rebuild byte identical and the Kaggle limits the notebook is written around are in [Research and provenance](README/DOCS/RESEARCH.md).

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
- [Research and provenance](README/DOCS/RESEARCH.md) - the literature behind the evaluation, the reproducible build, and how a reader checks which binary produced a table.

## Testing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

The suite has 440 tests. It covers convergence on sphere, Rosenbrock, Rastrigin, Ackley, ZDT1 to ZDT4, ZDT6, DTLZ1 and DTLZ2, the exact hypervolume identities, the weight lattice counts, constraint handling, reproducibility at one and four threads, the forecasting path on a synthetic panel, the detector fit from bank to artifact including its refusals, the segmentation path from panel to artifact including its refusals, the reader over the shipped models including its stdin and file inputs, the learned copy map from fit to gate to offline rebuild, and every CLI subcommand.

## License

MIT License, copyright 2026 Crissy Moon. See [LICENSE](LICENSE) for the full text.

Crissy Moon - [https://xcaliburmoon.net/](https://xcaliburmoon.net/)

Repository: [https://github.com/crissymoon/xcm-mmm](https://github.com/crissymoon/xcm-mmm)
