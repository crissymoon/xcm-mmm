# marketlab-ad-detectors

Seven text detectors over advertising copy, fitted in Rust. Five are declared cue indices and two are lexicons learned from a labelled bank. Built 2026-11-18T02:20:14Z from the files listed below, and checked by rebuilding them.

## Source

| file | rows | digest |
|---|---:|---|
| bank.jsonl | 2847 | `207b1e9abd22aa38` |
| ads.jsonl | 518 | `67516eae52831c9b` |
| cues.json | 5 axes | `8818d762970f5909` |

The bank is legal register from Federal Register abstracts and abstraction from grants.gov synopses, each set against advertisers' own copy as the negative side. The ads file is the same copy at document grain with its source split kept, so the index references are fitted on training ads and nothing else.

## Fitted axes

Held out numbers come from 5 folds assigned by the digest of the text, with the cut chosen inside each training part. `balanced` is the mean of recall on the positives and specificity on the negatives; `majority` is the score of answering no to everything; `shuffled` is the same fit on permuted labels.

| axis | terms | rows | recall | precision | balanced | majority | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| jargon | 120 | 2246 | 0.638 | 0.681 | 0.766 | 0.738 | 0.496 |
| legal | 120 | 2259 | 0.797 | 0.711 | 0.840 | 0.734 | 0.487 |

## Declared axes

No labelled corpus exists for these constructs, so nothing here is an accuracy. `prevalence` is the share of ads that fire, `halves_agree` is the share where two disjoint cue halves agree, `both` is the share where both halves fire, and `r` is their correlation over the ads where at least one did.

| axis | tag | prevalence | halves agree | both | r | checks |
|---|---|---:|---:|---:|---:|---:|
| fear | `#IntentionalFear` | 0.106 | 0.47 | 32 | -0.084 | 0/0 |
| followers | `#FollowerCount` | 0.006 | 0.82 | 4 | null | 3/3 |
| manipulation | `#MarketManipulation` | 0.000 | 1.00 | 0 | null | 6/6 |
| narrative | `#MediaNarrative` | 0.008 | 0.99 | 0 | null | 4/4 |
| publicity | `#PublicityPlay` | 0.000 | 0.97 | 0 | null | 3/3 |

## What the bank's labels are worth

| axis | rows | estimated label noise | ceiling on accuracy | smallest resolvable gain | resolves the floor |
|---|---:|---:|---:|---:|:---:|
| jargon | 2246 | 0.100 | 0.900 | 0.035 | no |
| legal | 2259 | 0.049 | 0.951 | 0.031 | no |

- **jargon**: 145 of 2,246 rows are confidently labelled against the model and 768 cleared neither threshold, so 1,478 stay in the joint. Scaled back up to the labelled totals that is 225 errors, 10.0 percent of the bank, 28.7 percent of the positive labels and 3.4 percent of the negative ones. A model that reproduced the true labels would score 0.900 against these ones, so a claim above that is a claim about the labels rather than the construct. The estimate assumes the rows left out carry the same error rate as the ones kept; 34.2 percent of the bank is left out, so that assumption is doing real work and the reading is an extrapolation rather than a count. At this shape the smallest difference that clears a 95 percent test at 80 percent power is 0.035, which is wider than the pre-registered floor of 0.030, so a gain at the floor would not be distinguishable from noise here. Resolving it would take about 3,129 rows at this class balance, against the 2,246 in hand.
- **legal**: 60 of 2,259 rows are confidently labelled against the model and 773 cleared neither threshold, so 1,486 stay in the joint. Scaled back up to the labelled totals that is 111 errors, 4.9 percent of the bank, 14.5 percent of the positive labels and 1.4 percent of the negative ones. A model that reproduced the true labels would score 0.951 against these ones, so a claim above that is a claim about the labels rather than the construct. The estimate assumes the rows left out carry the same error rate as the ones kept; 34.2 percent of the bank is left out, so that assumption is doing real work and the reading is an extrapolation rather than a count. At this shape the smallest difference that clears a 95 percent test at 80 percent power is 0.031, which is wider than the pre-registered floor of 0.030, so a gain at the floor would not be distinguishable from noise here. Resolving it would take about 2,348 rows at this class balance, against the 2,259 in hand.

The noise estimate is confident learning (Northcutt, Jiang and Chuang, 2021) on the same out-of-fold probabilities the held-out numbers came from, with a per-class threshold at the mean predicted probability of that class and the class totals scaled back up to the labelled counts. It measures the total labelling error, not its cause: a bank where an annotator misread the construct reads like one where labels were flipped at random. The ceiling is the accuracy a model that reproduced the true labels would score here, so a held-out number at or above it is measuring the labels rather than the construct. The resolution reading is the smallest balanced-accuracy gain this bank can see at 80 percent power, compared with the three point floor fixed before the fit; a bank that does not resolve the floor cannot produce evidence of a small improvement, only a large one.

## Reproduce

```bash
scripts/export-detect-data.py          # rewrite the three inputs
xcm-mmm detect train --bank data/detectors/bank.jsonl \
  --ads data/detectors/ads.jsonl --cues data/detectors/cues.json \
  --out data/detectors
xcm-mmm detect check --dir data/detectors
```

## Limits

- The legal and jargon banks are instruments and grant synopses set against ad copy. A high held out number says the two registers are separable, which is a fact about administrative English, not proof that a detector reads intent in an advertisement.
- The five declared axes have no labels. The audit measures whether a cue set agrees with itself; it cannot measure whether it is right, and a prevalence near zero on this corpus is a fact about Los Angeles political and issue advertising rather than about the construct.
- Fear ships its declared seed cues only. The Python artifact adds a threat lexicon learned from advertiser copy to the same component, so a fear score here runs lower than the one published beside the corpus.
- Levels are bands over the fitting corpus, so `strong` means strong relative to this bank. Comparing a reading across corpora is comparing two different rulers.
- The build time is a fact about the run, not the fit. The lexicon, the cuts and the references rebuild byte for byte; `built_at` does not.

7 axes. `model.json` is 34168 bytes and its digest is 0c10ba01b5121148.
