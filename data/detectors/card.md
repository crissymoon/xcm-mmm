# marketlab-ad-detectors

Seven text detectors over advertising copy, fitted in Rust. Five are declared cue indices and two are lexicons learned from a labelled bank. Built 2026-11-14T19:53:00Z from the files listed below, and checked by rebuilding them.

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

7 axes. `model.json` is 31876 bytes and its digest is 2869e24c26772ddf.
