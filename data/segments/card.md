# marketlab-behaviour-segments

Behavioural segments of advertisers, fitted in Rust. Built 2026-11-14T20:18:09Z from the entities listed below, and checked by rebuilding them.

A segment describes what an advertiser did in the panel, not what it is: ads started, the gaps between those starts, the spread of those gaps, the published spend band, copy length, terms bought against, creative reuse and copy concentration. Nothing descriptive enters a feature, and page names are carried for reading the result, never for fitting it.

## Source

| file | entities | ads | digest |
|---|---:|---:|---|
| features.jsonl | 251 | 1355 | `ec4bfd3dc8419f9a` |

Start dates run from 2022-10-25 to 2026-11-10. 493 of the ads arrived on the archive's test side and are kept in the same entities, because a segment of an advertiser is a description rather than a prediction and splitting an advertiser in half would describe nobody.

## Segments

251 entities, 4 groups, chosen from k = 3 to 6 by silhouette among the partitions that keep every group above the declared 5.0 percent floor. The chosen partition's smallest group holds 14.7 percent. `z` is the group's mean against the whole panel in standard deviations; the reading beside it is the same column in its own units.

| segment | label | entities | share | strongest readings |
|---:|---|---:|---:|---|
| 0 | creative reuse high | 57 | 0.227 | repeat +1.36 (0.95 share), burstiness -0.36 (0.04 coefficient of variation), concentrate +0.32 (0.90 share) |
| 1 | start gap spread high, ad volume high | 42 | 0.167 | burstiness +1.81 (1.04 coefficient of variation), ads +1.49 (12.54 ads), cadence +1.22 (10.22 days) |
| 2 | creative reuse low, ad volume low | 115 | 0.458 | repeat -0.78 (0.00 share), ads -0.74 (1.00 ads), concentrate +0.68 (1.00 share) |
| 3 | copy concentration low, creative reuse low | 37 | 0.147 | concentrate -1.46 (0.40 share), repeat -0.61 (0.08 share), cadence +0.38 (2.69 days) |

## Features

| feature | family | unit | transform | description |
|---|---|---|---|---|
| `ads` | volume | ads | ln(1 + x) | Ads the advertiser started inside the panel. |
| `cadence` | cadence | days | ln(1 + x) | Median gap between the days it started an ad. |
| `burstiness` | cadence | coefficient of variation | as it stands | Population standard deviation of the gaps over their mean, capped at 3. |
| `spend_mid` | value | usd per ad | ln(1 + x) | Mean midpoint of the spend band the archive published per ad. |
| `copy_words` | creative | words | ln(1 + x) | Median words per ad. |
| `terms` | targeting | terms per ad | as it stands | Mean number of terms the archive lists against an ad. |
| `repeat` | creative | share | as it stands | Share of its ads whose copy fingerprint appears more than once for it. |
| `concentrate` | creative | share | as it stands | Largest share of its ads that carry one fingerprint. |

Each column is standardised and clipped at 3.0 standard deviations before the fit. The mean and the standard deviation travel in the model, so a z can be recomputed by hand from the readings file.

## Selection

| k | silhouette | smallest group | inertia | sizes |
|---:|---:|---:|---:|---|
| 3 | 0.275 | 0.227 | 1042.0 | 137, 57, 57 |
| 4 **chosen** | 0.286 | 0.147 | 923.5 | 57, 42, 115, 37 |
| 5 | 0.223 | 0.116 | 818.7 | 29, 76, 58, 30, 58 |
| 6 | 0.234 | 0.116 | 745.6 | 54, 29, 43, 43, 52, 30 |

## Audit

There are no labels, so nothing below is an accuracy. What is measured is whether the partition stands up to its own protocol: the same fit on columns shuffled independently, and the same fit on resamples of the entities scored against the full panel by adjusted Rand index.

| question | reading | gate |
|---|---:|---|
| silhouette | 0.286 | at least 0.15 |
| shuffled control, mean | 0.168 | margin 0.08 |
| shuffled control, best of 20 | 0.185 | reported, not gated |
| resampling, mean agreement | 0.759 | at least 0.60 |
| resampling, tenth percentile | 0.552 | reported, not gated |
| resampling, worst of 50 | 0.537 | reported, not gated |

A silhouette near 0.29 is ordinary for behavioural data whose columns share a scale factor. It is a statement that the groups are separated, not that they are the only reading of the panel.

## Checks

- silhouette 0.286 beats the shuffled control at 0.168
- resampling keeps the partition at 0.759 mean agreement
- every group holds at least 14.7 percent of the entities
- every feature varies across the panel

## Reproduce

```bash
./xcm-mmm segment fit --features data/segments/features.jsonl --out data/segments
./xcm-mmm segment check --dir data/segments
```

## Read it

```bash
./xcm-mmm segment info
./xcm-mmm segment assign --entities data/segments/features.jsonl
```

## Limits

- The panel is a snapshot of what the archive published, weighted towards recent ads. Cadence is read inside the snapshot, so an advertiser buying steadily for years is seen only through the ads the archive kept.
- Spend and impressions are bands. Their midpoint is used, which is a coarse reading of value and is not a measurement of money spent.
- Reuse is read from the archive's copy fingerprint. Two ads that differ by a word count as two, so reuse understates what a reader would call the same creative.
- Copy concentration is bounded by the number of ads: an advertiser with one ad scores one by construction. That group is a real behaviour, the advertiser who bought once, but its concentration reading carries no information beyond its volume.
- Recency is deliberately not a feature. The panel is a snapshot weighted towards recent ads, so the distance from a last start to the snapshot day would measure the archive's coverage as much as the advertiser's activity.
- A group is a description of behaviour, not a target. Nothing here says that moving an advertiser between groups would change an outcome.

