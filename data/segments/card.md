# marketlab-behaviour-segments

Behavioural segments of advertisers, fitted in Rust. Built 2026-11-17T20:55:14Z from the entities listed below, and checked by rebuilding them.

A segment describes what an advertiser did in the panel, not what it is: ads started, the gaps between those starts, the spread of those gaps, the published spend band, copy length, terms bought against, creative reuse and copy concentration. Nothing descriptive enters a feature, and page names are carried for reading the result, never for fitting it.

## Source

| file | entities | ads | digest |
|---|---:|---:|---|
| features.jsonl | 251 | 1355 | `ec4bfd3dc8419f9a` |

Start dates run from 2022-10-25 to 2026-11-10. 493 of the ads arrived on the archive's test side and are kept in the same entities, because a segment of an advertiser is a description rather than a prediction and splitting an advertiser in half would describe nobody.

## Segments

251 entities, 4 groups, each group allowed a covariance per group. The search ran k = 3 to 6 in three families, and every group holds at least the declared 5.0 percent of the panel; the smallest here holds 7.2 percent. `z` is the group's mean against the whole panel in standard deviations; the reading beside it is the same column in its own units.

| segment | label | entities | share | strongest readings |
|---:|---|---:|---:|---|
| 0 | copy concentration low, creative reuse low | 18 | 0.072 | concentrate -1.29 (0.44 share), repeat -0.78 (0.00 share), cadence -0.48 (0.12 days) |
| 1 | start gap spread high, time between starts high | 65 | 0.259 | burstiness +1.20 (0.75 coefficient of variation), cadence +1.18 (9.93 days), ads +1.07 (8.36 ads) |
| 2 | creative reuse high, start gap spread low | 53 | 0.211 | repeat +1.38 (0.96 share), burstiness -0.46 (0.00 coefficient of variation), spend_mid +0.30 (265.44 usd per ad) |
| 3 | creative reuse low, ad volume low | 115 | 0.458 | repeat -0.78 (0.00 share), ads -0.74 (1.00 ads), concentrate +0.68 (1.00 share) |

## The two shares

Two readings are shares that pile on an edge: `repeat` at 0.00 holds 60.2 percent of the panel and `concentrate` at 1.00 holds 64.1 percent of the panel. A Gaussian can only imitate a pile with a group of almost no width, and then it reports a density where the data are a probability. Each is read here as a gate: the chance a member of a group sits on the edge, and the members inside it, whose mean moves with the other readings. `inside` is that mean in plain units, and the last column is the share of each group on the edge, in segment order. A group with nobody inside the edge repeats the edge value there rather than reporting a mean over no rows.

| reading | edge | panel on the edge | inside | groups on the edge |
|---|---:|---:|---:|---|
| repeat | 0.00 | 60.2% | 0.000, 0.771, 0.960, 0.000 | 100.0%, 27.7%, 0.0%, 100.0% |
| concentrate | 1.00 | 64.1% | 0.444, 0.436, 0.590, 1.000 | 0.0%, 12.3%, 71.7%, 100.0% |

Both readings are scored as the probability of the observed value on the lattice a share actually lives on: the possible values are multiples of one over the advertiser's ad count, so a cell is integrated rather than approximated by density times spacing. Multiplying a density by that spacing is how a continuous model appears to win a lattice it cannot represent, and it reports mass above one.

| reading | gate | Gaussian | Brier gate | Brier Gaussian | edge called right | panel rate |
|---|---:|---:|---:|---:|---:|---:|
| repeat | -0.891 | -1.000 | 0.0978 | 0.1164 | 84.1% | 60.1% |
| concentrate | -0.916 | -1.055 | 0.1070 | 0.1290 | 81.2% | 64.1% |

Held out on 5 folds, the gates give the observed values 0.248 nats more probability per entity than the Gaussian conditional inside the same groups, and calls the edge right on 82.7 percent of rows against 62.1 percent for the panel rate. The gate is refit on the training rows of each fold; the partition is the shipped one, so the level is optimistic and the paired difference is the reading. The gate does not move a group's boundary: a likelihood with an atom and one without are densities against different measures, so the partition is still chosen on the Gaussian likelihood and the gate is fitted to it afterwards.

The clip at 3 standard deviations still applies to ads, cadence, burstiness, spend_mid, copy_words, terms. It never fired on either share, so the pile was never a clipping problem: it is the Gaussian's need to spend a group on it. What the clip exemption guards is a share whose edge sits beyond the bound, which would fold the edge into the interior and leave the gate without a boundary value.

## Choice

full covariance at k = 4: 0.543 nats per entity on rows it did not see, spread 2.344 across folds, against -9.777 for the k-means partition this folder used to ship, a gain of 10.320. A fold model places its own held-out rows at 0.813 agreement with it, and it costs 179 parameters.

A k-means fit is the maximum likelihood fit of a mixture whose groups are spheres of one shared width. That is a claim that the eight readings are equally spread inside a group and that none of them co-vary. This panel says otherwise: the mean absolute correlation between columns is 0.21, and `ads` and `burstiness` at +0.74 run together. A sphere cannot use that, so three families were fitted and compared: one width per group, one width per group per column, and a covariance per group. Held-out likelihood chooses between families, because a comparison of predictions can only be settled by rows the fit never saw. Placement agreement is the gate that says the groups a fold model predicts are the groups that ship. And the comparison is paired by fold, because every candidate is read on the same rows: a larger k or a richer family had to add 0.020 nats per entity more than its own standard error before it was preferred, so fold noise cannot buy parameters.

BIC is reported because it is the reading a mixture is usually chosen on, and it is not the rule here, because it is read in sample: the lowest BIC on the grid is full at k = 5 with 224 parameters against 251 entities, and its held-out reading is -2.623 nats per entity against 0.543 for the fit that won. A criterion that lets a model hold nearly one parameter per entity is a reading to print, not a rule to follow.

ICL is printed beside BIC because BIC charges a model for the parameters it spends and nothing for the rows it cannot place: two overlapping components can raise the likelihood while every row sits near their boundary. ICL adds that ambiguity as `BIC - 2 * sum log p`, where `p` is the posterior the winning component gives a row, so it can only be BIC or worse. On this grid it does not change the answer. The lowest ICL is full at k = 5, reading -996 against -1000 on the same fit's BIC, and it is still the near-saturated candidate whose held-out reading is -2.623 against 0.543 for the fit that won. The charge is small here because a free covariance buys sharp posteriors: the chosen fit gives each of its own rows a mean winning posterior of 0.9994, so the penalty adds 0.3 across 251 entities, which is 0.001 per entity. A criterion that is BIC plus a term that nearly vanishes is, on this panel, BIC.

| family | k | held out, nats per entity | spread over folds | placement agreement | BIC | ICL | parameters | smallest group | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| spherical | 3 | -9.884 | 0.825 | 0.975 | 5020 | 5035 | 27 | 0.223 | set aside: k 3 reads -9.884 nats against -9.587 at k 5, and the paired gain does not clear the 0.020 margin and its own standard error |
| spherical | 4 | -9.786 | 0.776 | 0.823 | 4921 | 4934 | 36 | 0.112 | set aside: k 4 reads -9.786 nats against -9.587 at k 5, and the paired gain does not clear the 0.020 margin and its own standard error |
| spherical | 5 | -9.587 | 0.994 | 0.753 | 4868 | 4888 | 45 | 0.068 | set aside: -9.587 nats per entity against 0.543 for the fit that won, a gap of 10.130 |
| spherical | 6 | -9.566 | 0.990 | 0.807 | 4816 | 4862 | 54 | 0.064 | set aside: k 6 reads -9.566 nats against -9.587 at k 5, and the paired gain does not clear the 0.020 margin and its own standard error |
| diagonal | 3 | -2.145 | 1.532 | 0.923 | 710 | 711 | 50 | 0.251 | set aside: k 3 reads -2.145 nats against -0.935 at k 5, and the paired gain does not clear the 0.020 margin and its own standard error |
| diagonal | 4 | -1.316 | 1.711 | 0.785 | 638 | 639 | 67 | 0.151 | set aside: k 4 reads -1.316 nats against -0.935 at k 5, and the paired gain does not clear the 0.020 margin and its own standard error |
| diagonal | 5 | -0.935 | 1.498 | 0.626 | 250 | 255 | 84 | 0.132 | set aside: -0.935 nats per entity against 0.543 for the fit that won, a gap of 1.478 |
| diagonal | 6 | -0.100 | 1.528 | 0.745 | 115 | 115 | 101 | 0.024 | set aside: k 6 puts a group at 2.4 percent, below the 5.0 percent floor |
| full | 3 | -2.653 | 0.943 | 0.725 | 1701 | 1701 | 134 | 0.199 | set aside: k 3 reads -2.653 nats against 0.543 at k 4, and the paired gain does not clear the 0.020 margin and its own standard error |
| full **chosen** | 4 | 0.543 | 2.344 | 0.813 | 920 | 921 | 179 | 0.072 | chosen: 0.543 nats per entity held out at 0.813 placement agreement |
| full | 5 | -2.623 | 2.766 | 0.678 | -1000 | -996 | 224 | 0.072 | set aside: k 5 reads -2.622 nats against 0.543 at k 4, and the paired gain does not clear the 0.020 margin and its own standard error |
| full | 6 | -1.782 | 3.466 | 0.427 | -227 | -226 | 269 | 0.020 | set aside: k 6 puts a group at 2.0 percent, below the 5.0 percent floor |

Silhouette is a Euclidean reading, so it structurally rewards the spherical shape k-means produces: the chosen partition scores 0.222 on it against 0.286 for the k-means partition the folder used to ship, and 0.168 for the same protocol on columns shuffled independently. It clears the control. That is why the family is settled on likelihood and placement, and silhouette is kept as a floor and a comparison rather than as the selector.

Every component is bounded on shape as well as on width. The width floor refuses a component narrower than a thousandth of a standard deviation on any axis, and that floor reads the diagonal; a diagonal is not a shape, because two columns that move together to within rounding can both sit near unit variance while the covariance is nearly singular along the direction they share, and a density read through that direction is decided by one row. So the ratio between a component's longest axis and its shortest is capped at 1e12, and a component that exceeds it has its short axis lifted along the direction the data gave. That cap is not a statistical bound and the card will not pretend it is: at 251 rows over 8 columns the eigenvalue error is of order 0.46 against a longest axis of 2.57, so no ratio below about 6 could be separated from sampling error at all, and a cap picked near the resolution limit refuses structure rather than noise. Measured on this panel, a cap of ten thousand or a million both hand the choice to the diagonal family and cost the full covariance fit more than two nats per entity on held-out rows. The cap therefore sits where a covariance stops being one, at the edge of what a double-precision factorisation can invert, so that only a direction flat to twelve digits is refused. The worst shipped component sits at 6,089. That is far inside the cap, so no component needed lifting: the bound is a guard this panel never triggers, and the covariances below are the ones the maximisation step produced.

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

## The k-means reading

The sweep the folder used to gate on, kept so the old reading sits beside the new one. It is the same k-means protocol at every k: Euclidean silhouette, within-group inertia, and the size of the smallest group.

| k | silhouette | smallest group | inertia | sizes |
|---:|---:|---:|---:|---|
| 3 | 0.275 | 0.227 | 1042.0 | 137, 57, 57 |
| 4 **chosen** | 0.286 | 0.147 | 923.5 | 57, 42, 115, 37 |
| 5 | 0.223 | 0.116 | 818.7 | 29, 76, 58, 30, 58 |
| 6 | 0.234 | 0.116 | 745.6 | 54, 29, 43, 43, 52, 30 |

## Audit

There are no labels, so nothing here is a truth. What is measured is how well the fit predicts rows it never saw, and whether the groups survive the two protocols that try to break a partition: the same fit on columns shuffled independently, and the same fit on resamples of the entities scored against the full panel by adjusted Rand index.

| question | reading | gate |
|---|---:|---|
| held-out likelihood, nats per entity | 0.543 | gate: clear of the shuffled control by 0.020 |
| held-out spread over 5 folds | 2.344 | reported, not gated |
| held-out placement agreement | 0.813 | at least 0.60 |
| the same fit on shuffled columns, mean of 20 | -8.592 | reported, not gated |
| the same fit on shuffled columns, best | -6.922 | reported, not gated |
| the k-means partition this folder used to ship | -9.777 | the incumbent's held-out reading |
| silhouette | 0.222 | at least 0.15 |
| shuffled control, mean | 0.168 | margin 0.08 |
| shuffled control, best of 20 | 0.185 | reported, not gated |
| resampling, mean agreement | 0.712 | at least 0.60 |
| resampling, tenth percentile | 0.451 | reported, not gated |
| resampling, worst of 50 | 0.382 | reported, not gated |

### Which groups survive resampling

| group | holds | share | Jaccard, mean of 50 | worst | reading |
|---|---:|---:|---:|---:|---|
| copy concentration low, creative reuse low | 18 | 7.2 percent | 0.266 | 0.079 | below the 0.6 floor |
| start gap spread high, time between starts high | 65 | 25.9 percent | 0.589 | 0.352 | below the 0.6 floor |
| creative reuse high, start gap spread low | 53 | 21.1 percent | 0.626 | 0.203 | reproducible |
| creative reuse low, ad volume low | 115 | 45.8 percent | 0.893 | 0.653 | reproducible |

The adjusted Rand index above is one number for the whole partition, and it weighs every pair of entities: a large group that comes back exactly pays for a small group that does not. The per-group reading is the Jaccard index, the share of rows a group and its best match in a refit have in common over the rows either holds, so a group that splits in the refit scores low even though both halves together cover it. 2 of the 4 groups clear the 0.6 floor taken from Hennig (2007) and the mean across the partition is 0.593. A group below it is a region of this panel rather than a membership: an entity placed in one is described by the group it sits nearest, and nothing here claims that entity would land in the same group on a new draw. A group that does not return under resampling is a region of this panel rather than a membership that would reappear on a new draw, and the card says which is which rather than reporting the average alone.

A silhouette of 0.22 is ordinary for behavioural data whose columns share a scale factor, and a partition read through a covariance scores lower on it by construction: a stretched group is not a sphere, and the reading measures distance from a centre, not the shape a group actually has. The separated groups are 0.22 against 0.17 for columns with their pairing destroyed, so the structure is real; the shape is the part silhouette cannot see.

## How sharp the groups are

| winning posterior | entities |
|---|---:|
| at or above 0.900 | 250 |
| between 0.600 and 0.900 | 1 |
| below 0.600 | 0 |

The mean winning posterior is 0.9994, the tenth percentile 0.9999 and the lowest reading 0.8827, with a mean margin of 0.9988 over the runner-up and a lowest of 0.7655. Taken alone that would say the groups are almost perfectly separated. It is partly the clip. A component whose covariance carries an axis at the width floor has no spread in that direction, so the density it reports there is enormous for a row on the sheet and negligible off it; three of the four shipped components are in that state, because the z is clipped at three standard deviations and a group whose members all sit on the clip has no variation left. The flat readings are listed per group below.

Read again with every covariance axis raised to at least 0.5 standard deviations, so no component can lean on a flattened axis, the mean winning posterior falls to 0.9681 and 4 entities change group, which leaves 98.4 percent of the panel where it was. That is the honest split of the claim: the partition survives the clip being taken away, the confidence does not. A leave-one-fold-out model, fitted without each row's fold and matched to the shipped components by the ordering that minimises the distance between their centres, places 84.5 percent of the 251 rows it could read in the group this card gives them.

| group | entities | axes at the floor | readings flat inside the group | condition | mean posterior | held out keeps |
|---|---:|---:|---|---:|---:|---:|
| copy concentration low, creative reuse low | 18 | 4 | `burstiness`, `terms`, `repeat` | 6089 | 1.000 | 66.7 percent |
| start gap spread high, time between starts high | 65 | 0 | none | 18 | 1.000 | 78.5 percent |
| creative reuse high, start gap spread low | 53 | 1 | `burstiness` | 1361 | 0.997 | 64.1 percent |
| creative reuse low, ad volume low | 115 | 5 | `ads`, `cadence`, `burstiness`, `repeat`, `concentrate` | 1252 | 1.000 | 100.0 percent |

| this card's group, by the group a model fitted without its fold gave it | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 12 | 1 | 0 | 5 |
| 1 | 7 | 51 | 7 | 0 |
| 2 | 14 | 5 | 34 | 0 |
| 3 | 0 | 0 | 0 | 115 |

The largest movement is 14 entities from group 2 to group 0, two centres 13.88 apart under their pooled shape. Movement is not only proximity: a group of this size refits its own shape on four fifths of the panel, and a member that sits on the thin part of one group's sheet falls inside the other's. The reading to take from the table is which groups a new draw would reproduce and which it would rearrange, and it agrees with the per-group Jaccard above rather than repeating the posterior.

| pair | distance between centres | share a boundary would confuse |
|---|---:|---:|
| 0 and 1 | 4.74 | 0.018 |
| 0 and 2 | 13.88 | 0.000 |
| 0 and 3 | 24.47 | 0.000 |
| 1 and 2 | 4.02 | 0.045 |
| 1 and 3 | 6.57 | 0.001 |
| 2 and 3 | 15.29 | 0.000 |

The distance is Mahalanobis under the two groups' pooled shape and the share is `2 Phi(-d/2)`, the overlap of two equal Gaussians at that distance. It is the closest thing to a margin a partition without labels can have, and it is why the nearest pair matters more than the mean posterior does.

| entity | group | nearest rival | posterior | rival | margin | fitted without its fold | cushioned |
|---|---:|---:|---:|---:|---:|---:|---:|
| RivCo Supervisor Yxstian Gutierrez | 2 | 1 | 0.8827 | 0.1173 | 0.7655 | 1 | 2 |
| Our Future Los Angeles | 2 | 1 | 0.9813 | 0.0187 | 0.9626 | 2 | 2 |
| Barry Wernick for Dallas County Commissioner District 2 | 2 | 1 | 0.9966 | 0.0034 | 0.9933 | 2 | 2 |
| Angela Gonzales-Torres | 2 | 1 | 0.9969 | 0.0031 | 0.9938 | 2 | 2 |
| EAA | 2 | 1 | 0.9984 | 0.0016 | 0.9967 | 0 | 2 |
| Steven Wright 2024 For Congress | 2 | 1 | 0.9984 | 0.0016 | 0.9968 | 2 | 2 |

An entity listed here sits nearest a boundary. The held-out column is where a model that never saw it placed it and the cushioned column is where it lands once no component may use a flattened axis. When the two agree with the group this card gives, the placement is the entity's; when they disagree, the row is a region rather than a membership, and the card says which rather than rounding the disagreement away.

## Recency in the mixing weights

A mixture has two places a column can enter. The densities say what a group is like, and recency is kept out of them because the panel is a snapshot weighted towards recent ads: the distance from a last start measures the archive's coverage as much as the advertiser's activity. The mixing weights make a weaker claim, that a descriptive column changes which group a row comes from, and this is the audit of whether the panel supports it. The model writes `logit(pi_k(x))` as a multinomial logistic function of the descriptive columns and leaves every density on behaviour (Dayton and Macready, JASA 83, 1988).

| model | covariates | constant weights | gated weights | gain | standard error | worst mean move |
|---|---|---:|---:|---:|---:|---:|
| recency alone | recency (2.2%) | 0.543 | 0.544 | +0.001 | 0.005 | 0.001 sd |
| all three | recency (2.2%), span (75.0%), test_share (0.6%) | 0.543 | 0.678 | +0.135 | 0.208 | 0.003 sd |

The parenthesis after each covariate is the share of that column's own variance the four groups already explain. It is the reading to look at first: a column the partition already separates cannot add much to the weights.
The per entity spread is the only estimate of the noise this panel offers, so the gate must add twice its own standard error before the gain is called one, and no component mean may move more than a tenth of a standard deviation, because a moved mean means the groups describe something else. Both readings are held out on the same folds the family choice used, and both fits start from the same model, so the difference is the gate's doing.

Gating on recency adds 0.001 nats per entity on held-out rows against a standard error of 0.005, which is not a gain, so it stays out of the fit and the densities keep their constant weights.

Gating on recency, campaign span, test share adds 0.135 nats per entity on held-out rows against a standard error of 0.209, which is not a gain, so it stays out of the fit and the densities keep their constant weights.

The wider set is reported beside the pre-registered one rather than instead of it. Its span column is mechanically tied to ad volume, which every density already reads, and the fold to fold spread of its difference is 0.466 nats, which no reading at this row count can see through. That is why the acceptance is stated before the fit: a wider set offered as a second chance would find a number above zero sooner or later, and the number would mean nothing.

## Checks

- read through a full covariance at k = 4, chosen on 0.543 nats per entity held out
- 10.320 nats per entity better than the k-means partition the folder used to ship at -9.777
- held-out likelihood 0.543 stands clear of the same fit on shuffled columns at -8.592
- a fold model places its own held-out rows at 0.813 agreement with the shipped partition
- silhouette 0.222 beats the shuffled control at 0.168
- resampling keeps the partition at 0.712 mean agreement
- 2 of 4 groups clear the 0.6 Jaccard floor under resampling; group 0 at 0.266, group 1 at 0.589 below it, each read as a region rather than a membership
- every group holds at least 7.2 percent of the entities
- the repeat gate gives the observed value +0.109 nats over the Gaussian conditional law, and calls the edge right on 84.1 percent of held-out rows against 60.1 percent for the panel rate
- the concentrate gate gives the observed value +0.139 nats over the Gaussian conditional law, and calls the edge right on 81.2 percent of held-out rows against 64.1 percent for the panel rate
- the mean winning posterior is 0.9994 and falls to 0.9681 once no component may use a flattened axis, with 4 of 251 rows changing group
- a model fitted without each row's fold keeps 84.5 percent of them in the group this card gives
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
- Recency is deliberately not a feature. The panel is a snapshot weighted towards recent ads, so the distance from a last start to the snapshot day would measure the archive's coverage as much as the advertiser's activity. It is not in the mixing weights either, and that is measured rather than assumed: the four groups explain 2.2 percent of its own spread, and gating on it moves the held-out reading by +0.001 nats per entity against a standard error of 0.005.
- The accuracy reported is predictive: how well the fitted belief explains rows it was not fitted on, in nats per entity. It is not a measure of whether a group is the right group, because no label says which advertiser belongs where. A richer family can predict better and still be the less stable reading of the panel, which is why the resampling agreement is reported beside it.
- A group below the Jaccard floor is a region of this panel, not a membership. The fit still places every entity in it, because a segment has to answer for every advertiser, but nothing here says the same advertiser would land there on a new draw.
- The family choice is made on one panel. A diagonal covariance winning here says the columns of this archive are not equally spread inside an advertiser group; it is not a claim about advertising in general, and a different register or city could choose differently.
- A group is a description of behaviour, not a target. Nothing here says that moving an advertiser between groups would change an outcome.
- The gate describes the two shares inside a partition fitted without it. A partition fitted with the gate in the loop would draw different boundaries, and settling which is better needs a measure both models share; the held-out lattice score above is that measure for the shares alone, not for the partition.
- A winning posterior here is not a probability that the group is right. 3 of the 4 components carry axes at the width floor, which is what the clip leaves behind, and a density with a zero axis is enormous on the sheet and negligible off it. The cushioned reading and the held-out confusion above are the two corrections, and both are lower than the nominal number: read the table before quoting 0.999.
- The confidence a row carries and the stability of its group are separate readings. Group 3 holds 100 percent of its 115 entities when a model is fitted without them, while the smaller groups keep between 64 and 78 percent of theirs at a mean posterior of 0.997 or better. A stable large group and a small unstable one can both be sharp on this panel.
- The held-out vote places each row with a model fitted on four fifths of the panel, matched to the shipped groups by the ordering that minimises the distance between centres. A row that moves there is not mislabelled by this card: the two models answer the same question with different evidence, and the disagreement is the reading.

