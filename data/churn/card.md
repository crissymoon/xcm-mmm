# Churn baseline

xcm-mmm-churn-baseline v3, built 2026-11-18T02:20:10Z. A logistic over the telecom panel, fitted so a marketing
model has a retention baseline to beat rather than a number to quote.

## Panel

7,043 customers, 1,869 churned. Five folds assigned by a digest of the customer id,
so the split is the same on every machine.

## Window

Time zero is the join, so tenure month 1 is the first month at risk and number 0 cannot appear. 7,043 rows, 1,869 churn events, 356 rows still at risk at month 72 and therefore censored rather than counted as survivors. The series and the panel agree on the event count. Months below the 30-row at-risk floor: none.

The last survival reading is not compared with `1 - events / rows`. The series measures a hazard over a shrinking risk set, so a customer who stops contributing without churning leaves the denominator; the crude share divides by every row at once. The two agree only when nobody is censored before the last month.

## Held out

| reading | value |
|---|---:|
| AUC | 0.867 |
| AUC, tenure and monthly only | 0.801 |
| balanced accuracy at the fold cut | 0.785 |
| accuracy at the fold cut | 0.786 |
| majority baseline | 0.735 |
| folds | 5 |
| AUC 95% interval | [0.857, 0.876] |
| AUC 95% interval, DeLong logit | [0.858, 0.876] |
| AUC 95% interval, BCa | [0.858, 0.877] |
| AUC standard error | 0.0046 |
| Brier score | 0.134 [0.130, 0.138] |
| log loss | 0.418 |
| expected calibration error | 0.075 [0.069, 0.084] |
| expected calibration error after the beta map, in sample | 0.006 |
| conformal coverage at 90% nominal | 0.917 |
| conformal mean set size | 1.24 |
| gradient boosted trees, AUC | 0.898 [0.890, 0.905] |

On this snapshot the tree ensemble ranks better (0.898 against 0.867).
That is the tabular comparison the review asked to test rather than assume;
it is one panel, not a law about neural models.

Fold AUC: 0.864, 0.859, 0.876, 0.862, 0.878

## Uncertainty

Three intervals are reported and the widest is the one to quote, here the
bootstrap reading at [0.857, 0.876]. The bootstrap is a percentile interval over 400
seeded resamples. The DeLong interval is asymptotic and its bounds are read on
the logit scale, so they cannot leave (0, 1); it is centred on the continuity
corrected AUC rather than the raw reading, which is why its ends are not
equidistant from the point. BCa corrects the percentile interval for bias
(z0 +0.037) and for the skew of a bounded metric near a ceiling (acceleration
+0.0287), both read off the same resamples at no extra data cost. The logit is
not decoration: DeLong's normal interval is liberal when positives are under
about fifty, and without the transform the upper bound of a near perfect
classifier crosses one and stops bounding anything. The conformal threshold is
fitted on the first half of the held-out rows and coverage is measured on the
second half, so the coverage number is not the calibration set scored against
itself. The tree ensemble uses the same folds, features and target as the
logistic; it is the tabular comparison the card previously lacked, not a second
headline.

### Does the interval hold?

Coverage was measured by simulation at 251 rows with 50 positives, a
generating world whose true area under the curve is known in closed form
(0.760), and 2000 independent trials per method. Nominal is 95% and the band
is 0.93 to 0.97.

| interval | measured coverage | Monte Carlo 95% | mean width | inside the band |
|---|---:|---:|---:|:---:|
| bootstrap-percentile | 0.938 | [0.927, 0.948] | 0.144 | yes |
| bca | 0.936 | [0.924, 0.946] | 0.145 | yes |
| delong-logit | 0.947 | [0.936, 0.956] | 0.146 | yes |

The Monte Carlo column is the Wilson interval on the measured share, and it is not decoration. An earlier run of this simulation used 200 trials and reported 0.925 for DeLong against a 0.930 floor, which read as a failure of the analytic interval. Rerunning at 2,000 trials put the same estimator at 0.947 and inside the band, so the gap was the simulation's own noise: two hundred trials leave a standard error near 0.015, which is wider than any difference the band can resolve.

bootstrap-percentile, bca, delong-logit held nominal 0.95 inside [0.93, 0.97] at 251 rows with 50 positives

### Calibration

The expected calibration error is a mean of ten noisy bin ratios, so it is
reported with a percentile interval: 0.075 [0.069, 0.084]. beta map fitted on these rows: a 1.716, b -2.093, c +0.333, in-sample ECE 0.006 against raw 0.075. An in-sample error understates both, so the honest comparison needs a held-out split.

## Class conditional conformal

At 90% nominal each class is calibrated on its own rows, so the
guarantee is conditional on the class rather than marginal over the panel.
A rank of `ceil((n + 1)(1 - alpha))` needs at least `1 / alpha` calibration
rows of that class; a class with fewer is refused rather than given a number
it cannot support.

| class | calibration rows | coverage | coverage 95% Wilson | mean set size |
|---|---:|---:|---:|---:|
| churn | 974 | 0.896 | [0.876, 0.914] | 1.90 |
| stayed | 2547 | 0.907 | [0.895, 0.918] | 0.92 |

The pooled reading is 0.917 coverage at a mean set of 1.24. It is a
marginal number, and it hides the class the reader cares about: the churn
class covers less than nominal while the stayed class covers more, so the
marginal figure is a weighted average of an under-covering tail and an
over-covering body. The Wilson column is what makes the difference readable.
At 974 churn rows it reaches nominal, so the churn shortfall is at the edge
of resolution rather than established.

## Calibration

| bin | rows | predicted | observed |
|---|---:|---:|---:|
| 0.0-0.1 | 1582 | 0.064 | 0.013 |
| 0.1-0.2 | 1565 | 0.146 | 0.078 |
| 0.2-0.3 | 1209 | 0.251 | 0.189 |
| 0.3-0.4 | 968 | 0.348 | 0.352 |
| 0.4-0.5 | 755 | 0.447 | 0.556 |
| 0.5-0.6 | 594 | 0.552 | 0.714 |
| 0.6-0.7 | 370 | 0.629 | 0.846 |

## Support

1,869 churned rows across 14 coefficients, 133.5 events per feature. Adequate for a logistic on this feature set.

| training fraction | rows | held-out AUC |
|---|---:|---:|
| 25% | 1,409 | 0.867 |
| 50% | 2,817 | 0.867 |
| 75% | 4,226 | 0.867 |
| 100% | 5,634 | 0.868 |

Last-doubling gain 0.12 AUC points. A positive number means the bank was
still buying accuracy at the largest fraction; a flat one means the ceiling is
the features or the operator, not the row count. Neither reading transports to
a new operator.

## Labels and resolution

| reading | value |
|---|---:|
| confidently contradicted rows | 551 of 7,043 |
| cleared neither threshold, out of the joint | 2,469 |
| estimated label noise | 0.121 |
| estimated noise, churn labels | 0.128 |
| estimated noise, stayed labels | 0.118 |
| ceiling against these labels | 0.879 |
| smallest resolvable AUC gain at 80% power | 0.022 |
| pre-registered floor | 0.030 |
| paired AUC gain, trees against logistic | +0.031 [+0.026, +0.036] |

551 of 7,043 rows are confidently labelled against the model and 2,469 cleared neither threshold, so 4,574 stay in the joint. Scaled back up to the labelled totals that is 850 errors, 12.1 percent of the bank, 12.8 percent of the positive labels and 11.8 percent of the negative ones. A model that reproduced the true labels would score 0.879 against these ones, so a claim above that is a claim about the labels rather than the construct. The estimate assumes the rows left out carry the same error rate as the ones kept; 35.1 percent of the bank is left out, so that assumption is doing real work and the reading is an extrapolation rather than a count.

At this shape a difference of 0.022 is the smallest that clears a 95 percent test at 80 percent power, and the pre-registered floor of 0.030 sits above it, so a gain claimed at the floor is visible on this bank.

The tree comparison is read paired: both models score the same 7,043 rows, so the difference has its own error from the DeLong covariance rather than from two independent ones. The placements correlate at 0.85, which is why the interval on a gain of +0.031 is narrower than either model's own interval. The interval stays above zero, so on this panel the gain survives its own sampling error.

The noise estimate is confident learning (Northcutt, Jiang and Chuang, 2021) run on the out-of-fold probabilities the folds already produced, with a per-class threshold at the mean predicted probability of that class. It separates random from systematic labelling error only in the sense of measuring the total: a bank where one annotator misread the construct scores like one where labels were flipped by a coin, and only the first is a design problem. It is also only as good as the model behind it, which is why it is quoted beside the AUC rather than instead of it. Where a large share of the bank clears neither threshold, the scaling is what carries the estimate: the rows the model contradicts are counted, and the rest is the assumption that the rows it could not place are wrong at the same rate. On this panel that assumption is the difference between the 12.1 percent above and the 7.8 percent the contradiction count alone implies.

## Coefficients

On the standardised scale the fit uses. A sign is a direction, not a
cause.

| feature | weight |
|---|---:|
| two_year | -0.327 |
| tenure | -0.276 |
| referrals | -0.272 |
| monthly | +0.219 |
| one_year | -0.217 |
| fiber | +0.212 |
| dependents | -0.203 |
| credit_card | -0.152 |
| total | -0.120 |
| security | -0.110 |
| support | -0.102 |
| age | +0.102 |
| bank_withdrawal | +0.085 |
| dsl | -0.023 |

## The reason text

The 1,869 churn reasons are terse service labels, not customer narratives. The
lexicon reads the wording: `Attitude of support person` scores positive because
`support` is a trust word, and `Competitor had better devices` scores positive
because `better` is. Read the categories as an ordering of wording, not as a
measurement of how the customer felt. The labelled bank is where feeling is
measured; the reason text is where a driver is named.


## What is not in the model

- Churn reason text. All 1,869 reasons belong to customers who already
  left, so a model that read them would be predicting the past. The reasons
  are read on the emotion axes in `reasons.csv`, beside the panel rather than
  inside it.
- Dates. The table carries tenure, not calendar time, so this baseline
  cannot attribute a campaign; it can only say how much churn the panel
  explains before attribution starts.
- Geography. City and zip are in the source and are left out: at this
  sample size they encode the operator's footprint more than customer
  behaviour.

## Limits

- One operator, one snapshot. A coefficient here is not a market law. The
  interval is a statement about this panel's sampling noise, not about a new
  city or a new register.
- Right censoring: a customer who stayed twelve months is at risk for
  twelve months and no longer, which the series respects and the regression
  does not attempt to correct.
- `Joined` customers are censored rather than counted as non-churners.
- The coefficients are directions, not effects. The shipped panel has no
  treatment column, so it cannot answer an intervention question. A panel that
  carries one is read by `churn uplift`, which reports a cross-fitted T-learner
  and an inverse probability weighted average effect with bootstrap intervals,
  a Qini ranking, a decile table and a placebo p-value. That estimator still
  adjusts for observed covariates only; unmeasured confounding is not repaired.
- Selection on the forecasting path is nested when `forecast --nested` is
  used: the inner loop chooses and the outer loop scores. The plain command
  remains a single holdout and the report says which one ran.
- Drift and rollback are commands, not a daemon. `churn monitor` compares a
  reference panel with a current one by PSI and KS; `churn rollback` moves the
  artifact pointer back one version. A scheduler still has to call them.
- Calibration, conformal coverage and the tree comparison are measured on
  the same held-out rows as the AUC. None of them transports to a new operator.
- The label audit bounds what this bank's own labels can carry; it does not
  audit the construct. A ceiling of one would mean the labels are consistent with
  the model, not that they are right about churn.
- The resolution reading is a property of the panel's size and class
  balance. A gain the card does not claim is not proof of absence, only a
  difference this bank cannot see.
- Marginal coverage is not class conditional coverage. The pooled number
  holds across the held-out rows as a whole; the per-class numbers are the
  ones the card now quotes, and a class whose calibration rows cannot support
  the requested alpha is refused rather than given a reading.
