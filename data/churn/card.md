# Churn baseline

xcm-mmm-churn-baseline v2, built 2026-11-16T02:17:01Z. A logistic over the telecom panel, fitted so a marketing
model has a retention baseline to beat rather than a number to quote.

## Panel

7,043 customers, 1,869 churned. Five folds assigned by a digest of the customer id,
so the split is the same on every machine.

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
| AUC standard error | 0.0046 |
| Brier score | 0.134 |
| log loss | 0.418 |
| expected calibration error | 0.075 |
| conformal coverage at 90% nominal | 0.917 |
| conformal mean set size | 1.24 |
| gradient boosted trees, AUC | 0.898 [0.890, 0.905] |

On this snapshot the tree ensemble ranks better (0.898 against 0.867).
That is the tabular comparison the review asked to test rather than assume;
it is one panel, not a law about neural models.

Fold AUC: 0.864, 0.859, 0.876, 0.862, 0.878

## Uncertainty

The bootstrap interval is a percentile interval over the held-out rows,
400 resamples, seeded so a rebuild reproduces it. The DeLong interval is
asymptotic, and its bounds are read on the logit scale, so it cannot leave
(0, 1); it is centred on the continuity corrected AUC rather than on the raw
reading, which is why its two ends are not equidistant from the point. Quote
the wider of the two, here the bootstrap reading at [0.857, 0.876]. The logit is not
decoration: DeLong's normal interval is liberal when positives are under about
fifty, and without the transform the upper bound of a near perfect classifier
crosses one and stops bounding anything. The conformal threshold is fitted on
the first half of the held-out rows and coverage is measured on the second
half, so the coverage number is not the calibration set scored against itself.
The tree ensemble uses the same folds, features and target as the logistic; it
is the tabular comparison the card previously lacked, not a second headline.

## Class conditional conformal

At 90% nominal each class is calibrated on its own rows, so the
guarantee is conditional on the class rather than marginal over the panel.
A rank of `ceil((n + 1)(1 - alpha))` needs at least `1 / alpha` calibration
rows of that class; a class with fewer is refused rather than given a number
it cannot support.

| class | calibration rows | coverage | mean set size |
|---|---:|---:|---:|
| churn | 974 | 0.896 | 1.90 |
| stayed | 2547 | 0.907 | 0.92 |

The pooled reading is 0.917 coverage at a mean set of 1.24. It is a
marginal number, and it hides a class whose calibration rows are too few for
the requested tail.

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
- Marginal coverage is not class conditional coverage. The pooled number
  holds across the held-out rows as a whole; the per-class numbers are the
  ones the card now quotes, and a class whose calibration rows cannot support
  the requested alpha is refused rather than given a reading.
