# Churn baseline

xcm-mmm-churn-baseline v1, built 2026-11-14T22:59:27Z. A logistic over the telecom panel, fitted so a marketing
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

Fold AUC: 0.864, 0.859, 0.876, 0.862, 0.878

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

- One operator, one snapshot. A coefficient here is not a market law.
- Right censoring: a customer who stayed twelve months is at risk for
  twelve months and no longer, which the series respects and the regression
  does not attempt to correct.
- `Joined` customers are censored rather than counted as non-churners.
