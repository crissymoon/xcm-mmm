# Forecasting

[Back to the project](../../README.md)

This page explains the forecasting path, what each number means and where the accuracy actually comes from.

The short version: the search does not make a linear model cleverer. It chooses which columns the model is allowed to see, using a split that respects time, and it reports the result on rows the model has never seen. Most of the gain in a noisy panel comes from dropping columns that only looked useful in sample.

## Data

`forecast` accepts CSV or JSON Lines. The file must contain a header or object keys, at least two feature columns, and a target column named by `--target`.

CSV rules:

- Comma separated, one header row, values parsed as `f64`.
- A column is kept when at least four rows in five parse as a finite number, so an occasional blank does not remove a feature, while a date or label column is dropped.
- A row with a missing value in a kept column is skipped, not filled.
- The number of skipped rows is reported as `skipped_rows` in the JSON.

JSON Lines rules:

- One object per line, numeric fields only.
- The target must be numeric on every kept line.
- The feature schema comes from the first usable object, in sorted key order (the JSON map is ordered), and later objects missing a feature are skipped.
- String fields such as a date are ignored.

Features are standardised to zero mean and unit variance before any search. This matters for ridge: the penalty should treat every column on the same scale, or it will shrink the small-magnitude columns first for no good reason. The target is left in its own units, so the reported error is readable.

## Walk-forward folds

Random k-fold cross-validation is wrong for a time series, because a row can train on its own future. The split here is blocked:

```text
rows   0 ...................... n
       |---- train ----|-emb-|-- test --|
       |------- train -------|-emb-|-- test --|
       |----------- train ---------|-emb-|-- test --|
```

With `n` rows and `f` folds, the block size is `n / (f + 1)`. Fold `k`, for `k` from 1 to `f`, trains on rows before its test block and tests on that block. The embargo removes the last `embargo` rows of each training window, which is the guard against a label that overlaps its own future. For a one day ahead return, an embargo of one row is enough; for a five day forward return, use five.

Every fold reports its own error, R2, MAE and information coefficient, and the pooled numbers concatenate the test predictions. A model that only works in one regime shows up as uneven per fold numbers, which is a warning the pooled figure can hide.

## Ridge regression

For a chosen set of columns, the model is an ordinary least squares fit with a ridge penalty:

```text
minimise  ||y - X w - b||^2 + lambda ||w||^2
```

The normal equations are solved by Cholesky factorisation after adding `lambda` to the diagonal. Features are standardised, so `lambda = 1.0` is a mild shrinkage rather than a strong one. A singular system is reported as a failed fit, never as a silent zero.

Ridge is used because it is closed form, deterministic, and fast enough to fit four times for each of several thousand candidate feature sets. It is also honest: there is no early stopping, no random seed and no hidden validation split to leak.

## The two ways to search

The decision vector is a mask in `[0, 1]^p`. A feature is used when its mask value is above 0.5. The constraint `1 - count <= 0` keeps at least one feature alive.

**Single objective, `SparseAccuracy`.** Minimise

```text
cross-validated MSE + sparsity * (features / p)
```

This is what `de` solves. It returns one model, the best trade-off for the penalty you chose. The default penalty is 0.01, small enough to be a tie-breaker rather than a driver.

**Two objectives, `SelectionProblem`.** Minimise

```text
objective 1: cross-validated MSE
objective 2: features / p
```

This is what `moead` solves. It returns a front from one feature at the best error down to all features. Every point is a real model, scored on the same folds. The knee is the point furthest below the chord between the two ends of the front, which is the usual compromise between error and size; the best point is the smallest error on the curve. The CLI prints both.

A feature set can be scored without any search, which is useful for a baseline or a hypothesis:

```rust
let report = problem.report(&[0, 3, 7]);
println!("{}", report.summary());
```

## Reading the report

| Field | Meaning |
|---|---|
| `mse` | pooled mean squared error on test rows |
| `rmse` | square root of the pooled MSE, in target units |
| `mae` | pooled mean absolute error, in target units |
| `r2` | `1 - SSE / SST` against the test mean |
| `ic` | mean Spearman rank correlation between prediction and outcome across folds |
| `baseline_mse` | error of predicting each fold's training mean |
| `skill` | `1 - mse / baseline_mse`, the share of the baseline error removed |
| `folds` | per fold window, MSE, R2, MAE and IC |

Skill is the number to read first. A model with a high R2 on a target that barely moves can still have no skill; a model with a modest R2 and a positive skill has found something. The information coefficient is the one to read when the target is a return, because the ordering of predictions matters more than their level.

## Worked example

The example panel has 800 rows, 12 standardised features, four of which drive the target, and an autoregressive term that makes the split meaningful.

```bash
xcm-mmm synth --rows 800 --features 12 --informative 4 --seed 7 --out tmp/panel.csv
xcm-mmm forecast tmp/panel.csv --folds 4 --method both --budget 6000 --holdout 0.25
```

```text
rows        800   features 12   folds 4
model        feat          mse       r2      ic   skill
all            12     0.389071   0.6019   0.768   64.7%
de              4     0.325792   0.6667   0.810   70.5%
  keeps: x1, x2, x3, x4
moead-knee      2     0.480321   0.5086   0.722   56.5%
  keeps: x1, x3
moead-best      4     0.325792   0.6667   0.810   70.5%
  keeps: x1, x2, x3, x4
  front: 4 points, knee 2, best 0

holdout     rows 601..800   trained on 0..600   never seen by the search
model        feat          mse       r2      ic   skill
all            12     0.362740   0.7103   0.840   71.8%
de              4     0.366311   0.7075   0.838   71.5%
moead-knee      2     0.625203   0.5007   0.703   51.4%
moead-best      4     0.366311   0.7075   0.838   71.5%
```

The single objective search recovers exactly the four informative features. Inside the search window that looks like a clear win: R2 0.6019 for all twelve columns against 0.6667 for the four. The holdout table says otherwise. On the last 200 rows, which the search never saw, all twelve score 0.7103 and the selected four score 0.7075. The apparent gain was the search choosing columns on the same folds the report scores, which is the standard way feature selection flatters itself.

That is the reason the flag exists. Read the first table to see what the search believes, and the holdout table to see what it can deliver. When the two disagree, believe the second one.

The same test on a real panel of 324,539 rows and 36 numeric columns, search budget 2000, four folds, holdout of the last quarter:

| target | embargo rows | all features IC | selected IC | all R2 | selected R2 | features kept |
|---|---|---|---|---|---|---|
| 1 day return | 500 | 0.008 | 0.018 | -0.0068 | 0.0006 | 2 |
| 5 day return | 2500 | 0.040 | 0.051 | 0.0008 | 0.0048 | 3 |
| 21 day return | 10500 | 0.048 | 0.038 | 0.0053 | 0.0008 | 2 |

Two horizons improve and one gets worse, by margins that sit near the noise floor: overlapping labels mean the effective sample is roughly a fifth of the stated rows, so a difference of 0.01 in IC is under two standard errors. The honest reading is that selection on this panel is close to a wash, and the in-window table would have hidden that. A wide panel with correlated, mostly useless columns is where the search pays; a panel of well chosen features is where it costs.

Two concrete levers do move accuracy: dropping columns only helps when the dropped columns are actually noise, and the embargo has to cover the label horizon in rows, not days. A five day return on a panel of roughly 500 tickers per date needs about 2500 rows of embargo, not five.

## Where better accuracy comes from

In order of how much they usually matter:

1. **Split on time, never at random.** A random split lets the model see the future. Every number in this crate assumes a walk-forward split; keep it that way.
2. **Compare with the training mean.** The `all` row is there so a model has to earn its place. A skill near zero means the features added nothing, whatever R2 says.
3. **Drop columns that only help in sample.** Use the front, not a single number. A point with half the features and one percent more error is often the better model out of sample, and it is far less likely to break next quarter.
4. **Make the target stationary.** Predict a return or a change, not a level. A model that predicts the last price is accurate and useless.
5. **Match the embargo to the label horizon.** A five day forward return needs at least five rows of embargo. A leak here flatters every number downstream.
6. **Use more than four folds on a long panel.** Six to ten folds give a better view of regime stability, at the cost of a shorter first training block.
7. **Tune the ridge penalty on the folds, not on the report.** `--lambda` is a search setting; the report measures the model, not the penalty. Trying several values and reading the best report is a quiet way to overfit.
8. **Hold rows back before you believe a selection.** Run with `--holdout 0.25` and quote the second table. Without it the search picks columns using the same folds the report scores, and the selected rows win by construction.
9. **Keep the holdout at the end, in time.** The holdout is the last fraction of rows for that reason. Row order matters, so sort by date before feeding a panel in.
8. **Check IC as well as error.** A forecast can have a good R2 and no rank information, which is useless for ranking candidates.
9. **Add features with a reason.** Lags, spreads, ratios and regime flags derived from the same row are fine. Anything computed from future rows is not.
10. **Look at per fold detail.** If one fold carries all the skill, the model has found a regime, not a relationship.

What this crate does not do is model transaction costs, turnover, position sizing or non-stationarity in the coefficients. A feature set with positive IC is a candidate, not a strategy.

## Limits

- Ridge is linear. A nonlinear learner can be plugged in by implementing `Problem` over a different fit, but the mask decoding, the folds and the metrics stay the same.
- Missing values are handled by skipping rows, not by imputation. A panel with many gaps loses rows.
- The search treats each feature independently. Interaction features have to be built before the search.
- The fold count is fixed in advance. Very short panels return fewer folds than requested rather than empty windows.
- The search is stochastic. Two runs with different seeds can return different subsets when several are equally good; the report should be read as evidence, not as a unique answer.
- This is not a full marketing mix model. There is no adstock carry over, no saturation curve and no channel budget allocation. Those are feature construction and post fit steps; build the transformed columns first, then let the search choose among them.
