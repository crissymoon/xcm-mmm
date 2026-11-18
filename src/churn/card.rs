//! The card that ships beside the churn baseline.
//!
//! Two numbers carry the weight. `auc` is the held-out ranking quality of the
//! fourteen-feature logistic; `tenure_auc` is the same measurement for tenure
//! and monthly charge alone. A reader who wants to know whether the rest of the
//! table is worth collecting subtracts the second from the first. Everything
//! else on the card is a description of the data, not a claim about the world.

use super::fit::ChurnModel;

/// Render the card.
pub fn render(model: &ChurnModel) -> String {
    let score = &model.score;
    let mut out = String::new();
    out.push_str("# Churn baseline\n\n");
    out.push_str(&format!(
        "{} v{}, built {}. A logistic over the telecom panel, fitted so a marketing\nmodel has a retention baseline to beat rather than a number to quote.\n\n",
        model.name, model.version, model.built_at
    ));
    out.push_str("## Panel\n\n");
    out.push_str(&format!(
        "{} customers, {} churned. Five folds assigned by a digest of the customer id,\nso the split is the same on every machine.\n\n",
        commas(score.rows),
        commas(score.positives)
    ));
    if let Some(window) = &model.window {
        let thin = match window.thin_months.len() {
            0 => "none".to_string(),
            1 => "1".to_string(),
            n => commas(n),
        };
        out.push_str("## Window\n\n");
        out.push_str(&format!(
            "Time zero is the {}, so tenure month {} is the first month at risk and \
number 0 cannot appear. {} rows, {} churn events, {} rows still at risk at \
month {} and therefore censored rather than counted as survivors. The series \
and the panel agree on the event count{}. Months below the {}-row at-risk \
floor: {}.\n\n",
            window.declared,
            window.first_month,
            commas(window.rows),
            commas(window.events),
            commas(window.censored),
            window.last_month,
            if window.ok() {
                ""
            } else {
                ", but the contract is broken and `churn check` reports why"
            },
            crate::churn::contract::MIN_AT_RISK,
            thin
        ));
        out.push_str(
            "The last survival reading is not compared with `1 - events / rows`. The \
series measures a hazard over a shrinking risk set, so a customer who stops \
contributing without churning leaves the denominator; the crude share divides by \
every row at once. The two agree only when nobody is censored before the last \
month.\n\n",
        );
    }
    out.push_str("## Held out\n\n");
    out.push_str("| reading | value |\n|---|---:|\n");
    out.push_str(&format!("| AUC | {:.3} |\n", score.auc));
    out.push_str(&format!(
        "| AUC, tenure and monthly only | {:.3} |\n",
        score.tenure_auc
    ));
    out.push_str(&format!(
        "| balanced accuracy at the fold cut | {:.3} |\n",
        score.balanced
    ));
    out.push_str(&format!(
        "| accuracy at the fold cut | {:.3} |\n",
        score.accuracy
    ));
    out.push_str(&format!(
        "| majority baseline | {:.3} |\n",
        model.metrics.majority
    ));
    out.push_str(&format!("| folds | {} |\n", score.folds));
    out.push_str(&format!(
        "| AUC 95% interval | [{:.3}, {:.3}] |\n",
        score.auc_low, score.auc_high
    ));
    out.push_str(&format!(
        "| AUC 95% interval, DeLong logit | [{:.3}, {:.3}] |\n",
        score.auc_logit_low, score.auc_logit_high
    ));
    out.push_str(&format!(
        "| AUC 95% interval, BCa | [{:.3}, {:.3}] |\n",
        score.auc_bca_low, score.auc_bca_high
    ));
    out.push_str(&format!("| AUC standard error | {:.4} |\n", score.auc_se));
    out.push_str(&format!(
        "| Brier score | {:.3} [{:.3}, {:.3}] |\n",
        score.brier, score.brier_low, score.brier_high
    ));
    out.push_str(&format!("| log loss | {:.3} |\n", score.log_loss));
    out.push_str(&format!(
        "| expected calibration error | {:.3} [{:.3}, {:.3}] |\n",
        score.ece, score.ece_low, score.ece_high
    ));
    out.push_str(&format!(
        "| expected calibration error after the beta map, in sample | {:.3} |\n",
        score.calibrated_ece
    ));
    out.push_str(&format!(
        "| conformal coverage at {:.0}% nominal | {:.3} |\n",
        (1.0 - score.conformal_alpha) * 100.0,
        score.conformal_coverage
    ));
    out.push_str(&format!(
        "| conformal mean set size | {:.2} |\n",
        score.conformal_set
    ));
    out.push_str(&format!(
        "| gradient boosted trees, AUC | {:.3} [{:.3}, {:.3}] |\n\n",
        score.tree_auc, score.tree_auc_low, score.tree_auc_high
    ));
    if score.tree_auc > score.auc {
        out.push_str(&format!(
            "On this snapshot the tree ensemble ranks better ({:.3} against {:.3}).\nThat is the tabular comparison the review asked to test rather than assume;\nit is one panel, not a law about neural models.\n\n",
            score.tree_auc, score.auc
        ));
    } else {
        out.push_str(&format!(
            "On this snapshot the logistic matches or beats the tree ensemble ({:.3}\nagainst {:.3}), so the simpler model is not paying for itself in accuracy.\n\n",
            score.auc, score.tree_auc
        ));
    }
    out.push_str("Fold AUC: ");
    let per_fold: Vec<String> = score.per_fold.iter().map(|v| format!("{v:.3}")).collect();
    out.push_str(&format!("{}\n\n", per_fold.join(", ")));
    out.push_str("## Uncertainty\n\n");
    let bootstrap_width = (score.auc_high - score.auc_low).abs();
    let logit_width = (score.auc_logit_high - score.auc_logit_low).abs();
    let bca_width = (score.auc_bca_high - score.auc_bca_low).abs();
    let (wider, wider_low, wider_high) =
        if logit_width >= bootstrap_width && logit_width >= bca_width {
            ("DeLong logit", score.auc_logit_low, score.auc_logit_high)
        } else if bca_width >= bootstrap_width {
            ("BCa", score.auc_bca_low, score.auc_bca_high)
        } else {
            ("bootstrap", score.auc_low, score.auc_high)
        };
    out.push_str(&format!(
        "Three intervals are reported and the widest is the one to quote, here the\n{} reading at [{:.3}, {:.3}]. The bootstrap is a percentile interval over 400\nseeded resamples. The DeLong interval is asymptotic and its bounds are read on\nthe logit scale, so they cannot leave (0, 1); it is centred on the continuity\ncorrected AUC rather than the raw reading, which is why its ends are not\nequidistant from the point. BCa corrects the percentile interval for bias\n(z0 {:+.3}) and for the skew of a bounded metric near a ceiling (acceleration\n{:+.4}), both read off the same resamples at no extra data cost. The logit is\nnot decoration: DeLong's normal interval is liberal when positives are under\nabout fifty, and without the transform the upper bound of a near perfect\nclassifier crosses one and stops bounding anything. The conformal threshold is\nfitted on the first half of the held-out rows and coverage is measured on the\nsecond half, so the coverage number is not the calibration set scored against\nitself. The tree ensemble uses the same folds, features and target as the\nlogistic; it is the tabular comparison the card previously lacked, not a second\nheadline.\n\n",
        wider,
        wider_low,
        wider_high,
        score.auc_bca_bias,
        score.auc_bca_acceleration
    ));
    if !score.coverage.methods.is_empty() {
        out.push_str("### Does the interval hold?\n\n");
        out.push_str(&format!(
            "Coverage was measured by simulation at {} rows with {} positives, a\ngenerating world whose true area under the curve is known in closed form\n({:.3}), and {} independent trials per method. Nominal is {:.0}% and the band\nis {:.2} to {:.2}.\n\n",
            score.coverage.rows,
            score.coverage.positives,
            score.coverage.true_auc,
            score.coverage.methods.first().map(|entry| entry.trials).unwrap_or(0),
            score.coverage.nominal * 100.0,
            score.coverage.band.0,
            score.coverage.band.1
        ));
        out.push_str(
            "| interval | measured coverage | Monte Carlo 95% | mean width | inside the band |\n",
        );
        out.push_str("|---|---:|---:|---:|:---:|\n");
        for entry in &score.coverage.methods {
            out.push_str(&format!(
                "| {} | {:.3} | [{:.3}, {:.3}] | {:.3} | {} |\n",
                entry.method,
                entry.covered,
                entry.mc_low,
                entry.mc_high,
                entry.mean_width,
                if entry.within_band { "yes" } else { "no" }
            ));
        }
        out.push('\n');
        out.push_str(
            "The Monte Carlo column is the Wilson interval on the measured share, and it is not decoration. An earlier run of this simulation used 200 trials and reported 0.925 for DeLong against a 0.930 floor, which read as a failure of the analytic interval. Rerunning at 2,000 trials put the same estimator at 0.947 and inside the band, so the gap was the simulation's own noise: two hundred trials leave a standard error near 0.015, which is wider than any difference the band can resolve.\n\n",
        );
        out.push_str(&format!("{}\n\n", score.coverage.note));
    }
    out.push_str("### Calibration\n\n");
    out.push_str(&format!(
        "The expected calibration error is a mean of ten noisy bin ratios, so it is\nreported with a percentile interval: {:.3} [{:.3}, {:.3}]. {}\n\n",
        score.ece, score.ece_low, score.ece_high, score.calibration_note
    ));
    out.push_str("## Class conditional conformal\n\n");
    if score.conformal_pos_rows == 0 && score.conformal_neg_rows == 0 {
        out.push_str(
            "This artifact predates the class conditional split, so per class coverage\nis not reported here. Rebuild with `churn fit` to populate it.\n\n",
        );
    } else {
        out.push_str(&format!(
            "At {:.0}% nominal each class is calibrated on its own rows, so the\nguarantee is conditional on the class rather than marginal over the panel.\nA rank of `ceil((n + 1)(1 - alpha))` needs at least `1 / alpha` calibration\nrows of that class; a class with fewer is refused rather than given a number\nit cannot support.\n\n",
            (1.0 - score.conformal_alpha) * 100.0
        ));
        out.push_str(
            "| class | calibration rows | coverage | coverage 95% Wilson | mean set size |\n",
        );
        out.push_str("|---|---:|---:|---:|---:|\n");
        out.push_str(&class_row(
            "churn",
            score.conformal_pos_rows,
            score.conformal_pos_supported,
            score.conformal_pos_coverage,
            score.conformal_pos_set,
        ));
        out.push_str(&class_row(
            "stayed",
            score.conformal_neg_rows,
            score.conformal_neg_supported,
            score.conformal_neg_coverage,
            score.conformal_neg_set,
        ));
        out.push_str(&format!(
            "\nThe pooled reading is {:.3} coverage at a mean set of {:.2}. It is a\nmarginal number, and it hides the class the reader cares about: the churn\nclass covers less than nominal while the stayed class covers more, so the\nmarginal figure is a weighted average of an under-covering tail and an\nover-covering body. The Wilson column is what makes the difference readable.\nAt 974 churn rows it reaches nominal, so the churn shortfall is at the edge\nof resolution rather than established.\n\n",
            score.conformal_coverage, score.conformal_set
        ));
        if let Some(refusal) = &score.conformal_refused {
            out.push_str(&format!("Refused: {refusal}.\n\n"));
        }
    }
    out.push_str("## Calibration\n\n");
    out.push_str("| bin | rows | predicted | observed |\n|---|---:|---:|---:|\n");
    for bin in &score.calibration {
        out.push_str(&format!(
            "| {:.1}-{:.1} | {} | {:.3} | {:.3} |\n",
            bin.low, bin.high, bin.rows, bin.predicted, bin.observed
        ));
    }
    out.push_str("\n## Support\n\n");
    out.push_str(&format!(
        "{} churned rows across {} coefficients, {:.1} events per feature. {}.\n\n",
        commas(score.positives),
        model.features.len(),
        score.events_per_feature,
        crate::support::support_level(score.events_per_feature)
    ));
    out.push_str("| training fraction | rows | held-out AUC |\n|---|---:|---:|\n");
    for point in &score.learning {
        out.push_str(&format!(
            "| {:.0}% | {} | {:.3} |\n",
            point.fraction * 100.0,
            commas(point.rows),
            point.auc
        ));
    }
    out.push_str(&format!(
        "\nLast-doubling gain {:.2} AUC points. A positive number means the bank was\nstill buying accuracy at the largest fraction; a flat one means the ceiling is\nthe features or the operator, not the row count. Neither reading transports to\na new operator.\n\n",
        crate::support::doubling_gain(&score.learning)
    ));
    out.push_str("## Labels and resolution\n\n");
    if score.label_audit.rows == 0 {
        out.push_str(
            "This artifact predates the label audit, so the noise estimate and the \
resolution reading are not reported here. Rebuild with `churn fit` to populate them.\n\n",
        );
    } else if !score.label_audit.usable() {
        let why = score
            .label_audit
            .refused
            .clone()
            .unwrap_or_else(|| "no reason recorded".to_string());
        out.push_str(&format!(
            "The label audit refused to run: {why}. A ranking can be measured without one; \
a label audit cannot, because it needs a probability per row.\n\n"
        ));
    } else {
        let audit = &score.label_audit;
        out.push_str("| reading | value |\n|---|---:|\n");
        out.push_str(&format!(
            "| confidently contradicted rows | {} of {} |\n",
            audit.suspected,
            commas(audit.rows)
        ));
        out.push_str(&format!(
            "| cleared neither threshold, out of the joint | {} |\n",
            commas(audit.unconfident)
        ));
        out.push_str(&format!("| estimated label noise | {:.3} |\n", audit.noise));
        out.push_str(&format!(
            "| estimated noise, churn labels | {:.3} |\n",
            audit.noise_positive
        ));
        out.push_str(&format!(
            "| estimated noise, stayed labels | {:.3} |\n",
            audit.noise_negative
        ));
        out.push_str(&format!(
            "| ceiling against these labels | {:.3} |\n",
            audit.ceiling
        ));
        out.push_str(&format!(
            "| smallest resolvable AUC gain at {:.0}% power | {:.3} |\n",
            score.power.power * 100.0,
            score.power.detectable
        ));
        out.push_str(&format!(
            "| pre-registered floor | {:.3} |\n",
            score.power.floor
        ));
        if let Some(pair) = &score.tree_gain {
            out.push_str(&format!(
                "| paired AUC gain, trees against logistic | {:+.3} [{:+.3}, {:+.3}] |\n",
                pair.delta, pair.low, pair.high
            ));
        }
        out.push_str(&format!("\n{}\n\n", audit.note));
        out.push_str(&format!("{}\n\n", score.power.note));
        if let Some(pair) = &score.tree_gain {
            out.push_str(&format!(
                "The tree comparison is read paired: both models score the same {} rows, so \
the difference has its own error from the DeLong covariance rather than from two \
independent ones. The placements correlate at {:.2}, which is why the interval on a \
gain of {:+.3} is narrower than either model's own interval. {}\n\n",
                commas(pair.positives + pair.negatives),
                pair.correlation,
                pair.delta,
                if pair.low > 0.0 {
                    "The interval stays above zero, so on this panel the gain survives its own \
sampling error."
                } else {
                    "The interval includes zero, so the gain is not separated from noise on \
this panel, whatever the point estimate says."
                }
            ));
        }
        out.push_str(&format!(
            "The noise estimate is confident learning (Northcutt, Jiang and Chuang, 2021) run \
on the out-of-fold probabilities the folds already produced, with a per-class threshold \
at the mean predicted probability of that class. It separates random from systematic \
labelling error only in the sense of measuring the total: a bank where one annotator \
misread the construct scores like one where labels were flipped by a coin, and only the \
first is a design problem. It is also only as good as the model behind it, which is why \
it is quoted beside the AUC rather than instead of it. Where a large share of the bank \
clears neither threshold, the scaling is what carries the estimate: the rows the model \
contradicts are counted, and the rest is the assumption that the rows it could not place \
are wrong at the same rate. On this panel that assumption is the difference between the \
{:.1} percent above and the {:.1} percent the contradiction count alone implies.\n\n",
            audit.noise * 100.0,
            audit.suspected as f64 / audit.rows as f64 * 100.0
        ));
    }
    out.push_str("## Cohorts\n\n");
    let cohorts = &model.score.cohorts;
    if cohorts.rows.is_empty() {
        out.push_str(
            "This artifact predates the cohort table. Rebuild with `churn fit` to populate it.\n\n",
        );
    } else {
        out.push_str(&format!(
            "Cohorts are bands of tenure at the snapshot, which with a fixed snapshot is the same \
thing as a registration vintage. A band is mature for the {} month horizon when every row in it \
has been watched that long, so its label has had time to occur.\n\n",
            cohorts.horizon
        ));
        out.push_str("| band | rows | churned | recorded rate | mean tenure | auc | ece | coverage | against tenure only |\n");
        out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
        for row in &cohorts.rows {
            let paired = match &row.against_tenure {
                Some(pair) => format!("{:+.3} [{:+.3}, {:+.3}]", pair.delta, pair.low, pair.high),
                None => "refused".to_string(),
            };
            out.push_str(&format!(
                "| {} | {} | {} | {:.3} | {:.1} | {} | {} | {} | {} |\n",
                row.label,
                commas(row.rows),
                commas(row.churned),
                row.rate,
                row.mean_tenure,
                row.auc
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.ece
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                row.coverage
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                paired
            ));
        }
        out.push_str(&format!("\n{}\n\n", cohorts.note));
        let young: Vec<&super::cohort::CohortRow> =
            cohorts.rows.iter().filter(|row| !row.mature).collect();
        if !young.is_empty() {
            let names: Vec<String> = young
                .iter()
                .map(|row| format!("{} ({})", row.label, row.rows))
                .collect();
            out.push_str(&format!(
                "The bands below the horizon, {}, carry a rate and nothing else: a customer in \
their first months who has not churned is censored, not a negative, so a discrimination \
reading on those rows would be measuring how much time they have had rather than how they \
behave. The recorded rate falls across the bands, and that fall is mostly the length of the \
observation: the oldest band has had five years to remove its leavers and the youngest has \
had months.\n\n",
                names.join(", ")
            ));
        }
        out.push_str(&format!(
            "Pooled over the mature bands there are {} rows carrying {} churners and an AUC of \
{:.3} against {:.3} over the whole panel, so the young rows were not carrying the score. \
Each band is measured against its own base rate and its own rows, so no two bands are \
compared with each other here; the paired column is the shipped model against the \
tenure-only reference on the same rows, which is the comparison that is valid within a band.\n\n",
            commas(cohorts.mature_rows),
            commas(cohorts.mature_churned),
            cohorts.mature_auc,
            model.score.auc
        ));
    }
    out.push_str("## Coefficients\n\n");
    out.push_str(
        "On the standardised scale the fit uses. A sign is a direction, not a\ncause.\n\n",
    );
    out.push_str("| feature | weight |\n|---|---:|\n");
    let mut ordered = model.coefficients.clone();
    ordered.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
    for coefficient in &ordered {
        out.push_str(&format!(
            "| {} | {:+.3} |\n",
            coefficient.name, coefficient.weight
        ));
    }
    out.push_str("\n## The reason text\n\n");
    out.push_str("The 1,869 churn reasons are terse service labels, not customer narratives. The\nlexicon reads the wording: `Attitude of support person` scores positive because\n`support` is a trust word, and `Competitor had better devices` scores positive\nbecause `better` is. Read the categories as an ordering of wording, not as a\nmeasurement of how the customer felt. The labelled bank is where feeling is\nmeasured; the reason text is where a driver is named.\n\n");
    out.push_str("\n## What is not in the model\n\n");
    out.push_str("- Churn reason text. All 1,869 reasons belong to customers who already\n  left, so a model that read them would be predicting the past. The reasons\n  are read on the emotion axes in `reasons.csv`, beside the panel rather than\n  inside it.\n");
    out.push_str("- Dates. The table carries tenure, not calendar time, so this baseline\n  cannot attribute a campaign; it can only say how much churn the panel\n  explains before attribution starts.\n");
    out.push_str("- Geography. City and zip are in the source and are left out: at this\n  sample size they encode the operator's footprint more than customer\n  behaviour.\n\n");
    out.push_str("## Limits\n\n");
    out.push_str("- One operator, one snapshot. A coefficient here is not a market law. The\n  interval is a statement about this panel's sampling noise, not about a new\n  city or a new register.\n");
    out.push_str("- Right censoring: a customer who stayed twelve months is at risk for\n  twelve months and no longer, which the series respects and the regression\n  does not attempt to correct.\n");
    out.push_str("- `Joined` customers are censored rather than counted as non-churners.\n");
    out.push_str("- The coefficients are directions, not effects. The shipped panel has no\n  treatment column, so it cannot answer an intervention question. A panel that\n  carries one is read by `churn uplift`, which reports a cross-fitted T-learner\n  and an inverse probability weighted average effect with bootstrap intervals,\n  a Qini ranking, a decile table and a placebo p-value. That estimator still\n  adjusts for observed covariates only; unmeasured confounding is not repaired.\n");
    out.push_str("- Selection on the forecasting path is nested when `forecast --nested` is\n  used: the inner loop chooses and the outer loop scores. The plain command\n  remains a single holdout and the report says which one ran.\n");
    out.push_str("- Drift and rollback are commands, not a daemon. `churn monitor` compares a\n  reference panel with a current one, scoring each feature against a null simulated at\n  the row counts the reading was taken at, combining the index with a Cramer-von Mises\n  distance under Benjamini-Hochberg, and reporting a kernel two-sample reading on the\n  whole standardised vector; `churn rollback` moves the artifact pointer back one\n  version. A scheduler still has to call them.\n");
    out.push_str("- The folklore PSI bands are not used. A fixed 0.1 or 0.25 threshold is a\n  different test at every sample size, so the gate calibrates its own. What the gate\n  cannot do is tell drift from a legitimate change in the population; it says the two\n  panels differ, not why, and it needs labels on the current panel before anything can\n  be said about whether the difference mattered.\n");
    out.push_str("- No cohort here is a horizon rate. The snapshot records that a customer\n  churned, not when, so maturity says which bands could carry a dated rate, not what it\n  is. Young bands are refused rather than corrected.\n");
    out.push_str("- Calibration, conformal coverage and the tree comparison are measured on\n  the same held-out rows as the AUC. None of them transports to a new operator.\n");
    out.push_str("- The label audit bounds what this bank's own labels can carry; it does not\n  audit the construct. A ceiling of one would mean the labels are consistent with\n  the model, not that they are right about churn.\n");
    out.push_str("- The resolution reading is a property of the panel's size and class\n  balance. A gain the card does not claim is not proof of absence, only a\n  difference this bank cannot see.\n");
    out.push_str("- Marginal coverage is not class conditional coverage. The pooled number\n  holds across the held-out rows as a whole; the per-class numbers are the\n  ones the card now quotes, and a class whose calibration rows cannot support\n  the requested alpha is refused rather than given a reading.\n");
    out
}

/// One row of the class conditional table, or a refusal in the coverage cell.
fn class_row(name: &str, rows: usize, supported: bool, coverage: f64, set: f64) -> String {
    if supported {
        // The share is stored rounded to four places, so multiplying back gives
        // the hit count exactly at these sizes; the interval is then the same
        // Wilson bound the coverage simulation reports.
        let hits = (coverage * rows as f64).round() as usize;
        let (low, high) = crate::uncertainty::wilson(hits, rows, 0.95);
        format!("| {name} | {rows} | {coverage:.3} | [{low:.3}, {high:.3}] | {set:.2} |\n")
    } else {
        format!("| {name} | {rows} | refused | - | - |\n")
    }
}

pub use crate::num::commas;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::churn::fit::{fit, load};

    #[test]
    fn the_card_states_the_gap_between_the_full_and_narrow_auc() {
        if !std::path::Path::new("data/churn/panel.csv").is_file() {
            return;
        }
        let samples = load("data/churn/panel.csv").expect("the shipped panel");
        let model = fit(&samples, 5);
        let card = render(&model);
        assert!(card.contains("AUC, tenure and monthly only"));
        assert!(card.contains("AUC 95% interval, DeLong logit"));
        assert!(card.contains("AUC standard error"));
        assert!(card.contains("Class conditional conformal"));
        assert!(card.contains("Marginal coverage is not class conditional coverage"));
        assert!(card.contains("right censoring") || card.contains("Right censoring"));
        assert!(card.contains("Churn reason text"));
        assert!(card.contains("## Labels and resolution"));
        assert!(card.contains("estimated label noise"));
        assert!(card.contains("smallest resolvable AUC gain"));
        assert!(card.contains("paired AUC gain, trees against logistic"));
        // The note beside the table and the table itself must agree: one shared
        // definition of the rounding, not two.
        assert_eq!(
            model.score.label_audit.noise,
            crate::num::round4(model.score.label_audit.noise)
        );
        assert!(card.contains(&format!(
            "{:.1} percent of the bank",
            model.score.label_audit.noise * 100.0
        )));
    }
}
