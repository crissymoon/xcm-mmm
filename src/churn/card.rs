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
        out.push_str("| interval | measured coverage | mean width | inside the band |\n");
        out.push_str("|---|---:|---:|:---:|\n");
        for entry in &score.coverage.methods {
            out.push_str(&format!(
                "| {} | {:.3} | {:.3} | {} |\n",
                entry.method,
                entry.covered,
                entry.mean_width,
                if entry.within_band { "yes" } else { "no" }
            ));
        }
        out.push('\n');
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
        out.push_str("| class | calibration rows | coverage | mean set size |\n");
        out.push_str("|---|---:|---:|---:|\n");
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
            "\nThe pooled reading is {:.3} coverage at a mean set of {:.2}. It is a\nmarginal number, and it hides a class whose calibration rows are too few for\nthe requested tail.\n\n",
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
    out.push_str("- Drift and rollback are commands, not a daemon. `churn monitor` compares a\n  reference panel with a current one by PSI and KS; `churn rollback` moves the\n  artifact pointer back one version. A scheduler still has to call them.\n");
    out.push_str("- Calibration, conformal coverage and the tree comparison are measured on\n  the same held-out rows as the AUC. None of them transports to a new operator.\n");
    out.push_str("- Marginal coverage is not class conditional coverage. The pooled number\n  holds across the held-out rows as a whole; the per-class numbers are the\n  ones the card now quotes, and a class whose calibration rows cannot support\n  the requested alpha is refused rather than given a reading.\n");
    out
}

/// One row of the class conditional table, or a refusal in the coverage cell.
fn class_row(name: &str, rows: usize, supported: bool, coverage: f64, set: f64) -> String {
    if supported {
        format!("| {name} | {rows} | {coverage:.3} | {set:.2} |\n")
    } else {
        format!("| {name} | {rows} | refused | - |\n")
    }
}

fn commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

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
    }
}
