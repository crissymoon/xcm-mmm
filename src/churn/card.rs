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
    out.push_str(&format!("| folds | {} |\n\n", score.folds));
    out.push_str("Fold AUC: ");
    let per_fold: Vec<String> = score.per_fold.iter().map(|v| format!("{v:.3}")).collect();
    out.push_str(&format!("{}\n\n", per_fold.join(", ")));
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
    out.push_str("- One operator, one snapshot. A coefficient here is not a market law.\n");
    out.push_str("- Right censoring: a customer who stayed twelve months is at risk for\n  twelve months and no longer, which the series respects and the regression\n  does not attempt to correct.\n");
    out.push_str("- `Joined` customers are censored rather than counted as non-churners.\n");
    out
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
        assert!(card.contains("right censoring") || card.contains("Right censoring"));
        assert!(card.contains("Churn reason text"));
    }
}
