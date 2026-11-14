//! A retention baseline over the panel, and the AUC of reading tenure alone.
//!
//! This is not a churn competition entry. It is the reference an MMM team needs
//! before crediting a campaign with retention: how much of the churn in this
//! panel a five-feature logistic already explains, measured out of sample. Ten
//! features is what the table supports without a category explosion, and the
//! model is the same fixed step logistic the detectors use, so the fit is
//! deterministic and needs no dependency.
//!
//! Folds are assigned by a digest of the customer id, so a customer is in the
//! same fold on every machine and a rebuild reproduces the metric exactly. The
//! number that matters is `auc`; `tenure_auc` is the same score for tenure and
//! monthly charge only, and the gap between them is what the rest of the table
//! buys.

use std::collections::BTreeMap;
use std::fs;
use std::io;

use serde::{Deserialize, Serialize};

use super::csvx;
use crate::dataset::built_at;
use crate::detect::eval::{best_cut, evaluate, Metrics};
use crate::detect::logistic::{FitParams, Logistic};
use crate::digest::fnv64_str;

pub const MODEL: &str = "model.json";
pub const CARD: &str = "card.md";
pub const FOLDS: usize = 5;

/// One customer as the regression sees them.
pub struct Sample {
    pub id: String,
    pub churn: bool,
    pub values: Vec<f64>,
}

/// The feature names, in the order the values are built.
pub fn feature_names() -> Vec<String> {
    [
        "tenure",
        "monthly",
        "total",
        "age",
        "dependents",
        "referrals",
        "support",
        "security",
        "one_year",
        "two_year",
        "fiber",
        "dsl",
        "credit_card",
        "bank_withdrawal",
    ]
    .iter()
    .map(|name| name.to_string())
    .collect()
}

/// Read the panel and build the design matrix.
pub fn load(path: &str) -> io::Result<Vec<Sample>> {
    let table = csvx::read(path)?;
    let id = table.require("id")?;
    let churn = table.require("churn")?;
    let tenure = table.require("tenure")?;
    let monthly = table.require("monthly")?;
    let total = table.require("total")?;
    let age = table.require("age")?;
    let dependents = table.require("dependents")?;
    let referrals = table.require("referrals")?;
    let support = table.require("support")?;
    let security = table.require("security")?;
    let contract = table.require("contract")?;
    let internet = table.require("internet")?;
    let payment = table.require("payment")?;
    let mut out = Vec::with_capacity(table.rows.len());
    for (number, cells) in table.rows.iter().enumerate() {
        let at = number + 2;
        let value = |column: usize, name: &str| -> io::Result<f64> {
            cells[column].trim().parse::<f64>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("row {at} has a non numeric {name}: {:?}", cells[column]),
                )
            })
        };
        let contract = cells[contract].to_lowercase();
        let internet = cells[internet].to_lowercase();
        let payment = cells[payment].to_lowercase();
        let values = vec![
            value(tenure, "tenure")?,
            value(monthly, "monthly")?,
            value(total, "total")?,
            value(age, "age")?,
            value(dependents, "dependents")?,
            value(referrals, "referrals")?,
            if cells[support] == "1" { 1.0 } else { 0.0 },
            if cells[security] == "1" { 1.0 } else { 0.0 },
            f64::from(u8::from(contract.contains("one year"))),
            f64::from(u8::from(contract.contains("two year"))),
            f64::from(u8::from(internet.contains("fiber"))),
            f64::from(u8::from(internet.contains("dsl"))),
            f64::from(u8::from(payment.contains("credit card"))),
            f64::from(u8::from(payment.contains("bank"))),
        ];
        out.push(Sample {
            id: cells[id].clone(),
            churn: cells[churn].trim() == "1",
            values,
        });
    }
    if out.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path} holds no rows"),
        ));
    }
    Ok(out)
}

/// The fold a customer belongs to, from the id alone.
pub fn fold(id: &str, folds: usize) -> usize {
    let digest = fnv64_str(id);
    usize::from_str_radix(&digest[..8], 16).unwrap_or(0) % folds.max(1)
}

/// What the held-out folds said.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Score {
    pub rows: usize,
    pub positives: usize,
    pub auc: f64,
    pub balanced: f64,
    pub accuracy: f64,
    pub cut: f64,
    pub tenure_auc: f64,
    pub folds: usize,
    pub per_fold: Vec<f64>,
}

/// One coefficient, on the standardised scale the fit uses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Coefficient {
    pub name: String,
    pub weight: f64,
}

/// The fitted artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChurnModel {
    pub name: String,
    pub version: u32,
    pub built_at: String,
    pub features: Vec<String>,
    pub model: Logistic,
    pub coefficients: Vec<Coefficient>,
    pub score: Score,
    pub metrics: Metrics,
}

/// Fit the folds, choose a cut on the training side, and report the held-out score.
pub fn fit(samples: &[Sample], folds: usize) -> ChurnModel {
    let folds = folds.max(2);
    let names = feature_names();
    let labels: Vec<bool> = samples.iter().map(|s| s.churn).collect();
    let mut scores = vec![0.0; samples.len()];
    let mut narrow_scores = vec![0.0; samples.len()];
    let mut flags = vec![false; samples.len()];
    let mut cuts = Vec::new();
    let mut per_fold = Vec::new();
    for fold_at in 0..folds {
        let (train, test): (Vec<usize>, Vec<usize>) =
            (0..samples.len()).partition(|at| fold(&samples[*at].id, folds) != fold_at);
        if train.is_empty() || test.is_empty() {
            continue;
        }
        let rows: Vec<Vec<f64>> = train.iter().map(|at| samples[*at].values.clone()).collect();
        let train_labels: Vec<bool> = train.iter().map(|at| samples[*at].churn).collect();
        let model = Logistic::fit(&rows, &train_labels, &FitParams::default());
        let train_scores: Vec<f64> = rows.iter().map(|row| model.predict(row)).collect();
        let cut = best_cut(&train_scores, &train_labels);
        cuts.push(cut);
        let narrow_rows: Vec<Vec<f64>> = rows.iter().map(|row| vec![row[0], row[1]]).collect();
        let narrow = Logistic::fit(&narrow_rows, &train_labels, &FitParams::default());
        let fold_scores: Vec<f64> = test
            .iter()
            .map(|at| model.predict(&samples[*at].values))
            .collect();
        let fold_labels: Vec<bool> = test.iter().map(|at| samples[*at].churn).collect();
        per_fold.push(auc(&fold_scores, &fold_labels));
        for (score, at) in fold_scores.into_iter().zip(&test) {
            scores[*at] = score;
            flags[*at] = score >= cut;
        }
        for at in test {
            let values = &samples[at].values;
            narrow_scores[at] = narrow.predict(&[values[0], values[1]]);
        }
    }
    let predicted: Vec<bool> = scores.iter().map(|score| *score >= 0.5).collect();
    let metrics = evaluate(&labels, &predicted);
    let cut = cuts.iter().sum::<f64>() / cuts.len().max(1) as f64;
    let calibrated = evaluate(&labels, &flags);
    let rows_all: Vec<Vec<f64>> = samples.iter().map(|s| s.values.clone()).collect();
    let full = Logistic::fit(&rows_all, &labels, &FitParams::default());
    let coefficients: Vec<Coefficient> = names
        .iter()
        .enumerate()
        .map(|(at, name)| Coefficient {
            name: name.clone(),
            weight: round4(full.coefficients.get(at).copied().unwrap_or(0.0)),
        })
        .collect();
    let tenure_auc = auc(&narrow_scores, &labels);
    ChurnModel {
        name: "xcm-mmm-churn-baseline".to_string(),
        version: 1,
        built_at: built_at(),
        features: names,
        model: full,
        coefficients,
        score: Score {
            rows: samples.len(),
            positives: labels.iter().filter(|label| **label).count(),
            auc: round4(auc(&scores, &labels)),
            balanced: calibrated.balanced,
            accuracy: calibrated.accuracy,
            cut: round4(cut),
            tenure_auc: round4(tenure_auc),
            folds,
            per_fold: per_fold.iter().map(|value| round4(*value)).collect(),
        },
        metrics,
    }
}

/// Area under the ROC curve, ties counted at half credit.
pub fn auc(scores: &[f64], labels: &[bool]) -> f64 {
    if scores.len() != labels.len() || scores.is_empty() {
        return 0.5;
    }
    let positives = labels.iter().filter(|label| **label).count();
    let negatives = labels.len() - positives;
    if positives == 0 || negatives == 0 {
        return 0.5;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| scores[*a].total_cmp(&scores[*b]));
    let mut ranks = vec![0.0; scores.len()];
    let mut at = 0;
    while at < order.len() {
        let mut end = at;
        while end + 1 < order.len() && scores[order[end + 1]] == scores[order[at]] {
            end += 1;
        }
        let rank = (at + end + 2) as f64 / 2.0;
        for index in &order[at..=end] {
            ranks[*index] = rank;
        }
        at = end + 1;
    }
    let positive_ranks: f64 = labels
        .iter()
        .zip(&ranks)
        .filter(|(label, _)| **label)
        .map(|(_, rank)| *rank)
        .sum();
    (positive_ranks - (positives * (positives + 1)) as f64 / 2.0) / (positives * negatives) as f64
}

/// Fit the panel and write the artifact and its card.
pub fn build(dir: &str) -> io::Result<ChurnModel> {
    build_with(dir, FOLDS)
}

/// [`build`] with a fold count, for a caller that wants to test the split.
pub fn build_with(dir: &str, folds: usize) -> io::Result<ChurnModel> {
    let samples = load(&format!("{dir}/panel.csv"))?;
    let model = fit(&samples, folds);
    let text = serde_json::to_string_pretty(&model).map_err(io::Error::other)?;
    fs::write(format!("{dir}/{MODEL}"), format!("{text}\n"))?;
    fs::write(format!("{dir}/{CARD}"), super::card::render(&model))?;
    Ok(model)
}

/// Load a fitted artifact.
pub fn read(path: &str) -> io::Result<ChurnModel> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{path}: {e}")))
}

/// Coefficient weights by name, for a report.
pub fn weights(model: &ChurnModel) -> BTreeMap<String, f64> {
    model
        .coefficients
        .iter()
        .map(|c| (c.name.clone(), c.weight))
        .collect()
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: &str, churn: bool, tenure: f64) -> Sample {
        let mut values = vec![0.0; feature_names().len()];
        values[0] = tenure;
        values[1] = 60.0;
        Sample {
            id: id.to_string(),
            churn,
            values,
        }
    }

    #[test]
    fn auc_is_one_for_a_perfect_order_and_half_for_a_constant() {
        assert!((auc(&[0.9, 0.8, 0.2, 0.1], &[true, true, false, false]) - 1.0).abs() < 1e-9);
        assert!((auc(&[0.5, 0.5, 0.5], &[true, false, true]) - 0.5).abs() < 1e-9);
        assert!((auc(&[], &[]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_tie_is_half_credit() {
        assert!((auc(&[0.5, 0.5], &[true, false]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn folds_are_stable_and_in_range() {
        for id in ["1-AAA", "2-BBB", "abc-123"] {
            let at = fold(id, 5);
            assert!(at < 5);
            assert_eq!(at, fold(id, 5));
        }
    }

    #[test]
    fn a_fit_reports_more_than_the_base_rate() {
        let samples: Vec<Sample> = (0..60)
            .map(|at| sample(&format!("id-{at}"), at >= 30, at as f64))
            .collect();
        let model = fit(&samples, 5);
        assert_eq!(model.score.rows, 60);
        assert!(model.score.auc > 0.5);
        assert_eq!(model.features.len(), feature_names().len());
        assert_eq!(model.coefficients.len(), feature_names().len());
    }
}
