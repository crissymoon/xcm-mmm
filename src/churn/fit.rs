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
use crate::audit::{detectable_auc, AuditConfig, LabelAudit, Power};
use crate::data::Dataset;
use crate::dataset::built_at;
use crate::detect::eval::{best_cut, evaluate, Metrics};
use crate::detect::logistic::{FitParams, Logistic};
use crate::digest::fnv64_str;
use crate::learn::{Learner, TreeConfig};
use crate::rng::Rng;
use crate::support::{self, LearningPoint};
use crate::uncertainty::{
    bootstrap_auc, delong_auc, delong_pair, mondrian, MondrianConformal, PairedAuc,
};

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
#[serde(default)]
pub struct Score {
    pub rows: usize,
    pub positives: usize,
    pub auc: f64,
    pub auc_low: f64,
    pub auc_high: f64,
    pub auc_se: f64,
    pub auc_logit_low: f64,
    pub auc_logit_high: f64,
    pub balanced: f64,
    pub accuracy: f64,
    pub cut: f64,
    pub tenure_auc: f64,
    pub folds: usize,
    pub per_fold: Vec<f64>,
    pub brier: f64,
    pub log_loss: f64,
    pub ece: f64,
    /// Percentile interval on the expected calibration error.
    pub ece_low: f64,
    /// Upper end of that interval.
    pub ece_high: f64,
    /// Percentile interval on the Brier score.
    pub brier_low: f64,
    /// Upper end of that interval.
    pub brier_high: f64,
    /// Reliability after the beta map, in sample.
    pub calibrated_ece: f64,
    /// The fitted beta map, with the note that describes what it did.
    pub calibration_map: crate::uncertainty::BetaCalibration,
    /// Plain reading of the calibration table.
    pub calibration_note: String,
    /// Bias corrected and accelerated interval for AUC.
    pub auc_bca_low: f64,
    /// Upper end of the BCa interval.
    pub auc_bca_high: f64,
    /// Bias correction the BCa interval used.
    pub auc_bca_bias: f64,
    /// Acceleration the BCa interval used.
    pub auc_bca_acceleration: f64,
    /// Coverage simulation for the three intervals at the challenged shape.
    pub coverage: crate::uncertainty::CoverageReport,
    pub calibration: Vec<crate::uncertainty::CalibrationBin>,
    pub conformal_alpha: f64,
    pub conformal_threshold: f64,
    pub conformal_coverage: f64,
    pub conformal_set: f64,
    pub conformal_pos_rows: usize,
    pub conformal_pos_coverage: f64,
    pub conformal_pos_set: f64,
    pub conformal_pos_supported: bool,
    pub conformal_neg_rows: usize,
    pub conformal_neg_coverage: f64,
    pub conformal_neg_set: f64,
    pub conformal_neg_supported: bool,
    pub conformal_refused: Option<String>,
    pub tree_auc: f64,
    pub tree_auc_low: f64,
    pub tree_auc_high: f64,
    pub events_per_feature: f64,
    pub learning: Vec<LearningPoint>,
    /// How much of the bank's labelling is wrong, estimated from the same
    /// out-of-fold probabilities the AUC came from.
    pub label_audit: LabelAudit,
    /// The smallest AUC gain this panel resolves, beside the project floor.
    pub power: Power,
    /// The tree against the logistic, paired on the same rows. Absent when the
    /// two score vectors do not line up.
    pub tree_gain: Option<PairedAuc>,
    /// Every reading split by how long its outcome has been watched.
    pub cohorts: super::cohort::CohortTable,
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
    /// The observation window the panel was measured over. Absent on an
    /// artifact built before the contract existed, which reads as "unknown"
    /// rather than as a failure.
    #[serde(default)]
    pub window: Option<crate::churn::contract::TimeZero>,
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
    let cut_metrics = evaluate(&labels, &flags);
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
    let positives = labels.iter().filter(|label| **label).count();
    let learning = learning_curve(samples);
    let interval = bootstrap_auc(&scores, &labels, 400, 0.95, 0x5EED);
    let delong = delong_auc(&scores, &labels, 0.95);
    let accelerated = crate::uncertainty::bca(&scores, &labels, 400, 0.95, 0x5EED, 20, auc);
    let reliability =
        crate::uncertainty::calibration_report(&scores, &labels, 10, 200, 0.95, 0xCA1B);
    let coverage = crate::uncertainty::simulate_coverage(&coverage_config());
    let class_report = mondrian(&scores, &labels, 0.10);
    let pooled = &class_report.pooled;
    let churn_class = class_reading(&class_report, true);
    let stayed_class = class_reading(&class_report, false);
    let tree_scores = tree_baseline(samples, folds);
    let tree_interval = bootstrap_auc(&tree_scores, &labels, 200, 0.95, 0x7A11);
    let ids: Vec<String> = samples.iter().map(|sample| sample.id.clone()).collect();
    let label_audit = crate::audit::audit(&ids, &scores, &labels, &AuditConfig::default());
    let power = detectable_auc(
        auc(&scores, &labels),
        positives,
        samples.len() - positives,
        AuditConfig::default().floor,
        AuditConfig::default().power,
        AuditConfig::default().level,
    );
    let tree_gain = delong_pair(&tree_scores, &scores, &labels, 0.95);
    // The cohort table reads the same held-out scores, so a per-band reading
    // cannot disagree with the headline for any reason except the split itself.
    let cohorts = super::cohort::table(
        samples,
        &scores,
        &narrow_scores,
        class_report.pooled.threshold,
        super::cohort::HORIZON_MONTHS,
    );
    ChurnModel {
        name: "xcm-mmm-churn-baseline".to_string(),
        version: 3,
        built_at: built_at(),
        features: names.clone(),
        model: full,
        coefficients,
        score: Score {
            rows: samples.len(),
            positives,
            auc: round4(auc(&scores, &labels)),
            auc_low: round4(interval.low),
            auc_high: round4(interval.high),
            auc_se: round4(delong.se),
            auc_logit_low: round4(delong.low),
            auc_logit_high: round4(delong.high),
            balanced: cut_metrics.balanced,
            accuracy: cut_metrics.accuracy,
            cut: round4(cut),
            tenure_auc: round4(tenure_auc),
            folds,
            per_fold: per_fold.iter().map(|value| round4(*value)).collect(),
            brier: round4(reliability.raw.brier),
            log_loss: round4(reliability.raw.log_loss),
            ece: round4(reliability.raw.ece),
            ece_low: round4(reliability.ece.low),
            ece_high: round4(reliability.ece.high),
            brier_low: round4(reliability.brier.low),
            brier_high: round4(reliability.brier.high),
            calibrated_ece: round4(reliability.calibrated.ece),
            calibration_map: reliability.map.clone(),
            calibration_note: reliability.note.clone(),
            auc_bca_low: round4(accelerated.interval.low),
            auc_bca_high: round4(accelerated.interval.high),
            auc_bca_bias: round4(accelerated.bias),
            auc_bca_acceleration: round4(accelerated.acceleration),
            coverage,
            calibration: reliability.raw.bins.clone(),
            conformal_alpha: round4(pooled.alpha),
            conformal_threshold: round4(pooled.threshold),
            conformal_coverage: round4(pooled.coverage),
            conformal_set: round4(pooled.mean_set),
            conformal_pos_rows: churn_class.0,
            conformal_pos_coverage: round4(churn_class.1),
            conformal_pos_set: round4(churn_class.2),
            conformal_pos_supported: churn_class.3,
            conformal_neg_rows: stayed_class.0,
            conformal_neg_coverage: round4(stayed_class.1),
            conformal_neg_set: round4(stayed_class.2),
            conformal_neg_supported: stayed_class.3,
            conformal_refused: class_report.refused.clone(),
            tree_auc: round4(tree_interval.point),
            tree_auc_low: round4(tree_interval.low),
            tree_auc_high: round4(tree_interval.high),
            events_per_feature: round4(support::events_per_feature(positives, names.len())),
            learning,
            label_audit,
            power,
            tree_gain,
            cohorts,
        },
        metrics,
        window: None,
    }
}

/// The coverage simulation, run at the shape the intervals were challenged at.
///
/// Coverage is a property of the estimator at a bank shape, not of this panel,
/// so it is measured at 251 rows with 50 positives, where an asymptotic variance
/// is defensible and a failure is visible. Rerunning it at seven thousand rows
/// would cost minutes and answer the same question less sharply: every interval
/// covers at that size, which is why nobody quotes a small-bank result.
fn coverage_config() -> crate::uncertainty::CoverageConfig {
    crate::uncertainty::CoverageConfig::default()
}

/// The salt that gives the learning curve a split of its own.
///
/// A curve drawn on the folds that chose the model is not a second
/// measurement: its largest point is the headline itself, scored on the rows
/// the model was selected against. Salting the id partitions the panel again,
/// so the curve and the headline can disagree for sampling reasons instead of
/// agreeing by construction.
const CURVE_SALT: &str = "#curve";
const CURVE_FOLDS: usize = 4;

fn curve_fold(id: &str) -> usize {
    fold(&format!("{id}{CURVE_SALT}"), CURVE_FOLDS)
}

/// Held-out AUC and accuracy at six training fractions, on a split of their own.
///
/// A curve still rising at the largest fraction says the bank is limited by
/// rows. A flat curve below the label audit's ceiling says the ceiling is the
/// features or the operator. The subsets are drawn deterministically from the
/// curve's training side, so the diagnostic rebuilds with the artifact. The
/// accuracy column is the one the ceiling can be compared with; comparing the
/// area with an accuracy would be a scale error.
fn learning_curve(samples: &[Sample]) -> Vec<LearningPoint> {
    let mut out = Vec::with_capacity(support::FRACTIONS.len());
    for (index, fraction) in support::FRACTIONS.iter().enumerate() {
        let mut aucs = Vec::new();
        let mut accuracies = Vec::new();
        let mut rows = 0usize;
        let mut tested = 0usize;
        for fold_at in 0..CURVE_FOLDS {
            let train: Vec<usize> = (0..samples.len())
                .filter(|at| curve_fold(&samples[*at].id) != fold_at)
                .collect();
            let test: Vec<usize> = (0..samples.len())
                .filter(|at| curve_fold(&samples[*at].id) == fold_at)
                .collect();
            if train.is_empty() || test.is_empty() {
                continue;
            }
            let mut rng = Rng::new(0x1EAF_u64.wrapping_add(index as u64 * 31 + fold_at as u64));
            let mut order = train.clone();
            for i in (1..order.len()).rev() {
                let j = rng.below(i + 1);
                order.swap(i, j);
            }
            let keep = ((order.len() as f64 * fraction).ceil() as usize)
                .max(10)
                .min(order.len());
            let chosen = &order[..keep];
            let labels: Vec<bool> = chosen.iter().map(|at| samples[*at].churn).collect();
            if labels.iter().all(|label| *label) || labels.iter().all(|label| !*label) {
                continue;
            }
            let rows_fit: Vec<Vec<f64>> = chosen
                .iter()
                .map(|at| samples[*at].values.clone())
                .collect();
            let model = Logistic::fit(&rows_fit, &labels, &FitParams::default());
            let test_labels: Vec<bool> = test.iter().map(|at| samples[*at].churn).collect();
            if test_labels.iter().all(|label| *label) || test_labels.iter().all(|label| !*label) {
                continue;
            }
            let train_scores: Vec<f64> = rows_fit.iter().map(|row| model.predict(row)).collect();
            let cut = best_cut(&train_scores, &labels);
            let test_scores: Vec<f64> = test
                .iter()
                .map(|at| model.predict(&samples[*at].values))
                .collect();
            aucs.push(auc(&test_scores, &test_labels));
            let right = test_scores
                .iter()
                .zip(&test_labels)
                .filter(|(score, label)| (**score >= cut) == **label)
                .count();
            accuracies.push(right as f64 / test_scores.len().max(1) as f64);
            rows += chosen.len();
            tested += test_scores.len();
        }
        let (mean_auc, se_auc) = mean_and_se(&aucs);
        let (mean_accuracy, se_accuracy) = mean_and_se(&accuracies);
        // The fold spread says how sensitive the accuracy is to the split. The
        // binomial error says how precisely it is measured on the rows it was
        // measured on. A ceiling comparison has to clear the wider of the two.
        let binomial = (mean_accuracy * (1.0 - mean_accuracy) / tested.max(1) as f64).sqrt();
        out.push(LearningPoint {
            fraction: *fraction,
            rows: rows.checked_div(aucs.len()).unwrap_or(0),
            auc: round4(mean_auc),
            se: round4(se_auc),
            accuracy: round4(mean_accuracy),
            accuracy_se: round4(se_accuracy.max(binomial)),
        });
    }
    out
}

/// Mean and standard error across folds.
///
/// One fold reads `(value, 0)` rather than falling back to zero width, because
/// a single fold is not evidence of precision.
fn mean_and_se(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.5, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let variance = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, (variance / values.len() as f64).sqrt())
}

/// A gradient boosted tree baseline on the same folds and features.
///
/// The review's claim is that a tree ensemble often beats a linear model on
/// medium tabular data. This is how the crate tests it on the panel instead of
/// asserting it: same rows, same fold assignment, same target, a different
/// hypothesis class. Predictions are probabilities because the target is 0/1,
/// so the raw ensemble output is clipped to `[0, 1]` before ranking.
fn tree_baseline(samples: &[Sample], folds: usize) -> Vec<f64> {
    let names = feature_names();
    let all: Vec<usize> = (0..names.len()).collect();
    let learner = Learner::Trees(TreeConfig {
        estimators: 32,
        depth: 3,
        min_leaf: 20,
        learning_rate: 0.1,
    });
    let mut scores = vec![0.5; samples.len()];
    for fold_at in 0..folds {
        let train: Vec<usize> = (0..samples.len())
            .filter(|at| fold(&samples[*at].id, folds) != fold_at)
            .collect();
        let test: Vec<usize> = (0..samples.len())
            .filter(|at| fold(&samples[*at].id, folds) == fold_at)
            .collect();
        if train.len() < 20 || test.is_empty() {
            continue;
        }
        let features: Vec<Vec<f64>> = train.iter().map(|at| samples[*at].values.clone()).collect();
        let target: Vec<f64> = train
            .iter()
            .map(|at| f64::from(u8::from(samples[*at].churn)))
            .collect();
        let Ok(dataset) = Dataset::new(names.clone(), features, target) else {
            continue;
        };
        let Some(model) = learner.fit(&dataset, 0..dataset.len(), &all) else {
            continue;
        };
        for at in test {
            scores[at] = model.predict(&samples[at].values).clamp(0.0, 1.0);
        }
    }
    scores
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
    let mut model = fit(&samples, folds);
    model.window = super::contract::check_files(dir).ok();
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

/// Calibration rows, coverage, mean set size and support for one class.
///
/// A report with no rows for the class, which an artifact built before the
/// class conditional split carries, reads as unsupported rather than as zero
/// coverage.
fn class_reading(report: &MondrianConformal, label: bool) -> (usize, f64, f64, bool) {
    report
        .classes
        .iter()
        .find(|class| class.label == label)
        .map(|class| {
            (
                class.calibration_rows,
                class.coverage,
                class.mean_set,
                class.supported,
            )
        })
        .unwrap_or((0, 0.0, 0.0, false))
}

use crate::num::round4;

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
    fn an_older_artifact_without_the_new_fields_still_reads() {
        let samples: Vec<Sample> = (0..20)
            .map(|at| sample(&format!("id-{at}"), at % 2 == 0, at as f64))
            .collect();
        let model = fit(&samples, 2);
        let mut value = serde_json::to_value(&model).expect("json");
        let score = value["score"].as_object_mut().expect("a score object");
        for key in [
            "auc_se",
            "auc_logit_low",
            "auc_logit_high",
            "conformal_pos_rows",
            "conformal_pos_coverage",
            "conformal_pos_set",
            "conformal_pos_supported",
            "conformal_neg_rows",
            "conformal_neg_coverage",
            "conformal_neg_set",
            "conformal_neg_supported",
            "conformal_refused",
        ] {
            score.remove(key);
        }
        let older: ChurnModel = serde_json::from_value(value).expect("an older artifact");
        assert_eq!(older.score.auc, model.score.auc);
        assert_eq!(older.score.auc_se, 0.0);
        assert!(!older.score.conformal_pos_supported);
        assert!(older.score.conformal_refused.is_none());
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
        // Interleaved ids so the positional conformal split finds both classes
        // in its calibration half; churn follows tenure, so the panel is easy.
        let samples: Vec<Sample> = (0..60)
            .map(|index| {
                let at = if index % 2 == 0 {
                    30 + index / 2
                } else {
                    index / 2
                };
                sample(&format!("id-{at}"), at >= 30, at as f64)
            })
            .collect();
        let model = fit(&samples, 5);
        assert_eq!(model.score.rows, 60);
        assert!(model.score.auc > 0.5);
        assert_eq!(model.features.len(), feature_names().len());
        assert_eq!(model.coefficients.len(), feature_names().len());
        // The synthetic panel is separated by tenure, so the ranking is perfect
        // and the DeLong standard error is zero; the interval still stays
        // inside the unit interval through the logit transform.
        assert!(model.score.auc_se >= 0.0);
        assert!(model.score.auc_logit_low > 0.0 && model.score.auc_logit_high <= 1.0);
        assert!(model.score.conformal_pos_rows > 0);
        assert!(model.score.conformal_neg_rows > 0);
        assert!(model.score.conformal_pos_supported);
        assert!(model.score.conformal_refused.is_none());
        assert_eq!(
            model.score.conformal_pos_rows + model.score.conformal_neg_rows,
            model.score.rows / 2
        );
    }
}
