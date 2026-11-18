//! Confident learning: estimate how many labels are wrong, without a clean set.
//!
//! The estimator (Northcutt, Jiang and Chuang, *JAIR* 70, 2021) needs one thing
//! a fold loop already produces: an out-of-fold probability for every row. It
//! then counts only the rows the model is confident about, turns those counts
//! into an estimate of the joint distribution of given and true labels, and
//! reads the off diagonal as label noise.
//!
//! Three decisions are worth stating because they shape the number.
//!
//! The threshold for a class is the mean predicted probability of that class
//! over the rows given that label. This is the paper's per-class threshold and
//! it makes "confident" a property of the bank rather than a constant chosen
//! here: a bank the model finds easy raises its own bar.
//!
//! A row where no class clears its threshold is left out of the confident joint.
//! The reference implementation sends such a row to the negative class, which is
//! an artefact of taking the argmax of a boolean mask. Leaving it out and then
//! scaling the class totals back up to the labelled counts is the construction
//! in the paper, and it is the one that makes the off diagonal an estimate
//! rather than a count of rows the model happened to be sure about. A class
//! where every row is unconfident has nothing to scale and is refused.
//!
//! That scaling is also the bound: a model confidently wrong about every row of
//! a class cannot push that class's noise past its own size, because the class
//! total is fixed at the number of labels of that class.
//!
//! What it cannot do is separate random from systematic labelling error. A bank
//! where one annotator misunderstood the construct produces the same noise rate
//! as one where labels were flipped by a coin, and only the first is a design
//! problem. The ceiling is therefore a bound on accuracy against the recorded
//! labels, not against the construct.
//!
//! It is also only as good as the model that produced the probabilities. With
//! every row scored the same, the confident joint puts every row in one class,
//! the noise of the other class reads one and the ceiling collapses to the class
//! share. That is the correct answer to the question asked, since an
//! uninformative model supplies no evidence about any label, and it is why the
//! audit belongs beside the AUC rather than instead of it.

use serde::{Deserialize, Serialize};

use crate::num::round4;

/// Slack on a threshold comparison.
///
/// The threshold is the mean of the same probabilities it is compared against,
/// so rows land exactly on it by construction whenever a model emits few
/// distinct values. Summing several hundred of them moves the mean by an ulp,
/// and without this slack a row that is meant to be confident is excluded by the
/// last bit of its own sum.
const TIE: f64 = 1e-12;

/// Which class a row belongs to, in the order the counts are indexed.
pub const NEGATIVE: usize = 0;
/// The other class.
pub const POSITIVE: usize = 1;

/// How much of the audit to print, and the floor a gain has to clear.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Suspect rows to keep, worst margin first.
    pub top: usize,
    /// Level the power reading is quoted at.
    pub level: f64,
    /// Smallest gain worth claiming, pre-registered before the fit.
    pub floor: f64,
    /// Power the smallest detectable gain is quoted at.
    pub power: f64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            top: 8,
            level: 0.95,
            floor: 0.03,
            power: 0.80,
        }
    }
}

/// One row the model reads against its label.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Suspect {
    /// Row identifier, or `row n` when the caller supplied none.
    pub id: String,
    /// The label the bank gives the row.
    pub label: bool,
    /// Out-of-fold probability of the positive class.
    pub probability: f64,
    /// Probability of the given label minus the best rival, negative when the
    /// model prefers the other class.
    pub margin: f64,
}

/// What the labels are worth, as far as the model can tell.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LabelAudit {
    /// Rows audited.
    pub rows: usize,
    /// Rows labelled positive.
    pub positives: usize,
    /// Rows labelled negative.
    pub negatives: usize,
    /// Threshold the positive class had to clear.
    pub threshold_positive: f64,
    /// Threshold the negative class had to clear.
    pub threshold_negative: f64,
    /// Confident counts, indexed `[given][confident]`.
    pub confident: Vec<Vec<usize>>,
    /// Estimated joint counts after calibrating each row to the labelled total.
    pub joint: Vec<Vec<f64>>,
    /// Estimated share of positive labels that are wrong.
    pub noise_positive: f64,
    /// Estimated share of negative labels that are wrong.
    pub noise_negative: f64,
    /// Estimated share of all labels that are wrong.
    pub noise: f64,
    /// Best accuracy the bank can support against its own labels.
    pub ceiling: f64,
    /// Rows the model confidently contradicts.
    pub suspected: usize,
    /// Rows that cleared neither threshold and are out of the joint.
    pub unconfident: usize,
    /// Off diagonal count, rounded, as a whole number of rows.
    pub estimated_errors: usize,
    /// Worst margins, most negative first.
    pub suspects: Vec<Suspect>,
    /// Plain reading of the table.
    pub note: String,
    /// Why the audit did not run, when it did not.
    pub refused: Option<String>,
}

impl LabelAudit {
    /// An audit that could not be estimated, carrying the reason.
    pub fn refused(rows: usize, reason: impl Into<String>) -> Self {
        Self {
            rows,
            positives: 0,
            negatives: 0,
            threshold_positive: 0.0,
            threshold_negative: 0.0,
            confident: vec![vec![0; 2]; 2],
            joint: vec![vec![0.0; 2]; 2],
            noise_positive: 0.0,
            noise_negative: 0.0,
            noise: 0.0,
            ceiling: 1.0,
            suspected: 0,
            unconfident: 0,
            estimated_errors: 0,
            suspects: Vec::new(),
            note: String::new(),
            refused: Some(reason.into()),
        }
    }
}

/// Audit the labels with the out-of-fold probabilities that produced the score.
///
/// `ids` may be shorter than `scores`, in which case the missing rows are named
/// by position. The audit refuses rather than guessing when the two classes are
/// not both present, when the arrays disagree in length, or when a score is not
/// a probability: a ranking can be audited this way and a raw margin cannot.
pub fn audit(ids: &[String], scores: &[f64], labels: &[bool], cfg: &AuditConfig) -> LabelAudit {
    if scores.len() != labels.len() {
        return LabelAudit::refused(
            labels.len(),
            format!("{} scores against {} labels", scores.len(), labels.len()),
        );
    }
    if scores.is_empty() {
        return LabelAudit::refused(0, "no rows");
    }
    if let Some(at) = scores.iter().position(|score| !(0.0..=1.0).contains(score)) {
        return LabelAudit::refused(
            scores.len(),
            format!(
                "row {} scores {:.3}, which is not a probability",
                at + 1,
                scores[at]
            ),
        );
    }
    let positives = labels.iter().filter(|label| **label).count();
    let negatives = labels.len() - positives;
    if positives == 0 || negatives == 0 {
        return LabelAudit::refused(
            scores.len(),
            format!("{positives} rows of one class so one class has no threshold"),
        );
    }

    let positive_mean: f64 = scores
        .iter()
        .zip(labels)
        .filter(|(_, label)| **label)
        .map(|(score, _)| *score)
        .sum::<f64>()
        / positives as f64;
    let negative_mean: f64 = scores
        .iter()
        .zip(labels)
        .filter(|(_, label)| !**label)
        .map(|(score, _)| 1.0 - *score)
        .sum::<f64>()
        / negatives as f64;

    let mut confident = vec![vec![0usize; 2]; 2];
    let mut unconfident = 0usize;
    let mut suspects: Vec<Suspect> = Vec::with_capacity(scores.len());
    for (at, (score, label)) in scores.iter().zip(labels).enumerate() {
        let given = usize::from(*label);
        let clears_positive = *score >= positive_mean - TIE;
        let clears_negative = (1.0 - *score) >= negative_mean - TIE;
        let guessed = match (clears_positive, clears_negative) {
            (true, true) => usize::from(*score >= 1.0 - *score),
            (true, false) => POSITIVE,
            (false, true) => NEGATIVE,
            // Nothing clears its bar, so the row is not evidence either way and
            // is left out of the joint; the class totals are scaled back up.
            (false, false) => {
                unconfident += 1;
                suspects.push(Suspect {
                    id: ids
                        .get(at)
                        .cloned()
                        .unwrap_or_else(|| format!("row {}", at + 1)),
                    label: *label,
                    probability: round4(if *label { *score } else { 1.0 - *score }),
                    margin: round4(if *label {
                        2.0 * *score - 1.0
                    } else {
                        1.0 - 2.0 * *score
                    }),
                });
                continue;
            }
        };
        confident[given][guessed] += 1;
        let probability = if *label { *score } else { 1.0 - *score };
        let margin = if *label {
            2.0 * *score - 1.0
        } else {
            1.0 - 2.0 * *score
        };
        let id = ids
            .get(at)
            .cloned()
            .unwrap_or_else(|| format!("row {}", at + 1));
        suspects.push(Suspect {
            id,
            label: *label,
            probability: round4(probability),
            margin: round4(margin),
        });
    }

    let given_counts = [negatives, positives];
    let mut joint = vec![vec![0.0f64; 2]; 2];
    for given in [NEGATIVE, POSITIVE] {
        let total: usize = confident[given].iter().sum();
        if total == 0 {
            return LabelAudit::refused(
                scores.len(),
                format!("no confident rows in class {given}, so its noise is not estimable"),
            );
        }
        for guessed in [NEGATIVE, POSITIVE] {
            joint[given][guessed] =
                confident[given][guessed] as f64 * given_counts[given] as f64 / total as f64;
        }
    }

    let errors: f64 = joint[NEGATIVE][POSITIVE] + joint[POSITIVE][NEGATIVE];
    let noise = errors / scores.len() as f64;
    let noise_negative = (joint[NEGATIVE][POSITIVE] / negatives as f64).clamp(0.0, 1.0);
    let noise_positive = (joint[POSITIVE][NEGATIVE] / positives as f64).clamp(0.0, 1.0);
    let suspected = confident[NEGATIVE][POSITIVE] + confident[POSITIVE][NEGATIVE];

    suspects.sort_by(|a, b| a.margin.total_cmp(&b.margin));
    suspects.truncate(cfg.top);

    let note = format!(
        "{} of {} rows are confidently labelled against the model and {} cleared neither \
threshold, so {} stay in the joint. Scaled back up to the labelled totals that is {} \
errors, {:.1} percent of the bank, {:.1} percent of the positive labels and {:.1} percent \
of the negative ones. A model that reproduced the true labels would score {:.3} against \
these ones, so a claim above that is a claim about the labels rather than the \
construct. The estimate assumes the rows left out carry the same error rate as the \
ones kept; {:.1} percent of the bank is left out, so that assumption is doing real work \
and the reading is an extrapolation rather than a count.",
        suspected,
        crate::num::commas(scores.len()),
        crate::num::commas(unconfident),
        crate::num::commas(scores.len() - unconfident),
        errors.round(),
        noise * 100.0,
        noise_positive * 100.0,
        noise_negative * 100.0,
        1.0 - noise,
        unconfident as f64 / scores.len() as f64 * 100.0
    );

    LabelAudit {
        rows: scores.len(),
        positives,
        negatives,
        threshold_positive: round4(positive_mean),
        threshold_negative: round4(negative_mean),
        confident,
        joint: joint
            .iter()
            .map(|row| row.iter().map(|value| round4(*value)).collect())
            .collect(),
        noise_positive: round4(noise_positive),
        noise_negative: round4(noise_negative),
        noise: round4(noise),
        ceiling: round4(1.0 - noise),
        suspected,
        unconfident,
        estimated_errors: errors.round() as usize,
        suspects,
        note,
        refused: None,
    }
}

impl LabelAudit {
    /// Whether the audit produced a reading a card can quote.
    pub fn usable(&self) -> bool {
        self.refused.is_none()
    }
}
