//! What a held out fold says about a detector.
//!
//! Two numbers carry the weight here. `balanced` is the mean of recall on the
//! positives and specificity on the negatives, so a bank that is nine tenths
//! plain copy cannot be scored well by answering "no" every time. `majority` is
//! exactly that answer, reported beside it, so a detector that does not clear
//! its own majority class is visible as one rather than flattered by accuracy.

use serde::{Deserialize, Serialize};

use super::lexicon::round4;

/// A confusion matrix with the rates a reader needs and no others.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub rows: usize,
    pub positives: usize,
    pub predicted: usize,
    pub true_positive: usize,
    pub false_positive: usize,
    pub false_negative: usize,
    pub true_negative: usize,
    pub accuracy: f64,
    pub balanced: f64,
    pub recall: f64,
    pub precision: f64,
    /// Accuracy of answering "not this construct" for every row.
    pub majority: f64,
    /// Balanced accuracy of the same fit on shuffled labels. The control.
    pub shuffled: Option<f64>,
}

/// Score held out predictions against their labels.
pub fn evaluate(gold: &[bool], predicted: &[bool]) -> Metrics {
    let mut m = Metrics {
        rows: gold.len(),
        ..Default::default()
    };
    for (want, got) in gold.iter().zip(predicted) {
        if *want {
            m.positives += 1;
        }
        if *got {
            m.predicted += 1;
        }
        match (*want, *got) {
            (true, true) => m.true_positive += 1,
            (true, false) => m.false_negative += 1,
            (false, true) => m.false_positive += 1,
            (false, false) => m.true_negative += 1,
        }
    }
    let n = m.rows.max(1) as f64;
    let positives = m.positives.max(1) as f64;
    let negatives = (m.rows - m.positives).max(1) as f64;
    let recall = m.true_positive as f64 / positives;
    let specificity = m.true_negative as f64 / negatives;
    m.accuracy = round4((m.true_positive + m.true_negative) as f64 / n);
    m.balanced = round4((recall + specificity) / 2.0);
    m.recall = round4(recall);
    m.precision = round4(m.true_positive as f64 / m.predicted.max(1) as f64);
    m.majority = round4(negatives.max(positives) / n);
    m
}

/// The cut that maximises balanced accuracy, read off labelled scores.
///
/// A reading above the cut is a positive, so the candidate cuts are the
/// midpoints between neighbouring scores and the count of positives is read off
/// the top of the sorted list. Ties go to the lower cut, which keeps more of the
/// positives.
pub fn best_cut(scores: &[f64], gold: &[bool]) -> f64 {
    if scores.is_empty() || scores.len() != gold.len() {
        return 0.5;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| scores[*a].total_cmp(&scores[*b]));
    let positives = gold.iter().filter(|g| **g).count();
    let negatives = gold.len() - positives;
    if positives == 0 || negatives == 0 {
        return 0.5;
    }
    let mut best = (f64::NEG_INFINITY, round4(scores[order[0]] - 1e-6));
    let mut below = 0usize;
    for (position, index) in order.iter().enumerate() {
        if gold[*index] {
            below += 1;
        }
        let cut = if position + 1 < order.len() {
            round4((scores[*index] + scores[order[position + 1]]) / 2.0)
        } else {
            round4(scores[*index] + 1e-6)
        };
        let predicted = order.len() - (position + 1);
        let true_positive = positives - below;
        let false_positive = predicted - true_positive;
        let recall = true_positive as f64 / positives as f64;
        let specificity = (negatives - false_positive) as f64 / negatives as f64;
        let balanced = (recall + specificity) / 2.0;
        if balanced > best.0 {
            best = (balanced, cut);
        }
    }
    best.1
}

/// A deterministic shuffle of the labels, used as the control fit.
///
/// Fisher and Yates with a linear congruential source: no crate, no thread
/// dependence, and the same permutation every rebuild.
pub fn shuffled(gold: &[bool], seed: u64) -> Vec<bool> {
    let mut out = gold.to_vec();
    let mut state = seed | 1;
    for i in (1..out.len()).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = ((state >> 33) as usize) % (i + 1);
        out.swap(i, j);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_perfect_detector_scores_one_on_both_classes() {
        let m = evaluate(&[true, true, false, false], &[true, true, false, false]);
        assert_eq!(m.balanced, 1.0);
        assert_eq!(m.accuracy, 1.0);
        assert_eq!(m.majority, 0.5);
    }

    #[test]
    fn answering_no_every_time_clears_accuracy_and_fails_balanced() {
        let m = evaluate(&[true, false, false, false], &[false, false, false, false]);
        assert_eq!(m.accuracy, 0.75);
        assert_eq!(m.majority, 0.75);
        assert_eq!(m.balanced, 0.5);
        assert_eq!(m.recall, 0.0);
    }

    #[test]
    fn the_best_cut_separates_a_line() {
        let scores = [0.1, 0.4, 1.2, 1.5];
        let gold = [false, false, true, true];
        let cut = best_cut(&scores, &gold);
        assert!(cut > 0.4 && cut < 1.2, "{cut}");
        let predicted: Vec<bool> = scores.iter().map(|s| *s >= cut).collect();
        assert_eq!(evaluate(&gold, &predicted).balanced, 1.0);
    }

    #[test]
    fn a_cut_that_predicts_nothing_loses_to_one_that_separates() {
        let scores = [0.0, 0.0, 0.0, 2.0, 2.0];
        let gold = [false, false, false, true, true];
        let cut = best_cut(&scores, &gold);
        assert!(cut > 0.0 && cut <= 2.0, "{cut}");
        assert_eq!(evaluate(&gold, &[true; 5]).balanced, 0.5);
    }

    #[test]
    fn the_control_shuffle_keeps_the_class_balance() {
        let gold = [true, false, true, true, false, false, true];
        let permuted = shuffled(&gold, 7);
        assert_eq!(permuted.len(), gold.len());
        assert_eq!(
            permuted.iter().filter(|g| **g).count(),
            gold.iter().filter(|g| **g).count()
        );
        assert_eq!(permuted, shuffled(&gold, 7));
    }
}
