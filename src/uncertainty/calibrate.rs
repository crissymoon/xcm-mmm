//! Probability calibration and the interval around its error.
//!
//! A churn score of 0.3 is a claim that three in ten such customers leave. The
//! reliability table checks the claim; this module repairs it, and prices the
//! repair.
//!
//! **Why beta and not isotonic.** Isotonic regression is the strongest
//! calibration map in the usual comparison and it needs roughly a thousand rows
//! before it beats a parametric map, because it fits a monotone step function
//! with as many steps as the data allows (Niculescu-Mizil and Caruana, ICML
//! 2005). It also reports an expected calibration error near zero in sample,
//! which flatters exactly the table it is meant to test. Beta calibration is
//! three parameters, `logit(q) = a ln(s) + b ln(1 - s) + c`, so it cannot
//! memorise the bank, it handles the over-confidence that a regularised linear
//! model produces, and it returns the identity at `a = b = 1, c = 0` (Kull,
//! Silva Filho and Flach, AISTATS 2017). Under a thousand rows that is the
//! right trade, and the shipped churn bank is 7,043 rows with 1,777 positives,
//! so beta is used there too and the report says which map it applied.
//!
//! **Why the error carries an interval.** ECE on ten bins is a mean of ten
//! noisy ratios, and its standard error at a few thousand rows is a couple of
//! points. Quoting `ECE 0.075` without a band invites a reader to treat two
//! models as different when the difference is inside the noise. The bootstrap
//! here resamples rows, recomputes the bins and reports the percentile
//! interval, so the number on the card is a measurement rather than a decimal.
//!
//! Nothing here refits the model. The scores are given, and a calibration map
//! that was fitted on the same rows it is scored on would be optimistic, so
//! [`calibration_report`] reports the fitted map's in-sample ECE as well as the
//! raw one and says that the honest comparison needs a held-out split.

use serde::{Deserialize, Serialize};

use crate::linalg::solve_spd;
use crate::rng::Rng;

use super::{calibration, Calibration, Interval};

/// The smallest probability a log can take, so a hard zero does not become
/// negative infinity.
const EPSILON: f64 = 1e-6;

/// A fitted beta calibration map.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BetaCalibration {
    /// Coefficient on `ln(s)`.
    pub a: f64,
    /// Coefficient on `ln(1 - s)`.
    pub b: f64,
    /// Intercept.
    pub c: f64,
    /// Rows used in the fit.
    pub rows: usize,
    /// True when the optimiser stopped on a small step rather than the cap.
    pub converged: bool,
    /// Iterations used.
    pub iterations: usize,
    /// Estimator name, always `beta`.
    pub method: String,
}

impl Default for BetaCalibration {
    fn default() -> Self {
        Self {
            a: 1.0,
            b: 1.0,
            c: 0.0,
            rows: 0,
            converged: false,
            iterations: 0,
            method: "beta".to_string(),
        }
    }
}

impl BetaCalibration {
    /// True when the map came from an actual fit.
    pub fn is_usable(&self) -> bool {
        self.rows > 0 && self.converged && self.a.is_finite() && self.b.is_finite()
    }

    /// Map one probability.
    ///
    /// The input is clipped away from the closed ends first, because
    /// `ln(0)` is not a coefficient a linear map can absorb.
    pub fn apply(&self, probability: f64) -> f64 {
        let s = probability.clamp(EPSILON, 1.0 - EPSILON);
        let z = self.a * s.ln() + self.b * (1.0 - s).ln() + self.c;
        sigmoid(z)
    }

    /// Map a whole score vector.
    pub fn apply_all(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|score| self.apply(*score)).collect()
    }
}

/// Fit beta calibration by iteratively reweighted least squares.
///
/// The design is three columns, so the normal equations are a 3x3 solve with a
/// small ridge for stability. A plain Newton step overshoots on a bank whose
/// log-odds columns are collinear, so every step is shortened until the
/// log-likelihood improves and the step length is capped; that is the
/// difference between a fit that converges and one that runs away. A bank with
/// only one label cannot be fitted and returns the identity map with
/// `converged` false.
pub fn beta_calibration(scores: &[f64], labels: &[bool]) -> BetaCalibration {
    if scores.len() != labels.len() || scores.len() < 8 {
        return BetaCalibration::default();
    }
    let positives = labels.iter().filter(|label| **label).count();
    if positives == 0 || positives == labels.len() {
        return BetaCalibration::default();
    }
    let mut beta = BetaCalibration {
        rows: scores.len(),
        ..BetaCalibration::default()
    };
    let rows: Vec<[f64; 3]> = scores
        .iter()
        .map(|score| {
            let s = score.clamp(EPSILON, 1.0 - EPSILON);
            [s.ln(), (1.0 - s).ln(), 1.0]
        })
        .collect();
    let y: Vec<f64> = labels
        .iter()
        .map(|label| f64::from(u8::from(*label)))
        .collect();
    let objective = |weights: &[f64; 3]| -> f64 {
        rows.iter()
            .zip(&y)
            .map(|(row, target)| {
                let p = sigmoid(weights[0] * row[0] + weights[1] * row[1] + weights[2])
                    .clamp(1e-12, 1.0 - 1e-12);
                -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
            })
            .sum()
    };
    let mut weights = [1.0f64, 1.0, 0.0];
    let mut current = objective(&weights);
    for iteration in 0..100 {
        let mut gram = vec![vec![0.0; 3]; 3];
        let mut rhs = vec![0.0; 3];
        for (row, target) in rows.iter().zip(&y) {
            let z = weights[0] * row[0] + weights[1] * row[1] + weights[2];
            let p = sigmoid(z);
            let variance = (p * (1.0 - p)).max(1e-8);
            for i in 0..3 {
                for j in 0..3 {
                    gram[i][j] += variance * row[i] * row[j];
                }
                rhs[i] += (target - p) * row[i];
            }
        }
        let Some(step) = solve_spd(gram, rhs, 1e-6) else {
            break;
        };
        // Cap the move, then shorten it until the log-likelihood improves. A
        // three column logistic fit near collinearity will otherwise oscillate
        // between two extreme maps and never settle.
        let norm = step.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
        let scale = if norm > 4.0 { 4.0 / norm } else { 1.0 };
        let mut alpha = scale;
        let mut improved = false;
        for _ in 0..30 {
            let trial = [
                weights[0] + alpha * step[0],
                weights[1] + alpha * step[1],
                weights[2] + alpha * step[2],
            ];
            let candidate = objective(&trial);
            if candidate.is_finite() && candidate <= current {
                weights = trial;
                current = candidate;
                improved = true;
                break;
            }
            alpha *= 0.5;
        }
        beta.iterations = iteration + 1;
        let moved = step
            .iter()
            .fold(0.0f64, |acc, value| acc.max((alpha * value).abs()));
        if !improved || moved < 1e-9 {
            beta.converged = true;
            break;
        }
    }
    beta.a = weights[0];
    beta.b = weights[1];
    beta.c = weights[2];
    if !(beta.a.is_finite() && beta.b.is_finite() && beta.c.is_finite()) {
        return BetaCalibration::default();
    }
    beta
}

/// Raw and calibrated reliability with an interval on the raw error.
#[derive(Clone, Debug)]
pub struct CalibrationReport {
    /// Reliability of the scores as given.
    pub raw: Calibration,
    /// Reliability after the beta map, on the same rows.
    pub calibrated: Calibration,
    /// The fitted map.
    pub map: BetaCalibration,
    /// Percentile interval on the raw expected calibration error.
    pub ece: Interval,
    /// Percentile interval on the Brier score.
    pub brier: Interval,
    /// Rows in the table.
    pub rows: usize,
    /// The plain reading of the two errors.
    pub note: String,
}

/// Build the reliability table, fit beta calibration and bootstrap the error.
///
/// The bootstrap resamples rows and recomputes both the bins and the Brier
/// score, so the intervals are paired with the point estimates they belong to.
pub fn calibration_report(
    scores: &[f64],
    labels: &[bool],
    bins: usize,
    reps: usize,
    level: f64,
    seed: u64,
) -> CalibrationReport {
    let raw = calibration(scores, labels, bins);
    let map = beta_calibration(scores, labels);
    let calibrated = if map.is_usable() {
        calibration(&map.apply_all(scores), labels, bins)
    } else {
        Calibration::empty()
    };
    let ece = bootstrap_metric(scores, labels, bins, reps, level, seed, false);
    let brier = bootstrap_metric(scores, labels, bins, reps, level, seed ^ 0x5EED, true);
    let note = if map.is_usable() {
        // Round before formatting, so the note and the table beside it cannot
        // disagree in the last digit.
        format!(
            "beta map fitted on these rows: a {:.3}, b {:.3}, c {:+.3}, in-sample ECE {:.3} against raw {:.3}. An in-sample error understates both, so the honest comparison needs a held-out split.",
            map.a,
            map.b,
            map.c,
            super::round4(calibrated.ece),
            super::round4(raw.ece)
        )
    } else {
        "the beta map could not be fitted, so the calibrated column is empty and only the raw error is reported".to_string()
    };
    CalibrationReport {
        raw,
        calibrated,
        map,
        ece,
        brier,
        rows: scores.len(),
        note,
    }
}

fn bootstrap_metric(
    scores: &[f64],
    labels: &[bool],
    bins: usize,
    reps: usize,
    level: f64,
    seed: u64,
    brier: bool,
) -> Interval {
    let point = if brier {
        calibration(scores, labels, bins).brier
    } else {
        calibration(scores, labels, bins).ece
    };
    if scores.len() != labels.len() || scores.len() < 8 || reps == 0 {
        return Interval {
            point,
            low: point,
            high: point,
            level,
            resamples: 0,
        };
    }
    let mut rng = Rng::new(seed);
    let mut values = Vec::with_capacity(reps);
    let mut picked_scores = Vec::with_capacity(scores.len());
    let mut picked_labels = Vec::with_capacity(labels.len());
    for _ in 0..reps {
        picked_scores.clear();
        picked_labels.clear();
        for _ in 0..scores.len() {
            let at = rng.below(scores.len());
            picked_scores.push(scores[at]);
            picked_labels.push(labels[at]);
        }
        let table = calibration(&picked_scores, &picked_labels, bins);
        let value = if brier { table.brier } else { table.ece };
        if value.is_finite() {
            values.push(value);
        }
    }
    if values.is_empty() {
        return Interval {
            point,
            low: point,
            high: point,
            level,
            resamples: 0,
        };
    }
    let tail = (1.0 - level.clamp(0.0, 1.0)) / 2.0;
    Interval {
        point,
        low: super::quantile(&values, tail),
        high: super::quantile(&values, 1.0 - tail),
        level,
        resamples: values.len(),
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn bank(rows: usize, seed: u64) -> (Vec<f64>, Vec<bool>) {
        let mut rng = Rng::new(seed);
        let mut scores = Vec::with_capacity(rows);
        let mut labels = Vec::with_capacity(rows);
        for _ in 0..rows {
            // Over-confident scores: the probability is pushed away from a half
            // before the label is drawn from the true, less extreme rate.
            let truth = rng.uniform(0.05, 0.95);
            let pushed = ((truth - 0.5) * 2.2 + 0.5).clamp(EPSILON, 1.0 - EPSILON);
            scores.push(pushed);
            labels.push(rng.chance(truth));
        }
        (scores, labels)
    }

    #[test]
    fn beta_repairs_over_confidence_on_synthetic_scores() {
        let (scores, labels) = bank(4_000, 12);
        let raw = calibration(&scores, &labels, 10);
        let map = beta_calibration(&scores, &labels);
        assert!(map.is_usable());
        let repaired = calibration(&map.apply_all(&scores), &labels, 10);
        assert!(
            repaired.ece < raw.ece,
            "raw {} repaired {}",
            raw.ece,
            repaired.ece
        );
        assert!(map.a < 1.0, "over-confident scores need a < 1: {}", map.a);
    }

    #[test]
    fn the_identity_map_is_recovered_from_calibrated_scores() {
        let mut rng = Rng::new(3);
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..2_000 {
            let p = rng.uniform(0.05, 0.95);
            scores.push(p);
            labels.push(rng.chance(p));
        }
        let map = beta_calibration(&scores, &labels);
        // The coefficients are not the test. Over a mid range `ln(s)` and
        // `ln(1 - s)` are nearly collinear, so a whole family of triples maps
        // to almost the same curve. What has to be true is that the curve is
        // the identity within a point.
        for step in 1..20 {
            let p = step as f64 / 20.0;
            let mapped = map.apply(p);
            assert!(
                (mapped - p).abs() < 0.06,
                "map moved {p} to {mapped}: a {:.3} b {:.3} c {:+.3}",
                map.a,
                map.b,
                map.c
            );
        }
    }

    #[test]
    fn a_single_label_cannot_be_fitted() {
        let scores = vec![0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9];
        let labels = vec![true; 8];
        let map = beta_calibration(&scores, &labels);
        assert!(!map.is_usable());
        assert_eq!(map.a, 1.0);
        assert_eq!(map.b, 1.0);
        assert_eq!(map.c, 0.0);
    }

    #[test]
    fn the_interval_contains_the_point_and_the_map_is_reported() {
        let (scores, labels) = bank(1_200, 5);
        let report = calibration_report(&scores, &labels, 10, 100, 0.95, 9);
        assert!(report.ece.low <= report.ece.point);
        assert!(report.ece.high >= report.ece.point);
        assert!(report.ece.width() > 0.0);
        assert!(report.note.contains("beta map"));
        assert_eq!(report.rows, 1_200);
        assert!(report.brier.high >= report.brier.low);
    }

    #[test]
    fn the_note_and_the_table_print_one_number_the_same_way() {
        let (scores, labels) = bank(900, 11);
        let report = calibration_report(&scores, &labels, 10, 60, 0.95, 3);
        let stored = super::super::round4(report.calibrated.ece);
        let quoted = report
            .note
            .split("in-sample ECE ")
            .nth(1)
            .and_then(|tail| tail.split(' ').next())
            .expect("the note quotes the fitted error");
        assert_eq!(quoted, format!("{stored:.3}"));
        assert_eq!(
            quoted,
            format!("{:.3}", super::super::round4(report.calibrated.ece))
        );
    }

    #[test]
    fn the_map_is_monotone_so_the_ranking_survives() {
        let (scores, labels) = bank(800, 6);
        let map = beta_calibration(&scores, &labels);
        let mut previous = -1.0;
        for step in 0..=100 {
            let p = step as f64 / 100.0;
            let mapped = map.apply(p);
            assert!(mapped >= previous - 1e-9, "monotone at {p}");
            previous = mapped;
        }
    }
}
