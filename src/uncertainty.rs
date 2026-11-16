//! Interval estimates, calibration and conformal prediction.
//!
//! A point estimate is not a measurement. This module supplies the three
//! pieces the cards were missing: a percentile bootstrap interval for a ranking
//! metric, a reliability table with Brier score and expected calibration error,
//! and a split conformal wrapper that turns a probability into a prediction set
//! with a finite-sample coverage guarantee.
//!
//! Everything is deterministic. The bootstrap draws from [`crate::rng::Rng`]
//! with a caller supplied seed, so two runs on the same machine or on another
//! machine return the same interval. No clock is read.
//!
//! The guarantee attached to [`conformal`] is the usual split conformal one:
//! exchangeable calibration and test rows, and a miscoverage rate `alpha` that
//! is marginal rather than conditional. It is not a promise for a new operator,
//! a new city or a new register, which is exactly the boundary the cards state.

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

/// A point estimate with a percentile interval.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub point: f64,
    pub low: f64,
    pub high: f64,
    pub level: f64,
    pub resamples: usize,
}

impl Interval {
    /// Width of the interval, zero when it could not be estimated.
    pub fn width(&self) -> f64 {
        (self.high - self.low).max(0.0)
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

/// The `q` quantile of a slice, by linear interpolation on sorted values.
pub fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted.sort_by(f64::total_cmp);
    let q = q.clamp(0.0, 1.0);
    let position = q * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    if low == high {
        sorted[low]
    } else {
        let weight = position - low as f64;
        sorted[low] * (1.0 - weight) + sorted[high] * weight
    }
}

/// Percentile bootstrap of any statistic over paired rows.
///
/// The closure receives the resampled scores and labels. A resample that
/// cannot be scored is skipped rather than counted as zero, so a fold with one
/// class does not drag the interval towards a half.
pub fn bootstrap<F>(
    scores: &[f64],
    labels: &[bool],
    reps: usize,
    level: f64,
    seed: u64,
    mut statistic: F,
) -> Interval
where
    F: FnMut(&[f64], &[bool]) -> f64,
{
    let point = statistic(scores, labels);
    if scores.len() != labels.len() || scores.len() < 4 || reps == 0 {
        return Interval {
            point,
            low: point,
            high: point,
            level,
            resamples: 0,
        };
    }
    let mut rng = Rng::new(seed);
    let mut draws = Vec::with_capacity(reps);
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
        let value = statistic(&picked_scores, &picked_labels);
        if value.is_finite() {
            draws.push(value);
        }
    }
    if draws.is_empty() {
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
        low: quantile(&draws, tail),
        high: quantile(&draws, 1.0 - tail),
        level,
        resamples: draws.len(),
    }
}

/// Percentile bootstrap interval for AUC.
pub fn bootstrap_auc(
    scores: &[f64],
    labels: &[bool],
    reps: usize,
    level: f64,
    seed: u64,
) -> Interval {
    bootstrap(scores, labels, reps, level, seed, auc)
}

/// One reliability bin.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub low: f64,
    pub high: f64,
    pub rows: usize,
    pub predicted: f64,
    pub observed: f64,
}

/// A reliability table with the two scalar summaries a card should carry.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub rows: usize,
    pub bins: Vec<CalibrationBin>,
    pub brier: f64,
    pub log_loss: f64,
    pub ece: f64,
    pub max_gap: f64,
}

impl Calibration {
    /// Empty calibration for a model that could not be scored.
    pub fn empty() -> Self {
        Self::default()
    }

    /// True when a reliability curve can be read.
    pub fn is_usable(&self) -> bool {
        self.rows > 0 && !self.bins.is_empty()
    }
}

/// Reliability of a probability model over equal-width bins.
///
/// `bins` is clamped to `2..=20`. Rows outside `[0, 1]` are clipped, because a
/// probability is what the model claims and a value of 1.2 is a bug rather
/// than a different kind of evidence.
pub fn calibration(scores: &[f64], labels: &[bool], bins: usize) -> Calibration {
    if scores.len() != labels.len() || scores.is_empty() {
        return Calibration::empty();
    }
    let bins = bins.clamp(2, 20);
    let mut counts = vec![0usize; bins];
    let mut predicted = vec![0.0; bins];
    let mut observed = vec![0.0; bins];
    let mut brier = 0.0;
    let mut log_loss = 0.0;
    let mut clipped = Vec::with_capacity(scores.len());
    for (score, label) in scores.iter().zip(labels) {
        let p = score.clamp(0.0, 1.0);
        clipped.push(p);
        let y = f64::from(u8::from(*label));
        brier += (p - y) * (p - y);
        let safe = p.clamp(1e-12, 1.0 - 1e-12);
        log_loss -= if *label { safe.ln() } else { (1.0 - safe).ln() };
        let bucket = ((p * bins as f64).floor() as usize).min(bins - 1);
        counts[bucket] += 1;
        predicted[bucket] += p;
        observed[bucket] += y;
    }
    let n = scores.len() as f64;
    let mut table = Vec::with_capacity(bins);
    let mut ece = 0.0;
    let mut max_gap: f64 = 0.0;
    for bucket in 0..bins {
        if counts[bucket] == 0 {
            continue;
        }
        let count = counts[bucket] as f64;
        let mean_predicted = predicted[bucket] / count;
        let mean_observed = observed[bucket] / count;
        let gap = (mean_observed - mean_predicted).abs();
        ece += count / n * gap;
        max_gap = max_gap.max(gap);
        table.push(CalibrationBin {
            low: bucket as f64 / bins as f64,
            high: (bucket + 1) as f64 / bins as f64,
            rows: counts[bucket],
            predicted: mean_predicted,
            observed: mean_observed,
        });
    }
    Calibration {
        rows: scores.len(),
        bins: table,
        brier: brier / n,
        log_loss: log_loss / n,
        ece,
        max_gap,
    }
}

/// A split conformal result for a binary label.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Conformal {
    /// Miscoverage the caller asked for.
    pub alpha: f64,
    /// Rows used to choose the threshold.
    pub calibration_rows: usize,
    /// Rows used to measure coverage.
    pub test_rows: usize,
    /// The nonconformity threshold.
    pub threshold: f64,
    /// Empirical coverage on the test half.
    pub coverage: f64,
    /// Mean prediction set size, between zero and two.
    pub mean_set: f64,
    /// Share of test rows whose set holds both labels.
    pub both: f64,
    /// Share of test rows whose set is empty.
    pub empty: f64,
}

/// Split conformal for a binary classifier.
///
/// The first half of the rows, in the order given, chooses the threshold and
/// the second half measures coverage. Splitting by position rather than at
/// random keeps the result reproducible and makes it obvious that the coverage
/// number is not the calibration set scored against itself.
///
/// The nonconformity of a row is `1 - p(true label)`. A prediction set holds a
/// label when its nonconformity is at or below the threshold, so a confident
/// and correct row yields a singleton set and an uncertain row yields both
/// labels. The finite-sample correction `ceil((n + 1)(1 - alpha)) / n` is the
/// usual split conformal quantile.
pub fn conformal(scores: &[f64], labels: &[bool], alpha: f64) -> Conformal {
    if scores.len() != labels.len() || scores.len() < 8 {
        return Conformal::default();
    }
    let alpha = alpha.clamp(0.01, 0.5);
    let split = scores.len() / 2;
    let mut nonconformity = Vec::with_capacity(split);
    for (score, label) in scores[..split].iter().zip(&labels[..split]) {
        let p = score.clamp(0.0, 1.0);
        let p_true = if *label { p } else { 1.0 - p };
        nonconformity.push(1.0 - p_true);
    }
    nonconformity.sort_by(f64::total_cmp);
    let n = nonconformity.len();
    let rank = (((n + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    let threshold = nonconformity[rank.saturating_sub(1).min(n - 1)];

    let mut covered = 0usize;
    let mut size_total = 0.0;
    let mut both = 0usize;
    let mut empty = 0usize;
    let test = scores.len() - split;
    for (score, label) in scores[split..].iter().zip(&labels[split..]) {
        let p = score.clamp(0.0, 1.0);
        let include_one = (1.0 - p) <= threshold;
        let include_zero = p <= threshold;
        let size = f64::from(u8::from(include_zero)) + f64::from(u8::from(include_one));
        size_total += size;
        if size >= 2.0 {
            both += 1;
        }
        if size == 0.0 {
            empty += 1;
        }
        let truth_in = if *label { include_one } else { include_zero };
        if truth_in {
            covered += 1;
        }
    }
    Conformal {
        alpha,
        calibration_rows: split,
        test_rows: test,
        threshold,
        coverage: covered as f64 / test.max(1) as f64,
        mean_set: size_total / test.max(1) as f64,
        both: both as f64 / test.max(1) as f64,
        empty: empty as f64 / test.max(1) as f64,
    }
}

/// Prediction set for one probability under a conformal threshold.
pub fn conformal_set(probability: f64, threshold: f64) -> (bool, bool) {
    let p = probability.clamp(0.0, 1.0);
    (p <= threshold, (1.0 - p) <= threshold)
}

/// Effective sample size for a mean under a finite population, `n`.
pub fn effective_n(positives: usize, negatives: usize) -> f64 {
    let n = positives + negatives;
    if n == 0 {
        return 0.0;
    }
    let p = positives as f64 / n as f64;
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    1.0 / (1.0 / positives as f64 + 1.0 / negatives as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn separable() -> (Vec<f64>, Vec<bool>) {
        let mut rng = Rng::new(42);
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        for i in 0..200 {
            let positive = i % 2 == 0;
            labels.push(positive);
            let score = if positive {
                0.35 + 0.5 * rng.next_f64()
            } else {
                0.15 + 0.5 * rng.next_f64()
            };
            scores.push(score);
        }
        (scores, labels)
    }

    #[test]
    fn auc_is_perfect_for_a_perfect_order() {
        assert!((auc(&[0.9, 0.8, 0.2, 0.1], &[true, true, false, false]) - 1.0).abs() < 1e-9);
        assert!((auc(&[0.5, 0.5, 0.5], &[true, false, true]) - 0.5).abs() < 1e-9);
        assert!((auc(&[], &[]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_bootstrap_interval_brackets_the_point_estimate() {
        let (scores, labels) = separable();
        let interval = bootstrap_auc(&scores, &labels, 200, 0.95, 7);
        assert!(interval.point > 0.6);
        assert!(interval.low <= interval.point);
        assert!(interval.high >= interval.point);
        assert!(interval.high - interval.low > 0.0);
        assert_eq!(interval.resamples, 200);
    }

    #[test]
    fn the_bootstrap_is_deterministic() {
        let (scores, labels) = separable();
        let a = bootstrap_auc(&scores, &labels, 80, 0.9, 3);
        let b = bootstrap_auc(&scores, &labels, 80, 0.9, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn a_calibrated_model_has_a_small_gap() {
        let mut rng = Rng::new(9);
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..2000 {
            let p = rng.next_f64();
            scores.push(p);
            labels.push(rng.next_f64() < p);
        }
        let report = calibration(&scores, &labels, 10);
        assert_eq!(report.rows, 2000);
        assert!(report.ece < 0.05, "ece {}", report.ece);
        assert!(report.brier > 0.0 && report.brier < 0.3);
        assert!(report.is_usable());
    }

    #[test]
    fn conformal_covers_at_the_requested_rate() {
        let (scores, labels) = separable();
        let report = conformal(&scores, &labels, 0.1);
        assert!(report.coverage >= 0.8, "coverage {}", report.coverage);
        assert!(report.coverage <= 1.0);
        assert!(report.mean_set >= 0.0 && report.mean_set <= 2.0);
        assert!(report.threshold >= 0.0 && report.threshold <= 1.0);
        assert_eq!(report.calibration_rows + report.test_rows, scores.len());
    }

    #[test]
    fn a_singleton_set_names_one_label() {
        assert_eq!(conformal_set(0.95, 0.2), (false, true));
        assert_eq!(conformal_set(0.05, 0.2), (true, false));
        assert_eq!(conformal_set(0.5, 0.9), (true, true));
    }
}
