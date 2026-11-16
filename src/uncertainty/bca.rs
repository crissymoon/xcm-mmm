//! Bias corrected and accelerated bootstrap.
//!
//! The percentile interval is the weakest of the three ways to read a bootstrap
//! distribution. It assumes the resampled statistic is centred on the estimate
//! and symmetric, and a ranking metric on a small bank is neither: AUC is
//! bounded, it is estimated from a handful of positives, and its sampling
//! distribution is visibly skewed near the ceiling. BCa repairs both problems
//! with two numbers read off the same resampling the percentile interval
//! already produced (Efron, *JASA* 82, 1987; Efron and Tibshirani, *An
//! Introduction to the Bootstrap*, 1993, ch. 14).
//!
//! **The bias correction** `z0` asks how far the point estimate sits from the
//! median of the draws, on the normal scale. A positive `z0` means most
//! resamples land below the estimate, which is what a bounded metric near its
//! ceiling looks like.
//!
//! **The acceleration** `a` is a skewness of the leave-out values. It is read
//! from a grouped jackknife rather than a row-level one: deleting one row of
//! seven thousand and refitting the metric seven thousand times is the same
//! answer at a thousand times the cost, and with exchangeable rows the grouped
//! version is the standard cheap substitute. The group count defaults to 20,
//! which is enough to estimate a third moment and small enough to stay instant.
//!
//! If either correction comes back non-finite, or the corrected tail
//! probabilities leave `(0, 1)`, the result falls back to the percentile
//! interval and says so in `fallback`, because an interval that silently
//! degenerates is worse than one that reports what happened.

use crate::rng::Rng;
use crate::stats::{normal_cdf, normal_quantile};

use super::{draws, quantile, Interval};

/// A BCa interval and the two corrections that produced it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bca {
    /// The interval itself, with its point estimate and level.
    pub interval: Interval,
    /// Bias correction on the normal scale.
    pub bias: f64,
    /// Jackknife acceleration.
    pub acceleration: f64,
    /// Jackknife groups used.
    pub groups: usize,
    /// True when the corrections could not be applied and the percentile
    /// interval was returned instead.
    pub fallback: bool,
}

impl Bca {
    /// One line for a terminal.
    pub fn summary(&self) -> String {
        format!(
            "{:.4} [{:.4}, {:.4}] bca  z0 {:+.3}  a {:+.4}{}",
            self.interval.point,
            self.interval.low,
            self.interval.high,
            self.bias,
            self.acceleration,
            if self.fallback {
                "  (percentile fallback)"
            } else {
                ""
            }
        )
    }
}

/// BCa interval for a paired statistic over scores and labels.
///
/// `groups` is clamped to `2..=200`; more groups than rows cannot help.
pub fn bca<F>(
    scores: &[f64],
    labels: &[bool],
    reps: usize,
    level: f64,
    seed: u64,
    groups: usize,
    mut statistic: F,
) -> Bca
where
    F: FnMut(&[f64], &[bool]) -> f64,
{
    let point = statistic(scores, labels);
    let level = level.clamp(0.5, 0.999);
    let percentile = |draws: &[f64]| {
        let tail = (1.0 - level) / 2.0;
        (quantile(draws, tail), quantile(draws, 1.0 - tail))
    };
    if scores.len() != labels.len() || scores.len() < 8 || reps == 0 {
        return Bca {
            interval: Interval {
                point,
                low: point,
                high: point,
                level,
                resamples: 0,
            },
            bias: 0.0,
            acceleration: 0.0,
            groups: 0,
            fallback: true,
        };
    }
    let sample = draws(scores, labels, reps, seed, &mut statistic);
    if sample.len() < 8 || !point.is_finite() {
        return Bca {
            interval: Interval {
                point,
                low: point,
                high: point,
                level,
                resamples: sample.len(),
            },
            bias: 0.0,
            acceleration: 0.0,
            groups: 0,
            fallback: true,
        };
    }
    // Bias: share of draws below the point estimate, on the normal scale.
    let below = sample.iter().filter(|value| **value < point).count() as f64;
    let share = (below + 0.5) / (sample.len() as f64 + 1.0);
    let bias = normal_quantile(share.clamp(1e-9, 1.0 - 1e-9));
    // Acceleration: skewness of the leave-a-group-out values.
    let groups = groups.clamp(2, 200).min(scores.len());
    let (acceleration, _) = acceleration(scores, labels, groups, &mut statistic);
    let tail = (1.0 - level) / 2.0;
    let z_low = normal_quantile(tail);
    let z_high = normal_quantile(1.0 - tail);
    let shift = |z: f64| {
        let denominator = 1.0 - acceleration * (bias + z);
        if denominator.abs() < 1e-12 {
            return f64::NAN;
        }
        normal_cdf(bias + (bias + z) / denominator)
    };
    let (low_probability, high_probability) = (shift(z_low), shift(z_high));
    if !bias.is_finite()
        || !acceleration.is_finite()
        || !low_probability.is_finite()
        || !high_probability.is_finite()
        || low_probability <= 0.0
        || high_probability >= 1.0
        || low_probability >= high_probability
    {
        let (low, high) = percentile(&sample);
        return Bca {
            interval: Interval {
                point,
                low,
                high,
                level,
                resamples: sample.len(),
            },
            bias,
            acceleration,
            groups,
            fallback: true,
        };
    }
    let (low, high) = (
        quantile(&sample, low_probability),
        quantile(&sample, high_probability),
    );
    Bca {
        interval: Interval {
            point,
            low,
            high,
            level,
            resamples: sample.len(),
        },
        bias,
        acceleration,
        groups,
        fallback: false,
    }
}

/// Grouped jackknife acceleration and the leave-one-group-out values.
fn acceleration<F>(
    scores: &[f64],
    labels: &[bool],
    groups: usize,
    statistic: &mut F,
) -> (f64, Vec<f64>)
where
    F: FnMut(&[f64], &[bool]) -> f64,
{
    let n = scores.len();
    let width = n.div_ceil(groups).max(1);
    let mut values = Vec::new();
    let mut kept_scores = Vec::with_capacity(n);
    let mut kept_labels = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        let end = (start + width).min(n);
        kept_scores.clear();
        kept_labels.clear();
        for index in 0..start {
            kept_scores.push(scores[index]);
            kept_labels.push(labels[index]);
        }
        for index in end..n {
            kept_scores.push(scores[index]);
            kept_labels.push(labels[index]);
        }
        let value = statistic(&kept_scores, &kept_labels);
        if value.is_finite() {
            values.push(value);
        }
        start = end;
    }
    if values.len() < 3 {
        return (f64::NAN, values);
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let numerator: f64 = values
        .iter()
        .map(|value| (mean - value).powi(3))
        .sum::<f64>();
    let denominator: f64 = values
        .iter()
        .map(|value| (mean - value).powi(2))
        .sum::<f64>();
    let acceleration = if denominator > 0.0 {
        numerator / (6.0 * denominator.powf(1.5))
    } else {
        0.0
    };
    (acceleration, values)
}

/// Deterministic seed helper, so a caller does not have to invent one.
pub fn seed_for(name: &str) -> u64 {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for byte in name.as_bytes() {
        state = crate::rng::splitmix64(&mut state) ^ u64::from(*byte);
    }
    let mut rng = Rng::new(state);
    rng.next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uncertainty::auc;

    fn bank(rows: usize, positives: usize, seed: u64) -> (Vec<f64>, Vec<bool>) {
        let mut rng = Rng::new(seed);
        let mut scores = Vec::with_capacity(rows);
        let mut labels = Vec::with_capacity(rows);
        for index in 0..rows {
            let label = index < positives;
            labels.push(label);
            let centre = if label { 0.9 } else { 0.0 };
            scores.push(rng.normal(centre, 1.0));
        }
        (scores, labels)
    }

    #[test]
    fn bca_stays_inside_the_unit_interval_and_brackets_the_estimate() {
        let (scores, labels) = bank(600, 120, 11);
        let out = bca(&scores, &labels, 400, 0.95, 7, 20, auc);
        assert!(!out.fallback, "{}", out.summary());
        assert!(out.interval.low > 0.0 && out.interval.high < 1.0);
        assert!(out.interval.low <= out.interval.point);
        assert!(out.interval.high >= out.interval.point);
        assert!(out.interval.width() > 0.0);
        assert_eq!(out.groups, 20);
        assert!(out.acceleration.is_finite());
    }

    #[test]
    fn the_same_seed_gives_the_same_interval() {
        let (scores, labels) = bank(300, 60, 4);
        let first = bca(&scores, &labels, 200, 0.95, 3, 10, auc);
        let second = bca(&scores, &labels, 200, 0.95, 3, 10, auc);
        assert_eq!(first, second);
        assert_eq!(first.interval.point, second.interval.point);
    }

    #[test]
    fn a_degenerate_bank_reports_no_width_rather_than_inventing_one() {
        let scores = vec![0.5; 20];
        let labels = vec![true; 20];
        let out = bca(&scores, &labels, 100, 0.95, 1, 10, auc);
        assert_eq!(out.interval.width(), 0.0);
        assert_eq!(out.interval.low, out.interval.high);
    }

    #[test]
    fn the_acceleration_is_a_skewness_and_vanishes_on_symmetric_data() {
        let (scores, labels) = bank(400, 200, 21);
        let mut statistic = auc;
        let (a, values) = acceleration(&scores, &labels, 20, &mut statistic);
        assert_eq!(values.len(), 20);
        assert!(a.abs() < 0.2, "acceleration {a} on a symmetric bank");
    }
}
