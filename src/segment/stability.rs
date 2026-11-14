//! Two audits that do not need a label: resampling and a shuffled control.
//!
//! Segments are only worth shipping if the same entities keep landing together
//! when the panel is resampled, and if the partition is better than what the
//! same protocol finds in columns with their pairing destroyed. The first is
//! measured with the adjusted Rand index against the fit on the whole panel;
//! the second by refitting on every column independently permuted and comparing
//! the silhouette.
//!
//! Both run under the fit's own settings, so the comparison is like for like.
//! Neither is an accuracy: there is no label here to be right or wrong about.

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

use super::kmeans::{self, Fit};

/// How well the partition survives resampling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stability {
    pub bootstraps: usize,
    pub mean: f64,
    pub p10: f64,
    pub min: f64,
}

/// What the same protocol finds when the feature pairing is destroyed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Control {
    pub runs: usize,
    pub mean: f64,
    pub max: f64,
}

/// Adjusted Rand index between two labelings of the same rows.
///
/// One when the partitions are identical, zero when they agree no more than
/// chance. It is computed from the contingency counts with binomials in f64, so
/// a large panel cannot overflow the way integer binomials would.
pub fn ari(a: &[usize], b: &[usize]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut pairs = std::collections::BTreeMap::new();
    for (x, y) in a.iter().zip(b) {
        *pairs.entry((*x, *y)).or_insert(0usize) += 1;
    }
    let mut left = std::collections::BTreeMap::new();
    let mut right = std::collections::BTreeMap::new();
    for (x, y) in a.iter().zip(b) {
        *left.entry(*x).or_insert(0usize) += 1;
        *right.entry(*y).or_insert(0usize) += 1;
    }
    let choose2 = |count: usize| count as f64 * (count as f64 - 1.0) / 2.0;
    let index: f64 = pairs.values().map(|&c| choose2(c)).sum();
    let sum_left: f64 = left.values().map(|&c| choose2(c)).sum();
    let sum_right: f64 = right.values().map(|&c| choose2(c)).sum();
    let expected = sum_left * sum_right / choose2(a.len());
    let maximum = 0.5 * (sum_left + sum_right);
    if (maximum - expected).abs() < 1e-12 {
        1.0
    } else {
        (index - expected) / (maximum - expected)
    }
}

/// Resample entities with replacement, refit, and score the agreement of the
/// labels each resample gives the whole panel.
pub fn bootstrap(
    rows: &[Vec<f64>],
    k: usize,
    base: &Fit,
    runs: usize,
    restarts: usize,
    iterations: usize,
    seed: u64,
) -> Stability {
    let n = rows.len();
    let runs = runs.max(1);
    let mut scores = Vec::with_capacity(runs);
    for run in 0..runs {
        let mut rng = Rng::stream(seed ^ 0x5bf0_3641, run as u64 + 1);
        let draw: Vec<usize> = (0..n).map(|_| rng.below(n)).collect();
        let sample: Vec<Vec<f64>> = draw.iter().map(|&i| rows[i].clone()).collect();
        let fit = kmeans::fit(
            &sample,
            k,
            restarts,
            iterations,
            seed.wrapping_add(run as u64),
        );
        let labels = kmeans::assign(rows, &fit.centroids);
        scores.push(ari(&base.labels, &labels));
    }
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let rank = ((scores.len() as f64 * 0.1).ceil() as usize)
        .saturating_sub(1)
        .min(scores.len() - 1);
    Stability {
        bootstraps: runs,
        mean,
        p10: scores[rank],
        min: scores[0],
    }
}

/// Refit on columns shuffled independently and report the silhouette.
///
/// A column shuffle keeps every marginal exactly and destroys the joint
/// structure. If the real panel scores no better than its shuffled copies, the
/// groups are an artefact of the protocol.
pub fn control(
    rows: &[Vec<f64>],
    k: usize,
    runs: usize,
    restarts: usize,
    iterations: usize,
    seed: u64,
) -> Control {
    let runs = runs.max(1);
    let dim = rows[0].len();
    let mut scores = Vec::with_capacity(runs);
    for run in 0..runs {
        let mut rng = Rng::stream(seed ^ 0x9e37_79b9, run as u64 + 1);
        let mut shuffled = rows.to_vec();
        for column in 0..dim {
            for i in (1..shuffled.len()).rev() {
                let j = rng.below(i + 1);
                let value = shuffled[i][column];
                shuffled[i][column] = shuffled[j][column];
                shuffled[j][column] = value;
            }
        }
        let fit = kmeans::fit(
            &shuffled,
            k,
            restarts,
            iterations,
            seed.wrapping_add(run as u64),
        );
        scores.push(kmeans::silhouette(&shuffled, &fit.labels, k));
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Control { runs, mean, max }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_labelings_score_one_and_independent_ones_score_near_zero() {
        let a = vec![0, 0, 1, 1, 2, 2];
        assert_eq!(ari(&a, &a), 1.0);
        let shifted = vec![2, 2, 1, 1, 0, 0];
        assert_eq!(ari(&a, &shifted), 1.0);
        let mixed = vec![0, 1, 0, 1, 2, 0];
        assert!(ari(&a, &mixed) < 0.2);
    }

    #[test]
    fn ari_is_symmetric_and_finite_on_degenerate_input() {
        let a = vec![0, 0, 0, 0];
        let b = vec![0, 0, 1, 1];
        assert!((ari(&a, &b) - ari(&b, &a)).abs() < 1e-12);
        assert_eq!(ari(&[], &[]), 0.0);
    }

    fn blobs() -> Vec<Vec<f64>> {
        let mut rows = Vec::new();
        for cluster in 0..3 {
            for i in 0..15 {
                let t = i as f64 * 0.4;
                rows.push(vec![
                    cluster as f64 * 5.0 + 0.1 * t.sin(),
                    cluster as f64 * 5.0 + 0.1 * t.cos(),
                ]);
            }
        }
        rows
    }

    #[test]
    fn real_blobs_survive_resampling_and_beat_their_shuffled_control() {
        let rows = blobs();
        let fit = kmeans::fit(&rows, 3, 6, 100, 11);
        let stable = bootstrap(&rows, 3, &fit, 10, 4, 100, 11);
        assert!(stable.mean > 0.8, "mean {:.3}", stable.mean);
        let shuffled = control(&rows, 3, 6, 4, 100, 11);
        assert!(kmeans::silhouette(&rows, &fit.labels, 3) > shuffled.mean + 0.2);
    }

    #[test]
    fn a_control_keeps_every_marginal() {
        let rows = vec![
            vec![0.0, 5.0],
            vec![1.0, 6.0],
            vec![2.0, 7.0],
            vec![3.0, 8.0],
        ];
        let got = control(&rows, 2, 2, 2, 20, 9);
        assert_eq!(got.runs, 2);
        assert!(got.mean.is_finite() && got.max.is_finite());
    }
}
