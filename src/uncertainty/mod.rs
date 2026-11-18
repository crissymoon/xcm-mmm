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

pub mod bca;
pub mod calibrate;
pub mod coverage;

pub use bca::{bca, Bca};
pub use calibrate::{beta_calibration, calibration_report, BetaCalibration, CalibrationReport};
pub use coverage::{simulate as simulate_coverage, CoverageConfig, CoverageReport, MethodCoverage};

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
    let draws = draws(scores, labels, reps, seed, &mut statistic);
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

/// The bootstrap resamples a percentile interval is read from.
///
/// Split out so a caller that needs the whole distribution, rather than its two
/// tails, does not reimplement the resampling loop. BCa in
/// [`bca::bca`] is exactly such a caller: it maps the tails through a bias and
/// an acceleration correction before reading the same draws.
///
/// Non-finite values are dropped rather than clamped, so a resample that cannot
/// be scored does not drag an interval towards a half.
pub fn draws<F>(
    scores: &[f64],
    labels: &[bool],
    reps: usize,
    seed: u64,
    statistic: &mut F,
) -> Vec<f64>
where
    F: FnMut(&[f64], &[bool]) -> f64,
{
    if scores.len() != labels.len() || scores.len() < 4 || reps == 0 {
        return Vec::new();
    }
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(reps);
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
            out.push(value);
        }
    }
    out
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

/// Name reported by [`delong_auc`], so an artifact names its own estimator.
pub const DELONG_METHOD: &str = "delong-logit";

/// A DeLong standard error and interval for AUC, read on the logit scale.
///
/// `point` is the raw Mann-Whitney AUC with ties at half credit. `low` and
/// `high` are back transformed from a normal interval on the logit scale, so
/// the interval is centred on the continuity corrected AUC rather than on
/// `point`. That asymmetry is deliberate and visible on a near perfect
/// classifier, where the upper bound sits closer to the reading than the lower
/// bound does.
///
/// DeLong, DeLong and Clarke-Pearson (1988) supply the structural components
/// and the variance; Sun and Xu (2014) supply the fast placement algorithm
/// used here to count, for each positive, how many negatives it outranks. The
/// interval is asymptotic and liberal when positives are under about fifty, so
/// a caller should quote the wider of this interval and the percentile
/// bootstrap. The logit transform keeps the bounds inside the unit interval,
/// which a raw `point +/- z * se` does not: at an AUC of 0.995 with a hundred
/// positives the symmetric bounds cross one and the upper limit stops meaning
/// anything.
#[derive(Clone, Debug, PartialEq)]
pub struct DelongAuc {
    /// Raw Mann-Whitney AUC, ties at half credit.
    pub point: f64,
    /// Standard error of the AUC under the DeLong variance.
    pub se: f64,
    /// Lower end of the interval, back transformed from the logit scale.
    pub low: f64,
    /// Upper end of the interval, back transformed from the logit scale.
    pub high: f64,
    /// Nominal level the caller asked for.
    pub level: f64,
    /// Positive rows used.
    pub positives: usize,
    /// Negative rows used.
    pub negatives: usize,
    /// Estimator name, always `delong-logit`.
    pub method: &'static str,
}

/// AUC with a DeLong standard error and a logit scale interval.
///
/// With one class empty, fewer than four rows, or mismatched lengths the
/// reading is degenerate: `se` is zero and `low` and `high` both equal `point`,
/// which is how a caller tells "no noise estimated" from "noise estimated as
/// zero".
pub fn delong_auc(scores: &[f64], labels: &[bool], level: f64) -> DelongAuc {
    let level = level.clamp(0.5, 0.999_999);
    let positives = labels.iter().filter(|label| **label).count();
    let negatives = labels.len().saturating_sub(positives);
    let flat = |point: f64, se: f64, low: f64, high: f64| DelongAuc {
        point,
        se,
        low,
        high,
        level,
        positives,
        negatives,
        method: DELONG_METHOD,
    };
    if scores.len() != labels.len() || positives == 0 || negatives == 0 {
        return flat(0.5, 0.0, 0.5, 0.5);
    }
    let Some(placing) = placements(scores, labels) else {
        return flat(0.5, 0.0, 0.5, 0.5);
    };
    let point = placing.point;
    if scores.len() < 4 {
        return flat(point, 0.0, point, point);
    }
    let se = placing.se();
    let pairs = (positives * negatives) as f64;
    let corrected = ((point * pairs + 0.5) / (pairs + 1.0)).clamp(1e-6, 1.0 - 1e-6);
    let centre = (corrected / (1.0 - corrected)).ln();
    let scale = se / (corrected * (1.0 - corrected));
    let z = crate::stats::z_for_level(level);
    flat(
        point,
        se,
        back_logit(centre - z * scale),
        back_logit(centre + z * scale),
    )
}

/// The DeLong placements behind one score vector.
///
/// `v10` holds, for each positive, the share of negatives it outranks; `v01`
/// holds the mirror for each negative. The variance of the AUC needs only these
/// two vectors, and so does the covariance between two models scored on the same
/// rows, which is why they are computed once and shared.
struct Placements {
    point: f64,
    v10: Vec<f64>,
    v01: Vec<f64>,
    positives: usize,
    negatives: usize,
}

impl Placements {
    /// Standard error of this reading under the DeLong variance.
    fn se(&self) -> f64 {
        let variance = sample_var(&self.v10) / self.positives as f64
            + sample_var(&self.v01) / self.negatives as f64;
        variance.max(0.0).sqrt()
    }
}

/// Placements for `scores` against `labels`, or `None` when a class is empty.
fn placements(scores: &[f64], labels: &[bool]) -> Option<Placements> {
    if scores.len() != labels.len() {
        return None;
    }
    let positives = labels.iter().filter(|label| **label).count();
    let negatives = labels.len().checked_sub(positives)?;
    if positives == 0 || negatives == 0 {
        return None;
    }
    let mut positive_scores = Vec::with_capacity(positives);
    let mut negative_scores = Vec::with_capacity(negatives);
    for (score, label) in scores.iter().zip(labels) {
        if *label {
            positive_scores.push(*score);
        } else {
            negative_scores.push(*score);
        }
    }
    positive_scores.sort_by(f64::total_cmp);
    negative_scores.sort_by(f64::total_cmp);
    // The placements are kept in the caller's row order, not in sorted order.
    // A variance does not care, but the covariance between two models does: the
    // paired reading needs entry `i` of one vector to be the same row as entry
    // `i` of the other, and two models sort their rows differently.
    let mut v10 = vec![0.0; positives];
    let mut v01 = vec![0.0; negatives];
    let (mut at_positive, mut at_negative) = (0usize, 0usize);
    for (score, label) in scores.iter().zip(labels) {
        if *label {
            let (below, equal) = place(&negative_scores, *score);
            v10[at_positive] = (below as f64 + 0.5 * equal as f64) / negatives as f64;
            at_positive += 1;
        } else {
            let (below, equal) = place(&positive_scores, *score);
            let above = positives - below - equal;
            v01[at_negative] = (above as f64 + 0.5 * equal as f64) / positives as f64;
            at_negative += 1;
        }
    }
    let point = v10.iter().sum::<f64>() / positives as f64;
    Some(Placements {
        point,
        v10,
        v01,
        positives,
        negatives,
    })
}

/// Two AUCs, their difference and the paired DeLong interval on it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairedAuc {
    /// First reading, the one being credited.
    pub first: f64,
    /// Second reading, the reference.
    pub second: f64,
    /// `first - second`.
    pub delta: f64,
    /// Standard error of the difference, from the paired covariance.
    pub se: f64,
    /// Lower end of the interval on the difference.
    pub low: f64,
    /// Upper end of the interval on the difference.
    pub high: f64,
    /// Correlation between the two sets of placements, the reason the paired
    /// error is smaller than the independent one.
    pub correlation: f64,
    /// Nominal level.
    pub level: f64,
    /// Positive rows used, as a reminder of what the interval rests on.
    pub positives: usize,
    /// Negative rows used.
    pub negatives: usize,
}

/// The paired DeLong reading of `first - second` on the same rows.
///
/// Two models scored on the same rows are not independent, and comparing them as
/// if they were overstates the interval by a factor of up to `sqrt(2)`. DeLong's
/// covariance adds the cross terms of the placements:
/// `cov = C10 / m + C01 / n`, using the same structural vectors the single
/// reading uses. The interval is symmetric on the difference rather than on the
/// logit scale, because a difference can be zero or negative and a logit of it
/// has no meaning.
///
/// Returns `None` when the two vectors disagree in length or a class is empty,
/// which is the caller's cue to print nothing rather than a degenerate interval.
pub fn delong_pair(
    first: &[f64],
    second: &[f64],
    labels: &[bool],
    level: f64,
) -> Option<PairedAuc> {
    if first.len() != second.len() || first.len() != labels.len() {
        return None;
    }
    let a = placements(first, labels)?;
    let b = placements(second, labels)?;
    let m = a.positives as f64;
    let n = a.negatives as f64;
    let se_a = a.se();
    let se_b = b.se();
    let covariance = sample_cov(&a.v10, &b.v10) / m + sample_cov(&a.v01, &b.v01) / n;
    let variance = (se_a * se_a + se_b * se_b - 2.0 * covariance).max(0.0);
    let se = variance.sqrt();
    let delta = a.point - b.point;
    let z = crate::stats::z_for_level(level.clamp(0.5, 0.999_999));
    let correlation = if se_a > 0.0 && se_b > 0.0 {
        covariance / (se_a * se_b)
    } else {
        0.0
    };
    Some(PairedAuc {
        first: a.point,
        second: b.point,
        delta,
        se,
        low: delta - z * se,
        high: delta + z * se,
        correlation: correlation.clamp(-1.0, 1.0),
        level,
        positives: a.positives,
        negatives: a.negatives,
    })
}

/// How many sorted values sit below `score`, and how many equal it.
fn place(sorted: &[f64], score: f64) -> (usize, usize) {
    let below = sorted.partition_point(|value| value.total_cmp(&score).is_lt());
    let through = sorted.partition_point(|value| value.total_cmp(&score).is_le());
    (below, through - below)
}

/// Sample variance with `ddof` one, zero when there is nothing to estimate.
fn sample_var(values: &[f64]) -> f64 {
    sample_cov(values, values)
}

/// Sample covariance with `ddof` one, zero when the lengths do not match.
fn sample_cov(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || a.len() != b.len() {
        return 0.0;
    }
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let spread: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum();
    spread / (n - 1.0)
}

/// Inverse logit, kept strictly inside the unit interval.
fn back_logit(value: f64) -> f64 {
    (1.0 / (1.0 + (-value).exp())).clamp(1e-9, 1.0 - 1e-9)
}

/// One class of a Mondrian conformal split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassCoverage {
    /// The label this row describes.
    pub label: bool,
    /// Calibration rows of this class, on the first half.
    pub calibration_rows: usize,
    /// Test rows of this class, on the second half.
    pub test_rows: usize,
    /// Nonconformity threshold for this class.
    pub threshold: f64,
    /// Empirical coverage on the test rows of this class.
    pub coverage: f64,
    /// Mean prediction set size on the test rows of this class.
    pub mean_set: f64,
    /// False when the calibration rows cannot support `alpha`.
    pub supported: bool,
}

/// Mondrian, class conditional, split conformal for a binary label.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MondrianConformal {
    /// Miscoverage the caller asked for.
    pub alpha: f64,
    /// The pooled marginal reading, kept so the two sit side by side.
    pub pooled: Conformal,
    /// One row per class, churn then stayed.
    pub classes: Vec<ClassCoverage>,
    /// Why a class was refused, when one was.
    pub refused: Option<String>,
}

/// Split conformal with a threshold per class.
///
/// Vovk's Mondrian conformal gives a guarantee that is class conditional
/// rather than marginal: each class is calibrated on its own rows, so coverage
/// holds inside the churn rows and inside the stayed rows separately.
/// Angelopoulos and Bates (2021) state the same guarantee for the split case.
///
/// The rank `ceil((n_c + 1)(1 - alpha))` only exists when `alpha >= 1 / n_c`.
/// Below that the class has fewer calibration rows than the requested tail
/// needs, and a printed coverage number would be arithmetic without a
/// guarantee. Such a class is marked unsupported, falls back to the pooled
/// threshold so the set still means something, and names the smallest alpha it
/// could support. The refusal is the deliverable, not an error.
pub fn mondrian(scores: &[f64], labels: &[bool], alpha: f64) -> MondrianConformal {
    let alpha = alpha.clamp(1e-6, 0.5);
    let pooled = conformal(scores, labels, alpha);
    let mut classes = Vec::with_capacity(2);
    let mut refusals = Vec::new();
    if scores.len() != labels.len() || scores.len() < 8 {
        return MondrianConformal {
            alpha,
            pooled,
            classes,
            refused: None,
        };
    }
    let split = scores.len() / 2;
    for label in [true, false] {
        let mut calibration: Vec<f64> = scores[..split]
            .iter()
            .zip(&labels[..split])
            .filter(|(_, row)| **row == label)
            .map(|(score, row)| nonconformity(*score, *row))
            .collect();
        let n_c = calibration.len();
        let test_rows = labels[split..].iter().filter(|row| **row == label).count();
        let floor = smallest_alpha(n_c);
        let supported = alpha >= floor;
        let threshold = if supported {
            calibration.sort_by(f64::total_cmp);
            let rank = (((n_c + 1) as f64) * (1.0 - alpha)).ceil() as usize;
            calibration[rank.saturating_sub(1).min(n_c - 1)].clamp(0.0, 1.0)
        } else {
            refusals.push(refusal(label, n_c, alpha, floor));
            pooled.threshold
        };
        let (coverage, mean_set) = class_test(&scores[split..], &labels[split..], label, threshold);
        classes.push(ClassCoverage {
            label,
            calibration_rows: n_c,
            test_rows,
            threshold,
            coverage,
            mean_set,
            supported,
        });
    }
    MondrianConformal {
        alpha,
        pooled,
        classes,
        refused: if refusals.is_empty() {
            None
        } else {
            Some(refusals.join("; "))
        },
    }
}

/// Nonconformity of one row under its own label, `1 - p(true label)`.
fn nonconformity(score: f64, label: bool) -> f64 {
    let p = score.clamp(0.0, 1.0);
    if label {
        1.0 - p
    } else {
        p
    }
}

/// The smallest miscoverage a class with `n` calibration rows can support.
fn smallest_alpha(n: usize) -> f64 {
    if n == 0 {
        f64::INFINITY
    } else {
        1.0 / n as f64
    }
}

fn class_name(label: bool) -> &'static str {
    if label {
        "churn"
    } else {
        "stayed"
    }
}

fn refusal(label: bool, rows: usize, alpha: f64, floor: f64) -> String {
    if rows == 0 {
        format!(
            "{}: no calibration rows, no alpha is supported",
            class_name(label)
        )
    } else {
        format!(
            "{}: {} calibration rows support alpha {:.3} at the smallest, alpha {:.3} refused",
            class_name(label),
            rows,
            floor,
            alpha
        )
    }
}

/// Coverage and mean set size on one class of the test half.
fn class_test(scores: &[f64], labels: &[bool], label: bool, threshold: f64) -> (f64, f64) {
    let mut rows = 0usize;
    let mut covered = 0usize;
    let mut size_total = 0.0;
    for (score, row) in scores.iter().zip(labels) {
        if *row != label {
            continue;
        }
        let (include_zero, include_one) = conformal_set(*score, threshold);
        rows += 1;
        size_total += f64::from(u8::from(include_zero)) + f64::from(u8::from(include_one));
        let truth_in = if label { include_one } else { include_zero };
        if truth_in {
            covered += 1;
        }
    }
    let rows = rows.max(1) as f64;
    (covered as f64 / rows, size_total / rows)
}

/// Round to four decimals, the precision the artifacts store.
///
/// Every printed metric goes through this first, because a value rounded one
/// way in a table and formatted from full precision in the prose beside it can
/// disagree in the last digit. On the churn panel it did: the table said 0.006
/// and the note said 0.007 for the same error.
pub use crate::num::round4;

/// Wilson score interval for a share, `hits` out of `trials`.
///
/// A measured coverage of 0.925 from two hundred trials is not the same claim
/// as 0.925 from twenty thousand, and the difference is what decides whether an
/// interval missed its nominal level or might have. Wilson is used rather than
/// the normal approximation because the share is often near one, where the
/// normal interval runs past one and collapses to zero width at 100 percent.
pub fn wilson(hits: usize, trials: usize, level: f64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    let n = trials as f64;
    let p = hits as f64 / n;
    let z = normal_quantile(0.5 + level / 2.0);
    let denominator = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denominator;
    let spread = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt()) / denominator;
    ((centre - spread).max(0.0), (centre + spread).min(1.0))
}

/// The standard normal quantile for `p` in `(0, 1)`, by bisection on the cdf.
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let mut low = -10.0;
    let mut high = 10.0;
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if crate::stats::normal_cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
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

    #[test]
    fn wilson_keeps_a_share_inside_the_unit_interval() {
        let (low, high) = wilson(200, 200, 0.95);
        assert!((0.0..=1.0).contains(&low) && (0.0..=1.0).contains(&high));
        assert!(high <= 1.0);
        assert!(low < 1.0, "a full share still has a lower bound below one");
    }

    #[test]
    fn wilson_narrows_as_trials_grow() {
        let small = wilson(925, 1_000, 0.95);
        let large = wilson(9_250, 10_000, 0.95);
        assert!(large.1 - large.0 < small.1 - small.0);
        assert!(small.0 <= 0.925 && small.1 >= 0.925);
    }

    #[test]
    fn a_half_point_gap_needs_ten_thousand_trials_to_settle() {
        // 0.925 against a 0.930 floor. The standard error at n trials is
        // sqrt(0.925 * 0.075 / n), so excluding the floor takes about eleven
        // thousand trials. Two hundred and two thousand both reach it.
        let short = wilson(185, 200, 0.95);
        let medium = wilson(1_850, 2_000, 0.95);
        let long = wilson(18_500, 20_000, 0.95);
        assert!(short.1 > 0.930, "200 trials reach the floor: {short:?}");
        assert!(medium.1 > 0.930, "2000 trials still reach it: {medium:?}");
        assert!(long.1 < 0.930, "20000 trials separate it: {long:?}");
    }
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

    fn near_perfect() -> (Vec<f64>, Vec<bool>) {
        let mut rng = Rng::new(11);
        let mut scores = Vec::with_capacity(300);
        let mut labels = Vec::with_capacity(300);
        for _ in 0..300 {
            let positive = rng.next_f64() < 0.5;
            labels.push(positive);
            let base = rng.next_f64();
            scores.push(if positive {
                0.45 + 0.5 * base
            } else {
                0.5 * base
            });
        }
        (scores, labels)
    }

    fn balanced(n: usize, seed: u64) -> (Vec<f64>, Vec<bool>) {
        let mut rng = Rng::new(seed);
        let mut scores = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            let positive = rng.next_f64() < 0.5;
            labels.push(positive);
            let base = rng.next_f64();
            scores.push(if positive {
                0.45 + 0.5 * base
            } else {
                0.15 + 0.5 * base
            });
        }
        (scores, labels)
    }

    #[test]
    fn the_delong_point_is_the_rank_auc() {
        let cases: Vec<(Vec<f64>, Vec<bool>)> = vec![
            (vec![0.9, 0.8, 0.2, 0.1], vec![true, true, false, false]),
            (vec![0.5, 0.5, 0.5], vec![true, false, true]),
            (
                vec![0.1, 0.4, 0.35, 0.8, 0.2, 0.9],
                vec![true, false, false, true, false, true],
            ),
            (separable().0, separable().1),
        ];
        for (scores, labels) in cases {
            let report = delong_auc(&scores, &labels, 0.95);
            let rank = auc(&scores, &labels);
            assert!(
                (report.point - rank).abs() < 1e-9,
                "point {} against {}",
                report.point,
                rank
            );
            assert_eq!(report.method, DELONG_METHOD);
        }
    }

    #[test]
    fn the_delong_interval_brackets_a_near_perfect_reading() {
        let (scores, labels) = near_perfect();
        let report = delong_auc(&scores, &labels, 0.95);
        assert_eq!(report.positives + report.negatives, 300);
        assert!(report.point > 0.95, "point {}", report.point);
        assert!(report.se > 0.0, "se {}", report.se);
        assert!(
            report.low <= report.point,
            "low {} point {}",
            report.low,
            report.point
        );
        assert!(report.high >= report.point);
        assert!(report.low > 0.0 && report.high < 1.0);
        assert!(report.high - report.low > 0.0);
    }

    #[test]
    fn the_delong_standard_error_shrinks_as_rows_grow() {
        let build = |n: usize| {
            let mut rng = Rng::new(5);
            let mut scores = Vec::with_capacity(n);
            let mut labels = Vec::with_capacity(n);
            for at in 0..n {
                let positive = at % 2 == 0;
                labels.push(positive);
                let base = rng.next_f64();
                scores.push(if positive {
                    0.3 + 0.6 * base
                } else {
                    0.1 + 0.6 * base
                });
            }
            (scores, labels)
        };
        let (small_scores, small_labels) = build(100);
        let (large_scores, large_labels) = build(800);
        let small = delong_auc(&small_scores, &small_labels, 0.95);
        let large = delong_auc(&large_scores, &large_labels, 0.95);
        assert!(small.se > 0.0);
        assert!(
            large.se < small.se,
            "large se {} small se {}",
            large.se,
            small.se
        );
    }

    #[test]
    fn a_single_class_reports_no_error() {
        let report = delong_auc(&[0.1, 0.2, 0.3, 0.4], &[true, true, true, true], 0.95);
        assert_eq!(report.se, 0.0);
        assert_eq!(report.low, report.high);
        assert_eq!(report.positives, 4);
        assert_eq!(report.negatives, 0);
        let short = delong_auc(&[0.4, 0.2], &[true, false], 0.95);
        assert_eq!(short.se, 0.0);
        assert_eq!(short.low, short.point);
    }

    #[test]
    fn a_model_paired_with_itself_has_zero_difference_and_zero_error() {
        let (scores, labels) = balanced(120, 23);
        let pair = delong_pair(&scores, &scores, &labels, 0.95).expect("two classes");
        assert_eq!(pair.delta, 0.0);
        assert!(pair.se < 1e-12, "{}", pair.se);
        assert!((pair.correlation - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_paired_interval_is_narrower_than_two_independent_ones() {
        // Two correlated but different readings on the same rows. The paired
        // standard error must sit below the independent bound, which is the
        // whole reason for measuring the covariance.
        let mut rng = Rng::new(9);
        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut labels = Vec::new();
        for at in 0..300 {
            let positive = at % 3 == 0;
            labels.push(positive);
            let base = rng.next_f64();
            let other = rng.next_f64();
            // Overlapping arms: a separated pair has zero placement variance and
            // nothing to compare.
            let lift = if positive { 0.2 } else { 0.0 };
            first.push(0.3 + lift + 0.6 * base);
            second.push(0.3 + lift + 0.6 * other + 0.1 * base);
        }
        let a = delong_auc(&first, &labels, 0.95);
        let b = delong_auc(&second, &labels, 0.95);
        let pair = delong_pair(&first, &second, &labels, 0.95).expect("two classes");
        assert!(pair.se > 0.0);
        assert!(pair.se < (a.se * a.se + b.se * b.se).sqrt());
        assert!(pair.correlation > 0.0 && pair.correlation < 1.0);
        assert!(pair.low <= pair.delta && pair.high >= pair.delta);
    }

    #[test]
    fn a_paired_comparison_refuses_disagreeing_lengths() {
        let labels = [true, false, true, false];
        assert!(delong_pair(&[0.4, 0.2, 0.9, 0.1], &[0.4, 0.2], &labels, 0.95).is_none());
        assert!(delong_pair(
            &[0.4, 0.2, 0.9, 0.1],
            &[0.4, 0.2, 0.9, 0.1],
            &[true; 4],
            0.95
        )
        .is_none());
    }

    #[test]
    fn mondrian_measures_coverage_inside_each_class() {
        let (scores, labels) = balanced(200, 17);
        let report = mondrian(&scores, &labels, 0.1);
        assert_eq!(report.classes.len(), 2);
        for class in &report.classes {
            assert!(class.supported, "class {} refused", class.label);
            assert!(
                class.coverage >= 0.8,
                "class {} coverage {}",
                class.label,
                class.coverage
            );
            assert!(
                class.calibration_rows > 30,
                "class {}",
                class.calibration_rows
            );
            assert!(class.mean_set >= 1.0 && class.mean_set <= 2.0);
        }
        assert!(report.refused.is_none());
        assert!(report.pooled.coverage >= 0.8);
    }

    #[test]
    fn mondrian_refuses_an_alpha_the_calibration_rows_cannot_support() {
        let (scores, labels) = balanced(60, 23);
        let report = mondrian(&scores, &labels, 0.01);
        assert_eq!(report.classes.len(), 2);
        for class in &report.classes {
            assert!(!class.supported, "class {} claimed support", class.label);
            assert!(class.calibration_rows < 20);
            assert_eq!(class.threshold, report.pooled.threshold);
        }
        let refused = report.refused.expect("a refusal");
        assert!(refused.contains("alpha"), "{refused}");
        assert!(refused.contains("alpha 0.010 refused"), "{refused}");
    }
}
