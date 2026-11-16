//! The selection gate: how much of the winning score is selection itself.
//!
//! A search evaluates thousands of candidate subsets and keeps the best; the
//! best of many noisy trials is upward biased whether or not the data holds a
//! signal. This module records every distinct trial the forecast search scores
//! and converts the winning score into a probability against the null that the
//! trials were exchangeable noise.
//!
//! Two corrections keep the count honest. The first is deduplication: the
//! ledger counts distinct column sets, not evaluations. A search revisits the
//! same mask many times, through crossover, through a clone of the problem
//! inside a single objective wrapper, and through a restart; every one of
//! those repeats scores the same bet. Duplicate evaluations of one mask are
//! therefore not independent bets, they are the same bet taken twice, and
//! counting them would overstate the selection budget and make the gate too
//! pessimistic. The ledger keeps the best fold profile seen for each set and
//! counts the set once.
//!
//! The second is the effective trial count. Every candidate is scored on the
//! same folds, so the fold profiles of two nearby subsets move together;
//! treating forty correlated candidates as forty independent draws would
//! compare the winner against a maximum the search never actually took. The
//! mean pairwise correlation of the profiles shrinks the trial count to
//! `N / (1 + (N - 1) rho)`, floored at one. This is the Bailey and Lopez de
//! Prado correction, applied here to the trial count rather than to a Sharpe
//! ratio: correlated candidates spend less of the selection budget than
//! independent ones.
//!
//! Correlation is measured after each fold's mean across trials is removed.
//! Without that step every candidate looks correlated with every other one for
//! a reason that is not dependence: on any panel one fold is harder than
//! another, so every candidate scores worse on the same fold. Raw profiles on
//! the shipped style of panel correlate at about 0.7, which would call the
//! whole search one bet and flatter the winner.
//!
//! The two corrections pull in opposite directions and the gate quotes the
//! more conservative floor of the two. Deduplication shrinks the count, and a
//! smaller count lowers the expected maximum; removing the fold effect lowers
//! the correlation and so raises the effective count back up. Because neither
//! conversion is exact for a search whose dependence changes from one
//! candidate to the next, the gate reports both floors and decides on the
//! larger: a run must not become significant because its candidates happened
//! to look alike.

use std::collections::BTreeMap;

use crate::linalg::mean;
use crate::stats::{expected_max, expected_max_at, normal_cdf};

/// Fold scores at or above this are the failed fit marker of the forecast
/// module, never a real error, so the ledger drops them.
const MARKER_CEILING: f64 = 1.0e11;

/// At least this many distinct trials before a maximum has a distribution to be
/// compared against.
const MIN_TRIALS: usize = 3;

/// Pairwise correlation is read on a deterministic, evenly spaced subsample of
/// at most this many trials, so a long search costs a bounded amount of work
/// and two runs over the same ledger report the same number.
const CORRELATION_SAMPLE: usize = 128;

/// Highest dependence the correction will assume; at one the effective count
/// would collapse to a single trial and the gate would refuse to be useful.
const MAX_CORRELATION: f64 = 0.99;

/// Largest trial count the budget search is willing to consider.
const BUDGET_CEILING: usize = 1_000_000_000;

/// Every distinct candidate the search scored, with the best fold profile seen
/// for it.
///
/// The key is the sorted column set, so two masks that decode to the same
/// features are one bet. The value is the fold score vector with the lowest
/// mean, because a search that revisits a mask can only do so with a better fit
/// if something upstream changed, and the reported number is the winner of that
/// set rather than its first attempt.
#[derive(Clone, Debug, Default)]
pub struct Ledger {
    best: BTreeMap<Vec<usize>, Vec<f64>>,
}

impl Ledger {
    /// An empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one evaluation of a column set.
    ///
    /// Empty sets, empty fold vectors, non-finite scores and the failed fit
    /// marker are ignored: none of them is an observation of the candidate.
    /// A repeat of a set already present replaces the stored profile only when
    /// its mean is lower, so duplicates never raise the trial count.
    pub fn record(&mut self, columns: &[usize], folds: Vec<f64>) {
        if columns.is_empty() || folds.is_empty() {
            return;
        }
        if folds.iter().any(|v| !v.is_finite() || *v > MARKER_CEILING) {
            return;
        }
        let mut key = columns.to_vec();
        key.sort_unstable();
        let candidate = mean(&folds);
        match self.best.get_mut(&key) {
            Some(kept) if mean(kept) <= candidate => {}
            Some(kept) => *kept = folds,
            None => {
                self.best.insert(key, folds);
            }
        }
    }

    /// Number of distinct column sets recorded.
    pub fn len(&self) -> usize {
        self.best.len()
    }

    /// True when nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.best.is_empty()
    }

    /// Mean fold score of every distinct set, in column set order.
    ///
    /// The order is the `BTreeMap` order of the keys, which makes a ledger,
    /// and therefore a gate, reproducible across runs.
    pub fn means(&self) -> Vec<f64> {
        self.best.values().map(|f| mean(f)).collect()
    }

    /// The best set and its mean fold score, lowest mean first.
    pub fn best(&self) -> Option<(Vec<usize>, f64)> {
        let mut winner: Option<(Vec<usize>, f64)> = None;
        for (columns, scores) in &self.best {
            let candidate = mean(scores);
            if winner.as_ref().map(|(_, b)| candidate < *b).unwrap_or(true) {
                winner = Some((columns.clone(), candidate));
            }
        }
        winner
    }
}

/// What the selection gate measured about one search.
#[derive(Clone, Debug)]
pub struct Gate {
    /// Distinct column sets the search scored.
    pub trials: usize,
    /// Trial count after the correlation correction.
    pub effective_trials: f64,
    /// Mean pairwise correlation of the fold profiles, clamped to `[0, 0.99]`.
    pub mean_correlation: f64,
    /// Mean fold score of the winning set, lower is better.
    pub best: f64,
    /// Columns of the winning set, sorted.
    pub best_columns: Vec<usize>,
    /// Winning score in trial units: standard deviations of quality above the
    /// mean trial.
    pub z: f64,
    /// Expected maximum of that many standard normals under the null, and the
    /// number the verdict uses: the larger of the two floors below.
    pub expected_max: f64,
    /// Floor from the raw distinct trial count, which assumes independent
    /// candidates.
    pub expected_max_naive: f64,
    /// Floor from the effective trial count, which accounts for the measured
    /// dependence between candidates.
    pub expected_max_adjusted: f64,
    /// `z - expected_max`: how far the winner clears the noise floor.
    pub margin: f64,
    /// One sided probability that the best of that many null trials reaches
    /// the observed maximum.
    pub p: f64,
    /// Level the caller asked for, clamped to `[0, 1]`.
    pub level: f64,
    /// Human readable reading of the numbers above, naming the trial counts
    /// and the candidate budget the data supports.
    pub verdict: String,
}

impl Gate {
    /// Largest number of candidates whose expected maximum still sits below
    /// the observed winner.
    ///
    /// This is the budget rule: a search that evaluates more subsets than this
    /// cannot claim, at this level, that its winner is above what the same
    /// number of noise trials would produce.
    pub fn budget(&self) -> usize {
        supported_budget(self.z)
    }
}

/// Price a search: turn the winning mean score into a probability against the
/// null that the trials were exchangeable noise.
///
/// Returns `None` for fewer than three distinct trials, for a ledger whose
/// trial qualities have no spread (a constant ledger carries no maximum to
/// price), or for a non-finite statistic.
pub fn gate(ledger: &Ledger, level: f64) -> Option<Gate> {
    let level = if level.is_finite() {
        level.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let keys: Vec<&Vec<usize>> = ledger.best.keys().collect();
    let profiles: Vec<&[f64]> = ledger.best.values().map(|v| v.as_slice()).collect();
    let trials = profiles.len();
    if trials < MIN_TRIALS {
        return None;
    }
    let means: Vec<f64> = profiles.iter().copied().map(mean).collect();
    // Higher quality is a lower error, so the winner is the top of -mean.
    let quality: Vec<f64> = means.iter().map(|m| -m).collect();
    let q_mean = mean(&quality);
    let spread = standard_deviation(&quality);
    if spread <= 0.0 || !spread.is_finite() {
        return None;
    }
    let zs: Vec<f64> = quality.iter().map(|q| (q - q_mean) / spread).collect();
    let index = (0..trials).max_by(|&a, &b| zs[a].total_cmp(&zs[b]))?;
    let z = zs[index];
    if !z.is_finite() {
        return None;
    }

    let picked = stride(trials);
    let sample: Vec<&[f64]> = picked.iter().map(|&i| profiles[i]).collect();
    let mean_correlation = mean_pairwise_correlation(&sample).clamp(0.0, MAX_CORRELATION);

    let n = trials as f64;
    let effective_trials = (n / (1.0 + (n - 1.0) * mean_correlation)).max(1.0);
    let expected_max_naive = expected_max(trials);
    let expected_max_adjusted = expected_max_at(effective_trials);
    // Quote the conservative floor: the run has to clear the floor that assumes
    // independent candidates as well as the one that corrects for dependence.
    let expected_max = expected_max_naive.max(expected_max_adjusted);
    let margin = z - expected_max;
    let p = normal_cdf(-margin);
    let budget = supported_budget(z);
    let cleared = p <= 1.0 - level;
    let verdict = if cleared {
        format!(
            "selection clears the noise floor at level {level:.2}: {trials} distinct trials, {effective_trials:.1} effective, floor {expected_max:.2}, budget {budget} candidates"
        )
    } else {
        format!(
            "no better than the best of {trials} noise trials ({effective_trials:.1} effective, floor {expected_max:.2}), budget {budget} candidates"
        )
    };

    Some(Gate {
        trials,
        effective_trials,
        mean_correlation,
        best: means[index],
        best_columns: keys[index].clone(),
        z,
        expected_max,
        expected_max_naive,
        expected_max_adjusted,
        margin,
        p,
        level,
        verdict,
    })
}

/// Population standard deviation with `n - 1` in the denominator.
fn standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance =
        values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (values.len() - 1) as f64;
    variance.max(0.0).sqrt()
}

/// Evenly spaced indices over `0..n`, at most `CORRELATION_SAMPLE` of them.
fn stride(n: usize) -> Vec<usize> {
    if n <= CORRELATION_SAMPLE {
        return (0..n).collect();
    }
    let step = (n - 1) as f64 / (CORRELATION_SAMPLE - 1) as f64;
    let mut out: Vec<usize> = (0..CORRELATION_SAMPLE)
        .map(|i| (i as f64 * step).round() as usize)
        .collect();
    out.dedup();
    out
}

/// Dependence between candidates: the mean size of the angle between their fold
/// profiles, after the fold effect is removed.
///
/// Four decisions carry this number.
///
/// The profiles are read in logs. A fold error is a positive scale, so one fold
/// being twice as hard as another multiplies every candidate's score by the same
/// factor; in logs that is an additive shift, which is what centring can
/// remove.
///
/// Each fold's mean across trials is subtracted, which removes the shared
/// difficulty of that fold. Without it every candidate looks alike, because
/// they all score worse on the same fold.
///
/// What is left is compared by cosine, not by Pearson. Pearson would centre the
/// centred profiles a second time and divide by whatever variation remains,
/// which for candidates that differ only by level is rounding error: two such
/// candidates would score an arbitrary correlation near zero and the search
/// would be priced as independent draws it never took. The cosine keeps the
/// level, so a candidate that is uniformly better than another across every
/// fold scores one, which is the honest reading: the folds do not separate
/// them.
///
/// The size of the angle is the dependence, not its sign. Two candidates that
/// trade places across folds are as far from independent as two that move
/// together: both mean the fold split, not the candidate, is producing the
/// difference. The mean absolute cosine keeps the effective count monotone in
/// dependence and never above the trial count.
///
/// Returns one when no candidate-specific variation remains at all (every
/// profile has the same shape after centring), zero for a single profile or for
/// fewer than three folds, where centring leaves too little to compare.
fn mean_pairwise_correlation(profiles: &[&[f64]]) -> f64 {
    if profiles.len() < 2 {
        return 0.0;
    }
    let folds = profiles[0].len();
    if folds < 3 || profiles.iter().any(|p| p.len() != folds) {
        return 0.0;
    }
    let logged: Vec<Vec<f64>> = profiles
        .iter()
        .map(|profile| profile.iter().map(|v| v.max(1e-12).ln()).collect())
        .collect();
    let mut fold_mean = vec![0.0; folds];
    for profile in &logged {
        for (f, value) in profile.iter().enumerate() {
            fold_mean[f] += value;
        }
    }
    for value in &mut fold_mean {
        *value /= logged.len() as f64;
    }
    let centred: Vec<Vec<f64>> = logged
        .iter()
        .map(|profile| {
            profile
                .iter()
                .enumerate()
                .map(|(f, value)| value - fold_mean[f])
                .collect()
        })
        .collect();
    // A row with no spread after centring is a candidate whose fold profile is
    // a pure level shift: it carries no fold interaction to correlate, and a
    // correlation computed on it is undefined rather than zero. Those rows are
    // left out of the pairs; if every row is like that, the trials differ only
    // by level and the search is one bet.
    let spread = |row: &[f64]| -> f64 {
        let m = mean(row);
        row.iter().map(|v| (v - m) * (v - m)).sum::<f64>()
    };
    let active: Vec<&[f64]> = centred
        .iter()
        .filter(|row| spread(row) > 1e-18)
        .map(|row| row.as_slice())
        .collect();
    if active.len() < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    let mut pairs = 0usize;
    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            total += cosine(active[i], active[j]).abs();
            pairs += 1;
        }
    }
    if pairs == 0 {
        0.0
    } else {
        total / pairs as f64
    }
}

/// Cosine of the angle between two profiles, zero for a zero vector.
///
/// Defined on profiles that have already had the fold effect removed, and
/// deliberately not centred a second time: the level is part of what makes two
/// candidates the same bet.
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..n {
        num += a[i] * b[i];
        da += a[i] * a[i];
        db += b[i] * b[i];
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    (num / (da.sqrt() * db.sqrt())).clamp(-1.0, 1.0)
}

/// Largest trial count whose expected maximum is still below `z`.
///
/// Found by doubling and then bisecting, because the expected maximum grows
/// like `sqrt(2 log n)` and a linear walk would be slow for a wide winner.
fn supported_budget(z: f64) -> usize {
    if !z.is_finite() || z <= 0.0 {
        return 1;
    }
    let mut good = 1usize;
    let mut edge = 2usize;
    while edge <= BUDGET_CEILING && expected_max(edge) < z {
        good = edge;
        edge = edge.saturating_mul(2);
    }
    let mut bad = edge.min(BUDGET_CEILING.saturating_add(1));
    while good + 1 < bad {
        let mid = good + (bad - good) / 2;
        if expected_max(mid) < z {
            good = mid;
        } else {
            bad = mid;
        }
    }
    good
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn record_all(ledger: &mut Ledger, trials: usize, folds: usize, seed: u64) {
        let mut rng = Rng::new(seed);
        for t in 0..trials {
            let profile: Vec<f64> = (0..folds).map(|_| rng.normal(1.0, 0.05)).collect();
            ledger.record(&[t, 100 + t], profile);
        }
    }

    #[test]
    fn the_ledger_dedupes_and_keeps_the_best_profile() {
        let mut ledger = Ledger::new();
        ledger.record(&[0, 1], vec![1.0, 1.0]);
        ledger.record(&[1, 0], vec![0.5, 0.7]);
        ledger.record(&[0, 1], vec![0.9, 0.9]);
        assert_eq!(ledger.len(), 1);
        let (columns, best) = ledger.best().unwrap();
        assert_eq!(columns, vec![0, 1]);
        assert!((best - 0.6).abs() < 1e-12, "kept {best}");
        assert_eq!(ledger.means().len(), 1);
    }

    #[test]
    fn failed_and_empty_evaluations_are_not_trials() {
        let mut ledger = Ledger::new();
        ledger.record(&[], vec![1.0, 1.0]);
        ledger.record(&[3], Vec::new());
        ledger.record(&[4], vec![MARKER_CEILING * 2.0, 1.0]);
        ledger.record(&[5], vec![f64::NAN, 1.0]);
        ledger.record(&[6], vec![f64::INFINITY, 1.0]);
        assert!(ledger.is_empty());
        assert_eq!(ledger.best(), None);
    }

    #[test]
    fn duplicates_do_not_raise_the_trial_count() {
        let mut ledger = Ledger::new();
        for _ in 0..7 {
            ledger.record(&[0], vec![1.0, 1.1]);
            ledger.record(&[1], vec![1.2, 1.3]);
            ledger.record(&[2], vec![0.9, 1.0]);
            ledger.record(&[3], vec![1.4, 1.5]);
            ledger.record(&[4], vec![1.1, 1.2]);
        }
        assert_eq!(ledger.len(), 5);
        let g = gate(&ledger, 0.95).unwrap();
        assert_eq!(g.trials, 5);
    }

    #[test]
    fn a_clear_winner_clears_the_noise_floor() {
        let mut ledger = Ledger::new();
        record_all(&mut ledger, 40, 4, 17);
        ledger.record(&[999], vec![0.85, 0.86, 0.84, 0.85]);
        let g = gate(&ledger, 0.95).unwrap();
        assert_eq!(g.trials, 41);
        assert!(g.p < 0.05, "p {}", g.p);
        assert!(g.margin > 1.0, "margin {}", g.margin);
        assert!(g.budget() > 1_000, "budget {}", g.budget());
        assert!(g.verdict.contains("clears the noise floor"));
        assert_eq!(g.best_columns, vec![999]);
    }

    #[test]
    fn pure_noise_does_not_clear_the_noise_floor() {
        let mut ledger = Ledger::new();
        record_all(&mut ledger, 40, 4, 23);
        let g = gate(&ledger, 0.95).unwrap();
        assert_eq!(g.trials, 40);
        assert!(g.p > 0.05, "p {} z {}", g.p, g.z);
        assert!(g.margin < 1.5, "margin {}", g.margin);
        assert!(g.verdict.contains("no better than the best of 40"));
    }

    #[test]
    fn too_few_trials_and_a_constant_ledger_return_none() {
        let mut ledger = Ledger::new();
        ledger.record(&[0], vec![1.0, 2.0]);
        ledger.record(&[1], vec![2.0, 3.0]);
        assert!(gate(&ledger, 0.95).is_none());

        let mut flat = Ledger::new();
        for c in 0..5 {
            flat.record(&[c], vec![1.0, 1.0]);
        }
        assert_eq!(flat.len(), 5);
        assert!(gate(&flat, 0.95).is_none());
    }

    #[test]
    fn correlated_trials_spend_less_of_the_budget() {
        let base = [1.0, 1.2, 0.8, 1.1];
        let mut rng = Rng::new(5);
        let mut ledger = Ledger::new();
        for t in 0..30 {
            let scale = 1.0 + 0.01 * t as f64;
            let profile: Vec<f64> = base
                .iter()
                .map(|b| b * scale + 0.0001 * rng.normal(0.0, 1.0))
                .collect();
            ledger.record(&[t], profile);
        }
        let g = gate(&ledger, 0.95).unwrap();
        assert!(g.mean_correlation > 0.9, "rho {}", g.mean_correlation);
        assert!(
            g.effective_trials < g.trials as f64 * 0.5,
            "effective {} of {}",
            g.effective_trials,
            g.trials
        );
    }

    #[test]
    fn the_budget_rule_agrees_with_its_own_definition() {
        for z in [0.5, 1.5, 2.5, 4.0, 6.0] {
            let budget = supported_budget(z);
            assert!(expected_max(budget) < z, "z {z} budget {budget}");
            assert!(
                expected_max(budget + 1) >= z,
                "z {z} budget {budget} could grow"
            );
        }
        assert_eq!(supported_budget(0.0), 1);
        assert_eq!(supported_budget(f64::NAN), 1);
    }

    #[test]
    fn the_correlation_subsample_is_even_and_bounded() {
        assert_eq!(stride(10).len(), 10);
        let big = stride(10_000);
        assert_eq!(big.len(), CORRELATION_SAMPLE);
        assert_eq!(big[0], 0);
        assert_eq!(*big.last().unwrap(), 9_999);
        assert!(big.windows(2).all(|w| w[1] > w[0]));
    }
}
