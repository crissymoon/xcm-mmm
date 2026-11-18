//! A sequential reading on a batch series, and the refusals it needs.
//!
//! Two monitoring questions look alike and are not. "Is this week's rate
//! unusual?" needs one batch and a null, which [`super::gate`] supplies. "Has
//! the process moved?" needs a rule that accumulates evidence across batches
//! without firing every time a quiet week follows a loud one, and that rule has
//! to hold at every stopping time, not only at a sample size chosen in advance.
//!
//! ADWIN (Bifet and Gavalda, *SIAM* 2007) is the cheapest such rule. It keeps
//! an adaptive window of the most recent observations, compresses the window
//! into exponential buckets so the memory is `O(log width)`, and drops the
//! older half as soon as the two halves differ by more than a bound at level
//! `delta`. The window is the whole point: after a change it forgets the past,
//! so the reading does not have to be restarted by hand.
//!
//! Four decisions are documented here because each changes what the reading
//! means.
//!
//! **The bound is a union bound, so the level is per stream, not per test.**
//! Reading `n` split points is `n` chances at a false alarm, so the bound uses
//! `ln(2n/delta)`: a `delta` of 1e-6 at 72 batches is a 1e-6 chance of a cut
//! anywhere in those 72, not 1e-6 at each split.
//!
//! **Two bounds, and the choice is measured.** [`Bound::Range`] is Hoeffding's
//! inequality on the known range `[0, 1]`, which is the bound a rate invites
//! and is nearly blind on a rare one: at 2 percent it needs a difference of
//! tenths to fire regardless of the row count, and the differences a churn
//! monitor cares about are hundredths. [`Bound::Variance`] is the empirical
//! Bernstein bound (Maurer and Pontil, *Machine Learning* 75, 2009, theorem 4)
//! on the pooled variance of the two halves, which narrows with a rare rate
//! because the series carries its own spread. It is conservative for a reason
//! worth naming: the pooled variance of two halves that differ by the change
//! being tested is inflated by that change, so the test pays for its own
//! signal. [`calibrate`] scores both on the same stationary stream and the
//! report prints both, so the choice is a number rather than a preference.
//!
//! **`delta` is chosen on a stationary stream.** The published default, 0.002,
//! is a per-test level; used unchanged on a stream it is neither the false
//! alarm rate a dashboard has nor the sensitivity it could afford.
//! [`calibrate`] sweeps a grid, measures alarms per hundred batches on a
//! stream whose rate never moves, and takes the smallest `delta` whose measured
//! rate stays inside the caller's target. Sensitivity rises as `delta` falls,
//! so the target is a constraint and the choice is its tightest point.
//!
//! **A per-event alarm is refused, with arithmetic.** At a 2 percent rate on a
//! batch of two hundred labels there are four errors a batch. The relative
//! standard error of four events is `sqrt((1-p)/(np))`, 3.50, so a single event
//! moves the reading by three and a half times its own level and no rule
//! separates that from noise. [`refusal`] prints the arithmetic, and the
//! smallest batch count at which a stated relative shift is a fixed-sample
//! reach, which is the number a dashboard would have to meet before a weekly
//! alarm could be claimed.
//!
//! Nothing here is a daemon, and none of it is a claim about a live system: it
//! is a reading a scheduler calls, with the numbers it needs to decide whether
//! the reading is worth having.

mod calibrate;

#[cfg(test)]
mod tests;

pub use calibrate::{
    calibrate, refusal, report, Calibration, CalibrationConfig, Point, Refusal, Report, Shift,
    Stream,
};

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

/// Seed for every stream in this module, so a report is reproducible.
pub const SEQUENTIAL_SEED: u64 = 0x5E11_A1DE;

/// Which bound turns a mean difference into a test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bound {
    /// Hoeffding on the known range `[0, 1]`.
    Range,
    /// Empirical Bernstein on the pooled variance of the two halves.
    Variance,
}

impl Bound {
    /// One word, for a report line.
    pub fn name(&self) -> &'static str {
        match self {
            Bound::Range => "range",
            Bound::Variance => "variance",
        }
    }
}

/// An adaptive window with a cut rule.
#[derive(Clone, Debug)]
pub struct Config {
    pub bound: Bound,
    /// Level for a cut anywhere in the stream.
    pub delta: f64,
    /// Smallest half worth testing. A split that leaves fewer observations on
    /// either side is not a change, it is one row.
    pub min_sub: usize,
    /// Safety cap on buckets. The compression keeps `O(log width)` of them, so
    /// this never binds on a real stream and exists to bound the cost.
    pub max_buckets: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bound: Bound::Variance,
            delta: 1e-6,
            min_sub: 8,
            max_buckets: 64,
        }
    }
}

impl Config {
    /// The smallest difference a balanced split at this length resolves, given
    /// the series variance. This is the resolution the reading quotes, and it
    /// is only as trustworthy as the variance handed in.
    pub fn resolves(&self, window: usize, variance: f64) -> f64 {
        let half = window / 2;
        if half < self.min_sub.max(2) {
            return 1.0;
        }
        epsilon(
            self.bound, window, half, half, variance, variance, self.delta,
        )
    }
}

/// One bucket of the exponential histogram.
#[derive(Clone, Copy, Debug)]
struct Bucket {
    size: usize,
    total: f64,
    squares: f64,
}

impl Bucket {
    fn variance(&self) -> f64 {
        if self.size < 2 {
            return 0.0;
        }
        let n = self.size as f64;
        let mean = self.total / n;
        ((self.squares - n * mean * mean) / (n - 1.0)).max(0.0)
    }
}

/// A change the window reported.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cut {
    /// Observations consumed when the cut was found, counting from one.
    pub at: usize,
    /// Observations dropped from the older side.
    pub dropped: usize,
    /// Observations kept.
    pub kept: usize,
    /// Mean of the dropped side.
    pub before: f64,
    /// Mean of the kept side.
    pub after: f64,
    /// The bound the difference had to clear.
    pub epsilon: f64,
}

impl Cut {
    /// The difference the cut reports.
    pub fn difference(&self) -> f64 {
        self.after - self.before
    }

    /// One-based index of the first observation of the newer window, which is
    /// where the change is estimated to have started.
    pub fn start(&self) -> usize {
        self.at.saturating_sub(self.dropped) + 1
    }
}

/// An adaptive window.
///
/// The window holds the most recent observations only. Every push compresses
/// the bucket list and then tests the split points inside the window; a cut
/// drops the older side, which is what makes the reading forget.
#[derive(Clone, Debug)]
pub struct Adwin {
    config: Config,
    buckets: Vec<Bucket>,
    width: usize,
    total: f64,
    squares: f64,
    clock: usize,
    cuts: usize,
}

impl Adwin {
    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
            buckets: Vec::new(),
            width: 0,
            total: 0.0,
            squares: 0.0,
            clock: 0,
            cuts: 0,
        }
    }

    /// Observations currently in the window.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Mean of the window.
    pub fn mean(&self) -> f64 {
        if self.width == 0 {
            0.0
        } else {
            self.total / self.width as f64
        }
    }

    /// Variance of the whole window.
    pub fn variance(&self) -> f64 {
        if self.width < 2 {
            return 0.0;
        }
        let n = self.width as f64;
        let mean = self.total / n;
        ((self.squares - n * mean * mean) / (n - 1.0)).max(0.0)
    }

    /// Total cuts this instance has reported.
    pub fn cuts(&self) -> usize {
        self.cuts
    }

    /// Add one observation and report a cut if the window found one.
    ///
    /// The reading is a rate, so a value outside `[0, 1]` is refused rather
    /// than rescaled: the bound needs a range, and this module cannot know the
    /// range of a column it was not told about.
    pub fn push(&mut self, value: f64) -> Result<Option<Cut>, String> {
        if !value.is_finite() || value < 0.0 || value > 1.0 {
            return Err(format!(
                "a sequential reading takes a rate in [0, 1], not {value}"
            ));
        }
        if !(self.config.delta > 0.0 && self.config.delta < 1.0) {
            return Err(format!(
                "delta must lie strictly between 0 and 1, not {}",
                self.config.delta
            ));
        }
        self.clock += 1;
        self.buckets.push(Bucket {
            size: 1,
            total: value,
            squares: value * value,
        });
        self.width += 1;
        self.total += value;
        self.squares += value * value;
        self.compress();
        let cut = self.check();
        if cut.is_some() {
            self.cuts += 1;
        }
        Ok(cut)
    }

    /// Merge equal buckets from the newest end, which is what keeps the list
    /// logarithmic in the window.
    fn compress(&mut self) {
        loop {
            let len = self.buckets.len();
            if len < 2 {
                break;
            }
            if self.buckets[len - 1].size == self.buckets[len - 2].size {
                let newer = self.buckets.pop().expect("checked length");
                let older = self.buckets.last_mut().expect("checked length");
                older.size += newer.size;
                older.total += newer.total;
                older.squares += newer.squares;
                continue;
            }
            if len > self.config.max_buckets {
                // The safety valve. Merging the two oldest buckets is the only
                // way to bound the cost without dropping data outright, and it
                // never fires on a stream of realistic length.
                let oldest = self.buckets.remove(0);
                self.buckets[0].size += oldest.size;
                self.buckets[0].total += oldest.total;
                self.buckets[0].squares += oldest.squares;
                continue;
            }
            break;
        }
    }

    /// Test every split point in the window and keep the largest passing one.
    fn check(&mut self) -> Option<Cut> {
        let n = self.width;
        if n < 2 * self.config.min_sub {
            return None;
        }
        let mut newer = Bucket {
            size: 0,
            total: 0.0,
            squares: 0.0,
        };
        let mut best: Option<(f64, usize, Cut)> = None;
        for at in (0..self.buckets.len()).rev() {
            newer.size += self.buckets[at].size;
            newer.total += self.buckets[at].total;
            newer.squares += self.buckets[at].squares;
            let older_size = n - newer.size;
            if newer.size < self.config.min_sub {
                continue;
            }
            if older_size < self.config.min_sub {
                break;
            }
            let older = Bucket {
                size: older_size,
                total: self.total - newer.total,
                squares: self.squares - newer.squares,
            };
            let before = older.total / older_size as f64;
            let after = newer.total / newer.size as f64;
            let difference = (after - before).abs();
            let bound = epsilon(
                self.config.bound,
                n,
                older_size,
                newer.size,
                older.variance(),
                newer.variance(),
                self.config.delta,
            );
            if difference < bound {
                continue;
            }
            let keep = best
                .as_ref()
                .map(|(other, _, _)| difference > *other)
                .unwrap_or(true);
            if keep {
                best = Some((
                    difference,
                    at,
                    Cut {
                        at: self.clock,
                        dropped: older_size,
                        kept: newer.size,
                        before,
                        after,
                        epsilon: bound,
                    },
                ));
            }
        }
        let (_, split, cut) = best?;
        for bucket in self.buckets.drain(..split) {
            self.width -= bucket.size;
            self.total -= bucket.total;
            self.squares -= bucket.squares;
        }
        Some(cut)
    }
}

/// The bound a split has to clear.
///
/// `delta` is the level for a cut anywhere in a stream of length `n`, so the
/// log term carries `2n`: the `n - 1` split points and both signs.
fn epsilon(
    bound: Bound,
    n: usize,
    older: usize,
    newer: usize,
    older_variance: f64,
    newer_variance: f64,
    delta: f64,
) -> f64 {
    let total = n as f64;
    let m = (older * newer) as f64 / total;
    if m <= 0.0 {
        return 1.0;
    }
    let log = (2.0 * total / delta).ln();
    match bound {
        Bound::Range => (log / (2.0 * m)).sqrt(),
        Bound::Variance => {
            let a = (older - 1) as f64;
            let b = (newer - 1) as f64;
            if a + b <= 0.0 {
                return 1.0;
            }
            let pooled = (a * older_variance + b * newer_variance) / (a + b);
            let small = older.min(newer) as f64;
            let linear = if small > 1.0 {
                7.0 * log / (3.0 * (small - 1.0))
            } else {
                1.0
            };
            (2.0 * pooled * log / m).sqrt() + linear
        }
    }
}

/// The reading on one series.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Reading {
    pub points: usize,
    pub cuts: Vec<Cut>,
    /// Observations left in the window at the end.
    pub window: usize,
    /// Mean of that window.
    pub mean: f64,
    /// Smallest difference a balanced split of the final window resolves.
    pub resolves: f64,
}

/// Run the window over a series.
pub fn reading(series: &[f64], config: &Config) -> Result<Reading, String> {
    if series.len() < 2 * config.min_sub {
        return Err(format!(
            "a sequential reading needs at least {} observations, not {}",
            2 * config.min_sub,
            series.len()
        ));
    }
    let mut adwin = Adwin::new(config);
    let mut cuts = Vec::new();
    for value in series {
        if let Some(cut) = adwin.push(*value)? {
            cuts.push(cut);
        }
    }
    Ok(Reading {
        points: series.len(),
        cuts,
        window: adwin.width(),
        mean: adwin.mean(),
        resolves: config.resolves(adwin.width(), adwin.variance()),
    })
}

/// The same cuts under a shuffled order, which is the null a single series has
/// no other way to supply.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Shuffle {
    pub observed: usize,
    pub expected: f64,
    pub high: usize,
    pub p_value: f64,
    pub permutations: usize,
    pub verdict: String,
}

/// Count cuts on the series and on shuffles of it.
///
/// A series whose order carries structure cuts more often than the same values
/// in a random order. A series that is stationary cuts as often as its own
/// shuffle. This is not a test for a change point; it is a test of whether the
/// reading on this series is about the order at all, which is the question a
/// declining hazard raises and a weekly rate does not.
pub fn shuffle(
    series: &[f64],
    config: &Config,
    permutations: usize,
    seed: u64,
) -> Result<Shuffle, String> {
    let observed = reading(series, config)?.cuts.len();
    let permutations = permutations.max(1);
    let mut rng = Rng::new(seed);
    let mut work: Vec<f64> = Vec::with_capacity(series.len());
    let mut counts = Vec::with_capacity(permutations);
    for _ in 0..permutations {
        work.clear();
        work.extend_from_slice(series);
        rng.shuffle(&mut work);
        counts.push(reading(&work, config)?.cuts.len());
    }
    let expected = counts.iter().sum::<usize>() as f64 / permutations as f64;
    let at_least = counts.iter().filter(|count| **count >= observed).count();
    let p_value = (1.0 + at_least as f64) / (1.0 + permutations as f64);
    let mut sorted = counts.clone();
    sorted.sort_unstable();
    let high = sorted[(permutations * 95 / 100).min(permutations - 1)];
    let verdict = if observed > high {
        format!(
            "the order carries structure: {observed} cuts against at most {high} in 95 percent of {} shuffles",
            crate::num::commas(permutations)
        )
    } else {
        format!(
            "the order carries no measurable structure: {observed} cuts against a shuffled mean of {}",
            crate::num::round4(expected)
        )
    };
    Ok(Shuffle {
        observed,
        expected: crate::num::round4(expected),
        high,
        p_value: crate::num::round4(p_value),
        permutations,
        verdict,
    })
}
