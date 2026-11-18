//! Calibration of the level, the streams it is measured on, and the refusals.

use super::{reading, shuffle, Adwin, Bound, Config, Reading, Shuffle, SEQUENTIAL_SEED};

use serde::{Deserialize, Serialize};

use crate::rng::Rng;
use crate::stats::normal_quantile;
use crate::uncertainty::wilson;

/// The shape of a stream used for calibration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Stream {
    /// Rate per batch.
    pub rate: f64,
    /// Labels in a batch, so a batch reading is a binomial draw rather than a
    /// single label. A per-event stream would be `batch_rows = 1`, and the
    /// calibration reports what that costs.
    pub batch_rows: usize,
    /// Batches simulated.
    pub batches: usize,
    /// Independent runs.
    pub trials: usize,
}

impl Default for Stream {
    fn default() -> Self {
        Self {
            rate: 0.02,
            batch_rows: 200,
            batches: 1000,
            // A hundred streams is what it takes for an upper bound on a share
            // of zero to sit under a five percent budget: forty trials leave
            // 0.088, which cannot establish the level at all.
            trials: 100,
        }
    }
}

/// A stream that moves at a stated batch.
///
/// The shifted stream gets its own length and its own trial count, because the
/// horizon a change needs is much longer than the horizon a false alarm needs
/// and the two measurements should not be forced to share one.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Shift {
    /// Rate after the change.
    pub to: f64,
    /// Batch at which the rate changes.
    pub at: usize,
    /// Batches simulated.
    pub batches: usize,
    /// Independent runs.
    pub trials: usize,
}

impl Default for Shift {
    fn default() -> Self {
        Self {
            to: 0.06,
            at: 1500,
            batches: 4000,
            trials: 20,
        }
    }
}

/// How the calibration is run.
#[derive(Clone, Debug)]
pub struct CalibrationConfig {
    pub stationary: Stream,
    pub shift: Option<Shift>,
    /// Grid of `delta`, ascending.
    pub grid: Vec<f64>,
    /// Share of stationary streams allowed to raise a cut. This is the unit a
    /// uniform bound promises in, so it is the unit a budget belongs in.
    pub target_share: f64,
    pub min_sub: usize,
    pub permutations: usize,
    pub seed: u64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            stationary: Stream::default(),
            shift: Some(Shift::default()),
            grid: vec![0.001, 0.01, 0.05, 0.2, 0.5],
            target_share: 0.05,
            min_sub: 8,
            permutations: 200,
            seed: SEQUENTIAL_SEED,
        }
    }
}

/// One grid point, scored on both streams.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Point {
    pub delta: f64,
    /// Stationary streams run.
    pub streams: usize,
    /// Streams that raised at least one cut.
    pub raised: usize,
    /// Share of streams that raised a cut.
    pub share: f64,
    /// Wilson interval on that share, because a measured share without one is
    /// the mistake this repository keeps having to fix.
    pub low: f64,
    pub high: f64,
    pub batches: usize,
    pub mean_window: f64,
    /// Median batches from the change to the first cut, when any trial found
    /// one.
    pub delay: Option<f64>,
    /// Share of trials that found a cut at all.
    pub detected: f64,
}

/// The calibration.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Calibration {
    pub points: Vec<Point>,
    pub chosen: usize,
    pub reason: String,
    pub target_share: f64,
}

/// Score a grid of `delta` on a stationary stream and on a shifted one.
pub fn calibrate(config: &CalibrationConfig) -> Calibration {
    let mut points = Vec::with_capacity(config.grid.len());
    for (at, delta) in config.grid.iter().enumerate() {
        let adwin = Config {
            bound: Bound::Variance,
            delta: *delta,
            min_sub: config.min_sub,
            max_buckets: 64,
        };
        let mut rng = Rng::stream(config.seed, at as u64 + 1);
        let mut raised = 0usize;
        let mut window = 0.0;
        for _ in 0..config.stationary.trials {
            let mut detector = Adwin::new(&adwin);
            let mut fired = false;
            for _ in 0..config.stationary.batches {
                let value = batch(
                    &mut rng,
                    config.stationary.rate,
                    config.stationary.batch_rows,
                );
                if detector.push(value).ok().flatten().is_some() {
                    fired = true;
                }
            }
            if fired {
                raised += 1;
            }
            window += detector.width() as f64 / config.stationary.trials as f64;
        }
        let (low, high) = wilson(raised, config.stationary.trials, 0.95);
        let (delay, detected) = match config.shift {
            None => (None, 0.0),
            Some(shift) => shifted(config, &adwin, shift),
        };
        points.push(Point {
            delta: *delta,
            streams: config.stationary.trials,
            raised,
            share: crate::num::round4(raised as f64 / config.stationary.trials.max(1) as f64),
            low: crate::num::round4(low),
            high: crate::num::round4(high),
            batches: config.stationary.batches,
            mean_window: crate::num::round4(window),
            delay: delay.map(crate::num::round4),
            detected: crate::num::round4(detected),
        });
    }
    // A larger delta is a looser bound: it cuts sooner and it cuts more often,
    // so sensitivity and the false-alarm share rise together. A level has to
    // hold twice, though: in its own promise, that at most `delta` of all
    // streams ever raise a cut, and in the measurement that the promise is not
    // being violated. The measurement alone cannot rank this grid, because a
    // valid uniform bound on a bounded rate is conservative enough that none
    // of these levels fires at all; what it can do is rule out a level whose
    // promise would break the budget, which is what the grid points above it
    // do. So the choice is the largest level that clears both.
    let admissible =
        |point: &Point| point.delta <= config.target_share && point.high <= config.target_share;
    let (chosen, reason) = match points.iter().rposition(admissible) {
        Some(at) => {
            let point = &points[at];
            (
                at,
                format!(
                    "delta {} is the largest level that promises at most {:.2} of streams in error and is measured inside that, with {} of {} stationary streams raising a cut and an upper bound of {:.3}{}",
                    point.delta,
                    config.target_share,
                    point.raised,
                    point.streams,
                    point.high,
                    if at + 1 < points.len() {
                        "; the looser grid points are excluded by their own promise, which the measurement cannot rank at this length"
                    } else {
                        ""
                    }
                ),
            )
        }
        // No point clears both conditions, so the fallback is the quietest end
        // of the grid rather than the loosest: a level that cannot be
        // established should err towards fewer false alarms.
        None => (
            0,
            format!(
                "no grid point both promises and measures inside {:.2} streams in error at {} trials, so the quietest level, {}, is used and the budget is not established at this length",
                config.target_share,
                config.stationary.trials,
                points.first().map(|point| point.delta).unwrap_or(0.0)
            ),
        ),
    };
    Calibration {
        points,
        chosen,
        reason,
        target_share: config.target_share,
    }
}

/// Run trial batch streams that change at `shift.at`.
fn shifted(config: &CalibrationConfig, adwin: &Config, shift: Shift) -> (Option<f64>, f64) {
    let stream = config.stationary;
    let mut rng = Rng::stream(config.seed, 0x51F7);
    let mut delays = Vec::new();
    for _ in 0..shift.trials {
        let mut detector = Adwin::new(adwin);
        let mut found = None;
        for batch_at in 0..shift.batches {
            let rate = if batch_at < shift.at {
                stream.rate
            } else {
                shift.to
            };
            let value = batch(&mut rng, rate, stream.batch_rows);
            if let Ok(Some(cut)) = detector.push(value) {
                if cut.at > shift.at && found.is_none() {
                    found = Some((cut.at - shift.at) as f64);
                }
            }
        }
        if let Some(delay) = found {
            delays.push(delay);
        }
    }
    let detected = delays.len() as f64 / shift.trials.max(1) as f64;
    let delay = if delays.is_empty() {
        None
    } else {
        delays.sort_by(f64::total_cmp);
        Some(delays[delays.len() / 2])
    };
    (delay, detected)
}

/// One batch reading: a binomial draw over `rows` labels.
///
/// The draw walks the geometric gaps between successes rather than flipping
/// `rows` coins, which is exact and costs `O(hits)` instead of `O(rows)`. At a
/// two percent rate over two hundred labels that is four draws rather than two
/// hundred, which is what makes a calibration at thousands of batches cheap
/// enough to ship.
fn batch(rng: &mut Rng, rate: f64, rows: usize) -> f64 {
    if rows == 0 || rate <= 0.0 {
        return 0.0;
    }
    if rate >= 1.0 {
        return 1.0;
    }
    let log = (1.0 - rate).ln();
    let mut hits = 0usize;
    let mut at = 0usize;
    loop {
        let gap = (rng.open_f64().ln() / log).floor() as usize + 1;
        at += gap;
        if at > rows {
            break;
        }
        hits += 1;
    }
    hits as f64 / rows as f64
}

/// What a per-event alarm would need, and why it is refused.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Refusal {
    pub rate: f64,
    pub batch_rows: usize,
    /// Expected events in one batch.
    pub events: f64,
    /// Standard error of one batch's rate, relative to the rate itself.
    pub relative_se: f64,
    /// The same for a batch of one label, which is what a per-event rule sees.
    pub event_se: f64,
    /// The relative shift the batch arithmetic is quoted at.
    pub shift: f64,
    /// Rows a fixed-sample test needs to resolve that shift at the stated
    /// power, from the two-proportion formula.
    pub rows: f64,
    /// The same in batches of the stated size.
    pub batches: f64,
    pub verdict: String,
}

/// The arithmetic a per-event alert would have to beat.
///
/// The rows figure is the fixed-sample two-proportion size at level `level` and
/// power `power` for a rate moving by `shift` proportionally. It is a floor on
/// what any rule could need, and a sequential rule that must hold at every
/// stopping time needs more, which the calibration measures rather than
/// asserts.
pub fn refusal(rate: f64, batch_rows: usize, shift: f64, level: f64, power: f64) -> Refusal {
    let events = rate * batch_rows as f64;
    let relative = |rows: f64| -> f64 {
        if rows <= 0.0 || rate <= 0.0 {
            f64::INFINITY
        } else {
            ((1.0 - rate) / (rows * rate)).sqrt()
        }
    };
    let relative_se = relative(batch_rows as f64);
    let event_se = relative(1.0);
    let moved = rate * (1.0 + shift);
    let p = rate.clamp(1e-9, 1.0 - 1e-9);
    let q = moved.clamp(1e-9, 1.0 - 1e-9);
    let a = normal_quantile(1.0 - level / 2.0);
    let b = normal_quantile(power);
    let difference = (q - p).abs();
    let needed = if difference <= 0.0 {
        f64::INFINITY
    } else {
        ((a * (2.0 * p * (1.0 - p)).sqrt() + b * (p * (1.0 - p) + q * (1.0 - q)).sqrt())
            / difference)
            .powi(2)
    };
    let batches = needed / batch_rows.max(1) as f64;
    let verdict = format!(
        "a per-event alarm is refused: one label at a rate of {:.2} carries a relative standard error of {:.0} percent, and a batch of {} carries {:.0} percent, so a single event is not a signal; a {:.0} percent shift needs at least {} rows, {:.1} batches, before even a fixed-sample test resolves it",
        rate,
        100.0 * event_se,
        crate::num::commas(batch_rows),
        100.0 * relative_se,
        100.0 * shift,
        crate::num::commas(needed.ceil() as usize),
        batches
    );
    Refusal {
        rate,
        batch_rows,
        events: crate::num::round4(events),
        relative_se: crate::num::round4(relative_se),
        event_se: crate::num::round4(event_se),
        shift,
        rows: crate::num::round4(needed),
        batches: crate::num::round4(batches),
        verdict,
    }
}

/// The whole reading: calibration, the series, the shuffle null and the
/// refusals.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Report {
    pub bound: String,
    pub delta: f64,
    pub target_share: f64,
    pub stream: Stream,
    pub shift: Option<Shift>,
    pub calibration: Calibration,
    pub reading: Reading,
    pub shuffle: Shuffle,
    pub refusal: Refusal,
    pub note: String,
    /// The fixed-sample floor beside the measured sequential delay, which is
    /// the price of a rule that has to hold at every stopping time.
    pub comparison: String,
    /// What shape the calibration was run at, and why that is not the series'.
    pub assumption: String,
}

impl Report {
    /// Whether the bound could see a change as large as the level it is
    /// watching. Below this the absence of a cut says nothing about the series.
    pub fn informative(&self) -> bool {
        self.reading.resolves <= self.reading.mean.max(1e-9)
    }

    /// The shuffle reading, with its informativeness stated.
    pub fn shuffle_line(&self) -> String {
        if self.informative() {
            self.shuffle.verdict.clone()
        } else {
            format!(
                "not informative: at a resolution of {:.4} against a window mean of {:.4}, a cut needs a jump of {:.1} times the level, so the null is the bound's arithmetic rather than evidence about the series",
                self.reading.resolves,
                self.reading.mean,
                self.reading.resolves / self.reading.mean.max(1e-9)
            )
        }
    }
}

impl Stream {
    /// A stream of `rows` labels per batch.
    pub fn new(rate: f64, batch_rows: usize) -> Self {
        Self {
            rate,
            batch_rows,
            ..Stream::default()
        }
    }
}

/// Calibrate, read the series, and say what the reading is and is not.
pub fn report(series: &[f64], config: &CalibrationConfig) -> Result<Report, String> {
    let calibration = calibrate(config);
    let chosen = calibration
        .points
        .get(calibration.chosen)
        .ok_or("the calibration produced no grid points")?;
    let adwin = Config {
        bound: Bound::Variance,
        delta: chosen.delta,
        min_sub: config.min_sub,
        max_buckets: 64,
    };
    let reading = reading(series, &adwin)?;
    let shuffled = shuffle(series, &adwin, config.permutations, config.seed ^ 0x9E37)?;
    let moved = config
        .shift
        .map(|shift| shift.to)
        .unwrap_or(config.stationary.rate);
    let refusal = refusal(
        config.stationary.rate,
        config.stationary.batch_rows,
        moved / config.stationary.rate.max(1e-9) - 1.0,
        0.05,
        0.80,
    );
    let ratio = reading.resolves / reading.mean.max(1e-9);
    let note = if reading.cuts.is_empty() {
        format!(
            "no cut in {} batches, where the bound resolves a difference of {:.4}, which is {ratio:.1} times the window mean of {:.4}: at this length the absence of a cut is arithmetic rather than evidence",
            reading.points, reading.resolves, reading.mean
        )
    } else {
        format!(
            "{} cuts in {} batches at a resolution of {:.4}, which is {ratio:.1} times the window mean of {:.4}",
            reading.cuts.len(),
            reading.points,
            reading.resolves,
            reading.mean
        )
    };
    let comparison = match chosen.delay {
        Some(delay) if delay.is_finite() && delay > 0.0 => format!(
            "a fixed-sample test resolves the same shift in {} rows, {:.1} batches; the sequential rule takes {delay:.0} batches, {:.0} times as long, which is the price of a bound that has to hold at every stopping time",
            crate::num::commas(refusal.rows.ceil() as usize),
            refusal.batches,
            delay / refusal.batches.max(0.1)
        ),
        _ => "the shifted stream never cut inside its horizon, so the sequential cost is not measured here"
            .to_string(),
    };
    Ok(Report {
        bound: adwin.bound.name().to_string(),
        delta: chosen.delta,
        target_share: calibration.target_share,
        stream: config.stationary,
        shift: config.shift,
        calibration,
        reading,
        shuffle: shuffled,
        refusal,
        note,
        comparison,
        assumption: format!(
            "The level is calibrated at {} labels a batch at a rate of {:.2}, which is the operating shape a weekly dashboard has; the series is then read at the level that calibration chose, not at its own row counts.",
            crate::num::commas(config.stationary.batch_rows),
            config.stationary.rate
        ),
    })
}
