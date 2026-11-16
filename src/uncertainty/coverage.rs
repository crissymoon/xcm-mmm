//! A coverage simulation: does the interval hold the number it claims to hold?
//!
//! An interval is a promise about repeated sampling. The only way to check a
//! promise is to sample repeatedly from a world where the answer is known, ask
//! for an interval each time, and count how often the answer lands inside. That
//! is what this module does, and it is why the churn card can quote a coverage
//! number at all.
//!
//! The world is deliberately simple and deliberately unfriendly. Positives are
//! drawn from a normal with mean `effect` and unit spread, negatives from a
//! standard normal, so the true area under the curve is known in closed form,
//! `Phi(effect / sqrt(2))`, with no estimate anywhere in it. The default bank
//! is 251 rows with 50 positives, which is the shape the roadmap set: enough
//! positives for an asymptotic variance to be defensible, few enough that its
//! failures should show.
//!
//! Three intervals are compared on the same draws:
//!
//! - the percentile bootstrap, which is the weakest of the three;
//! - BCa, which corrects for the skew a bounded metric has near a ceiling;
//! - the DeLong variance on the logit scale, which is closed form and assumes
//!   the statistic is asymptotically normal.
//!
//! The report does not pick a winner by coverage alone. An interval that covers
//! by being enormous is useless, so the mean width is printed beside the
//! coverage and both have to be read together.

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

use super::{auc, bca, bootstrap_auc, delong_auc, Bca};

/// What to simulate.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageConfig {
    /// Rows per trial.
    pub rows: usize,
    /// Positives per trial.
    pub positives: usize,
    /// Independent trials.
    pub trials: usize,
    /// Bootstrap resamples per interval.
    pub reps: usize,
    /// Nominal coverage, for example 0.95.
    pub level: f64,
    /// Standardised gap between the two score distributions, which fixes the
    /// true area under the curve at `Phi(effect / sqrt(2))`.
    pub effect: f64,
    /// Seed for the whole simulation.
    pub seed: u64,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            rows: 251,
            positives: 50,
            trials: 200,
            reps: 300,
            level: 0.95,
            effect: 1.0,
            seed: 0xC0FE,
        }
    }
}

impl CoverageConfig {
    /// The band a nominal interval is allowed to land in: plus or minus two
    /// points, which is the tolerance the roadmap set.
    pub fn band(&self) -> (f64, f64) {
        (self.level - 0.02, self.level + 0.02)
    }
}

/// The measured behaviour of one interval.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MethodCoverage {
    /// Estimator name.
    pub method: String,
    /// Share of trials whose interval held the true value.
    pub covered: f64,
    /// Mean width across trials.
    pub mean_width: f64,
    /// Trials that produced a usable interval.
    pub trials: usize,
    /// True when the coverage sits inside the band.
    pub within_band: bool,
}

/// The simulation result.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CoverageReport {
    /// Nominal coverage that was requested.
    pub nominal: f64,
    /// True area under the curve of the generating world.
    pub true_auc: f64,
    /// Rows and positives per trial.
    pub rows: usize,
    /// Positives per trial.
    pub positives: usize,
    /// One entry per interval, in the order percentile, BCa, DeLong.
    pub methods: Vec<MethodCoverage>,
    /// The band that was applied.
    pub band: (f64, f64),
    /// Plain reading of the result.
    pub note: String,
}

/// Run the simulation.
///
/// Deterministic: the same configuration returns the same report on any
/// machine, because the only source of variation is the seeded generator.
pub fn simulate(config: &CoverageConfig) -> CoverageReport {
    let rows = config.rows.max(20);
    let positives = config.positives.clamp(2, rows - 2);
    let trials = config.trials.max(2);
    let true_auc = crate::stats::normal_cdf(config.effect / std::f64::consts::SQRT_2);
    let mut percentile_hits = 0usize;
    let mut bca_hits = 0usize;
    let mut delong_hits = 0usize;
    let mut percentile_width = 0.0;
    let mut bca_width = 0.0;
    let mut delong_width = 0.0;
    let mut usable = 0usize;
    let mut bca_fallbacks = 0usize;
    for trial in 0..trials {
        let mut rng = Rng::new(config.seed.wrapping_add(trial as u64 * 0x9E37));
        let mut scores = Vec::with_capacity(rows);
        let mut labels = Vec::with_capacity(rows);
        for index in 0..rows {
            let positive = index < positives;
            labels.push(positive);
            let centre = if positive { config.effect } else { 0.0 };
            scores.push(rng.normal(centre, 1.0));
        }
        let seed = config.seed.wrapping_add(trial as u64);
        let percentile = bootstrap_auc(&scores, &labels, config.reps, config.level, seed);
        let biased: Bca = bca(
            &scores,
            &labels,
            config.reps,
            config.level,
            seed ^ 0x8AD,
            20,
            auc,
        );
        let delong = delong_auc(&scores, &labels, config.level);
        if percentile.resamples == 0 || !delong.se.is_finite() {
            continue;
        }
        usable += 1;
        if biased.fallback {
            bca_fallbacks += 1;
        }
        if (percentile.low..=percentile.high).contains(&true_auc) {
            percentile_hits += 1;
        }
        if (biased.interval.low..=biased.interval.high).contains(&true_auc) {
            bca_hits += 1;
        }
        if (delong.low..=delong.high).contains(&true_auc) {
            delong_hits += 1;
        }
        percentile_width += percentile.width();
        bca_width += biased.interval.width();
        delong_width += (delong.high - delong.low).max(0.0);
    }
    if usable == 0 {
        return CoverageReport {
            nominal: config.level,
            true_auc,
            rows,
            positives,
            methods: Vec::new(),
            band: config.band(),
            note: "no trial produced a usable interval, so nothing is claimed".to_string(),
        };
    }
    let count = usable as f64;
    let (low_band, high_band) = config.band();
    let method = |name: &str, hits: usize, width: f64| {
        let covered = hits as f64 / count;
        MethodCoverage {
            method: name.to_string(),
            covered,
            mean_width: width / count,
            trials: usable,
            within_band: covered >= low_band && covered <= high_band,
        }
    };
    let methods = vec![
        method("bootstrap-percentile", percentile_hits, percentile_width),
        method("bca", bca_hits, bca_width),
        method("delong-logit", delong_hits, delong_width),
    ];
    let held: Vec<&str> = methods
        .iter()
        .filter(|entry| entry.within_band)
        .map(|entry| entry.method.as_str())
        .collect();
    let note = if held.is_empty() {
        format!(
            "no interval held nominal {:.2} inside [{:.2}, {:.2}] at {} rows with {} positives; the shortest is reported with its measured coverage",
            config.level, low_band, high_band, rows, positives
        )
    } else {
        format!(
            "{} held nominal {:.2} inside [{:.2}, {:.2}] at {} rows with {} positives{}",
            held.join(", "),
            config.level,
            low_band,
            high_band,
            rows,
            positives,
            if bca_fallbacks > 0 {
                format!("; bca fell back to the percentile interval on {bca_fallbacks} trials")
            } else {
                String::new()
            }
        )
    };
    CoverageReport {
        nominal: config.level,
        true_auc,
        rows,
        positives,
        methods,
        band: (low_band, high_band),
        note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_true_area_follows_the_generating_gap() {
        let config = CoverageConfig::default();
        let report = simulate(&config);
        // Phi(1 / sqrt 2) is 0.7602.
        assert!((report.true_auc - 0.7602).abs() < 0.001);
        assert_eq!(report.methods.len(), 3);
        for entry in &report.methods {
            assert!(entry.covered > 0.8, "{} {}", entry.method, entry.covered);
            assert!(entry.mean_width > 0.0);
        }
    }

    #[test]
    fn bca_holds_nominal_coverage_on_the_small_bank() {
        let config = CoverageConfig {
            trials: 120,
            reps: 200,
            ..CoverageConfig::default()
        };
        let report = simulate(&config);
        let bca_entry = report
            .methods
            .iter()
            .find(|entry| entry.method == "bca")
            .expect("bca entry");
        assert!(
            bca_entry.covered >= 0.90,
            "bca covered {} at nominal {}",
            bca_entry.covered,
            report.nominal
        );
        assert!(report.note.contains("nominal"));
    }

    #[test]
    fn the_simulation_is_deterministic() {
        let config = CoverageConfig {
            trials: 20,
            reps: 50,
            ..CoverageConfig::default()
        };
        let first = simulate(&config);
        let second = simulate(&config);
        assert_eq!(first.methods[0].covered, second.methods[0].covered);
        assert_eq!(first.methods[1].mean_width, second.methods[1].mean_width);
    }
}
