//! How much evidence a fitted bank actually supplies.
//!
//! A sample size is not a support statement. Two numbers make the boundary
//! visible: events per feature, which says whether a logistic has enough
//! positives to estimate its coefficients, and a learning curve, which says
//! whether more rows of the same kind would move the held-out score. A curve
//! that is still rising is a data-limited bank; a flat curve is a signal that
//! the ceiling is the features or the operator, not the row count.
//!
//! The curve is deliberately modest: six fractions, a split of its own so the
//! diagnostic never scores itself on the folds the headline was chosen on, and
//! the same model as the headline. Four points on the scoring folds could not
//! tell a ceiling from a blind design, and a curve measured on the same rows it
//! was selected on is not a measurement. Neither reading transports to a new
//! operator.
//!
//! The ceiling it is compared against is the label audit's, which is an
//! accuracy, so the curve carries an accuracy beside its area. Comparing an
//! area with an accuracy would be a scale error dressed as a finding.

use serde::{Deserialize, Serialize};

/// The training fractions a curve is drawn at.
///
/// Six, because four points cannot distinguish a curve that has flattened from
/// one measured too coarsely, and the first point is low enough that the rise
/// is visible rather than assumed.
pub const FRACTIONS: [f64; 6] = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00];

/// One point on a learning curve.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LearningPoint {
    pub fraction: f64,
    pub rows: usize,
    pub auc: f64,
    /// Standard error of [`LearningPoint::auc`] across the curve's folds.
    /// Absent on a curve drawn before the reading was reported.
    #[serde(default)]
    pub se: f64,
    /// Accuracy at the cut the fold chose on its own training rows, on the
    /// same scale as the label audit's ceiling. Absent before the ceiling was
    /// compared.
    #[serde(default)]
    pub accuracy: f64,
    /// Standard error used for the ceiling comparison: the wider of the spread
    /// across the curve's folds and the binomial error of the accuracy on the
    /// pooled test rows. A fold spread alone measures how sensitive the reading
    /// is to the split, not how precisely it is measured.
    #[serde(default)]
    pub accuracy_se: f64,
}

/// The curve against the label audit's ceiling.
#[derive(Clone, Debug, PartialEq)]
pub struct CeilingReading {
    /// Accuracy at the largest fraction.
    pub curve: f64,
    /// Standard error of that number across the curve's folds.
    pub se: f64,
    /// The accuracy a model that reproduced the true labels would score.
    pub ceiling: f64,
    /// Ceiling minus curve. Positive means room remains.
    pub gap: f64,
    /// Accuracy gained per doubling of rows at the last step.
    pub doubling: f64,
    /// One sentence a card can print without further arithmetic.
    pub verdict: &'static str,
}

/// Positives per estimated coefficient.
pub fn events_per_feature(positives: usize, features: usize) -> f64 {
    if features == 0 {
        return 0.0;
    }
    positives as f64 / features as f64
}

/// A short reading of the events-per-feature rule of thumb.
pub fn support_level(events: f64) -> &'static str {
    if events >= 20.0 {
        "Adequate for a logistic on this feature set"
    } else if events >= 10.0 {
        "Thin: coefficients will move between snapshots"
    } else {
        "Too thin for a stable logistic; treat the ranking as a prior"
    }
}

/// The change per unit of fraction between the last two points.
fn per_fraction(points: &[LearningPoint], value: impl Fn(&LearningPoint) -> f64) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let a = &points[points.len() - 2];
    let b = &points[points.len() - 1];
    if b.fraction <= a.fraction {
        return 0.0;
    }
    (value(b) - value(a)) / (b.fraction - a.fraction)
}

/// The change per doubling of rows between the last two points.
fn per_doubling(points: &[LearningPoint], value: impl Fn(&LearningPoint) -> f64) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let a = &points[points.len() - 2];
    let b = &points[points.len() - 1];
    if a.fraction <= 0.0 || b.fraction <= a.fraction {
        return 0.0;
    }
    let doublings = (b.fraction / a.fraction).log2();
    (value(b) - value(a)) / doublings
}

/// The slope between the last two learning-curve points, in AUC per fraction.
///
/// Positive means the held-out score was still rising at the largest fraction,
/// which is the signature of a bank that is limited by rows rather than by the
/// hypothesis class.
pub fn slope(points: &[LearningPoint]) -> f64 {
    per_fraction(points, |point| point.auc)
}

/// The slope as a percentage of AUC per doubling of rows.
pub fn doubling_gain(points: &[LearningPoint]) -> f64 {
    per_doubling(points, |point| point.auc) * 100.0
}

/// The accuracy gained per doubling of rows at the last step, in points.
pub fn accuracy_doubling_gain(points: &[LearningPoint]) -> f64 {
    per_doubling(points, |point| point.accuracy) * 100.0
}

/// A fitted learning-curve asymptote, and what it costs to believe it.
///
/// The form is `a - b * n^-c`, the inverse-power law a curve of this kind is
/// usually written as (Cortes et al., NeurIPS 1993). Three parameters on six
/// points is thin, so the standard error is reported and the card prints the
/// last measured point beside the asymptote: when the two are not separated,
/// the honest reading is that the curve has flattened, not that the asymptote
/// has been discovered.
#[derive(Clone, Debug, PartialEq)]
pub struct Asymptote {
    /// The limit the fit extrapolates to.
    pub asymptote: f64,
    /// The exponent `c`, larger for a curve that saturates sooner.
    pub rate: f64,
    /// Standard error of the intercept, from the linear fit at the chosen rate.
    /// Optimistic on six points and reported as such.
    pub se: f64,
    /// Lower end of the 95 percent interval.
    pub low: f64,
    /// Upper end.
    pub high: f64,
    /// Accuracy at the largest fraction, for comparison.
    pub last: f64,
    /// One sentence a card can print.
    pub note: &'static str,
}

/// Fit the inverse-power law to the accuracy column.
///
/// `None` below six points, which is a refusal rather than a fit: two or three
/// parameters on four points is interpolation with a straight face. Also `None`
/// when the rows are not positive and increasing, because then `n^-c` is not a
/// curve.
pub fn asymptote(points: &[LearningPoint]) -> Option<Asymptote> {
    if points.len() < FRACTIONS.len() {
        return None;
    }
    if points
        .windows(2)
        .any(|pair| pair[1].rows <= pair[0].rows || pair[0].rows == 0)
    {
        return None;
    }
    if points.iter().any(|point| point.accuracy <= 0.0) {
        return None;
    }
    let values: Vec<f64> = points.iter().map(|point| point.accuracy).collect();
    let rows: Vec<f64> = points.iter().map(|point| point.rows as f64).collect();
    let mut best: Option<(f64, f64, f64, f64, f64)> = None;
    // A grid over the exponent: for a fixed exponent the model is a straight
    // line in n^-c, so the intercept and slope are a two-parameter least
    // squares fit and only the exponent needs searching.
    for step in 5..=300 {
        let rate = step as f64 / 100.0;
        let xs: Vec<f64> = rows.iter().map(|n| n.powf(-rate)).collect();
        let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
        let mean_y = values.iter().sum::<f64>() / values.len() as f64;
        let sxx: f64 = xs.iter().map(|x| (x - mean_x) * (x - mean_x)).sum();
        if sxx <= 0.0 {
            continue;
        }
        let sxy: f64 = xs
            .iter()
            .zip(&values)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let slope = sxy / sxx;
        let intercept = mean_y - slope * mean_x;
        // A positive `b` means the curve rises toward the limit.
        if slope >= 0.0 {
            continue;
        }
        let sse: f64 = xs
            .iter()
            .zip(&values)
            .map(|(x, y)| {
                let fitted = intercept + slope * x;
                (y - fitted) * (y - fitted)
            })
            .sum();
        if best.map(|(best_sse, ..)| sse < best_sse).unwrap_or(true) {
            best = Some((sse, intercept, rate, mean_x, sxx));
        }
    }
    let last = &points[points.len() - 1];
    let Some((sse, intercept, rate, mean_x, sxx)) = best else {
        // No exponent fits a rising curve. Reporting the last point as the
        // asymptote is the honest reading: there is nothing above it to fit.
        return Some(Asymptote {
            asymptote: last.accuracy,
            rate: 0.0,
            se: last.accuracy_se,
            low: last.accuracy - 1.96 * last.accuracy_se,
            high: last.accuracy + 1.96 * last.accuracy_se,
            last: last.accuracy,
            note: "The curve is flat or falling, so there is no rising power law to fit; the last measured point is reported in its place.",
        });
    };
    let freedom = (points.len() as f64 - 3.0).max(1.0);
    let variance = sse / freedom;
    let se = (variance * (1.0 / points.len() as f64 + mean_x * mean_x / sxx)).sqrt();
    // A fit whose residuals are larger than the reading's own error is not a
    // description of these points, however small its standard error looks.
    let rmse = (sse / points.len() as f64).sqrt();
    let tolerance = 2.0 * last.accuracy_se.max(1e-9);
    let note = if rmse > tolerance {
        "The power law does not describe these points: its residuals are larger than the reading's own error, so the extrapolated limit is reported and not used."
    } else if (intercept - last.accuracy).abs() <= 2.0 * se.max(1e-9) {
        "The asymptote is not separated from the last measured point, so the curve has flattened rather than reached a discovered limit."
    } else {
        "The asymptote is separated from the last measured point, which is what a curve still buying accuracy looks like."
    };
    Some(Asymptote {
        asymptote: intercept,
        rate,
        se,
        low: intercept - 1.96 * se,
        high: intercept + 1.96 * se,
        last: last.accuracy,
        note,
    })
}

/// Read the curve against the label audit's ceiling.
///
/// `None` when there is no curve or no ceiling, because a comparison against a
/// number the audit refused is not a weaker finding, it is no finding.
///
/// The three verdicts are separate on purpose. A curve at the ceiling cannot be
/// improved by rows of the same kind. A curve below the ceiling and still
/// rising says rows would help. A curve below the ceiling and flat says the
/// remaining gap is the features or the operator, and more rows of this kind
/// buys nothing. Each needs two standard errors to clear, and a one standard
/// error difference is reported as unresolved rather than promoted.
pub fn against_ceiling(points: &[LearningPoint], ceiling: f64) -> Option<CeilingReading> {
    let last = points.last()?;
    if ceiling <= 0.0 || last.accuracy <= 0.0 {
        return None;
    }
    let gap = ceiling - last.accuracy;
    let doubling = accuracy_doubling_gain(points);
    let error = if last.accuracy_se > 0.0 {
        last.accuracy_se
    } else {
        last.se
    }
    .max(1e-9);
    let verdict = if gap.abs() <= 2.0 * error {
        "The gap is unresolved: it sits inside two standard errors of the reading, which is not the same as being absent."
    } else if gap > 0.0 && doubling > 2.0 * error * 100.0 {
        "The curve is below the ceiling and still rising, so more rows of this kind would help."
    } else if gap > 0.0 {
        "The curve is below the ceiling but flat, so the remaining gap is the features or the operator, not the row count."
    } else {
        "The curve is above the estimated ceiling, which means the reading is measuring the labels rather than the construct."
    };
    Some(CeilingReading {
        curve: last.accuracy,
        se: error,
        ceiling,
        gap,
        doubling,
        verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(fraction: f64, rows: usize, auc: f64, accuracy: f64) -> LearningPoint {
        LearningPoint {
            fraction,
            rows,
            auc,
            se: 0.01,
            accuracy,
            accuracy_se: 0.01,
        }
    }

    #[test]
    fn events_per_feature_refuses_a_zero_denominator() {
        assert_eq!(events_per_feature(10, 0), 0.0);
        assert!((events_per_feature(100, 10) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn the_slope_reads_the_last_two_points() {
        let points = vec![
            point(0.25, 25, 0.6, 0.55),
            point(0.5, 50, 0.7, 0.60),
            point(1.0, 100, 0.8, 0.65),
        ];
        assert!((slope(&points) - 0.2).abs() < 1e-9);
        assert!((doubling_gain(&points) - 10.0).abs() < 1e-9);
        assert!((accuracy_doubling_gain(&points) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn a_flat_curve_has_no_doubling_gain() {
        let points = vec![point(0.5, 50, 0.8, 0.7), point(1.0, 100, 0.8, 0.7)];
        assert_eq!(doubling_gain(&points), 0.0);
        assert_eq!(accuracy_doubling_gain(&points), 0.0);
    }

    #[test]
    fn the_fractions_are_six_and_rising() {
        assert_eq!(FRACTIONS.len(), 6);
        assert!(FRACTIONS.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(FRACTIONS[0], 0.10);
        assert_eq!(FRACTIONS[5], 1.00);
    }

    #[test]
    fn a_curve_at_the_ceiling_says_rows_cannot_help() {
        let points = vec![point(0.5, 50, 0.80, 0.876), point(1.0, 100, 0.80, 0.878)];
        let reading = against_ceiling(&points, 0.879).unwrap();
        assert!((reading.gap - 0.001).abs() < 1e-9);
        assert!(reading.verdict.contains("unresolved"));
    }

    #[test]
    fn the_ceiling_error_is_the_wider_of_the_two_readings() {
        let mut last = point(1.0, 100, 0.80, 0.790);
        last.accuracy_se = 0.012;
        let reading = against_ceiling(&[point(0.5, 50, 0.80, 0.780), last], 0.879).unwrap();
        assert!((reading.se - 0.012).abs() < 1e-9);
        // A gap under twice that error is unresolved even though the fold
        // spread alone would have called it separated.
        let mut near = point(1.0, 100, 0.80, 0.870);
        near.accuracy_se = 0.012;
        let reading = against_ceiling(&[point(0.5, 50, 0.86, 0.860), near], 0.879).unwrap();
        assert!(reading.verdict.contains("unresolved"));
    }

    #[test]
    fn a_curve_below_the_ceiling_and_rising_says_rows_would_help() {
        let points = vec![point(0.5, 50, 0.70, 0.700), point(1.0, 100, 0.80, 0.760)];
        let reading = against_ceiling(&points, 0.879).unwrap();
        assert!(reading.gap > 0.0);
        assert!((reading.doubling - 6.0).abs() < 1e-9);
        assert!(reading.verdict.contains("would help"));
    }

    #[test]
    fn a_curve_below_the_ceiling_and_flat_names_the_features() {
        let points = vec![point(0.5, 50, 0.70, 0.700), point(1.0, 100, 0.70, 0.702)];
        let reading = against_ceiling(&points, 0.879).unwrap();
        assert!(reading.verdict.contains("features or the operator"));
    }

    #[test]
    fn a_curve_above_the_ceiling_is_reading_the_labels() {
        let points = vec![point(0.5, 50, 0.90, 0.900), point(1.0, 100, 0.90, 0.905)];
        let reading = against_ceiling(&points, 0.879).unwrap();
        assert!(reading.gap < 0.0);
        assert!(reading.verdict.contains("measuring the labels"));
    }

    #[test]
    fn a_comparison_without_a_ceiling_is_refused() {
        let points = vec![point(0.5, 50, 0.8, 0.7), point(1.0, 100, 0.8, 0.7)];
        assert!(against_ceiling(&points, 0.0).is_none());
        assert!(against_ceiling(&[], 0.879).is_none());
        // A curve drawn before the accuracy column existed reads as no ceiling
        // comparison rather than as a comparison with zero.
        let legacy = vec![LearningPoint {
            fraction: 1.0,
            rows: 100,
            auc: 0.8,
            se: 0.0,
            accuracy: 0.0,
            accuracy_se: 0.0,
        }];
        assert!(against_ceiling(&legacy, 0.879).is_none());
    }

    #[test]
    fn the_curve_fit_recovers_a_known_power_law() {
        // accuracy = 0.9 - 0.2 * n^-0.5, which is a curve that saturates.
        let points: Vec<LearningPoint> = [100.0, 200.0, 350.0, 500.0, 750.0, 1000.0]
            .iter()
            .enumerate()
            .map(|(at, n)| {
                let n: f64 = *n;
                let accuracy = 0.9 - 0.2 * n.powf(-0.5);
                LearningPoint {
                    fraction: FRACTIONS[at],
                    rows: n as usize,
                    auc: accuracy,
                    se: 0.002,
                    accuracy,
                    accuracy_se: 0.002,
                }
            })
            .collect();
        let reading = asymptote(&points).unwrap();
        assert!(
            (reading.asymptote - 0.9).abs() < 0.005,
            "asymptote {}",
            reading.asymptote
        );
        assert!((reading.rate - 0.5).abs() < 0.06, "rate {}", reading.rate);
        assert!(reading.low < reading.asymptote && reading.asymptote < reading.high);
        assert!(reading.note.contains("separated"));
    }

    #[test]
    fn a_flat_curve_has_an_asymptote_it_cannot_separate() {
        let points: Vec<LearningPoint> = [500.0, 1000.0, 1800.0, 2600.0, 3900.0, 5200.0]
            .iter()
            .enumerate()
            .map(|(at, n)| LearningPoint {
                fraction: FRACTIONS[at],
                rows: *n as usize,
                auc: 0.867,
                se: 0.002,
                accuracy: 0.790,
                accuracy_se: 0.005,
            })
            .collect();
        let reading = asymptote(&points).unwrap();
        assert!((reading.asymptote - 0.790).abs() < 1e-9);
        assert!(reading.note.contains("flat or falling"), "{}", reading.note);
        assert!((reading.last - 0.790).abs() < 1e-9);
    }

    #[test]
    fn a_noisy_curve_refuses_the_extrapolation_it_cannot_describe() {
        // A curve that wobbles by more than its own error has no power law in
        // it, and a small standard error on a bad fit is not a finding.
        let accuracies = [0.800, 0.740, 0.810, 0.750, 0.800, 0.790];
        let points: Vec<LearningPoint> = [500.0, 1000.0, 1800.0, 2600.0, 3900.0, 5200.0]
            .iter()
            .zip(accuracies)
            .enumerate()
            .map(|(at, (n, accuracy))| LearningPoint {
                fraction: FRACTIONS[at],
                rows: *n as usize,
                auc: accuracy,
                se: 0.002,
                accuracy,
                accuracy_se: 0.005,
            })
            .collect();
        let reading = asymptote(&points).unwrap();
        assert!(
            reading.note.contains("does not describe"),
            "{}",
            reading.note
        );
    }

    #[test]
    fn the_curve_fit_refuses_fewer_than_six_points_or_a_broken_axis() {
        let four: Vec<LearningPoint> = [0.25, 0.5, 0.75, 1.0]
            .iter()
            .enumerate()
            .map(|(at, fraction)| LearningPoint {
                fraction: *fraction,
                rows: 100 * (at + 1),
                auc: 0.8,
                se: 0.01,
                accuracy: 0.75,
                accuracy_se: 0.01,
            })
            .collect();
        assert!(asymptote(&four).is_none());
        let mut broken = four.clone();
        broken.push(LearningPoint {
            fraction: 1.0,
            rows: 100,
            auc: 0.8,
            se: 0.01,
            accuracy: 0.75,
            accuracy_se: 0.01,
        });
        broken.push(LearningPoint {
            fraction: 1.0,
            rows: 200,
            auc: 0.8,
            se: 0.01,
            accuracy: 0.75,
            accuracy_se: 0.01,
        });
        assert!(asymptote(&broken).is_none());
    }
}
