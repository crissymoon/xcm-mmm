//! How much evidence a fitted bank actually supplies.
//!
//! A sample size is not a support statement. Two numbers make the boundary
//! visible: events per feature, which says whether a logistic has enough
//! positives to estimate its coefficients, and a learning curve, which says
//! whether more rows of the same kind would move the held-out score. A curve
//! that is still rising is a data-limited bank; a flat curve is a signal that
//! the ceiling is the features or the operator, not the row count.
//!
//! The curve is deliberately modest: four fractions, the same folds, the same
//! model and the same metric as the headline. It is a diagnostic, not a
//! forecast of what a new city would buy.

use serde::{Deserialize, Serialize};

/// One point on a learning curve.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LearningPoint {
    pub fraction: f64,
    pub rows: usize,
    pub auc: f64,
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

/// The slope between the last two learning-curve points.
///
/// Positive means the held-out score was still rising at the largest fraction,
/// which is the signature of a bank that is limited by rows rather than by the
/// hypothesis class.
pub fn slope(points: &[LearningPoint]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let a = &points[points.len() - 2];
    let b = &points[points.len() - 1];
    if b.fraction <= a.fraction {
        return 0.0;
    }
    (b.auc - a.auc) / (b.fraction - a.fraction)
}

/// The slope as a percentage of AUC per doubling of rows.
pub fn doubling_gain(points: &[LearningPoint]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let a = &points[points.len() - 2];
    let b = &points[points.len() - 1];
    if a.fraction <= 0.0 || b.fraction <= a.fraction {
        return 0.0;
    }
    let doublings = (b.fraction / a.fraction).log2();
    (b.auc - a.auc) / doublings * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_per_feature_refuses_a_zero_denominator() {
        assert_eq!(events_per_feature(10, 0), 0.0);
        assert!((events_per_feature(100, 10) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn the_slope_reads_the_last_two_points() {
        let points = vec![
            LearningPoint {
                fraction: 0.25,
                rows: 25,
                auc: 0.6,
            },
            LearningPoint {
                fraction: 0.5,
                rows: 50,
                auc: 0.7,
            },
            LearningPoint {
                fraction: 1.0,
                rows: 100,
                auc: 0.8,
            },
        ];
        assert!((slope(&points) - 0.2).abs() < 1e-9);
        assert!((doubling_gain(&points) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn a_flat_curve_has_no_doubling_gain() {
        let points = vec![
            LearningPoint {
                fraction: 0.5,
                rows: 50,
                auc: 0.8,
            },
            LearningPoint {
                fraction: 1.0,
                rows: 100,
                auc: 0.8,
            },
        ];
        assert_eq!(doubling_gain(&points), 0.0);
    }
}
