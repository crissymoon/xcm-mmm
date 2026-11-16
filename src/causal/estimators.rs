//! Estimators that read the cross-fitted nuisances of a causal fit.
//!
//! Everything here is a pure function of vectors: the doubly robust
//! pseudo-outcome, its influence function, the overlap weighted and Crump
//! trimmed effects, the E-value and the policy value curve. The fold loop that
//! produces the nuisances lives in the parent module and the parent decides
//! what to publish, so a change to one estimand cannot disturb the others.

use serde::{Deserialize, Serialize};

use super::{mean, variance, CausalData};

/// Smallest propensity used in a denominator.
pub(crate) const PROPENSITY_FLOOR: f64 = 1e-3;

/// Cap on either bias correction term of the doubly robust pseudo-outcome.
pub(crate) const CORRECTION_CAP: f64 = 10.0;

/// Rows the Crump rule must leave before its estimate is reported.
pub(crate) const CRUMP_FLOOR: usize = 50;

/// Value of the uplift ranking at one treat-the-top-fraction rule.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PolicyPoint {
    /// Share of the ranked table the rule treats, from 0.0 to 1.0.
    pub fraction: f64,
    /// Rows the rule treats.
    pub rows: usize,
    /// Doubly robust mean of the pseudo-outcome over those rows.
    pub value: f64,
}

/// Doubly robust pseudo-outcome for every row.
///
/// `psi = mu1 - mu0 + t (y - mu1) / e - (1 - t) (y - mu0) / (1 - e)` with `e`
/// clamped to `[1e-3, 1 - 1e-3]`. Each correction term is capped at
/// `CORRECTION_CAP` because the outcome is a probability, and the number of
/// rows that reached the cap is returned so the caller can report it rather
/// than drop those rows silently.
pub(crate) fn pseudo_outcomes(
    data: &CausalData,
    mu0: &[f64],
    mu1: &[f64],
    propensity: &[f64],
) -> (Vec<f64>, usize) {
    let mut psi = Vec::with_capacity(data.rows.len());
    let mut clipped = 0usize;
    for (at, row) in data.rows.iter().enumerate() {
        let e = propensity[at].clamp(PROPENSITY_FLOOR, 1.0 - PROPENSITY_FLOOR);
        let treated_term = if row.treatment {
            (row.outcome - mu1[at]) / e
        } else {
            0.0
        };
        let control_term = if row.treatment {
            0.0
        } else {
            -(row.outcome - mu0[at]) / (1.0 - e)
        };
        if treated_term.abs() > CORRECTION_CAP || control_term.abs() > CORRECTION_CAP {
            clipped += 1;
        }
        psi.push(
            mu1[at] - mu0[at]
                + treated_term.clamp(-CORRECTION_CAP, CORRECTION_CAP)
                + control_term.clamp(-CORRECTION_CAP, CORRECTION_CAP),
        );
    }
    (psi, clipped)
}

/// Mean pseudo-outcome on the selected rows and its influence function error.
///
/// The influence function of a sample mean is `psi - mean`, so the standard
/// error is `sd(psi) / sqrt(n)`. Neyman orthogonality is what lets the nuisance
/// error be left out of this formula without a first order bias.
pub(crate) fn dr_mean(psi: &[f64], index: &[usize]) -> (f64, f64) {
    let values: Vec<f64> = index.iter().map(|at| psi[*at]).collect();
    let point = mean(&values);
    let error = (variance(&values) / values.len().max(1) as f64).sqrt();
    (
        point,
        if error.is_finite() {
            error.max(0.0)
        } else {
            0.0
        },
    )
}

/// Overlap weighted effect (Li, Morgan and Zaslavsky, 2018).
///
/// Weights `w = e (1 - e)` scaled to mean one put the estimate on the rows
/// where both arms are plausible, which is the population a decision can act
/// on. The standard error uses the influence function `w (psi - value)`, with
/// the result floored at zero because the trimmed read can be negative on a
/// degenerate weight vector.
pub(crate) fn overlap_weighted(psi: &[f64], propensity: &[f64]) -> (f64, f64) {
    let mut weights: Vec<f64> = propensity
        .iter()
        .map(|p| {
            let p = p.clamp(PROPENSITY_FLOOR, 1.0 - PROPENSITY_FLOOR);
            p * (1.0 - p)
        })
        .collect();
    let raw = mean(&weights);
    if !raw.is_finite() || raw <= 0.0 {
        return (mean(psi), 0.0);
    }
    for weight in weights.iter_mut() {
        *weight /= raw;
    }
    let n = psi.len().max(1) as f64;
    let value = psi
        .iter()
        .zip(&weights)
        .map(|(p, weight)| weight * p)
        .sum::<f64>()
        / n;
    let spread: Vec<f64> = psi
        .iter()
        .zip(&weights)
        .map(|(p, weight)| weight * (p - value))
        .collect();
    let error = (variance(&spread) / n).sqrt();
    (
        value,
        if error.is_finite() {
            error.max(0.0)
        } else {
            0.0
        },
    )
}

/// Crump, Hotz, Imbens and Mitnik (2009) trimming rule.
///
/// With `t = 1 / (e (1 - e))` sorted ascending, take the longest prefix whose
/// last member is at most twice the prefix mean. That prefix fixes the bound
/// `a = 0.5 - sqrt(0.25 - 1 / t_k)`, which retains every row with
/// `a <= e <= 1 - a`. Returns the bound and the retained row index. The first
/// prefix always satisfies the rule, so a bound always exists.
pub(crate) fn crump_rule(propensity: &[f64]) -> (f64, Vec<usize>) {
    let t: Vec<f64> = propensity
        .iter()
        .map(|p| {
            let p = p.clamp(PROPENSITY_FLOOR, 1.0 - PROPENSITY_FLOOR);
            1.0 / (p * (1.0 - p))
        })
        .collect();
    let mut order: Vec<usize> = (0..t.len()).collect();
    order.sort_by(|a, b| t[*a].total_cmp(&t[*b]));
    let mut sum = 0.0;
    let mut keep = 0usize;
    for (position, at) in order.iter().enumerate() {
        sum += t[*at];
        if t[*at] <= 2.0 * sum / (position + 1) as f64 {
            keep = position + 1;
        }
    }
    let last = t[*order.get(keep.saturating_sub(1)).unwrap_or(&0)];
    let inside = 0.25 - 1.0 / last;
    let bound = if inside >= 0.0 {
        (0.5 - inside.sqrt()).max(0.0)
    } else {
        0.0
    };
    let kept: Vec<usize> = (0..propensity.len())
        .filter(|at| {
            let p = propensity[*at];
            p >= bound && p <= 1.0 - bound
        })
        .collect();
    (bound, kept)
}

/// E-value for a risk ratio (VanderWeele and Ding, 2017).
fn e_value(risk_ratio: f64) -> f64 {
    let rr = risk_ratio.max(1e-6);
    if rr >= 1.0 {
        rr + (rr * (rr - 1.0)).sqrt()
    } else {
        let inverse = 1.0 / rr;
        inverse + (inverse * (inverse - 1.0)).sqrt()
    }
}

/// E-value of a risk difference, converted at the control arm risk `p0`.
///
/// `p0` is floored at `1e-6` so a panel with no control events reports a large
/// E-value instead of an infinity.
pub(crate) fn e_value_of_difference(difference: f64, p0: f64) -> f64 {
    let base = p0.max(1e-6);
    e_value(((base + difference) / base).max(1e-6))
}

/// Policy value of treating the top fraction of the uplift ranking.
///
/// The first point is treating nobody, worth zero; the rest are the deciles of
/// the ranking and the value is the mean pseudo-outcome over the rows the rule
/// would treat. The last point is treating every row.
pub(crate) fn policy_curve(uplift: &[f64], psi: &[f64]) -> Vec<PolicyPoint> {
    let mut order: Vec<usize> = (0..uplift.len()).collect();
    order.sort_by(|a, b| uplift[*b].total_cmp(&uplift[*a]));
    let n = order.len();
    let mut curve = Vec::with_capacity(11);
    curve.push(PolicyPoint {
        fraction: 0.0,
        rows: 0,
        value: 0.0,
    });
    if n == 0 {
        return curve;
    }
    for step in 1..=10 {
        let fraction = step as f64 / 10.0;
        let rows = ((n as f64 * fraction).round() as usize).clamp(1, n);
        let value = mean(&order[..rows].iter().map(|at| psi[*at]).collect::<Vec<_>>());
        curve.push(PolicyPoint {
            fraction,
            rows,
            value,
        });
    }
    curve
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_e_value_grows_with_the_risk_ratio() {
        assert!((e_value(1.0) - 1.0).abs() < 1e-12);
        assert!(e_value(2.0) > 2.0);
        assert!((e_value(0.5) - e_value(2.0)).abs() < 1e-12);
    }

    #[test]
    fn a_flat_pseudo_outcome_has_no_policy_gain() {
        let psi = vec![0.2; 100];
        let uplift = vec![0.0; 100];
        let curve = policy_curve(&uplift, &psi);
        assert_eq!(curve.len(), 11);
        for point in curve.iter().skip(1) {
            assert!((point.value - 0.2).abs() < 1e-12);
        }
        assert_eq!(curve[0].rows, 0);
        assert_eq!(curve[0].value, 0.0);
    }

    #[test]
    fn a_ranked_pseudo_outcome_falls_as_the_rule_widens() {
        let psi: Vec<f64> = (0..100).map(|i| 1.0 - i as f64 / 100.0).collect();
        let uplift = psi.clone();
        let curve = policy_curve(&uplift, &psi);
        for pair in curve.windows(2).skip(1) {
            assert!(pair[0].value >= pair[1].value);
        }
        assert_eq!(curve.last().map(|point| point.rows), Some(100));
    }
}
