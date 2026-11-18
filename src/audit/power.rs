//! The smallest gain a bank can resolve, and the floor worth claiming.
//!
//! A standard error says how precisely one number was measured. A reader's next
//! question is different: how large would a difference have to be before this
//! bank could see it at all. This module answers that, in the units the card
//! quotes, beside a floor fixed before the fit.
//!
//! For AUC the standard error is Hanley and McNeil's (*Radiology* 143, 1982),
//! which is the DeLong variance written in closed form for a single reading. The
//! smallest detectable difference between two independent estimates is
//! `(z + z_power) * sqrt(2) * se`: the `sqrt(2)` is there because a difference of
//! two noisy numbers is noisier than either. That is the conservative form.
//!
//! The conservative form is the wrong one when both models were scored on the
//! same rows, which is the usual case here: a paired test divides by
//! `sqrt(2(1 - rho))` instead and is sharper as the two readings agree.
//! [`crate::uncertainty::delong_pair`] measures `rho` on the placements and is
//! the reading to quote for an observed difference. This module answers the
//! prior question: with no comparison yet, what difference could this bank
//! detect. A card that quotes an observed gain should quote the paired interval
//! and use this number only as the bar the gain had to clear.
//!
//! Both functions scale the row count rather than searching for it: with the
//! class balance held fixed the standard error falls as `1 / sqrt(n)`, so the
//! rows needed for a floor follow from the rows in hand by one ratio.

use serde::{Deserialize, Serialize};

use crate::num::round4;
use crate::stats::normal_quantile;

/// A bank shape, its standard error and the gain it can resolve.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Power {
    /// The metric the delta is quoted in, AUC or a share.
    pub reading: f64,
    /// Whether the delta is an AUC or a share, for the caller's label.
    pub kind: String,
    /// Rows used.
    pub rows: usize,
    /// Events that carry the variance: positives for AUC, all rows for a share.
    pub events: usize,
    /// Standard error of the reading at this shape.
    pub se: f64,
    /// Smallest difference detectable at the stated power.
    pub detectable: f64,
    /// Smallest difference worth claiming, fixed before the fit.
    pub floor: f64,
    /// Power the detectable figure is quoted at.
    pub power: f64,
    /// Level of the test.
    pub level: f64,
    /// Whether the bank resolves the floor.
    pub resolvable: bool,
    /// Rows that would resolve the floor, holdng the class balance fixed.
    pub rows_for_floor: usize,
    /// Plain reading of the three numbers.
    pub note: String,
}

/// Variance of a single AUC, Hanley and McNeil (1982), equation 2.
///
/// `q1 = A / (2 - A)` and `q2 = 2 A^2 / (1 + A)` are the two placement
/// probabilities the trapezoidal variance needs. The reading is clamped so a
/// perfect separation does not produce a negative variance from rounding.
pub fn hanley_mcneil(auc: f64, positives: usize, negatives: usize) -> f64 {
    if positives == 0 || negatives == 0 {
        return 0.0;
    }
    let a = auc.clamp(0.0, 1.0);
    let q1 = a / (2.0 - a);
    let q2 = 2.0 * a * a / (1.0 + a);
    let np = positives as f64;
    let nn = negatives as f64;
    let variance =
        (a * (1.0 - a) + (np - 1.0) * (q1 - a * a) + (nn - 1.0) * (q2 - a * a)) / (np * nn);
    variance.max(0.0).sqrt()
}

/// `z` for the two sided level plus `z` for the power, the multiplier a
/// difference needs to clear both.
fn multiplier(level: f64, power: f64) -> f64 {
    let level = level.clamp(0.5, 0.999_999);
    let power = power.clamp(0.5, 0.999_999);
    normal_quantile(1.0 - (1.0 - level) / 2.0) + normal_quantile(power)
}

/// The smallest AUC gain this bank resolves at `power`, and the rows a floor needs.
pub fn detectable_auc(
    auc: f64,
    positives: usize,
    negatives: usize,
    floor: f64,
    power: f64,
    level: f64,
) -> Power {
    let se = hanley_mcneil(auc, positives, negatives);
    assemble(
        "auc",
        auc,
        positives + negatives,
        positives,
        se,
        floor,
        power,
        level,
    )
}

/// The same reading for a share, such as an accuracy or a balanced accuracy.
///
/// The variance is the binomial one, `p (1 - p) / n`. A balanced accuracy is a
/// mean of two rates measured on their own rows, so its true variance is smaller
/// than this; the binomial form is the conservative bound and is the one a card
/// should print when the class sizes differ, because it is the one that does not
/// need them.
pub fn detectable_share(share: f64, rows: usize, floor: f64, power: f64, level: f64) -> Power {
    let se = if rows == 0 {
        0.0
    } else {
        let p = share.clamp(1e-9, 1.0 - 1e-9);
        (p * (1.0 - p) / rows as f64).sqrt()
    };
    assemble("share", share, rows, rows, se, floor, power, level)
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    kind: &'static str,
    reading: f64,
    rows: usize,
    events: usize,
    se: f64,
    floor: f64,
    power: f64,
    level: f64,
) -> Power {
    let k = multiplier(level, power);
    let detectable = k * std::f64::consts::SQRT_2 * se;
    let needed = if floor <= 0.0 || se <= 0.0 || rows == 0 {
        rows
    } else {
        let target = floor / (k * std::f64::consts::SQRT_2);
        let factor = (se / target).powi(2);
        ((rows as f64) * factor).ceil() as usize
    };
    let resolvable = detectable <= floor;
    let note = if resolvable {
        format!(
            "At this shape a difference of {:.3} is the smallest that clears a {} percent \
test at {} percent power, and the pre-registered floor of {:.3} sits above it, so a gain \
claimed at the floor is visible on this bank.",
            round4(detectable),
            (level * 100.0).round(),
            (power * 100.0).round(),
            round4(floor)
        )
    } else {
        format!(
            "At this shape the smallest difference that clears a {} percent test at {} \
percent power is {:.3}, which is wider than the pre-registered floor of {:.3}, so a gain \
at the floor would not be distinguishable from noise here. Resolving it would take about \
{} rows at this class balance, against the {} in hand.",
            (level * 100.0).round(),
            (power * 100.0).round(),
            round4(detectable),
            round4(floor),
            crate::num::commas(needed),
            crate::num::commas(rows)
        )
    };
    Power {
        reading: round4(reading),
        kind: kind.to_string(),
        rows,
        events,
        se: round4(se),
        detectable: round4(detectable),
        floor: round4(floor),
        power,
        level,
        resolvable,
        rows_for_floor: needed,
        note,
    }
}
