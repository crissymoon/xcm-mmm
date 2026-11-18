//! A cohort table, where every reading carries the age of its window.
//!
//! The churn panel is one snapshot. A customer who joined three months ago and
//! is still there has been watched for three months; a customer who has been
//! there five years has been watched for five. Their recorded labels are not on
//! the same clock, so a rate computed across the whole panel mixes customers
//! who had a week to churn with customers who had years. The shipped panel
//! shows exactly that: the rate falls from 53.3 percent among customers in
//! their first six months to 6.6 percent beyond five years, and the fall is
//! mostly the length of the observation rather than a change in behaviour.
//!
//! So this table never compares a young cohort with a mature one. Cohorts are
//! bands of tenure at the snapshot, which with a fixed snapshot is the same
//! thing as a registration vintage. A band is *mature* for a horizon when every
//! row in it has been watched for the whole horizon, so the label has had time
//! to occur; an unconverted row inside the window is censored, not a negative
//! (Chapelle, *KDD* 2014, states the selection problem for a fixed window).
//! Readings that need a complete label are refused on a band that is not
//! mature, with the reason printed instead of the number.
//!
//! Two further limits are structural and stated rather than repaired. The
//! snapshot records *that* a customer churned, not when, so no band can be
//! turned into a horizon-dated rate; maturity says which bands are old enough
//! for such a rate to be computable at all, not what it is. And each band's
//! discrimination is measured against that band's own base rate, so a band
//! comparison is a comparison of two different tasks.

use serde::{Deserialize, Serialize};

use super::fit::Sample;
use crate::uncertainty::{calibration, conformal_set, delong_pair, PairedAuc};

/// Months a labelled outcome needs in order to be complete.
///
/// Twelve is the horizon the panel's own series is read over and the shortest
/// window that clears the first-year churn in this data. A band whose lower
/// edge is at least this many months is mature; a band that straddles it is
/// not, which is why two of the seven bands below carry no readings.
pub const HORIZON_MONTHS: u32 = 12;

/// One tenure band.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CohortRow {
    /// Human label, such as `13-24 months`.
    pub label: String,
    pub low: u32,
    /// Upper edge, `99` for the open band.
    pub high: u32,
    pub rows: usize,
    pub churned: usize,
    /// Recorded churn share. Not a hazard: see the module note.
    pub rate: f64,
    /// Mean tenure in the band, the age every reading below is conditional on.
    pub mean_tenure: f64,
    /// True when every row has been watched for the whole horizon.
    pub mature: bool,
    /// Rows watched for fewer than [`HORIZON_MONTHS`] months.
    pub censored: usize,
    pub auc: Option<f64>,
    pub ece: Option<f64>,
    /// Share of the band whose label sits inside the pooled conformal set.
    pub coverage: Option<f64>,
    /// The shipped model against the tenure-only reference on the same rows.
    pub against_tenure: Option<PairedAuc>,
    /// Reason a reading is missing, when one is.
    pub refused: Option<String>,
}

/// The bands plus one pooled reading over the mature ones.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CohortTable {
    pub horizon: u32,
    pub rows: Vec<CohortRow>,
    /// Rows in mature bands.
    pub mature_rows: usize,
    /// Churners in mature bands.
    pub mature_churned: usize,
    /// Held-out AUC pooled over the mature bands, which is the whole panel
    /// minus the customers too young to have had twelve months.
    pub mature_auc: f64,
    pub note: String,
}

/// The bands, from first six months to five years and beyond.
pub fn bands() -> Vec<(u32, u32)> {
    vec![
        (0, 6),
        (7, 12),
        (13, 24),
        (25, 36),
        (37, 48),
        (49, 60),
        (61, 99),
    ]
}

fn label(low: u32, high: u32) -> String {
    if high >= 99 {
        format!("{low}+ months")
    } else {
        format!("{low}-{high} months")
    }
}

/// Build the table.
///
/// `scores` are the held-out probabilities for the shipped model, `reference`
/// the same rows scored by the tenure-only baseline, and `threshold` the pooled
/// conformal threshold whose coverage the table reports.
pub fn table(
    samples: &[Sample],
    scores: &[f64],
    reference: &[f64],
    threshold: f64,
    horizon: u32,
) -> CohortTable {
    let horizon = horizon.max(1);
    if samples.len() != scores.len() || samples.len() != reference.len() {
        return CohortTable {
            horizon,
            rows: Vec::new(),
            note: "the score vectors do not line up with the panel, so no cohort reading is \
                   possible"
                .to_string(),
            ..CohortTable::default()
        };
    }
    let mut rows = Vec::new();
    for (low, high) in bands() {
        let inside: Vec<usize> = (0..samples.len())
            .filter(|at| {
                let tenure = samples[*at].values.first().copied().unwrap_or(0.0);
                tenure >= f64::from(low) && tenure <= f64::from(high)
            })
            .collect();
        if inside.is_empty() {
            continue;
        }
        let row_scores: Vec<f64> = inside.iter().map(|at| scores[*at]).collect();
        let row_reference: Vec<f64> = inside.iter().map(|at| reference[*at]).collect();
        let labels: Vec<bool> = inside.iter().map(|at| samples[*at].churn).collect();
        let churned = labels.iter().filter(|label| **label).count();
        let mean_tenure = inside
            .iter()
            .map(|at| samples[*at].values.first().copied().unwrap_or(0.0))
            .sum::<f64>()
            / inside.len() as f64;
        let censored = inside
            .iter()
            .filter(|at| samples[**at].values.first().copied().unwrap_or(0.0) < f64::from(horizon))
            .count();
        let mature = low >= horizon;
        let mut row = CohortRow {
            label: label(low, high),
            low,
            high,
            rows: inside.len(),
            churned,
            rate: churned as f64 / inside.len() as f64,
            mean_tenure,
            mature,
            censored,
            auc: None,
            ece: None,
            coverage: None,
            against_tenure: None,
            refused: None,
        };
        if !mature {
            row.refused = Some(format!(
                "watched for less than {horizon} months, so an unconverted row is censored rather than negative"
            ));
            rows.push(row);
            continue;
        }
        if churned == 0 || churned == inside.len() {
            row.refused = Some(format!("one class only, {churned} of {}", inside.len()));
            rows.push(row);
            continue;
        }
        row.auc = Some(crate::uncertainty::auc(&row_scores, &labels));
        row.ece = Some(calibration(&row_scores, &labels, 5).ece);
        row.coverage = Some(coverage(&row_scores, &labels, threshold));
        row.against_tenure = delong_pair(&row_scores, &row_reference, &labels, 0.95);
        rows.push(row);
    }
    let mature: Vec<usize> = (0..samples.len())
        .filter(|at| samples[*at].values.first().copied().unwrap_or(0.0) >= f64::from(horizon))
        .collect();
    let mature_labels: Vec<bool> = mature.iter().map(|at| samples[*at].churn).collect();
    let mature_scores: Vec<f64> = mature.iter().map(|at| scores[*at]).collect();
    let mature_auc =
        if mature_labels.iter().any(|label| *label) && mature_labels.iter().any(|label| !*label) {
            crate::uncertainty::auc(&mature_scores, &mature_labels)
        } else {
            0.0
        };
    let refused = rows.iter().filter(|row| !row.mature).count();
    let note = format!(
        "{} of {} rows have been watched for the whole {horizon} month horizon and carry the readings; \
         {refused} bands are younger than that and are printed with their reason instead. A snapshot cannot \
         date a churn, so no band here is a {horizon} month rate, and two bands are never compared because \
         each is measured against its own base rate.",
        crate::num::commas(mature.len()),
        crate::num::commas(samples.len())
    );
    CohortTable {
        horizon,
        rows,
        mature_rows: mature.len(),
        mature_churned: mature_labels.iter().filter(|label| **label).count(),
        mature_auc,
        note,
    }
}

fn coverage(scores: &[f64], labels: &[bool], threshold: f64) -> f64 {
    let mut covered = 0usize;
    for (score, label) in scores.iter().zip(labels) {
        // `conformal_set` returns the negative label first: a set holds the
        // negative when the probability is low enough and the positive when
        // `1 - probability` is.
        let (in_negative, in_positive) = conformal_set(*score, threshold);
        if (*label && in_positive) || (!*label && in_negative) {
            covered += 1;
        }
    }
    covered as f64 / scores.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(tenure: f64, churn: bool) -> Sample {
        Sample {
            id: format!("id-{tenure}-{}", u8::from(churn)),
            churn,
            values: vec![tenure],
        }
    }

    /// A young band and a mature one, with scores that separate them.
    fn panel() -> Vec<Sample> {
        let mut out = Vec::new();
        for i in 0..40 {
            out.push(sample(f64::from(i % 6), i % 2 == 0));
        }
        for i in 0..80 {
            out.push(sample(20.0 + f64::from(i % 5), i % 2 == 0));
        }
        out
    }

    #[test]
    fn a_young_band_is_printed_with_its_reason() {
        let samples = panel();
        let scores: Vec<f64> = samples
            .iter()
            .map(|s| if s.churn { 0.8 } else { 0.2 })
            .collect();
        let table = table(&samples, &scores, &scores, 0.1, HORIZON_MONTHS);
        let young = table
            .rows
            .iter()
            .find(|row| row.label == "0-6 months")
            .expect("young band");
        assert!(!young.mature);
        assert!(young.auc.is_none());
        assert!(young.refused.is_some());
        assert_eq!(young.censored, young.rows);
    }

    #[test]
    fn a_mature_band_carries_every_reading() {
        let samples = panel();
        let scores: Vec<f64> = samples
            .iter()
            .map(|s| if s.churn { 0.8 } else { 0.2 })
            .collect();
        let table = table(&samples, &scores, &scores, 0.1, HORIZON_MONTHS);
        let old = table
            .rows
            .iter()
            .find(|row| row.label == "13-24 months")
            .expect("mature band");
        assert!(old.mature);
        assert_eq!(old.censored, 0);
        assert!(old.auc.unwrap() > 0.99);
        assert!(old.coverage.is_some());
        let paired = old.against_tenure.as_ref().expect("paired reading");
        assert!(paired.se.abs() < 1e-9, "the same scores cannot differ");
        assert_eq!(table.mature_rows, 80);
    }

    #[test]
    fn the_mature_pool_excludes_the_young_rows() {
        let samples = panel();
        let scores: Vec<f64> = samples
            .iter()
            .map(|s| if s.churn { 0.8 } else { 0.2 })
            .collect();
        let table = table(&samples, &scores, &scores, 0.1, HORIZON_MONTHS);
        assert_eq!(table.mature_rows, 80);
        assert!(table.mature_auc > 0.99);
        assert!(table.note.contains("12 month horizon"));
        assert!(table.note.contains("1 bands are younger"));
        assert_eq!(table.rows.len(), 2);
    }

    #[test]
    fn a_band_with_one_class_refuses_its_readings() {
        let samples: Vec<Sample> = (0..30).map(|i| sample(20.0 + f64::from(i), true)).collect();
        let scores = vec![0.9; samples.len()];
        let table = table(&samples, &scores, &scores, 0.1, HORIZON_MONTHS);
        let row = &table.rows[0];
        assert!(row.auc.is_none());
        assert!(row.refused.as_deref().unwrap_or("").contains("one class"));
    }
}
