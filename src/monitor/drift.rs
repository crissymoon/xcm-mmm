//! Distribution drift between a reference panel and a current one.
//!
//! Two readings per feature, because they fail in different directions. The
//! population stability index is a symmetrised Kullback-Leibler divergence on
//! quantile bins, so it is sensitive to a shift in the body of a distribution
//! and blind to a change confined to a tail the bins merge. The
//! Kolmogorov-Smirnov distance is the largest gap between the two empirical
//! cdfs, so it catches exactly that kind of tail move and says nothing about
//! how much of the sample moved.
//!
//! Both are descriptive on their own. [`crate::monitor::gate`] is the layer
//! that turns them into a decision, and it is separate because a threshold is a
//! claim about a sample size rather than a property of the statistic.

use crate::data::Dataset;

/// Reference quantile bin edges, `bins + 1` of them.
///
/// Computed from one sorted copy of the values rather than one sort per edge,
/// because the gate's null resamples this on every feature and the difference
/// between eleven sorts and one is the difference between a second and a
/// minute on a report.
pub fn edges(values: &[f64], bins: usize) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return vec![0.0, 0.0];
    }
    sorted.sort_by(f64::total_cmp);
    let bins = bins.clamp(2, 50);
    (0..=bins)
        .map(|i| interpolate(&sorted, i as f64 / bins as f64))
        .collect()
}

/// Linear interpolation on a sorted slice, matching
/// [`crate::uncertainty::quantile`] value for value.
fn interpolate(sorted: &[f64], q: f64) -> f64 {
    let position = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    if low == high {
        return sorted[low];
    }
    let weight = position - low as f64;
    sorted[low] * (1.0 - weight) + sorted[high] * weight
}

/// Counts per bin, with the same edge rule the index has always used.
pub fn counts(values: &[f64], edges: &[f64]) -> Vec<usize> {
    let bins = edges.len().saturating_sub(1).max(1);
    let mut out = vec![0usize; bins];
    for value in values {
        out[bucket(*value, edges)] += 1;
    }
    out
}

fn bucket(value: f64, edges: &[f64]) -> usize {
    let last = edges.len().saturating_sub(2);
    if value <= edges[0] {
        return 0;
    }
    for (i, edge) in edges.iter().enumerate().skip(1) {
        if value <= *edge {
            return (i - 1).min(last);
        }
    }
    last
}

/// Population stability index between two samples.
///
/// Bins come from the reference quantiles so a heavy tail does not collapse
/// into one bucket. A share of zero is floored at `1e-6`, which keeps the
/// logarithm finite without pretending the bin was observed.
pub fn psi(reference: &[f64], current: &[f64], bins: usize) -> f64 {
    if reference.len() < 2 || current.len() < 2 {
        return 0.0;
    }
    let edges = edges(reference, bins);
    let a = counts(reference, &edges);
    let b = counts(current, &edges);
    psi_counts(&a, reference.len() as f64, &b, current.len() as f64)
}

/// The index arithmetic itself, on two sets of shares.
///
/// Exposed because the gate's null draws shares and must score them with the
/// same floors and the same expression; a second copy of this loop would be a
/// second definition of the statistic.
pub fn psi_shares(reference: &[f64], current: &[f64]) -> f64 {
    let mut total = 0.0;
    for (r, c) in reference.iter().zip(current) {
        let r = r.max(1e-6);
        let c = c.max(1e-6);
        total += (c - r) * (c / r).ln();
    }
    total
}

/// [`psi_shares`] on counts.
pub fn psi_counts(
    reference: &[usize],
    reference_rows: f64,
    current: &[usize],
    current_rows: f64,
) -> f64 {
    if reference_rows <= 0.0 || current_rows <= 0.0 {
        return 0.0;
    }
    let r: Vec<f64> = reference
        .iter()
        .map(|count| *count as f64 / reference_rows)
        .collect();
    let c: Vec<f64> = current
        .iter()
        .map(|count| *count as f64 / current_rows)
        .collect();
    psi_shares(&r, &c)
}

/// Columns that both panels carry, matched by name.
///
/// A column present on one side only is returned in the second vector rather
/// than treated as zero. Comparing by name, not position, is what makes the
/// reading survive a column reorder.
pub type PairedColumns = (Vec<(String, Vec<f64>, Vec<f64>)>, Vec<String>);

pub fn paired_columns(reference: &Dataset, current: &Dataset) -> PairedColumns {
    let mut current_index = std::collections::BTreeMap::new();
    for (index, name) in current.names.iter().enumerate() {
        current_index.insert(name.clone(), index);
    }
    let mut columns = Vec::new();
    let mut missing = Vec::new();
    for (index, name) in reference.names.iter().enumerate() {
        let Some(&other) = current_index.get(name) else {
            missing.push(name.clone());
            continue;
        };
        let a: Vec<f64> = reference.features.iter().map(|row| row[index]).collect();
        let b: Vec<f64> = current.features.iter().map(|row| row[other]).collect();
        columns.push((name.clone(), a, b));
    }
    (columns, missing)
}

/// Two-sample Kolmogorov-Smirnov statistic.
pub fn ks(reference: &[f64], current: &[f64]) -> f64 {
    if reference.is_empty() || current.is_empty() {
        return 0.0;
    }
    let mut a: Vec<f64> = reference
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let mut b: Vec<f64> = current.iter().copied().filter(|v| v.is_finite()).collect();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let (mut i, mut j) = (0usize, 0usize);
    let mut worst = 0.0f64;
    while i < a.len() || j < b.len() {
        let value = match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) => x.min(*y),
            (Some(x), None) => *x,
            (None, Some(y)) => *y,
            (None, None) => break,
        };
        while i < a.len() && a[i] <= value {
            i += 1;
        }
        while j < b.len() && b[j] <= value {
            j += 1;
        }
        let gap = (i as f64 / a.len() as f64 - j as f64 / b.len() as f64).abs();
        worst = worst.max(gap);
    }
    worst
}

/// One feature's drift reading.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FeatureDrift {
    pub name: String,
    pub psi: f64,
    pub ks: f64,
    pub reference_mean: f64,
    pub current_mean: f64,
    pub mean_shift: f64,
    pub drifted: bool,
}

/// All feature readings for one comparison.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DriftReport {
    pub reference_rows: usize,
    pub current_rows: usize,
    pub threshold: f64,
    pub features: Vec<FeatureDrift>,
    pub missing: Vec<String>,
    pub max_psi: f64,
    pub drifted: usize,
}

impl DriftReport {
    /// Names of the features whose PSI crossed the threshold.
    pub fn alarms(&self) -> Vec<&str> {
        self.features
            .iter()
            .filter(|feature| feature.drifted)
            .map(|feature| feature.name.as_str())
            .collect()
    }
}

/// Compare a reference panel with a current panel by feature name.
///
/// A column missing from either side is named in `missing` rather than treated
/// as zero. Comparing by name, not position, is what makes the report survive a
/// column reorder.
pub fn compare(
    reference: &Dataset,
    current: &Dataset,
    threshold: f64,
    bins: usize,
) -> Result<DriftReport, String> {
    if reference.features.is_empty() || current.features.is_empty() {
        return Err("drift needs at least one row on each side".to_string());
    }
    let (columns, missing) = paired_columns(reference, current);
    let mut features = Vec::with_capacity(columns.len());
    for (name, a, b) in &columns {
        let reading = psi(a, b, bins);
        let distance = ks(a, b);
        let reference_mean = a.iter().sum::<f64>() / a.len() as f64;
        let current_mean = b.iter().sum::<f64>() / b.len() as f64;
        features.push(FeatureDrift {
            name: name.clone(),
            psi: reading,
            ks: distance,
            reference_mean,
            current_mean,
            mean_shift: current_mean - reference_mean,
            drifted: reading >= threshold,
        });
    }
    if features.is_empty() {
        return Err("the two panels share no feature name".to_string());
    }
    let max_psi = features
        .iter()
        .map(|feature| feature.psi)
        .fold(0.0f64, f64::max);
    let drifted = features.iter().filter(|feature| feature.drifted).count();
    Ok(DriftReport {
        reference_rows: reference.len(),
        current_rows: current.len(),
        threshold,
        features,
        missing,
        max_psi,
        drifted,
    })
}

/// Coverage in one batch.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CoveragePoint {
    pub batch: usize,
    pub rows: usize,
    pub coverage: f64,
    pub nominal: f64,
    pub alarm: bool,
}

/// Rolling coverage over consecutive batches.
///
/// `window` is the number of batches in the rolling mean. With one batch the
/// series is the batch coverage itself. The alarm fires when the rolling mean
/// falls below `nominal - tolerance`.
pub fn coverage_series(
    covered: &[bool],
    batch: usize,
    window: usize,
    nominal: f64,
    tolerance: f64,
) -> Vec<CoveragePoint> {
    if covered.is_empty() || batch == 0 {
        return Vec::new();
    }
    let batch = batch.max(1);
    let window = window.max(1);
    let mut out = Vec::new();
    let chunks: Vec<&[bool]> = covered.chunks(batch).collect();
    for (index, _chunk) in chunks.iter().enumerate() {
        let start = index.saturating_sub(window - 1);
        let recent = &chunks[start..=index];
        let rows: usize = recent.iter().map(|part| part.len()).sum();
        let hits: usize = recent
            .iter()
            .map(|part| part.iter().filter(|value| **value).count())
            .sum();
        let reading = hits as f64 / rows.max(1) as f64;
        out.push(CoveragePoint {
            batch: index,
            rows,
            coverage: reading,
            nominal,
            alarm: reading < nominal - tolerance,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_samples_have_no_drift() {
        let a: Vec<f64> = (0..200).map(|i| i as f64).collect();
        assert!(psi(&a, &a, 10).abs() < 1e-9);
        assert!(ks(&a, &a).abs() < 1e-9);
    }

    #[test]
    fn a_shifted_sample_has_positive_drift() {
        let a: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..200).map(|i| i as f64 + 1000.0).collect();
        assert!(psi(&a, &b, 10) > 1.0, "psi {}", psi(&a, &b, 10));
        assert!(ks(&a, &b) > 0.9);
    }

    #[test]
    fn edges_interpolate_like_the_shared_quantile() {
        let values: Vec<f64> = (0..97).map(|i| (i as f64 * 0.37).sin()).collect();
        for q in [0.0, 0.1, 0.25, 0.5, 0.9, 1.0] {
            let shared = crate::uncertainty::quantile(&values, q);
            let local = interpolate(
                &{
                    let mut sorted = values.clone();
                    sorted.sort_by(f64::total_cmp);
                    sorted
                },
                q,
            );
            assert!((shared - local).abs() < 1e-12, "{q}: {shared} {local}");
        }
    }

    #[test]
    fn coverage_alarms_when_a_batch_falls_below_nominal() {
        let mut covered = vec![true; 40];
        covered.extend(vec![false; 10]);
        let points = coverage_series(&covered, 10, 2, 0.9, 0.05);
        assert_eq!(points.len(), 5);
        assert!(!points[0].alarm);
        assert!(points.last().unwrap().alarm);
    }
}
