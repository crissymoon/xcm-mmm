//! Drift, coverage and rollback for a model that is already in service.
//!
//! The repository builds artifacts and checks that they rebuild. That is a
//! property of the build, not of a deployment. This module supplies the three
//! operational pieces a deployed model needs and a card can report:
//!
//! * [`psi`] and [`ks`] compare a reference panel with a current one, feature
//!   by feature, and name the columns whose distribution moved.
//! * [`coverage_series`] tracks conformal coverage over batches and raises an
//!   alarm when a rolling window falls under the nominal rate.
//! * [`Registry`] keeps versioned copies of an artifact and can move the
//!   pointer back to the previous version, so a rollback is a command rather
//!   than a rebuild.
//!
//! Nothing here watches a live system by itself. The module is the measurement
//! and the rollback path; a scheduler still has to call it. That boundary is
//! stated on the card rather than hidden.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::data::Dataset;

/// Population stability index between two samples.
///
/// Bins come from the reference quantiles so a heavy tail does not collapse
/// into one bucket. A share of zero is floored at `1e-6`, which keeps the
/// logarithm finite without pretending the bin was observed.
pub fn psi(reference: &[f64], current: &[f64], bins: usize) -> f64 {
    if reference.len() < 2 || current.len() < 2 {
        return 0.0;
    }
    let bins = bins.clamp(2, 50);
    let mut edges: Vec<f64> = (0..=bins)
        .map(|i| {
            let q = i as f64 / bins as f64;
            crate::uncertainty::quantile(reference, q)
        })
        .collect();
    for edge in edges.iter_mut() {
        if !edge.is_finite() {
            *edge = 0.0;
        }
    }
    let mut reference_counts = vec![0usize; bins];
    let mut current_counts = vec![0usize; bins];
    for value in reference {
        reference_counts[bucket(*value, &edges)] += 1;
    }
    for value in current {
        current_counts[bucket(*value, &edges)] += 1;
    }
    let reference_n = reference.len() as f64;
    let current_n = current.len() as f64;
    let mut total = 0.0;
    for i in 0..bins {
        let r = (reference_counts[i] as f64 / reference_n).max(1e-6);
        let c = (current_counts[i] as f64 / current_n).max(1e-6);
        total += (c - r) * (c / r).ln();
    }
    total
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
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    let mut current_index = std::collections::BTreeMap::new();
    for (index, name) in current.names.iter().enumerate() {
        current_index.insert(name.clone(), index);
    }
    let mut features = Vec::new();
    let mut missing = Vec::new();
    for (index, name) in reference.names.iter().enumerate() {
        let Some(&other) = current_index.get(name) else {
            missing.push(name.clone());
            continue;
        };
        let a: Vec<f64> = reference.features.iter().map(|row| row[index]).collect();
        let b: Vec<f64> = current.features.iter().map(|row| row[other]).collect();
        let reading = psi(&a, &b, bins);
        let distance = ks(&a, &b);
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
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    let mut chunks: Vec<&[bool]> = covered.chunks(batch).collect();
    if chunks.is_empty() {
        chunks.push(covered);
    }
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

/// A versioned artifact directory.
#[derive(Clone, Debug)]
pub struct Registry {
    root: PathBuf,
}

/// One published version.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Version {
    pub id: u32,
    pub file: String,
}

impl Registry {
    /// Open or create a registry at `root`.
    pub fn open(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("versions"))?;
        Ok(Self { root })
    }

    /// Publish a model file as the next version and make it current.
    pub fn publish(&self, model: &str) -> io::Result<Version> {
        let source = Path::new(model);
        if !source.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{model} is not a file"),
            ));
        }
        let next = self
            .list()?
            .last()
            .map(|version| version.id + 1)
            .unwrap_or(1);
        let directory = self.root.join("versions").join(format!("v{next:04}"));
        fs::create_dir_all(&directory)?;
        let name = source
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "model.json".to_string());
        fs::copy(source, directory.join(&name))?;
        self.set_current(next, &name)?;
        Ok(Version {
            id: next,
            file: name,
        })
    }

    /// The current version, if anything was published.
    pub fn current(&self) -> io::Result<Option<Version>> {
        let text = match fs::read_to_string(self.root.join("current.json")) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))
    }

    /// Every published version, oldest first.
    pub fn list(&self) -> io::Result<Vec<Version>> {
        let mut out = Vec::new();
        let directory = self.root.join("versions");
        if !directory.is_dir() {
            return Ok(out);
        }
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let Ok(id) = name.trim_start_matches('v').parse::<u32>() else {
                continue;
            };
            let mut files = fs::read_dir(entry.path())?
                .filter_map(|file| file.ok())
                .map(|file| file.file_name().to_string_lossy().to_string())
                .filter(|file| file.ends_with(".json"))
                .collect::<Vec<_>>();
            files.sort();
            if let Some(file) = files.into_iter().next() {
                out.push(Version { id, file });
            }
        }
        out.sort_by_key(|version| version.id);
        Ok(out)
    }

    /// Move the current pointer back one version and restore `target`.
    ///
    /// The target file is overwritten with the previous version's bytes, and
    /// the registry is unchanged apart from the pointer, so a second rollback
    /// can return to the version that was current before. Returns the version
    /// that was restored.
    pub fn rollback(&self, target: &str) -> io::Result<Version> {
        let current = self
            .current()?
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "nothing is published"))?;
        let versions = self.list()?;
        let previous = versions
            .iter()
            .rev()
            .find(|version| version.id < current.id)
            .cloned()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "there is no earlier version to roll back to",
                )
            })?;
        let source = self
            .root
            .join("versions")
            .join(format!("v{:04}", previous.id))
            .join(&previous.file);
        fs::copy(source, target)?;
        self.set_current(previous.id, &previous.file)?;
        Ok(previous)
    }

    fn set_current(&self, id: u32, file: &str) -> io::Result<()> {
        let version = Version {
            id,
            file: file.to_string(),
        };
        let text = serde_json::to_string(&version).map_err(io::Error::other)?;
        fs::write(self.root.join("current.json"), format!("{text}\n"))
    }
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
    fn coverage_alarms_when_a_batch_falls_below_nominal() {
        let mut covered = vec![true; 40];
        covered.extend(vec![false; 10]);
        let points = coverage_series(&covered, 10, 2, 0.9, 0.05);
        assert_eq!(points.len(), 5);
        assert!(!points[0].alarm);
        assert!(points.last().unwrap().alarm);
    }

    #[test]
    fn a_registry_publishes_and_rolls_back() {
        let dir = std::env::temp_dir().join(format!("xcm-mmm-reg-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let first = dir.join("first.json");
        let second = dir.join("second.json");
        fs::write(&first, "{\"version\":1}").unwrap();
        fs::write(&second, "{\"version\":2}").unwrap();
        let registry = Registry::open(dir.join("registry")).unwrap();
        let a = registry.publish(first.to_str().unwrap()).unwrap();
        let b = registry.publish(second.to_str().unwrap()).unwrap();
        assert_eq!(a.id, 1);
        assert_eq!(b.id, 2);
        assert_eq!(registry.current().unwrap().unwrap().id, 2);
        assert_eq!(registry.list().unwrap().len(), 2);
        let target = dir.join("live.json");
        fs::write(&target, "{\"version\":2}").unwrap();
        let restored = registry.rollback(target.to_str().unwrap()).unwrap();
        assert_eq!(restored.id, 1);
        assert!(fs::read_to_string(&target)
            .unwrap()
            .contains("\"version\":1"));
        assert_eq!(registry.current().unwrap().unwrap().id, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rolling_back_without_history_is_refused() {
        let dir = std::env::temp_dir().join(format!("xcm-mmm-reg-one-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.json");
        fs::write(&model, "{}").unwrap();
        let registry = Registry::open(dir.join("registry")).unwrap();
        registry.publish(model.to_str().unwrap()).unwrap();
        assert!(registry.rollback(model.to_str().unwrap()).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
