//! The artifact a fit ships and the reader that assigns an entity to it.
//!
//! The file is plain JSON because a segment is meant to be argued with: it
//! carries the feature table, the standardisation, the centroids, the profile
//! of every group in both z and plain units, the assignment of every entity it
//! was fitted on, and the two audits. Nothing about a segment is prose only.
//!
//! A segment name is derived from the profile rather than typed in: the two
//! strongest readings, said high or low. A label that a rebuild could not
//! reproduce would be a claim outside the artifact.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::ad::Day;
use super::feature::{FeatureSpec, Scaling};
use super::kmeans::{Fit, Sweep};
use super::stability::{Control, Stability};
use super::SegmentConfig;

/// Where the observations came from, with the digest of the file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub path: String,
    pub entities: usize,
    pub ads: usize,
    pub test_ads: usize,
    pub first_day: String,
    pub last_day: String,
    pub fnv64: String,
}

/// One reading of a segment, in z and in its own units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProfileRow {
    pub name: String,
    pub label: String,
    pub family: String,
    pub unit: String,
    pub z: f64,
    pub mean: f64,
}

/// One group of advertisers who behaved alike.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub id: usize,
    pub label: String,
    pub size: usize,
    pub share: f64,
    pub profile: Vec<ProfileRow>,
}

/// The numbers a reader scans first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    pub entities: usize,
    pub features: usize,
    pub k: usize,
    pub inertia: f64,
    pub silhouette: f64,
    pub min_share: f64,
    pub sweep: Vec<Sweep>,
    pub stability: Stability,
    pub control: Control,
    pub flat: Vec<String>,
}

/// The fitted artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    pub version: u32,
    pub built_at: String,
    pub source: Source,
    pub config: SegmentConfig,
    pub features: Vec<FeatureSpec>,
    pub scaling: Scaling,
    pub k: usize,
    pub centroids: Vec<Vec<f64>>,
    pub segments: Vec<Segment>,
    pub assignments: BTreeMap<String, usize>,
    pub metrics: Metrics,
}

/// One shipped file and the digest that proves it is the one described.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileStamp {
    pub name: String,
    pub rows: usize,
    pub bytes: u64,
    pub fnv64: String,
}

/// One segment, reduced to the line a manifest carries.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SegmentStamp {
    pub id: usize,
    pub label: String,
    pub size: usize,
}

/// The folder's own account of itself.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: u32,
    pub built_at: String,
    pub files: Vec<FileStamp>,
    pub config: SegmentConfig,
    pub k: usize,
    pub segments: Vec<SegmentStamp>,
    pub checks: Vec<String>,
}

impl Artifact {
    pub fn load(path: &str) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        serde_json::from_str(&text)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{path}: {e}")))
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let text = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, format!("{text}\n"))
    }

    /// Place one set of observations against the fitted standardisation.
    pub fn z(&self, obs: &BTreeMap<String, f64>) -> io::Result<Vec<f64>> {
        let mut row = Vec::with_capacity(self.features.len());
        for (column, spec) in self.features.iter().enumerate() {
            let value = obs.get(&spec.name).copied().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("no reading for {}", spec.name),
                )
            })?;
            row.push(self.scaling.z(column, spec.apply(value)));
        }
        Ok(row)
    }

    /// The nearest segment and the distance to its centre.
    pub fn classify(&self, obs: &BTreeMap<String, f64>) -> io::Result<(usize, f64)> {
        let row = self.z(obs)?;
        let (id, distance) = super::kmeans::nearest(&row, &self.centroids);
        Ok((id, distance.sqrt()))
    }

    pub fn segment(&self, id: usize) -> Option<&Segment> {
        self.segments.iter().find(|s| s.id == id)
    }

    /// Rebuild the profile of a partition from the raw matrix.
    pub fn profiles(&self, fit: &Fit, rows: &super::feature::Matrix) -> Vec<Segment> {
        segments_of(fit, &self.features, rows)
    }
}

/// Read every group back from a partition: size, share and both profiles.
pub fn segments_of(
    fit: &Fit,
    specs: &[FeatureSpec],
    rows: &super::feature::Matrix,
) -> Vec<Segment> {
    let n = rows.raw.len().max(1);
    let mut out = Vec::with_capacity(fit.k);
    for cluster in 0..fit.k {
        let members: Vec<usize> = (0..rows.raw.len())
            .filter(|&i| fit.labels[i] == cluster)
            .collect();
        let mut profile: Vec<ProfileRow> = specs
            .iter()
            .enumerate()
            .map(|(column, spec)| {
                let mean = if members.is_empty() {
                    0.0
                } else {
                    spec.invert(
                        members.iter().map(|&i| rows.raw[i][column]).sum::<f64>()
                            / members.len() as f64,
                    )
                };
                ProfileRow {
                    name: spec.name.clone(),
                    label: spec.label.clone(),
                    family: spec.family.clone(),
                    unit: spec.unit.clone(),
                    z: fit.centroids[cluster][column],
                    mean,
                }
            })
            .collect();
        profile.sort_by(|a, b| {
            b.z.abs()
                .partial_cmp(&a.z.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.name.cmp(&b.name))
        });
        let label = label_of(&profile);
        out.push(Segment {
            id: cluster,
            label,
            size: members.len(),
            share: members.len() as f64 / n as f64,
            profile,
        });
    }
    out
}

/// A name read off the profile: the two strongest readings, above a floor.
///
/// Nothing is typed per segment, so a rebuild names every group the same way.
pub fn label_of(profile: &[ProfileRow]) -> String {
    let mut ranked: Vec<&ProfileRow> = profile.iter().filter(|r| r.z.abs() >= 0.4).collect();
    ranked.sort_by(|a, b| {
        b.z.abs()
            .partial_cmp(&a.z.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });
    let words: Vec<String> = ranked
        .iter()
        .take(2)
        .map(|r| format!("{} {}", r.label, if r.z >= 0.0 { "high" } else { "low" }))
        .collect();
    if words.is_empty() {
        "near the mean".to_string()
    } else {
        words.join(", ")
    }
}

/// UTC timestamp in ISO 8601, so the card carries a readable build time.
pub fn now_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let rem = secs % 86_400;
    format!(
        "{}T{:02}:{:02}:{:02}Z",
        Day((secs / 86_400) as i64),
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Round to four places, the precision a reader is expected to act on.
pub fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(pairs: &[(&str, f64)]) -> Vec<ProfileRow> {
        pairs
            .iter()
            .map(|(name, z)| ProfileRow {
                name: (*name).into(),
                label: (*name).into(),
                family: "test".into(),
                unit: "x".into(),
                z: *z,
                mean: 0.0,
            })
            .collect()
    }

    #[test]
    fn a_label_reads_the_two_strongest_readings() {
        let got = label_of(&profile(&[
            ("burstiness", 1.4),
            ("concentrate", -1.1),
            ("ads", 0.2),
        ]));
        assert_eq!(got, "burstiness high, concentrate low");
    }

    #[test]
    fn a_profile_with_nothing_above_the_floor_says_so() {
        assert_eq!(
            label_of(&profile(&[("ads", 0.1), ("terms", -0.2)])),
            "near the mean"
        );
    }

    #[test]
    fn a_label_is_independent_of_the_order_the_profile_arrived_in() {
        let a = label_of(&profile(&[("ads", 0.9), ("repeat", 1.2)]));
        let b = label_of(&profile(&[("repeat", 1.2), ("ads", 0.9)]));
        assert_eq!(a, b);
    }

    #[test]
    fn a_timestamp_looks_like_an_iso_date() {
        let stamp = now_iso8601();
        assert_eq!(stamp.len(), 20);
        assert!(stamp.ends_with('Z'));
        assert!(Day::parse(&stamp[..10]).is_some());
    }
}
