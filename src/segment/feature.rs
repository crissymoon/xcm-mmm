//! The feature table and the standardisation the fit reads.
//!
//! Eight readings describe what an advertiser did: how many ads it started, how
//! far apart the starts were, how uneven that spacing was, what it paid at the
//! band midpoint, how long its copy ran, how many terms it bought against, how
//! much of its copy repeated, and how far one fingerprint dominated it.
//!
//! The table is data, not prose: it travels in the artifact with the transform
//! it declares, so a reader can reproduce a z by hand. Standardisation is the
//! whole of the scaling, and the z is clipped at three standard deviations
//! because a single advertiser with an almost constant reading would otherwise
//! become a segment of its own and pull a centroid with it.
//!
//! A reading declared as a share may also declare an edge, and a share with a
//! large pile on that edge is held out of the clip and described by a gate: see
//! [`super::gate`] for why a pile is a probability and not a density, and for the
//! measurement that decides which shares get one.

use std::io;

use serde::{Deserialize, Serialize};

use super::ad::{mean, spread, Entity};

/// How a reading in its own units becomes the column the fit sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transform {
    /// `ln(1 + x)`, for counts and dollars, whose tails are long.
    Log1p,
    /// The reading as it stands, for shares and a capped dispersion.
    Identity,
}

/// One column of the fit, described well enough to rebuild it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureSpec {
    /// The key in an entity's observations.
    pub name: String,
    /// The words a segment name uses.
    pub label: String,
    /// Volume, cadence, value, creative or targeting.
    pub family: String,
    /// The unit of the untransformed reading.
    pub unit: String,
    pub transform: Transform,
    /// Where a share piles up, in plain units, when it is a share at all.
    ///
    /// This is a declaration about the reading, not a decision: the fit reads the
    /// panel and gates a declared edge only when enough rows sit on it.
    #[serde(default)]
    pub boundary: Option<f64>,
    pub description: String,
}

impl FeatureSpec {
    /// The value the fit sees for one observation.
    pub fn apply(&self, value: f64) -> f64 {
        self.transform.apply(value)
    }

    /// The reading read back from a fitted value, in its own units.
    pub fn invert(&self, value: f64) -> f64 {
        self.transform.invert(value)
    }
}

impl Transform {
    pub fn apply(&self, value: f64) -> f64 {
        match self {
            Transform::Log1p => (1.0 + value).ln(),
            Transform::Identity => value,
        }
    }

    /// The inverse of `apply`, so a profile can be quoted in dollars and words
    /// rather than in the space the fit worked in.
    pub fn invert(&self, value: f64) -> f64 {
        match self {
            Transform::Log1p => value.exp_m1(),
            Transform::Identity => value,
        }
    }
}

/// The declared table. Order is the column order of the fit and never changes
/// silently: a z is read against the same column it was written from.
pub fn specs() -> Vec<FeatureSpec> {
    let row = |name: &str,
               label: &str,
               family: &str,
               unit: &str,
               transform: Transform,
               description: &str| FeatureSpec {
        name: name.into(),
        label: label.into(),
        family: family.into(),
        unit: unit.into(),
        transform,
        boundary: None,
        description: description.into(),
    };
    // A share always takes its edge with it, so the two that pile up are named
    // once rather than tested against a literal at the point of use.
    let share = |name: &str, label: &str, boundary: f64, description: &str| FeatureSpec {
        name: name.into(),
        label: label.into(),
        family: "creative".into(),
        unit: "share".into(),
        transform: Transform::Identity,
        boundary: Some(boundary),
        description: description.into(),
    };
    vec![
        row(
            "ads",
            "ad volume",
            "volume",
            "ads",
            Transform::Log1p,
            "Ads the advertiser started inside the panel.",
        ),
        row(
            "cadence",
            "time between starts",
            "cadence",
            "days",
            Transform::Log1p,
            "Median gap between the days it started an ad.",
        ),
        row(
            "burstiness",
            "start gap spread",
            "cadence",
            "coefficient of variation",
            Transform::Identity,
            "Population standard deviation of the gaps over their mean, capped at 3.",
        ),
        row(
            "spend_mid",
            "banded spend",
            "value",
            "usd per ad",
            Transform::Log1p,
            "Mean midpoint of the spend band the archive published per ad.",
        ),
        row(
            "copy_words",
            "copy length",
            "creative",
            "words",
            Transform::Log1p,
            "Median words per ad.",
        ),
        row(
            "terms",
            "targeting terms",
            "targeting",
            "terms per ad",
            Transform::Identity,
            "Mean number of terms the archive lists against an ad.",
        ),
        share(
            "repeat",
            "creative reuse",
            0.0,
            "Share of its ads whose copy fingerprint appears more than once for it.",
        ),
        share(
            "concentrate",
            "copy concentration",
            1.0,
            "Largest share of its ads that carry one fingerprint.",
        ),
    ]
}

/// A fitted standardisation, the z matrix and the raw matrix beside it.
///
/// The raw matrix is kept because a profile reads better in dollars and words
/// than in z, and because the card quotes both.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scaling {
    pub mean: Vec<f64>,
    pub sd: Vec<f64>,
    pub clip: f64,
    /// Columns whose pile is described by a gate, so the clip is not applied.
    ///
    /// A share with a boundary at 1 and a small spread can put its edge beyond
    /// the clip, and clipping it would fold the edge into the interior and make
    /// the gate's boundary value meaningless.
    #[serde(default)]
    pub clip_exempt: Vec<usize>,
}

impl Scaling {
    pub fn z(&self, column: usize, value: f64) -> f64 {
        let sd = self.sd.get(column).copied().unwrap_or(1.0);
        let mu = self.mean.get(column).copied().unwrap_or(0.0);
        let z = if sd > 1e-9 { (value - mu) / sd } else { 0.0 };
        if self.clip_exempt.contains(&column) {
            z
        } else {
            z.clamp(-self.clip, self.clip)
        }
    }
}

/// The two matrices a fit reads: columns in units, and columns as z.
pub struct Matrix {
    pub specs: Vec<FeatureSpec>,
    pub raw: Vec<Vec<f64>>,
    pub z: Vec<Vec<f64>>,
    pub scaling: Scaling,
    /// Columns that do not vary, which carry no signal and are reported.
    pub flat: Vec<String>,
    /// Shares whose pile is large enough to be read as a gate.
    pub gates: Vec<super::gate::GateSpec>,
}

impl Matrix {
    /// Build from entities. A missing observation is a fault, not a zero: the
    /// extractor wrote the file and has to have written every column.
    pub fn build(
        entities: &[Entity],
        specs: &[FeatureSpec],
        clip: f64,
        gate_share: f64,
    ) -> io::Result<Self> {
        if entities.is_empty() {
            return Err(bad("no entities to fit"));
        }
        let columns = specs.len();
        let mut raw = Vec::with_capacity(entities.len());
        for entity in entities {
            let mut row = Vec::with_capacity(columns);
            for spec in specs {
                let value = entity.obs.get(&spec.name).copied().ok_or_else(|| {
                    bad(&format!(
                        "entity {} has no reading for {}",
                        entity.id, spec.name
                    ))
                })?;
                if !value.is_finite() {
                    return Err(bad(&format!(
                        "entity {} has a non finite {}",
                        entity.id, spec.name
                    )));
                }
                row.push(spec.apply(value));
            }
            raw.push(row);
        }

        let mut mu = vec![0.0; columns];
        let mut sigma = vec![0.0; columns];
        for column in 0..columns {
            let values: Vec<f64> = raw.iter().map(|row| row[column]).collect();
            mu[column] = mean(&values);
            sigma[column] = spread(&values);
        }
        let gates = super::gate::detect(specs, &raw, &mu, &sigma, gate_share);
        let scaling = Scaling {
            mean: mu,
            sd: sigma,
            clip: clip.max(0.5),
            clip_exempt: gates.iter().map(|gate| gate.column).collect(),
        };
        let z = raw
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(column, value)| scaling.z(column, *value))
                    .collect()
            })
            .collect();
        let flat = specs
            .iter()
            .enumerate()
            .filter(|(column, _)| scaling.sd[*column] <= 1e-9)
            .map(|(_, spec)| spec.name.clone())
            .collect();
        Ok(Self {
            specs: specs.to_vec(),
            raw,
            z,
            scaling,
            flat,
            gates,
        })
    }

    pub fn len(&self) -> usize {
        self.raw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }
}

fn bad(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::ad::Entity;
    use std::collections::BTreeMap;

    fn entity(id: &str, ads: f64, cadence: f64) -> Entity {
        let digit = id
            .bytes()
            .rfind(|b| b.is_ascii_digit())
            .map(|b| (b - b'0') as f64)
            .unwrap_or(1.0);
        let mut obs = BTreeMap::new();
        for (i, spec) in specs().iter().enumerate() {
            obs.insert(spec.name.clone(), digit + i as f64 * 0.5);
        }
        obs.insert("ads".into(), ads);
        obs.insert("cadence".into(), cadence);
        Entity {
            id: id.into(),
            name: id.into(),
            ads: ads as usize,
            test_ads: 0,
            first: "2026-01-01".into(),
            last: "2026-02-01".into(),
            obs,
        }
    }

    #[test]
    fn the_table_has_one_row_per_reading_and_no_duplicates() {
        let table = specs();
        assert_eq!(table.len(), 8);
        let mut names: Vec<&String> = table.iter().map(|s| &s.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), table.len());
        assert!(table.iter().all(|s| !s.description.is_empty()));
    }

    #[test]
    fn a_z_is_clipped_and_reads_against_its_own_column() {
        let entities: Vec<Entity> = (0..6)
            .map(|i| entity(&format!("e{i}"), i as f64 + 1.0, (i * i) as f64))
            .collect();
        let matrix = Matrix::build(&entities, &specs(), 3.0, 0.5).unwrap();
        for row in &matrix.z {
            assert!(row.iter().all(|z| z.abs() <= 3.0));
        }
        assert!(matrix.flat.is_empty());
    }

    #[test]
    fn a_flat_column_is_reported_rather_than_dividing_by_zero() {
        let entities: Vec<Entity> = (0..5)
            .map(|i| entity(&format!("e{i}"), i as f64, 4.0))
            .collect();
        let matrix = Matrix::build(&entities, &specs(), 3.0, 0.5).unwrap();
        assert!(matrix.flat.contains(&"cadence".to_string()));
        assert!(matrix.z.iter().all(|row| row[1] == 0.0));
    }

    #[test]
    fn a_missing_reading_is_a_fault() {
        let mut one = entity("e1", 1.0, 2.0);
        one.obs.remove("terms");
        assert!(Matrix::build(&[one], &specs(), 3.0, 0.5).is_err());
    }
}
