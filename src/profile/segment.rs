//! How one text compares with the behavioural segments, on the part a text can
//! supply.
//!
//! The segment model reads eight things about an advertiser, and only one of
//! them is visible in a single piece of copy: how long the copy runs. Targeting
//! terms are visible only when the caller names the buy, and volume, cadence,
//! spend, reuse and concentration are behaviour over time that no text holds.
//!
//! So this module does not pretend to assign a text to a segment. It computes
//! the distance to every centroid **over the observed columns only**, names the
//! columns it could not observe, and lets the reader see how thin the evidence
//! is. A one column match is a hint about copy length, not a classification,
//! and the note in the view says exactly that.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::segment::Artifact;

/// Columns a text can supply, and how the text supplies them.
///
/// The names are the shipped feature names, checked against the artifact at
/// read time: a model whose table no longer carries one of these simply
/// reports nothing for it rather than inventing a column.
const OBSERVABLE: [(&str, Basis); 2] = [("copy_words", Basis::Words), ("terms", Basis::Terms)];

#[derive(Clone, Copy)]
enum Basis {
    /// Words per ad, which for one text is its word count.
    Words,
    /// Terms bought against, which only the caller knows.
    Terms,
}

/// One text placed against every segment centre, on the observed columns.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentView {
    /// Labels of the dimensions the text supplied.
    pub basis: Vec<String>,
    /// Labels of the dimensions that need behaviour no text holds.
    pub unobserved: Vec<String>,
    /// Every segment, nearest first.
    pub ranked: Vec<Placement>,
    /// Gap in rmse between the nearest and second nearest centre. A gap this
    /// small relative to the distances themselves is noise, not a match.
    pub margin: f64,
    /// What the ranking is and is not evidence for.
    pub note: String,
}

/// One segment's distance from the text, with the readings that made it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Placement {
    pub id: usize,
    pub label: String,
    /// Share of the fitted panel this segment holds.
    pub share: f64,
    /// Root mean squared z distance over the observed columns only.
    pub rmse: f64,
    pub readings: Vec<Reading>,
}

/// One observed column: the text's value, the segment's mean, and the z gap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reading {
    pub name: String,
    pub label: String,
    pub unit: String,
    pub value: f64,
    pub segment: f64,
    pub z: f64,
}

/// Place a text against a fitted segment model.
///
/// `terms` is the number of targeting terms the ad was bought against, when
/// the caller knows it. `None` leaves that column unobserved rather than zero.
pub fn place(artifact: &Artifact, words: usize, terms: Option<f64>) -> SegmentView {
    let mut observed: BTreeMap<&str, f64> = BTreeMap::new();
    for (name, basis) in OBSERVABLE {
        if !artifact.features.iter().any(|spec| spec.name == name) {
            continue;
        }
        match basis {
            Basis::Words => {
                observed.insert(name, words as f64);
            }
            Basis::Terms => {
                if let Some(value) = terms {
                    observed.insert(name, value.max(0.0));
                }
            }
        }
    }

    let columns: Vec<usize> = artifact
        .features
        .iter()
        .enumerate()
        .filter(|(_, spec)| observed.contains_key(spec.name.as_str()))
        .map(|(column, _)| column)
        .collect();

    let basis = columns
        .iter()
        .map(|&column| artifact.features[column].label.clone())
        .collect();
    let unobserved = artifact
        .features
        .iter()
        .filter(|spec| !observed.contains_key(spec.name.as_str()))
        .map(|spec| spec.label.clone())
        .collect::<Vec<_>>();

    let mut ranked = Vec::with_capacity(artifact.segments.len());
    if !columns.is_empty() {
        for (id, centroid) in artifact.centroids.iter().enumerate() {
            let segment = artifact.segment(id);
            let mut readings = Vec::with_capacity(columns.len());
            let mut total = 0.0;
            for &column in &columns {
                let spec = &artifact.features[column];
                let value = observed[spec.name.as_str()];
                let z = artifact.scaling.z(column, spec.apply(value));
                let gap = z - centroid[column];
                total += gap * gap;
                // The centroid is a mean on the transformed scale, so reading
                // it back in units means undoing the standardisation first.
                let mean = artifact.scaling.mean.get(column).copied().unwrap_or(0.0);
                let sd = artifact.scaling.sd.get(column).copied().unwrap_or(0.0);
                readings.push(Reading {
                    name: spec.name.clone(),
                    label: spec.label.clone(),
                    unit: spec.unit.clone(),
                    value,
                    segment: spec.invert(mean + sd * centroid[column]),
                    z,
                });
            }
            ranked.push(Placement {
                id,
                label: segment.map(|s| s.label.clone()).unwrap_or_default(),
                share: segment.map(|s| s.share).unwrap_or(0.0),
                rmse: (total / columns.len() as f64).sqrt(),
                readings,
            });
        }
    }
    ranked.sort_by(|a, b| a.rmse.total_cmp(&b.rmse).then_with(|| a.id.cmp(&b.id)));
    let margin = if ranked.len() > 1 {
        ranked[1].rmse - ranked[0].rmse
    } else {
        0.0
    };

    let note = if columns.is_empty() {
        format!(
            "this model carries no column a text can supply ({} of {} dimensions unobserved)",
            unobserved.len(),
            artifact.features.len()
        )
    } else {
        format!(
            "{} of {} dimensions observed; the distance compares only those, so this ranks copy length, it does not assign a segment",
            columns.len(),
            artifact.features.len()
        )
    };

    SegmentView {
        basis,
        unobserved,
        ranked,
        margin,
        note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped() -> Artifact {
        Artifact::load(&format!(
            "{}/{}",
            crate::segment::FOLDER,
            crate::segment::MODEL
        ))
        .unwrap()
    }

    #[test]
    fn the_observable_names_are_the_ones_the_shipped_table_uses() {
        let artifact = shipped();
        for (name, _) in OBSERVABLE {
            assert!(
                artifact.features.iter().any(|spec| spec.name == name),
                "no column named {name}"
            );
        }
    }

    #[test]
    fn a_text_observes_copy_length_and_names_the_rest() {
        let artifact = shipped();
        let view = place(&artifact, 240, None);
        assert_eq!(view.basis.len(), 1);
        assert_eq!(view.unobserved.len(), artifact.features.len() - 1);
        assert_eq!(view.ranked.len(), artifact.segments.len());
        assert!(view.ranked[0].rmse <= view.ranked[1].rmse);
    }

    #[test]
    fn terms_are_observed_only_when_given() {
        let artifact = shipped();
        assert_eq!(place(&artifact, 100, None).basis.len(), 1);
        assert_eq!(place(&artifact, 100, Some(12.0)).basis.len(), 2);
    }
}
