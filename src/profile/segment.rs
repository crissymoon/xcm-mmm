//! Marketing copy scored against the behavioural segments.
//!
//! The segment model reads eight things about an advertiser, and two of them
//! leave a trace in a single piece of text: copy length, and whether the copy
//! repeats a phrase the way a reusable creative repeats across ads. Targeting
//! terms are readable only when the caller names the buy. Volume, cadence,
//! spend, burstiness and cross-ad concentration are behaviour over time that
//! no text holds, and are named as unobserved rather than guessed.
//!
//! Three things keep the score honest:
//!
//! * The distance runs over the observed columns only, and every placement
//!   names the columns it could read.
//! * Columns are weighted by how far apart the segment centres actually sit on
//!   them. Copy length moves by about a tenth of a standard deviation across
//!   the four segments and reuse moves by a full one, so reuse decides the fit
//!   and copy length barely does. A flat unweighted mean would let a column
//!   that separates nothing pull as hard as the one that separates everything.
//! * `reach` reports how much of the model's between-segment separation the
//!   observed columns carry, and the note repeats it, so a fit of 34% against
//!   four segments is not mistaken for a classification.
//! * Each column's gap is measured from the closest segment on that column, not
//!   from zero. A twenty word text is far from every segment's copy length, and
//!   that common offset says nothing about which segment it resembles; left in,
//!   it would inflate every distance and make the softmax look more decisive
//!   than the evidence.
//!
//! The fit is a softmax over squared distances with the spread of those
//! distances as its width. It always sums to 100 because it is a comparison
//! between the segments on this text, not a probability that any is right.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::segment::Artifact;

/// One text's readings, as [`super::language`] measured them.
#[derive(Clone, Debug, Default)]
pub struct Signal {
    pub words: usize,
    /// Terms per ad, when the caller knows the buy. A copy cannot show the
    /// audience it was bought against, so this stays unobserved without it.
    pub terms: Option<f64>,
    /// Share of the copy's words inside a phrase it uses more than once.
    pub reuse: f64,
    /// The most repeated four word phrase, shown as the hook when present.
    pub phrase: String,
}

/// Columns a text can supply, and how it supplies them.
///
/// The names are the shipped feature names checked against the artifact at read
/// time: a model whose table no longer carries one reports nothing for it
/// rather than inventing a column.
const OBSERVABLE: [(&str, Basis); 3] = [
    ("copy_words", Basis::Words),
    ("terms", Basis::Terms),
    ("repeat", Basis::Reuse),
];

#[derive(Clone, Copy)]
enum Basis {
    /// Words per ad, which for one text is its word count.
    Words,
    /// Terms bought against, which only the caller knows.
    Terms,
    /// Cross-ad reuse, proxied by the share of the copy inside a phrase it
    /// uses more than once.
    Reuse,
}

impl Basis {
    /// Where the number came from, for the view and for a reader checking it.
    fn source(self) -> &'static str {
        match self {
            Basis::Words => "text",
            Basis::Terms => "terms",
            Basis::Reuse => "phrase reuse",
        }
    }
}

/// One text placed against every segment centre, on the observed columns.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentView {
    /// Labels of the dimensions the text supplied.
    pub basis: Vec<String>,
    /// Labels of the dimensions that need behaviour no text holds.
    pub unobserved: Vec<String>,
    /// Share of the between-segment separation the observed columns carry.
    pub reach: f64,
    /// Every segment, best fit first. `fit` sums to 100 across the list.
    pub ranked: Vec<Placement>,
    /// Gap in weighted distance between the nearest and second nearest centre.
    pub margin: f64,
    /// The phrase the copy repeats, when it repeats one: the evidence behind
    /// the reuse proxy.
    pub hook: String,
    /// What the ranking is and is not evidence for.
    pub note: String,
}

/// One segment's fit for the text, with the readings that made it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Placement {
    pub id: usize,
    pub label: String,
    /// Share of the fitted panel this segment holds.
    pub share: f64,
    /// Fit in percent, relative to the other segments on this text.
    pub fit: f64,
    /// Root mean squared z gap over the observed columns, weighted by how far
    /// apart the segment centres sit on each column.
    pub distance: f64,
    /// Columns the text sits closest to this segment on, nearest first.
    pub matches: Vec<String>,
    /// Columns it sits furthest from, furthest first.
    pub clashes: Vec<String>,
    pub readings: Vec<Reading>,
}

/// One observed column: the text's value, the segment's mean, the z gap and
/// the weight that gap carried.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reading {
    pub name: String,
    pub label: String,
    pub unit: String,
    /// How the text supplied the column.
    pub source: String,
    pub value: f64,
    pub segment: f64,
    pub z: f64,
    /// The text's z minus the segment centre: positive means the text runs
    /// above the segment on this column.
    pub gap: f64,
    pub weight: f64,
}

/// Score a text against a fitted segment model.
pub fn place(artifact: &Artifact, signal: &Signal) -> SegmentView {
    let mut observed: BTreeMap<&str, (f64, &'static str)> = BTreeMap::new();
    for (name, basis) in OBSERVABLE {
        if !artifact.features.iter().any(|spec| spec.name == name) {
            continue;
        }
        match basis {
            Basis::Words => {
                observed.insert(name, (signal.words as f64, basis.source()));
            }
            Basis::Terms => {
                if let Some(value) = signal.terms {
                    observed.insert(name, (value.max(0.0), basis.source()));
                }
            }
            Basis::Reuse => {
                observed.insert(name, (signal.reuse.clamp(0.0, 1.0), basis.source()));
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

    let weights = separation(artifact);
    let reach = {
        let seen: f64 = columns.iter().map(|&column| weights[column]).sum();
        let all: f64 = weights.iter().sum();
        if all > 1e-9 {
            seen / all
        } else {
            0.0
        }
    };

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

    // The text's z on each observed column, and the smallest squared gap any
    // segment manages on it. Subtracting that floor leaves only the part of
    // the distance that separates one segment from another.
    let zs: Vec<f64> = columns
        .iter()
        .map(|&column| {
            let spec = &artifact.features[column];
            let (value, _) = observed[spec.name.as_str()];
            artifact.scaling.z(column, spec.apply(value))
        })
        .collect();
    let floors: Vec<f64> = columns
        .iter()
        .enumerate()
        .map(|(at, &column)| {
            artifact
                .centroids
                .iter()
                .filter_map(|centre| centre.get(column))
                .map(|centre| (zs[at] - centre) * (zs[at] - centre))
                .fold(f64::INFINITY, f64::min)
                .min(f64::MAX)
        })
        .collect();

    let mut ranked = Vec::with_capacity(artifact.segments.len());
    if !columns.is_empty() && artifact.centroids.len() == artifact.segments.len() {
        for (id, centroid) in artifact.centroids.iter().enumerate() {
            let segment = artifact.segment(id);
            let mut readings = Vec::with_capacity(columns.len());
            let mut total = 0.0;
            let mut mass = 0.0;
            for (at, &column) in columns.iter().enumerate() {
                let spec = &artifact.features[column];
                let (value, source) = observed[spec.name.as_str()];
                let z = zs[at];
                let gap = z - centroid[column];
                let weight = weights[column].max(0.0);
                let floor = if floors[at].is_finite() {
                    floors[at]
                } else {
                    0.0
                };
                total += weight * (gap * gap - floor).max(0.0);
                mass += weight;
                // The centroid is a mean on the transformed scale, so reading
                // it back in units means undoing the standardisation first.
                let mean = artifact.scaling.mean.get(column).copied().unwrap_or(0.0);
                let sd = artifact.scaling.sd.get(column).copied().unwrap_or(0.0);
                readings.push(Reading {
                    name: spec.name.clone(),
                    label: spec.label.clone(),
                    unit: spec.unit.clone(),
                    source: source.to_string(),
                    value,
                    segment: spec.invert(mean + sd * centroid[column]),
                    z,
                    gap,
                    weight,
                });
            }
            let distance = if mass > 1e-9 {
                (total / mass).sqrt()
            } else {
                (total / columns.len() as f64).sqrt()
            };
            let mut order: Vec<usize> = (0..readings.len()).collect();
            order.sort_by(|a, b| {
                let pull = |at: usize| readings[at].gap.abs() * readings[at].weight;
                pull(*a)
                    .total_cmp(&pull(*b))
                    .then_with(|| readings[*a].label.cmp(&readings[*b].label))
            });
            // Only a column that carried a real share of the distance can be
            // said to match or clash: a column with a tenth of the weight of
            // the strongest one moves the ranking by rounding.
            let top_weight = readings.iter().map(|r| r.weight).fold(0.0, f64::max);
            let carries = |at: usize| readings[at].weight >= top_weight * 0.25;
            let live: Vec<usize> = order.iter().copied().filter(|at| carries(*at)).collect();
            let (matches, clashes) = if live.len() > 1 {
                (
                    live.iter()
                        .take(2)
                        .map(|at| readings[*at].label.clone())
                        .collect(),
                    live.iter()
                        .rev()
                        .take(2)
                        .map(|at| readings[*at].label.clone())
                        .collect(),
                )
            } else {
                (Vec::new(), Vec::new())
            };
            ranked.push(Placement {
                id,
                label: segment.map(|s| s.label.clone()).unwrap_or_default(),
                share: segment.map(|s| s.share).unwrap_or(0.0),
                fit: 0.0,
                distance,
                matches,
                clashes,
                readings,
            });
        }
    }
    ranked.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.id.cmp(&b.id))
    });

    let spread: f64 = if ranked.is_empty() {
        0.0
    } else {
        ranked
            .iter()
            .map(|at| at.distance * at.distance)
            .sum::<f64>()
            / ranked.len() as f64
    };
    let width = spread.max(1e-6);
    let scores: Vec<f64> = ranked
        .iter()
        .map(|at| (-(at.distance * at.distance) / (2.0 * width)).exp())
        .collect();
    let sum: f64 = scores.iter().sum();
    let even = 100.0 / ranked.len().max(1) as f64;
    for (at, score) in ranked.iter_mut().zip(scores.iter()) {
        at.fit = if sum > 0.0 {
            (score / sum) * 100.0
        } else {
            even
        };
    }

    let margin = if ranked.len() > 1 {
        ranked[1].distance - ranked[0].distance
    } else {
        0.0
    };

    let note = if columns.is_empty() {
        format!(
            "this model carries no column a text can supply ({} of {} dimensions unobserved)",
            unobserved.len(),
            artifact.features.len()
        )
    } else if reach < 0.25 {
        format!(
            "the copy supplies {} of {} dimensions and those carry {:.0}% of the separation between segments: the fit ranks creative style, it does not assign a segment, because volume, cadence and spend decide that",
            columns.len(),
            artifact.features.len(),
            reach * 100.0
        )
    } else {
        format!(
            "the copy supplies {} of {} dimensions and those carry {:.0}% of the separation between segments; the fit is relative to the other segments, not a probability that either is right",
            columns.len(),
            artifact.features.len(),
            reach * 100.0
        )
    };

    SegmentView {
        basis,
        unobserved,
        reach,
        ranked,
        margin,
        hook: signal.phrase.clone(),
        note,
    }
}

/// How far apart the segment centres sit on each column, in z.
///
/// This is the between-segment standard deviation of the centroids. It is the
/// weighting the fit uses because it is exactly the amount of separation a
/// column contributes to the model, so a column whose centres all sit at zero
/// cannot outvote one whose centres span the axis.
fn separation(artifact: &Artifact) -> Vec<f64> {
    let k = artifact.centroids.len().max(1) as f64;
    (0..artifact.features.len())
        .map(|column| {
            let mean: f64 = artifact
                .centroids
                .iter()
                .map(|centre| centre.get(column).copied().unwrap_or(0.0))
                .sum::<f64>()
                / k;
            let spread: f64 = artifact
                .centroids
                .iter()
                .map(|centre| {
                    let value = centre.get(column).copied().unwrap_or(0.0);
                    (value - mean) * (value - mean)
                })
                .sum::<f64>()
                / k;
            spread.sqrt()
        })
        .collect()
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
    fn a_text_supplies_three_columns_and_names_the_rest() {
        let artifact = shipped();
        let view = place(
            &artifact,
            &Signal {
                words: 240,
                terms: None,
                reuse: 0.2,
                phrase: String::new(),
            },
        );
        assert_eq!(view.basis.len(), 2);
        assert_eq!(view.unobserved.len(), artifact.features.len() - 2);
        assert_eq!(view.ranked.len(), artifact.segments.len());
        assert!(view
            .ranked
            .windows(2)
            .all(|w| w[0].distance <= w[1].distance));
        let total: f64 = view.ranked.iter().map(|at| at.fit).sum();
        assert!((total - 100.0).abs() < 0.001, "{total}");
    }

    #[test]
    fn reuse_decides_the_fit_because_it_separates_the_segments() {
        let artifact = shipped();
        let weights = separation(&artifact);
        let names: Vec<&str> = artifact.features.iter().map(|s| s.name.as_str()).collect();
        let at = |name: &str| names.iter().position(|n| *n == name).unwrap();
        assert!(
            weights[at("repeat")] > weights[at("copy_words")] * 5.0,
            "reuse {} vs copy length {}",
            weights[at("repeat")],
            weights[at("copy_words")]
        );
        assert!(
            weights[at("concentrate")] > weights[at("terms")] * 5.0,
            "concentration {} vs terms {}",
            weights[at("concentrate")],
            weights[at("terms")]
        );
    }

    #[test]
    fn a_repeating_text_fits_the_reuse_segment_better_than_a_flat_one() {
        let artifact = shipped();
        let signal = |reuse: f64| Signal {
            words: 70,
            terms: None,
            reuse,
            phrase: String::new(),
        };
        let repeating = place(&artifact, &signal(0.9));
        let flat = place(&artifact, &signal(0.0));
        assert_eq!(repeating.ranked[0].label, "creative reuse high");
        assert_ne!(flat.ranked[0].id, repeating.ranked[0].id);
    }

    #[test]
    fn terms_are_given_or_derived_and_never_dropped() {
        let artifact = shipped();
        let base = Signal {
            words: 100,
            terms: None,
            reuse: 0.1,
            phrase: String::new(),
        };
        let derived = place(&artifact, &base);
        let given = place(
            &artifact,
            &Signal {
                terms: Some(12.0),
                ..base
            },
        );
        assert_eq!(derived.basis.len(), 2);
        assert_eq!(given.basis.len(), 3);
        let reading = given.ranked[0]
            .readings
            .iter()
            .find(|reading| reading.name == "terms")
            .unwrap();
        assert_eq!(reading.value, 12.0);
        assert_eq!(reading.source, "terms");
    }

    #[test]
    fn reach_says_the_copy_sees_a_small_share_of_the_model() {
        let artifact = shipped();
        let view = place(
            &artifact,
            &Signal {
                words: 70,
                terms: None,
                reuse: 0.3,
                phrase: String::new(),
            },
        );
        assert!(view.reach > 0.0 && view.reach < 1.0, "{}", view.reach);
        assert!(view.note.contains("does not assign") || view.note.contains("not a probability"));
    }
}
