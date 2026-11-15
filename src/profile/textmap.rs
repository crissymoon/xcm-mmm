//! A learned map from copy to segment, as an artifact a report can read.
//!
//! The geometric view in [`super::segment`] places a text on the two columns it
//! can supply. This module learns the placement instead: a multinomial logistic
//! map from what the copy reads as, and which words it uses, onto the segment an
//! advertiser was assigned by its behaviour.
//!
//! Two units are fitted and they are not interchangeable:
//!
//! * `ad`, one ad's copy against the label of the advertiser that ran it. The
//!   measured signal is thin, because volume, cadence, spend and burstiness are
//!   the columns that separate the segments and none of them is in a sentence.
//!   The unit is shipped with its measurement and is withheld from a report
//!   unless it clears the gate.
//! * `brand`, the mean of an advertiser's copy set against its own label. A set
//!   of ads shows cross-ad reuse and vocabulary breadth that one ad cannot, and
//!   this unit clears the gate.
//!
//! Nothing here is an accuracy unless it was measured on advertisers the map
//! did not see: the folds are grouped by page id, so every ad of a held out
//! advertiser is held out with it. The artifact carries the per-class recall,
//! the majority rate it has to beat and the shuffled-label control it was
//! measured against, and [`Unit::usable`] is the gate's answer rather than a
//! judgement typed in.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::segment::Transform;
use crate::text::hash;

use super::language::LanguageView;

/// File the map ships as, inside the segment folder.
pub const TEXTMAP: &str = "textmap.json";
/// The derived per-ad rows the map is fitted from. No ad copy travels here:
/// only the readings and the vocabulary buckets, so a rebuild needs no panel.
pub const TEXT_FEATURES: &str = "text_features.jsonl";

/// Buckets the vocabulary block hashes into. Small enough that a brand's ad
/// set fills every bucket, large enough that common words land apart.
pub const HASH_DIM: usize = 96;

/// The units a map can be fitted at, smallest first.
pub const UNITS: [&str; 2] = ["ad", "brand"];

/// One surface column a text supplies without being told anything else.
#[derive(Clone, Copy, Debug)]
pub struct Column {
    pub name: &'static str,
    pub label: &'static str,
    pub transform: Transform,
    pub description: &'static str,
}

/// The columns read off one text, in the order they enter the map.
///
/// These are the readings [`LanguageView`] already carries, so a report and a
/// fit cannot disagree about a text's length or its reuse.
pub const SURFACE: [Column; 11] = [
    Column {
        name: "log_words",
        label: "copy length",
        transform: Transform::Log1p,
        description: "words in the copy",
    },
    Column {
        name: "distinct",
        label: "distinct words",
        transform: Transform::Identity,
        description: "distinct lowercased words",
    },
    Column {
        name: "type_token",
        label: "type/token ratio",
        transform: Transform::Identity,
        description: "distinct words over words",
    },
    Column {
        name: "sentence_words",
        label: "words per sentence",
        transform: Transform::Log1p,
        description: "mean words per sentence",
    },
    Column {
        name: "second_person",
        label: "second person",
        transform: Transform::Identity,
        description: "share of you and your",
    },
    Column {
        name: "numbers",
        label: "numbers per 100",
        transform: Transform::Log1p,
        description: "digit runs per hundred words",
    },
    Column {
        name: "questions",
        label: "questions",
        transform: Transform::Identity,
        description: "question marks",
    },
    Column {
        name: "exclamations",
        label: "exclamations",
        transform: Transform::Identity,
        description: "exclamation marks",
    },
    Column {
        name: "emotional_density",
        label: "emotional density",
        transform: Transform::Identity,
        description: "share of words above the neutral prior",
    },
    Column {
        name: "reuse",
        label: "phrase reuse",
        transform: Transform::Identity,
        description: "share of words inside a repeated four word phrase",
    },
    Column {
        name: "focus",
        label: "top phrase focus",
        transform: Transform::Identity,
        description: "share of words inside the most repeated phrase",
    },
];

/// Columns that only a set of ads can supply, added to the brand unit.
pub const SET_COLUMN: Column = Column {
    name: "ads_gathered",
    label: "ads in the set",
    transform: Transform::Log1p,
    description: "ads supplied for one advertiser",
};

/// The surface readings of one view, transformed, in [`SURFACE`] order.
pub fn surface(view: &LanguageView) -> Vec<f64> {
    vec![
        SURFACE[0].transform.apply(view.words as f64),
        SURFACE[1].transform.apply(view.distinct as f64),
        SURFACE[2].transform.apply(view.type_token),
        SURFACE[3].transform.apply(view.sentence_words),
        SURFACE[4].transform.apply(view.second_person),
        SURFACE[5].transform.apply(view.numbers_per_100),
        SURFACE[6].transform.apply(view.questions as f64),
        SURFACE[7].transform.apply(view.exclamations as f64),
        SURFACE[8].transform.apply(view.emotional_density),
        SURFACE[9].transform.apply(view.reuse),
        SURFACE[10].transform.apply(view.focus),
    ]
}

/// The vocabulary block: content words hashed into [`HASH_DIM`] buckets and
/// scaled to unit length, so a long ad does not shout over a short one.
pub fn vocabulary(words: &[String]) -> Vec<f64> {
    let mut block = vec![0.0; HASH_DIM];
    for word in words {
        if word.chars().count() < 3 || crate::detect::token::is_function(word) {
            continue;
        }
        let bucket = (hash::word_hash(word) % HASH_DIM as u64) as usize;
        block[bucket] += 1.0;
    }
    let norm = block.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 1e-9 {
        for value in &mut block {
            *value /= norm;
        }
    }
    block
}

/// One ad's row: surface readings then the vocabulary block.
pub fn features(view: &LanguageView, words: &[String]) -> Vec<f64> {
    let mut row = surface(view);
    row.extend(vocabulary(words));
    row
}

/// One advertiser's row: the mean of its ads, the size of the set, and the
/// vocabulary renormalised after averaging.
pub fn brand(rows: &[Vec<f64>], ads: usize) -> Vec<f64> {
    let mut out = vec![0.0; SURFACE.len() + HASH_DIM + 1];
    if rows.is_empty() {
        return out;
    }
    for row in rows {
        for (column, value) in row.iter().enumerate() {
            out[column] += value;
        }
    }
    let n = rows.len() as f64;
    for value in out.iter_mut().take(SURFACE.len() + HASH_DIM) {
        *value /= n;
    }
    let norm = out[SURFACE.len()..SURFACE.len() + HASH_DIM]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm > 1e-9 {
        for value in out[SURFACE.len()..SURFACE.len() + HASH_DIM].iter_mut() {
            *value /= norm;
        }
    }
    out[SURFACE.len() + HASH_DIM] = SET_COLUMN.transform.apply(ads as f64);
    out
}

/// Column names for one unit, in the order the weights use them.
pub fn columns(unit: &str) -> Vec<String> {
    let mut names: Vec<String> = SURFACE.iter().map(|c| c.name.to_string()).collect();
    names.extend((0..HASH_DIM).map(|bucket| format!("vocab_{bucket:02}")));
    if unit == "brand" {
        names.push(SET_COLUMN.name.to_string());
    }
    names
}

/// How a text map is fitted. Carried into the artifact so a rebuild uses the
/// settings the shipped map was built with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextConfig {
    pub name: String,
    /// Folds for the grouped cross-validation, and the only number the gate
    /// trusts.
    pub folds: usize,
    pub iterations: usize,
    pub learning_rate: f64,
    /// L2 penalty on the weights, tuned on the grouped folds.
    pub l2: f64,
    /// How far above chance the balanced accuracy has to stand.
    pub margin: f64,
    /// How much of the distance from chance to perfect the map has to cover.
    pub min_lift: f64,
    /// Shuffled-label controls, each a full cross-validation.
    pub controls: usize,
    /// Smallest unit the fit will touch.
    pub min_rows: usize,
    pub seed: u64,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            name: "marketlab-copy-segments".to_string(),
            folds: 5,
            iterations: 900,
            learning_rate: 0.6,
            l2: 0.03,
            margin: 0.02,
            min_lift: 0.10,
            controls: 5,
            min_rows: 60,
            seed: 20261114,
        }
    }
}

/// Where the rows came from.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub panel: String,
    pub model: String,
    pub ads: usize,
    pub entities: usize,
    pub fnv64: String,
}

/// One class as the fit measured it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassRow {
    pub id: usize,
    pub label: String,
    pub share: f64,
    /// Rows of this class in the training unit.
    pub train: usize,
    /// Share of this class the held-out folds recovered.
    pub recall: f64,
}

/// The panel's own split, reported beside the grouped folds because the two
/// disagree and the reason matters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitReading {
    pub train: usize,
    pub test: usize,
    pub accuracy: f64,
}

/// One fitted unit: the standardisation, the weights and what they achieved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Unit {
    pub unit: String,
    pub rows: usize,
    pub classes: Vec<ClassRow>,
    /// Share of held-out rows given the right segment.
    pub accuracy: f64,
    /// Mean per-class recall, the reading a majority class cannot inflate.
    pub balanced: f64,
    /// One over the number of classes: what guessing would score on `balanced`.
    pub chance: f64,
    /// Share of held-out rows the largest class alone would get right.
    pub majority: f64,
    /// `(balanced - chance) / (1 - chance)`.
    pub lift: f64,
    /// Mean balanced accuracy of the shuffled-label controls.
    pub control: f64,
    /// The gate's answer, and the only reason a report shows this unit.
    pub usable: bool,
    pub confusion: Vec<Vec<usize>>,
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub mean: Vec<f64>,
    pub sd: Vec<f64>,
    /// The panel split, when the unit could be read that way.
    pub split: Option<SplitReading>,
    pub note: String,
}

/// The fitted map, as it ships.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextMap {
    pub name: String,
    pub version: u32,
    pub built_at: String,
    pub source: Source,
    pub config: TextConfig,
    /// Columns per unit, so a weight vector reads without counting buckets.
    pub columns: BTreeMap<String, Vec<String>>,
    pub units: BTreeMap<String, Unit>,
    pub note: String,
}

/// One segment's share of the map's belief about a text or a set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scored {
    pub id: usize,
    pub label: String,
    pub share: f64,
    /// Percent, and the ranked list sums to 100.
    pub probability: f64,
}

/// A map's answer about one input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ranking {
    pub unit: String,
    pub top: usize,
    pub top_label: String,
    pub ranked: Vec<Scored>,
}

impl TextMap {
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

    /// The fitted unit, when it exists and cleared its gate.
    pub fn usable(&self, unit: &str) -> Option<&Unit> {
        self.units.get(unit).filter(|unit| unit.usable)
    }

    /// The fitted unit whether or not it cleared the gate, so a report can say
    /// why it was withheld.
    pub fn unit(&self, unit: &str) -> Option<&Unit> {
        self.units.get(unit)
    }

    /// Score one row against a unit. `None` when the row is the wrong width or
    /// the unit was not cleared for use.
    pub fn predict(&self, unit: &str, row: &[f64]) -> Option<Ranking> {
        let fitted = self.usable(unit)?;
        if fitted.weights.is_empty() || fitted.weights[0].len() != row.len() {
            return None;
        }
        let z: Vec<f64> = row
            .iter()
            .enumerate()
            .map(|(column, value)| (value - fitted.mean[column]) / fitted.sd[column])
            .collect();
        let mut logits: Vec<f64> = fitted
            .weights
            .iter()
            .zip(&fitted.bias)
            .map(|(weights, bias)| {
                bias + weights
                    .iter()
                    .zip(&z)
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>()
            })
            .collect();
        let top = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for logit in &mut logits {
            *logit = (*logit - top).exp();
        }
        let total: f64 = logits.iter().sum();
        let mut ranked: Vec<Scored> = fitted
            .classes
            .iter()
            .zip(&logits)
            .map(|(class, mass)| Scored {
                id: class.id,
                label: class.label.clone(),
                share: class.share,
                probability: if total > 1e-12 {
                    mass / total * 100.0
                } else {
                    0.0
                },
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        let first = ranked.first()?;
        Some(Ranking {
            unit: unit.to_string(),
            top: first.id,
            top_label: first.label.clone(),
            ranked,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(words: usize, reuse: f64) -> LanguageView {
        LanguageView {
            words,
            distinct: words / 2,
            content_terms: words / 3,
            type_token: 0.5,
            sentence_words: 8.0,
            second_person: 0.1,
            numbers_per_100: 2.0,
            questions: 0,
            exclamations: 1,
            emotional_density: 0.2,
            reuse,
            focus: 0.0,
            top_phrase: String::new(),
            too_short: false,
        }
    }

    fn unit() -> Unit {
        let classes = vec![
            ClassRow {
                id: 0,
                label: "a".into(),
                share: 0.5,
                train: 2,
                recall: 0.5,
            },
            ClassRow {
                id: 1,
                label: "b".into(),
                share: 0.5,
                train: 2,
                recall: 0.5,
            },
        ];
        let width = columns("ad").len();
        Unit {
            unit: "ad".into(),
            rows: 4,
            classes,
            accuracy: 0.6,
            balanced: 0.5,
            chance: 0.5,
            majority: 0.5,
            lift: 0.0,
            control: 0.5,
            usable: true,
            confusion: vec![vec![1, 1], vec![1, 1]],
            weights: vec![vec![0.0; width], vec![0.5; width]],
            bias: vec![0.0, 0.0],
            mean: vec![0.0; width],
            sd: vec![1.0; width],
            split: None,
            note: String::new(),
        }
    }

    #[test]
    fn a_row_is_the_surface_columns_then_the_vocabulary() {
        let words: Vec<String> = ["act", "now", "act", "now", "today"]
            .iter()
            .map(|w| w.to_string())
            .collect();
        let row = features(&view(5, 0.4), &words);
        assert_eq!(row.len(), SURFACE.len() + HASH_DIM);
        assert!((row[0] - 5.0_f64.ln_1p()).abs() < 1e-9);
        let norm = row[SURFACE.len()..]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_brand_row_averages_its_ads_and_counts_the_set() {
        let rows = vec![
            features(&view(4, 0.0), &["a".into()]),
            features(&view(8, 0.0), &["a".into()]),
        ];
        let row = brand(&rows, 2);
        assert_eq!(row.len(), SURFACE.len() + HASH_DIM + 1);
        assert!((row[0] - (4.0_f64.ln_1p() + 8.0_f64.ln_1p()) / 2.0).abs() < 1e-9);
        assert!((row[SURFACE.len() + HASH_DIM] - 2.0_f64.ln_1p()).abs() < 1e-9);
    }

    #[test]
    fn an_empty_brand_row_is_all_zero_rather_than_a_nan() {
        let row = brand(&[], 0);
        assert!(row.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn probabilities_sum_to_one_hundred_and_rank_the_highest_first() {
        let row = vec![1.0; columns("ad").len()];
        let mut map = TextMap {
            name: "t".into(),
            version: 1,
            built_at: String::new(),
            source: Source::default(),
            config: TextConfig::default(),
            columns: BTreeMap::new(),
            units: BTreeMap::new(),
            note: String::new(),
        };
        map.units.insert("ad".into(), unit());
        let ranking = map.predict("ad", &row).expect("usable unit");
        assert_eq!(ranking.top, 1);
        let total: f64 = ranking.ranked.iter().map(|s| s.probability).sum();
        assert!((total - 100.0).abs() < 1e-9);
    }

    #[test]
    fn a_withheld_unit_has_no_prediction() {
        let row = vec![0.0; columns("ad").len()];
        let mut fitted = unit();
        fitted.usable = false;
        let mut map = TextMap {
            name: "t".into(),
            version: 1,
            built_at: String::new(),
            source: Source::default(),
            config: TextConfig::default(),
            columns: BTreeMap::new(),
            units: BTreeMap::from([("ad".to_string(), fitted)]),
            note: String::new(),
        };
        assert!(map.predict("ad", &row).is_none());
        map.units.get_mut("ad").expect("unit").usable = true;
        assert!(map.predict("ad", &[0.0; 3]).is_none());
    }

    #[test]
    fn column_names_line_up_with_the_weight_width() {
        assert_eq!(columns("ad").len(), SURFACE.len() + HASH_DIM);
        assert_eq!(columns("brand").len(), SURFACE.len() + HASH_DIM + 1);
        assert_eq!(columns("ad")[SURFACE.len()], "vocab_00");
    }
}
