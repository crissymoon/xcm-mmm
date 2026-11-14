//! The artifact that ships: seven axes, the fits behind them, and a reader.
//!
//! The artifact is plain JSON because the point of it is to be read and
//! disputed. A lexicon axis carries its terms, the z each term earned, the cut
//! that turns a score into a label and the held out numbers that say whether
//! the cut transfers. An index axis carries its cue table, the component
//! quantiles that turn a density into a capped share, its cuts and its audit.
//!
//! [`Detector`] is the compiled form: the same artifact with every cue pattern
//! built once. It is what scores text, and it borrows the artifact rather than
//! rewriting it, so what is scored is exactly what is on disk.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::eval::Metrics;
use super::feature::{FeatureSpec, Features};
use super::index::{self, Index, IndexSpec};
use super::lexicon::{round4, Lexicon};
use super::logistic::Logistic;
use super::{token, TrainConfig};
use crate::text::tokenize;

/// One axis of the artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum Axis {
    /// A lexicon learned from a labelled bank, scored by register.
    Lexicon(LexiconAxis),
    /// A declared cue index, fitted only in its component reference.
    Index(IndexAxis),
}

/// A learned axis: a lexicon the features read, and the logistic over them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LexiconAxis {
    pub tag: String,
    /// `sentence` or `document`. A sentence axis is read sentence by sentence
    /// and aggregated, because a paragraph can carry one clause and one joke.
    pub grain: String,
    /// The readings the fit sees, and the patterns behind the surface ones.
    pub features: FeatureSpec,
    /// The learned density. It is a feature, and it is also the evidence a
    /// reader can point at, so it travels whole.
    pub lexicon: Lexicon,
    pub model: Logistic,
    /// Firing cut on the probability, then the clear and strong bands above it.
    pub cuts: Vec<f64>,
    pub metrics: Metrics,
}

/// A declared axis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexAxis {
    pub tag: String,
    pub question: String,
    pub components: BTreeMap<String, f64>,
    pub cues: BTreeMap<String, String>,
    pub gate: Vec<String>,
    pub min_groups: usize,
    pub reference: BTreeMap<String, Vec<f64>>,
    pub scale: BTreeMap<String, f64>,
    pub cuts: Vec<f64>,
    pub cuts_note: String,
    pub audit: Audit,
    pub checks: Vec<CheckRun>,
}

/// What the index audit can measure without labels.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Audit {
    pub rows: usize,
    pub fired: usize,
    pub prevalence: f64,
    pub halves_live: usize,
    pub halves_both_fired: usize,
    pub halves_agree: f64,
    pub halves_correlation: Option<f64>,
    pub groups: BTreeMap<String, usize>,
}

/// One declared example and whether the rule behaved on it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckRun {
    pub text: String,
    pub expected: bool,
    pub score: f64,
    pub passed: bool,
}

/// The whole on-disk artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    pub version: u32,
    pub built_at: String,
    pub source: Source,
    pub config: TrainConfig,
    pub axes: BTreeMap<String, Axis>,
}

/// Digests of the three files the fit read.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Source {
    pub bank: String,
    pub bank_rows: usize,
    pub bank_fnv64: String,
    pub ads: String,
    pub ads_rows: usize,
    pub ads_fnv64: String,
    pub cues: String,
    pub cues_axes: usize,
    pub cues_fnv64: String,
}

/// One axis, reduced to the numbers a reader scans first.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AxisStamp {
    pub method: String,
    pub tag: String,
    pub rows: usize,
    pub balanced: f64,
    pub prevalence: f64,
}

/// The folder's own account of itself.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: u32,
    pub built_at: String,
    pub files: Vec<FileStamp>,
    pub config: TrainConfig,
    pub axes: BTreeMap<String, AxisStamp>,
}

/// One shipped file and the digest that proves it is the one described.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FileStamp {
    pub name: String,
    pub rows: usize,
    pub bytes: u64,
    pub fnv64: String,
}

/// One axis reading for one text.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reading {
    pub axis: String,
    pub method: String,
    pub score: f64,
    pub level: String,
    pub label: bool,
    pub hits: Vec<String>,
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
}

/// An artifact with its cue patterns and feature patterns compiled.
pub struct Detector {
    artifact: Artifact,
    indexes: BTreeMap<String, Index>,
    features: BTreeMap<String, Features>,
}

impl Detector {
    /// Load the artifact and compile every cue table in it.
    pub fn load(path: &str) -> io::Result<Self> {
        Self::from_artifact(Artifact::load(path)?)
    }

    pub fn from_artifact(artifact: Artifact) -> io::Result<Self> {
        let mut indexes = BTreeMap::new();
        let mut features = BTreeMap::new();
        for (name, axis) in &artifact.axes {
            if let Axis::Lexicon(lex) = axis {
                let compiled = Features::compile(lex.features.clone()).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("{name}: {e}"))
                })?;
                features.insert(name.clone(), compiled);
            }
            if let Axis::Index(index) = axis {
                let spec = IndexSpec {
                    tag: index.tag.clone(),
                    question: index.question.clone(),
                    components: index.components.clone(),
                    cues: index.cues.clone(),
                    gate: index.gate.clone(),
                    min_groups: index.min_groups,
                    halves: Vec::new(),
                    checks: Vec::new(),
                    lexicon_in: None,
                };
                let compiled = Index::compile(spec).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("{name}: {e}"))
                })?;
                indexes.insert(name.clone(), compiled);
            }
        }
        Ok(Self {
            artifact,
            indexes,
            features,
        })
    }

    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    /// Every axis, scored on one text.
    pub fn score(&self, text: &str) -> Vec<Reading> {
        let mut out = Vec::new();
        for (name, axis) in &self.artifact.axes {
            match axis {
                Axis::Lexicon(lex) => out.push(self.score_lexicon(name, lex, text)),
                Axis::Index(ind) => out.push(self.score_index(name, ind, text)),
            }
        }
        out
    }

    fn score_lexicon(&self, name: &str, axis: &LexiconAxis, text: &str) -> Reading {
        let words = token::tokens(text);
        let probability = |body: &str, words: &[String]| -> f64 {
            let values = self
                .features
                .get(name)
                .map(|features| features.values(body, words, Some(&axis.lexicon)))
                .unwrap_or_else(|| vec![axis.lexicon.register(words)]);
            axis.model.predict(&values)
        };
        let score = if axis.grain == "sentence" {
            let mut scores: Vec<f64> = tokenize::sentences(text, 8)
                .iter()
                .filter(|s| token::tokens(s).len() <= 45)
                .map(|s| {
                    let body = token::tokens(s);
                    probability(s, &body)
                })
                .collect();
            scores.sort_by(|a, b| b.total_cmp(a));
            scores.truncate(3);
            if scores.is_empty() {
                probability(text, &words)
            } else {
                round4(scores.iter().sum::<f64>() / scores.len() as f64)
            }
        } else {
            round4(probability(text, &words))
        };
        Reading {
            axis: name.to_string(),
            method: "lexicon".to_string(),
            score,
            level: index::level_lexicon(score, &axis.cuts).to_string(),
            label: score >= axis.cuts.first().copied().unwrap_or(0.5),
            hits: axis
                .lexicon
                .hits(&words, 8)
                .into_iter()
                .map(|h| h.term)
                .collect(),
        }
    }

    fn score_index(&self, name: &str, axis: &IndexAxis, text: &str) -> Reading {
        let score = match self.indexes.get(name) {
            Some(compiled) => {
                let words = token::tokens(text);
                let parts = compiled.parts(text, &words, None);
                compiled.score(&parts, &axis.reference)
            }
            None => 0.0,
        };
        let hits = match self.indexes.get(name) {
            Some(compiled) => compiled.hits(text, 6),
            None => Vec::new(),
        };
        Reading {
            axis: name.to_string(),
            method: "index".to_string(),
            score,
            level: index::level_index(score, &axis.cuts).to_string(),
            label: score > 0.0,
            hits,
        }
    }

    /// Score every row of a JSON Lines file that carries a `text` field, and
    /// write one row per input row with the readings attached.
    pub fn score_rows(&self, input: &str, out: &str) -> io::Result<usize> {
        let text = fs::read_to_string(input)?;
        let mut written = 0;
        let mut lines = String::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let mut value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{input}: {e}")))?;
            let body = value
                .get("text")
                .and_then(|t| t.as_str())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{input}: a row has no text"),
                    )
                })?
                .to_string();
            let readings = self.score(&body);
            let mut axes = serde_json::Map::new();
            for reading in readings {
                let value = serde_json::to_value(&reading).map_err(io::Error::other)?;
                axes.insert(reading.axis.clone(), value);
            }
            if let Some(object) = value.as_object_mut() {
                object.insert("axes".to_string(), serde_json::Value::Object(axes));
            }
            lines.push_str(&serde_json::to_string(&value).map_err(io::Error::other)?);
            lines.push('\n');
            written += 1;
        }
        fs::write(out, lines)?;
        Ok(written)
    }
}

/// True when a JSON value differs from another, ignoring the build time.
///
/// The build time is a fact about when the fit ran, not about the fit, so a
/// rebuild a week later must still compare clean.
pub fn same_fit(left: &Artifact, right: &Artifact) -> bool {
    left.config == right.config
        && left.source.bank_fnv64 == right.source.bank_fnv64
        && left.source.ads_fnv64 == right.source.ads_fnv64
        && left.source.cues_fnv64 == right.source.cues_fnv64
        && serde_json::to_string(&left.axes).ok() == serde_json::to_string(&right.axes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::lexicon::FitReport;

    fn artifact() -> Artifact {
        let lexicon = Lexicon {
            terms: BTreeMap::from([("shall".to_string(), 4.0)]),
            cap: 6.0,
            prior_tokens: 20.0,
            report: FitReport::default(),
        };
        let axis = LexiconAxis {
            tag: String::new(),
            grain: "sentence".into(),
            features: FeatureSpec {
                names: vec!["register".into()],
                patterns: BTreeMap::new(),
            },
            lexicon,
            model: Logistic {
                mean: vec![0.0],
                scale: vec![1.0],
                coefficients: vec![4.0],
                intercept: -2.0,
                clip: 4.0,
            },
            cuts: vec![0.5, 0.7, 0.9],
            metrics: Metrics::default(),
        };
        Artifact {
            name: "test".into(),
            version: 1,
            built_at: "now".into(),
            source: Source::default(),
            config: TrainConfig::default(),
            axes: BTreeMap::from([("legal".to_string(), Axis::Lexicon(axis))]),
        }
    }

    #[test]
    fn a_lexicon_axis_scores_a_sentence_and_names_its_terms() {
        let detector = Detector::from_artifact(artifact()).unwrap();
        let readings = detector.score("The department shall adopt the rule.");
        assert_eq!(readings.len(), 1);
        assert!(readings[0].label, "{:?}", readings[0]);
        assert_eq!(readings[0].hits, ["shall"]);
    }

    #[test]
    fn a_plain_sentence_does_not_fire() {
        let detector = Detector::from_artifact(artifact()).unwrap();
        let readings = detector.score("Join us on Saturday.");
        assert!(!readings[0].label);
        assert_eq!(readings[0].level, "none");
    }

    #[test]
    fn the_artifact_round_trips_through_json() {
        let a = artifact();
        let text = serde_json::to_string(&a).unwrap();
        let b: Artifact = serde_json::from_str(&text).unwrap();
        assert!(same_fit(&a, &b));
        assert_eq!(b.axes.len(), 1);
    }

    #[test]
    fn a_different_build_time_is_not_a_different_fit() {
        let a = artifact();
        let mut b = a.clone();
        b.built_at = "later".into();
        assert!(same_fit(&a, &b));
    }
}
