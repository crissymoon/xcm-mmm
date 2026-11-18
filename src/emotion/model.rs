//! An emotion reading over the eight Plutchik axes, for any English word.
//!
//! Three things decide a word's reading, in order. A listed word is read from
//! the seed lexicon or from the adjustment the labelled bank learned. An
//! unlisted inflection is folded onto a listed stem ([`super::lemma`]). A word
//! that is still unknown is read by character analogy to the words that are
//! ([`super::analogy`]). Every path ends somewhere, so a word with no evidence
//! at all is read as neutral rather than skipped: that is what "handles every
//! English word" means here, and [`super::coverage`] counts which tier carried
//! how many of a dictionary's words.
//!
//! Scoring runs over tokens, not characters. A negator flips the next three
//! affect words through a declared map (`not happy` reads as sadness, not as
//! joy with a minus sign). An amplifier or diminisher scales the next word and
//! only the next word. The per-word readings are pooled with a length prior,
//! so one furious adjective in a long letter does not read like a furious
//! letter, and every score lands on the same scale for every text.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::analogy::{Analogy, AnalogyConfig};
use super::lemma;
use crate::detect::token;
use crate::digest::fnv64_str;

/// Which lookup answered for a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Exact,
    Listed,
    Lemma,
    Prefix,
    Suffix,
    Analogy,
    Prior,
}

impl Tier {
    pub const ALL: [Tier; 7] = [
        Tier::Exact,
        Tier::Listed,
        Tier::Lemma,
        Tier::Prefix,
        Tier::Suffix,
        Tier::Analogy,
        Tier::Prior,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Exact => "exact",
            Tier::Listed => "listed",
            Tier::Lemma => "lemma",
            Tier::Prefix => "prefix",
            Tier::Suffix => "suffix",
            Tier::Analogy => "analogy",
            Tier::Prior => "prior",
        }
    }
}

/// A derivational ending that keeps or flips the base word's reading.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SuffixRule {
    pub ending: String,
    pub flip: bool,
}

/// The words that change how their neighbour reads.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Modifiers {
    pub negators: Vec<String>,
    pub amplifiers: Vec<String>,
    pub diminishers: Vec<String>,
}

/// What the labelled bank said about the model it adjusted.
///
/// `serde(default)` so an artifact written before the resolution reading still
/// loads: the newer fields come back as zero and the card says they are missing
/// rather than the file failing to parse.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BankMetrics {
    pub rows: usize,
    pub labels: usize,
    pub fit_accuracy: f64,
    pub leave_one_out: f64,
    pub majority: f64,
    pub per_label: BTreeMap<String, f64>,
    /// Smallest leave-one-out accuracy gain this bank resolves at 80 percent
    /// power, in accuracy points.
    pub detectable: f64,
    /// Whether that clears the three point floor fixed before the fit.
    pub resolves_floor: bool,
    /// Rows that would resolve the floor at this accuracy.
    pub rows_for_floor: usize,
    /// Plain reading of the two numbers, including why no label audit runs here.
    pub resolution_note: String,
}

/// The files the fit read, with the digest that proves it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub seed: String,
    pub seed_words: usize,
    pub seed_fnv64: String,
    pub bank: String,
    pub bank_rows: usize,
    pub bank_fnv64: String,
    pub listed: String,
    pub listed_rows: usize,
    pub listed_fnv64: String,
}

/// One word's contribution to a reading.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hit {
    pub word: String,
    pub stem: String,
    pub tier: String,
    pub weight: f64,
    pub valence: f64,
}

/// What one text reads as.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reading {
    pub valence: f64,
    pub arousal: f64,
    pub intensity: f64,
    pub dominant: String,
    pub emotions: BTreeMap<String, f64>,
    pub hits: Vec<Hit>,
}

impl Reading {
    pub fn share(&self, emotion: &str) -> f64 {
        self.emotions.get(emotion).copied().unwrap_or(0.0)
    }

    pub fn is_neutral(&self) -> bool {
        self.dominant == "neutral"
    }

    /// A short line for a report: dominant, valence, arousal.
    pub fn line(&self) -> String {
        format!(
            "{:<12} valence {:+.2}  arousal {:.2}  intensity {:.2}",
            self.dominant, self.valence, self.arousal, self.intensity
        )
    }
}

/// The artifact that ships.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionModel {
    pub name: String,
    pub version: u32,
    pub built_at: String,
    pub emotions: Vec<String>,
    pub valence_weights: BTreeMap<String, f64>,
    pub arousal_weights: BTreeMap<String, f64>,
    pub flip: BTreeMap<String, String>,
    pub seed: BTreeMap<String, Vec<f64>>,
    pub learned: BTreeMap<String, Vec<f64>>,
    /// A valence-only list for words the seed does not name.
    pub listed: BTreeMap<String, f64>,
    pub listed_positive: BTreeMap<String, f64>,
    pub listed_negative: BTreeMap<String, f64>,
    /// Words the neutral rows carry: each dampens the reading of that word.
    pub neutral: BTreeMap<String, f64>,
    pub blend: f64,
    pub prefixes: Vec<String>,
    pub prefix_keep: BTreeSet<String>,
    pub suffixes: Vec<SuffixRule>,
    pub modifiers: Modifiers,
    /// Polarity below which a text is labelled neutral.
    pub neutral_band: f64,
    /// Smallest total affect mass that may be labelled at all.
    pub min_mass: f64,
    pub prior: f64,
    pub analogy: AnalogyConfig,
    pub source: Source,
    pub metrics: BankMetrics,
    pub learned_top: BTreeMap<String, Vec<String>>,
}

impl EmotionModel {
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

    /// Seed and learned weights merged, learned scaled by the blend and the
    /// whole entry damped by any neutrality the bank learned for the word.
    pub fn entries(&self) -> BTreeMap<String, Vec<f64>> {
        let mut out = self.seed.clone();
        for (word, weights) in &self.learned {
            let entry = out
                .entry(word.clone())
                .or_insert_with(|| vec![0.0; self.emotions.len()]);
            for (slot, value) in entry.iter_mut().zip(weights) {
                *slot += value * self.blend;
            }
        }
        for (word, damp) in &self.neutral {
            if let Some(entry) = out.get_mut(word) {
                for slot in entry {
                    *slot *= (1.0 - damp).clamp(0.0, 1.0);
                }
            }
        }
        out
    }

    pub fn scanner(&self) -> Scanner {
        let entries = self.entries();
        let analogy = Analogy::build(&entries, self.analogy.clone());
        Scanner {
            model: self.clone(),
            entries,
            analogy,
            negators: self.modifiers.negators.iter().cloned().collect(),
            amplifiers: self.modifiers.amplifiers.iter().cloned().collect(),
            diminishers: self.modifiers.diminishers.iter().cloned().collect(),
            valence: self
                .emotions
                .iter()
                .map(|e| self.valence_weights.get(e).copied().unwrap_or(0.0))
                .collect(),
            arousal: self
                .emotions
                .iter()
                .map(|e| self.arousal_weights.get(e).copied().unwrap_or(0.0))
                .collect(),
        }
    }

    pub fn emotion_index(&self, name: &str) -> Option<usize> {
        self.emotions.iter().position(|e| e == name)
    }
}

/// One word's answer: its weights, the tier, and the term the tier used.
#[derive(Clone, Debug)]
pub struct Resolved {
    pub weights: Vec<f64>,
    pub tier: Tier,
    pub term: String,
    pub similarity: f64,
}

/// A model with its lookup tables and modifier sets built once.
pub struct Scanner {
    model: EmotionModel,
    entries: BTreeMap<String, Vec<f64>>,
    analogy: Analogy,
    negators: BTreeSet<String>,
    amplifiers: BTreeSet<String>,
    diminishers: BTreeSet<String>,
    valence: Vec<f64>,
    arousal: Vec<f64>,
}

impl Scanner {
    pub fn model(&self) -> &EmotionModel {
        &self.model
    }

    pub fn entries(&self) -> &BTreeMap<String, Vec<f64>> {
        &self.entries
    }

    pub fn analogy(&self) -> &Analogy {
        &self.analogy
    }

    pub fn is_negator(&self, word: &str) -> bool {
        self.negators.contains(word)
    }

    pub fn is_amplifier(&self, word: &str) -> bool {
        self.amplifiers.contains(word)
    }

    pub fn is_diminisher(&self, word: &str) -> bool {
        self.diminishers.contains(word)
    }

    /// The reading of one word, through the first tier that answers.
    pub fn resolve(&self, word: &str) -> Resolved {
        let width = self.model.emotions.len();
        if let Some(weights) = self.entries.get(word) {
            return Resolved {
                weights: weights.clone(),
                tier: Tier::Exact,
                term: word.to_string(),
                similarity: 1.0,
            };
        }
        if let Some(valence) = self.model.listed.get(word) {
            let distribution = if *valence >= 0.0 {
                &self.model.listed_positive
            } else {
                &self.model.listed_negative
            };
            let mut weights = vec![0.0; width];
            for (emotion, share) in distribution {
                if let Some(slot) = self.model.emotion_index(emotion) {
                    weights[slot] += share * valence.abs();
                }
            }
            if weights.iter().any(|weight| *weight > 0.0) {
                return Resolved {
                    weights,
                    tier: Tier::Listed,
                    term: word.to_string(),
                    similarity: valence.abs(),
                };
            }
        }
        if token::is_function(word) || word.chars().count() < 4 {
            return Resolved {
                weights: vec![0.0; width],
                tier: Tier::Prior,
                term: String::new(),
                similarity: 0.0,
            };
        }
        for candidate in lemma::candidates(word).into_iter().skip(1) {
            if let Some(weights) = self.entries.get(&candidate.stem) {
                return Resolved {
                    weights: weights.clone(),
                    tier: Tier::Lemma,
                    term: candidate.stem,
                    similarity: 1.0,
                };
            }
        }
        for prefix in &self.model.prefixes {
            let Some(rest) = word.strip_prefix(prefix.as_str()) else {
                continue;
            };
            if rest.chars().count() < 3 || self.model.prefix_keep.contains(word) {
                continue;
            }
            let Some(base) = self.entries.get(rest) else {
                continue;
            };
            let mut weights = vec![0.0; width];
            for (at, emotion) in self.model.emotions.iter().enumerate() {
                let target = self
                    .model
                    .flip
                    .get(emotion)
                    .map(String::as_str)
                    .unwrap_or("fear");
                if let Some(slot) = self.model.emotion_index(target) {
                    weights[slot] += base[at];
                }
            }
            return Resolved {
                weights,
                tier: Tier::Prefix,
                term: rest.to_string(),
                similarity: 1.0,
            };
        }
        for rule in &self.model.suffixes {
            let Some(base) = word.strip_suffix(rule.ending.as_str()) else {
                continue;
            };
            if base.chars().count() < 3 {
                continue;
            }
            let Some(weights) = self.entries.get(base) else {
                continue;
            };
            let mut out = vec![0.0; width];
            if rule.flip {
                for (at, emotion) in self.model.emotions.iter().enumerate() {
                    let target = self
                        .model
                        .flip
                        .get(emotion)
                        .map(String::as_str)
                        .unwrap_or("fear");
                    if let Some(slot) = self.model.emotion_index(target) {
                        out[slot] += weights[at];
                    }
                }
            } else {
                out.clone_from(weights);
            }
            return Resolved {
                weights: out,
                tier: Tier::Suffix,
                term: base.to_string(),
                similarity: 1.0,
            };
        }
        if let Some(read) = self.analogy.read(word) {
            return Resolved {
                weights: read.weights,
                tier: Tier::Analogy,
                term: read
                    .neighbours
                    .first()
                    .map(|n| n.word.clone())
                    .unwrap_or_default(),
                similarity: read.confidence,
            };
        }
        Resolved {
            weights: vec![0.0; width],
            tier: Tier::Prior,
            term: String::new(),
            similarity: 0.0,
        }
    }

    /// Read a text.
    pub fn read(&self, text: &str) -> Reading {
        let width = self.model.emotions.len();
        let words = token::tokens(text);
        let mut mass = vec![0.0; width];
        let mut total = 0.0;
        let mut valence_sum = 0.0;
        let mut arousal_sum = 0.0;
        let mut hits: Vec<Hit> = Vec::new();
        let mut negate = 0usize;
        let mut factor = 1.0;
        for word in &words {
            if self.is_negator(word) {
                negate = 3;
                factor = 1.0;
                continue;
            }
            if self.is_amplifier(word) {
                factor = 1.6;
                continue;
            }
            if self.is_diminisher(word) {
                factor = 0.6;
                continue;
            }
            let resolved = self.resolve(word);
            if resolved.tier == Tier::Prior {
                continue;
            }
            let mut weights = resolved.weights;
            for weight in &mut weights {
                *weight *= factor;
            }
            factor = 1.0;
            if negate > 0 {
                let mut flipped = vec![0.0; width];
                for (at, emotion) in self.model.emotions.iter().enumerate() {
                    let target = self
                        .model
                        .flip
                        .get(emotion)
                        .map(String::as_str)
                        .unwrap_or("fear");
                    if let Some(slot) = self.model.emotion_index(target) {
                        flipped[slot] += weights[at];
                    }
                }
                weights = flipped;
                negate -= 1;
            }
            let word_mass: f64 = weights.iter().sum();
            if word_mass <= 0.0 {
                continue;
            }
            let word_valence: f64 = weights
                .iter()
                .zip(&self.valence)
                .map(|(w, v)| w * v)
                .sum::<f64>()
                / word_mass;
            let word_arousal: f64 = weights
                .iter()
                .zip(&self.arousal)
                .map(|(w, a)| w * a)
                .sum::<f64>()
                / word_mass;
            for (slot, weight) in mass.iter_mut().zip(&weights) {
                *slot += weight;
            }
            total += word_mass;
            valence_sum += word_valence * word_mass;
            arousal_sum += word_arousal * word_mass;
            hits.push(Hit {
                word: word.clone(),
                stem: resolved.term,
                tier: resolved.tier.as_str().to_string(),
                weight: round4(word_mass),
                valence: round4(word_valence),
            });
        }
        let prior = self.model.prior.max(0.0);
        let scale = total + prior;
        let emotions: BTreeMap<String, f64> = self
            .model
            .emotions
            .iter()
            .enumerate()
            .map(|(at, emotion)| (emotion.clone(), round4(mass[at] / scale)))
            .collect();
        let intensity = emotions.values().copied().fold(0.0, f64::max);
        let valence = round4((valence_sum / scale).clamp(-1.0, 1.0));
        let dominant = if total >= self.model.min_mass && valence.abs() >= self.model.neutral_band {
            emotions
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(emotion, _)| emotion.clone())
                .unwrap_or_else(|| "neutral".to_string())
        } else {
            "neutral".to_string()
        };
        hits.sort_by(|a, b| {
            b.weight
                .total_cmp(&a.weight)
                .then_with(|| a.word.cmp(&b.word))
        });
        hits.truncate(8);
        Reading {
            valence,
            arousal: round4((arousal_sum / scale).clamp(0.0, 1.0)),
            intensity: round4(intensity),
            dominant,
            emotions,
            hits,
        }
    }
}

/// Four decimal places, so a rebuild writes the same bytes.
pub use crate::num::round4;

/// True when two artifacts are the same fit, ignoring when the fit ran.
pub fn same_fit(left: &EmotionModel, right: &EmotionModel) -> bool {
    left.seed == right.seed
        && left.learned == right.learned
        && left.neutral == right.neutral
        && left.blend == right.blend
        && left.source.seed_fnv64 == right.source.seed_fnv64
        && left.source.bank_fnv64 == right.source.bank_fnv64
        && left.source.listed_fnv64 == right.source.listed_fnv64
        && left.metrics == right.metrics
}

/// Digest of a file's bytes, for the manifest and the source block.
pub fn file_digest(path: &str) -> io::Result<String> {
    fs::read(path).map(|bytes| crate::digest::fnv64(&bytes))
}

/// Digest of a string, for a source file that was read rather than named.
pub fn text_digest(text: &str) -> String {
    fnv64_str(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> EmotionModel {
        let emotions = vec![
            "joy".to_string(),
            "sadness".to_string(),
            "anger".to_string(),
            "fear".to_string(),
        ];
        let mut valence_weights = BTreeMap::new();
        valence_weights.insert("joy".into(), 1.0);
        valence_weights.insert("sadness".into(), -1.0);
        valence_weights.insert("anger".into(), -0.9);
        valence_weights.insert("fear".into(), -0.8);
        let mut arousal_weights = BTreeMap::new();
        for (emotion, value) in [
            ("joy", 0.6),
            ("sadness", 0.2),
            ("anger", 1.0),
            ("fear", 1.0),
        ] {
            arousal_weights.insert(emotion.to_string(), value);
        }
        let mut seed = BTreeMap::new();
        seed.insert("happy".to_string(), vec![1.0, 0.0, 0.0, 0.0]);
        seed.insert("sad".to_string(), vec![0.0, 1.0, 0.0, 0.0]);
        seed.insert("angry".to_string(), vec![0.0, 0.0, 1.0, 0.0]);
        seed.insert("run".to_string(), vec![0.0, 0.0, 0.0, 1.0]);
        let mut flip = BTreeMap::new();
        flip.insert("joy".to_string(), "sadness".to_string());
        flip.insert("sadness".to_string(), "joy".to_string());
        flip.insert("anger".to_string(), "fear".to_string());
        flip.insert("fear".to_string(), "anger".to_string());
        EmotionModel {
            name: "test".into(),
            version: 1,
            built_at: "now".into(),
            emotions,
            valence_weights,
            arousal_weights,
            flip,
            seed,
            learned: BTreeMap::new(),
            listed: BTreeMap::new(),
            listed_positive: BTreeMap::new(),
            listed_negative: BTreeMap::new(),
            neutral: BTreeMap::new(),
            blend: 0.5,
            prefixes: vec!["un".into()],
            prefix_keep: BTreeSet::new(),
            suffixes: vec![SuffixRule {
                ending: "less".into(),
                flip: true,
            }],
            modifiers: Modifiers {
                negators: vec!["not".into(), "never".into()],
                amplifiers: vec!["very".into()],
                diminishers: vec!["slightly".into()],
            },
            neutral_band: 0.12,
            min_mass: 0.4,
            prior: 1.0,
            analogy: AnalogyConfig::default(),
            source: Source::default(),
            metrics: BankMetrics::default(),
            learned_top: BTreeMap::new(),
        }
    }

    #[test]
    fn a_listed_word_reads_on_its_axis() {
        let scanner = model().scanner();
        let reading = scanner.read("I am happy");
        assert_eq!(reading.dominant, "joy");
        assert!(reading.valence > 0.0);
        assert_eq!(reading.hits[0].tier, "exact");
    }

    #[test]
    fn an_inflection_reads_through_its_stem() {
        let scanner = model().scanner();
        let reading = scanner.read("we are running");
        assert_eq!(reading.dominant, "fear");
        assert_eq!(reading.hits[0].tier, "lemma");
        assert_eq!(reading.hits[0].stem, "run");
    }

    #[test]
    fn a_negator_flips_the_word_that_follows() {
        let scanner = model().scanner();
        let reading = scanner.read("not happy");
        assert!(reading.valence < 0.0);
        assert_eq!(reading.dominant, "sadness");
    }

    #[test]
    fn an_amplifier_lifts_and_a_diminisher_lowers() {
        let scanner = model().scanner();
        let plain = scanner.read("angry");
        let lifted = scanner.read("very angry");
        let lowered = scanner.read("slightly angry");
        assert!(lifted.arousal >= plain.arousal);
        assert!(lifted.intensity > lowered.intensity);
    }

    #[test]
    fn an_unknown_word_still_gets_a_reading_or_a_prior() {
        let scanner = model().scanner();
        let known = scanner.read("happily");
        assert_eq!(known.dominant, "joy");
        let unknown = scanner.read("xylophone");
        assert_eq!(unknown.dominant, "neutral");
        assert!(unknown.valence.abs() < 0.01);
    }

    #[test]
    fn a_negated_prefix_word_reads_against_its_base() {
        let scanner = model().scanner();
        let base = scanner.read("happy");
        let flipped = scanner.read("unhappy");
        assert!(flipped.valence < 0.0);
        assert!(base.valence > 0.0);
    }

    #[test]
    fn an_empty_text_reads_neutral() {
        let scanner = model().scanner();
        let reading = scanner.read("");
        assert_eq!(reading.dominant, "neutral");
        assert_eq!(reading.intensity, 0.0);
        assert!(reading.hits.is_empty());
    }
}
