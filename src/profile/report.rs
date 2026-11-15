//! One report over one text: shape, semantics, emotion, detectors, segment.
//!
//! Every number here is either counted from the text or read from a shipped
//! model. The only arithmetic this module does is the kind a reader can check
//! by hand: a mean, a share, and the two Flesch formulas. Nothing is a weighted
//! blend of two scores, because a blend would need a weight nobody fitted.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::detect::feature::syllables;
use crate::detect::{token, Axis as DetectorAxis};
use crate::text::tokenize;

use super::context::Loaded;
use super::language::{self, LanguageView};
use super::segment::{place, SegmentView, Signal};

/// How long the text is, and how it divides.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Shape {
    pub chars: usize,
    pub words: usize,
    pub sentences: usize,
    pub words_per_sentence: f64,
}

/// The measurable part of meaning: syllables, register and the surface
/// readings the two learned detectors carry, plus the two Flesch formulas.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Semantics {
    pub syllables_per_word: f64,
    /// Share of words seven characters or longer.
    pub long_word_share: f64,
    /// Share of words that are you, your, yours, yourself or yourselves.
    pub second_person_share: f64,
    pub questions: usize,
    pub exclamations: usize,
    /// Flesch reading ease, on the formula's own 0 to 120 scale.
    pub flesch_ease: f64,
    /// Flesch-Kincaid grade level, in US school years.
    pub grade_level: f64,
    /// Named surface readings, straight from the learned axes.
    pub surface: BTreeMap<String, f64>,
}

/// What the text feels like, on the eight Plutchik axes.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Emotion {
    pub dominant: String,
    pub valence: f64,
    pub arousal: f64,
    pub intensity: f64,
    pub shares: BTreeMap<String, f64>,
    pub negators: usize,
    pub amplifiers: usize,
    pub diminishers: usize,
}

/// One word that carried emotional mass, and the axis it leans on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trigger {
    pub word: String,
    pub axis: String,
    pub tier: String,
    pub weight: f64,
    pub valence: f64,
}

/// One of the seven detectors, with the evidence behind its score.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Axis {
    pub name: String,
    pub method: String,
    /// The question a cue index was declared to answer; empty for an axis
    /// learned from the bank.
    pub question: String,
    pub score: f64,
    pub level: String,
    pub label: bool,
    pub hits: Vec<String>,
    /// Components that must fire before the score can leave zero.
    pub gate: Vec<String>,
    pub gate_open: bool,
    /// Cue hits per hundred words, before the weights. Empty for a learned
    /// axis, which has no cue groups.
    pub parts: BTreeMap<String, f64>,
}

/// The models a report was read from, so a stored report says what scored it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Models {
    pub detectors: String,
    pub emotion: String,
    pub segments: Option<String>,
}

/// Everything this crate can say about one text.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub text: String,
    pub models: Models,
    pub shape: Shape,
    pub semantics: Semantics,
    pub emotion: Emotion,
    pub language: LanguageView,
    pub triggers: Vec<Trigger>,
    pub detectors: Vec<Axis>,
    pub segment: Option<SegmentView>,
    /// Anything the models could not see, stated rather than filled in.
    pub notes: Vec<String>,
}

/// What the caller wants shown, beside the text.
#[derive(Clone, Debug)]
pub struct Options {
    /// Trigger words to report. Zero reports every word the reading kept.
    pub triggers: usize,
    /// Targeting terms per ad, when the text is one ad in a known buy.
    pub terms: Option<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            triggers: 6,
            terms: None,
        }
    }
}

/// Read one text with the models already open.
pub fn build(loaded: &Loaded, text: &str, options: &Options) -> Report {
    let words = token::tokens(text);
    let count = words.len();
    let sentences = tokenize::sentences(text, 1).len().max(1);
    let shape = Shape {
        chars: text.chars().count(),
        words: count,
        sentences,
        words_per_sentence: ratio(count as f64, sentences as f64),
    };

    let syllables_total: usize = words.iter().map(|word| syllables(word).max(1)).sum();
    let semantics = Semantics {
        syllables_per_word: ratio(syllables_total as f64, count as f64),
        long_word_share: ratio(
            words
                .iter()
                .filter(|word| word.chars().count() >= 7)
                .count() as f64,
            count as f64,
        ),
        second_person_share: ratio(
            words
                .iter()
                .filter(|word| {
                    matches!(
                        word.as_str(),
                        "you" | "your" | "yours" | "yourself" | "yourselves"
                    )
                })
                .count() as f64,
            count as f64,
        ),
        questions: text.matches('?').count(),
        exclamations: text.matches('!').count(),
        flesch_ease: round(
            206.835
                - 1.015 * shape.words_per_sentence
                - 84.6 * ratio(syllables_total as f64, count as f64),
        ),
        grade_level: round(
            0.39 * shape.words_per_sentence + 11.8 * ratio(syllables_total as f64, count as f64)
                - 15.59,
        ),
        surface: surface_readings(loaded, text),
    };

    let reading = loaded.emotion.read(text);
    let emotion = Emotion {
        dominant: reading.dominant.clone(),
        valence: reading.valence,
        arousal: reading.arousal,
        intensity: reading.intensity,
        shares: reading.emotions.clone(),
        negators: words
            .iter()
            .filter(|word| loaded.emotion.is_negator(word))
            .count(),
        amplifiers: words
            .iter()
            .filter(|word| loaded.emotion.is_amplifier(word))
            .count(),
        diminishers: words
            .iter()
            .filter(|word| loaded.emotion.is_diminisher(word))
            .count(),
    };

    let lang = language::read(text, &words, sentences, reading.hits.len());

    let take = if options.triggers == 0 {
        reading.hits.len()
    } else {
        options.triggers.min(reading.hits.len())
    };
    let triggers = reading
        .hits
        .iter()
        .take(take)
        .map(|hit| Trigger {
            word: hit.word.clone(),
            axis: axis_of(loaded, &hit.word),
            tier: hit.tier.clone(),
            weight: hit.weight,
            valence: hit.valence,
        })
        .collect();

    let detectors = loaded
        .detectors
        .score(text)
        .into_iter()
        .map(|reading| {
            let (question, gate) = match loaded.detectors.artifact().axes.get(&reading.axis) {
                Some(DetectorAxis::Index(index)) => (index.question.clone(), index.gate.clone()),
                _ => (String::new(), Vec::new()),
            };
            let parts = loaded
                .detectors
                .parts(&reading.axis, text)
                .unwrap_or_default();
            let gate_open = gate.is_empty()
                || gate
                    .iter()
                    .any(|name| parts.get(name).copied().unwrap_or(0.0) > 0.0);
            Axis {
                name: reading.axis,
                method: reading.method,
                question,
                score: reading.score,
                level: reading.level,
                label: reading.label,
                hits: reading.hits,
                gate,
                gate_open,
                parts,
            }
        })
        .collect();

    let segment = loaded.segments.as_ref().map(|artifact| {
        place(
            artifact,
            &Signal {
                words: count,
                terms: options.terms,
                reuse: lang.reuse,
                phrase: lang.top_phrase.clone(),
            },
        )
    });

    let mut notes = Vec::new();
    if count < 5 {
        notes.push("under five words: the emotion model has little mass to read and the sentence detectors read a fragment".to_string());
    }
    if loaded.segments.is_none() {
        notes.push(format!(
            "no segment model at {}: run segment fit to add that view",
            loaded.segments_path
        ));
    }
    if semantics.surface.is_empty() {
        notes.push("the learned axes returned no surface readings".to_string());
    }

    Report {
        text: text.to_string(),
        models: Models {
            detectors: loaded.detectors_path.clone(),
            emotion: loaded.emotion_path.clone(),
            segments: if loaded.segments.is_some() {
                Some(loaded.segments_path.clone())
            } else {
                None
            },
        },
        shape,
        semantics,
        emotion,
        language: lang,
        triggers,
        detectors,
        segment,
        notes,
    }
}

/// Merge every learned axis's own surface readings into one named map.
///
/// The names come from the artifact, so a refit that renames a reading shows
/// up in the report rather than being silently replaced by a zero.
fn surface_readings(loaded: &Loaded, text: &str) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for (name, axis) in &loaded.detectors.artifact().axes {
        if matches!(axis, DetectorAxis::Lexicon(_)) {
            if let Some(readings) = loaded.detectors.features(name, text) {
                out.extend(readings);
            }
        }
    }
    out
}

/// The axis a word's own weights point at, for the trigger table.
///
/// This is the seed's lean before modifiers and negation, which is what a
/// reader wants from a word list: the reading that put the word here.
fn axis_of(loaded: &Loaded, word: &str) -> String {
    let resolved = loaded.emotion.resolve(word);
    let emotions = &loaded.emotion.model().emotions;
    let mut best: Option<(usize, f64)> = None;
    for (at, weight) in resolved.weights.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        if best.map(|(_, top)| *weight > top).unwrap_or(true) {
            best = Some((at, *weight));
        }
    }
    best.and_then(|(at, _)| emotions.get(at).cloned())
        .unwrap_or_else(|| "neutral".to_string())
}

fn ratio(part: f64, whole: f64) -> f64 {
    if whole.abs() < 1e-9 {
        0.0
    } else {
        part / whole
    }
}

fn round(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_text_divides_by_nothing() {
        let loaded = Loaded::open(
            crate::profile::DETECTORS,
            crate::profile::EMOTION,
            crate::profile::SEGMENTS,
        )
        .unwrap();
        let report = build(&loaded, "", &Options::default());
        assert_eq!(report.shape.words, 0);
        assert_eq!(report.shape.words_per_sentence, 0.0);
        assert_eq!(report.semantics.syllables_per_word, 0.0);
        assert!(report.triggers.is_empty());
    }

    #[test]
    fn plain_words_read_easy_and_long_ones_do_not() {
        let loaded = Loaded::open(
            crate::profile::DETECTORS,
            crate::profile::EMOTION,
            crate::profile::SEGMENTS,
        )
        .unwrap();
        let easy = build(&loaded, "The cat sat on the mat.", &Options::default());
        assert!(easy.semantics.flesch_ease > 100.0, "{:?}", easy.semantics);
        assert_eq!(easy.shape.sentences, 1);
        let hard = build(
            &loaded,
            "Notwithstanding the aforementioned stipulations, compliance with the \
             documentation obligations is required.",
            &Options::default(),
        );
        assert!(
            hard.semantics.flesch_ease < easy.semantics.flesch_ease,
            "{} vs {}",
            hard.semantics.flesch_ease,
            easy.semantics.flesch_ease
        );
        assert!(hard.semantics.surface.contains_key("authority"));
    }

    #[test]
    fn triggers_are_capped_and_carry_a_tier() {
        let loaded = Loaded::open(
            crate::profile::DETECTORS,
            crate::profile::EMOTION,
            crate::profile::SEGMENTS,
        )
        .unwrap();
        let text = "Dreadful collapse, ruin and terror, then joy.";
        let report = build(
            &loaded,
            text,
            &Options {
                triggers: 2,
                terms: None,
            },
        );
        assert_eq!(report.triggers.len(), 2);
        assert!(report.triggers.iter().all(|t| !t.tier.is_empty()));
        let every = build(
            &loaded,
            text,
            &Options {
                triggers: 0,
                terms: None,
            },
        );
        assert!(every.triggers.len() >= report.triggers.len());
    }

    #[test]
    fn every_detector_reports_a_level_and_a_gate_state() {
        let loaded = Loaded::open(
            crate::profile::DETECTORS,
            crate::profile::EMOTION,
            crate::profile::SEGMENTS,
        )
        .unwrap();
        let report = build(&loaded, "Act now or lose everything.", &Options::default());
        assert_eq!(report.detectors.len(), 7);
        for axis in &report.detectors {
            assert!(!axis.level.is_empty(), "{}", axis.name);
            if axis.method == "index" {
                assert!(!axis.gate.is_empty(), "{}", axis.name);
            }
        }
    }
}
