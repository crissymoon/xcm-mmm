//! The readings a lexicon axis is fitted on.
//!
//! A lexicon density alone is a strong but blunt feature: it says the register
//! is formal without saying whether the formality is a citation, an obligation
//! or a nominalisation. The published detector fits four features per axis, one
//! of them the learned density and the rest surface counts, and this module
//! computes the same set from patterns that travel in `cues.json` rather than
//! living in the code:
//!
//! * `register`: capped lexicon mass per hundred words, with a length prior;
//! * `citation`, `authority`, `nominal`, `agentless`, `directive`, `victim`,
//!   `urgency`: grammars of a claim, counted per hundred words;
//! * `syllables`: mean syllables per word, the standard readability reading of
//!   abstraction;
//! * `intensity`: exclamation and question marks plus shouted words, per
//!   hundred words.
//!
//! An unknown feature name is a refusal. A feature that silently read zero would
//! look exactly like an axis where the reading never occurs, and the difference
//! matters when the fit comes out weak.

use std::collections::BTreeMap;

use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

use super::lexicon::Lexicon;

/// The feature list and patterns for one axis, as they travel in `cues.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FeatureSpec {
    pub names: Vec<String>,
    #[serde(default)]
    pub patterns: BTreeMap<String, String>,
}

/// A spec with its patterns compiled.
#[derive(Clone, Debug)]
pub struct Features {
    spec: FeatureSpec,
    patterns: BTreeMap<String, Regex>,
}

/// The names this module knows how to compute.
pub const NAMES: [&str; 10] = [
    "register",
    "citation",
    "authority",
    "nominal",
    "agentless",
    "directive",
    "victim",
    "urgency",
    "syllables",
    "intensity",
];

impl Features {
    pub fn compile(spec: FeatureSpec) -> Result<Self, String> {
        let mut patterns = BTreeMap::new();
        for name in &spec.names {
            if !NAMES.contains(&name.as_str()) {
                return Err(format!("unknown feature {name}"));
            }
            if let Some(pattern) = spec.patterns.get(name) {
                let compiled = RegexBuilder::new(pattern)
                    .case_insensitive(true)
                    .build()
                    .map_err(|e| format!("feature {name}: {e}"))?;
                patterns.insert(name.clone(), compiled);
            }
        }
        Ok(Self { spec, patterns })
    }

    pub fn names(&self) -> &[String] {
        &self.spec.names
    }

    /// One row of values, in the order the spec names them.
    pub fn values(&self, text: &str, words: &[String], lexicon: Option<&Lexicon>) -> Vec<f64> {
        let count = words.len().max(1) as f64;
        self.spec
            .names
            .iter()
            .map(|name| match name.as_str() {
                "register" => lexicon.map(|lex| lex.register(words)).unwrap_or(0.0),
                "syllables" => words.iter().map(|w| syllables(w) as f64).sum::<f64>() / count,
                "intensity" => {
                    let marks = text.chars().filter(|c| *c == '!' || *c == '?').count();
                    let shouted = text
                        .split_whitespace()
                        .filter(|w| {
                            w.chars().filter(|c| c.is_alphabetic()).count() > 3
                                && w.chars()
                                    .filter(|c| c.is_alphabetic())
                                    .all(|c| c.is_uppercase())
                        })
                        .count();
                    100.0 * (marks + shouted) as f64 / count
                }
                other => {
                    let hits = self
                        .patterns
                        .get(other)
                        .map(|pattern| pattern.find_iter(text).count())
                        .unwrap_or(0);
                    100.0 * hits as f64 / count
                }
            })
            .collect()
    }
}

/// Vowel groups in a word, at least one, the standard syllable estimate.
pub fn syllables(word: &str) -> usize {
    let mut count = 0;
    let mut inside = false;
    for c in word.chars() {
        let vowel = matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y');
        if vowel && !inside {
            count += 1;
        }
        inside = vowel;
    }
    count.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> FeatureSpec {
        FeatureSpec {
            names: vec!["register".into(), "citation".into(), "syllables".into()],
            patterns: BTreeMap::from([(
                "citation".to_string(),
                r"(?i)(\bsection\s*\d|\bu\.?s\.?c\.?)".to_string(),
            )]),
        }
    }

    #[test]
    fn an_unknown_feature_is_refused() {
        let bad = FeatureSpec {
            names: vec!["vibes".into()],
            patterns: BTreeMap::new(),
        };
        assert!(Features::compile(bad).is_err());
    }

    #[test]
    fn a_context_free_reading_counts_per_hundred_words() {
        let features = Features::compile(spec()).unwrap();
        let text = "Under Section 12 the agency shall act for ten words here";
        let words = crate::detect::token::tokens(text);
        let values = features.values(text, &words, None);
        assert!(values[1] > 0.0);
        assert_eq!(values[0], 0.0);
    }

    #[test]
    fn syllables_are_vowel_groups_with_a_floor() {
        assert_eq!(syllables("strength"), 1);
        assert_eq!(syllables("administration"), 5);
        assert_eq!(syllables(""), 1);
    }

    #[test]
    fn intensity_counts_marks_and_shouts() {
        let spec = FeatureSpec {
            names: vec!["intensity".into()],
            patterns: BTreeMap::new(),
        };
        let features = Features::compile(spec).unwrap();
        let text = "THIS IS URGENT! act now?";
        let words = crate::detect::token::tokens(text);
        assert!(features.values(text, &words, None)[0] > 0.0);
    }
}
