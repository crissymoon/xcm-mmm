//! Counting what "handles every English word" actually means.
//!
//! The claim worth making is not that the lexicon lists every word; it cannot.
//! It is that every word gets a reading from a named tier, and that the tiers
//! above the neutral prior carry most of a dictionary. This module reads a word
//! list, asks the scanner for one word at a time, and counts the answer. It also
//! counts the words that came back with real evidence rather than a near zero,
//! because a tier that always answers neutral is a tier in name only.
//!
//! The default list is `/usr/share/dict/words` on macOS. A caller can point at
//! any newline separated list; the report is the same shape.

use std::collections::BTreeMap;
use std::fs;
use std::io;

use super::model::{Scanner, Tier};

/// Evidence below this weight is not called a reading.
pub const INFORMATIVE: f64 = 0.15;

/// What one word list resolved to.
#[derive(Clone, Debug, Default)]
pub struct Coverage {
    pub words: usize,
    pub tiers: BTreeMap<&'static str, usize>,
    pub informative: BTreeMap<&'static str, usize>,
    pub examples: BTreeMap<&'static str, Vec<String>>,
}

impl Coverage {
    pub fn total(&self, tier: Tier) -> usize {
        self.tiers.get(tier.as_str()).copied().unwrap_or(0)
    }

    /// Words read from evidence, at any tier above the prior.
    pub fn handled(&self) -> usize {
        self.tiers
            .iter()
            .filter(|(tier, _)| **tier != Tier::Prior.as_str())
            .map(|(_, count)| count)
            .sum()
    }

    /// Words whose reading is more than a near zero.
    pub fn informative_total(&self) -> usize {
        self.informative.values().sum()
    }

    pub fn share(&self, count: usize) -> f64 {
        if self.words == 0 {
            0.0
        } else {
            count as f64 / self.words as f64
        }
    }
}

/// What a corpus of running text resolved to, by token rather than by type.
#[derive(Clone, Debug, Default)]
pub struct Corpus {
    pub tokens: usize,
    /// Tokens outside the closed class, the ones that can carry affect.
    pub content: usize,
    /// Content tokens that fell to the neutral prior.
    pub prior_content: usize,
    pub tiers: BTreeMap<&'static str, usize>,
}

impl Corpus {
    pub fn handled(&self) -> usize {
        self.tiers
            .iter()
            .filter(|(tier, _)| **tier != Tier::Prior.as_str())
            .map(|(_, count)| count)
            .sum()
    }

    pub fn share(&self, count: usize) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            count as f64 / self.tokens as f64
        }
    }

    /// Content tokens read above the prior, over content tokens.
    pub fn content_share(&self) -> f64 {
        if self.content == 0 {
            0.0
        } else {
            (self.content - self.prior_content) as f64 / self.content as f64
        }
    }
}

/// Measure every token of a JSON Lines corpus, using each row's `text` field.
///
/// A dictionary weights `abacate` the same as `the`. A corpus does not, and the
/// question a reader actually has is whether the words they write are covered.
pub fn measure_corpus(scanner: &Scanner, path: &str) -> io::Result<Corpus> {
    let text = fs::read_to_string(path)?;
    let mut corpus = Corpus::default();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let body = match serde_json::from_str::<serde_json::Value>(line) {
            Ok(value) => value
                .get("text")
                .and_then(|text| text.as_str())
                .unwrap_or(line)
                .to_string(),
            Err(_) => line.to_string(),
        };
        for word in crate::detect::token::tokens(&body) {
            corpus.tokens += 1;
            let resolved = scanner.resolve(&word);
            let tier = resolved.tier.as_str();
            *corpus.tiers.entry(tier).or_insert(0) += 1;
            if !crate::detect::token::is_function(&word) {
                corpus.content += 1;
                if resolved.tier == Tier::Prior {
                    corpus.prior_content += 1;
                }
            }
        }
    }
    Ok(corpus)
}

/// Measure a word list.
pub fn measure(scanner: &Scanner, path: &str, examples: usize) -> io::Result<Coverage> {
    let text = fs::read_to_string(path)?;
    let mut coverage = Coverage::default();
    for line in text.lines() {
        let word = line.trim();
        if word.is_empty() {
            continue;
        }
        coverage.words += 1;
        let resolved = scanner.resolve(word);
        let tier = resolved.tier.as_str();
        *coverage.tiers.entry(tier).or_insert(0) += 1;
        if resolved.weights.iter().copied().fold(0.0, f64::max) >= INFORMATIVE {
            *coverage.informative.entry(tier).or_insert(0) += 1;
        } else if resolved.tier != Tier::Prior
            && coverage.examples.get(tier).map(Vec::len).unwrap_or(0) < examples
        {
            coverage
                .examples
                .entry(tier)
                .or_default()
                .push(word.to_string());
        }
    }
    Ok(coverage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emotion::model::EmotionModel;

    fn scanner() -> Option<Scanner> {
        let path = "data/emotion/model.json";
        if !std::path::Path::new(path).is_file() {
            return None;
        }
        Some(EmotionModel::load(path).ok()?.scanner())
    }

    #[test]
    fn every_listed_word_gets_a_tier_and_the_tiers_add_up() {
        let Some(scanner) = scanner() else {
            return;
        };
        let coverage = measure(&scanner, "data/emotion/seed.json", 3).expect("a measure");
        let counted: usize = coverage.tiers.values().sum();
        assert_eq!(counted, coverage.words);
        assert!(coverage.words > 0);
    }

    #[test]
    fn an_unknown_word_falls_to_the_prior_rather_than_vanishing() {
        let Some(scanner) = scanner() else {
            return;
        };
        let covered = measure(&scanner, "data/emotion/seed.json", 1).expect("a measure");
        assert_eq!(
            covered.handled() + covered.total(Tier::Prior),
            covered.words,
            "handled and prior must account for every word"
        );
    }

    #[test]
    fn a_word_the_seed_lists_is_never_left_at_the_prior() {
        let Some(scanner) = scanner() else {
            return;
        };
        let model = scanner.model();
        for word in model.seed.keys() {
            assert_ne!(
                scanner.resolve(word).tier,
                Tier::Prior,
                "{word} is seeded but unread"
            );
        }
    }
}
