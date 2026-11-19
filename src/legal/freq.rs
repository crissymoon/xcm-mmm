//! How much of general English a reader has already met.
//!
//! The glossary counts drafting. It cannot tell `under` from `hereunder`: both
//! are one hit, both are avoidable, and a density that treats them alike says a
//! contract is as hard to read as the number of wordy phrases in it. That is
//! wrong in the one way that matters to a reader, because `under` is in every
//! newspaper and `hereunder` is in no newspaper at all. Complexity in the
//! literature is graded rather than binary for the same reason: CompLex
//! (Shardlow et al., 2021) collects a five point difficulty rating per word and
//! finds annotators agreeing on the middle of the scale, and the CWI shared
//! tasks (Paetzold & Specia 2016; Yimam et al., 2018) mark a word complex when
//! a reader may not know it. Rarity is the cheapest observable stand-in for
//! that: a word nobody writes outside a courthouse is a word a reader has to
//! stop at.
//!
//! The measurement is a frequency table over general English, written by
//! `scripts/lex-freq.py` from the Brown corpus: 1,161,192 tokens of American
//! English from 1961. Two limits follow from that and are not argued away here.
//! It is old, so a word coined since 1961 reads as rare, and it is small, so a
//! word absent from it is not proven unknown, only unused in a million words of
//! edited prose. Both push in the same direction, which is why the reading
//! reports a word rather than a score: `hereunder` and `witnesseth` are the
//! words it names, and a reader can disagree with the list by reading it.
//!
//! The cut is a threshold in `settings.json` (`rarity.max_per_million`), not a
//! constant here. At one per million, a word must be absent from the corpus or
//! appear once to be reported. `zipf` is exposed for a caller that wants the
//! graded value: it is `3 + log10(per million)`, the convention the word
//! frequency literature uses, so the scale runs from about 7 for `the` to 0 for
//! a word no general corpus holds.

use std::collections::BTreeMap;
use std::fs;

use serde::{Deserialize, Serialize};

/// A frequency table over general English, loaded beside the glossary.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Freq {
    /// The corpus the counts came from, carried with the numbers so a reading
    /// can say what it was measured against.
    #[serde(default)]
    pub source: String,
    /// Tokens the corpus holds.
    #[serde(default)]
    pub tokens: usize,
    /// Occurrences per million tokens. A word not in the map was never used.
    #[serde(default)]
    pub per_million: BTreeMap<String, f64>,
}

impl Freq {
    /// Read `freq.json`. The file is generated, so a missing or malformed one is
    /// a refusal that names the script rather than a silent empty table: an
    /// empty table would report every word as unknown.
    pub fn load(path: &str) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| {
            format!("{path}: {e} (regenerate the frequency table with scripts/lex-freq.py)")
        })?;
        let freq: Freq = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        if freq.per_million.is_empty() {
            return Err(format!("{path}: the frequency table holds no words"));
        }
        Ok(freq)
    }

    /// Occurrences per million tokens, zero for a word the corpus never used.
    pub fn per_million(&self, word: &str) -> f64 {
        self.per_million
            .get(&word.to_lowercase())
            .copied()
            .unwrap_or(0.0)
    }

    /// The graded value: `3 + log10(per million)`, floored at zero.
    ///
    /// A word at 1,000 per million scores 6, one at 10 scores 4, one the corpus
    /// does not hold scores 0. The scale is the one word-frequency work reports,
    /// which is why it is worth naming a value at all.
    pub fn zipf(&self, word: &str) -> f64 {
        let rate = self.per_million(word);
        if rate <= 0.0 {
            0.0
        } else {
            (3.0 + rate.log10()).max(0.0)
        }
    }

    /// The least frequent word in a phrase, with its rate.
    ///
    /// Function words need no stop list: `of` and `in` are never the rarest word
    /// in a phrase a legal drafter wrote, and a rule that skipped them could not
    /// report `in furtherance of`, which is rare because of `furtherance`.
    /// Ties keep the earlier word, so the answer is stable across runs.
    pub fn rarest(&self, phrase: &str) -> Option<(String, f64)> {
        let mut best: Option<(String, f64)> = None;
        for word in phrase.split(|c: char| !c.is_alphabetic()) {
            if word.is_empty() {
                continue;
            }
            let rate = self.per_million(word);
            let better = match &best {
                Some((_, seen)) => rate < *seen,
                None => true,
            };
            if better {
                best = Some((word.to_lowercase(), rate));
            }
        }
        best
    }

    /// The word that makes a phrase unknown to a general reader, when the
    /// phrase holds one: the rarest word, reported only if it is under `below`
    /// occurrences per million.
    pub fn unknown(&self, phrase: &str, below: f64) -> Option<String> {
        match self.rarest(phrase) {
            Some((word, rate)) if rate < below => Some(word),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn freq() -> Freq {
        let mut per_million = BTreeMap::new();
        per_million.insert("under".to_string(), 600.0);
        per_million.insert("hereunder".to_string(), 0.4);
        per_million.insert("in".to_string(), 9000.0);
        per_million.insert("the".to_string(), 7000.0);
        per_million.insert("of".to_string(), 3000.0);
        per_million.insert("absence".to_string(), 45.0);
        per_million.insert("furtherance".to_string(), 0.0);
        Freq {
            source: "a corpus".to_string(),
            tokens: 1_000_000,
            per_million,
        }
    }

    #[test]
    fn a_word_the_corpus_never_used_has_no_rate() {
        assert_eq!(freq().per_million("witnesseth"), 0.0);
    }

    #[test]
    fn the_rarest_word_is_the_one_that_makes_the_phrase_hard() {
        assert_eq!(
            freq().rarest("in furtherance of").map(|(w, _)| w),
            Some("furtherance".to_string())
        );
        assert_eq!(
            freq().rarest("prior to").map(|(w, _)| w),
            Some("prior".to_string())
        );
    }

    #[test]
    fn a_function_word_is_never_reported_as_the_rare_one() {
        let freq = freq();
        assert_eq!(
            freq.rarest("in the absence of").map(|(w, _)| w),
            Some("absence".to_string())
        );
    }

    #[test]
    fn the_cut_belongs_to_the_settings_and_not_to_this_module() {
        let freq = freq();
        assert_eq!(
            freq.unknown("hereunder", 1.0),
            Some("hereunder".to_string())
        );
        assert_eq!(freq.unknown("hereunder", 0.1), None);
        assert_eq!(freq.unknown("under", 1.0), None);
    }

    #[test]
    fn the_graded_value_falls_with_frequency_and_stops_at_zero() {
        let freq = freq();
        assert!(freq.zipf("in") > freq.zipf("under"), "in > under");
        assert!(
            freq.zipf("under") > freq.zipf("hereunder"),
            "under > hereunder"
        );
        assert_eq!(freq.zipf("witnesseth"), 0.0);
    }

    #[test]
    fn a_phrase_with_no_letters_holds_no_word() {
        assert!(freq().rarest("---").is_none());
    }
}
