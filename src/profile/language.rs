//! Marketing copy read as copy.
//!
//! The segment model is fitted on what an advertiser did, and only four of its
//! eight columns leave a trace in a single piece of text: how long the copy
//! runs, how many distinct terms it buys against, whether the copy repeats
//! itself, and how far one phrase dominates it. The first two are counted
//! directly. The last two are proxies, because an advertiser's `repeat` is the
//! share of its ads whose fingerprint recurs and its `concentrate` is the share
//! of its ads carrying one fingerprint, neither of which one ad can show. The
//! proxies keep the same units and the same meaning at one step smaller:
//! the share of a text's four word shingles that recur, and the share of them
//! held by the most common shingle.
//!
//! Everything else here is a surface reading of the copy as marketing: how
//! directly it addresses the reader, how specific it is, how hard it punches
//! per sentence and how much emotional mass it carries. These are reported,
//! never blended into a single invented score: a reader can check every one of
//! them against the text.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::detect::token;
use crate::text::hash;

/// Shingle width for the reuse proxies. Four words is the shortest window that
/// survives a rewritten subject line and still names a phrase.
const SHINGLE: usize = 4;

/// What the copy reads as, before it is placed against a segment.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LanguageView {
    /// Words in the copy, which is the `copy length` column.
    pub words: usize,
    /// Distinct lowercased words.
    pub distinct: usize,
    /// Distinct words outside the closed grammatical class.
    pub content_terms: usize,
    /// Distinct words over words.
    pub type_token: f64,
    /// Mean words per sentence, the punch reading.
    pub sentence_words: f64,
    /// Share of words that are you, your, yours, yourself or yourselves.
    pub second_person: f64,
    /// Digit runs per hundred words, the specificity reading.
    pub numbers_per_100: f64,
    pub questions: usize,
    pub exclamations: usize,
    /// Share of words the emotion model read above the neutral prior.
    pub emotional_density: f64,
    /// Share of the copy's words inside a four word phrase it uses more than
    /// once. This is the `creative reuse` column at one ad's scale.
    pub reuse: f64,
    /// Share of the copy's words inside its single most repeated phrase.
    pub focus: f64,
    /// The most common four word phrase, empty when none recurs.
    pub top_phrase: String,
    /// True when the copy is shorter than one shingle and the reuse proxies
    /// cannot be read at all.
    pub too_short: bool,
}

impl LanguageView {
    /// Whether a surface read is present rather than zero for want of a word.
    pub fn sentence_rate(&self, hits: usize) -> f64 {
        if self.sentence_words <= 0.0 {
            0.0
        } else {
            let sentences = (self.words as f64 / self.sentence_words).max(1.0);
            hits as f64 / sentences
        }
    }
}

/// Read one text as marketing copy.
///
/// `words` must be the tokens the report already counted, so the shape block
/// and this block cannot disagree about how long the text is. `emotional` is
/// the number of words the emotion model kept, which is a model output rather
/// than a count this module could make itself.
pub fn read(text: &str, words: &[String], sentences: usize, emotional: usize) -> LanguageView {
    let copies = words.len().max(1);
    let distinct: BTreeSet<&str> = words.iter().map(String::as_str).collect();
    let content_terms = distinct
        .iter()
        .filter(|word| word.len() > 1 && !token::is_function(word))
        .count();

    let lowered: Vec<u64> = words.iter().map(|word| hash::word_hash(word)).collect();
    let shingles = windows(&lowered, SHINGLE);
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for shingle in &shingles {
        *counts.entry(*shingle).or_insert(0) += 1;
    }
    let (top, top_count) = counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(key, count)| (*key, *count))
        .unwrap_or((0, 0));
    let reuse = coverage(&shingles, &counts, words.len(), |count| count > 1);
    // With no repeated phrase the commonest window is every window, and
    // calling that total concentration would read a flat pitch as a hook.
    let focus = if top_count > 1 {
        coverage(&shingles, &counts, words.len(), |count| count == top_count)
    } else {
        0.0
    };
    let top_phrase = if top_count > 1 {
        phrase_of(words, &shingles, top).unwrap_or_default()
    } else {
        String::new()
    };

    LanguageView {
        words: words.len(),
        distinct: distinct.len(),
        content_terms,
        type_token: distinct.len() as f64 / copies as f64,
        sentence_words: words.len() as f64 / sentences.max(1) as f64,
        second_person: words
            .iter()
            .filter(|word| {
                matches!(
                    word.as_str(),
                    "you" | "your" | "yours" | "yourself" | "yourselves"
                )
            })
            .count() as f64
            / copies as f64,
        numbers_per_100: digit_runs(text) as f64 * 100.0 / copies as f64,
        questions: text.matches('?').count(),
        exclamations: text.matches('!').count(),
        emotional_density: emotional as f64 / copies as f64,
        reuse,
        focus,
        top_phrase,
        too_short: shingles.is_empty(),
    }
}

/// Share of word positions covered by windows the filter keeps.
///
/// A position counts once however many repeated windows cover it, so the
/// reading stays a share of the copy and cannot exceed one. It is the same
/// unit as the panel's `repeat`, which is a share of ads.
fn coverage(
    shingles: &[u64],
    counts: &BTreeMap<u64, usize>,
    words: usize,
    keep: impl Fn(usize) -> bool,
) -> f64 {
    if shingles.is_empty() || words == 0 {
        return 0.0;
    }
    let mut covered = vec![false; words];
    for (at, shingle) in shingles.iter().enumerate() {
        let count = counts.get(shingle).copied().unwrap_or(0);
        if !keep(count) {
            continue;
        }
        for slot in covered.iter_mut().skip(at).take(SHINGLE) {
            *slot = true;
        }
    }
    covered.iter().filter(|hit| **hit).count() as f64 / words as f64
}

/// Every window's hash, duplicates kept.
///
/// [`hash::shingles`] returns a set for MinHash, which is the right shape for
/// comparing two documents and the wrong shape here: reuse is a count, so the
/// repeated window has to survive the hashing.
fn windows(words: &[u64], k: usize) -> Vec<u64> {
    if words.len() < k {
        return Vec::new();
    }
    words
        .windows(k)
        .map(|window| {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for &word in window {
                h = (h ^ word).wrapping_mul(0x0100_0000_01b3);
            }
            h
        })
        .collect()
}

/// The words of the shingle whose hash is `key`, read back from the windows.
fn phrase_of(words: &[String], shingles: &[u64], key: u64) -> Option<String> {
    let at = shingles.iter().position(|shingle| *shingle == key)?;
    Some(words.get(at..at + SHINGLE)?.join(" "))
}

/// Numbers in the text, so `2 for $49.99` counts two and `2026` counts one.
///
/// A dot or comma between two digits stays inside the number, because `49.99`
/// is one price and counting it as two overstates how specific the copy is.
fn digit_runs(text: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut runs = 0;
    let mut at = 0;
    while at < chars.len() {
        if !chars[at].is_ascii_digit() {
            at += 1;
            continue;
        }
        runs += 1;
        while at < chars.len() {
            let digit = chars[at].is_ascii_digit();
            let separator = matches!(chars[at], '.' | ',')
                && chars.get(at + 1).is_some_and(|c| c.is_ascii_digit());
            if digit || separator {
                at += 1;
            } else {
                break;
            }
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        token::tokens(text)
    }

    #[test]
    fn a_repeated_hook_is_visible_as_reuse_and_focus() {
        let text = "Act now, act now, act now, the seats are going.";
        let tokens = words(text);
        let view = read(text, &tokens, 1, 4);
        assert!(view.reuse > 0.0, "{view:?}");
        assert!(view.focus > 0.2, "{view:?}");
        assert_eq!(view.top_phrase, "act now act now");
        assert!(!view.too_short);
    }

    #[test]
    fn a_flat_pitch_has_no_reuse() {
        let text = "We opened a small office and hired two people this spring.";
        let tokens = words(text);
        let view = read(text, &tokens, 1, 0);
        assert_eq!(view.reuse, 0.0);
        assert_eq!(view.top_phrase, "");
        assert!(view.content_terms >= 6, "{view:?}");
    }

    #[test]
    fn a_fragment_that_cannot_hold_a_shingle_says_so() {
        let text = "Buy now";
        let tokens = words(text);
        let view = read(text, &tokens, 1, 1);
        assert!(view.too_short);
        assert_eq!(view.reuse, 0.0);
        assert_eq!(view.focus, 0.0);
    }

    #[test]
    fn numbers_are_counted_as_runs_not_digits() {
        assert_eq!(digit_runs("2 for $49.99 in 2026"), 3);
        assert_eq!(digit_runs("no numbers here"), 0);
    }
}
