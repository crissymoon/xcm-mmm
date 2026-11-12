//! Quality scoring: measurable statistics, published rules, and a small
//! logistic model that can be fitted from labelled keep and drop examples.
//!
//! Three layers, each independent of the others:
//!
//! * [`Stats`] measures a document. Nothing is judged here, so the numbers can
//!   be compared across corpora.
//! * [`Rules`] applies thresholds of the kind Gopher and C4 publish, returning
//!   a named [`Flag`] per failure. A flag is a reason, not a score, so a
//!   rejected document can be explained in one line.
//! * [`QualityModel`] is a logistic regression over the same statistics, fitted
//!   from examples the caller trusts. It is nine features, so it cannot
//!   overfit a small set and its weights can be read out loud.
//!
//! The composite score in [`Score`] is the fallback when no labelled examples
//! exist. It is a weighted sum of bounded components, each of which says one
//! thing: length, letters, variety, repetition, structure and function words.

use std::collections::{HashMap, HashSet};

use crate::text::tokenize;

/// The 120 most common English words. A document with almost none of them is a
/// list, a table fragment or a keyword soup rather than prose.
pub const STOPWORDS: &[&str] = &[
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "his",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "may",
    "might",
    "more",
    "most",
    "much",
    "must",
    "my",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "said",
    "same",
    "she",
    "should",
    "since",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "two",
    "under",
    "up",
    "us",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "among",
    "between",
    "during",
    "against",
    "per",
    "each",
    "both",
    "being",
    "having",
    "within",
    "without",
    "across",
    "still",
    "yet",
    "though",
    "however",
    "therefore",
];

fn is_stopword(lowered: &str) -> bool {
    STOPWORDS.contains(&lowered)
}

/// Measurements of one document. Every field is a count or a ratio over the
/// document as given; nothing is imputed.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub chars: usize,
    pub words: usize,
    pub unique: usize,
    pub sentences: usize,
    pub mean_word_len: f64,
    /// Share of characters that are letters.
    pub alpha: f64,
    /// Share of characters that are digits.
    pub digit: f64,
    /// Share of characters that are punctuation.
    pub punct: f64,
    /// Share of letters that are uppercase.
    pub upper: f64,
    /// Share of words found in [`STOPWORDS`].
    pub stop: f64,
    /// Share of lines that repeat an earlier line.
    pub dup_line: f64,
    /// Frequency of the most repeated content word, over all words.
    pub top_token: f64,
    /// Share of lines that start like a list item.
    pub bullets: f64,
    pub terminal: bool,
    /// Type to token ratio.
    pub unique_frac: f64,
    pub mean_sentence_words: f64,
}

impl Stats {
    /// Measure a document.
    pub fn of(text: &str) -> Self {
        let chars = text.chars().count();
        let mut words = 0usize;
        let mut word_chars = 0usize;
        let mut stops = 0usize;
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut seen: HashSet<String> = HashSet::new();
        for word in tokenize::words(text) {
            words += 1;
            word_chars += word.chars().count();
            let lowered = tokenize::lowercase_word(word);
            seen.insert(lowered.clone());
            if is_stopword(&lowered) {
                stops += 1;
            } else {
                *counts.entry(lowered).or_insert(0) += 1;
            }
        }
        let upper = text.chars().filter(|c| c.is_uppercase()).count();
        let alpha = text.chars().filter(|c| c.is_alphabetic()).count();
        let digit = text.chars().filter(|c| c.is_ascii_digit()).count();
        let punct = text
            .chars()
            .filter(|c| c.is_ascii_punctuation() && *c != '_')
            .count();
        let lines: Vec<String> = text
            .lines()
            .map(|line| tokenize::lowercase_word(line.trim()))
            .filter(|line| !line.is_empty())
            .collect();
        let mut line_seen: HashSet<&str> = HashSet::new();
        let dup = lines
            .iter()
            .filter(|line| !line_seen.insert(line.as_str()))
            .count();
        let bullets = text
            .lines()
            .filter(|line| {
                let t = line.trim_start();
                t.starts_with("- ")
                    || t.starts_with("* ")
                    || t.starts_with("\u{2022}")
                    || t.split_once('.')
                        .map(|(n, _)| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
                        == Some(true)
            })
            .count();
        let top = counts.values().copied().max().unwrap_or(0);
        let sentences = tokenize::sentences(text, 1).len().max(1);
        let last = text.trim_end().chars().last().unwrap_or(' ');
        let denom = chars.max(1) as f64;
        let word_count = words.max(1);
        Self {
            chars,
            words,
            unique: seen.len(),
            sentences,
            mean_word_len: word_chars as f64 / word_count as f64,
            alpha: alpha as f64 / denom,
            digit: digit as f64 / denom,
            punct: punct as f64 / denom,
            upper: if alpha == 0 {
                0.0
            } else {
                upper as f64 / alpha as f64
            },
            stop: stops as f64 / word_count as f64,
            dup_line: if lines.is_empty() {
                0.0
            } else {
                dup as f64 / lines.len() as f64
            },
            top_token: top as f64 / word_count as f64,
            bullets: if lines.is_empty() {
                0.0
            } else {
                bullets as f64 / lines.len() as f64
            },
            terminal: matches!(last, '.' | '!' | '?' | '"' | '\'' | ')' | ':'),
            unique_frac: seen.len() as f64 / word_count as f64,
            mean_sentence_words: word_count as f64 / sentences as f64,
        }
    }
}

/// Why a document failed a rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flag {
    TooShort,
    TooLong,
    WordsTooShort,
    WordsTooLong,
    MachineLike,
    NoFunctionWords,
    SymbolHeavy,
    RepeatedLines,
    RepeatedWord,
    NoTerminator,
    ListHeavy,
}

impl Flag {
    /// One line a report can print.
    pub fn reason(self) -> &'static str {
        match self {
            Flag::TooShort => "fewer words than the floor",
            Flag::TooLong => "more words than the ceiling",
            Flag::WordsTooShort => "mean word length below the floor",
            Flag::WordsTooLong => "mean word length above the ceiling",
            Flag::MachineLike => "uppercase share is not prose",
            Flag::NoFunctionWords => "almost no function words",
            Flag::SymbolHeavy => "punctuation and digits dominate",
            Flag::RepeatedLines => "too many repeated lines",
            Flag::RepeatedWord => "one word carries the document",
            Flag::NoTerminator => "no sentence terminator",
            Flag::ListHeavy => "mostly list items",
        }
    }
}

/// Thresholds. Defaults follow the published Gopher and C4 rules where the
/// corpus allows it, and are loosened for the short forms (headlines and ad
/// copy) that this crate is pointed at. A caller that needs the strict web
/// values can set them explicitly.
#[derive(Clone, Debug)]
pub struct Rules {
    pub min_words: usize,
    pub max_words: usize,
    pub min_mean_word_len: f64,
    pub max_mean_word_len: f64,
    pub min_stop: f64,
    pub max_upper: f64,
    pub max_symbol: f64,
    pub max_dup_line: f64,
    pub max_top_token: f64,
    pub max_bullets: f64,
    pub require_terminal: bool,
    pub terminal_exempt_words: usize,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            min_words: 5,
            max_words: 3000,
            min_mean_word_len: 2.8,
            max_mean_word_len: 12.0,
            min_stop: 0.04,
            max_upper: 0.45,
            max_symbol: 0.22,
            max_dup_line: 0.30,
            max_top_token: 0.20,
            max_bullets: 0.60,
            require_terminal: false,
            terminal_exempt_words: 14,
        }
    }
}

impl Rules {
    /// Every rule the document fails, in a stable order.
    pub fn flags(&self, s: &Stats) -> Vec<Flag> {
        let mut out = Vec::new();
        if s.words < self.min_words {
            out.push(Flag::TooShort);
        }
        if s.words > self.max_words {
            out.push(Flag::TooLong);
        }
        if s.words >= 3 && s.mean_word_len < self.min_mean_word_len {
            out.push(Flag::WordsTooShort);
        }
        if s.mean_word_len > self.max_mean_word_len {
            out.push(Flag::WordsTooLong);
        }
        if s.upper > self.max_upper && s.words >= 10 {
            out.push(Flag::MachineLike);
        }
        if s.stop < self.min_stop && s.words >= 8 {
            out.push(Flag::NoFunctionWords);
        }
        if s.punct + s.digit > self.max_symbol {
            out.push(Flag::SymbolHeavy);
        }
        if s.dup_line > self.max_dup_line {
            out.push(Flag::RepeatedLines);
        }
        if s.top_token > self.max_top_token && s.words >= 10 {
            out.push(Flag::RepeatedWord);
        }
        if self.require_terminal && !s.terminal && s.words > self.terminal_exempt_words {
            out.push(Flag::NoTerminator);
        }
        if s.bullets > self.max_bullets {
            out.push(Flag::ListHeavy);
        }
        out
    }

    /// Whether a document passes every rule.
    pub fn passes(&self, s: &Stats) -> bool {
        self.flags(s).is_empty()
    }
}

/// Weights of the composite score. They sum to one by construction; the
/// constructor normalises them, so a caller can pass any positive numbers.
#[derive(Clone, Debug)]
pub struct Weights {
    pub length: f64,
    pub lexical: f64,
    pub diversity: f64,
    pub repetition: f64,
    pub structure: f64,
    pub function_words: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            length: 0.22,
            lexical: 0.20,
            diversity: 0.20,
            repetition: 0.14,
            structure: 0.12,
            function_words: 0.12,
        }
    }
}

impl Weights {
    fn total(&self) -> f64 {
        self.length
            + self.lexical
            + self.diversity
            + self.repetition
            + self.structure
            + self.function_words
    }
}

/// The composite score, with each component kept so a report can show which
/// one moved.
#[derive(Clone, Debug, Default)]
pub struct Score {
    pub length: f64,
    pub lexical: f64,
    pub diversity: f64,
    pub repetition: f64,
    pub structure: f64,
    pub function_words: f64,
    pub total: f64,
}

fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

/// Score a document in [0, 1]. The length component peaks at 40 words, which is
/// one headline plus a lead sentence, and decays logarithmically either side.
pub fn score(s: &Stats, w: &Weights) -> Score {
    let peak = 40.0f64;
    let ratio = (s.words as f64 + 1.0) / peak;
    let length = clamp01(1.0 - (ratio.ln().abs() / peak.ln()));
    let lexical = clamp01((s.alpha - 0.45) / 0.40);
    let diversity = if s.words < 8 {
        0.8
    } else {
        clamp01((s.unique_frac - 0.35) / 0.45)
    };
    let repetition = clamp01(1.0 - s.top_token / 0.18);
    let structure = if s.mean_sentence_words >= 6.0 && s.mean_sentence_words <= 45.0 {
        1.0
    } else if s.mean_sentence_words < 6.0 {
        clamp01(s.mean_sentence_words / 6.0)
    } else {
        clamp01(45.0 / s.mean_sentence_words)
    };
    let function_words = clamp01(s.stop / 0.20);
    let total_weight = w.total().max(1e-12);
    let total = (length * w.length
        + lexical * w.lexical
        + diversity * w.diversity
        + repetition * w.repetition
        + structure * w.structure
        + function_words * w.function_words)
        / total_weight;
    Score {
        length,
        lexical,
        diversity,
        repetition,
        structure,
        function_words,
        total,
    }
}

/// Feature names of [`QualityModel`], in weight order.
pub const MODEL_FEATURES: &[&str] = &[
    "log_words",
    "mean_word_len",
    "alpha",
    "digit",
    "stop",
    "unique_frac",
    "dup_line",
    "top_token",
    "sentences_per_100",
];

fn features(s: &Stats) -> [f64; 9] {
    [
        (s.words as f64 + 1.0).ln(),
        s.mean_word_len,
        s.alpha,
        s.digit,
        s.stop,
        s.unique_frac,
        s.dup_line,
        s.top_token,
        s.sentences as f64 * 100.0 / s.words.max(1) as f64,
    ]
}

/// A logistic model over [`MODEL_FEATURES`], fitted by full batch gradient
/// descent with L2 shrinkage. Deterministic from any starting point of zeros,
/// so two runs on the same rows produce the same weights.
#[derive(Clone, Debug)]
pub struct QualityModel {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub weights: Vec<f64>,
    pub bias: f64,
    /// Mean log loss on the training rows at the last epoch.
    pub loss: f64,
}

impl QualityModel {
    /// Fit from negatives and positives. Both slices must be non-empty; the
    /// positives and negatives are balanced by weight so a 9 to 1 corpus does
    /// not collapse to the majority class.
    pub fn fit(neg: &[Stats], pos: &[Stats], epochs: usize, lr: f64, l2: f64) -> Option<Self> {
        if neg.is_empty() || pos.is_empty() {
            return None;
        }
        let dim = MODEL_FEATURES.len();
        let mut all: Vec<[f64; 9]> = Vec::with_capacity(neg.len() + pos.len());
        all.extend(neg.iter().map(features));
        all.extend(pos.iter().map(features));
        let mut mean = vec![0.0; dim];
        for row in &all {
            for (i, v) in row.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= all.len() as f64;
        }
        let mut scale = vec![0.0; dim];
        for row in &all {
            for (i, v) in row.iter().enumerate() {
                scale[i] += (v - mean[i]).powi(2);
            }
        }
        for s in &mut scale {
            *s = (*s / all.len() as f64).sqrt().max(1e-6);
        }
        let standardise = |row: &[f64; 9]| -> Vec<f64> {
            row.iter()
                .enumerate()
                .map(|(i, v)| (v - mean[i]) / scale[i])
                .collect()
        };
        let x: Vec<Vec<f64>> = all.iter().map(standardise).collect();
        let n_neg = neg.len();
        let w_pos = all.len() as f64 / (2.0 * pos.len() as f64);
        let w_neg = all.len() as f64 / (2.0 * n_neg as f64);
        let y: Vec<(f64, f64)> = (0..all.len())
            .map(|i| {
                if i < n_neg {
                    (0.0, w_neg)
                } else {
                    (1.0, w_pos)
                }
            })
            .collect();
        let mut weights = vec![0.0; dim];
        let mut bias = 0.0;
        let mut loss = 0.0;
        for _ in 0..epochs.max(1) {
            let mut grad = vec![0.0; dim];
            let mut grad_bias = 0.0;
            loss = 0.0;
            for (row, &(label, sample_weight)) in x.iter().zip(y.iter()) {
                let mut z = bias;
                for (i, v) in row.iter().enumerate() {
                    z += weights[i] * v;
                }
                let p = 1.0 / (1.0 + (-z).exp());
                let p = p.clamp(1e-9, 1.0 - 1e-9);
                let error = (p - label) * sample_weight;
                for (i, v) in row.iter().enumerate() {
                    grad[i] += error * v;
                }
                grad_bias += error;
                loss -= (label * p.ln() + (1.0 - label) * (1.0 - p).ln()) * sample_weight;
            }
            let n = all.len() as f64;
            for i in 0..dim {
                weights[i] -= lr * (grad[i] / n + l2 * weights[i]);
            }
            bias -= lr * grad_bias / n;
            loss /= n;
        }
        Some(Self {
            mean,
            scale,
            weights,
            bias,
            loss,
        })
    }

    /// Probability that a document is a keep.
    pub fn predict(&self, s: &Stats) -> f64 {
        let row = features(s);
        let mut z = self.bias;
        for (i, v) in row.iter().enumerate() {
            z += self.weights[i] * (v - self.mean[i]) / self.scale[i];
        }
        1.0 / (1.0 + (-z).exp())
    }

    /// Weights as (name, weight) pairs, in the fitted order.
    pub fn named_weights(&self) -> Vec<(&'static str, f64)> {
        MODEL_FEATURES
            .iter()
            .copied()
            .zip(self.weights.iter().copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROSE: &str = "The Federal Reserve held interest rates steady on Wednesday and said it would watch inflation before moving again. Officials pointed to a cooling labour market and slower wage growth as evidence that policy is working.";

    #[test]
    fn prose_scores_above_a_keyword_soup() {
        let prose = Stats::of(PROSE);
        let soup = Stats::of("rates rates rates fed fed inflation inflation cuts cuts hikes hikes");
        let w = Weights::default();
        assert!(score(&prose, &w).total > score(&soup, &w).total);
    }

    #[test]
    fn rules_flag_the_soup_but_not_the_prose() {
        let rules = Rules::default();
        assert!(rules.passes(&Stats::of(PROSE)));
        let flags = rules.flags(&Stats::of("buy buy buy buy buy buy buy buy buy buy buy"));
        assert!(flags.contains(&Flag::RepeatedWord), "{flags:?}");
    }

    #[test]
    fn model_separates_two_clear_families() {
        let neg: Vec<Stats> = (0..12)
            .map(|i| Stats::of(&format!("click here now !! offer {} {} !!", i, i)))
            .collect();
        let pos: Vec<Stats> = (0..12)
            .map(|i| {
                Stats::of(&format!(
                    "The company said revenue rose {i} percent in the quarter as demand held up across its industrial and retail units."
                ))
            })
            .collect();
        let model = QualityModel::fit(&neg, &pos, 400, 0.3, 1e-3).expect("fitted");
        let hi = model.predict(&Stats::of(PROSE));
        let lo = model.predict(&Stats::of("CLICK HERE 24/7 !!!"));
        assert!(hi > 0.5 && lo < 0.5, "{hi} {lo}");
        assert_eq!(model.named_weights().len(), MODEL_FEATURES.len());
    }

    #[test]
    fn stats_are_honest_about_repetition() {
        let s = Stats::of("a b a b a b a b");
        assert!(s.words == 8);
        assert!(s.unique == 2);
        assert!(s.unique_frac < 0.3);
    }
}
