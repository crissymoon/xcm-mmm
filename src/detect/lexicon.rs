//! A lexicon learned by smoothed log odds, with the filters that keep it honest.
//!
//! Monroe, Colaresi and Quinn (2008): each term is scored by the log ratio of
//! its rate in the positive bank to its rate in the negative bank, shrunk
//! towards zero by a Dirichlet prior built from the pooled counts. A term seen
//! once in a bank of two thousand words is therefore pulled towards the middle
//! instead of dominating the density, which is what a raw ratio would do.
//!
//! Two filters are deliberately one sided, and both are about what the corpus
//! actually is rather than about the model:
//!
//! * `min_df` counts occurrences in the positive bank only. A drafting word that
//!   never appears in plain copy is the clearest evidence a lexicon can carry,
//!   and demanding it on both sides would discard exactly those terms.
//! * `max_plain_df` is a document frequency, which needs enough documents to
//!   mean anything. With five negatives its granularity is 0.2 and comparing it
//!   to 0.015 would drop every shared word, so the filter is inert below
//!   `min_plain_docs`.
//!
//! The shared `max_rate` filter is a frequency ceiling on both sides. A term
//! that appears in 2 percent of the tokens of either bank is style, not
//! evidence, and it is in every sentence of that register anyway.
//!
//! Scoring a text is [`Lexicon::register`]: capped lexicon mass per hundred
//! words, with a length prior, so one long sentence and one short one land on
//! the same scale.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::token::{self, is_function};

/// Fit settings, in one place, carried into the artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitConfig {
    /// Terms kept, highest z first.
    pub top_terms: usize,
    /// Occurrences a term needs in the positive bank.
    pub min_df: usize,
    /// Frequency ceiling per token on either side.
    pub max_rate: f64,
    /// Document frequency ceiling in the negative bank.
    pub max_plain_df: f64,
    /// Negatives needed before that ceiling is read at all.
    pub min_plain_docs: usize,
    /// Dirichlet prior mass per candidate term.
    pub alpha: f64,
    /// Largest z a single term may carry.
    pub cap: f64,
    /// Length prior of the register, in words.
    pub prior_tokens: f64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            top_terms: 120,
            min_df: 3,
            max_rate: 0.02,
            max_plain_df: 0.015,
            min_plain_docs: 20,
            alpha: 0.01,
            cap: 6.0,
            prior_tokens: 20.0,
        }
    }
}

/// What the fit saw, so a reader can tell a 600 term bank from a 20 term one.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FitReport {
    pub positive_rows: usize,
    pub negative_rows: usize,
    pub positive_tokens: usize,
    pub negative_tokens: usize,
    pub candidates: usize,
    pub kept: usize,
    pub dropped_function: usize,
    pub dropped_rare: usize,
    pub dropped_common: usize,
}

/// A fitted lexicon: term to z, plus the two numbers scoring needs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Lexicon {
    pub terms: BTreeMap<String, f64>,
    pub cap: f64,
    pub prior_tokens: f64,
    pub report: FitReport,
}

impl Lexicon {
    /// Lexicon mass per hundred words, capped per occurrence and length adjusted.
    pub fn register(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }
        let mass: f64 = words
            .iter()
            .filter_map(|word| self.terms.get(word))
            .map(|z| z.min(self.cap))
            .sum();
        100.0 * mass / (words.len() as f64 + self.prior_tokens)
    }

    /// [`Lexicon::register`] for raw text.
    pub fn register_text(&self, text: &str) -> f64 {
        self.register(&token::tokens(text))
    }

    /// The contributing terms of one text, heaviest first, at most `take`.
    pub fn hits(&self, words: &[String], take: usize) -> Vec<Hit> {
        let mut found: BTreeMap<&str, usize> = BTreeMap::new();
        for word in words {
            if self.terms.contains_key(word.as_str()) {
                *found.entry(word.as_str()).or_insert(0) += 1;
            }
        }
        let mut hits: Vec<Hit> = found
            .into_iter()
            .map(|(term, count)| Hit {
                term: term.to_string(),
                count,
                z: self.terms.get(term).copied().unwrap_or(0.0),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.z.total_cmp(&a.z)
                .then_with(|| b.count.cmp(&a.count))
                .then_with(|| a.term.cmp(&b.term))
        });
        hits.truncate(take);
        hits
    }
}

/// One term that carried a reading.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hit {
    pub term: String,
    pub count: usize,
    pub z: f64,
}

/// Fit a lexicon on tokenised banks.
pub fn fit(positive: &[Vec<String>], negative: &[Vec<String>], cfg: &FitConfig) -> Lexicon {
    let (pos, pos_total) = counts(positive);
    let (neg, neg_total) = counts(negative);
    let plain_df = if negative.len() >= cfg.min_plain_docs {
        document_frequency(negative)
    } else {
        BTreeMap::new()
    };
    let neg_docs = negative.len().max(1) as f64;

    let mut report = FitReport {
        positive_rows: positive.len(),
        negative_rows: negative.len(),
        positive_tokens: pos_total,
        negative_tokens: neg_total,
        ..Default::default()
    };

    let mut candidates: Vec<&String> = Vec::new();
    for (word, count) in &pos {
        if word.chars().count() < 3 {
            continue;
        }
        if is_function(word) {
            report.dropped_function += 1;
            continue;
        }
        if *count < cfg.min_df {
            report.dropped_rare += 1;
            continue;
        }
        let share = *count as f64 / pos_total.max(1) as f64;
        let plain = *neg.get(word).unwrap_or(&0) as f64 / neg_total.max(1) as f64;
        let plain_docs = plain_df.get(word).copied().unwrap_or(0.0) / neg_docs;
        if share >= cfg.max_rate || plain >= cfg.max_rate || plain_docs > cfg.max_plain_df {
            report.dropped_common += 1;
            continue;
        }
        candidates.push(word);
    }
    report.candidates = candidates.len();

    let spread = cfg.alpha * candidates.len().max(1) as f64;
    let mut scored: Vec<(String, f64)> = Vec::new();
    for word in candidates {
        let seen = *neg.get(word).unwrap_or(&0) as f64;
        let first = (*pos.get(word).unwrap_or(&0) as f64 + cfg.alpha)
            / (pos_total as f64 + spread - *pos.get(word).unwrap_or(&0) as f64 - cfg.alpha);
        let second = (seen + cfg.alpha) / (neg_total as f64 + spread - seen - cfg.alpha);
        let delta = first.max(1e-12).ln() - second.max(1e-12).ln();
        let sigma = (1.0 / (*pos.get(word).unwrap_or(&0) as f64 + cfg.alpha)
            + 1.0 / (seen + cfg.alpha))
            .sqrt();
        let z = delta / sigma;
        if z > 0.0 {
            scored.push((word.clone(), round4(z.min(cfg.cap))));
        }
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(cfg.top_terms);

    report.kept = scored.len();
    Lexicon {
        terms: scored.into_iter().collect(),
        cap: cfg.cap,
        prior_tokens: cfg.prior_tokens,
        report,
    }
}

fn counts(bank: &[Vec<String>]) -> (BTreeMap<String, usize>, usize) {
    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    let mut total = 0;
    for row in bank {
        for word in row {
            *out.entry(word.clone()).or_insert(0) += 1;
            total += 1;
        }
    }
    (out, total)
}

/// Fraction of documents in which a term appears, not how often it appears.
fn document_frequency(bank: &[Vec<String>]) -> BTreeMap<String, f64> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for row in bank {
        let mut unique: Vec<&str> = row.iter().map(String::as_str).collect();
        unique.sort_unstable();
        unique.dedup();
        for word in unique {
            *seen.entry(word).or_insert(0) += 1;
        }
    }
    let docs = bank.len().max(1) as f64;
    seen.into_iter()
        .map(|(word, n)| (word.to_string(), n as f64 / docs))
        .collect()
}

/// Four decimal places, so a rebuild writes the same bytes.
pub fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bank(rows: &[&str]) -> Vec<Vec<String>> {
        rows.iter().map(|text| token::tokens(text)).collect()
    }

    /// A fixture bank is a handful of sentences, so the frequency ceiling that
    /// guards a real corpus would drop every term in it. Raise it here and keep
    /// the rest of the defaults, which is what the tests are about.
    fn tiny() -> FitConfig {
        FitConfig {
            min_df: 1,
            max_rate: 0.9,
            max_plain_df: 0.9,
            ..Default::default()
        }
    }

    #[test]
    fn a_drafting_word_never_seen_in_plain_copy_still_gets_a_weight() {
        let pos = bank(&[
            "the department shall promulgate regulations pursuant to the act",
            "the commission shall adopt standards pursuant to the act",
            "the agency shall issue guidance pursuant to the act",
            "the director shall act pursuant to the act",
        ]);
        let neg = bank(&[
            "vote for us on tuesday",
            "join our campaign this weekend",
            "we fixed the road last spring",
            "call your neighbours today",
        ]);
        let lex = fit(&pos, &neg, &tiny());
        assert!(lex.terms.contains_key("pursuant"), "{:?}", lex.terms);
        assert!(lex.register_text("pursuant to the act") > 0.0);
        assert_eq!(lex.register_text("vote for us"), 0.0);
    }

    #[test]
    fn shrinkage_keeps_a_single_sighting_below_a_repeated_one() {
        let pos = bank(&[
            "ordinance shall apply",
            "ordinance shall apply",
            "ordinance shall apply",
            "ordinance shall apply",
            "ordinance shall apply",
            "zoning variance once",
        ]);
        let neg = bank(&[
            "join us", "join us", "join us", "join us", "join us", "join us",
        ]);
        let lex = fit(&pos, &neg, &tiny());
        let ordinance = lex.terms["ordinance"];
        let variance = *lex.terms.get("zoning").unwrap_or(&0.0);
        assert!(ordinance > variance, "{ordinance} {variance}");
    }

    #[test]
    fn the_closed_class_never_becomes_a_term() {
        let pos = bank(&[
            "the the the shall shall shall",
            "the shall the shall the shall",
        ]);
        let neg = bank(&["vote vote vote", "join join join"]);
        let lex = fit(&pos, &neg, &tiny());
        assert!(!lex.terms.contains_key("the"));
        assert!(!lex.terms.contains_key("shall"));
        assert!(lex.report.dropped_function > 0);
    }

    #[test]
    fn register_is_a_density_so_length_only_helps_the_length_prior() {
        let lex = Lexicon {
            terms: BTreeMap::from([("alpha".to_string(), 2.0)]),
            cap: 6.0,
            prior_tokens: 20.0,
            report: FitReport::default(),
        };
        let short = lex.register_text("alpha");
        let long = lex.register_text("alpha filler one two three four five six seven");
        assert!(short > long);
        assert!((short - 100.0 * 2.0 / 21.0).abs() < 1e-9);
    }
}
